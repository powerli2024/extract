#!/usr/bin/env python3
"""
MossFormer2_SS_16K 语音分离封装（ClearerVoice / clearvoice）。

关键优化：CLEARVOICE_PYTHON 模式下使用**常驻子进程**，模型只加载一次。
（旧实现每条样本重启进程+重载模型，GPU 利用率会接近 0 且极慢。）

环境变量:
  CLEARVOICE_PYTHON   — 必须与 PYTHON_BIN 同一 VE 解释器（已 pip install clearvoice）
  CLEARVOICE_ROOT     — 可选，仅当用源码树而不是 pip 包时
  MOSS_CKPT_DIR       — 含 MossFormer2_SS_16K/ 的 checkpoints 目录
                        （默认优先 <VB上级>/checkpoints，勿放 VB/ 内）
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from utils_audio import load_audio, peak_normalize, save_audio

_DAEMON_CODE = r"""
import json, sys, os, traceback, gc
from pathlib import Path
import numpy as np
import soundfile as sf

# 协议通道：stdout 只发 JSON；库日志全部进 stderr（避免污染 readline）
_PROTO_OUT = sys.stdout
sys.stdout = sys.stderr

# CUDA 上下文创建前清掉 expandable_segments，否则顶满显存会狂刷 W803
_conf = str(os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "") or "")
if _conf:
    _parts = [p.strip() for p in _conf.split(",") if p.strip() and "expandable_segments" not in p]
    if _parts:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(_parts)
    else:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

def log(msg):
    print(msg, file=sys.stderr, flush=True)

def emit(obj):
    _PROTO_OUT.write(json.dumps(obj) + "\n")
    _PROTO_OUT.flush()

def split_two_speaker(arr):
    raw = tuple(np.asarray(arr).shape)
    a = np.asarray(arr)
    if a.ndim == 3:
        a = np.squeeze(a)
    if a.ndim == 1:
        raise RuntimeError("sep output collapsed to 1-D from %s" % (raw,))
    if a.ndim != 2:
        raise RuntimeError("unexpected sep output shape %s -> %s" % (raw, a.shape))
    n0, n1 = int(a.shape[0]), int(a.shape[1])
    if n0 == 2 and n1 != 2:
        s1, s2 = a[0], a[1]
    elif n1 == 2 and n0 != 2:
        s1, s2 = a[:, 0], a[:, 1]
    elif n0 == 2 and n1 == 2:
        s1, s2 = a[0], a[1]
    else:
        raise RuntimeError("cannot find 2-speaker axis in %s" % (raw,))
    return np.asarray(s1, dtype=np.float32).reshape(-1), np.asarray(s2, dtype=np.float32).reshape(-1)

def _iter_torch_modules(cv):
    import torch
    models = []
    seen = set()

    def add(obj, depth=0):
        if obj is None or depth > 5:
            return
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)
        if isinstance(obj, torch.nn.Module):
            models.append(obj)
            return
        if isinstance(obj, dict):
            for v in obj.values():
                add(v, depth + 1)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                add(v, depth + 1)

    add(cv)
    if hasattr(cv, "__dict__"):
        for v in list(cv.__dict__.values()):
            add(v, 1)
    return models

def release_gpu(cv):
    import torch
    for m in _iter_torch_modules(cv):
        try:
            if hasattr(m, "cpu"):
                m.cpu()
            elif hasattr(m, "to"):
                m.to("cpu")
        except Exception as e:
            log("[clearvoice-daemon] release warn: %s" % e)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    log("[clearvoice-daemon] GPU released (model on CPU)")

def acquire_gpu(cv):
    import torch
    if not torch.cuda.is_available():
        return
    for m in _iter_torch_modules(cv):
        try:
            if hasattr(m, "cuda"):
                m.cuda()
            elif hasattr(m, "to"):
                m.to("cuda")
        except Exception as e:
            log("[clearvoice-daemon] acquire warn: %s" % e)
    log("[clearvoice-daemon] GPU acquired")

def load_cv(model_name):
    import torch
    ckpt_parent = os.environ.get("MOSS_CKPT_PARENT", "").strip()
    if ckpt_parent:
        os.chdir(ckpt_parent)
        log("[clearvoice-daemon] cwd=%s" % os.getcwd())
    # 限制本进程显存占比；MOSS_GPU_FRAC=0/负数 表示不限制（尽量吃满）
    try:
        frac = float(os.environ.get("MOSS_GPU_FRAC", "0.90"))
    except Exception:
        frac = 0.90
    if torch.cuda.is_available() and 0 < frac <= 1.0:
        torch.cuda.set_per_process_memory_fraction(frac, device=0)
        log("[clearvoice-daemon] CUDA memory_fraction=%.2f" % frac)
    elif torch.cuda.is_available():
        log("[clearvoice-daemon] CUDA memory_fraction=unlimited")
    from clearvoice import ClearVoice
    log("[clearvoice-daemon] loading ClearVoice model=%s ..." % model_name)
    cv = ClearVoice(task="speech_separation", model_names=[model_name])
    log("[clearvoice-daemon] model ready")
    return cv

def separate_one(cv, wav_path, out1, out2, sr, max_sec=0):
    import librosa
    import torch
    audio, file_sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if int(file_sr) != int(sr):
        audio = librosa.resample(audio.astype(np.float32), orig_sr=file_sr, target_sr=sr)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)

    def _run():
        output_wav = cv(audio, False)
        arr = np.asarray(output_wav)
        s1, s2 = split_two_speaker(arr)
        sf.write(out1, s1.astype(np.float32), sr)
        sf.write(out2, s2.astype(np.float32), sr)

    try:
        _run()
    except Exception as e1:
        msg = str(e1).lower()
        if torch.cuda.is_available() and ("out of memory" in msg or "oom" in msg):
            log("[clearvoice-daemon] OOM on separate → empty_cache + retry once")
            gc.collect()
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            try:
                _run()
                return
            except Exception:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

def main():
    model_name = os.environ.get("MOSS_MODEL", "MossFormer2_SS_16K")
    try:
        cv = load_cv(model_name)
    except Exception as e:
        emit({"ok": False, "error": "load_failed: " + str(e), "trace": traceback.format_exc()[-1500:]})
        return
    emit({"ok": True, "event": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get("cmd", "separate")
            if cmd == "quit":
                emit({"ok": True, "event": "bye"})
                break
            if cmd == "release_gpu":
                release_gpu(cv)
                emit({"ok": True, "event": "released"})
                continue
            if cmd == "acquire_gpu":
                acquire_gpu(cv)
                emit({"ok": True, "event": "acquired"})
                continue
            if cmd == "empty_cache":
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
                    torch.cuda.empty_cache()
                emit({"ok": True, "event": "empty_cache"})
                continue
            if cmd == "separate_batch":
                # 一批连续分离，减少 IPC 往返，提高 GPU 连续利用率
                results = []
                for it in req.get("items", []):
                    try:
                        separate_one(
                            cv,
                            it["input_wav"],
                            it["out_spk1"],
                            it["out_spk2"],
                            int(it.get("sr", 16000)),
                            max_sec=float(it.get("max_sec", 0) or 0),
                        )
                        results.append({"ok": True, "spk1": it["out_spk1"], "spk2": it["out_spk2"]})
                    except Exception as e:
                        results.append({
                            "ok": False,
                            "error": str(e),
                            "trace": traceback.format_exc()[-800:],
                        })
                emit({"ok": True, "event": "batch", "results": results})
                continue
            # max_sec 仅保留协议兼容；separate_one 始终处理完整音频。
            max_sec = float(req.get("max_sec", 0) or 0)
            separate_one(
                cv,
                req["input_wav"],
                req["out_spk1"],
                req["out_spk2"],
                int(req.get("sr", 16000)),
                max_sec=max_sec,
            )
            emit({"ok": True, "spk1": req["out_spk1"], "spk2": req["out_spk2"]})
        except Exception as e:
            emit({"ok": False, "error": str(e), "trace": traceback.format_exc()[-1500:]})

if __name__ == "__main__":
    main()
"""


def _vm_root() -> Path:
    """VM 包根（本文件在 VM/scripts/）。"""
    return Path(__file__).resolve().parents[1]


def _vb_root() -> Path:
    """兼容旧名：等同 VM 包根。"""
    return _vm_root()


def _media_root() -> Path:
    """VM 上一级：放 checkpoints / outputs / 数据集。"""
    return _vm_root().parent


def _looks_like_lfs_pointer(path: Path) -> bool:
    try:
        if path.stat().st_size > 2048:
            return False
        head = path.read_text(encoding="utf-8", errors="ignore")[:80]
        return head.startswith("version https://git-lfs.github.com")
    except Exception:
        return False


def resolve_moss_ckpt_dir(model_name: str = "MossFormer2_SS_16K") -> Path | None:
    """返回含 <model_name>/ 的 checkpoints 目录（优先 VB 外）。"""
    env = os.environ.get("MOSS_CKPT_DIR", "").strip()
    root = os.environ.get("CLEARVOICE_ROOT", "").strip()
    media = _media_root()
    candidates: list[Path] = []
    if env:
        p = Path(env).expanduser()
        candidates.append(p)
        if p.name == model_name:
            candidates.append(p.parent)
    candidates.extend(
        [
            Path("/root/autodl-tmp/checkpoints"),
            media / "checkpoints",
            Path("/root/checkpoints"),
            Path.home() / "checkpoints",
            # 兼容旧路径（不推荐）
            _vb_root() / "checkpoints",
            Path.cwd() / "checkpoints",
            Path.cwd() / "clearvoice" / "checkpoints",
        ]
    )
    if root:
        r = Path(root)
        candidates.extend(
            [
                r / "clearvoice" / "checkpoints",
                r / "checkpoints",
            ]
        )
    try:
        import clearvoice as cv_pkg  # type: ignore

        pkg = Path(cv_pkg.__file__).resolve().parent
        candidates.append(pkg / "checkpoints")
    except Exception:
        pass

    seen = set()
    for c in candidates:
        try:
            c = c.resolve()
        except Exception:
            continue
        if c in seen:
            continue
        seen.add(c)
        if (c / model_name).is_dir():
            return c
        if c.name == model_name and c.is_dir():
            return c.parent
    return None


def ensure_moss_checkpoint(model_name: str = "MossFormer2_SS_16K") -> Path:
    """
    保证存在 <ckpt_dir>/<model>/last_best_checkpoint.pt。
    返回 checkpoints 的父目录供 ClearVoice daemon chdir。
    """
    ckpt_dir = resolve_moss_ckpt_dir(model_name)
    media = _media_root()
    if ckpt_dir is None:
        raise FileNotFoundError(
            f"未找到 {model_name}。请放到 VB 外:\n"
            f"  cd VM && ./download_models.sh\n"
            f"  # → {media}/checkpoints/{model_name}/last_best_checkpoint.pt\n"
            f"或: export MOSS_CKPT_DIR={media}/checkpoints"
        )

    model_dir = ckpt_dir / model_name
    pt = model_dir / "last_best_checkpoint.pt"
    bare = model_dir / "last_best_checkpoint"

    if pt.exists() and _looks_like_lfs_pointer(pt):
        raise RuntimeError(f"{pt} 是 Git LFS 指针，请重新 ./download_models.sh")
    if not pt.exists():
        if bare.exists() and not _looks_like_lfs_pointer(bare):
            try:
                pt.symlink_to(bare.name)
            except OSError:
                import shutil

                shutil.copy2(bare, pt)
            print(f"[moss] 已链接 {bare.name} → {pt.name}", flush=True)
        else:
            raise FileNotFoundError(
                f"缺少 {pt}\n请运行: cd VM && ./download_models.sh"
            )

    # ClearVoice reload_for_eval: last_best_checkpoint 是【文本指针】，内容为权重文件名
    # （例如 "last_best_checkpoint.pt"），绝不能是 .pt 的 symlink/二进制副本
    def _ensure_pointer(pointer: Path, weight_name: str = "last_best_checkpoint.pt") -> None:
        need = weight_name.strip() + "\n"
        if pointer.is_symlink():
            pointer.unlink()
        elif pointer.is_file():
            try:
                cur = pointer.read_text(encoding="utf-8").strip()
                # 已是合法指针且指向的文件存在
                if cur and (pointer.parent / cur).is_file() and not _looks_like_lfs_pointer(
                    pointer.parent / cur
                ):
                    return
            except UnicodeDecodeError:
                # 误把二进制当成指针 → 重写
                pointer.unlink()
        pointer.write_text(need, encoding="utf-8")
        print(f"[moss] ClearVoice 文本指针 → {pointer} 内容={weight_name!r}", flush=True)

    _ensure_pointer(bare, "last_best_checkpoint.pt")

    if ckpt_dir.name == "checkpoints":
        parent = ckpt_dir.parent
    else:
        parent = media
        link_root = parent / "checkpoints" / model_name
        link_root.mkdir(parents=True, exist_ok=True)
        link_pt = link_root / "last_best_checkpoint.pt"
        if not link_pt.exists():
            try:
                link_pt.symlink_to(pt.resolve())
            except OSError:
                import shutil

                shutil.copy2(pt, link_pt)
        _ensure_pointer(link_root / "last_best_checkpoint", "last_best_checkpoint.pt")

    expected_pt = parent / "checkpoints" / model_name / "last_best_checkpoint.pt"
    expected_bare = parent / "checkpoints" / model_name / "last_best_checkpoint"
    if not expected_pt.exists():
        raise FileNotFoundError(f"权重路径仍不可用: {expected_pt}")
    _ensure_pointer(expected_bare, "last_best_checkpoint.pt")
    return parent


def _try_import_clearvoice():
    from clearvoice import ClearVoice  # type: ignore

    return ClearVoice


class MossFormer2Separator:
    def __init__(
        self,
        model_name: str = "MossFormer2_SS_16K",
        peak: float = 0.7,
        device: str = "cuda:0",
    ):
        self.model_name = model_name
        self.peak = peak
        self.device = device
        self._cv = None
        self._mode = "none"
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._workdir: tempfile.TemporaryDirectory | None = None
        self._init_error = ""
        self._init_backend()

    def _init_backend(self) -> None:
        cv_py = os.environ.get("CLEARVOICE_PYTHON", "").strip()
        if cv_py:
            py = Path(cv_py).expanduser()
            if not py.is_file():
                root = os.environ.get("CLEARVOICE_ROOT", "").strip()
                for base in (Path.cwd(), Path(root) if root else None):
                    if base is None:
                        continue
                    cand = (base / cv_py).resolve()
                    if cand.is_file():
                        py = cand
                        break
            if not py.is_file():
                # 无效路径：不要直接 failed，回退到当前进程 import clearvoice
                print(
                    f"[moss][WARN] CLEARVOICE_PYTHON 不存在: {cv_py}，"
                    "尝试当前 Python in-process import clearvoice ...",
                    flush=True,
                )
                os.environ.pop("CLEARVOICE_PYTHON", None)
                cv_py = ""
            else:
                os.environ["CLEARVOICE_PYTHON"] = str(py.resolve())
                self._mode = "subprocess"
                try:
                    self._start_daemon()
                except Exception as e:
                    self._mode = "failed"
                    self._init_error = f"启动 ClearVoice 常驻进程失败: {e}"
                return
        try:
            ClearVoice = _try_import_clearvoice()
            ckpt_parent = ensure_moss_checkpoint(self.model_name)
            print(
                f"[moss] loading ClearVoice in-process cwd={ckpt_parent} ...",
                flush=True,
            )
            old_cwd = os.getcwd()
            try:
                os.chdir(ckpt_parent)
                self._cv = ClearVoice(
                    task="speech_separation",
                    model_names=[self.model_name],
                )
            finally:
                os.chdir(old_cwd)
            self._mode = "inprocess"
            self._init_error = ""
            print("[moss] ClearVoice ready", flush=True)
        except Exception as e:
            self._import_error = e
            self._init_error = str(e)
            self._mode = "failed"

    def _daemon_env(self, ckpt_parent: Path) -> dict:
        env = os.environ.copy()
        root = os.environ.get("CLEARVOICE_ROOT", "").strip()
        if root:
            root = str(Path(root).resolve())
            env["CLEARVOICE_ROOT"] = root
        # 去掉会遮蔽 pip 包的 PYTHONPATH
        if "PYTHONPATH" in env:
            parts = []
            for p in env["PYTHONPATH"].split(os.pathsep):
                if not p:
                    continue
                if root and Path(p).resolve() == Path(root).resolve():
                    continue
                parts.append(p)
            if parts:
                env["PYTHONPATH"] = os.pathsep.join(parts)
            else:
                env.pop("PYTHONPATH", None)
        env["MOSS_MODEL"] = self.model_name
        env["MOSS_CKPT_PARENT"] = str(ckpt_parent)
        env["MOSS_GPU_FRAC"] = os.environ.get("MOSS_GPU_FRAC", "0.90")
        env["PYTHONUNBUFFERED"] = "1"
        # 同卡顶满时 expandable_segments 会狂刷 W803
        conf = str(env.get("PYTORCH_CUDA_ALLOC_CONF", "") or "")
        if (not conf.strip()) or ("expandable_segments" in conf):
            env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        # 避免 libgomp 刷屏
        if not env.get("OMP_NUM_THREADS") or not str(env.get("OMP_NUM_THREADS", "")).isdigit():
            env["OMP_NUM_THREADS"] = "8"
        return env

    def _start_daemon(self) -> None:
        py = os.environ["CLEARVOICE_PYTHON"]
        try:
            ckpt_parent = ensure_moss_checkpoint(self.model_name)
        except Exception as e:
            raise RuntimeError(f"MossFormer2 权重未就绪: {e}") from e

        self._workdir = tempfile.TemporaryDirectory(prefix="vb_moss_daemon_")
        worker = Path(self._workdir.name) / "daemon.py"
        worker.write_text(_DAEMON_CODE, encoding="utf-8")
        print(
            f"[moss] starting ClearVoice daemon: {py}\n"
            f"[moss] checkpoint cwd={ckpt_parent}  "
            f"(期望 {ckpt_parent}/checkpoints/{self.model_name}/last_best_checkpoint.pt)",
            flush=True,
        )
        self._proc = subprocess.Popen(
            [py, str(worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(ckpt_parent),
            env=self._daemon_env(ckpt_parent),
        )
        atexit.register(self.close)

        # 异步把 stderr 打到主进程，方便看下载进度
        def _pump_err():
            assert self._proc is not None and self._proc.stderr is not None
            for line in self._proc.stderr:
                sys.stderr.write("[clearvoice] " + line)
                sys.stderr.flush()

        threading.Thread(target=_pump_err, daemon=True).start()

        assert self._proc.stdout is not None
        # 跳过空行/非 JSON（库偶发污染）；等 ready 或 load_failed
        msg = None
        deadline = time.time() + float(os.environ.get("MOSS_DAEMON_READY_SEC", "600"))
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"ClearVoice daemon 已退出 code={self._proc.returncode}（加载失败，见上方 [clearvoice] 日志）"
                )
            line = self._proc.stdout.readline()
            if not line:
                time.sleep(0.05)
                continue
            line = line.strip()
            if not line or not line.startswith("{"):
                # 误入 stdout 的文本：转到 stderr 显示后继续等
                print(f"[clearvoice][stdout-skip] {line[:300]}", file=sys.stderr, flush=True)
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                print(f"[clearvoice][stdout-skip] {line[:300]}", file=sys.stderr, flush=True)
                continue
            break
        if msg is None:
            raise RuntimeError(
                "ClearVoice daemon 等待 ready 超时（可调 MOSS_DAEMON_READY_SEC）"
            )
        if not msg.get("ok"):
            raise RuntimeError(msg.get("error", "daemon load failed"))
        print("[moss] ClearVoice daemon ready (model kept in memory)", flush=True)

    def status_message(self) -> str:
        if self._mode == "inprocess":
            return "inprocess（当前 Python 已加载 ClearVoice，模型常驻）"
        if self._mode == "subprocess":
            return (
                f"daemon（CLEARVOICE_PYTHON={os.environ.get('CLEARVOICE_PYTHON')}，"
                "模型常驻，勿每条重启）"
            )
        return "failed — " + (self._init_error or str(getattr(self, "_import_error", "")))

    def separate(
        self,
        wav: np.ndarray,
        sr: int = 16000,
        max_sec: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        wav = peak_normalize(np.asarray(wav, dtype=np.float32), peak=self.peak)
        if self._mode == "inprocess":
            return self._separate_inprocess(wav, sr, max_sec=max_sec)
        if self._mode == "subprocess":
            return self._separate_daemon(wav, sr, max_sec=max_sec)
        raise RuntimeError("ClearVoice 不可用: " + self.status_message())

    def _separate_inprocess(
        self, wav: np.ndarray, sr: int, max_sec: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        audio = wav.reshape(1, -1).astype(np.float32)
        output_wav = self._cv(audio, False)
        arr = np.asarray(output_wav)
        from sep_common import split_two_speaker_wav

        s1, s2 = split_two_speaker_wav(arr)
        return peak_normalize(s1, self.peak), peak_normalize(s2, self.peak)

    def _tmp_dir(self):
        """优先 /dev/shm，减少磁盘 IO。"""
        shm = Path("/dev/shm")
        if shm.is_dir() and os.access(shm, os.W_OK):
            return tempfile.TemporaryDirectory(prefix="vb_moss_req_", dir=str(shm))
        return tempfile.TemporaryDirectory(prefix="vb_moss_req_")

    def _separate_daemon(
        self, wav: np.ndarray, sr: int, max_sec: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._proc is None or self._proc.poll() is not None:
            self._start_daemon()
        assert self._proc is not None and self._proc.stdin and self._proc.stdout
        with self._lock:
            with self._tmp_dir() as td:
                td_p = Path(td)
                in_wav = td_p / "in.wav"
                out1 = td_p / "spk1.wav"
                out2 = td_p / "spk2.wav"
                save_audio(in_wav, wav, sr)
                req = {
                    "cmd": "separate",
                    "input_wav": str(in_wav),
                    "out_spk1": str(out1),
                    "out_spk2": str(out2),
                    "sr": sr,
                    "max_sec": float(max_sec or 0),
                }
                def _ask_once() -> dict:
                    assert self._proc is not None and self._proc.stdin and self._proc.stdout
                    self._proc.stdin.write(json.dumps(req) + "\n")
                    self._proc.stdin.flush()
                    line = self._proc.stdout.readline()
                    if not line:
                        raise RuntimeError("ClearVoice daemon 无响应（可能已崩溃）")
                    return json.loads(line)

                msg = _ask_once()
                err = str(msg.get("error") or "")
                # OOM：empty_cache 后自动再试一次
                if (not msg.get("ok")) and (
                    "out of memory" in err.lower() or "oom" in err.lower()
                ):
                    print("[moss] OOM → empty_cache 后重试 1 次", flush=True)
                    try:
                        self._proc.stdin.write(
                            json.dumps({"cmd": "empty_cache"}) + "\n"
                        )
                        self._proc.stdin.flush()
                        self._proc.stdout.readline()
                    except Exception:
                        pass
                    msg = _ask_once()
                if not msg.get("ok"):
                    raise RuntimeError(
                        f"ClearVoice 分离失败: {msg.get('error')}\n{msg.get('trace', '')}"
                    )
                s1, _ = load_audio(out1, sr)
                s2, _ = load_audio(out2, sr)
                return peak_normalize(s1, self.peak), peak_normalize(s2, self.peak)

    def separate_many(
        self,
        wavs: list[np.ndarray],
        sr: int = 16000,
        max_sec: float = 0.0,
    ) -> list:
        """
        一批连续分离（单次 IPC），拉长 ClearVoice GPU 忙碌段，提高利用率。
        返回与 wavs 等长；失败项为 Exception。
        """
        if not wavs:
            return []
        if self._mode == "inprocess":
            out: list = []
            for w in wavs:
                try:
                    out.append(self._separate_inprocess(w, sr, max_sec=max_sec))
                except Exception as e:
                    out.append(e)
            return out
        if self._mode != "subprocess":
            raise RuntimeError("ClearVoice 不可用: " + self.status_message())
        if self._proc is None or self._proc.poll() is not None:
            self._start_daemon()
        assert self._proc is not None and self._proc.stdin and self._proc.stdout

        with self._lock:
            with self._tmp_dir() as td:
                td_p = Path(td)
                items = []
                for i, wav in enumerate(wavs):
                    wav = peak_normalize(np.asarray(wav, dtype=np.float32), peak=self.peak)
                    in_wav = td_p / f"in_{i}.wav"
                    out1 = td_p / f"spk1_{i}.wav"
                    out2 = td_p / f"spk2_{i}.wav"
                    save_audio(in_wav, wav, sr)
                    items.append(
                        {
                            "input_wav": str(in_wav),
                            "out_spk1": str(out1),
                            "out_spk2": str(out2),
                            "sr": sr,
                            "max_sec": float(max_sec or 0),
                        }
                    )
                req = {"cmd": "separate_batch", "items": items}
                self._proc.stdin.write(json.dumps(req) + "\n")
                self._proc.stdin.flush()
                line = self._proc.stdout.readline()
                if not line:
                    raise RuntimeError("ClearVoice daemon 无响应（batch）")
                msg = json.loads(line)
                if not msg.get("ok"):
                    raise RuntimeError(msg.get("error", "separate_batch failed"))
                results = msg.get("results") or []
                out = []
                for i, r in enumerate(results):
                    if not r.get("ok"):
                        out.append(RuntimeError(r.get("error", "sep failed")))
                        continue
                    s1, _ = load_audio(r["spk1"], sr)
                    s2, _ = load_audio(r["spk2"], sr)
                    out.append(
                        (peak_normalize(s1, self.peak), peak_normalize(s2, self.peak))
                    )
                while len(out) < len(wavs):
                    out.append(RuntimeError("missing batch result"))
                return out

    def empty_cache(self) -> None:
        if self._mode == "subprocess":
            try:
                self._daemon_cmd("empty_cache")
            except Exception:
                pass
        else:
            try:
                import gc
                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def _daemon_cmd(self, cmd: str) -> None:
        if self._mode != "subprocess" or self._proc is None:
            return
        if self._proc.poll() is not None:
            return
        assert self._proc.stdin and self._proc.stdout
        with self._lock:
            self._proc.stdin.write(json.dumps({"cmd": cmd}) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError(f"ClearVoice daemon 无响应: {cmd}")
            msg = json.loads(line)
            if not msg.get("ok"):
                raise RuntimeError(f"ClearVoice {cmd} 失败: {msg.get('error')}")

    def release_gpu(self) -> None:
        """ASR 前把 MossFormer 挪到 CPU，腾出显存。"""
        if self._mode == "subprocess":
            try:
                self._daemon_cmd("release_gpu")
            except Exception as e:
                print(f"[moss] release_gpu 失败（继续）: {e}", flush=True)
        elif self._mode == "inprocess" and self._cv is not None:
            try:
                import gc
                import torch

                for attr in ("model", "models"):
                    obj = getattr(self._cv, attr, None)
                    if obj is None:
                        continue
                    items = obj.values() if isinstance(obj, dict) else (obj if isinstance(obj, (list, tuple)) else [obj])
                    for m in items:
                        if hasattr(m, "cpu"):
                            m.cpu()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"[moss] inprocess release_gpu 失败: {e}", flush=True)

    def acquire_gpu(self) -> None:
        """ASR 后把 MossFormer 搬回 GPU。"""
        if self._mode == "subprocess":
            try:
                self._daemon_cmd("acquire_gpu")
            except Exception as e:
                print(f"[moss] acquire_gpu 失败（继续）: {e}", flush=True)
        elif self._mode == "inprocess" and self._cv is not None:
            try:
                for attr in ("model", "models"):
                    obj = getattr(self._cv, attr, None)
                    if obj is None:
                        continue
                    items = obj.values() if isinstance(obj, dict) else (obj if isinstance(obj, (list, tuple)) else [obj])
                    for m in items:
                        if hasattr(m, "cuda"):
                            m.cuda()
            except Exception as e:
                print(f"[moss] inprocess acquire_gpu 失败: {e}", flush=True)

    def close(self) -> None:
        try:
            self.release_gpu()
        except Exception:
            pass
        proc = self._proc
        self._proc = None
        self._cv = None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin:
                proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                proc.stdin.flush()
                proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        if self._workdir is not None:
            try:
                self._workdir.cleanup()
            except Exception:
                pass
            self._workdir = None

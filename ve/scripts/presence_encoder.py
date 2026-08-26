#!/usr/bin/env python3
"""Presence 声纹编码器：优先 ERes2NetV2（中文短时），回退 Wespeaker ResNet34-LM。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from audio_io import cosine_sim
from paths import (
    default_campplus_dir,
    default_ecapa_presence_dir,
    default_spk_chs_dir,
    default_vblink100_dir,
    default_vblink_dir,
    setup_sys_path,
)

setup_sys_path()


class PresenceEncoder:
    """统一接口: embed(wav) -> np.ndarray[D]。"""

    name: str = "base"

    def embed(self, wav: np.ndarray, sr: int = 16000) -> np.ndarray:
        raise NotImplementedError

    def embed_batch(self, wavs: list[np.ndarray], sr: int = 16000) -> list[np.ndarray]:
        return [self.embed(w, sr) for w in wavs]

    def sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return cosine_sim(self.embed(a), self.embed(b))


def _diagnose_python() -> str:
    import sys

    return f"python={sys.executable}"


def _import_modelscope_sv():
    """导入 ModelScope speaker-verification pipeline；失败时给出装到当前解释器的明确提示。"""
    import sys

    try:
        import modelscope as ms
    except ImportError as e:
        raise ImportError(
            f"当前解释器未安装 modelscope（{_diagnose_python()}）。\n"
            f"请用同一解释器安装: {sys.executable} -m pip install -U modelscope\n"
            f"（不要只 pip install；conda 环境内须 conda activate 后再装）\n"
            f"原始错误: {e}"
        ) from e

    try:
        from modelscope.pipelines import pipeline
    except Exception as e:
        raise ImportError(
            f"modelscope 在 {_diagnose_python()} 可 import，但 pipelines 失败: "
            f"{type(e).__name__}: {e}\n"
            f"modelscope={getattr(ms, '__file__', None)} version={getattr(ms, '__version__', '?')}"
        ) from e

    # Tasks 枚举在部分版本路径不同
    tasks_sv = None
    try:
        from modelscope.utils.constant import Tasks

        tasks_sv = getattr(Tasks, "speaker_verification", None) or getattr(
            Tasks, "speaker-verification", None
        )
    except Exception:
        Tasks = None  # type: ignore
    if tasks_sv is None:
        tasks_sv = "speaker-verification"
    return pipeline, tasks_sv, ms


class ERes2NetV2Encoder(PresenceEncoder):
    """ModelScope iic/speech_eres2netv2_sv_zh-cn_16k-common。"""

    name = "eres2netv2_zh"

    def __init__(self, model_dir: str | Path | None = None, device: str = "cuda:0"):
        self.device = device
        self.model_dir = Path(model_dir) if model_dir else None
        self._sv = None
        self._load()

    def _load(self) -> None:
        import sys

        print(f"[INFO] ERes2NetV2: {_diagnose_python()}", flush=True)
        pipeline, task, ms = _import_modelscope_sv()
        print(
            f"[INFO] modelscope={getattr(ms, '__version__', '?')} "
            f"← {getattr(ms, '__file__', None)}",
            flush=True,
        )

        model_id = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
        model_ref: str = model_id
        if self.model_dir and self.model_dir.is_dir():
            tip = self.model_dir / "MODELSCOPE_PATH.txt"
            if tip.is_file():
                model_ref = tip.read_text(encoding="utf-8").strip() or model_id
            elif (self.model_dir / "config.yaml").is_file() or (
                self.model_dir / "configuration.json"
            ).is_file():
                model_ref = str(self.model_dir)
        print(f"[INFO] load PresenceEncoder ERes2NetV2 ← {model_ref}", flush=True)

        last_err: Exception | None = None
        # 兼容不同 modelscope 的 device / revision 参数
        attempts = [
            dict(task=task, model=model_ref, model_revision="master", device=self.device),
            dict(task=task, model=model_ref, device=self.device),
            dict(task=task, model=model_ref, model_revision="master"),
            dict(task=task, model=model_ref),
            dict(task="speaker-verification", model=model_ref, device=self.device),
            dict(task="speaker-verification", model=model_id),
        ]
        for kwargs in attempts:
            try:
                self._sv = pipeline(**kwargs)
                print(f"[INFO] ERes2NetV2 pipeline OK kwargs={list(kwargs)}", flush=True)
                return
            except TypeError as e:
                last_err = e
                continue
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(
            f"ERes2NetV2 pipeline 构建失败（{_diagnose_python()}）: {last_err}"
        ) from last_err

    def embed(self, wav: np.ndarray, sr: int = 16000) -> np.ndarray:
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        last_out: Any = None
        # 优先内存输入，避免批量打分时每条都落盘
        mem_calls = (
            lambda: self._sv([wav], output_emb=True),
            lambda: self._sv([wav], extract_emb=True),
            lambda: self._sv([{"array": wav, "sampling_rate": int(sr)}], output_emb=True),
            lambda: self._sv([{"array": wav, "sampling_rate": int(sr)}]),
            lambda: self._sv([wav]),
        )
        for call in mem_calls:
            try:
                out = call()
            except TypeError:
                continue
            except Exception:
                continue
            last_out = out
            emb = self._parse_emb(out)
            if emb is not None:
                return emb

        import soundfile as sf
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            sf.write(tmp, wav, sr)
            file_calls = (
                lambda: self._sv([tmp], output_emb=True),
                lambda: self._sv([tmp], extract_emb=True),
                lambda: self._sv([tmp]),
                lambda: self._sv([tmp, tmp]),
            )
            for call in file_calls:
                try:
                    out = call()
                except TypeError:
                    continue
                except Exception:
                    continue
                last_out = out
                emb = self._parse_emb(out)
                if emb is not None:
                    return emb
        finally:
            Path(tmp).unlink(missing_ok=True)

        raise RuntimeError(
            f"ERes2NetV2 输出无 embedding: type={type(last_out)} "
            f"keys={list(last_out) if isinstance(last_out, dict) else None}"
        )

    def embed_batch(self, wavs: list[np.ndarray], sr: int = 16000) -> list[np.ndarray]:
        """优先让 ModelScope 一次编码同长度/短 CMD 的多条流；不支持时安全回退单条。"""
        xs = [np.asarray(w, dtype=np.float32).reshape(-1) for w in wavs]
        if not xs:
            return []
        calls = (
            lambda: self._sv(xs, output_emb=True),
            lambda: self._sv(xs, extract_emb=True),
            lambda: self._sv([{"array": x, "sampling_rate": int(sr)} for x in xs], output_emb=True),
            lambda: self._sv(xs),
        )
        for call in calls:
            try:
                out = call()
                embs = self._parse_embs(out, len(xs))
                if embs is not None:
                    return embs
            except Exception:
                continue
        return [self.embed(x, sr) for x in xs]

    @staticmethod
    def _parse_emb(out: Any) -> np.ndarray | None:
        if isinstance(out, dict):
            for k in ("embs", "embedding", "spk_embedding", "emb"):
                if k in out and out[k] is not None:
                    arr = np.asarray(out[k], dtype=np.float32)
                    if arr.ndim == 2:
                        arr = arr[0]
                    return arr.reshape(-1)
        if isinstance(out, (list, tuple)) and out:
            return ERes2NetV2Encoder._parse_emb(out[0])
        if isinstance(out, np.ndarray):
            arr = np.asarray(out, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[0]
            return arr.reshape(-1)
        return None

    @staticmethod
    def _parse_embs(out: Any, n: int) -> list[np.ndarray] | None:
        """只接受严格一一对应的 batch embedding，避免把验证分数误当嵌入。"""
        if isinstance(out, dict):
            for key in ("embs", "embedding", "spk_embedding", "emb"):
                value = out.get(key)
                if value is None:
                    continue
                arr = np.asarray(value, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[0] == n:
                    return [arr[i].reshape(-1) for i in range(n)]
                if n == 1 and arr.ndim in (1, 2):
                    return [arr.reshape(-1) if arr.ndim == 1 else arr[0].reshape(-1)]
        if isinstance(out, (list, tuple)) and len(out) == n:
            parsed = [ERes2NetV2Encoder._parse_emb(x) for x in out]
            if all(x is not None for x in parsed):
                return [x for x in parsed if x is not None]
        return None


class ResNet34Encoder(PresenceEncoder):
    """回退：cnceleb_resnet34_LM（VE/scripts/spk_encoder_resnet34.py，不依赖 VD 仓在场）。"""

    name = "resnet34_lm"

    def __init__(self, model_dir: str | Path, device: str = "cuda:0"):
        setup_sys_path()
        try:
            from spk_encoder_resnet34 import FrozenSpeakerEncoder
        except ImportError as e:
            raise ImportError(
                "无法导入 spk_encoder_resnet34。"
                "请确认 VE/scripts/spk_encoder_resnet34.py 存在，"
                "并已 pip install wespeaker（或 onnxruntime）。"
                f" 原始错误: {e}"
            ) from e

        print(f"[INFO] load PresenceEncoder ResNet34-LM ← {model_dir}", flush=True)
        self._enc = FrozenSpeakerEncoder(model_dir, device=device)

    def embed(self, wav: np.ndarray, sr: int = 16000) -> np.ndarray:
        return self._enc.embed_numpy(wav)


class CAMPlusEncoder(ERes2NetV2Encoder):
    """ModelScope iic/speech_campplus_sv_zh-cn_16k-common。"""

    name = "campplus_zh"

    def _load(self) -> None:
        import sys

        print(f"[INFO] CAM++: {_diagnose_python()}", flush=True)
        pipeline, task, ms = _import_modelscope_sv()
        model_id = "iic/speech_campplus_sv_zh-cn_16k-common"
        model_ref: str = model_id
        if self.model_dir and self.model_dir.is_dir():
            tip = self.model_dir / "MODELSCOPE_PATH.txt"
            if tip.is_file():
                model_ref = tip.read_text(encoding="utf-8").strip() or model_id
            elif (self.model_dir / "config.yaml").is_file() or (
                self.model_dir / "configuration.json"
            ).is_file():
                model_ref = str(self.model_dir)
        print(f"[INFO] load PresenceEncoder CAM++ ← {model_ref}", flush=True)
        last_err: Exception | None = None
        attempts = [
            dict(task=task, model=model_ref, model_revision="master", device=self.device),
            dict(task=task, model=model_ref, device=self.device),
            dict(task=task, model=model_ref),
            dict(task="speaker-verification", model=model_id),
        ]
        for kwargs in attempts:
            try:
                self._sv = pipeline(**kwargs)
                print(f"[INFO] CAM++ pipeline OK kwargs={list(kwargs)}", flush=True)
                return
            except TypeError as e:
                last_err = e
                continue
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"CAM++ pipeline 失败（{sys.executable}）: {last_err}") from last_err


class WespeakerLocalEncoder(PresenceEncoder):
    """WeSpeaker load_model_local：VoxBlink2 SimAM-ResNet 等。"""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        name: str = "wespeaker_local",
        device: str = "cuda:0",
    ):
        setup_sys_path()
        try:
            from spk_encoder_resnet34 import _patch_torchaudio

            _patch_torchaudio()
        except Exception:
            pass
        import wespeaker

        model_dir = Path(model_dir)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"WeSpeaker 本地目录不存在: {model_dir}")
        print(f"[INFO] load PresenceEncoder {name} ← {model_dir}", flush=True)
        self.name = name
        self._m = wespeaker.load_model_local(str(model_dir))
        # device: 'cuda' / 'cpu' / 'cuda:0'
        dev = device
        if device.startswith("cuda"):
            dev = "cuda"
        try:
            self._m.set_device(dev)
        except Exception as e:
            print(f"[WARN] set_device({dev}) failed: {e}", flush=True)

    def embed(self, wav: np.ndarray, sr: int = 16000) -> np.ndarray:
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        pcm_fn = getattr(self._m, "extract_embedding_from_pcm", None)
        if callable(pcm_fn):
            try:
                import torch

                emb = pcm_fn(torch.from_numpy(wav), int(sr))
                return np.asarray(emb, dtype=np.float32).reshape(-1)
            except Exception:
                pass
        import soundfile as sf
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            sf.write(tmp, wav, sr)
            emb = self._m.extract_embedding(tmp)
        finally:
            Path(tmp).unlink(missing_ok=True)
        return np.asarray(emb, dtype=np.float32).reshape(-1)


def create_presence_encoder(
    backend: str = "eres2netv2",
    *,
    eres_dir: str | Path | None = None,
    resnet_dir: str | Path | None = None,
    campplus_dir: str | Path | None = None,
    vblink_dir: str | Path | None = None,
    ecapa_presence_dir: str | Path | None = None,
    vblink100_dir: str | Path | None = None,
    device: str = "cuda:0",
) -> PresenceEncoder:
    backend = (backend or "eres2netv2").lower().strip()
    if backend in ("eres2netv2", "eres", "eres2net"):
        try:
            return ERes2NetV2Encoder(model_dir=eres_dir, device=device)
        except Exception as e:
            import os
            import sys

            print(f"[ERR] ERes2NetV2 加载失败  python={sys.executable}", flush=True)
            print(f"[ERR]   {type(e).__name__}: {e}", flush=True)
            print(
                f"[HINT] 装到当前解释器: {sys.executable} -m pip install -U modelscope",
                flush=True,
            )
            print(
                "[HINT] 或: cd /root/extract/ve && ./setup_env.sh && "
                "ONLY=eres2netv2 ./download_presence_encoders.sh",
                flush=True,
            )
            cause = getattr(e, "__cause__", None) or getattr(e, "__context__", None)
            if cause is not None:
                print(f"[ERR]   cause: {type(cause).__name__}: {cause}", flush=True)
            if os.environ.get("ALLOW_PRESENCE_FALLBACK", "").strip() in ("1", "true", "yes"):
                print("[WARN] ALLOW_PRESENCE_FALLBACK=1 → 回退 ResNet34-LM", flush=True)
                backend = "resnet34"
            else:
                raise RuntimeError(
                    "锁定 Presence 是 ERes2NetV2，禁止静默回退 ResNet。"
                    "装 modelscope 到当前 PYTHON_BIN，或显式 PRESENCE_BACKEND=resnet34_lm。"
                ) from e
    if backend in ("campplus", "cam++", "campplus_zh"):
        # 禁止用 eres_dir 兜底；旧逻辑会让 CAM++ 臂静默加载 ERes 权重。
        return CAMPlusEncoder(
            model_dir=campplus_dir or default_campplus_dir(), device=device
        )
    if backend in ("ecapa", "ecapa_tdnn", "ecapa1024", "ecapa1024_lm"):
        return WespeakerLocalEncoder(
            ecapa_presence_dir or default_ecapa_presence_dir(),
            name="ecapa1024_lm_voxceleb",
            device=device,
        )
    if backend in ("vblink2", "vblink", "vblinkp", "samresnet34"):
        return WespeakerLocalEncoder(
            vblink_dir or default_vblink_dir(),
            name="vblink2_samresnet34",
            device=device,
        )
    if backend in ("vblink100", "vblink2_100", "samresnet100", "vblink2_samresnet100"):
        return WespeakerLocalEncoder(
            vblink100_dir or default_vblink100_dir(),
            name="vblink2_samresnet100",
            device=device,
        )
    if backend in ("resnet34", "resnet34_lm", "wespeaker"):
        return ResNet34Encoder(resnet_dir or default_spk_chs_dir(), device=device)
    raise ValueError(
        f"未知 presence backend: {backend}；"
        "可选: eres2netv2 | campplus | ecapa_tdnn | resnet34_lm | "
        "vblink2 | vblink2_samresnet100"
    )

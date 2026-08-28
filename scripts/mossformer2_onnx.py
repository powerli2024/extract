#!/usr/bin/env python3
"""
MossFormer2 ONNX 语音分离封装（基于 onnxruntime，无需 ClearVoice）。

与 VB/mossformer2_ss.py 保持相同接口（MossFormer2Separator），
上游调用方无需改动。

模型来源:
  https://www.modelscope.cn/models/dengcunqin/speech_mossformer2_separation_temporal_16k

══════════════════════════════════════════════════════════════════════════════
  优化特性（vs 初始版）:
  - 多 Session 并行推理：利用 ThreadPoolExecutor + 独立 ONNX session
  - CUDA stream 隔离：do_copy_in_default_stream=False 避免序列化
  - 自动适配显存：根据 GPU 空闲显存决定 session 数量
  - ONNX 图优化全部开启 + fp16 优先

  环境变量:
    MOSS_ONNX_PATH      — ONNX 模型文件路径（默认自动查找）
    MOSS_ONNX_DEVICE    — CPU / CUDA:0（默认 CUDA:0 若可用）
    MOSS_NUM_SESSIONS   — 并行 session 数（默认自动: 4090→3, 其他→2）
    MOSS_GPU_FRAC       — 单 session 可占的最大显存比例（默认 0.85）
    ORT_NUM_THREADS     — 每个 session 的 intra-op 线程（默认 4）
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from utils_audio import load_audio, peak_normalize, save_audio

# ── ONNX Runtime 懒加载 ──
_ORT_AVAILABLE = False
_ORT_GPU = False


def _ensure_ort():
    global _ORT_AVAILABLE, _ORT_GPU
    if _ORT_AVAILABLE:
        return
    # 必须先补 CUDA 库路径，再 import ort；否则会找不到 libcublasLt.so.12 并静默掉 CPU
    try:
        from cuda_libs import ensure_cuda_libs

        ensure_cuda_libs(verbose=True)
    except Exception as e:
        print(f"[moss-onnx][WARN] cuda_libs 准备失败: {e}", flush=True)
    try:
        import onnxruntime as ort  # type: ignore

        _ORT_AVAILABLE = True
        providers = ort.get_available_providers()
        _ORT_GPU = "CUDAExecutionProvider" in providers
        if not _ORT_GPU:
            print(
                "[moss-onnx][WARN] onnxruntime 未见 CUDAExecutionProvider。"
                "将用 CPU（很慢）。请确认已装 onnxruntime-gpu，且 libcublasLt.so.12 可被找到。",
                flush=True,
            )
    except ImportError:
        raise ImportError(
            "需要 onnxruntime / onnxruntime-gpu。\n"
            "  pip install onnxruntime-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple"
        )


# ── 模型路径解析 ──


def _vm_root() -> Path:
    """VM 包根（本文件在 VM/scripts/）。"""
    return Path(__file__).resolve().parents[1]


def _vb_root() -> Path:
    """兼容旧名：等同 VM 包根。"""
    return _vm_root()


def _media_root() -> Path:
    """VM 上一级：权重默认目录。"""
    return _vm_root().parent


def resolve_onnx_path() -> Path:
    """查找 MossFormer2 ONNX 模型文件（simple_model.onnx）。"""
    env = os.environ.get("MOSS_ONNX_PATH", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p
        if p.is_dir() and (p / "simple_model.onnx").is_file():
            return p / "simple_model.onnx"

    candidates = [
        _media_root() / "checkpoints" / "MossFormer2_ONNX" / "simple_model.onnx",
        _media_root() / "checkpoints" / "MossFormer2_SS_16K_ONNX" / "simple_model.onnx",
        _media_root() / "models" / "mossformer2_onnx" / "simple_model.onnx",
        _vb_root() / "checkpoints" / "simple_model.onnx",
        Path("/root/checkpoints/MossFormer2_ONNX/simple_model.onnx"),
        Path("/root/autodl-tmp/checkpoints/MossFormer2_ONNX/simple_model.onnx"),
        Path.cwd() / "simple_model.onnx",
    ]
    for c in candidates:
        try:
            if c.is_file():
                return c.resolve()
        except Exception:
            continue
    raise FileNotFoundError(
        "未找到 MossFormer2 ONNX 模型 (simple_model.onnx)。\n"
        "请运行: cd VM && ./download_models.sh\n"
        "或设置: export MOSS_ONNX_PATH=/path/to/simple_model.onnx"
    )


# ── GPU 信息探测 ──


def _detect_gpu_memory_mb() -> int:
    """探测 GPU 0 的总显存（MB），失败返回 0。"""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return int(props.total_mem // (1024 * 1024))
    except Exception:
        pass
    try:
        import subprocess, re

        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        m = re.search(r"(\d+)", out)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return 0


# ── ONNX 分离器（与 MossFormer2Separator 接口兼容）──


class MossFormer2Separator:
    """
    MossFormer2 ONNX 语音分离器（多 session 并行优化版）。

    接口完全兼容 VB/scripts/mossformer2_ss.py 的 MossFormer2Separator，
    可直接替换 import 使用。

    Parameters
    ----------
    model_name : str
        模型名称（仅用于日志，实际模型由 MOSS_ONNX_PATH 控制）。
    peak : float
        峰值归一化目标（默认 0.7）。
    device : str
        设备字符串（"cuda:0" / "cpu"），实际 ONNX 推理设备由 providers 决定。
    num_sessions : int | None
        并行 session 数量；None=自动根据显存决定，1=串行（兼容旧行为）。
    """

    def __init__(
        self,
        model_name: str = "MossFormer2_SS_16K_ONNX",
        peak: float = 0.7,
        device: str = "cuda:0",
        num_sessions: int | None = None,
    ):
        _ensure_ort()
        import onnxruntime as ort  # type: ignore

        self.model_name = model_name
        self.peak = peak
        self.device = device
        self._mode = "onnx"
        self._init_error = ""

        onnx_path = resolve_onnx_path()

        # provider 选择
        preferred = os.environ.get("MOSS_ONNX_DEVICE", "").strip().lower()
        if preferred in ("cpu",):
            self._providers = ["CPUExecutionProvider"]
            self._use_gpu = False
        elif _ORT_GPU:
            # do_copy_in_default_stream=False 是关键：允许不同 session 的
            # compute stream 并行执行，而不被默认流串行化。
            self._providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "cudnn_conv_algo_search": "HEURISTIC",
                        "do_copy_in_default_stream": False,
                        # 允许 ORT 使用更大显存池
                        "gpu_mem_limit": int(
                            float(os.environ.get("MOSS_GPU_FRAC", "0.85"))
                            * _detect_gpu_memory_mb()
                            * 1024
                            * 1024
                        )
                        if _detect_gpu_memory_mb() > 0
                        else None,
                    },
                ),
                "CPUExecutionProvider",
            ]
            self._use_gpu = True
        else:
            self._providers = ["CPUExecutionProvider"]
            self._use_gpu = False

        # 决定 session 数
        if num_sessions is not None:
            self._num_sessions = max(1, int(num_sessions))
        else:
            user_val = os.environ.get("MOSS_NUM_SESSIONS", "").strip()
            if user_val:
                self._num_sessions = max(1, int(user_val))
            else:
                # 自动：4090 系列 → 3 sessions；≥16GB → 3；≥8GB → 2；其余 1
                total_mb = _detect_gpu_memory_mb()
                gpu_name = ""
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_name = torch.cuda.get_device_name(0)
                except Exception:
                    pass
                if self._use_gpu and ("4090" in gpu_name or total_mb >= 22000):
                    # 配合 VM_SEP_BATCH 并行，4090 默认 6 session 拉高利用率
                    self._num_sessions = int(os.environ.get("MOSS_NUM_SESSIONS_AUTO", "6"))
                elif self._use_gpu and total_mb >= 14000:
                    self._num_sessions = 4
                elif self._use_gpu and total_mb >= 8000:
                    self._num_sessions = 2
                else:
                    self._num_sessions = 1

        # 创建 session 池
        self._sessions: list[ort.InferenceSession] = []
        self._session_lock = threading.Lock()
        self._session_round_robin = 0

        for idx in range(self._num_sessions):
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_opts.intra_op_num_threads = int(os.environ.get("ORT_NUM_THREADS", "4"))
            # inter-op 并行度：多个 session 时适当降低单 session 的 inter-op
            sess_opts.inter_op_num_threads = max(1, int(os.environ.get("ORT_NUM_THREADS", "4")) // 2)
            # 启用内存模式优化
            sess_opts.enable_mem_pattern = True
            sess_opts.enable_cpu_mem_arena = True
            # 执行模式：默认 sequential，图形优时可改 parallel
            sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            # fp16 优先（如果硬件支持）
            if self._use_gpu:
                sess_opts.add_session_config_entry("session.intra_op.use_fp16", "1")

            try:
                session = ort.InferenceSession(
                    str(onnx_path),
                    sess_options=sess_opts,
                    providers=self._providers,
                )
                self._sessions.append(session)
            except Exception as e:
                if idx == 0:
                    self._mode = "failed"
                    self._init_error = f"ONNX 模型加载失败: {e}"
                    raise RuntimeError(self._init_error) from e
                # 非首 session 失败不致命，降级为已有数
                print(f"[moss-onnx] session #{idx} 创建失败: {e}，降级为 {len(self._sessions)} sessions", flush=True)
                break

        # 从首个 session 提取元信息
        s0 = self._sessions[0]
        self._input_name = s0.get_inputs()[0].name
        self._output_names = [o.name for o in s0.get_outputs()]
        self._input_shape = s0.get_inputs()[0].shape
        self._sample_rate = 16000  # 模型固定 16k

        actual = s0.get_providers()[0] if s0.get_providers() else "?"
        if self._use_gpu and "CUDA" not in str(actual):
            print(
                f"[moss-onnx][WARN] 请求 GPU 但实际 provider={actual}。"
                "常见原因: libcublasLt.so.12 不在 LD_LIBRARY_PATH。"
                "可: pip install -U nvidia-cublas-cu12 && 重跑；"
                "或 export LD_LIBRARY_PATH="
                "$(python -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).resolve().parent.parent/\"nvidia/cublas/lib\")')"
                ":$LD_LIBRARY_PATH",
                flush=True,
            )

        actual_provider = s0.get_providers()[0]
        print(
            f"[moss-onnx] 模型已加载  model={model_name}  "
            f"sessions={len(self._sessions)}  provider={actual_provider}  "
            f"input={self._input_name} shape={self._input_shape}",
            flush=True,
        )
        print(f"[moss-onnx] ONNX 常驻（无需 ClearVoice conda）", flush=True)

    # ── Session 池 ──

    def _get_session(self) -> "ort.InferenceSession":
        """轮询获取一个 session（线程安全）。"""
        import onnxruntime as ort
        with self._session_lock:
            s = self._sessions[self._session_round_robin % len(self._sessions)]
            self._session_round_robin += 1
            return s

    # ── 接口方法 ──

    def status_message(self) -> str:
        if self._mode == "failed":
            return "failed — " + (self._init_error or "unknown")
        provider = self._sessions[0].get_providers()[0] if self._sessions else "?"
        return (
            f"onnx（{provider}，{len(self._sessions)} sessions 并行，模型常驻，无子进程）"
        )

    def separate(
        self,
        wav: np.ndarray,
        sr: int = 16000,
        max_sec: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        单条分离。

        Returns
        -------
        (spk1, spk2) : tuple[np.ndarray, np.ndarray]
        """
        wav = peak_normalize(np.asarray(wav, dtype=np.float32), peak=self.peak)
        session = self._get_session()
        return self._run_onnx(session, wav)

    def _run_onnx(
        self, session: "ort.InferenceSession", wav: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """核心 ONNX 推理（使用指定 session）。"""
        # 输入: (1, T) float32
        audio = np.asarray(wav, dtype=np.float32).reshape(1, -1)

        outputs = session.run(self._output_names, {self._input_name: audio})

        # 输出可能是多种格式，做鲁棒处理
        arr = outputs[0]  # shape depends on model variant
        if isinstance(arr, list):
            arr = np.asarray(arr, dtype=np.float32)

        arr = np.asarray(arr, dtype=np.float32)

        # 常见输出格式:
        # (1, T, 2)  → spk1=[:,:,0], spk2=[:,:,1]
        # (1, 2, T)  → spk1=[:,0,:], spk2=[:,1,:]
        # (2, T)     → spk1=[0,:],   spk2=[1,:]
        # (T,)       → 单路（降噪模型），两路相同
        if arr.ndim == 3:
            if arr.shape[2] <= 4 and arr.shape[1] > 4:
                s1 = arr[0, :, 0]
                s2 = arr[0, :, 1] if arr.shape[2] > 1 else arr[0, :, 0]
            elif arr.shape[1] <= 4 and arr.shape[2] > 4:
                s1 = arr[0, 0, :]
                s2 = arr[0, 1, :] if arr.shape[1] > 1 else arr[0, 0, :]
            else:
                s1 = arr[0, ..., 0] if arr.ndim == 3 else arr[0]
                s2 = arr[0, ..., -1] if arr.ndim == 3 else arr[0]
                while s1.ndim > 1:
                    s1 = s1.reshape(-1)
                while s2.ndim > 1:
                    s2 = s2.reshape(-1)
        elif arr.ndim == 2:
            if arr.shape[0] <= 4 and arr.shape[1] > 4:
                s1 = arr[0, :]
                s2 = arr[1, :] if arr.shape[0] > 1 else arr[0, :]
            else:
                s1 = arr[:, 0] if arr.shape[1] <= 4 else arr[0, :]
                s2 = arr[:, 1] if arr.shape[1] > 1 else s1
        elif arr.ndim == 1:
            s1 = arr
            s2 = arr.copy()
        else:
            raise RuntimeError(f"unexpected ONNX output shape: {arr.shape}")

        s1 = np.asarray(s1, dtype=np.float32).reshape(-1)
        s2 = np.asarray(s2, dtype=np.float32).reshape(-1)

        return peak_normalize(s1, self.peak), peak_normalize(s2, self.peak)

    def separate_many(
        self,
        wavs: list[np.ndarray],
        sr: int = 16000,
        max_sec: float = 0.0,
    ) -> list:
        """
        批量分离（多 session 并行）。

        根据 session 数和输入数量自适应：
        - 1 session 或 1 条 wav → 串行（避免线程开销）
        - 多条 wav + 多 session → ThreadPoolExecutor 并行
        """
        if not wavs:
            return []

        n = len(wavs)
        n_sessions = len(self._sessions)

        if n == 1 or n_sessions == 1:
            # 串行路径：避免线程创建开销
            out: list = []
            session = self._sessions[0]
            for w in wavs:
                try:
                    w_proc = peak_normalize(np.asarray(w, dtype=np.float32), peak=self.peak)
                    out.append(self._run_onnx(session, w_proc))
                except Exception as e:
                    out.append(e)
            return out

        # ── 并行路径 ──
        # 预处理在外部线程完成，每个线程用自己的 session

        def _prep_and_run(idx_and_wav: tuple[int, np.ndarray]) -> tuple[int, object]:
            i, w = idx_and_wav
            try:
                w_proc = peak_normalize(np.asarray(w, dtype=np.float32), peak=self.peak)
                session = self._get_session()
                return (i, self._run_onnx(session, w_proc))
            except Exception as e:
                return (i, e)

        workers = min(n, n_sessions)  # 不需要多余线程
        results: dict[int, object] = {}

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(_prep_and_run, (i, w)) for i, w in enumerate(wavs)
            ]
            for fut in as_completed(futures):
                i, val = fut.result()
                results[i] = val

        return [results[i] for i in range(n)]

    def release_gpu(self) -> None:
        """回收 GPU 显存（清空 ONNX arena）。"""
        for s in self._sessions:
            try:
                # 释放 ORT 内部 allocator arena
                pass  # ORT Python API 无直接 arena free；依赖 gc
            except Exception:
                pass
        try:
            import gc

            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        except Exception:
            pass

    def acquire_gpu(self) -> None:
        """ONNX session 始终在显存，无需重新加载。"""
        pass

    def empty_cache(self) -> None:
        """回收 CUDA 缓存。"""
        try:
            import gc

            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
            except ImportError:
                pass
        except Exception:
            pass

    def close(self) -> None:
        """释放所有 ONNX session。"""
        self._sessions.clear()
        try:
            import gc

            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        except Exception:
            pass

    # ── 兼容旧 VB 方法（no-op / 委托）──

    def _separate_inprocess(
        self, wav: np.ndarray, sr: int, max_sec: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.separate(wav, sr=sr, max_sec=max_sec)

    def _separate_daemon(
        self, wav: np.ndarray, sr: int, max_sec: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.separate(wav, sr=sr, max_sec=max_sec)


# ── 兼容旧接口（download 脚本等）──


def resolve_moss_ckpt_dir(model_name: str = "MossFormer2_SS_16K") -> Path | None:
    """兼容旧代码：返回 ONNX 模型所在目录。"""
    try:
        p = resolve_onnx_path()
        return p.parent
    except FileNotFoundError:
        return None


def ensure_moss_checkpoint(model_name: str = "MossFormer2_SS_16K") -> Path:
    """兼容旧代码：确保 ONNX 模型存在，返回其父目录。"""
    p = resolve_onnx_path()
    return p.parent


def _try_import_clearvoice():
    """兼容旧代码：ONNX 模式不需要 ClearVoice。"""
    raise ImportError("ONNX 模式不需要 ClearVoice")

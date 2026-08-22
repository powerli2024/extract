#!/usr/bin/env python3
"""
ASR 后端（已迁入 VM/scripts，供本流水线使用）。

支持:
  - qwen3_1.7b  → Qwen/Qwen3-ASR-1.7B（默认）
  - qwen3_0.6b  → Qwen/Qwen3-ASR-0.6B
  - paraformer  → FunASR paraformer-zh（可选；需额外依赖）

VM 运行默认离线：须通过 ASR_MODEL_DIR / QWEN3_ASR_DIR 指向本地权重。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from cer_metrics import normalize_for_cer

# 别名 → HF / ModelScope 模型 ID
QWEN3_MODEL_IDS = {
    "qwen3_0.6b": "Qwen/Qwen3-ASR-0.6B",
    "qwen3_0.6B": "Qwen/Qwen3-ASR-0.6B",
    "qwen3-0.6b": "Qwen/Qwen3-ASR-0.6B",
    "Qwen3-ASR-0.6B": "Qwen/Qwen3-ASR-0.6B",
    "qwen3_1.7b": "Qwen/Qwen3-ASR-1.7B",
    "qwen3_1.7B": "Qwen/Qwen3-ASR-1.7B",
    "qwen3-1.7b": "Qwen/Qwen3-ASR-1.7B",
    "Qwen3-ASR-1.7B": "Qwen/Qwen3-ASR-1.7B",
}

_CJK = re.compile(r"[\u4e00-\u9fff]")


def resolve_qwen_model_id(name: str) -> str:
    if name in QWEN3_MODEL_IDS:
        return QWEN3_MODEL_IDS[name]
    # 本地目录或完整 hub id
    return name


def guess_language(wake_text: str) -> str | None:
    """根据唤醒文本粗判强制语言（Qwen3 支持 language=Chinese/English）。"""
    t = normalize_for_cer(wake_text)
    if not t:
        return None
    cjk = len(_CJK.findall(t))
    if cjk >= max(1, len(t) // 2):
        return "Chinese"
    # hi colmo 等英文唤醒
    if re.fullmatch(r"[a-z0-9]+", t):
        return "English"
    return None


def create_asr_backend(
    backend: str = "qwen3_1.7b",
    device: str = "cuda:0",
    model_dir: str | None = None,
    dtype: str = "bfloat16",
    max_new_tokens: int = 64,
) -> "BaseASRBackend":
    """
    backend:
      qwen3_1.7b | qwen3_0.6b | paraformer
    model_dir:
      本地权重目录（优先）；否则用 hub id 自动下载
    """
    b = backend.strip().lower().replace("-", "_")
    if b.startswith("qwen3") or "qwen3_asr" in b or b in ("0.6b", "1.7b"):
        if b in ("1.7b", "qwen3_1.7b", "qwen3_1.7"):
            mid = "Qwen/Qwen3-ASR-1.7B"
        elif b in ("0.6b", "qwen3_0.6b", "qwen3_0.6"):
            mid = "Qwen/Qwen3-ASR-0.6B"
        else:
            mid = resolve_qwen_model_id(backend)
        return Qwen3ASRBackend(
            model_id=mid,
            device=device,
            model_dir=model_dir,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
        )
    if b in ("paraformer", "paraformer_zh", "funasr"):
        return ParaformerASRBackend(device=device, model_dir=model_dir)
    # 允许直接传 hub id / 本地路径当 Qwen
    if "Qwen3-ASR" in backend or (model_dir and Path(model_dir).is_dir()):
        return Qwen3ASRBackend(
            model_id=resolve_qwen_model_id(backend),
            device=device,
            model_dir=model_dir,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
        )
    raise ValueError(
        f"未知 ASR backend={backend!r}，可选: qwen3_0.6b | qwen3_1.7b | paraformer"
    )


class BaseASRBackend:
    name: str = "base"

    def transcribe(
        self,
        wav: np.ndarray,
        sr: int = 16000,
        language: str | None = None,
        wake_text: str | None = None,
        context: str | None = None,
    ) -> tuple[str, float]:
        raise NotImplementedError


class Qwen3ASRBackend(BaseASRBackend):
    """qwen-asr 包 + transformers backend。"""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-ASR-1.7B",
        device: str = "cuda:0",
        model_dir: str | None = None,
        dtype: str = "bfloat16",
        max_new_tokens: int = 64,
    ):
        import torch
        try:
            from qwen_asr import Qwen3ASRModel
        except ModuleNotFoundError as e:
            import sys

            raise ModuleNotFoundError(
                f"当前解释器没有 qwen_asr: {sys.executable}\n"
                f"请: conda activate ve && pip install -U qwen-asr\n"
                f"或: ./setup_env.sh && source ./env.sh\n"
                f"然后确认: {sys.executable} -c 'import qwen_asr'"
            ) from e

        self.name = Path(model_dir or model_id).name
        self.sr = 16000
        local = model_dir or os.environ.get("QWEN3_ASR_DIR", "").strip()
        # 若设置了通用缓存根，尝试子目录
        if not local:
            root = os.environ.get("QWEN3_ASR_ROOT", "").strip()
            if root:
                cand = Path(root) / Path(model_id).name
                if cand.is_dir():
                    local = str(cand)

        if local and Path(local).exists():
            load_path = local
        elif os.environ.get("HF_HUB_OFFLINE", "").strip() in ("1", "true", "TRUE") or os.environ.get(
            "VM_ALLOW_DOWNLOAD", "0"
        ).strip() not in ("1", "true", "TRUE", "yes"):
            raise FileNotFoundError(
                f"ASR 本地权重不存在（运行期禁止下载）: local={local!r} model_id={model_id}\n"
                f"请设置 ASR_MODEL_DIR 指向已下载目录，或先准备 /root/Qwen3-ASR-1.7B"
            )
        else:
            load_path = model_id
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(dtype.lower(), torch.bfloat16)

        # share 模式默认尽量吃满；ASR_GPU_FRAC<=0 表示不限制
        try:
            asr_frac = float(os.environ.get("ASR_GPU_FRAC", "0.90"))
            if torch.cuda.is_available() and 0 < asr_frac <= 1.0:
                torch.cuda.set_per_process_memory_fraction(asr_frac)
        except Exception:
            asr_frac = None

        # 更大 batch → 更高 GPU util；OOM 时 transcribe_many 自动减半
        try:
            max_bs = int(os.environ.get("ASR_MAX_BATCH", "12"))
        except Exception:
            max_bs = 12
        max_bs = max(1, min(max_bs, 32))

        self.model = Qwen3ASRModel.from_pretrained(
            load_path,
            dtype=torch_dtype,
            device_map=device,
            max_inference_batch_size=max_bs,
            max_new_tokens=max_new_tokens,
        )
        self._load_path = load_path
        self._default_chunk = max_bs
        # 抑制 pad_token 刷屏
        try:
            import transformers

            transformers.logging.set_verbosity_error()
        except Exception:
            pass
        try:
            gen = getattr(self.model, "model", None) or self.model
            for obj in (gen, getattr(gen, "generation_config", None), getattr(self.model, "generation_config", None)):
                if obj is None:
                    continue
                eos = getattr(obj, "eos_token_id", None)
                if eos is not None and hasattr(obj, "pad_token_id"):
                    obj.pad_token_id = eos
        except Exception:
            pass

    def transcribe(
        self,
        wav: np.ndarray,
        sr: int = 16000,
        language: str | None = None,
        wake_text: str | None = None,
        context: str | None = None,
    ) -> tuple[str, float]:
        texts = self.transcribe_many(
            [wav], sr=sr, language=language, wake_text=wake_text, context=context
        )
        return texts[0]

    def transcribe_many(
        self,
        wavs: list[np.ndarray],
        sr: int = 16000,
        language: str | None = None,
        wake_text: str | None = None,
        chunk_size: int | None = None,
        context: str | None = None,
    ) -> list[tuple[str, float]]:
        """批量转写；按 chunk 切分，OOM 时自动减半重试。"""
        if not wavs:
            return []
        lang = language
        if lang is None and wake_text:
            lang = guess_language(wake_text)
        ctx = (context or wake_text or "").strip() or None
        if chunk_size is None:
            chunk_size = int(getattr(self, "_default_chunk", 8))

        def _one_chunk(chunk: list[np.ndarray]) -> list[tuple[str, float]]:
            audios = [(np.asarray(w, dtype=np.float32).reshape(-1), int(sr)) for w in chunk]
            # qwen-asr: 尝试 context / prompt；不支持则降级
            results = None
            last_err = None
            for kwargs in (
                {"audio": audios, "language": lang, "context": ctx},
                {"audio": audios, "language": lang, "prompt": ctx},
                {"audio": audios, "language": lang},
            ):
                if kwargs.get("context") is None and "context" in kwargs:
                    kwargs = {k: v for k, v in kwargs.items() if k != "context"}
                if kwargs.get("prompt") is None and "prompt" in kwargs:
                    kwargs = {k: v for k, v in kwargs.items() if k != "prompt"}
                try:
                    results = self.model.transcribe(**kwargs)
                    break
                except TypeError as e:
                    last_err = e
                    continue
            if results is None:
                raise last_err or RuntimeError("ASR transcribe failed")
            out: list[tuple[str, float]] = []
            for i in range(len(chunk)):
                if not results or i >= len(results):
                    out.append(("", 0.0))
                    continue
                text = getattr(results[i], "text", "") or ""
                text = normalize_for_cer(text)
                conf = 0.0 if not text else float(min(0.95, 0.4 + 0.03 * min(len(text), 20)))
                out.append((text, conf))
            return out

        chunk_size = max(1, int(chunk_size))
        all_out: list[tuple[str, float]] = []
        i = 0
        while i < len(wavs):
            cs = chunk_size
            while cs >= 1:
                try:
                    all_out.extend(_one_chunk(wavs[i : i + cs]))
                    i += cs
                    break
                except Exception as e:
                    if "out of memory" in str(e).lower() or "CUDA" in str(type(e).__name__):
                        try:
                            import torch
                            import gc

                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                        if cs == 1:
                            raise
                        cs = max(1, cs // 2)
                        continue
                    raise
        return all_out


def asr_move_to_cpu(asr) -> None:
    """临时腾出 ASR 显存，供长音频 ClearVoice 分离。"""
    import gc

    import torch

    model = getattr(asr, "model", None)
    if model is None:
        return
    try:
        inner = getattr(model, "model", None) or getattr(model, "thinker", None) or model
        if hasattr(inner, "cpu"):
            inner.cpu()
        elif hasattr(model, "to"):
            model.to("cpu")
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def asr_move_to_gpu(asr, device: str = "cuda:0") -> None:
    model = getattr(asr, "model", None)
    if model is None:
        return
    try:
        inner = getattr(model, "model", None) or getattr(model, "thinker", None) or model
        if hasattr(inner, "to"):
            inner.to(device)
        elif hasattr(model, "to"):
            model.to(device)
    except Exception:
        pass


class ParaformerASRBackend(BaseASRBackend):
    """FunASR Paraformer（可选回退；默认流水线用 Qwen3-ASR，不依赖 funasr）。"""

    def __init__(self, device: str = "cuda:0", model_dir: str | None = None):
        try:
            from funasr import AutoModel
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "paraformer 后端需要 funasr。默认请用 qwen3_1.7b；"
                "若确需 Paraformer: pip install funasr"
            ) from e

        self.name = "paraformer-zh"
        self.sr = 16000
        kwargs: dict[str, Any] = {
            "model": model_dir or "paraformer-zh",
            "model_revision": "v2.0.4",
            "device": device,
        }
        if model_dir and Path(model_dir).exists():
            kwargs["model"] = model_dir
        self.model = AutoModel(**kwargs)

    def transcribe(
        self,
        wav: np.ndarray,
        sr: int = 16000,
        language: str | None = None,
        wake_text: str | None = None,
        context: str | None = None,
    ) -> tuple[str, float]:
        import os
        import tempfile

        from utils_audio import save_audio

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            save_audio(tmp, wav.astype(np.float32), 16000)
            results = self.model.generate(
                input=tmp,
                batch_size=1,
                beam_size=3,
            )
            if not results:
                return "", 0.0
            raw = results[0] if isinstance(results, list) else results
            text = raw.get("text", "") if isinstance(raw, dict) else str(raw)
            text = normalize_for_cer(text)
            conf = 0.0
            if isinstance(raw, dict):
                for key in ("score", "confidence"):
                    if raw.get(key) is not None:
                        conf = float(raw[key])
                        break
            if conf <= 0 and text:
                conf = min(0.95, 0.35 + 0.02 * len(text))
            return text, conf
        finally:
            os.unlink(tmp)

#!/usr/bin/env python3
"""音频 I/O 与 OOM 重试工具。"""

from __future__ import annotations

import functools
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

T = TypeVar("T")


def load_audio(path: str | Path, sr: int = 16000) -> tuple[np.ndarray, int]:
    import soundfile as sf

    path = Path(path)
    wav, file_sr = sf.read(str(path), always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if int(file_sr) != int(sr):
        wav = resample_wav(wav, int(file_sr), int(sr), method="librosa_default")
    return wav.astype(np.float32), sr


def resample_wav(
    wav: np.ndarray,
    orig_sr: int,
    target_sr: int,
    *,
    method: str = "poly",
) -> np.ndarray:
    """采样率转换（USEF 8 kHz ↔ 流水线 16 kHz 时用）。

    method:
      poly          — scipy.signal.resample_poly（2:1 首选，有抗混叠）
      soxr_hq       — librosa/soxr 高质量
      kaiser_best   — librosa
      kaiser_fast   — librosa 较快
      librosa_default — 兼容旧 load_audio（librosa 默认 res_type）

    禁止在调用方做「隔点抽取」；详见 NOTES_USEF_RESAMPLE.md。
    """
    x = np.asarray(wav, dtype=np.float32).reshape(-1)
    o, t = int(orig_sr), int(target_sr)
    if o == t:
        return x
    m = (method or "poly").lower().strip()

    if m == "poly":
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(o, t)
        up, down = t // g, o // g
        y = resample_poly(x, up, down)
        return np.asarray(y, dtype=np.float32)

    import librosa

    res_type = {
        "soxr_hq": "soxr_hq",
        "kaiser_best": "kaiser_best",
        "kaiser_fast": "kaiser_fast",
        "librosa_default": None,
    }.get(m)
    if m not in ("soxr_hq", "kaiser_best", "kaiser_fast", "librosa_default"):
        raise ValueError(
            f"未知 resample method={method!r}；"
            f"可选: poly | soxr_hq | kaiser_best | kaiser_fast | librosa_default"
        )
    kwargs = {}
    if res_type is not None:
        kwargs["res_type"] = res_type
    y = librosa.resample(x, orig_sr=o, target_sr=t, **kwargs)
    return np.asarray(y, dtype=np.float32)

def save_audio(path: str | Path, wav: np.ndarray, sr: int = 16000) -> None:
    import soundfile as sf

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(p), np.asarray(wav, dtype=np.float32), sr)


def peak_normalize(wav: np.ndarray, peak: float = 0.95) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    p = float(np.max(np.abs(wav)) + 1e-12)
    if p < 1e-8:
        return wav
    return (wav * (peak / p)).astype(np.float32)


def truncate_wav(wav: np.ndarray, sr: int, max_sec: float) -> np.ndarray:
    if max_sec <= 0:
        return wav
    n = int(max_sec * sr)
    if len(wav) <= n:
        return wav
    return wav[:n].astype(np.float32)


def vad_crop_speech(
    wav: np.ndarray,
    sr: int = 16000,
    *,
    frame_ms: float = 20.0,
    pad_ms: float = 100.0,
    thr_ratio: float = 0.12,
    min_speech_ms: float = 200.0,
    max_sec: float = 4.0,
) -> tuple[np.ndarray, dict]:
    """能量 VAD：裁掉首尾静音，只保留语音段（供 enroll embed）。

    阈值用噪声底 + 高分位能量，避免单帧爆破音把阈值抬过高。
    检测失败则回退原波形。返回 (cropped_wav, meta)。
    """
    w = np.asarray(wav, dtype=np.float32).reshape(-1)
    meta: dict = {
        "vad": "energy",
        "orig_n": int(len(w)),
        "orig_sec": round(len(w) / float(sr), 4),
        "cropped": False,
    }
    if len(w) < int(0.05 * sr):
        meta["reason"] = "too_short"
        return w, meta

    frame = max(1, int(sr * frame_ms / 1000.0))
    n_frames = len(w) // frame
    if n_frames < 3:
        meta["reason"] = "few_frames"
        return w, meta

    frames = w[: n_frames * frame].reshape(n_frames, frame)
    energy = np.mean(frames.astype(np.float64) ** 2, axis=1)
    noise = float(np.percentile(energy, 20))
    peak = float(np.percentile(energy, 95))  # 抗单帧尖峰
    if peak <= noise * 1.5:
        # 能量太平：整段都当语音，只做 max_sec 截断
        cropped = w
        meta["reason"] = "flat_energy"
    else:
        thr = noise + (peak - noise) * float(thr_ratio)
        speech = energy >= thr
        idx = np.flatnonzero(speech)
        if idx.size == 0:
            meta["reason"] = "no_speech"
            return w, meta
        pad = max(0, int(pad_ms / frame_ms))
        i0 = max(0, int(idx[0]) - pad)
        i1 = min(n_frames, int(idx[-1]) + 1 + pad)
        start = i0 * frame
        end = min(len(w), i1 * frame)
        cropped = w[start:end]
        meta["start"] = int(start)
        meta["end"] = int(end)
        meta["thr_energy"] = round(thr, 8)
        meta["speech_frames"] = int(idx.size)

    min_n = int(sr * min_speech_ms / 1000.0)
    if len(cropped) < min_n:
        meta["reason"] = "speech_too_short"
        return w, meta

    if max_sec and max_sec > 0:
        max_n = int(max_sec * sr)
        if len(cropped) > max_n:
            hop = max(1, max_n // 4)
            best_i, best_e = 0, -1.0
            for i in range(0, len(cropped) - max_n + 1, hop):
                seg = cropped[i : i + max_n]
                e = float(np.dot(seg, seg))
                if e > best_e:
                    best_e, best_i = e, i
            tail = len(cropped) - max_n
            if tail > 0:
                e = float(np.dot(cropped[tail:], cropped[tail:]))
                if e > best_e:
                    best_i = tail
            cropped = cropped[best_i : best_i + max_n]

    meta.update(
        {
            "cropped": True,
            "out_n": int(len(cropped)),
            "out_sec": round(len(cropped) / float(sr), 4),
        }
    )
    return cropped.astype(np.float32), meta


def is_oom(err: BaseException | str) -> bool:
    s = str(err).lower()
    return (
        "out of memory" in s
        or "oom" in s
        or "cudaerrormemoryallocation" in s
        or "cuda out of memory" in s
    )


def empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def with_oom_retry(
    fn: Callable[..., T],
    *,
    max_retries: int = 3,
    shorten_sec: list[float] | None = None,
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> Callable[..., T]:
    """装饰器：OOM 时清空显存并可选缩短音频时长后重试。

    被装饰函数若接收关键字 max_sep_sec / max_sec，会在重试时改写。
    """
    shorten = shorten_sec or [6.0, 4.0, 3.0]

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        last: BaseException | None = None
        for attempt in range(max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last = e
                if not is_oom(e) or attempt >= max_retries:
                    raise
                empty_cuda_cache()
                sec = shorten[min(attempt, len(shorten) - 1)]
                if "max_sep_sec" in kwargs:
                    kwargs["max_sep_sec"] = min(float(kwargs["max_sep_sec"] or sec), sec)
                if "max_sec" in kwargs:
                    kwargs["max_sec"] = min(float(kwargs["max_sec"] or sec), sec)
                if on_retry:
                    on_retry(attempt + 1, e)
                time.sleep(0.5)
        assert last is not None
        raise last

    return wrapper


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = float(np.linalg.norm(a) + 1e-8)
    nb = float(np.linalg.norm(b) + 1e-8)
    return float(np.dot(a / na, b / nb))

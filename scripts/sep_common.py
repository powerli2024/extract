"""Shared BSS helpers: OOM retry, speaker-unpack, GPU release.

Used by s1–s8 so cascade / full / gated do not drift.
"""

from __future__ import annotations

import gc
from typing import Any

import numpy as np

from utils_audio import truncate_wav


def is_oom(err: BaseException | str) -> bool:
    s = str(err).lower()
    return "out of memory" in s or "oom" in s or "cudaerrormemoryallocation" in s


def oom_retry_sec(max_sep_sec: float) -> float:
    """Unified OOM fallback length (seconds).

    max_sep_sec<=0 means 'no pre-truncate' on the first try; retry still
    uses 3s energy window. If the first try was already <=3s, cut to 2s.
    """
    shorter = 3.0
    if max_sep_sec > 0:
        shorter = min(float(max_sep_sec), 3.0)
        if shorter >= float(max_sep_sec) - 1e-6:
            shorter = min(shorter, 2.0)
    return float(shorter)


def empty_cache(sep: Any) -> None:
    fn = getattr(sep, "empty_cache", None)
    if callable(fn):
        try:
            fn()
        except Exception:
            pass


def close_sep(sep: Any) -> None:
    if sep is None:
        return
    for name in ("close", "release_gpu"):
        fn = getattr(sep, name, None)
        if callable(fn):
            try:
                fn()
            except Exception as e:
                print(f"[WARN] sep.{name}: {e}", flush=True)
    try:
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def split_two_speaker_wav(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpack ClearVoice / MossFormer output into (spk1, spk2).

    Handles (2,T), (T,2), (1,2,T), (2,1,T), (1,T,2). A size-1 batch
    axis is squeezed first so (1,2,T) does **not** copy spk1 onto spk2.
    """
    raw_shape = tuple(np.asarray(arr).shape)
    a = np.asarray(arr)
    if a.ndim == 3:
        a = np.squeeze(a)
    if a.ndim == 1:
        raise RuntimeError(f"sep output collapsed to 1-D from {raw_shape}")
    if a.ndim != 2:
        raise RuntimeError(f"unexpected sep output shape {raw_shape} -> {a.shape}")
    n0, n1 = int(a.shape[0]), int(a.shape[1])
    if n0 == 2 and n1 != 2:
        s1, s2 = a[0], a[1]
    elif n1 == 2 and n0 != 2:
        s1, s2 = a[:, 0], a[:, 1]
    elif n0 == 2 and n1 == 2:
        s1, s2 = a[0], a[1]
    else:
        raise RuntimeError(f"cannot find 2-speaker axis in {raw_shape}")
    s1 = np.asarray(s1, dtype=np.float32).reshape(-1)
    s2 = np.asarray(s2, dtype=np.float32).reshape(-1)
    return s1, s2


def separate_one_with_oom_retry(sep: Any, wav: np.ndarray, sr: int, max_sep_sec: float):
    try:
        return sep.separate(wav, sr=sr, max_sec=max_sep_sec)
    except Exception as e:
        if not is_oom(e):
            raise
        empty_cache(sep)
        shorter = oom_retry_sec(max_sep_sec)
        print(f"[WARN] sep OOM → retry max_sep_sec={shorter}", flush=True)
        w = wav
        if shorter > 0:
            w = truncate_wav(wav, sr=sr, max_sec=shorter, mode="energy")
        return sep.separate(w, sr=sr, max_sec=shorter)


def separate_batch_resilient(
    sep: Any,
    wavs: list,
    *,
    sr: int,
    max_sep_sec: float,
) -> list:
    """Per-item results; one OOM must not mark the whole batch failed.

    `separate_many` may return Exception per slot or raise. Failures are
    retried with `separate_one_with_oom_retry`.
    """
    n = len(wavs)
    if n == 0:
        return []
    outs: list = [None] * n
    used_many = False
    if n > 1 and hasattr(sep, "separate_many"):
        try:
            many = sep.separate_many(wavs, sr=sr, max_sec=max_sep_sec)
            if len(many) != n:
                raise RuntimeError(f"separate_many len {len(many)} != {n}")
            outs = list(many)
            used_many = True
        except Exception:
            used_many = False
            outs = [None] * n
    for i, wav in enumerate(wavs):
        val = outs[i] if used_many else None
        if val is not None and not isinstance(val, Exception):
            continue
        try:
            outs[i] = separate_one_with_oom_retry(sep, wav, sr, max_sep_sec)
        except Exception as e:
            outs[i] = e
    return outs

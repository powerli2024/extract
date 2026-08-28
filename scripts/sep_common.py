"""Shared BSS helpers: lossless full-utterance inference and GPU release.

Used by s1–s8 so cascade / full / gated do not drift.
"""

from __future__ import annotations

import gc
from typing import Any

import numpy as np

def is_oom(err: BaseException | str) -> bool:
    s = str(err).lower()
    return "out of memory" in s or "oom" in s or "cudaerrormemoryallocation" in s


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


def _exact_length_pair(
    out: Any, n_samples: int, sr: int
) -> tuple[np.ndarray, np.ndarray]:
    """Require two streams and preserve the complete input time axis.

    A model may differ by a few tail samples because of convolution padding.  We
    only trim/pad the *model output* to the input length; input audio is never
    cropped.  A large mismatch is treated as a bad model result.
    """
    if not isinstance(out, (tuple, list)) or len(out) != 2:
        raise RuntimeError(f"separator must return two streams, got {type(out).__name__}")
    fixed = []
    tolerance = max(32, int(float(sr) * 0.020))
    for i, stream in enumerate(out, 1):
        w = np.asarray(stream, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(w)):
            raise RuntimeError(f"spk{i} contains non-finite samples")
        if abs(len(w) - n_samples) > tolerance:
            raise RuntimeError(
                f"spk{i} length mismatch: output={len(w)} input={n_samples} "
                f"(refuse possible truncation)"
            )
        if len(w) < n_samples:
            w = np.pad(w, (0, n_samples - len(w)))
        elif len(w) > n_samples:
            w = w[:n_samples]
        fixed.append(w.astype(np.float32, copy=False))
    return fixed[0], fixed[1]


def separate_one_with_oom_retry(sep: Any, wav: np.ndarray, sr: int, max_sep_sec: float):
    """Separate a complete utterance; OOM retry never shortens the input.

    ``max_sep_sec`` is accepted for CLI compatibility but deliberately ignored.
    It used to energy-crop long inputs, which made CER/ranking incomparable.
    """
    w = np.asarray(wav, dtype=np.float32).reshape(-1)
    try:
        return _exact_length_pair(sep.separate(w, sr=sr, max_sec=0.0), len(w), sr)
    except Exception as e:
        if not is_oom(e):
            raise
        empty_cache(sep)
        print(
            f"[WARN] sep OOM → clear cache and retry full utterance "
            f"duration={len(w) / float(sr):.3f}s (no truncation)",
            flush=True,
        )
        return _exact_length_pair(sep.separate(w, sr=sr, max_sec=0.0), len(w), sr)


def separate_batch_resilient(
    sep: Any,
    wavs: list,
    *,
    sr: int,
    max_sep_sec: float,
) -> list:
    """Full-length inference with sorted batches and recursive batch backoff.

    The caller groups similar lengths into full parallel batches.  A failed/OOM
    batch is bisected until the bad item is isolated.  No retry truncates audio.
    """
    n = len(wavs)
    if n == 0:
        return []
    outs: list = [None] * n
    def run_group(indices: list[int]) -> None:
        if len(indices) > 1 and hasattr(sep, "separate_many"):
            try:
                values = sep.separate_many(
                    [wavs[i] for i in indices], sr=sr, max_sec=0.0
                )
                if len(values) != len(indices):
                    raise RuntimeError(
                        f"separate_many len {len(values)} != {len(indices)}"
                    )
                failed = []
                for i, value in zip(indices, values):
                    if isinstance(value, Exception) or value is None:
                        failed.append(i)
                    else:
                        try:
                            outs[i] = _exact_length_pair(
                                value, len(np.asarray(wavs[i]).reshape(-1)), sr
                            )
                        except Exception:
                            failed.append(i)
                for i in failed:
                    run_group([i])
                return
            except Exception as e:
                if is_oom(e):
                    empty_cache(sep)
                mid = len(indices) // 2
                run_group(indices[:mid])
                run_group(indices[mid:])
                return
        i = indices[0]
        try:
            outs[i] = separate_one_with_oom_retry(
                sep, wavs[i], sr, max_sep_sec
            )
        except Exception as e:
            outs[i] = e

    run_group(list(range(n)))
    return outs

#!/usr/bin/env python3
"""方案 D：过 Presence 门后直接把 CMD mix 送 ASR（不做 TSE / 选路）。"""

from __future__ import annotations

from typing import Any

import numpy as np

from audio_io import peak_normalize


class MixPassthroughExtractor:
    """mixture → peak_normalize → 原样输出。"""

    name = "mix_passthrough"

    def __init__(self, *, peak: float = 0.95, device: str = "cuda:0"):
        del device
        self.peak = float(peak)
        print("[INFO] MixPassthrough ready (CMD mix → ASR)", flush=True)

    def extract(
        self,
        mixture: np.ndarray,
        enroll: np.ndarray,
        *,
        sr: int = 16000,
        max_sec: float = 0.0,
        **_kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del enroll, sr, max_sec
        mix = np.asarray(mixture, dtype=np.float32).reshape(-1)
        out = peak_normalize(mix, peak=self.peak)
        meta = {
            "tse_backend": self.name,
            "routed_stream": "mix",
            "n_samples": int(out.shape[0]),
        }
        return out, meta

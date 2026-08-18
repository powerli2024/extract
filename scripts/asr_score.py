#!/usr/bin/env python3
"""多流 ASR + 神谕打分（仅本地 ASR 权重）。"""

from __future__ import annotations

from typing import Any

import numpy as np

from cer_pinyin import oracle_of, pack_streams, score_streams
from local_models import assert_runtime_models
from paths import setup_sys_path

setup_sys_path()


def create_asr(device: str = "cuda:0", model_dir: str | None = None):
    import os
    from pathlib import Path

    locs = assert_runtime_models(
        need_onnx=False, need_cv=False, need_asr=True, asr_model_dir=model_dir
    )
    p = Path(locs["asr"])
    print(f"[INFO] local ASR → {p} (HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')})")
    from asr_backend import create_asr_backend

    return create_asr_backend(
        backend="qwen3_1.7b",
        device=device,
        model_dir=str(p.resolve()),
    )


def asr_hyps(asr, wavs: dict[str, np.ndarray], wake: str, sr: int = 16000) -> dict[str, str]:
    names = list(wavs.keys())
    arrs = [wavs[n] for n in names]
    if hasattr(asr, "transcribe_many"):
        pairs = asr.transcribe_many(arrs, wake_text=wake)
    else:
        pairs = [asr.transcribe(w, wake_text=wake) for w in arrs]
    return {n: (hyp or "") for n, (hyp, _c) in zip(names, pairs)}


def score_wavs(
    asr,
    wavs: dict[str, np.ndarray],
    wake: str,
    sr: int = 16000,
) -> dict[str, Any]:
    hyps = asr_hyps(asr, wavs, wake, sr)
    cers = score_streams(hyps, wake)
    o, oc = oracle_of(cers)
    return {
        "oracle_stream": o,
        "oracle_cer": round(oc, 4),
        "oracle_hyp": cers[o]["hyp"],
        "metric": cers[o]["metric"],
        "streams": pack_streams(cers),
    }

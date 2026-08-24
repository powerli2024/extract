from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from enroll_quality import QualityPolicy, assess, signal_metrics  # noqa: E402


def test_silence_is_rejected() -> None:
    metrics = signal_metrics(np.zeros(16000, dtype=np.float32), 16000)
    decision, _score, reasons, _review = assess(metrics, QualityPolicy())
    assert decision == "reject"
    assert "level_too_low" in reasons


def test_clean_keyword_signal_has_finite_metrics() -> None:
    sr = 16000
    t = np.arange(int(1.2 * sr), dtype=np.float32) / sr
    # Speech-like multi-tone with a short quiet lead-in; not a MOS surrogate.
    wav = 0.12 * np.sin(2 * np.pi * 220 * t) + 0.04 * np.sin(2 * np.pi * 3100 * t)
    wav[: int(0.1 * sr)] *= 0.02
    metrics = signal_metrics(wav.astype(np.float32), sr)
    assert metrics["duration_sec"] == 1.2
    assert metrics["nonfinite_ratio"] == 0
    assert metrics["occupied_bandwidth_hz"] > 2500
    assert metrics["clip_ratio"] == 0


def test_embedding_stability_is_review_until_labeled_validation() -> None:
    metrics = {
        "nonfinite_ratio": 0.0,
        "duration_sec": 1.2,
        "active_sec": 0.9,
        "rms_dbfs": -18.0,
        "clip_ratio": 0.0,
        "dc_abs": 0.0,
        "speech_ratio": 0.75,
        "occupied_bandwidth_hz": 4000.0,
        "embedding_stability": 0.4,
    }
    decision, _score, reasons, _review = assess(metrics, QualityPolicy())
    assert decision == "review"
    assert reasons == []
    assert "unstable_speaker_embedding" in _review

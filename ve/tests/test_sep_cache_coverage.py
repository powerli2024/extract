from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audio_io import save_audio  # noqa: E402
from run_extract import load_reusable_d1_streams, sep_cache_coverage  # noqa: E402


def test_sep_cache_reports_all_decision_groups_and_missing(tmp_path: Path) -> None:
    root = tmp_path / "sep_streams"
    good = root / "d1" / "pos" / "pos_0"
    good.mkdir(parents=True)
    for name in ("mix.wav", "d1_spk1.wav", "d1_spk2.wav"):
        (good / name).write_bytes(b"x")
    rows = [
        {"uid": "pos_0", "split": "pos", "decision": "accept"},
        {"uid": "neg_0", "split": "neg", "decision": "reject"},
    ]
    report = sep_cache_coverage(rows, root, 1)
    assert report["groups"]["pos_accept"]["d1_pair_saved"] == 1
    assert report["groups"]["neg_reject_or_error"]["mix_saved"] == 0
    assert report["missing_count"] == 1


def test_reusable_streams_require_matching_mix_fingerprint(tmp_path: Path) -> None:
    root = tmp_path / "sep_streams"
    utt = root / "d1" / "pos" / "pos_1"
    wav = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    for name in ("mix", "d1_spk1", "d1_spk2"):
        save_audio(utt / f"{name}.wav", wav, 16000)
    streams, reason = load_reusable_d1_streams(root, "pos", "pos_1", wav, 16000)
    assert reason == "hit"
    assert set(streams or {}) == {"mix", "d1_spk1", "d1_spk2"}
    streams, reason = load_reusable_d1_streams(root, "pos", "pos_1", wav * 0.5, 16000)
    assert streams is None
    assert reason == "mix_fingerprint_mismatch"

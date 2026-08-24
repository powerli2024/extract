from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from optimize_gate_for_score import (  # noqa: E402
    evaluate,
    load_thresholds,
    optimize_thresholds,
    stream_score,
)


def _row(uid: str, split: str, lang: str, mix: float, s1: float, s2: float):
    return {
        "uid": uid,
        "split": split,
        "label": "present" if split == "pos" else "absent",
        "lang": lang,
        "sim_streams": {"mix": mix, "d1_spk1": s1, "d1_spk2": s2},
    }


def test_strict_score_caps_unsafe_separation_rescue() -> None:
    row = _row("p", "pos", "zh", 0.20, 0.50, 0.30)
    assert stream_score(row, "max") == 0.50
    assert abs(stream_score(row, "strict_rescue") - 0.30) < 1e-9


def test_optimizer_uses_real_asr_benefit_not_frr_proxy() -> None:
    rows = [
        _row("p_good", "pos", "zh", 0.60, 0.60, 0.10),
        _row("p_bad", "pos", "zh", 0.40, 0.40, 0.10),
        _row("n", "neg", "zh", 0.45, 0.45, 0.10),
        _row("p_en", "pos", "en", 0.60, 0.60, 0.10),
        _row("n_en", "neg", "en", 0.20, 0.20, 0.10),
    ]
    asr = {
        "p_good": {"uid": "p_good", "status": "ok", "n": 10, "edit_distance": 0},
        # Accepting p_bad provides no CER gain, so it must not justify accepting n.
        "p_bad": {"uid": "p_bad", "status": "ok", "n": 10, "edit_distance": 10},
        "p_en": {"uid": "p_en", "status": "ok", "n": 10, "edit_distance": 0},
    }
    thresholds = optimize_thresholds(rows, asr, policy="max")
    assert thresholds["zh"] > 0.45
    metrics = evaluate(rows, asr, policy="max", thresholds=thresholds)
    assert metrics["rr"] == 1.0
    assert metrics["contest_score"] > 0.5


def test_load_locked_threshold_shape(tmp_path: Path) -> None:
    path = tmp_path / "thr.json"
    path.write_text('{"thr_by_lang":{"zh":0.2,"en":0.3}}', encoding="utf-8")
    assert load_thresholds(path) == {"zh": 0.2, "en": 0.3, "default": 0.2}

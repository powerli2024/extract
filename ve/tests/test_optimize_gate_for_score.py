from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from optimize_gate_for_score import (  # noqa: E402
    _dist,
    evaluate,
    load_thresholds,
    optimize_thresholds,
    stream_score,
    main,
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


def test_distribution_keeps_spread_for_stability_audit() -> None:
    got = _dist([0.1, 0.2, 0.3])
    assert got["p50"] == 0.2
    assert got["std"] > 0.0


def test_main_writes_runtime_threshold_for_global_and_lang_modes(
    tmp_path: Path, monkeypatch
) -> None:
    import json

    decisions = []
    asr = []
    for lang in ("zh", "en"):
        for i in range(2):
            uid = f"p_{lang}_{i}"
            decisions.append(_row(uid, "pos", lang, 0.7 - i * .1, 0.65, 0.1) | {"cmd_text": "打开"})
            asr.append({"uid": uid, "decision": "accept", "cmd_text": "打开", "status": "ok", "n": 2, "edit_distance": 0})
            decisions.append(_row(f"n_{lang}_{i}", "neg", lang, 0.2 + i * .1, 0.25, 0.1) | {"cmd_text": ""})
    dec_path = tmp_path / "dec.jsonl"
    asr_path = tmp_path / "asr.jsonl"
    base_path = tmp_path / "base.json"
    out = tmp_path / "out"
    dec_path.write_text("".join(json.dumps(x) + "\n" for x in decisions), encoding="utf-8")
    asr_path.write_text("".join(json.dumps(x) + "\n" for x in asr), encoding="utf-8")
    base_path.write_text(json.dumps({"thr_by_lang": {"zh": .5, "en": .5}}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "optimize", "--decisions", str(dec_path), "--asr-all-pos", str(asr_path),
        "--baseline-thr", str(base_path), "--out-dir", str(out),
        "--policies", "max", "--threshold-modes", "global,lang_split",
        "--holdout-frac", "0.5", "--seeds", "2", "--strict",
    ])
    assert main() == 0
    threshold = json.loads((out / "recommended_thr.json").read_text(encoding="utf-8"))
    assert threshold["thr_mode"] in {"global", "lang_split"}
    assert threshold["stream_policy"] == "max"
    assert set(threshold["thr_by_lang"]) == {"zh", "en", "default"}

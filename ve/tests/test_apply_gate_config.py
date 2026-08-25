from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from apply_gate_config import apply_rows  # noqa: E402


def test_apply_gate_uses_language_threshold_and_preserves_rows() -> None:
    rows = [
        {"uid": "z", "lang": "zh", "sim_streams": {"mix": .25, "d1_spk1": .2}},
        {"uid": "e", "lang": "en", "sim_streams": {"mix": .25, "d1_spk1": .2}},
    ]
    config = {
        "stream_policy": "max",
        "thr_by_lang": {"zh": .2, "en": .3, "default": .2},
    }
    got = apply_rows(rows, config)
    assert got[0]["decision"] == "accept"
    assert got[1]["decision"] == "reject"
    assert rows[0].get("decision") is None

from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from asr_cer import build_summary  # noqa: E402


def test_summary_cer_uses_reference_character_weighting() -> None:
    records = [
        {"uid": "short", "lang": "zh", "decision": "accept", "status": "ok", "n": 2,
         "edit_distance": 2, "cer": 1.0, "ref_norm": "甲乙"},
        {"uid": "long", "lang": "zh", "decision": "accept", "status": "ok", "n": 10,
         "edit_distance": 0, "cer": 0.0, "ref_norm": "甲乙丙丁戊己庚辛壬癸"},
    ]
    summary = build_summary(records, rr=1.0, meta={})
    assert summary["cer_total"] == round(2 / 12, 6)
    assert summary["cer_total_macro"] == 0.5
    assert summary["by_lang"]["zh"]["cer_total"] == round(2 / 12, 6)

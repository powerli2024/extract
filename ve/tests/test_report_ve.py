from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from report_ve import summarize  # noqa: E402


def test_presence_report_exposes_rr_frr_counts_and_proxy() -> None:
    rows = [
        {"uid": "p0", "split": "pos", "label": "present", "decision": "reject"},
        {"uid": "p1", "split": "pos", "label": "present", "decision": "accept"},
        {"uid": "n0", "split": "neg", "label": "absent", "decision": "reject"},
        {"uid": "n1", "split": "neg", "label": "absent", "decision": "accept"},
    ]
    overall = summarize(rows)["overall"]
    assert overall["rr"] == 0.5
    assert overall["frr"] == 0.5
    assert overall["n_pos_false_reject"] == 1
    assert overall["presence_proxy_score"] == 0.5
    assert "contest_score" not in overall

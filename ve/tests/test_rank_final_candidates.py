from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from rank_final_candidates import score  # noqa: E402


def test_final_score_is_character_weighted_and_reject_aware() -> None:
    rows = {
        "p1": {"split": "pos", "errors": 1, "ref_chars": 2, "rejected": False},
        "p2": {"split": "pos", "errors": 8, "ref_chars": 8, "rejected": True},
        "n1": {"split": "neg", "rejected": True},
        "n2": {"split": "neg", "rejected": False},
    }
    got = score(rows, list(rows))
    assert got["rr"] == 0.5
    assert got["cer"] == 0.9
    assert got["contest_score"] == 0.3

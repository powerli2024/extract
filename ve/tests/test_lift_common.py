from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from lift_common import duration_mismatch  # noqa: E402


def test_empty_hypothesis_retries_even_for_short_cmd() -> None:
    assert duration_mismatch("", 1.2)
    assert duration_mismatch("   ", 2.0)


def test_short_nonempty_cmd_is_not_forced_to_retry() -> None:
    assert not duration_mismatch("开空调", 1.2)

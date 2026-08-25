from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_extract import extracted_filename  # noqa: E402


def test_sep_route_filename_contains_routed_stream() -> None:
    assert extracted_filename("pos_1", "sep_route", {"routed_stream": "spk1"}) == "pos_1__spk1.wav"
    assert extracted_filename("pos_1", "adaptive_route", {"routed_stream": "d1_spk2"}) == "pos_1__d1_spk2.wav"


def test_non_route_filename_stays_compatible() -> None:
    assert extracted_filename("pos_1", "wesep_bsrnn", {"routed_stream": "spk1"}) == "pos_1.wav"

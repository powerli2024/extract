from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_cmd_se_gate_decisions import replace_gate_scores  # noqa: E402
from prepare_cmd_se_asr_arm import select_export  # noqa: E402


def test_se_condition_changes_gate_scores_and_best_better_are_distinct() -> None:
    ranked = [{
        "uid": "u",
        "raw_similarity": {"mix": .2, "d1_spk1": .3, "d1_spk2": .1},
        "se48k_similarity": {"mix": .4, "d1_spk1": .25, "d1_spk2": .2},
        "exported": {"se48k": [
            {"source_stream": "mix", "path": "/a", "similarity": .4},
            {"source_stream": "d1_spk1", "path": "/b", "similarity": .25},
        ]},
    }]
    output, missing = replace_gate_scores([{"uid": "u", "sim_streams": {"mix": .2}}], ranked, "se48k")
    assert not missing
    assert output[0]["presence_score"] == .4
    assert output[0]["sim_streams"]["mix"] == .4
    assert select_export(ranked[0], "se48k", "best")["path"] == "/a"
    assert select_export(ranked[0], "se48k", "better")["path"] == "/b"

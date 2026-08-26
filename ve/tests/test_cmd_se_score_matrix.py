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


def test_mix_top_se48k_uses_only_enhanced_mix_score() -> None:
    ranked = [{
        "uid": "u",
        "raw_similarity": {"mix": .35, "d1_spk1": .30, "d1_spk2": .20},
        # 即使 SE spk1 更高，也只能使用 SE mix，不能退化为三流 max。
        "se48k_similarity": {"mix": .42, "d1_spk1": .80, "d1_spk2": .10},
    }]
    output, missing = replace_gate_scores([{"uid": "u"}], ranked, "mix_top_se48k")
    assert not missing
    assert output[0]["presence_score"] == .42
    assert output[0]["sim_streams"] == {"mix": .42}
    assert output[0]["cmd_se_gate_meta"]["used_se48k_mix"] is True
    assert output[0]["cmd_se_gate_meta"]["raw_selected_stream"] == "mix"


def test_mix_top_se48k_keeps_raw_best_spk_when_mix_is_not_strictly_better() -> None:
    ranked = [{
        "uid": "u",
        "raw_similarity": {"mix": .30, "d1_spk1": .30, "d1_spk2": .20},
        "se48k_similarity": {"mix": .90},
    }]
    output, missing = replace_gate_scores([{"uid": "u"}], ranked, "mix_top_se48k")
    assert not missing
    assert output[0]["presence_score"] == .30
    assert output[0]["cmd_se_gate_meta"]["used_se48k_mix"] is False
    assert output[0]["cmd_se_gate_meta"]["raw_selected_stream"] == "d1_spk1"

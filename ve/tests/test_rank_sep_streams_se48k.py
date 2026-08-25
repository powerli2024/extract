from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from rank_sep_streams_se48k import output_name, rank_sources


def test_rank_sources_is_deterministic():
    assert rank_sources({"d1_spk2": 0.4, "mix": 0.4, "d1_spk1": 0.5}) == [
        "d1_spk1", "d1_spk2", "mix"
    ]


def test_output_name_carries_only_audio_slot_labels():
    assert output_name("pos_1", "se48k", "best", "d1_spk2") == "pos_1__se48k__best__d1_spk2.wav"

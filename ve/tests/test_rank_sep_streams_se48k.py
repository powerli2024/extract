from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from rank_sep_streams_se48k import output_name, rank_sources
from rank_sep_streams_se48k import enhance_with_oom_retry
import numpy as np


def test_rank_sources_is_deterministic():
    assert rank_sources({"d1_spk2": 0.4, "mix": 0.4, "d1_spk1": 0.5}) == [
        "d1_spk1", "d1_spk2", "mix"
    ]


def test_output_name_carries_only_audio_slot_labels():
    assert output_name("pos_1", "se48k", "best", "d1_spk2") == "pos_1__se48k__best__d1_spk2.wav"


def test_unequal_streams_fall_back_to_single_inference():
    class FakeSE:
        def enhance_many_48k(self, waves):
            raise AssertionError("unequal waves must not enter batch")

        def enhance_48k(self, wav):
            return wav

    logs = []
    waves = {"mix": np.zeros(10, dtype=np.float32), "d1_spk1": np.zeros(8, dtype=np.float32)}
    output, mode = enhance_with_oom_retry(FakeSE(), waves, log=logs.append)
    assert mode == "single_unequal"
    assert [len(output[k]) for k in waves] == [10, 8]
    assert logs == ["unequal_lengths_retry_single"]

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from presence_gate import PresenceGate  # noqa: E402
from calibrate_presence import sweep_thresholds  # noqa: E402


def _vec(cosine: float) -> np.ndarray:
    return np.asarray([cosine, math.sqrt(1.0 - cosine**2)], dtype=np.float32)


class FakeEncoder:
    name = "fake"

    def __init__(self, scores: dict[int, float]):
        self.scores = scores

    def embed(self, wav: np.ndarray, sr: int = 16000) -> np.ndarray:
        key = int(np.asarray(wav).reshape(-1)[0])
        if key == 1:
            return np.asarray([1.0, 0.0], dtype=np.float32)
        return _vec(self.scores[key])


class FakeSeparator:
    def separate(self, wav: np.ndarray, sr: int = 16000):
        return np.asarray([3.0], dtype=np.float32), np.asarray([4.0], dtype=np.float32)


def _gate(policy: str, scores: dict[int, float]) -> PresenceGate:
    return PresenceGate(
        FakeEncoder(scores),
        separator=FakeSeparator(),
        use_sep=True,
        sep_depth=1,
        enroll_vad=False,
        stream_policy=policy,
        rescue_high_margin=0.08,
        rescue_floor_margin=0.10,
        rescue_dominance=0.05,
    )


def test_max_mix_and_strict_rescue_have_distinct_scores() -> None:
    scores = {2: 0.20, 3: 0.50, 4: 0.30}
    enroll = np.asarray([1.0], dtype=np.float32)
    cmd = np.asarray([2.0], dtype=np.float32)
    assert abs(_gate("max", scores).score(enroll, cmd).score - 0.50) < 1e-5
    assert abs(_gate("mix", scores).score(enroll, cmd).score - 0.20) < 1e-5
    strict = _gate("strict_rescue", scores).score(enroll, cmd, thr=0.29)
    assert abs(strict.score - 0.30) < 1e-5
    assert strict.reject is False
    assert strict.rescue_eligible is True


def test_strict_rescue_effective_score_matches_floor_constraint() -> None:
    scores = {2: 0.20, 3: 0.50, 4: 0.30}
    result = _gate("strict_rescue", scores).score(
        np.asarray([1.0], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
        thr=0.31,
    )
    assert result.reject is True


def test_strict_rescue_requires_stream_dominance() -> None:
    scores = {2: 0.20, 3: 0.50, 4: 0.47}
    result = _gate("strict_rescue", scores).score(
        np.asarray([1.0], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
        thr=0.25,
    )
    assert abs(result.score - 0.20) < 1e-5
    assert result.reject is True
    assert result.rescue_eligible is False


def test_fixed_far_calibration_respects_security_target() -> None:
    scores = [
        ("present", 0.70),
        ("present", 0.55),
        ("present", 0.35),
        ("absent", 0.40),
        ("absent", 0.30),
        ("absent", 0.20),
        ("absent", 0.10),
    ]
    result = sweep_thresholds(scores, select_by="far", target_far=0.0)
    assert result["recommended"]["far"] == 0.0
    assert result["recommended"]["thr"] > 0.40

#!/usr/bin/env python3
"""Dependency-light tests for the optional DAE-TSE integration boundary."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from candidate_sources import resolve_candidate_sources, write_signature
from export_kws_handoff import discover_indexes
from paths import STAGE_DIRS
from stage_dae_tse import assert_isolated_out_dir


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_wav(path: Path, frames: int = 1600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.linspace(-0.1, 0.1, frames, dtype=np.float32), 16000)


def make_meta(root: Path) -> None:
    raw0 = root / "data/pos/kws_0.wav"
    raw1 = root / "data/pos/kws_1.wav"
    write_wav(raw0)
    write_wav(raw1)
    write_jsonl(
        root / "pos/meta/items.jsonl",
        [
            {"uid": "pos_0", "split": "pos", "id": 0, "wake_text": "你好", "kws_path": str(raw0)},
            {"uid": "pos_1", "split": "pos", "id": 1, "wake_text": "hello", "kws_path": str(raw1)},
        ],
    )


def test_raw_and_s1_resolution() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        make_meta(root)
        raw = resolve_candidate_sources(
            root, source_stage="raw", source_thr="", split="pos"
        )
        assert len(raw) == 2 and all(row["source_stream"] == "original" for row in raw)

        stage = root / "pos" / STAGE_DIRS["s1"]
        write_wav(stage / "wav/pos_0_spk2.wav")
        write_wav(stage / "wav/pos_1_peak.wav")
        write_jsonl(
            stage / "index.jsonl",
            [
                {"uid": "pos_0", "oracle_stream": "spk2", "oracle_cer": 0.0},
                {"uid": "pos_1", "oracle_stream": "original", "oracle_cer": 0.0},
            ],
        )
        s1 = resolve_candidate_sources(
            root, source_stage="s1", source_thr="", split="pos"
        )
        assert Path(s1[0]["source_wav"]).name == "pos_0_spk2.wav"
        assert Path(s1[1]["source_wav"]).name == "pos_1_peak.wav"


def test_gated_stage_is_an_explicit_subset() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        make_meta(root)
        stage = root / "pos" / STAGE_DIRS["s7"] / "thr_a"
        write_wav(stage / "wav/pos_1_spk1.wav")
        write_jsonl(
            stage / "index.jsonl",
            [{"uid": "pos_1", "oracle_stream": "spk1", "oracle_cer": 0.25}],
        )
        rows = resolve_candidate_sources(
            root, source_stage="s7", source_thr="a", split="pos"
        )
        assert [row["uid"] for row in rows] == ["pos_1"]


def test_signature_refuses_mixed_resume() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "signature.json"
        first = write_signature(path, {"model": "a", "data": "x"})
        same = write_signature(path, {"model": "a", "data": "x"})
        assert first["signature_sha256"] == same["signature_sha256"]
        try:
            write_signature(path, {"model": "b", "data": "x"})
        except RuntimeError as exc:
            assert "use a new DAE_OUT" in str(exc)
        else:
            raise AssertionError("signature mismatch was accepted")


def test_dae_output_cannot_pollute_handoff_tree() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assert_isolated_out_dir(root, root / "experiments/dae")
        assert_isolated_out_dir(root, root.parent / "dae_external")
        try:
            assert_isolated_out_dir(root, root / "pos/dae")
        except RuntimeError as exc:
            assert "must not be inside" in str(exc)
        else:
            raise AssertionError("DAE output was allowed inside pos stage tree")


def test_handoff_uses_frozen_stage_allowlist() -> None:
    with tempfile.TemporaryDirectory() as directory:
        split_root = Path(directory) / "pos"
        official = split_root / STAGE_DIRS["s1"]
        experiment = split_root / "dae_tse_experiment"
        write_jsonl(official / "index.jsonl", [{"uid": "pos_0"}])
        write_jsonl(experiment / "index.jsonl", [{"uid": "pos_0"}])
        found = discover_indexes(split_root)
        assert [label for label, _ in found] == [STAGE_DIRS["s1"]]


def main() -> None:
    tests = [
        test_raw_and_s1_resolution,
        test_gated_stage_is_an_explicit_subset,
        test_signature_refuses_mixed_resume,
        test_dae_output_cannot_pollute_handoff_tree,
        test_handoff_uses_frozen_stage_allowlist,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()

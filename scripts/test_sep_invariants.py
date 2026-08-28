#!/usr/bin/env python3
"""Offline invariants for extract@sep BSS (no GPU)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sep_common import (  # noqa: E402
    separate_batch_resilient,
    separate_one_with_oom_retry,
    split_two_speaker_wav,
)
from stage_resume import stage_complete  # noqa: E402
from gate_policy import build_gate_plan  # noqa: E402
from audit_sep_run import check_index  # noqa: E402


def test_split_two_speaker_batch_speakers_time() -> None:
    t = 80
    spk1 = np.linspace(-1.0, 0.5, t, dtype=np.float32)
    spk2 = np.linspace(0.9, -0.2, t, dtype=np.float32)
    arr = np.stack([spk1, spk2], axis=0)[None, ...]  # (1, 2, T) — the ClearVoice bug shape
    a, b = split_two_speaker_wav(arr)
    assert a.shape == (t,) and b.shape == (t,)
    assert np.allclose(a, spk1)
    assert np.allclose(b, spk2)
    assert not np.allclose(a, b)


def test_split_two_speaker_speakers_batch_time() -> None:
    t = 40
    spk1 = np.ones(t, dtype=np.float32)
    spk2 = np.zeros(t, dtype=np.float32) + 0.3
    arr = np.stack([spk1, spk2], axis=0)[:, None, :]  # (2, 1, T)
    a, b = split_two_speaker_wav(arr)
    assert np.allclose(a, spk1) and np.allclose(b, spk2)


def test_kws_chinese_uses_pinyin_cer_not_strict_char_cer() -> None:
    try:
        from cer_pinyin import cer_value
    except ModuleNotFoundError as exc:
        if exc.name == "editdistance":
            return  # setup_env/check_env enforce this runtime dependency.
        raise
    assert cer_value("科目", "科慕", use_pinyin=True) == 0.0
    assert cer_value("科目", "科慕", use_pinyin=False) > 0.0


def test_gate_plan_deduplicates_same_uid_cohort() -> None:
    rows = [
        {"uid": f"pos_{i}", "oracle_cer": 0.0 if i < 19 else 0.25}
        for i in range(20)
    ]
    plan = build_gate_plan(rows, {"a": 0.0, "b": 0.0, "c": 0.025})
    assert plan["a"]["duplicate_of"] is None
    assert plan["b"]["duplicate_of"] == "a"
    assert plan["c"]["duplicate_of"] is None
    assert len(plan["a"]["rows"]) == 20
    assert len(plan["c"]["rows"]) == 1


class _Sep:
    def __init__(self) -> None:
        self.n_many = 0
        self.n_one = 0

    def separate_many(self, wavs, sr=16000, max_sec=0.0):
        self.n_many += 1
        raise RuntimeError("CUDA out of memory batch")

    def separate(self, wav, sr=16000, max_sec=0.0):
        self.n_one += 1
        if self.n_one % 2 == 1:
            raise RuntimeError("CUDA out of memory")
        assert max_sec == 0.0
        w = np.asarray(wav, dtype=np.float32).reshape(-1)
        return w, w + 0.01

    def empty_cache(self) -> None:
        pass


def test_batch_oom_retries_per_item() -> None:
    sep = _Sep()
    wavs = [np.ones(1600, dtype=np.float32) for _ in range(3)]
    outs = separate_batch_resilient(sep, wavs, sr=16000, max_sep_sec=6.0)
    assert len(outs) == 3
    assert all(not isinstance(x, Exception) for x in outs)
    assert sep.n_many == 2  # failed 3-item batch is bisected; 2-item half stays batched
    assert sep.n_one == 6  # every item retries once, always at full length
    assert all(len(x[0]) == len(wavs[i]) for i, x in enumerate(outs))


def test_max_sep_sec_never_truncates_long_audio() -> None:
    class IdentitySep:
        def separate(self, wav, sr=16000, max_sec=0.0):
            assert max_sec == 0.0
            return wav, -np.asarray(wav)

    wav = np.linspace(-1, 1, 16000 * 7 + 123, dtype=np.float32)
    a, b = separate_one_with_oom_retry(IdentitySep(), wav, 16000, 3.0)
    assert len(a) == len(wav) and len(b) == len(wav)
    assert np.allclose(a, wav) and np.allclose(b, -wav)


def test_audit_rejects_truncated_separator_output() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "stage"
        (root / "wav").mkdir(parents=True)
        source = Path(td) / "source.wav"
        sf.write(source, np.zeros(16000 * 4, dtype=np.float32), 16000)
        sf.write(root / "wav" / "pos_1_peak.wav", np.zeros(16000 * 3), 16000)
        row = {
            "uid": "pos_1", "kws_path": str(source), "wake_text": "你好",
            "oracle_cer": 0.0, "metric": "pinyin", "streams": {"original": {}},
        }
        index = root / "index.jsonl"
        index.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        stat = check_index(index)
        assert stat["n_duration_errors"] == 1


def test_full_stage_errors_not_complete() -> None:
    os.environ["VM_SKIP_DONE"] = "1"
    os.environ.pop("VM_FORCE", None)
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        root = tmp_path / "pos" / "s1_onnx_full"
        root.mkdir(parents=True)
        rows = [{"uid": f"pos_{i}", "error": "oom"} for i in range(5)]
        (root / "index.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
        (root / "summary.json").write_text(
            json.dumps({"n_items": 5, "n_ok": 0, "n_fail": 5}), encoding="utf-8"
        )
        assert stage_complete(tmp_path, "s1", "pos", 5) is False


def test_gated_limit_summary_not_complete() -> None:
    os.environ["VM_SKIP_DONE"] = "1"
    os.environ.pop("VM_FORCE", None)
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        parent = tmp_path / "pos" / "s1_onnx_full"
        parent.mkdir(parents=True)
        parent_rows = [{"uid": f"pos_{i}", "oracle_cer": 0.2} for i in range(10)]
        (parent / "index.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in parent_rows), encoding="utf-8"
        )
        gated = tmp_path / "pos" / "s5_onnx_then_cv_gate"
        thr = gated / "thr_a"
        thr.mkdir(parents=True)
        smoke = [{"uid": f"pos_{i}", "oracle_cer": 0.1} for i in range(2)]
        (thr / "index.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in smoke), encoding="utf-8"
        )
        (gated / "summary.json").write_text(
            json.dumps(
                {
                    "parent_stage": "s1",
                    "thr": {"a": 0.1, "b": 0.15, "c": 0.2},
                    "by_thr": {
                        "a": {"n_subset": 2, "index": str(thr / "index.jsonl")},
                        "b": {"n_subset": 0},
                        "c": {"n_subset": 0},
                    },
                    "limit": 2,
                    "partial": True,
                    "catalog_n": 10,
                }
            ),
            encoding="utf-8",
        )
        assert stage_complete(tmp_path, "s5", "pos", 1, gated=True) is False


def test_gated_recompute_parent_even_without_partial_flag() -> None:
    """Old smoke dumps with no partial flag still must not skip."""
    os.environ["VM_SKIP_DONE"] = "1"
    os.environ.pop("VM_FORCE", None)
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        parent = tmp_path / "pos" / "s1_onnx_full"
        parent.mkdir(parents=True)
        parent_rows = [{"uid": f"pos_{i}", "oracle_cer": 0.5} for i in range(8)]
        (parent / "index.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in parent_rows), encoding="utf-8"
        )
        gated = tmp_path / "pos" / "s5_onnx_then_cv_gate"
        thr = gated / "thr_a"
        thr.mkdir(parents=True)
        (thr / "index.jsonl").write_text(
            json.dumps({"uid": "pos_0", "oracle_cer": 0.0}) + "\n", encoding="utf-8"
        )
        (gated / "summary.json").write_text(
            json.dumps(
                {
                    "parent_stage": "s1",
                    "thr": {"a": 0.1, "b": 0.2, "c": 0.3},
                    "by_thr": {
                        "a": {"n_subset": 1, "index": str(thr / "index.jsonl")},
                        "b": {"n_subset": 0},
                        "c": {"n_subset": 0},
                    },
                }
            ),
            encoding="utf-8",
        )
        assert stage_complete(tmp_path, "s5", "pos", 1, gated=True) is False


def test_gated_duplicate_alias_uses_canonical_index_for_completion() -> None:
    os.environ["VM_SKIP_DONE"] = "1"
    os.environ.pop("VM_FORCE", None)
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        parent = tmp_path / "pos" / "s1_onnx_full"
        parent.mkdir(parents=True)
        parent_rows = [
            {"uid": "pos_0", "oracle_cer": 0.0},
            {"uid": "pos_1", "oracle_cer": 0.5},
            {"uid": "pos_2", "oracle_cer": 1.0},
        ]
        (parent / "index.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in parent_rows), encoding="utf-8"
        )
        gated = tmp_path / "pos" / "s5_onnx_then_cv_gate"
        a = gated / "thr_a"
        c = gated / "thr_c"
        a.mkdir(parents=True)
        c.mkdir(parents=True)
        (a / "index.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in parent_rows), encoding="utf-8"
        )
        (c / "index.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in parent_rows[1:]), encoding="utf-8"
        )
        summary = {
            "parent_stage": "s1", "catalog_n": 3, "limit": 0, "partial": False,
            "thr": {"a": 0.0, "b": 0.0, "c": 0.5},
            "by_thr": {
                "a": {"n_subset": 3, "index": str(a / "index.jsonl")},
                "b": {"n_subset": 3, "duplicate_of": "a", "index": str(a / "index.jsonl")},
                "c": {"n_subset": 2, "index": str(c / "index.jsonl")},
            },
        }
        (gated / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        assert stage_complete(tmp_path, "s5", "pos", 1, gated=True) is True


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    raise SystemExit(1 if failed else 0)

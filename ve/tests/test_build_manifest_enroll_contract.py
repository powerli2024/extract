from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_manifest import build_items  # noqa: E402


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def test_strict_manifest_rejects_index_audio_outside_selected_best_sep(tmp_path: Path) -> None:
    data = tmp_path / "data"
    best = tmp_path / "best"
    old = tmp_path / "old"
    for p in (data / "pos", data / "neg", best / "pos", best / "neg", old / "pos"):
        p.mkdir(parents=True, exist_ok=True)
    (data / "pos" / "cmd_0.wav").write_bytes(b"cmd")
    (data / "neg" / "cmd_0.wav").write_bytes(b"cmd")
    (best / "neg" / "kws_0.wav").write_bytes(b"new-neg")
    (old / "pos" / "kws_0.wav").write_bytes(b"old-pos")
    _jsonl(data / "pos.jsonl", [{"id": 0, "唤醒音频": "pos/kws_0.wav", "识别音频": "pos/cmd_0.wav"}])
    _jsonl(data / "neg.jsonl", [{"id": 0, "唤醒音频": "neg/kws_0.wav", "识别音频": "neg/cmd_0.wav"}])
    _jsonl(best / "index.jsonl", [
        {"uid": "pos_0", "split": "pos", "ok": True, "dest_wav": str(old / "pos" / "kws_0.wav")},
        {"uid": "neg_0", "split": "neg", "ok": True, "dest_rel": "neg/kws_0.wav"},
    ])
    items, qc = build_items(data_dir=data, best_sep=best, splits=["pos", "neg"], strict_best_sep=True)
    assert [x["uid"] for x in items] == ["neg_0"]
    assert qc["external_best_sep_enroll"][0]["uid"] == "pos_0"

#!/usr/bin/env python3
"""为 best_sep 音频目录生成 extract 所需的 index.jsonl。

示例:
  python ve/scripts/create_best_sep_index.py \
    --best-sep D:/data/sep_3 --data-dir D:/data/datasetA

要求 best_sep 下存在 pos/、neg/；音频文件名和 datasetA 中的“唤醒音频”一致。
索引使用 dest_rel 相对路径，因此可直接复制到 AutoDL 使用。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON: {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL record must be an object: {path}:{line_no}")
            rows.append(row)
    return rows


def make_records(data_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for split in ("pos", "neg"):
        for row in read_jsonl(data_dir / f"{split}.jsonl"):
            if "id" not in row or not row.get("唤醒音频"):
                raise ValueError(f"{split}.jsonl contains a row without id/唤醒音频")
            uid = f"{split}_{row['id']}"
            if uid in seen:
                raise ValueError(f"duplicate dataset UID: {uid}")
            seen.add(uid)
            rel = str(row["唤醒音频"]).replace("\\", "/").lstrip("/")
            parts = Path(rel).parts
            if not parts or parts[0] != split:
                raise ValueError(f"{uid} wake path must start with {split}/: {rel}")
            records.append({"uid": uid, "split": split, "ok": True, "dest_rel": rel})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate strict best_sep index.jsonl")
    parser.add_argument("--best-sep", type=Path, required=True, help="directory containing pos/ and neg/")
    parser.add_argument("--data-dir", type=Path, required=True, help="datasetA directory")
    parser.add_argument("--overwrite", action="store_true", help="overwrite an existing index.jsonl")
    args = parser.parse_args()

    best_sep = args.best_sep.resolve()
    data_dir = args.data_dir.resolve()
    if not best_sep.is_dir():
        raise SystemExit(f"best_sep directory not found: {best_sep}")
    output = best_sep / "index.jsonl"
    if output.exists() and not args.overwrite:
        raise SystemExit(f"index already exists: {output}; pass --overwrite to replace it")

    records = make_records(data_dir)
    expected = {str(r["dest_rel"]) for r in records}
    missing = sorted(rel for rel in expected if not (best_sep / rel).is_file())
    if missing:
        raise SystemExit(f"missing {len(missing)} expected WAV files; first: {missing[:5]}")

    actual = {
        p.relative_to(best_sep).as_posix()
        for split in ("pos", "neg")
        for p in (best_sep / split).glob("*.wav")
    }
    extra = sorted(actual - expected)
    if extra:
        raise SystemExit(f"found {len(extra)} unexpected WAV files; first: {extra[:5]}")

    with output.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} records to {output}")


if __name__ == "__main__":
    main()

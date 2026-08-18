#!/usr/bin/env python3
"""收集 kws：按 split 分树写 meta。

uid = {split}_{id}（jsonl 的 id，禁止行号）。
正式 datasetA/{pos,neg}.jsonl：id 唯一且与 kws_{id}.wav 一致。
若 DATA_DIR 指向含重复 id 的旧副本，直接失败（不静默消歧）。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from paths import (
    assert_split,
    default_data_dir,
    default_vm_out,
    ensure_meta,
    split_root,
)

_KWS_NAME = re.compile(r"^kws_(\d+)\.wav$", re.I)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            obj["_idx"] = i
            rows.append(obj)
    return rows


def collect_one_split(
    data_dir: Path,
    split: str,
    *,
    allow_id_mismatch: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """返回 (ok, missing, warnings)。uid={split}_{id}。"""
    split = assert_split(split)
    data_dir = data_dir.resolve()
    jl = data_dir / f"{split}.jsonl"
    if not jl.is_file():
        raise FileNotFoundError(f"缺少 {jl}")

    raw_rows = _load_jsonl(jl)
    id_counts: Counter[int] = Counter()
    for row in raw_rows:
        if "id" not in row or row["id"] is None:
            raise SystemExit(
                f"[ERR] {jl} 第 {row['_idx']} 行缺少 id（禁止用行号当 uid）"
            )
        id_counts[int(row["id"])] += 1

    bad_ids = sorted(i for i, n in id_counts.items() if n > 1)
    if bad_ids:
        raise SystemExit(
            f"[ERR] {jl} 存在重复 id（共 {len(bad_ids)} 个），无法用 uid={{split}}_{{id}}。\n"
            f"  例: id={bad_ids[0]} 出现 {id_counts[bad_ids[0]]} 次。\n"
            f"  请使用干净的 datasetA/{{pos,neg}}.jsonl"
            f"（勿用 audio_annotation/data 下的旧副本）。"
        )

    ok: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    seen_uid: set[str] = set()
    seen_kws: set[str] = set()

    for row in raw_rows:
        sid = int(row["id"])
        kws_rel = (row.get("唤醒音频") or row.get("kws_path") or "").strip()
        if not kws_rel:
            raise SystemExit(f"[ERR] {split} id={sid} 无唤醒音频路径")
        kws_rel = kws_rel.replace("\\", "/")
        parts = kws_rel.split("/")
        if parts[0] != split:
            raise SystemExit(
                f"[ERR] split={split} 与路径前缀不一致: {kws_rel} (id={sid})"
            )
        fname = parts[-1]
        m = _KWS_NAME.match(fname)
        if not m:
            raise SystemExit(f"[ERR] 非法 kws 文件名: {kws_rel}")
        kws_num = int(m.group(1))

        if kws_rel in seen_kws:
            raise SystemExit(f"[ERR] 同 split 重复 kws 路径: {kws_rel}")
        seen_kws.add(kws_rel)

        if sid != kws_num:
            msg = f"id={sid} 与文件名 {fname} 不一致"
            if allow_id_mismatch:
                warnings.append(
                    {
                        "type": "id_filename_mismatch",
                        "split": split,
                        "id": sid,
                        "kws_num": kws_num,
                        "kws_rel": kws_rel,
                        "msg": msg,
                    }
                )
                print(f"[WARN] {msg}（仍用 uid={split}_{sid}）")
            else:
                raise SystemExit(
                    f"[ERR] {msg}。正式数据应 id==kws 编号；"
                    f"确认 DATA_DIR 指向 datasetA 而非旧副本。"
                )

        uid = f"{split}_{sid}"
        if uid in seen_uid:
            raise SystemExit(f"[ERR] 重复 uid: {uid}")
        seen_uid.add(uid)

        kws_path = (data_dir / kws_rel).resolve()
        try:
            kws_path.relative_to((data_dir / split).resolve())
        except ValueError as e:
            raise SystemExit(f"[ERR] kws 不在 {split}/ 下: {kws_path}") from e

        item = {
            "uid": uid,
            "split": split,
            "id": sid,
            "kws_num": kws_num,
            "idx": row["_idx"],
            "kws_rel": kws_rel,
            "kws_path": str(kws_path),
            "wake_text": row.get("唤醒文本") or row.get("wake_text") or "",
            "uid_rule": "jsonl_id",
        }
        if not kws_path.is_file():
            missing.append({**item, "error": "kws_path missing"})
            continue
        ok.append(item)
    return ok, missing, warnings


def write_split_meta(
    vm_out: Path,
    split: str,
    items: list[dict],
    missing: list[dict],
    warnings: list[dict],
) -> Path:
    meta = ensure_meta(vm_out, split)
    with (meta / "items.jsonl").open("w", encoding="utf-8") as fw:
        for it in items:
            if it["split"] != split:
                raise SystemExit(f"[ERR] 写入 {split} 时混入 {it['split']}: {it['uid']}")
            if it["uid"] != f"{split}_{it['id']}":
                raise SystemExit(f"[ERR] uid 与 id 不一致: {it}")
            fw.write(json.dumps(it, ensure_ascii=False) + "\n")
    with (meta / "missing.jsonl").open("w", encoding="utf-8") as fw:
        for it in missing:
            fw.write(json.dumps(it, ensure_ascii=False) + "\n")
    with (meta / "warnings.jsonl").open("w", encoding="utf-8") as fw:
        for it in warnings:
            fw.write(json.dumps(it, ensure_ascii=False) + "\n")
    summary = {
        "split": split,
        "n_ok": len(items),
        "n_missing": len(missing),
        "n_warnings": len(warnings),
        "uid_rule": "uid={split}_{id}",
        "split_root": str(split_root(vm_out, split).resolve()),
    }
    (meta / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[OK] meta[{split}] → {meta}  ok={len(items)} missing={len(missing)} warn={len(warnings)}"
    )
    return meta


def load_items(vm_out: Path, split: str) -> list[dict]:
    split = assert_split(split)
    p = split_root(vm_out, split) / "meta" / "items.jsonl"
    if not p.is_file():
        raise FileNotFoundError(f"先跑 collect --splits {split}: {p}")
    items = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    for it in items:
        if it.get("split") != split:
            raise SystemExit(
                f"[ERR] meta 污染: 期望 split={split} 实际 {it.get('split')} uid={it.get('uid')}"
            )
        if it.get("uid") != f"{split}_{it.get('id')}":
            raise SystemExit(f"[ERR] uid 与 id 不一致: {it.get('uid')} id={it.get('id')}")
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect KWS per split into isolated trees")
    ap.add_argument("--data-dir", type=str, default="")
    ap.add_argument("--vm-out", type=str, default="")
    ap.add_argument("--splits", type=str, default="pos,neg")
    ap.add_argument("--limit", type=int, default=0, help="每个 split 各自 limit")
    ap.add_argument(
        "--allow-id-mismatch",
        action="store_true",
        help="允许 id 与 kws 文件名不一致（默认严格失败）",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    vm_out = Path(args.vm_out) if args.vm_out else default_vm_out()
    splits = [assert_split(s.strip()) for s in args.splits.split(",") if s.strip()]

    all_summary: dict[str, Any] = {
        "data_dir": str(data_dir.resolve()),
        "splits": {},
        "uid_rule": "uid={split}_{id}",
    }
    for split in splits:
        items, missing, warnings = collect_one_split(
            data_dir, split, allow_id_mismatch=args.allow_id_mismatch
        )
        if args.limit > 0:
            items = items[: args.limit]
        write_split_meta(vm_out, split, items, missing, warnings)
        all_summary["splits"][split] = {
            "n_ok": len(items),
            "n_missing": len(missing),
            "n_warnings": len(warnings),
        }

    root_meta = vm_out / "meta"
    root_meta.mkdir(parents=True, exist_ok=True)
    (root_meta / "collect_summary.json").write_text(
        json.dumps(all_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[OK] collect_summary", all_summary)


if __name__ == "__main__":
    main()

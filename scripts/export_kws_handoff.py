#!/usr/bin/env python3
"""Build best_sep + kws_handoff.json for github.com/powerli2024/kws.

No MMS-FA. Within-stage oracle already prefers original (cer_pinyin.oracle_of).
Across stages: min oracle_cer, prefer non-original on ties (matches the
existing pos_neg/best_sep collector).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import STAGE_DIRS, VALID_SPLITS, default_vm_out

SKIP_DIRS = {"meta", "reports", "best_sep", "packs"}
STREAM_TO_TAG = {"original": "peak"}
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
SCHEMA = "kws_sep_handoff/v1"
EXTRACT_REPO = "https://github.com/powerli2024/extract"
EXTRACT_BRANCH = "sep"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def discover_indexes(split_root: Path) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    if not split_root.is_dir():
        return found
    for child in sorted(split_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir() or child.name in SKIP_DIRS or child.name.startswith("."):
            continue
        idx = child / "index.jsonl"
        if idx.is_file():
            found.append((child.name, idx))
        for thr in sorted(child.glob("thr_*")):
            tidx = thr / "index.jsonl"
            if thr.is_dir() and tidx.is_file():
                found.append((f"{child.name}/{thr.name}", tidx))
    return found


def wav_rel(stage_label: str, uid: str, stream: str) -> str:
    tag = STREAM_TO_TAG.get(stream, stream)
    if "/" in stage_label:
        stage, thr = stage_label.split("/", 1)
        return f"{stage}/{thr}/wav/{uid}_{tag}.wav"
    return f"{stage_label}/wav/{uid}_{tag}.wav"


def pick_across_stages(cands: list[dict[str, Any]]) -> dict[str, Any]:
    return min(
        cands,
        key=lambda r: (
            float(r["oracle_cer"]),
            0 if r["oracle_stream"] != "original" else 1,
            int(r["order"]),
            str(r["best_stage"]),
        ),
    )


def export_split(vm_out: Path, split: str, *, copy_wavs: bool = True) -> list[dict[str, Any]]:
    split_root = vm_out / split
    indexes = discover_indexes(split_root)
    if not indexes:
        raise FileNotFoundError(f"no stage index.jsonl under {split_root}")
    by_uid: dict[str, list[dict[str, Any]]] = {}
    for order, (label, path) in enumerate(indexes):
        for row in load_jsonl(path):
            if row.get("error") or row.get("oracle_cer") is None:
                continue
            uid = str(row["uid"])
            rec = {
                "uid": uid,
                "split": split,
                "id": row.get("id"),
                "kws_rel": row.get("kws_rel"),
                "kws_path": row.get("kws_path"),
                "wake_text": row.get("wake_text") or "",
                "lang": "zh" if CJK_RE.search(row.get("wake_text") or "") else "en",
                "metric": row.get("metric"),
                "best_stage": label,
                "oracle_stream": row["oracle_stream"],
                "oracle_cer": float(row["oracle_cer"]),
                "oracle_hyp": row.get("oracle_hyp"),
                "order": order,
                "streams": row.get("streams") or {},
            }
            by_uid.setdefault(uid, []).append(rec)
    out: list[dict[str, Any]] = []
    collect = vm_out / "best_sep"
    for uid, cands in sorted(by_uid.items(), key=lambda kv: kv[0]):
        best = pick_across_stages(cands)
        dest_rel = f"{split}/{uid}.wav"
        src_rel = wav_rel(best["best_stage"], uid, str(best["oracle_stream"]))
        src = split_root / src_rel
        dest = collect / dest_rel
        rec = {
            "split": split,
            "uid": uid,
            "id": best["id"],
            "kws_rel": best["kws_rel"],
            "wake_text": best["wake_text"],
            "lang": best["lang"],
            "metric": "cer_py" if best["lang"] == "zh" else "cer_char",
            "best_stage": best["best_stage"],
            "oracle_stream": best["oracle_stream"],
            "oracle_cer": round(float(best["oracle_cer"]), 4),
            "oracle_hyp": best["oracle_hyp"],
            "src_wav": str(src),
            "src_wav_rel": src_rel,
            "dest_wav": str(dest),
            "dest_rel": dest_rel,
            "selector": "oracle_cer_across_stages",
            "mms_fa": False,
        }
        if not src.is_file():
            rec["ok"] = False
            rec["error"] = f"missing_src:{src}"
        elif not copy_wavs:
            rec["ok"] = True
            rec["copied"] = False
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            rec["ok"] = True
            rec["copied"] = True
            rec["bytes"] = dest.stat().st_size
        out.append(rec)
    return out


def write_handoff(vm_out: Path, records: list[dict[str, Any]], splits: list[str]) -> Path:
    payload = {
        "schema": SCHEMA,
        "extract_repo": EXTRACT_REPO,
        "extract_branch": EXTRACT_BRANCH,
        "mms_fa": False,
        "selector_within_stage": "oracle_cer_prefer_original",
        "selector_across_stages": "min_oracle_cer_prefer_sep_then_stage_order",
        "peak_norm": 0.7,
        "truncate_max_sec": 6.0,
        "truncate_mode": "energy",
        "stage_dirs": STAGE_DIRS,
        "splits": splits,
        "vm_out": str(vm_out),
        "n_records": sum(1 for r in records if r.get("ok")),
        "n_ok": sum(1 for r in records if r.get("ok")),
        "n_fail": sum(1 for r in records if not r.get("ok")),
        "kws_repo": "https://github.com/powerli2024/kws",
        "kws_next": [
            f"python scripts/rebuild_best_sep.py --pos-neg {vm_out}",
            f"python scripts/analyze_dual_zero.py --pos-neg {vm_out}",
        ],
    }
    path = vm_out / "kws_handoff.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (vm_out / "best_sep" / "kws_handoff.json").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--vm-out", type=Path, default=None)
    p.add_argument("--splits", default="pos,neg")
    p.add_argument(
        "--no-copy",
        action="store_true",
        help="write index/handoff without copying wavs (src must still exist)",
    )
    p.add_argument(
        "--compare-existing",
        action="store_true",
        help="compare picker to existing best_sep/index.jsonl; do not write",
    )
    return p.parse_args()


def compare_existing(vm_out: Path, records: list[dict[str, Any]]) -> int:
    old_path = vm_out / "best_sep" / "index.jsonl"
    if not old_path.is_file():
        print(f"[ERR] no {old_path}", flush=True)
        return 1
    old = {str(r["uid"]): r for r in load_jsonl(old_path)}
    n_miss = n_stage = n_stream = 0
    for rec in records:
        uid = rec["uid"]
        prev = old.get(uid)
        if prev is None:
            n_miss += 1
            continue
        if prev.get("best_stage") != rec["best_stage"]:
            n_stage += 1
        if prev.get("oracle_stream") != rec["oracle_stream"]:
            n_stream += 1
    print(
        f"[compare] n={len(records)} existing={len(old)} "
        f"missing_in_old={n_miss} stage_diff={n_stage} stream_diff={n_stream}",
        flush=True,
    )
    return 1 if n_miss or n_stage or n_stream else 0


def main() -> int:
    args = parse_args()
    vm_out = (args.vm_out or default_vm_out()).expanduser().resolve()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for s in splits:
        if s not in VALID_SPLITS:
            raise SystemExit(f"bad split {s}")
    copy_wavs = not args.no_copy and not args.compare_existing
    all_rec: list[dict[str, Any]] = []
    for split in splits:
        recs = export_split(vm_out, split, copy_wavs=copy_wavs)
        all_rec.extend(recs)
        print(f"[OK] {split} n={len(recs)} ok={sum(1 for r in recs if r.get('ok'))}", flush=True)
    if args.compare_existing:
        return compare_existing(vm_out, all_rec)
    collect = vm_out / "best_sep"
    ok_rec = [r for r in all_rec if r.get("ok")]
    fail_rec = [r for r in all_rec if not r.get("ok")]
    write_jsonl(collect / "index.jsonl", ok_rec)
    write_jsonl(collect / "failed.jsonl", fail_rec)
    (collect / "summary.json").write_text(
        json.dumps(
            {
                "n_records": len(ok_rec),
                "n_ok": len(ok_rec),
                "n_fail": len(fail_rec),
                "splits": splits,
                "dest_dir": str(collect),
                "index_excludes_failed": True,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    handoff = write_handoff(vm_out, all_rec, splits)
    n_fail = len(fail_rec)
    print(
        f"[OK] {collect / 'index.jsonl'} n_ok={len(ok_rec)}  "
        f"failed={collect / 'failed.jsonl'} n_fail={n_fail}  handoff={handoff}",
        flush=True,
    )
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

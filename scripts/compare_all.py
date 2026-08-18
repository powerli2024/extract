#!/usr/bin/env python3
"""按 split 汇总各阶段 summary，并写全局 reports/compare_all.json。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from paths import STAGE_DIRS, VALID_SPLITS, assert_split, default_vm_out, ensure_reports, stage_dir


def collect_split(vm_out: Path, split: str) -> dict:
    split = assert_split(split)
    out = {"split": split, "stages": {}}
    for key in STAGE_DIRS:
        sp = stage_dir(vm_out, key, split) / "summary.json"
        if sp.is_file():
            out["stages"][key] = json.loads(sp.read_text(encoding="utf-8"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vm-out", type=str, default="")
    ap.add_argument("--splits", type=str, default="pos,neg")
    args = ap.parse_args()
    vm_out = Path(args.vm_out) if args.vm_out else default_vm_out()
    splits = [assert_split(s.strip()) for s in args.splits.split(",") if s.strip()]
    if not splits:
        splits = list(VALID_SPLITS)

    by_split = {}
    for split in splits:
        by_split[split] = collect_split(vm_out, split)
        split_reports = ensure_reports(vm_out, split)
        p = split_reports / "compare_all.json"
        p.write_text(json.dumps(by_split[split], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] {split} → {p}")
        for k, v in by_split[split]["stages"].items():
            if "mean_oracle_cer" in v:
                print(f"  {k}: mean={v.get('mean_oracle_cer')} n_ok={v.get('n_ok')}")
            elif "by_thr" in v:
                print(f"  {k}: thr={v.get('thr')}")

    root = {
        "vm_out": str(vm_out.resolve()),
        "splits": by_split,
        "note": "pos/neg 分树；中间音频互不混写",
    }
    reports = ensure_reports(vm_out)
    path = reports / "compare_all.json"
    path.write_text(json.dumps(root, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] global → {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create an all-positive ASR arm from one CMD-SE condition/slot."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def select_export(row: dict[str, Any], condition: str, slot: str) -> dict[str, Any] | None:
    choices = (row.get("exported") or {}).get(condition) or []
    index = 0 if slot == "best" else 1
    return choices[index] if len(choices) > index else None


def main() -> int:
    p = argparse.ArgumentParser(description="CMD-SE 导出候选 -> all-positive ASR VE_OUT")
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--cmd-se-results", type=Path, required=True)
    p.add_argument("--condition", choices=["raw", "se48k"], required=True)
    p.add_argument("--slot", choices=["best", "better"], required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()
    samples = load_jsonl(args.samples)
    ranked_rows = load_jsonl(args.cmd_se_results)
    ranked = {str(r.get("uid")): r for r in ranked_rows if r.get("uid")}
    decisions: list[dict[str, Any]] = []
    errors: list[str] = []
    for sample in samples:
        uid = str(sample["uid"])
        row = ranked.get(uid) or {}
        choice = select_export(row, args.condition, args.slot)
        path = Path(str(choice.get("path"))) if choice else Path("")
        ok = row.get("status") == "ok" and choice is not None and path.is_file()
        if not ok:
            errors.append(uid)
        decisions.append({
            "uid": uid,
            "split": sample.get("split"),
            "label": sample.get("label"),
            "lang": sample.get("lang"),
            "cmd_text": sample.get("cmd_text"),
            "decision": "accept" if ok else "extract_error",
            "reject_decision": False,
            "extracted_wav": str(path.resolve()) if ok else None,
            "extracted_stream": choice.get("source_stream") if ok and choice else None,
            "cmd_se_condition": args.condition,
            "cmd_se_slot": args.slot,
            "cmd_se_similarity": choice.get("similarity") if ok and choice else None,
        })
    if args.strict and errors:
        raise SystemExit(f"CMD-SE candidate coverage failed n={len(errors)} first={errors[:10]}")
    (args.out_dir / "manifest").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.samples, args.out_dir / "manifest" / "samples.jsonl")
    (args.out_dir / "results" / "all_results.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in decisions), encoding="utf-8"
    )
    print(
        f"[OK] CMD-SE ASR arm condition={args.condition} slot={args.slot} "
        f"rows={len(decisions)} errors={len(errors)} -> {args.out_dir}"
    )
    return 2 if args.strict and errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

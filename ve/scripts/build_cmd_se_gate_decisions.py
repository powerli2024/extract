#!/usr/bin/env python3
"""Replace raw gate similarity streams with raw/SE48K CMD-SE similarities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def replace_gate_scores(
    base: list[dict[str, Any]], ranked_rows: list[dict[str, Any]], condition: str
) -> tuple[list[dict[str, Any]], list[str]]:
    ranked = {str(r.get("uid")): r for r in ranked_rows if r.get("uid")}
    key = "raw_similarity" if condition == "raw" else "se48k_similarity"
    output: list[dict[str, Any]] = []
    missing: list[str] = []
    for original in base:
        uid = str(original.get("uid"))
        scores = (ranked.get(uid) or {}).get(key) or {}
        if not scores:
            missing.append(uid)
        row = dict(original)
        row["sim_streams"] = {str(k): float(v) for k, v in scores.items()}
        row["presence_score"] = max(row["sim_streams"].values()) if scores else None
        row["cmd_se_gate_condition"] = condition
        output.append(row)
    return output, missing


def main() -> int:
    p = argparse.ArgumentParser(description="构造 CMD-SE condition 对应的门控分数行")
    p.add_argument("--base-decisions", type=Path, required=True)
    p.add_argument("--cmd-se-results", type=Path, required=True)
    p.add_argument("--condition", choices=["raw", "se48k"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()
    base = load_jsonl(args.base_decisions)
    ranked_rows = load_jsonl(args.cmd_se_results)
    output, missing = replace_gate_scores(base, ranked_rows, args.condition)
    if args.strict and missing:
        raise SystemExit(f"CMD-SE gate coverage failed n={len(missing)} first={missing[:10]}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in output), encoding="utf-8"
    )
    print(f"[OK] CMD-SE gate condition={args.condition} rows={len(output)} -> {args.out}")
    return 2 if args.strict and missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Apply a recommended gate config to cached similarity rows without labels at runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from optimize_gate_for_score import stream_score


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def apply_rows(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    thresholds = load_thresholds_from_data(config)
    policy = str(config.get("stream_policy") or "max")
    margins = (
        float(config.get("rescue_high_margin", .08)),
        float(config.get("rescue_floor_margin", .10)),
        float(config.get("rescue_dominance", .05)),
    )
    out = []
    for original in rows:
        row = dict(original)
        lang = str(row.get("lang") or "zh")
        value = stream_score(
            row, policy, rescue_high_margin=margins[0],
            rescue_floor_margin=margins[1], rescue_dominance=margins[2],
        )
        threshold = float(thresholds.get(lang, thresholds["default"]))
        reject = value < threshold
        row.update({
            "presence_score": value, "presence_thr": threshold,
            "stream_policy": policy, "decision": "reject" if reject else "accept",
            "reject_decision": reject,
            "reject_reason": "speaker_absent" if reject else "",
            "gate_overlay_source": "recommended_thr.json",
        })
        out.append(row)
    return out


def load_thresholds_from_data(data: dict[str, Any]) -> dict[str, float]:
    raw = data.get("thr_by_lang") or {"default": data.get("presence_thr")}
    result = {str(k): float(v) for k, v in raw.items() if v is not None}
    if "default" not in result:
        result["default"] = result.get("zh", next(iter(result.values())))
    return result


def main() -> int:
    p = argparse.ArgumentParser(description="将门控配置应用到已有 sim_streams")
    p.add_argument("--decisions", type=Path, required=True)
    p.add_argument("--thr-file", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()
    rows = load_jsonl(args.decisions)
    config = json.loads(args.thr_file.read_text(encoding="utf-8"))
    output = apply_rows(rows, config)
    missing = [str(r.get("uid")) for r in rows if not (r.get("sim_streams") or {})]
    if args.strict and missing:
        raise SystemExit(f"missing sim_streams: n={len(missing)} first={missing[:5]}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in output), encoding="utf-8"
    )
    print(f"[OK] applied gate rows={len(output)} policy={config.get('stream_policy')} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

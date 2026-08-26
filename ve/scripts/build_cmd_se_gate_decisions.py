#!/usr/bin/env python3
"""Build raw/SE48K and conditional mix-top CMD-SE gate scores."""

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
    output: list[dict[str, Any]] = []
    missing: list[str] = []
    for original in base:
        uid = str(original.get("uid"))
        ranked_row = ranked.get(uid) or {}
        row = dict(original)
        gate_meta: dict[str, Any] = {}
        if condition in {"raw", "se48k"}:
            key = "raw_similarity" if condition == "raw" else "se48k_similarity"
            scores = ranked_row.get(key) or {}
            valid = bool(scores)
            gate_scores = {str(k): float(v) for k, v in scores.items()}
            presence_score = max(gate_scores.values()) if gate_scores else None
        elif condition == "mix_top_se48k":
            raw = {str(k): float(v) for k, v in (ranked_row.get("raw_similarity") or {}).items()}
            se48k = {
                str(k): float(v)
                for k, v in (ranked_row.get("se48k_similarity") or {}).items()
            }
            spk = {k: v for k, v in raw.items() if k != "mix"}
            valid = "mix" in raw and bool(spk)
            presence_score = None
            if valid:
                # “mix 优于 spk1/spk2”使用严格大于；并列时不触发 SE，避免
                # 相等分数因字典顺序被当成 mix 胜出。
                best_spk, best_spk_score = max(spk.items(), key=lambda kv: (kv[1], kv[0]))
                mix_score = raw["mix"]
                use_se_mix = mix_score > best_spk_score
                selected_stream = "mix" if use_se_mix else best_spk
                if use_se_mix and "mix" not in se48k:
                    valid = False
                else:
                    presence_score = se48k["mix"] if use_se_mix else best_spk_score
                gate_meta = {
                    "raw_selected_stream": selected_stream,
                    "raw_mix_score": mix_score,
                    "raw_best_spk_stream": best_spk,
                    "raw_best_spk_score": best_spk_score,
                    "used_se48k_mix": use_se_mix,
                    "se48k_mix_score": se48k.get("mix"),
                    "effective_score": presence_score,
                }
            # 这是单一门控证据，不允许后续 max 再把三个 raw 流混进来。
            gate_scores = {"mix": float(presence_score)} if presence_score is not None else {}
        else:
            raise ValueError(f"unknown CMD-SE gate condition={condition}")
        if not valid:
            missing.append(uid)
        row["sim_streams"] = gate_scores
        row["presence_score"] = presence_score
        row["cmd_se_gate_condition"] = condition
        if gate_meta:
            row["cmd_se_gate_meta"] = gate_meta
        output.append(row)
    return output, missing


def main() -> int:
    p = argparse.ArgumentParser(description="构造 CMD-SE condition 对应的门控分数行")
    p.add_argument("--base-decisions", type=Path, required=True)
    p.add_argument("--cmd-se-results", type=Path, required=True)
    p.add_argument(
        "--condition", choices=["raw", "se48k", "mix_top_se48k"], required=True,
        help="mix_top_se48k: raw mix 严格胜过两分离流时改用 SE(mix) 相似度，否则用 raw 最佳分离流",
    )
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

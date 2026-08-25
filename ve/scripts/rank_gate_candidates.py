#!/usr/bin/env python3
"""Rank KWS enrollment/gate candidates by conservative repeated-holdout score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_candidate(spec: str) -> tuple[str, Path]:
    name, sep, raw = spec.partition("=")
    if not sep or not name.strip() or not raw.strip():
        raise ValueError(f"candidate must be NAME=DIR: {spec!r}")
    return name.strip(), Path(raw.strip()).resolve()


def main() -> int:
    p = argparse.ArgumentParser(description="汇总多组 KWS 的真实 CER 门控优化结果")
    p.add_argument("--candidate", action="append", required=True, help="NAME=优化报告目录")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for spec in args.candidate:
        name, root = parse_candidate(spec)
        path = root / "gate_score_optimization.json"
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{name}: {path}: {exc}")
            continue
        cov = report.get("coverage") or {}
        cov_ok = not cov.get("missing_asr") and not cov.get("duplicate_asr") and not cov.get("invalid_asr")
        recommended = report.get("recommendation") or {}
        for config, holdout in (report.get("holdout") or {}).items():
            distributions = holdout.get("distributions") or holdout
            policy = str(holdout.get("policy") or config.split("@", 1)[0])
            threshold_mode = str(holdout.get("threshold_mode") or "lang_split")
            score = distributions.get("contest_score") or {}
            delta = distributions.get("score_delta_vs_baseline") or {}
            stable = float(delta.get("p05", 0.0)) > 0.0
            rows.append({
                "kws": name,
                "policy": policy,
                "threshold_mode": threshold_mode,
                "configuration": config,
                "recommended": (
                    recommended.get("policy") == policy
                    and recommended.get("threshold_mode", "lang_split") == threshold_mode
                ),
                "eligible": bool(cov_ok and stable),
                "score_mean": score.get("mean"),
                "score_p05": score.get("p05"),
                "score_p95": score.get("p95"),
                "delta_mean": delta.get("mean"),
                "delta_p05": delta.get("p05"),
                "rr_mean": (distributions.get("rr") or {}).get("mean"),
                "cer_mean": (distributions.get("cer_pos_micro") or {}).get("mean"),
                "frr_mean": (distributions.get("frr") or {}).get("mean"),
                "thr_zh_p50": (distributions.get("thr_zh") or {}).get("p50"),
                "thr_zh_p05": (distributions.get("thr_zh") or {}).get("p05"),
                "thr_zh_p95": (distributions.get("thr_zh") or {}).get("p95"),
                "thr_en_p50": (distributions.get("thr_en") or {}).get("p50"),
                "report_dir": str(root),
            })

    rows.sort(key=lambda r: (
        int(r["eligible"]),
        float(r["score_p05"] if r["score_p05"] is not None else -1),
        float(r["score_mean"] if r["score_mean"] is not None else -1),
    ), reverse=True)
    eligible = [r for r in rows if r["eligible"]]
    shortlist = eligible[: max(1, args.top_k)]
    payload = {
        "ranking_contract": "coverage clean; paired delta p05>0; rank score p05 then mean",
        "warning": "screening rank is not final score; shortlisted arms must run strict downstream evaluation",
        "n_rows": len(rows),
        "n_eligible": len(eligible),
        "shortlist": shortlist,
        "rows": rows,
        "errors": errors,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "gate_ranking.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md = [
        "# KWS / Presence gate screening",
        "",
        "Only coverage-clean candidates with paired holdout delta p05 > 0 are eligible.",
        "",
        "| rank | KWS | policy | thr mode | eligible | score p05 | score mean | delta p05 | RR | CER | FRR | zh thr p50 [p05,p95] |",
        "|---:|---|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(rows, 1):
        fmt = lambda v: "-" if v is None else f"{float(v):.6f}"
        md.append(
            f"| {i} | {row['kws']} | {row['policy']} | {row['threshold_mode']} | {'Y' if row['eligible'] else 'N'} | "
            f"{fmt(row['score_p05'])} | {fmt(row['score_mean'])} | {fmt(row['delta_p05'])} | "
            f"{fmt(row['rr_mean'])} | {fmt(row['cer_mean'])} | {fmt(row['frr_mean'])} | "
            f"{fmt(row['thr_zh_p50'])} [{fmt(row['thr_zh_p05'])},{fmt(row['thr_zh_p95'])}] |"
        )
    md += ["", "## Shortlist", ""]
    md += [f"- {r['kws']} / {r['policy']} / {r['threshold_mode']}" for r in shortlist] or ["- No stable candidate; retain locked baseline."]
    if errors:
        md += ["", "## Errors", ""] + [f"- {e}" for e in errors]
    (args.out_dir / "gate_ranking.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"[OK] ranking -> {args.out_dir / 'gate_ranking.json'}")
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

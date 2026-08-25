#!/usr/bin/env python3
"""Paired bootstrap ranking for strict full downstream evaluation arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def parse_spec(spec: str) -> tuple[str, Path]:
    name, sep, raw = spec.partition("=")
    if not sep or not name or not raw:
        raise ValueError(f"candidate must be NAME=VE_OUT: {spec!r}")
    return name, Path(raw).resolve()


def score(rows: dict[str, dict[str, Any]], uids: list[str]) -> dict[str, float]:
    neg = [rows[u] for u in uids if rows[u]["split"] == "neg"]
    pos = [rows[u] for u in uids if rows[u]["split"] == "pos"]
    rr = sum(bool(r["rejected"]) for r in neg) / max(1, len(neg))
    errors = sum(int(r["errors"]) for r in pos)
    refs = sum(int(r["ref_chars"]) for r in pos)
    cer = errors / max(1, refs)
    return {"rr": rr, "cer": cer, "contest_score": 0.5 * (rr + 1.0 - cer)}


def dist(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {k: float(v) for k, v in {
        "mean": np.mean(x), "p05": np.quantile(x, .05),
        "p50": np.quantile(x, .50), "p95": np.quantile(x, .95),
    }.items()}


def main() -> int:
    p = argparse.ArgumentParser(description="严格 final_eval 候选的配对 bootstrap 排名")
    p.add_argument("--candidate", action="append", required=True, help="NAME=VE_OUT；第一项是基线")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--replicates", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260825)
    args = p.parse_args()

    arms: list[dict[str, Any]] = []
    base_samples: dict[str, dict[str, Any]] | None = None
    errors: list[str] = []
    for spec in args.candidate:
        name, root = parse_spec(spec)
        try:
            summary = load_json(root / "reports/final_eval/summary.json")
            final_rows = load_jsonl(root / "reports/final_eval/rows.jsonl")
            samples = load_jsonl(root / "manifest/samples.jsonl")
            sample_map = {str(r["uid"]): r for r in samples}
            row_map = {str(r["uid"]): r for r in final_rows}
            cov = summary.get("coverage") or {}
            if cov.get("errors") or cov.get("duplicate_decisions") or cov.get("duplicate_asr"):
                raise ValueError("strict coverage is not clean")
            if set(sample_map) != set(row_map):
                raise ValueError("manifest/final rows UID mismatch")
            if base_samples is None:
                base_samples = sample_map
            elif set(base_samples) != set(sample_map):
                raise ValueError("candidate UID set differs from baseline")
            arms.append({"name": name, "root": str(root), "summary": summary, "rows": row_map})
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    if not arms or base_samples is None:
        raise SystemExit("no eligible final candidate")

    strata: list[list[str]] = []
    for split in ("pos", "neg"):
        for lang in ("zh", "en"):
            bucket = [u for u, s in base_samples.items()
                      if str(s.get("split")) == split and str(s.get("lang") or "zh") == lang]
            if bucket:
                strata.append(bucket)
    rng = np.random.default_rng(args.seed)
    boot: dict[str, list[float]] = {a["name"]: [] for a in arms}
    delta: dict[str, list[float]] = {a["name"]: [] for a in arms}
    for _ in range(max(1, args.replicates)):
        sampled: list[str] = []
        for bucket in strata:
            sampled.extend(rng.choice(bucket, size=len(bucket), replace=True).tolist())
        scores = {a["name"]: score(a["rows"], sampled)["contest_score"] for a in arms}
        base = scores[arms[0]["name"]]
        for name, value in scores.items():
            boot[name].append(value)
            delta[name].append(value - base)

    ranked: list[dict[str, Any]] = []
    all_uids = list(base_samples)
    for idx, arm in enumerate(arms):
        overall = score(arm["rows"], all_uids)
        by_lang = {}
        for lang in ("zh", "en"):
            ids = [u for u, s in base_samples.items() if str(s.get("lang") or "zh") == lang]
            by_lang[lang] = score(arm["rows"], ids)
        d = dist(delta[arm["name"]])
        ranked.append({
            "name": arm["name"], "root": arm["root"], "baseline": idx == 0,
            "eligible_gain": idx == 0 or d["p05"] > 0.0,
            **overall, "bootstrap_score": dist(boot[arm["name"]]),
            "paired_delta": d, "by_lang": by_lang,
            "worst_lang_score": min(x["contest_score"] for x in by_lang.values()),
        })
    ranked.sort(key=lambda r: (
        int(r["eligible_gain"]), r["contest_score"], r["worst_lang_score"]
    ), reverse=True)
    payload = {
        "contract": "strict coverage; paired stratified bootstrap; candidate gain requires delta p05>0",
        "baseline": arms[0]["name"], "replicates": args.replicates,
        "ranking": ranked, "errors": errors,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "final_ranking.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Strict final candidate ranking", "",
        f"Baseline: `{arms[0]['name']}`; paired bootstrap replicates: {args.replicates}.", "",
        "| rank | arm | stable gain | RR | CER | score | delta p05 | score p05-p95 | zh score | en score |",
        "|---:|---|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(ranked, 1):
        b, d = row["bootstrap_score"], row["paired_delta"]
        lines.append(
            f"| {i} | {row['name']} | {'Y' if row['eligible_gain'] else 'N'} | "
            f"{row['rr']:.6f} | {row['cer']:.6f} | **{row['contest_score']:.6f}** | "
            f"{d['p05']:.6f} | {b['p05']:.6f}-{b['p95']:.6f} | "
            f"{row['by_lang']['zh']['contest_score']:.6f} | {row['by_lang']['en']['contest_score']:.6f} |"
        )
    if errors:
        lines += ["", "## Excluded", ""] + [f"- {x}" for x in errors]
    (args.out_dir / "final_ranking.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

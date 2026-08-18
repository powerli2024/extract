#!/usr/bin/env python3
"""CER 分布报告 + 默认阈值 a=P50,b=P75,c=P90。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return ys[f]
    return ys[f] + (ys[c] - ys[f]) * (k - f)


def hist_buckets(cers: list[float]) -> dict[str, int]:
    edges = [0.0, 0.1, 0.3, 0.5, 1.0, 1e9]
    labels = ["0-0.1", "0.1-0.3", "0.3-0.5", "0.5-1.0", ">=1.0"]
    counts = {lb: 0 for lb in labels}
    for c in cers:
        for i in range(len(edges) - 1):
            if edges[i] <= c < edges[i + 1] or (i == len(edges) - 2 and c >= edges[i]):
                counts[labels[i]] += 1
                break
    return counts


def compute_dist(cers: list[float], thr_override: list[float] | None = None) -> dict[str, Any]:
    n = len(cers)
    mean = sum(cers) / n if n else 0.0
    a = percentile(cers, 50)
    b = percentile(cers, 75)
    c = percentile(cers, 90)
    if thr_override and len(thr_override) == 3:
        a, b, c = thr_override
    return {
        "n": n,
        "mean": round(mean, 4),
        "p50": round(percentile(cers, 50), 4),
        "p75": round(percentile(cers, 75), 4),
        "p90": round(percentile(cers, 90), 4),
        "thr": {"a": round(a, 4), "b": round(b, 4), "c": round(c, 4)},
        "n_ge_a": sum(1 for x in cers if x >= a),
        "n_ge_b": sum(1 for x in cers if x >= b),
        "n_ge_c": sum(1 for x in cers if x >= c),
        "hist": hist_buckets(cers),
        "cer_eq_0_rate": round(sum(1 for x in cers if x == 0) / n, 4) if n else 0.0,
    }


def report_from_index(
    index_path: Path,
    out_json: Path,
    out_md: Path,
    thr_override: list[float] | None = None,
) -> dict[str, Any]:
    cers = []
    with index_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            if "oracle_cer" in o:
                cers.append(float(o["oracle_cer"]))
    dist = compute_dist(cers, thr_override)
    dist["index"] = str(index_path.resolve())
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(dist, ensure_ascii=False, indent=2), encoding="utf-8")
    thr = dist["thr"]
    md = [
        f"# CER 分布 — `{index_path.parent.name}`",
        "",
        f"- n={dist['n']} mean={dist['mean']}",
        f"- P50={dist['p50']} P75={dist['p75']} P90={dist['p90']}",
        f"- 默认阈值 a={thr['a']} b={thr['b']} c={thr['c']}",
        f"- n≥a={dist['n_ge_a']} n≥b={dist['n_ge_b']} n≥c={dist['n_ge_c']}",
        f"- CER==0 rate={dist['cer_eq_0_rate']}",
        "",
        "## hist",
        "",
    ]
    for k, v in dist["hist"].items():
        md.append(f"- {k}: {v}")
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[OK] CER dist → {out_json}")
    return dist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", default="")
    ap.add_argument("--thr", type=str, default="", help="override a,b,c")
    args = ap.parse_args()
    thr = None
    if args.thr.strip():
        parts = [float(x) for x in args.thr.split(",")]
        if len(parts) != 3:
            raise SystemExit("--thr 需要 a,b,c 三个数")
        thr = parts
    out_md = Path(args.out_md) if args.out_md else Path(args.out_json).with_suffix(".md")
    report_from_index(Path(args.index), Path(args.out_json), out_md, thr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""汇总 VP 打分细胞：RR / FRR / FAR / proxy（不跑 ASR）。

读取 score_encoders_on_sep 或 calibrate 产物：
  <root>/**/recommended_thr.json
  可选同目录 scores.jsonl — 若存在且 --holdout-frac>0 则重算 holdout 指标
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def metrics_at(rows: list[dict[str, Any]], thr: float) -> dict[str, float]:
    tp = fp = tn = fn = 0
    for r in rows:
        y = 1 if r.get("label") in ("present", "pos", 1, True) else 0
        s = float(r["presence_score"])
        d = 1 if s >= thr else 0
        if y == 1 and d == 1:
            tp += 1
        elif y == 0 and d == 1:
            fp += 1
        elif y == 0 and d == 0:
            tn += 1
        else:
            fn += 1
    n_pos, n_neg = tp + fn, tn + fp
    rr = tn / n_neg if n_neg else 0.0
    frr = fn / n_pos if n_pos else 0.0
    far = fp / n_neg if n_neg else 0.0
    return {
        "rr": rr,
        "frr": frr,
        "far": far,
        "proxy": 0.5 * rr + 0.5 * (1.0 - frr),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "fn": fn,
        "fp": fp,
    }


def load_scores(p: Path) -> list[dict[str, Any]]:
    rows = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def holdout_eval(rows: list[dict[str, Any]], thr: float, frac: float, seed: int) -> dict[str, Any] | None:
    if frac <= 0 or len(rows) < 20:
        return None
    rng = random.Random(seed)
    pos = [r for r in rows if r.get("label") in ("present", "pos", 1, True)]
    neg = [r for r in rows if r.get("label") not in ("present", "pos", 1, True)]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_hp = max(1, int(round(len(pos) * frac)))
    n_hn = max(1, int(round(len(neg) * frac)))
    ho = pos[:n_hp] + neg[:n_hn]
    return metrics_at(ho, thr)


def find_cells(root: Path) -> list[Path]:
    return sorted(root.rglob("recommended_thr.json"))


def main() -> int:
    ap = argparse.ArgumentParser(description="VP 细胞汇总")
    ap.add_argument("--root", type=Path, required=True, help="含各 cell 报告的根目录")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--holdout-seed", type=int, default=0)
    args = ap.parse_args()
    root = args.root.resolve()
    cells = find_cells(root)
    if not cells:
        raise SystemExit(f"未找到 recommended_thr.json: {root}")

    table: list[dict[str, Any]] = []
    for thr_path in cells:
        cell_dir = thr_path.parent
        thr_obj = json.loads(thr_path.read_text(encoding="utf-8"))
        rel = str(cell_dir.relative_to(root)).replace("\\", "/")
        row: dict[str, Any] = {
            "cell": rel,
            "encoder": thr_obj.get("encoder") or thr_obj.get("backend"),
            "arm": thr_obj.get("arm"),
            "enroll_vad": thr_obj.get("enroll_vad"),
            "thr": thr_obj.get("presence_thr"),
            "rr_file": thr_obj.get("rr"),
            "frr_file": thr_obj.get("frr"),
            "proxy_file": thr_obj.get("contest_score"),
            "n": thr_obj.get("n_scored"),
            "note": "file 指标多为全量 in-sample，偏乐观",
        }
        scores_p = cell_dir / "scores.jsonl"
        if scores_p.is_file() and args.holdout_frac > 0:
            rows = load_scores(scores_p)
            thr = float(thr_obj.get("presence_thr") or 0.0)
            ho = holdout_eval(rows, thr, args.holdout_frac, args.holdout_seed)
            if ho:
                row["holdout"] = ho
        table.append(row)

    table.sort(key=lambda r: (-float(r.get("holdout", {}).get("proxy") or r.get("proxy_file") or 0), r["cell"]))
    payload = {
        "root": str(root),
        "n_cells": len(table),
        "holdout_frac": args.holdout_frac,
        "metric": "proxy=0.5*RR+0.5*(1-FRR)；非 VE 竞赛分",
        "table": table,
        "best_by_holdout_or_file": table[0] if table else None,
    }
    out = args.out or (root / "reports" / "matrix.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = [
        "# VP 检测矩阵",
        "",
        f"- root: `{root}`",
        f"- cells: {len(table)}",
        f"- holdout_frac: {args.holdout_frac}（在**已选定的 file-thr** 上评估；不是重新选 τ）",
        "- matrix 的 file-thr 来自全量 in-sample，holdout 列偏乐观；live+HOLDOUT_FRAC 才是 calib 选 τ。",
        "",
        "> proxy 只用于 VP 内部排序，不是 VE `0.5*RR+0.5*(1-CER)`。",
        "",
        "| cell | encoder | arm | vad | thr | RR_file | FRR_file | proxy_file | holdout_proxy | holdout_RR | holdout_FRR |",
        "|------|---------|-----|-----|-----|---------|----------|------------|---------------|------------|-------------|",
    ]
    for r in table:
        ho = r.get("holdout") or {}
        md.append(
            f"| `{r['cell']}` | {r.get('encoder')} | {r.get('arm')} | {r.get('enroll_vad')} | "
            f"{r.get('thr')} | {r.get('rr_file')} | {r.get('frr_file')} | {r.get('proxy_file')} | "
            f"{ho.get('proxy', '')} | {ho.get('rr', '')} | {ho.get('frr', '')} |"
        )
    md_path = out.with_suffix(".md")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "md": str(md_path), "n": len(table)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

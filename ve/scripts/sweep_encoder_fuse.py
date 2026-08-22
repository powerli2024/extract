#!/usr/bin/env python3
"""对 encoder×arm scores 做 Presence 融合规则的精确离线最优扫描。

目标: max contest = 0.5*RR + 0.5*(1-FRR)，FRR 当 CER 代理。

规则族:
  single           — 单编码器 thr
  dual_or          — sp>=tp OR ss>=ts（联合扫 tp,ts）
  gray_rescue      — sp>=tp；或 (tp-m)<=sp<tp 且 ss>=ts（联合扫 tp,ts,m）
  dual_and         — sp>=tp AND ss>=ts
  weighted         — a*sp+(1-a)*ss >= thr（扫 a,thr；分数未标准化）

用法:
  python scripts/sweep_encoder_fuse.py \\
    --reports-dir /path/to/ve_encoder_cmp/reports \\
    --out /path/to/fuse_best.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


def load_cell(reports: Path, enc: str, arm: str) -> dict[str, tuple[float, int]]:
    """uid -> (score, y) where y=1 present."""
    p = reports / f"{enc}__{arm}" / "scores.jsonl"
    if not p.is_file():
        # 兼容 enc 短名 vs 带 _zh 后缀
        alts = list(reports.glob(f"{enc}*__{arm}/scores.jsonl"))
        if not alts:
            raise FileNotFoundError(p)
        p = alts[0]
    out: dict[str, tuple[float, int]] = {}
    with p.open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            y = 1 if o["label"] in ("present", "pos", 1, True) else 0
            out[str(o["uid"])] = (float(o["presence_score"]), y)
    return out


def list_cells(reports: Path) -> list[tuple[str, str]]:
    cells = []
    for d in sorted(reports.iterdir()):
        if not d.is_dir() or "__" not in d.name:
            continue
        if not (d / "scores.jsonl").is_file():
            continue
        enc, arm = d.name.split("__", 1)
        cells.append((enc, arm))
    return cells


def contest_from_accept(accept: np.ndarray, y: np.ndarray) -> dict[str, float]:
    yb = y.astype(bool)
    ab = accept.astype(bool)
    n_pos = int(yb.sum())
    n_neg = int((~yb).sum())
    tp = int((ab & yb).sum())
    fp = int((ab & ~yb).sum())
    tn = int((~ab & ~yb).sum())
    fn = int((~ab & yb).sum())
    rr = tn / n_neg if n_neg else 0.0
    frr = fn / n_pos if n_pos else 0.0
    return {
        "rr": rr,
        "frr": frr,
        "far": fp / n_neg if n_neg else 0.0,
        "contest": 0.5 * rr + 0.5 * (1.0 - frr),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


def thr_candidates(scores: np.ndarray, max_n: int = 400) -> np.ndarray:
    """分数点 + 均匀网格，去重后限幅。"""
    s = np.asarray(scores, dtype=np.float64)
    uniq = np.unique(np.round(s, 6))
    lo, hi = float(s.min()), float(s.max())
    if hi > lo:
        grid = np.round(np.linspace(lo, hi, 201), 6)
    else:
        grid = np.array([lo], dtype=np.float64)
    # 阈值略低于/等于每个分数点：用分数本身即可（>= thr）
    cands = np.unique(np.concatenate([uniq, grid]))
    if len(cands) > max_n:
        # 保留分位数 + 均匀子样
        qs = np.round(np.quantile(s, np.linspace(0, 1, max_n // 2)), 6)
        step = max(1, len(cands) // (max_n // 2))
        cands = np.unique(np.concatenate([qs, cands[::step]]))
    return cands


def best_single(sp: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for thr in thr_candidates(sp):
        m = contest_from_accept(sp >= thr, y)
        if best is None or m["contest"] > best["contest"] + 1e-15 or (
            abs(m["contest"] - best["contest"]) < 1e-15 and thr > best["thr"]
        ):
            best = {"rule": "single", "thr": float(thr), **m}
    assert best is not None
    return best


def best_dual_or(sp: np.ndarray, ss: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    tp_c = thr_candidates(sp, 350)
    ts_c = thr_candidates(ss, 350)
    for tp in tp_c:
        ap = sp >= tp
        for ts in ts_c:
            m = contest_from_accept(ap | (ss >= ts), y)
            if best is None or m["contest"] > best["contest"] + 1e-15:
                best = {
                    "rule": "dual_or",
                    "thr_primary": float(tp),
                    "thr_secondary": float(ts),
                    **m,
                }
    assert best is not None
    return best


def best_dual_and(sp: np.ndarray, ss: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for tp in thr_candidates(sp, 350):
        ap = sp >= tp
        for ts in thr_candidates(ss, 350):
            m = contest_from_accept(ap & (ss >= ts), y)
            if best is None or m["contest"] > best["contest"] + 1e-15:
                best = {
                    "rule": "dual_and",
                    "thr_primary": float(tp),
                    "thr_secondary": float(ts),
                    **m,
                }
    assert best is not None
    return best


def best_gray_rescue(
    sp: np.ndarray,
    ss: np.ndarray,
    y: np.ndarray,
    margins: np.ndarray | None = None,
) -> dict[str, Any]:
    """accept if sp>=tp OR ((tp-m)<=sp<tp AND ss>=ts)."""
    if margins is None:
        margins = np.round(np.arange(0.0, 0.301, 0.01), 6)
    best: dict[str, Any] | None = None
    tp_c = thr_candidates(sp, 300)
    ts_c = thr_candidates(ss, 250)
    for tp in tp_c:
        above = sp >= tp
        for mgn in margins:
            lo = tp - float(mgn)
            in_gray = (sp >= lo) & (sp < tp)
            if not in_gray.any() and mgn > 0:
                # 灰带为空时退化为 single@tp；仍评估
                pass
            for ts in ts_c:
                acc = above | (in_gray & (ss >= ts))
                met = contest_from_accept(acc, y)
                if best is None or met["contest"] > best["contest"] + 1e-15:
                    best = {
                        "rule": "gray_rescue",
                        "thr_primary": float(tp),
                        "thr_secondary": float(ts),
                        "margin": float(mgn),
                        **met,
                    }
    assert best is not None
    return best


def best_weighted(sp: np.ndarray, ss: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for a in np.round(np.linspace(0.0, 1.0, 21), 4):
        mix = a * sp + (1.0 - a) * ss
        for thr in thr_candidates(mix, 400):
            m = contest_from_accept(mix >= thr, y)
            if best is None or m["contest"] > best["contest"] + 1e-15:
                best = {
                    "rule": "weighted",
                    "alpha_primary": float(a),
                    "thr": float(thr),
                    **m,
                }
    assert best is not None
    return best


def align_pair(
    a: dict[str, tuple[float, int]], b: dict[str, tuple[float, int]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    uids = sorted(set(a) & set(b))
    sp = np.array([a[u][0] for u in uids], dtype=np.float64)
    ss = np.array([b[u][0] for u in uids], dtype=np.float64)
    y = np.array([a[u][1] for u in uids], dtype=np.int8)
    # 一致性检查
    for u in uids:
        if a[u][1] != b[u][1]:
            raise SystemExit(f"label mismatch uid={u}")
    return sp, ss, y


def default_fuse_reports_dir() -> Path:
    env = os.environ.get("VE_ENCODER_CMP_REPORTS", "").strip()
    if env:
        return Path(env).expanduser()
    cands = [
        Path("/root/autodl-tmp/ve_encoder_cmp/reports"),
        Path(r"D:\media\qqqqqqqqqqqqqq\ve_encoder_cmp_reports\ve_encoder_cmp\reports"),
    ]
    for c in cands:
        if c.is_dir():
            return c
    return cands[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="精确离线扫描 encoder 融合规则")
    p.add_argument(
        "--reports-dir",
        type=Path,
        default=default_fuse_reports_dir(),
        help="encoder×arm 的 reports 目录；也可用 VE_ENCODER_CMP_REPORTS",
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--arms",
        default="sep_once,sep_multi,no_sep",
        help="逗号分隔，只扫这些 arm",
    )
    p.add_argument(
        "--primaries",
        default="",
        help="主编码器过滤，空=全部",
    )
    p.add_argument(
        "--secondaries",
        default="",
        help="副编码器过滤，空=全部（不含 primary）",
    )
    p.add_argument("--skip-weighted", action="store_true")
    p.add_argument("--skip-and", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    reports = Path(args.reports_dir).resolve()
    if not reports.is_dir():
        raise SystemExit(
            f"reports-dir 不存在: {reports}\n"
            "请传 --reports-dir 或设置 VE_ENCODER_CMP_REPORTS"
            "（Linux/AutoDL 常见: /root/autodl-tmp/ve_encoder_cmp/reports）"
        )

    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    cells = [(e, a) for e, a in list_cells(reports) if a in arms]
    encs = sorted({e for e, _ in cells})
    primaries = (
        [x.strip() for x in args.primaries.split(",") if x.strip()]
        if args.primaries.strip()
        else encs
    )
    secondaries = (
        [x.strip() for x in args.secondaries.split(",") if x.strip()]
        if args.secondaries.strip()
        else encs
    )

    print(f"[INFO] reports={reports}")
    print(f"[INFO] encoders={encs} arms={arms}")

    results: list[dict[str, Any]] = []
    t0 = time.time()

    # --- singles ---
    for enc, arm in cells:
        data = load_cell(reports, enc, arm)
        uids = sorted(data)
        sp = np.array([data[u][0] for u in uids])
        y = np.array([data[u][1] for u in uids], dtype=np.int8)
        b = best_single(sp, y)
        b.update({"primary": enc, "secondary": None, "arm": arm, "n": len(uids)})
        results.append(b)
        print(
            f"[single] {enc}/{arm}: contest={b['contest']:.6f} "
            f"RR={b['rr']:.4f} FRR={b['frr']:.4f} thr={b['thr']}",
            flush=True,
        )

    # --- pairs ---
    for arm in arms:
        avail = [e for e in encs if (reports / f"{e}__{arm}" / "scores.jsonl").is_file()]
        # 也匹配可能的命名差异
        if len(avail) < 2:
            # glob
            avail = sorted({e for e, a in cells if a == arm})
        for p_enc, s_enc in combinations(avail, 2):
            # 双向：谁当 primary
            for primary, secondary in ((p_enc, s_enc), (s_enc, p_enc)):
                if primary not in primaries or secondary not in secondaries:
                    continue
                A = load_cell(reports, primary, arm)
                B = load_cell(reports, secondary, arm)
                sp, ss, y = align_pair(A, B)
                print(
                    f"\n=== pair {primary} + {secondary} @ {arm} n={len(y)} ===",
                    flush=True,
                )

                for name, fn in (
                    ("dual_or", best_dual_or),
                    ("gray_rescue", best_gray_rescue),
                ):
                    t1 = time.time()
                    b = fn(sp, ss, y)
                    b.update(
                        {
                            "primary": primary,
                            "secondary": secondary,
                            "arm": arm,
                            "n": int(len(y)),
                            "elapsed_sec": round(time.time() - t1, 2),
                        }
                    )
                    results.append(b)
                    print(
                        f"[{name}] contest={b['contest']:.6f} RR={b['rr']:.4f} "
                        f"FRR={b['frr']:.4f} params={ {k:b[k] for k in b if k.startswith('thr') or k=='margin'} } "
                        f"({b['elapsed_sec']}s)",
                        flush=True,
                    )

                if not args.skip_and:
                    t1 = time.time()
                    b = best_dual_and(sp, ss, y)
                    b.update(
                        {
                            "primary": primary,
                            "secondary": secondary,
                            "arm": arm,
                            "n": int(len(y)),
                            "elapsed_sec": round(time.time() - t1, 2),
                        }
                    )
                    results.append(b)
                    print(
                        f"[dual_and] contest={b['contest']:.6f} RR={b['rr']:.4f} "
                        f"FRR={b['frr']:.4f} ({b['elapsed_sec']}s)",
                        flush=True,
                    )

                if not args.skip_weighted:
                    t1 = time.time()
                    b = best_weighted(sp, ss, y)
                    b.update(
                        {
                            "primary": primary,
                            "secondary": secondary,
                            "arm": arm,
                            "n": int(len(y)),
                            "elapsed_sec": round(time.time() - t1, 2),
                        }
                    )
                    results.append(b)
                    print(
                        f"[weighted] contest={b['contest']:.6f} alpha={b['alpha_primary']} "
                        f"thr={b['thr']} ({b['elapsed_sec']}s)",
                        flush=True,
                    )

    results.sort(key=lambda r: (-r["contest"], r.get("frr", 1), r.get("far", 1)))
    best = results[0]
    payload = {
        "reports_dir": str(reports),
        "elapsed_sec": round(time.time() - t0, 2),
        "n_configs": len(results),
        "best": best,
        "top20": results[:20],
        "note": (
            "contest = 0.5*RR + 0.5*(1-FRR)；FRR 作 CER 代理。"
            "gray_rescue: accept if sp>=tp OR ((tp-m)<=sp<tp AND ss>=ts)。"
            "本扫基于当前 reports（未必含 enroll VAD）；VAD 后需重打分再扫。"
        ),
    }

    out = args.out or (reports / "fuse_best_offline.json")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = [
        "# Presence 融合离线精确扫描",
        "",
        f"- reports: `{reports}`",
        f"- configs evaluated: {len(results)}",
        f"- elapsed: {payload['elapsed_sec']}s",
        "",
        "## Best",
        "",
        "```json",
        json.dumps(best, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Top 20",
        "",
        "| rank | contest | RR | FRR | rule | primary | secondary | arm | params |",
        "|------|---------|----|-----|------|---------|-----------|-----|--------|",
    ]
    for i, r in enumerate(results[:20], 1):
        params = {
            k: r[k]
            for k in (
                "thr",
                "thr_primary",
                "thr_secondary",
                "margin",
                "alpha_primary",
            )
            if k in r
        }
        md.append(
            f"| {i} | {r['contest']:.6f} | {r['rr']:.4f} | {r['frr']:.4f} | "
            f"{r['rule']} | {r['primary']} | {r.get('secondary')} | {r['arm']} | `{params}` |"
        )
    md_path = out.with_suffix(".md")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\n======== BEST ========")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"[OK] {out}")
    print(f"[OK] {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

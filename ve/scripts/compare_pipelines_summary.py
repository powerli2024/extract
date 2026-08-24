#!/usr/bin/env python3
"""汇总各 PIPELINE 的 Presence / CER / contest 对照表。

用法（AutoDL）:
  python scripts/compare_pipelines_summary.py
  python scripts/compare_pipelines_summary.py --roots /root/autodl-tmp/ve_mix,/root/autodl-tmp/ve_ps4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOTS = [
    "/root/autodl-tmp/ve_mix",
    "/root/autodl-tmp/ve_ps4",
    "/root/autodl-tmp/ve_wesep",
    "/root/autodl-tmp/ve_sep_route",
    "/root/autodl-tmp/ve_adaptive_route",
    "/root/autodl-tmp/ve_cond_tasnet",
]


def _load_json(p: Path) -> dict[str, Any] | None:
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def row_for(root: Path) -> dict[str, Any]:
    name = root.name
    asr = _load_json(root / "reports" / "asr_cer" / "summary.json")
    ve = _load_json(root / "reports" / "summary.json")
    thr = _load_json(root / "reports" / "presence_calib" / "recommended_thr.json")
    final = _load_json(root / "reports" / "final_eval" / "summary.json")

    out: dict[str, Any] = {
        "pipeline": name.replace("ve_", "", 1) if name.startswith("ve_") else name,
        "root": str(root),
        "ok_asr": asr is not None,
        "ok_extract": ve is not None,
        "ok_final": final is not None,
    }
    if ve:
        ov = ve.get("overall") or {}
        sp = ve.get("splits") or {}
        pos = sp.get("pos") or {}
        neg = sp.get("neg") or {}
        meta = ve.get("meta") or {}
        out.update(
            {
                "tse": meta.get("tse_backend") or meta.get("pipeline"),
                "n": ov.get("n"),
                "n_accept": ov.get("n_accept"),
                "rr": neg.get("rr") if neg.get("rr") is not None else ov.get("rr"),
                "frr": pos.get("frr"),
                "far": neg.get("far"),
                "latency_p50_ms": (ov.get("latency_ms") or {}).get("p50"),
            }
        )
    if asr:
        out.update(
            {
                "cer_total": asr.get("cer_total"),
                "cer_accepted": asr.get("cer_accepted_mean"),
                "contest": asr.get("contest_score_new"),
                "rr_asr": asr.get("rr"),
                "n_pos_asr": asr.get("n_pos"),
                "n_accepted_asr": asr.get("n_accepted"),
                "by_lang": asr.get("by_lang"),
            }
        )
    if final:
        coverage = final.get("coverage") or {}
        out.update(
            {
                "rr_final": final.get("rr_neg"),
                "cer_final": final.get("cer_pos_micro"),
                "contest_final": final.get("contest_score"),
                "coverage_errors": len(coverage.get("errors") or [])
                + int(coverage.get("duplicate_decisions") or 0)
                + int(coverage.get("duplicate_asr") or 0),
            }
        )
        out["eligible"] = out["coverage_errors"] == 0
    elif thr and out.get("rr") is not None and out.get("frr") is not None:
        # 无 ASR 时用 FRR 代理 CER（与校准口径一致，仅供对照）
        rr = float(out["rr"])
        frr = float(out["frr"])
        out["contest_proxy_frr"] = round(0.5 * rr + 0.5 * (1.0 - frr), 6)
    return out


def fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--roots",
        default=",".join(DEFAULT_ROOTS),
        help="逗号分隔的 VE_OUT 根目录",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("/root/autodl-tmp/ve_compare/pipelines_summary.md"),
    )
    args = p.parse_args()
    roots = [Path(x.strip()) for x in args.roots.split(",") if x.strip()]
    rows = [row_for(r) for r in roots if r.is_dir()]

    lines = [
        "# VE PIPELINE 对照",
        "",
        "竞赛分: `0.5*RR + 0.5*(1-CER_total)`；误拒 pos 记 CER=1。",
        "Presence 各方案应接近（共享 thr）；差异主要来自提取质量 → CER。",
        "",
        "| pipeline | eligible | RR | FRR | FAR | CER_micro | CER_accept | **score** | coverage errors | n_accept | p50_ms |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ranked = sorted(
        rows,
        key=lambda r: (
            -(1 if r.get("eligible") else 0),
            -(r["contest_final"] if r.get("eligible") and r.get("contest_final") is not None else -1.0),
            r.get("cer_final") if r.get("cer_final") is not None else 9.0,
        ),
    )
    for r in ranked:
        lines.append(
            "| {pipe} | {eligible} | {rr} | {frr} | {far} | {cer} | {cera} | **{c}** | {cov} | {na} | {lat} |".format(
                pipe=r["pipeline"],
                eligible="Y" if r.get("eligible") else "N",
                rr=fmt(r.get("rr_final") if r.get("rr_final") is not None else r.get("rr")),
                frr=fmt(r.get("frr")),
                far=fmt(r.get("far")),
                cer=fmt(r.get("cer_final")),
                cera=fmt(r.get("cer_accepted")),
                c=fmt(r.get("contest_final")) if r.get("eligible") else "-",
                cov=fmt(r.get("coverage_errors"), 0),
                na=fmt(r.get("n_accept") or r.get("n_accepted_asr"), 0),
                lat=fmt(r.get("latency_p50_ms"), 1),
            )
        )

    lines += ["", "## 详情路径", ""]
    for r in ranked:
        lines.append(f"- `{r['pipeline']}`: `{r['root']}/reports/final_eval/summary.md`")
        bl = r.get("by_lang")
        if bl:
            lines.append(f"  - by_lang: `{json.dumps(bl, ensure_ascii=False)}`")

    missing = [str(r) for r in roots if not r.is_dir()]
    if missing:
        lines += ["", "## 缺失目录", ""] + [f"- `{m}`" for m in missing]

    no_asr = [r["pipeline"] for r in rows if not r.get("ok_asr")]
    if no_asr:
        lines += [
            "",
            "## 尚未跑 ASR",
            "",
            "```bash",
        ]
        for name in no_asr:
            lines.append(f"VE_OUT=/root/autodl-tmp/ve_{name} ./run_asr_cer.sh")
        lines.append("```")

    text = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)
    print(f"[OK] → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

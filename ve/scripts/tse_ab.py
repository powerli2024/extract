#!/usr/bin/env python3
"""Phase-2：在 present 子集上对 TSE 骨干做 A/B（默认 PS4 vs 记录位）。

当前仓库已接入的生产骨干仅为 PS4 BSRNN。本脚本：
  1) 对 force-extract 的 present 结果汇总质量参考分（若有 sim）
  2) 预留第二骨干接口；无权重时写出「待接入」报告，不阻断 Phase-1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paths import default_ve_out, ensure_dir, setup_sys_path

setup_sys_path()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize_backend(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    present_acc = [
        r
        for r in rows
        if r.get("label") == "present" and r.get("decision") == "accept"
    ]
    return {
        "backend": name,
        "n_present_accept": len(present_acc),
        "n_extract_error": sum(1 for r in rows if r.get("decision") == "extract_error"),
        "note": "质量指标（SI-SDR/DNSMOS）需额外参考音频；此处仅统计可提取性",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TSE A/B 报告（Phase-2）")
    p.add_argument("--ve-out", type=Path, default=None)
    p.add_argument(
        "--primary-results",
        type=Path,
        default=None,
        help="主骨干 all_results.jsonl（默认 ve_out/results）",
    )
    p.add_argument(
        "--secondary-results",
        type=Path,
        default=None,
        help="第二骨干结果；缺省则标记 pending",
    )
    p.add_argument("--secondary-name", default="usef_tse_or_tfgridnet")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ve_out = (args.ve_out or default_ve_out()).resolve()
    out_dir = ensure_dir(ve_out / "reports" / "tse_ab")
    primary = args.primary_results or (ve_out / "results" / "all_results.jsonl")
    rows_a = load_jsonl(primary)
    rep: dict[str, Any] = {
        "phase": 2,
        "status": "partial",
        "primary": summarize_backend(rows_a, "ps4_bsrnn"),
        "secondary": None,
        "recommendation": (
            "Phase-1 继续使用 PS4 BSRNN。"
            "待 USEF-TSE / TF-GridNet 公开可复现权重后，"
            "用 --force-extract 在 present 子集跑第二骨干，"
            "将结果 jsonl 传入 --secondary-results 再生成对比。"
        ),
        "how_to_plug_second_backend": {
            "steps": [
                "实现 scripts/tse_<name>.py，接口与 PS4Extractor.extract 一致",
                "run_extract.py --tse-backend <name> --force-extract --splits pos",
                "python scripts/tse_ab.py --secondary-results <path>",
            ]
        },
    }
    if args.secondary_results and args.secondary_results.is_file():
        rows_b = load_jsonl(args.secondary_results)
        rep["secondary"] = summarize_backend(rows_b, args.secondary_name)
        rep["status"] = "compared"
    else:
        rep["secondary"] = {
            "backend": args.secondary_name,
            "status": "pending_no_weights",
        }

    (out_dir / "ab_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md = [
        "# TSE A/B（Phase-2）",
        "",
        f"- status: **{rep['status']}**",
        f"- primary: `{json.dumps(rep['primary'], ensure_ascii=False)}`",
        f"- secondary: `{json.dumps(rep['secondary'], ensure_ascii=False)}`",
        "",
        rep["recommendation"],
        "",
    ]
    (out_dir / "ab_report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"[OK] {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""统一的最终离线计分器。

线上推理只输出 decision / ASR 假设；本文件才读取 pos/neg 标签和参考 CMD 文本。
正式 CER 采用官方定义：总 (S+I+D) / 总参考字符数；被拒或处理失败的正样本
按该条 errors=N（即单条 CER=1）计。宏平均同时保留，仅供诊断。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from asr_cer import normalize_for_cer
from paths import default_ve_out, ensure_dir


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def decision_is_reject(row: dict[str, Any]) -> bool:
    if "reject" in row:
        return bool(row["reject"])
    return str(row.get("final_decision") or row.get("decision") or "").startswith("reject") or bool(row.get("reject_decision"))


def asr_errors(row: dict[str, Any] | None, ref: str) -> tuple[int, int, float, str]:
    n_ref = len(ref)
    if not row or row.get("status") != "ok":
        return n_ref, n_ref, 1.0, "asr_missing_or_error"
    n = int(row.get("n") or n_ref)
    if n <= 0:
        return 0, 0, 0.0, "ok_empty_ref"
    dist = row.get("edit_distance")
    if dist is None:
        return n, n, 1.0, "asr_missing_distance"
    err = int(dist)
    return err, n, err / n, "ok"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VE 最终统一评测（标签仅在此处使用）")
    p.add_argument("--ve-out", type=Path, default=None)
    p.add_argument("--samples", type=Path, default=None)
    p.add_argument("--decisions", type=Path, default=None,
                   help="决策 jsonl；默认 results/all_results.jsonl。overlay 必须显式传入，避免复用旧结果。")
    p.add_argument("--asr", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--strict", action="store_true", help="UID 覆盖或 ASR 不完整时返回非零")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ve = (args.ve_out or default_ve_out()).resolve()
    samples_path = (args.samples or ve / "manifest" / "samples.jsonl").resolve()
    decisions_path = (
        args.decisions.resolve()
        if args.decisions
        else ve / "results" / "all_results.jsonl"
    )
    asr_path = (args.asr or ve / "reports" / "asr_cer" / "asr_results.jsonl").resolve()
    out_dir = ensure_dir((args.out_dir or ve / "reports" / "final_eval").resolve())

    samples = load_jsonl(samples_path)
    decision_rows = load_jsonl(decisions_path)
    asr_rows = load_jsonl(asr_path)
    if not samples:
        raise SystemExit(f"空或不存在 samples: {samples_path}")
    dec = {str(r.get("uid")): r for r in decision_rows if r.get("uid")}
    asr = {str(r.get("uid")): r for r in asr_rows if r.get("uid")}
    duplicate_dec = len(decision_rows) - len(dec)
    duplicate_asr = len(asr_rows) - len(asr)

    rows: list[dict[str, Any]] = []
    n_pos = n_neg = n_rej_neg = 0
    errors = chars = 0
    macro_cers: list[float] = []
    coverage_errors: list[str] = []
    for sample in samples:
        uid = str(sample["uid"])
        split = str(sample.get("split"))
        drow = dec.get(uid)
        missing_decision = drow is None
        if drow is None:
            coverage_errors.append(f"missing_decision:{uid}")
            rejected = False  # fail closed for RR; pos will obtain CER=1 below
        else:
            rejected = decision_is_reject(drow)
        item: dict[str, Any] = {"uid": uid, "split": split, "rejected": rejected}
        if split == "pos":
            n_pos += 1
            ref = normalize_for_cer(sample.get("cmd_text"))
            if rejected or missing_decision:
                err, n, cer, status = len(ref), len(ref), 1.0, "rejected"
            else:
                err, n, cer, status = asr_errors(asr.get(uid), ref)
                if status != "ok" and status != "ok_empty_ref":
                    coverage_errors.append(f"{status}:{uid}")
            errors += err
            chars += n
            macro_cers.append(cer)
            item.update({"status": status, "errors": err, "ref_chars": n, "cer": cer})
        elif split == "neg":
            n_neg += 1
            n_rej_neg += int(rejected)
        else:
            coverage_errors.append(f"unknown_split:{uid}:{split}")
        rows.append(item)

    extra_decisions = sorted(set(dec) - {str(s["uid"]) for s in samples})
    if extra_decisions:
        coverage_errors.extend(f"extra_decision:{uid}" for uid in extra_decisions)
    cer_micro = errors / chars if chars else 0.0
    cer_macro = sum(macro_cers) / len(macro_cers) if macro_cers else 0.0
    rr = n_rej_neg / n_neg if n_neg else 0.0
    summary = {
        "metric_version": "official_character_weighted_v1",
        "metric": "score=0.5*RR_neg + 0.5*(1-CER_pos_micro); rejected pos has errors=N",
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_neg_rejected": n_rej_neg,
        "rr_neg": rr,
        "cer_pos_micro": cer_micro,
        "cer_pos_macro_diagnostic": cer_macro,
        "pos_total_errors": errors,
        "pos_total_ref_chars": chars,
        "contest_score": 0.5 * rr + 0.5 * (1.0 - cer_micro),
        "paths": {"samples": str(samples_path), "decisions": str(decisions_path), "asr": str(asr_path)},
        "coverage": {
            "n_samples": len(samples), "n_decision_rows": len(decision_rows), "n_asr_rows": len(asr_rows),
            "duplicate_decisions": duplicate_dec, "duplicate_asr": duplicate_asr,
            "errors": coverage_errors,
        },
    }
    (out_dir / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text(
        "# VE final evaluation\n\n"
        f"- **CER_pos_micro** = {cer_micro:.6f} ({errors}/{chars})\n"
        f"- CER_pos_macro (diagnostic) = {cer_macro:.6f}\n"
        f"- **RR_neg** = {rr:.6f} ({n_rej_neg}/{n_neg})\n"
        f"- **contest_score** = {summary['contest_score']:.6f}\n"
        f"- coverage errors = {len(coverage_errors)}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.strict and (coverage_errors or duplicate_dec or duplicate_asr):
        print("[ERR] strict final evaluation coverage failed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

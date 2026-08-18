#!/usr/bin/env python3
"""过 ASR 后叠：长句非任务加拒 ∪ camp/窗否决。只加拒。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from lift_common import camp_veto, contest_metrics, extra_reject_text, window_veto
from paths import default_ve_out, ensure_dir


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="文本加拒 + camp/窗否决（只加拒）")
    p.add_argument("--ve-out", type=Path, default=None)
    p.add_argument("--asr-pos", type=Path, default=None, help="asr_results.jsonl")
    p.add_argument("--neg", type=Path, default=None)
    p.add_argument("--veto-margin", type=float, default=0.12)
    p.add_argument("--no-text", action="store_true")
    p.add_argument("--no-camp", action="store_true")
    p.add_argument("--window-veto", action="store_true")
    args = p.parse_args()

    ve = (args.ve_out or default_ve_out()).resolve()
    asr_pos = args.asr_pos or (ve / "reports" / "asr_cer" / "asr_results.jsonl")
    neg_path = args.neg or (ve / "results" / "neg_results.jsonl")
    pos = load_jsonl(asr_pos) if asr_pos.is_file() else []
    neg = load_jsonl(neg_path) if neg_path.is_file() else []

    def pack(r: dict[str, Any], split: str) -> dict[str, Any]:
        score = float(r.get("presence_score") or r.get("eres") or 0.0)
        thr = float(r.get("presence_thr") or r.get("thr") or 0.0)
        hyp = r.get("asr_text") or r.get("hyp") or ""
        camp = r.get("veto_score")
        if camp is None:
            camp = r.get("camp")
        bw = (r.get("best_window") or {}).get("score") if isinstance(r.get("best_window"), dict) else r.get("best_window_score")
        sw = (r.get("second_window") or {}).get("score") if isinstance(r.get("second_window"), dict) else r.get("second_window_score")
        base = str(r.get("decision") or "").startswith("reject") or bool(r.get("reject_decision"))
        return {
            "uid": r.get("uid"),
            "split": split,
            "cer": r.get("cer"),
            "score": score,
            "thr": thr,
            "hyp": hyp,
            "camp": None if camp is None else float(camp),
            "best_window_score": bw,
            "second_window_score": sw,
            "base_rej": base or (score < thr),
        }

    rows = [pack(r, "pos") for r in pos] + [pack(r, "neg") for r in neg]

    def pred(r: dict[str, Any]) -> bool:
        if r["base_rej"]:
            return True
        if (not args.no_text) and extra_reject_text(r["score"], r["thr"], r.get("hyp")):
            return True
        if (not args.no_camp) and camp_veto(r["score"], r.get("camp"), r["thr"], margin=args.veto_margin):
            return True
        if args.window_veto and window_veto(
            r.get("best_window_score"), r.get("second_window_score"), r["thr"],
            margin=args.veto_margin,
        ):
            return True
        return False

    m0 = contest_metrics(rows, lambda r: r["base_rej"])
    m1 = contest_metrics(rows, pred)
    n_extra_pos = sum(1 for r in rows if r["split"] == "pos" and (not r["base_rej"]) and pred(r))
    n_extra_neg = sum(1 for r in rows if r["split"] == "neg" and (not r["base_rej"]) and pred(r))
    out = {
        "baseline": m0,
        "overlay": m1,
        "d_contest": round(m1["contest"] - m0["contest"], 6),
        "n_extra_pos": n_extra_pos,
        "n_extra_neg": n_extra_neg,
        "go": bool((m1["contest"] - m0["contest"]) >= 0.005 and n_extra_pos <= 5),
    }
    od = ensure_dir(ve / "reports" / "lift_overlay")
    (od / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[OK] {od / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

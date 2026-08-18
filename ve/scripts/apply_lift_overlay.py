#!/usr/bin/env python3
"""过 ASR 后叠：长句非任务加拒 ∪ camp/窗否决。只加拒。

--locked-thr 用冻结 τ（zh 0.29305 / en 0.357868，按唤醒词语言）重判余弦门，
再叠文本加拒。新放行的原误拒 pos 若从未 ASR，保守记 CER=1，并报 n_need_asr。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from lift_common import (
    EN_THR,
    ZH_THR,
    camp_veto,
    contest_metrics,
    extra_reject_text,
    round_metrics,
    thr_of_lang,
    window_veto,
)
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
    p.add_argument("--asr-neg", type=Path, default=None, help="neg FA 的 asr_results.jsonl（补 hyp）")
    p.add_argument("--neg", type=Path, default=None)
    p.add_argument("--samples", type=Path, default=None)
    p.add_argument("--tag", default=None, help="写出 reports/lift_overlay/<tag>.json")
    p.add_argument("--veto-margin", type=float, default=0.12)
    p.add_argument("--locked-thr", action="store_true", help="用冻结 zh/en τ 替代提取文件里的 thr")
    p.add_argument("--no-text", action="store_true")
    p.add_argument("--no-camp", action="store_true")
    p.add_argument("--window-veto", action="store_true")
    args = p.parse_args()

    ve = (args.ve_out or default_ve_out()).resolve()
    asr_pos = args.asr_pos or (ve / "reports" / "asr_cer" / "asr_results.jsonl")
    neg_path = args.neg or (ve / "results" / "neg_results.jsonl")
    samples_path = args.samples or (ve / "manifest" / "samples.jsonl")
    pos = load_jsonl(asr_pos) if asr_pos.is_file() else []
    neg = load_jsonl(neg_path) if neg_path.is_file() else []
    neg_hyp = {}
    asr_neg = args.asr_neg or (ve / "reports" / "asr_neg_fa" / "asr_results.jsonl")
    if asr_neg.is_file():
        for r in load_jsonl(asr_neg):
            uid = r.get("uid")
            if uid:
                neg_hyp[uid] = r.get("asr_text") or r.get("hyp") or ""
    lang_of = {}
    if samples_path.is_file():
        for s in load_jsonl(samples_path):
            lang_of[s.get("uid")] = s.get("lang") or "zh"

    def pack(r: dict[str, Any], split: str) -> dict[str, Any]:
        score = float(r.get("presence_score") or r.get("eres") or 0.0)
        lang = r.get("lang") or lang_of.get(r.get("uid")) or "zh"
        file_thr = float(r.get("presence_thr") or r.get("thr") or 0.0)
        thr = thr_of_lang(lang) if args.locked_thr else file_thr
        hyp = r.get("asr_text") or r.get("hyp") or ""
        if split == "neg" and r.get("uid") in neg_hyp:
            hyp = neg_hyp[r.get("uid")] or hyp
        camp = r.get("veto_score")
        if camp is None:
            camp = r.get("camp")
        bw = (r.get("best_window") or {}).get("score") if isinstance(r.get("best_window"), dict) else r.get("best_window_score")
        sw = (r.get("second_window") or {}).get("score") if isinstance(r.get("second_window"), dict) else r.get("second_window_score")
        file_rej = str(r.get("decision") or "").startswith("reject") or bool(r.get("reject_decision"))
        cosine_rej = score < thr
        orig_accept = (not file_rej) and str(r.get("decision") or "") == "accept"
        if split == "pos" and not orig_accept:
            # 原误拒没有真实 ASR
            cer = 1.0
        else:
            cer = r.get("cer")
        return {
            "uid": r.get("uid"),
            "split": split,
            "lang": lang,
            "cer": cer,
            "score": score,
            "thr": thr,
            "file_thr": file_thr,
            "hyp": hyp,
            "camp": None if camp is None else float(camp),
            "best_window_score": bw,
            "second_window_score": sw,
            "file_rej": file_rej,
            "base_rej": cosine_rej,
            "orig_accept": orig_accept,
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

    m_file = contest_metrics(rows, lambda r: r["file_rej"])
    m_cos = contest_metrics(rows, lambda r: r["base_rej"])
    m1 = contest_metrics(rows, pred)
    n_extra_pos = sum(1 for r in rows if r["split"] == "pos" and (not r["base_rej"]) and pred(r))
    n_extra_neg = sum(1 for r in rows if r["split"] == "neg" and (not r["base_rej"]) and pred(r))
    n_need_asr = sum(
        1 for r in rows
        if r["split"] == "pos" and (not r["base_rej"]) and (not r["orig_accept"])
    )
    tag = args.tag or ("locked_text" if args.locked_thr else "holdout_text")
    out = {
        "tag": tag,
        "locked_thr": bool(args.locked_thr),
        "zh_thr": ZH_THR,
        "en_thr": EN_THR,
        "file_extract": round_metrics(m_file),
        "cosine_only": round_metrics(m_cos),
        "overlay": round_metrics(m1),
        "d_contest_vs_file": round(m1["contest"] - m_file["contest"], 6),
        "d_contest_vs_cosine": round(m1["contest"] - m_cos["contest"], 6),
        "n_extra_pos": n_extra_pos,
        "n_extra_neg": n_extra_neg,
        "n_need_asr": n_need_asr,
        "note": "n_need_asr>0 时新放行 pos 仍按 CER=1 保守估计",
        "go_vs_file": bool((m1["contest"] - m_file["contest"]) >= 0.005 and n_extra_pos <= 5),
    }
    od = ensure_dir(ve / "reports" / "lift_overlay")
    jp = od / f"{tag}.json"
    jp.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [
        f"# lift overlay `{tag}`",
        "",
        f"- locked_thr={args.locked_thr} zh={ZH_THR} en={EN_THR}",
        f"- file contest={out['file_extract']['contest']} RR={out['file_extract']['rr']}",
        f"- cosine contest={out['cosine_only']['contest']} RR={out['cosine_only']['rr']}",
        f"- overlay contest={out['overlay']['contest']} RR={out['overlay']['rr']} CER={out['overlay']['cer']}",
        f"- Δ vs file={out['d_contest_vs_file']}  extra_pos={n_extra_pos} extra_neg={n_extra_neg} need_asr={n_need_asr}",
        "",
    ]
    (od / f"{tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[OK] {jp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

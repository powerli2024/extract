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
from asr_cer import normalize_for_cer


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _row_id(r: dict[str, Any]) -> int | None:
    if r.get("id") is not None:
        try:
            return int(r["id"])
        except (TypeError, ValueError):
            return None
    uid = str(r.get("uid") or "")
    if "_" in uid:
        tail = uid.rsplit("_", 1)[-1]
        if tail.isdigit():
            return int(tail)
    return None


def overlay_reason(
    r: dict[str, Any],
    *,
    use_text: bool,
    use_camp: bool,
    use_window: bool,
    veto_margin: float,
) -> str:
    if r.get("base_rej"):
        return "speaker_absent"
    if use_text and extra_reject_text(r["score"], r["thr"], r.get("hyp")):
        return "len_and_nontask_gray"
    if use_camp and camp_veto(r["score"], r.get("camp"), r["thr"], margin=veto_margin):
        return "camp_veto"
    if use_window and window_veto(
        r.get("best_window_score"), r.get("second_window_score"), r["thr"],
        margin=veto_margin,
    ):
        return "window_veto"
    return "speaker_absent"


def extra_reject_tag(*, use_text: bool, use_camp: bool, use_window: bool) -> str:
    parts: list[str] = []
    if use_text:
        parts.append("len_and_nontask_gray")
    if use_camp:
        parts.append("camp_veto")
    if use_window:
        parts.append("window_veto")
    return "+".join(parts) if parts else "none"


def write_result_json(
    path: Path,
    rows: list[dict[str, Any]],
    pred_fn,
    metrics: dict[str, Any],
    *,
    extra_reject: str = "len_and_nontask_gray",
) -> None:
    """竞赛 pos 列表：拒识 content 空、cer=1；汇总使用官方字符加权 CER。"""
    pos = [r for r in rows if r.get("split") == "pos"]
    pos.sort(key=lambda r: (_row_id(r) is None, _row_id(r) or 0, str(r.get("uid") or "")))
    results = []
    for r in pos:
        rid = _row_id(r)
        if rid is None:
            continue
        label = str(r.get("label") or r.get("cmd_text") or "")
        if pred_fn(r):
            results.append({"id": rid, "content": "", "label": label, "cer": 1.0})
        else:
            cer = r.get("cer")
            results.append({
                "id": rid,
                "content": str(r.get("hyp") or ""),
                "label": label,
                "cer": 1.0 if cer is None else float(cer),
            })
    payload = {
        "results": results,
        "avg_cer": float(metrics["cer"]),
        "avg_rr": float(metrics["rr"]),
        "contest": float(metrics["contest"]),
        "zh_thr": ZH_THR,
        "en_thr": EN_THR,
        "extra_reject": extra_reject,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def official_metrics(rows: list[dict[str, Any]], pred_fn) -> dict[str, float | int]:
    """官方 CER：总(S+I+D)/总参考字符数；拒绝/失败正样本 errors=N。"""
    errors = chars = n_pos = n_neg = n_rej_neg = 0
    for r in rows:
        if r.get("split") == "neg":
            n_neg += 1
            n_rej_neg += int(bool(pred_fn(r)))
            continue
        if r.get("split") != "pos":
            continue
        n_pos += 1
        ref = normalize_for_cer(r.get("cmd_text") or r.get("label"))
        n_ref = len(ref)
        if pred_fn(r) or r.get("asr_status") != "ok":
            errors += n_ref
            chars += n_ref
            continue
        n = int(r.get("ref_chars") or n_ref)
        dist = r.get("edit_distance")
        if n <= 0:
            continue
        if dist is None:
            errors += n
        else:
            errors += int(dist)
        chars += n
    cer = errors / chars if chars else 0.0
    rr = n_rej_neg / n_neg if n_neg else 0.0
    return {
        "cer": cer,
        "rr": rr,
        "contest": 0.5 * rr + 0.5 * (1.0 - cer),
        "total_errors": errors,
        "total_ref_chars": chars,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_rej_neg": n_rej_neg,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="文本加拒 + camp/窗否决（只加拒）")
    p.add_argument("--ve-out", type=Path, default=None)
    p.add_argument("--asr-pos", type=Path, default=None, help="asr_results.jsonl")
    p.add_argument("--asr-neg", type=Path, default=None, help="neg FA 的 asr_results.jsonl（补 hyp）")
    p.add_argument("--no-neg-asr", action="store_true", help="本轮不读取任何历史 neg ASR 结果")
    p.add_argument("--neg", type=Path, default=None)
    p.add_argument("--samples", type=Path, default=None)
    p.add_argument("--tag", default=None, help="写出 reports/lift_overlay/<tag>.json")
    p.add_argument("--veto-margin", type=float, default=0.12)
    p.add_argument("--locked-thr", action="store_true", help="用冻结 zh/en τ 替代提取文件里的 thr")
    p.add_argument("--no-text", action="store_true")
    p.add_argument("--no-camp", action="store_true")
    p.add_argument("--window-veto", action="store_true")
    p.add_argument("--write-result", action="store_true",
                   help="写出 reports/submit/result.json（竞赛 pos 列表）")
    p.add_argument("--result-json", type=Path, default=None)
    args = p.parse_args()

    ve = (args.ve_out or default_ve_out()).resolve()
    asr_pos = args.asr_pos or (ve / "reports" / "asr_cer" / "asr_results.jsonl")
    neg_path = args.neg or (ve / "results" / "neg_results.jsonl")
    samples_path = args.samples or (ve / "manifest" / "samples.jsonl")
    pos = load_jsonl(asr_pos) if asr_pos.is_file() else []
    neg = load_jsonl(neg_path) if neg_path.is_file() else []
    neg_hyp = {}
    asr_neg = args.asr_neg or (ve / "reports" / "asr_neg_fa" / "asr_results.jsonl")
    if (not args.no_neg_asr) and asr_neg.is_file():
        for r in load_jsonl(asr_neg):
            uid = r.get("uid")
            if uid:
                neg_hyp[uid] = r.get("asr_text") or r.get("hyp") or ""
    lang_of = {}
    sample_meta: dict[str, dict[str, Any]] = {}
    if samples_path.is_file():
        for s in load_jsonl(samples_path):
            uid = s.get("uid")
            if not uid:
                continue
            lang_of[uid] = s.get("lang") or "zh"
            sample_meta[uid] = s

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
        sm = sample_meta.get(r.get("uid")) or {}
        return {
            "uid": r.get("uid"),
            "id": r.get("id") if r.get("id") is not None else sm.get("id"),
            "split": split,
            "lang": lang,
            "cer": cer,
            "asr_status": r.get("status"),
            "edit_distance": r.get("edit_distance"),
            "ref_chars": r.get("n"),
            "score": score,
            "thr": thr,
            "file_thr": file_thr,
            "hyp": hyp,
            "label": r.get("cmd_text") or r.get("label") or sm.get("cmd_text") or "",
            "cmd_text": r.get("cmd_text") or sm.get("cmd_text") or "",
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
    m_file_official = official_metrics(rows, lambda r: r["file_rej"])
    m_cos_official = official_metrics(rows, lambda r: r["base_rej"])
    m1_official = official_metrics(rows, pred)
    n_extra_pos = sum(1 for r in rows if r["split"] == "pos" and (not r["base_rej"]) and pred(r))
    n_extra_neg = sum(1 for r in rows if r["split"] == "neg" and (not r["base_rej"]) and pred(r))
    # overlay 仍拒的原误拒 pos 不需要 ASR；只统计 overlay 新放行且从未接受过的 pos
    n_need_asr = sum(
        1 for r in rows
        if r["split"] == "pos" and (not pred(r)) and (not r["orig_accept"])
    )
    extra_tag = extra_reject_tag(
        use_text=not args.no_text,
        use_camp=not args.no_camp,
        use_window=bool(args.window_veto),
    )
    tag = args.tag or ("locked_text" if args.locked_thr else "holdout_text")
    out = {
        "tag": tag,
        "locked_thr": bool(args.locked_thr),
        "zh_thr": ZH_THR,
        "en_thr": EN_THR,
        "file_extract_macro_diagnostic": round_metrics(m_file),
        "cosine_only_macro_diagnostic": round_metrics(m_cos),
        "overlay_macro_diagnostic": round_metrics(m1),
        "file_extract_official": round_metrics(m_file_official),
        "cosine_only_official": round_metrics(m_cos_official),
        "overlay_official": round_metrics(m1_official),
        "d_contest_vs_file": round(float(m1_official["contest"]) - float(m_file_official["contest"]), 6),
        "d_contest_vs_cosine": round(float(m1_official["contest"]) - float(m_cos_official["contest"]), 6),
        "n_extra_pos": n_extra_pos,
        "n_extra_neg": n_extra_neg,
        "n_need_asr": n_need_asr,
        "extra_reject": extra_tag,
        "note": "overlay_macro_diagnostic 仅供历史对照；overlay_official 是正式字符加权口径。n_need_asr>0 时新放行 pos 仍按 CER=1 保守估计。",
        "go_vs_file": bool((float(m1_official["contest"]) - float(m_file_official["contest"])) >= 0.005 and n_extra_pos <= 5),
    }
    od = ensure_dir(ve / "reports" / "lift_overlay")
    jp = od / f"{tag}.json"
    jp.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [
        f"# lift overlay `{tag}`",
        "",
        f"- locked_thr={args.locked_thr} zh={ZH_THR} en={EN_THR}",
        f"- file official contest={out['file_extract_official']['contest']} RR={out['file_extract_official']['rr']} CER={out['file_extract_official']['cer']}",
        f"- cosine official contest={out['cosine_only_official']['contest']} RR={out['cosine_only_official']['rr']} CER={out['cosine_only_official']['cer']}",
        f"- overlay macro diagnostic contest={out['overlay_macro_diagnostic']['contest']} RR={out['overlay_macro_diagnostic']['rr']} CER={out['overlay_macro_diagnostic']['cer']}",
        f"- overlay official contest={out['overlay_official']['contest']} RR={out['overlay_official']['rr']} CER={out['overlay_official']['cer']}",
        f"- Δ vs file={out['d_contest_vs_file']}  extra_pos={n_extra_pos} extra_neg={n_extra_neg} need_asr={n_need_asr}",
        "",
    ]
    (od / f"{tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    dump_rows = []
    for r in rows:
        rej = pred(r)
        dump_rows.append({
            "uid": r.get("uid"),
            "id": r.get("id"),
            "split": r.get("split"),
            "lang": r.get("lang"),
            "score": r.get("score"),
            "thr": r.get("thr"),
            "reject": rej,
            "reject_reason": overlay_reason(
                r,
                use_text=not args.no_text,
                use_camp=not args.no_camp,
                use_window=bool(args.window_veto),
                veto_margin=float(args.veto_margin),
            ) if rej else "",
            "extra_reject": bool((not r["base_rej"]) and rej),
            "hyp": r.get("hyp") or "",
            "cer": 1.0 if (r.get("split") == "pos" and rej) else r.get("cer"),
        })
    write_jsonl(od / f"{tag}_rows.jsonl", dump_rows)
    if args.write_result or tag == "submit":
        rp = args.result_json or (ve / "reports" / "submit" / "result.json")
        write_result_json(rp, rows, pred, m1_official, extra_reject=extra_tag)
        print(f"[OK] result.json → {rp} contest={round(float(m1_official['contest']), 6)} RR={round(float(m1_official['rr']), 6)}")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[OK] {jp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

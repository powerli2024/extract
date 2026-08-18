#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 asr_results.jsonl 汇总 pos CER / neg 转写统计 → analysis.json + analysis.md + per_utt.jsonl。"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

ARMS = ("no_sep", "sep_once", "sep_multi")
CER_OK = ("ok", "empty_hyp")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def _mean(xs: list[float]) -> Optional[float]:
    return round(sum(xs) / len(xs), 6) if xs else None


def _std(xs: list[float]) -> Optional[float]:
    if not xs:
        return None
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return round(math.sqrt(var), 6)


def _quantile(xs: list[float], p: float) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return round(s[idx], 6)


def _hist(cers: list[float]) -> dict[str, int]:
    h = Counter()
    for c in cers:
        if c <= 0:
            h["=0"] += 1
        elif c < 0.25:
            h["(0,0.25)"] += 1
        elif c < 0.5:
            h["[0.25,0.5)"] += 1
        elif c < 1:
            h["[0.5,1)"] += 1
        else:
            h["=1"] += 1
    return dict(h)


def _cer_block(cers: list[float]) -> dict[str, Any]:
    return {
        "n": len(cers),
        "mean": _mean(cers),
        "std": _std(cers),
        "p50": _quantile(cers, 0.5),
        "p90": _quantile(cers, 0.9),
        "p95": _quantile(cers, 0.95),
        "hist": _hist(cers),
        "exact_match_rate": round(sum(1 for c in cers if c <= 0) / len(cers), 6) if cers else None,
        "cer_ge_1_rate": round(sum(1 for c in cers if c >= 1) / len(cers), 6) if cers else None,
    }


def is_pos(r: dict[str, Any]) -> bool:
    return r.get("split") == "pos" or r.get("label") == "present"


def usable(r: dict[str, Any]) -> bool:
    return r.get("status") in CER_OK and r.get("cer") is not None


def group_by_utt(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    g: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        uid = r.get("uid")
        if uid:
            g[str(uid)].append(r)
    return g


def mix_rec(recs: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    for r in recs:
        if r.get("arm") == "no_sep" and r.get("stream") == "mix" and r.get("status") in CER_OK:
            return r
        if r.get("arm") == "no_sep" and r.get("stream") == "mix" and r.get("status") == "ok":
            return r
    for r in recs:
        if r.get("arm") == "no_sep" and r.get("status") in CER_OK:
            return r
    return None


def arm_recs(recs: list[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    return [r for r in recs if r.get("arm") == arm and r.get("status") in CER_OK]


def oracle(recs: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    good = [r for r in recs if r.get("cer") is not None]
    if not good:
        return None
    return min(good, key=lambda r: (float(r["cer"]), str(r.get("stream") or "")))


def worst(recs: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    good = [r for r in recs if r.get("cer") is not None]
    if not good:
        return None
    return max(good, key=lambda r: (float(r["cer"]), str(r.get("stream") or "")))


def compare_oracle_vs_mix(
    utts: dict[str, list[dict[str, Any]]], arm: str
) -> dict[str, Any]:
    win = lose = tie = 0
    deltas: list[float] = []
    win_d: list[float] = []
    lose_d: list[float] = []
    rows = []
    for uid, recs in utts.items():
        m = mix_rec(recs)
        o = oracle(arm_recs(recs, arm))
        if not m or m.get("cer") is None or not o:
            continue
        mc, oc = float(m["cer"]), float(o["cer"])
        d = oc - mc
        deltas.append(d)
        if abs(d) < 1e-9:
            tie += 1
        elif d < 0:
            win += 1
            win_d.append(d)
        else:
            lose += 1
            lose_d.append(d)
        rows.append((uid, mc, oc, d, o.get("stream"), m.get("lang") or recs[0].get("lang")))
    n = win + lose + tie
    return {
        "n_utt_both": n,
        "win_sep_better": win,
        "tie": tie,
        "lose_sep_worse": lose,
        "win_rate": round(win / n, 6) if n else None,
        "lose_rate": round(lose / n, 6) if n else None,
        "mean_delta_oracle_minus_mix": _mean(deltas),
        "mean_delta_when_win": _mean(win_d),
        "mean_delta_when_lose": _mean(lose_d),
        "median_delta": _quantile(deltas, 0.5),
    }


def arm_pos_stats(utts: dict[str, list[dict[str, Any]]], arm: str) -> dict[str, Any]:
    pool: list[float] = []
    oracles: list[float] = []
    worsts: list[float] = []
    n_streams: list[int] = []
    best_names: Counter[str] = Counter()
    empty_utt = 0
    n_utt = 0
    stream_cers: dict[str, list[float]] = defaultdict(list)
    for recs in utts.values():
        rs = [r for r in arm_recs(recs, arm) if r.get("cer") is not None]
        if not rs:
            continue
        n_utt += 1
        n_streams.append(len(rs))
        cers = [float(r["cer"]) for r in rs]
        pool.extend(cers)
        o = oracle(rs)
        w = worst(rs)
        if o:
            oracles.append(float(o["cer"]))
            best_names[str(o.get("stream") or "?")] += 1
            if float(o["cer"]) >= 1.0 or not (o.get("hyp_norm") or ""):
                empty_utt += 1
        if w:
            worsts.append(float(w["cer"]))
        for r in rs:
            stream_cers[str(r.get("stream") or "?")].append(float(r["cer"]))
    by_stream = {
        k: _cer_block(v) for k, v in sorted(stream_cers.items(), key=lambda kv: -len(kv[1]))
    }
    return {
        "n_utt": n_utt,
        "n_stream_rows": len(pool),
        "mean_n_streams": _mean([float(x) for x in n_streams]),
        "pool_all_streams": _cer_block(pool),
        "oracle_min": _cer_block(oracles),
        "worst_max": _cer_block(worsts),
        "oracle_empty_or_cer1_rate": round(empty_utt / n_utt, 6) if n_utt else None,
        "best_stream_counts": dict(best_names.most_common()),
        "by_stream": by_stream,
    }


def by_lang_pos(utts: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    buckets: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for uid, recs in utts.items():
        lang = None
        for r in recs:
            lang = r.get("lang") or lang
        lang = str(lang or "unk")
        buckets[lang][uid] = recs
    out: dict[str, Any] = {}
    for lang, sub in buckets.items():
        mix_c = []
        once_o = []
        multi_o = []
        for recs in sub.values():
            m = mix_rec(recs)
            if m and m.get("cer") is not None:
                mix_c.append(float(m["cer"]))
            o1 = oracle(arm_recs(recs, "sep_once"))
            if o1:
                once_o.append(float(o1["cer"]))
            o2 = oracle(arm_recs(recs, "sep_multi"))
            if o2:
                multi_o.append(float(o2["cer"]))
        out[lang] = {
            "n_utt": len(sub),
            "mix": _cer_block(mix_c),
            "sep_once_oracle": _cer_block(once_o),
            "sep_multi_oracle": _cer_block(multi_o),
            "mix_vs_sep_once": compare_oracle_vs_mix(sub, "sep_once"),
            "mix_vs_sep_multi": compare_oracle_vs_mix(sub, "sep_multi"),
        }
    return out


def pick_global_oracle(recs: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    good = [r for r in recs if r.get("status") in CER_OK and r.get("cer") is not None]
    if not good:
        return None
    return min(good, key=lambda r: (float(r["cer"]), str(r.get("arm")), str(r.get("stream"))))


def pos_analysis(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pos_rows = [r for r in rows if is_pos(r)]
    utts = group_by_utt(pos_rows)
    mix_cers = []
    per_utt: list[dict[str, Any]] = []
    best_arm: Counter[str] = Counter()
    worst20_src: list[tuple[float, dict[str, Any]]] = []
    gain20_src: list[tuple[float, dict[str, Any]]] = []

    for uid, recs in utts.items():
        m = mix_rec(recs)
        o1 = oracle(arm_recs(recs, "sep_once"))
        o2 = oracle(arm_recs(recs, "sep_multi"))
        og = pick_global_oracle(recs)
        lang = (m or recs[0]).get("lang")
        mix_cer = float(m["cer"]) if m and m.get("cer") is not None else None
        if mix_cer is not None:
            mix_cers.append(mix_cer)
        row = {
            "uid": uid,
            "split": "pos",
            "lang": lang,
            "wake_text": (m or recs[0]).get("wake_text"),
            "cmd_text": (m or recs[0]).get("cmd_text"),
            "mix_cer": mix_cer,
            "mix_hyp": (m or {}).get("asr_text") if m else None,
            "sep_once_oracle_cer": float(o1["cer"]) if o1 else None,
            "sep_once_best_stream": o1.get("stream") if o1 else None,
            "sep_once_best_hyp": o1.get("asr_text") if o1 else None,
            "sep_once_n_streams": len(arm_recs(recs, "sep_once")),
            "sep_multi_oracle_cer": float(o2["cer"]) if o2 else None,
            "sep_multi_best_stream": o2.get("stream") if o2 else None,
            "sep_multi_best_hyp": o2.get("asr_text") if o2 else None,
            "sep_multi_n_streams": len(arm_recs(recs, "sep_multi")),
            "oracle_all_cer": float(og["cer"]) if og else None,
            "oracle_all_arm": og.get("arm") if og else None,
            "oracle_all_stream": og.get("stream") if og else None,
        }
        if mix_cer is not None and row["sep_once_oracle_cer"] is not None:
            row["delta_once_minus_mix"] = round(row["sep_once_oracle_cer"] - mix_cer, 6)
        if mix_cer is not None and row["sep_multi_oracle_cer"] is not None:
            row["delta_multi_minus_mix"] = round(row["sep_multi_oracle_cer"] - mix_cer, 6)
        if og:
            best_arm[str(og.get("arm"))] += 1
        per_utt.append(row)
        if mix_cer is not None:
            worst20_src.append((mix_cer, row))
        d = row.get("delta_once_minus_mix")
        if d is not None:
            gain20_src.append((float(d), row))

    worst20 = [
        {
            "uid": r["uid"],
            "lang": r["lang"],
            "mix_cer": r["mix_cer"],
            "cmd_text": r["cmd_text"],
            "mix_hyp": r["mix_hyp"],
            "sep_once_oracle_cer": r["sep_once_oracle_cer"],
            "sep_once_best_stream": r["sep_once_best_stream"],
        }
        for _, r in sorted(worst20_src, key=lambda x: -x[0])[:20]
    ]
    best_gains = [
        {
            "uid": r["uid"],
            "lang": r["lang"],
            "mix_cer": r["mix_cer"],
            "sep_once_oracle_cer": r["sep_once_oracle_cer"],
            "delta_once_minus_mix": r.get("delta_once_minus_mix"),
            "sep_once_best_stream": r["sep_once_best_stream"],
            "cmd_text": r["cmd_text"],
            "mix_hyp": r["mix_hyp"],
            "sep_once_best_hyp": r["sep_once_best_hyp"],
        }
        for _, r in sorted(gain20_src, key=lambda x: x[0])[:20]
    ]
    worst_gains = [
        {
            "uid": r["uid"],
            "lang": r["lang"],
            "mix_cer": r["mix_cer"],
            "sep_once_oracle_cer": r["sep_once_oracle_cer"],
            "delta_once_minus_mix": r.get("delta_once_minus_mix"),
            "sep_once_best_stream": r["sep_once_best_stream"],
            "cmd_text": r["cmd_text"],
        }
        for _, r in sorted(gain20_src, key=lambda x: -x[0])[:20]
    ]

    # 全局 oracle 相对 mix
    all_oracles = [r["oracle_all_cer"] for r in per_utt if r.get("oracle_all_cer") is not None]
    analysis = {
        "n_utt": len(utts),
        "n_rows": len(pos_rows),
        "mix": _cer_block(mix_cers),
        "arms": {a: arm_pos_stats(utts, a) for a in ARMS},
        "mix_vs_sep_once_oracle": compare_oracle_vs_mix(utts, "sep_once"),
        "mix_vs_sep_multi_oracle": compare_oracle_vs_mix(utts, "sep_multi"),
        "oracle_all": {
            **_cer_block(all_oracles),
            "best_arm_counts": dict(best_arm),
            "note": "每条 utt 在 no_sep+sep_once+sep_multi 全部轨上取 min CER（乐观上界，不是可部署路由）",
        },
        "by_lang": by_lang_pos(utts),
        "worst_20_mix_cer": worst20,
        "top_20_sep_once_gains": best_gains,
        "top_20_sep_once_hurts": worst_gains,
        "takeaways": pos_takeaways(mix_cers, utts, best_arm),
    }
    return analysis, per_utt


def pos_takeaways(
    mix_cers: list[float],
    utts: dict[str, list[dict[str, Any]]],
    best_arm: Counter[str],
) -> list[str]:
    notes = []
    mix_m = _mean(mix_cers)
    once = compare_oracle_vs_mix(utts, "sep_once")
    multi = compare_oracle_vs_mix(utts, "sep_multi")
    once_stats = arm_pos_stats(utts, "sep_once")
    if mix_m is not None:
        notes.append(f"pos mix 平均 CER={mix_m}（竞赛口径是逐条平均，不是字符微平均）。")
    om = (once_stats.get("oracle_min") or {}).get("mean")
    pm = (once_stats.get("pool_all_streams") or {}).get("mean")
    if om is not None and mix_m is not None:
        notes.append(
            f"sep_once oracle min-CER={om}（相对 mix {om - mix_m:+.4f}）；"
            f"若随机/平均用所有 d1 轨则 pool CER={pm}（通常差于 mix，说明分离多数轨在伤 ASR）。"
        )
    if once.get("n_utt_both"):
        notes.append(
            f"sep_once oracle vs mix：更好 {once['win_sep_better']}/"
            f"{once['n_utt_both']}（win_rate={once['win_rate']}），"
            f"更差 {once['lose_sep_worse']}，平 {once['tie']}；"
            f"mean Δ(oracle-mix)={once['mean_delta_oracle_minus_mix']}。"
        )
    if multi.get("n_utt_both"):
        notes.append(
            f"sep_multi oracle vs mix：更好 {multi['win_sep_better']}/"
            f"{multi['n_utt_both']}，更差 {multi['lose_sep_worse']}；"
            f"mean Δ={multi['mean_delta_oracle_minus_mix']}。"
        )
    if best_arm:
        tot = sum(best_arm.values())
        mix_w = best_arm.get("no_sep", 0)
        notes.append(
            f"全局 oracle 最佳臂计数：{dict(best_arm)}；"
            f"仍选 mix 的比例={mix_w / tot:.3f}。"
            if tot
            else "无全局 oracle。"
        )
    notes.append(
        "oracle 需要事先知道哪条轨 CER 最低，竞赛时没有 cmd_text 不能这么选路；"
        "Presence 选路是另一套信号，本报告只回答「分离有没有 ASR 信息量」。"
    )
    return notes


def neg_arm_stats(utts: dict[str, list[dict[str, Any]]], arm: str) -> dict[str, Any]:
    lens: list[float] = []
    empty_streams = 0
    n_streams = 0
    utt_all_empty = 0
    utt_any_text = 0
    utt_n = 0
    wake_hit = 0
    long_hyps: list[tuple[int, str, str, str]] = []
    by_stream_empty: dict[str, list[int]] = defaultdict(list)
    for uid, recs in utts.items():
        rs = arm_recs(recs, arm)
        if not rs:
            continue
        utt_n += 1
        empties = 0
        any_t = False
        for r in rs:
            n_streams += 1
            hn = r.get("hyp_norm") or ""
            nch = int(r.get("hyp_nchars") or len(hn))
            lens.append(float(nch))
            is_empty = bool(r.get("empty_hyp") or not hn)
            by_stream_empty[str(r.get("stream") or "?")].append(1 if is_empty else 0)
            if is_empty:
                empty_streams += 1
                empties += 1
            else:
                any_t = True
            if r.get("wake_overlap"):
                wake_hit += 1
            if nch >= 8:
                long_hyps.append((nch, uid, str(r.get("stream")), str(r.get("asr_text") or "")))
        if empties == len(rs):
            utt_all_empty += 1
        if any_t:
            utt_any_text += 1
    long_hyps.sort(key=lambda x: -x[0])
    return {
        "n_utt": utt_n,
        "n_streams": n_streams,
        "empty_stream_rate": round(empty_streams / n_streams, 6) if n_streams else None,
        "utt_all_streams_empty_rate": round(utt_all_empty / utt_n, 6) if utt_n else None,
        "utt_any_nonempty_rate": round(utt_any_text / utt_n, 6) if utt_n else None,
        "mean_hyp_nchars": _mean(lens),
        "mean_hyp_nchars_nonzero": _mean([x for x in lens if x > 0]),
        "wake_overlap_stream_rate": round(wake_hit / n_streams, 6) if n_streams else None,
        "empty_rate_by_stream": {
            k: round(sum(v) / len(v), 6) for k, v in sorted(by_stream_empty.items())
        },
        "top_20_longest_hyps": [
            {"nchars": n, "uid": u, "stream": s, "asr_text": t} for n, u, s, t in long_hyps[:20]
        ],
    }


def neg_analysis(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    neg_rows = [r for r in rows if not is_pos(r)]
    utts = group_by_utt(neg_rows)
    per_utt = []
    for uid, recs in utts.items():
        m = mix_rec(recs)
        o1 = arm_recs(recs, "sep_once")
        o2 = arm_recs(recs, "sep_multi")
        mix_h = (m.get("hyp_norm") or "") if m else ""
        once_any = any((r.get("hyp_norm") or "") for r in o1)
        multi_any = any((r.get("hyp_norm") or "") for r in o2)
        per_utt.append(
            {
                "uid": uid,
                "split": "neg",
                "lang": (m or recs[0]).get("lang"),
                "wake_text": (m or recs[0]).get("wake_text"),
                "mix_hyp": (m or {}).get("asr_text") if m else None,
                "mix_nchars": len(mix_h),
                "mix_empty": not bool(mix_h),
                "mix_wake_overlap": bool(m.get("wake_overlap")) if m else False,
                "sep_once_any_nonempty": once_any,
                "sep_once_n_streams": len(o1),
                "sep_multi_any_nonempty": multi_any,
                "sep_multi_n_streams": len(o2),
            }
        )
    mix_empty = [r for r in per_utt if r.get("mix_empty")]
    once_extra = [
        r
        for r in per_utt
        if r.get("mix_empty") and r.get("sep_once_any_nonempty")
    ]
    analysis = {
        "n_utt": len(utts),
        "n_rows": len(neg_rows),
        "note": "neg 无 cmd_text，不算 CER；看空转写、长度、是否像唤醒词。分离若把噪声听成命令，会在竞赛 accept 后抬 CER。",
        "arms": {a: neg_arm_stats(utts, a) for a in ARMS},
        "mix_empty_but_sep_once_speaks_rate": (
            round(len(once_extra) / len(mix_empty), 6) if mix_empty else None
        ),
        "n_mix_empty": len(mix_empty),
        "n_mix_empty_sep_once_speaks": len(once_extra),
        "takeaways": neg_takeaways(utts, mix_empty, once_extra),
    }
    return analysis, per_utt


def neg_takeaways(
    utts: dict[str, list[dict[str, Any]]],
    mix_empty: list[dict[str, Any]],
    once_extra: list[dict[str, Any]],
) -> list[str]:
    notes = []
    mix = neg_arm_stats(utts, "no_sep")
    once = neg_arm_stats(utts, "sep_once")
    notes.append(
        f"neg mix 空转写率={mix.get('empty_stream_rate')}，"
        f"sep_once 空轨率={once.get('empty_stream_rate')}，"
        f"utt 上「所有 d1 轨都空」={once.get('utt_all_streams_empty_rate')}。"
    )
    if mix_empty:
        notes.append(
            f"mix 本身为空的 {len(mix_empty)} 条里，有 {len(once_extra)} 条 "
            f"({analysis_pct(len(once_extra), len(mix_empty))}) 在 sep_once 某轨上转出了字——"
            "分离可能把干扰说成命令，Presence 若误放行会伤 CER。"
        )
    notes.append("wake_overlap：转写与唤醒词高度重合，常见于模型把噪声/唤醒残影听成唤醒。")
    return notes


def analysis_pct(a: int, b: int) -> str:
    return f"{a / b:.1%}" if b else "n/a"


def status_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    c = Counter(str(r.get("status") or "none") for r in rows)
    arms = Counter(str(r.get("arm") or "") for r in rows)
    splits = Counter(str(r.get("split") or "") for r in rows)
    return {
        "n_rows": len(rows),
        "by_status": dict(c),
        "by_arm": dict(arms),
        "by_split": dict(splits),
    }


def md_report(doc: dict[str, Any]) -> str:
    pos = doc.get("pos") or {}
    neg = doc.get("neg") or {}
    cov = doc.get("coverage") or {}
    lines = [
        "# asr_probe 分析",
        "",
        "全量 pos/neg × `no_sep` / `sep_once` / `sep_multi` 的 Qwen3-ASR。"
        "不做拒识；pos 相对 `cmd_text` 算 CER；neg 只看转写形态。",
        "",
        "## 覆盖",
        "",
        "```json",
        json.dumps(cov, ensure_ascii=False, indent=2),
        "```",
        "",
        "## pos（有 cmd_text）",
        "",
        f"- utt 数: **{pos.get('n_utt')}**  行数: {pos.get('n_rows')}",
        f"- mix 平均 CER: **{(pos.get('mix') or {}).get('mean')}**  "
        f"exact={ (pos.get('mix') or {}).get('exact_match_rate') }",
        "",
    ]
    for a in ARMS:
        st = (pos.get("arms") or {}).get(a) or {}
        o = st.get("oracle_min") or {}
        p = st.get("pool_all_streams") or {}
        lines.append(
            f"### {a}\n\n"
            f"- utt={st.get('n_utt')}  stream 行={st.get('n_stream_rows')}  "
            f"均轨数={st.get('mean_n_streams')}\n"
            f"- 所有轨 pool CER mean={p.get('mean')}  oracle min mean={o.get('mean')}  "
            f"worst max mean={(st.get('worst_max') or {}).get('mean')}\n"
            f"- oracle 最佳轨计数: `{st.get('best_stream_counts')}`\n"
        )
    lines += [
        "### mix vs 分离 oracle",
        "",
        f"- sep_once: `{json.dumps(pos.get('mix_vs_sep_once_oracle'), ensure_ascii=False)}`",
        f"- sep_multi: `{json.dumps(pos.get('mix_vs_sep_multi_oracle'), ensure_ascii=False)}`",
        "",
        "### 按语言",
        "",
        "```json",
        json.dumps(pos.get("by_lang"), ensure_ascii=False, indent=2),
        "```",
        "",
        "### 结论要点",
        "",
    ]
    for t in pos.get("takeaways") or []:
        lines.append(f"- {t}")
    lines += [
        "",
        "### mix CER 最差 20",
        "",
        "```json",
        json.dumps(pos.get("worst_20_mix_cer"), ensure_ascii=False, indent=2),
        "```",
        "",
        "### sep_once oracle 相对 mix 收益最大 20（Δ 最负）",
        "",
        "```json",
        json.dumps(pos.get("top_20_sep_once_gains"), ensure_ascii=False, indent=2),
        "```",
        "",
        "## neg（无识别文本）",
        "",
        f"- utt 数: **{neg.get('n_utt')}**  行数: {neg.get('n_rows')}",
        f"- mix 空但 sep_once 某轨有字: {neg.get('n_mix_empty_sep_once_speaks')} / "
        f"{neg.get('n_mix_empty')}  (rate={neg.get('mix_empty_but_sep_once_speaks_rate')})",
        "",
    ]
    for t in neg.get("takeaways") or []:
        lines.append(f"- {t}")
    lines += [
        "",
        "```json",
        json.dumps(neg.get("arms"), ensure_ascii=False, indent=2),
        "```",
        "",
        "## 怎么用到竞赛",
        "",
        "- 若 oracle 也打不赢 mix：分离不该进 ASR，Presence 最多用 sep 打分。",
        "- 若 oracle 明显赢但 pool 明显输：信息在「选对轨」，没有可靠选路就不要把 d1/d2 送给 ASR。",
        "- neg 上分离更爱「听出字」：误放行时 CER 风险更大。",
        "",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="asr_probe 汇总")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    path = args.results.resolve()
    if not path.is_file():
        raise SystemExit(f"找不到 {path}")
    out_dir = (args.out_dir or path.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(path)
    if not rows:
        raise SystemExit(f"空 jsonl: {path}")

    pos, pos_utt = pos_analysis(rows)
    neg, neg_utt = neg_analysis(rows)
    doc = {
        "source": str(path),
        "coverage": status_counts(rows),
        "pos": pos,
        "neg": neg,
    }
    (out_dir / "analysis.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "analysis.md").write_text(md_report(doc), encoding="utf-8")
    with (out_dir / "per_utt.jsonl").open("w", encoding="utf-8") as f:
        for r in pos_utt + neg_utt:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] {out_dir / 'analysis.json'}")
    print(f"[OK] {out_dir / 'analysis.md'}")
    print(f"[OK] {out_dir / 'per_utt.jsonl'}  pos_utt={len(pos_utt)} neg_utt={len(neg_utt)}")
    # 控制台摘要
    print("--- pos mix ---")
    print(json.dumps(pos.get("mix"), ensure_ascii=False, indent=2))
    print("--- mix vs sep_once oracle ---")
    print(json.dumps(pos.get("mix_vs_sep_once_oracle"), ensure_ascii=False, indent=2))
    print("--- takeaways ---")
    for t in (pos.get("takeaways") or []) + (neg.get("takeaways") or []):
        print("*", t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

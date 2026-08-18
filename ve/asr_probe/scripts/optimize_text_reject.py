#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在「只加拒、不救回」约束下，用 holdout 锁文本拒识参数，使竞赛分高于余弦基线。"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refine_linguistic_patterns import classify_v2

ROOT = Path(r"d:\media\datasetA\sssss")
ZH_THR = 0.29305
EN_THR = 0.357868


def load_all():
    asr = json.loads((ROOT / "no_sep.json").read_text(encoding="utf-8"))
    scores = {}
    for line in (ROOT / "scores (1).jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            scores[r["uid"]] = r
    rows = []
    for r in asr:
        s = float(scores[r["uid"]]["presence_score"])
        lang = r.get("lang") or "zh"
        thr = ZH_THR if lang == "zh" else EN_THR
        v2 = classify_v2(r.get("asr_text") or r.get("hyp_norm"))
        r2 = dict(r)
        r2.update({
            "score": s,
            "thr": thr,
            "lang": lang,
            "base_rej": s < thr,
            "v2": v2,
            "margin": s - thr,  # <0 拒，>0 过门
        })
        rows.append(r2)
    return rows


def metrics(rows, pred) -> dict:
    pos_c = []
    n_pos = n_fr = n_neg = n_rej = 0
    for r in rows:
        rej = pred(r)
        if r["split"] == "pos":
            n_pos += 1
            if rej:
                n_fr += 1
                pos_c.append(1.0)
            else:
                c = r.get("cer")
                pos_c.append(1.0 if c is None else float(c))
        else:
            n_neg += 1
            if rej:
                n_rej += 1
    rr = n_rej / n_neg if n_neg else 0
    frr = n_fr / n_pos if n_pos else 1
    cer = sum(pos_c) / len(pos_c) if pos_c else 1
    return {
        "rr": rr, "frr": frr, "cer": cer,
        "contest": 0.5 * rr + 0.5 * (1 - cer),
        "n_fr": n_fr, "n_rej_neg": n_rej, "n_pos": n_pos, "n_neg": n_neg,
    }


def stratified_split(rows, frac=0.3, seed=7):
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[(r["split"], r["lang"])].append(r)
    train, test = [], []
    for g in by.values():
        g = list(g)
        rng.shuffle(g)
        n = max(1, int(round(len(g) * frac)))
        test.extend(g[:n])
        train.extend(g[n:])
    return train, test


def make_pred(kind: str, delta: float, L: int):
    """只允许把「过门」改成拒识，不允许把拒识改成接受。"""

    def pred(r):
        if r["base_rej"]:
            return True
        # 过门样本才考虑文本加拒
        v2 = r["v2"]
        m = r["margin"]
        in_upper_gray = 0 <= m <= delta
        tts = v2["primary"] == "设备播报"
        chatty = v2["primary"] in ("闲聊/背景", "其他", "空转写")
        long = v2["len"] >= L
        nontask = not v2["task_oriented"]
        if kind == "tts_anywhere":
            return tts
        if kind == "tts_gray":
            return in_upper_gray and tts
        if kind == "len_gray":
            return in_upper_gray and long
        if kind == "nontask_gray":
            return in_upper_gray and nontask
        if kind == "chat_gray":
            return in_upper_gray and chatty
        if kind == "len_or_nontask_gray":
            return in_upper_gray and (long or nontask)
        if kind == "len_and_nontask_gray":
            return in_upper_gray and long and nontask
        if kind == "len_or_chat_gray":
            return in_upper_gray and (long or chatty)
        if kind == "len_and_chat_gray":
            return in_upper_gray and long and chatty
        if kind == "len_anywhere":
            return long
        if kind == "len_gray_or_tts":
            return tts or (in_upper_gray and long)
        if kind == "len_or_chat_gray_or_tts":
            return tts or (in_upper_gray and (long or chatty))
        return False

    return pred


def region_stats(rows) -> dict:
    """语言学差异必须看「过门子集」，那才是文本加拒能碰到的地方。"""
    fa = [r for r in rows if r["split"] == "neg" and not r["base_rej"]]
    tp = [r for r in rows if r["split"] == "pos" and not r["base_rej"]]
    fr = [r for r in rows if r["split"] == "pos" and r["base_rej"]]
    tn = [r for r in rows if r["split"] == "neg" and r["base_rej"]]

    def pack(xs, name):
        if not xs:
            return {"n": 0}
        lens = [r["v2"]["len"] for r in xs]
        prim = Counter(r["v2"]["primary"] for r in xs)
        n = len(xs)
        def rate(fn):
            return round(sum(1 for r in xs if fn(r)) / n, 4)
        return {
            "n": n,
            "len_mean": round(sum(lens) / n, 2),
            "len_p50": sorted(lens)[n // 2],
            "len_p90": sorted(lens)[min(n - 1, int(round(0.9 * (n - 1))))],
            "task_rate": rate(lambda r: r["v2"]["task_oriented"]),
            "len_ge12": rate(lambda r: r["v2"]["len"] >= 12),
            "len_ge14": rate(lambda r: r["v2"]["len"] >= 14),
            "len_ge16": rate(lambda r: r["v2"]["len"] >= 16),
            "len_ge18": rate(lambda r: r["v2"]["len"] >= 18),
            "chat_or_other": rate(lambda r: r["v2"]["primary"] in ("闲聊/背景", "其他", "设备播报", "空转写")),
            "tts": rate(lambda r: r["v2"]["primary"] == "设备播报"),
            "primary": dict(prim.most_common()),
        }

    # 过门后：len>=L 的精确率（预测 neg）
    prec = {}
    for L in (12, 14, 16, 18, 20):
        hit_neg = sum(1 for r in fa if r["v2"]["len"] >= L)
        hit_pos = sum(1 for r in tp if r["v2"]["len"] >= L)
        prec[f"len>={L}_on_accept"] = {
            "n_fa_hit": hit_neg, "n_tp_hit": hit_pos,
            "prec_as_neg": round(hit_neg / (hit_neg + hit_pos), 4) if hit_neg + hit_pos else None,
            "fa_recall": round(hit_neg / len(fa), 4) if fa else None,
        }
    return {"FA_accepted_neg": pack(fa, "fa"), "TP_accepted_pos": pack(tp, "tp"),
            "FR_rejected_pos": pack(fr, "fr"), "TN_rejected_neg": pack(tn, "tn"),
            "length_as_neg_on_accepts": prec}


def main() -> int:
    rows = load_all()
    base = metrics(rows, lambda r: r["base_rej"])
    diag = region_stats(rows)
    train, test = stratified_split(rows)

    kinds = [
        "len_gray", "nontask_gray", "chat_gray",
        "len_or_nontask_gray", "len_and_nontask_gray",
        "len_or_chat_gray", "len_and_chat_gray",
        "len_gray_or_tts", "len_or_chat_gray_or_tts",
        "tts_anywhere", "len_anywhere",
    ]
    deltas = [0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
    Ls = [12, 13, 14, 15, 16, 17, 18, 20]

    cand = []
    for kind in kinds:
        d_iter = [0.0] if kind in ("tts_anywhere", "len_anywhere") else deltas
        L_iter = [16] if kind in ("tts_anywhere", "nontask_gray", "chat_gray") else Ls
        for d in d_iter:
            for L in L_iter:
                pred = make_pred(kind, d, L)
                tr, te = metrics(train, pred), metrics(test, pred)
                btr, bte = metrics(train, lambda r: r["base_rej"]), metrics(test, lambda r: r["base_rej"])
                cand.append({
                    "kind": kind, "delta": d, "L": L,
                    "train_C": round(tr["contest"], 6),
                    "test_C": round(te["contest"], 6),
                    "train_dC": round(tr["contest"] - btr["contest"], 6),
                    "test_dC": round(te["contest"] - bte["contest"], 6),
                    "train_rr": round(tr["rr"], 4),
                    "test_rr": round(te["rr"], 4),
                    "train_frr": round(tr["frr"], 4),
                    "test_frr": round(te["frr"], 4),
                    "train_cer": round(tr["cer"], 4),
                    "test_cer": round(te["cer"], 4),
                    "full": metrics(rows, pred),
                })

    # 选择：test 必须涨分；train 也必须不掉（防过拟合）；FRR 增幅限制
    bte = metrics(test, lambda r: r["base_rej"])
    btr = metrics(train, lambda r: r["base_rej"])
    viable = [
        c for c in cand
        if c["test_dC"] > 1e-6
        and c["train_dC"] >= -1e-4
        and (c["test_frr"] - bte["frr"]) <= 0.02
        and (c["train_frr"] - btr["frr"]) <= 0.02
    ]
    viable.sort(key=lambda c: (c["test_C"], c["train_C"], -c["test_frr"]), reverse=True)
    # 即便不满足约束，也保留 test 最优便于对照
    cand.sort(key=lambda c: (c["test_C"], c["train_dC"]), reverse=True)

    best = viable[0] if viable else cand[0]
    pred = make_pred(best["kind"], best["delta"], best["L"])
    full = metrics(rows, pred)

    extra_pos, extra_neg = [], []
    for r in rows:
        if r["base_rej"]:
            continue
        if pred(r):
            (extra_pos if r["split"] == "pos" else extra_neg).append(r)

    # 写 result.json（从 backup 的 id/label 对齐）
    backup_p = ROOT / "result_lang_split_backup.json"
    src = json.loads(backup_p.read_text(encoding="utf-8")) if backup_p.is_file() else json.loads((ROOT / "result.json").read_text(encoding="utf-8"))
    pos_map = {r["id"]: r for r in rows if r["split"] == "pos"}
    new_results = []
    for old in src["results"]:
        p = pos_map[old["id"]]
        if pred(p):
            new_results.append({"id": old["id"], "content": "", "label": old["label"], "cer": 1.0})
        else:
            hyp = p.get("asr_text") or ""
            cer = p.get("cer")
            new_results.append({
                "id": old["id"],
                "content": hyp,
                "label": old["label"],
                "cer": 1.0 if cer is None else float(cer),
            })
    out = {
        "results": new_results,
        "avg_cer": full["cer"],
        "avg_rr": full["rr"],
        "contest": full["contest"],
    }
    (ROOT / "result.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def brief(xs, k=12):
        return [{
            "uid": r["uid"], "id": r.get("id"), "lang": r["lang"],
            "score": round(r["score"], 4), "margin": round(r["margin"], 4),
            "primary": r["v2"]["primary"], "len": r["v2"]["len"],
            "task": r["v2"]["task_oriented"],
            "asr": (r.get("asr_text") or "")[:48],
            "cer": r.get("cer"),
        } for r in xs[:k]]

    summary = {
        "principle": "文本只加拒、不救回。身份由余弦决定；文本只抓过门后的长叠话/闲聊/设备播报。",
        "baseline_full": base,
        "chosen": {
            "kind": best["kind"], "delta": best["delta"], "L": best["L"],
            "train": {k: best[k] for k in ("train_C", "train_dC", "train_rr", "train_frr", "train_cer")},
            "test": {k: best[k] for k in ("test_C", "test_dC", "test_rr", "test_frr", "test_cer")},
            "full": full,
        },
        "n_viable": len(viable),
        "top_viable": [{k: c[k] for k in ("kind", "delta", "L", "train_C", "test_C", "train_dC", "test_dC", "test_rr", "test_frr")} for c in viable[:8]],
        "top_any_test": [{k: c[k] for k in ("kind", "delta", "L", "train_C", "test_C", "train_dC", "test_dC")} for c in cand[:8]],
        "accept_region_linguistics": diag,
        "n_extra_pos_reject": len(extra_pos),
        "n_extra_neg_reject": len(extra_neg),
        "extra_pos_reject": brief(extra_pos, 15),
        "extra_neg_reject": brief(extra_neg, 20),
    }
    (ROOT / "text_reject_holdout.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "baseline_contest": round(base["contest"], 6),
        "baseline_rr": round(base["rr"], 6),
        "baseline_cer": round(base["cer"], 6),
        "chosen": best["kind"], "delta": best["delta"], "L": best["L"],
        "full_contest": round(full["contest"], 6),
        "full_rr": round(full["rr"], 6),
        "full_cer": round(full["cer"], 6),
        "full_frr": round(full["frr"], 6),
        "d_contest": round(full["contest"] - base["contest"], 6),
        "n_extra_neg": len(extra_neg),
        "n_extra_pos": len(extra_pos),
        "n_viable": len(viable),
        "test_dC": best["test_dC"],
        "train_dC": best["train_dC"],
        "FA": diag["FA_accepted_neg"],
        "TP": diag["TP_accepted_pos"],
        "len_prec": diag["length_as_neg_on_accepts"],
        "top_viable": summary["top_viable"][:5],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

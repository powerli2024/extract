#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Linguistic + presence-score analysis of datasetA/sssss/no_sep.json.

Runtime-legal features only (no cmd_text at test). Holdout to check overfit.
"""
from __future__ import annotations

import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

CJK = re.compile(r"[\u4e00-\u9fff]")
LATIN = re.compile(r"[a-zA-Z]")
DIGIT = re.compile(r"\d|[零一二两三四五六七八九十百千万半]")
PUNCT = str.maketrans("", "", "，。！？、；：""''「」『』（）【】《》…·—–-.,!?;:\"'`")

DEVICES = [
    "空调", "灯光", "灯", "洗碗机", "洗衣机", "冰箱", "电视", "热水器",
    "新风", "地暖", "窗帘", "风扇", "加湿器", "净化器", "烤箱", "油烟机",
    "扫地机器人", "扫地机", "音响", "音箱", "插座", "开关", "空调",
    "airconditioner", "light", "lamp", "dishwasher", "washer", "fridge",
    "tv", "heater", "curtain", "fan", "humidifier", "speaker", "oven",
]
ACTS = [
    "打开", "关闭", "关掉", "开启", "暂停", "开始", "停止", "调到", "调成",
    "调高", "调低", "升高", "降低", "设置", "切换", "启动", "关机", "开机",
    "turnon", "turnoff", "set", "pause", "stop", "start", "increase",
    "decrease", "switch",
]
SLOTS = [
    "温度", "风量", "亮度", "湿度", "模式", "制热", "制冷", "送风", "除湿",
    "自动", "静音", "百分", "度", "档", "temperature", "brightness",
    "volume", "mode", "percent", "heat", "cool", "fan",
]


def nfkc(s: Optional[str]) -> str:
    if not s:
        return ""
    t = unicodedata.normalize("NFKC", str(s))
    t = "".join(ch for ch in t if not ch.isspace())
    t = t.translate(PUNCT)
    return t.lower().strip()


def cmd_lang(text: str) -> str:
    t = nfkc(text)
    if not t:
        return "empty"
    cjk = len(CJK.findall(t))
    lat = len(LATIN.findall(t))
    if cjk == 0 and lat == 0:
        return "other"
    if cjk > 0 and lat > 0:
        return "mix"
    if cjk >= max(1, (cjk + lat) // 2):
        return "zh"
    return "en"


def has_any(t: str, vocab: list[str]) -> bool:
    return any(v in t for v in vocab)


def looks_command(t: str) -> bool:
    if len(t) < 2:
        return False
    return has_any(t, DEVICES) or has_any(t, ACTS) or has_any(t, SLOTS)


def wake_in_hyp(wake: str, hyp: str) -> bool:
    w, h = nfkc(wake), nfkc(hyp)
    if not w or not h:
        return False
    return w in h or h in w


def wake_only(wake: str, hyp: str) -> bool:
    w, h = nfkc(wake), nfkc(hyp)
    if not w or not h:
        return False
    return h == w or h.replace(w, "") == "" or (w in h and len(h) <= len(w) + 2)


def edit_ratio(a: str, b: str) -> float:
    a, b = nfkc(a), nfkc(b)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
            prev = cur
    return min(1.0, dp[n] / max(m, 1))


def mean(xs: list[float]) -> Optional[float]:
    return round(sum(xs) / len(xs), 6) if xs else None


def quantile(xs: list[float], p: float) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return round(s[idx], 6)


def hist_cer(xs: list[float]) -> dict[str, int]:
    h = Counter()
    for c in xs:
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


def ngrams(text: str, n: int) -> list[str]:
    t = nfkc(text)
    if len(t) < n:
        return []
    return [t[i : i + n] for i in range(len(t) - n + 1)]


def load_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def feats(row: dict[str, Any]) -> dict[str, Any]:
    hyp = nfkc(row.get("hyp_norm") or row.get("asr_text"))
    wake = nfkc(row.get("wake_text"))
    cmd = nfkc(row.get("cmd_text") or "")
    hlang = cmd_lang(hyp)
    wlang = str(row.get("lang") or cmd_lang(wake))
    clang = cmd_lang(cmd) if cmd else "na"
    return {
        "empty": not hyp,
        "hyp_len": len(hyp),
        "hyp_lang": hlang,
        "wake_lang": wlang,
        "cmd_lang": clang,
        "wake_cmd_lang_mismatch": clang in ("zh", "en") and wlang in ("zh", "en") and clang != wlang,
        "wake_hyp_lang_mismatch": hlang in ("zh", "en") and wlang in ("zh", "en") and hlang != wlang,
        "cmd_hyp_lang_mismatch": clang in ("zh", "en") and hlang in ("zh", "en") and clang != hlang,
        "looks_command": looks_command(hyp),
        "has_device": has_any(hyp, DEVICES),
        "has_act": has_any(hyp, ACTS),
        "has_slot": has_any(hyp, SLOTS),
        "has_digit": bool(DIGIT.search(hyp)),
        "wake_in_hyp": wake_in_hyp(wake, hyp),
        "wake_only": wake_only(wake, hyp),
        "code_switch_hyp": hlang == "mix",
        "cer": row.get("cer"),
        "dur": row.get("dur_sec"),
        "chars_per_sec": (len(hyp) / row["dur_sec"]) if row.get("dur_sec") else None,
    }


def contest(rr: float, frr: float) -> float:
    # proxy: pos reject CER=1, accept CER=0
    return round(0.5 * rr + 0.5 * (1.0 - frr), 6)


def real_contest(rr: float, pos_cers: list[float]) -> float:
    m = sum(pos_cers) / len(pos_cers) if pos_cers else 1.0
    return round(0.5 * rr + 0.5 * (1.0 - m), 6)


def sweep_thr(pos_s: list[float], neg_s: list[float]) -> dict[str, Any]:
    cands = sorted(set(round(x, 4) for x in pos_s + neg_s))
    if not cands:
        return {"thr": None, "rr": 0, "frr": 1, "proxy": 0}
    best = None
    for thr in cands:
        frr = sum(1 for x in pos_s if x < thr) / len(pos_s) if pos_s else 1
        rr = sum(1 for x in neg_s if x < thr) / len(neg_s) if neg_s else 0
        pxy = contest(rr, frr)
        row = {"thr": thr, "rr": round(rr, 6), "frr": round(frr, 6), "proxy": pxy}
        if best is None or pxy > best["proxy"]:
            best = row
    return best or {"thr": None, "rr": 0, "frr": 1, "proxy": 0}


def apply_rule(
    rows: list[dict[str, Any]],
    scores: dict[str, dict[str, Any]],
    *,
    mode: str,
    zh_thr: float,
    en_thr: float,
    gray: float,
) -> dict[str, Any]:
    """Decide reject if score < lang_thr, with optional ASR gray-zone overlay.

    mode:
      score_only
      gray_reject_empty      — in gray zone, reject if empty hyp
      gray_reject_not_cmd    — in gray zone, reject if not looks_command
      gray_reject_wake_only
      gray_accept_cmd        — in gray zone, accept if looks_command else score
      gray_reject_empty_or_wake
    """
    tp_cer: list[float] = []  # per pos utt contest CER
    n_neg = n_rej = 0
    n_pos = n_fr = 0
    n_gray = 0
    n_overlay_flip = 0
    for r in rows:
        uid = r["uid"]
        sc = scores.get(uid)
        if not sc:
            continue
        lang = r.get("lang") or "zh"
        thr = zh_thr if lang == "zh" else en_thr
        score = float(sc["presence_score"])
        f = r["_f"]
        below = score < thr
        in_gray = abs(score - thr) <= gray
        reject = below
        if mode != "score_only" and in_gray:
            n_gray += 1
            asr_reject = False
            asr_accept = False
            if mode == "gray_reject_empty":
                asr_reject = f["empty"]
            elif mode == "gray_reject_not_cmd":
                asr_reject = not f["looks_command"]
            elif mode == "gray_reject_wake_only":
                asr_reject = f["wake_only"] or f["empty"]
            elif mode == "gray_reject_empty_or_wake":
                asr_reject = f["empty"] or f["wake_only"] or f["wake_in_hyp"]
            elif mode == "gray_accept_cmd":
                asr_accept = f["looks_command"] and not f["empty"]
                asr_reject = f["empty"] or f["wake_only"]
            if asr_reject:
                if not reject:
                    n_overlay_flip += 1
                reject = True
            elif asr_accept:
                if reject:
                    n_overlay_flip += 1
                reject = False
        is_pos = r["split"] == "pos"
        if is_pos:
            n_pos += 1
            if reject:
                n_fr += 1
                tp_cer.append(1.0)
            else:
                tp_cer.append(float(r.get("cer") if r.get("cer") is not None else 1.0))
        else:
            n_neg += 1
            if reject:
                n_rej += 1
    frr = n_fr / n_pos if n_pos else 1
    rr = n_rej / n_neg if n_neg else 0
    return {
        "mode": mode,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "rr": round(rr, 6),
        "frr": round(frr, 6),
        "cer_total": round(sum(tp_cer) / len(tp_cer), 6) if tp_cer else None,
        "contest_real": real_contest(rr, tp_cer),
        "contest_proxy": contest(rr, frr),
        "n_gray": n_gray,
        "n_overlay_flip": n_overlay_flip,
        "zh_thr": zh_thr,
        "en_thr": en_thr,
        "gray": gray,
    }


def holdout_eval(
    rows: list[dict[str, Any]],
    scores: dict[str, dict[str, Any]],
    modes: list[str],
    seed: int = 7,
    frac: float = 0.3,
) -> dict[str, Any]:
    rng = random.Random(seed)
    by_key: dict[tuple, list] = defaultdict(list)
    for r in rows:
        by_key[(r["split"], r.get("lang") or "zh")].append(r)
    train, test = [], []
    for _, grp in by_key.items():
        g = list(grp)
        rng.shuffle(g)
        n_te = max(1, int(round(len(g) * frac)))
        test.extend(g[:n_te])
        train.extend(g[n_te:])

    def lang_thrs(subset: list[dict[str, Any]]) -> tuple[float, float]:
        out = {}
        for lang in ("zh", "en"):
            pos = [
                float(scores[r["uid"]]["presence_score"])
                for r in subset
                if r["split"] == "pos" and (r.get("lang") or "zh") == lang and r["uid"] in scores
            ]
            neg = [
                float(scores[r["uid"]]["presence_score"])
                for r in subset
                if r["split"] == "neg" and (r.get("lang") or "zh") == lang and r["uid"] in scores
            ]
            out[lang] = sweep_thr(pos, neg)["thr"] or 0.3
        return out["zh"], out["en"]

    zh_t, en_t = lang_thrs(train)
    rows_out = []
    for g in (0.0, 0.04, 0.06, 0.08, 0.10, 0.12):
        for mode in modes:
            if mode == "score_only" and g != 0.0:
                continue
            tr = apply_rule(train, scores, mode=mode, zh_thr=zh_t, en_thr=en_t, gray=g)
            te = apply_rule(test, scores, mode=mode, zh_thr=zh_t, en_thr=en_t, gray=g)
            rows_out.append(
                {
                    "gray": g,
                    "mode": mode,
                    "train_contest": tr["contest_real"],
                    "test_contest": te["contest_real"],
                    "train_rr": tr["rr"],
                    "test_rr": te["rr"],
                    "train_frr": tr["frr"],
                    "test_frr": te["frr"],
                    "train_cer": tr["cer_total"],
                    "test_cer": te["cer_total"],
                    "test_n_gray": te["n_gray"],
                    "test_flips": te["n_overlay_flip"],
                    "gap": round(tr["contest_real"] - te["contest_real"], 6),
                }
            )
    rows_out.sort(key=lambda x: (-x["test_contest"], x["gap"]))
    return {
        "n_train": len(train),
        "n_test": len(test),
        "zh_thr_train": zh_t,
        "en_thr_train": en_t,
        "top": rows_out[:12],
        "score_only": [x for x in rows_out if x["mode"] == "score_only"][0],
        "all": rows_out,
    }


def main() -> int:
    root = Path(r"d:\media\datasetA\sssss")
    asr = load_json(root / "no_sep.json")
    scores_rows = load_jsonl(root / "scores (1).jsonl")
    scores = {r["uid"]: r for r in scores_rows}

    for r in asr:
        r["_f"] = feats(r)

    pos = [r for r in asr if r["split"] == "pos"]
    neg = [r for r in asr if r["split"] == "neg"]

    pos_cer = [float(r["cer"]) for r in pos if r.get("cer") is not None]
    status = Counter(r.get("status") for r in asr)

    # templates from pos cmd
    cmd_counter = Counter(nfkc(r.get("cmd_text")) for r in pos)
    wake_counter = Counter(nfkc(r.get("wake_text")) for r in asr)
    device_hits = Counter()
    for r in pos:
        t = nfkc(r.get("cmd_text"))
        for d in DEVICES:
            if d in t:
                device_hits[d] += 1

    # language matrix
    wake_cmd = Counter()
    wake_hyp_pos = Counter()
    wake_hyp_neg = Counter()
    for r in pos:
        f = r["_f"]
        wake_cmd[(f["wake_lang"], f["cmd_lang"])] += 1
        wake_hyp_pos[(f["wake_lang"], f["hyp_lang"])] += 1
    for r in neg:
        f = r["_f"]
        wake_hyp_neg[(f["wake_lang"], f["hyp_lang"])] += 1

    def rate(rows, key):
        n = len(rows)
        return round(sum(1 for r in rows if r["_f"][key]) / n, 6) if n else None

    def mean_len(rows):
        return mean([r["_f"]["hyp_len"] for r in rows])

    # CER by linguistic buckets (pos)
    def cer_bucket(pred):
        xs = [float(r["cer"]) for r in pos if pred(r) and r.get("cer") is not None]
        return {"n": len(xs), "mean": mean(xs), "p50": quantile(xs, 0.5)}

    buckets = {
        "all": cer_bucket(lambda r: True),
        "wake_cmd_match": cer_bucket(lambda r: not r["_f"]["wake_cmd_lang_mismatch"]),
        "wake_cmd_mismatch": cer_bucket(lambda r: r["_f"]["wake_cmd_lang_mismatch"]),
        "hyp_cmd_lang_mismatch": cer_bucket(lambda r: r["_f"]["cmd_hyp_lang_mismatch"]),
        "hyp_has_wake": cer_bucket(lambda r: r["_f"]["wake_in_hyp"]),
        "code_switch_hyp": cer_bucket(lambda r: r["_f"]["code_switch_hyp"]),
        "cmd_zh": cer_bucket(lambda r: r["_f"]["cmd_lang"] == "zh"),
        "cmd_en": cer_bucket(lambda r: r["_f"]["cmd_lang"] == "en"),
        "wake_zh": cer_bucket(lambda r: r["_f"]["wake_lang"] == "zh"),
        "wake_en": cer_bucket(lambda r: r["_f"]["wake_lang"] == "en"),
        "len_le4": cer_bucket(lambda r: r["_f"]["hyp_len"] <= 4),
        "len_5_12": cer_bucket(lambda r: 5 <= r["_f"]["hyp_len"] <= 12),
        "len_ge13": cer_bucket(lambda r: r["_f"]["hyp_len"] >= 13),
        "looks_command": cer_bucket(lambda r: r["_f"]["looks_command"]),
        "not_command": cer_bucket(lambda r: not r["_f"]["looks_command"]),
    }

    # substitution patterns: aligned strings
    sub_pairs = Counter()
    for r in pos:
        ra, ha = r.get("ref_aligned") or "", r.get("hyp_aligned") or ""
        if not ra or not ha or len(ra) != len(ha):
            continue
        i = 0
        while i < len(ra):
            if ra[i] == "〔" or (ra[i] not in "〔〕_" and ha[i] not in "〔〕_" and ra[i] != ha[i]):
                # skip; use simple char loop for 〔x〕
                pass
            i += 1
        # parse 〔x〕 markers
        def toks(s):
            out = []
            i = 0
            while i < len(s):
                if s[i] == "〔":
                    j = s.find("〕", i)
                    out.append(("diff", s[i + 1 : j] if j > 0 else s[i + 1 :]))
                    i = j + 1 if j > 0 else i + 1
                elif s[i] == "_":
                    out.append(("gap", "_"))
                    i += 1
                else:
                    out.append(("ok", s[i]))
                    i += 1
            return out
        rt, ht = toks(ra), toks(ha)
        if len(rt) == len(ht):
            for a, b in zip(rt, ht):
                if a[0] == "diff" and b[0] == "diff" and a[1] and b[1]:
                    sub_pairs[f"{a[1]}→{b[1]}"] += 1

    # top cmd ngrams
    ng2 = Counter()
    ng3 = Counter()
    for r in pos:
        t = nfkc(r.get("cmd_text"))
        ng2.update(ngrams(t, 2))
        ng3.update(ngrams(t, 3))

    # score vs linguistic features
    def score_of(r):
        sc = scores.get(r["uid"])
        return float(sc["presence_score"]) if sc else None

    def feat_by_split(key):
        out = {}
        for name, rows in (("pos", pos), ("neg", neg)):
            hit = [score_of(r) for r in rows if r["_f"][key] and score_of(r) is not None]
            miss = [score_of(r) for r in rows if (not r["_f"][key]) and score_of(r) is not None]
            out[name] = {
                "feat_rate": rate(rows, key),
                "score_when_true_mean": mean(hit),
                "score_when_false_mean": mean(miss),
                "n_true": len(hit),
                "n_false": len(miss),
            }
        return out

    feat_keys = [
        "empty",
        "looks_command",
        "wake_in_hyp",
        "wake_only",
        "wake_hyp_lang_mismatch",
        "code_switch_hyp",
        "has_device",
        "has_act",
        "has_digit",
    ]
    feat_sep = {k: feat_by_split(k) for k in feat_keys}

    # likelihood ratio style: P(feat|pos)/P(feat|neg)
    lr = {}
    for k in feat_keys:
        p = rate(pos, k) or 0
        n = rate(neg, k) or 0
        lr[k] = {
            "p_pos": p,
            "p_neg": n,
            "lr_pos": round(p / n, 3) if n else None,
            "abs_gap": round(p - n, 4),
        }

    # gray-zone usefulness: score near lang thr
    # first get full-set lang thrs
    zh_pos = [score_of(r) for r in pos if r.get("lang") == "zh" and score_of(r) is not None]
    zh_neg = [score_of(r) for r in neg if r.get("lang") == "zh" and score_of(r) is not None]
    en_pos = [score_of(r) for r in pos if r.get("lang") == "en" and score_of(r) is not None]
    en_neg = [score_of(r) for r in neg if r.get("lang") == "en" and score_of(r) is not None]
    zh_thr = sweep_thr(zh_pos, zh_neg)["thr"]
    en_thr = sweep_thr(en_pos, en_neg)["thr"]

    def in_gray(r, g=0.08):
        s = score_of(r)
        if s is None:
            return False
        thr = zh_thr if r.get("lang") == "zh" else en_thr
        return abs(s - thr) <= g

    gray_rows = [r for r in asr if in_gray(r, 0.08) and r["uid"] in scores]
    gray_pos = [r for r in gray_rows if r["split"] == "pos"]
    gray_neg = [r for r in gray_rows if r["split"] == "neg"]
    gray_lr = {}
    for k in feat_keys:
        p = rate(gray_pos, k) or 0
        n = rate(gray_neg, k) or 0
        gray_lr[k] = {"p_pos": p, "p_neg": n, "abs_gap": round(p - n, 4), "n_pos": len(gray_pos), "n_neg": len(gray_neg)}

    modes = [
        "score_only",
        "gray_reject_empty",
        "gray_reject_wake_only",
        "gray_reject_empty_or_wake",
        "gray_reject_not_cmd",
        "gray_accept_cmd",
    ]
    # full-set (optimistic) + holdout
    full_rules = []
    for g in (0.0, 0.06, 0.08, 0.10):
        for mode in modes:
            if mode == "score_only" and g != 0.0:
                continue
            full_rules.append(
                apply_rule(asr, scores, mode=mode, zh_thr=zh_thr, en_thr=en_thr, gray=g)
            )
    full_rules.sort(key=lambda x: -x["contest_real"])
    ho = holdout_eval(asr, scores, modes)

    # error taxonomy for pos with cer>0
    err_types = Counter()
    for r in pos:
        cer = r.get("cer")
        if cer is None or cer <= 0:
            continue
        f = r["_f"]
        cmd = nfkc(r.get("cmd_text"))
        hyp = nfkc(r.get("hyp_norm") or r.get("asr_text"))
        if not hyp:
            err_types["empty_hyp"] += 1
        elif f["cmd_hyp_lang_mismatch"]:
            err_types["lang_wrong"] += 1
        elif f["wake_in_hyp"]:
            err_types["wake_inserted"] += 1
        elif f["code_switch_hyp"]:
            err_types["code_switch"] += 1
        elif looks_command(hyp) and looks_command(cmd):
            # number / slot errors if both commands
            if DIGIT.search(cmd) and nfkc(cmd) != hyp:
                err_types["slot_or_paraphrase"] += 1
            else:
                err_types["paraphrase_or_sub"] += 1
        else:
            err_types["unrelated_hyp"] += 1

    # unique commands / speaker-independent schema
    doc = {
        "meta": {
            "n": len(asr),
            "n_pos": len(pos),
            "n_neg": len(neg),
            "n_with_score": sum(1 for r in asr if r["uid"] in scores),
            "status": dict(status),
            "model": pos[0].get("model") if pos else None,
            "arm": "no_sep/mix",
            "note": "lang 字段来自唤醒词，不是命令语言",
        },
        "pos_asr": {
            "cer_mean": mean(pos_cer),
            "cer_p50": quantile(pos_cer, 0.5),
            "cer_p90": quantile(pos_cer, 0.9),
            "hist": hist_cer(pos_cer),
            "exact_match": round(sum(1 for c in pos_cer if c <= 0) / len(pos_cer), 6),
            "empty_rate": rate(pos, "empty"),
            "mean_hyp_len": mean_len(pos),
            "mean_cmd_len": mean([len(nfkc(r.get("cmd_text"))) for r in pos]),
            "unique_cmd": len(cmd_counter),
            "top_cmd": cmd_counter.most_common(15),
            "top_device": device_hits.most_common(15),
            "top_bigrams": ng2.most_common(12),
            "top_trigrams": ng3.most_common(12),
            "top_subs": sub_pairs.most_common(20),
            "error_taxonomy": dict(err_types),
            "cer_buckets": buckets,
        },
        "wakes": {
            "unique": len(wake_counter),
            "top": wake_counter.most_common(20),
        },
        "lang_matrix": {
            "wake_x_cmd_pos": {f"{a}->{b}": n for (a, b), n in wake_cmd.most_common()},
            "wake_x_hyp_pos": {f"{a}->{b}": n for (a, b), n in wake_hyp_pos.most_common()},
            "wake_x_hyp_neg": {f"{a}->{b}": n for (a, b), n in wake_hyp_neg.most_common()},
            "pos_wake_cmd_mismatch_rate": rate(pos, "wake_cmd_lang_mismatch"),
            "pos_wake_hyp_mismatch_rate": rate(pos, "wake_hyp_lang_mismatch"),
            "neg_wake_hyp_mismatch_rate": rate(neg, "wake_hyp_lang_mismatch"),
        },
        "neg_asr": {
            "empty_rate": rate(neg, "empty"),
            "mean_hyp_len": mean_len(neg),
            "looks_command": rate(neg, "looks_command"),
            "wake_in_hyp": rate(neg, "wake_in_hyp"),
            "wake_only": rate(neg, "wake_only"),
            "has_device": rate(neg, "has_device"),
            "code_switch": rate(neg, "code_switch_hyp"),
            "top_hyp": Counter(nfkc(r.get("hyp_norm") or r.get("asr_text") or "") or "<EMPTY>" for r in neg).most_common(20),
        },
        "feature_separation": {"full": lr, "by_score_mean": feat_sep, "gray_band_0.08": gray_lr},
        "presence_lang_thr": {
            "zh": sweep_thr(zh_pos, zh_neg),
            "en": sweep_thr(en_pos, en_neg),
            "n_gray_0.08": len(gray_rows),
            "gray_pos": len(gray_pos),
            "gray_neg": len(gray_neg),
        },
        "fusion_fullset_optimistic": full_rules[:10],
        "fusion_holdout_30": ho,
        "practice": {},
    }

    # recommended practice from holdout
    so = ho["score_only"]
    better = [x for x in ho["all"] if x["test_contest"] >= so["test_contest"] - 1e-9]
    best = ho["top"][0]
    doc["practice"] = {
        "do_not": [
            "不要用 cmd_text / CER 当运行时拒识特征（测试集没有识别文本标签）。",
            "不要用「像不像家电命令」做硬拒识：neg 里也大量是别人的真实命令。",
            "不要按唤醒词语言切命令语言阈值当语种真相：本数据唤醒与命令经常跨语。",
            "不要在全量上扫 ASR 规则再报竞赛分（这是全部训练数据，会过拟合）。",
        ],
        "do": [
            "声纹仍是主门：lang_split 只按唤醒词分阈值（enroll 侧），不把 hyp 语种当说话人语种。",
            "ASR 只允许在余弦灰区做小动作，且只用运行时可知量：空转写、是否几乎只有唤醒词。",
            "阈值与灰区宽度在 holdout 上锁死，提交前不再用全量重扫。",
            "接受后的波形仍走 mix ASR；语言学 lexicon 只用于灰区，不改变提取。",
        ],
        "holdout_score_only": so,
        "holdout_best": best,
        "holdout_best_beats_score_only": best["test_contest"] - so["test_contest"],
    }

    out = Path(r"d:\media\datasetA\sssss\linguistics_analysis.json")
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: doc[k] for k in ("meta", "practice")}, ensure_ascii=False, indent=2))
    print("wrote", out)
    print("pos cer", doc["pos_asr"]["cer_mean"], "exact", doc["pos_asr"]["exact_match"])
    print("lang matrix", doc["lang_matrix"]["wake_x_cmd_pos"])
    print("lr", json.dumps(lr, ensure_ascii=False, indent=2))
    print("holdout best", best)
    print("score_only", so)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

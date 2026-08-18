#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

PUNCT = str.maketrans("", "", "，。！？、；：\"\"''「」『』（）【】《》…·—–-.,!?;:\"'`")
ACTS = ["打开", "关闭", "关掉", "开启", "暂停", "调到", "调成", "调高", "调低", "设置", "切换", "启动", "关机", "开机", "关上"]
TTS = ["主人当前", "语音功能已关闭", "遥控器"]
QA = ["吃什么", "哪些食物"]
MEDIA = ["播放", "下一首", "观影"]
HVAC = ["空调", "风速", "温度", "制热", "制冷", "防直吹"]


def nfkc(s) -> str:
    if not s:
        return ""
    t = unicodedata.normalize("NFKC", str(s))
    t = "".join(ch for ch in t if not ch.isspace()).translate(PUNCT)
    return t.lower().strip()


def genre(t: str) -> str:
    if any(x in t for x in TTS):
        return "device_tts"
    if any(x in t for x in QA):
        return "food_qa"
    if any(x in t for x in MEDIA):
        return "media"
    if any(x in t for x in HVAC):
        return "hvac"
    if any(x in t for x in ACTS):
        return "imperative"
    if len(t) >= 16:
        return "long_other"
    return "short_other"


def q(xs, p):
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[i]


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def main() -> int:
    asr = json.loads(Path(r"d:\media\datasetA\sssss\no_sep.json").read_text(encoding="utf-8"))
    scores = {}
    for line in Path(r"d:\media\datasetA\sssss\scores (1).jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            scores[r["uid"]] = r
    for r in asr:
        r["h"] = nfkc(r.get("hyp_norm") or r.get("asr_text"))
        r["c"] = nfkc(r.get("cmd_text") or "")
        r["g"] = genre(r["h"])
        r["s"] = float(scores[r["uid"]]["presence_score"])
        r["len"] = len(r["h"])
    pos = [r for r in asr if r["split"] == "pos"]
    neg = [r for r in asr if r["split"] == "neg"]

    def len_stats(rows):
        xs = [r["len"] for r in rows]
        return {k: round(v, 3) for k, v in {
            "mean": mean(xs), "p25": q(xs, 0.25), "p50": q(xs, 0.5),
            "p75": q(xs, 0.75), "p90": q(xs, 0.9), "p95": q(xs, 0.95),
        }.items()}

    bins = [(0, 4), (5, 8), (9, 12), (13, 16), (17, 24), (25, 99)]
    len_bin = []
    for a, b in bins:
        pp = sum(1 for r in pos if a <= r["len"] <= b) / len(pos)
        nn = sum(1 for r in neg if a <= r["len"] <= b) / len(neg)
        len_bin.append({"bin": f"{a}-{b}", "pos": round(pp, 4), "neg": round(nn, 4), "gap": round(pp - nn, 4)})

    xs, ys = [], []
    for r in pos:
        if r.get("cer") is not None:
            xs.append(r["s"])
            ys.append(float(r["cer"]))
    mx, my = mean(xs), mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    corr = round(num / den, 4) if den else None
    sc = sorted(xs)
    qs = [q(sc, p) for p in (0, 0.25, 0.5, 0.75)]
    cer_q = []
    for i in range(4):
        lo, hi = qs[i], (qs[i + 1] if i < 3 else 9)
        sel = [float(r["cer"]) for r in pos if r.get("cer") is not None and lo <= r["s"] < hi]
        cer_q.append({"q": i + 1, "score_lo": round(lo, 4), "n": len(sel), "cer": round(mean(sel), 4)})

    def contest_real(rows, pred):
        pos_c = []
        n_neg = n_rej = 0
        for r in rows:
            rej = pred(r)
            if r["split"] == "pos":
                pos_c.append(1.0 if rej else float(r.get("cer") if r.get("cer") is not None else 1))
            else:
                n_neg += 1
                if rej:
                    n_rej += 1
        rr = n_neg and n_rej / n_neg
        cer = mean(pos_c)
        frr = sum(1 for r in rows if r["split"] == "pos" and pred(r)) / sum(1 for r in rows if r["split"] == "pos")
        return {"rr": rr, "frr": frr, "cer": cer, "contest": 0.5 * rr + 0.5 * (1 - cer)}

    rng = random.Random(7)
    by = defaultdict(list)
    for r in asr:
        by[(r["split"], r.get("lang") or "zh")].append(r)
    train, test = [], []
    for g in by.values():
        g = list(g)
        rng.shuffle(g)
        n = max(1, int(round(len(g) * 0.3)))
        test.extend(g[:n])
        train.extend(g[n:])

    def lang_thr(subset):
        out = {}
        for lang in ("zh", "en"):
            pos_s = [r["s"] for r in subset if r["split"] == "pos" and (r.get("lang") or "zh") == lang]
            neg_s = [r["s"] for r in subset if r["split"] == "neg" and (r.get("lang") or "zh") == lang]
            best = None
            for thr in sorted(set(round(x, 4) for x in pos_s + neg_s)):
                frr = sum(x < thr for x in pos_s) / len(pos_s)
                rr = sum(x < thr for x in neg_s) / len(neg_s)
                pxy = 0.5 * rr + 0.5 * (1 - frr)
                if best is None or pxy > best[0]:
                    best = (pxy, thr)
            out[lang] = best[1]
        return out["zh"], out["en"]

    zt, et = lang_thr(train)

    def thr_of(r):
        return zt if (r.get("lang") or "zh") == "zh" else et

    def score_only(r):
        return r["s"] < thr_of(r)

    def gray(r, g):
        return abs(r["s"] - thr_of(r)) <= g

    specs = [("score_only", score_only)]
    for g in (0.05, 0.08, 0.10):
        specs.append((f"gray{g}_len>=18", lambda r, g=g: True if gray(r, g) and r["len"] >= 18 else score_only(r)))
        specs.append((f"gray{g}_len>=16", lambda r, g=g: True if gray(r, g) and r["len"] >= 16 else score_only(r)))
        specs.append((f"gray{g}_tts", lambda r, g=g: True if gray(r, g) and r["g"] == "device_tts" else score_only(r)))
        specs.append((f"gray{g}_noact", lambda r, g=g: True if gray(r, g) and not any(a in r["h"] for a in ACTS) else score_only(r)))
        specs.append((f"gray{g}_digit", lambda r, g=g: True if gray(r, g) and bool(re.search(r"\d|[零一二两三四五六七八九十]", r["h"])) else score_only(r)))
    specs.append(("hard_tts", lambda r: True if r["g"] == "device_tts" else score_only(r)))
    specs.append(("hard_len>=24", lambda r: True if r["len"] >= 24 else score_only(r)))
    specs.append(("hard_len>=20", lambda r: True if r["len"] >= 20 else score_only(r)))

    rows = []
    for name, fn in specs:
        tr, te = contest_real(train, fn), contest_real(test, fn)
        rows.append({
            "name": name,
            "train_contest": round(tr["contest"], 4),
            "test_contest": round(te["contest"], 4),
            "train_rr": round(tr["rr"], 4),
            "test_rr": round(te["rr"], 4),
            "train_frr": round(tr["frr"], 4),
            "test_frr": round(te["frr"], 4),
            "train_cer": round(tr["cer"], 4),
            "test_cer": round(te["cer"], 4),
            "gap": round(tr["contest"] - te["contest"], 4),
        })
    rows.sort(key=lambda x: -x["test_contest"])

    gp = Counter(r["g"] for r in pos)
    gn = Counter(r["g"] for r in neg)
    gc = Counter(genre(r["c"]) for r in pos)

    out = {
        "len_pos": len_stats(pos),
        "len_neg": len_stats(neg),
        "len_bin": len_bin,
        "genre_hyp_pos": dict(gp),
        "genre_hyp_neg": dict(gn),
        "genre_cmd_pos": dict(gc),
        "tts_pos": sum(1 for r in pos if r["g"] == "device_tts"),
        "tts_neg": sum(1 for r in neg if r["g"] == "device_tts"),
        "pos_len_ge20": {
            "n": sum(1 for r in pos if r["len"] >= 20),
            "cer": round(mean([float(r["cer"]) for r in pos if r["len"] >= 20 and r.get("cer") is not None]) or 0, 4),
        },
        "neg_len_ge20": sum(1 for r in neg if r["len"] >= 20),
        "corr_score_cer": corr,
        "cer_by_score_quartile": cer_q,
        "holdout_thrs": {"zh": zt, "en": et, "n_train": len(train), "n_test": len(test)},
        "holdout": rows,
        "score_only": next(x for x in rows if x["name"] == "score_only"),
    }
    p = Path(r"d:\media\datasetA\sssss\linguistics_pass2.json")
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("len_pos", "len_neg", "corr_score_cer", "cer_by_score_quartile", "genre_hyp_pos", "genre_hyp_neg", "tts_pos", "tts_neg", "pos_len_ge20", "neg_len_ge20", "holdout_thrs", "score_only")}, ensure_ascii=False, indent=2))
    print("--- holdout ---")
    for r in rows[:12]:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

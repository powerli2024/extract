#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""融合余弦 lang_split 与 v2 文本判别，更新 result.json 并计算竞赛分。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refine_linguistic_patterns import classify_v2  # noqa: E402

ROOT = Path(r"d:\media\datasetA\sssss")
ZH_THR = 0.29305
EN_THR = 0.357868
DELTA = 0.05
LEN_L = 16


def thr_of(lang: str) -> float:
    return ZH_THR if (lang or "zh") == "zh" else EN_THR


def load() -> tuple[list, list, dict, dict]:
    asr = json.loads((ROOT / "no_sep.json").read_text(encoding="utf-8"))
    result = json.loads((ROOT / "result.json").read_text(encoding="utf-8"))
    scores = {}
    for line in (ROOT / "scores (1).jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            scores[r["uid"]] = r
    pos = [x for x in asr if x["split"] == "pos"]
    neg = [x for x in asr if x["split"] == "neg"]
    for r in pos + neg:
        r["score"] = float(scores[r["uid"]]["presence_score"])
        r["thr"] = thr_of(r.get("lang") or "zh")
        r["v2"] = classify_v2(r.get("asr_text") or r.get("hyp_norm"))
        r["base_rej"] = r["score"] < r["thr"]
    by_id = {x["id"]: x for x in result["results"]}
    return pos, neg, by_id, result


def fused_reject(r: dict, *, rescue: bool, extra: bool) -> bool:
    """灰区双向修正：近阈短任务句救回；近阈长句/非任务加拒。"""
    below = r["base_rej"]
    gray = abs(r["score"] - r["thr"]) <= DELTA
    v2 = r["v2"]
    short_task = v2["task_oriented"] and v2["len"] < LEN_L
    long_or_nontask = (v2["len"] >= LEN_L) or (not v2["task_oriented"])
    if extra and (not below) and gray and long_or_nontask:
        return True
    if rescue and below and gray and short_task:
        return False
    return below


def metrics(pos, neg, pred) -> dict:
    pos_c = []
    n_fr = 0
    for r in pos:
        rej = pred(r)
        if rej:
            n_fr += 1
            pos_c.append(1.0)
        else:
            c = r.get("cer")
            pos_c.append(1.0 if c is None else float(c))
    n_rej = sum(1 for r in neg if pred(r))
    rr = n_rej / len(neg)
    frr = n_fr / len(pos)
    cer = sum(pos_c) / len(pos_c)
    return {
        "rr": rr,
        "frr": frr,
        "n_fr": n_fr,
        "n_rej_neg": n_rej,
        "cer": cer,
        "contest": 0.5 * rr + 0.5 * (1.0 - cer),
    }


def main() -> int:
    pos, neg, by_id, result = load()
    variants = {
        "baseline": lambda r: r["base_rej"],
        "rescue_only": lambda r: fused_reject(r, rescue=True, extra=False),
        "extra_only": lambda r: fused_reject(r, rescue=False, extra=True),
        "both": lambda r: fused_reject(r, rescue=True, extra=True),
    }
    table = {k: metrics(pos, neg, fn) for k, fn in variants.items()}

    pred = variants["both"]
    recovered = []
    new_neg = []
    extra_pos_rej = []
    unrescue_neg = []  # rescued neg (RR 损失)

    for r in pos:
        b, f = r["base_rej"], pred(r)
        if b and not f:
            recovered.append(r)
        if (not b) and f:
            extra_pos_rej.append(r)
    for r in neg:
        b, f = r["base_rej"], pred(r)
        if (not b) and f:
            new_neg.append(r)
        if b and not f:
            unrescue_neg.append(r)

    # 写 result.json：按融合决策重填 pos
    new_results = []
    pos_by_id = {p["id"]: p for p in pos}
    for old in result["results"]:
        p = pos_by_id[old["id"]]
        rej = pred(p)
        if rej:
            new_results.append({
                "id": old["id"],
                "content": "",
                "label": old["label"],
                "cer": 1.0,
            })
        else:
            hyp = p.get("asr_text") if p.get("asr_text") is not None else (old.get("content") or "")
            cer = p.get("cer")
            if cer is None:
                cer = 1.0 if not str(hyp).strip() else float(old.get("cer") or 1.0)
            new_results.append({
                "id": old["id"],
                "content": hyp or "",
                "label": old["label"],
                "cer": float(cer),
            })

    m = table["both"]
    out = {
        "results": new_results,
        "avg_cer": m["cer"],
        "avg_rr": m["rr"],
        "contest": m["contest"],
    }
    (ROOT / "result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    def brief(rows, k=8):
        xs = []
        for r in rows[:k]:
            xs.append({
                "uid": r["uid"],
                "id": r["id"],
                "lang": r.get("lang"),
                "score": round(r["score"], 4),
                "thr": r["thr"],
                "primary": r["v2"]["primary"],
                "len": r["v2"]["len"],
                "task": r["v2"]["task_oriented"],
                "asr": (r.get("asr_text") or "")[:40],
                "cmd": r.get("cmd_text"),
                "cer": r.get("cer"),
            })
        return xs

    sidecar = {
        "policy": {
            "baseline": "lang_split raw_max",
            "zh_thr": ZH_THR,
            "en_thr": EN_THR,
            "gray_delta": DELTA,
            "len_L": LEN_L,
            "rescue": "灰区且 score<τ 且 短任务句 → 接受",
            "extra_reject": "灰区且 score≥τ 且 (len≥16 或非任务) → 拒识",
        },
        "baseline": table["baseline"],
        "variants": table,
        "n_recovered_pos": len(recovered),
        "n_extra_pos_reject": len(extra_pos_rej),
        "n_new_neg_reject": len(new_neg),
        "n_unrescued_neg": len(unrescue_neg),
        "recovered_pos": brief(recovered, 30),
        "new_neg_reject": brief(new_neg, 20),
        "extra_pos_reject": brief(extra_pos_rej, 15),
        "contest": m["contest"],
        "avg_cer": m["cer"],
        "avg_rr": m["rr"],
    }
    (ROOT / "result_fusion_sidecar.json").write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps({
        "variants": {k: {kk: round(vv, 6) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in table.items()},
        "n_recovered_pos": len(recovered),
        "n_new_neg_reject": len(new_neg),
        "n_extra_pos_reject": len(extra_pos_rej),
        "n_unrescued_neg": len(unrescue_neg),
        "final_contest": round(m["contest"], 6),
        "final_avg_cer": round(m["cer"], 6),
        "final_avg_rr": round(m["rr"], 6),
        "baseline_contest": round(table["baseline"]["contest"], 6),
        "recovered_ids": [r["id"] for r in recovered],
        "new_neg_uids": [r["uid"] for r in new_neg],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

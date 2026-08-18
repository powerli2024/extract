#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下一刀离线验收：锁定门控上的 CER=1 桶、T1 对照 jsonl、T4 camp/窗否决（只加拒）。

不改部署 τ。验收只认真实 contest；T4 在 holdout 上选 margin。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

ROOT_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(ROOT_SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lift_common import (  # noqa: E402
    CAMP_VETO_MARGIN,
    extra_reject_text,
    camp_veto,
    contest_metrics,
    round_metrics,
    stratified_holdout,
    window_veto,
)
from refine_linguistic_patterns import classify_v2  # noqa: E402

DEFAULT_SSSSS = Path(r"d:\media\datasetA\sssss")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_locked_rows(sssss: Path) -> list[dict[str, Any]]:
    asr = json.loads((sssss / "no_sep.json").read_text(encoding="utf-8"))
    sims = {r["uid"]: r for r in load_jsonl(sssss / "encoder_sims.jsonl")}
    scores: dict[str, dict] = {}
    sp = sssss / "scores (1).jsonl"
    if sp.is_file():
        for r in load_jsonl(sp):
            scores[r["uid"]] = r
    rows = []
    for r in asr:
        uid = r["uid"]
        sim = sims.get(uid) or {}
        sc = scores.get(uid) or {}
        eres = float(sim.get("eres_sep_once") or sc.get("presence_score") or 0.0)
        camp = sim.get("camp_sep_once")
        camp_f = None if camp is None else float(camp)
        thr = float(sim.get("locked_thr") or (0.29305 if (r.get("lang") or "zh") == "zh" else 0.357868))
        hyp = r.get("asr_text") or r.get("hyp_norm") or ""
        v2 = classify_v2(hyp)
        rows.append({
            **r,
            "score": eres,
            "eres": eres,
            "camp": camp_f,
            "thr": thr,
            "hyp": hyp,
            "v2": v2,
            "base_rej": eres < thr,
        })
    return rows


def locked_pred(r: dict[str, Any]) -> bool:
    if r["base_rej"]:
        return True
    return extra_reject_text(r["score"], r["thr"], r.get("hyp"))


def apply_alt_cer(rows: list[dict[str, Any]], alt_path: Path) -> list[dict[str, Any]]:
    """T1：用另一套 mix 转写替换已接受 pos 的 CER；门控不变。"""
    by_uid: dict[str, dict] = {}
    if alt_path.suffix == ".json":
        blob = json.loads(alt_path.read_text(encoding="utf-8"))
        items = blob.get("results") or blob.get("asr") or blob
        if isinstance(items, list):
            for it in items:
                uid = it.get("uid") or (f"pos_{it['id']}" if "id" in it else None)
                if uid:
                    by_uid[str(uid)] = it
    else:
        for it in load_jsonl(alt_path):
            if it.get("uid"):
                by_uid[str(it["uid"])] = it
    out = []
    for r in rows:
        n = dict(r)
        alt = by_uid.get(str(r["uid"]))
        if alt is not None and r.get("split") == "pos":
            if alt.get("cer") is not None:
                n["cer"] = float(alt["cer"])
            if alt.get("asr_text") is not None:
                n["hyp"] = alt["asr_text"]
                n["asr_text"] = alt["asr_text"]
        out.append(n)
    return out


def pred_with_veto(margin: float, *, use_camp: bool, use_window: bool):
    def pred(r: dict[str, Any]) -> bool:
        if locked_pred(r):
            return True
        if use_camp and camp_veto(r["eres"], r.get("camp"), r["thr"], margin=margin):
            return True
        if use_window:
            bw = r.get("best_window_score")
            sw = r.get("second_window_score")
            if window_veto(bw, sw, r["thr"], margin=margin):
                return True
        return False
    return pred


def extra_counts(rows, pred) -> dict[str, int]:
    n_extra_pos = n_extra_neg = 0
    for r in rows:
        base, now = locked_pred(r), pred(r)
        if (not base) and now:
            if r["split"] == "pos":
                n_extra_pos += 1
            else:
                n_extra_neg += 1
    return {"n_extra_pos": n_extra_pos, "n_extra_neg": n_extra_neg}


def main() -> int:
    p = argparse.ArgumentParser(description="T1/T4 离线真实 contest")
    p.add_argument("--sssss", type=Path, default=DEFAULT_SSSSS)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--alt-asr", type=Path, default=None,
                   help="T1 另一套 asr jsonl/json（uid+cer）；门控锁死")
    p.add_argument("--alt-name", default="alt")
    p.add_argument("--holdout-frac", type=float, default=0.3)
    p.add_argument("--holdout-seed", type=int, default=7)
    args = p.parse_args()

    sssss = args.sssss.expanduser().resolve()
    rows = load_locked_rows(sssss)
    locked = contest_metrics(rows, locked_pred)
    cosine_only = contest_metrics(rows, lambda r: r["base_rej"])

    # T1 基线：接受集合 CER=1 桶（auto / 现有 mix）
    accept_pos = [r for r in rows if r["split"] == "pos" and not locked_pred(r)]
    t1_auto = {
        "name": "locked+mix_auto",
        **round_metrics(locked, 4),
        "n_accept_pos": len(accept_pos),
        "cer_hist_accept": locked["cer_hist_accept"],
        "n_cer1_accept": locked["n_cer1_accept"],
        "note": "language=auto, context=none；Chinese/领域对照须 --alt-asr 或 run_asr_cer.sh",
    }

    t1_alt = None
    if args.alt_asr:
        alt_rows = apply_alt_cer(rows, args.alt_asr)
        m = contest_metrics(alt_rows, locked_pred)
        t1_alt = {
            "name": args.alt_name,
            **round_metrics(m, 4),
            "d_contest": round(m["contest"] - locked["contest"], 4),
            "d_cer1_accept": int(m["n_cer1_accept"] - locked["n_cer1_accept"]),
        }

    train, test = stratified_holdout(rows, args.holdout_frac, args.holdout_seed)
    margins = [0.08, 0.10, 0.12, 0.15, 0.20]
    t4_grid = []
    for mg in margins:
        pred = pred_with_veto(mg, use_camp=True, use_window=False)
        tr, te = contest_metrics(train, pred), contest_metrics(test, pred)
        rec = {
            "margin": mg,
            "train": round_metrics(tr, 4),
            "test": round_metrics(te, 4),
            "full": round_metrics(contest_metrics(rows, pred), 4),
            "train_dC": round(tr["contest"] - contest_metrics(train, locked_pred)["contest"], 4),
            "test_dC": round(te["contest"] - contest_metrics(test, locked_pred)["contest"], 4),
            **extra_counts(rows, pred),
        }
        t4_grid.append(rec)

    # 选 train 涨分、test 不掉、额外 pos 误拒尽量少
    viable = [g for g in t4_grid if g["train_dC"] >= 0 and g["test_dC"] >= 0]
    if viable:
        best_t4 = max(viable, key=lambda g: (g["test_dC"], g["train_dC"], -g["n_extra_pos"]))
    else:
        best_t4 = {
            "margin": None,
            "note": "全部 margin 的 train 或 test 真实 contest 下降；不接入 VETO_CAMP",
        }
    go = bool(
        viable
        and best_t4.get("test_dC", -1) >= 0.005
        and best_t4.get("n_extra_pos", 99) <= 5
    )

    out = {
        "n": len(rows),
        "cosine_only": round_metrics(cosine_only, 4),
        "locked": t1_auto,
        "t1_alt": t1_alt,
        "t4_camp_veto": {
            "grid": t4_grid,
            "best": best_t4,
            "go": go,
            "go_rule": "holdout test_dC>=0.005 且额外 pos 误拒<=5；只否决不救援",
            "default_margin": CAMP_VETO_MARGIN,
        },
        "closed": [
            "不救 FN",
            "不换全量 TSE",
            "不把代理 contest 当提交分",
        ],
    }
    out_path = args.out or (sssss / "next_lift_eval.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[OK] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

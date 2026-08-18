#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从当前锁死点继续加严拒识，看竞赛分是否还能涨。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimize_text_reject import load_all, make_pred, metrics, stratified_split

ROOT = Path(r"d:\media\datasetA\sssss")


def sim_only(add: float):
    def pred(r):
        return r["score"] < (r["thr"] + add)
    return pred


def overlay(delta: float, L: int, nontask: bool, tau_add: float = 0.0):
    def pred(r):
        thr = r["thr"] + tau_add
        if r["score"] < thr:
            return True
        m = r["score"] - thr
        ok_text = r["v2"]["len"] >= L and ((not r["v2"]["task_oriented"]) if nontask else True)
        if 0 <= m <= delta and ok_text:
            return True
        return False
    return pred


def row(name, m, ref):
    return {
        "name": name,
        "contest": round(m["contest"], 4),
        "dC": round(m["contest"] - ref["contest"], 4),
        "rr": round(m["rr"], 3),
        "frr": round(m["frr"], 3),
        "cer": round(m["cer"], 3),
        "n_fr": m["n_fr"],
        "n_rej_neg": m["n_rej_neg"],
    }


def main() -> int:
    rows = load_all()
    train, test = stratified_split(rows)
    base = metrics(rows, lambda r: r["base_rej"])
    locked_pred = make_pred("len_and_nontask_gray", 0.10, 15)
    locked = metrics(rows, locked_pred)

    tau_full = []
    for add in (0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25):
        tau_full.append(row(f"tau+{add:.2f}", metrics(rows, sim_only(add)), base))

    tau_on_locked = []
    for add in (0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10):
        tau_on_locked.append(row(f"lock+tau+{add:.2f}", metrics(rows, overlay(0.10, 15, True, add)), locked))

    text_full = []
    specs = [
        (0.10, 15, True),
        (0.12, 15, True),
        (0.15, 15, True),
        (0.20, 15, True),
        (0.10, 14, True),
        (0.10, 13, True),
        (0.10, 12, True),
        (0.10, 15, False),
        (0.08, 15, False),
        (0.12, 14, False),
        (0.10, 12, False),
        (0.20, 12, False),
    ]
    btr = metrics(train, locked_pred)
    bte = metrics(test, locked_pred)
    for d, L, nt in specs:
        pred = overlay(d, L, nt, 0)
        m = metrics(rows, pred)
        tr = metrics(train, pred)
        te = metrics(test, pred)
        rec = row(f"d={d:.2f}_L={L}_nt={int(nt)}", m, locked)
        rec["train_dC"] = round(tr["contest"] - btr["contest"], 4)
        rec["test_dC"] = round(te["contest"] - bte["contest"], 4)
        text_full.append(rec)

    tau_ho = []
    tbase_tr = metrics(train, lambda r: r["base_rej"])
    tbase_te = metrics(test, lambda r: r["base_rej"])
    for add in (0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20):
        tr = metrics(train, sim_only(add))
        te = metrics(test, sim_only(add))
        tau_ho.append({
            "add": add,
            "train_C": round(tr["contest"], 4),
            "test_C": round(te["contest"], 4),
            "train_dC": round(tr["contest"] - tbase_tr["contest"], 4),
            "test_dC": round(te["contest"] - tbase_te["contest"], 4),
            "test_rr": round(te["rr"], 3),
            "test_frr": round(te["frr"], 3),
        })

    # remaining FA after locked: if we reject ALL remaining FA, what's the FRR cost of matching score region?
    remaining_fa = [r for r in rows if r["split"] == "neg" and not locked_pred(r)]
    remaining_tp = [r for r in rows if r["split"] == "pos" and not locked_pred(r)]
    out = {
        "locked": row("locked", locked, base),
        "baseline": row("baseline", base, base),
        "n_remaining_fa": len(remaining_fa),
        "n_remaining_tp": len(remaining_tp),
        "tau_only": tau_full,
        "raise_tau_on_locked": tau_on_locked,
        "stricter_text": text_full,
        "holdout_tau_only": tau_ho,
        "best_tau_only": max(tau_full, key=lambda x: x["contest"]),
        "best_text": max(text_full, key=lambda x: x["contest"]),
    }
    (ROOT / "stricter_reject_sweep.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

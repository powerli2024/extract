#!/usr/bin/env python3
"""Optimize Presence policy directly for the official RR/CER score.

Requires ASR results for *all positive CMDs* from one fixed downstream audio
backend (normally full mix).  This avoids the invalid CER:=FRR proxy and makes
lowering the Presence threshold auditable: every newly accepted positive has a
real edit distance.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from paths import normalize_presence_label


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stream_score(
    row: dict[str, Any],
    policy: str,
    *,
    rescue_high_margin: float = 0.08,
    rescue_floor_margin: float = 0.10,
    rescue_dominance: float = 0.05,
) -> float:
    sims = row.get("sim_streams") or {}
    mix = float(sims.get("mix", row.get("sim_enroll_mix", 0.0)))
    sep = sorted(
        (float(v) for k, v in sims.items() if k not in {"mix", "mix_window", "peak"}),
        reverse=True,
    )
    if policy == "mix":
        return mix
    if policy == "max":
        return max([mix, *sep])
    if policy != "strict_rescue":
        raise ValueError(f"unknown policy={policy}")
    if len(sep) >= 2 and sep[0] - sep[1] >= rescue_dominance:
        return max(mix, min(sep[0] - rescue_high_margin, mix + rescue_floor_margin))
    return mix


def asr_cost(row: dict[str, Any] | None) -> tuple[int, int]:
    """Return (accepted errors, reference chars); ASR failure is CER=1."""
    if not row:
        raise KeyError("missing ASR row")
    n = int(row.get("n") or 0)
    if n <= 0:
        raise ValueError(f"invalid ASR ref length uid={row.get('uid')} n={n}")
    if row.get("status") != "ok" or row.get("edit_distance") is None:
        return n, n
    return int(row["edit_distance"]), n


def evaluate(
    rows: list[dict[str, Any]],
    asr: dict[str, dict[str, Any]],
    *,
    policy: str,
    thresholds: dict[str, float],
    margins: tuple[float, float, float] = (0.08, 0.10, 0.05),
) -> dict[str, Any]:
    n_pos = n_neg = n_neg_reject = n_pos_reject = errors = refs = 0
    unknown_asr: list[str] = []
    high, floor, dominance = margins
    for row in rows:
        label = normalize_presence_label(row.get("label"), split=row.get("split"))
        lang = str(row.get("lang") or "zh")
        score = stream_score(
            row,
            policy,
            rescue_high_margin=high,
            rescue_floor_margin=floor,
            rescue_dominance=dominance,
        )
        rejected = score < float(thresholds.get(lang, thresholds.get("default", 0.0)))
        if label == "absent":
            n_neg += 1
            n_neg_reject += int(rejected)
            continue
        n_pos += 1
        n_pos_reject += int(rejected)
        uid = str(row["uid"])
        try:
            accepted_errors, n = asr_cost(asr.get(uid))
        except (KeyError, ValueError):
            accepted_errors, n = 0, 0
            unknown_asr.append(uid)
        if n <= 0:
            continue
        refs += n
        errors += n if rejected else accepted_errors
    rr = n_neg_reject / max(1, n_neg)
    cer = errors / max(1, refs)
    return {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "rr": rr,
        "frr": n_pos_reject / max(1, n_pos),
        "cer_pos_micro": cer,
        "pos_errors": errors,
        "pos_ref_chars": refs,
        "contest_score": 0.5 * rr + 0.5 * (1.0 - cer),
        "missing_asr_uids": unknown_asr,
    }


def optimize_thresholds(
    rows: list[dict[str, Any]],
    asr: dict[str, dict[str, Any]],
    *,
    policy: str,
    margins: tuple[float, float, float] = (0.08, 0.10, 0.05),
) -> dict[str, float]:
    """Optimize independent zh/en thresholds for the official additive score."""
    n_neg = sum(
        normalize_presence_label(r.get("label"), split=r.get("split")) == "absent"
        for r in rows
    )
    total_ref = 0
    for row in rows:
        if normalize_presence_label(row.get("label"), split=row.get("split")) == "present":
            _err, n = asr_cost(asr.get(str(row["uid"])))
            total_ref += n
    if not n_neg or not total_ref:
        raise ValueError("threshold optimization requires both negatives and positive ASR refs")

    high, floor, dominance = margins
    result: dict[str, float] = {}
    for lang in ("zh", "en"):
        events: list[tuple[float, float]] = []
        for row in rows:
            if str(row.get("lang") or "zh") != lang:
                continue
            score = stream_score(
                row,
                policy,
                rescue_high_margin=high,
                rescue_floor_margin=floor,
                rescue_dominance=dominance,
            )
            label = normalize_presence_label(row.get("label"), split=row.get("split"))
            if label == "absent":
                delta = -0.5 / n_neg
            else:
                err, n = asr_cost(asr.get(str(row["uid"])))
                # All-reject is the starting point. Accepting saves n-err.
                delta = 0.5 * (n - err) / total_ref
            events.append((score, delta))
        if not events:
            continue
        events.sort(key=lambda x: x[0], reverse=True)
        gain = 0.0
        # Above max => reject all. Prefer higher threshold on exact ties.
        best_gain = 0.0
        best_thr = events[0][0] + 1e-6
        i = 0
        while i < len(events):
            score = events[i][0]
            while i < len(events) and events[i][0] == score:
                gain += events[i][1]
                i += 1
            if gain > best_gain + 1e-15 or (
                abs(gain - best_gain) <= 1e-15 and score > best_thr
            ):
                best_gain, best_thr = gain, score
        result[lang] = float(best_thr)
    if "zh" not in result and result:
        result["zh"] = next(iter(result.values()))
    result["default"] = result.get("zh", next(iter(result.values())))
    return result


def stratified_split(
    rows: list[dict[str, Any]], frac: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for label in ("present", "absent"):
        for lang in ("zh", "en"):
            bucket = [
                r
                for r in rows
                if normalize_presence_label(r.get("label"), split=r.get("split")) == label
                and str(r.get("lang") or "zh") == lang
            ]
            rng.shuffle(bucket)
            n_test = max(1, int(round(frac * len(bucket)))) if bucket else 0
            test.extend(bucket[:n_test])
            train.extend(bucket[n_test:])
    return train, test


def _dist(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p50": float(np.quantile(x, 0.50)),
        "p95": float(np.quantile(x, 0.95)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="用真实 mix ASR CER 优化 Presence 门控")
    p.add_argument("--decisions", type=Path, required=True, help="含 sim_streams 的 all_results.jsonl")
    p.add_argument("--asr-all-pos", type=Path, required=True, help="所有正样本均已 ASR 的 asr_results.jsonl")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--policies", default="max,strict_rescue,mix")
    p.add_argument("--holdout-frac", type=float, default=0.30)
    p.add_argument("--seeds", type=int, default=100)
    p.add_argument("--rescue-high-margin", type=float, default=0.08)
    p.add_argument("--rescue-floor-margin", type=float, default=0.10)
    p.add_argument("--rescue-dominance", type=float, default=0.05)
    p.add_argument("--strict", action="store_true")
    p.add_argument(
        "--baseline-thr",
        type=Path,
        default=None,
        help="冻结 max 门控阈值 JSON；提供后报告配对 holdout 分数差",
    )
    return p.parse_args()


def load_thresholds(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("thr_by_lang") or data.get("thresholds") or data
    out = {str(k): float(v) for k, v in raw.items() if isinstance(v, (int, float))}
    if "default" not in out and "zh" in out:
        out["default"] = out["zh"]
    if not out:
        raise ValueError(f"threshold JSON has no numeric thresholds: {path}")
    return out


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.decisions.resolve())
    asr_rows = load_jsonl(args.asr_all_pos.resolve())
    asr = {str(r["uid"]): r for r in asr_rows if r.get("uid")}
    pos_uids = {
        str(r["uid"])
        for r in rows
        if normalize_presence_label(r.get("label"), split=r.get("split")) == "present"
    }
    pos_rows = {
        str(r["uid"]): r
        for r in rows
        if normalize_presence_label(r.get("label"), split=r.get("split")) == "present"
    }
    missing = sorted(pos_uids - set(asr))
    duplicate_asr = len(asr_rows) - len(asr)
    invalid_asr: list[str] = []
    for uid in sorted(pos_uids & set(asr)):
        try:
            if str(asr[uid].get("decision")) != "accept":
                raise ValueError("not an all-positive forced-accept ASR row")
            if str(asr[uid].get("cmd_text") or "") != str(pos_rows[uid].get("cmd_text") or ""):
                raise ValueError("cmd_text mismatch between decisions and ASR")
            asr_cost(asr[uid])
        except (KeyError, ValueError) as exc:
            invalid_asr.append(f"{uid}:{exc}")
    if args.strict and (missing or duplicate_asr or invalid_asr):
        raise SystemExit(
            "ASR coverage invalid: "
            f"missing={len(missing)} duplicate={duplicate_asr} invalid={len(invalid_asr)}"
        )

    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    baseline_thresholds = load_thresholds(args.baseline_thr.resolve()) if args.baseline_thr else None
    margins = (
        float(args.rescue_high_margin),
        float(args.rescue_floor_margin),
        float(args.rescue_dominance),
    )
    report: dict[str, Any] = {
        "metric": "0.5*RR_neg + 0.5*(1-CER_pos_micro); rejected positive has errors=N",
        "asr_backend_contract": "all positive CMDs must use the same fixed downstream waveform backend",
        "coverage": {
            "n_decisions": len(rows),
            "n_asr": len(asr_rows),
            "n_pos": len(pos_uids),
            "missing_asr": missing,
            "duplicate_asr": duplicate_asr,
            "invalid_asr": invalid_asr,
        },
        "margins": {
            "high": margins[0],
            "floor": margins[1],
            "dominance": margins[2],
        },
        "full_data_diagnostic": {},
        "holdout": {},
        "baseline": {
            "policy": "max",
            "thresholds": baseline_thresholds,
            "path": str(args.baseline_thr.resolve()) if args.baseline_thr else None,
        },
    }

    cv_rows: dict[str, list[dict[str, Any]]] = {p: [] for p in policies}
    for policy in policies:
        thresholds = optimize_thresholds(rows, asr, policy=policy, margins=margins)
        report["full_data_diagnostic"][policy] = {
            "thresholds": thresholds,
            "metrics": evaluate(rows, asr, policy=policy, thresholds=thresholds, margins=margins),
            "warning": "in-sample optimum; do not deploy without holdout stability",
        }

    for seed in range(max(1, int(args.seeds))):
        train, test = stratified_split(rows, float(args.holdout_frac), seed)
        baseline_metrics = (
            evaluate(test, asr, policy="max", thresholds=baseline_thresholds, margins=margins)
            if baseline_thresholds
            else None
        )
        for policy in policies:
            thresholds = optimize_thresholds(train, asr, policy=policy, margins=margins)
            metrics = evaluate(test, asr, policy=policy, thresholds=thresholds, margins=margins)
            cv_rows[policy].append(
                {
                    **metrics,
                    "thr_zh": thresholds.get("zh"),
                    "thr_en": thresholds.get("en"),
                    "score_delta_vs_baseline": (
                        metrics["contest_score"] - baseline_metrics["contest_score"]
                        if baseline_metrics
                        else 0.0
                    ),
                }
            )

    for policy, vals in cv_rows.items():
        report["holdout"][policy] = {
            key: _dist([float(v[key]) for v in vals])
            for key in (
                "contest_score", "score_delta_vs_baseline", "rr",
                "cer_pos_micro", "frr", "thr_zh", "thr_en",
            )
        }
    best = max(
        policies,
        key=lambda p: report["holdout"][p]["contest_score"]["mean"],
    )
    stable = (
        baseline_thresholds is None
        or report["holdout"][best]["score_delta_vs_baseline"]["p05"] > 0.0
    )
    report["recommendation"] = {
        "policy": best if stable else "max_baseline",
        "best_experimental_policy": best,
        "basis": (
            "highest mean holdout score and p05 delta above baseline is positive"
            if stable
            else "no stable gain: paired holdout score delta p05 is not positive"
        ),
        "deploy_thresholds_candidate": (
            report["full_data_diagnostic"][best]["thresholds"]
            if stable
            else baseline_thresholds
        ),
        "requires_independent_domain_validation": True,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "gate_score_optimization.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Presence optimization with real ASR CER",
        "",
        "| policy | score mean | score p05-p95 | delta vs baseline | RR | CER | FRR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in policies:
        h = report["holdout"][policy]
        lines.append(
            f"| {policy} | {h['contest_score']['mean']:.6f} | "
            f"{h['contest_score']['p05']:.6f}-{h['contest_score']['p95']:.6f} | "
            f"{h['score_delta_vs_baseline']['mean']:.6f} "
            f"[{h['score_delta_vs_baseline']['p05']:.6f},"
            f"{h['score_delta_vs_baseline']['p95']:.6f}] | "
            f"{h['rr']['mean']:.6f} | {h['cer_pos_micro']['mean']:.6f} | "
            f"{h['frr']['mean']:.6f} |"
        )
    lines += ["", f"Recommended policy: **{report['recommendation']['policy']}**", ""]
    (args.out_dir / "gate_score_optimization.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 2 if args.strict and (missing or duplicate_asr or invalid_asr) else 0


if __name__ == "__main__":
    raise SystemExit(main())

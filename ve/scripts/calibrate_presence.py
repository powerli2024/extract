#!/usr/bin/env python3
"""Presence 阈值校准：支持 sep_depth=0/1/2+，可选保存分离中间 wav。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from audio_io import load_audio
from paths import (
    default_cohort_dir,
    default_eres2net_dir,
    default_spk_chs_dir,
    default_test_cohort_dir,
    default_ve_out,
    ensure_dir,
    setup_sys_path,
)
from presence_encoder import create_presence_encoder
from presence_gate import PresenceGate, try_create_onnx_separator

setup_sys_path()


def load_samples(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def stratified_limit(samples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """按 pos/neg 分层，避免 jsonl 前 N 条全是 pos 导致 RR 虚假为 1。"""
    if limit <= 0 or len(samples) <= limit:
        return samples
    pos = [r for r in samples if r.get("split") == "pos" or r.get("label") == "present"]
    neg = [r for r in samples if r.get("split") == "neg" or r.get("label") == "absent"]
    if not neg:
        print("[WARN] 无 neg/absent，LIMIT 下 RR/FAR 无意义", flush=True)
        return samples[:limit]
    n_pos_all, n_neg_all = len(pos), len(neg)
    n_neg = max(1, int(round(limit * n_neg_all / max(1, n_pos_all + n_neg_all))))
    n_neg = min(n_neg, n_neg_all, limit - 1)
    n_pos = min(n_pos_all, limit - n_neg)
    take_pos, take_neg = pos[:n_pos], neg[:n_neg]
    merged: list[dict[str, Any]] = []
    i = j = 0
    while i < len(take_pos) or j < len(take_neg):
        if i < len(take_pos):
            merged.append(take_pos[i])
            i += 1
        if j < len(take_neg):
            merged.append(take_neg[j])
            j += 1
    print(
        f"[INFO] stratified limit={limit} → pos={n_pos} neg={n_neg} "
        f"(pool pos={n_pos_all} neg={n_neg_all})",
        flush=True,
    )
    return merged


def contest_score(rr: float, cer: float) -> float:
    return 0.5 * float(rr) + 0.5 * (1.0 - float(cer))


def sweep_thresholds(
    scores: list[tuple[str, float]],
    *,
    target_frr: float = 0.02,
    select_by: str = "contest",
) -> dict[str, Any]:
    presents = sorted([s for lab, s in scores if lab == "present"])
    absents = sorted([s for lab, s in scores if lab == "absent"])
    n_p, n_a = len(presents), len(absents)
    if n_p == 0:
        raise SystemExit("无 present 样本，无法校准")

    all_s = presents + absents
    lo, hi = float(min(all_s)), float(max(all_s))
    # 分数点 + 覆盖 [lo,hi] 的网格（兼容 raw cosine 与 Z-Norm）
    grid: list[float] = []
    if hi > lo:
        step = max((hi - lo) / 200.0, 1e-4)
        x = lo
        while x <= hi + 1e-12:
            grid.append(round(x, 6))
            x += step
    else:
        grid = [round(lo, 6)]
    # 旧 cosine 习惯网格仍保留，不影响 Z-Norm（会被真实分位点主导）
    grid.extend(i / 100 for i in range(-50, 151))
    cands = sorted(set([round(x, 6) for x in all_s] + grid))

    def metrics_at(thr: float) -> dict[str, float]:
        frr = sum(1 for s in presents if s < thr) / n_p
        far = (sum(1 for s in absents if s >= thr) / n_a) if n_a else 0.0
        rr = 1.0 - far
        cer = frr
        return {
            "thr": thr,
            "frr": frr,
            "far": far,
            "rr": rr,
            "cer": cer,
            "neg_reject_rate": rr,
            "contest_score": contest_score(rr, cer),
        }

    curve = [metrics_at(t) for t in cands]
    select_by = (select_by or "contest").lower()
    if select_by == "contest":
        chosen = max(curve, key=lambda m: (m["contest_score"], m["thr"]))
    else:
        feasible = [m for m in curve if m["frr"] <= target_frr + 1e-12]
        if feasible:
            chosen = min(feasible, key=lambda m: (m["far"], -m["thr"]))
        else:
            min_frr = min(m["frr"] for m in curve)
            feasible = [m for m in curve if abs(m["frr"] - min_frr) < 1e-9]
            chosen = min(feasible, key=lambda m: (m["far"], -m["thr"]))

    eer = None
    best_gap = 1e9
    for m in curve:
        gap = abs(m["frr"] - m["far"])
        if gap < best_gap:
            best_gap = gap
            eer = {"thr": m["thr"], "eer": (m["frr"] + m["far"]) / 2, **m}

    return {
        "n_present": n_p,
        "n_absent": n_a,
        "target_frr": target_frr,
        "select_by": select_by,
        "metric": "0.5*RR + 0.5*(1-CER); CER=1 if present mis-rejected else 0 (presence-only)",
        "recommended": chosen,
        "eer_approx": eer,
        "curve_sample": curve[:: max(1, len(curve) // 40)],
        "mean_score_present": sum(presents) / n_p,
        "mean_score_absent": (sum(absents) / n_a) if n_a else None,
        "best_contest_score": chosen["contest_score"],
        "score_range": {"min": lo, "max": hi},
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="校准 Presence 阈值")
    p.add_argument("--samples", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--presence-backend", default="eres2netv2")
    p.add_argument("--eres-dir", type=Path, default=None)
    p.add_argument("--spk-chs-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--use-sep", action="store_true", help="等价 --sep-depth 1（若未显式设 depth）")
    p.add_argument(
        "--sep-depth",
        type=int,
        default=-1,
        help="0=不分离 1=一次分离 2+=级联多次；默认 -1 表示由 --use-sep 决定",
    )
    p.add_argument(
        "--save-sep-wavs",
        action="store_true",
        help="保存分离中间轨到 --sep-wav-dir 或 out-dir/../sep_streams",
    )
    p.add_argument("--sep-wav-dir", type=Path, default=None)
    p.add_argument("--target-frr", type=float, default=0.02)
    p.add_argument(
        "--select-by",
        default="contest",
        choices=("contest", "frr"),
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--thr", type=float, default=0.0, help="仅打分用，校准本身扫 thr")
    p.add_argument(
        "--cohort-dir",
        type=Path,
        default=None,
        help="干净路人（enroll）；也可 COHORT_DIR",
    )
    p.add_argument(
        "--test-cohort-dir",
        type=Path,
        default=None,
        help="CMD 路人（test/AS-Norm）；也可 TEST_COHORT_DIR，默认 mix500",
    )
    p.add_argument("--enroll-znorm", action="store_true")
    p.add_argument("--test-znorm", action="store_true")
    p.add_argument(
        "--asnorm",
        action="store_true",
        help="完整 AS-Norm：0.5*(z_A+z_B)，需 enroll+test cohort",
    )
    p.add_argument("--cohort-per-spk", type=int, default=2)
    p.add_argument("--cohort-max-files", type=int, default=400)
    p.add_argument("--test-cohort-max-files", type=int, default=500)
    p.add_argument("--cohort-seed", type=int, default=0)
    p.add_argument("--znorm-eps", type=float, default=1e-3)
    p.add_argument(
        "--enroll-vad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enroll 能量 VAD 裁剪静音后再 embed（默认开）",
    )
    p.add_argument("--enroll-vad-max-sec", type=float, default=4.0)
    p.add_argument(
        "--lang-split",
        action="store_true",
        help="按 KWS 语言(zh/en)分别扫 thr（当前最佳拒识）",
    )
    p.add_argument(
        "--holdout-frac",
        type=float,
        default=0.0,
        help=">0 时仅在 (1-frac) 校准子集上选 thr，并在 holdout 上报 contest（防同集过拟合）",
    )
    p.add_argument("--holdout-seed", type=int, default=0)
    p.add_argument(
        "--cmd-windows",
        default="off",
        help="off | slide | energy：CMD 滑窗打分（须重扫 τ）",
    )
    p.add_argument("--win-sec", type=float, default=0.8)
    p.add_argument("--hop-sec", type=float, default=0.4)
    p.add_argument("--win-pad-ms", type=float, default=80.0)
    return p.parse_args()


def resolve_sep_depth(args: argparse.Namespace) -> int:
    if args.sep_depth >= 0:
        return int(args.sep_depth)
    return 1 if args.use_sep else 0


def resolve_norm_mode(args: argparse.Namespace) -> str:
    if args.asnorm:
        return "asnorm"
    if args.test_znorm and args.enroll_znorm:
        return "asnorm"
    if args.test_znorm or args.test_cohort_dir is not None:
        if args.enroll_znorm or args.cohort_dir is not None:
            return "asnorm"
        return "test_znorm"
    if args.enroll_znorm or args.cohort_dir is not None:
        return "enroll_znorm"
    return "raw"


def main() -> int:
    args = parse_args()
    sep_depth = resolve_sep_depth(args)
    ve_out = default_ve_out()
    samples_path = args.samples or (ve_out / "manifest" / "samples.jsonl")
    out_dir = ensure_dir(args.out_dir or (ve_out / "reports" / "presence_calib"))
    samples = load_samples(samples_path)
    if args.limit and args.limit > 0:
        samples = stratified_limit(samples, int(args.limit))
    n_neg = sum(1 for r in samples if r.get("label") == "absent" or r.get("split") == "neg")
    if n_neg == 0 and (args.limit and args.limit > 0):
        raise SystemExit("LIMIT 子集无 neg，无法校准 RR；请更新分层抽样或 LIMIT=0")

    enc = create_presence_encoder(
        args.presence_backend,
        eres_dir=args.eres_dir or default_eres2net_dir(),
        resnet_dir=args.spk_chs_dir or default_spk_chs_dir(),
        device=args.device,
    )
    need_sep = sep_depth >= 1
    sep = try_create_onnx_separator(peak=0.95, device=args.device) if need_sep else None
    if need_sep and sep is None:
        raise SystemExit("sep_depth>=1 需要 MossFormer，请先 ./download_moss_onnx.sh")

    norm_mode = resolve_norm_mode(args)
    score_norm = None
    enroll_dir = test_dir = None
    if norm_mode != "raw":
        from cohort_znorm import build_score_normalizer

        enroll_dir = (
            Path(args.cohort_dir) if args.cohort_dir else default_cohort_dir()
        )
        test_dir = (
            Path(args.test_cohort_dir)
            if args.test_cohort_dir
            else default_test_cohort_dir()
        )
        score_norm = build_score_normalizer(
            enc,
            mode=norm_mode,  # type: ignore[arg-type]
            enroll_dir=enroll_dir if norm_mode in ("enroll_znorm", "asnorm") else None,
            test_dir=test_dir if norm_mode in ("test_znorm", "asnorm") else None,
            enroll_per_spk=int(args.cohort_per_spk),
            enroll_max_files=int(args.cohort_max_files),
            test_max_files=int(args.test_cohort_max_files),
            seed=int(args.cohort_seed),
            eps=float(args.znorm_eps),
        )
        print(f"[INFO] score_norm={norm_mode}", flush=True)

    gate = PresenceGate(
        enc,
        thr=0.0,
        use_sep=need_sep,
        separator=sep,
        sep_depth=sep_depth if sep is not None else 0,
        score_normalizer=score_norm,
        enroll_vad=bool(args.enroll_vad),
        enroll_vad_max_sec=float(args.enroll_vad_max_sec),
        cmd_window_mode=str(getattr(args, "cmd_windows", "off") or "off"),
        win_sec=float(getattr(args, "win_sec", 0.8)),
        hop_sec=float(getattr(args, "hop_sec", 0.4)),
        win_pad_ms=float(getattr(args, "win_pad_ms", 80.0)),
    )
    actual_depth = gate.sep_depth
    print(
        f"[INFO] enroll_vad={gate.enroll_vad} max_sec={gate.enroll_vad_max_sec} "
        f"cmd_windows={gate.cmd_window_mode}",
        flush=True,
    )

    sep_root = None
    if args.save_sep_wavs and actual_depth >= 1:
        sep_root = ensure_dir(
            args.sep_wav_dir
            or (out_dir.parent.parent / "sep_streams" / f"d{actual_depth}")
        )
        print(f"[INFO] save sep wavs → {sep_root}")

    scored: list[tuple[str, float]] = []
    detail: list[dict[str, Any]] = []
    t0 = time.time()
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore

    iterator = (
        tqdm(samples, desc=f"calib:d{actual_depth}", unit="utt", mininterval=0.5)
        if tqdm is not None
        else samples
    )
    for i, it in enumerate(iterator):
        enroll, sr = load_audio(it["enroll_wav"])
        cmd, _ = load_audio(it["cmd_wav"])
        save_dir = None
        if sep_root is not None:
            save_dir = sep_root / str(it.get("split", "x")) / str(it["uid"])
        pr = gate.score(
            enroll, cmd, enroll_key=it["uid"], sr=sr, save_dir=save_dir
        )
        scored.append((it["label"], pr.score))
        detail.append(
            {
                "uid": it["uid"],
                "label": it["label"],
                "split": it["split"],
                "lang": it.get("lang"),
                "wake_text": it.get("wake_text"),
                **pr.to_dict(),
            }
        )
        n_done = i + 1
        if n_done % 500 == 0 or n_done == len(samples):
            msg = f"[INFO] scored {n_done}/{len(samples)} depth={actual_depth}"
            if tqdm is not None:
                tqdm.write(msg)
            else:
                print(msg, flush=True)

    holdout_frac = float(args.holdout_frac or 0.0)
    calib_detail = detail
    holdout_detail: list[dict[str, Any]] = []
    if holdout_frac > 0:
        import random as _rnd

        if not (0.0 < holdout_frac < 0.9):
            raise SystemExit("--holdout-frac 须在 (0, 0.9)")
        rng = _rnd.Random(int(args.holdout_seed))
        pos_d = [r for r in detail if r.get("label") == "present"]
        neg_d = [r for r in detail if r.get("label") == "absent"]
        rng.shuffle(pos_d)
        rng.shuffle(neg_d)
        n_ho_p = max(1, int(round(len(pos_d) * holdout_frac)))
        n_ho_n = max(1, int(round(len(neg_d) * holdout_frac)))
        # 留足校准样本
        n_ho_p = min(n_ho_p, max(0, len(pos_d) - 10))
        n_ho_n = min(n_ho_n, max(0, len(neg_d) - 5))
        holdout_detail = pos_d[:n_ho_p] + neg_d[:n_ho_n]
        ho_uids = {str(r["uid"]) for r in holdout_detail}
        calib_detail = [r for r in detail if str(r["uid"]) not in ho_uids]
        print(
            f"[INFO] holdout_frac={holdout_frac} seed={args.holdout_seed} "
            f"calib={len(calib_detail)} holdout={len(holdout_detail)} "
            f"(pos_ho={n_ho_p} neg_ho={n_ho_n})",
            flush=True,
        )
        if len(calib_detail) < 20:
            raise SystemExit("holdout 后校准子集过小")

    scored_calib = [
        (str(r["label"]), float(r["presence_score"])) for r in calib_detail
    ]
    cal = sweep_thresholds(
        scored_calib, target_frr=args.target_frr, select_by=args.select_by
    )
    # 全量扫 thr 仅作对照（不写入 recommended，避免同集乐观偏差被当部署值）
    cal_full = sweep_thresholds(
        scored, target_frr=args.target_frr, select_by=args.select_by
    )
    cal["presence_backend"] = enc.name
    cal["use_sep"] = actual_depth >= 1
    cal["sep_depth"] = actual_depth
    cal["save_sep_wavs"] = bool(sep_root)
    cal["sep_wav_dir"] = str(sep_root) if sep_root else None
    cal["score_norm"] = norm_mode
    cal["enroll_vad"] = bool(args.enroll_vad)
    cal["enroll_vad_max_sec"] = float(args.enroll_vad_max_sec)
    cal["holdout_frac"] = holdout_frac
    cal["holdout_seed"] = int(args.holdout_seed)
    cal["full_data_oracle"] = {
        "thr": cal_full["recommended"]["thr"],
        "contest_score": cal_full["recommended"]["contest_score"],
        "rr": cal_full["recommended"]["rr"],
        "frr": cal_full["recommended"]["frr"],
        "note": "同集最优；仅对照，易乐观",
    }
    if score_norm is not None:
        cal["norm_meta"] = score_norm.to_meta()
        if enroll_dir is not None and norm_mode in ("enroll_znorm", "asnorm"):
            cal["enroll_cohort_dir"] = str(enroll_dir)
        if test_dir is not None and norm_mode in ("test_znorm", "asnorm"):
            cal["test_cohort_dir"] = str(test_dir)
    cal["elapsed_sec"] = round(time.time() - t0, 2)
    cal["n_scored"] = len(scored)
    cal["n_calib"] = len(calib_detail)
    cal["n_holdout"] = len(holdout_detail)

    # 全局 recommended（单 thr）；选 thr 仅用 calib 子集
    rec = cal["recommended"]
    thr_payload: dict[str, Any] = {
        "presence_thr": rec["thr"],
        "frr": rec["frr"],
        "far": rec["far"],
        "rr": rec["rr"],
        "cer": rec["cer"],
        "contest_score": rec["contest_score"],
        "neg_reject_rate": rec["neg_reject_rate"],
        "target_frr": args.target_frr,
        "select_by": args.select_by,
        "backend": enc.name,
        "use_sep": cal["use_sep"],
        "sep_depth": actual_depth,
        "score_norm": cal["score_norm"],
        "enroll_vad": bool(args.enroll_vad),
        "enroll_vad_max_sec": float(args.enroll_vad_max_sec),
        "thr_mode": "global",
        "metric": "0.5*RR + 0.5*(1-CER)",
        "holdout_frac": holdout_frac,
        "holdout_seed": int(args.holdout_seed),
        "metrics_scope": "calib_subset" if holdout_frac > 0 else "full_in_sample",
        "full_data_oracle": cal["full_data_oracle"],
        "warning": (
            "contest/thr 在标定数据上选得；holdout_frac=0 时为同集乐观估计，"
            "不保证验证集更优。建议 --holdout-frac 0.3 看泛化。"
            if holdout_frac <= 0
            else "thr 在 calib 子集上选择；contest_holdout 才是更可信的泛化估计。"
        ),
    }

    if holdout_frac > 0 and holdout_detail:
        thr_v = float(rec["thr"])
        n_p = n_a = n_fr = n_fa = 0
        for r in holdout_detail:
            s = float(r["presence_score"])
            if r.get("label") == "present":
                n_p += 1
                if s < thr_v:
                    n_fr += 1
            else:
                n_a += 1
                if s >= thr_v:
                    n_fa += 1
        frr_h = n_fr / max(1, n_p)
        far_h = n_fa / max(1, n_a)
        rr_h = 1.0 - far_h
        contest_h = 0.5 * rr_h + 0.5 * (1.0 - frr_h)
        thr_payload["holdout"] = {
            "rr": rr_h,
            "frr": frr_h,
            "far": far_h,
            "cer": frr_h,
            "contest_score": contest_h,
            "n_present": n_p,
            "n_absent": n_a,
            "thr": thr_v,
        }
        cal["holdout"] = thr_payload["holdout"]
        print(
            f"[INFO] holdout contest={contest_h:.4f} RR={rr_h:.4f} FRR={frr_h:.4f} "
            f"(calib contest={rec['contest_score']:.4f} | "
            f"full_oracle={cal_full['recommended']['contest_score']:.4f})",
            flush=True,
        )

    if args.lang_split:
        from presence_thr import build_lang_split_recommendation

        ls = build_lang_split_recommendation(
            calib_detail,
            samples=samples,
            sweep_fn=sweep_thresholds,
            target_frr=args.target_frr,
            select_by=args.select_by,
        )
        cal["lang_split"] = ls
        if ls.get("ok"):
            thr_payload["thr_mode"] = "lang_split"
            thr_payload["thr_by_lang"] = ls["thr_by_lang"]
            thr_payload["by_lang"] = ls["by_lang"]
            pooled = ls["pooled"]
            thr_payload["presence_thr"] = ls["presence_thr"]
            thr_payload["rr"] = pooled["rr"]
            thr_payload["frr"] = pooled["frr"]
            thr_payload["far"] = pooled["far"]
            thr_payload["cer"] = pooled["cer"]
            thr_payload["contest_score"] = pooled["contest_score"]
            thr_payload["neg_reject_rate"] = pooled["rr"]
            print(
                f"[INFO] lang_split thr={ls['thr_by_lang']} "
                f"pooled contest={pooled['contest_score']:.4f}",
                flush=True,
            )
            if holdout_frac > 0 and holdout_detail:
                from presence_thr import thr_for_sample

                n_p = n_a = n_fr = n_fa = 0
                for r in holdout_detail:
                    uid = str(r.get("uid") or "")
                    samp = next((s for s in samples if str(s.get("uid")) == uid), r)
                    thr_v = thr_for_sample(
                        samp,
                        default_thr=float(ls["presence_thr"]),
                        thr_meta=ls,
                    )
                    s = float(r["presence_score"])
                    if r.get("label") == "present":
                        n_p += 1
                        if s < thr_v:
                            n_fr += 1
                    else:
                        n_a += 1
                        if s >= thr_v:
                            n_fa += 1
                frr_h = n_fr / max(1, n_p)
                far_h = n_fa / max(1, n_a)
                rr_h = 1.0 - far_h
                contest_h = 0.5 * rr_h + 0.5 * (1.0 - frr_h)
                thr_payload["holdout"] = {
                    "rr": rr_h,
                    "frr": frr_h,
                    "far": far_h,
                    "cer": frr_h,
                    "contest_score": contest_h,
                    "n_present": n_p,
                    "n_absent": n_a,
                    "thr_mode": "lang_split",
                }
                cal["holdout"] = thr_payload["holdout"]
                print(
                    f"[INFO] lang_split holdout contest={contest_h:.4f} "
                    f"RR={rr_h:.4f} FRR={frr_h:.4f}",
                    flush=True,
                )
        else:
            print(f"[WARN] lang_split 失败: {ls.get('reason')}，回退全局 thr", flush=True)

    (out_dir / "calibration.json").write_text(
        json.dumps(cal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (out_dir / "scores.jsonl").open("w", encoding="utf-8") as f:
        for row in detail:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    thr_path = out_dir / "recommended_thr.json"
    thr_path.write_text(
        json.dumps(thr_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    md = [
        "# Presence 校准报告",
        "",
        f"- backend: `{enc.name}`",
        f"- sep_depth: **{actual_depth}** (0=无分离, 1=一次, 2+=级联)",
        f"- use_sep: {cal['use_sep']}",
        f"- score_norm: `{cal['score_norm']}`",
        f"- enroll_vad: **{bool(args.enroll_vad)}** (max_sec={args.enroll_vad_max_sec})",
        f"- thr_mode: `{thr_payload.get('thr_mode')}`",
        f"- holdout_frac: {holdout_frac}",
        f"- save_sep_wavs: {cal['save_sep_wavs']} → `{cal.get('sep_wav_dir')}`",
        f"- n_present={cal['n_present']} n_absent={cal['n_absent']}",
        f"- select_by: `{args.select_by}`",
        f"- **竞赛分(校准子集)** `0.5*RR + 0.5*(1-CER)` = **{thr_payload['contest_score']:.4f}**",
        f"  - RR={thr_payload['rr']:.4f}  CER(=FRR)={thr_payload['cer']:.4f}",
        f"- **recommended thr={thr_payload['presence_thr']}** "
        f"FRR={thr_payload['frr']:.4f} FAR={thr_payload['far']:.4f}",
        f"- full_data_oracle contest={cal['full_data_oracle']['contest_score']:.4f} "
        f"thr={cal['full_data_oracle']['thr']}（同集乐观，仅对照）",
        f"- elapsed: {cal['elapsed_sec']}s",
        "",
        f"> {thr_payload.get('warning', '')}",
        "",
    ]
    if thr_payload.get("holdout"):
        ho = thr_payload["holdout"]
        md += [
            f"- **holdout contest={ho['contest_score']:.4f}** "
            f"RR={ho['rr']:.4f} FRR={ho['frr']:.4f} "
            f"(n_pos={ho['n_present']} n_neg={ho['n_absent']})",
            "",
        ]
    if thr_payload.get("thr_mode") == "lang_split":
        md.append(f"- thr_by_lang: `{thr_payload.get('thr_by_lang')}`")
        md.append("")
    if score_norm is not None:
        md.append(
            f"拒识：score_norm=`{norm_mode}`；raw=`max_k sim`；`< thr` → speaker_absent。"
        )
        if norm_mode == "asnorm":
            md.append("AS-Norm: `0.5*((raw-μ_A)/σ_A + (raw-μ_B)/σ_B)`，B=获胜轨。")
    else:
        md.append(
            "拒识：`presence_score = max_k sim(enroll, stream_k)`；`< thr` → speaker_absent。"
        )
    (out_dir / "calibration.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(thr_payload, ensure_ascii=False, indent=2))
    print(f"[OK] wrote {out_dir} sep_depth={actual_depth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

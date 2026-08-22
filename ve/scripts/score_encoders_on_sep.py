#!/usr/bin/env python3
"""多编码器 × 已有 sep_streams 打分（不再跑 MossFormer）。

arms:
  no_sep    — 只用 mix.wav
  sep_once  — mix + d1_*（一次分离）
  sep_multi — d2 目录下全部轨（级联）；若无 d2 则跳过

用法:
  python scripts/score_encoders_on_sep.py \\
    --samples $VE_OUT/manifest/samples.jsonl \\
    --sep-root /root/autodl-tmp/ve_gate_cmp/sep_streams \\
    --encoders eres2netv2,campplus,resnet34_lm \\
    --arms no_sep,sep_once,sep_multi \\
    --out-dir /root/autodl-tmp/ve_encoder_cmp
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from audio_io import cosine_sim, load_audio, vad_crop_speech
from calibrate_presence import stratified_limit, sweep_thresholds
from paths import (
    default_campplus_dir,
    default_eres2net_dir,
    default_spk_chs_dir,
    default_vblink_dir,
    default_ve_out,
    ensure_dir,
    normalize_presence_label,
    setup_sys_path,
)
from presence_encoder import create_presence_encoder

setup_sys_path()

ARM_DEPTH = {
    "no_sep": 0,
    "sep_once": 1,
    "sep_1": 1,
    "1_sep": 1,
    "sep_multi": 2,
    "dual_sep": 2,
    "sep_2": 2,
    "2_sep": 2,
}


def load_samples(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return float(np.sqrt(np.mean(x**2) + 1e-12))


def resolve_utt_sep_dir(sep_root: Path, depth: int, split: str, uid: str) -> Path:
    return Path(sep_root) / f"d{depth}" / str(split) / str(uid)


def load_streams_for_arm(
    sep_root: Path,
    arm: str,
    split: str,
    uid: str,
    *,
    cmd_wav_fallback: str | Path | None = None,
    min_rms: float = 1e-4,
) -> dict[str, np.ndarray] | None:
    """从已有 sep_streams 读波形。返回 None 表示该 arm 不可用。"""
    depth = ARM_DEPTH[arm]
    if depth == 0:
        # 优先 d1/.../mix.wav，否则用原始 cmd
        d1 = resolve_utt_sep_dir(sep_root, 1, split, uid)
        mix_p = d1 / "mix.wav"
        if mix_p.is_file():
            w, _ = load_audio(mix_p)
            return {"mix": w}
        if cmd_wav_fallback and Path(cmd_wav_fallback).is_file():
            w, _ = load_audio(cmd_wav_fallback)
            return {"mix": w}
        return None

    utt_dir = resolve_utt_sep_dir(sep_root, depth, split, uid)
    if not utt_dir.is_dir():
        return None
    streams: dict[str, np.ndarray] = {}
    for p in sorted(utt_dir.glob("*.wav")):
        name = p.stem
        if name == "peak":
            continue
        if depth == 1:
            # 一次分离：只要 mix + d1_*
            if name != "mix" and not name.startswith("d1_"):
                continue
        # depth>=2：该目录下全部轨（compare_sep_reject 已按 depth 落盘）
        w, _ = load_audio(p)
        if name != "mix" and _rms(w) < min_rms:
            continue
        streams[name] = w
    if "mix" not in streams:
        return None
    return streams


def score_streams(
    enc: Any,
    enroll_emb: np.ndarray,
    streams: dict[str, np.ndarray],
    *,
    sr: int = 16000,
) -> tuple[float, str, dict[str, float]]:
    sims: dict[str, float] = {}
    for name, w in streams.items():
        emb = enc.embed(w, sr)
        sims[name] = cosine_sim(enroll_emb, emb)
    best_name = max(sims, key=lambda k: sims[k])
    return float(sims[best_name]), best_name, sims


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="多编码器 × 已有 sep_streams 打分")
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument(
        "--sep-root",
        type=Path,
        required=True,
        help="含 d1/ d2/ 的 sep_streams 根目录",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--encoders",
        default="eres2netv2,campplus,resnet34_lm",
        help="逗号分隔: eres2netv2,campplus,resnet34_lm,vblink2",
    )
    p.add_argument(
        "--arms",
        default="no_sep,sep_once,sep_multi",
        help="逗号分隔: no_sep,sep_once,sep_multi（或 1_sep,dual_sep）",
    )
    p.add_argument("--eres-dir", type=Path, default=None)
    p.add_argument("--spk-chs-dir", type=Path, default=None)
    p.add_argument("--campplus-dir", type=Path, default=None)
    p.add_argument("--vblink-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--select-by", default="contest", choices=("contest", "frr"))
    p.add_argument("--target-frr", type=float, default=0.02)
    p.add_argument(
        "--enroll-vad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enroll 能量 VAD 裁剪（默认开；--no-enroll-vad 关闭）",
    )
    p.add_argument("--enroll-vad-max-sec", type=float, default=4.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    encoders = [x.strip() for x in args.encoders.split(",") if x.strip()]
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    for a in arms:
        if a not in ARM_DEPTH:
            raise SystemExit(f"未知 arm={a}；可选 {list(ARM_DEPTH)}")

    sep_root = Path(args.sep_root).resolve()
    if not sep_root.is_dir():
        raise SystemExit(f"sep-root 不存在: {sep_root}")

    out_dir = ensure_dir(
        args.out_dir or (default_ve_out().parent / "ve_encoder_cmp")
    )
    samples = load_samples(Path(args.samples))
    if args.limit and args.limit > 0:
        samples = stratified_limit(samples, int(args.limit))

    # 探测哪些 arm 有数据
    probe = samples[0]
    available_arms: list[str] = []
    for arm in arms:
        st = load_streams_for_arm(
            sep_root,
            arm,
            str(probe.get("split", "pos")),
            str(probe["uid"]),
            cmd_wav_fallback=probe.get("cmd_wav"),
        )
        if st is None and ARM_DEPTH[arm] >= 2:
            print(f"[WARN] arm={arm} 首条无 d2 目录，将按样本跳过缺失项", flush=True)
            available_arms.append(arm)  # still try per-utt
        elif st is None and ARM_DEPTH[arm] == 0:
            print(f"[WARN] arm={arm} 找不到 mix，检查 sep-root 或 samples.cmd_wav", flush=True)
            available_arms.append(arm)
        else:
            available_arms.append(arm)
            print(
                f"[INFO] arm={arm} ok example streams={list(st.keys()) if st else []}",
                flush=True,
            )

    print(f"[INFO] out={out_dir} n_samples={len(samples)} encoders={encoders}")
    print(f"[INFO] sep_root={sep_root}")

    matrix: dict[str, Any] = {
        "n_samples": len(samples),
        "sep_root": str(sep_root),
        "encoders": encoders,
        "arms": available_arms,
        "enroll_vad": bool(args.enroll_vad),
        "enroll_vad_max_sec": float(args.enroll_vad_max_sec),
        "cells": {},
    }
    print(
        f"[INFO] enroll_vad={args.enroll_vad} max_sec={args.enroll_vad_max_sec}",
        flush=True,
    )

    t_all = time.time()
    for enc_name in encoders:
        print(f"\n=== encoder={enc_name} ===", flush=True)
        try:
            enc = create_presence_encoder(
                enc_name,
                eres_dir=args.eres_dir or default_eres2net_dir(),
                resnet_dir=args.spk_chs_dir or default_spk_chs_dir(),
                campplus_dir=args.campplus_dir or default_campplus_dir(),
                vblink_dir=args.vblink_dir or default_vblink_dir(),
                device=args.device,
            )
        except Exception as e:
            print(f"[ERR] 加载 {enc_name} 失败: {e}", flush=True)
            matrix["cells"][enc_name] = {"error": str(e)}
            continue

        # 预计算 enroll
        enroll_cache: dict[str, np.ndarray] = {}
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None  # type: ignore

        for arm in available_arms:
            scored: list[tuple[str, float]] = []
            details: list[dict[str, Any]] = []
            n_miss = 0
            t0 = time.time()
            it = (
                tqdm(samples, desc=f"{enc.name}:{arm}", unit="utt", mininterval=0.5)
                if tqdm
                else samples
            )
            for row in it:
                uid = str(row["uid"])
                split = str(row.get("split", "x"))
                try:
                    label = normalize_presence_label(row.get("label"), split=split)
                except KeyError:
                    label = "present" if str(split) == "pos" else "absent"
                if uid not in enroll_cache:
                    ew, sr = load_audio(row["enroll_wav"])
                    if args.enroll_vad:
                        ew, _vad_meta = vad_crop_speech(
                            ew, sr, max_sec=float(args.enroll_vad_max_sec)
                        )
                    enroll_cache[uid] = enc.embed(ew, sr)
                e = enroll_cache[uid]
                streams = load_streams_for_arm(
                    sep_root,
                    arm,
                    split,
                    uid,
                    cmd_wav_fallback=row.get("cmd_wav"),
                )
                if streams is None:
                    n_miss += 1
                    continue
                score, best, sims = score_streams(enc, e, streams)
                scored.append((label, score))
                details.append(
                    {
                        "uid": uid,
                        "label": label,
                        "split": split,
                        "encoder": enc.name,
                        "arm": arm,
                        "sep_depth": ARM_DEPTH[arm],
                        "presence_score": round(score, 6),
                        "best_stream": best,
                        "sim_streams": {k: round(v, 6) for k, v in sims.items()},
                        "n_streams": len(sims),
                    }
                )

            if len(scored) < 10:
                print(
                    f"[WARN] {enc.name}/{arm}: 有效样本过少 n={len(scored)} miss={n_miss}",
                    flush=True,
                )
                continue

            cal = sweep_thresholds(
                scored, target_frr=args.target_frr, select_by=args.select_by
            )
            rec = cal["recommended"]
            cell_dir = ensure_dir(out_dir / "reports" / f"{enc.name}__{arm}")
            (cell_dir / "calibration.json").write_text(
                json.dumps(cal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            with (cell_dir / "scores.jsonl").open("w", encoding="utf-8") as f:
                for d in details:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
            thr_obj = {
                "encoder": enc.name,
                "arm": arm,
                "sep_depth": ARM_DEPTH[arm],
                "presence_thr": rec["thr"],
                "rr": rec["rr"],
                "frr": rec["frr"],
                "far": rec["far"],
                "cer": rec["cer"],
                "contest_score": rec["contest_score"],
                "n_scored": len(scored),
                "n_miss_streams": n_miss,
                "elapsed_sec": round(time.time() - t0, 2),
                "select_by": args.select_by,
                "enroll_vad": bool(args.enroll_vad),
            }
            (cell_dir / "recommended_thr.json").write_text(
                json.dumps(thr_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            matrix["cells"].setdefault(enc.name, {})[arm] = thr_obj
            print(
                f"[OK] {enc.name}/{arm}: contest={rec['contest_score']:.4f} "
                f"RR={rec['rr']:.4f} FRR={rec['frr']:.4f} thr={rec['thr']} "
                f"n={len(scored)} miss={n_miss}",
                flush=True,
            )

    matrix["elapsed_sec"] = round(time.time() - t_all, 2)

    # 汇总表
    table_rows: list[dict[str, Any]] = []
    for enc_name, arms_map in matrix["cells"].items():
        if not isinstance(arms_map, dict) or "error" in arms_map:
            continue
        for arm, thr_obj in arms_map.items():
            table_rows.append(
                {
                    "encoder": enc_name,
                    "arm": arm,
                    "contest": round(thr_obj["contest_score"], 4),
                    "rr": round(thr_obj["rr"], 4),
                    "frr": round(thr_obj["frr"], 4),
                    "thr": thr_obj["presence_thr"],
                    "n": thr_obj["n_scored"],
                }
            )
    table_rows.sort(key=lambda r: (-r["contest"], r["frr"]))
    matrix["table"] = table_rows
    if table_rows:
        matrix["best"] = table_rows[0]

    reports = ensure_dir(out_dir / "reports")
    (reports / "encoder_sep_matrix.json").write_text(
        json.dumps(matrix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md = [
        "# 多编码器 × sep_streams 对照",
        "",
        f"- sep_root: `{sep_root}`",
        f"- n_samples: {len(samples)}",
        f"- enroll_vad: `{bool(args.enroll_vad)}` (max_sec={args.enroll_vad_max_sec})",
        "",
        "| encoder | arm | thr | RR | FRR | contest | n |",
        "|---------|-----|-----|----|-----|---------|---|",
    ]
    for r in table_rows:
        md.append(
            f"| {r['encoder']} | {r['arm']} | {r['thr']} | {r['rr']} | "
            f"{r['frr']} | **{r['contest']}** | {r['n']} |"
        )
    if table_rows:
        b = table_rows[0]
        md += [
            "",
            f"**最佳**: `{b['encoder']}` / `{b['arm']}` "
            f"contest={b['contest']} thr={b['thr']}",
        ]
    md += [
        "",
        "说明: 不重跑分离；`no_sep`=仅 mix，`sep_once`=d1 轨，`sep_multi`=d2 轨。",
        "分数 = max_k cosine(enroll, stream_k)；thr 按 contest 重扫。",
    ]
    (reports / "encoder_sep_matrix.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"best": matrix.get("best"), "n_cells": len(table_rows)}, ensure_ascii=False, indent=2))
    print(f"[OK] {reports / 'encoder_sep_matrix.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

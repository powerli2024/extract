#!/usr/bin/env python3
"""VE 端到端：Presence 拒识 → 提取方案之一。

PIPELINE / --tse-backend:
  ps4         — PS4 BSRNN（默认）
  wesep_bsrnn — WeSep 官方 bsrnn_ecapa_vox1
  sep_route   — MossFormer 分离 + enroll 声纹选路（强制 use_sep）
  adaptive_route — 分离流声纹显著优于 mix 时才选分离流（强制 use_sep）
  mix         — CMD mix 直通 ASR（不做 TSE）

产物:
  VE_OUT/extracted/{split}/{uid}.wav
  VE_OUT/results/{split}_results.jsonl
  VE_OUT/reports/...
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

from audio_io import load_audio, save_audio
from calibrate_presence import stratified_limit
from paths import (
    default_cohort_dir,
    default_eres2net_dir,
    default_ps4_weights,
    default_spk_chs_dir,
    default_test_cohort_dir,
    default_ve_out,
    default_wesep_dir,
    ensure_dir,
    setup_sys_path,
)
from presence_encoder import create_presence_encoder
from presence_gate import PresenceGate, try_create_onnx_separator
from presence_thr import load_thr_file, thr_for_sample
from report_ve import write_run_reports
from tse_factory import create_tse

setup_sys_path()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_backend(name: str) -> str:
    b = (name or "ps4").lower().strip()
    aliases = {
        "ps4": "ps4",
        "ps4_bsrnn": "ps4",
        "bsrnn": "ps4",
        "wesep": "wesep_bsrnn",
        "wesep_bsrnn": "wesep_bsrnn",
        "wesep_bsrnn_ecapa": "wesep_bsrnn",
        "sep_route": "sep_route",
        "mossformer": "sep_route",
        "route": "sep_route",
        "sep": "sep_route",
        "adaptive_route": "adaptive_route",
        "adaptive": "adaptive_route",
        "mix_sep_route": "adaptive_route",
        "cond_tasnet": "cond_tasnet",
        "condtasnet": "cond_tasnet",
        "tasnet": "cond_tasnet",
        "cond-tasnet": "cond_tasnet",
        "mix": "mix",
        "passthrough": "mix",
        "mix_passthrough": "mix",
        "cmd": "mix",
        "none": "mix",
    }
    if b not in aliases:
        raise SystemExit(
            f"未知 --tse-backend={name!r}；可选: ps4 | wesep_bsrnn | sep_route | adaptive_route | mix | cond_tasnet"
        )
    return aliases[b]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VE Presence-gated 提取")
    p.add_argument("--samples", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--splits", default="pos,neg")
    p.add_argument("--presence-backend", default="eres2netv2")
    p.add_argument("--eres-dir", type=Path, default=None)
    p.add_argument("--spk-chs-dir", type=Path, default=None)
    p.add_argument("--presence-thr", type=float, default=-1.0, help="<0 则读校准文件")
    p.add_argument("--thr-file", type=Path, default=None)
    p.add_argument("--use-sep", action="store_true", help="Presence 用 MossFormer（默认 depth=1）")
    p.add_argument(
        "--sep-depth",
        type=int,
        default=-1,
        help="0=不分离 1=一次 2+=级联多次；-1 表示由 --use-sep / sep_route 决定",
    )
    p.add_argument(
        "--save-sep-wavs",
        action="store_true",
        help="保存分离中间轨到 VE_OUT/sep_streams/d{depth}/{split}/{uid}/",
    )
    p.add_argument(
        "--tse-backend",
        default="ps4",
        help="ps4 | wesep_bsrnn | sep_route | adaptive_route | mix | cond_tasnet",
    )
    p.add_argument("--cond-tasnet-ckpt", type=Path, default=None)
    p.add_argument("--ecapa-dir", type=Path, default=None)
    p.add_argument("--tasnet-chunk-sec", type=float, default=4.0)
    p.add_argument("--route-min-gain", type=float, default=0.03,
                   help="adaptive_route 选择分离流所需的最小声纹增益")
    p.add_argument("--ps4-weights", type=Path, default=None)
    p.add_argument("--wesep-dir", type=Path, default=None, help="兼容旧参数")
    p.add_argument("--wesep-model-dir", type=Path, default=None)
    p.add_argument("--wesep-language", default="english")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip-tse", action="store_true", help="只跑 Presence（调试）")
    p.add_argument("--force-extract", action="store_true", help="忽略门控强制提取（对照）")
    p.add_argument("--write-reject-debug-wav", action="store_true")
    p.add_argument("--cohort-dir", type=Path, default=None)
    p.add_argument("--test-cohort-dir", type=Path, default=None)
    p.add_argument(
        "--enroll-znorm",
        action="store_true",
        help="enroll Z-Norm；thr 含 score_norm 时也会自动开",
    )
    p.add_argument("--test-znorm", action="store_true")
    p.add_argument("--asnorm", action="store_true")
    p.add_argument("--no-enroll-znorm", action="store_true", help="兼容旧开关，强制 raw")
    p.add_argument("--no-score-norm", action="store_true")
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
        "--cmd-windows",
        default="off",
        help="off | slide | energy：CMD 滑窗/能量段打分；默认 ASR 用 argmax 窗",
    )
    p.add_argument("--win-sec", type=float, default=0.8)
    p.add_argument("--hop-sec", type=float, default=0.4)
    p.add_argument("--win-pad-ms", type=float, default=80.0)
    p.add_argument(
        "--asr-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="滑窗开启时是否把 mix ASR 裁到 argmax 窗（T2 默认开；T2b 用 --no-asr-crop）",
    )
    p.add_argument(
        "--veto-backend",
        default="",
        help="灰区第二路否决编码器（如 campplus）；空=关闭。只否决不救援",
    )
    p.add_argument("--veto-margin", type=float, default=0.12)
    p.add_argument("--veto-gray", type=float, default=0.10)
    p.add_argument(
        "--veto-windows",
        action="store_true",
        help="灰区且次优窗明显低于最优窗时否决",
    )
    return p.parse_args()


def resolve_sep_depth(args: argparse.Namespace, backend: str) -> int:
    if args.sep_depth >= 0:
        return int(args.sep_depth)
    if backend in ("sep_route", "adaptive_route") or args.use_sep:
        return 1
    return 0


def main() -> int:
    args = parse_args()
    backend = normalize_backend(args.tse_backend)
    sep_depth = resolve_sep_depth(args, backend)
    use_sep = sep_depth >= 1

    ve_out = (args.out_dir or default_ve_out()).resolve()
    ensure_dir(ve_out)
    samples_path = args.samples
    if samples_path is None:
        for cand in (
            ve_out / "manifest" / "samples.jsonl",
            default_ve_out() / "manifest" / "samples.jsonl",
            ve_out.parent / "manifest" / "samples.jsonl",
        ):
            if cand.is_file():
                samples_path = cand
                break
        if samples_path is None:
            samples_path = ve_out / "manifest" / "samples.jsonl"
    else:
        samples_path = samples_path.resolve()
    if not samples_path.is_file():
        raise SystemExit(f"找不到 samples.jsonl: {samples_path}（请先 build_manifest.py）")

    thr_file = args.thr_file or (
        ve_out / "reports" / "presence_calib" / "recommended_thr.json"
    )
    if not thr_file.is_file():
        alt = ve_out.parent / "reports" / "presence_calib" / "recommended_thr.json"
        if alt.is_file():
            thr_file = alt
        else:
            shared = Path(
                "/root/autodl-tmp/ve_presence_best/reports/presence_calib/recommended_thr.json"
            )
            if shared.is_file():
                thr_file = shared
            else:
                thr_file = (
                    default_ve_out() / "reports" / "presence_calib" / "recommended_thr.json"
                )

    thr_meta: dict[str, Any] = {}
    if args.presence_thr >= 0:
        thr_default = float(args.presence_thr)
    else:
        thr_default, thr_meta = load_thr_file(thr_file, 0.25)

    splits = {s.strip() for s in args.splits.split(",") if s.strip()}
    samples = [r for r in load_jsonl(samples_path) if r.get("split") in splits]
    if args.limit and args.limit > 0:
        samples = stratified_limit(samples, int(args.limit))

    want_mode = "raw"
    if not args.no_score_norm and not args.no_enroll_znorm:
        meta_mode = str(thr_meta.get("score_norm") or "raw")
        if args.asnorm or meta_mode == "asnorm":
            want_mode = "asnorm"
        elif args.test_znorm or meta_mode == "test_znorm":
            want_mode = "test_znorm"
        elif args.enroll_znorm or meta_mode == "enroll_znorm" or args.cohort_dir:
            want_mode = "enroll_znorm"
        if args.test_cohort_dir and want_mode == "enroll_znorm":
            want_mode = "asnorm"
        if args.test_cohort_dir and want_mode == "raw":
            want_mode = "test_znorm"
        if meta_mode == "raw" and not (
            args.asnorm or args.enroll_znorm or args.test_znorm or args.cohort_dir
        ):
            want_mode = "raw"

    thr_mode = thr_meta.get("thr_mode") or "global"
    from ve_tags import assert_thr_runtime_compatible

    for w in assert_thr_runtime_compatible(
        thr_meta, enroll_vad=bool(args.enroll_vad), strict=True
    ):
        print(f"[WARN] {w}", flush=True)
    print(f"[INFO] VE_OUT={ve_out}")
    print(
        f"[INFO] samples={len(samples)} thr_default={thr_default} thr_mode={thr_mode} "
        f"backend={backend} sep_depth={sep_depth} save_sep={args.save_sep_wavs} "
        f"score_norm={want_mode} enroll_vad={bool(args.enroll_vad)}"
    )
    if thr_mode == "lang_split":
        print(f"[INFO] thr_by_lang={thr_meta.get('thr_by_lang')}", flush=True)
    if thr_meta.get("holdout"):
        print(f"[INFO] thr 来自 holdout 校准: {thr_meta.get('holdout')}", flush=True)
    print("[INFO] reject_policy=speaker_absent_only")

    enc = create_presence_encoder(
        args.presence_backend,
        eres_dir=args.eres_dir or default_eres2net_dir(),
        resnet_dir=args.spk_chs_dir or default_spk_chs_dir(),
        device=args.device,
    )
    sep = try_create_onnx_separator(peak=0.95, device=args.device) if use_sep else None
    if use_sep and sep is None:
        raise SystemExit(
            "sep_depth>=1 需要 MossFormer。请 ./download_moss_onnx.sh（extract/scripts/mossformer2_onnx.py）"
        )
    if backend in ("sep_route", "adaptive_route") and sep is None:
        raise SystemExit(f"PIPELINE={backend} 需要 MossFormer")

    score_norm = None
    if want_mode != "raw":
        from cohort_znorm import build_score_normalizer

        score_norm = build_score_normalizer(
            enc,
            mode=want_mode,  # type: ignore[arg-type]
            enroll_dir=(
                Path(args.cohort_dir) if args.cohort_dir else default_cohort_dir()
            )
            if want_mode in ("enroll_znorm", "asnorm")
            else None,
            test_dir=(
                Path(args.test_cohort_dir)
                if args.test_cohort_dir
                else default_test_cohort_dir()
            )
            if want_mode in ("test_znorm", "asnorm")
            else None,
            enroll_per_spk=int(args.cohort_per_spk),
            enroll_max_files=int(args.cohort_max_files),
            test_max_files=int(args.test_cohort_max_files),
            seed=int(args.cohort_seed),
            eps=float(args.znorm_eps),
        )

    veto_enc = None
    veto_name = str(getattr(args, "veto_backend", "") or "").strip()
    if veto_name:
        veto_enc = create_presence_encoder(
            veto_name,
            eres_dir=args.eres_dir or default_eres2net_dir(),
            resnet_dir=args.spk_chs_dir or default_spk_chs_dir(),
            campplus_dir=args.eres_dir or default_eres2net_dir(),
            device=args.device,
        )
        print(f"[INFO] veto encoder={veto_enc.name} margin={args.veto_margin}", flush=True)

    gate = PresenceGate(
        enc,
        thr=thr_default,
        use_sep=bool(sep),
        separator=sep,
        sep_depth=sep_depth if sep is not None else 0,
        score_normalizer=score_norm,
        enroll_vad=bool(args.enroll_vad),
        enroll_vad_max_sec=float(args.enroll_vad_max_sec),
        cmd_window_mode=str(args.cmd_windows or "off"),
        win_sec=float(args.win_sec),
        hop_sec=float(args.hop_sec),
        win_pad_ms=float(args.win_pad_ms),
        veto_encoder=veto_enc,
        veto_margin=float(args.veto_margin),
        veto_gray=float(args.veto_gray),
        veto_windows=bool(args.veto_windows),
    )
    actual_depth = gate.sep_depth
    print(
        f"[INFO] enroll_vad={gate.enroll_vad} max_sec={gate.enroll_vad_max_sec} "
        f"cmd_windows={gate.cmd_window_mode} veto_windows={gate.veto_windows}",
        flush=True,
    )

    extractor = None
    if not args.skip_tse:
        extractor = create_tse(
            backend,
            weights=args.ps4_weights or default_ps4_weights(),
            device=args.device,
            wesep_dir=args.wesep_dir or default_wesep_dir(),
            wesep_model_dir=args.wesep_model_dir,
            wesep_language=args.wesep_language,
            separator=sep,
            encoder=enc,
            cond_tasnet_ckpt=args.cond_tasnet_ckpt,
            ecapa_dir=args.ecapa_dir,
            tasnet_chunk_sec=float(args.tasnet_chunk_sec),
            route_min_gain=float(args.route_min_gain),
        )

    results_dir = ensure_dir(ve_out / "results")
    extracted_dir = ensure_dir(ve_out / "extracted")
    debug_dir = ensure_dir(ve_out / "debug_reject") if args.write_reject_debug_wav else None
    sep_root = (
        ensure_dir(ve_out / "sep_streams" / f"d{actual_depth}")
        if args.save_sep_wavs and actual_depth >= 1
        else None
    )

    by_split: dict[str, list[dict[str, Any]]] = {s: [] for s in splits}
    t_run0 = time.time()

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore

    iterator = (
        tqdm(samples, desc=f"extract:{backend}", unit="utt", mininterval=0.5)
        if tqdm is not None
        else samples
    )
    for i, it in enumerate(iterator):
        uid = it["uid"]
        split = it["split"]
        thr = thr_for_sample(it, default_thr=thr_default, thr_meta=thr_meta)
        t0 = time.time()
        rec: dict[str, Any] = {
            "uid": uid,
            "split": split,
            "id": it.get("id"),
            "label": it.get("label"),
            "lang": it.get("lang"),
            "wake_text": it.get("wake_text"),
            "cmd_text": it.get("cmd_text"),
            "enroll_wav": it.get("enroll_wav"),
            "cmd_wav": it.get("cmd_wav"),
            "presence_backend": enc.name,
            "tse_backend": None if extractor is None else extractor.name,
            "reject_policy": "speaker_absent_only",
            "pipeline": backend,
            "thr_mode": thr_mode,
        }
        try:
            enroll, sr = load_audio(it["enroll_wav"])
            cmd, _ = load_audio(it["cmd_wav"])
            save_dir = None
            if sep_root is not None:
                save_dir = sep_root / split / uid
            pr, streams, enroll_emb = gate.score_with_streams(
                enroll, cmd, enroll_key=uid, sr=sr, thr=thr, save_dir=save_dir
            )
            rec.update(pr.to_dict())
            rec["presence_thr"] = thr
            rec["presence_ms"] = round((time.time() - t0) * 1000, 1)

            do_extract = (not pr.reject) or args.force_extract
            if args.skip_tse:
                do_extract = False

            if pr.reject and not args.force_extract:
                rec["decision"] = "reject"
                rec["reject_reason"] = pr.reason or "speaker_absent"
                rec["extracted_wav"] = None
                if debug_dir and extractor is not None:
                    try:
                        if backend in ("sep_route", "adaptive_route"):
                            dbg, meta = extractor.extract(
                                cmd,
                                enroll,
                                sr=sr,
                                streams=streams,
                                enroll_emb=enroll_emb,
                                preferred_stream=pr.best_stream,
                            )
                        else:
                            dbg, meta = extractor.extract(cmd, enroll, sr=sr)
                        dp = debug_dir / split / f"{uid}.wav"
                        save_audio(dp, dbg, sr)
                        rec["debug_tse_wav"] = str(dp)
                        rec["debug_tse_meta"] = meta
                    except Exception as e:
                        rec["debug_tse_error"] = str(e)
            elif do_extract and extractor is not None:
                t1 = time.time()
                try:
                    if backend in ("sep_route", "adaptive_route"):
                        out, meta = extractor.extract(
                            cmd,
                            enroll,
                            sr=sr,
                            streams=streams,
                            enroll_emb=enroll_emb,
                            preferred_stream=pr.best_stream,
                        )
                    else:
                        out, meta = extractor.extract(cmd, enroll, sr=sr)
                    if pr.best_window and backend == "mix" and getattr(args, "asr_crop", True):
                        from window_geom import crop_with_pad

                        out, wmeta = crop_with_pad(
                            out,
                            int(pr.best_window["start"]),
                            int(pr.best_window["end"]),
                            sr,
                            pad_ms=float(args.win_pad_ms),
                        )
                        meta = dict(meta or {})
                        meta["asr_crop"] = wmeta
                    elif pr.best_window and backend == "mix":
                        meta = dict(meta or {})
                        meta["asr_crop"] = {
                            "skipped": True,
                            "reason": "full_mix",
                            "best_start": int(pr.best_window["start"]),
                            "best_end": int(pr.best_window["end"]),
                        }
                    out_path = extracted_dir / split / f"{uid}.wav"
                    save_audio(out_path, out, sr)
                    rec["decision"] = "accept"
                    rec["reject_decision"] = False
                    rec["reject_reason"] = ""
                    rec["extracted_wav"] = str(out_path.resolve())
                    rec["tse_meta"] = meta
                    rec["tse_ms"] = round((time.time() - t1) * 1000, 1)
                except Exception as e:
                    rec["decision"] = "extract_error"
                    rec["extract_error"] = str(e)
                    rec["extract_traceback"] = traceback.format_exc(limit=5)
            else:
                rec["decision"] = "accept_no_tse" if not pr.reject else "reject"
            rec["elapsed_ms"] = round((time.time() - t0) * 1000, 1)
        except Exception as e:
            rec["decision"] = "pipeline_error"
            rec["error"] = str(e)
            rec["traceback"] = traceback.format_exc(limit=8)
            rec["elapsed_ms"] = round((time.time() - t0) * 1000, 1)

        by_split.setdefault(split, []).append(rec)
        n_done = i + 1
        if n_done % 500 == 0 or n_done == len(samples):
            msg = (
                f"[INFO] {n_done}/{len(samples)} last={uid} "
                f"decision={rec.get('decision')} score={rec.get('presence_score')} "
                f"thr={rec.get('presence_thr')}"
            )
            if tqdm is not None:
                tqdm.write(msg)
            else:
                print(msg, flush=True)

    all_rows: list[dict[str, Any]] = []
    for split, rows in by_split.items():
        path = results_dir / f"{split}_results.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                all_rows.append(r)
        print(f"[OK] {path} n={len(rows)}")

    with (results_dir / "all_results.jsonl").open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = write_run_reports(
        ve_out / "reports",
        all_rows,
        meta={
            "presence_thr": thr_default,
            "thr_mode": thr_mode,
            "thr_by_lang": thr_meta.get("thr_by_lang"),
            "presence_backend": enc.name,
            "use_sep": actual_depth >= 1,
            "sep_depth": actual_depth,
            "save_sep_wavs": bool(sep_root),
            "score_norm": want_mode,
            "tse_backend": None if extractor is None else extractor.name,
            "pipeline": backend,
            "elapsed_sec": round(time.time() - t_run0, 2),
            "n_samples": len(all_rows),
            "ve_out": str(ve_out),
            "force_extract": args.force_extract,
            "skip_tse": args.skip_tse,
            "thr_file": str(thr_file) if thr_file else None,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] reports → {ve_out / 'reports'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""实验 5–8：单 split 按第一轮 CER 分布阈值 a/b/c 子集再分离。

- --retry-failed：各 thr_* 只补 error/缺失，保留成功
- ClearVoice 路径：先分离关 daemon，再 ASR（减同卡 OOM）
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from asr_score import create_asr, score_wavs
from paths import (
    THR_NAMES,
    assert_split,
    default_vm_out,
    ensure_reports,
    ensure_stage,
    setup_sys_path,
    stage_dir,
    wav_path,
)
from progress_log import StageProgress
from report_cer_dist import compute_dist, report_from_index
from sep_common import close_sep, separate_batch_resilient, separate_one_with_oom_retry
from stage_resume import (
    count_index_rows,
    load_index_by_uid,
    partition_items,
    retry_failed,
    skip_message,
    stage_complete,
    write_index_merged,
)

setup_sys_path()
from utils_audio import load_audio, save_audio  # noqa: E402


class MissingParentWavError(FileNotFoundError):
    pass


def _load_parent_index(vm_out: Path, parent_stage: str, split: str) -> list[dict]:
    p = stage_dir(vm_out, parent_stage, split) / "index.jsonl"
    if not p.is_file():
        raise FileNotFoundError(f"需要先跑 {split}/{parent_stage}: {p}")
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        if o.get("error"):
            continue
        if o.get("split") and o["split"] != split:
            raise SystemExit(f"[ERR] 父 index 混入异 split: {o.get('uid')}")
        if "oracle_cer" in o and o["oracle_cer"] is not None:
            rows.append(o)
    return rows


def _resolve_thr(rows: list[dict], thr_arg: str) -> dict[str, float]:
    cers = [float(r["oracle_cer"]) for r in rows]
    if thr_arg.strip():
        parts = [float(x) for x in thr_arg.split(",")]
        if len(parts) != 3:
            raise SystemExit("--thr 需要 a,b,c")
        return {"a": parts[0], "b": parts[1], "c": parts[2]}
    return compute_dist(cers)["thr"]


def _subset(rows: list[dict], thr: float) -> list[dict]:
    return [r for r in rows if float(r["oracle_cer"]) >= thr]


def _base_rec(it: dict, parent_stage: str, thr_name: str, thr_val: float) -> dict:
    return {
        **{
            k: it[k]
            for k in ("uid", "split", "id", "kws_rel", "kws_path", "wake_text")
            if k in it
        },
        "parent_stage": parent_stage,
        "parent_oracle_cer": it["oracle_cer"],
        "thr_name": thr_name,
        "thr": thr_val,
    }


def _run_thr_subset(
    *,
    mode: str,
    sep,
    rows: list[dict],
    parent_root: Path,
    out_root: Path,
    peak: float,
    max_sep_sec: float,
    parent_stage: str,
    thr_name: str,
    thr_val: float,
    split: str,
    asr_model_dir: str,
    device: str,
    use_cv: bool,
) -> dict[str, Any]:
    """对单个 thr 子集：续跑/补失败；CV 则两阶段 sep→ASR。"""
    (out_root / "wav").mkdir(parents=True, exist_ok=True)
    index_path = out_root / "index.jsonl"
    order = [str(r["uid"]) for r in rows]

    force = os.environ.get("VM_FORCE", "").strip() in ("1", "true", "TRUE", "yes")
    existing = {} if force else load_index_by_uid(index_path)
    todo, keep = partition_items(rows, existing, only_failed=retry_failed() or not force)
    # partition always skips ok; when not retry and not force but incomplete stage,
    # stage_complete already false so we may re-enter with partial index — keep ok.
    if not todo and keep:
        write_index_merged(index_path, keep, order)
        cers = [r["oracle_cer"] for r in keep.values() if r.get("oracle_cer") is not None]
        return {
            "thr_name": thr_name,
            "thr": thr_val,
            "n_subset": len(rows),
            "n_ok": len(keep),
            "n_fail": 0,
            "mean_parent_cer": round(
                sum(float(r["oracle_cer"]) for r in rows) / len(rows), 4
            )
            if rows
            else None,
            "mean_oracle_cer": round(sum(cers) / len(cers), 4) if cers else None,
            "index": str(index_path.resolve()),
        }

    print(
        f"[INFO] {split}/thr_{thr_name} keep_ok={len(keep)} todo={len(todo)} mode={mode}",
        flush=True,
    )
    rows_by_uid: dict[str, dict] = dict(keep)
    pending: list[tuple[dict, dict, int]] = []
    tag = f"{split}/{thr_name}-sep"
    prog = StageProgress(len(todo), tag)

    for it in todo:
        uid = it["uid"]
        if not str(uid).startswith(f"{split}_"):
            raise SystemExit(f"[ERR] uid 前缀与 split 不符: {uid} vs {split}")
        try:
            if mode == "cv_peak":
                pp = wav_path(parent_root, uid, "peak")
                if not pp.is_file():
                    raise MissingParentWavError(
                        f"缺少父阶段 peak wav: {pp}（gated 禁止回退 kws 一阶）"
                    )
                peak_wav, sr = load_audio(pp, 16000)
                s1, s2 = separate_one_with_oom_retry(sep, peak_wav, sr, max_sep_sec)
                for t, w in (("peak", peak_wav), ("spk1", s1), ("spk2", s2)):
                    save_audio(wav_path(out_root, uid, t), w, sr)
                cands = {"original": peak_wav, "spk1": s1, "spk2": s2}
            else:
                p_peak = wav_path(parent_root, uid, "peak")
                p1 = wav_path(parent_root, uid, "spk1")
                p2 = wav_path(parent_root, uid, "spk2")
                if not (p_peak.is_file() and p1.is_file() and p2.is_file()):
                    raise MissingParentWavError(f"缺少父阶段 wav: {uid}")
                peak_wav, sr = load_audio(p_peak, 16000)
                s1, _ = load_audio(p1, 16000)
                s2, _ = load_audio(p2, 16000)
                save_audio(wav_path(out_root, uid, "peak"), peak_wav, sr)
                save_audio(wav_path(out_root, uid, "spk1"), s1, sr)
                save_audio(wav_path(out_root, uid, "spk2"), s2, sr)
                outs = separate_batch_resilient(
                    sep, [s1, s2], sr=sr, max_sep_sec=max_sep_sec
                )
                if isinstance(outs[0], Exception):
                    raise outs[0]
                if isinstance(outs[1], Exception):
                    raise outs[1]
                (a1, a2), (b1, b2) = outs[0], outs[1]
                for t, w in (
                    ("spk1_r1", a1),
                    ("spk1_r2", a2),
                    ("spk2_r1", b1),
                    ("spk2_r2", b2),
                ):
                    save_audio(wav_path(out_root, uid, t), w, sr)
                cands = {
                    "original": peak_wav,
                    "spk1": s1,
                    "spk2": s2,
                    "spk1_r1": a1,
                    "spk1_r2": a2,
                    "spk2_r1": b1,
                    "spk2_r2": b2,
                }
            pending.append((it, cands, sr))
            prog.tick(uid=uid, ok=True, cer=None)
        except Exception as e:
            rows_by_uid[uid] = {
                "uid": uid,
                "split": split,
                "error": str(e),
                "thr_name": thr_name,
            }
            prog.tick(uid=uid, ok=False, err=str(e))
        if prog.i % 20 == 0:
            write_index_merged(index_path, rows_by_uid, order)

    prog.close()

    if not pending:
        write_index_merged(index_path, rows_by_uid, order)
        ok_rows = [
            rows_by_uid[u] for u in order if u in rows_by_uid and not rows_by_uid[u].get("error")
        ]
        fail_n = sum(1 for u in order if u in rows_by_uid and rows_by_uid[u].get("error"))
        return {
            "thr_name": thr_name,
            "thr": thr_val,
            "n_subset": len(rows),
            "n_ok": len(ok_rows),
            "n_fail": fail_n,
            "mean_parent_cer": round(
                sum(float(r["oracle_cer"]) for r in rows) / len(rows), 4
            )
            if rows
            else None,
            "mean_oracle_cer": None,
            "index": str(index_path.resolve()),
        }

    print(f"[INFO] thr_{thr_name} release sep → ASR (n={len(pending)})", flush=True)
    close_sep(sep)

    asr = create_asr(device=device, model_dir=asr_model_dir.strip() or None)
    prog2 = StageProgress(len(pending), f"{split}/{thr_name}-asr")
    for it, cands, sr in pending:
        uid = it["uid"]
        wake = it.get("wake_text") or ""
        try:
            if wake.strip():
                scored = score_wavs(asr, cands, wake, sr)
            else:
                scored = {
                    "oracle_stream": None,
                    "oracle_cer": None,
                    "oracle_hyp": None,
                    "metric": "no_wake",
                    "streams": {},
                }
            rows_by_uid[uid] = {**_base_rec(it, parent_stage, thr_name, thr_val), **scored}
            prog2.tick(uid=uid, ok=True, cer=scored.get("oracle_cer"))
        except Exception as e:
            rows_by_uid[uid] = {
                "uid": uid,
                "split": split,
                "error": f"asr_score: {e}",
                "thr_name": thr_name,
            }
            prog2.tick(uid=uid, ok=False, err=str(e))
        if prog2.i % 20 == 0:
            write_index_merged(index_path, rows_by_uid, order)

    prog2.close()
    write_index_merged(index_path, rows_by_uid, order)

    ok_rows = [
        rows_by_uid[u] for u in order if u in rows_by_uid and not rows_by_uid[u].get("error")
    ]
    fail_n = sum(1 for u in order if u in rows_by_uid and rows_by_uid[u].get("error"))
    cers = [r["oracle_cer"] for r in ok_rows if r.get("oracle_cer") is not None]
    return {
        "thr_name": thr_name,
        "thr": thr_val,
        "n_subset": len(rows),
        "n_ok": len(ok_rows),
        "n_fail": fail_n,
        "mean_parent_cer": round(
            sum(float(r["oracle_cer"]) for r in rows) / len(rows), 4
        )
        if rows
        else None,
        "mean_oracle_cer": round(sum(cers) / len(cers), 4) if cers else None,
        "index": str(index_path.resolve()),
    }


def _gated_has_fails(vm_out: Path, stage: str, split: str) -> bool:
    root = stage_dir(vm_out, stage, split)
    for name in THR_NAMES:
        ip = root / f"thr_{name}" / "index.jsonl"
        if not ip.is_file():
            continue
        total, ok = count_index_rows(ip)
        if total > ok:
            return True
    return False


def run_gated(stage: str, vm_out: Path, split: str, args: argparse.Namespace) -> None:
    split = assert_split(split)
    cfg = {
        "s5": {"parent": "s1", "mode": "cv_peak", "sep": "cv"},
        "s6": {"parent": "s1", "mode": "refine", "sep": "onnx"},
        "s7": {"parent": "s2", "mode": "refine", "sep": "onnx"},
        "s8": {"parent": "s2", "mode": "refine", "sep": "cv"},
    }[stage]
    parent = cfg["parent"]

    if stage_complete(vm_out, stage, split, n_expected=1, gated=True):
        if retry_failed() and _gated_has_fails(vm_out, stage, split):
            print(f"[INFO] {split}/{stage} 有失败，进入 --retry-failed", flush=True)
        else:
            skip_message(stage, split)
            return

    rows_all = _load_parent_index(vm_out, parent, split)
    catalog_n = len(rows_all)
    thr_map = _resolve_thr(rows_all, args.thr)
    rows = rows_all[: args.limit] if args.limit > 0 else rows_all
    print(f"[INFO] {split}/{stage} parent={parent} thr={thr_map}")

    reports = ensure_reports(vm_out, split)
    report_from_index(
        stage_dir(vm_out, parent, split) / "index.jsonl",
        reports / f"{parent}_cer_dist.json",
        reports / f"{parent}_cer_dist.md",
        [thr_map["a"], thr_map["b"], thr_map["c"]],
    )

    stage_root = ensure_stage(vm_out, stage, split)
    parent_root = stage_dir(vm_out, parent, split)
    use_cv = cfg["sep"] == "cv"

    by_thr = {}
    for name in THR_NAMES:
        thr_val = float(thr_map[name])
        sub = _subset(rows, thr_val)
        sub_dir = stage_root / f"thr_{name}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {split}/{stage} thr_{name}={thr_val} n={len(sub)} ===")
        if not sub:
            by_thr[name] = {"thr": thr_val, "n_subset": 0, "n_ok": 0, "n_fail": 0}
            continue

        sep = None
        try:
            if cfg["sep"] == "onnx":
                from sep_onnx import create_onnx_separator

                sep = create_onnx_separator(peak=args.peak, device=args.device)
            else:
                from sep_cv import create_cv_separator

                os.environ.setdefault("MOSS_GPU_FRAC", "0.92")
                sep = create_cv_separator(peak=args.peak, device=args.device)

            summary = _run_thr_subset(
                mode=cfg["mode"],
                sep=sep,
                rows=sub,
                parent_root=parent_root,
                out_root=sub_dir,
                peak=args.peak,
                max_sep_sec=args.max_sep_sec,
                parent_stage=parent,
                thr_name=name,
                thr_val=thr_val,
                split=split,
                asr_model_dir=args.asr_model_dir,
                device=args.device,
                use_cv=use_cv,
            )
        finally:
            close_sep(sep)
        by_thr[name] = summary
        (sub_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    overall = {
        "stage": stage,
        "split": split,
        "parent_stage": parent,
        "mode": cfg["mode"],
        "sep_backend": cfg["sep"],
        "thr": thr_map,
        "by_thr": by_thr,
        "catalog_n": int(catalog_n),
        "limit": int(args.limit or 0),
        "partial": bool(args.limit and int(args.limit) > 0),
        "stage_root": str(stage_root.resolve()),
    }
    (stage_root / "summary.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] {split}/{stage}", json.dumps(overall, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="VM gated re-sep (per split)")
    ap.add_argument("--stage", required=True, choices=["s5", "s6", "s7", "s8"])
    ap.add_argument("--split", required=True, choices=["pos", "neg"])
    ap.add_argument("--vm-out", type=str, default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--thr", type=str, default="", help="a,b,c override")
    ap.add_argument("--peak", type=float, default=0.7)
    ap.add_argument("--max-sep-sec", type=float, default=6.0)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--asr-model-dir", type=str, default=os.environ.get("ASR_MODEL_DIR", ""))
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--retry-failed",
        action="store_true",
        help="各 thr_* 只重跑 error/缺失样本",
    )
    args = ap.parse_args()
    if args.force:
        os.environ["VM_FORCE"] = "1"
    if args.retry_failed:
        os.environ["VM_RETRY_FAILED"] = "1"
    vm_out = Path(args.vm_out) if args.vm_out else default_vm_out()
    run_gated(args.stage, vm_out, args.split, args)


if __name__ == "__main__":
    main()

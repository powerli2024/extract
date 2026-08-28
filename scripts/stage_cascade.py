#!/usr/bin/env python3
"""实验 3/4：单 split cascade；复用同 split 下父阶段 wav。

- 断点续跑 / --retry-failed：只补失败与缺失
- ClearVoice(s4)：先二阶分离并关闭 daemon，再加载 ASR（避免同卡 OOM）
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from asr_score import create_asr, score_wavs
from collect_kws import load_items
from paths import (
    assert_split,
    default_vm_out,
    ensure_stage,
    setup_sys_path,
    stage_dir,
    wav_path,
)
from progress_log import StageProgress
from sep_common import close_sep, separate_batch_resilient, separate_one_with_oom_retry
from stage_resume import (
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
    """Cascade 必须吃父阶段 wav；禁止回退成一阶分离。"""


def _load_parent_streams(
    *,
    parent: Path,
    uid: str,
    stage_root: Path,
):
    p_peak = wav_path(parent, uid, "peak")
    p1 = wav_path(parent, uid, "spk1")
    p2 = wav_path(parent, uid, "spk2")
    missing = [str(p) for p in (p_peak, p1, p2) if not p.is_file()]
    if missing:
        raise MissingParentWavError(
            f"cascade 父阶段 wav 缺失 uid={uid} parent={parent.name}: {missing}"
        )
    peak_wav, sr = load_audio(p_peak, 16000)
    s1, _ = load_audio(p1, 16000)
    s2, _ = load_audio(p2, 16000)
    save_audio(wav_path(stage_root, uid, "peak"), peak_wav, sr)
    save_audio(wav_path(stage_root, uid, "spk1"), s1, sr)
    save_audio(wav_path(stage_root, uid, "spk2"), s2, sr)
    return peak_wav, s1, s2, sr


def _cascade_second(sep, s1, s2, sr: int, max_sep_sec: float):
    """二阶：对 s1/s2 再分离。单条失败不拖垮另一条。"""
    outs = separate_batch_resilient(sep, [s1, s2], sr=sr, max_sep_sec=max_sep_sec)
    if isinstance(outs[0], Exception):
        raise outs[0]
    if isinstance(outs[1], Exception):
        raise outs[1]
    return outs[0], outs[1]


def run_cascade(
    *,
    stage: str,
    parent_stage: str,
    backend: str,
    vm_out: Path,
    split: str,
    items: list[dict],
    peak: float,
    max_sep_sec: float,
    device: str,
    asr_model_dir: str,
    catalog_n: int,
    limit: int,
) -> Path:
    split = assert_split(split)
    stage_root = ensure_stage(vm_out, stage, split)
    parent = stage_dir(vm_out, parent_stage, split)
    index_path = stage_root / "index.jsonl"
    order = [str(it["uid"]) for it in items]

    if stage_complete(vm_out, stage, split, catalog_n):
        skip_message(stage, split, f" index={index_path}")
        return index_path

    existing = (
        {}
        if os.environ.get("VM_FORCE", "").strip() in ("1", "true", "TRUE", "yes")
        else load_index_by_uid(index_path)
    )
    only_failed = retry_failed()
    todo, keep = partition_items(items, existing, only_failed=only_failed)

    if not todo:
        print(
            f"[SKIP] {split}/{stage} 无待办（成功 {len(keep)}/{len(items)}）",
            flush=True,
        )
        if keep:
            write_index_merged(index_path, keep, order)
        return index_path

    print(
        f"[INFO] {split}/{stage} resume: keep_ok={len(keep)} todo={len(todo)} "
        f"parent={parent_stage} retry_failed={only_failed}",
        flush=True,
    )

    if backend == "onnx":
        from sep_onnx import create_onnx_separator

        sep = create_onnx_separator(peak=peak, device=device)
    else:
        from sep_cv import create_cv_separator

        os.environ.setdefault("MOSS_GPU_FRAC", "0.92")
        sep = create_cv_separator(peak=peak, device=device)

    rows_by_uid: dict[str, dict] = dict(keep)
    # phase1: uid -> (it, cands_dict, sr) 已落盘二阶 wav
    pending_score: list[tuple[dict, dict, int]] = []
    prog = StageProgress(len(todo), f"{split}/{stage}-sep")

    for it in todo:
        if it["split"] != split:
            raise SystemExit(f"[ERR] item split 混入: {it}")
        uid = it["uid"]
        try:
            peak_wav, s1, s2, sr = _load_parent_streams(
                parent=parent,
                uid=uid,
                stage_root=stage_root,
            )
            (a1, a2), (b1, b2) = _cascade_second(sep, s1, s2, sr, max_sep_sec)
            for tag, w in (
                ("spk1_r1", a1),
                ("spk1_r2", a2),
                ("spk2_r1", b1),
                ("spk2_r2", b2),
            ):
                save_audio(wav_path(stage_root, uid, tag), w, sr)
            cands = {
                "original": peak_wav,
                "spk1": s1,
                "spk2": s2,
                "spk1_r1": a1,
                "spk1_r2": a2,
                "spk2_r1": b1,
                "spk2_r2": b2,
            }
            pending_score.append((it, cands, sr))
            prog.tick(uid=uid, ok=True, cer=None)
        except Exception as e:
            rows_by_uid[uid] = {
                **{k: it[k] for k in ("uid", "split", "id", "kws_rel", "kws_path", "wake_text")},
                "stage": stage,
                "error": str(e),
            }
            prog.tick(uid=uid, ok=False, err=str(e))
        if prog.i % 20 == 0:
            write_index_merged(index_path, rows_by_uid, order)

    prog.close()
    print(
        f"[INFO] {split}/{stage} phase1 done sep_ok={len(pending_score)}; release sep → ASR",
        flush=True,
    )
    close_sep(sep)
    sep = None

    asr = create_asr(device=device, model_dir=asr_model_dir.strip() or None)
    prog2 = StageProgress(len(pending_score), f"{split}/{stage}-asr")
    for it, cands, sr in pending_score:
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
            rows_by_uid[uid] = {
                **{k: it[k] for k in ("uid", "split", "id", "kws_rel", "kws_path", "wake_text")},
                "stage": stage,
                "backend": backend,
                "parent_stage": parent_stage,
                **scored,
            }
            prog2.tick(uid=uid, ok=True, cer=scored.get("oracle_cer"))
        except Exception as e:
            rows_by_uid[uid] = {
                **{k: it[k] for k in ("uid", "split", "id", "kws_rel", "kws_path", "wake_text")},
                "stage": stage,
                "error": f"asr_score: {e}",
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
    ok_cers = [r["oracle_cer"] for r in ok_rows if r.get("oracle_cer") is not None]
    summary = {
        "stage": stage,
        "split": split,
        "backend": backend,
        "parent_stage": parent_stage,
        "n_items": len(items),
        "catalog_n": int(catalog_n),
        "limit": int(limit or 0),
        "partial": bool(limit and int(limit) > 0),
        "n_ok": len(ok_rows),
        "n_fail": fail_n,
        "mean_oracle_cer": round(sum(ok_cers) / len(ok_cers), 4) if ok_cers else None,
        "index": str(index_path.resolve()),
    }
    (stage_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] {split}/{stage}", summary, flush=True)
    return index_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["s3", "s4"])
    ap.add_argument("--split", required=True, choices=["pos", "neg"])
    ap.add_argument("--vm-out", type=str, default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--peak", type=float, default=0.7)
    ap.add_argument(
        "--max-sep-sec", type=float, default=0.0,
        help="deprecated compatibility option; full utterances are always preserved",
    )
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--asr-model-dir", type=str, default=os.environ.get("ASR_MODEL_DIR", ""))
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--retry-failed",
        action="store_true",
        help="只重跑 error/缺失样本，保留成功结果",
    )
    args = ap.parse_args()
    if args.force:
        os.environ["VM_FORCE"] = "1"
    if args.retry_failed:
        os.environ["VM_RETRY_FAILED"] = "1"

    vm_out = Path(args.vm_out) if args.vm_out else default_vm_out()
    items_all = load_items(vm_out, args.split)
    catalog_n = len(items_all)
    items = items_all[: args.limit] if args.limit > 0 else items_all
    common = dict(
        vm_out=vm_out,
        split=args.split,
        items=items,
        peak=args.peak,
        max_sep_sec=args.max_sep_sec,
        device=args.device,
        asr_model_dir=args.asr_model_dir,
        catalog_n=catalog_n,
        limit=int(args.limit or 0),
    )

    if args.stage == "s3":
        run_cascade(stage="s3", parent_stage="s1", backend="onnx", **common)
    else:
        run_cascade(stage="s4", parent_stage="s2", backend="cv", **common)


if __name__ == "__main__":
    main()

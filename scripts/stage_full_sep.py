#!/usr/bin/env python3
"""实验 1/2：单 split 全量一阶分离（pos/neg 分树）。

- 默认跳过已完成阶段（不覆盖）；VM_FORCE=1 强制全量重跑
- 断点续跑：已有成功 uid 自动跳过
- --retry-failed / VM_RETRY_FAILED=1：只重跑 error / 缺失样本
- ClearVoice：先分离并关闭 daemon 释放显存，再加载 ASR（避免同卡 OOM）
- ONNX：VM_SEP_BATCH 条并行 separate_many
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
    ensure_reports,
    ensure_stage,
    setup_sys_path,
    wav_path,
)
from progress_log import StageProgress
from report_cer_dist import report_from_index
from stage_resume import (
    load_index_by_uid,
    partition_items,
    retry_failed,
    skip_message,
    stage_complete,
    write_index_merged,
)

setup_sys_path()
from utils_audio import load_audio, peak_normalize, save_audio, truncate_wav  # noqa: E402


def _sep_batch_size(backend: str) -> int:
    env = os.environ.get("VM_SEP_BATCH", "").strip()
    if env.isdigit():
        return max(1, int(env))
    return 8 if backend == "onnx" else 1


def _is_oom(err: BaseException | str) -> bool:
    s = str(err).lower()
    return "out of memory" in s or "oom" in s or "cudaerrormemoryallocation" in s


def _separate_one(sep, peak_wav, sr: int, max_sep_sec: float):
    """分离；OOM 时 empty_cache + 缩短时长再试。"""
    try:
        return sep.separate(peak_wav, sr=sr, max_sec=max_sep_sec)
    except Exception as e:
        if not _is_oom(e):
            raise
        if hasattr(sep, "empty_cache"):
            try:
                sep.empty_cache()
            except Exception:
                pass
        # 难样本：再压到 ≤3s
        shorter = 3.0
        if max_sep_sec > 0:
            shorter = min(float(max_sep_sec), 3.0)
        if max_sep_sec > 0 and shorter >= float(max_sep_sec) - 1e-6:
            # 已经很短仍 OOM：再砍到 2s
            shorter = min(shorter, 2.0)
        print(f"[WARN] sep OOM → retry max_sep_sec={shorter}", flush=True)
        wav = peak_wav
        if shorter > 0:
            wav = truncate_wav(peak_wav, sr=sr, max_sec=shorter, mode="energy")
        return sep.separate(wav, sr=sr, max_sec=shorter)


def _close_sep(sep) -> None:
    for name in ("close", "release_gpu"):
        fn = getattr(sep, name, None)
        if callable(fn):
            try:
                fn()
            except Exception as e:
                print(f"[WARN] sep.{name}: {e}", flush=True)
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_full_sep(
    *,
    stage: str,
    backend: str,
    vm_out: Path,
    split: str,
    items: list[dict],
    peak: float,
    max_sep_sec: float,
    device: str,
    asr_model_dir: str,
) -> Path:
    split = assert_split(split)
    stage_root = ensure_stage(vm_out, stage, split)
    index_path = stage_root / "index.jsonl"
    order = [str(it["uid"]) for it in items]

    if stage_complete(vm_out, stage, split, len(items)):
        skip_message(stage, split, f" index={index_path}")
        return index_path

    existing = {} if os.environ.get("VM_FORCE", "").strip() in ("1", "true", "TRUE", "yes") else load_index_by_uid(index_path)
    only_failed = retry_failed()
    todo, keep = partition_items(items, existing, only_failed=only_failed)

    if not todo:
        print(
            f"[SKIP] {split}/{stage} 无待办（成功 {len(keep)}/{len(items)}）。"
            f"若要全量覆盖用 --force",
            flush=True,
        )
        if keep:
            write_index_merged(index_path, keep, order)
        return index_path

    print(
        f"[INFO] {split}/{stage} resume: keep_ok={len(keep)} todo={len(todo)} "
        f"total={len(items)} retry_failed={only_failed}",
        flush=True,
    )

    if backend == "onnx":
        from sep_onnx import create_onnx_separator

        sep = create_onnx_separator(peak=peak, device=device)
    elif backend == "cv":
        from sep_cv import create_cv_separator

        # ClearVoice 独占 GPU 阶段：降低显存上限，稍后关闭再上 ASR
        os.environ.setdefault("MOSS_GPU_FRAC", "0.92")
        sep = create_cv_separator(peak=peak, device=device)
    else:
        raise ValueError(backend)

    rows_by_uid: dict[str, dict] = dict(keep)
    # phase1 产物：uid -> (it, peak, s1, s2, sr) 或 error
    pending_score: list[tuple[dict, object, object, object, int]] = []
    prog = StageProgress(len(todo), f"{split}/{stage}-sep")
    batch_n = _sep_batch_size(backend)
    print(
        f"[INFO] {split}/{stage} phase1=separate sep_batch={batch_n} backend={backend}",
        flush=True,
    )

    for start in range(0, len(todo), batch_n):
        batch = todo[start : start + batch_n]
        prepared: list[tuple[dict, object, int]] = []

        for it in batch:
            if it["split"] != split:
                raise SystemExit(f"[ERR] item split 混入: {it}")
            try:
                raw, sr = load_audio(it["kws_path"], 16000)
                peak_wav = peak_normalize(raw, peak=peak)
                if max_sep_sec > 0:
                    peak_wav = truncate_wav(
                        peak_wav, sr=sr, max_sec=max_sep_sec, mode="energy"
                    )
                prepared.append((it, peak_wav, sr))
            except Exception as e:
                prepared.append((it, e, 16000))

        to_sep_idx = [i for i, (_, w, _) in enumerate(prepared) if not isinstance(w, Exception)]
        sep_outs: dict[int, object] = {}
        if to_sep_idx:
            # ClearVoice 逐条（含 OOM 降级）；ONNX 可 batch
            if backend == "onnx" and hasattr(sep, "separate_many") and len(to_sep_idx) > 1:
                wavs = [prepared[i][1] for i in to_sep_idx]
                sr0 = prepared[to_sep_idx[0]][2]
                try:
                    outs = sep.separate_many(wavs, sr=sr0, max_sec=max_sep_sec)
                    for j, oi in enumerate(to_sep_idx):
                        sep_outs[oi] = outs[j]
                except Exception as e:
                    for oi in to_sep_idx:
                        sep_outs[oi] = e
            else:
                for oi in to_sep_idx:
                    it_i, wav_i, sr_i = prepared[oi]
                    try:
                        sep_outs[oi] = _separate_one(sep, wav_i, sr_i, max_sep_sec)
                    except Exception as e:
                        sep_outs[oi] = e

        for i, (it, peak_or_err, sr) in enumerate(prepared):
            uid = it["uid"]
            try:
                if isinstance(peak_or_err, Exception):
                    raise peak_or_err
                peak_wav = peak_or_err
                out = sep_outs.get(i)
                if isinstance(out, Exception):
                    raise out
                if out is None:
                    s1, s2 = _separate_one(sep, peak_wav, sr, max_sep_sec)
                else:
                    s1, s2 = out
                save_audio(wav_path(stage_root, uid, "peak"), peak_wav, sr)
                save_audio(wav_path(stage_root, uid, "spk1"), s1, sr)
                save_audio(wav_path(stage_root, uid, "spk2"), s2, sr)
                pending_score.append((it, peak_wav, s1, s2, sr))
                prog.tick(uid=uid, ok=True, cer=None)
            except Exception as e:
                rows_by_uid[uid] = {
                    **{
                        k: it[k]
                        for k in (
                            "uid",
                            "split",
                            "id",
                            "kws_rel",
                            "kws_path",
                            "wake_text",
                        )
                    },
                    "stage": stage,
                    "error": str(e),
                }
                prog.tick(uid=uid, ok=False, err=str(e))

        write_index_merged(index_path, rows_by_uid, order)

    prog.close()

    # 释放分离模型显存，再上 ASR（解决 CV+ASR 同卡 OOM）
    print(f"[INFO] {split}/{stage} phase1 done sep_ok={len(pending_score)}; release sep → load ASR", flush=True)
    _close_sep(sep)
    sep = None

    asr = create_asr(device=device, model_dir=asr_model_dir.strip() or None)
    prog2 = StageProgress(len(pending_score), f"{split}/{stage}-asr")
    n_no_wake = 0
    for it, peak_wav, s1, s2, sr in pending_score:
        uid = it["uid"]
        wake = it.get("wake_text") or ""
        try:
            if not wake.strip():
                n_no_wake += 1
                scored = {
                    "oracle_stream": None,
                    "oracle_cer": None,
                    "oracle_hyp": None,
                    "metric": "no_wake",
                    "streams": {},
                    "note": "无唤醒文本，仅保存分离音频",
                }
            else:
                scored = score_wavs(
                    asr,
                    {"original": peak_wav, "spk1": s1, "spk2": s2},
                    wake,
                    sr,
                )
            rows_by_uid[uid] = {
                **{
                    k: it[k]
                    for k in ("uid", "split", "id", "kws_rel", "kws_path", "wake_text")
                },
                "stage": stage,
                "backend": backend,
                **scored,
            }
            prog2.tick(uid=uid, ok=True, cer=scored.get("oracle_cer"))
        except Exception as e:
            rows_by_uid[uid] = {
                **{
                    k: it[k]
                    for k in (
                        "uid",
                        "split",
                        "id",
                        "kws_rel",
                        "kws_path",
                        "wake_text",
                    )
                },
                "stage": stage,
                "error": f"asr_score: {e}",
            }
            prog2.tick(uid=uid, ok=False, err=str(e))
        if prog2.i % 20 == 0:
            write_index_merged(index_path, rows_by_uid, order)

    prog2.close()
    write_index_merged(index_path, rows_by_uid, order)

    ok_rows = [rows_by_uid[u] for u in order if u in rows_by_uid and not rows_by_uid[u].get("error")]
    fail_n = sum(1 for u in order if u in rows_by_uid and rows_by_uid[u].get("error"))
    ok_cers = [r["oracle_cer"] for r in ok_rows if r.get("oracle_cer") is not None]
    summary = {
        "stage": stage,
        "split": split,
        "backend": backend,
        "n_items": len(items),
        "n_ok": len(ok_rows),
        "n_fail": fail_n,
        "n_no_wake": sum(1 for r in ok_rows if r.get("metric") == "no_wake"),
        "mean_oracle_cer": round(sum(ok_cers) / len(ok_cers), 4) if ok_cers else None,
        "index": str(index_path.resolve()),
        "stage_root": str(stage_root.resolve()),
    }
    (stage_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if ok_cers:
        reports = ensure_reports(vm_out, split)
        report_from_index(
            index_path,
            reports / f"{stage}_cer_dist.json",
            reports / f"{stage}_cer_dist.md",
        )
    print(f"[OK] {split}/{stage}", summary, flush=True)
    return index_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["s1", "s2"])
    ap.add_argument("--split", required=True, choices=["pos", "neg"])
    ap.add_argument("--vm-out", type=str, default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--peak", type=float, default=0.7)
    ap.add_argument("--max-sep-sec", type=float, default=6.0)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--asr-model-dir", type=str, default=os.environ.get("ASR_MODEL_DIR", ""))
    ap.add_argument("--force", action="store_true", help="强制全量重跑并覆盖")
    ap.add_argument(
        "--retry-failed",
        action="store_true",
        help="只重跑 index 里带 error 或缺失的样本（保留成功结果）",
    )
    args = ap.parse_args()
    if args.force:
        os.environ["VM_FORCE"] = "1"
    if args.retry_failed:
        os.environ["VM_RETRY_FAILED"] = "1"

    vm_out = Path(args.vm_out) if args.vm_out else default_vm_out()
    items = load_items(vm_out, args.split)
    if args.limit > 0:
        items = items[: args.limit]
    backend = "onnx" if args.stage == "s1" else "cv"
    run_full_sep(
        stage=args.stage,
        backend=backend,
        vm_out=vm_out,
        split=args.split,
        items=items,
        peak=args.peak,
        max_sep_sec=args.max_sep_sec,
        device=args.device,
        asr_model_dir=args.asr_model_dir,
    )


if __name__ == "__main__":
    main()

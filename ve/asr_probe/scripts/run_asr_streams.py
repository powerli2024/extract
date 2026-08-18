#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对 pos/neg 的 mix / d1 / d2 全部轨跑 Qwen3-ASR（不做 Presence / TSE）。

产出 asr_results.jsonl：一行 = (uid, arm, stream)。
断点键：uid|arm|stream。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

VE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VE_ROOT / "scripts"))

from asr_cer import (  # noqa: E402
    MODEL_ID,
    Qwen3ASRBackend,
    compute_cer,
    guess_language,
    normalize_for_cer,
    resolve_asr_dir,
)
from audio_io import load_audio  # noqa: E402

NORM_VER = "asr-probe-v1"
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def rec_key(uid: str, arm: str, stream: str) -> str:
    return f"{uid}|{arm}|{stream}"


def _rms(x: Any) -> float:
    import numpy as np

    a = np.asarray(x, dtype=np.float32).reshape(-1)
    return float((a.astype("float64") ** 2).mean() ** 0.5 + 1e-12)


def list_stream_wavs(
    sep_root: Path,
    arm: str,
    split: str,
    uid: str,
    cmd_wav: Optional[str],
    *,
    include_mix_in_sep: bool,
) -> list[tuple[str, Path]]:
    """列出该臂要转写的 (stream_name, wav_path)。

    no_sep     : mix（d1/mix.wav 或 cmd_wav）
    sep_once   : d1_*（默认不含 mix，避免与 no_sep 重复 ASR）
    sep_multi  : d2 下非 peak 轨（默认不含 mix）
    """
    if arm not in ARM_DEPTH:
        raise ValueError(f"未知 arm={arm}")
    depth = ARM_DEPTH[arm]
    if depth == 0:
        mix_p = sep_root / "d1" / str(split) / str(uid) / "mix.wav"
        if mix_p.is_file():
            return [("mix", mix_p)]
        if cmd_wav and Path(cmd_wav).is_file():
            return [("mix", Path(cmd_wav))]
        return []

    utt_dir = sep_root / f"d{depth}" / str(split) / str(uid)
    if not utt_dir.is_dir():
        return []
    out: list[tuple[str, Path]] = []
    for p in sorted(utt_dir.glob("*.wav")):
        name = p.stem
        if name == "peak":
            continue
        if name == "mix" and not include_mix_in_sep:
            continue
        if depth == 1 and name != "mix" and not name.startswith("d1_"):
            continue
        out.append((name, p))
    return out


def collect_jobs(
    samples: list[dict[str, Any]],
    sep_root: Path,
    arms: list[str],
    *,
    include_mix_in_sep: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    missing: dict[str, int] = {f"no_{a}": 0 for a in arms}
    n_by_arm: dict[str, int] = {a: 0 for a in arms}
    n_utt_by_arm: dict[str, int] = {a: 0 for a in arms}
    for s in samples:
        uid = str(s["uid"])
        split = str(s.get("split") or "")
        cmd = s.get("cmd_wav")
        for arm in arms:
            wavs = list_stream_wavs(
                sep_root, arm, split, uid, cmd, include_mix_in_sep=include_mix_in_sep
            )
            if not wavs:
                missing[f"no_{arm}"] += 1
                continue
            n_utt_by_arm[arm] += 1
            for stream, path in wavs:
                n_by_arm[arm] += 1
                jobs.append(
                    {
                        "uid": uid,
                        "split": split,
                        "id": s.get("id"),
                        "label": s.get("label"),
                        "lang": s.get("lang"),
                        "wake_text": s.get("wake_text"),
                        "cmd_text": s.get("cmd_text"),
                        "cmd_wav": cmd,
                        "arm": arm,
                        "stream": stream,
                        "wav": str(path.resolve()),
                    }
                )
    plan = {
        "n_samples": len(samples),
        "n_jobs": len(jobs),
        "n_jobs_by_arm": n_by_arm,
        "n_utt_with_streams_by_arm": n_utt_by_arm,
        "n_utt_missing_arm": missing,
        "include_mix_in_sep": include_mix_in_sep,
        "arms": arms,
        "sep_root": str(sep_root),
    }
    return jobs, plan


def wake_overlap(wake: Optional[str], hyp_norm: str) -> bool:
    wn = normalize_for_cer(wake)
    if not wn or not hyp_norm:
        return False
    if wn in hyp_norm or hyp_norm in wn:
        return True
    return float(compute_cer(wn, hyp_norm)["cer"]) <= 0.3


def build_record(
    job: dict[str, Any],
    *,
    hyp: Optional[str],
    status: str,
    error: Optional[str],
    asr_ms: Optional[float],
    rms: Optional[float],
    dur_sec: Optional[float],
    language: Optional[str],
    model: str,
) -> dict[str, Any]:
    hyp_s = "" if hyp is None else str(hyp)
    hyp_norm = normalize_for_cer(hyp_s)
    ref = job.get("cmd_text")
    ref_s = "" if ref is None else str(ref)
    is_pos = str(job.get("split") or "") == "pos" or str(job.get("label") or "") == "present"
    rec: dict[str, Any] = {
        "uid": job["uid"],
        "split": job.get("split"),
        "id": job.get("id"),
        "label": job.get("label"),
        "lang": job.get("lang"),
        "wake_text": job.get("wake_text"),
        "cmd_text": ref,
        "arm": job["arm"],
        "stream": job["stream"],
        "wav": job.get("wav"),
        "dur_sec": dur_sec,
        "rms": None if rms is None else round(float(rms), 8),
        "language": language,
        "asr_text": hyp_s if status not in ("missing_wav", "low_rms") else None,
        "hyp_norm": hyp_norm if status not in ("missing_wav", "low_rms") else None,
        "hyp_nchars": len(hyp_norm) if hyp_norm else 0,
        "empty_hyp": bool(status in ("ok", "empty_hyp") and not hyp_norm),
        "wake_overlap": wake_overlap(job.get("wake_text"), hyp_norm) if hyp_norm else False,
        "ref_norm": None,
        "s": None,
        "d": None,
        "i": None,
        "n": None,
        "edit_distance": None,
        "cer": None,
        "ref_aligned": None,
        "hyp_aligned": None,
        "status": status,
        "error": error,
        "asr_ms": asr_ms,
        "norm_ver": NORM_VER,
        "model": model,
    }
    if is_pos and ref is not None and status in ("ok", "empty_hyp"):
        ref_norm = normalize_for_cer(ref_s)
        detail = compute_cer(ref_norm, hyp_norm)
        rec.update(
            {
                "ref_norm": ref_norm,
                "s": detail["s"],
                "d": detail["d"],
                "i": detail["i"],
                "n": detail["n"],
                "edit_distance": detail["dist"],
                "cer": detail["cer"],
                "ref_aligned": detail["ref_aligned"],
                "hyp_aligned": detail["hyp_aligned"],
            }
        )
        if not hyp_norm:
            rec["status"] = "empty_hyp"
    return rec


def transcribe_chunk(
    asr: Optional[Qwen3ASRBackend],
    wavs: list[Any],
    *,
    language: Optional[str],
    wake: Optional[str],
    fake: Optional[str],
    refs: list[str],
) -> list[tuple[Optional[str], str, Optional[str]]]:
    """返回 [(hyp, status, error), ...]。"""
    if fake:
        from asr_cer import fake_hyp

        out = []
        for ref in refs:
            h = fake_hyp(ref or "", fake) if ref else ""
            out.append((h, "ok" if h else "empty_hyp", None))
        return out
    assert asr is not None
    try:
        hyps = asr.transcribe_many(wavs, language=language, wake_text=wake)
        return [(h, "ok" if (h or "").strip() else "empty_hyp", None) for h in hyps]
    except Exception as e:  # noqa: BLE001
        if len(wavs) == 1:
            return [(None, "asr_error", f"{e}\n{traceback.format_exc(limit=4)}")]
        # OOM / batch 失败：逐条重试
        rows = []
        for w, ref in zip(wavs, refs):
            rows.extend(
                transcribe_chunk(
                    asr, [w], language=language, wake=wake, fake=fake, refs=[ref]
                )
            )
        return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="asr_probe：全量 pos/neg × 流 ASR")
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--sep-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--arms", default="no_sep,sep_once,sep_multi")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--limit", type=int, default=0, help="每 split 最多 N 条 utt（pos/neg 各 N）")
    p.add_argument("--min-rms", type=float, default=1e-4)
    p.add_argument(
        "--include-mix-in-sep",
        action="store_true",
        help="sep_once/sep_multi 也转 mix（默认跳过，与 no_sep 去重）",
    )
    p.add_argument("--language", default=None)
    p.add_argument("--guess-language", action="store_true")
    p.add_argument("--use-wake-context", action="store_true")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--fake-asr", choices=["identity", "perturb"], default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    for a in arms:
        if a not in ARM_DEPTH:
            raise SystemExit(f"未知 arm={a}；可选 {list(ARM_DEPTH)}")

    samples_path = args.samples.resolve()
    sep_root = args.sep_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "asr_results.jsonl"

    if not samples_path.is_file():
        raise SystemExit(f"找不到 samples.jsonl: {samples_path}")
    if not (sep_root / "d1").is_dir():
        raise SystemExit(f"SEP_ROOT 缺少 d1/: {sep_root}")

    samples = load_jsonl(samples_path)
    if args.limit and args.limit > 0:
        pos = [s for s in samples if s.get("split") == "pos"][: args.limit]
        neg = [s for s in samples if s.get("split") == "neg"][: args.limit]
        other = [s for s in samples if s.get("split") not in ("pos", "neg")][: args.limit]
        samples = pos + neg + other
        print(f"[INFO] LIMIT={args.limit} → pos={len(pos)} neg={len(neg)} other={len(other)}")

    jobs, plan = collect_jobs(
        samples, sep_root, arms, include_mix_in_sep=bool(args.include_mix_in_sep)
    )
    (out_dir / "jobs_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)

    done: set[str] = set()
    if args.resume and out_path.is_file():
        n_stale = 0
        for r in load_jsonl(out_path):
            if r.get("norm_ver") != NORM_VER:
                n_stale += 1
                continue
            k = rec_key(str(r.get("uid") or ""), str(r.get("arm") or ""), str(r.get("stream") or ""))
            if r.get("status") in ("ok", "empty_hyp", "low_rms", "missing_wav"):
                done.add(k)
        print(f"[INFO] resume: 复用 {len(done)} 条；跳过口径不符 {n_stale}", flush=True)

    pending = [j for j in jobs if rec_key(j["uid"], j["arm"], j["stream"]) not in done]
    print(f"[INFO] jobs={len(jobs)} pending={len(pending)} already={len(done)}", flush=True)

    asr: Optional[Qwen3ASRBackend] = None
    model_name = MODEL_ID
    if pending and not args.fake_asr:
        model_dir = resolve_asr_dir(str(args.model_dir) if args.model_dir else None)
        if not model_dir:
            if os.environ.get("ASR_ALLOW_DOWNLOAD", "0").strip().lower() in ("1", "true", "yes"):
                model_dir = MODEL_ID
                print(f"[WARN] 本地无权重，ASR_ALLOW_DOWNLOAD=1 → {MODEL_ID}")
            else:
                raise SystemExit(
                    "找不到 Qwen3-ASR-1.7B。export ASR_MODEL_DIR=... 或先 ./download_qwen3_asr.sh"
                )
        print(f"[INFO] ASR weights: {model_dir}", flush=True)
        asr = Qwen3ASRBackend(
            model_dir,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            max_batch=max(1, int(args.batch)),
        )
        model_name = Path(model_dir).name if Path(str(model_dir)).is_dir() else MODEL_ID

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    # 按 (language, wake) 分桶，便于真正 batch
    def lang_of(job: dict[str, Any]) -> Optional[str]:
        if args.language:
            return args.language
        if args.guess_language:
            return guess_language(job.get("wake_text"))
        return None

    def wake_of(job: dict[str, Any]) -> Optional[str]:
        if not args.use_wake_context:
            return None
        t = (job.get("wake_text") or "").strip()
        return t or None

    pbar = tqdm(total=len(pending), desc="asr_probe", unit="stream") if tqdm and pending else None
    t0 = time.time()
    n_written = 0
    batch_n = max(1, int(args.batch))

    def flush_records(recs: list[dict[str, Any]]) -> None:
        nonlocal n_written
        if not recs:
            return
        with out_path.open("a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_written += len(recs)

    i = 0
    while i < len(pending):
        seed = pending[i]
        lang = lang_of(seed)
        wake = wake_of(seed)
        chunk_jobs: list[dict[str, Any]] = []
        chunk_wavs: list[Any] = []
        chunk_meta: list[dict[str, Any]] = []
        while i < len(pending) and len(chunk_jobs) < batch_n:
            j = pending[i]
            if lang_of(j) != lang or wake_of(j) != wake:
                if chunk_jobs:
                    break
                lang = lang_of(j)
                wake = wake_of(j)
            wav_p = Path(j["wav"])
            if not wav_p.is_file():
                rec = build_record(
                    j,
                    hyp=None,
                    status="missing_wav",
                    error="wav 不存在",
                    asr_ms=None,
                    rms=None,
                    dur_sec=None,
                    language=lang,
                    model=model_name,
                )
                flush_records([rec])
                i += 1
                if pbar:
                    pbar.update(1)
                continue
            try:
                w, sr = load_audio(wav_p, sr=16000)
            except Exception as e:  # noqa: BLE001
                rec = build_record(
                    j,
                    hyp=None,
                    status="asr_error",
                    error=f"load_audio: {e}",
                    asr_ms=None,
                    rms=None,
                    dur_sec=None,
                    language=lang,
                    model=model_name,
                )
                flush_records([rec])
                i += 1
                if pbar:
                    pbar.update(1)
                continue
            rms = _rms(w)
            dur = round(len(w) / float(sr), 3)
            if j["stream"] != "mix" and rms < float(args.min_rms):
                rec = build_record(
                    j,
                    hyp="",
                    status="low_rms",
                    error=f"rms={rms:.2e} < {args.min_rms}",
                    asr_ms=None,
                    rms=rms,
                    dur_sec=dur,
                    language=lang,
                    model=model_name,
                )
                flush_records([rec])
                i += 1
                if pbar:
                    pbar.update(1)
                continue
            chunk_jobs.append(j)
            chunk_wavs.append(w)
            chunk_meta.append({"rms": rms, "dur": dur})
            i += 1

        if not chunk_jobs:
            continue
        refs = [str(j.get("cmd_text") or "") for j in chunk_jobs]
        t1 = time.time()
        results = transcribe_chunk(
            asr,
            chunk_wavs,
            language=lang,
            wake=wake,
            fake=args.fake_asr,
            refs=refs,
        )
        elapsed = (time.time() - t1) * 1000.0
        recs = []
        per_ms = elapsed / max(1, len(chunk_jobs))
        for j, meta, (hyp, status, err) in zip(chunk_jobs, chunk_meta, results):
            recs.append(
                build_record(
                    j,
                    hyp=hyp,
                    status=status,
                    error=err,
                    asr_ms=round(per_ms, 1),
                    rms=meta["rms"],
                    dur_sec=meta["dur"],
                    language=lang,
                    model=model_name,
                )
            )
        flush_records(recs)
        if pbar:
            pbar.update(len(chunk_jobs))

    if pbar:
        pbar.close()
    print(
        f"[OK] wrote +{n_written} → {out_path}  elapsed={time.time() - t0:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

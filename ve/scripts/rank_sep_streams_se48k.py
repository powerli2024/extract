#!/usr/bin/env python3
"""对已有 MossFormer 分离三流做 MossFormer2_SE_48K 增强并按注册声纹重排。

输入固定为 sep_streams/d1/{split}/{uid}/{mix,d1_spk1,d1_spk2}.wav。
输出每个样本恰好最多四条到 extracted/{split}/：raw 与 se48k 各取相似度前二。
文件名包含 raw/se48k 和 best/better；JSONL 不使用这两个主观标签，只保留可复核
的来源流、全部相似度、导出的路径和采样率链路。
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from audio_io import cosine_sim, load_audio, resample_wav, save_audio, vad_crop_speech
from calibrate_presence import stratified_limit
from moss_se48k import MossFormer2SE48K
from paths import default_eres2net_dir, ensure_dir, setup_sys_path
from presence_encoder import create_presence_encoder

setup_sys_path()

STREAMS = ("mix", "d1_spk1", "d1_spk2")
LABELS = ("best", "better")  # 只允许出现在音频文件名中。


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rank_sources(scores: dict[str, float]) -> list[str]:
    """确定性排序：相似度降序，再按流名，防止相等分数导致结果漂移。"""
    return sorted(scores, key=lambda k: (-float(scores[k]), k))


def output_name(uid: str, condition: str, label: str, source: str) -> str:
    if condition not in ("raw", "se48k"):
        raise ValueError(condition)
    if label not in LABELS:
        raise ValueError(label)
    if source not in STREAMS:
        raise ValueError(source)
    return f"{uid}__{condition}__{label}__{source}.wav"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MossFormer 三流 → SE-48k → 注册声纹前二导出")
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--sep-root", type=Path, required=True, help="VE_OUT/sep_streams")
    p.add_argument("--out-dir", type=Path, required=True, help="独立 VE_OUT；不可覆盖原 run_all 输出")
    p.add_argument("--clearvoice-root", type=Path, required=True, help="ClearerVoice-Studio 仓根")
    p.add_argument("--presence-backend", default="eres2netv2")
    p.add_argument("--eres-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--enroll-vad", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--enroll-vad-max-sec", type=float, default=4.0)
    p.add_argument("--strict", action="store_true", help="任一三流缺失或 SE 失败即返回非零")
    return p.parse_args()


def _stream_paths(sep_root: Path, split: str, uid: str) -> dict[str, Path]:
    base = sep_root / "d1" / split / uid
    return {name: base / f"{name}.wav" for name in STREAMS}


def main() -> int:
    args = parse_args()
    samples = load_jsonl(args.samples)
    if args.limit:
        samples = stratified_limit(samples, int(args.limit))
    sep_root = args.sep_root.resolve()
    if not sep_root.is_dir():
        raise SystemExit(f"--sep-root 不存在: {sep_root}")
    out_dir = ensure_dir(args.out_dir)
    extracted = ensure_dir(out_dir / "extracted")
    result_path = ensure_dir(out_dir / "results") / "se48k_ranked_results.jsonl"

    enc = create_presence_encoder(
        args.presence_backend, eres_dir=args.eres_dir or default_eres2net_dir(), device=args.device
    )
    se = MossFormer2SE48K(clearvoice_root=args.clearvoice_root, device=args.device)
    print(f"[INFO] n={len(samples)} sep_root={sep_root} out={out_dir}", flush=True)
    print("[INFO] resample=poly 16000->48000->16000; exported audio is 16 kHz for SV/ASR", flush=True)

    enroll_cache: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    n_ok = n_error = 0
    t0 = time.time()
    for idx, sample in enumerate(samples, 1):
        uid, split = str(sample["uid"]), str(sample.get("split", "pos"))
        rec: dict[str, Any] = {
            "uid": uid, "split": split, "label": sample.get("label"),
            "source_sep_dir": str(sep_root / "d1" / split / uid),
            "presence_backend": enc.name,
            "processing": {
                "raw_stream_sr": 16000, "se_model": "MossFormer2_SE_48K",
                "se_model_sr": 48000, "resample": "poly", "export_sr": 16000,
            },
        }
        try:
            if uid not in enroll_cache:
                ew, esr = load_audio(sample["enroll_wav"])
                if args.enroll_vad:
                    ew, _ = vad_crop_speech(ew, esr, max_sec=float(args.enroll_vad_max_sec))
                enroll_cache[uid] = enc.embed(ew, esr)
            enroll_emb = enroll_cache[uid]
            paths = _stream_paths(sep_root, split, uid)
            missing = [name for name, p in paths.items() if not p.is_file()]
            if missing:
                raise FileNotFoundError(f"missing streams={missing}")

            raw: dict[str, np.ndarray] = {}
            enhanced: dict[str, np.ndarray] = {}
            for name, path in paths.items():
                raw[name], sr = load_audio(path)
                if sr != 16000:
                    raise RuntimeError(f"{path} load sr={sr}, expected 16000")
                up = resample_wav(raw[name], 16000, 48000, method="poly")
                y48 = se.enhance_48k(up)
                enhanced[name] = resample_wav(y48, 48000, 16000, method="poly")

            raw_scores = {name: float(cosine_sim(enroll_emb, enc.embed(w, 16000))) for name, w in raw.items()}
            se_scores = {name: float(cosine_sim(enroll_emb, enc.embed(w, 16000))) for name, w in enhanced.items()}
            raw_keep, se_keep = rank_sources(raw_scores)[:2], rank_sources(se_scores)[:2]
            exports: dict[str, list[dict[str, Any]]] = {"raw": [], "se48k": []}
            for condition, waves, selected in (("raw", raw, raw_keep), ("se48k", enhanced, se_keep)):
                scores = raw_scores if condition == "raw" else se_scores
                for label, source in zip(LABELS, selected):
                    path = extracted / split / output_name(uid, condition, label, source)
                    save_audio(path, waves[source], 16000)
                    # 不写 best/better 字段；文件名本身表达保留槽位。
                    exports[condition].append({"source_stream": source, "similarity": round(scores[source], 6), "path": str(path)})
            rec.update({
                "raw_similarity": {k: round(v, 6) for k, v in raw_scores.items()},
                "se48k_similarity": {k: round(v, 6) for k, v in se_scores.items()},
                "exported": exports,
                "status": "ok",
            })
            n_ok += 1
        except Exception as e:
            n_error += 1
            rec.update({"status": "error", "error": str(e), "traceback": traceback.format_exc(limit=3)})
            print(f"[ERR] {uid}: {e}", flush=True)
        rows.append(rec)
        if idx % 25 == 0 or idx == len(samples):
            print(f"[INFO] {idx}/{len(samples)} ok={n_ok} error={n_error}", flush=True)

    with result_path.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary = {"n": len(rows), "n_ok": n_ok, "n_error": n_error, "elapsed_sec": round(time.time() - t0, 2), "results": str(result_path)}
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports" / "se48k_ranked_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 2 if args.strict and n_error else 0


if __name__ == "__main__":
    raise SystemExit(main())

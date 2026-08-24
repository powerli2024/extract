#!/usr/bin/env python3
"""Label-free quality audit for KWS enrollment audio.

This module intentionally does not inspect CMD audio or pos/neg labels.  It
measures whether an enrollment waveform is technically usable by the speaker
encoder.  The individual measurements are the contract; the 0-100 score is a
transparent convenience summary, not a learned predictor of downstream FAR.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from audio_io import cosine_sim, load_audio, vad_crop_speech


EPS = 1e-12


@dataclass(frozen=True)
class QualityPolicy:
    min_duration_sec: float = 0.30
    review_min_duration_sec: float = 0.55
    max_duration_sec: float = 4.0
    min_active_sec: float = 0.20
    review_min_active_sec: float = 0.35
    min_rms_dbfs: float = -42.0
    review_rms_dbfs: float = -34.0
    max_rms_dbfs: float = -6.0
    max_clip_ratio: float = 0.01
    review_clip_ratio: float = 0.001
    max_dc_abs: float = 0.03
    min_speech_ratio: float = 0.20
    review_speech_ratio: float = 0.35
    min_embedding_stability: float = 0.55
    review_embedding_stability: float = 0.72


def _db(x: float) -> float:
    return float(20.0 * math.log10(max(float(x), EPS)))


def _frame_rms(wav: np.ndarray, sr: int, frame_ms: float = 20.0) -> np.ndarray:
    n = max(1, int(round(sr * frame_ms / 1000.0)))
    if wav.size < n:
        padded = np.pad(wav, (0, n - wav.size))
        return np.asarray([np.sqrt(np.mean(padded.astype(np.float64) ** 2) + EPS)])
    m = wav.size // n
    frames = wav[: m * n].reshape(m, n).astype(np.float64)
    return np.sqrt(np.mean(frames**2, axis=1) + EPS)


def _speech_activity(frame_db: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Return an energy-based speech mask and its diagnostic percentiles."""
    p20 = float(np.percentile(frame_db, 20))
    p90 = float(np.percentile(frame_db, 90))
    contrast = p90 - p20
    if contrast < 4.0:
        # A nearly flat, non-silent short keyword is commonly all speech.
        threshold = p20 - 1.0
    else:
        threshold = p20 + 0.25 * contrast
    threshold = max(threshold, -52.0)
    return frame_db >= threshold, p20, p90, contrast


def _occupied_bandwidth_hz(wav: np.ndarray, sr: int, fraction: float = 0.95) -> float:
    if wav.size < 32:
        return 0.0
    x = wav.astype(np.float64) - float(np.mean(wav))
    x *= np.hanning(x.size)
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    # Exclude DC/very-low-frequency handling noise.
    power[freqs < 80.0] = 0.0
    total = float(np.sum(power))
    if total <= EPS:
        return 0.0
    idx = int(np.searchsorted(np.cumsum(power), fraction * total))
    return float(freqs[min(idx, freqs.size - 1)])


def signal_metrics(wav: np.ndarray, sr: int = 16000) -> dict[str, float]:
    x = np.asarray(wav, dtype=np.float32).reshape(-1)
    duration = x.size / float(sr)
    finite = np.isfinite(x)
    nonfinite_ratio = float(1.0 - np.mean(finite)) if x.size else 1.0
    if x.size:
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + EPS)) if x.size else 0.0
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    frame_rms = _frame_rms(x, sr)
    frame_db = np.asarray([_db(v) for v in frame_rms], dtype=np.float64)
    speech, noise_db, speech_db, contrast = _speech_activity(frame_db)
    speech_ratio = float(np.mean(speech)) if speech.size else 0.0
    active_sec = duration * speech_ratio
    return {
        "duration_sec": round(duration, 4),
        "active_sec": round(active_sec, 4),
        "speech_ratio": round(speech_ratio, 6),
        "silence_ratio": round(1.0 - speech_ratio, 6),
        "rms_dbfs": round(_db(rms), 3),
        "peak_dbfs": round(_db(peak), 3),
        "clip_ratio": round(float(np.mean(np.abs(x) >= 0.999)) if x.size else 0.0, 8),
        "dc_abs": round(abs(float(np.mean(x))) if x.size else 0.0, 8),
        "nonfinite_ratio": round(nonfinite_ratio, 8),
        "noise_floor_dbfs_proxy": round(noise_db, 3),
        "speech_level_dbfs_proxy": round(speech_db, 3),
        "energy_contrast_db": round(contrast, 3),
        "occupied_bandwidth_hz": round(_occupied_bandwidth_hz(x, sr), 1),
    }


def embedding_metrics(wav: np.ndarray, encoder: Any, sr: int = 16000) -> dict[str, Any]:
    """Measure speaker-embedding robustness without any CMD or identity label."""
    x = np.asarray(wav, dtype=np.float32).reshape(-1)
    full = encoder.embed(x, sr)
    variants: list[tuple[str, np.ndarray]] = []

    cropped, meta = vad_crop_speech(x, sr, max_sec=4.0)
    if cropped.size >= int(0.35 * sr) and cropped.size != x.size:
        variants.append(("vad_crop", cropped))

    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + EPS))
    if x.size and rms > 1e-6:
        # Deterministic 20 dB perturbation: tests embedding robustness, not SNR.
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(x.size).astype(np.float32)
        noise_rms = float(np.sqrt(np.mean(noise.astype(np.float64) ** 2) + EPS))
        noisy = np.clip(x + noise * (rms / (10.0 * noise_rms)), -1.0, 1.0)
        variants.append(("noise_20db", noisy.astype(np.float32)))

    sims: dict[str, float] = {}
    for name, variant in variants:
        sims[name] = round(cosine_sim(full, encoder.embed(variant, sr)), 6)
    stability = float(min(sims.values())) if sims else 1.0
    return {
        "embedding_stability": round(stability, 6),
        "embedding_variant_similarity": sims,
        "vad_crop": meta,
        "backend": getattr(encoder, "name", type(encoder).__name__),
    }


def assess(metrics: dict[str, Any], policy: QualityPolicy) -> tuple[str, float, list[str], list[str]]:
    reject: list[str] = []
    review: list[str] = []
    score = 100.0

    def hard(condition: bool, reason: str, penalty: float) -> None:
        nonlocal score
        if condition:
            reject.append(reason)
            score -= penalty

    def soft(condition: bool, reason: str, penalty: float) -> None:
        nonlocal score
        if condition:
            review.append(reason)
            score -= penalty

    hard(metrics["nonfinite_ratio"] > 0, "nonfinite_samples", 50)
    hard(metrics["duration_sec"] < policy.min_duration_sec, "too_short", 35)
    soft(
        policy.min_duration_sec <= metrics["duration_sec"] < policy.review_min_duration_sec,
        "short_duration",
        10,
    )
    soft(metrics["duration_sec"] > policy.max_duration_sec, "too_long", 8)
    hard(metrics["active_sec"] < policy.min_active_sec, "insufficient_active_speech", 35)
    soft(
        policy.min_active_sec <= metrics["active_sec"] < policy.review_min_active_sec,
        "limited_active_speech",
        10,
    )
    hard(metrics["rms_dbfs"] < policy.min_rms_dbfs, "level_too_low", 30)
    hard(metrics["rms_dbfs"] > policy.max_rms_dbfs, "level_too_high", 25)
    hard(metrics["clip_ratio"] > policy.max_clip_ratio, "severe_clipping", 30)
    hard(metrics["dc_abs"] > policy.max_dc_abs, "dc_offset", 20)
    hard(metrics["speech_ratio"] < policy.min_speech_ratio, "mostly_silence", 25)

    soft(metrics["rms_dbfs"] < policy.review_rms_dbfs, "level_low", 8)
    soft(metrics["clip_ratio"] > policy.review_clip_ratio, "some_clipping", 8)
    soft(metrics["speech_ratio"] < policy.review_speech_ratio, "much_silence", 8)
    # A short vowel-heavy wake phrase can legitimately have a low spectral
    # roll-off. Bandwidth stays in raw metrics but never changes the decision.

    stability = metrics.get("embedding_stability")
    if stability is not None:
        # Stability is model/domain dependent.  Never hard-reject by this
        # heuristic alone before an independent labeled-CMD validation.
        soft(stability < policy.min_embedding_stability, "unstable_speaker_embedding", 15)
        soft(
            policy.min_embedding_stability <= stability < policy.review_embedding_stability,
            "speaker_embedding_needs_review",
            10,
        )

    # A known wake phrase is valid enrollment-side information, unlike CMD labels.
    text_cer = metrics.get("wake_text_cer")
    if text_cer is not None:
        hard(float(text_cer) > 0.5, "wake_phrase_mismatch", 35)
        soft(0.2 < float(text_cer) <= 0.5, "wake_phrase_uncertain", 12)

    if reject:
        decision = "reject"
    elif review:
        decision = "review"
    else:
        decision = "pass"
    return decision, round(max(0.0, score), 1), sorted(set(reject)), sorted(set(review))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def _manifest_items(path: Path) -> Iterable[dict[str, Any]]:
    for i, row in enumerate(_read_jsonl(path)):
        audio = row.get("enroll_path") or row.get("enroll_wav") or row.get("audio") or row.get("path")
        if not audio:
            yield {"uid": row.get("uid", str(i)), "error": "missing_audio_path"}
            continue
        audio_path = Path(str(audio))
        if not audio_path.is_absolute():
            audio_path = (path.parent / audio_path).resolve()
        cer = row.get("wake_text_cer")
        if cer is None and isinstance(row.get("enroll_meta"), dict):
            cer = row["enroll_meta"].get("oracle_cer")
        yield {"uid": str(row.get("uid", i)), "audio": str(audio_path), "wake_text_cer": cer}


def _make_encoder(args: argparse.Namespace) -> Any | None:
    if args.backend == "none":
        return None
    from presence_encoder import create_presence_encoder

    return create_presence_encoder(
        args.backend,
        eres_dir=args.eres_dir,
        campplus_dir=args.campplus_dir,
        resnet_dir=args.resnet_dir,
        device=args.device,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="无 CMD/正负标签的 KWS 注册音频质量评估")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", type=Path, help="JSONL；路径字段 enroll_path/enroll_wav/audio/path")
    src.add_argument("--audio-dir", type=Path, help="递归扫描 wav/flac/ogg")
    src.add_argument("--audio", type=Path, help="评估单个音频")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--backend", default="eres2netv2", choices=("none", "eres2netv2", "campplus", "resnet34_lm"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--eres-dir", type=Path, default=os.environ.get("ERES2NET_DIR") or os.environ.get("ERES_DIR"))
    p.add_argument("--campplus-dir", type=Path, default=os.environ.get("CAMPPLUS_DIR"))
    p.add_argument("--resnet-dir", type=Path, default=os.environ.get("SPK_CHS_DIR"))
    p.add_argument("--policy", type=Path, default=None, help="覆盖 QualityPolicy 的 JSON")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--fail-on-reject", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    policy_data: dict[str, Any] = {}
    if args.policy:
        policy_data = json.loads(args.policy.read_text(encoding="utf-8"))
    policy = QualityPolicy(**policy_data)
    encoder = _make_encoder(args)

    if args.manifest:
        items = list(_manifest_items(args.manifest.resolve()))
    elif args.audio_dir:
        exts = {".wav", ".flac", ".ogg"}
        paths = sorted(p for p in args.audio_dir.resolve().rglob("*") if p.suffix.lower() in exts)
        items = [{"uid": p.stem, "audio": str(p)} for p in paths]
    else:
        p = args.audio.resolve()
        items = [{"uid": p.stem, "audio": str(p)}]
    if args.limit > 0:
        items = items[: args.limit]
    if not items:
        raise SystemExit("没有找到可评估的注册音频")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items, 1):
        uid = str(item.get("uid", index))
        row: dict[str, Any] = {"uid": uid, "audio": item.get("audio")}
        try:
            if item.get("error"):
                raise ValueError(str(item["error"]))
            wav, sr = load_audio(str(item["audio"]), 16000)
            row.update(signal_metrics(wav, sr))
            if item.get("wake_text_cer") is not None:
                row["wake_text_cer"] = float(item["wake_text_cer"])
            if encoder is not None:
                row.update(embedding_metrics(wav, encoder, sr))
            decision, score, hard_reasons, review_reasons = assess(row, policy)
            row.update({
                "decision": decision,
                "quality_score": score,
                "reject_reasons": hard_reasons,
                "review_reasons": review_reasons,
            })
        except Exception as exc:
            row.update({
                "decision": "reject",
                "quality_score": 0.0,
                "reject_reasons": ["analysis_error"],
                "review_reasons": [],
                "error": f"{type(exc).__name__}: {exc}",
            })
        rows.append(row)
        if index % 100 == 0 or index == len(items):
            print(f"[INFO] {index}/{len(items)} last={uid} decision={row['decision']}", flush=True)

    report = args.out_dir / "enroll_quality.jsonl"
    with report.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = {name: sum(r["decision"] == name for r in rows) for name in ("pass", "review", "reject")}
    valid_scores = [float(r["quality_score"]) for r in rows]
    reject_reason_counts = Counter(
        reason for row in rows for reason in row.get("reject_reasons", [])
    )
    review_reason_counts = Counter(
        reason for row in rows for reason in row.get("review_reasons", [])
    )
    metric_quantiles: dict[str, dict[str, float]] = {}
    for key in (
        "duration_sec",
        "active_sec",
        "speech_ratio",
        "rms_dbfs",
        "clip_ratio",
        "occupied_bandwidth_hz",
        "embedding_stability",
    ):
        vals = [float(row[key]) for row in rows if row.get(key) is not None]
        if vals:
            q = np.percentile(np.asarray(vals, dtype=np.float64), [5, 50, 95])
            metric_quantiles[key] = {
                "p05": round(float(q[0]), 6),
                "p50": round(float(q[1]), 6),
                "p95": round(float(q[2]), 6),
            }
    summary = {
        "contract": "label_free_enrollment_quality_v1",
        "n": len(rows),
        "counts": counts,
        "reject_reason_counts": dict(reject_reason_counts.most_common()),
        "review_reason_counts": dict(review_reason_counts.most_common()),
        "metric_quantiles": metric_quantiles,
        "quality_score_mean": round(float(np.mean(valid_scores)), 3) if valid_scores else None,
        "backend": args.backend,
        "uses_cmd_audio": False,
        "uses_pos_neg_label": False,
        "policy": asdict(policy),
        "report": str(report.resolve()),
        "note": "Raw metrics are authoritative; quality_score is a transparent policy summary, not a FAR predictor.",
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if args.fail_on_reject and counts["reject"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

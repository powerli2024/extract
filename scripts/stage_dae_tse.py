#!/usr/bin/env python3
"""Evaluate DAE-TSE as an isolated text-conditioned candidate view.

Outputs live under ``VM_OUT/experiments`` and are intentionally invisible to
the normal s1-s8 handoff. The DAE environment performs inference; the regular
extract-sep environment owns source resolution, signatures, ASR/CER and audit.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from candidate_sources import (
    audio_contract,
    canonical_sha256,
    directory_fingerprint,
    git_fingerprint,
    load_jsonl,
    resolve_candidate_sources,
    sha256_file,
    python_runtime_fingerprint,
    write_signature,
)
from paths import STAGE_DIRS, assert_split, default_vm_out
from progress_log import StageProgress
from dae_cue_loader import resolve_cue_helper


CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vm-out", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--splits", default="pos,neg")
    parser.add_argument("--source-stage", default="s1", choices=["raw", *STAGE_DIRS])
    parser.add_argument("--source-thr", default="", choices=["", "a", "b", "c"])
    parser.add_argument("--dae-python", type=Path, required=True)
    parser.add_argument("--dae-repo", type=Path, required=True)
    parser.add_argument("--dae-cue-helper", type=Path, default=None)
    parser.add_argument("--dae-config", type=Path, required=True)
    parser.add_argument("--dae-checkpoint", type=Path, required=True)
    parser.add_argument("--asr-model-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-peak", type=float, default=0.70)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-asr", action="store_true")
    return parser.parse_args()


def default_out_dir(vm_out: Path, source_stage: str, source_thr: str) -> Path:
    label = source_stage + (f"_thr_{source_thr}" if source_thr else "")
    return vm_out / "experiments" / f"dae_tse_from_{label}"


def assert_isolated_out_dir(vm_out: Path, out_dir: Path) -> None:
    if out_dir == vm_out:
        raise RuntimeError("DAE_OUT cannot be VM_OUT itself")
    for split in ("pos", "neg", "best_sep"):
        forbidden = (vm_out / split).resolve()
        if out_dir == forbidden or forbidden in out_dir.parents:
            raise RuntimeError(
                f"DAE_OUT must not be inside handoff/data tree {forbidden}; "
                "use VM_OUT/experiments/... or a separate directory"
            )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows = load_jsonl(path)
    result = {str(row.get("uid")): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"duplicate uid in {path}")
    return result


def prepare_sources(args, vm_out: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if args.source_stage in {"s5", "s6", "s7", "s8"} and not args.source_thr:
        raise RuntimeError(f"{args.source_stage} is gated; --source-thr a|b|c is required")
    splits = [assert_split(value.strip()) for value in args.splits.split(",") if value.strip()]
    all_rows: list[dict[str, Any]] = []
    for split in splits:
        rows = resolve_candidate_sources(
            vm_out,
            source_stage=args.source_stage,
            source_thr=args.source_thr,
            split=split,
        )
        if args.limit > 0:
            rows = rows[: args.limit]
        all_rows.extend(rows)
    applicable = [
        row for row in all_rows if CJK_RE.search(str(row.get("wake_text") or ""))
    ]
    return all_rows, applicable


def build_manifest(applicable: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for row in applicable:
        destination = out_dir / "wav" / str(row["split"]) / f"{row['uid']}_dae_tse.wav"
        rows.append(
            {
                "uid": row["uid"],
                "split": row["split"],
                "wake_text": row["wake_text"],
                "input_wav": row["source_wav"],
                "input_sha256": row["source_sha256"],
                "expected_frames": row["source_frames"],
                "expected_sample_rate": 16000,
                "output_wav": str(destination.resolve()),
            }
        )
    return rows


def signature_payload(
    args,
    all_rows,
    manifest,
    vm_out: Path,
    out_dir: Path,
    cue_helper: Path,
) -> dict[str, Any]:
    runtime_script = Path(__file__).with_name("dae_tse_infer_manifest.py")
    source_identity = [
        {
            "uid": row["uid"],
            "source_sha256": row["source_sha256"],
            "source_frames": row["source_frames"],
            "wake_text": row.get("wake_text") or "",
        }
        for row in all_rows
    ]
    return {
        "schema": "extract_sep_dae_tse_experiment/v1",
        "production_approved": False,
        "extract_repo": git_fingerprint(Path(__file__).resolve().parents[1]),
        "orchestrator_python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "vm_out": str(vm_out.resolve()),
        "out_dir": str(out_dir.resolve()),
        "source_stage": args.source_stage,
        "source_thr": args.source_thr,
        "splits": args.splits,
        "limit": int(args.limit),
        "n_source": len(all_rows),
        "n_applicable_zh": len(manifest),
        "source_manifest_sha256": canonical_sha256(source_identity),
        "dae_python": python_runtime_fingerprint(args.dae_python.resolve()),
        "dae_repo": git_fingerprint(args.dae_repo.resolve()),
        "dae_cue_helper": {
            "path": str(cue_helper.resolve()),
            "sha256": sha256_file(cue_helper),
        },
        "dae_config": {
            "path": str(args.dae_config.resolve()),
            "sha256": sha256_file(args.dae_config),
        },
        "dae_checkpoint": {
            "path": str(args.dae_checkpoint.resolve()),
            "sha256": sha256_file(args.dae_checkpoint),
        },
        "dae_runtime_sha256": sha256_file(runtime_script),
        "asr_model": directory_fingerprint(args.asr_model_dir.resolve()),
        "device": args.device,
        "input_peak": float(args.input_peak),
        "audio_contract": "mono_16k_full_source_exact_length",
        "metric_contract": "zh_toneless_pinyin_cer;en_not_applicable",
        "selection_contract": "dae_only_on_strict_cer_improvement;production_false",
    }


def run_dae(args, manifest_path: Path, result_path: Path) -> None:
    if not args.dae_python.is_file():
        raise FileNotFoundError(args.dae_python)
    command = [
        str(args.dae_python),
        str(Path(__file__).with_name("dae_tse_infer_manifest.py")),
        "--manifest",
        str(manifest_path),
        "--dae-repo",
        str(args.dae_repo),
        "--cue-helper",
        str(args.dae_cue_helper),
        "--config",
        str(args.dae_config),
        "--checkpoint",
        str(args.dae_checkpoint),
        "--result-jsonl",
        str(result_path),
        "--device",
        args.device,
        "--input-peak",
        str(args.input_peak),
    ]
    print("[INFO] launching isolated DAE-TSE runtime", flush=True)
    subprocess.run(command, check=True)


def validate_dae_outputs(manifest: list[dict[str, Any]], result_path: Path) -> None:
    results = {str(row["uid"]): row for row in load_jsonl(result_path)}
    failures = []
    for row in manifest:
        result = results.get(str(row["uid"]))
        if result is None or result.get("status") == "error":
            failures.append({"uid": row["uid"], "result": result})
            continue
        try:
            audio_contract(Path(row["output_wav"]), expected_frames=int(row["expected_frames"]))
        except Exception as exc:
            failures.append({"uid": row["uid"], "error": str(exc)})
    if failures:
        raise RuntimeError(
            f"DAE output audit failed for {len(failures)}/{len(manifest)}: "
            + json.dumps(failures[:10], ensure_ascii=False)
        )


def score_candidates(args, all_rows, manifest, out_dir: Path, signature: dict[str, Any]) -> dict:
    from asr_score import create_asr, score_wavs
    from utils_audio import load_audio

    manifest_by_uid = {str(row["uid"]): row for row in manifest}
    index_path = out_dir / "index.jsonl"
    existing = load_existing(index_path)
    rows_by_uid: dict[str, dict[str, Any]] = {}
    pending = []
    for source in all_rows:
        uid = str(source["uid"])
        dae_item = manifest_by_uid.get(uid)
        if dae_item is None:
            rows_by_uid[uid] = {
                "uid": uid,
                "split": source["split"],
                "wake_text": source.get("wake_text") or "",
                "source_stage": source["source_stage"],
                "source_thr": source["source_thr"],
                "source_stream": source["source_stream"],
                "source_wav": source["source_wav"],
                "source_sha256": source["source_sha256"],
                "applicable": False,
                "status": "fallback_source_unsupported_language",
                "selected_view": "source",
                "production_approved": False,
            }
            continue
        dae_path = Path(dae_item["output_wav"])
        dae_sha = sha256_file(dae_path)
        prior = existing.get(uid)
        if (
            prior
            and prior.get("signature_sha256") == signature["signature_sha256"]
            and prior.get("source_sha256") == source["source_sha256"]
            and prior.get("dae_sha256") == dae_sha
            and prior.get("status") == "scored"
        ):
            rows_by_uid[uid] = prior
        else:
            pending.append((source, dae_path, dae_sha))

    asr = create_asr(device=args.device, model_dir=str(args.asr_model_dir.resolve())) if pending else None
    progress = StageProgress(len(pending), "dae-tse-asr")
    for source, dae_path, dae_sha in pending:
        uid = str(source["uid"])
        try:
            source_wav, sample_rate = load_audio(source["source_wav"], 16000)
            dae_wav, _ = load_audio(dae_path, 16000)
            scored = score_wavs(
                asr,
                {"original": source_wav, "dae_tse": dae_wav},
                str(source.get("wake_text") or ""),
                sample_rate,
            )
            source_cer = float(scored["streams"]["original"]["cer"])
            dae_cer = float(scored["streams"]["dae_tse"]["cer"])
            selected = "dae_tse" if dae_cer < source_cer else "source"
            record = {
                "uid": uid,
                "split": source["split"],
                "wake_text": source.get("wake_text") or "",
                "source_stage": source["source_stage"],
                "source_thr": source["source_thr"],
                "source_stream": source["source_stream"],
                "source_wav": source["source_wav"],
                "source_sha256": source["source_sha256"],
                "dae_wav": str(dae_path.resolve()),
                "dae_sha256": dae_sha,
                "applicable": True,
                "status": "scored",
                "source_cer": source_cer,
                "dae_cer": dae_cer,
                "delta_dae_minus_source": round(dae_cer - source_cer, 4),
                "selected_view": selected,
                "selection_reason": (
                    "dae_strict_cer_improvement" if selected == "dae_tse" else "source_tie_or_better"
                ),
                "metric": scored["metric"],
                "streams": scored["streams"],
                "signature_sha256": signature["signature_sha256"],
                "speaker_safety_checked": False,
                "production_approved": False,
            }
            rows_by_uid[uid] = record
            progress.tick(uid=uid, ok=True, cer=min(source_cer, dae_cer))
        except Exception as exc:
            progress.tick(uid=uid, ok=False, err=str(exc))
            raise
    progress.close()
    ordered = [rows_by_uid[str(row["uid"])] for row in all_rows]
    write_jsonl(index_path, ordered)
    paired = [row for row in ordered if row.get("status") == "scored"]
    improved = [row for row in paired if row["dae_cer"] < row["source_cer"]]
    tied = [row for row in paired if row["dae_cer"] == row["source_cer"]]
    worse = [row for row in paired if row["dae_cer"] > row["source_cer"]]
    mean = lambda key: (
        sum(float(row[key]) for row in paired) / len(paired) if paired else None
    )
    selected_mean = (
        sum(min(float(row["source_cer"]), float(row["dae_cer"])) for row in paired)
        / len(paired)
        if paired
        else None
    )
    return {
        "schema": "extract_sep_dae_tse_report/v1",
        "status": "EXPERIMENT_COMPLETE_NEEDS_SPEAKER_AND_DOWNSTREAM",
        "production_approved": False,
        "signature_sha256": signature["signature_sha256"],
        "n_total": len(ordered),
        "n_applicable": len(paired),
        "n_fallback_unsupported_language": len(ordered) - len(paired),
        "n_dae_strict_improve": len(improved),
        "n_tie": len(tied),
        "n_dae_worse": len(worse),
        "mean_source_cer_applicable": mean("source_cer"),
        "mean_dae_cer_applicable": mean("dae_cer"),
        "mean_oracle_selected_cer_applicable": selected_mean,
        "speaker_safety_checked": False,
        "next_gate": "run raw-speaker cosine and frozen KWS/Presence/CMD evaluation",
        "index": str(index_path.resolve()),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 < args.input_peak <= 1.0:
        raise ValueError(f"--input-peak must be in (0,1], got {args.input_peak}")
    vm_out = (args.vm_out or default_vm_out()).resolve()
    out_dir = (args.out_dir or default_out_dir(vm_out, args.source_stage, args.source_thr)).resolve()
    assert_isolated_out_dir(vm_out, out_dir)
    all_rows, applicable = prepare_sources(args, vm_out)
    if not all_rows:
        raise RuntimeError("no source rows")
    manifest = build_manifest(applicable, out_dir)
    manifest_path = out_dir / "dae_manifest.jsonl"
    write_jsonl(manifest_path, manifest)
    args.dae_cue_helper = resolve_cue_helper(args.dae_repo.resolve(), args.dae_cue_helper)
    payload = signature_payload(args, all_rows, manifest, vm_out, out_dir, args.dae_cue_helper)
    signature = write_signature(out_dir / "signature.json", payload)
    print(
        f"[INFO] DAE experiment out={out_dir} source={args.source_stage} "
        f"thr={args.source_thr or '-'} total={len(all_rows)} zh={len(applicable)}",
        flush=True,
    )
    if args.prepare_only:
        print(f"[OK] prepare-only manifest={manifest_path}")
        return
    result_path = out_dir / "dae_runtime_results.jsonl"
    run_dae(args, manifest_path, result_path)
    validate_dae_outputs(manifest, result_path)
    if args.skip_asr:
        summary = {
            "schema": "extract_sep_dae_tse_report/v1",
            "status": "SMOKE_ONLY_NO_ASR",
            "production_approved": False,
            "signature_sha256": signature["signature_sha256"],
            "n_total": len(all_rows),
            "n_dae_ok": len(manifest),
        }
    else:
        summary = score_candidates(args, all_rows, manifest, out_dir, signature)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[OK] DAE-TSE experiment summary={summary_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the pinned Chinese DAE-TSE model once over an extract-sep manifest.

This file normally runs in the same pinned cu124 environment as extract-sep.
It owns model construction only; experiment routing, signatures, ASR and
acceptance remain in ``stage_dae_tse.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from dae_cue_loader import load_text_to_phone_labels, resolve_cue_helper


SAMPLE_RATE = 16000
PSOK_ID = 71
PEOK_ID = 72


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dae-repo", type=Path, required=True)
    parser.add_argument("--cue-helper", type=Path, default=None)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result-jsonl", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-peak", type=float, default=0.70)
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError("DAE manifest is empty")
    return rows


def load_wave(path: Path, expected_frames: int, input_peak: float) -> torch.Tensor:
    wave, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE:
        raise RuntimeError(f"DAE input must be 16 kHz: {path} sr={sample_rate}")
    wave = np.asarray(wave.mean(axis=1), dtype=np.float32)
    if wave.size != expected_frames or not wave.size or not np.isfinite(wave).all():
        raise RuntimeError(
            f"invalid DAE input: {path} frames={wave.size} expected={expected_frames}"
        )
    peak = float(np.max(np.abs(wave)))
    if peak > 1e-8:
        wave *= float(input_peak) / peak
    return torch.from_numpy(wave)


def write_wave(path: Path, wave: torch.Tensor, expected_frames: int) -> None:
    array = wave.detach().cpu().float().numpy().reshape(-1)
    if array.size != expected_frames:
        raise RuntimeError(
            f"DAE output length mismatch: frames={array.size} expected={expected_frames}"
        )
    if not np.isfinite(array).all():
        raise RuntimeError("DAE output contains non-finite samples")
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    if peak > 0.99:
        array *= 0.99 / peak
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), array, SAMPLE_RATE, subtype="PCM_16")


def load_runtime(
    dae_repo: Path,
    config_path: Path,
    checkpoint_path: Path,
    device,
    cue_helper_path: Path | None = None,
):
    if not dae_repo.is_dir():
        raise FileNotFoundError(dae_repo)
    for path in (config_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    sys.path.insert(0, str(dae_repo.resolve()))
    import yaml
    from wesep.models import get_model

    cue_helper = resolve_cue_helper(dae_repo, cue_helper_path)
    text_to_phone_labels = load_text_to_phone_labels(cue_helper)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_name = config["model"]["tse_model"]
    model_args = dict(config["model_args"]["tse_model"])
    # Match the challengecup/official inference path: DAE-TSE uses text cues
    # as its target condition, so an optional speaker-model bootstrap must not
    # trigger a second weight download or an unrelated enrollment dependency.
    if "spk_model_init" in model_args:
        model_args["spk_model_init"] = False
    recipe_root = config_path.resolve().parents[2]
    previous_cwd = Path.cwd()
    try:
        os.chdir(recipe_root)
        model = get_model(model_name)(**model_args)
    finally:
        os.chdir(previous_cwd)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("models", [None])[0]
    if state is None:
        state = checkpoint.get("model")
    if state is None:
        raise RuntimeError("DAE checkpoint has neither models[0] nor model state")
    loaded = model.load_state_dict(state, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(f"non-strict DAE state load: {loaded}")
    model = model.to(device).eval()
    if hasattr(model, "asr_encoder") and hasattr(model.asr_encoder, "tcasr_encoder"):
        model.asr_encoder.tcasr_encoder.eval()
    return model, text_to_phone_labels, checkpoint.get("epoch")


def text_to_cue(text: str, mapper) -> list[int]:
    labels, unknown = mapper(text)
    if unknown:
        raise RuntimeError(f"unsupported DAE cue {text!r}: unknown phones={unknown}")
    phones = [int(phone) for character in labels for phone in character]
    if not phones:
        raise RuntimeError(f"empty DAE cue for {text!r}")
    return [PSOK_ID, *phones, PEOK_ID]


def make_additional(cue: list[int], device) -> dict:
    labels = torch.tensor([cue], dtype=torch.long, device=device)
    lengths = torch.tensor([len(cue)], dtype=torch.long, device=device)
    return {"data_type": "TC-ASR", "data": {"kw_label": labels, "kw_len": lengths}}


def output_is_resumable(path: Path, frames: int) -> bool:
    try:
        info = sf.info(str(path))
        return info.samplerate == SAMPLE_RATE and info.frames == frames and info.channels >= 1
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    if not 0.0 < args.input_peak <= 1.0:
        raise ValueError(f"--input-peak must be in (0,1], got {args.input_peak}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for DAE-TSE but torch.cuda.is_available() is false")
    rows = load_manifest(args.manifest)
    model, mapper, checkpoint_epoch = load_runtime(
        args.dae_repo.resolve(),
        args.config.resolve(),
        args.checkpoint.resolve(),
        device,
        args.cue_helper.resolve() if args.cue_helper else None,
    )
    results = []
    failures = 0
    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            uid = str(row["uid"])
            source = Path(row["input_wav"])
            destination = Path(row["output_wav"])
            frames = int(row["expected_frames"])
            record = {
                "uid": uid,
                "input_wav": str(source),
                "output_wav": str(destination),
                "checkpoint_epoch": checkpoint_epoch,
            }
            try:
                if output_is_resumable(destination, frames):
                    record["status"] = "resumed"
                else:
                    wave = load_wave(source, frames, args.input_peak).to(device)
                    cue = text_to_cue(str(row["wake_text"]), mapper)
                    enroll = torch.zeros((1, 1), dtype=torch.float32, device=device)
                    output = model(
                        wave.unsqueeze(0), enroll, make_additional(cue, device)
                    )
                    if isinstance(output, (list, tuple)):
                        output = output[0]
                    enhanced = output[0]
                    write_wave(destination, enhanced, frames)
                    record["status"] = "processed"
                    record["cue_length"] = len(cue)
                    record["input_peak"] = args.input_peak
            except Exception as exc:
                failures += 1
                record["status"] = "error"
                record["error"] = f"{type(exc).__name__}: {exc}"
            results.append(record)
            print(
                f"\r[DAE-TSE] {index}/{len(rows)} failures={failures} uid={uid}",
                end="",
                flush=True,
            )
    print(flush=True)
    args.result_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.result_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results),
        encoding="utf-8",
    )
    if failures:
        raise SystemExit(f"DAE-TSE failed for {failures}/{len(rows)} rows")


if __name__ == "__main__":
    main()

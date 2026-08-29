#!/usr/bin/env python3
"""Resolve frozen KWS candidate audio and guard resumable experiment signatures."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import soundfile as sf

from paths import assert_split, stage_dir, wav_path


STREAM_TO_TAG = {"original": "peak"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"broken JSONL {path}:{number}") from exc
    return rows


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def directory_fingerprint(path: Path) -> dict[str, Any]:
    """Cheap identity for large model directories; file checkpoints use full SHA256."""
    if not path.is_dir():
        return {"path": str(path.resolve()), "exists": False}
    files = []
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        stat = item.stat()
        files.append(
            {
                "rel": item.relative_to(path).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return {
        "path": str(path.resolve()),
        "exists": True,
        "n_files": len(files),
        "listing_sha256": canonical_sha256(files),
    }


def git_fingerprint(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    try:
        commit = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        diff = subprocess.run(
            ["git", "-C", str(path), "diff", "--binary", "--no-ext-diff"],
            check=True,
            capture_output=True,
        ).stdout
        staged_diff = subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "diff",
                "--cached",
                "--binary",
                "--no-ext-diff",
            ],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot fingerprint git repository {path}") from exc
    return {
        "path": str(path.resolve()),
        "commit": commit,
        "dirty_tracked": bool(status.strip()),
        "tracked_diff_sha256": hashlib.sha256(diff + staged_diff).hexdigest(),
    }


def python_runtime_fingerprint(python: Path) -> dict[str, Any]:
    if not python.is_file():
        raise FileNotFoundError(python)
    probe = (
        "import json,sys,torch,soundfile,yaml,wesep;"
        "print(json.dumps({'python':sys.version,'executable':sys.executable,"
        "'torch':torch.__version__,'cuda':torch.version.cuda,"
        "'wesep_file':getattr(wesep,'__file__',None),"
        "'soundfile':getattr(soundfile,'__version__',None),"
        "'yaml':getattr(yaml,'__version__',None)}))"
    )
    try:
        output = subprocess.run(
            [str(python), "-c", probe],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        payload = json.loads(next(line for line in reversed(output) if line.strip()))
    except (OSError, subprocess.CalledProcessError, StopIteration, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"DAE Python preflight failed: {python}; require torch/soundfile/yaml/wesep"
        ) from exc
    return payload


def audio_contract(path: Path, *, expected_frames: int | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    info = sf.info(str(path))
    if info.samplerate != 16000:
        raise RuntimeError(f"expected 16 kHz WAV: {path} sr={info.samplerate}")
    if info.channels < 1 or info.frames <= 0:
        raise RuntimeError(f"invalid WAV: {path}")
    if expected_frames is not None and info.frames != expected_frames:
        raise RuntimeError(
            f"length mismatch: {path} frames={info.frames} expected={expected_frames}"
        )
    return {
        "frames": int(info.frames),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "duration_sec": float(info.frames / info.samplerate),
    }


def _stage_index_root(
    vm_out: Path, source_stage: str, source_thr: str, split: str
) -> Path:
    root = stage_dir(vm_out, source_stage, split)
    if source_thr:
        root = root / f"thr_{source_thr}"
    return root


def resolve_candidate_sources(
    vm_out: Path,
    *,
    source_stage: str,
    source_thr: str,
    split: str,
) -> list[dict[str, Any]]:
    """Resolve one frozen source WAV per collected UID.

    ``source_stage=raw`` reads the collect manifest. Other values read that
    stage's oracle stream. Gated stages require ``source_thr``.
    """
    split = assert_split(split)
    items_path = vm_out / split / "meta" / "items.jsonl"
    items = load_jsonl(items_path)
    if source_stage == "raw":
        rows = []
        for item in items:
            source = Path(str(item["kws_path"])).resolve()
            contract = audio_contract(source)
            rows.append(
                {
                    **item,
                    "source_stage": "raw",
                    "source_thr": "",
                    "source_stream": "original",
                    "source_wav": str(source),
                    "source_frames": contract["frames"],
                    "source_sha256": sha256_file(source),
                }
            )
        return rows

    root = _stage_index_root(vm_out, source_stage, source_thr, split)
    index_rows = load_jsonl(root / "index.jsonl")
    by_uid = {str(row.get("uid")): row for row in index_rows}
    if len(by_uid) != len(index_rows):
        raise RuntimeError(f"duplicate uid in {root / 'index.jsonl'}")
    item_by_uid = {str(item["uid"]): item for item in items}
    if len(item_by_uid) != len(items):
        raise RuntimeError(f"duplicate uid in {items_path}")
    if source_thr:
        # Gated stages intentionally contain only the triggered parent cohort.
        selected_items = []
        for row in index_rows:
            uid = str(row.get("uid") or "")
            item = item_by_uid.get(uid)
            if item is None:
                raise RuntimeError(f"stage uid {uid} is absent from {items_path}")
            selected_items.append(item)
    else:
        selected_items = items
    resolved = []
    for item in selected_items:
        uid = str(item["uid"])
        row = by_uid.get(uid)
        if row is None:
            raise RuntimeError(f"missing {uid} in {root / 'index.jsonl'}")
        if row.get("error") or row.get("oracle_cer") is None:
            raise RuntimeError(f"source stage is not successful for {uid}: {row}")
        stream = str(row.get("oracle_stream") or "")
        if not stream:
            raise RuntimeError(f"missing oracle_stream for {uid}")
        source = wav_path(root, uid, STREAM_TO_TAG.get(stream, stream)).resolve()
        contract = audio_contract(source)
        resolved.append(
            {
                **item,
                "source_stage": source_stage,
                "source_thr": source_thr,
                "source_stream": stream,
                "source_stage_cer": float(row["oracle_cer"]),
                "source_wav": str(source),
                "source_frames": contract["frames"],
                "source_sha256": sha256_file(source),
            }
        )
    return resolved


def write_signature(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    complete = {**payload, "signature_sha256": canonical_sha256(payload)}
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("signature_sha256") != complete["signature_sha256"]:
            raise RuntimeError(
                "experiment signature mismatch; preserve this evidence and use a new DAE_OUT\n"
                f"existing={existing.get('signature_sha256')}\n"
                f"requested={complete['signature_sha256']}"
            )
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(complete, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return complete

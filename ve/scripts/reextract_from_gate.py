#!/usr/bin/env python3
"""复用已有 Presence 决策，只对 accept 重跑 TSE（T5 Cond-TasNet 对照）。

不重扫 τ、不改 RR/FRR。新 VE_OUT，禁止覆盖源 mix 目录。
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

from audio_io import load_audio, save_audio
from paths import default_ve_out, ensure_dir
from tse_factory import create_tse


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description="复用门控，重跑 TSE")
    p.add_argument("--src-ve-out", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--tse-backend", default="cond_tasnet")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cond-tasnet-ckpt", type=Path, default=None)
    p.add_argument("--ecapa-dir", type=Path, default=None)
    p.add_argument("--tasnet-chunk-sec", type=float, default=4.0)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    src = args.src_ve_out.resolve()
    out = (args.out_dir or default_ve_out()).resolve()
    if out == src:
        raise SystemExit(f"[ERR] 输出目录不能等于源: {src}")
    if "ve_mix_novad" in str(out).replace("\\", "/"):
        raise SystemExit("[ERR] 禁止写入 ve_mix_novad")
    src_res = src / "results" / "all_results.jsonl"
    samples_p = src / "manifest" / "samples.jsonl"
    if not src_res.is_file() or not samples_p.is_file():
        raise SystemExit(f"[ERR] 源不齐: {src}")

    ensure_dir(out / "manifest")
    ensure_dir(out / "results")
    ensure_dir(out / "extracted" / "pos")
    ensure_dir(out / "extracted" / "neg")
    shutil.copy2(samples_p, out / "manifest" / "samples.jsonl")
    qc = src / "manifest" / "qc.json"
    if qc.is_file():
        shutil.copy2(qc, out / "manifest" / "qc.json")
    calib_src = src / "reports" / "presence_calib" / "recommended_thr.json"
    if calib_src.is_file():
        ensure_dir(out / "reports" / "presence_calib")
        shutil.copy2(calib_src, out / "reports" / "presence_calib" / "recommended_thr.json")

    extractor = create_tse(
        args.tse_backend,
        device=args.device,
        cond_tasnet_ckpt=args.cond_tasnet_ckpt,
        ecapa_dir=args.ecapa_dir,
        tasnet_chunk_sec=float(args.tasnet_chunk_sec),
    )
    rows = load_jsonl(src_res)
    if args.limit and args.limit > 0:
        acc = [r for r in rows if r.get("decision") == "accept"]
        keep = {r["uid"] for r in acc[: args.limit]}
        keep |= {r["uid"] for r in rows if r.get("decision") != "accept"}
        rows = [r for r in rows if r.get("uid") in keep]

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore

    n_ok = n_fail = 0
    out_rows: list[dict[str, Any]] = []
    it = tqdm(rows, desc=f"tse:{extractor.name}", unit="utt") if tqdm else rows
    for rec0 in it:
        rec = dict(rec0)
        rec["pipeline"] = args.tse_backend
        rec["tse_backend"] = extractor.name
        rec["gate_src"] = str(src)
        dec = rec.get("decision")
        split = rec.get("split") or "pos"
        uid = rec.get("uid")
        if dec != "accept":
            rec["extracted_wav"] = None
            out_rows.append(rec)
            continue
        cmd = rec.get("cmd_wav")
        enroll = rec.get("enroll_wav")
        if not cmd or not Path(str(cmd)).is_file() or not enroll or not Path(str(enroll)).is_file():
            rec["decision"] = "extract_error"
            rec["extract_error"] = "missing cmd/enroll wav"
            n_fail += 1
            out_rows.append(rec)
            continue
        t1 = time.time()
        try:
            mix, sr = load_audio(cmd)
            enr, _ = load_audio(enroll)
            wav, meta = extractor.extract(mix, enr, sr=sr)
            dest = out / "extracted" / split / f"{uid}.wav"
            save_audio(dest, wav, sr)
            rec["extracted_wav"] = str(dest.resolve())
            rec["tse_meta"] = meta
            rec["tse_ms"] = round((time.time() - t1) * 1000, 1)
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            rec["decision"] = "extract_error"
            rec["extract_error"] = str(e)
            n_fail += 1
        out_rows.append(rec)

    write_jsonl(out / "results" / "all_results.jsonl", out_rows)
    write_jsonl(out / "results" / "pos_results.jsonl", [r for r in out_rows if r.get("split") == "pos"])
    write_jsonl(out / "results" / "neg_results.jsonl", [r for r in out_rows if r.get("split") == "neg"])
    print(f"[OK] accept extracted={n_ok} fail={n_fail} → {out}")
    return 0 if n_ok or not any(r.get("decision") == "accept" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

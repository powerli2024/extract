#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载 Qwen3-ASR-1.7B（先 HuggingFace，失败自动回退 ModelScope）。

默认下载到 AutoDL 数据盘（与 download_qwen3_asr.sh / run_all.sh / README 一致）：
    ${ASR_MODEL_DIR:-${QWEN3_ASR_DIR:-/root/autodl-tmp/Qwen3-ASR-1.7B}}

用法:
    python scripts/download_qwen3_asr.py                        # 默认 1.7B → /root/autodl-tmp/Qwen3-ASR-1.7B
    python scripts/download_qwen3_asr.py --source modelscope    # 强制走 ModelScope
    python scripts/download_qwen3_asr.py --model-id Qwen/Qwen3-ASR-2B
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

DEFAULT_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
# 关键文件校验（权重可能是单个大 safetensors 或多个分片）
REQUIRED_FILES = ("config.json", "tokenizer_config.json", "generation_config.json")


def default_out_dir() -> Path:
    for key in ("ASR_MODEL_DIR", "QWEN3_ASR_DIR", "VE_ASR_MODEL_DIR"):
        env = os.environ.get(key, "").strip()
        if env:
            return Path(env).expanduser()
    return Path("/root/autodl-tmp/Qwen3-ASR-1.7B")


def verify(out_dir: Path) -> tuple[list[str], bool]:
    missing = [f for f in REQUIRED_FILES if not (out_dir / f).is_file()]
    has_weights = bool(
        list(out_dir.glob("*.safetensors"))
        or list(out_dir.glob("*.bin"))
        or (out_dir / "model").is_dir()
    )
    return missing, has_weights


def report_size(out_dir: Path) -> None:
    if not out_dir.is_dir():
        return
    files = [p for p in out_dir.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    print(f"[OK] 目录大小: {total / 1024 ** 3:.2f} GB, 文件数: {len(files)}")
    for p in sorted(files, key=lambda q: q.stat().st_size, reverse=True)[:8]:
        print(f"     {p.relative_to(out_dir)}  {p.stat().st_size / 1024 ** 2:.1f} MB")


def dl_hf(model_id: str, out_dir: Path) -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=model_id, local_dir=str(out_dir), resume_download=True, max_workers=4
    )


def dl_modelscope(model_id: str, out_dir: Path) -> None:
    from modelscope import snapshot_download

    snapshot_download(model_id, local_dir=str(out_dir))


def main() -> int:
    ap = argparse.ArgumentParser(description="下载 Qwen3-ASR-1.7B 到 autodl-tmp 指定位置")
    ap.add_argument("--model-id", default=os.environ.get("ASR_MODEL_ID", DEFAULT_MODEL_ID))
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--source", choices=["auto", "hf", "modelscope"], default="auto",
                    help="auto: 先 HF 再 ModelScope；hf/modelscope: 只走指定源")
    args = ap.parse_args()

    out_dir = (args.out_dir or default_out_dir()).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] model_id = {args.model_id}")
    print(f"[INFO] 目标目录 = {out_dir}")
    print(f"[INFO] HF_ENDPOINT = {os.environ.get('HF_ENDPOINT', '(未设置, 直连 HF)')}")
    print(f"[INFO] HF_HOME = {os.environ.get('HF_HOME', '(默认)')}")
    print(f"[INFO] MODELSCOPE_CACHE = {os.environ.get('MODELSCOPE_CACHE', '(默认)')}")

    missing, has_weights = verify(out_dir)
    if not missing and has_weights:
        print("[OK] 模型已存在且完整，跳过下载。")
        report_size(out_dir)
        return 0

    if args.source in ("auto", "hf"):
        print("\n[1/2] 尝试 HuggingFace ...")
        try:
            t0 = time.time()
            dl_hf(args.model_id, out_dir)
            missing, has_weights = verify(out_dir)
            if not missing and has_weights:
                print(f"[OK] HuggingFace 下载完成, 用时 {time.time() - t0:.1f}s")
                report_size(out_dir)
                return 0
            print(f"[WARN] HF 下载不完整: missing={missing} weights={has_weights}")
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] HuggingFace 下载失败: {type(e).__name__}: {e}")

    if args.source in ("auto", "modelscope"):
        print("\n[2/2] 尝试 ModelScope ...")
        try:
            t0 = time.time()
            dl_modelscope(args.model_id, out_dir)
            missing, has_weights = verify(out_dir)
            if not missing and has_weights:
                print(f"[OK] ModelScope 下载完成, 用时 {time.time() - t0:.1f}s")
                report_size(out_dir)
                return 0
            print(f"[WARN] ModelScope 下载不完整: missing={missing} weights={has_weights}")
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] ModelScope 下载失败: {type(e).__name__}: {e}")

    print("\n[ERR] 两个源都失败。手动方案:")
    print(f"  1) huggingface-cli download {args.model_id} --local-dir {out_dir}")
    print(f"  2) modelscope download --model {args.model_id} --local_dir {out_dir}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

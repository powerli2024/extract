#!/usr/bin/env python3
"""ONNX MossFormer2 薄封装（仅本地权重，禁止运行期下载）。"""

from __future__ import annotations

from local_models import assert_runtime_models
from paths import setup_sys_path

setup_sys_path()


def create_onnx_separator(peak: float = 0.7, device: str = "cuda:0"):
    locs = assert_runtime_models(need_onnx=True, need_cv=False, need_asr=False)
    print(f"[INFO] local ONNX → {locs['onnx']} (offline)")
    # 先补 CUDA 库，再加载 ORT
    from cuda_libs import ensure_cuda_libs

    ensure_cuda_libs(verbose=True)
    from mossformer2_onnx import MossFormer2Separator

    sep = MossFormer2Separator(peak=peak, device=device)
    print("[INFO]", sep.status_message())
    actual = ""
    try:
        actual = sep._sessions[0].get_providers()[0]  # noqa: SLF001
    except Exception:
        pass
    if device.startswith("cuda") and actual and "CUDA" not in actual:
        print(
            f"[WARN] ONNX 实际在 {actual} 上跑（非 CUDA）。"
            "请检查 libcublasLt.so.12 / onnxruntime-gpu。",
            flush=True,
        )
    return sep

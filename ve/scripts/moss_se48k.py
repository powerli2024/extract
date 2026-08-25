#!/usr/bin/env python3
"""ClearVoice MossFormer2_SE_48K 的小型适配层。

三流缓存属于 16 kHz；这个模型必须接收 48 kHz。因此调用方必须先上采样，
并在送入声纹 / ASR 前再回采样到 16 kHz。这里不做任何隐式重采样，避免实验
口径被悄悄改变。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


class MossFormer2SE48K:
    """使用官方 ClearVoice 推理接口的 48 kHz 语音增强器。"""

    sample_rate = 48000

    def __init__(self, *, clearvoice_root: str | Path, device: str = "cuda:0"):
        root = Path(clearvoice_root).expanduser().resolve()
        package_root = root / "clearvoice"
        checkpoint = package_root / "checkpoints" / "MossFormer2_SE_48K" / "last_best_checkpoint"
        if not package_root.is_dir():
            raise FileNotFoundError(
                f"找不到 ClearerVoice-Studio: {package_root}；先运行 ./download_moss_se48k.sh"
            )
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"找不到 MossFormer2_SE_48K 权重: {checkpoint}；先运行 ./download_moss_se48k.sh"
            )

        # 官方 config 的 checkpoint_dir 是相对 clearvoice/ 的路径。
        os.chdir(package_root)
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
        if device.startswith("cuda:"):
            import torch

            torch.cuda.set_device(int(device.split(":", 1)[1]))
        from clearvoice import ClearVoice

        self._model = ClearVoice(
            task="speech_enhancement", model_names=["MossFormer2_SE_48K"]
        )
        self.root = root
        self.device = device

    def enhance_48k(self, wav: np.ndarray) -> np.ndarray:
        """输入和输出均为单声道 float32 48 kHz；长度须由调用方校验。"""
        return self.enhance_many_48k([wav])[0]

    def enhance_many_48k(self, wavs: list[np.ndarray]) -> list[np.ndarray]:
        """等长 48 kHz 波形批推理；CMD 三流可合成一个 batch，减少 GPU 调度开销。"""
        if not wavs:
            return []
        xs = [np.asarray(w, dtype=np.float32).reshape(-1) for w in wavs]
        lengths = [len(w) for w in xs]
        if not all(lengths):
            raise ValueError("empty waveform")
        if len(set(lengths)) != 1:
            raise ValueError(f"enhance_many_48k requires equal lengths, got={lengths}")
        out = self._model(np.stack(xs, axis=0))
        # ClearVoice 的 tensor-to-tensor 接口在不同版本可能返回 Tensor / ndarray / list。
        if hasattr(out, "detach"):
            out = out.detach().cpu().numpy()
        if isinstance(out, (list, tuple)):
            if len(out) != 1:
                raise RuntimeError(f"unexpected ClearVoice output list length={len(out)}")
            out = out[0]
        y = np.asarray(out, dtype=np.float32)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        if y.ndim != 2 or y.shape[0] != len(xs) or y.shape[1] < 1:
            raise RuntimeError("ClearVoice returned empty waveform")
        return [np.asarray(y[i, : lengths[i]], dtype=np.float32) for i in range(len(xs))]

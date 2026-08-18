#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WeSep 预训练 TSE 适配器（bsrnn_ecapa_vox1，官方高层 API 封装）。

官方唯一预训练模型：bsrnn_ecapa_vox1（BSRNN + ECAPA，VoxCeleb1），
由 wesep.cli.extractor.load_model("english") 从 ModelScope 自动下载到 ~/.wesep/english/。

接口与 PS4Extractor 完全一致：
    ex = WesepExtractor(device="cuda:0")              # 自动下载模型
    ex = WesepExtractor(model_dir="/path/to/dir")      # 用本地已下载目录
    out, meta = ex.extract(mixture, enroll, sr=16000)

环境（download_wesep.sh 一键准备）：
    wesep 仓库  VE_MODEL_DIR/wesep（pip install -e）
    依赖        silero-vad pyyaml soundfile numpy tqdm torch torchaudio
    模型        首次 load_model("english") 自动下载（ModelScope）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from paths import default_model_dir, default_wesep_root, setup_sys_path

setup_sys_path()


def _ensure_wesep_importable() -> None:
    for c in (
        Path(os.environ.get("WESEP_ROOT", "")) if os.environ.get("WESEP_ROOT") else None,
        default_wesep_root(),
        default_model_dir() / "wesep",
        Path("/root/autodl-tmp/wesep"),
        Path("/root/wesep"),
    ):
        if c and (c / "wesep").is_dir() and str(c) not in sys.path:
            sys.path.insert(0, str(c))
            break


class WesepExtractor:
    name = "wesep_bsrnn_ecapa"

    def __init__(
        self,
        *,
        model_dir: str | Path | None = None,
        language: str = "english",
        device: str = "cuda:0",
        resample_rate: int = 16000,
        apply_vad: bool = False,
        output_norm: bool = True,
        match_rms: bool = False,
    ):
        _ensure_wesep_importable()
        from wesep.cli.extractor import Extractor, load_model, load_model_local

        self.device = device
        self.match_rms = match_rms
        if model_dir and Path(model_dir).is_dir():
            self._ext = load_model_local(str(Path(model_dir).resolve()))
            print(f"[INFO] WeSep TSE <- 本地目录 {model_dir}", flush=True)
        else:
            self._ext = load_model(language)
            print(f"[INFO] WeSep TSE <- 自动下载 bsrnn_ecapa_vox1（language={language}）", flush=True)
        self._ext.set_device(device)
        self._ext.set_resample_rate(int(resample_rate))
        self._ext.set_vad(bool(apply_vad))
        self._ext.set_output_norm(bool(output_norm))
        self._ext.set_wavform_norm(True)
        self.resample_rate = int(resample_rate)
        print("[INFO] WeSep BSRNN+ECAPA 就绪（bsrnn_ecapa_vox1）", flush=True)

    def extract(
        self,
        mixture: np.ndarray,
        enroll: np.ndarray,
        *,
        sr: int = 16000,
        max_sec: float = 0.0,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        import torch

        mix = np.asarray(mixture, dtype=np.float32).reshape(1, -1)
        enr = np.asarray(enroll, dtype=np.float32).reshape(1, -1)
        mix_t = torch.from_numpy(mix)
        enr_t = torch.from_numpy(enr)

        out = self._ext.extract_speech_from_pcm(mix_t, int(sr), enr_t, int(sr))
        if out is None:
            raise RuntimeError(
                "WeSep 提取返回 None（可能 joint_training=False，或 VAD 判定 enroll 全静音）"
            )
        out_np = out.squeeze(0).detach().cpu().numpy().astype(np.float32)

        meta: dict[str, Any] = {
            "tse_backend": self.name,
            "model": "bsrnn_ecapa_vox1",
            "resample_rate": self.resample_rate,
            "apply_vad": self._ext.apply_vad,
            "output_norm": self._ext.output_norm,
            "rms_scale": 1.0,
        }
        if self.match_rms:
            out_np, scale = self._match_rms_to_ref(out_np, mix.reshape(-1))
            meta["rms_scale"] = float(scale)
        return out_np, meta

    @staticmethod
    def _match_rms_to_ref(out_wav: np.ndarray, ref_wav: np.ndarray, max_scale: float = 8.0):
        y = np.asarray(out_wav, dtype=np.float32).reshape(-1)
        ref = np.asarray(ref_wav, dtype=np.float32).reshape(-1)
        r_out = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
        r_ref = float(np.sqrt(np.mean(ref ** 2)) + 1e-12)
        if r_out < 1e-8:
            return y, 0.0
        scale = float(np.clip(r_ref / r_out, 1.0 / max_scale, max_scale))
        y2 = y * scale
        peak = float(np.max(np.abs(y2)) + 1e-12)
        if peak > 0.99:
            y2 = y2 * (0.99 / peak)
        return y2.astype(np.float32), scale


def create_tse(backend: str = "wesep_bsrnn", **kwargs: Any) -> WesepExtractor:
    return WesepExtractor(**kwargs)


if __name__ == "__main__":
    print("tse_wesep: 请通过 run_extract.py --tse-backend wesep_bsrnn 调用")

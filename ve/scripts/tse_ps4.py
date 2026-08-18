#!/usr/bin/env python3
"""PS4 BSRNN TSE：优先用 HF 官方 inference.py（无 wesep / 无训练依赖）。

参考:
  https://huggingface.co/TaurenMountain/PS4
权重目录默认: $VE_MODEL_DIR/PS4/{checkpoint_epoch037.pt,inference.py}
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

from audio_io import empty_cuda_cache, is_oom, peak_normalize, truncate_wav
from paths import default_model_dir, default_ps4_weights, setup_sys_path

setup_sys_path()


def _match_rms_to_ref(out_wav: np.ndarray, ref_wav: np.ndarray, max_scale: float = 8.0):
    y = np.asarray(out_wav, dtype=np.float32).reshape(-1)
    ref = np.asarray(ref_wav, dtype=np.float32).reshape(-1)
    r_out = float(np.sqrt(np.mean(y**2)) + 1e-12)
    r_ref = float(np.sqrt(np.mean(ref**2)) + 1e-12)
    if r_out < 1e-8:
        return y, 0.0
    scale = float(np.clip(r_ref / r_out, 1.0 / max_scale, max_scale))
    y2 = y * scale
    peak = float(np.max(np.abs(y2)) + 1e-12)
    if peak > 0.99:
        y2 = y2 * (0.99 / peak)
    return y2.astype(np.float32), scale


def _ps4_dir_from_weights(weights: Path) -> Path:
    return weights.parent if weights.suffix == ".pt" else weights


def _load_hf_inference(ps4_dir: Path):
    """加载同目录 inference.py（HF 自包含 BSRNN，无需 wesep）。"""
    inf = ps4_dir / "inference.py"
    if not inf.is_file():
        return None
    spec = importlib.util.spec_from_file_location("ps4_hf_inference", inf)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    # 避免与其它 inference 冲突
    sys.modules["ps4_hf_inference"] = mod
    spec.loader.exec_module(mod)
    return mod


class PS4Extractor:
    """mixture + enrollment → target wav。"""

    name = "ps4_bsrnn"

    def __init__(
        self,
        weights: str | Path | None = None,
        *,
        device: str = "cuda:0",
        wesep_dir: str | Path | None = None,  # 兼容旧参数；HF 路径不用
        config_path: str = "",
        peak_norm_cmd: bool = True,
        peak: float = 0.95,
        match_rms: bool = True,
    ):
        del wesep_dir, config_path  # 推理-only / HF 路径不需要
        self.device = device
        self.weights = Path(weights) if weights else default_ps4_weights()
        self.peak_norm_cmd = peak_norm_cmd
        self.peak = peak
        self.match_rms = match_rms
        self.model = None
        self._backend = ""
        self._hf = None
        self._load()

    def _load(self) -> None:
        import torch

        if not self.weights.is_file():
            # 允许只给目录
            cand = _ps4_dir_from_weights(self.weights) / "checkpoint_epoch037.pt"
            if cand.is_file():
                self.weights = cand
            else:
                raise FileNotFoundError(
                    f"PS4 权重不存在: {self.weights}\n"
                    "请运行: ./download_models.sh  （下载到 $VE_MODEL_DIR/PS4/）"
                )

        ps4_dir = _ps4_dir_from_weights(self.weights)
        self._hf = _load_hf_inference(ps4_dir)
        if self._hf is not None:
            device = torch.device(self.device)
            model = self._hf.build_model(device)
            self._hf.load_checkpoint(str(self.weights), model, device)
            self.model = model
            self._backend = "hf_inference.py"
            print(f"[INFO] PS4 TSE ready ← {self.weights} ({self._backend})", flush=True)
            return

        # 回退：VD wesep 路径（仅当未下载 inference.py）
        print("[WARN] 未找到 PS4/inference.py，尝试 wesep 回退…", flush=True)
        try:
            from run_ps4_eval import load_ps4_model

            self.model = load_ps4_model(str(self.weights), device=self.device, config_path="")
            if self.model is None:
                raise RuntimeError("wesep load_ps4_model 返回 None")
            self._backend = "wesep"
            print(f"[INFO] PS4 TSE ready ← {self.weights} ({self._backend})", flush=True)
        except Exception as e:
            raise RuntimeError(
                "PS4 加载失败。请确保已下载 HF 包到 "
                f"{default_model_dir() / 'PS4'}（含 checkpoint_epoch037.pt + inference.py）\n"
                f"原因: {e}"
            ) from e

    def extract(
        self,
        mixture: np.ndarray,
        enroll: np.ndarray,
        *,
        sr: int = 16000,
        max_sec: float = 0.0,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        import torch

        mix = np.asarray(mixture, dtype=np.float32).reshape(-1)
        enr = np.asarray(enroll, dtype=np.float32).reshape(-1)
        if self.peak_norm_cmd:
            mix = peak_normalize(mix, peak=self.peak)
        if max_sec and max_sec > 0:
            mix = truncate_wav(mix, sr, max_sec)
            enr = truncate_wav(enr, sr, min(max_sec, 4.0) if max_sec > 4 else max_sec)

        meta: dict[str, Any] = {
            "tse_backend": self.name,
            "ps4_loader": self._backend,
            "max_sec": max_sec,
            "oom_retried": False,
        }

        def _run(mix_np: np.ndarray, enr_np: np.ndarray) -> np.ndarray:
            if self._hf is not None:
                mix_t = torch.from_numpy(mix_np).unsqueeze(0)
                enr_t = torch.from_numpy(enr_np).unsqueeze(0)
                out_t = self._hf.extract_speaker(
                    self.model, mix_t, enr_t, torch.device(self.device)
                )
                return out_t.squeeze(0).cpu().numpy().astype(np.float32)
            from run_ps4_eval import ps4_extract

            return ps4_extract(self.model, mix_np, enr_np, sr=sr, device=self.device)

        try:
            out = _run(mix, enr)
        except Exception as e:
            if not is_oom(e):
                raise
            meta["oom_retried"] = True
            empty_cuda_cache()
            for sec in (6.0, 4.0, 3.0, 2.0):
                try:
                    mix2 = truncate_wav(mix, sr, sec)
                    out = _run(mix2, enr)
                    meta["max_sec"] = sec
                    mix = mix2
                    break
                except Exception as e2:
                    if not is_oom(e2):
                        raise
                    empty_cuda_cache()
            else:
                raise RuntimeError(f"PS4 OOM after retries: {e}") from e

        scale = 1.0
        if self.match_rms:
            out, scale = _match_rms_to_ref(out, mix)
        meta["rms_scale"] = float(scale)
        return np.asarray(out, dtype=np.float32), meta


def create_tse(backend: str = "ps4", **kwargs: Any) -> PS4Extractor:
    backend = (backend or "ps4").lower()
    if backend in ("ps4", "bsrnn", "ps4_bsrnn"):
        return PS4Extractor(**kwargs)
    raise ValueError(f"未知 TSE backend: {backend}（仅支持 ps4）")

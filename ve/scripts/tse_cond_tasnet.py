#!/usr/bin/env python3
"""Cond-TasNet TSE：enroll → ECAPA emb → FiLM 条件分离 CMD mix。

条件向量必须与训练一致（ECAPA-VoxCeleb 192-d），不要用 Presence 的 eres 向量。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from audio_io import peak_normalize
from cond_tasnet_model import CondTasNetInferencer
from paths import default_cond_tasnet_ckpt, default_ecapa_dir


def _load_ecapa(ecapa_dir: Path, device: str):
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier  # type: ignore

    savedir = str(ecapa_dir)
    kwargs: dict[str, Any] = {"run_opts": {"device": device}}
    if (ecapa_dir / "hyperparams.yaml").is_file():
        return EncoderClassifier.from_hparams(
            source=savedir, savedir=savedir, **kwargs
        )
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=savedir,
        **kwargs,
    )


class CondTasNetExtractor:
    name = "cond_tasnet"

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        ecapa_dir: str | Path | None = None,
        device: str = "cuda:0",
        chunk_sec: float = 4.0,
        peak: float = 0.95,
        **_kwargs: Any,
    ):
        self.device = device
        self.peak = float(peak)
        ckpt = Path(checkpoint) if checkpoint else default_cond_tasnet_ckpt()
        if not ckpt.is_file():
            raise SystemExit(
                f"找不到 Cond-TasNet 权重: {ckpt}\n"
                "  放到 $VE_MODEL_DIR/cond_tasnet/best.pt 或设 COND_TASNET_CKPT"
            )
        self.ckpt = ckpt.resolve()
        self.infer = CondTasNetInferencer(str(self.ckpt), device=device, chunk_sec=chunk_sec)
        ecapa = Path(ecapa_dir) if ecapa_dir else default_ecapa_dir()
        ecapa.mkdir(parents=True, exist_ok=True)
        self.ecapa = _load_ecapa(ecapa, device)
        self.ecapa.eval()
        print(f"[INFO] CondTasNet ready ckpt={self.ckpt} ecapa={ecapa}", flush=True)

    def _embed(self, wav: np.ndarray) -> np.ndarray:
        import torch

        signal = torch.from_numpy(np.asarray(wav, dtype=np.float32)).float().unsqueeze(0)
        signal = signal.to(self.device)
        with torch.inference_mode():
            emb = self.ecapa.encode_batch(signal)
        return emb.squeeze().detach().cpu().numpy().astype(np.float32)

    def extract(
        self,
        mixture: np.ndarray,
        enroll: np.ndarray,
        *,
        sr: int = 16000,
        max_sec: float = 0.0,
        **_kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del sr, max_sec
        mix = np.asarray(mixture, dtype=np.float32).reshape(-1)
        enr = np.asarray(enroll, dtype=np.float32).reshape(-1)
        emb = self._embed(enr)
        est = self.infer.separate(mix, emb).numpy().astype(np.float32).reshape(-1)
        out = peak_normalize(est, peak=self.peak)
        meta = {
            "tse_backend": self.name,
            "ckpt": str(self.ckpt),
            "arch": getattr(self.infer, "arch", {}),
            "emb_dim": int(emb.shape[-1]),
            "n_samples": int(out.shape[0]),
        }
        return out, meta

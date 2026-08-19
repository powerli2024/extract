#!/usr/bin/env python3
"""Speaker-conditioned Conv-TasNet（V3 小网 + V3.5 SepBlock），按 checkpoint 自动选结构。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(channels)
        self.prelu = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prelu(self.bn(self.conv(x)))


class SepBlock(nn.Module):
    def __init__(self, B: int, H: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.in_conv = nn.Conv1d(B, H, 1)
        self.depth_conv = nn.Conv1d(H, H, kernel_size, padding=padding, dilation=dilation, groups=H)
        self.bn = nn.BatchNorm1d(H)
        self.prelu = nn.PReLU()
        self.out_conv = nn.Conv1d(H, B, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.prelu(self.in_conv(x))
        y = self.prelu(self.bn(self.depth_conv(y)))
        y = self.out_conv(y)
        return self.prelu(y + residual)


class SpeakerConditionedConvTasNet(nn.Module):
    def __init__(
        self,
        *,
        n_src: int = 1,
        N: int = 256,
        L: int = 20,
        B: int = 128,
        H: int = 512,
        P: int = 3,
        X: int = 6,
        emb_dim: int = 192,
        block: str = "v3",
    ):
        super().__init__()
        self.n_src = n_src
        self.N, self.L = N, L
        self.block = block
        self.encoder = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, padding=L // 2)
        self.film = nn.Linear(emb_dim, 2 * N)
        self.bottleneck = nn.Conv1d(N, B, 1)
        blocks: list[nn.Module] = []
        for _ in range(X):
            for r in range(P):
                if block == "v35":
                    blocks.append(SepBlock(B, H, kernel_size=3, dilation=2 ** r))
                else:
                    blocks.append(TemporalBlock(B, kernel_size=3, dilation=2 ** r))
        self.tcn = nn.Sequential(*blocks)
        self.mask_conv = nn.Conv1d(B, N * n_src, 1)
        self.prelu = nn.PReLU()
        self.decoder = nn.ConvTranspose1d(N, 1, kernel_size=L, stride=L // 2, padding=L // 2)

    def forward(self, mixture: torch.Tensor, spk_emb: torch.Tensor) -> torch.Tensor:
        x = mixture.unsqueeze(1)
        w = self.prelu(self.encoder(x))
        gamma, beta = self.film(spk_emb).chunk(2, dim=-1)
        w = gamma.unsqueeze(-1) * w + beta.unsqueeze(-1)
        y = self.bottleneck(w)
        y = self.tcn(y)
        m = torch.sigmoid(self.mask_conv(y))
        m = m[:, : self.N, :]
        z = w * m
        out = self.decoder(z).squeeze(1)
        t = mixture.shape[-1]
        if out.shape[-1] > t:
            out = out[..., :t]
        elif out.shape[-1] < t:
            out = F.pad(out, (0, t - out.shape[-1]))
        return out


def _state_dict_of(ckpt: Any) -> dict[str, Any]:
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if isinstance(ckpt, dict) and all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt
    raise ValueError("checkpoint 需含 model state_dict")


def infer_arch(sd: dict[str, Any]) -> dict[str, Any]:
    n, _, l = sd["encoder.weight"].shape
    emb_dim = int(sd["film.weight"].shape[1])
    b = int(sd["bottleneck.weight"].shape[0])
    idxs = []
    for k in sd:
        if k.startswith("tcn.") and k.endswith(".weight"):
            try:
                idxs.append(int(k.split(".")[1]))
            except ValueError:
                continue
    n_blocks = (max(idxs) + 1) if idxs else 1
    if "tcn.0.in_conv.weight" in sd:
        h = int(sd["tcn.0.in_conv.weight"].shape[0])
        p = 4
        for cand in (3, 4, 5, 6, 8):
            if n_blocks % cand == 0:
                p = cand
                break
        x = max(1, n_blocks // p)
        return {"block": "v35", "N": n, "L": l, "B": b, "H": h, "P": p, "X": x, "emb_dim": emb_dim}
    h = int(sd["tcn.0.conv.weight"].shape[0])
    p = 3
    for cand in (3, 4, 5, 6):
        if n_blocks % cand == 0:
            p = cand
            break
    x = max(1, n_blocks // p)
    return {"block": "v3", "N": n, "L": l, "B": b, "H": h, "P": p, "X": x, "emb_dim": emb_dim}


class CondTasNetInferencer:
    def __init__(self, checkpoint: str, device: str = "cuda:0", chunk_sec: float = 4.0):
        self.device = device
        self.chunk_sec = float(chunk_sec)
        try:
            ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint, map_location=device)
        sd = _state_dict_of(ckpt)
        inferred = infer_arch(sd)
        arch = dict(ckpt.get("arch") or {}) if isinstance(ckpt, dict) else {}
        if arch:
            allowed = {"n_src", "N", "L", "B", "H", "P", "X", "emb_dim"}
            arch = {k: v for k, v in arch.items() if k in allowed}
            for k, v in inferred.items():
                if k == "block" or k not in arch:
                    arch[k] = v
        else:
            arch = inferred
        self.arch = arch
        self.model = SpeakerConditionedConvTasNet(**arch).to(device)
        self.model.load_state_dict(sd, strict=True)
        self.model.eval()
        self.sr = 16000
        print(f"[INFO] Cond-TasNet {Path(checkpoint).name} arch={arch}", flush=True)

    @torch.inference_mode()
    def separate(self, mixture, spk_emb, chunk_sec: float | None = None) -> torch.Tensor:
        import numpy as np

        chunk_sec = float(chunk_sec or self.chunk_sec)
        if isinstance(mixture, np.ndarray):
            mixture = torch.from_numpy(mixture).float()
        if isinstance(spk_emb, np.ndarray):
            spk_emb = torch.from_numpy(spk_emb).float()
        mixture = mixture.to(self.device)
        spk_emb = spk_emb.to(self.device).unsqueeze(0) if spk_emb.dim() == 1 else spk_emb
        chunk_len = int(chunk_sec * self.sr)
        if mixture.numel() <= chunk_len:
            return self.model(mixture.unsqueeze(0), spk_emb).squeeze(0).cpu()
        hop = max(1, chunk_len // 2)
        out = torch.zeros_like(mixture)
        weight = torch.zeros_like(mixture)
        for start in range(0, int(mixture.shape[-1]), hop):
            end = min(start + chunk_len, int(mixture.shape[-1]))
            chunk = mixture[start:end]
            if chunk.numel() < chunk_len // 4:
                break
            pad_len = chunk_len - int(chunk.shape[-1])
            if pad_len > 0:
                chunk = F.pad(chunk, (0, pad_len))
            est = self.model(chunk.unsqueeze(0), spk_emb).squeeze(0)
            est = est[: end - start]
            win = torch.hann_window(int(est.shape[-1]), device=self.device)
            out[start:end] += est * win
            weight[start:end] += win
        weight = torch.clamp(weight, min=1e-8)
        return (out / weight).cpu()

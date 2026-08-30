#!/usr/bin/env python3
"""方案 C：MossFormer 分离 → enroll 声纹选路 → 输出 best stream（不做 PS4/WeSep）。"""

from __future__ import annotations

from typing import Any

import numpy as np

from audio_io import cosine_sim, preprocess_for_separation
from presence_encoder import PresenceEncoder


class SepRouteExtractor:
    """mixture + enroll → 选声纹最像的分离轨。"""

    name = "sep_route_mossformer"

    def __init__(
        self,
        separator: Any,
        *,
        encoder: PresenceEncoder | None = None,
        device: str = "cuda:0",
        peak: float = 0.70,
    ):
        if separator is None:
            raise ValueError("SepRouteExtractor 需要 separator")
        self.separator = separator
        self.encoder = encoder
        self.device = device
        self.peak = peak
        print(f"[INFO] SepRoute TSE ready (MossFormer 选路) device={device}", flush=True)

    def _separate(self, mixture: np.ndarray, sr: int) -> dict[str, np.ndarray]:
        mix = np.asarray(mixture, dtype=np.float32).reshape(-1)
        peak, sr = preprocess_for_separation(mix, sr, peak=self.peak)
        streams: dict[str, np.ndarray] = {"mix": mix, "peak": peak}
        if hasattr(self.separator, "separate"):
            s1, s2 = self.separator.separate(peak, sr=sr)
            streams["spk1"] = np.asarray(s1, dtype=np.float32).reshape(-1)
            streams["spk2"] = np.asarray(s2, dtype=np.float32).reshape(-1)
        else:
            raise RuntimeError("separator 无 separate() 接口")
        return streams

    def extract(
        self,
        mixture: np.ndarray,
        enroll: np.ndarray,
        *,
        sr: int = 16000,
        max_sec: float = 0.0,
        streams: dict[str, np.ndarray] | None = None,
        enroll_emb: np.ndarray | None = None,
        preferred_stream: str | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del max_sec  # 选路路径暂不截断；OOM 由 ORT 侧处理
        # Presence 未分出 spk 轨时再分一次（sep 失败回退 / use_sep 关掉等）
        reused = (
            streams is not None
            and ("spk1" in streams or "d1_spk1" in streams)
            and ("spk2" in streams or "d1_spk2" in streams)
        )
        if not reused:
            streams = self._separate(mixture, sr)
            preferred_stream = None  # 新分离结果需重新选路
        else:
            # 统一别名，供选路
            if "spk1" not in streams and "d1_spk1" in streams:
                streams = dict(streams)
                streams["spk1"] = streams["d1_spk1"]
                streams["spk2"] = streams["d1_spk2"]
            if preferred_stream == "d1_spk1":
                preferred_stream = "spk1"
            elif preferred_stream == "d1_spk2":
                preferred_stream = "spk2"
        if preferred_stream and preferred_stream in streams and preferred_stream != "peak":
            # Presence 已选好的轨直接用，避免再 embed
            name = preferred_stream
            sims: dict[str, float] = {preferred_stream: 1.0}
        else:
            if self.encoder is None:
                raise RuntimeError("未提供 preferred_stream 时 SepRoute 需要 encoder")
            e = enroll_emb if enroll_emb is not None else self.encoder.embed(enroll, sr)
            sims = {}
            best_name, best_s = "mix", -1.0
            for name, w in streams.items():
                if name == "peak":
                    continue
                s = cosine_sim(e, self.encoder.embed(w, sr))
                sims[name] = float(s)
                if s > best_s:
                    best_s, best_name = s, name
            name = best_name
        out = np.asarray(streams[name], dtype=np.float32).reshape(-1)
        meta = {
            "tse_backend": self.name,
            "routed_stream": name,
            "sim_streams": {k: round(v, 6) for k, v in sims.items()},
            "n_streams": len(streams),
            "reused_presence_streams": reused,
        }
        return out, meta

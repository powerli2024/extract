#!/usr/bin/env python3
"""标签无关的 mix/分离流自适应路由。

仅当分离流相对原始 mix 的注册声纹相似度有明确增益时才使用它，避免把
盲分离失真或错误说话人无条件送入 ASR。正、负样本执行完全相同的规则。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from audio_io import cosine_sim, peak_normalize
from presence_encoder import PresenceEncoder


class AdaptiveRouteExtractor:
    name = "adaptive_mix_sep_route"

    def __init__(self, *, encoder: PresenceEncoder, min_gain: float = 0.03,
                 peak: float = 0.95, device: str = "cuda:0", **_kwargs: Any):
        del device
        if encoder is None:
            raise ValueError("adaptive_route requires a presence encoder")
        self.encoder = encoder
        self.min_gain = float(min_gain)
        self.peak = float(peak)
        print(f"[INFO] AdaptiveRoute ready (sep gain >= {self.min_gain:.3f} → sep, else mix)", flush=True)

    def extract(
        self,
        mixture: np.ndarray,
        enroll: np.ndarray,
        *,
        sr: int = 16000,
        streams: dict[str, np.ndarray] | None = None,
        enroll_emb: np.ndarray | None = None,
        **_kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        mix = np.asarray(mixture, dtype=np.float32).reshape(-1)
        candidates = dict(streams or {})
        candidates["mix"] = mix
        e = enroll_emb if enroll_emb is not None else self.encoder.embed(enroll, sr)
        sims: dict[str, float] = {}
        for name, wav in candidates.items():
            if name == "peak":
                continue
            w = np.asarray(wav, dtype=np.float32).reshape(-1)
            if w.size < int(0.15 * sr):
                continue
            sims[name] = float(cosine_sim(e, self.encoder.embed(w, sr)))

        mix_score = float(sims.get("mix", -1.0))
        separated = [(score, name) for name, score in sims.items() if name != "mix"]
        best_sep_score, best_sep_name = max(separated, default=(-1.0, ""))
        use_sep = bool(best_sep_name and best_sep_score >= mix_score + self.min_gain)
        chosen = best_sep_name if use_sep else "mix"
        out = np.asarray(candidates[chosen], dtype=np.float32).reshape(-1)
        return peak_normalize(out, peak=self.peak), {
            "tse_backend": self.name,
            "routed_stream": chosen,
            "route_rule": "sep_if_spk_gain_ge_min_gain_else_mix",
            "route_min_gain": self.min_gain,
            "mix_spk_score": round(mix_score, 6),
            "best_sep_stream": best_sep_name or None,
            "best_sep_spk_score": round(best_sep_score, 6) if best_sep_name else None,
            "sep_spk_gain": round(best_sep_score - mix_score, 6) if best_sep_name else None,
            "sim_streams": {k: round(v, 6) for k, v in sims.items()},
        }

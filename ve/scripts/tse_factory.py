#!/usr/bin/env python3
"""TSE / 选路后端工厂：ps4 | wesep_bsrnn | sep_route | adaptive_route | mix | cond_tasnet。"""

from __future__ import annotations

from typing import Any


def create_tse(backend: str = "ps4", **kwargs: Any):
    """按名创建提取器。

    - ps4 / ps4_bsrnn: HF PS4
    - wesep / wesep_bsrnn: 官方 WeSep bsrnn_ecapa_vox1
    - sep_route / mossformer: MossFormer 分离后按 enroll 声纹选路
    - adaptive_route: 仅在分离流声纹明显优于 mix 时选择分离流
    - mix / passthrough: CMD mix 直接送 ASR（不做提取）
    """
    b = (backend or "ps4").lower().strip()
    if b in ("ps4", "ps4_bsrnn", "bsrnn"):
        from tse_ps4 import PS4Extractor

        return PS4Extractor(
            weights=kwargs.get("weights"),
            device=kwargs.get("device", "cuda:0"),
            wesep_dir=kwargs.get("wesep_dir"),
        )
    if b in ("wesep", "wesep_bsrnn", "wesep_bsrnn_ecapa"):
        from tse_wesep import WesepExtractor

        return WesepExtractor(
            model_dir=kwargs.get("wesep_model_dir") or kwargs.get("model_dir"),
            language=kwargs.get("wesep_language", "english"),
            device=kwargs.get("device", "cuda:0"),
            match_rms=bool(kwargs.get("match_rms", False)),
        )
    if b in ("sep_route", "mossformer", "route", "sep"):
        from tse_sep_route import SepRouteExtractor

        sep = kwargs.get("separator")
        if sep is None:
            raise RuntimeError(
                "sep_route 需要 MossFormer 分离器。请设置 USE_SEP=1 / PIPELINE=sep_route，"
                "并保证 VM ONNX 或 ClearVoice 权重可用（见 download_moss_onnx.sh）"
            )
        return SepRouteExtractor(
            separator=sep,
            encoder=kwargs.get("encoder"),
            device=kwargs.get("device", "cuda:0"),
            peak=float(kwargs.get("peak", 0.95)),
        )
    if b in ("adaptive_route", "adaptive", "mix_sep_route"):
        from tse_adaptive_route import AdaptiveRouteExtractor

        return AdaptiveRouteExtractor(
            encoder=kwargs.get("encoder"),
            min_gain=float(kwargs.get("route_min_gain", 0.03)),
            peak=float(kwargs.get("peak", 0.95)),
            device=kwargs.get("device", "cuda:0"),
        )
    if b in ("cond_tasnet", "condtasnet", "tasnet", "cond-tasnet"):
        from tse_cond_tasnet import CondTasNetExtractor

        return CondTasNetExtractor(
            checkpoint=kwargs.get("cond_tasnet_ckpt") or kwargs.get("checkpoint"),
            ecapa_dir=kwargs.get("ecapa_dir"),
            device=kwargs.get("device", "cuda:0"),
            chunk_sec=float(kwargs.get("chunk_sec") or kwargs.get("tasnet_chunk_sec") or 4.0),
        )
    if b in ("mix", "passthrough", "mix_passthrough", "cmd", "none"):
        from tse_mix import MixPassthroughExtractor

        return MixPassthroughExtractor(
            peak=float(kwargs.get("peak", 0.95)),
            device=kwargs.get("device", "cuda:0"),
        )
    raise ValueError(
        f"未知 tse-backend={backend!r}；可选: ps4 | wesep_bsrnn | sep_route | adaptive_route | mix | cond_tasnet"
    )

#!/usr/bin/env python3
"""运行标签：VAD / sep / thr 模式，用于校准与 VE_OUT 并存、防 thr 串用。"""

from __future__ import annotations

from typing import Any


def vad_tag(enroll_vad: bool) -> str:
    return "vad" if enroll_vad else "novad"


def calib_dir_name(
    *,
    backend: str = "eres2netv2",
    use_sep: bool = True,
    lang_split: bool = True,
    enroll_vad: bool = True,
    score_norm: str = "raw",
) -> str:
    """例: presence_calib_eres2netv2_sep1_ls_vad_raw"""
    sep = "sep1" if use_sep else "nosep"
    ls = "ls" if lang_split else "gthr"
    return (
        f"presence_calib_{backend}_{sep}_{ls}_{vad_tag(enroll_vad)}_{score_norm}"
    )


def assert_thr_runtime_compatible(
    thr_meta: dict[str, Any] | None,
    *,
    enroll_vad: bool,
    strict: bool = True,
) -> list[str]:
    """校准产物与当前 extract 设置是否一致。返回警告列表；strict 时不一致即抛错。"""
    meta = thr_meta or {}
    warns: list[str] = []
    if "enroll_vad" not in meta:
        warns.append(
            "thr 文件无 enroll_vad 字段（旧校准）；无法保证与当前 VAD 设置一致"
        )
    else:
        thr_vad = bool(meta["enroll_vad"])
        if thr_vad != bool(enroll_vad):
            msg = (
                f"thr.enroll_vad={thr_vad} 与当前 enroll_vad={enroll_vad} 不一致；"
                f"分数尺度不同，禁止串用（请换对应 presence_calib_* 目录）"
            )
            if strict:
                raise SystemExit(f"[ERR] {msg}")
            warns.append(msg)
    return warns

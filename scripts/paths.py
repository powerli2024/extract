#!/usr/bin/env python3
"""VM 路径与 sys.path：默认仅使用本包 scripts（已 vendoring 分离/ASR 代码）。"""

from __future__ import annotations

import os
from pathlib import Path

STAGE_DIRS = {
    "s1": "s1_onnx_full",
    "s2": "s2_cv_full",
    "s3": "s3_onnx_cascade",
    "s4": "s4_cv_cascade",
    "s5": "s5_onnx_then_cv_gate",
    "s6": "s6_onnx_then_onnx_gate",
    "s7": "s7_cv_then_onnx_gate",
    "s8": "s8_cv_then_cv_gate",
}

THR_NAMES = ("a", "b", "c")
VALID_SPLITS = ("pos", "neg")

# 已迁入本包的运行时模块（不再强制依赖外部 VB/VB_onnx）
VENDORED_MODULES = (
    "utils_audio.py",
    "cer_metrics.py",
    "asr_backend.py",
    "mossformer2_ss.py",
    "mossformer2_onnx.py",
)


def vm_root() -> Path:
    return Path(__file__).resolve().parents[1]


def media_root() -> Path:
    """VM 上一级（AutoDL 上常为 /root）。"""
    return vm_root().parent


def default_data_dir() -> Path:
    env = os.environ.get("DATA_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    for c in (
        Path("/root/autodl-tmp/datasetA"),
        Path("/root/datasetA"),
        Path("/root/autodl-tmp/vb_heavy/datasetA"),
        media_root() / "datasetA",
    ):
        if c.is_dir():
            return c.resolve()
    return (media_root() / "datasetA").resolve()


def default_vm_out() -> Path:
    env = os.environ.get("VM_OUT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    if Path("/root/autodl-tmp").is_dir():
        if (vm_root() / ".sep-only").is_file():
            return Path("/root/autodl-tmp/kws_sep").resolve()
        return Path("/root/autodl-tmp/vm").resolve()
    return (media_root() / "vm_out").resolve()


def assert_split(split: str) -> str:
    s = (split or "").strip()
    if s not in VALID_SPLITS:
        raise ValueError(f"非法 split={split!r}，仅允许 {VALID_SPLITS}")
    return s


def split_root(vm_out: Path, split: str) -> Path:
    return vm_out / assert_split(split)


def stage_dir(vm_out: Path, stage: str, split: str) -> Path:
    name = STAGE_DIRS.get(stage, stage)
    return split_root(vm_out, split) / name


def ensure_stage(vm_out: Path, stage: str, split: str) -> Path:
    d = stage_dir(vm_out, stage, split)
    (d / "wav").mkdir(parents=True, exist_ok=True)
    return d


def ensure_meta(vm_out: Path, split: str) -> Path:
    m = split_root(vm_out, split) / "meta"
    m.mkdir(parents=True, exist_ok=True)
    return m


def ensure_reports(vm_out: Path, split: str | None = None) -> Path:
    if split:
        r = split_root(vm_out, split) / "reports"
    else:
        r = vm_out / "reports"
    r.mkdir(parents=True, exist_ok=True)
    return r


def wav_path(stage_root: Path, uid: str, tag: str) -> Path:
    if "_" not in uid:
        raise ValueError(f"uid 缺少 split 前缀: {uid}")
    return stage_root / "wav" / f"{uid}_{tag}.wav"


def assert_vendored() -> Path:
    here = Path(__file__).resolve().parent
    missing = [n for n in VENDORED_MODULES if not (here / n).is_file()]
    if missing:
        raise FileNotFoundError(
            f"VM 缺少已整合模块: {missing}（应位于 {here}）"
        )
    return here


def setup_sys_path() -> None:
    """仅注入本包 scripts；可选追加外部 VB 路径作兼容回退。"""
    import sys

    here = assert_vendored()
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    # 兼容：若仍设置了 VB_DIR / VB_ONNX_DIR，放在后面，不覆盖本包同名模块
    if os.environ.get("VM_ALLOW_EXTERNAL_VB", "").strip() in ("1", "true", "YES"):
        for key in ("VB_DIR", "VB_ONNX_DIR"):
            root = os.environ.get(key, "").strip()
            if not root:
                continue
            p = Path(root).expanduser() / "scripts"
            if p.is_dir() and str(p) not in sys.path:
                sys.path.append(str(p))

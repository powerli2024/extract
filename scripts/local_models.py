#!/usr/bin/env python3
"""本地权重解析：运行期禁止下载，缺文件直接报错并提示 download_models.sh。"""

from __future__ import annotations

import os
from pathlib import Path

from paths import media_root


def enable_offline_env() -> None:
    """强制 HF/Transformers 离线；仅在 VM 运行脚本里调用。"""
    if os.environ.get("VM_ALLOW_DOWNLOAD", "").strip() in ("1", "true", "TRUE", "yes"):
        return
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    # 防止 modelscope 偷偷拉
    os.environ.setdefault("MODELSCOPE_SDK_DEBUG", "0")


def _hint() -> str:
    return "请先单独运行: cd VM && ./download_models.sh （运行脚本不会自动下载）"


def resolve_onnx_local() -> Path:
    env = os.environ.get("MOSS_ONNX_PATH", "").strip()
    cands: list[Path] = []
    if env:
        p = Path(env).expanduser()
        cands.append(p if p.suffix else p / "simple_model.onnx")
        if p.is_dir():
            cands.append(p / "simple_model.onnx")
    ckpt = Path(os.environ.get("MOSS_CKPT_DIR", "/root/checkpoints")).expanduser()
    media = media_root()
    cands += [
        ckpt / "MossFormer2_ONNX" / "simple_model.onnx",
        Path("/root/autodl-tmp/checkpoints/MossFormer2_ONNX/simple_model.onnx"),
        media / "checkpoints" / "MossFormer2_ONNX" / "simple_model.onnx",
    ]
    for c in cands:
        try:
            if c.is_file() and c.stat().st_size > 1024:
                return c.resolve()
        except OSError:
            continue
    raise FileNotFoundError(f"本地未找到 MossFormer2 ONNX。\n{_hint()}")


def resolve_cv_ckpt_local(model_name: str = "MossFormer2_SS_16K") -> Path:
    ckpt = Path(os.environ.get("MOSS_CKPT_DIR", "/root/checkpoints")).expanduser()
    media = media_root()
    cands = [
        ckpt / model_name / "last_best_checkpoint.pt",
        Path("/root/autodl-tmp/checkpoints") / model_name / "last_best_checkpoint.pt",
        media / "checkpoints" / model_name / "last_best_checkpoint.pt",
    ]
    for c in cands:
        try:
            if c.is_file() and c.stat().st_size > 1024:
                return c.resolve()
        except OSError:
            continue
    raise FileNotFoundError(f"本地未找到 ClearVoice {model_name}。\n{_hint()}")


def resolve_asr_local(model_dir: str | None = None) -> Path:
    cands: list[Path] = []
    if model_dir and str(model_dir).strip():
        cands.append(Path(str(model_dir).strip()).expanduser())
    for key in ("ASR_MODEL_DIR", "QWEN3_ASR_DIR"):
        v = os.environ.get(key, "").strip()
        if v:
            cands.append(Path(v).expanduser())
    cands += [
        Path("/root/Qwen3-ASR-1.7B"),
        media_root() / "Qwen3-ASR-1.7B",
        Path("/root/autodl-tmp/Qwen3-ASR-1.7B"),
    ]
    for c in cands:
        try:
            if c.is_dir() and (
                (c / "config.json").is_file()
                or (c / "model.safetensors").is_file()
                or any(c.glob("*.safetensors"))
                or any(c.glob("*.bin"))
            ):
                return c.resolve()
        except OSError:
            continue
    raise FileNotFoundError(
        f"本地未找到 Qwen3-ASR 目录（需含 config.json / 权重）。\n{_hint()}\n"
        f"或设置: export ASR_MODEL_DIR=/path/to/Qwen3-ASR-1.7B"
    )


def assert_runtime_models(
    *,
    need_onnx: bool = False,
    need_cv: bool = False,
    need_asr: bool = True,
    asr_model_dir: str | None = None,
) -> dict[str, str]:
    enable_offline_env()
    out: dict[str, str] = {"offline": "1" if os.environ.get("HF_HUB_OFFLINE") == "1" else "0"}
    if need_onnx:
        out["onnx"] = str(resolve_onnx_local())
        os.environ["MOSS_ONNX_PATH"] = out["onnx"]
    if need_cv:
        out["cv_ckpt"] = str(resolve_cv_ckpt_local())
    if need_asr:
        out["asr"] = str(resolve_asr_local(asr_model_dir))
        os.environ["ASR_MODEL_DIR"] = out["asr"]
        os.environ["QWEN3_ASR_DIR"] = out["asr"]
    return out

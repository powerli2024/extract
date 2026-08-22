#!/usr/bin/env python3
"""下载 VE 推理模型到 VE_MODEL_DIR（默认 /root/autodl-tmp/ve_models）。

中文为主、仅推理：
  - PS4: HuggingFace TaurenMountain/PS4（checkpoint + inference.py）
  - Presence: ERes2NetV2-zh（ModelScope）+ cnceleb ResNet34-LM 回退
明确不下：Whisper / DNSMOS / VoxCeleb / MossFormer ONNX / wesep 训练仓
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def _info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def _ok(msg: str) -> None:
    print(f"[ OK ] {msg}", flush=True)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}", flush=True)


def model_root() -> Path:
    env = os.environ.get("VE_MODEL_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    if Path("/root/autodl-tmp").is_dir():
        return Path("/root/autodl-tmp/ve_models").resolve()
    return (Path(__file__).resolve().parents[2] / "ve_models").resolve()


def ensure_hf_env() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    if Path("/root/autodl-tmp").is_dir():
        os.environ.setdefault("HF_HOME", "/root/autodl-tmp/cache/huggingface")
        os.environ.setdefault("MODELSCOPE_CACHE", "/root/autodl-tmp/cache/modelscope")
        Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
        Path(os.environ["MODELSCOPE_CACHE"]).mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src)
    except Exception:
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


def ps4_ready(ps4_dir: Path) -> bool:
    """PS4Extractor 只认 checkpoint_epoch037.pt；任意 checkpoint*.pt 不算就绪。"""
    ckpt = ps4_dir / "checkpoint_epoch037.pt"
    inf = ps4_dir / "inference.py"
    return ckpt.is_file() and inf.is_file()


def download_ps4(ps4_dir: Path) -> None:
    """下载官方推理包：https://huggingface.co/TaurenMountain/PS4"""
    ps4_dir.mkdir(parents=True, exist_ok=True)
    if ps4_ready(ps4_dir):
        _ok(f"PS4 已就绪 → {ps4_dir}")
        return

    legacy = Path("/root/autodl-tmp/ps4_models/PS4")
    if legacy.is_dir() and (legacy / "checkpoint_epoch037.pt").is_file():
        _info(f"复用 {legacy}")
        for name in ("checkpoint_epoch037.pt", "inference.py", "README.md"):
            src = legacy / name
            if src.is_file():
                link_or_copy(src, ps4_dir / name)
        if ps4_ready(ps4_dir):
            _ok(f"PS4 复用完成 → {ps4_dir}")
            return

    _info("下载 PS4 推理包 TaurenMountain/PS4 → " + str(ps4_dir))
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as e:
        raise RuntimeError("需要 huggingface_hub：pip install huggingface_hub") from e

    # 先拉必需文件；失败再整仓 snapshot
    required = ("checkpoint_epoch037.pt", "inference.py")
    try:
        for fn in required:
            path = hf_hub_download(
                repo_id="TaurenMountain/PS4",
                filename=fn,
                local_dir=str(ps4_dir),
            )
            _info(f"got {fn} → {path}")
    except Exception as e:
        _warn(f"按文件下载失败 ({e})，改用 snapshot_download")
        snapshot_download(
            repo_id="TaurenMountain/PS4",
            local_dir=str(ps4_dir),
        )

    if not ps4_ready(ps4_dir):
        raise RuntimeError(
            f"PS4 下载后校验失败。需要 {ps4_dir}/checkpoint_epoch037.pt 与 inference.py\n"
            "请检查 HF_ENDPOINT / 网络，或手动放入上述文件。"
        )
    _ok(f"PS4 下载完成 → {ps4_dir}")
    for p in sorted(ps4_dir.iterdir()):
        if p.is_file():
            print(f"       {p.name}\t{p.stat().st_size}", flush=True)


def normalize_wespeaker(dir_: Path) -> None:
    target = dir_ / "avg_model.pt"
    if target.is_file():
        return
    for name in ("avg_model", "model_5.pt"):
        src = dir_ / name
        if src.is_file():
            try:
                target.symlink_to(src.name)
            except Exception:
                shutil.copy2(src, target)
            return
    pts = sorted(dir_.glob("model_*.pt"))
    if pts:
        shutil.copy2(pts[-1], target)


def download_cnceleb(chs_dir: Path) -> None:
    """中文 Presence 回退编码器（推理用；非 PS4 训练依赖）。"""
    if chs_dir.is_dir() and (chs_dir / "config.yaml").is_file():
        normalize_wespeaker(chs_dir)
        _ok(f"cnceleb_resnet34_LM 已就绪 → {chs_dir}")
        return

    legacy = Path("/root/autodl-tmp/ps4_models/cnceleb_resnet34_LM")
    if legacy.is_dir() and (legacy / "config.yaml").is_file():
        _info(f"复用 {legacy}")
        link_or_copy(legacy, chs_dir)
        normalize_wespeaker(chs_dir)
        _ok(f"cnceleb 复用完成 → {chs_dir}")
        return

    _info("下载 Wespeaker/wespeaker-cnceleb-resnet34-LM")
    from huggingface_hub import snapshot_download

    chs_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        "Wespeaker/wespeaker-cnceleb-resnet34-LM",
        local_dir=str(chs_dir),
    )
    normalize_wespeaker(chs_dir)
    if not (chs_dir / "config.yaml").is_file():
        raise RuntimeError(f"cnceleb 下载后缺 config.yaml: {chs_dir}")
    _ok(f"cnceleb 下载完成 → {chs_dir}")


def download_eres2net(eres_dir: Path) -> None:
    eres_dir.mkdir(parents=True, exist_ok=True)
    marker = eres_dir / "MODELSCOPE_PATH.txt"
    if marker.is_file() or any(eres_dir.rglob("*.pt")) or any(eres_dir.rglob("*.bin")):
        _ok(f"ERes2NetV2 已就绪 → {eres_dir}")
        return
    _info("预下载 ModelScope ERes2NetV2-zh（Presence 主编码器）")
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        _warn("modelscope 未安装，跳过 ERes2NetV2；可设 PRESENCE_BACKEND=resnet34")
        return
    path = snapshot_download(
        "iic/speech_eres2netv2_sv_zh-cn_16k-common",
        cache_dir=str(eres_dir / "_ms_cache"),
    )
    marker.write_text(str(path) + "\n", encoding="utf-8")
    _ok(f"ERes2NetV2 → {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VE 推理模型下载（中文为主，无训练组件）")
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--skip-ps4", action="store_true")
    p.add_argument("--skip-cnceleb", action="store_true")
    p.add_argument("--skip-eres2net", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ensure_hf_env()
    root = (args.model_dir or model_root()).resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["VE_MODEL_DIR"] = str(root)
    _info(f"VE_MODEL_DIR={root}")
    _info(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}")
    _info("不下：Whisper / DNSMOS / VoxCeleb / MossFormer ONNX / wesep（HF inference.py 自包含）")

    errors: list[str] = []
    if not args.skip_ps4:
        try:
            download_ps4(root / "PS4")
        except Exception as e:
            _fail(str(e))
            errors.append("PS4")
    if not args.skip_cnceleb:
        try:
            download_cnceleb(root / "cnceleb_resnet34_LM")
        except Exception as e:
            _fail(str(e))
            errors.append("cnceleb")
    if not args.skip_eres2net:
        try:
            download_eres2net(root / "eres2netv2_zh")
        except Exception as e:
            _warn(str(e))
            # ERes2Net 失败不阻断：可回退 resnet34

    print("---", flush=True)
    for p in sorted(root.iterdir()):
        print(f"  {p.name}", flush=True)

    if errors:
        _fail("必需模型失败: " + ", ".join(errors))
        return 2
    _ok("download_models 完成（推理-only）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

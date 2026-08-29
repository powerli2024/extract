#!/usr/bin/env python3
"""Strict read-only verification for the unified Ubuntu 20.04 / cu124 env."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


EXPECTED = {
    "torch": "2.6.0",
    "torchaudio": "2.6.0",
    "torchvision": "0.21.0",
    "onnxruntime-gpu": "1.20.2",
    "clearvoice": "0.1.2",
    "qwen-asr": "0.0.6",
    "transformers": "4.57.6",
    "accelerate": "1.12.0",
    "modelscope": "1.39.1",
    "funasr": "1.4.6",
    "kaldiio": "2.18.1",
    "addict": "2.4.0",
    "numpy": "1.26.4",
    "librosa": "0.10.2.post1",
    "soundfile": "0.12.1",
}


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dae-repo", type=Path)
    parser.add_argument("--dae-config", type=Path)
    parser.add_argument("--dae-checkpoint", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--require-dae", action="store_true")
    args = parser.parse_args()

    hard: list[str] = []
    warnings: list[str] = []
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {},
    }

    if sys.version_info[:2] != (3, 10):
        hard.append(f"Python 必须是 3.10.x，当前为 {sys.version.split()[0]}")

    for name, expected in EXPECTED.items():
        try:
            actual = package_version(name)
            report["packages"][name] = actual  # type: ignore[index]
            if actual.split("+")[0] != expected:
                hard.append(f"{name}={actual}，期望 {expected}")
        except importlib.metadata.PackageNotFoundError:
            hard.append(f"缺少包 {name}")

    try:
        import torch

        report["torch_cuda_build"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        if torch.version.cuda != "12.4":
            hard.append(f"torch CUDA build={torch.version.cuda!r}，期望 '12.4'")
        if not torch.cuda.is_available():
            hard.append("torch.cuda.is_available()=False")
        else:
            report["gpu"] = torch.cuda.get_device_name(0)
            sample = torch.ones(16, device="cuda")
            if float((sample * 2).sum().cpu()) != 32.0:
                hard.append("CUDA 张量计算结果异常")
    except Exception as exc:
        hard.append(f"PyTorch CUDA 自检失败: {type(exc).__name__}: {exc}")

    try:
        import torch  # noqa: F401  # 先加载 PyTorch 携带的 CUDA/cuDNN 库
        import onnxruntime as ort

        providers = ort.get_available_providers()
        report["ort_providers"] = providers
        if "CUDAExecutionProvider" not in providers:
            hard.append(f"ONNX Runtime 无 CUDAExecutionProvider: {providers}")
    except Exception as exc:
        hard.append(f"ONNX Runtime 自检失败: {type(exc).__name__}: {exc}")

    for module in (
        "qwen_asr",
        "clearvoice",
        "modelscope",
        "funasr",
        "kaldiio",
        "addict",
        "editdistance",
        "pypinyin",
    ):
        try:
            importlib.import_module(module)
        except Exception as exc:
            hard.append(f"import {module} 失败: {type(exc).__name__}: {exc}")

    dae_status: dict[str, object] = {"requested": bool(args.dae_repo)}
    if args.dae_repo:
        repo = args.dae_repo.resolve()
        required = [repo / "pyproject.toml", repo / "tools" / "build_zh_text_cues.py"]
        if args.dae_config:
            required.append(args.dae_config.resolve())
        if args.dae_checkpoint:
            required.append(args.dae_checkpoint.resolve())
        missing = [str(path) for path in required if not path.is_file()]
        dae_status["repo"] = str(repo)
        dae_status["missing_assets"] = missing
        try:
            import wesep  # noqa: F401
            import kce  # noqa: F401
            dae_status["imports"] = "ok"
        except Exception as exc:
            missing.append(f"Python import wesep/kce: {type(exc).__name__}: {exc}")
        if missing:
            message = "DAE-TSE 未就绪: " + "; ".join(missing)
            (hard if args.require_dae else warnings).append(message)
        else:
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "from tools.build_zh_text_cues import text_to_phone_labels; "
                        "print(text_to_phone_labels('你好'))",
                    ],
                    cwd=repo,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                dae_status["zh_cue_smoke"] = result.stdout.strip()
            except Exception as exc:
                message = f"DAE 中文 cue 自检失败: {type(exc).__name__}: {exc}"
                (hard if args.require_dae else warnings).append(message)
    report["dae_tse"] = dae_status
    report["hard_errors"] = hard
    report["warnings"] = warnings
    report["status"] = "FAIL" if hard else ("WARN" if warnings else "PASS")

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    return 1 if hard else (2 if warnings else 0)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""在 import onnxruntime 之前，把 CUDA 动态库路径补进进程环境。

ORT-GPU（CUDA12/cuDNN9）仍会 dlopen 若干 *.so.11 / *.so.12：
  libcublasLt.so.12 / libcufft.so.11 / libcudnn.so.9 ...
这些通常来自 pip 包 nvidia-*-cu12，位于 site-packages/nvidia/*/lib/。
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

_NVIDIA_SUBLIBS = (
    "nvidia/cublas/lib",
    "nvidia/cuda_runtime/lib",
    "nvidia/cuda_nvrtc/lib",
    "nvidia/cudnn/lib",
    "nvidia/cufft/lib",
    "nvidia/curand/lib",
    "nvidia/cusolver/lib",
    "nvidia/cusparse/lib",
    "nvidia/nccl/lib",
    "nvidia/nvjitlink/lib",
    "torch/lib",
)

# 顺序有关：先基础 runtime / nvjitlink，再 cublas/cufft/cudnn
_PRELOAD_NAMES = (
    "libnvJitLink.so.12",
    "libnvJitLink.so.11",
    "libcudart.so.12",
    "libcudart.so.11",
    "libcublasLt.so.12",
    "libcublasLt.so.11",
    "libcublas.so.12",
    "libcublas.so.11",
    "libcufft.so.11",  # ORT 常见硬依赖（即便 CUDA12 wheel）
    "libcufft.so.12",
    "libcudnn.so.9",
    "libcudnn.so.8",
    "libcurand.so.10",
    "libcusolver.so.11",
    "libcusparse.so.12",
)


def _fix_omp_threads() -> None:
    v = str(os.environ.get("OMP_NUM_THREADS", "")).strip()
    if not v.isdigit() or int(v) <= 0:
        os.environ["OMP_NUM_THREADS"] = "8"


def _site_roots() -> list[Path]:
    roots: list[Path] = []
    try:
        import site

        for p in site.getsitepackages():
            roots.append(Path(p))
        us = site.getusersitepackages()
        if us:
            roots.append(Path(us))
    except Exception:
        pass
    try:
        import torch

        roots.append(Path(torch.__file__).resolve().parent.parent)
    except Exception:
        pass
    # sys.path 里的 site-packages
    for p in sys.path:
        if p and "site-packages" in p.replace("\\", "/"):
            roots.append(Path(p))
    out: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        try:
            s = str(r.resolve())
        except Exception:
            continue
        if s not in seen and Path(s).is_dir():
            seen.add(s)
            out.append(Path(s))
    return out


def _candidate_lib_dirs() -> list[Path]:
    dirs: list[Path] = []
    for site in _site_roots():
        for sub in _NVIDIA_SUBLIBS:
            p = site / sub
            if p.is_dir():
                dirs.append(p)

    for p in (
        Path("/usr/local/cuda/lib64"),
        Path("/usr/local/cuda/targets/x86_64-linux/lib"),
        Path("/usr/lib/x86_64-linux-gnu"),
        Path(os.environ["CUDA_HOME"]) / "lib64" if os.environ.get("CUDA_HOME") else None,
        Path(os.environ["CUDA_PATH"]) / "lib64" if os.environ.get("CUDA_PATH") else None,
    ):
        if p is not None and p.is_dir():
            dirs.append(p)

    out: list[Path] = []
    seen: set[str] = set()
    for d in dirs:
        try:
            s = str(d.resolve())
        except Exception:
            continue
        if s not in seen:
            seen.add(s)
            out.append(Path(s))
    return out


def _prepend_ld_library_path(dirs: list[Path]) -> None:
    if not dirs:
        return
    parts = [str(d) for d in dirs]
    cur = os.environ.get("LD_LIBRARY_PATH", "").strip()
    if cur:
        parts.append(cur)
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(parts)


def _find_lib(dirs: list[Path], name: str) -> Path | None:
    """精确 soname，否则 glob 版本后缀（libcublasLt.so.12.4.5.1）。"""
    extras: tuple[str, ...] = ()
    if name == "libcufft.so.11":
        extras = ("libcufft.so.12",)
    for d in dirs:
        p = d / name
        if p.is_file():
            return p
        for g in sorted(d.glob(name + "*")):
            if g.is_file():
                return g
        for extra in extras:
            p = d / extra
            if p.is_file():
                return p
            for g in sorted(d.glob(extra + "*")):
                if g.is_file():
                    return g
    return None


def _soname_alias_dir(dirs: list[Path]) -> Path | None:
    """ORT 常 dlopen libcufft.so.11；CUDA12 wheel 只有 libcufft.so.12* 时做同名软链。"""
    root = Path(__file__).resolve().parents[1] / ".runtime" / "cuda_soname"
    want = {
        "libcufft.so.11": ("libcufft.so.11", "libcufft.so.12"),
        "libcublasLt.so.12": ("libcublasLt.so.12",),
    }
    made = False
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    for soname, stems in want.items():
        dest = root / soname
        if dest.is_file() or dest.is_symlink():
            made = True
            continue
        src = None
        for stem in stems:
            src = _find_lib(dirs, stem)
            if src is not None:
                break
        if src is None:
            continue
        try:
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            dest.symlink_to(src.resolve())
            made = True
        except OSError:
            continue
    return root if made else None


def _preload(dirs: list[Path]) -> list[str]:
    loaded: list[str] = []
    seen: set[str] = set()
    for name in _PRELOAD_NAMES:
        lib = _find_lib(dirs, name)
        if lib is None:
            continue
        key = str(lib.resolve())
        if key in seen:
            continue
        try:
            ctypes.CDLL(key, mode=ctypes.RTLD_GLOBAL)
            loaded.append(key)
            seen.add(key)
        except OSError:
            continue
    return loaded


def _missing_critical(dirs: list[Path]) -> list[str]:
    """检查 ORT 最常缺的几个库是否存在于候选目录。"""
    need = ("libcublasLt.so.12", "libcufft.so.11", "libcudnn.so.9")
    miss = []
    for name in need:
        if _find_lib(dirs, name) is not None:
            continue
        if name.startswith("libcudnn") and any(
            list(d.glob("libcudnn.so.9*")) for d in dirs
        ):
            continue
        miss.append(name)
    return miss


def ensure_cuda_libs(*, verbose: bool = True) -> dict:
    """在首次 import onnxruntime 前调用。"""
    _fix_omp_threads()
    dirs = _candidate_lib_dirs()
    alias = _soname_alias_dir(dirs)
    if alias is not None:
        dirs = [alias, *dirs]
    _prepend_ld_library_path(dirs)
    loaded = _preload(dirs)
    missing = _missing_critical(dirs)
    info = {
        "lib_dirs": [str(d) for d in dirs],
        "preloaded": loaded,
        "missing_critical": missing,
        "LD_LIBRARY_PATH_head": os.environ.get("LD_LIBRARY_PATH", "")[:400],
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "executable": sys.executable,
    }
    if verbose:
        print(
            f"[cuda_libs] dirs={len(dirs)} preloaded={len(loaded)} "
            f"missing={missing or 'none'} OMP={info['OMP_NUM_THREADS']}",
            flush=True,
        )
        if missing:
            print(
                "[cuda_libs][WARN] 缺少: "
                + ", ".join(missing)
                + "\n  请在当前 PYTHON（不要 pip -U nvidia-*）:\n"
                f"  {sys.executable} -m pip install -r requirements-ort.txt",
                flush=True,
            )
    return info


def ort_cuda_ok() -> bool:
    ensure_cuda_libs(verbose=False)
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False

#!/usr/bin/env python3
"""ClearVoice MossFormer2 薄封装（仅本地权重，禁止运行期下载）。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from local_models import assert_runtime_models
from paths import setup_sys_path

setup_sys_path()

_PROBE = (
    "import importlib.util as u\n"
    "miss=[]\n"
    "miss.append('clearvoice') if u.find_spec('clearvoice') is None else None\n"
    "miss.append('torch') if u.find_spec('torch') is None else None\n"
    "if miss:\n"
    "    raise SystemExit('missing:' + ','.join([m for m in miss if m]))\n"
    "import clearvoice, torch\n"
    "print('ok', torch.__version__)\n"
)


def _probe(py: str) -> tuple[bool, str]:
    p = Path(py).expanduser()
    if not p.exists():
        return False, "path_missing"
    # conda 的 python 常为 symlink；resolve 后仍须可执行
    try:
        r = subprocess.run(
            [str(p), "-c", _PROBE],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS") or "8"},
        )
    except Exception as e:
        return False, f"exec_error:{e}"
    if r.returncode == 0:
        return True, (r.stdout or "").strip()
    err = ((r.stderr or "") + "\n" + (r.stdout or "")).strip()
    # 取最后一行关键信息
    last = [ln for ln in err.splitlines() if ln.strip()]
    return False, (last[-1] if last else f"exit={r.returncode}")[:200]


def _resolve_clearvoice_python() -> str | None:
    """找到能同时 import clearvoice 与 torch 的解释器。"""
    cands: list[str] = []
    env = os.environ.get("CLEARVOICE_PYTHON", "").strip()
    if env:
        cands.append(env)
    cands.append(sys.executable)
    cands += [
        "/root/miniconda3/envs/qwen3-asr/bin/python",
        "/root/autodl-tmp/envs/qwen3-asr/bin/python",
        "/root/miniconda3/envs/ClearerVoice-Studio/bin/python",
        "/root/autodl-tmp/envs/ClearerVoice-Studio/bin/python",
        "/root/miniconda3/envs/ClearerVoice-Studio/bin/python3",
        "/root/miniconda3/envs/clearvoice/bin/python",
        "/root/autodl-tmp/envs/clearvoice/bin/python",
        "/root/miniconda3/envs/ClearVoice/bin/python",
    ]
    seen: set[str] = set()
    for c in cands:
        if not c or c in seen:
            continue
        seen.add(c)
        ok, detail = _probe(c)
        mark = "OK" if ok else "no"
        print(f"[INFO] probe clearvoice+torch [{mark}] {c} → {detail}", flush=True)
        if ok:
            return str(Path(c).expanduser().resolve())
    return None


def create_cv_separator(peak: float = 0.7, device: str = "cuda:0"):
    locs = assert_runtime_models(need_onnx=False, need_cv=True, need_asr=False)
    print(f"[INFO] local ClearVoice ckpt → {locs['cv_ckpt']} (offline)")

    py = _resolve_clearvoice_python()
    if py:
        os.environ["CLEARVOICE_PYTHON"] = py
        print(f"[INFO] CLEARVOICE_PYTHON={py}", flush=True)
    else:
        os.environ.pop("CLEARVOICE_PYTHON", None)
        print(
            "[WARN] 未找到同时具备 clearvoice+torch 的 Python。\n"
            "  在同一环境里必须两者都有，例如:\n"
            "    conda activate qwen3-asr && pip install clearvoice\n"
            "    # 或\n"
            "    conda activate ClearerVoice-Studio && pip install clearvoice torch torchaudio\n"
            "    export CLEARVOICE_PYTHON=$CONDA_PREFIX/bin/python\n"
            "将尝试当前环境 in-process …",
            flush=True,
        )

    from mossformer2_ss import MossFormer2Separator

    sep = MossFormer2Separator(
        model_name="MossFormer2_SS_16K", peak=peak, device=device
    )
    msg = sep.status_message()
    print("[INFO]", msg)
    if "failed" in (msg or "").lower():
        raise RuntimeError(
            f"ClearVoice 不可用: {msg}\n"
            "同一 Python 需要: import clearvoice 且 import torch\n"
            "  conda activate qwen3-asr && pip install clearvoice\n"
            "  export CLEARVOICE_PYTHON=$CONDA_PREFIX/bin/python"
        )
    return sep

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
    """只用 VE 解释器（CLEARVOICE_PYTHON == PYTHON_BIN）。不探测独立 ClearerVoice-Studio。"""
    env_name = os.environ.get("VM_CONDA_ENV", "ve").strip() or "ve"
    cands: list[str] = []
    for key in ("CLEARVOICE_PYTHON", "PYTHON_BIN"):
        v = os.environ.get(key, "").strip()
        if v:
            cands.append(v)
    cands.append(sys.executable)
    cands += [
        f"/root/miniconda3/envs/{env_name}/bin/python",
        f"/root/anaconda3/envs/{env_name}/bin/python",
        "/root/miniconda3/envs/ve/bin/python",
        "/root/anaconda3/envs/ve/bin/python",
        "/root/autodl-tmp/envs/ve/bin/python",
        # 历史 VE 名，仅回退
        "/root/miniconda3/envs/qwen3-asr/bin/python",
        "/root/autodl-tmp/envs/qwen3-asr/bin/python",
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
            "[WARN] VE python 不能同时 import clearvoice+torch。\n"
            "  conda activate ve && pip install -r requirements-optional.txt\n"
            "  或: ./setup_env.sh && source ./env.sh\n"
            "将尝试当前解释器 in-process …",
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
            "同一 VE python 需要: import clearvoice 且 import torch\n"
            "  conda activate ve && pip install -r requirements-optional.txt\n"
            "  或: ./setup_env.sh"
        )
    return sep

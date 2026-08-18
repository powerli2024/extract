#!/usr/bin/env bash
# 安装/搭建环境（会 conda create + pip）。检查请用 ./check_env.sh
# 默认创建独立环境 qwen3-asr，并安装:
#   torch / torchaudio / onnxruntime-gpu / qwen-asr / 轻量依赖
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo " VM setup_env (install)"
echo " ROOT=$ROOT"
echo "============================================"

if [[ -f "${CLEARVOICE_ENV_FILE:-}" ]]; then
  # shellcheck disable=SC1090
  source "$CLEARVOICE_ENV_FILE" || true
elif [[ -f /root/VB/.env_clearvoice ]]; then
  # shellcheck disable=SC1091
  source /root/VB/.env_clearvoice || true
fi

export CLEARVOICE_ROOT="${CLEARVOICE_ROOT:-/root/ClearerVoice-Studio}"
# 仅当文件存在时才预设；否则留给运行时探测 / in-process
if [[ -z "${CLEARVOICE_PYTHON:-}" ]]; then
  for c in \
    /root/miniconda3/envs/ClearerVoice-Studio/bin/python \
    /root/miniconda3/envs/clearvoice/bin/python
  do
    if [[ -x "$c" ]]; then export CLEARVOICE_PYTHON="$c"; break; fi
  done
fi
if [[ -d /root/autodl-tmp ]]; then
  export VM_OUT="${VM_OUT:-/root/autodl-tmp/vm}"
  export MOSS_CKPT_DIR="${MOSS_CKPT_DIR:-/root/autodl-tmp/checkpoints}"
  export DATA_DIR="${DATA_DIR:-/root/autodl-tmp/datasetA}"
  if [[ -z "${ASR_MODEL_DIR:-}" ]]; then
    if [[ -d /root/autodl-tmp/Qwen3-ASR-1.7B ]]; then
      export ASR_MODEL_DIR=/root/autodl-tmp/Qwen3-ASR-1.7B
    elif [[ -d /root/Qwen3-ASR-1.7B ]]; then
      export ASR_MODEL_DIR=/root/Qwen3-ASR-1.7B
    else
      export ASR_MODEL_DIR=/root/autodl-tmp/Qwen3-ASR-1.7B
    fi
  fi
  export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
  export TORCH_HOME="${TORCH_HOME:-/root/autodl-tmp/cache/torch}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/cache/pip}"
  mkdir -p "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR"
else
  export VM_OUT="${VM_OUT:-$ROOT/../vm_out}"
  export MOSS_CKPT_DIR="${MOSS_CKPT_DIR:-/root/checkpoints}"
  export DATA_DIR="${DATA_DIR:-/root/datasetA}"
  export ASR_MODEL_DIR="${ASR_MODEL_DIR:-/root/Qwen3-ASR-1.7B}"
fi

ENV_NAME="${VM_CONDA_ENV:-qwen3-asr}"
PY_VER="${VM_PYTHON_VERSION:-3.12}"
# CUDA wheel 标签：可 export TORCH_CUDA=cu124 / cu121 / cpu
TORCH_CUDA="${TORCH_CUDA:-}"
PIP_MIRROR=(-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn)

CONDA_BIN="${CONDA_BIN:-}"
if [[ -z "$CONDA_BIN" ]]; then
  for c in \
    /root/miniconda3/bin/conda \
    /root/anaconda3/bin/conda \
    "$(command -v conda || true)"
  do
    [[ -n "$c" && -x "$c" ]] && CONDA_BIN="$c" && break
  done
fi

resolve_env_python() {
  local name="$1"
  for c in \
    "/root/miniconda3/envs/${name}/bin/python" \
    "/root/anaconda3/envs/${name}/bin/python"
  do
    [[ -x "$c" ]] && { echo "$c"; return 0; }
  done
  if [[ -n "$CONDA_BIN" ]]; then
    local p
    p="$("$CONDA_BIN" run -n "$name" which python 2>/dev/null || true)"
    [[ -n "$p" && -x "$p" ]] && { echo "$p"; return 0; }
  fi
  return 1
}

detect_torch_cuda_tag() {
  if [[ -n "$TORCH_CUDA" ]]; then
    echo "$TORCH_CUDA"
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "cpu"
    return 0
  fi
  # e.g. "CUDA Version: 12.4"
  local ver
  ver="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)"
  if [[ -z "$ver" ]]; then
    echo "cu124"
    return 0
  fi
  local major minor
  major="${ver%%.*}"
  minor="${ver#*.}"
  minor="${minor%%.*}"
  if [[ "$major" -gt 12 ]] || { [[ "$major" -eq 12 ]] && [[ "${minor:-0}" -ge 4 ]]; }; then
    echo "cu124"
  elif [[ "$major" -eq 12 ]]; then
    echo "cu121"
  elif [[ "$major" -eq 11 ]]; then
    echo "cu118"
  else
    echo "cu124"
  fi
}

need_reinstall_torch() {
  # 缺包 / CPU-only 但机器有 GPU → 需要重装
  "$PYTHON_BIN" - <<'PY'
import sys
try:
    import torch
except Exception:
    sys.exit(1)
try:
    import torchaudio  # noqa: F401
except Exception:
    sys.exit(1)
has_gpu = False
try:
    import subprocess
    has_gpu = subprocess.call(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
except Exception:
    pass
if has_gpu and not torch.cuda.is_available():
    sys.exit(2)
sys.exit(0)
PY
}

install_torch_stack() {
  local tag="$1"
  echo "[INFO] 安装 torch / torchaudio （tag=$tag）..."
  if [[ "$tag" == "cpu" ]]; then
    "$PYTHON_BIN" -m pip install -U torch torchaudio "${PIP_MIRROR[@]}" \
      || "$PYTHON_BIN" -m pip install -U torch torchaudio
    return
  fi
  local indexes=(
    "https://mirrors.aliyun.com/pytorch-wheels/${tag}"
    "https://download.pytorch.org/whl/${tag}"
  )
  local ok=0
  local idx
  for idx in "${indexes[@]}"; do
    echo "[INFO] try: pip install torch torchaudio --index-url $idx"
    if "$PYTHON_BIN" -m pip install -U torch torchaudio --index-url "$idx"; then
      ok=1
      break
    fi
    echo "[WARN] 来源失败: $idx"
  done
  if [[ "$ok" -ne 1 ]]; then
    echo "[WARN] 专用 CUDA wheel 失败，回退清华源（可能是 CPU 版）..."
    "$PYTHON_BIN" -m pip install -U torch torchaudio "${PIP_MIRROR[@]}" || true
  fi
}

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if p="$(resolve_env_python "$ENV_NAME" 2>/dev/null)"; then
    PYTHON_BIN="$p"
    echo "[INFO] 复用已有环境: $ENV_NAME -> $PYTHON_BIN"
  elif [[ -n "$CONDA_BIN" ]]; then
    echo "[INFO] 未找到 conda 环境 '$ENV_NAME'，正在创建 (python=$PY_VER) ..."
    "$CONDA_BIN" create -n "$ENV_NAME" "python=$PY_VER" -y
    PYTHON_BIN="$(resolve_env_python "$ENV_NAME")"
    echo "[INFO] 已创建: $PYTHON_BIN"
  else
    echo "[WARN] 未找到 conda，回退到当前 python3（不推荐）"
    PYTHON_BIN="$(command -v python3 || command -v python)"
  fi
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
echo "[INFO] PYTHON_BIN=$PYTHON_BIN"

echo "[INFO] pip upgrade ..."
"$PYTHON_BIN" -m pip install -U pip setuptools wheel "${PIP_MIRROR[@]}" >/dev/null || true

# 1) torch / torchaudio（ASR + 通用 CUDA）
TORCH_TAG="$(detect_torch_cuda_tag)"
echo "[INFO] TORCH_CUDA tag=$TORCH_TAG (可用 TORCH_CUDA=cu124 覆盖)"
set +e
need_reinstall_torch
trc=$?
set -e
if [[ "$trc" -ne 0 ]]; then
  install_torch_stack "$TORCH_TAG"
else
  echo "[ OK ] torch / torchaudio 已可用"
fi

# 1.5) pip 自带的 CUDA 运行库（给 onnxruntime-gpu 找 libcublasLt.so.12）
if [[ "$TORCH_TAG" != "cpu" ]]; then
  echo "[INFO] 安装/补齐 nvidia CUDA pip 库（ORT 需要 cublas/cufft/cudnn）..."
  "$PYTHON_BIN" -m pip install -U \
    nvidia-cublas-cu12 \
    nvidia-cufft-cu12 \
    nvidia-cudnn-cu12 \
    nvidia-cuda-runtime-cu12 \
    nvidia-cuda-nvrtc-cu12 \
    nvidia-nvjitlink-cu12 \
    nvidia-curand-cu12 \
    nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 \
    "${PIP_MIRROR[@]}" || echo "[WARN] nvidia-*-cu12 部分安装失败，请检查网络后重试"
fi

# 2) onnxruntime-gpu（ONNX 分离）
echo "[INFO] 安装 onnxruntime-gpu ..."
if "$PYTHON_BIN" -c "import onnxruntime as ort; print(ort.get_available_providers())" 2>/dev/null | grep -qi CUDA; then
  echo "[ OK ] onnxruntime 已含 CUDAExecutionProvider"
else
  # 先卸 CPU 包避免冲突
  "$PYTHON_BIN" -m pip uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
  if ! "$PYTHON_BIN" -m pip install -U onnxruntime-gpu "${PIP_MIRROR[@]}"; then
    echo "[WARN] onnxruntime-gpu 失败，尝试官方源..."
    "$PYTHON_BIN" -m pip install -U onnxruntime-gpu || \
      "$PYTHON_BIN" -m pip install -U onnxruntime "${PIP_MIRROR[@]}" || \
      echo "[ERR] onnxruntime 安装失败"
  fi
fi

# 验证 ORT 在补齐 LD_LIBRARY_PATH 后能否真正用 CUDA
echo "[INFO] 验证 ORT CUDA（含 torch nvidia lib 路径）..."
"$PYTHON_BIN" - <<'PY' || echo "[WARN] ORT CUDA 验证未通过，运行时会尝试再注入路径"
import os, sys
from pathlib import Path
try:
    import torch
    site = Path(torch.__file__).resolve().parent.parent
    dirs=[]
    for sub in ("nvidia/cublas/lib","nvidia/cuda_runtime/lib","nvidia/cudnn/lib","nvidia/nvjitlink/lib"):
        p=site/sub
        if p.is_dir(): dirs.append(str(p))
    if dirs:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(dirs + [os.environ.get("LD_LIBRARY_PATH","")])
except Exception as e:
    print("torch path prep fail", e)
import onnxruntime as ort
print("providers=", ort.get_available_providers())
assert "CUDAExecutionProvider" in ort.get_available_providers(), "no CUDA EP"
print("ORT CUDA OK")
PY

# 3) 本仓库 requirements.txt（音频 / CER / qwen-asr 及其传递依赖）
REQ_FILE="$ROOT/requirements.txt"
echo "[INFO] pip: $REQ_FILE ..."
if [[ -f "$REQ_FILE" ]]; then
  "$PYTHON_BIN" -m pip install -U -r "$REQ_FILE" "${PIP_MIRROR[@]}" \
    || "$PYTHON_BIN" -m pip install -U -r "$REQ_FILE"
else
  echo "[WARN] 缺少 $REQ_FILE，回退逐包安装"
  "$PYTHON_BIN" -m pip install -U \
    numpy scipy soxr soundfile librosa audioread \
    editdistance pypinyin tqdm packaging \
    huggingface_hub sentencepiece protobuf safetensors einops \
    "${PIP_MIRROR[@]}"
fi

# 4) qwen-asr（requirements 已含；此处再确认，失败则单独重试）
echo "[INFO] 确认 qwen-asr ..."
if "$PYTHON_BIN" -c "import qwen_asr" 2>/dev/null; then
  echo "[ OK ] qwen_asr 已可 import"
else
  "$PYTHON_BIN" -m pip install -U "qwen-asr" "${PIP_MIRROR[@]}" \
    || "$PYTHON_BIN" -m pip install -U "qwen-asr"
fi

mkdir -p "$VM_OUT/pos" "$VM_OUT/neg" "$VM_OUT/reports" "$VM_OUT/packs" "$MOSS_CKPT_DIR"

mkdir -p "$ROOT/.runtime"
echo "$PYTHON_BIN" >"$ROOT/.runtime/python_bin"
cat >"$ROOT/.runtime/env.sh" <<EOF
# auto-generated by setup_env.sh — also: source $ROOT/env.sh
export PYTHON_BIN="$PYTHON_BIN"
export VM_CONDA_ENV="$ENV_NAME"
export DATA_DIR="\${DATA_DIR:-$DATA_DIR}"
export VM_OUT="\${VM_OUT:-$VM_OUT}"
export ASR_MODEL_DIR="\${ASR_MODEL_DIR:-$ASR_MODEL_DIR}"
export MOSS_CKPT_DIR="\${MOSS_CKPT_DIR:-$MOSS_CKPT_DIR}"
export CLEARVOICE_ROOT="\${CLEARVOICE_ROOT:-$CLEARVOICE_ROOT}"
EOF
# 仅当探测到真实路径时写入，避免把不存在的默认写进 env.sh
if [[ -n "${CLEARVOICE_PYTHON:-}" && -x "${CLEARVOICE_PYTHON}" ]]; then
  echo "export CLEARVOICE_PYTHON=\"\${CLEARVOICE_PYTHON:-$CLEARVOICE_PYTHON}\"" >>"$ROOT/.runtime/env.sh"
fi
cat >>"$ROOT/.runtime/env.sh" <<EOF
export PATH="\$(dirname "$PYTHON_BIN"):\$PATH"
EOF
cp -f "$ROOT/.runtime/env.sh" "$ROOT/env.sh"

echo "[INFO] 写出环境快照 → $VM_OUT/meta/env_snapshot.txt"
mkdir -p "$VM_OUT/meta"
{
  echo "PYTHON_BIN=$PYTHON_BIN"
  echo "VM_CONDA_ENV=$ENV_NAME"
  echo "TORCH_CUDA=$TORCH_TAG"
  echo "DATA_DIR=$DATA_DIR"
  echo "VM_OUT=$VM_OUT"
  echo "CLEARVOICE_ROOT=$CLEARVOICE_ROOT"
  echo "CLEARVOICE_PYTHON=$CLEARVOICE_PYTHON"
  echo "MOSS_CKPT_DIR=$MOSS_CKPT_DIR"
  echo "ASR_MODEL_DIR=$ASR_MODEL_DIR"
  echo "MOSS_ONNX_PATH=${MOSS_ONNX_PATH:-}"
  "$PYTHON_BIN" - <<'PY' || true
import importlib
mods = (
    "torch", "torchaudio", "onnxruntime", "qwen_asr",
    "editdistance", "pypinyin", "soundfile", "numpy", "scipy",
    "librosa", "soxr", "tqdm", "huggingface_hub",
)
for m in mods:
    try:
        mod = importlib.import_module(m)
        extra = ""
        if m == "torch":
            import torch
            extra = f" cuda={torch.cuda.is_available()} ver={torch.__version__}"
        if m == "onnxruntime":
            import onnxruntime as ort
            extra = f" providers={ort.get_available_providers()}"
        print(f"import {m}=OK{extra}")
    except Exception as e:
        print(f"import {m}=FAIL {e}")
PY
  date -Iseconds 2>/dev/null || date
} >"$VM_OUT/meta/env_snapshot.txt"

echo ""
echo "===== 关键结果摘要 ====="
"$PYTHON_BIN" - <<'PY' || true
import importlib
checks = {
    "torch": "torch",
    "torchaudio": "torchaudio",
    "onnxruntime": "onnxruntime",
    "qwen_asr": "qwen_asr",
}
for name, modn in checks.items():
    try:
        importlib.import_module(modn)
        print(f"[ OK ] {name}")
    except Exception as e:
        print(f"[ERR ] {name}: {e}")
try:
    import torch
    print(f"       torch={torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"       gpu={torch.cuda.get_device_name(0)}")
except Exception:
    pass
try:
    import onnxruntime as ort
    print(f"       ort providers={ort.get_available_providers()}")
except Exception:
    pass
PY

echo ""
echo "setup 完成。PYTHON_BIN=$PYTHON_BIN"
echo "接下来:"
echo "  1) ./download_models.sh"
echo "  2) ./check_env.sh"
echo "  3) source ./env.sh && ./run_all.sh --limit 20"
echo ""
echo "手动覆盖 CUDA wheel 示例:"
echo "  TORCH_CUDA=cu124 ./setup_env.sh"
echo "  TORCH_CUDA=cu121 ./setup_env.sh"

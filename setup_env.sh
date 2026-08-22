#!/usr/bin/env bash
# 安装/搭建环境（conda create + pip）。检查请用 ./check_env.sh
# extract@sep：全部装进 VE 环境（默认 conda 名 ve），含 torch / ORT / qwen-asr / clearvoice。
# 不新建 qwen3-asr，也不用独立 ClearerVoice-Studio。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/paths_defaults.sh"
EXTRACT_PICK_DEFER=1
# shellcheck disable=SC1091
source "$ROOT/pick_python.sh"
if [[ -f "$ROOT/ve/.env_ve" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/ve/.env_ve" || true
fi

echo "============================================"
echo " VM setup_env (install into VE)"
echo " ROOT=$ROOT"
echo "============================================"

export CLEARVOICE_ROOT="${CLEARVOICE_ROOT:-}"
mkdir -p "${HF_HOME:-/tmp}" "${TORCH_HOME:-/tmp}" "${PIP_CACHE_DIR:-/tmp}" 2>/dev/null || true

ENV_NAME="${VM_CONDA_ENV:-ve}"
PY_VER="${VM_PYTHON_VERSION:-3.12}"
# CUDA wheel 标签：可 export TORCH_CUDA=cu124 / cu121 / cpu
TORCH_CUDA="${TORCH_CUDA:-}"
TUNA_INDEX="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
TUNA_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
PIP_MIRROR=(-i "$TUNA_INDEX" --trusted-host "$TUNA_HOST")
PIP_CONF="$ROOT/pip.conf"

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

maybe_network_turbo() {
  # AutoDL 学术加速：仅官方 pytorch/HF 回退时需要
  if [[ -f /etc/network_turbo ]]; then
    # shellcheck disable=SC1091
    source /etc/network_turbo || true
    echo "[INFO] 已 source /etc/network_turbo"
  fi
}

apply_pip_tuna() {
  # 全局默认清华源：环境变量 + PIP_CONFIG_FILE，避免漏掉 -i 的 pip 走到官方
  export PIP_INDEX_URL="$TUNA_INDEX"
  export PIP_TRUSTED_HOST="$TUNA_HOST"
  export PIP_ROOT_USER_ACTION="${PIP_ROOT_USER_ACTION:-ignore}"
  export PIP_DISABLE_PIP_VERSION_CHECK=1
  if [[ -f "$PIP_CONF" ]]; then
    export PIP_CONFIG_FILE="$PIP_CONF"
  fi
  echo "[INFO] pip 默认源 = 清华 $TUNA_INDEX"
}

vm_pip() {
  "$PYTHON_BIN" -m pip install "${PIP_MIRROR[@]}" "$@"
}

torch_cuda_ok() {
  "$PYTHON_BIN" - <<'PY'
import torch, torchaudio  # noqa: F401
import os, sys
want_gpu = os.environ.get("_VM_WANT_GPU", "1") == "1"
if want_gpu and not torch.cuda.is_available():
    sys.exit(2)
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
PY
}

install_torch_stack() {
  local tag="$1"
  echo "[INFO] 安装 torch / torchaudio （tag=$tag；优先清华 PyPI）..."
  export PIP_ROOT_USER_ACTION="${PIP_ROOT_USER_ACTION:-ignore}"
  export _VM_WANT_GPU=1

  echo "[INFO] [1/2] 清华源 pip install torch torchaudio"
  if vm_pip torch torchaudio; then
    if [[ "$tag" == "cpu" ]] || torch_cuda_ok; then
      echo "[ OK ] 清华源 torch 可用"
      return
    fi
    echo "[WARN] 清华源装上了 torch，但 CUDA 不可用，将卸掉后改用官方 ${tag} wheel"
    "$PYTHON_BIN" -m pip uninstall -y torch torchaudio torchvision >/dev/null 2>&1 || true
  else
    echo "[WARN] 清华源安装 torch 失败，尝试官方 CUDA 索引"
  fi

  if [[ "$tag" == "cpu" ]]; then
    return
  fi

  # 官方 CUDA 索引会替换 PyPI；只作为清华无 GPU wheel 时的回退
  local idx="${TORCH_INDEX:-https://download.pytorch.org/whl/${tag}}"
  maybe_network_turbo
  echo "[INFO] [2/2] 回退官方 CUDA 索引: --index-url $idx"
  if "$PYTHON_BIN" -m pip install torch torchaudio --index-url "$idx"; then
    if torch_cuda_ok; then
      echo "[ OK ] 官方 ${tag} torch 可用"
      return
    fi
  fi
  echo "[ERR] torch CUDA 安装失败。可: source /etc/network_turbo && TORCH_INDEX=$idx ./setup_env.sh"
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

apply_pip_tuna
echo "[INFO] pip upgrade ..."
vm_pip -U pip setuptools wheel >/dev/null || true

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

# 1.5+2) onnxruntime-gpu + nvidia CUDA pip 库（ORT 找 libcublasLt.so.12）
# 禁止 pip -U nvidia-*：torch 2.6+cu124 钉死 12.4.127，-U 会升到 12.6
ORT_REQ="$ROOT/requirements-ort.txt"
echo "[INFO] pip: $ORT_REQ → $PYTHON_BIN"
if [[ "$TORCH_TAG" == "cpu" ]]; then
  echo "[WARN] CPU 机器：装 onnxruntime（无 GPU EP）"
  vm_pip onnxruntime || "$PYTHON_BIN" -m pip install onnxruntime
else
  if "$PYTHON_BIN" -c "import onnxruntime as ort; print(ort.get_available_providers())" 2>/dev/null | grep -qi CUDA; then
    echo "[ OK ] onnxruntime 已含 CUDAExecutionProvider"
  else
    "$PYTHON_BIN" -m pip uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
    if [[ -f "$ORT_REQ" ]]; then
      vm_pip -r "$ORT_REQ" || "$PYTHON_BIN" -m pip install -r "$ORT_REQ"
    else
      vm_pip onnxruntime-gpu || "$PYTHON_BIN" -m pip install onnxruntime-gpu
    fi
  fi
fi

echo "[INFO] 硬检查: $PYTHON_BIN -c 'import onnxruntime'"
if ! "$PYTHON_BIN" -c "import onnxruntime as ort; print('ort', getattr(ort,'__version__', '?'), ort.get_available_providers())"; then
  echo "[ERR] $PYTHON_BIN 没有 onnxruntime。"
  echo "  不要只 pip install -r requirements.txt（里面没有 ORT）。"
  echo "  请: $PYTHON_BIN -m pip install -r $ORT_REQ"
  echo "  或重新: ./setup_env.sh"
  exit 1
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
    for sub in ("nvidia/cublas/lib","nvidia/cuda_runtime/lib","nvidia/cudnn/lib","nvidia/nvjitlink/lib","nvidia/cufft/lib"):
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
  vm_pip -U -r "$REQ_FILE" \
    || "$PYTHON_BIN" -m pip install -U -r "$REQ_FILE"
else
  echo "[WARN] 缺少 $REQ_FILE，回退逐包安装"
  vm_pip -U \
    numpy scipy soxr soundfile librosa audioread \
    editdistance pypinyin tqdm packaging \
    huggingface_hub sentencepiece protobuf safetensors einops
fi

# 4) qwen-asr（requirements 已含；此处再确认，失败则单独重试）
echo "[INFO] 确认 qwen-asr ..."
if "$PYTHON_BIN" -c "import qwen_asr" 2>/dev/null; then
  echo "[ OK ] qwen_asr 已可 import"
else
  vm_pip -U "qwen-asr" \
    || "$PYTHON_BIN" -m pip install -U "qwen-asr"
fi

# 5) ClearVoice 装进同一 VE python（s2/s4/s5/s7/s8）
OPT_REQ="$ROOT/requirements-optional.txt"
echo "[INFO] pip: $OPT_REQ (clearvoice → VE) ..."
if [[ -f "$OPT_REQ" ]]; then
  vm_pip -r "$OPT_REQ" || "$PYTHON_BIN" -m pip install -r "$OPT_REQ" \
    || echo "[WARN] clearvoice 安装失败；s1/s3/s6 仍可跑，s2+ 需要: $PYTHON_BIN -m pip install clearvoice"
fi
if "$PYTHON_BIN" -c "import clearvoice" 2>/dev/null; then
  echo "[ OK ] clearvoice 已在 VE python"
else
  echo "[WARN] VE python 尚不能 import clearvoice"
fi
export CLEARVOICE_PYTHON="$PYTHON_BIN"

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
export CLEARVOICE_ROOT="\${CLEARVOICE_ROOT:-}"
export CLEARVOICE_PYTHON="$PYTHON_BIN"
export PATH="\$(dirname "$PYTHON_BIN"):\$PATH"
export PIP_INDEX_URL="${TUNA_INDEX}"
export PIP_TRUSTED_HOST="${TUNA_HOST}"
export PIP_CONFIG_FILE="${PIP_CONF}"
export PIP_ROOT_USER_ACTION="ignore"
EOF
cp -f "$ROOT/.runtime/env.sh" "$ROOT/env.sh"

# 与 ve/.env_ve 对齐 PYTHON_BIN（不覆盖已有 VE 路径配置）
if [[ -d "$ROOT/ve" ]]; then
  if [[ ! -f "$ROOT/ve/.env_ve" ]]; then
    cat > "$ROOT/ve/.env_ve" <<VEEOF
export VE_ROOT="$ROOT/ve"
export PYTHON_BIN="$PYTHON_BIN"
export CLEARVOICE_PYTHON="$PYTHON_BIN"
export PATH="$(dirname "$PYTHON_BIN"):\$PATH"
export DATA_DIR="${DATA_DIR}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/root/autodl-tmp/cache/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/cache/pip}"
VEEOF
    echo "[OK] wrote $ROOT/ve/.env_ve"
  fi
fi

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
    "librosa", "soxr", "tqdm", "huggingface_hub", "clearvoice",
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

if ! "$PYTHON_BIN" -c "import onnxruntime" 2>/dev/null; then
  echo "[ERR] setup 结束时 $PYTHON_BIN 仍无 onnxruntime，已中止。"
  exit 1
fi

echo ""
echo "setup 完成。PYTHON_BIN=$PYTHON_BIN  （VE 环境，conda 名 $ENV_NAME）"
echo "ClearVoice 与 ASR 同一解释器: CLEARVOICE_PYTHON=$PYTHON_BIN"
echo "接下来:"
echo "  1) source ./env.sh          # 或: conda activate ve && source ve/.env_ve"
echo "  2) ./download_models.sh"
echo "  3) ./check_env.sh"
echo "  4) ./run_sep.sh --limit 20"
echo ""
echo "手动覆盖 CUDA wheel 示例:"
echo "  TORCH_CUDA=cu124 ./setup_env.sh"
echo "  TORCH_CUDA=cu121 ./setup_env.sh"

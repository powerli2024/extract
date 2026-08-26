#!/usr/bin/env bash
# VE 环境安装：全部进 conda env ve（与 Presence / mix ASR 同一 PYTHON_BIN）。
# 不新建 qwen3-asr，也不用独立 ClearerVoice-Studio。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "============================================"
echo " VE setup_env (conda env ve)"
echo " ROOT=$ROOT"
echo "============================================"

if [[ -d /root/autodl-tmp ]]; then
  export VE_OUT="${VE_OUT:-/root/autodl-tmp/ve}"
  export VE_MODEL_DIR="${VE_MODEL_DIR:-/root/autodl-tmp/ve_models}"
  export DATA_DIR="${DATA_DIR:-/root/autodl-tmp/datasetA}"
  export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
  export TORCH_HOME="${TORCH_HOME:-/root/autodl-tmp/cache/torch}"
  export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/root/autodl-tmp/cache/modelscope}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/cache/pip}"
  mkdir -p "$VE_OUT" "$VE_MODEL_DIR" "$HF_HOME" "$TORCH_HOME" "$MODELSCOPE_CACHE" "$PIP_CACHE_DIR"
else
  export VE_OUT="${VE_OUT:-$ROOT/../ve_out}"
  export VE_MODEL_DIR="${VE_MODEL_DIR:-$ROOT/../ve_models}"
  export DATA_DIR="${DATA_DIR:-$ROOT/../datasetA}"
fi

# 干净 KWS enroll：已设 BEST_SEP_DIR 则用；否则选第一个现成目录，仍可稍后覆盖
if [[ -z "${BEST_SEP_DIR:-}" ]]; then
  for c in \
    /root/autodl-tmp/pos_neg/best_sep \
    /root/autodl-tmp/kws_sep/best_sep \
    /root/autodl-tmp/best_sep \
    "$ROOT/../pos_neg/best_sep" \
    "$ROOT/../kws_sep/best_sep"
  do
    if [[ -f "$c/index.jsonl" || -d "$c/pos" ]]; then
      export BEST_SEP_DIR="$c"
      break
    fi
  done
fi

if [[ -z "${OMP_NUM_THREADS:-}" || ! "${OMP_NUM_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=4
fi
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

PIP_MIRROR=(-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn)
ENV_NAME="${VM_CONDA_ENV:-ve}"
PY_VER="${VM_PYTHON_VERSION:-3.12}"

if [[ -f /etc/network_turbo ]]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo || true
fi

if [[ -f "$ROOT/.env_ve" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.env_ve" || true
fi

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
    "/root/anaconda3/envs/${name}/bin/python" \
    "/root/autodl-tmp/envs/${name}/bin/python"
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

if [[ -z "${PYTHON_BIN:-}" || ! -x "${PYTHON_BIN:-}" ]]; then
  if p="$(resolve_env_python "$ENV_NAME" 2>/dev/null)"; then
    PYTHON_BIN="$p"
    echo "[INFO] 复用 conda env $ENV_NAME → $PYTHON_BIN"
  elif [[ -n "$CONDA_BIN" ]]; then
    echo "[INFO] 未找到 conda 环境 '$ENV_NAME'，正在创建 (python=$PY_VER) ..."
    "$CONDA_BIN" create -n "$ENV_NAME" "python=$PY_VER" -y
    PYTHON_BIN="$(resolve_env_python "$ENV_NAME")"
    echo "[INFO] 已创建: $PYTHON_BIN"
  fi
fi
export PYTHON_BIN
# shellcheck disable=SC1091
source "$ROOT/pick_python.sh"
echo "[INFO] Python=$PYTHON_BIN"
"$PYTHON_BIN" -c "import sys; print(sys.version)"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true

if [[ "${SKIP_TORCH:-}" != "1" ]] && ! "$PYTHON_BIN" -c "import torch" 2>/dev/null; then
  TORCH_CUDA="${TORCH_CUDA:-}"
  if [[ -z "$TORCH_CUDA" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    ver="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)"
    case "$ver" in
      12.4*|12.5*|12.6*|12.8*) TORCH_CUDA=cu124 ;;
      12.1*|12.2*|12.3*) TORCH_CUDA=cu121 ;;
      11.*) TORCH_CUDA=cu118 ;;
      *) TORCH_CUDA=cu124 ;;
    esac
  fi
  TORCH_CUDA="${TORCH_CUDA:-cpu}"
  echo "[INFO] 当前解释器无 torch，安装 TORCH_CUDA=$TORCH_CUDA"
  if [[ "$TORCH_CUDA" == "cpu" ]]; then
    "$PYTHON_BIN" -m pip install -U torch torchaudio "${PIP_MIRROR[@]}"
  else
    "$PYTHON_BIN" -m pip install -U torch torchaudio --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" || \
      "$PYTHON_BIN" -m pip install -U torch torchaudio "${PIP_MIRROR[@]}"
  fi
else
  echo "[INFO] 复用已有 torch"
  "$PYTHON_BIN" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
fi

"$PYTHON_BIN" -m pip install -U -r "$ROOT/requirements.txt" "${PIP_MIRROR[@]}"
ensure_modelscope "$PYTHON_BIN"
"$PYTHON_BIN" -m pip install -U huggingface_hub addict "${PIP_MIRROR[@]}"
if ! "$PYTHON_BIN" -c "import qwen_asr" 2>/dev/null; then
  echo "[INFO] 安装 qwen-asr → $PYTHON_BIN"
  "$PYTHON_BIN" -m pip install -U "qwen-asr" "${PIP_MIRROR[@]}" \
    || "$PYTHON_BIN" -m pip install -U "qwen-asr"
fi
if [[ "${INSTALL_WESPEAKER:-0}" == "1" ]]; then
  "$PYTHON_BIN" -c "import wespeaker" 2>/dev/null || \
    "$PYTHON_BIN" -m pip install -q "git+https://github.com/wenet-e2e/wespeaker.git" || \
    echo "[WARN] wespeaker 安装失败"
fi

# Presence USE_SEP / sep_route 需要 ORT。禁止 pip -U nvidia-*（会把 12.4 升到 12.6）
ORT_REQ="$ROOT/requirements_moss_ort.txt"
if [[ "${SKIP_MOSS_ORT:-0}" != "1" ]]; then
  if "$PYTHON_BIN" -c "import onnxruntime" 2>/dev/null; then
    echo "[INFO] 复用已有 onnxruntime"
  else
    echo "[INFO] pip: $ORT_REQ → $PYTHON_BIN（不要 -U nvidia-*）"
    "$PYTHON_BIN" -m pip install -r "$ORT_REQ" "${PIP_MIRROR[@]}" \
      || "$PYTHON_BIN" -m pip install -r "$ORT_REQ"
  fi
fi

hard_fail_import() {
  local mod="$1"
  if ! "$PYTHON_BIN" -c "import $mod" 2>/dev/null; then
    echo "[ERR] $PYTHON_BIN 不能 import $mod"
    echo "  请确认 conda activate ve 后重新 ./setup_env.sh"
    exit 1
  fi
  echo "[ OK ] import $mod"
}
hard_fail_import torch
hard_fail_import modelscope
hard_fail_import qwen_asr
hard_fail_import onnxruntime

export PYTHONPATH="$ROOT/scripts:${ROOT}/../scripts:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export MOSS_ONNX_PATH="${MOSS_ONNX_PATH:-/root/autodl-tmp/checkpoints/MossFormer2_ONNX/simple_model.onnx}"
export WESEP_ROOT="${WESEP_ROOT:-$VE_MODEL_DIR/wesep}"
export BEST_SEP_DIR="${BEST_SEP_DIR:-}"

if [[ ! -f "$ROOT/.env_ve" ]]; then
  cat > "$ROOT/.env_ve" <<EOF
export VE_ROOT="$ROOT"
export VE_OUT_BASE="${VE_OUT:-/root/autodl-tmp/ve}"
export PYTHON_BIN="$PYTHON_BIN"
export PATH="$(dirname "$PYTHON_BIN"):\$PATH"
export VE_MODEL_DIR="$VE_MODEL_DIR"
export DATA_DIR="$DATA_DIR"
export BEST_SEP_DIR="$BEST_SEP_DIR"
export HF_HOME="${HF_HOME:-}"
export TORCH_HOME="${TORCH_HOME:-}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-}"
export HF_ENDPOINT="\${HF_ENDPOINT:-https://hf-mirror.com}"
export OMP_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$MKL_NUM_THREADS"
export PYTHONPATH="$ROOT/scripts:${ROOT}/../scripts:\${PYTHONPATH:-}"
export PS4_WEIGHTS="\${PS4_WEIGHTS:-$VE_MODEL_DIR/PS4/checkpoint_epoch037.pt}"
export SPK_CHS_DIR="\${SPK_CHS_DIR:-$VE_MODEL_DIR/cnceleb_resnet34_LM}"
export ERES2NET_DIR="\${ERES2NET_DIR:-$VE_MODEL_DIR/eres2netv2_zh}"
export CAMPPLUS_DIR="\${CAMPPLUS_DIR:-$VE_MODEL_DIR/campplus_zh}"
export ECAPA_PRESENCE_DIR="\${ECAPA_PRESENCE_DIR:-$VE_MODEL_DIR/voxceleb_ecapa1024_LM}"
export VBLINK100_DIR="\${VBLINK100_DIR:-$VE_MODEL_DIR/vblink2_samresnet100}"
export WESEP_ROOT="\${WESEP_ROOT:-$VE_MODEL_DIR/wesep}"
export MOSS_ONNX_PATH="\${MOSS_ONNX_PATH:-/root/autodl-tmp/checkpoints/MossFormer2_ONNX/simple_model.onnx}"
export PIPELINE="\${PIPELINE:-mix}"
export ENROLL_VAD="\${ENROLL_VAD:-0}"
export LOCKED_THR="\${LOCKED_THR:-1}"
export EXTRA_REJECT="\${EXTRA_REJECT:-1}"
EOF
  echo "[OK] wrote $ROOT/.env_ve"
else
  echo "[INFO] 保留已有 $ROOT/.env_ve（不覆盖 PYTHON_BIN）"
fi

echo "source $ROOT/.env_ve"
echo "下一步:"
echo "  conda activate ve"
echo "  source $ROOT/.env_ve"
echo "  ONLY=eres2netv2 ./download_presence_encoders.sh"
echo "  ./download_moss_onnx.sh"
echo "  PIPELINE=mix ./check_env.sh"
echo "  ENROLL_VAD=0 PIPELINE=mix LOCKED_THR=1 EXTRA_REJECT=1 ./run_all.sh"
echo "  已有 ve_mix_novad: VE_OUT=/root/autodl-tmp/ve_mix_novad ./run_next_lift.sh submit"
echo "禁止 FORCE_CALIB 覆盖锁定 τ。"

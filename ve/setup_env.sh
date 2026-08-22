#!/usr/bin/env bash
# VE 环境安装（AutoDL：数据与模型默认 /root/autodl-tmp）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "============================================"
echo " VE setup_env"
echo " ROOT=$ROOT"
echo "============================================"

if [[ -d /root/autodl-tmp ]]; then
  export VE_OUT="${VE_OUT:-/root/autodl-tmp/ve}"
  export VE_MODEL_DIR="${VE_MODEL_DIR:-/root/autodl-tmp/ve_models}"
  export DATA_DIR="${DATA_DIR:-/root/autodl-tmp/datasetA}"
  export BEST_SEP_DIR="${BEST_SEP_DIR:-/root/autodl-tmp/pos_neg/best_sep}"
  export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
  export TORCH_HOME="${TORCH_HOME:-/root/autodl-tmp/cache/torch}"
  export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/root/autodl-tmp/cache/modelscope}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/cache/pip}"
  mkdir -p "$VE_OUT" "$VE_MODEL_DIR" "$HF_HOME" "$TORCH_HOME" "$MODELSCOPE_CACHE" "$PIP_CACHE_DIR"
else
  export VE_OUT="${VE_OUT:-$ROOT/../ve_out}"
  export VE_MODEL_DIR="${VE_MODEL_DIR:-$ROOT/../ve_models}"
  export DATA_DIR="${DATA_DIR:-$ROOT/../datasetA}"
  export BEST_SEP_DIR="${BEST_SEP_DIR:-$ROOT/../pos_neg/best_sep}"
fi

# 非法 OMP 会导致 libgomp 报错（AutoDL 常见）
if [[ -z "${OMP_NUM_THREADS:-}" || ! "${OMP_NUM_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=4
fi
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

PIP_MIRROR=(-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn)

# 网络：HF/git 可 turbo；pip 前必须 unset
if [[ -f /etc/network_turbo ]]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo || true
fi

# 与 extract 根共用 conda env ve（ve/.env_ve 的 PYTHON_BIN）；不要另开 qwen3-asr
# shellcheck disable=SC1091
source "$ROOT/pick_python.sh"
echo "[INFO] Python=$PYTHON_BIN"
"$PYTHON_BIN" -c "import sys; print(sys.version)"

# pip 不用代理
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true

# torch：ve 里通常已有，默认不重装（避免拆 ASR）
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
  echo "[INFO] 复用已有 torch（SKIP_TORCH 或已 import）"
  "$PYTHON_BIN" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
fi

"$PYTHON_BIN" -m pip install -U -r "$ROOT/requirements.txt" "${PIP_MIRROR[@]}"
ensure_modelscope "$PYTHON_BIN"
"$PYTHON_BIN" -m pip install -U huggingface_hub addict "${PIP_MIRROR[@]}"
# wespeaker 仅 ResNet 回退需要；锁定 eres 可不装
if [[ "${INSTALL_WESPEAKER:-0}" == "1" ]]; then
  "$PYTHON_BIN" -c "import wespeaker" 2>/dev/null || \
    "$PYTHON_BIN" -m pip install -q "git+https://github.com/wenet-e2e/wespeaker.git" || \
    echo "[WARN] wespeaker 安装失败"
fi

# MossFormer ORT：ve 常已有；没有才装
if [[ "${SKIP_MOSS_ORT:-}" != "1" ]] && "$PYTHON_BIN" -c "import onnxruntime" 2>/dev/null; then
  echo "[INFO] 复用已有 onnxruntime"
  SKIP_MOSS_ORT=1
fi
if [[ "${SKIP_MOSS_ORT:-0}" != "1" ]]; then
  echo "[INFO] 安装 onnxruntime-gpu + nvidia-*-cu12（ORT CUDA）..."
  "$PYTHON_BIN" -m pip install -U -r "$ROOT/requirements_moss_ort.txt" "${PIP_MIRROR[@]}" \
    || echo "[WARN] Moss ORT 依赖安装失败；可稍后: pip install -r requirements_moss_ort.txt"
else
  echo "[INFO] SKIP_MOSS_ORT=1，跳过 ORT/CUDA wheel"
fi

# PS4 优先 HF inference.py；VD/tools + VM/scripts（sep_route）可选
export PYTHONPATH="$ROOT/scripts:${PYTHONPATH:-}"
if [[ -d "$ROOT/../VD/tools" ]]; then
  export PYTHONPATH="$ROOT/../VD/tools:$PYTHONPATH"
fi
if [[ -d "$ROOT/../scripts" ]]; then
  export PYTHONPATH="$ROOT/../scripts:$PYTHONPATH"
fi
if [[ -d "$ROOT/../VM/scripts" ]]; then
  export PYTHONPATH="$ROOT/../VM/scripts:$PYTHONPATH"
fi
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export MOSS_ONNX_PATH="${MOSS_ONNX_PATH:-/root/autodl-tmp/checkpoints/MossFormer2_ONNX/simple_model.onnx}"
export WESEP_ROOT="${WESEP_ROOT:-$VE_MODEL_DIR/wesep}"

cat > "$ROOT/.env_ve" <<EOF
export VE_ROOT="$ROOT"
export VE_OUT_BASE="${VE_OUT:-/root/autodl-tmp/ve}"
# 勿写死 VE_OUT：run_all 按 PIPELINE 设为 ve_mix / ve_ps4 / …
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
export PYTHONPATH="$ROOT/scripts:${ROOT}/../scripts:${ROOT}/../VD/tools:${ROOT}/../VM/scripts:\${PYTHONPATH:-}"
export PS4_WEIGHTS="\${PS4_WEIGHTS:-$VE_MODEL_DIR/PS4/checkpoint_epoch037.pt}"
export SPK_CHS_DIR="\${SPK_CHS_DIR:-$VE_MODEL_DIR/cnceleb_resnet34_LM}"
export ERES2NET_DIR="\${ERES2NET_DIR:-$VE_MODEL_DIR/eres2netv2_zh}"
export WESEP_ROOT="\${WESEP_ROOT:-$VE_MODEL_DIR/wesep}"
export MOSS_ONNX_PATH="\${MOSS_ONNX_PATH:-/root/autodl-tmp/checkpoints/MossFormer2_ONNX/simple_model.onnx}"
EOF

echo "[OK] wrote $ROOT/.env_ve"
echo "source $ROOT/.env_ve"
echo "下一步:"
echo "  conda activate ve"
echo "  source $ROOT/.env_ve"
echo "  ONLY=eres2netv2 ./download_presence_encoders.sh"
echo "  ./download_moss_onnx.sh"
echo "  PIPELINE=mix ./check_env.sh"
echo "  ENROLL_VAD=0 PIPELINE=mix FORCE_CALIB=1 HOLDOUT_FRAC=0.3 ./run_all.sh"

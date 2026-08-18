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

ENV_NAME="${VE_CONDA_ENV:-ve}"
# 3.10+ 即可（含 str|Path 等）；不必 3.12。已有环境可跳过 create。
PY_VER="${VE_PYTHON_VERSION:-3.10}"
PIP_MIRROR=(-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn)

# 网络：HF/git 可 turbo；pip 前必须 unset
if [[ -f /etc/network_turbo ]]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo || true
fi

CONDA_BIN=""
for c in /root/miniconda3/bin/conda /root/anaconda3/bin/conda "$(command -v conda || true)"; do
  [[ -n "$c" && -x "$c" ]] && CONDA_BIN="$c" && break
done

if [[ -n "$CONDA_BIN" ]]; then
  if ! "$CONDA_BIN" env list | grep -qE "^${ENV_NAME}\\s"; then
    "$CONDA_BIN" create -n "$ENV_NAME" "python=${PY_VER}" -y
  fi
  # shellcheck disable=SC1091
  source "$("$CONDA_BIN" info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
fi

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
echo "[INFO] Python=$PYTHON_BIN"

# pip 不用代理
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true

# torch（按 CUDA）
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
echo "[INFO] TORCH_CUDA=$TORCH_CUDA"
if [[ "$TORCH_CUDA" == "cpu" ]]; then
  "$PYTHON_BIN" -m pip install -U torch torchaudio "${PIP_MIRROR[@]}"
else
  "$PYTHON_BIN" -m pip install -U torch torchaudio --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" || \
    "$PYTHON_BIN" -m pip install -U torch torchaudio "${PIP_MIRROR[@]}"
fi

"$PYTHON_BIN" -m pip install -U -r "$ROOT/requirements.txt" "${PIP_MIRROR[@]}"
"$PYTHON_BIN" -m pip install -U modelscope "${PIP_MIRROR[@]}" || echo "[WARN] modelscope 安装失败，将回退 ResNet34"
"$PYTHON_BIN" -m pip install -U huggingface_hub addict "${PIP_MIRROR[@]}"
# wespeaker（ResNet34 回退 / 质量参考）
"$PYTHON_BIN" -c "import wespeaker" 2>/dev/null || \
  "$PYTHON_BIN" -m pip install -q "git+https://github.com/wenet-e2e/wespeaker.git" || \
  echo "[WARN] wespeaker 安装失败"

# MossFormer ORT：默认装（可用 SKIP_MOSS_ORT=1 跳过，约 1.5GB+）
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
# 勿写死 VE_OUT：run_all 按 PIPELINE 设为 ve_ps4 / ve_wesep / ve_sep_route
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
echo "  ./download_models.sh && ./check_env.sh"
echo "  PIPELINE=ps4 ./run_all.sh"
echo "  ./download_wesep.sh && PIPELINE=wesep ./run_all.sh"
echo "  ./download_moss_onnx.sh && PIPELINE=sep_route ./run_all.sh"

#!/usr/bin/env bash
# 下载 MossFormer2 ONNX → /root/autodl-tmp/checkpoints/MossFormer2_ONNX/simple_model.onnx
# 供 PIPELINE=sep_route / USE_SEP=1
#
# 来源: ModelScope dengcunqin/speech_mossformer2_separation_temporal_16k
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$ROOT/.env_ve" ]] && source "$ROOT/.env_ve" || true

MODEL_NAME="MossFormer2_ONNX"
if [[ -d /root/autodl-tmp ]]; then
  OUT_DIR="${OUT_DIR:-/root/autodl-tmp/checkpoints/$MODEL_NAME}"
else
  OUT_DIR="${OUT_DIR:-$ROOT/../checkpoints/$MODEL_NAME}"
fi
MS_URL="${MS_URL:-https://www.modelscope.cn/models/dengcunqin/speech_mossformer2_separation_temporal_16k/resolve/master/simple_model.onnx}"

# 同仓 extract 根或旁路 VM 的下载脚本
_delegate=""
if [[ -x "$ROOT/../download_mossformer2_onnx.sh" ]]; then
  _delegate="$ROOT/../download_mossformer2_onnx.sh"
elif [[ -x "$ROOT/../VM/download_mossformer2_onnx.sh" ]]; then
  _delegate="$ROOT/../VM/download_mossformer2_onnx.sh"
fi
if [[ -n "$_delegate" ]]; then
  echo "[INFO] 委托 $_delegate OUT_DIR=$OUT_DIR"
  OUT_DIR="$OUT_DIR" "$_delegate"
  export MOSS_ONNX_PATH="$OUT_DIR/simple_model.onnx"
  echo "MOSS_ONNX_PATH=$MOSS_ONNX_PATH"
  exit 0
fi

file_size() {
  local f="$1"
  if [[ ! -f "$f" ]]; then echo 0; return; fi
  stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0
}

mkdir -p "$OUT_DIR"
ONNX="$OUT_DIR/simple_model.onnx"
TMP="$OUT_DIR/simple_model.onnx.partial"

if [[ -f "$ONNX" ]]; then
  sz="$(file_size "$ONNX")"
  if [[ "$sz" -gt 10485760 ]]; then
    echo "[OK] 已存在: $ONNX ($sz bytes)"
    echo "export MOSS_ONNX_PATH=$ONNX"
    exit 0
  fi
  echo "[WARN] 已有文件过小 ($sz)，重新下载"
fi

rm -f "$ONNX" "$TMP"
echo "[INFO] 下载: $MS_URL"
if command -v wget >/dev/null 2>&1; then
  wget -c --show-progress -O "$TMP" "$MS_URL"
elif command -v curl >/dev/null 2>&1; then
  curl -L --retry 5 --retry-delay 2 -C - -o "$TMP" "$MS_URL"
else
  echo "[ERR] 需要 wget 或 curl"; exit 1
fi

sz="$(file_size "$TMP")"
if [[ ! -f "$TMP" ]] || [[ "$sz" -lt 10485760 ]]; then
  echo "[ERR] 下载无效: $TMP ($sz bytes)"; rm -f "$TMP"; exit 1
fi
mv -f "$TMP" "$ONNX"
echo "[OK] $ONNX ($sz bytes)"

# onnxruntime-gpu + nvidia CUDA12 libs（ORT 找 libcublasLt.so.12 等）
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true
PIP_MIRROR=(-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn)
echo "[INFO] 安装/更新 requirements_moss_ort.txt ..."
"$PYTHON_BIN" -m pip install -U -r "$ROOT/requirements_moss_ort.txt" "${PIP_MIRROR[@]}" \
  || echo "[WARN] Moss ORT 依赖安装失败，请手动: pip install -r requirements_moss_ort.txt"

echo "export MOSS_ONNX_PATH=$ONNX"
echo "下一步: 确保 VM/scripts 在 PYTHONPATH，然后 LIMIT=128 ./compare_sep_reject.sh"

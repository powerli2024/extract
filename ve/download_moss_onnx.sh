#!/usr/bin/env bash
# 下载 MossFormer2 ONNX → /root/autodl-tmp/checkpoints/MossFormer2_ONNX/simple_model.onnx
# 供 PIPELINE=sep_route / Presence USE_SEP=1
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$ROOT/.env_ve" ]] && source "$ROOT/.env_ve" || true
# shellcheck disable=SC1091
source "$ROOT/pick_python.sh"

MODEL_NAME="MossFormer2_ONNX"
if [[ -d /root/autodl-tmp ]]; then
  OUT_DIR="${OUT_DIR:-/root/autodl-tmp/checkpoints/$MODEL_NAME}"
else
  OUT_DIR="${OUT_DIR:-$ROOT/../checkpoints/$MODEL_NAME}"
fi
MS_URL="${MS_URL:-https://www.modelscope.cn/models/dengcunqin/speech_mossformer2_separation_temporal_16k/resolve/master/simple_model.onnx}"

_delegate=""
if [[ -x "$ROOT/../download_mossformer2_onnx.sh" ]]; then
  _delegate="$ROOT/../download_mossformer2_onnx.sh"
fi
if [[ -n "$_delegate" ]]; then
  echo "[INFO] 委托 $_delegate OUT_DIR=$OUT_DIR"
  OUT_DIR="$OUT_DIR" "$_delegate"
else
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
    if [[ "$sz" -le 10485760 ]]; then
      echo "[WARN] 已有文件过小 ($sz)，重新下载"
      rm -f "$ONNX" "$TMP"
    fi
  fi
  if [[ ! -f "$ONNX" ]]; then
    echo "[INFO] 下载: $MS_URL"
    if command -v wget >/dev/null 2>&1; then
      wget -c --show-progress -O "$TMP" "$MS_URL"
    elif command -v curl >/dev/null 2>&1; then
      curl -L --retry 5 --retry-delay 2 -C - -o "$TMP" "$MS_URL"
    else
      echo "[ERR] 需要 wget 或 curl" >&2
      exit 1
    fi
    sz="$(file_size "$TMP")"
    if [[ ! -f "$TMP" ]] || [[ "$sz" -lt 10485760 ]]; then
      echo "[ERR] 下载无效: $TMP ($sz bytes)" >&2
      rm -f "$TMP"
      exit 1
    fi
    mv -f "$TMP" "$ONNX"
  fi
fi

export MOSS_ONNX_PATH="$OUT_DIR/simple_model.onnx"
echo "MOSS_ONNX_PATH=$MOSS_ONNX_PATH"

ORT_REQ="$ROOT/requirements_moss_ort.txt"
echo "[INFO] pip: $ORT_REQ → $PYTHON_BIN（不要 -U nvidia-*）"
if ! "$PYTHON_BIN" -c "import onnxruntime" 2>/dev/null; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true
  PIP_MIRROR=(-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn)
  "$PYTHON_BIN" -m pip install -r "$ORT_REQ" "${PIP_MIRROR[@]}" \
    || "$PYTHON_BIN" -m pip install -r "$ORT_REQ"
fi
if ! "$PYTHON_BIN" -c "import onnxruntime as ort; print('ort', getattr(ort,'__version__','?'), ort.get_available_providers())"; then
  echo "[ERR] $PYTHON_BIN 没有 onnxruntime。请: $PYTHON_BIN -m pip install -r $ORT_REQ" >&2
  exit 1
fi
echo "下一步: PIPELINE=mix ./check_env.sh"

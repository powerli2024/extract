#!/usr/bin/env bash
# 下载 MossFormer2 ONNX → /root/autodl-tmp/checkpoints/MossFormer2_ONNX/simple_model.onnx
# 供 PIPELINE=sep_route / Presence USE_SEP=1
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="MossFormer2_ONNX"
if [[ -d /root/autodl-tmp ]]; then
  OUT_DIR="${OUT_DIR:-/root/autodl-tmp/checkpoints/$MODEL_NAME}"
else
  OUT_DIR="${OUT_DIR:-$PROJECT_DIR/checkpoints/$MODEL_NAME}"
fi
MS_URL="${MS_URL:-https://www.modelscope.cn/models/dengcunqin/speech_mossformer2_separation_temporal_16k/resolve/master/simple_model.onnx}"

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

if [[ -f /etc/network_turbo ]]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo || true
fi

rm -f "$ONNX" "$TMP"
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
echo "[OK] $ONNX ($sz bytes)"
echo "export MOSS_ONNX_PATH=$ONNX"

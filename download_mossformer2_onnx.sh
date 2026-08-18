#!/usr/bin/env bash
# 内部脚本：下载 MossFormer2 ONNX。请用仓库根目录 ./download_models.sh
# 输出: $MOSS_CKPT_DIR/MossFormer2_ONNX/simple_model.onnx
# 来源: ModelScope dengcunqin/speech_mossformer2_separation_temporal_16k
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$PROJECT_DIR/paths_defaults.sh"
MODEL_NAME="MossFormer2_ONNX"
OUT_DIR="${OUT_DIR:-$MOSS_CKPT_DIR/$MODEL_NAME}"

# ModelScope 直链（resolve/master，无需 git-lfs）
MS_URL="${MS_URL:-https://www.modelscope.cn/models/dengcunqin/speech_mossformer2_separation_temporal_16k/resolve/master/simple_model.onnx}"

log_info() { printf '[INFO %s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
log_ok()   { printf '[ OK  %s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
log_warn() { printf '[WARN %s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
log_err()  { printf '[ERR  %s] %s\n' "$(date '+%H:%M:%S')" "$*" >&2; }

file_size() {
  local f="$1"
  if [[ ! -f "$f" ]]; then echo 0; return; fi
  stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0
}

mkdir -p "$OUT_DIR"
ONNX="$OUT_DIR/simple_model.onnx"
TMP="$OUT_DIR/simple_model.onnx.partial"

# 已有有效文件（>10MB 认为有效）
if [[ -f "$ONNX" ]]; then
  sz="$(file_size "$ONNX")"
  if [[ "$sz" -gt 10485760 ]]; then
    log_ok "已存在: $ONNX ($sz bytes)"
    exit 0
  fi
  log_warn "已有文件过小 ($sz bytes)，重新下载"
fi

rm -f "$ONNX" "$TMP"

download_url() {
  local url="$1"
  local dest="$2"
  log_info "下载: $url"
  if command -v wget >/dev/null 2>&1; then
    wget -c --show-progress -O "$dest" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -L --retry 5 --retry-delay 2 -C - -o "$dest" "$url"
  else
    log_err "需要 wget 或 curl"
    return 1
  fi
}

# 下载
if ! download_url "$MS_URL" "$TMP"; then
  log_err "ModelScope 下载失败"
  log_err "请确认网络连通，或手动下载放到:"
  log_err "  $ONNX"
  log_err "URL: $MS_URL"
  rm -f "$TMP"
  exit 1
fi

sz="$(file_size "$TMP")"
if [[ ! -f "$TMP" ]] || [[ "$sz" -lt 10485760 ]]; then
  log_err "下载文件无效: $TMP ($sz bytes)"
  rm -f "$TMP"
  exit 1
fi

mv -f "$TMP" "$ONNX"
log_ok "完成: $ONNX ($sz bytes)"

# 导出环境变量提示
echo ""
echo "=== 下载完成 ==="
echo "ONNX 模型: $ONNX"
echo ""
echo "使用前设置（可选，自动查找不需要）:"
echo "  export MOSS_ONNX_PATH=$ONNX"
echo ""
ls -lh "$OUT_DIR"

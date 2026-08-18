#!/usr/bin/env bash
# 内部脚本：下载 MossFormer2 ONNX。请用仓库根目录 ./download_models.sh
# 先检查 OUT_DIR/simple_model.onnx 是否合格，合格则跳过。
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$PROJECT_DIR/paths_defaults.sh"
# shellcheck disable=SC1091
source "$PROJECT_DIR/download_lib.sh"
MODEL_NAME="MossFormer2_ONNX"
OUT_DIR="${OUT_DIR:-$MOSS_CKPT_DIR/$MODEL_NAME}"
MS_URL="${MS_URL:-https://www.modelscope.cn/models/dengcunqin/speech_mossformer2_separation_temporal_16k/resolve/master/simple_model.onnx}"

log_info() { printf '[INFO %s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
log_ok()   { printf '[ OK  %s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
log_warn() { printf '[WARN %s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
log_err()  { printf '[ERR  %s] %s\n' "$(date '+%H:%M:%S')" "$*" >&2; }

ONNX="$OUT_DIR/simple_model.onnx"
TMP="$OUT_DIR/simple_model.onnx.partial"

log_info "检查 ONNX 目录: $OUT_DIR"
if [[ -n "${OUT_DIR}" && "$OUT_DIR" != "/" ]]; then
  mkdir -p "$OUT_DIR"
else
  log_err "OUT_DIR 非法: ${OUT_DIR:-<empty>}"
  exit 1
fi

if [[ "${VM_FORCE_DOWNLOAD:-0}" != "1" ]] && is_onnx_ok "$ONNX"; then
  log_ok "已存在且合格，跳过下载: $ONNX ($(file_size "$ONNX") bytes)"
  exit 0
fi

if [[ "${VM_FORCE_DOWNLOAD:-0}" == "1" ]]; then
  log_info "--force：将重新下载 $ONNX"
elif [[ -e "$ONNX" ]]; then
  log_warn "已有文件不合格 ($(explain_bad "$ONNX"))，将重新下载"
fi

if [[ -f /etc/network_turbo ]]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo || true
  log_info "已 source /etc/network_turbo"
fi

rm -f "$TMP"
if [[ -f "$ONNX" ]] && ! is_onnx_ok "$ONNX"; then
  rm -f "$ONNX"
fi

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

log_info "开始下载 → $ONNX"
if ! download_url "$MS_URL" "$TMP"; then
  log_err "ModelScope 下载失败"
  log_err "请放到: $ONNX"
  log_err "URL: $MS_URL"
  rm -f "$TMP"
  exit 1
fi

if ! is_onnx_ok "$TMP"; then
  log_err "下载文件无效: $TMP ($(explain_bad "$TMP"))"
  rm -f "$TMP"
  exit 1
fi

mv -f "$TMP" "$ONNX"
log_ok "完成: $ONNX ($(file_size "$ONNX") bytes)"
ls -lh "$OUT_DIR"

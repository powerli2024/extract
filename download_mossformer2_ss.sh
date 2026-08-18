#!/usr/bin/env bash
# 内部脚本：下载 MossFormer2_SS_16K。请用仓库根目录 ./download_models.sh
# 先检查 OUT_DIR 下文件是否合格，合格则跳过。
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$PROJECT_DIR/paths_defaults.sh"
# shellcheck disable=SC1091
source "$PROJECT_DIR/download_lib.sh"
MODEL="${MODEL:-MossFormer2_SS_16K}"
OUT_DIR="${OUT_DIR:-$MOSS_SS_DIR}"
HF_REPO="${HF_REPO:-alibabasglab/$MODEL}"
HF_URL="${HF_URL:-https://huggingface.co/${HF_REPO}/resolve/main/last_best_checkpoint.pt}"
MS_URL="${MS_URL:-https://www.modelscope.cn/models/iic/ClearerVoice-Studio/resolve/master/clearvoice/checkpoints/${MODEL}/last_best_checkpoint.pt}"

log_info() { printf '[INFO %s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
log_ok()   { printf '[ OK  %s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
log_warn() { printf '[WARN %s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
log_err()  { printf '[ERR  %s] %s\n' "$(date '+%H:%M:%S')" "$*" >&2; }

PT="$OUT_DIR/last_best_checkpoint.pt"
BARE="$OUT_DIR/last_best_checkpoint"
TMP="$OUT_DIR/last_best_checkpoint.pt.partial"

log_info "检查 ClearVoice 目录: $OUT_DIR"
if [[ -n "${OUT_DIR}" && "$OUT_DIR" != "/" ]]; then
  mkdir -p "$OUT_DIR"
else
  log_err "OUT_DIR 非法: ${OUT_DIR:-<empty>}"
  exit 1
fi

if [[ "${VM_FORCE_DOWNLOAD:-0}" != "1" ]] && is_pt_ok "$PT"; then
  ensure_ss_pointer "$OUT_DIR"
  log_ok "已存在且合格，跳过下载: $PT ($(file_size "$PT") bytes)"
  exit 0
fi

if [[ "${VM_FORCE_DOWNLOAD:-0}" == "1" ]]; then
  log_info "--force：将重新下载 $PT"
elif [[ -e "$PT" ]]; then
  log_warn "已有文件不合格 ($(explain_bad "$PT"))，将重新下载"
fi

if [[ "${VM_FORCE_DOWNLOAD:-0}" != "1" ]] && ! is_pt_ok "$PT" && is_pt_ok "$BARE"; then
  mv -f "$BARE" "$PT"
  ensure_ss_pointer "$OUT_DIR"
  log_ok "已将二进制 last_best_checkpoint 重命名为 .pt，跳过下载"
  exit 0
fi

if [[ -f /etc/network_turbo ]]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo || true
  log_info "已 source /etc/network_turbo"
fi

rm -f "$TMP"
# 只删不合格的目标，避免误删合格文件（force 时才删）
if [[ -f "$PT" ]] && ! is_pt_ok "$PT"; then
  rm -f "$PT"
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

log_info "开始下载 → $PT"
if download_url "$MS_URL" "$TMP"; then
  :
else
  log_warn "ModelScope 失败，尝试 HuggingFace"
  rm -f "$TMP"
  download_url "$HF_URL" "$TMP" || true
fi

if ! is_pt_ok "$TMP"; then
  log_err "下载失败或文件无效: $TMP ($(explain_bad "$TMP"))"
  log_err "请放到: $PT"
  rm -f "$TMP"
  exit 1
fi

mv -f "$TMP" "$PT"
ensure_ss_pointer "$OUT_DIR"
log_ok "完成: $PT ($(file_size "$PT") bytes)"
ls -lh "$OUT_DIR"

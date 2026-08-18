#!/usr/bin/env bash
# 下载 MossFormer2_SS_16K → <VB上级>/checkpoints/MossFormer2_SS_16K/last_best_checkpoint.pt
# 默认不落在 VB/ 内
#
# 优先 wget/curl 直连 HuggingFace（避开 huggingface-cli / xet 401）
# 网络: 先 source /etc/network_turbo
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDIA_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
MODEL="${MODEL:-MossFormer2_SS_16K}"
# 权重在 VB 外: ../checkpoints/MossFormer2_SS_16K/
OUT_DIR="${OUT_DIR:-$MEDIA_ROOT/checkpoints/$MODEL}"
HF_REPO="${HF_REPO:-alibabasglab/$MODEL}"
# 直链（resolve/main，不走 xet CLI）
HF_URL="${HF_URL:-https://huggingface.co/${HF_REPO}/resolve/main/last_best_checkpoint.pt}"
MS_URL="${MS_URL:-https://www.modelscope.cn/models/iic/ClearerVoice-Studio/resolve/master/clearvoice/checkpoints/${MODEL}/last_best_checkpoint.pt}"

log_info() { printf '[INFO %s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
log_ok()   { printf '[ OK  %s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
log_warn() { printf '[WARN %s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
log_err()  { printf '[ERR  %s] %s\n' "$(date '+%H:%M:%S')" "$*" >&2; }

file_size() {
  local f="$1"
  if [[ ! -f "$f" ]]; then echo 0; return; fi
  stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0
}

is_lfs_pointer() {
  local f="$1"
  [[ -f "$f" ]] || return 1
  local sz
  sz="$(file_size "$f")"
  [[ "$sz" -lt 2048 ]] || return 1
  head -c 40 "$f" 2>/dev/null | grep -q 'git-lfs' && return 0
  return 1
}

mkdir -p "$OUT_DIR"
PT="$OUT_DIR/last_best_checkpoint.pt"
BARE="$OUT_DIR/last_best_checkpoint"
TMP="$OUT_DIR/last_best_checkpoint.pt.partial"

if [[ -f /etc/network_turbo ]]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo
  log_info "已 source /etc/network_turbo（直连 HuggingFace）"
fi

write_pointer() {
  # ClearVoice: last_best_checkpoint 是文本，内容为权重文件名（不能是 .pt 的 symlink）
  printf '%s\n' "last_best_checkpoint.pt" >"$BARE"
  log_ok "文本指针 $BARE → last_best_checkpoint.pt"
}

# 已有有效 .pt
if [[ -f "$PT" ]] && ! is_lfs_pointer "$PT"; then
  sz="$(file_size "$PT")"
  if [[ "$sz" -gt 1048576 ]]; then
    write_pointer
    log_ok "已存在: $PT ($sz bytes)"
    exit 0
  fi
fi

# 无后缀大文件（旧布局）→ 挪成 .pt，再写文本指针
if [[ -f "$BARE" ]] && ! is_lfs_pointer "$BARE"; then
  sz="$(file_size "$BARE")"
  if [[ "$sz" -gt 1048576 ]]; then
    if [[ ! -f "$PT" ]]; then
      mv -f "$BARE" "$PT"
      log_ok "已将二进制 last_best_checkpoint 重命名为 .pt"
    fi
    write_pointer
    exit 0
  fi
  log_warn "last_best_checkpoint 过小($sz)，当作无效 LFS 指针"
fi

rm -f "$PT" "$TMP"

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

# 1) HuggingFace 直链
if download_url "$HF_URL" "$TMP"; then
  :
else
  log_warn "HuggingFace 失败，尝试 ModelScope"
  rm -f "$TMP"
  download_url "$MS_URL" "$TMP" || true
fi

sz="$(file_size "$TMP")"
if [[ ! -f "$TMP" ]] || [[ "$sz" -lt 1048576 ]] || is_lfs_pointer "$TMP"; then
  log_err "下载失败或文件无效: $TMP ($sz bytes)"
  log_err "请确认已 source /etc/network_turbo，或手动下载放到:"
  log_err "  $PT"
  log_err "HF: $HF_URL"
  rm -f "$TMP"
  exit 1
fi

mv -f "$TMP" "$PT"
# 清掉无效 LFS / 错误 symlink，写入 ClearVoice 所需文本指针
rm -f "$BARE"
write_pointer
log_ok "完成: $PT ($sz bytes)"
ls -lh "$OUT_DIR"

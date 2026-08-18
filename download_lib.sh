#!/usr/bin/env bash
# 被 download_*.sh source：文件大小 / 格式 / 规范路径检查。不单独执行。

file_size() {
  local f="$1"
  if [[ ! -f "$f" ]]; then echo 0; return; fi
  stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0
}

_head_ascii() {
  local f="$1" n="${2:-64}"
  head -c "$n" "$f" 2>/dev/null | tr -d '\0' || true
}

is_html_or_json_error() {
  local f="$1"
  [[ -f "$f" ]] || return 1
  local h
  h="$(_head_ascii "$f" 80)"
  [[ "$h" == "<!"* || "$h" == "<html"* || "$h" == "<HTML"* ]] && return 0
  [[ "$h" == *"git-lfs"* ]] && return 0
  return 1
}

is_lfs_pointer() {
  local f="$1"
  [[ -f "$f" ]] || return 1
  local sz
  sz="$(file_size "$f")"
  [[ "$sz" -lt 2048 ]] || return 1
  head -c 64 "$f" 2>/dev/null | grep -q 'git-lfs' && return 0
  return 1
}

# ONNX: 常规文件或指向文件的链接、>10MB、非 HTML/LFS
is_onnx_ok() {
  local f="$1"
  [[ -f "$f" ]] || return 1
  [[ -d "$f" ]] && return 1
  local sz
  sz="$(file_size "$f")"
  [[ "$sz" -gt 10485760 ]] || return 1
  is_html_or_json_error "$f" && return 1
  is_lfs_pointer "$f" && return 1
  if head -c 8 "$f" 2>/dev/null | grep -Eq '^[{<]'; then
    return 1
  fi
  return 0
}

# ClearVoice .pt: >10MB，ZIP(PK) 或 pickle(\x80)，非 LFS/HTML
is_pt_ok() {
  local f="$1"
  [[ -f "$f" ]] || return 1
  [[ -d "$f" ]] && return 1
  local sz
  sz="$(file_size "$f")"
  [[ "$sz" -gt 10485760 ]] || return 1
  is_html_or_json_error "$f" && return 1
  is_lfs_pointer "$f" && return 1
  local mag
  mag="$(head -c 2 "$f" 2>/dev/null || true)"
  # PK = zip 包装的 torch.save；\x80 = pickle
  if [[ "$mag" == "PK" ]]; then
    return 0
  fi
  local b0
  b0="$(head -c 1 "$f" 2>/dev/null | od -An -tx1 2>/dev/null | tr -d ' \n' || true)"
  [[ "$b0" == "80" ]] && return 0
  # 个别权重不是上述魔数，但已经很大且不是 HTML：仍接受
  [[ "$sz" -gt 104857600 ]] && return 0
  return 1
}

# ClearVoice 文本指针：内容必须是权重文件名，不能是二进制/symlink
is_ss_pointer_ok() {
  local pointer="$1"
  [[ -f "$pointer" && ! -L "$pointer" ]] || return 1
  local sz
  sz="$(file_size "$pointer")"
  [[ "$sz" -gt 0 && "$sz" -lt 512 ]] || return 1
  grep -qx 'last_best_checkpoint.pt' "$pointer" 2>/dev/null
}

is_asr_ok() {
  local d="$1"
  [[ -d "$d" ]] || return 1
  [[ -f "$d/config.json" ]] || return 1
  local h
  h="$(_head_ascii "$d/config.json" 20)"
  [[ "$h" == "{"* ]] || return 1
  local w
  for w in \
    "$d/model.safetensors" \
    "$d/model.safetensors.index.json" \
    "$d/pytorch_model.bin" \
    "$d/pytorch_model.bin.index.json"
  do
    if [[ -f "$w" ]]; then
      local sz
      sz="$(file_size "$w")"
      # index.json 可以很小；权重文件要大
      if [[ "$w" == *.json ]]; then
        # 有分片：至少存在一个 >10MB 的 safetensors/bin
        local shard
        shard="$(find "$d" -maxdepth 1 \( -name '*.safetensors' -o -name '*.bin' \) -size +10M 2>/dev/null | head -1)"
        [[ -n "$shard" ]] && return 0
      elif [[ "$sz" -gt 10485760 ]]; then
        return 0
      fi
    fi
  done
  local shard
  shard="$(find "$d" -maxdepth 1 \( -name '*.safetensors' -o -name '*.bin' \) -size +10M 2>/dev/null | head -1)"
  [[ -n "$shard" ]]
}

explain_bad() {
  local f="$1"
  if [[ ! -e "$f" ]]; then echo "不存在"; return; fi
  if [[ -d "$f" && ! -f "$f" ]]; then echo "是目录不是文件"; return; fi
  local sz
  sz="$(file_size "$f")"
  if is_lfs_pointer "$f"; then echo "Git LFS 指针 (${sz}B)"; return; fi
  if is_html_or_json_error "$f"; then echo "HTML/错误页 (${sz}B)"; return; fi
  echo "过小或格式不对 (${sz}B)"
}

# 在候选路径里找一份已合格的文件；找到则打印路径
find_ok_file() {
  local kind="$1"
  shift
  local c
  for c in "$@"; do
    [[ -n "$c" ]] || continue
    case "$kind" in
      onnx) is_onnx_ok "$c" && { echo "$c"; return 0; } ;;
      pt)   is_pt_ok "$c" && { echo "$c"; return 0; } ;;
    esac
  done
  return 1
}

# 把已有合格文件接到规范路径：同盘优先硬链/符号链接，避免再下
adopt_to_canonical() {
  local src="$1"
  local dest="$2"
  [[ -f "$src" ]] || return 1
  mkdir -p "$(dirname "$dest")"
  if [[ "$(readlink -f "$src" 2>/dev/null || echo "$src")" == "$(readlink -f "$dest" 2>/dev/null || echo "")" ]]; then
    return 0
  fi
  if [[ -e "$dest" ]]; then
    rm -f "$dest"
  fi
  if ln -f "$src" "$dest" 2>/dev/null; then
    echo "[INFO] 硬链接 $src → $dest"
    return 0
  fi
  ln -sfn "$src" "$dest"
  echo "[INFO] 符号链接 $src → $dest"
}

ensure_ss_pointer() {
  local dir="$1"
  local pointer="$dir/last_best_checkpoint"
  mkdir -p "$dir"
  if [[ -L "$pointer" ]]; then
    rm -f "$pointer"
  fi
  if [[ -f "$pointer" ]] && ! is_ss_pointer_ok "$pointer"; then
    local sz
    sz="$(file_size "$pointer")"
    if [[ "$sz" -gt 1048576 ]]; then
      # 旧布局：无后缀大文件其实是权重
      if [[ ! -f "$dir/last_best_checkpoint.pt" ]]; then
        mv -f "$pointer" "$dir/last_best_checkpoint.pt"
        echo "[INFO] 将二进制 last_best_checkpoint 重命名为 .pt"
      else
        rm -f "$pointer"
      fi
    else
      rm -f "$pointer"
    fi
  fi
  printf '%s\n' "last_best_checkpoint.pt" >"$pointer"
}

#!/usr/bin/env bash
# 权重唯一下载入口。不要单独跑 download_mossformer2_*.sh。
#
# 流程：先检查规范路径与文件是否合格 → 已合格则跳过 → 其它目录有合格文件则链接过来
#      → 仍缺再下载。
#
# 目标:
#   $MOSS_CKPT_DIR/MossFormer2_ONNX/simple_model.onnx
#   $MOSS_CKPT_DIR/MossFormer2_SS_16K/last_best_checkpoint.pt
#   $ASR_MODEL_DIR/                    # 加 --asr 才下载
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/paths_defaults.sh"
# shellcheck disable=SC1091
source "$ROOT/download_lib.sh"

WANT_ONNX=1
WANT_SS=1
WANT_ASR=0
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --onnx-only) WANT_ONNX=1; WANT_SS=0; WANT_ASR=0; shift ;;
    --ss-only)   WANT_ONNX=0; WANT_SS=1; WANT_ASR=0; shift ;;
    --asr|--with-asr) WANT_ASR=1; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help)
      cat <<EOF
用法: ./download_models.sh [--onnx-only|--ss-only] [--asr] [--force]
  先检查目录/文件是否已合格，合格则跳过，不重复下载。
  --force  忽略已有文件，强制重下。
当前:
  MOSS_CKPT_DIR=$MOSS_CKPT_DIR
  ASR_MODEL_DIR=$ASR_MODEL_DIR
EOF
      exit 0
      ;;
    *) echo "未知参数: $1  （./download_models.sh --help）" >&2; exit 1 ;;
  esac
done

ONNX_CANON="$MOSS_CKPT_DIR/MossFormer2_ONNX/simple_model.onnx"
SS_CANON="$MOSS_SS_DIR/last_best_checkpoint.pt"

onnx_alts=(
  "$ONNX_CANON"
  /root/autodl-tmp/checkpoints/MossFormer2_ONNX/simple_model.onnx
  /root/checkpoints/MossFormer2_ONNX/simple_model.onnx
  "$VM_ROOT/../checkpoints/MossFormer2_ONNX/simple_model.onnx"
)
ss_alts=(
  "$SS_CANON"
  /root/autodl-tmp/checkpoints/MossFormer2_SS_16K/last_best_checkpoint.pt
  /root/checkpoints/MossFormer2_SS_16K/last_best_checkpoint.pt
  "$VM_ROOT/../checkpoints/MossFormer2_SS_16K/last_best_checkpoint.pt"
)
asr_alts=(
  "$ASR_MODEL_DIR"
  /root/autodl-tmp/Qwen3-ASR-1.7B
  /root/Qwen3-ASR-1.7B
)

NEED_ONNX=0
NEED_SS=0
NEED_ASR=0

echo "============================================"
echo " download_models  （先检查，再决定是否下载）"
echo " MOSS_CKPT_DIR=$MOSS_CKPT_DIR"
echo " ASR_MODEL_DIR=$ASR_MODEL_DIR"
echo "============================================"
echo ""
echo "---- 预检查 ----"

check_onnx() {
  if [[ "$FORCE" != "1" ]] && is_onnx_ok "$ONNX_CANON"; then
    echo "[SKIP] ONNX 已合格  $(file_size "$ONNX_CANON") bytes  $ONNX_CANON"
    return 0
  fi
  if [[ "$FORCE" == "1" ]]; then
    echo "[INFO] --force：将重下 ONNX"
    NEED_ONNX=1
    return 0
  fi
  if [[ -e "$ONNX_CANON" ]]; then
    echo "[WARN] ONNX 规范路径不合格: $ONNX_CANON ($(explain_bad "$ONNX_CANON"))"
  else
    echo "[INFO] ONNX 规范路径不存在: $ONNX_CANON"
  fi
  local found
  found="$(find_ok_file onnx "${onnx_alts[@]}")" || found=""
  if [[ -n "$found" ]]; then
    echo "[INFO] 在其它目录找到合格 ONNX: $found"
    adopt_to_canonical "$found" "$ONNX_CANON"
    if is_onnx_ok "$ONNX_CANON"; then
      echo "[SKIP] ONNX 已接到规范路径，不下载"
      return 0
    fi
  fi
  echo "[NEED] ONNX 需要下载"
  NEED_ONNX=1
}

check_ss() {
  if [[ "$FORCE" != "1" ]] && is_pt_ok "$SS_CANON"; then
    ensure_ss_pointer "$MOSS_SS_DIR"
    echo "[SKIP] ClearVoice .pt 已合格  $(file_size "$SS_CANON") bytes  $SS_CANON"
    return 0
  fi
  if [[ "$FORCE" == "1" ]]; then
    echo "[INFO] --force：将重下 ClearVoice .pt"
    NEED_SS=1
    return 0
  fi
  if [[ -e "$SS_CANON" ]]; then
    echo "[WARN] ClearVoice 规范路径不合格: $SS_CANON ($(explain_bad "$SS_CANON"))"
  else
    echo "[INFO] ClearVoice 规范路径不存在: $SS_CANON"
  fi
  local found
  found="$(find_ok_file pt "${ss_alts[@]}")" || found=""
  if [[ -n "$found" ]]; then
    echo "[INFO] 在其它目录找到合格 .pt: $found"
    mkdir -p "$MOSS_SS_DIR"
    adopt_to_canonical "$found" "$SS_CANON"
    ensure_ss_pointer "$MOSS_SS_DIR"
    if is_pt_ok "$SS_CANON"; then
      echo "[SKIP] ClearVoice 已接到规范路径，不下载"
      return 0
    fi
  fi
  echo "[NEED] ClearVoice .pt 需要下载"
  NEED_SS=1
}

check_asr() {
  if [[ "$FORCE" == "1" && "$WANT_ASR" == "1" ]]; then
    echo "[INFO] --force：将重下 ASR"
    NEED_ASR=1
    return 0
  fi
  if is_asr_ok "$ASR_MODEL_DIR"; then
    echo "[SKIP] ASR 已合格  $ASR_MODEL_DIR"
    return 0
  fi
  if [[ -e "$ASR_MODEL_DIR" ]]; then
    echo "[WARN] ASR 目录不完整: $ASR_MODEL_DIR （需 config.json + 权重 *.safetensors/*.bin）"
  else
    echo "[INFO] ASR 目录不存在: $ASR_MODEL_DIR"
  fi
  local d
  for d in "${asr_alts[@]}"; do
    [[ -n "$d" ]] || continue
    if is_asr_ok "$d"; then
      echo "[INFO] 在其它目录找到合格 ASR: $d"
      mkdir -p "$(dirname "$ASR_MODEL_DIR")"
      if [[ "$(readlink -f "$d" 2>/dev/null || echo "$d")" != "$(readlink -f "$ASR_MODEL_DIR" 2>/dev/null || true)" ]]; then
        if [[ -e "$ASR_MODEL_DIR" && ! -L "$ASR_MODEL_DIR" ]]; then
          echo "[WARN] $ASR_MODEL_DIR 已存在且不完整，请手动核对；暂用 $d"
          export ASR_MODEL_DIR="$d"
        else
          ln -sfn "$d" "$ASR_MODEL_DIR"
          echo "[INFO] 符号链接 $d → $ASR_MODEL_DIR"
        fi
      fi
      if is_asr_ok "$ASR_MODEL_DIR"; then
        echo "[SKIP] ASR 已接到规范路径，不下载"
        return 0
      fi
    fi
  done
  if [[ "$WANT_ASR" == "1" ]]; then
    echo "[NEED] ASR 需要下载"
    NEED_ASR=1
  else
    echo "[WARN] ASR 未就绪（默认不下载）。需要时: ./download_models.sh --asr"
  fi
}

if [[ "$WANT_ONNX" == "1" ]]; then check_onnx; fi
if [[ "$WANT_SS" == "1" ]]; then check_ss; fi
check_asr

if [[ "$NEED_ONNX" != "1" && "$NEED_SS" != "1" && "$NEED_ASR" != "1" ]]; then
  echo ""
  echo "[ OK ] 所需权重已在规范目录，无需下载"
  echo "检查: ./check_env.sh"
  exit 0
fi

echo ""
echo "---- 开始下载缺的部分 ----"
if [[ "$FORCE" == "1" ]]; then
  export VM_FORCE_DOWNLOAD=1
fi
if [[ -f /etc/network_turbo ]]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo || true
  echo "[INFO] 已 source /etc/network_turbo"
fi
mkdir -p "$MOSS_CKPT_DIR"

if [[ "$NEED_SS" == "1" ]]; then
  echo "[INFO] 下载 ClearVoice SS → $MOSS_SS_DIR"
  OUT_DIR="$MOSS_SS_DIR" bash "$ROOT/download_mossformer2_ss.sh" \
    || echo "[WARN] ClearVoice SS 下载失败（s2/s4/s5/s7/s8 需要）"
fi

if [[ "$NEED_ONNX" == "1" ]]; then
  echo "[INFO] 下载 MossFormer2 ONNX → $(dirname "$ONNX_CANON")"
  OUT_DIR="$(dirname "$ONNX_CANON")" bash "$ROOT/download_mossformer2_onnx.sh" \
    || echo "[WARN] ONNX 下载失败（s1/s3/s6 需要）"
fi

if [[ "$NEED_ASR" == "1" ]]; then
  echo "[INFO] 下载 Qwen3-ASR-1.7B → $ASR_MODEL_DIR （体积很大）"
  mkdir -p "$ASR_MODEL_DIR"
  if command -v modelscope >/dev/null 2>&1; then
    modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir "$ASR_MODEL_DIR" \
      || echo "[WARN] modelscope 下载 ASR 失败"
  else
    py="${PYTHON_BIN:-python3}"
    if "$py" -c "import modelscope" 2>/dev/null; then
      "$py" - <<PY
from modelscope import snapshot_download
snapshot_download("Qwen/Qwen3-ASR-1.7B", local_dir=r"${ASR_MODEL_DIR}")
PY
    else
      echo "[WARN] 未安装 modelscope，ASR 请手动放到 $ASR_MODEL_DIR"
      echo "       pip install modelscope && ./download_models.sh --asr"
    fi
  fi
  if is_asr_ok "$ASR_MODEL_DIR"; then
    echo "[ OK ] ASR 下载后校验通过"
  else
    echo "[WARN] ASR 下载后仍不完整: $ASR_MODEL_DIR"
  fi
fi

echo ""
echo "---- 完成后复核 ----"
if [[ "$WANT_ONNX" == "1" ]]; then
  if is_onnx_ok "$ONNX_CANON"; then
    echo "[ OK ] ONNX  $(file_size "$ONNX_CANON") bytes  $ONNX_CANON"
  else
    echo "[WARN] ONNX 仍不合格  $ONNX_CANON ($(explain_bad "$ONNX_CANON"))"
  fi
fi
if [[ "$WANT_SS" == "1" ]]; then
  if is_pt_ok "$SS_CANON"; then
    echo "[ OK ] ClearVoice .pt  $(file_size "$SS_CANON") bytes  $SS_CANON"
  else
    echo "[WARN] ClearVoice .pt 仍不合格  $SS_CANON ($(explain_bad "$SS_CANON"))"
  fi
fi
if is_asr_ok "$ASR_MODEL_DIR"; then
  echo "[ OK ] ASR  $ASR_MODEL_DIR"
else
  echo "[WARN] ASR 未就绪  $ASR_MODEL_DIR"
fi
echo ""
echo "检查: ./check_env.sh"

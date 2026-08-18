#!/usr/bin/env bash
# 权重唯一下载入口。不要直接跑 download_mossformer2_*.sh（未设 OUT_DIR 会写到错误目录）。
#
# 目标布局（AutoDL 数据盘）:
#   $MOSS_CKPT_DIR/MossFormer2_ONNX/simple_model.onnx
#   $MOSS_CKPT_DIR/MossFormer2_SS_16K/last_best_checkpoint.pt
#   $ASR_MODEL_DIR/                    # Qwen3-ASR-1.7B，默认不自动拉
#
# 用法:
#   ./download_models.sh              # ONNX + ClearVoice .pt
#   ./download_models.sh --onnx-only
#   ./download_models.sh --ss-only
#   ./download_models.sh --asr        # 额外拉 ASR（很大）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/paths_defaults.sh"

WANT_ONNX=1
WANT_SS=1
WANT_ASR=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --onnx-only) WANT_ONNX=1; WANT_SS=0; WANT_ASR=0; shift ;;
    --ss-only)   WANT_ONNX=0; WANT_SS=1; WANT_ASR=0; shift ;;
    --asr|--with-asr) WANT_ASR=1; shift ;;
    -h|--help)
      sed -n '2,16p' "$0" | sed 's/^# \?//'
      echo "当前: MOSS_CKPT_DIR=$MOSS_CKPT_DIR"
      echo "      ASR_MODEL_DIR=$ASR_MODEL_DIR"
      exit 0
      ;;
    *) echo "未知参数: $1  （./download_models.sh --help）" >&2; exit 1 ;;
  esac
done

mkdir -p "$MOSS_CKPT_DIR" "$(dirname "$ASR_MODEL_DIR")"
if [[ -f /etc/network_turbo ]]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo || true
  echo "[INFO] 已 source /etc/network_turbo"
fi

echo "============================================"
echo " download_models"
echo " MOSS_CKPT_DIR=$MOSS_CKPT_DIR"
echo " ASR_MODEL_DIR=$ASR_MODEL_DIR"
echo "============================================"
echo "将写入:"
echo "  $MOSS_CKPT_DIR/MossFormer2_ONNX/simple_model.onnx"
echo "  $MOSS_CKPT_DIR/MossFormer2_SS_16K/last_best_checkpoint.pt"
echo "  $ASR_MODEL_DIR/   (仅 --asr)"
echo ""

if [[ "$WANT_SS" == "1" ]]; then
  echo "[INFO] ClearVoice SS → $MOSS_SS_DIR"
  OUT_DIR="$MOSS_SS_DIR" bash "$ROOT/download_mossformer2_ss.sh" \
    || echo "[WARN] ClearVoice SS 下载失败（s2/s4/s5/s7/s8 需要）"
fi

if [[ "$WANT_ONNX" == "1" ]]; then
  echo "[INFO] MossFormer2 ONNX → $MOSS_CKPT_DIR/MossFormer2_ONNX"
  OUT_DIR="$MOSS_CKPT_DIR/MossFormer2_ONNX" bash "$ROOT/download_mossformer2_onnx.sh" \
    || echo "[WARN] ONNX 下载失败（s1/s3/s6 需要）"
fi

if [[ "$WANT_ASR" == "1" ]]; then
  if [[ -f "$ASR_MODEL_DIR/config.json" ]]; then
    echo "[ OK ] ASR 已在: $ASR_MODEL_DIR"
  else
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
  fi
else
  if [[ -f "$ASR_MODEL_DIR/config.json" ]]; then
    echo "[ OK ] ASR 已在: $ASR_MODEL_DIR"
  else
    echo "[WARN] 未找到 ASR: $ASR_MODEL_DIR"
    echo "       自行放置，或: ./download_models.sh --asr"
  fi
fi

echo ""
ok_or_warn() {
  local f="$1" msg="$2"
  if [[ -f "$f" ]]; then
    local sz
    sz="$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)"
    echo "[ OK ] $msg  ($sz bytes)  $f"
  else
    echo "[WARN] 缺少 $msg  $f"
  fi
}
ok_or_warn "$MOSS_CKPT_DIR/MossFormer2_ONNX/simple_model.onnx" "ONNX"
ok_or_warn "$MOSS_SS_DIR/last_best_checkpoint.pt" "ClearVoice .pt"
if [[ -f "$ASR_MODEL_DIR/config.json" ]]; then
  echo "[ OK ] ASR  $ASR_MODEL_DIR"
else
  echo "[WARN] ASR 未就绪  $ASR_MODEL_DIR"
fi
echo ""
echo "下载结束。检查: ./check_env.sh"

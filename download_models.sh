#!/usr/bin/env bash
# 唯一下载入口（与 run_* 分离）。调用本包内 download_mossformer2_*.sh。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDIA="$(cd "$ROOT/.." && pwd)"

export MOSS_CKPT_DIR="${MOSS_CKPT_DIR:-/root/checkpoints}"
if [[ ! -d "$(dirname "$MOSS_CKPT_DIR")" ]] && [[ -d /root/autodl-tmp ]]; then
  export MOSS_CKPT_DIR="${MOSS_CKPT_DIR:-/root/autodl-tmp/checkpoints}"
fi
mkdir -p "$MOSS_CKPT_DIR"

echo "============================================"
echo " VM download_models（独立下载；与运行分离）"
echo " ROOT=$ROOT"
echo "============================================"

# ClearVoice SS
SS_SH="$ROOT/download_mossformer2_ss.sh"
if [[ -f "$SS_SH" ]]; then
  echo "[INFO] ClearVoice SS via $SS_SH"
  OUT_DIR="${MOSS_CKPT_DIR}/MossFormer2_SS_16K" bash "$SS_SH" || echo "[WARN] ClearVoice SS 下载失败"
else
  echo "[WARN] 缺少 $SS_SH"
fi

# ONNX
ONNX_SH="$ROOT/download_mossformer2_onnx.sh"
if [[ -f "$ONNX_SH" ]]; then
  echo "[INFO] ONNX via $ONNX_SH"
  OUT_DIR="${MOSS_CKPT_DIR}/MossFormer2_ONNX" bash "$ONNX_SH" || echo "[WARN] ONNX 下载失败"
else
  echo "[WARN] 缺少 $ONNX_SH"
fi

ASR_DIR="${ASR_MODEL_DIR:-/root/Qwen3-ASR-1.7B}"
if [[ -d "$ASR_DIR" ]]; then
  echo "[ OK ] ASR 已在本地: $ASR_DIR"
else
  echo "[WARN] 未找到 $ASR_DIR"
  echo "       请自行将 Qwen3-ASR-1.7B 放到该路径，或 export ASR_MODEL_DIR=..."
  echo "       （本脚本不自动拉取体积很大的 ASR；run_* 也不会下载）"
fi

FOUND_ONNX=""
for c in \
  "${MOSS_ONNX_PATH:-}" \
  "$MOSS_CKPT_DIR/MossFormer2_ONNX/simple_model.onnx" \
  "$MEDIA/checkpoints/MossFormer2_ONNX/simple_model.onnx" \
  /root/autodl-tmp/checkpoints/MossFormer2_ONNX/simple_model.onnx
do
  if [[ -n "$c" && -f "$c" ]]; then
    FOUND_ONNX="$c"
    export MOSS_ONNX_PATH="$c"
    echo "[ OK ] MOSS_ONNX_PATH=$c"
    break
  fi
done
[[ -n "$FOUND_ONNX" ]] || echo "[WARN] 仍未找到 simple_model.onnx"

FOUND_CV=""
for c in \
  "$MOSS_CKPT_DIR/MossFormer2_SS_16K/last_best_checkpoint.pt" \
  "$MEDIA/checkpoints/MossFormer2_SS_16K/last_best_checkpoint.pt" \
  /root/autodl-tmp/checkpoints/MossFormer2_SS_16K/last_best_checkpoint.pt
do
  if [[ -f "$c" ]]; then
    FOUND_CV="$c"
    echo "[ OK ] ClearVoice ckpt $c"
    break
  fi
done
[[ -n "$FOUND_CV" ]] || echo "[WARN] 仍未找到 ClearVoice last_best_checkpoint.pt"

echo ""
echo "download_models 结束。运行: ./check_env.sh && ./run_all.sh"

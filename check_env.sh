#!/usr/bin/env bash
# 只检查、不安装。退出码：0=可用，1=有硬错误，2=有警告但仍可尝试跑部分阶段。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/paths_defaults.sh"

HARD=0
WARN=0

ok() { echo "[ OK ] $*"; }
warn() { echo "[WARN] $*"; WARN=$((WARN + 1)); }
err() { echo "[ERR ] $*"; HARD=$((HARD + 1)); }

echo "============================================"
echo " VM check_env (read-only)"
echo " ROOT=$ROOT"
echo "============================================"

if [[ -f "${VB_DIR:-/root/VB}/.env_clearvoice" ]]; then
  # shellcheck disable=SC1090
  source "${VB_DIR:-/root/VB}/.env_clearvoice" || true
fi

export CLEARVOICE_ROOT="${CLEARVOICE_ROOT:-/root/ClearerVoice-Studio}"
if [[ -n "${CLEARVOICE_PYTHON:-}" && ! -x "${CLEARVOICE_PYTHON}" ]]; then
  unset CLEARVOICE_PYTHON
fi
if [[ -z "${CLEARVOICE_PYTHON:-}" ]]; then
  for c in \
    /root/miniconda3/envs/ClearerVoice-Studio/bin/python \
    /root/miniconda3/envs/clearvoice/bin/python \
    /root/miniconda3/envs/ClearVoice/bin/python
  do
    if [[ -x "$c" ]]; then export CLEARVOICE_PYTHON="$c"; break; fi
  done
fi

if [[ -f "$ROOT/.runtime/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.runtime/env.sh" || true
fi
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=""
  for c in \
    /root/miniconda3/envs/qwen3-asr/bin/python \
    "$(command -v python3 || true)" \
    "$(command -v python || true)"
  do
    [[ -n "$c" && -x "$c" ]] && PYTHON_BIN="$c" && break
  done
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "--- Python / 包 ---"
if [[ -x "$PYTHON_BIN" ]] || command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  ok "PYTHON_BIN=$PYTHON_BIN"
  if [[ "$PYTHON_BIN" == */envs/qwen3-asr/* ]] || [[ "$PYTHON_BIN" == */envs/"${VM_CONDA_ENV:-qwen3-asr}"/* ]]; then
    ok "使用专用 conda env"
  else
    warn "未使用 qwen3-asr env（当前 $PYTHON_BIN）。建议重跑: ./setup_env.sh"
  fi
  "$PYTHON_BIN" - <<'PY' || err "主 Python 缺包 — 请重新 ./setup_env.sh"
import importlib
import subprocess

hard = (
    "torch",
    "torchaudio",
    "onnxruntime",
    "qwen_asr",
    "editdistance",
    "pypinyin",
    "soundfile",
    "numpy",
    "scipy",
    "librosa",
    "tqdm",
)
for m in hard:
    importlib.import_module(m)
    print("  import", m, "ok")

import torch
print(f"  torch={torch.__version__} cuda={torch.cuda.is_available()}")
has_gpu = subprocess.call(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
if has_gpu and not torch.cuda.is_available():
    raise SystemExit("检测到 GPU 但 torch.cuda.is_available()=False，请重装 CUDA 版 torch")

import onnxruntime as ort
prov = ort.get_available_providers()
print(f"  ort providers={prov}")
if has_gpu and "CUDAExecutionProvider" not in prov:
    raise SystemExit("GPU 机器缺少 CUDAExecutionProvider，请重装 onnxruntime-gpu")
PY
else
  err "找不到 PYTHON_BIN"
fi

echo "--- 数据 ---"
if [[ -d "$DATA_DIR" ]]; then
  ok "DATA_DIR=$DATA_DIR"
  for s in pos neg; do
    if [[ -f "$DATA_DIR/$s.jsonl" ]]; then
      ok "$s.jsonl"
    else
      err "缺少 $DATA_DIR/$s.jsonl"
    fi
    if [[ -d "$DATA_DIR/$s" ]]; then
      n=$(find "$DATA_DIR/$s" -maxdepth 1 -name 'kws_*.wav' 2>/dev/null | wc -l | tr -d ' ')
      ok "$DATA_DIR/$s/ kws_*.wav ≈ $n"
    else
      err "缺少目录 $DATA_DIR/$s/"
    fi
  done
else
  err "DATA_DIR 不存在: $DATA_DIR"
fi

echo "--- VM 自带运行时模块（已整合，无需外部 VB/VB_onnx）---"
need_mods=(utils_audio.py cer_metrics.py asr_backend.py mossformer2_ss.py mossformer2_onnx.py)
for f in "${need_mods[@]}"; do
  [[ -f "$ROOT/scripts/$f" ]] && ok "scripts/$f" || err "缺少 scripts/$f"
done
[[ -f "$ROOT/download_mossformer2_ss.sh" ]] && ok "download_mossformer2_ss.sh" || warn "缺少 SS 下载脚本"
[[ -f "$ROOT/download_mossformer2_onnx.sh" ]] && ok "download_mossformer2_onnx.sh" || warn "缺少 ONNX 下载脚本"

echo "--- 权重 ---"
ok "MOSS_CKPT_DIR=$MOSS_CKPT_DIR"
ok "ASR_MODEL_DIR=$ASR_MODEL_DIR"
onnx_found=""
for c in \
  "${MOSS_ONNX_PATH:-}" \
  "$MOSS_CKPT_DIR/MossFormer2_ONNX/simple_model.onnx" \
  /root/autodl-tmp/checkpoints/MossFormer2_ONNX/simple_model.onnx \
  /root/checkpoints/MossFormer2_ONNX/simple_model.onnx
do
  if [[ -n "$c" && -f "$c" ]]; then onnx_found="$c"; break; fi
done
[[ -n "$onnx_found" ]] && ok "MOSS_ONNX=$onnx_found" || warn "未找到 MossFormer2 ONNX（s1/s3/s6 需要）"

cv_found=""
for c in \
  "$MOSS_CKPT_DIR/MossFormer2_SS_16K/last_best_checkpoint.pt" \
  /root/autodl-tmp/checkpoints/MossFormer2_SS_16K/last_best_checkpoint.pt
do
  if [[ -f "$c" ]]; then cv_found="$c"; break; fi
done
[[ -n "$cv_found" ]] && ok "ClearVoice ckpt=$cv_found" || warn "未找到 ClearVoice SS ckpt（s2/s4/s5/s8 需要）"

if [[ -x "$CLEARVOICE_PYTHON" ]]; then
  ok "CLEARVOICE_PYTHON=$CLEARVOICE_PYTHON"
else
  warn "CLEARVOICE_PYTHON 不可执行: $CLEARVOICE_PYTHON"
fi
[[ -d "$CLEARVOICE_ROOT" ]] && ok "CLEARVOICE_ROOT=$CLEARVOICE_ROOT" || warn "CLEARVOICE_ROOT 不存在: $CLEARVOICE_ROOT"

if [[ -d "$ASR_MODEL_DIR" ]]; then
  ok "ASR_MODEL_DIR=$ASR_MODEL_DIR"
else
  warn "ASR_MODEL_DIR 不存在: $ASR_MODEL_DIR（有唤醒文本时打分需要）"
fi

echo "--- 输出隔离约定 ---"
ok "VM_OUT=$VM_OUT （将写 $VM_OUT/{pos,neg}/... 分树）"
ok "uid={split}_{id}；中间音频禁止跨 split 混写"

echo "--- GPU ---"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L 2>/dev/null | head -3 || warn "nvidia-smi 异常"
else
  warn "无 nvidia-smi（可用 CPU，但很慢）"
fi

echo ""
echo "HARD=$HARD WARN=$WARN"
if [[ "$HARD" -gt 0 ]]; then
  echo "结论: 环境未就绪 → 先 ./setup_env.sh && ./download_models.sh"
  exit 1
fi
if [[ "$WARN" -gt 0 ]]; then
  echo "结论: 可跑部分阶段，但有警告；缺权重请 ./download_models.sh"
  exit 2
fi
echo "结论: 检查通过，可 ./run_all.sh"
exit 0

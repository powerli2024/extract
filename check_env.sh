#!/usr/bin/env bash
# 只检查、不安装。退出码：0=可用，1=有硬错误，2=有警告但仍可尝试跑部分阶段。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/paths_defaults.sh"
# shellcheck disable=SC1091
source "$ROOT/pick_python.sh"

HARD=0
WARN=0

ok() { echo "[ OK ] $*"; }
warn() { echo "[WARN] $*"; WARN=$((WARN + 1)); }
err() { echo "[ERR ] $*"; HARD=$((HARD + 1)); }

echo "============================================"
echo " VM check_env (read-only)  unified cu124 python"
echo " ROOT=$ROOT"
echo "============================================"

if [[ -f "$ROOT/.runtime/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.runtime/env.sh" || true
fi
PYTHON_BIN="$(extract_pick_python)"
export PYTHON_BIN
export CLEARVOICE_PYTHON="$PYTHON_BIN"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "--- 固定平台合同 ---"
if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  if [[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "20.04" ]]; then
    ok "OS=${PRETTY_NAME}"
  else
    err "要求 Ubuntu 20.04，当前 ${PRETTY_NAME:-unknown}"
  fi
else
  err "找不到 /etc/os-release"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d '[:space:]')"
  if [[ "$(printf '%s\n%s\n' 525.60.13 "$driver_version" | sort -V | head -n1)" == "525.60.13" ]]; then
    ok "NVIDIA driver=$driver_version"
  else
    err "CUDA 12.x wheel 运行时要求 driver>=525.60.13，当前 $driver_version"
  fi
  if [[ "${VM_REQUIRE_CUDA_TOOLKIT:-0}" == "1" ]]; then
    if [[ "$(printf '%s\n%s\n' 550.54.14 "$driver_version" | sort -V | head -n1)" != "550.54.14" ]]; then
      err "完整 CUDA Toolkit 12.4 要求 driver>=550.54.14"
    elif ! command -v nvcc >/dev/null 2>&1; then
      err "VM_REQUIRE_CUDA_TOOLKIT=1 但找不到 nvcc"
    elif [[ "$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | tail -n1)" != "12.4" ]]; then
      err "nvcc 不是 12.4"
    else
      ok "CUDA Toolkit nvcc=12.4"
    fi
  fi
else
  err "缺少 nvidia-smi"
fi

echo "--- Python / 包（须为 VE 解释器）---"
if [[ -x "$PYTHON_BIN" ]] || command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  ok "PYTHON_BIN=$PYTHON_BIN"
  if [[ "$PYTHON_BIN" == */envs/"${VM_CONDA_ENV:-ve-cu124}"/* ]]; then
    ok "conda env=${VM_CONDA_ENV:-ve-cu124}"
  elif [[ -f "$ROOT/ve/.env_ve" ]]; then
    ok "来自 ve/.env_ve（与 Presence VE 同一解释器）"
  else
    warn "未检测到统一 conda env。建议: VM_CONDA_ENV=ve-cu124 ./setup_env.sh"
  fi
  verify_args=()
  if [[ -n "${DAE_TSE_REPO:-}" && -d "${DAE_TSE_REPO:-}" ]]; then
    verify_args+=(--dae-repo "$DAE_TSE_REPO")
    [[ -n "${DAE_TSE_CONFIG:-}" ]] && verify_args+=(--dae-config "$DAE_TSE_CONFIG")
    [[ -n "${DAE_TSE_CHECKPOINT:-}" ]] && verify_args+=(--dae-checkpoint "$DAE_TSE_CHECKPOINT")
    [[ "${REQUIRE_DAE_TSE:-0}" == "1" ]] && verify_args+=(--require-dae)
  fi
  set +e
  "$PYTHON_BIN" "$ROOT/environment/verify_cuda124.py" "${verify_args[@]}"
  verify_rc=$?
  set -e
  if [[ "$verify_rc" -eq 1 ]]; then
    err "统一 cu124 环境严格检查失败"
  elif [[ "$verify_rc" -eq 2 ]]; then
    echo "[INFO] 主环境通过；DAE-TSE 是可选候选，资产/兼容性未齐时保持 NO_GO"
  else
    ok "统一 cu124 环境严格检查通过"
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
  if "$PYTHON_BIN" -c "import clearvoice" >/dev/null 2>&1; then
    ok "clearvoice 已在 VE python（s2/s4/s5/s7/s8）"
  else
    warn "VE python 不能 import clearvoice；s2+ 需要: $PYTHON_BIN -m pip install -r requirements-optional.txt"
  fi
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

ok "CLEARVOICE_PYTHON=$CLEARVOICE_PYTHON （须与 PYTHON_BIN 同一 VE 环境）"
if [[ -n "$CLEARVOICE_PYTHON" && "$CLEARVOICE_PYTHON" != "$PYTHON_BIN" ]]; then
  warn "CLEARVOICE_PYTHON 与 PYTHON_BIN 不同；sep 分支应只用 VE python"
fi

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
echo "结论: 检查通过，可 ./run_sep.sh"
exit 0

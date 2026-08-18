#!/usr/bin/env bash
# 单阶段入口（运行）。不安装、不下载；代码已自包含于 VM/scripts。
# 默认跳过已完成阶段（不覆盖）；强制: --force 或 VM_FORCE=1
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$ROOT/scripts"
MEDIA="$(cd "$ROOT/.." && pwd)"

# 尽早修 OMP，避免任何 python 启动前 libgomp 报警
if [[ -z "${OMP_NUM_THREADS:-}" || ! "${OMP_NUM_THREADS}" =~ ^[0-9]+$ || "${OMP_NUM_THREADS}" -le 0 ]]; then
  export OMP_NUM_THREADS=8
fi
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"

STAGE="${1:-}"
[[ -n "$STAGE" ]] || {
  echo "用法: $0 collect|s1..s8|compare|eval|analyze [--force] [--retry-failed] [args...]" >&2
  exit 1
}
shift || true

if [[ -f "${VB_DIR:-/root/VB}/.env_clearvoice" ]]; then
  # shellcheck disable=SC1090
  source "${VB_DIR:-/root/VB}/.env_clearvoice" || true
fi
export CLEARVOICE_ROOT="${CLEARVOICE_ROOT:-/root/ClearerVoice-Studio}"

# AutoDL：大文件默认数据盘；未迁移时仍回退 /root/*
if [[ -d /root/autodl-tmp ]]; then
  [[ -f /root/autodl-tmp/env_paths.sh ]] && source /root/autodl-tmp/env_paths.sh || true
  [[ -f "$ROOT/.runtime/autodl_tmp_paths.sh" ]] && source "$ROOT/.runtime/autodl_tmp_paths.sh" || true
  export VM_OUT="${VM_OUT:-/root/autodl-tmp/vm}"
  if [[ -z "${MOSS_CKPT_DIR:-}" ]]; then
    if [[ -f /root/autodl-tmp/checkpoints/MossFormer2_ONNX/simple_model.onnx \
       || -f /root/autodl-tmp/checkpoints/MossFormer2_SS_16K/last_best_checkpoint.pt ]]; then
      export MOSS_CKPT_DIR=/root/autodl-tmp/checkpoints
    elif [[ -d /root/checkpoints ]]; then
      export MOSS_CKPT_DIR=/root/checkpoints
    else
      export MOSS_CKPT_DIR=/root/autodl-tmp/checkpoints
    fi
  fi
  if [[ -z "${DATA_DIR:-}" ]]; then
    if [[ -f /root/autodl-tmp/datasetA/pos.jsonl || -d /root/autodl-tmp/datasetA ]]; then
      export DATA_DIR=/root/autodl-tmp/datasetA
    else
      export DATA_DIR=/root/datasetA
    fi
  fi
  if [[ -z "${ASR_MODEL_DIR:-}" ]]; then
    if [[ -d /root/autodl-tmp/Qwen3-ASR-1.7B ]]; then
      export ASR_MODEL_DIR=/root/autodl-tmp/Qwen3-ASR-1.7B
    else
      export ASR_MODEL_DIR=/root/Qwen3-ASR-1.7B
    fi
  fi
  export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
  export TORCH_HOME="${TORCH_HOME:-/root/autodl-tmp/cache/torch}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/cache/pip}"
else
  export VM_OUT="${VM_OUT:-$MEDIA/vm_out}"
  export MOSS_CKPT_DIR="${MOSS_CKPT_DIR:-/root/checkpoints}"
  export DATA_DIR="${DATA_DIR:-/root/datasetA}"
  export ASR_MODEL_DIR="${ASR_MODEL_DIR:-/root/Qwen3-ASR-1.7B}"
fi

# 无效 CLEARVOICE_PYTHON 会阻断 ClearVoice：仅保留真实存在的路径
if [[ -n "${CLEARVOICE_PYTHON:-}" && ! -x "${CLEARVOICE_PYTHON}" ]]; then
  echo "[WARN] CLEARVOICE_PYTHON 无效，已清除: $CLEARVOICE_PYTHON"
  unset CLEARVOICE_PYTHON
fi
if [[ -z "${CLEARVOICE_PYTHON:-}" ]]; then
  for c in \
    /root/miniconda3/envs/ClearerVoice-Studio/bin/python \
    /root/miniconda3/envs/clearvoice/bin/python \
    /root/miniconda3/envs/ClearVoice/bin/python
  do
    if [[ -x "$c" ]]; then
      export CLEARVOICE_PYTHON="$c"
      break
    fi
  done
fi

# ONNX 吞吐：4090 默认更多 session + 批分离
export MOSS_NUM_SESSIONS="${MOSS_NUM_SESSIONS:-6}"
export VM_SEP_BATCH="${VM_SEP_BATCH:-8}"
export VM_SKIP_DONE="${VM_SKIP_DONE:-1}"

if [[ -f "$ROOT/.runtime/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.runtime/env.sh" || true
fi

# source 之后再修一次（平台/env 常把 OMP 设成空串或非法值）
if [[ -z "${OMP_NUM_THREADS:-}" || ! "${OMP_NUM_THREADS}" =~ ^[0-9]+$ || "${OMP_NUM_THREADS}" -le 0 ]]; then
  export OMP_NUM_THREADS=8
fi
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"
unset TRANSFORMERS_CACHE 2>/dev/null || true

python_has_qwen_asr() {
  local py="$1"
  [[ -x "$py" ]] || return 1
  "$py" -c "import qwen_asr" >/dev/null 2>&1
}

pick_python() {
  local cands=()
  [[ -n "${PYTHON_BIN:-}" ]] && cands+=("$PYTHON_BIN")
  [[ -f "$ROOT/.runtime/python_bin" ]] && cands+=("$(tr -d '[:space:]' <"$ROOT/.runtime/python_bin")")
  cands+=(
    "/root/miniconda3/envs/${VM_CONDA_ENV:-qwen3-asr}/bin/python"
    /root/miniconda3/envs/qwen3-asr/bin/python
    /root/autodl-tmp/envs/qwen3-asr/bin/python
    "$(command -v python3 || true)"
    "$(command -v python || true)"
  )
  local c
  for c in "${cands[@]}"; do
    [[ -n "$c" && -x "$c" ]] || continue
    if python_has_qwen_asr "$c"; then
      echo "$c"
      return 0
    fi
  done
  return 1
}

if ! PYTHON_BIN="$(pick_python)"; then
  echo "[ERR] 当前没有可用的 Python 含 qwen_asr。请 ./setup_env.sh && source ./env.sh" >&2
  exit 1
fi
echo "[INFO] PYTHON_BIN=$PYTHON_BIN"
"$PYTHON_BIN" -c "import qwen_asr, torch; print('[INFO] qwen_asr OK; torch', torch.__version__, 'cuda', torch.cuda.is_available())"

export PYTHONPATH="${SCRIPTS}:${PYTHONPATH:-}"
eval "$("$PYTHON_BIN" - <<'PY'
from pathlib import Path
import os
dirs = []
try:
    import torch
    site = Path(torch.__file__).resolve().parent.parent
    for sub in (
        "nvidia/cublas/lib",
        "nvidia/cufft/lib",
        "nvidia/cudnn/lib",
        "nvidia/cuda_runtime/lib",
        "nvidia/cuda_nvrtc/lib",
        "nvidia/nvjitlink/lib",
        "nvidia/curand/lib",
        "nvidia/cusolver/lib",
        "nvidia/cusparse/lib",
        "torch/lib",
    ):
        p = site / sub
        if p.is_dir():
            dirs.append(str(p))
except Exception:
    pass
for p in ("/usr/local/cuda/lib64", "/usr/local/cuda/targets/x86_64-linux/lib"):
    if Path(p).is_dir():
        dirs.append(p)
cur = os.environ.get("LD_LIBRARY_PATH", "")
merged = os.pathsep.join(dirs + ([cur] if cur else []))
print(f'export LD_LIBRARY_PATH={merged!r}')
if dirs:
    print(f'echo "[INFO] CUDA lib dirs (+{len(dirs)}): {dirs[0]} ..."')
else:
    print('echo "[WARN] 未找到 nvidia/cublas 等 CUDA lib；ORT 可能掉 CPU"')
PY
)"

if [[ "${VM_ALLOW_DOWNLOAD:-0}" != "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export VM_ALLOW_DOWNLOAD=0
fi

for f in utils_audio.py asr_backend.py mossformer2_ss.py mossformer2_onnx.py; do
  [[ -f "$SCRIPTS/$f" ]] || { echo "[ERR] 缺少整合模块 $SCRIPTS/$f" >&2; exit 1; }
done

LIMIT=0
THR=""
PEAK=0.7
MAX_SEP_SEC=6
DEVICE=cuda:0
SPLITS="${SPLITS:-pos,neg}"
SPLIT_ONE=""
FORCE=0
RETRY_FAILED=0
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit) LIMIT="$2"; shift 2 ;;
    --thr) THR="$2"; shift 2 ;;
    --peak) PEAK="$2"; shift 2 ;;
    --max-sep-sec) MAX_SEP_SEC="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --asr-model-dir) ASR_MODEL_DIR="$2"; shift 2 ;;
    --splits) SPLITS="$2"; shift 2 ;;
    --split) SPLIT_ONE="$2"; shift 2 ;;
    --vm-out) VM_OUT="$2"; shift 2 ;;
    --force) FORCE=1; export VM_FORCE=1; shift ;;
    --retry-failed) RETRY_FAILED=1; export VM_RETRY_FAILED=1; shift ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

if [[ -n "$SPLIT_ONE" ]]; then
  SPLITS="$SPLIT_ONE"
fi

echo "============================================"
echo " VM stage=$STAGE  VM_OUT=$VM_OUT"
echo " SPLITS=$SPLITS  LIMIT=$LIMIT  FORCE=$FORCE SKIP_DONE=${VM_SKIP_DONE}"
echo " ONNX: MOSS_NUM_SESSIONS=$MOSS_NUM_SESSIONS VM_SEP_BATCH=$VM_SEP_BATCH"
echo " CLEARVOICE_PYTHON=${CLEARVOICE_PYTHON:-<unset, will probe>}"
echo " offline: HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}"
echo "============================================"

run_one_split() {
  local split="$1"
  echo ""
  echo "---- split=$split ----"
  local force_flag=()
  local retry_flag=()
  [[ "$FORCE" == "1" ]] && force_flag=(--force)
  [[ "$RETRY_FAILED" == "1" ]] && retry_flag=(--retry-failed)
  case "$STAGE" in
    s1|s2)
      "$PYTHON_BIN" "$SCRIPTS/stage_full_sep.py" --stage "$STAGE" --split "$split" \
        --vm-out "$VM_OUT" --limit "$LIMIT" --peak "$PEAK" \
        --max-sep-sec "$MAX_SEP_SEC" --device "$DEVICE" \
        --asr-model-dir "$ASR_MODEL_DIR" \
        ${force_flag[@]+"${force_flag[@]}"} \
        ${retry_flag[@]+"${retry_flag[@]}"} ${EXTRA[@]+"${EXTRA[@]}"}
      ;;
    s3|s4)
      "$PYTHON_BIN" "$SCRIPTS/stage_cascade.py" --stage "$STAGE" --split "$split" \
        --vm-out "$VM_OUT" --limit "$LIMIT" --peak "$PEAK" \
        --max-sep-sec "$MAX_SEP_SEC" --device "$DEVICE" \
        --asr-model-dir "$ASR_MODEL_DIR" \
        ${force_flag[@]+"${force_flag[@]}"} \
        ${retry_flag[@]+"${retry_flag[@]}"} ${EXTRA[@]+"${EXTRA[@]}"}
      ;;
    s5|s6|s7|s8)
      local thr_flag=()
      [[ -n "$THR" ]] && thr_flag=(--thr "$THR")
      "$PYTHON_BIN" "$SCRIPTS/stage_gated.py" --stage "$STAGE" --split "$split" \
        --vm-out "$VM_OUT" --limit "$LIMIT" --peak "$PEAK" \
        --max-sep-sec "$MAX_SEP_SEC" --device "$DEVICE" \
        --asr-model-dir "$ASR_MODEL_DIR" \
        ${thr_flag[@]+"${thr_flag[@]}"} \
        ${force_flag[@]+"${force_flag[@]}"} \
        ${retry_flag[@]+"${retry_flag[@]}"} ${EXTRA[@]+"${EXTRA[@]}"}
      ;;
    *)
      echo "[ERR] run_one_split 不支持 $STAGE" >&2
      exit 1
      ;;
  esac
}

case "$STAGE" in
  collect)
    "$PYTHON_BIN" "$SCRIPTS/collect_kws.py" \
      --data-dir "$DATA_DIR" --vm-out "$VM_OUT" --splits "$SPLITS" \
      --limit "$LIMIT" ${EXTRA[@]+"${EXTRA[@]}"}
    ;;
  s1|s2|s3|s4|s5|s6|s7|s8)
    IFS=',' read -r -a SPLIT_ARR <<< "$SPLITS"
    for sp in "${SPLIT_ARR[@]}"; do
      sp="${sp#"${sp%%[![:space:]]*}"}"
      sp="${sp%"${sp##*[![:space:]]}"}"
      [[ -n "$sp" ]] || continue
      run_one_split "$sp"
    done
    ;;
  compare)
    "$PYTHON_BIN" "$SCRIPTS/compare_all.py" --vm-out "$VM_OUT" --splits "$SPLITS"
    ;;
  eval|report)
    "$PYTHON_BIN" "$SCRIPTS/eval_report.py" --vm-out "$VM_OUT" --splits "$SPLITS"
    ;;
  analyze|analysis)
    "$PYTHON_BIN" "$SCRIPTS/analyze_results.py" --vm-out "$VM_OUT" --splits "$SPLITS"
    ;;
  *)
    echo "[ERR] unknown stage $STAGE" >&2
    exit 1
    ;;
esac

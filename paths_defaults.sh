#!/usr/bin/env bash
# 被其它 *.sh source：统一 AutoDL 数据盘路径。只填「尚未设置」的变量。
# 仓库根 = 本文件所在目录。
if [[ -z "${VM_ROOT:-}" ]]; then
  VM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
export VM_ROOT
export VM_CONDA_ENV="${VM_CONDA_ENV:-ve}"

# 迁移脚本写出的路径优先
if [[ -f /root/autodl-tmp/env_paths.sh ]]; then
  # shellcheck disable=SC1091
  source /root/autodl-tmp/env_paths.sh || true
fi
if [[ -f "${VM_ROOT}/.runtime/autodl_tmp_paths.sh" ]]; then
  # shellcheck disable=SC1091
  source "${VM_ROOT}/.runtime/autodl_tmp_paths.sh" || true
fi

if [[ -d /root/autodl-tmp ]]; then
  if [[ -f "${VM_ROOT}/.sep-only" ]]; then
    export VM_OUT="${VM_OUT:-/root/autodl-tmp/kws_sep}"
  else
    export VM_OUT="${VM_OUT:-/root/autodl-tmp/vm}"
  fi
  export DATA_DIR="${DATA_DIR:-}"
  if [[ -z "${DATA_DIR}" ]]; then
    if [[ -f /root/autodl-tmp/datasetA/pos.jsonl || -d /root/autodl-tmp/datasetA ]]; then
      export DATA_DIR=/root/autodl-tmp/datasetA
    elif [[ -f /root/datasetA/pos.jsonl ]]; then
      export DATA_DIR=/root/datasetA
    else
      export DATA_DIR=/root/autodl-tmp/datasetA
    fi
  fi
  if [[ -z "${MOSS_CKPT_DIR:-}" ]]; then
    if [[ -f /root/autodl-tmp/checkpoints/MossFormer2_ONNX/simple_model.onnx \
       || -f /root/autodl-tmp/checkpoints/MossFormer2_SS_16K/last_best_checkpoint.pt ]]; then
      export MOSS_CKPT_DIR=/root/autodl-tmp/checkpoints
    elif [[ -f /root/checkpoints/MossFormer2_ONNX/simple_model.onnx \
         || -f /root/checkpoints/MossFormer2_SS_16K/last_best_checkpoint.pt ]]; then
      export MOSS_CKPT_DIR=/root/checkpoints
    else
      export MOSS_CKPT_DIR=/root/autodl-tmp/checkpoints
    fi
  fi
  if [[ -z "${ASR_MODEL_DIR:-}" ]]; then
    if [[ -d /root/autodl-tmp/Qwen3-ASR-1.7B ]]; then
      export ASR_MODEL_DIR=/root/autodl-tmp/Qwen3-ASR-1.7B
    elif [[ -d /root/Qwen3-ASR-1.7B ]]; then
      export ASR_MODEL_DIR=/root/Qwen3-ASR-1.7B
    else
      export ASR_MODEL_DIR=/root/autodl-tmp/Qwen3-ASR-1.7B
    fi
  fi
  export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
  export TORCH_HOME="${TORCH_HOME:-/root/autodl-tmp/cache/torch}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/cache/pip}"
else
  export VM_OUT="${VM_OUT:-$VM_ROOT/../vm_out}"
  export MOSS_CKPT_DIR="${MOSS_CKPT_DIR:-/root/checkpoints}"
  export DATA_DIR="${DATA_DIR:-/root/datasetA}"
  export ASR_MODEL_DIR="${ASR_MODEL_DIR:-/root/Qwen3-ASR-1.7B}"
fi

export MOSS_ONNX_PATH="${MOSS_ONNX_PATH:-$MOSS_CKPT_DIR/MossFormer2_ONNX/simple_model.onnx}"
export MOSS_SS_DIR="${MOSS_SS_DIR:-$MOSS_CKPT_DIR/MossFormer2_SS_16K}"

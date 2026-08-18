#!/usr/bin/env bash
# VE 真实 CER（Qwen3-ASR-1.7B）
# 用法:
#   VE_OUT=/root/autodl-tmp/ve_ps4 ./run_asr_cer.sh
#   VE_OUT=/root/autodl-tmp/ve_wesep ./run_asr_cer.sh --limit 20
#   PIPELINE=ps4|wesep|sep_route|mix ./run_asr_cer.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$ROOT/.env_ve" ]] && source "$ROOT/.env_ve" || true
export PYTHONPATH="$ROOT/scripts:${ROOT}/../scripts:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export ASR_MODEL_DIR="${ASR_MODEL_DIR:-${QWEN3_ASR_DIR:-/root/autodl-tmp/Qwen3-ASR-1.7B}}"

# 与 run_all 对齐：可用 PIPELINE 推断 VE_OUT
if [[ -z "${VE_OUT:-}" ]]; then
  if [[ -n "${PIPELINE:-}" ]]; then
    pl="$(echo "$PIPELINE" | tr '[:upper:]' '[:lower:]')"
    case "$pl" in
      wesep|wesep_bsrnn) pl=wesep ;;
      sep_route|mossformer|route|sep) pl=sep_route ;;
      mix|passthrough|cmd|none) pl=mix ;;
      *) pl=ps4 ;;
    esac
    VE_OUT="/root/autodl-tmp/ve_${pl}"
  else
    VE_OUT="${VE_OUT_BASE:-/root/autodl-tmp/ve}_ps4"
    if [[ ! -d "$VE_OUT/results" && -d /root/autodl-tmp/ve/results ]]; then
      VE_OUT=/root/autodl-tmp/ve
    fi
  fi
fi
export VE_OUT

echo "=== VE run_asr_cer ==="
echo "VE_OUT=$VE_OUT"
echo "ASR 模型目录: $ASR_MODEL_DIR"

ASR_CLI=(
  --ve-out "$VE_OUT"
  --model-dir "$ASR_MODEL_DIR"
  --device "${DEVICE:-cuda:0}"
)
[[ -n "${ASR_LANGUAGE:-}" ]] && ASR_CLI+=(--language "$ASR_LANGUAGE")
[[ "${ASR_HOTWORDS:-0}" == "1" ]] && ASR_CLI+=(--hotwords)
[[ "${ASR_DOMAIN_CONTEXT:-0}" == "1" ]] && ASR_CLI+=(--domain-context)
[[ "${ASR_RETRY_MISMATCH:-0}" == "1" ]] && ASR_CLI+=(--retry-mismatch)

"${PYTHON_BIN:-python}" "$ROOT/scripts/asr_cer.py" "${ASR_CLI[@]}" "$@"

#!/usr/bin/env bash
# 下载 Qwen3-ASR-1.7B 到 ASR_MODEL_DIR（默认 /root/autodl-tmp/Qwen3-ASR-1.7B，与 VM 一致）
# 用法:
#   ./download_qwen3_asr.sh                        # HF 优先，失败自动回退 ModelScope
#   ./download_qwen3_asr.sh --source modelscope    # 强制走 ModelScope
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$ROOT/.env_ve" ]] && source "$ROOT/.env_ve" || true
export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/root/autodl-tmp/cache/modelscope}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export ASR_MODEL_DIR="${ASR_MODEL_DIR:-${QWEN3_ASR_DIR:-/root/autodl-tmp/Qwen3-ASR-1.7B}}"

echo "=== 下载 Qwen3-ASR-1.7B → $ASR_MODEL_DIR ==="
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true
"${PYTHON_BIN:-python}" "$ROOT/scripts/download_qwen3_asr.py" \
  --model-id Qwen/Qwen3-ASR-1.7B \
  --out-dir "$ASR_MODEL_DIR" \
  "$@"

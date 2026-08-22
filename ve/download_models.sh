#!/usr/bin/env bash
# 下载 VE 推理模型到 /root/autodl-tmp/ve_models（或 VE_MODEL_DIR）
# PS4: https://huggingface.co/TaurenMountain/PS4 （仅 checkpoint + inference.py）
# 不下：Whisper / DNSMOS / MossFormer / wesep 训练仓
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
[[ -f "$ROOT/.env_ve" ]] && source "$ROOT/.env_ve" || true

MODEL_DIR="${VE_MODEL_DIR:-/root/autodl-tmp/ve_models}"
export VE_MODEL_DIR="$MODEL_DIR"
mkdir -p "$MODEL_DIR"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/datasetA}"

if [[ -f /etc/network_turbo ]]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo || true
fi
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/root/autodl-tmp/cache/modelscope}"
mkdir -p "$HF_HOME" "$MODELSCOPE_CACHE"

echo "[INFO] MODEL_DIR=$MODEL_DIR"
echo "[INFO] DATA_DIR=$DATA_DIR"
echo "[INFO] BEST_SEP_DIR=${BEST_SEP_DIR:-"(auto: pos_neg/best_sep 或 kws_sep/best_sep)"}"
echo "[INFO] HF_ENDPOINT=$HF_ENDPOINT"

check_data() {
  local name="$1" path="$2" hint="$3"
  if [[ -d "$path" ]]; then
    echo "[ OK ] $name → $path"
  else
    echo "[WARN] 缺少 $name: $path"
    echo "       $hint"
  fi
}
check_data "datasetA" "$DATA_DIR" \
  "请将仓库 datasetA/ 同步或软链到 /root/autodl-tmp/datasetA（勿放系统盘）"
if [[ -n "${BEST_SEP_DIR:-}" ]]; then
  check_data "best_sep" "$BEST_SEP_DIR" \
    "export BEST_SEP_DIR=/path/to/best_sep（sep 产物）"
  if [[ -d "$BEST_SEP_DIR" && ! -f "$BEST_SEP_DIR/index.jsonl" ]]; then
    echo "[WARN] best_sep 缺 index.jsonl: $BEST_SEP_DIR/index.jsonl"
  fi
else
  echo "[INFO] BEST_SEP_DIR 未设；build_manifest 将扫描 pos_neg/best_sep 与 kws_sep/best_sep"
fi

link_if_missing() {
  local dst="$1" src="$2"
  if [[ ! -e "$dst" && -e "$src" ]]; then
    mkdir -p "$(dirname "$dst")"
    ln -s "$src" "$dst"
    echo "[INFO] 软链 $dst -> $src"
  fi
}
if [[ -d /root/autodl-tmp ]]; then
  link_if_missing "$DATA_DIR" "$ROOT/../datasetA"
  if [[ -n "${BEST_SEP_DIR:-}" ]]; then
    link_if_missing "$BEST_SEP_DIR" "$ROOT/../pos_neg/best_sep"
  fi
fi

# 确保 huggingface_hub 可用
"$PYTHON_BIN" -c "import huggingface_hub" 2>/dev/null || \
  "$PYTHON_BIN" -m pip install -U huggingface_hub \
    -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

echo "[INFO] 调用 scripts/download_models.py …"
"$PYTHON_BIN" "$ROOT/scripts/download_models.py" --model-dir "$MODEL_DIR"
rc=$?

PS4_DIR="$MODEL_DIR/PS4"
if [[ ! -f "$PS4_DIR/checkpoint_epoch037.pt" ]] || [[ ! -f "$PS4_DIR/inference.py" ]]; then
  echo "[FAIL] PS4 未就绪。需要:"
  echo "       $PS4_DIR/checkpoint_epoch037.pt"
  echo "       $PS4_DIR/inference.py"
  echo "       来源: https://huggingface.co/TaurenMountain/PS4"
  exit 2
fi

echo "[OK] download_models.sh 完成"
echo "     PS4=$PS4_DIR"
ls -lah "$PS4_DIR" || true
exit "$rc"

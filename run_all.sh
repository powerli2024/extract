#!/usr/bin/env bash
# 运行全流程（不安装、不下载）。pos/neg 分树；中间音频隔离。
# 环境: ./setup_env.sh && ./download_models.sh && ./check_env.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDIA="$(cd "$ROOT/.." && pwd)"

STAGES_DEFAULT="collect,s1,s2,s3,s4,s5,s6,s7,s8,compare,eval,analyze"
STAGES="$STAGES_DEFAULT"
PACK="${PACK:-1}"
SPLITS="${SPLITS:-pos,neg}"
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stages) STAGES="$2"; shift 2 ;;
    --limit) EXTRA+=(--limit "$2"); shift 2 ;;
    --thr) EXTRA+=(--thr "$2"); shift 2 ;;
    --no-pack) PACK=0; shift ;;
    --splits) SPLITS="$2"; shift 2 ;;
    --split) SPLITS="$2"; shift 2 ;;
    --vm-out) VM_OUT="$2"; shift 2 ;;
    --force) EXTRA+=(--force); export VM_FORCE=1; shift ;;
    --retry-failed) EXTRA+=(--retry-failed); export VM_RETRY_FAILED=1; shift ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

if [[ -f "${VB_DIR:-/root/VB}/.env_clearvoice" ]]; then
  # shellcheck disable=SC1090
  source "${VB_DIR:-/root/VB}/.env_clearvoice" || true
fi
if [[ -d /root/autodl-tmp ]]; then
  export VM_OUT="${VM_OUT:-/root/autodl-tmp/vm}"
else
  export VM_OUT="${VM_OUT:-$MEDIA/vm_out}"
fi
export DATA_DIR="${DATA_DIR:-/root/datasetA}"
export SPLITS

if [[ "${VM_ALLOW_DOWNLOAD:-0}" != "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
fi

export VM_SKIP_DONE="${VM_SKIP_DONE:-1}"
export MOSS_NUM_SESSIONS="${MOSS_NUM_SESSIONS:-6}"
export VM_SEP_BATCH="${VM_SEP_BATCH:-8}"

echo "============================================"
echo " VM run_all"
echo " STAGES=$STAGES"
echo " SPLITS=$SPLITS"
echo " VM_OUT=$VM_OUT"
echo " SKIP_DONE=$VM_SKIP_DONE (已完成阶段默认跳过，不覆盖)"
echo " ONNX sessions=$MOSS_NUM_SESSIONS batch=$VM_SEP_BATCH"
echo "============================================"

IFS=',' read -r -a ARR <<< "$STAGES"
for s in "${ARR[@]}"; do
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  [[ -n "$s" ]] || continue
  echo ""
  echo "######## STAGE $s ########"
  bash "$ROOT/run_stage.sh" "$s" --splits "$SPLITS" --vm-out "$VM_OUT" ${EXTRA[@]+"${EXTRA[@]}"}
done

if [[ "$PACK" == "1" ]]; then
  mkdir -p "$VM_OUT/packs"
  TS="$(date +%Y%m%d_%H%M%S)"
  ARCHIVE="$VM_OUT/packs/vm_${TS}.tar.gz"
  PARENT="$(cd "$(dirname "$VM_OUT")" && pwd)"
  BASE="$(basename "$VM_OUT")"
  echo "[INFO] 打包 → $ARCHIVE"
  if tar -czf "$ARCHIVE" -C "$PARENT" --exclude="$BASE/packs" "$BASE"; then
    :
  else
    tar -czf "$ARCHIVE" -C "$PARENT" "$BASE"
  fi
  ln -sfn "$(basename "$ARCHIVE")" "$VM_OUT/packs/vm_latest.tar.gz" 2>/dev/null || true
  echo "[ OK ] $ARCHIVE"
fi

echo "全部完成 → $VM_OUT"
echo "报告: $VM_OUT/reports/eval_report.md  以及 $VM_OUT/{pos,neg}/reports/"

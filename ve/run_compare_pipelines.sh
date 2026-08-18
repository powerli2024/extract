#!/usr/bin/env bash
# 四方案对照：共享 Presence 校准（按 VAD/sep/lang 分桶），再分别提取 + ASR
#
#   ps4 | wesep | sep_route | mix
#
# 用法（AutoDL）:
#   LIMIT=64 SKIP_ASR=1 ./run_compare_pipelines.sh
#   ENROLL_VAD=0 ./run_compare_pipelines.sh          # 与 VAD=1 并存，不同 VE_OUT
#   HOLDOUT_FRAC=0.3 ./run_compare_pipelines.sh       # thr 在 holdout 上评估
#   PIPELINES=ps4,mix ./run_compare_pipelines.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
[[ -f "$ROOT/.env_ve" ]] && source "$ROOT/.env_ve" || true

export ENROLL_VAD="${ENROLL_VAD:-1}"
export USE_SEP="${USE_SEP:-1}"
export LANG_SPLIT="${LANG_SPLIT:-1}"
export PRESENCE_BACKEND="${PRESENCE_BACKEND:-eres2netv2}"
export HOLDOUT_FRAC="${HOLDOUT_FRAC:-0}"
export FORCE_CALIB="${FORCE_CALIB:-0}"
export CALIB_ROOT="${CALIB_ROOT:-/root/autodl-tmp/ve_presence_best}"

if [[ "$ENROLL_VAD" == "1" ]]; then VAD_TAG=vad; else VAD_TAG=novad; fi
if [[ "$USE_SEP" == "1" ]]; then SEP_TAG=sep1; else SEP_TAG=nosep; fi
if [[ "$LANG_SPLIT" == "1" ]]; then LS_TAG=ls; else LS_TAG=gthr; fi
export CALIB_DIR="${CALIB_DIR:-$CALIB_ROOT/reports/presence_calib_${PRESENCE_BACKEND}_${SEP_TAG}_${LS_TAG}_${VAD_TAG}_raw}"

LIMIT="${LIMIT:-0}"
SKIP_ASR="${SKIP_ASR:-0}"
PIPELINES_CSV="${PIPELINES:-ps4,wesep,sep_route,mix}"

echo "=== run_compare_pipelines ==="
echo "CALIB_DIR=$CALIB_DIR ENROLL_VAD=$ENROLL_VAD HOLDOUT_FRAC=$HOLDOUT_FRAC"
echo "PIPELINES=$PIPELINES_CSV LIMIT=$LIMIT SKIP_ASR=$SKIP_ASR"
echo "产物目录: /root/autodl-tmp/ve_<pipeline>_${VAD_TAG}/"

IFS=',' read -r -a PLS <<< "$PIPELINES_CSV"
first=1
for pl in "${PLS[@]}"; do
  pl="$(echo "$pl" | tr '[:upper:]' '[:lower:]' | xargs)"
  [[ -z "$pl" ]] && continue
  echo
  echo "########## PIPELINE=$pl VAD=$VAD_TAG ##########"
  if [[ "$first" == "1" ]]; then
    SKIP_CALIB=0
    first=0
  else
    SKIP_CALIB=1
  fi
  if [[ "$FORCE_CALIB" == "1" && "$SKIP_CALIB" == "0" ]]; then
    export FORCE_CALIB=1
  else
    export FORCE_CALIB=0
  fi
  # 不显式锁死 VE_OUT：交给 run_all 按 pipeline+vad 命名，保证并存
  PIPELINE="$pl" \
    SKIP_CALIB="$SKIP_CALIB" \
    LIMIT="$LIMIT" \
    SKIP_ASR="$SKIP_ASR" \
    bash "$ROOT/run_all.sh"
done

echo
echo "=== 汇总 CER 报告路径 ==="
for pl in "${PLS[@]}"; do
  pl="$(echo "$pl" | tr '[:upper:]' '[:lower:]' | xargs)"
  [[ -z "$pl" ]] && continue
  s="/root/autodl-tmp/ve_${pl}_${VAD_TAG}/reports/asr_cer/summary.md"
  if [[ -f "$s" ]]; then
    echo "---- $pl ($VAD_TAG) ----"
    head -n 25 "$s" || true
  else
    echo "[ ] $pl 尚无 asr_cer: $s"
  fi
done
echo "thr: $CALIB_DIR/recommended_thr.json"
echo "done."

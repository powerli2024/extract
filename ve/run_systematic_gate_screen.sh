#!/usr/bin/env bash
# 多组 KWS × 多门控策略的第一阶段筛选。下游波形固定为原始 CMD mix，避免混入 TSE 差异。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$ROOT/.env_ve" ]] && source "$ROOT/.env_ve" || true

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/datasetA}"
EXP_ROOT="${EXP_ROOT:-/root/autodl-tmp/ve_systematic}"
KWS_CANDIDATES="${KWS_CANDIDATES:-}"
SEEDS="${SEEDS:-100}"
HOLDOUT_FRAC="${HOLDOUT_FRAC:-0.30}"
TOP_K="${TOP_K:-3}"
THRESHOLD_MODES="${THRESHOLD_MODES:-global,lang_split}"
if [[ -z "$KWS_CANDIDATES" ]]; then
  echo '[ERR] KWS_CANDIDATES 为空；格式: name=/abs/kws1;name2=/abs/kws2' >&2
  exit 2
fi
mkdir -p "$EXP_ROOT"
IFS=';' read -r -a ITEMS <<< "$KWS_CANDIDATES"

first_spec="${ITEMS[0]}"
first_dir="${first_spec#*=}"
ALL_POS_MIX_OUT="${ALL_POS_MIX_OUT:-$EXP_ROOT/all_pos_mix}"
if [[ ! -f "$ALL_POS_MIX_OUT/reports/asr_cer/asr_results.jsonl" || "${FORCE_ALL_POS_ASR:-0}" == "1" ]]; then
  echo '>>> shared all-positive mix ASR'
  BEST_SEP_DIR="$first_dir" DATA_DIR="$DATA_DIR" ALL_POS_MIX_OUT="$ALL_POS_MIX_OUT" \
    STRICT_ENROLL=1 LIMIT=0 ASR_RESUME=0 bash "$ROOT/run_all_pos_mix_asr.sh"
else
  echo "[REUSE] all-positive mix ASR: $ALL_POS_MIX_OUT/reports/asr_cer/asr_results.jsonl"
fi

RANK_ARGS=()
for spec in "${ITEMS[@]}"; do
  name="${spec%%=*}"
  kws="${spec#*=}"
  if [[ -z "$name" || "$name" == "$kws" || ! -d "$kws" ]]; then
    echo "[ERR] invalid KWS candidate: $spec" >&2; exit 2
  fi
  arm="$EXP_ROOT/gate/$name"
  echo ">>> KWS=$name presence scoring"
  BEST_SEP_DIR="$kws" VE_OUT="$arm" DATA_DIR="$DATA_DIR" PIPELINE=mix \
    LOCKED_THR=1 ENROLL_VAD=0 CMD_SE=0 SAVE_SEP_WAVS=0 EXTRA_REJECT=0 \
    SKIP_ASR=1 STRICT_ENROLL=1 STRICT_EVAL=0 LIMIT=0 ASR_RESUME=0 \
    SEP_REUSE_ROOT="${SEP_REUSE_ROOT:-}" STRICT_SEP_REUSE="${STRICT_SEP_REUSE:-0}" \
    bash "$ROOT/run_all.sh"
  opt="$arm/reports/gate_score_opt"
  "$PYTHON_BIN" "$ROOT/scripts/optimize_gate_for_score.py" \
    --decisions "$arm/results/all_results.jsonl" \
    --asr-all-pos "$ALL_POS_MIX_OUT/reports/asr_cer/asr_results.jsonl" \
    --baseline-thr "$ROOT/configs/locked_thr.json" --out-dir "$opt" \
    --holdout-frac "$HOLDOUT_FRAC" --seeds "$SEEDS" \
    --threshold-modes "$THRESHOLD_MODES" --no-enroll-vad --strict
  RANK_ARGS+=(--candidate "$name=$opt")
done

"$PYTHON_BIN" "$ROOT/scripts/rank_gate_candidates.py" \
  "${RANK_ARGS[@]}" --top-k "$TOP_K" --out-dir "$EXP_ROOT/ranking"
echo "[OK] gate screen complete: $EXP_ROOT/ranking/gate_ranking.md"

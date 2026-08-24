#!/usr/bin/env bash
# 对所有正样本强制走原始 CMD mix ASR，为门控降阈值/策略提供真实 CER。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$ROOT/.env_ve" ]] && source "$ROOT/.env_ve" || true

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/datasetA}"
OUT="${ALL_POS_MIX_OUT:-/root/autodl-tmp/ve_goal/all_pos_mix}"
SAMPLES="${SAMPLES:-}"
DEVICE="${DEVICE:-cuda:0}"
LIMIT="${LIMIT:-0}"

mkdir -p "$OUT/manifest" "$OUT/results" "$OUT/extracted" "$OUT/reports"
if [[ -z "$SAMPLES" ]]; then
  BEST_SEP_ARGS=()
  [[ -n "${BEST_SEP_DIR:-}" ]] && BEST_SEP_ARGS+=(--best-sep "$BEST_SEP_DIR")
  [[ "${STRICT_ENROLL:-1}" == "1" ]] && BEST_SEP_ARGS+=(--strict-best-sep)
  "$PYTHON_BIN" "$ROOT/scripts/build_manifest.py" \
    --data-dir "$DATA_DIR" "${BEST_SEP_ARGS[@]}" --out-dir "$OUT/manifest"
  SAMPLES="$OUT/manifest/samples.jsonl"
else
  cp -f "$SAMPLES" "$OUT/manifest/samples.jsonl"
fi

ARGS=(
  --samples "$SAMPLES" --out-dir "$OUT" --splits pos
  --presence-backend "${PRESENCE_BACKEND:-eres2netv2}"
  --thr-file "$ROOT/configs/locked_thr.json"
  --device "$DEVICE" --tse-backend mix --force-extract
  --use-sep --sep-depth 1 --no-enroll-vad --no-score-norm --stream-policy max
)
[[ "$LIMIT" != "0" ]] && ARGS+=(--limit "$LIMIT")
"$PYTHON_BIN" "$ROOT/scripts/run_extract.py" "${ARGS[@]}"

ASR_ARGS=(--no-resume)
[[ "$LIMIT" != "0" ]] && ASR_ARGS+=(--limit "$LIMIT")
ASR_RETRY_MISMATCH="${ASR_RETRY_MISMATCH:-1}" VE_OUT="$OUT" \
  bash "$ROOT/run_asr_cer.sh" "${ASR_ARGS[@]}"

echo "[OK] all-pos mix ASR: $OUT/reports/asr_cer/asr_results.jsonl"
echo "下一步（DECISIONS 需来自同一 manifest、且含 mix/d1 sim_streams）:"
echo "python $ROOT/scripts/optimize_gate_for_score.py --decisions \"\$DECISIONS\" --asr-all-pos '$OUT/reports/asr_cer/asr_results.jsonl' --baseline-thr '$ROOT/configs/locked_thr.json' --out-dir '$OUT/reports/gate_score_opt' --strict"

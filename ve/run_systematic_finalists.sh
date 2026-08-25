#!/usr/bin/env bash
# 对入围 KWS/策略逐个下游臂做 all-positive ASR、臂内阈值优化和严格计分。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$ROOT/.env_ve" ]] && source "$ROOT/.env_ve" || true
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/datasetA}"
EXP_ROOT="${EXP_ROOT:-/root/autodl-tmp/ve_systematic}"
FINALISTS="${FINALISTS:-}"
PIPELINES="${PIPELINES:-mix,sep_route,adaptive_route,wesep,ps4,cond_tasnet}"
if [[ -z "$FINALISTS" ]]; then
  echo '[ERR] FINALISTS 为空；格式: name|/kws|/gate_score_opt|policy;name2|...' >&2
  exit 2
fi
IFS=';' read -r -a ITEMS <<< "$FINALISTS"
RANK_ARGS=()
BASELINE_RANK_ARGS=()
first_finalist=1
for spec in "${ITEMS[@]}"; do
  IFS='|' read -r name kws opt policy <<< "$spec"
  if [[ -z "$name" || ! -d "$kws" || ! -f "$opt/recommended_thr.json" || -z "$policy" ]]; then
    echo "[ERR] invalid finalist: $spec" >&2; exit 2
  fi
  arm_root="$EXP_ROOT/final/$name"
  screen_arm="$(cd "$opt/../.." && pwd)"
  screen_decisions="$screen_arm/results/all_results.jsonl"
  screen_samples="$screen_arm/manifest/samples.jsonl"
  [[ -f "$screen_decisions" && -f "$screen_samples" ]] || {
    echo "[ERR] finalist 缺少第一阶段 decisions/manifest: $screen_arm" >&2; exit 2;
  }
  echo ">>> finalist=$name policy=$policy pipelines=$PIPELINES"
  IFS=',' read -r -a PLS <<< "$PIPELINES"
  for pl in "${PLS[@]}"; do
    pl="$(echo "$pl" | tr '[:upper:]' '[:lower:]' | xargs)"
    [[ -n "$pl" ]] || continue
    pl_root="$arm_root/$pl"
    all_pos="$pl_root/all_pos"
    gate_opt="$pl_root/gate_opt"
    eval_root="$pl_root/eval"
    echo ">>> $name/$pl all-positive extraction + ASR"
    if [[ "${FORCE_ALL_POS:-0}" != "1" \
          && -f "$all_pos/results/all_results.jsonl" \
          && -f "$all_pos/reports/asr_cer/asr_results.jsonl" ]]; then
      echo "[REUSE] $name/$pl all-positive extracted + ASR（后续 strict coverage 会复核）"
    else
      BEST_SEP_DIR="$kws" DATA_DIR="$DATA_DIR" ALL_POS_BACKEND="$pl" \
        ALL_POS_MIX_OUT="$all_pos" SEP_REUSE_ROOT="${SEP_REUSE_ROOT:-}" \
        STRICT_SEP_REUSE="${STRICT_SEP_REUSE:-0}" ASR_RESUME=0 \
        bash "$ROOT/run_all_pos_mix_asr.sh"
    fi
    "$PYTHON_BIN" "$ROOT/scripts/optimize_gate_for_score.py" \
      --decisions "$screen_decisions" \
      --asr-all-pos "$all_pos/reports/asr_cer/asr_results.jsonl" \
      --baseline-thr "$ROOT/configs/locked_thr.json" --out-dir "$gate_opt" \
      --policies "$policy" --threshold-modes "${THRESHOLD_MODES:-global,lang_split}" \
      --holdout-frac "${HOLDOUT_FRAC:-0.30}" --seeds "${SEEDS:-200}" \
      --no-enroll-vad --strict
    "$PYTHON_BIN" "$ROOT/scripts/apply_gate_config.py" \
      --decisions "$screen_decisions" --thr-file "$gate_opt/recommended_thr.json" \
      --out "$eval_root/results/all_results.jsonl" --strict
    mkdir -p "$eval_root/manifest"
    cp -f "$screen_samples" "$eval_root/manifest/samples.jsonl"
    "$PYTHON_BIN" "$ROOT/scripts/final_evaluate.py" \
      --ve-out "$eval_root" --samples "$screen_samples" \
      --decisions "$eval_root/results/all_results.jsonl" \
      --asr "$all_pos/reports/asr_cer/asr_results.jsonl" \
      --out-dir "$eval_root/reports/final_eval" --strict
    RANK_ARGS+=(--candidate "$name.$pl=$eval_root")
    if [[ "$first_finalist" == "1" && "$pl" == "${BASELINE_PIPELINE:-sep_route}" ]]; then
      baseline_root="$EXP_ROOT/final/locked_baseline_$pl"
      "$PYTHON_BIN" "$ROOT/scripts/apply_gate_config.py" \
        --decisions "$screen_decisions" --thr-file "$ROOT/configs/locked_thr.json" \
        --out "$baseline_root/results/all_results.jsonl" --strict
      mkdir -p "$baseline_root/manifest"
      cp -f "$screen_samples" "$baseline_root/manifest/samples.jsonl"
      "$PYTHON_BIN" "$ROOT/scripts/final_evaluate.py" \
        --ve-out "$baseline_root" --samples "$screen_samples" \
        --decisions "$baseline_root/results/all_results.jsonl" \
        --asr "$all_pos/reports/asr_cer/asr_results.jsonl" \
        --out-dir "$baseline_root/reports/final_eval" --strict
      BASELINE_RANK_ARGS=(--candidate "locked.$pl=$baseline_root")
    fi
  done
  first_finalist=0
done
if [[ "${#BASELINE_RANK_ARGS[@]}" == "0" ]]; then
  echo "[ERR] BASELINE_PIPELINE=${BASELINE_PIPELINE:-sep_route} 不在 PIPELINES=$PIPELINES" >&2
  exit 2
fi
"$PYTHON_BIN" "$ROOT/scripts/rank_final_candidates.py" \
  "${BASELINE_RANK_ARGS[@]}" "${RANK_ARGS[@]}" --out-dir "$EXP_ROOT/final_ranking" \
  --replicates "${BOOTSTRAP_REPLICATES:-2000}"
echo "[OK] final ranking: $EXP_ROOT/final_ranking/final_ranking.md"

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
AUTO_PREPARE_FINALISTS="${AUTO_PREPARE_FINALISTS:-1}"
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
  if [[ -z "$name" || -z "$kws" || -z "$policy" ]]; then
    echo "[ERR] invalid finalist fields: name=${name:-<empty>} kws=${kws:-<empty>} policy=${policy:-<empty>}" >&2
    echo "      expected: name|/absolute/kws|/absolute/gate_score_opt|policy" >&2
    exit 2
  fi
  if [[ ! -d "$kws" ]]; then
    echo "[ERR] finalist KWS directory does not exist: $kws" >&2
    exit 2
  fi
  # `auto`/空路径使用本 EXP_ROOT 的标准第一阶段目录；也兼容直接传
  # recommended_thr.json 文件，而不只接受其父目录。
  if [[ -z "$opt" || "$opt" == "auto" ]]; then
    opt="$EXP_ROOT/gate/$name/reports/gate_score_opt"
  elif [[ -f "$opt" && "$(basename "$opt")" == "recommended_thr.json" ]]; then
    opt="$(dirname "$opt")"
  fi
  if [[ ! -f "$opt/recommended_thr.json" ]]; then
    echo "[MISS] finalist gate config: $opt/recommended_thr.json" >&2
    if [[ "$AUTO_PREPARE_FINALISTS" != "1" ]]; then
      echo "[ERR] set AUTO_PREPARE_FINALISTS=1 to build the missing gate screen" >&2
      exit 2
    fi
    echo ">>> auto-prepare gate screen for finalist=$name"
    KWS_CANDIDATES="$name=$kws" EXP_ROOT="$EXP_ROOT" TOP_K=1 \
      SEEDS="${SEEDS:-200}" HOLDOUT_FRAC="${HOLDOUT_FRAC:-0.30}" \
      THRESHOLD_MODES="${THRESHOLD_MODES:-global,lang_split}" \
      SEP_REUSE_ROOT="${SEP_REUSE_ROOT:-}" \
      STRICT_SEP_REUSE="${STRICT_SEP_REUSE:-0}" \
      bash "$ROOT/run_systematic_gate_screen.sh"
  fi
  if [[ ! -f "$opt/recommended_thr.json" ]]; then
    echo "[ERR] gate auto-prepare finished but config is still missing: $opt/recommended_thr.json" >&2
    exit 2
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
  if [[ "${RUN_CMD_SE:-1}" == "1" ]]; then
    cmd_se_root="$arm_root/cmd_se"
    SCREEN_ARM="$screen_arm" SEP_REUSE_ROOT="${SEP_REUSE_ROOT:-}" \
      CMD_SE_EXP_ROOT="$cmd_se_root" \
      CMD_SE_ARMS="${CMD_SE_ARMS:-raw:raw:best,raw:se48k:best,se48k:raw:best,se48k:se48k:best,mix_top_se48k:raw:best,raw:se48k:better,se48k:se48k:better}" \
      CMD_SE_POLICIES="$policy" CLEARVOICE_ROOT="${CLEARVOICE_ROOT:-/root/autodl-tmp/ClearerVoice-Studio}" \
      SEEDS="${SEEDS:-200}" HOLDOUT_FRAC="${HOLDOUT_FRAC:-0.30}" \
      THRESHOLD_MODES="${THRESHOLD_MODES:-global,lang_split}" \
      bash "$ROOT/run_cmd_se_score_matrix.sh"
    IFS=',' read -r -a SE_ARMS <<< "${CMD_SE_ARMS:-raw:raw:best,raw:se48k:best,se48k:raw:best,se48k:se48k:best,mix_top_se48k:raw:best,raw:se48k:better,se48k:se48k:better}"
    for se_spec in "${SE_ARMS[@]}"; do
      if [[ "$se_spec" == *:*:* ]]; then
        IFS=':' read -r gate_cond audio_cond slot <<< "$se_spec"
      else
        audio_cond="${se_spec%_*}"; gate_cond="$audio_cond"; slot="${se_spec##*_}"
      fi
      se_arm="${gate_cond}gate_${audio_cond}_${slot}"
      RANK_ARGS+=(--candidate "$name.cmd_se.$se_arm=$cmd_se_root/arms/$se_arm/eval")
    done
  fi
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

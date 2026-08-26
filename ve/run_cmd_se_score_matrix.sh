#!/usr/bin/env bash
# CMD-SE 的正式 RR/FRR/CER 因子实验：gate condition × ASR audio condition × slot。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$ROOT/.env_ve" ]] && source "$ROOT/.env_ve" || true
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
SCREEN_ARM="${SCREEN_ARM:-}"
SEP_REUSE_ROOT="${SEP_REUSE_ROOT:-}"
OUT_ROOT="${CMD_SE_EXP_ROOT:-/root/autodl-tmp/ve_systematic/cmd_se}"
CLEARVOICE_ROOT="${CLEARVOICE_ROOT:-/root/autodl-tmp/ClearerVoice-Studio}"
CMD_SE_ARMS="${CMD_SE_ARMS:-raw:raw:best,raw:se48k:best,se48k:raw:best,se48k:se48k:best,mix_top_se48k:raw:best,raw:se48k:better,se48k:se48k:better}"
CMD_SE_POLICIES="${CMD_SE_POLICIES:-max}"
[[ -f "$SCREEN_ARM/manifest/samples.jsonl" && -f "$SCREEN_ARM/results/all_results.jsonl" ]] || {
  echo "[ERR] SCREEN_ARM 缺少 manifest/results: $SCREEN_ARM" >&2; exit 2;
}
[[ -d "$SEP_REUSE_ROOT/d1" ]] || {
  echo "[ERR] SEP_REUSE_ROOT 必须是含 d1/ 的 sep_streams: $SEP_REUSE_ROOT" >&2; exit 2;
}
mkdir -p "$OUT_ROOT"
CMD_SE_OUT="$OUT_ROOT/ranked"
CMD_SE_RESULTS="$CMD_SE_OUT/results/se48k_ranked_results.jsonl"
RANK_ARGS=(
  --samples "$SCREEN_ARM/manifest/samples.jsonl"
  --sep-root "$SEP_REUSE_ROOT" --out-dir "$CMD_SE_OUT"
  --clearvoice-root "$CLEARVOICE_ROOT" --presence-backend "${PRESENCE_BACKEND:-eres2netv2}"
  --device "${DEVICE:-cuda:0}" --no-enroll-vad --strict
)
[[ "${FORCE_CMD_SE:-0}" == "1" ]] && RANK_ARGS+=(--no-resume) || RANK_ARGS+=(--resume)
"$PYTHON_BIN" "$ROOT/scripts/rank_sep_streams_se48k.py" "${RANK_ARGS[@]}"

IFS=',' read -r -a ARMS <<< "$CMD_SE_ARMS"
for arm_spec in "${ARMS[@]}"; do
  if [[ "$arm_spec" == *:*:* ]]; then
    IFS=':' read -r gate_condition audio_condition slot <<< "$arm_spec"
  else
    # 兼容旧写法 raw_best / se48k_better：gate 与 audio condition 相同。
    audio_condition="${arm_spec%_*}"
    gate_condition="$audio_condition"
    slot="${arm_spec##*_}"
  fi
  case "$gate_condition:$audio_condition:$slot" in
    raw:raw:best|raw:raw:better|raw:se48k:best|raw:se48k:better|se48k:raw:best|se48k:raw:better|se48k:se48k:best|se48k:se48k:better|mix_top_se48k:raw:best) ;;
    *) echo "[ERR] invalid CMD_SE arm=$arm_spec" >&2; exit 2 ;;
  esac
  arm="${gate_condition}gate_${audio_condition}_${slot}"
  arm_root="$OUT_ROOT/arms/$arm"
  # ASR 音频只由 audio_condition + slot 决定，与 gate_condition 无关。
  # 优先复用任一旧门控臂的同音频结果，避免新门控策略重复转写 1364 条。
  asr_root=""
  for legacy_asr in "$OUT_ROOT"/arms/*gate_"${audio_condition}_${slot}"/all_pos_asr; do
    if [[ -f "$legacy_asr/results/all_results.jsonl" \
          && -f "$legacy_asr/reports/asr_cer/asr_results.jsonl" ]]; then
      asr_root="$legacy_asr"
      break
    fi
  done
  if [[ -z "$asr_root" ]]; then
    asr_root="$OUT_ROOT/shared_asr/${audio_condition}_${slot}"
  fi
  gate_rows="$arm_root/gate_decisions.jsonl"
  gate_opt="$arm_root/gate_opt"
  eval_root="$arm_root/eval"
  echo ">>> CMD-SE arm=$arm (gate=$gate_condition audio=$audio_condition slot=$slot)"
  echo "[INFO] shared ASR root=$asr_root"
  "$PYTHON_BIN" "$ROOT/scripts/prepare_cmd_se_asr_arm.py" \
    --samples "$SCREEN_ARM/manifest/samples.jsonl" --cmd-se-results "$CMD_SE_RESULTS" \
    --condition "$audio_condition" --slot "$slot" --out-dir "$asr_root" --strict
  # 文件存在不等于覆盖完整。默认让 asr_cer 的 resume 逐 UID 校验配置、
  # decision 与 status；完整项零推理复用，缺失/失败项自动补跑。
  ASR_REUSE_ARG=--resume
  if [[ "${FORCE_CMD_SE_ASR:-0}" == "1" ]]; then
    ASR_REUSE_ARG=--no-resume
    echo "[FORCE] full CMD-SE ASR rerun: $asr_root"
  elif [[ -f "$asr_root/reports/asr_cer/asr_results.jsonl" ]]; then
    echo "[CHECK] CMD-SE ASR coverage/resume: $asr_root/reports/asr_cer/asr_results.jsonl"
  else
    echo "[MISS] CMD-SE ASR; creating: $asr_root/reports/asr_cer/asr_results.jsonl"
  fi
  VE_OUT="$asr_root" ASR_RETRY_MISMATCH="${ASR_RETRY_MISMATCH:-1}" \
    bash "$ROOT/run_asr_cer.sh" "$ASR_REUSE_ARG" --require-accepted-ok
  "$PYTHON_BIN" "$ROOT/scripts/build_cmd_se_gate_decisions.py" \
    --base-decisions "$SCREEN_ARM/results/all_results.jsonl" \
    --cmd-se-results "$CMD_SE_RESULTS" --condition "$gate_condition" \
    --out "$gate_rows" --strict
  "$PYTHON_BIN" "$ROOT/scripts/optimize_gate_for_score.py" \
    --decisions "$gate_rows" \
    --asr-all-pos "$asr_root/reports/asr_cer/asr_results.jsonl" \
    --out-dir "$gate_opt" \
    --policies "$CMD_SE_POLICIES" --threshold-modes "${THRESHOLD_MODES:-global,lang_split}" \
    --holdout-frac "${HOLDOUT_FRAC:-0.30}" --seeds "${SEEDS:-200}" \
    --no-enroll-vad --strict
  "$PYTHON_BIN" "$ROOT/scripts/apply_gate_config.py" \
    --decisions "$gate_rows" --thr-file "$gate_opt/recommended_thr.json" \
    --out "$eval_root/results/all_results.jsonl" --strict
  mkdir -p "$eval_root/manifest"
  cp -f "$SCREEN_ARM/manifest/samples.jsonl" "$eval_root/manifest/samples.jsonl"
  "$PYTHON_BIN" "$ROOT/scripts/final_evaluate.py" \
    --ve-out "$eval_root" --samples "$SCREEN_ARM/manifest/samples.jsonl" \
    --decisions "$eval_root/results/all_results.jsonl" \
    --asr "$asr_root/reports/asr_cer/asr_results.jsonl" \
    --out-dir "$eval_root/reports/final_eval" --strict
done
echo "[OK] CMD-SE strict matrix -> $OUT_ROOT/arms"

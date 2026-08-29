#!/usr/bin/env bash
# Smoke in the final output tree, resume to full, then run strict audits.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:---full}"
case "$MODE" in --smoke|--full|--all) ;; *) echo "usage: $0 [--smoke|--full|--all]"; exit 2;; esac

# Capture caller overrides before env.sh fills legacy defaults.
REQUESTED_VM_OUT="${VM_OUT:-/root/autodl-tmp/kws_sep_dedup}"
REQUESTED_DATA_DIR="${DATA_DIR:-/root/autodl-tmp/datasetA}"
REQUESTED_ASR_MODEL_DIR="${ASR_MODEL_DIR:-/root/autodl-tmp/Qwen3-ASR-1.7B}"
REQUESTED_MOSS_CKPT_DIR="${MOSS_CKPT_DIR:-/root/autodl-tmp/checkpoints}"
[[ -f "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
source "$ROOT/pick_python.sh"
export VM_OUT="$REQUESTED_VM_OUT"
export DATA_DIR="$REQUESTED_DATA_DIR"
export ASR_MODEL_DIR="$REQUESTED_ASR_MODEL_DIR"
export MOSS_CKPT_DIR="$REQUESTED_MOSS_CKPT_DIR"

echo "[INFO] PYTHON_BIN=$PYTHON_BIN"
echo "[INFO] VM_OUT=$VM_OUT (existing successful rows are resumed; nothing is deleted)"
"$PYTHON_BIN" "$ROOT/scripts/test_sep_invariants.py"
"$PYTHON_BIN" "$ROOT/scripts/test_dae_tse_integration.py"
bash "$ROOT/check_env.sh"

if [[ "$MODE" == "--smoke" || "$MODE" == "--all" ]]; then
  echo "[INFO] smoke s1-s8, 20 UIDs per split"
  bash "$ROOT/run_sep.sh" --limit 20
  "$PYTHON_BIN" "$ROOT/scripts/audit_sep_run.py" --vm-out "$VM_OUT" --expected-uids 40
fi

if [[ "$MODE" == "--full" || "$MODE" == "--all" ]]; then
  echo "[INFO] resume the same tree to full coverage"
  bash "$ROOT/run_sep.sh"
  "$PYTHON_BIN" "$ROOT/scripts/audit_sep_run.py" --vm-out "$VM_OUT" --expected-uids 1838
  if [[ -f /root/kws/scripts/compare_all_stages.py ]]; then
    echo "[INFO] validate extract-sep -> kws input contract"
    "$PYTHON_BIN" /root/kws/scripts/audit_sep_input.py \
      --pos-neg "$VM_OUT" --expected-uids 1838 --check-duration --require-handoff \
      --out "$VM_OUT/reports/kws_input_audit.json"
    echo "[INFO] systematic independent-stage comparison"
    "$PYTHON_BIN" /root/kws/scripts/compare_all_stages.py \
      --pos-neg "$VM_OUT" --expected-uids 1838 --hash-wav \
      --out-json "$VM_OUT/reports/all_stage_comparison.json" \
      --out-md "$VM_OUT/reports/all_stage_comparison.md"
    "$PYTHON_BIN" /root/kws/scripts/rank_same_uid_audio.py \
      --pos-neg "$VM_OUT" --expected-uids 1838 --top-k 20 \
      --out-jsonl "$VM_OUT/reports/same_uid_audio_rank.jsonl" \
      --out-summary "$VM_OUT/reports/same_uid_audio_rank_summary.json" \
      --out-conflicts "$VM_OUT/reports/same_uid_audio_score_conflicts.jsonl" \
      --out-md "$VM_OUT/reports/same_uid_audio_rank.md"
    OLD_VM_OUT="${OLD_VM_OUT:-/root/autodl-tmp/kws_sep}"
    if [[ "$OLD_VM_OUT" != "$VM_OUT" && -f "$OLD_VM_OUT/best_sep/index.jsonl" ]]; then
      "$PYTHON_BIN" /root/kws/scripts/compare_best_sep.py \
        --dir "old=$OLD_VM_OUT/best_sep" \
        --dir "dedup=$VM_OUT/best_sep" \
        --baseline old \
        --out "$VM_OUT/reports/old_vs_dedup_best_sep.json"
    fi
  else
    echo "[WARN] /root/kws systematic comparison scripts not found; strict extract audit is complete"
  fi
fi

echo "[OK] rerun/audit complete: $VM_OUT"

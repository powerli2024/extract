#!/usr/bin/env bash
# Optional DAE-TSE experiment. It never changes the default s1-s8 handoff.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/paths_defaults.sh"
[[ -f "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
# shellcheck disable=SC1091
source "$ROOT/pick_python.sh"

DAE_PYTHON_BIN="${DAE_PYTHON_BIN:-$PYTHON_BIN}"
DAE_TSE_REPO="${DAE_TSE_REPO:-/root/autodl-tmp/projects/DAE-TSE}"
DAE_TSE_CONFIG="${DAE_TSE_CONFIG:-$DAE_TSE_REPO/examples/librimix/dae-tse/exp/backbone/config.yaml}"
DAE_TSE_CHECKPOINT="${DAE_TSE_CHECKPOINT:-/root/autodl-fs/midea-dae/models/dae_zh_v1/model.pt}"
DAE_SOURCE_STAGE="${DAE_SOURCE_STAGE:-s1}"
DAE_SOURCE_THR="${DAE_SOURCE_THR:-}"
DAE_OUT="${DAE_OUT:-$VM_OUT/experiments/dae_tse_from_${DAE_SOURCE_STAGE}${DAE_SOURCE_THR:+_thr_$DAE_SOURCE_THR}}"
LIMIT="${LIMIT:-0}"
DEVICE="${DEVICE:-cuda:0}"
DAE_INPUT_PEAK="${DAE_INPUT_PEAK:-0.70}"

if [[ ! -f "$VM_OUT/pos/meta/items.jsonl" || ! -f "$VM_OUT/neg/meta/items.jsonl" ]]; then
  echo "[INFO] collect metadata first"
  bash "$ROOT/run_stage.sh" collect --vm-out "$VM_OUT"
fi

ARGS=(
  --vm-out "$VM_OUT"
  --out-dir "$DAE_OUT"
  --source-stage "$DAE_SOURCE_STAGE"
  --dae-python "$DAE_PYTHON_BIN"
  --dae-repo "$DAE_TSE_REPO"
  --dae-config "$DAE_TSE_CONFIG"
  --dae-checkpoint "$DAE_TSE_CHECKPOINT"
  --asr-model-dir "$ASR_MODEL_DIR"
  --device "$DEVICE"
  --input-peak "$DAE_INPUT_PEAK"
  --limit "$LIMIT"
)
[[ -n "$DAE_SOURCE_THR" ]] && ARGS+=(--source-thr "$DAE_SOURCE_THR")

echo "[INFO] optional DAE-TSE candidate; default handoff remains unchanged"
echo "[INFO] source=$DAE_SOURCE_STAGE thr=${DAE_SOURCE_THR:--} out=$DAE_OUT"
exec "$PYTHON_BIN" "$ROOT/scripts/stage_dae_tse.py" "${ARGS[@]}" "$@"

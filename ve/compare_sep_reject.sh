#!/usr/bin/env bash
# 拒识三组对照：不做分离 / 一次分离 / 多次分离；默认保存中间 wav
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$ROOT/.env_ve" ]] && source "$ROOT/.env_ve" || true

export PYTHONPATH="$ROOT/scripts:${ROOT}/../scripts:${ROOT}/../VM/scripts:${ROOT}/../VD/tools:${PYTHONPATH:-}"
export MOSS_ONNX_PATH="${MOSS_ONNX_PATH:-/root/autodl-tmp/checkpoints/MossFormer2_ONNX/simple_model.onnx}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
DEVICE="${DEVICE:-cuda:0}"
LIMIT="${LIMIT:-0}"
OUT="${VE_OUT:-/root/autodl-tmp/ve_gate_cmp}"
SAMPLES="${SAMPLES:-}"
COHORT_DIR="${COHORT_DIR:-}"
ENROLL_ZNORM="${ENROLL_ZNORM:-0}"

mkdir -p "$OUT"

if [[ ! -f "$MOSS_ONNX_PATH" ]]; then
  echo "[INFO] 下载 MossFormer ONNX ..."
  "$ROOT/download_moss_onnx.sh" || true
fi

# 找 samples
if [[ -z "$SAMPLES" ]]; then
  for c in \
    "$OUT/manifest/samples.jsonl" \
    /root/autodl-tmp/ve_ps4/manifest/samples.jsonl \
    /root/autodl-tmp/ve/manifest/samples.jsonl \
    /root/autodl-tmp/ve_ps4_usep/manifest/samples.jsonl
  do
    if [[ -f "$c" ]]; then SAMPLES="$c"; break; fi
  done
fi

if [[ -z "${SAMPLES:-}" || ! -f "$SAMPLES" ]]; then
  echo "[INFO] 无 manifest，先 build ..."
  DATA_DIR="${DATA_DIR:-/root/autodl-tmp/datasetA}"
  BEST_SEP_DIR="${BEST_SEP_DIR:-/root/autodl-tmp/pos_neg/best_sep}"
  "$PYTHON_BIN" "$ROOT/scripts/build_manifest.py" \
    --data-dir "$DATA_DIR" --best-sep "$BEST_SEP_DIR" --out-dir "$OUT/manifest"
  SAMPLES="$OUT/manifest/samples.jsonl"
fi

echo "=== compare_sep_reject ==="
echo "SAMPLES=$SAMPLES OUT=$OUT LIMIT=$LIMIT"

ARGS=(
  --samples "$SAMPLES"
  --out-dir "$OUT"
  --device "$DEVICE"
  --save-sep-wavs
  --select-by contest
  --depths 0,1,2
)
[[ "$LIMIT" != "0" ]] && ARGS+=(--limit "$LIMIT")
if [[ "$ENROLL_ZNORM" == "1" ]] || [[ -n "$COHORT_DIR" ]]; then
  COHORT_DIR="${COHORT_DIR:-/root/autodl-tmp/clean_kws}"
  ARGS+=(--enroll-znorm --cohort-dir "$COHORT_DIR")
  echo "[INFO] enroll Z-Norm ON cohort=$COHORT_DIR"
fi

"$PYTHON_BIN" "$ROOT/scripts/compare_sep_reject.py" "${ARGS[@]}"

echo
echo "报告: $OUT/reports/sep_reject_compare.md"
echo "中间音: $OUT/sep_streams/d1|d2/{pos,neg}/{uid}/"
ls -lah "$OUT/reports" || true

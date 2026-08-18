#!/usr/bin/env bash
# Enroll 侧 Z-Norm 校准对照：同一设定下 raw vs enroll_znorm
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$ROOT/.env_ve" ]] && source "$ROOT/.env_ve" || true

export PYTHONPATH="$ROOT/scripts:${ROOT}/../scripts:${ROOT}/../VM/scripts:${ROOT}/../VD/tools:${PYTHONPATH:-}"
export MOSS_ONNX_PATH="${MOSS_ONNX_PATH:-/root/autodl-tmp/checkpoints/MossFormer2_ONNX/simple_model.onnx}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
DEVICE="${DEVICE:-cuda:0}"
LIMIT="${LIMIT:-0}"
OUT="${VE_OUT:-/root/autodl-tmp/ve_gate_znorm}"
export OUT
SAMPLES="${SAMPLES:-}"
COHORT_DIR="${COHORT_DIR:-/root/autodl-tmp/clean_kws}"
SEP_DEPTH="${SEP_DEPTH:-1}"
PRESENCE_BACKEND="${PRESENCE_BACKEND:-eres2netv2}"

mkdir -p "$OUT"

if [[ ! -d "$COHORT_DIR" ]] || [[ -z "$(find "$COHORT_DIR" -name '*.wav' 2>/dev/null | head -1)" ]]; then
  echo "[ERR] 找不到路人 wav：COHORT_DIR=$COHORT_DIR"
  echo "      例: unzip clean_kws.zip -d /root/autodl-tmp/"
  exit 1
fi

if [[ -z "$SAMPLES" ]]; then
  for c in \
    "$OUT/manifest/samples.jsonl" \
    /root/autodl-tmp/ve_ps4/manifest/samples.jsonl \
    /root/autodl-tmp/ve_gate_cmp_eres/manifest/samples.jsonl \
    /root/autodl-tmp/ve/manifest/samples.jsonl
  do
    if [[ -f "$c" ]]; then SAMPLES="$c"; break; fi
  done
fi

if [[ -z "${SAMPLES:-}" || ! -f "$SAMPLES" ]]; then
  DATA_DIR="${DATA_DIR:-/root/autodl-tmp/datasetA}"
  BEST_SEP_DIR="${BEST_SEP_DIR:-/root/autodl-tmp/pos_neg/best_sep}"
  "$PYTHON_BIN" "$ROOT/scripts/build_manifest.py" \
    --data-dir "$DATA_DIR" --best-sep "$BEST_SEP_DIR" --out-dir "$OUT/manifest"
  SAMPLES="$OUT/manifest/samples.jsonl"
fi

echo "=== calibrate enroll Z-Norm ==="
echo "SAMPLES=$SAMPLES OUT=$OUT COHORT=$COHORT_DIR SEP_DEPTH=$SEP_DEPTH LIMIT=$LIMIT"

RAW_ARGS=(
  --samples "$SAMPLES"
  --out-dir "$OUT/reports/presence_calib_raw"
  --presence-backend "$PRESENCE_BACKEND"
  --device "$DEVICE"
  --sep-depth "$SEP_DEPTH"
  --select-by contest
)
ZN_ARGS=(
  --samples "$SAMPLES"
  --out-dir "$OUT/reports/presence_calib_znorm"
  --presence-backend "$PRESENCE_BACKEND"
  --device "$DEVICE"
  --sep-depth "$SEP_DEPTH"
  --select-by contest
  --cohort-dir "$COHORT_DIR"
  --enroll-znorm
  --cohort-per-spk "${COHORT_PER_SPK:-2}"
  --cohort-max-files "${COHORT_MAX_FILES:-400}"
)
[[ "$LIMIT" != "0" ]] && RAW_ARGS+=(--limit "$LIMIT") && ZN_ARGS+=(--limit "$LIMIT")

echo ">>> raw (no Z-Norm)"
"$PYTHON_BIN" "$ROOT/scripts/calibrate_presence.py" "${RAW_ARGS[@]}"

echo ">>> enroll Z-Norm"
"$PYTHON_BIN" "$ROOT/scripts/calibrate_presence.py" "${ZN_ARGS[@]}"

echo
echo "对比 recommended_thr:"
"$PYTHON_BIN" - <<PY
import json
from pathlib import Path
out = Path(r"$OUT")
for name in ("presence_calib_raw", "presence_calib_znorm"):
    p = out / "reports" / name / "recommended_thr.json"
    if not p.is_file():
        print(f"{name}: MISSING")
        continue
    o = json.loads(p.read_text(encoding="utf-8"))
    print(
        f"{name}: contest={o.get('contest_score')} RR={o.get('rr')} "
        f"FRR={o.get('frr')} thr={o.get('presence_thr')} norm={o.get('score_norm')}"
    )
PY

echo "报告: $OUT/reports/presence_calib_{raw,znorm}/"

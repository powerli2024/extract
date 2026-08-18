#!/usr/bin/env bash
# 多编码器 × 已有 sep_streams（no_sep / sep_once / sep_multi）打分，不重跑 Moss
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$ROOT/.env_ve" ]] && source "$ROOT/.env_ve" || true

export PYTHONPATH="$ROOT/scripts:${ROOT}/../scripts:${ROOT}/../VM/scripts:${ROOT}/../VD/tools:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
DEVICE="${DEVICE:-cuda:0}"
LIMIT="${LIMIT:-0}"
ENROLL_VAD="${ENROLL_VAD:-1}"
if [[ "$ENROLL_VAD" == "1" ]]; then VAD_TAG=vad; else VAD_TAG=novad; fi
if [[ -n "${VE_OUT:-}" ]]; then
  OUT="$VE_OUT"
else
  OUT="/root/autodl-tmp/ve_encoder_cmp_${VAD_TAG}"
fi
SAMPLES="${SAMPLES:-}"
SEP_ROOT="${SEP_ROOT:-}"
ENCODERS="${ENCODERS:-eres2netv2,campplus,resnet34_lm}"
ARMS="${ARMS:-no_sep,sep_once,sep_multi}"

mkdir -p "$OUT"

if [[ -z "$SAMPLES" ]]; then
  for c in \
    /root/autodl-tmp/ve_gate_cmp/manifest/samples.jsonl \
    /root/autodl-tmp/ve_gate_cmp_eres/manifest/samples.jsonl \
    /root/autodl-tmp/ve_gate_znorm/manifest/samples.jsonl \
    /root/autodl-tmp/ve_ps4/manifest/samples.jsonl \
    "$OUT/manifest/samples.jsonl"
  do
    if [[ -f "$c" ]]; then SAMPLES="$c"; break; fi
  done
fi

if [[ -z "$SEP_ROOT" ]]; then
  for c in \
    /root/autodl-tmp/ve_gate_cmp/sep_streams \
    /root/autodl-tmp/ve_gate_cmp_eres/sep_streams \
    /root/autodl-tmp/ve_gate_znorm/sep_streams
  do
    if [[ -d "$c/d1" ]]; then SEP_ROOT="$c"; break; fi
  done
fi

if [[ -z "${SAMPLES:-}" || ! -f "$SAMPLES" ]]; then
  echo "[ERR] 找不到 samples.jsonl，请 export SAMPLES=..."
  exit 1
fi
if [[ -z "${SEP_ROOT:-}" || ! -d "$SEP_ROOT/d1" ]]; then
  echo "[ERR] 找不到 sep_streams（需含 d1/），请 export SEP_ROOT=..."
  echo "      例: /root/autodl-tmp/ve_gate_cmp/sep_streams"
  exit 1
fi

echo "=== score_encoders_on_sep ==="
echo "SAMPLES=$SAMPLES"
echo "SEP_ROOT=$SEP_ROOT"
echo "ENCODERS=$ENCODERS ARMS=$ARMS OUT=$OUT LIMIT=$LIMIT ENROLL_VAD=$ENROLL_VAD"

ARGS=(
  --samples "$SAMPLES"
  --sep-root "$SEP_ROOT"
  --out-dir "$OUT"
  --encoders "$ENCODERS"
  --arms "$ARMS"
  --device "$DEVICE"
  --select-by contest
)
[[ "$LIMIT" != "0" ]] && ARGS+=(--limit "$LIMIT")
if [[ "$ENROLL_VAD" == "1" ]]; then
  ARGS+=(--enroll-vad)
else
  ARGS+=(--no-enroll-vad)
fi
# 默认落在 VE_MODEL_DIR 下的标准子目录（download_presence_encoders.sh 产物）
ERES_DIR="${ERES_DIR:-${VE_MODEL_DIR:-/root/autodl-tmp/ve_models}/eres2netv2_zh}"
CAMPPLUS_DIR="${CAMPPLUS_DIR:-${VE_MODEL_DIR:-/root/autodl-tmp/ve_models}/campplus_zh}"
SPK_CHS_DIR="${SPK_CHS_DIR:-${VE_MODEL_DIR:-/root/autodl-tmp/ve_models}/cnceleb_resnet34_LM}"
VBLINK_DIR="${VBLINK_DIR:-${VE_MODEL_DIR:-/root/autodl-tmp/ve_models}/vblink2_samresnet34}"
ARGS+=(--eres-dir "$ERES_DIR")
ARGS+=(--campplus-dir "$CAMPPLUS_DIR")
ARGS+=(--spk-chs-dir "$SPK_CHS_DIR")
[[ -d "$VBLINK_DIR" ]] && ARGS+=(--vblink-dir "$VBLINK_DIR")

"$PYTHON_BIN" "$ROOT/scripts/score_encoders_on_sep.py" "${ARGS[@]}"

if [[ "${SKIP_FUSE:-0}" == "1" ]]; then
  echo "[INFO] SKIP_FUSE=1，跳过 sweep_encoder_fuse"
else
  echo
  echo "=== fuse sweep (offline optimal) ==="
  "$PYTHON_BIN" "$ROOT/scripts/sweep_encoder_fuse.py" \
    --reports-dir "$OUT/reports" \
    --out "$OUT/reports/fuse_best_offline.json" \
    --arms "$ARMS"
  echo "融合最优: $OUT/reports/fuse_best_offline.md"
fi

echo
echo "报告: $OUT/reports/encoder_sep_matrix.md"
ls -lah "$OUT/reports" | head -40

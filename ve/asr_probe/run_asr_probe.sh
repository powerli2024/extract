#!/usr/bin/env bash
# 全量 pos/neg × no_sep / sep_once / sep_multi → Qwen3-ASR + 分析 JSON
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VE_ROOT="$(cd "$HERE/.." && pwd)"
[[ -f "$VE_ROOT/.env_ve" ]] && source "$VE_ROOT/.env_ve" || true

usage() {
  cat <<'EOF'
asr_probe — 对 mix / d1 / d2 全部轨跑 ASR（含 pos 与 neg）

  ./run_asr_probe.sh [--help]
  LIMIT=8 SKIP_ANALYZE=1 ./run_asr_probe.sh

环境:
  ASR_PROBE_OUT   默认 /root/autodl-tmp/asr_probe
  SAMPLES         默认自动找 samples.jsonl
  SEP_ROOT        含 d1/ 的 sep_streams
  DATA_DIR        无 samples 时 build_manifest
  ASR_MODEL_DIR   Qwen3-ASR-1.7B
  ARMS            默认 no_sep,sep_once,sep_multi
  LIMIT           0=全量
  DEVICE          cuda:0
  BATCH           ASR batch，默认 8
  SKIP_ANALYZE=1  只转写不汇总
EOF
}

for a in "$@"; do
  case "$a" in
    -h|--help|help) usage; exit 0 ;;
    -*) echo "[ERR] $a  用环境变量。 $0 --help" >&2; exit 2 ;;
  esac
done

export PYTHONPATH="$VE_ROOT/scripts:${VE_ROOT}/../scripts:${VE_ROOT}/../VM/scripts:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
OUT="${ASR_PROBE_OUT:-/root/autodl-tmp/asr_probe}"
SAMPLES="${SAMPLES:-}"
SEP_ROOT="${SEP_ROOT:-}"
ARMS="${ARMS:-no_sep,sep_once,sep_multi}"
LIMIT="${LIMIT:-0}"
DEVICE="${DEVICE:-cuda:0}"
BATCH="${BATCH:-8}"
ASR_MODEL_DIR="${ASR_MODEL_DIR:-${QWEN3_ASR_DIR:-/root/autodl-tmp/Qwen3-ASR-1.7B}}"

mkdir -p "$OUT"

if [[ -z "$SAMPLES" ]]; then
  for c in \
    "$OUT/manifest/samples.jsonl" \
    /root/autodl-tmp/vp/manifest/samples.jsonl \
    /root/autodl-tmp/ve_gate_cmp/manifest/samples.jsonl \
    /root/autodl-tmp/ve_mix_vad/manifest/samples.jsonl \
    /root/autodl-tmp/ve_mix/manifest/samples.jsonl
  do
    if [[ -f "$c" ]]; then SAMPLES="$c"; break; fi
  done
fi
if [[ -z "${SAMPLES:-}" || ! -f "$SAMPLES" ]]; then
  echo ">>> build_manifest → $OUT/manifest <<<"
  "$PYTHON_BIN" "$VE_ROOT/scripts/build_manifest.py" \
    --data-dir "${DATA_DIR:-/root/autodl-tmp/datasetA}" \
    --best-sep "${BEST_SEP_DIR:-/root/autodl-tmp/pos_neg/best_sep}" \
    --out-dir "$OUT/manifest"
  SAMPLES="$OUT/manifest/samples.jsonl"
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
if [[ -z "${SEP_ROOT:-}" || ! -d "$SEP_ROOT/d1" ]]; then
  echo "[ERR] 需要 SEP_ROOT（含 d1/）。export SEP_ROOT=/root/autodl-tmp/ve_gate_cmp/sep_streams"
  exit 1
fi

echo "=== asr_probe ==="
echo "SAMPLES=$SAMPLES"
echo "SEP_ROOT=$SEP_ROOT"
echo "OUT=$OUT ARMS=$ARMS LIMIT=$LIMIT"

ARGS=(
  --samples "$SAMPLES"
  --sep-root "$SEP_ROOT"
  --out-dir "$OUT"
  --arms "$ARMS"
  --device "$DEVICE"
  --model-dir "$ASR_MODEL_DIR"
  --batch "$BATCH"
)
[[ "$LIMIT" != "0" ]] && ARGS+=(--limit "$LIMIT")

"$PYTHON_BIN" "$HERE/scripts/run_asr_streams.py" "${ARGS[@]}"

if [[ "${SKIP_ANALYZE:-0}" != "1" ]]; then
  "$PYTHON_BIN" "$HERE/scripts/analyze_asr_streams.py" \
    --results "$OUT/asr_results.jsonl" \
    --out-dir "$OUT"
fi

echo "[OK] $OUT/asr_results.jsonl"
echo "[OK] $OUT/analysis.json"
ls -lah "$OUT" | head -20

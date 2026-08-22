#!/usr/bin/env bash
# raw / enroll_znorm / asnorm 三方对照（A=clean_kws, B=mix500）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$ROOT/.env_ve" ]] && source "$ROOT/.env_ve" || true

export PYTHONPATH="$ROOT/scripts:${ROOT}/../scripts:${ROOT}/../VM/scripts:${ROOT}/../VD/tools:${PYTHONPATH:-}"
export MOSS_ONNX_PATH="${MOSS_ONNX_PATH:-/root/autodl-tmp/checkpoints/MossFormer2_ONNX/simple_model.onnx}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
DEVICE="${DEVICE:-cuda:0}"
LIMIT="${LIMIT:-0}"
OUT="${VE_OUT:-/root/autodl-tmp/ve_gate_asnorm}"
SAMPLES="${SAMPLES:-}"
COHORT_DIR="${COHORT_DIR:-/root/autodl-tmp/clean_kws}"
TEST_COHORT_DIR="${TEST_COHORT_DIR:-/root/autodl-tmp/mix500}"
SEP_DEPTH="${SEP_DEPTH:-1}"
PRESENCE_BACKEND="${PRESENCE_BACKEND:-eres2netv2}"

mkdir -p "$OUT"

need_wavs() {
  local d="$1"
  [[ -d "$d" ]] && [[ -n "$(find "$d" -name '*.wav' 2>/dev/null | head -1)" ]]
}

if ! need_wavs "$COHORT_DIR"; then
  echo "[ERR] enroll 路人缺失: COHORT_DIR=$COHORT_DIR (clean_kws)"
  exit 1
fi
if ! need_wavs "$TEST_COHORT_DIR"; then
  echo "[ERR] test 路人缺失: TEST_COHORT_DIR=$TEST_COHORT_DIR"
  echo "      例: mkdir -p /root/autodl-tmp/mix500 && unrar x mix500.rar /root/autodl-tmp/mix500/"
  echo "      或: 7z x mix500.rar -o/root/autodl-tmp/mix500"
  exit 1
fi

if [[ -z "$SAMPLES" ]]; then
  for c in \
    "$OUT/manifest/samples.jsonl" \
    /root/autodl-tmp/ve_gate_znorm/manifest/samples.jsonl \
    /root/autodl-tmp/ve_gate_cmp_eres/manifest/samples.jsonl \
    /root/autodl-tmp/ve_ps4/manifest/samples.jsonl \
    /root/autodl-tmp/ve/manifest/samples.jsonl
  do
    if [[ -f "$c" ]]; then SAMPLES="$c"; break; fi
  done
fi

if [[ -z "${SAMPLES:-}" || ! -f "$SAMPLES" ]]; then
  DATA_DIR="${DATA_DIR:-/root/autodl-tmp/datasetA}"
  BEST_SEP_ARGS=()
  [[ -n "${BEST_SEP_DIR:-}" ]] && BEST_SEP_ARGS+=(--best-sep "$BEST_SEP_DIR")
  "$PYTHON_BIN" "$ROOT/scripts/build_manifest.py" \
    --data-dir "$DATA_DIR" "${BEST_SEP_ARGS[@]}" --out-dir "$OUT/manifest"
  SAMPLES="$OUT/manifest/samples.jsonl"
fi

echo "=== calibrate AS-Norm ==="
echo "SAMPLES=$SAMPLES OUT=$OUT"
echo "A_cohort=$COHORT_DIR  B_cohort=$TEST_COHORT_DIR  SEP_DEPTH=$SEP_DEPTH LIMIT=$LIMIT"

base=(
  --samples "$SAMPLES"
  --presence-backend "$PRESENCE_BACKEND"
  --device "$DEVICE"
  --sep-depth "$SEP_DEPTH"
  --select-by contest
)
[[ "$LIMIT" != "0" ]] && base+=(--limit "$LIMIT")

run_one() {
  local name="$1"; shift
  echo ">>> $name"
  "$PYTHON_BIN" "$ROOT/scripts/calibrate_presence.py" \
    --out-dir "$OUT/reports/$name" \
    "${base[@]}" \
    "$@"
}

# 若已有 raw 结果可跳过：SKIP_RAW=1
if [[ "${SKIP_RAW:-0}" != "1" ]]; then
  run_one presence_calib_raw
else
  echo ">>> skip raw (SKIP_RAW=1)"
fi

if [[ "${SKIP_ENROLL:-0}" != "1" ]]; then
  run_one presence_calib_enroll_znorm \
    --enroll-znorm --cohort-dir "$COHORT_DIR" \
    --cohort-per-spk "${COHORT_PER_SPK:-2}" \
    --cohort-max-files "${COHORT_MAX_FILES:-400}"
else
  echo ">>> skip enroll_znorm"
fi

run_one presence_calib_asnorm \
  --asnorm \
  --cohort-dir "$COHORT_DIR" \
  --test-cohort-dir "$TEST_COHORT_DIR" \
  --cohort-per-spk "${COHORT_PER_SPK:-2}" \
  --cohort-max-files "${COHORT_MAX_FILES:-400}" \
  --test-cohort-max-files "${TEST_COHORT_MAX_FILES:-500}"

echo
echo "对比 recommended_thr:"
"$PYTHON_BIN" - <<PY
import json
from pathlib import Path
out = Path(r"$OUT")
names = (
    "presence_calib_raw",
    "presence_calib_enroll_znorm",
    "presence_calib_asnorm",
)
# 兼容旧目录名
alt = {"presence_calib_enroll_znorm": "presence_calib_znorm"}
for name in names:
    p = out / "reports" / name / "recommended_thr.json"
    if not p.is_file() and name in alt:
        p = out / "reports" / alt[name] / "recommended_thr.json"
    if not p.is_file():
        # 也扫 ve_gate_znorm 的 raw
        p2 = Path("/root/autodl-tmp/ve_gate_znorm/reports") / name.replace("presence_calib_enroll_znorm","presence_calib_znorm") / "recommended_thr.json"
        if name == "presence_calib_raw":
            p2 = Path("/root/autodl-tmp/ve_gate_znorm/reports/presence_calib_raw/recommended_thr.json")
        if p2.is_file():
            p = p2
    if not p.is_file():
        print(f"{name}: MISSING")
        continue
    o = json.loads(p.read_text(encoding="utf-8"))
    print(
        f"{name}: contest={o.get('contest_score')} RR={o.get('rr')} "
        f"FRR={o.get('frr')} thr={o.get('presence_thr')} norm={o.get('score_norm')}"
    )
PY

echo "报告: $OUT/reports/presence_calib_{raw,enroll_znorm,asnorm}/"

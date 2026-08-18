#!/usr/bin/env bash
# 确认 MossFormer 可用 → 可选 LIMIT 校准对比（无 sep vs use_sep）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$ROOT/.env_ve" ]] && source "$ROOT/.env_ve" || true

export PYTHONPATH="$ROOT/scripts:${ROOT}/../scripts:${ROOT}/../VM/scripts:${ROOT}/../VD/tools:${PYTHONPATH:-}"
export MOSS_ONNX_PATH="${MOSS_ONNX_PATH:-/root/autodl-tmp/checkpoints/MossFormer2_ONNX/simple_model.onnx}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
DEVICE="${DEVICE:-cuda:0}"
LIMIT="${LIMIT:-64}"
VE_BASE="${VE_OUT_BASE:-/root/autodl-tmp/ve}"

echo "=== smoke_sep + presence use_sep 对比 ==="
echo "MOSS_ONNX_PATH=$MOSS_ONNX_PATH"

if [[ ! -f "$MOSS_ONNX_PATH" ]]; then
  echo "[INFO] 缺少 ONNX，尝试 ./download_moss_onnx.sh"
  "$ROOT/download_moss_onnx.sh"
fi

step() { echo; echo ">>> $* <<<"; }

step "1/3 smoke_sep（合成双音）"
"$PYTHON_BIN" "$ROOT/scripts/smoke_sep.py" --device "$DEVICE" \
  --out-dir "${VE_BASE}_sep_smoke/wavs"

# 若有 manifest，用真实一条再测
MANIFEST="${MANIFEST:-/root/autodl-tmp/ve_ps4/manifest/samples.jsonl}"
if [[ ! -f "$MANIFEST" ]]; then
  MANIFEST="/root/autodl-tmp/ve/manifest/samples.jsonl"
fi
if [[ -f "$MANIFEST" ]]; then
  step "1b/3 smoke_sep（manifest 首条 pos）"
  read -r CMD_WAV ENROLL_WAV < <("$PYTHON_BIN" - <<PY
import json
from pathlib import Path
p = Path("$MANIFEST")
for line in p.open(encoding="utf-8"):
    r = json.loads(line)
    if r.get("split") == "pos":
        print(r["cmd_wav"], r["enroll_wav"])
        break
PY
)
  if [[ -n "${CMD_WAV:-}" && -f "$CMD_WAV" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/smoke_sep.py" --device "$DEVICE" \
      --wav "$CMD_WAV" --enroll "$ENROLL_WAV" \
      --out-dir "${VE_BASE}_sep_smoke/wavs_real"
  else
    echo "[WARN] 未能从 manifest 取到 wav，跳过真实样本"
  fi
else
  echo "[WARN] 无 manifest，跳过真实样本 Presence 对比"
fi

if [[ "${SKIP_CALIB:-0}" == "1" ]]; then
  echo "[INFO] SKIP_CALIB=1，结束"
  exit 0
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "[ERR] 需要 samples.jsonl 才能校准对比。请先 PIPELINE=ps4 跑过 manifest，或:"
  echo "  python scripts/build_manifest.py"
  exit 1
fi

step "2/3 calibrate 无 sep（LIMIT=$LIMIT）"
"$PYTHON_BIN" "$ROOT/scripts/calibrate_presence.py" \
  --samples "$MANIFEST" \
  --out-dir "${VE_BASE}_calib_nosep/reports/presence_calib" \
  --device "$DEVICE" \
  --limit "$LIMIT" \
  --target-frr 0.02 \
  --select-by contest

step "3/3 calibrate USE_SEP=1（LIMIT=$LIMIT）"
"$PYTHON_BIN" "$ROOT/scripts/calibrate_presence.py" \
  --samples "$MANIFEST" \
  --out-dir "${VE_BASE}_calib_usep/reports/presence_calib" \
  --device "$DEVICE" \
  --use-sep \
  --limit "$LIMIT" \
  --target-frr 0.02 \
  --select-by contest

echo
echo "=== 对比 recommended_thr.json ==="
"$PYTHON_BIN" - <<PY
import json
from pathlib import Path
for label, p in [
    ("no_sep", Path("${VE_BASE}_calib_nosep/reports/presence_calib/recommended_thr.json")),
    ("use_sep", Path("${VE_BASE}_calib_usep/reports/presence_calib/recommended_thr.json")),
]:
    if not p.is_file():
        print(label, "MISSING", p)
        continue
    o = json.loads(p.read_text(encoding="utf-8"))
    print(f"--- {label} ---")
    print(json.dumps(o, ensure_ascii=False, indent=2)[:800])
PY

echo
echo "听感: ${VE_BASE}_sep_smoke/wavs/"
echo "全量拒识对比（确认 smoke 通过后）:"
echo "  USE_SEP=1 PIPELINE=ps4 VE_OUT=/root/autodl-tmp/ve_ps4_usep ./run_all.sh"

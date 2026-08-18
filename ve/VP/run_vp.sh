#!/usr/bin/env bash
# VP：只做说话人在场检测（拒识），不跑 TSE / ASR
set -euo pipefail
VP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VE_ROOT="$(cd "$VP_ROOT/.." && pwd)"
[[ -f "$VE_ROOT/.env_ve" ]] && source "$VE_ROOT/.env_ve" || true

usage() {
  cat <<'EOF'
VP run_vp.sh — 说话人在场 / 拒识实验

用法:
  ./run_vp.sh [--help|-h]
  MODE=matrix ./run_vp.sh          # 默认：已有 sep_streams 上打编码器×流×VAD
  MODE=live  ./run_vp.sh           # 无预计算流时：calibrate_presence 现打（慢）

环境变量:
  VP_OUT          默认 /root/autodl-tmp/vp
  SAMPLES         默认 $VP_OUT/manifest/samples.jsonl（无则 build_manifest）
  SEP_ROOT        含 d1/ 的 sep_streams；matrix 模式必需
  ENCODERS        默认 eres2netv2,campplus,resnet34_lm
  ARMS            默认 no_sep,sep_once,sep_multi
  VAD_MODES       默认 0,1   （0=novad 1=vad；matrix 会各打一套）
  LIMIT           0=全量
  DEVICE          cuda:0
  HOLDOUT_FRAC    汇总 holdout / live 选 τ；默认 0.3
  SKIP_FUSE       matrix 默认 1（V3 再设 0 跑融合扫）
  MODE            matrix | live

  live 单细胞（MODE=live）:
  PRESENCE_BACKEND  默认 eres2netv2
  USE_SEP           默认 1
  LANG_SPLIT        默认 1
  ENROLL_VAD        默认 0（VP 默认对照，不默认开 VAD）
  HOLDOUT_FRAC      校准选 τ；默认 0.3
  FORCE_CALIB       1=重校准

产物:
  $VP_OUT/manifest/
  $VP_OUT/matrix_{vad,novad}/reports/     # MODE=matrix
  $VP_OUT/live_<tag>/                     # MODE=live
  $VP_OUT/reports/matrix.md               # 汇总

设计见 DESIGN.md。本脚本不跑 ASR。
EOF
}

for _a in "$@"; do
  case "$_a" in
    -h|--help|help) usage; exit 0 ;;
    -*) echo "[ERR] 未知选项 $_a；配置用环境变量。 $0 --help" >&2; exit 2 ;;
    *) echo "[ERR] 不支持位置参数 $_a" >&2; exit 2 ;;
  esac
done

export PYTHONPATH="$VE_ROOT/scripts:${VE_ROOT}/../scripts:${VE_ROOT}/../VD/tools:${VE_ROOT}/../VM/scripts:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
DEVICE="${DEVICE:-cuda:0}"
LIMIT="${LIMIT:-0}"
MODE="${MODE:-matrix}"
VP_OUT="${VP_OUT:-/root/autodl-tmp/vp}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/datasetA}"
BEST_SEP_DIR="${BEST_SEP_DIR:-/root/autodl-tmp/pos_neg/best_sep}"
ENCODERS="${ENCODERS:-eres2netv2,campplus,resnet34_lm}"
ARMS="${ARMS:-no_sep,sep_once,sep_multi}"
VAD_MODES="${VAD_MODES:-0,1}"
HOLDOUT_FRAC="${HOLDOUT_FRAC:-0.3}"
SEP_ROOT="${SEP_ROOT:-}"

mkdir -p "$VP_OUT/manifest" "$VP_OUT/reports" "$VP_OUT/logs"

echo "=== VP run ==="
echo "MODE=$MODE VP_OUT=$VP_OUT VE_ROOT=$VE_ROOT"

if [[ ! -f "${SAMPLES:-}" ]]; then
  if [[ -f "$VP_OUT/manifest/samples.jsonl" ]]; then
    SAMPLES="$VP_OUT/manifest/samples.jsonl"
  else
    echo ">>> build_manifest <<<"
    "$PYTHON_BIN" "$VE_ROOT/scripts/build_manifest.py" \
      --data-dir "$DATA_DIR" --best-sep "$BEST_SEP_DIR" \
      --out-dir "$VP_OUT/manifest"
    SAMPLES="$VP_OUT/manifest/samples.jsonl"
  fi
fi
echo "SAMPLES=$SAMPLES"

if [[ "$MODE" == "matrix" ]]; then
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
    echo "[ERR] MODE=matrix 需要 SEP_ROOT（含 d1/）。或改 MODE=live 现打分离。"
    exit 1
  fi
  echo "SEP_ROOT=$SEP_ROOT"
  IFS=',' read -r -a VADS <<< "$VAD_MODES"
  for v in "${VADS[@]}"; do
    v="$(echo "$v" | xargs)"
    [[ -z "$v" ]] && continue
    if [[ "$v" == "1" ]]; then tag=vad; eva=1; else tag=novad; eva=0; fi
    cell_out="$VP_OUT/matrix_${tag}"
    echo ">>> score_encoders VAD=$eva → $cell_out <<<"
    # 融合是 DESIGN V3；矩阵阶段默认不扫 fuse
    VE_OUT="$cell_out" ENROLL_VAD="$eva" SAMPLES="$SAMPLES" SEP_ROOT="$SEP_ROOT" \
      ENCODERS="$ENCODERS" ARMS="$ARMS" LIMIT="$LIMIT" DEVICE="$DEVICE" \
      SKIP_FUSE="${SKIP_FUSE:-1}" \
      bash "$VE_ROOT/score_encoders_on_sep.sh"
  done
  echo ">>> summarize <<<"
  "$PYTHON_BIN" "$VP_ROOT/scripts/summarize_cells.py" \
    --root "$VP_OUT" --out "$VP_OUT/reports/matrix.json" \
    --holdout-frac "$HOLDOUT_FRAC"
  echo "[OK] $VP_OUT/reports/matrix.md"
  exit 0
fi

if [[ "$MODE" == "live" ]]; then
  PRESENCE_BACKEND="${PRESENCE_BACKEND:-eres2netv2}"
  USE_SEP="${USE_SEP:-1}"
  LANG_SPLIT="${LANG_SPLIT:-1}"
  ENROLL_VAD="${ENROLL_VAD:-0}"
  if [[ "$ENROLL_VAD" == "1" ]]; then vtag=vad; else vtag=novad; fi
  if [[ "$USE_SEP" == "1" ]]; then stag=sep1; else stag=nosep; fi
  if [[ "$LANG_SPLIT" == "1" ]]; then ltag=ls; else ltag=gthr; fi
  CAL_DIR="$VP_OUT/live_${PRESENCE_BACKEND}_${stag}_${ltag}_${vtag}"
  mkdir -p "$CAL_DIR"
  if [[ -f "$CAL_DIR/recommended_thr.json" && "${FORCE_CALIB:-0}" != "1" ]]; then
    echo "[INFO] 复用 $CAL_DIR （FORCE_CALIB=1 重跑）"
  else
    CAL_ARGS=(
      --samples "$SAMPLES"
      --out-dir "$CAL_DIR"
      --presence-backend "$PRESENCE_BACKEND"
      --device "$DEVICE"
      --select-by contest
      --holdout-frac "$HOLDOUT_FRAC"
    )
    [[ "$USE_SEP" == "1" ]] && CAL_ARGS+=(--use-sep --sep-depth 1)
    [[ "$LANG_SPLIT" == "1" ]] && CAL_ARGS+=(--lang-split)
    [[ "$ENROLL_VAD" == "1" ]] && CAL_ARGS+=(--enroll-vad) || CAL_ARGS+=(--no-enroll-vad)
    [[ "$LIMIT" != "0" ]] && CAL_ARGS+=(--limit "$LIMIT")
    echo ">>> calibrate_presence ${CAL_ARGS[*]} <<<"
    "$PYTHON_BIN" "$VE_ROOT/scripts/calibrate_presence.py" "${CAL_ARGS[@]}"
  fi
  "$PYTHON_BIN" "$VP_ROOT/scripts/summarize_cells.py" \
    --root "$VP_OUT" --out "$VP_OUT/reports/matrix.json" \
    --holdout-frac "$HOLDOUT_FRAC"
  echo "[OK] live → $CAL_DIR"
  echo "[OK] $VP_OUT/reports/matrix.md"
  exit 0
fi

echo "[ERR] MODE=$MODE 应为 matrix 或 live"
exit 1

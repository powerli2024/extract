#!/usr/bin/env bash
# 下一刀入口（不改默认提交）。一次只动一个因子。
# 验收：真实 contest / RR / CER_total，并看 CER=1 桶。
#
# T1 解码（同一 mix 门控与波形）:
#   ./run_next_lift.sh t1
# T2 滑窗（须 FORCE_CALIB，另开 VE_OUT）:
#   ./run_next_lift.sh t2
# T3 时长不匹配二次解码（叠在已有 mix 提取上）:
#   ./run_next_lift.sh t3
# T4 离线 camp 否决（本地可跑，用 sssss 标定）:
#   ./run_next_lift.sh t4
# 门控: 相对锁定 +0.005 才改默认。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$ROOT/.env_ve" ]] && source "$ROOT/.env_ve" || true
# shellcheck disable=SC1091
source "$ROOT/pick_python.sh"
export PYTHONPATH="$ROOT/scripts:${ROOT}/../scripts:${PYTHONPATH:-}"
DEVICE="${DEVICE:-cuda:0}"
ASR_MODEL_DIR="${ASR_MODEL_DIR:-${QWEN3_ASR_DIR:-/root/autodl-tmp/Qwen3-ASR-1.7B}}"
SSSSS="${SSSSS_DIR:-}"
if [[ -z "$SSSSS" ]]; then
  for d in /root/autodl-tmp/datasetA/sssss "$ROOT/../../datasetA/sssss" "$ROOT/../datasetA/sssss"; do
    if [[ -d "$d" ]]; then SSSSS="$d"; break; fi
  done
  SSSSS="${SSSSS:-/root/autodl-tmp/datasetA/sssss}"
fi
cmd="${1:-help}"

# T1/T3 叠在已完成的 mix 提取上，不新建门控。
mix_ok() {
  local d="$1"
  [[ -f "$d/manifest/samples.jsonl" && -f "$d/results/all_results.jsonl" ]]
}

list_mix_outs() {
  local d
  for d in /root/autodl-tmp/ve_mix_novad /root/autodl-tmp/ve_mix_vad \
           /root/autodl-tmp/ve_mix /root/autodl-tmp/ve_presence_best; do
    if mix_ok "$d"; then
      echo "  $d"
    fi
  done
  # 其它 ve_* 目录
  shopt -s nullglob
  for d in /root/autodl-tmp/ve_*; do
    [[ -d "$d" ]] || continue
    mix_ok "$d" || continue
    case "$d" in
      */ve_mix_novad|*/ve_mix_vad|*/ve_mix|*/ve_presence_best) continue ;;
    esac
    echo "  $d"
  done
  shopt -u nullglob
}

require_mix_out() {
  local ve="${1:?}"
  if mix_ok "$ve"; then
    echo "$ve"
    return 0
  fi
  echo "[ERR] T1/T3 需要已完成的 mix 提取（manifest/samples.jsonl + results/all_results.jsonl）。" >&2
  echo "      当前 VE_OUT=$ve 不齐。" >&2
  echo >&2
  echo "这台机上已有的 VE 产物:" >&2
  local found
  found="$(list_mix_outs || true)"
  if [[ -n "$found" ]]; then
    echo "$found" >&2
    echo "换目录再跑: VE_OUT=<上面某一行> ./run_next_lift.sh $cmd" >&2
  else
    echo "  （没有找到任何 mix 产物）" >&2
    echo >&2
    echo "先跑锁定 mix（会写出 ve_mix_novad）:" >&2
    echo "  cd $ROOT" >&2
    echo "  ENROLL_VAD=0 PIPELINE=mix PRESENCE_BACKEND=eres2netv2 USE_SEP=1 LANG_SPLIT=1 \\" >&2
    echo "  FORCE_CALIB=1 HOLDOUT_FRAC=0.3 ./run_all.sh" >&2
  fi
  echo >&2
  echo "找一下: ls /root/autodl-tmp/ve_*/manifest/samples.jsonl" >&2
  echo "ve.sh 在 extract 根: cd /root/extract && ./ve.sh t1" >&2
  echo "在 ve/ 里请用: ./run_next_lift.sh t1  或  ./ve.sh t1（本目录也有入口）" >&2
  exit 2
}

t1() {
  local ve
  ve="$(require_mix_out "${VE_OUT:-/root/autodl-tmp/ve_mix_novad}")"
  echo "=== T1 decode ablation  VE_OUT=$ve ==="
  echo "[arm] auto (baseline, 若已有 asr_cer 可跳过)"
  # Chinese
  OUT="${ve}/reports/asr_cer_zh"
  mkdir -p "$OUT"
  "$PYTHON_BIN" "$ROOT/scripts/asr_cer.py" \
    --ve-out "$ve" --model-dir "$ASR_MODEL_DIR" --device "$DEVICE" \
    --language Chinese --out-dir "$OUT"
  echo "[arm] Chinese+domain-context"
  OUT2="${ve}/reports/asr_cer_zh_domain"
  mkdir -p "$OUT2"
  "$PYTHON_BIN" "$ROOT/scripts/asr_cer.py" \
    --ve-out "$ve" --model-dir "$ASR_MODEL_DIR" --device "$DEVICE" \
    --language Chinese --domain-context --out-dir "$OUT2"
  "$PYTHON_BIN" "$ROOT/asr_probe/scripts/eval_next_lift.py" \
    --sssss "$SSSSS" \
    --alt-asr "$OUT2/asr_results.jsonl" --alt-name zh_domain \
    --out "${ve}/reports/t1_eval.json" || true
  echo "对比: $ve/reports/asr_cer/summary.md vs asr_cer_zh vs asr_cer_zh_domain"
  echo "主看 n_cer1_accepted 与 contest_score_new"
}

t2() {
  echo "=== T2 CMD windows (slide) ==="
  echo "须重扫 τ：CMD_WINDOWS 改变分数几何"
  ENROLL_VAD=0 PIPELINE=mix \
  PRESENCE_BACKEND=eres2netv2 USE_SEP=1 LANG_SPLIT=1 \
  CMD_WINDOWS=slide WIN_SEC="${WIN_SEC:-0.8}" HOP_SEC="${HOP_SEC:-0.4}" \
  FORCE_CALIB=1 HOLDOUT_FRAC=0.3 \
  VE_OUT="${VE_OUT:-/root/autodl-tmp/ve_mix_win}" \
  "$ROOT/run_all.sh"
}

t3() {
  local ve
  ve="$(require_mix_out "${VE_OUT:-/root/autodl-tmp/ve_mix_novad}")"
  echo "=== T3 retry on duration mismatch  VE_OUT=$ve ==="
  OUT="${ve}/reports/asr_cer_retry"
  mkdir -p "$OUT"
  "$PYTHON_BIN" "$ROOT/scripts/asr_cer.py" \
    --ve-out "$ve" --model-dir "$ASR_MODEL_DIR" --device "$DEVICE" \
    --retry-mismatch --out-dir "$OUT"
  echo "看 $OUT/summary.md 的 CER=1 桶；默认回退 mix hyp"
}

t4() {
  echo "=== T4 camp veto (offline, 只否决) ==="
  "$PYTHON_BIN" "$ROOT/asr_probe/scripts/eval_next_lift.py" \
    --sssss "$SSSSS" \
    --out "${2:-$SSSSS/next_lift_eval.json}"
  echo "Go: test_dC>=0.005 且 n_extra_pos<=5 才把 VETO_CAMP=1 接入 extract"
}

case "$cmd" in
  t1) t1 ;;
  t2) t2 ;;
  t3) t3 ;;
  t4) t4 ;;
  all)
    t4
    echo "T1/T2/T3 需 AutoDL 上的 mix 提取 + Qwen3；先 t4 离线，再 t1。"
    ;;
  help|-h|--help|*)
    sed -n '2,16p' "$0"
    echo "用法: $0 t1|t2|t3|t4|all"
    exit 0
    ;;
esac

#!/usr/bin/env bash
# 下一刀入口（不改默认提交）。一次只动一个因子。
# 验收：真实 contest / RR / CER_total，并看 CER=1 桶。
#
# T0 冻结 τ + 叠话长句加拒（不重跑 ASR）:
#   ./run_next_lift.sh t0
# T1 解码（同一 mix 门控与波形）:
#   ./run_next_lift.sh t1
# T1b 热词 context（Qwen3 Vocabulary，不是指令）:
#   ./run_next_lift.sh t1b
# T2 滑窗 Presence + ASR 裁窗（须 FORCE_CALIB，另开 VE_OUT，勿覆盖 novad）:
#   ./run_next_lift.sh t2
# T2b 滑窗 Presence + 整段 mix ASR（优先复用 T2 决策，不重扫 τ）:
#   ./run_next_lift.sh t2b
# T3 时长不匹配二次解码（叠在已有 mix 提取上）:
#   ./run_next_lift.sh t3
# T4 离线 camp 否决（本地可跑，用 sssss 标定）:
#   ./run_next_lift.sh t4
# T5 Cond-TasNet（复用 mix 门控，另开 VE_OUT，勿覆盖 novad）:
#   ./run_next_lift.sh t5
# 提交：冻结 τ + 叠话加拒 → reports/submit/result.json（不重跑 Presence）:
#   VE_OUT=/root/autodl-tmp/ve_mix_novad ./run_next_lift.sh submit
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

t0() {
  local ve
  ve="$(require_mix_out "${VE_OUT:-/root/autodl-tmp/ve_mix_novad}")"
  echo "=== T0 叠话加拒 overlay  VE_OUT=$ve ==="
  local asr="${ve}/reports/asr_cer/asr_results.jsonl"
  if [[ ! -f "$asr" ]]; then
    echo "[ERR] 需要已有 $asr （先跑完 mix 的 asr_cer）" >&2
    exit 2
  fi
  if [[ "${SKIP_NEG_ASR:-0}" != "1" ]]; then
    echo "[arm] ASR 过门 neg（叠话加拒打在 FA 长 hyp 上）"
    "$PYTHON_BIN" "$ROOT/scripts/asr_cer.py" \
      --ve-out "$ve" --model-dir "$ASR_MODEL_DIR" --device "$DEVICE" \
      --neg-fa --out-dir "${ve}/reports/asr_neg_fa"
  fi
  echo "[arm] holdout τ + 文本加拒（--no-camp）"
  "$PYTHON_BIN" "$ROOT/scripts/apply_lift_overlay.py" \
    --ve-out "$ve" --asr-pos "$asr" --no-camp --tag holdout_text
  echo "[arm] 冻结锁定 τ + 文本加拒"
  "$PYTHON_BIN" "$ROOT/scripts/apply_lift_overlay.py" \
    --ve-out "$ve" --asr-pos "$asr" --no-camp --locked-thr --tag locked_text
  echo "看 $ve/reports/lift_overlay/holdout_text.md 与 locked_text.md"
  echo "n_need_asr>0 时冻结 τ 新放行的 pos 尚未 ASR，contest 偏保守"
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

t1b() {
  local ve
  ve="$(require_mix_out "${VE_OUT:-/root/autodl-tmp/ve_mix_novad}")"
  echo "=== T1b hotword context  VE_OUT=$ve ==="
  OUT="${ve}/reports/asr_cer_hotwords"
  mkdir -p "$OUT"
  "$PYTHON_BIN" "$ROOT/scripts/asr_cer.py" \
    --ve-out "$ve" --model-dir "$ASR_MODEL_DIR" --device "$DEVICE" \
    --hotwords --out-dir "$OUT"
  echo "对比: $ve/reports/asr_cer/summary.md vs asr_cer_hotwords vs asr_cer_zh_domain"
  echo "主看 CER=1 桶；泄漏: grep -c 'Vocabulary:' $OUT/asr_results.jsonl"
}

t2() {
  echo "=== T2 CMD windows (slide) ==="
  echo "须重扫 τ：CMD_WINDOWS 改变分数几何；ASR 裁 argmax 窗"
  if [[ "${VE_OUT:-}" == *ve_mix_novad* ]]; then
    echo "[ERR] 禁止覆盖 ve_mix_novad。T2 请用 VE_OUT=/root/autodl-tmp/ve_mix_win" >&2
    exit 2
  fi
  ENROLL_VAD=0 PIPELINE=mix \
  PRESENCE_BACKEND=eres2netv2 USE_SEP=1 LANG_SPLIT=1 \
  CMD_WINDOWS=slide WIN_SEC="${WIN_SEC:-0.8}" HOP_SEC="${HOP_SEC:-0.4}" \
  FORCE_CALIB=1 HOLDOUT_FRAC=0.3 \
  VE_OUT="${VE_OUT:-/root/autodl-tmp/ve_mix_win}" \
  "$ROOT/run_all.sh"
}

t2b() {
  # 只改 ASR 输入：Presence 仍用滑窗 max；ASR 打原始 CMD 整段。
  local src="${T2_SRC:-/root/autodl-tmp/ve_mix_win}"
  echo "=== T2b 滑窗 Presence + 整段 mix ASR ==="
  echo "禁止覆盖 ve_mix_novad；对照 T2 裁窗 CER，不要直接跟 novad 的 RR 比增益"
  if mix_ok "$src"; then
    local out="${src}/reports/asr_cer_fullmix"
    mkdir -p "$out"
    echo "[fast] 复用 $src 的 Presence 决策，ASR --wav-source mix → $out"
    "$PYTHON_BIN" "$ROOT/scripts/asr_cer.py" \
      --ve-out "$src" --model-dir "$ASR_MODEL_DIR" --device "$DEVICE" \
      --wav-source mix --out-dir "$out"
    echo "对比:"
    echo "  T2 裁窗: $src/reports/asr_cer/summary.md"
    echo "  T2b 整段: $out/summary.md"
    echo "  锁定 mix: /root/autodl-tmp/ve_mix_novad/reports/asr_cer/summary.md"
    echo "主看 CER=1 桶与 contest。RR 来自 T2 窗 τ，与 novad 不可互换。"
    return 0
  fi
  if [[ "${VE_OUT:-}" == *ve_mix_novad* ]]; then
    echo "[ERR] 无 $src 且 VE_OUT 指向 novad。改成 VE_OUT=/root/autodl-tmp/ve_mix_win_fullasr" >&2
    exit 2
  fi
  echo "[full] 未找到 $src，重跑 extract（ASR_CROP=0，另开目录）"
  local calib="/root/autodl-tmp/ve_presence_best/reports/presence_calib_eres2netv2_sep1_ls_novad_raw_win/recommended_thr.json"
  local skip_c=0
  if [[ -f "$calib" ]]; then
    echo "[INFO] 复用已有窗 τ: $calib"
    skip_c=1
  fi
  ENROLL_VAD=0 PIPELINE=mix \
  PRESENCE_BACKEND=eres2netv2 USE_SEP=1 LANG_SPLIT=1 \
  CMD_WINDOWS=slide ASR_CROP=0 WIN_SEC="${WIN_SEC:-0.8}" HOP_SEC="${HOP_SEC:-0.4}" \
  SKIP_CALIB="$skip_c" FORCE_CALIB="$([[ "$skip_c" == 1 ]] && echo 0 || echo 1)" HOLDOUT_FRAC=0.3 \
  VE_OUT="${VE_OUT:-/root/autodl-tmp/ve_mix_win_fullasr}" \
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

submit() {
  local ve
  ve="$(require_mix_out "${VE_OUT:-/root/autodl-tmp/ve_mix_novad}")"
  echo "=== submit 冻结 τ + 叠话加拒  VE_OUT=$ve ==="
  local asr="${ve}/reports/asr_cer/asr_results.jsonl"
  if [[ ! -f "$asr" ]]; then
    echo "[ERR] 需要已有 $asr" >&2
    exit 2
  fi
  if [[ "${SKIP_NEG_ASR:-0}" != "1" ]]; then
    echo "[arm] ASR 过门 neg"
    "$PYTHON_BIN" "$ROOT/scripts/asr_cer.py" \
      --ve-out "$ve" --model-dir "$ASR_MODEL_DIR" --device "$DEVICE" \
      --neg-fa --out-dir "${ve}/reports/asr_neg_fa"
  fi
  echo "[arm] locked τ + 文本加拒 → result.json"
  "$PYTHON_BIN" "$ROOT/scripts/apply_lift_overlay.py" \
    --ve-out "$ve" --asr-pos "$asr" --no-camp --locked-thr --write-result --tag submit
  echo "提交文件: $ve/reports/submit/result.json"
  echo "对照 T0 locked_text contest 应 ≈ 0.7389"
}

t5() {
  local src="${T5_SRC:-/root/autodl-tmp/ve_mix_novad}"
  local ve="${VE_OUT:-/root/autodl-tmp/ve_condtasnet_novad}"
  echo "=== T5 Cond-TasNet（复用 Presence，只换提取） ==="
  if [[ "$ve" == *ve_mix_novad* ]]; then
    echo "[ERR] 禁止覆盖 ve_mix_novad。T5 默认 VE_OUT=/root/autodl-tmp/ve_condtasnet_novad" >&2
    exit 2
  fi
  if ! mix_ok "$src"; then
    echo "[ERR] 需要已有 mix 门控: $src" >&2
    exit 2
  fi
  echo "[arm] 复用 $src 决策 → $ve"
  local extra=()
  [[ -n "${COND_TASNET_CKPT:-}" ]] && extra+=(--cond-tasnet-ckpt "$COND_TASNET_CKPT")
  [[ -n "${ECAPA_DIR:-}" ]] && extra+=(--ecapa-dir "$ECAPA_DIR")
  "$PYTHON_BIN" "$ROOT/scripts/reextract_from_gate.py" \
    --src-ve-out "$src" --out-dir "$ve" --tse-backend cond_tasnet \
    --device "$DEVICE" "${extra[@]}"
  echo "[arm] ASR"
  "$PYTHON_BIN" "$ROOT/scripts/asr_cer.py" \
    --ve-out "$ve" --model-dir "$ASR_MODEL_DIR" --device "$DEVICE"
  echo "[arm] 冻结 τ + 叠话加拒"
  if [[ "${SKIP_NEG_ASR:-0}" != "1" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/asr_cer.py" \
      --ve-out "$ve" --model-dir "$ASR_MODEL_DIR" --device "$DEVICE" \
      --neg-fa --out-dir "${ve}/reports/asr_neg_fa"
  fi
  "$PYTHON_BIN" "$ROOT/scripts/apply_lift_overlay.py" \
    --ve-out "$ve" --asr-pos "${ve}/reports/asr_cer/asr_results.jsonl" \
    --no-camp --locked-thr --write-result --tag submit
  echo "对照 mix 锁定: /root/autodl-tmp/ve_mix_novad/reports/lift_overlay/locked_text.json"
  echo "T5: $ve/reports/asr_cer/summary.md 与 $ve/reports/submit/result.json"
  echo "Go: 真实 contest ≥ 锁定 0.739 + 0.005 才改默认"
}

case "$cmd" in
  t0) t0 ;;
  t1) t1 ;;
  t1b) t1b ;;
  t2) t2 ;;
  t2b) t2b ;;
  t3) t3 ;;
  t4) t4 ;;
  t5) t5 ;;
  submit) submit ;;
  all)
    t4
    echo "T0 不需 GPU；T1/T1b/T2/T2b/T3/T5 需 mix 提取 + Qwen3。"
    ;;
  help|-h|--help|*)
    sed -n '2,26p' "$0"
    echo "用法: $0 t0|t1|t1b|t2|t2b|t3|t4|t5|submit|all"
    exit 0
    ;;
esac

#!/usr/bin/env bash
# VE 全流程：manifest → Presence 校准 → 提取 → ASR CER
# 用法: ./run_all.sh [--help|-h]
# 参数一律通过环境变量传入（见 --help）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
[[ -f "$ROOT/.env_ve" ]] && source "$ROOT/.env_ve" || true
# shellcheck disable=SC1091
source "$ROOT/pick_python.sh"

usage() {
  cat <<'EOF'
VE run_all.sh — Presence-gated TSE 全流程

用法:
  ./run_all.sh [--help|-h]
  PIPELINE=mix LIMIT=64 SKIP_ASR=1 ./run_all.sh
  ENROLL_VAD=0 HOLDOUT_FRAC=0.3 PIPELINE=ps4 ./run_all.sh

说明: 本脚本不解析位置参数；配置全部用环境变量（可写在 .env_ve）。

════════════════════════════════════════════════════════════
PIPELINE（提取后端）
════════════════════════════════════════════════════════════
  PIPELINE=ps4|wesep|sep_route|adaptive_route|mix|cond_tasnet     默认: ps4

  ps4         HF PS4 BSRNN（需 download_models.sh）
  wesep       WeSep bsrnn_ecapa（需 download_wesep.sh）
  sep_route   MossFormer 分离 + 声纹选路（需 Moss ONNX + extract/scripts）
  adaptive_route 仅当分离流对注册声纹的增益≥ROUTE_MIN_GAIN时选分离流；否则 mix
  mix         CMD 直通 ASR（当前端到端最强基线）
  cond_tasnet Cond-TasNet（需 COND_TASNET_CKPT + ECAPA；另开 VE_OUT）

  规划中（尚未接入）:
  usef        USEF-TSE（公开权重为 8 kHz；见下方「USEF 降采样」）

════════════════════════════════════════════════════════════
路径 / 设备 / 子集
════════════════════════════════════════════════════════════
  VE_OUT              输出根；默认 /root/autodl-tmp/ve_${PIPELINE}_${vad|novad}
  DATA_DIR            datasetA；默认 /root/autodl-tmp/datasetA
  BEST_SEP_DIR        干净 KWS enroll（sep 产物）。自行指定；未设则扫描
                      pos_neg/best_sep、kws_sep/best_sep 等现成目录
                      例: BEST_SEP_DIR=/root/autodl-tmp/kws_sep/best_sep ./run_all.sh
  STRICT_ENROLL       1=正式评测要求 best_sep 覆盖每一个 UID（默认 1）；
                      任何缺失、ok=false 或音频缺失均失败，绝不回退原始 KWS
  STRICT_EVAL         1=最终计分必须全 UID 覆盖、无重复且每个接受 pos 有 ASR（默认 1）
  SAMPLES             可选，跳过重建时指定 samples.jsonl
  DEVICE              默认 cuda:0
  LIMIT               分层抽样条数；0=全量（默认）
  PYTHON_BIN          python 可执行文件
  MOSS_ONNX_PATH      MossFormer ONNX
  PS4_WEIGHTS         PS4 checkpoint
  WESEP_MODEL_DIR     WeSep 模型目录
  WESEP_ROOT          额外 PYTHONPATH
  ASR_MODEL_DIR / QWEN3_ASR_DIR   ASR 权重

════════════════════════════════════════════════════════════
Presence 门控
════════════════════════════════════════════════════════════
  PRESENCE_BACKEND    eres2netv2（默认）| campplus | resnet34_lm | …
  USE_SEP             1=Presence 用一次分离 max-cosine（默认 1）
                      sep_route 强制为 1
  LANG_SPLIT          1=按 zh/en 分 thr（默认 1）
  ENROLL_VAD          1=enroll 能量 VAD；0=关闭。提交默认 0（LOCKED_THR=1 会强制 0）
                      → VE_OUT / 校准目录带 _vad 或 _novad，结果可并存
  LOCKED_THR          1=用仓库冻结 τ（zh 0.29305 / en 0.357868），不扫 thr（默认 1）
                      FORCE_CALIB=1 或 CMD_WINDOWS≠off 时自动关闭
  EXTRA_REJECT        1=ASR 后叠话长句加拒并写 result.json（默认 1）
  TARGET_FRR          校准辅助目标；默认 0.02
  HOLDOUT_FRAC        >0 时仅在 calib 子集选 thr，并报 holdout contest
                      （建议 0.3；默认 0=同集扫 thr，易乐观）
  CMD_WINDOWS         off|slide|energy  CMD 滑窗打分（默认 off）
                      开了必须 FORCE_CALIB=1（分数几何变了）
                      默认 ASR 裁 argmax 窗；ASR_CROP=0 则整段 mix ASR（T2b）
  VETO_CAMP=1         灰区 camp 否决（只加拒）
  VETO_WINDOWS=1      灰区次优窗否决
  ASR_LANGUAGE        强制 Qwen3 language（如 Chinese）；空=自动
  ASR_HOTWORDS=1      Qwen3 热词表 context（推荐于指令式 domain）
  ASR_DOMAIN_CONTEXT=1  智能家居指令 context，不用唤醒词
  ASR_RETRY_MISMATCH=1  hyp 与时长不匹配时二次解码，回退 mix
  ASR_RESUME         1=仅在音频与配置未变时手动复用 ASR；正式跑分默认 0（禁用复用）
  ROUTE_MIN_GAIN  adaptive_route 最小声纹增益，默认 0.03
  HOLDOUT_SEED        holdout 随机种子；默认 0
  ASNORM              1=AS-Norm（需 cohort）
  ENROLL_ZNORM        1=仅 enroll Z-Norm
  COHORT_DIR          enroll cohort；默认 clean_kws
  TEST_COHORT_DIR     test cohort；默认 mix500

════════════════════════════════════════════════════════════
校准目录（多 PIPELINE 共享 thr）
════════════════════════════════════════════════════════════
  CALIB_ROOT          默认 /root/autodl-tmp/ve_presence_best
  CALIB_DIR           默认
    $CALIB_ROOT/reports/presence_calib_<backend>_<sep|nosep>_<ls|gthr>_<vad|novad>_<norm>
  SKIP_CALIB=1        强制跳过校准（须已有 THR）
  FORCE_CALIB=1       强制重跑校准（同时关闭 LOCKED_THR）
  LOCKED_THR=1        使用 ve/configs/locked_thr.json，跳过校准

════════════════════════════════════════════════════════════
流程开关
════════════════════════════════════════════════════════════
  SKIP_ASR=1          只到 extract，不跑 ASR CER
  对照多后端:         ./run_compare_pipelines.sh

════════════════════════════════════════════════════════════
产物
════════════════════════════════════════════════════════════
  $VE_OUT/manifest/samples.jsonl
  $VE_OUT/results/{pos,neg,all}_results.jsonl
  $VE_OUT/extracted/{pos,neg}/*.wav
  $VE_OUT/reports/presence_calib/  asr_cer/  lift_overlay/  submit/result.json  logs/

════════════════════════════════════════════════════════════
USEF-TSE 与 16 kHz→8 kHz 降采样（规划，接入前必读）
════════════════════════════════════════════════════════════
  公开 USEF 权重按 8 kHz 训练；本流水线 wav / ASR 为 16 kHz。
  若「直接使用」预训练 ckpt，必须成对处理:

    16k mix/enroll  ──resample↓──►  8k ──USEF──► 8k est ──resample↑──► 16k → ASR

  降采样方法（按推荐优先级）:
    1) resample_poly(up=1, down=2) 或 soxr/librosa kaiser_best
       — 有抗混叠低通，2:1 有理倍数，相位稳定；enroll/mix 必须同一实现
    2) torchaudio.functional.resample(16000→8000) 高质量模式
    3) librosa.resample(..., res_type="kaiser_fast") — 更快，高频略差
    禁止: 隔点抽取 / 无低通的线性抽稀 — 混叠会进语音带，伤分离与 CER

  主要影响:
    • Nyquist 降至 4 kHz：擦音/齿音/部分中文辅音能量丢失，ASR CER 常变差
    • 升回 16 kHz 无法恢复已丢高频；ASR 看到的是带宽受限语音
    • Presence 仍建议在 16 kHz 上打分；仅 TSE 段走 8 kHz，避免 thr 尺度被拖垮
    • 与 mix@16k 基线比时，必须同一 Presence 决策集，否则 RR/CER 不可比

  接入前建议冒烟:
    LIMIT 子集上对比 (a) mix@16k (b) USEF+soxr (c) USEF+poly
    看 SI 代理（若有）与真实 ASR CER；若 (b)(c) 均不及 mix，则勿换默认 PIPELINE。

  许可: USEF 官方 CC BY-NC 4.0（非商业）。细节见 NOTES_USEF_RESAMPLE.md

════════════════════════════════════════════════════════════
示例
════════════════════════════════════════════════════════════
  PIPELINE=mix LIMIT=64 SKIP_ASR=1 ./run_all.sh
  PIPELINE=ps4 ./run_all.sh
  ENROLL_VAD=0 PIPELINE=mix ./run_all.sh
  HOLDOUT_FRAC=0.3 FORCE_CALIB=1 PIPELINE=mix ./run_all.sh
  PIPELINES=mix,ps4 ./run_compare_pipelines.sh

更多: README.md / AUTODL.md
EOF
}

for _arg in "$@"; do
  case "$_arg" in
    -h|--help|help)
      usage
      exit 0
      ;;
    -*)
      echo "[ERR] 未知选项: $_arg（配置请用环境变量）。试: $0 --help" >&2
      exit 2
      ;;
    *)
      echo "[ERR] 不支持位置参数: $_arg。试: $0 --help" >&2
      exit 2
      ;;
  esac
done

PIPELINE_RAW="${PIPELINE:-ps4}"
PIPELINE_LC="$(echo "$PIPELINE_RAW" | tr '[:upper:]' '[:lower:]')"
case "$PIPELINE_LC" in
  ps4|ps4_bsrnn|bsrnn) PIPELINE=ps4; TSE_BACKEND=ps4 ;;
  wesep|wesep_bsrnn|wesep_bsrnn_ecapa) PIPELINE=wesep; TSE_BACKEND=wesep_bsrnn ;;
  sep_route|mossformer|route|sep) PIPELINE=sep_route; TSE_BACKEND=sep_route ;;
  adaptive_route|adaptive|mix_sep_route) PIPELINE=adaptive_route; TSE_BACKEND=adaptive_route ;;
  mix|passthrough|cmd|none) PIPELINE=mix; TSE_BACKEND=mix ;;
  cond_tasnet|condtasnet|tasnet|cond-tasnet) PIPELINE=cond_tasnet; TSE_BACKEND=cond_tasnet ;;
  usef|usef_tse|usef-tse)
    echo "[ERR] PIPELINE=usef 尚未接入。请先阅读: $0 --help（USEF 降采样一节）与 NOTES_USEF_RESAMPLE.md"
    exit 1
    ;;
  *)
    echo "[ERR] 未知 PIPELINE=${PIPELINE_RAW}；可选: ps4 | wesep | sep_route | adaptive_route | mix | cond_tasnet"
    echo "      运行 $0 --help 查看全部环境变量"
    exit 1
    ;;
esac

export PYTHONPATH="$ROOT/scripts:${ROOT}/../scripts:${ROOT}/../VM/scripts:${ROOT}/../VD/tools:${PYTHONPATH:-}"
if [[ -n "${WESEP_ROOT:-}" ]]; then
  export PYTHONPATH="${WESEP_ROOT}:${PYTHONPATH}"
elif [[ -d /root/autodl-tmp/ve_models/wesep/wesep ]]; then
  export PYTHONPATH="/root/autodl-tmp/ve_models/wesep:${PYTHONPATH}"
fi

DATA_DIR="${DATA_DIR:-/root/autodl-tmp/datasetA}"
BEST_SEP_ARGS=()
if [[ -n "${BEST_SEP_DIR:-}" ]]; then
  BEST_SEP_ARGS+=(--best-sep "$BEST_SEP_DIR")
fi
DEVICE="${DEVICE:-cuda:0}"
LIMIT="${LIMIT:-0}"
STRICT_ENROLL="${STRICT_ENROLL:-1}"
STRICT_EVAL="${STRICT_EVAL:-1}"
ASR_RESUME="${ASR_RESUME:-0}"
TARGET_FRR="${TARGET_FRR:-0.02}"
PRESENCE_BACKEND="${PRESENCE_BACKEND:-eres2netv2}"
# 最新最佳拒识：Presence 用一次分离 + 语言分 thr；默认 raw（不开 AS-Norm）
USE_SEP="${USE_SEP:-1}"
LANG_SPLIT="${LANG_SPLIT:-1}"
ENROLL_VAD="${ENROLL_VAD:-1}"
HOLDOUT_FRAC="${HOLDOUT_FRAC:-0}"
MOSS_ONNX_PATH="${MOSS_ONNX_PATH:-/root/autodl-tmp/checkpoints/MossFormer2_ONNX/simple_model.onnx}"
export MOSS_ONNX_PATH

# sep_route 强制 Presence 用分离（须在打标签前）
if [[ "$PIPELINE" == "sep_route" || "$PIPELINE" == "adaptive_route" ]]; then
  USE_SEP=1
fi

# 提交默认：冻结 τ + 叠话加拒。实验臂 FORCE_CALIB / 滑窗会关掉锁定 τ。
LOCKED_THR="${LOCKED_THR:-1}"
EXTRA_REJECT="${EXTRA_REJECT:-1}"
CMD_WINDOWS="${CMD_WINDOWS:-off}"
ASR_CROP="${ASR_CROP:-1}"
case "$(echo "$CMD_WINDOWS" | tr '[:upper:]' '[:lower:]')" in
  1|true|on|yes|slide) CMD_WINDOWS=slide; WIN_TAG=win ;;
  energy|vad|seg) CMD_WINDOWS=energy; WIN_TAG=wenergy ;;
  *) CMD_WINDOWS=off; WIN_TAG=nowin ;;
esac
if [[ "${FORCE_CALIB:-0}" == "1" || "$CMD_WINDOWS" != "off" ]]; then
  if [[ "$LOCKED_THR" == "1" ]]; then
    echo "[INFO] FORCE_CALIB/CMD_WINDOWS → LOCKED_THR=0（实验臂，不用提交冻结 τ）"
  fi
  LOCKED_THR=0
fi
if [[ "$LOCKED_THR" == "1" ]]; then
  if [[ "$ENROLL_VAD" == "1" ]]; then
    echo "[INFO] LOCKED_THR=1 对应 ENROLL_VAD=0（提交配方）"
    ENROLL_VAD=0
  fi
fi

# VAD / 非 VAD 并存：默认 VE_OUT 与校准目录带标签，避免互相覆盖
if [[ "$ENROLL_VAD" == "1" ]]; then VAD_TAG=vad; else VAD_TAG=novad; fi
if [[ "$USE_SEP" == "1" ]]; then SEP_TAG=sep1; else SEP_TAG=nosep; fi
if [[ "$LANG_SPLIT" == "1" ]]; then LS_TAG=ls; else LS_TAG=gthr; fi
SCORE_NORM_TAG=raw
[[ "${ASNORM:-0}" == "1" ]] && SCORE_NORM_TAG=asnorm
[[ "${ENROLL_ZNORM:-0}" == "1" && "$SCORE_NORM_TAG" == "raw" ]] && SCORE_NORM_TAG=enroll_znorm
VETO_CAMP="${VETO_CAMP:-0}"
VETO_WINDOWS="${VETO_WINDOWS:-0}"

if [[ -z "${VE_OUT:-}" || "${VE_OUT}" == "/root/autodl-tmp/ve" ]]; then
  VE_OUT="/root/autodl-tmp/ve_${PIPELINE}_${VAD_TAG}"
  if [[ "$WIN_TAG" != "nowin" ]]; then
    VE_OUT="${VE_OUT}_${WIN_TAG}"
    [[ "$ASR_CROP" == "0" ]] && VE_OUT="${VE_OUT}_fullasr"
  fi
fi
export VE_OUT

# 共享校准目录（多 PIPELINE 对照时只校准一次）；按 backend/sep/lang/vad/norm/win 分桶
CALIB_ROOT="${CALIB_ROOT:-/root/autodl-tmp/ve_presence_best}"
CALIB_STEM="presence_calib_${PRESENCE_BACKEND}_${SEP_TAG}_${LS_TAG}_${VAD_TAG}_${SCORE_NORM_TAG}"
if [[ "$WIN_TAG" != "nowin" ]]; then
  CALIB_STEM="${CALIB_STEM}_${WIN_TAG}"
fi
CALIB_DIR="${CALIB_DIR:-$CALIB_ROOT/reports/${CALIB_STEM}}"
THR_FILE="$CALIB_DIR/recommended_thr.json"
LOCKED_THR_FILE="$ROOT/configs/locked_thr.json"
if [[ "$LOCKED_THR" == "1" ]]; then
  if [[ ! -f "$LOCKED_THR_FILE" ]]; then
    echo "[ERR] LOCKED_THR=1 但缺少 $LOCKED_THR_FILE"; exit 1
  fi
  THR_FILE="$LOCKED_THR_FILE"
  SKIP_CALIB=1
fi

mkdir -p "$VE_OUT"/{manifest,results,extracted,reports,logs}
mkdir -p "$CALIB_DIR"

ts() { date '+%Y%m%d_%H%M%S'; }
LOG="$VE_OUT/logs/run_$(ts).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== VE run_all ==="
echo "PIPELINE=$PIPELINE TSE_BACKEND=$TSE_BACKEND"
echo "Presence: backend=$PRESENCE_BACKEND USE_SEP=$USE_SEP LANG_SPLIT=$LANG_SPLIT ENROLL_VAD=$ENROLL_VAD ASNORM=${ASNORM:-0} HOLDOUT_FRAC=$HOLDOUT_FRAC LOCKED_THR=$LOCKED_THR EXTRA_REJECT=$EXTRA_REJECT"
echo "VE_OUT=$VE_OUT CALIB_DIR=$CALIB_DIR VAD_TAG=$VAD_TAG CMD_WINDOWS=$CMD_WINDOWS ASR_CROP=$ASR_CROP"
echo "DATA_DIR=$DATA_DIR BEST_SEP=${BEST_SEP_DIR:-"(auto scan)"} DEVICE=$DEVICE"
echo "MOSS_ONNX_PATH=$MOSS_ONNX_PATH"
echo "PYTHON_BIN=$PYTHON_BIN"
"$PYTHON_BIN" -c "import sys; print('sys.executable=', sys.executable)"
case "$(echo "$PRESENCE_BACKEND" | tr '[:upper:]' '[:lower:]')" in
  eres2netv2|eres|eres2net|campplus|cam++|campplus_zh)
    if ! ensure_modelscope "$PYTHON_BIN"; then
      echo "[ERR] Presence 需要 modelscope，且必须装进上面这个 PYTHON_BIN（conda env ve）。"
      exit 1
    fi
    ;;
esac

nvidia-smi || true

need_moss=0
[[ "$USE_SEP" == "1" ]] && need_moss=1
[[ "$PIPELINE" == "sep_route" || "$PIPELINE" == "adaptive_route" ]] && need_moss=1

if [[ "$PIPELINE" == "wesep" ]]; then
  if ! "$PYTHON_BIN" -c "from wesep.cli.extractor import load_model" 2>/dev/null; then
    echo "[ERR] WeSep 未就绪。请先: ./download_wesep.sh"
    exit 1
  fi
fi
if [[ "$need_moss" == "1" ]]; then
  if [[ ! -f "$MOSS_ONNX_PATH" ]]; then
    echo "[ERR] 缺少 MossFormer ONNX: $MOSS_ONNX_PATH"
    echo "      请先: ./download_moss_onnx.sh"
    exit 1
  fi
  if ! "$PYTHON_BIN" -c "import onnxruntime" 2>/dev/null; then
    echo "[ERR] 需要 onnxruntime-gpu。pip install onnxruntime-gpu"
    exit 1
  fi
  if ! "$PYTHON_BIN" -c "import sys; sys.path.insert(0,'$ROOT/../scripts'); from mossformer2_onnx import MossFormer2Separator" 2>/dev/null \
    && ! "$PYTHON_BIN" -c "import sys; sys.path.insert(0,'$ROOT/../VM/scripts'); from mossformer2_onnx import MossFormer2Separator" 2>/dev/null \
    && ! "$PYTHON_BIN" -c "from mossformer2_onnx import MossFormer2Separator" 2>/dev/null; then
    echo "[ERR] 无法 import mossformer2_onnx。extract 仓应有 ../scripts/mossformer2_onnx.py"
    exit 1
  fi
fi
if [[ "$PIPELINE" == "ps4" ]]; then
  PS4="${PS4_WEIGHTS:-/root/autodl-tmp/ve_models/PS4/checkpoint_epoch037.pt}"
  if [[ ! -f "$PS4" ]]; then
    echo "[WARN] 未找到 PS4 权重: $PS4 （extract 时可能失败）"
  fi
fi
if [[ "$PIPELINE" == "cond_tasnet" ]]; then
  CK="${COND_TASNET_CKPT:-/root/autodl-tmp/ve_models/cond_tasnet/best.pt}"
  if [[ ! -f "$CK" ]]; then
    for alt in /root/autodl-tmp/models/cond_tasnet/best.pt \
               /root/autodl-tmp/ve_models/cond_tasnet_v35/best.pt; do
      [[ -f "$alt" ]] && CK="$alt" && break
    done
  fi
  if [[ ! -f "$CK" ]]; then
    echo "[ERR] 缺少 Cond-TasNet 权重。设 COND_TASNET_CKPT 或放到:"
    echo "      /root/autodl-tmp/ve_models/cond_tasnet/best.pt"
    exit 1
  fi
  export COND_TASNET_CKPT="$CK"
  echo "[INFO] COND_TASNET_CKPT=$CK"
fi

step() { echo; echo ">>> $* <<<"; }

step "[1/7] build_manifest"
if [[ "$STRICT_ENROLL" == "1" ]]; then
  BEST_SEP_ARGS+=(--strict-best-sep)
fi
"$PYTHON_BIN" "$ROOT/scripts/build_manifest.py" \
  --data-dir "$DATA_DIR" \
  "${BEST_SEP_ARGS[@]}" \
  --out-dir "$VE_OUT/manifest"
# 对照时复用同一份 samples：若 CALIB_ROOT 尚无 manifest，拷一份
if [[ ! -f "$CALIB_ROOT/manifest/samples.jsonl" ]]; then
  mkdir -p "$CALIB_ROOT/manifest"
  cp -f "$VE_OUT/manifest/samples.jsonl" "$CALIB_ROOT/manifest/samples.jsonl" || true
  [[ -f "$VE_OUT/manifest/qc.json" ]] && cp -f "$VE_OUT/manifest/qc.json" "$CALIB_ROOT/manifest/" || true
fi
SAMPLES="${SAMPLES:-$VE_OUT/manifest/samples.jsonl}"

step "[2/7] calibrate_presence (shared thr)"
if [[ "$LOCKED_THR" == "1" ]]; then
  echo "[INFO] LOCKED_THR=1 使用冻结 τ: $THR_FILE （跳过校准）"
elif [[ "${SKIP_CALIB:-0}" == "1" && -f "$THR_FILE" ]]; then
  echo "[INFO] SKIP_CALIB=1 且已有 $THR_FILE"
elif [[ -f "$THR_FILE" && "${FORCE_CALIB:-0}" != "1" ]]; then
  echo "[INFO] 复用已有校准: $THR_FILE （FORCE_CALIB=1 可重跑）"
else
  CAL_ARGS=(
    --samples "$SAMPLES"
    --out-dir "$CALIB_DIR"
    --presence-backend "$PRESENCE_BACKEND"
    --device "$DEVICE"
    --target-frr "$TARGET_FRR"
    --select-by contest
  )
  [[ "$USE_SEP" == "1" ]] && CAL_ARGS+=(--use-sep --sep-depth 1)
  [[ "$LANG_SPLIT" == "1" ]] && CAL_ARGS+=(--lang-split)
  [[ "$ENROLL_VAD" == "1" ]] && CAL_ARGS+=(--enroll-vad) || CAL_ARGS+=(--no-enroll-vad)
  if [[ "$CMD_WINDOWS" != "off" ]]; then
    CAL_ARGS+=(--cmd-windows "$CMD_WINDOWS" --win-sec "${WIN_SEC:-0.8}" --hop-sec "${HOP_SEC:-0.4}")
  fi
  [[ "$LIMIT" != "0" ]] && CAL_ARGS+=(--limit "$LIMIT")
  if [[ -n "${HOLDOUT_FRAC:-}" && "$HOLDOUT_FRAC" != "0" ]]; then
    CAL_ARGS+=(--holdout-frac "$HOLDOUT_FRAC")
    [[ -n "${HOLDOUT_SEED:-}" ]] && CAL_ARGS+=(--holdout-seed "$HOLDOUT_SEED")
  fi
  if [[ "${ASNORM:-0}" == "1" ]]; then
    COHORT_DIR="${COHORT_DIR:-/root/autodl-tmp/clean_kws}"
    TEST_COHORT_DIR="${TEST_COHORT_DIR:-/root/autodl-tmp/mix500}"
    CAL_ARGS+=(--asnorm --cohort-dir "$COHORT_DIR" --test-cohort-dir "$TEST_COHORT_DIR")
    echo "[INFO] AS-Norm A=$COHORT_DIR B=$TEST_COHORT_DIR"
  elif [[ "${ENROLL_ZNORM:-0}" == "1" ]] || [[ -n "${COHORT_DIR:-}" ]]; then
    COHORT_DIR="${COHORT_DIR:-/root/autodl-tmp/clean_kws}"
    CAL_ARGS+=(--enroll-znorm --cohort-dir "$COHORT_DIR")
  fi
  "$PYTHON_BIN" "$ROOT/scripts/calibrate_presence.py" "${CAL_ARGS[@]}"
fi
if [[ ! -f "$THR_FILE" ]]; then
  echo "[ERR] 缺少 $THR_FILE"; exit 1
fi
# 同步一份到本 PIPELINE reports，方便查看
mkdir -p "$VE_OUT/reports/presence_calib"
cp -f "$THR_FILE" "$VE_OUT/reports/presence_calib/recommended_thr.json" || true
[[ -f "$CALIB_DIR/calibration.md" ]] && cp -f "$CALIB_DIR/calibration.md" "$VE_OUT/reports/presence_calib/" || true

step "[3/7] run_extract pipeline=$PIPELINE"
EXT_ARGS=(
  --samples "$SAMPLES"
  --out-dir "$VE_OUT"
  --presence-backend "$PRESENCE_BACKEND"
  --thr-file "$THR_FILE"
  --device "$DEVICE"
  --tse-backend "$TSE_BACKEND"
  --no-score-norm
)
[[ "$USE_SEP" == "1" ]] && EXT_ARGS+=(--use-sep --sep-depth 1)
[[ "$ENROLL_VAD" == "1" ]] && EXT_ARGS+=(--enroll-vad) || EXT_ARGS+=(--no-enroll-vad)
[[ "$LIMIT" != "0" ]] && EXT_ARGS+=(--limit "$LIMIT")
[[ -n "${WESEP_MODEL_DIR:-}" ]] && EXT_ARGS+=(--wesep-model-dir "$WESEP_MODEL_DIR")
if [[ "${ASNORM:-0}" == "1" ]]; then
  COHORT_DIR="${COHORT_DIR:-/root/autodl-tmp/clean_kws}"
  TEST_COHORT_DIR="${TEST_COHORT_DIR:-/root/autodl-tmp/mix500}"
  EXT_ARGS=(
    --samples "$SAMPLES"
    --out-dir "$VE_OUT"
    --presence-backend "$PRESENCE_BACKEND"
    --thr-file "$THR_FILE"
    --device "$DEVICE"
    --tse-backend "$TSE_BACKEND"
    --asnorm --cohort-dir "$COHORT_DIR" --test-cohort-dir "$TEST_COHORT_DIR"
  )
  [[ "$USE_SEP" == "1" ]] && EXT_ARGS+=(--use-sep --sep-depth 1)
  [[ "$ENROLL_VAD" == "1" ]] && EXT_ARGS+=(--enroll-vad) || EXT_ARGS+=(--no-enroll-vad)
  [[ "$LIMIT" != "0" ]] && EXT_ARGS+=(--limit "$LIMIT")
  [[ -n "${WESEP_MODEL_DIR:-}" ]] && EXT_ARGS+=(--wesep-model-dir "$WESEP_MODEL_DIR")
fi
if [[ "$CMD_WINDOWS" != "off" ]]; then
  EXT_ARGS+=(--cmd-windows "$CMD_WINDOWS" --win-sec "${WIN_SEC:-0.8}" --hop-sec "${HOP_SEC:-0.4}" --win-pad-ms "${WIN_PAD_MS:-80}")
  [[ "$ASR_CROP" == "0" ]] && EXT_ARGS+=(--no-asr-crop)
fi
if [[ "$VETO_CAMP" == "1" ]]; then
  EXT_ARGS+=(--veto-backend "${VETO_BACKEND:-campplus}" --veto-margin "${VETO_MARGIN:-0.12}")
fi
[[ "$VETO_WINDOWS" == "1" ]] && EXT_ARGS+=(--veto-windows)
[[ "$PIPELINE" == "cond_tasnet" && -n "${COND_TASNET_CKPT:-}" ]] && EXT_ARGS+=(--cond-tasnet-ckpt "$COND_TASNET_CKPT")
[[ "$PIPELINE" == "adaptive_route" ]] && EXT_ARGS+=(--route-min-gain "${ROUTE_MIN_GAIN:-0.03}")
[[ -n "${ECAPA_DIR:-}" ]] && EXT_ARGS+=(--ecapa-dir "$ECAPA_DIR")

echo "[CMD] $PYTHON_BIN $ROOT/scripts/run_extract.py ${EXT_ARGS[*]}"
echo "[INFO] samples=$SAMPLES thr=$THR_FILE"
if [[ ! -f "$SAMPLES" ]]; then
  echo "[ERR] samples 不存在: $SAMPLES"; exit 1
fi
if [[ ! -f "$ROOT/scripts/presence_thr.py" ]]; then
  echo "[ERR] 缺少 $ROOT/scripts/presence_thr.py — 请同步最新 VE/scripts"
  exit 1
fi
if [[ ! -f "$ROOT/scripts/tse_mix.py" ]]; then
  echo "[ERR] 缺少 $ROOT/scripts/tse_mix.py — 请同步最新 VE/scripts"
  exit 1
fi

set +e
PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$ROOT/scripts/run_extract.py" "${EXT_ARGS[@]}"
ext_ec=$?
set -e
if [[ "$ext_ec" -ne 0 ]]; then
  echo "[ERR] run_extract 失败 exit=$ext_ec"
  exit "$ext_ec"
fi
if [[ ! -f "$VE_OUT/results/all_results.jsonl" ]]; then
  echo "[ERR] run_extract 未写出 $VE_OUT/results/all_results.jsonl"
  echo "      请确认已同步: run_extract.py / presence_thr.py / tse_mix.py / tse_factory.py"
  exit 1
fi
n_res="$("$PYTHON_BIN" -c "print(sum(1 for _ in open('$VE_OUT/results/all_results.jsonl',encoding='utf-8')))")"
echo "[OK] extract 完成 n=$n_res → $VE_OUT/results/all_results.jsonl"

step "[4/7] tse_ab placeholder report"
"$PYTHON_BIN" "$ROOT/scripts/tse_ab.py" --ve-out "$VE_OUT" || true

step "[5/7] asr_cer"
if [[ "${SKIP_ASR:-0}" == "1" ]]; then
  echo "[INFO] SKIP_ASR=1；稍后: VE_OUT=$VE_OUT ./run_asr_cer.sh"
else
  ASR_ARGS=(--ve-out "$VE_OUT" --device "$DEVICE")
  [[ -n "${ASR_MODEL_DIR:-}${QWEN3_ASR_DIR:-}" ]] && \
    ASR_ARGS+=(--model-dir "${ASR_MODEL_DIR:-$QWEN3_ASR_DIR}")
  [[ "$LIMIT" != "0" ]] && ASR_ARGS+=(--limit "$LIMIT")
  [[ -n "${ASR_LANGUAGE:-}" ]] && ASR_ARGS+=(--language "$ASR_LANGUAGE")
  [[ "${ASR_HOTWORDS:-0}" == "1" ]] && ASR_ARGS+=(--hotwords)
  [[ "${ASR_DOMAIN_CONTEXT:-0}" == "1" ]] && ASR_ARGS+=(--domain-context)
  [[ "${ASR_RETRY_MISMATCH:-0}" == "1" ]] && ASR_ARGS+=(--retry-mismatch)
  [[ "$ASR_RESUME" != "1" ]] && ASR_ARGS+=(--no-resume)
  if ! "$PYTHON_BIN" "$ROOT/scripts/asr_cer.py" "${ASR_ARGS[@]}"; then
    if [[ "$STRICT_EVAL" == "1" ]]; then
      echo "[ERR] ASR 失败；STRICT_EVAL=1 禁止复用历史报告继续跑分"; exit 1
    fi
    echo "[WARN] asr_cer 失败（非严格模式保留 extract 结果）"
  fi
fi

step "[6/7] extra_reject + submit result.json"
FINAL_DECISIONS="$VE_OUT/results/all_results.jsonl"
if [[ "$EXTRA_REJECT" != "1" ]]; then
  echo "[INFO] EXTRA_REJECT=0，跳过叠话加拒"
elif [[ ! -f "$VE_OUT/reports/asr_cer/asr_results.jsonl" ]]; then
  echo "[WARN] 无 asr_cer jsonl，跳过 overlay。稍后: VE_OUT=$VE_OUT ./run_next_lift.sh submit"
else
  OVERLAY_NO_NEG_ASR=0
  if [[ "${SKIP_NEG_ASR:-0}" != "1" ]]; then
    echo "[arm] ASR 过门 neg（叠话加拒）"
    NEG_ASR_ARGS=(--ve-out "$VE_OUT" --model-dir "${ASR_MODEL_DIR:-${QWEN3_ASR_DIR:-/root/autodl-tmp/Qwen3-ASR-1.7B}}" \
      --device "$DEVICE" --neg-fa --out-dir "$VE_OUT/reports/asr_neg_fa")
    [[ "$ASR_RESUME" != "1" ]] && NEG_ASR_ARGS+=(--no-resume)
    if ! "$PYTHON_BIN" "$ROOT/scripts/asr_cer.py" "${NEG_ASR_ARGS[@]}"; then
      if [[ "$STRICT_EVAL" == "1" ]]; then
        echo "[ERR] neg-fa ASR 失败；STRICT_EVAL=1 禁止使用历史 neg ASR"; exit 1
      fi
      echo "[WARN] neg-fa ASR 失败；本轮禁用文本加拒的 neg hyp"
      OVERLAY_NO_NEG_ASR=1
    fi
  else
    OVERLAY_NO_NEG_ASR=1
  fi
  OVERLAY_ARGS=(--ve-out "$VE_OUT" --asr-pos "$VE_OUT/reports/asr_cer/asr_results.jsonl" --no-camp --write-result --tag submit)
  [[ "$OVERLAY_NO_NEG_ASR" == "1" ]] && OVERLAY_ARGS+=(--no-neg-asr)
  [[ "$LOCKED_THR" == "1" ]] && OVERLAY_ARGS+=(--locked-thr)
  "$PYTHON_BIN" "$ROOT/scripts/apply_lift_overlay.py" "${OVERLAY_ARGS[@]}"
  OVERLAY_ROWS="$VE_OUT/reports/lift_overlay/submit_rows.jsonl"
  if [[ ! -f "$OVERLAY_ROWS" ]]; then
    echo "[ERR] overlay 未写出 $OVERLAY_ROWS"; exit 1
  fi
  FINAL_DECISIONS="$OVERLAY_ROWS"
fi

step "[7/7] done"
if [[ "${SKIP_ASR:-0}" != "1" ]]; then
  step "[final] official character-weighted evaluation"
  FINAL_ARGS=(--ve-out "$VE_OUT" --decisions "$FINAL_DECISIONS")
  [[ "$STRICT_EVAL" == "1" ]] && FINAL_ARGS+=(--strict)
  "$PYTHON_BIN" "$ROOT/scripts/final_evaluate.py" "${FINAL_ARGS[@]}"
fi
echo "PIPELINE=$PIPELINE"
echo "thr: $THR_FILE"
echo "reports: $VE_OUT/reports"
echo "extracted: $VE_OUT/extracted"
echo "log: $LOG"
ls -lah "$VE_OUT/reports" || true

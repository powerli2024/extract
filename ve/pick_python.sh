#!/usr/bin/env bash
# 选定 VE 用的解释器，并可选把 Presence 依赖装进同一个 bin。
# 须先设 ROOT=ve 目录。source 本文件后导出 PYTHON_BIN。
#
# 优先级：已设 PYTHON_BIN → extract env.sh / .runtime → qwen3-asr → CONDA_PREFIX → python3
# 不新建名为 ve 的 conda 环境（AutoDL 上 Qwen3 已在 qwen3-asr）。
_ve_pick_python() {
  local c p
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    if [[ -x "$PYTHON_BIN" ]]; then
      echo "$PYTHON_BIN"
      return 0
    fi
    p="$(command -v "$PYTHON_BIN" 2>/dev/null || true)"
    if [[ -n "$p" && -x "$p" ]]; then
      echo "$p"
      return 0
    fi
  fi
  if [[ -n "${ROOT:-}" && -f "$ROOT/../.runtime/python_bin" ]]; then
    p="$(tr -d '\r\n' <"$ROOT/../.runtime/python_bin")"
    [[ -x "$p" ]] && { echo "$p"; return 0; }
  fi
  if [[ -n "${ROOT:-}" && -f "$ROOT/../env.sh" ]]; then
    # shellcheck disable=SC1091
    p="$(
      set +u
      # shellcheck disable=SC1090
      source "$ROOT/../env.sh" >/dev/null 2>&1 || true
      echo "${PYTHON_BIN:-}"
    )"
    [[ -n "$p" && -x "$p" ]] && { echo "$p"; return 0; }
  fi
  for c in \
    /root/miniconda3/envs/qwen3-asr/bin/python \
    /root/miniconda3/envs/qwen3-asr/bin/python3 \
    /root/anaconda3/envs/qwen3-asr/bin/python \
    "${CONDA_PREFIX:-}/bin/python"
  do
    [[ -n "$c" && -x "$c" ]] && { echo "$c"; return 0; }
  done
  command -v python3 || command -v python
}

PYTHON_BIN="$(_ve_pick_python)"
export PYTHON_BIN
export PATH="$(dirname "$PYTHON_BIN"):${PATH:-/usr/bin}"

ensure_modelscope() {
  local py="${1:-$PYTHON_BIN}"
  local mirror=(
    -i "${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
    --trusted-host "${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
  )
  # 仅有 modelscope 不够：speaker-verification pipeline 还要 audio extra / funasr
  local need=0
  "$py" -c "import modelscope" 2>/dev/null || need=1
  "$py" -c "from modelscope.pipelines import pipeline" 2>/dev/null || need=1
  "$py" -c "import funasr" 2>/dev/null || need=1
  if [[ "$need" == "0" ]]; then
    "$py" -c "import modelscope,sys; print('[OK] modelscope', getattr(modelscope,'__version__','?'), '←', modelscope.__file__); print('[OK] python', sys.executable)"
    return 0
  fi
  echo "[INFO] 安装 modelscope[audio] + funasr → $py"
  "$py" -m pip install -U "modelscope[audio]" funasr kaldiio addict huggingface_hub "${mirror[@]}" \
    || "$py" -m pip install -U modelscope funasr kaldiio addict huggingface_hub "${mirror[@]}" || {
    echo "[ERR] modelscope/audio 安装失败。手动:"
    echo "      $py -m pip install -U 'modelscope[audio]' funasr kaldiio"
    return 1
  }
  "$py" -c "from modelscope.pipelines import pipeline; import modelscope,sys; print('[OK] modelscope', getattr(modelscope,'__version__','?'), 'pipeline OK'); print('[OK] python', sys.executable)"
}

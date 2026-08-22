#!/usr/bin/env bash
# extract@sep：只用 VE 解释器。source 后导出 PYTHON_BIN。
# 优先级：已设 PYTHON_BIN → ve/.env_ve → conda env ve → .runtime → 旧名 qwen3-asr（仅回退）
# 不新建 ClearerVoice-Studio / qwen3-asr；clearvoice 装进同一 PYTHON_BIN。
if [[ -z "${VM_ROOT:-}" ]]; then
  VM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
export VM_CONDA_ENV="${VM_CONDA_ENV:-ve}"

_extract_ve_python_from_file() {
  local f="$1" p
  [[ -f "$f" ]] || return 1
  p="$(
    set +u
    # shellcheck disable=SC1090
    source "$f" >/dev/null 2>&1 || true
    echo "${PYTHON_BIN:-}"
  )"
  [[ -n "$p" && -x "$p" ]] || return 1
  echo "$p"
}

extract_pick_python() {
  local c p
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    if [[ -x "$PYTHON_BIN" ]]; then
      echo "$PYTHON_BIN"
      return 0
    fi
    p="$(command -v "$PYTHON_BIN" 2>/dev/null || true)"
    [[ -n "$p" && -x "$p" ]] && { echo "$p"; return 0; }
  fi
  if p="$(_extract_ve_python_from_file "$VM_ROOT/ve/.env_ve")"; then
    echo "$p"
    return 0
  fi
  if [[ -f "$VM_ROOT/.runtime/python_bin" ]]; then
    p="$(tr -d '\r\n' <"$VM_ROOT/.runtime/python_bin")"
    [[ -x "$p" ]] && { echo "$p"; return 0; }
  fi
  if p="$(_extract_ve_python_from_file "$VM_ROOT/env.sh")"; then
    echo "$p"
    return 0
  fi
  for c in \
    "/root/miniconda3/envs/${VM_CONDA_ENV}/bin/python" \
    "/root/miniconda3/envs/${VM_CONDA_ENV}/bin/python3" \
    "/root/anaconda3/envs/${VM_CONDA_ENV}/bin/python" \
    "/root/autodl-tmp/envs/${VM_CONDA_ENV}/bin/python" \
    /root/miniconda3/envs/ve/bin/python \
    /root/miniconda3/envs/ve/bin/python3
  do
    [[ -n "$c" && -x "$c" ]] && { echo "$c"; return 0; }
  done
  # 历史 VE 解释器名（仅回退，不再新建）
  for c in \
    /root/miniconda3/envs/qwen3-asr/bin/python \
    /root/anaconda3/envs/qwen3-asr/bin/python \
    "${CONDA_PREFIX:-}/bin/python"
  do
    [[ -n "$c" && -x "$c" ]] && { echo "$c"; return 0; }
  done
  command -v python3 || command -v python
}

if [[ "${EXTRACT_PICK_DEFER:-}" != "1" ]]; then
  if [[ -f "$VM_ROOT/ve/.env_ve" ]]; then
    # shellcheck disable=SC1091
    source "$VM_ROOT/ve/.env_ve" || true
  fi
  PYTHON_BIN="$(extract_pick_python)"
  export PYTHON_BIN
  if [[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]]; then
    export PATH="$(dirname "$PYTHON_BIN"):${PATH:-/usr/bin}"
  fi
  export CLEARVOICE_PYTHON="$PYTHON_BIN"
fi
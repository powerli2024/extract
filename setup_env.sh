#!/usr/bin/env bash
# Ubuntu 20.04 / Python 3.10 / PyTorch cu124 统一环境构建。
# 主流程、ClearVoice、Qwen-ASR、ONNX Runtime 和 DAE-TSE 共用同一解释器。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/paths_defaults.sh"

ENV_NAME="${VM_CONDA_ENV:-ve-cu124}"
PYTHON_VERSION="${VM_PYTHON_VERSION:-3.10}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
PIP_INDEX="${VM_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TRUSTED="${VM_PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
DAE_TSE_REPO="${DAE_TSE_REPO:-/root/DAE-TSE}"
DAE_TSE_COMMIT="${DAE_TSE_COMMIT:-9799ec6f04b3547bf891c95644e78dfd37d5d7a1}"
DAE_TSE_CONFIG="${DAE_TSE_CONFIG:-$DAE_TSE_REPO/examples/librimix/dae-tse/exp/backbone/config.yaml}"
DAE_TSE_CHECKPOINT="${DAE_TSE_CHECKPOINT:-/root/autodl-fs/midea-dae/models/dae_zh_v1/model.pt}"
DAE_TSE_CUE_HELPER="${DAE_TSE_CUE_HELPER:-}"
INSTALL_DAE="${VM_INSTALL_DAE_TSE:-0}"
REQUIRE_DAE="${VM_REQUIRE_DAE_TSE:-0}"
INSTALL_SYSTEM="${VM_INSTALL_SYSTEM_PACKAGES:-0}"
RECREATE_ENV="${VM_RECREATE_ENV:-0}"
REQUIRE_TOOLKIT="${VM_REQUIRE_CUDA_TOOLKIT:-0}"
REQ="$ROOT/environment/requirements-cu124.txt"
CONSTRAINTS="$ROOT/environment/constraints-cu124.txt"

ok() { echo "[ OK ] $*"; }
warn() { echo "[WARN] $*"; }
die() { echo "[ERR ] $*" >&2; exit 1; }

dae_issue() {
  if [[ "$REQUIRE_DAE" == "1" ]]; then
    die "$*"
  fi
  warn "$*；主环境继续，DAE-TSE 保持 NO_GO"
}

version_ge() {
  [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" == "$2" ]]
}

find_conda() {
  local candidate
  for candidate in \
    "${CONDA_BIN:-}" \
    /root/miniconda3/bin/conda \
    /root/anaconda3/bin/conda \
    "$(command -v conda 2>/dev/null || true)"
  do
    [[ -n "$candidate" && -x "$candidate" ]] && { echo "$candidate"; return 0; }
  done
  return 1
}

env_python() {
  "$CONDA_BIN" run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)' 2>/dev/null | tail -n1
}

echo "============================================================"
echo " extract-sep unified env: Ubuntu 20.04 / Python 3.10 / cu124"
echo " conda=$ENV_NAME  DAE=$INSTALL_DAE  REQUIRE_DAE=$REQUIRE_DAE  NVCC=$REQUIRE_TOOLKIT"
echo "============================================================"

[[ -r /etc/os-release ]] || die "找不到 /etc/os-release；本脚本只支持 Ubuntu 20.04"
# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "20.04" ]]; then
  die "宿主机必须为 Ubuntu 20.04，当前为 ${PRETTY_NAME:-unknown}"
fi
ok "OS=${PRETTY_NAME}"

command -v nvidia-smi >/dev/null 2>&1 || die "缺少 nvidia-smi / NVIDIA 驱动"
DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d '[:space:]')"
[[ -n "$DRIVER_VERSION" ]] || die "无法读取 NVIDIA 驱动版本"
version_ge "$DRIVER_VERSION" "525.60.13" \
  || die "CUDA 12.x wheel 运行时要求 Linux driver >=525.60.13，当前 $DRIVER_VERSION"
if version_ge "$DRIVER_VERSION" "550.54.14"; then
  ok "NVIDIA driver=$DRIVER_VERSION（达到 CUDA Toolkit 12.4 GA 推荐线）"
else
  warn "NVIDIA driver=$DRIVER_VERSION：满足 CUDA 12.x minor compatibility，但低于 Toolkit 12.4 GA 的 550.54.14"
fi
if [[ "$REQUIRE_TOOLKIT" == "1" ]]; then
  version_ge "$DRIVER_VERSION" "550.54.14" \
    || die "VM_REQUIRE_CUDA_TOOLKIT=1 要求 driver>=550.54.14"
  command -v nvcc >/dev/null 2>&1 || die "VM_REQUIRE_CUDA_TOOLKIT=1 但找不到 nvcc"
  NVCC_RELEASE="$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | tail -n1)"
  [[ "$NVCC_RELEASE" == "12.4" ]] || die "要求 nvcc 12.4，当前 ${NVCC_RELEASE:-unknown}"
  ok "nvcc=$NVCC_RELEASE"
fi

if [[ "$INSTALL_SYSTEM" == "1" ]]; then
  command -v apt-get >/dev/null 2>&1 || die "VM_INSTALL_SYSTEM_PACKAGES=1 但没有 apt-get"
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential ffmpeg git git-lfs libsndfile1 locales sox zlib1g
fi

missing_commands=()
for command_name in ffmpeg git sox; do
  command -v "$command_name" >/dev/null 2>&1 || missing_commands+=("$command_name")
done
if (( ${#missing_commands[@]} )); then
  die "缺少系统命令: ${missing_commands[*]}。以 root 运行: VM_INSTALL_SYSTEM_PACKAGES=1 bash ./setup_env.sh"
fi

CONDA_BIN="$(find_conda)" || die "未找到 conda；请先安装 Miniconda"
ok "conda=$CONDA_BIN"

if "$CONDA_BIN" env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  if [[ "$RECREATE_ENV" == "1" ]]; then
    echo "[INFO] 按 VM_RECREATE_ENV=1 删除并重建 conda 环境: $ENV_NAME"
    "$CONDA_BIN" env remove -n "$ENV_NAME" -y
    "$CONDA_BIN" create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
  else
    CURRENT_PY="$($CONDA_BIN run -n "$ENV_NAME" python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    [[ "$CURRENT_PY" == "3.10" ]] \
      || die "已有 $ENV_NAME 使用 Python $CURRENT_PY；请换环境名，或明确执行 VM_RECREATE_ENV=1 bash ./setup_env.sh"
    ok "复用 conda env=$ENV_NAME (Python $CURRENT_PY)"
  fi
else
  "$CONDA_BIN" create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

PYTHON_BIN="$(env_python)"
[[ -x "$PYTHON_BIN" ]] || die "无法解析 $ENV_NAME 的 Python: $PYTHON_BIN"
ok "PYTHON_BIN=$PYTHON_BIN"

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_ROOT_USER_ACTION=ignore
"$PYTHON_BIN" -m pip install \
  --index-url "$PIP_INDEX" --trusted-host "$PIP_TRUSTED" \
  pip==25.1.1 setuptools==80.9.0 wheel==0.45.1

echo "[INFO] 安装官方 PyTorch cu124 固定组合"
"$PYTHON_BIN" -m pip install --index-url "$TORCH_INDEX" \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

echo "[INFO] 安装统一环境其余固定依赖"
"$PYTHON_BIN" -m pip install \
  --index-url "$PIP_INDEX" --trusted-host "$PIP_TRUSTED" \
  --constraint "$CONSTRAINTS" --requirement "$REQ"

if [[ "$INSTALL_DAE" == "1" ]]; then
  DAE_SOURCE_READY=1
  if [[ ! -e "$DAE_TSE_REPO" ]]; then
    mkdir -p "$(dirname "$DAE_TSE_REPO")"
    if ! git clone https://github.com/GnafiY/DAE-TSE.git "$DAE_TSE_REPO" \
      || ! git -C "$DAE_TSE_REPO" checkout --detach "$DAE_TSE_COMMIT"; then
      dae_issue "无法获取固定 DAE-TSE 源码 $DAE_TSE_COMMIT"
      DAE_SOURCE_READY=0
    fi
  fi
  if [[ "$DAE_SOURCE_READY" == "1" && ! -d "$DAE_TSE_REPO/.git" ]]; then
    dae_issue "DAE_TSE_REPO 不是 Git 克隆: $DAE_TSE_REPO"
    DAE_SOURCE_READY=0
  fi
  if [[ "$DAE_SOURCE_READY" == "1" ]]; then
    DAE_HEAD="$(git -C "$DAE_TSE_REPO" rev-parse HEAD)"
    if [[ "$DAE_HEAD" != "$DAE_TSE_COMMIT" ]]; then
      dae_issue "DAE-TSE commit=$DAE_HEAD，期望 $DAE_TSE_COMMIT；未自动切换现有代码"
      DAE_SOURCE_READY=0
    elif [[ -n "$(git -C "$DAE_TSE_REPO" status --porcelain --untracked-files=no)" ]]; then
      dae_issue "DAE-TSE tracked files 有本地修改；请保存修改或另建固定 commit 克隆"
      DAE_SOURCE_READY=0
    fi
  fi
  if [[ "$DAE_SOURCE_READY" == "1" ]]; then
    if "$PYTHON_BIN" -m pip install --no-deps --editable "$DAE_TSE_REPO" \
      && "$PYTHON_BIN" -m pip install --no-deps --editable "$DAE_TSE_REPO/kce"; then
      ok "DAE-TSE wesep/kce 已装入同一环境"
    else
      dae_issue "DAE-TSE wesep/kce 安装失败"
    fi
  fi
fi

"$PYTHON_BIN" -m pip check

mkdir -p "$ROOT/.runtime" "$VM_OUT/meta"
printf '%s\n' "$PYTHON_BIN" >"$ROOT/.runtime/python_bin"
cat >"$ROOT/.runtime/env.sh" <<EOF
# generated by setup_env.sh
export PYTHON_BIN="$PYTHON_BIN"
export CLEARVOICE_PYTHON="$PYTHON_BIN"
export DAE_PYTHON_BIN="$PYTHON_BIN"
export VM_CONDA_ENV="$ENV_NAME"
export DAE_TSE_REPO="${DAE_TSE_REPO}"
export DAE_TSE_COMMIT="${DAE_TSE_COMMIT}"
export DAE_TSE_CONFIG="${DAE_TSE_CONFIG}"
export DAE_TSE_CHECKPOINT="${DAE_TSE_CHECKPOINT}"
export DAE_TSE_CUE_HELPER="${DAE_TSE_CUE_HELPER}"
export DATA_DIR="\${DATA_DIR:-$DATA_DIR}"
export VM_OUT="\${VM_OUT:-$VM_OUT}"
export ASR_MODEL_DIR="\${ASR_MODEL_DIR:-$ASR_MODEL_DIR}"
export MOSS_CKPT_DIR="\${MOSS_CKPT_DIR:-$MOSS_CKPT_DIR}"
export HF_HOME="\${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
export TORCH_HOME="\${TORCH_HOME:-/root/autodl-tmp/cache/torch}"
export PIP_CACHE_DIR="\${PIP_CACHE_DIR:-/root/autodl-tmp/cache/pip}"
export PATH="$(dirname "$PYTHON_BIN"):\$PATH"
EOF
cp -f "$ROOT/.runtime/env.sh" "$ROOT/env.sh"

VERIFY_ARGS=(--json-out "$VM_OUT/meta/env_cuda124_verify.json")
if [[ "$INSTALL_DAE" == "1" ]]; then
  VERIFY_ARGS+=(--dae-repo "$DAE_TSE_REPO")
  [[ -n "${DAE_TSE_CONFIG:-}" ]] && VERIFY_ARGS+=(--dae-config "$DAE_TSE_CONFIG")
  [[ -n "${DAE_TSE_CHECKPOINT:-}" ]] && VERIFY_ARGS+=(--dae-checkpoint "$DAE_TSE_CHECKPOINT")
  [[ -n "${DAE_TSE_CUE_HELPER:-}" ]] && VERIFY_ARGS+=(--dae-cue-helper "$DAE_TSE_CUE_HELPER")
  [[ "$REQUIRE_DAE" == "1" ]] && VERIFY_ARGS+=(--require-dae)
fi
set +e
"$PYTHON_BIN" "$ROOT/environment/verify_cuda124.py" "${VERIFY_ARGS[@]}"
VERIFY_RC=$?
set -e
if [[ "$VERIFY_RC" -eq 1 ]]; then
  die "统一环境硬检查失败；查看 $VM_OUT/meta/env_cuda124_verify.json"
elif [[ "$VERIFY_RC" -eq 2 ]]; then
  warn "主环境可用，但 DAE 中文资产尚未齐全；查看验证报告"
fi

"$PYTHON_BIN" -m pip freeze >"$VM_OUT/meta/requirements-lock-cu124.txt"
echo ""
ok "统一环境构建完成"
echo "  source $ROOT/env.sh"
echo "  bash $ROOT/check_env.sh"
echo "  完整锁文件: $VM_OUT/meta/requirements-lock-cu124.txt"
echo "  验证报告:   $VM_OUT/meta/env_cuda124_verify.json"

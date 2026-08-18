#!/usr/bin/env bash
# wesep（WeSep）安装 + 预训练 TSE 模型（bsrnn_ecapa_vox1，ModelScope）
# 仓库: https://github.com/wenet-e2e/wesep.git
# 供: PIPELINE=wesep ./run_all.sh
#
# 依赖原则（AutoDL / 与 torch·modelscope 共存）:
#   - numpy 钉 1.26.x（禁止被 scipy 拉到 2.x，否则 sklearn/hdbscan ABI 炸）
#   - scipy 1.11–1.13（有 coo_array，且支持 numpy1）
#   - transformers/peft 对齐（wespeaker 导入链会碰 peft）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$ROOT/.env_ve" ]] && source "$ROOT/.env_ve" || true
export VE_MODEL_DIR="${VE_MODEL_DIR:-/root/autodl-tmp/ve_models}"
export WESEP_ROOT="${WESEP_ROOT:-$VE_MODEL_DIR/wesep}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
mkdir -p "$VE_MODEL_DIR"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true
PIP=( -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn )

# 1) 克隆 wesep 仓库
if [[ ! -d "$WESEP_ROOT/wesep" ]]; then
  echo "[1/3] 克隆 wenet-e2e/wesep → $WESEP_ROOT ..."
  git clone --depth 1 https://github.com/wenet-e2e/wesep.git "$WESEP_ROOT"
else
  echo "[1/3] wesep 仓库已存在: $WESEP_ROOT"
fi

# 2) 安装依赖
echo "[2/3] 安装依赖（钉 numpy<2，避免 ABI 炸）..."
"$PYTHON_BIN" -m pip install -U \
  "setuptools>=65,<81" wheel packaging \
  "numpy>=1.26.4,<2" \
  "scipy>=1.11.4,<1.14" \
  "scikit-learn>=1.3,<1.7" \
  "transformers>=4.46.0,<5.0" "peft>=0.13.2,<0.18" \
  pyyaml soundfile librosa tqdm silero-vad \
  "${PIP[@]}"

# 按当前 numpy 重编/重装二进制扩展（修 dtype size changed）
# 注意：不可直接 force-reinstall sklearn（会再次拉 numpy2）；先钉 numpy/scipy，再 --no-deps
echo "[INFO] 钉 numpy1 + scipy1.13，再无依赖重装 sklearn/hdbscan ..."
"$PYTHON_BIN" -m pip install --force-reinstall --no-cache-dir \
  "numpy==1.26.4" "scipy==1.13.1" \
  "${PIP[@]}"
"$PYTHON_BIN" -m pip install --force-reinstall --no-cache-dir --no-deps \
  "scikit-learn==1.5.2" "hdbscan==0.8.40" \
  "${PIP[@]}"
"$PYTHON_BIN" -m pip install -U joblib threadpoolctl "${PIP[@]}"

if ! "$PYTHON_BIN" -c "from wespeaker.models.speaker_model import get_speaker_model" 2>/dev/null; then
  echo "[INFO] 安装/修复 wespeaker ..."
  "$PYTHON_BIN" -m pip install -U "wespeaker" "${PIP[@]}" || true
fi

if [[ -f "$WESEP_ROOT/requirements.txt" ]]; then
  grep -v -i -E '^torch|^torchaudio|^transformers|^peft|^numpy|^scipy|^#' \
    "$WESEP_ROOT/requirements.txt" > /tmp/wesep_req.txt || true
  if [[ -s /tmp/wesep_req.txt ]]; then
    "$PYTHON_BIN" -m pip install -r /tmp/wesep_req.txt "${PIP[@]}" || echo "[WARN] requirements 部分安装失败"
  fi
fi
# 再次钉回 numpy1（防止上面 requirements 又拉到 numpy2）
"$PYTHON_BIN" -m pip install "numpy>=1.26.4,<2" "scipy>=1.11.4,<1.14" "${PIP[@]}"

"$PYTHON_BIN" -m pip install -e "$WESEP_ROOT" "${PIP[@]}" || echo "[WARN] pip install -e wesep 失败，将用 PYTHONPATH"

echo "[INFO] 自检 import 链 ..."
export PYTHONPATH="$WESEP_ROOT:${PYTHONPATH:-}"
"$PYTHON_BIN" - <<'PY' || {
  echo "[ERR] wesep import 失败。一键修复 ABI："
  echo "  pip install 'numpy>=1.26.4,<2' 'scipy>=1.11.4,<1.14'"
  echo "  pip install --force-reinstall --no-cache-dir scikit-learn hdbscan"
  exit 1
}
import numpy as np
import scipy
print(f"[OK] numpy={np.__version__} scipy={scipy.__version__}")
assert np.__version__.startswith("1."), f"需要 numpy1.x，当前 {np.__version__}"
from scipy.sparse import coo_array  # noqa: F401
import pkg_resources  # noqa: F401
from wespeaker.models.speaker_model import get_speaker_model  # noqa: F401
from wesep.cli.extractor import load_model  # noqa: F401
print("[OK] wesep / wespeaker import 通过")
PY

# 3) 预下载预训练模型
echo "[3/3] 下载预训练模型 bsrnn_ecapa_vox1 ..."
"$PYTHON_BIN" - <<'PY'
from wesep.cli.extractor import load_model
m = load_model("english")
m.set_device("cpu")
print("[OK] wesep 预训练模型 bsrnn_ecapa_vox1 就绪（~/.wesep/english/）")
PY

echo ""
echo "=== 环境摘要 ==="
echo "WESEP_ROOT=$WESEP_ROOT"
"$PYTHON_BIN" -c "import numpy,scipy; print('numpy',numpy.__version__,'scipy',scipy.__version__)"
echo "预训练模型: bsrnn_ecapa_vox1 → ~/.wesep/english/"
echo ""
echo "下一步: PIPELINE=wesep ./run_all.sh"

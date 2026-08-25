#!/usr/bin/env bash
# 安装官方 ClearVoice 代码，并把官方 ModelScope 权重放到其约定的位置。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CLEARVOICE_ROOT="${CLEARVOICE_ROOT:-/root/autodl-tmp/ClearerVoice-Studio}"
MODEL_DIR="$CLEARVOICE_ROOT/clearvoice/checkpoints/MossFormer2_SE_48K"
MODEL_ID="${MOSS_SE48K_MODEL_ID:-alibabasglab/MossFormer2_SE_48K}"

if [[ ! -d "$CLEARVOICE_ROOT/.git" ]]; then
  git clone --depth 1 https://github.com/modelscope/ClearerVoice-Studio.git "$CLEARVOICE_ROOT"
fi

"$PYTHON_BIN" -m pip install -e "$CLEARVOICE_ROOT/clearvoice"
export MOSS_SE48K_MODEL_DIR="$MODEL_DIR"
export MOSS_SE48K_MODEL_ID="$MODEL_ID"
"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
from modelscope import snapshot_download

model_id = os.environ["MOSS_SE48K_MODEL_ID"]
target = Path(os.environ["MOSS_SE48K_MODEL_DIR"])
target.parent.mkdir(parents=True, exist_ok=True)
snapshot_download(model_id, local_dir=str(target))
need = target / "last_best_checkpoint"
if not need.is_file():
    raise SystemExit(f"[ERR] 下载后缺少 {need}")
print(f"[OK] MossFormer2_SE_48K -> {target}")
PY

echo "[OK] CLEARVOICE_ROOT=$CLEARVOICE_ROOT"
echo "[NEXT] cd $ROOT && CLEARVOICE_ROOT=$CLEARVOICE_ROOT ./smoke_moss_se48k.sh <one_16k_wav>"

#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then echo "usage: $0 <one_16k_wav>" >&2; exit 64; fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CLEARVOICE_ROOT="${CLEARVOICE_ROOT:-/root/autodl-tmp/ClearerVoice-Studio}"
PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}" CLEARVOICE_ROOT="$CLEARVOICE_ROOT" "$PYTHON_BIN" -X faulthandler - "$1" <<'PY'
import os, sys
from pathlib import Path
from audio_io import load_audio, resample_wav, save_audio
from moss_se48k import MossFormer2SE48K

inp = Path(sys.argv[1]).resolve()
w, sr = load_audio(inp)
print(f"[INFO] input={inp} sr={sr} samples={len(w)}")
model = MossFormer2SE48K(clearvoice_root=os.environ["CLEARVOICE_ROOT"])
y48 = model.enhance_48k(resample_wav(w, 16000, 48000, method="poly"))
y16 = resample_wav(y48, 48000, 16000, method="poly")
out = inp.with_name(inp.stem + "__se48k_smoke.wav")
save_audio(out, y16, 16000)
print(f"[OK] output={out} sr=16000 samples={len(y16)}")
PY

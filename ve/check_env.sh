#!/usr/bin/env bash
# 环境自检（不安装）— 支持 PIPELINE=ps4|wesep|sep_route
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
[[ -f "$ROOT/.env_ve" ]] && source "$ROOT/.env_ve" || true
# shellcheck disable=SC1091
source "$ROOT/pick_python.sh"

export VE_ROOT="${VE_ROOT:-$ROOT}"
export VE_OUT="${VE_OUT:-/root/autodl-tmp/ve}"
export VE_MODEL_DIR="${VE_MODEL_DIR:-/root/autodl-tmp/ve_models}"
export DATA_DIR="${DATA_DIR:-/root/autodl-tmp/datasetA}"
export BEST_SEP_DIR="${BEST_SEP_DIR:-/root/autodl-tmp/pos_neg/best_sep}"
export PS4_WEIGHTS="${PS4_WEIGHTS:-$VE_MODEL_DIR/PS4/checkpoint_epoch037.pt}"
export MOSS_ONNX_PATH="${MOSS_ONNX_PATH:-/root/autodl-tmp/checkpoints/MossFormer2_ONNX/simple_model.onnx}"
export WESEP_ROOT="${WESEP_ROOT:-$VE_MODEL_DIR/wesep}"
export PYTHONPATH="$ROOT/scripts:${ROOT}/../scripts:${ROOT}/../VD/tools:${ROOT}/../VM/scripts:${WESEP_ROOT}:${PYTHONPATH:-}"

PIPELINE_RAW="${PIPELINE:-mix}"
PIPELINE_LC="$(echo "$PIPELINE_RAW" | tr '[:upper:]' '[:lower:]')"

echo "=== VE check_env (PIPELINE=$PIPELINE_LC) ==="
echo "Python: $PYTHON_BIN"
"$PYTHON_BIN" -V
"$PYTHON_BIN" -c "import sys; print('executable', sys.executable)"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || echo "no nvidia-smi"

if grep -q $'\r' "$ROOT/setup_env.sh" 2>/dev/null; then
  echo "[FAIL] setup_env.sh 含 CR（CRLF）；Linux 上请: sed -i 's/\r$//' $ROOT/*.sh"
  exit 2
fi
echo "[ OK ] shell scripts LF (no CR in setup_env.sh)"

export PIPELINE_LC
"$PYTHON_BIN" - <<'PY'
import os, sys
from pathlib import Path

ok = True
pipeline = os.environ.get("PIPELINE_LC", "all")

def check(name, cond, hint=""):
    global ok
    if cond:
        print(f"[ OK ] {name}")
    else:
        print(f"[FAIL] {name} {hint}")
        ok = False

def warn(name, cond, hint=""):
    if cond:
        print(f"[ OK ] {name}")
    else:
        print(f"[WARN] {name} {hint}")

try:
    import torch
    check("torch", True)
    print(f"       torch={torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"       gpu={torch.cuda.get_device_name(0)} mem={torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
except Exception as e:
    check("torch", False, str(e))

for mod in ("numpy", "soundfile", "yaml", "librosa", "torchaudio"):
    try:
        __import__(mod)
        check(mod, True)
    except Exception as e:
        check(mod, False, str(e))

try:
    import modelscope  # noqa: F401
    check("modelscope", True)
    print(f"       modelscope={getattr(modelscope, '__version__', '?')} ← {modelscope.__file__}")
except Exception as e:
    check("modelscope", False, f"{e} → {sys.executable} -m pip install -U modelscope 或 ./setup_env.sh")

ve_root = Path(os.environ.get("VE_ROOT", ".")).resolve()
sys.path.insert(0, str(ve_root / "scripts"))
for extra in (
    ve_root.parent / "scripts",
    Path("/root/extract/scripts"),
    Path(os.environ.get("WESEP_ROOT", "")),
):
    if extra and Path(extra).is_dir() and str(extra) not in sys.path:
        sys.path.append(str(extra))

ve_out = Path(os.environ.get("VE_OUT", "/root/autodl-tmp/ve"))
model = Path(os.environ.get("VE_MODEL_DIR", "/root/autodl-tmp/ve_models"))
data = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/datasetA"))
best = Path(os.environ.get("BEST_SEP_DIR", "/root/autodl-tmp/pos_neg/best_sep"))
ps4_dir = model / "PS4"
ps4 = Path(os.environ.get("PS4_WEIGHTS", str(ps4_dir / "checkpoint_epoch037.pt")))
inf = ps4_dir / "inference.py"
eres = model / "eres2netv2_zh"
chs = Path(os.environ.get("SPK_CHS_DIR", str(model / "cnceleb_resnet34_LM")))
moss = Path(os.environ.get("MOSS_ONNX_PATH", "/root/autodl-tmp/checkpoints/MossFormer2_ONNX/simple_model.onnx"))
wesep_root = Path(os.environ.get("WESEP_ROOT", str(model / "wesep")))

need_ps4 = pipeline in ("ps4", "ps4_bsrnn")
need_wesep = pipeline in ("wesep", "wesep_bsrnn")
need_sep = pipeline in ("sep_route", "mossformer", "route") or os.environ.get(
    "USE_SEP", "1"
).strip() in ("1", "true", "yes")
need_tasnet = pipeline in ("cond_tasnet", "condtasnet", "tasnet", "cond-tasnet")

check(
    f"DATA_DIR {data}",
    data.is_dir() and (data / "pos").is_dir(),
    "同步 datasetA → /root/autodl-tmp/datasetA",
)
check(
    f"BEST_SEP {best}",
    best.is_dir() and (best / "index.jsonl").is_file(),
    "同步 pos_neg/best_sep → /root/autodl-tmp/pos_neg/best_sep",
)

if need_ps4:
    check(f"PS4 checkpoint {ps4}", ps4.is_file(), "运行 ./download_models.sh")
    check(f"PS4 inference.py {inf}", inf.is_file(), "HF 包须含 inference.py")
    if ps4.is_file() and inf.is_file():
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("ps4_hf_inference", inf)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            check("PS4 HF inference.py import", hasattr(mod, "build_model") and hasattr(mod, "extract_speaker"))
        except Exception as e:
            check("PS4 HF inference.py import", False, str(e))

if os.environ.get("PRESENCE_BACKEND", "eres2netv2").lower() in (
    "resnet34", "resnet34_lm", "wespeaker",
):
    check(
        f"cnceleb {chs}",
        chs.is_dir() and (chs / "config.yaml").is_file(),
        "ONLY=resnet34_lm ./download_presence_encoders.sh",
    )
eres_ok = eres.is_dir() and (
    (eres / "MODELSCOPE_PATH.txt").is_file()
    or (eres / "configuration.json").is_file()
    or any(eres.rglob("*.pt"))
    or any(eres.rglob("*.bin"))
)
check(
    f"ERes2Net dir {eres}",
    eres_ok,
    "ONLY=eres2netv2 ./download_presence_encoders.sh（须同一 PYTHON_BIN 已装 modelscope）",
)

if need_wesep:
    warn(f"WESEP_ROOT {wesep_root}", (wesep_root / "wesep").is_dir(), "运行 ./download_wesep.sh")
    try:
        from wesep.cli.extractor import load_model  # noqa: F401
        check("wesep.cli.extractor import", True)
    except Exception as e:
        check("wesep.cli.extractor import", False, f"{e} → ./download_wesep.sh")

if need_sep:
    check(f"MossFormer ONNX {moss}", moss.is_file(), "运行 ./download_moss_onnx.sh")
    try:
        import onnxruntime as ort  # noqa: F401
        check("onnxruntime", True)
        print(f"       providers={ort.get_available_providers()}")
    except Exception as e:
        check("onnxruntime", False, f"{e} → pip install onnxruntime-gpu")
    try:
        from mossformer2_onnx import MossFormer2Separator  # noqa: F401
        check("mossformer2_onnx import", True)
    except Exception as e:
        check(
            "mossformer2_onnx import",
            False,
            f"{e} → 确认 extract/scripts/mossformer2_onnx.py 或 ./download_moss_onnx.sh",
        )

if need_tasnet:
    from paths import default_cond_tasnet_ckpt, default_ecapa_dir
    ck = Path(os.environ.get("COND_TASNET_CKPT") or default_cond_tasnet_ckpt())
    check(f"Cond-TasNet ckpt {ck}", ck.is_file(), "放到 $VE_MODEL_DIR/cond_tasnet/best.pt 或 COND_TASNET_CKPT")
    try:
        import speechbrain  # noqa: F401
        check("speechbrain", True)
    except Exception as e:
        check("speechbrain", False, f"{e} → pip install speechbrain")

# 工厂可解析
try:
    from tse_factory import create_tse
    check("tse_factory import", True)
except Exception as e:
    check("tse_factory import", False, str(e))

if Path("/root/autodl-tmp").is_dir():
    for label, p in (
        ("DATA_DIR", data),
        ("BEST_SEP", best),
        ("VE_MODEL_DIR", model),
        ("VE_OUT", ve_out),
    ):
        try:
            resolved = p.resolve()
            on_tmp = str(resolved).startswith("/root/autodl-tmp")
            if on_tmp:
                print(f"[ OK ] {label} on autodl-tmp")
            else:
                print(f"[WARN] {label} 不在 autodl-tmp: {resolved}")
        except Exception as e:
            print(f"[WARN] {label} resolve: {e}")

print("VE_OUT=", ve_out)
print("VE_MODEL_DIR=", model)
print("PIPELINE=", pipeline)
print("用法: PIPELINE=mix ./check_env.sh")
sys.exit(0 if ok else 2)
PY

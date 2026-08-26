#!/usr/bin/env bash
# 下载 Presence 声纹编码器到 /root/autodl-tmp/ve_models
#   eres2netv2  — ModelScope 中文短时
#   campplus    — ModelScope 中文轻量
#   resnet34_lm — HF CNCeleb ResNet34-LM
#   vblink2     — WeSpeaker VoxBlink2 SimAM-ResNet34（多语）
#   ecapa_tdnn  — WeSpeaker VoxCeleb ECAPA-TDNN 1024 LM
#   vblink100   — WeSpeaker VoxBlink2 SimAM-ResNet100（多语，重点对照）
#
# 用法:
#   ./download_presence_encoders.sh
#   ONLY=eres2netv2,vblink2 ./download_presence_encoders.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$ROOT/.env_ve" ]] && source "$ROOT/.env_ve" || true
# shellcheck disable=SC1091
source "$ROOT/pick_python.sh"

MODEL_DIR="${VE_MODEL_DIR:-/root/autodl-tmp/ve_models}"
ONLY="${ONLY:-eres2netv2}"

if [[ -f /etc/network_turbo ]]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo || true
fi
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/root/autodl-tmp/cache/modelscope}"
mkdir -p "$MODEL_DIR" "$HF_HOME" "$MODELSCOPE_CACHE"

want() {
  local name="$1"
  [[ ",$ONLY," == *",$name,"* ]]
}

echo "============================================"
echo " download_presence_encoders"
echo " MODEL_DIR=$MODEL_DIR"
echo " PYTHON_BIN=$PYTHON_BIN"
echo " ONLY=$ONLY"
echo "============================================"

if ! ensure_modelscope "$PYTHON_BIN"; then
  echo "[ERR] 下载 ERes/CAM++ 需要 modelscope"
  exit 1
fi

"$PYTHON_BIN" -c "import huggingface_hub" 2>/dev/null || \
  "$PYTHON_BIN" -m pip install -U huggingface_hub \
    -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

# ---------- eres2netv2 / campplus：ModelScope ----------
download_ms() {
  local key="$1" model_id="$2" dst="$3"
  mkdir -p "$dst"
  if [[ -f "$dst/MODELSCOPE_PATH.txt" ]] || [[ -f "$dst/configuration.json" ]] || [[ -f "$dst/config.yaml" ]]; then
    echo "[ OK ] $key 已存在 → $dst"
    return 0
  fi
  echo "[INFO] ModelScope 拉取 $model_id → $dst"
  "$PYTHON_BIN" - <<PY
from pathlib import Path
import os
dst = Path(r"$dst")
dst.mkdir(parents=True, exist_ok=True)
model_id = "$model_id"
try:
    from modelscope.hub.snapshot_download import snapshot_download
except ImportError:
    raise SystemExit("请: pip install -U modelscope")
cache = snapshot_download(model_id, cache_dir=os.environ.get("MODELSCOPE_CACHE"))
tip = dst / "MODELSCOPE_PATH.txt"
tip.write_text(str(cache).rstrip() + "\n", encoding="utf-8")
# 也写一份软链提示
print(f"[OK] {model_id} cache={cache}")
print(f"[OK] tip → {tip}")
PY
}

# ---------- resnet34_lm：HuggingFace ----------
download_resnet34() {
  local dst="$MODEL_DIR/cnceleb_resnet34_LM"
  mkdir -p "$dst"
  if ls "$dst"/*.onnx >/dev/null 2>&1 || [[ -f "$dst/avg_model.pt" ]]; then
    echo "[ OK ] resnet34_lm 已存在 → $dst"
    return 0
  fi
  echo "[INFO] HF 拉取 Wespeaker/wespeaker-cnceleb-resnet34-LM → $dst"
  "$PYTHON_BIN" - <<PY
from huggingface_hub import snapshot_download
from pathlib import Path
dst = Path(r"$dst")
snapshot_download(
    repo_id="Wespeaker/wespeaker-cnceleb-resnet34-LM",
    local_dir=str(dst),
    local_dir_use_symlinks=False,
)
print("[OK] resnet34_lm →", dst)
PY
}

# ---------- vblink2 ----------
# 失败原因说明：HF 上不存在 Wespeaker/wespeaker-voxblink2-samresnet34（会 401）。
# 官方包在 ModelScope 数据集 wenet/wespeaker_pretrained_models（与 wespeaker Hub 一致）。
download_vblink2() {
  local dst="$MODEL_DIR/vblink2_samresnet34"
  mkdir -p "$dst"
  if [[ -f "$dst/avg_model.pt" && -f "$dst/config.yaml" ]]; then
    echo "[ OK ] vblink2 已存在 → $dst"
    return 0
  fi

  echo "[INFO] 下载 VoxBlink2 SimAM-ResNet34 …"
  echo "[INFO] 正确源: ModelScope dataset / wenet.org.cn / HF:gaunernst/...（勿用 Wespeaker/wespeaker-voxblink2-*）"

  VB_DST="$dst" VB_ZIP="$MODEL_DIR/_tmp_voxblink2_samresnet34.zip" "$PYTHON_BIN" - <<'PY'
import json, os, shutil, sys, urllib.request, zipfile
from pathlib import Path

dst = Path(os.environ["VB_DST"])
zip_path = Path(os.environ["VB_ZIP"])
dst.mkdir(parents=True, exist_ok=True)

# 清残缺
if not ((dst / "avg_model.pt").is_file() and (dst / "config.yaml").is_file()):
    for p in list(dst.iterdir()):
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
zip_path.unlink(missing_ok=True)

def is_good_zip(p: Path) -> bool:
    if not p.is_file() or p.stat().st_size < 1_000_000:
        return False
    return p.read_bytes()[:2] == b"PK"

def normalize_dir(root: Path) -> bool:
    """确保 root 下有 avg_model.pt + config.yaml。"""
    if (root / "avg_model.pt").is_file() and (root / "config.yaml").is_file():
        return True
    pts = list(root.rglob("avg_model.pt")) or list(root.rglob("*.pt"))
    cfgs = list(root.rglob("config.yaml"))
    if pts:
        shutil.copy2(pts[0], root / "avg_model.pt")
    if cfgs:
        shutil.copy2(cfgs[0], root / "config.yaml")
    return (root / "avg_model.pt").is_file() and (root / "config.yaml").is_file()

def try_modelscope_dataset() -> bool:
    api = "https://modelscope.cn/api/v1/datasets/wenet/wespeaker_pretrained_models/oss/tree"
    key = "voxblink2_samresnet34.zip"
    print("[INFO] ModelScope dataset API …", flush=True)
    try:
        with urllib.request.urlopen(api, timeout=90) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"[WARN] API 失败: {e}", flush=True)
        return False
    info = next((d for d in (payload.get("Data") or []) if d.get("Key") == key), None)
    if not info or not info.get("Url"):
        print(f"[WARN] 未找到 Key={key}", flush=True)
        return False
    url = info["Url"]
    print(f"[INFO] 下载 OSS …", flush=True)
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as e:
        print(f"[WARN] OSS 下载失败: {e}", flush=True)
        return False
    return is_good_zip(zip_path)

def try_wenet_org() -> bool:
    url = "https://wenet.org.cn/downloads?models=wespeaker&version=voxblink2_samresnet34.zip"
    print(f"[INFO] wenet.org.cn …", flush=True)
    zip_path.unlink(missing_ok=True)
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as e:
        print(f"[WARN] wenet.org.cn 失败: {e}", flush=True)
        return False
    return is_good_zip(zip_path)

def try_hf_mirror() -> bool:
    print("[INFO] HF 社区镜像 gaunernst/wespeaker-voxblink2-samresnet34 …", flush=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[WARN] 无 huggingface_hub", flush=True)
        return False
    try:
        snapshot_download(
            repo_id="gaunernst/wespeaker-voxblink2-samresnet34",
            local_dir=str(dst),
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        print(f"[WARN] HF 镜像失败: {e}", flush=True)
        return False
    return normalize_dir(dst)

def extract_zip() -> bool:
    if not is_good_zip(zip_path):
        return False
    print(f"[INFO] 解压 {zip_path} → {dst}", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst)
    zip_path.unlink(missing_ok=True)
    return normalize_dir(dst)

ok = False
if try_modelscope_dataset() and extract_zip():
    ok = True
elif try_wenet_org() and extract_zip():
    ok = True
elif try_hf_mirror():
    ok = True

if ok and normalize_dir(dst):
    print(f"[ OK ] vblink2 → {dst}", flush=True)
    sys.exit(0)

print("[FAIL] vblink2 所有源失败。手动:", flush=True)
print("  python -c \"from wespeaker.cli.hub import Hub; print(Hub.get_model('vblinkp'))\"", flush=True)
print("  # 然后将 ~/.wespeaker/vblinkp/{avg_model.pt,config.yaml} 拷到", dst, flush=True)
sys.exit(1)
PY
}

# ---------- WeSpeaker 官方 zip：ECAPA / SimAM-ResNet100 ----------
download_wespeaker_zip() {
  local key="$1" version="$2" dst="$3"
  mkdir -p "$dst"
  if [[ -f "$dst/avg_model.pt" && -f "$dst/config.yaml" ]]; then
    echo "[ OK ] $key 已存在 → $dst"
    return 0
  fi
  echo "[INFO] 下载 $key: $version → $dst"
  WS_DST="$dst" WS_VERSION="$version" WS_ZIP="$MODEL_DIR/_tmp_${key}.zip" \
    "$PYTHON_BIN" - <<'PY'
import os, shutil, urllib.request, zipfile
from pathlib import Path

dst = Path(os.environ["WS_DST"])
version = os.environ["WS_VERSION"]
zip_path = Path(os.environ["WS_ZIP"])
url = f"https://wenet.org.cn/downloads?models=wespeaker&version={version}"
dst.mkdir(parents=True, exist_ok=True)
zip_path.unlink(missing_ok=True)
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=180) as src, zip_path.open("wb") as out:
    shutil.copyfileobj(src, out)
if zip_path.stat().st_size < 1_000_000 or zip_path.read_bytes()[:2] != b"PK":
    raise SystemExit(f"下载不是有效 zip: {zip_path} size={zip_path.stat().st_size}")
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(dst)
zip_path.unlink(missing_ok=True)
pts = list(dst.rglob("avg_model.pt")) or list(dst.rglob("*.pt"))
cfgs = list(dst.rglob("config.yaml"))
if not (dst / "avg_model.pt").is_file() and pts:
    shutil.copy2(pts[0], dst / "avg_model.pt")
if not (dst / "config.yaml").is_file() and cfgs:
    shutil.copy2(cfgs[0], dst / "config.yaml")
if not ((dst / "avg_model.pt").is_file() and (dst / "config.yaml").is_file()):
    raise SystemExit(f"模型包缺 avg_model.pt/config.yaml: {dst}")
print(f"[ OK ] {version} → {dst}")
PY
}

# ---------- run ----------
fail=0
if want eres2netv2; then
  download_ms eres2netv2 "iic/speech_eres2netv2_sv_zh-cn_16k-common" \
    "$MODEL_DIR/eres2netv2_zh" || fail=1
fi
if want campplus; then
  download_ms campplus "iic/speech_campplus_sv_zh-cn_16k-common" \
    "$MODEL_DIR/campplus_zh" || fail=1
fi
if want resnet34_lm; then
  download_resnet34 || fail=1
fi
if want vblink2; then
  download_vblink2 || fail=1
fi
if want ecapa_tdnn; then
  download_wespeaker_zip ecapa_tdnn "voxceleb_ECAPA1024_LM.zip" \
    "$MODEL_DIR/voxceleb_ecapa1024_LM" || fail=1
fi
if want vblink100; then
  download_wespeaker_zip vblink100 "voxblink2_samresnet100.zip" \
    "$MODEL_DIR/vblink2_samresnet100" || fail=1
fi

echo
echo "======= 路径（给 score_encoders_on_sep.sh）======="
echo "export VE_MODEL_DIR=$MODEL_DIR"
echo "export ERES_DIR=$MODEL_DIR/eres2netv2_zh"
echo "export CAMPPLUS_DIR=$MODEL_DIR/campplus_zh"
echo "export SPK_CHS_DIR=$MODEL_DIR/cnceleb_resnet34_LM"
echo "export VBLINK_DIR=$MODEL_DIR/vblink2_samresnet34"
echo "export ECAPA_PRESENCE_DIR=$MODEL_DIR/voxceleb_ecapa1024_LM"
echo "export VBLINK100_DIR=$MODEL_DIR/vblink2_samresnet100"
echo
echo "若只要补 vblink2:  ONLY=vblink2 bash ./download_presence_encoders.sh"
echo "ENCODERS=eres2netv2,campplus,ecapa_tdnn,vblink2_samresnet100 bash ./score_encoders_on_sep.sh"
echo

if [[ "$fail" == "1" ]]; then
  echo "[WARN] 部分模型未就绪，见上方 FAIL"
  exit 2
fi
echo "[OK] download_presence_encoders 完成"
exit 0

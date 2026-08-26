#!/usr/bin/env python3
"""VE 路径：默认数据与模型落在 /root/autodl-tmp（AutoDL 数据盘）。"""

from __future__ import annotations

import os
from pathlib import Path

VALID_SPLITS = ("pos", "neg")


def ve_root() -> Path:
    return Path(__file__).resolve().parents[1]


def media_root() -> Path:
    """数据旁路根：ve 的上一级（extract 仓）或再上一级（旧 media/VE）。"""
    return ve_root().parent


def extract_root() -> Path:
    """extract.git 根（ve 的父目录）。VE 嵌在 extract/ve 时 VM scripts 在此。"""
    p = ve_root()
    if (p.parent / "scripts" / "mossformer2_onnx.py").is_file():
        return p.parent
    if (p.parent / "vm" / "scripts" / "mossformer2_onnx.py").is_file():
        return p.parent
    return p.parent


def vm_scripts_dirs() -> list[Path]:
    """MossFormer ONNX 所在 scripts：优先同仓 extract/scripts。"""
    er = extract_root()
    return [
        er / "scripts",
        er / "vm" / "scripts",
        media_root() / "VM" / "scripts",
        Path("/root/extract/scripts"),
        Path("/root/VM/scripts"),
        Path("/root/autodl-tmp/VM/scripts"),
        ve_root().parent / "VM" / "scripts",
    ]


def has_autodl_tmp() -> bool:
    return Path("/root/autodl-tmp").is_dir()


def default_ve_out() -> Path:
    env = os.environ.get("VE_OUT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    if has_autodl_tmp():
        return Path("/root/autodl-tmp/ve").resolve()
    return (media_root() / "ve_out").resolve()


def default_data_dir() -> Path:
    env = os.environ.get("DATA_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    for c in (
        Path("/root/autodl-tmp/datasetA"),
        Path("/root/datasetA"),
        media_root() / "datasetA",
    ):
        if c.is_dir():
            return c.resolve()
    return (media_root() / "datasetA").resolve()


def looks_like_best_sep(p: Path) -> bool:
    """干净 KWS enroll 目录：有 index.jsonl，或 pos/neg 下有 wav。"""
    try:
        if not p.is_dir():
            return False
    except OSError:
        return False
    if (p / "index.jsonl").is_file():
        return True
    for sub in ("pos", "neg"):
        d = p / sub
        try:
            if d.is_dir() and any(d.glob("*.wav")):
                return True
        except OSError:
            continue
    return False


def best_sep_candidates() -> list[Path]:
    """BEST_SEP_DIR 优先，其余为常见 sep 产物位置（含 kws_sep 与 pos_neg）。"""
    raw: list[Path] = []
    env = os.environ.get("BEST_SEP_DIR", "").strip()
    if env:
        raw.append(Path(env).expanduser())
    raw.extend(
        [
            Path("/root/autodl-tmp/pos_neg/best_sep"),
            Path("/root/autodl-tmp/kws_sep/best_sep"),
            Path("/root/autodl-tmp/best_sep"),
            media_root() / "pos_neg" / "best_sep",
            media_root() / "kws_sep" / "best_sep",
            Path("D:/media/pos_neg/best_sep"),
            Path("D:/media/kws_sep/best_sep"),
        ]
    )
    out: list[Path] = []
    seen: set[str] = set()
    for c in raw:
        try:
            r = c.resolve()
        except OSError:
            continue
        key = str(r).replace("\\", "/").lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def default_best_sep() -> Path:
    """干净 KWS enroll。显式 BEST_SEP_DIR 优先；否则选第一个像 best_sep 的目录。"""
    env = os.environ.get("BEST_SEP_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    for c in best_sep_candidates():
        if looks_like_best_sep(c):
            return c
    return (media_root() / "pos_neg" / "best_sep").resolve()


def require_best_sep(p: Path | None = None) -> Path:
    """校验 enroll 目录；失败时列出可切换的候选。"""
    chosen = (p or default_best_sep()).resolve()
    found = [c for c in best_sep_candidates() if looks_like_best_sep(c)]
    if looks_like_best_sep(chosen):
        others = [c for c in found if c != chosen]
        if others:
            print(
                "[INFO] 其它干净 KWS 候选: "
                + ", ".join(str(x) for x in others)
                + "  → export BEST_SEP_DIR=... 或 --best-sep 切换",
                flush=True,
            )
        return chosen
    lines = "\n".join(f"  {c}" for c in found) if found else "  （未发现现成目录）"
    raise SystemExit(
        f"找不到干净 KWS enroll 目录: {chosen}\n"
        "main 只读 sep 产物，不跑 KWS BSS。请自行指定：\n"
        "  export BEST_SEP_DIR=/path/to/best_sep\n"
        "  python scripts/build_manifest.py --best-sep /path/to/best_sep\n"
        "目录须含 index.jsonl 或 pos/*.wav。已发现:\n"
        f"{lines}"
    )


def default_cohort_dir() -> Path:
    """干净路人（enroll Z-Norm）目录；环境变量 COHORT_DIR。"""
    env = os.environ.get("COHORT_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    for c in (
        Path("/root/autodl-tmp/clean_kws"),
        Path("/root/autodl-tmp/cohort/clean_kws"),
        media_root() / "clean_kws",
        media_root() / "_clean_kws_inspect" / "clean_kws",
    ):
        if c.is_dir() and any(c.rglob("*.wav")):
            return c.resolve()
    return Path("/root/autodl-tmp/clean_kws").resolve()


def default_test_cohort_dir() -> Path:
    """CMD 域路人（test Z-Norm / AS-Norm）；环境变量 TEST_COHORT_DIR。"""
    env = os.environ.get("TEST_COHORT_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    for c in (
        Path("/root/autodl-tmp/mix500"),
        Path("/root/autodl-tmp/cohort/mix500"),
        media_root() / "mix500",
        media_root() / "_mix500_inspect",
        media_root() / "datasetA" / "mix500",
    ):
        if c.is_dir() and any(c.rglob("*.wav")):
            return c.resolve()
    return Path("/root/autodl-tmp/mix500").resolve()


def default_model_dir() -> Path:
    env = os.environ.get("VE_MODEL_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    if has_autodl_tmp():
        return Path("/root/autodl-tmp/ve_models").resolve()
    return (media_root() / "ve_models").resolve()


def default_ps4_weights() -> Path:
    env = os.environ.get("PS4_WEIGHTS", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    for c in (
        default_model_dir() / "PS4" / "checkpoint_epoch037.pt",
        Path("/root/autodl-tmp/ps4_models/PS4/checkpoint_epoch037.pt"),
        Path("/root/autodl-tmp/ve_models/PS4/checkpoint_epoch037.pt"),
    ):
        if c.is_file():
            return c.resolve()
    return (default_model_dir() / "PS4" / "checkpoint_epoch037.pt").resolve()


def default_eres2net_dir() -> Path:
    # ERES_DIR：download_presence_encoders.sh / score_encoders_on_sep.sh
    # ERES2NET_DIR：旧环境变量名（setup_env / .env_ve）
    env = (
        os.environ.get("ERES_DIR", "").strip()
        or os.environ.get("ERES2NET_DIR", "").strip()
    )
    if env:
        return Path(env).expanduser().resolve()
    return (default_model_dir() / "eres2netv2_zh").resolve()


def default_campplus_dir() -> Path:
    env = os.environ.get("CAMPPLUS_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (default_model_dir() / "campplus_zh").resolve()


def default_ecapa_presence_dir() -> Path:
    """WeSpeaker VoxCeleb ECAPA-TDNN 1024 LM（Presence 对照臂）。"""
    env = os.environ.get("ECAPA_PRESENCE_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (default_model_dir() / "voxceleb_ecapa1024_LM").resolve()


def default_vblink100_dir() -> Path:
    """WeSpeaker VoxBlink2 SimAM-ResNet100 多语模型。"""
    env = os.environ.get("VBLINK100_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (default_model_dir() / "vblink2_samresnet100").resolve()


def default_cond_tasnet_ckpt() -> Path:
    env = os.environ.get("COND_TASNET_CKPT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    for c in (
        default_model_dir() / "cond_tasnet" / "best.pt",
        default_model_dir() / "cond_tasnet_v35" / "best.pt",
        Path("/root/autodl-tmp/models/cond_tasnet/best.pt"),
        Path("/root/autodl-tmp/ve_models/cond_tasnet/best.pt"),
        media_root() / "models" / "cond_tasnet" / "best.pt",
        media_root() / "V3" / "outputs" / "cond_tasnet" / "best.pt",
    ):
        if c.is_file():
            return c.resolve()
    return (default_model_dir() / "cond_tasnet" / "best.pt").resolve()


def default_ecapa_dir() -> Path:
    env = os.environ.get("ECAPA_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    for c in (
        default_model_dir() / "ecapa",
        Path("/root/autodl-tmp/models/ecapa"),
        Path("/root/autodl-tmp/ve_models/ecapa"),
        media_root() / "models" / "ecapa",
    ):
        if c.is_dir() and (c / "hyperparams.yaml").is_file():
            return c.resolve()
    return (default_model_dir() / "ecapa").resolve()


def default_vblink_dir() -> Path:
    """WeSpeaker VoxBlink2 SimAM-ResNet34 目录（avg_model.pt + config.yaml）。"""
    env = os.environ.get("VBLINK_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    for c in (
        default_model_dir() / "vblink2_samresnet34",
        Path("/root/autodl-tmp/ve_models/vblink2_samresnet34"),
        Path("/root/autodl-tmp/models/vblink2_samresnet34"),
    ):
        if c.is_dir() and (c / "avg_model.pt").is_file():
            return c.resolve()
    return (default_model_dir() / "vblink2_samresnet34").resolve()


def normalize_presence_label(raw: object = None, *, split: object = None) -> str:
    """统一成 present/absent。接受 present/pos/1 与 absent/neg/0；缺字段时用 split。"""
    s = str(raw or "").strip().lower()
    if s in ("present", "pos", "1", "true", "yes"):
        return "present"
    if s in ("absent", "neg", "0", "false", "no"):
        return "absent"
    sp = str(split or "").strip().lower()
    if sp in ("pos", "present"):
        return "present"
    if sp in ("neg", "absent"):
        return "absent"
    raise KeyError(f"无法判定 present/absent: label={raw!r} split={split!r}")


def default_spk_chs_dir() -> Path:
    env = os.environ.get("SPK_CHS_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    for c in (
        default_model_dir() / "cnceleb_resnet34_LM",
        Path("/root/autodl-tmp/ps4_models/cnceleb_resnet34_LM"),
    ):
        if c.is_dir():
            return c.resolve()
    return (default_model_dir() / "cnceleb_resnet34_LM").resolve()


def default_wesep_dir() -> Path:
    """旧 REAL-TSE wesep 路径（PS4 回退）；官方 WeSep 见 default_wesep_root。"""
    env = os.environ.get("WESEP_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    for c in (
        Path("/root/autodl-tmp/REAL-TSE-Challenge/wesep_real_tse"),
        Path("/root/REAL-TSE-Challenge/wesep_real_tse"),
        media_root() / "REAL-TSE-Challenge" / "wesep_real_tse",
        media_root() / "VD" / "REAL-TSE-Challenge" / "wesep_real_tse",
    ):
        if c.is_dir():
            return c.resolve()
    return Path("/root/autodl-tmp/REAL-TSE-Challenge/wesep_real_tse")


def default_wesep_root() -> Path:
    """wenet-e2e/wesep 仓库根（含 wesep/ 包；download_wesep.sh 安装）。"""
    env = os.environ.get("WESEP_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    for c in (
        default_model_dir() / "wesep",
        Path("/root/autodl-tmp/wesep"),
        Path("/root/wesep"),
        media_root() / "wesep",
    ):
        if (c / "wesep").is_dir():
            return c.resolve()
    return (default_model_dir() / "wesep").resolve()


def default_moss_onnx_path() -> Path:
    env = os.environ.get("MOSS_ONNX_PATH", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_dir():
            p = p / "simple_model.onnx"
        return p.resolve()
    for c in (
        Path("/root/autodl-tmp/checkpoints/MossFormer2_ONNX/simple_model.onnx"),
        Path("/root/checkpoints/MossFormer2_ONNX/simple_model.onnx"),
        media_root() / "checkpoints" / "MossFormer2_ONNX" / "simple_model.onnx",
        default_model_dir() / "MossFormer2_ONNX" / "simple_model.onnx",
    ):
        if c.is_file():
            return c.resolve()
    return Path("/root/autodl-tmp/checkpoints/MossFormer2_ONNX/simple_model.onnx")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_sys_path() -> None:
    import sys

    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    # VD/tools：本机 media 旁、AutoDL /root/media、仅拷了 VE 时的常见位置
    candidates = [
        media_root() / "VD" / "tools",
        Path("/root/media/VD/tools"),
        Path("/root/VD/tools"),
        ve_root().parent / "VD" / "tools",
        ve_root().parent / "media" / "VD" / "tools",
    ]
    env = os.environ.get("VD_TOOLS", "").strip()
    if env:
        candidates.insert(0, Path(env))
    for vd_tools in candidates:
        if vd_tools.is_dir() and str(vd_tools) not in sys.path:
            sys.path.append(str(vd_tools))
            break
    # VM/scripts：MossFormer ONNX（sep_route / USE_SEP）
    vm_cands = vm_scripts_dirs()
    env_vm = os.environ.get("VM_SCRIPTS", "").strip()
    if env_vm:
        vm_cands.insert(0, Path(env_vm))
    for vm_scripts in vm_cands:
        if vm_scripts.is_dir() and str(vm_scripts) not in sys.path:
            sys.path.append(str(vm_scripts))
            break
    # 官方 WeSep 源码树
    wr = default_wesep_root()
    if (wr / "wesep").is_dir() and str(wr) not in sys.path:
        sys.path.insert(0, str(wr))

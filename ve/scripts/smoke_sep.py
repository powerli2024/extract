#!/usr/bin/env python3
"""冒烟：确认 MossFormer 分离器能加载并产出两路有效波形。

用法（AutoDL）:
  cd /root/extract/ve
  source .env_ve
  ./download_moss_onnx.sh          # 若尚无 ONNX
  python scripts/smoke_sep.py
  python scripts/smoke_sep.py --wav /root/autodl-tmp/datasetA/pos/cmd_0.wav
  python scripts/smoke_sep.py --wav CMD.wav --enroll ENROLL.wav --device cuda:0

退出码: 0=OK，2=缺权重/依赖，3=分离结果异常。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from paths import default_moss_onnx_path, setup_sys_path  # noqa: E402

setup_sys_path()


def _synth_mix(sr: int = 16000, sec: float = 2.0) -> np.ndarray:
    """两正弦叠加，便于检查两路是否都非静音。"""
    t = np.arange(int(sr * sec), dtype=np.float32) / sr
    a = 0.4 * np.sin(2 * np.pi * 220.0 * t)
    b = 0.4 * np.sin(2 * np.pi * 440.0 * t)
    return (a + b).astype(np.float32)


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return float(np.sqrt(np.mean(x**2) + 1e-12))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MossFormer 分离冒烟")
    p.add_argument("--wav", type=Path, default=None, help="真实 CMD wav；默认合成双音")
    p.add_argument("--enroll", type=Path, default=None, help="可选：再测 Presence use_sep")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--peak", type=float, default=0.95)
    p.add_argument("--out-dir", type=Path, default=None, help="写出 spk1/spk2 wav")
    p.add_argument("--max-sec", type=float, default=6.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    onnx = default_moss_onnx_path()
    print(f"[INFO] MOSS_ONNX_PATH expect → {onnx} exists={onnx.is_file()}")
    print(f"[INFO] device={args.device}")

    from presence_gate import try_create_onnx_separator

    t0 = time.time()
    sep = try_create_onnx_separator(peak=args.peak, device=args.device)
    if sep is None:
        print("[FAIL] 分离器创建失败。请:")
        print("  1) ./download_moss_onnx.sh")
        print("  2) 确认 /root/extract/scripts/mossformer2_onnx.py")
        print("  3) pip install onnxruntime-gpu")
        print("  4) export MOSS_ONNX_PATH=/root/autodl-tmp/checkpoints/MossFormer2_ONNX/simple_model.onnx")
        return 2
    print(f"[OK] separator ready in {time.time() - t0:.2f}s  name={type(sep).__name__}")

    sr = 16000
    if args.wav is not None:
        from audio_io import load_audio

        if not args.wav.is_file():
            print(f"[FAIL] wav 不存在: {args.wav}")
            return 2
        mix, sr = load_audio(args.wav)
        print(f"[INFO] load {args.wav} sr={sr} n={len(mix)} dur={len(mix)/sr:.2f}s")
    else:
        mix = _synth_mix(sr=sr, sec=2.0)
        print(f"[INFO] synth mix n={len(mix)} rms={_rms(mix):.4f}")

    if args.max_sec and args.max_sec > 0 and len(mix) > int(args.max_sec * sr):
        mix = mix[: int(args.max_sec * sr)]

    t1 = time.time()
    try:
        s1, s2 = sep.separate(mix, sr=sr)
    except Exception as e:
        print(f"[FAIL] separate() 异常: {e}")
        return 3
    ms = (time.time() - t1) * 1000
    s1 = np.asarray(s1, dtype=np.float32).reshape(-1)
    s2 = np.asarray(s2, dtype=np.float32).reshape(-1)
    print(
        f"[OK] separate {ms:.0f}ms  "
        f"len mix/s1/s2={len(mix)}/{len(s1)}/{len(s2)}  "
        f"rms mix/s1/s2={_rms(mix):.4f}/{_rms(s1):.4f}/{_rms(s2):.4f}"
    )

    ok = True
    if len(s1) < sr // 4 or len(s2) < sr // 4:
        print("[FAIL] 输出过短")
        ok = False
    if _rms(s1) < 1e-4 and _rms(s2) < 1e-4:
        print("[FAIL] 两路近乎静音")
        ok = False
    if not np.isfinite(s1).all() or not np.isfinite(s2).all():
        print("[FAIL] 输出含 NaN/Inf")
        ok = False
    # 合成音：至少一路应明显有能量；真实 wav 不强制两路都响
    if args.wav is None and max(_rms(s1), _rms(s2)) < 0.01:
        print("[FAIL] 合成双音分离后能量过低")
        ok = False

    if args.out_dir is not None:
        from audio_io import save_audio

        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_audio(out / "mix.wav", mix, sr)
        save_audio(out / "spk1.wav", s1, sr)
        save_audio(out / "spk2.wav", s2, sr)
        print(f"[OK] wrote {out}/{{mix,spk1,spk2}}.wav")

    if args.enroll is not None:
        if not args.enroll.is_file():
            print(f"[FAIL] enroll 不存在: {args.enroll}")
            return 2
        from audio_io import load_audio
        from presence_encoder import create_presence_encoder
        from presence_gate import PresenceGate
        from paths import default_eres2net_dir, default_spk_chs_dir

        enroll, _ = load_audio(args.enroll)
        enc = create_presence_encoder(
            "eres2netv2",
            eres_dir=default_eres2net_dir(),
            resnet_dir=default_spk_chs_dir(),
            device=args.device,
        )
        gate0 = PresenceGate(enc, thr=0.0, use_sep=False, separator=None)
        gate1 = PresenceGate(enc, thr=0.0, use_sep=True, separator=sep)
        r0 = gate0.score(enroll, mix, sr=sr)
        r1, streams, _ = gate1.score_with_streams(enroll, mix, sr=sr)
        print(
            f"[INFO] presence mix-only score={r0.score:.4f} best={r0.best_stream} "
            f"sims={r0.sim_streams}"
        )
        print(
            f"[INFO] presence +sep   score={r1.score:.4f} best={r1.best_stream} "
            f"sims={r1.sim_streams} streams={list(streams.keys())}"
        )
        if "spk1" not in streams or "spk2" not in streams:
            print("[FAIL] Presence 未拿到 spk 轨")
            ok = False
        else:
            print("[OK] Presence use_sep 三流打分正常")

    if ok:
        print("[OK] smoke_sep PASSED — 分离模型可用，可开 USE_SEP=1 做拒识对比")
        return 0
    print("[FAIL] smoke_sep 未通过")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())

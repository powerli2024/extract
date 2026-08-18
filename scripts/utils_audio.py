#!/usr/bin/env python3
"""音频工具：16k 加载 + 峰值归一化（默认 peak=0.7）。"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def load_audio(path: str | Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    wav, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return wav.astype(np.float32), target_sr


def save_audio(path: str | Path, wav: np.ndarray, sr: int = 16000) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav.astype(np.float32), sr)


def peak_normalize(wav: np.ndarray, peak: float = 0.7) -> np.ndarray:
    """等比例拉响到峰值 peak（默认 0.7）。"""
    if len(wav) == 0:
        return wav.astype(np.float32)
    p = float(np.max(np.abs(wav)))
    if p < 1e-8:
        return wav.astype(np.float32)
    return (wav * (peak / p)).astype(np.float32)


def preprocess_audio(wav: np.ndarray, peak: float = 0.7) -> np.ndarray:
    if wav.dtype != np.float32:
        wav = wav.astype(np.float32)
    return peak_normalize(wav, peak=peak)


def audio_duration(wav: np.ndarray, sr: int = 16000) -> float:
    return float(len(wav)) / float(sr)


def truncate_wav(
    wav: np.ndarray,
    sr: int = 16000,
    max_sec: float = 6.0,
    mode: str = "energy",
) -> np.ndarray:
    """
    限制时长，降低 MossFormer 注意力显存（长音频易 OOM）。
    mode=head: 取开头；energy: 取能量最高的窗（唤醒段更稳）。
    """
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    max_n = int(max(0.5, float(max_sec)) * int(sr))
    if len(wav) <= max_n:
        return wav
    if mode == "head":
        return wav[:max_n].copy()
    # 滑窗能量最大
    hop = max(1, max_n // 4)
    best_i, best_e = 0, -1.0
    for i in range(0, len(wav) - max_n + 1, hop):
        seg = wav[i : i + max_n]
        e = float(np.dot(seg, seg))
        if e > best_e:
            best_e, best_i = e, i
    # 末尾再比一次
    tail = len(wav) - max_n
    if tail > 0:
        seg = wav[tail:]
        e = float(np.dot(seg, seg))
        if e > best_e:
            best_i = tail
    return wav[best_i : best_i + max_n].copy()


def waveform_energy(wav: np.ndarray) -> float:
    if len(wav) == 0:
        return 0.0
    return float(np.mean(wav.astype(np.float64) ** 2))


def speech_ratio_energy(wav: np.ndarray, frame: int = 320, thr_ratio: float = 0.1) -> float:
    """
    粗略语音占比：帧能量超过全局均值*thr_ratio 的比例。
    不依赖 webrtcvad，避免额外依赖。
    """
    if len(wav) < frame:
        return 0.0 if float(np.max(np.abs(wav))) < 1e-4 else 1.0
    n = len(wav) // frame
    if n <= 0:
        return 0.0
    frames = wav[: n * frame].reshape(n, frame)
    e = np.mean(frames.astype(np.float64) ** 2, axis=1)
    mean_e = float(np.mean(e)) + 1e-12
    return float(np.mean(e > mean_e * thr_ratio))

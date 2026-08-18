#!/usr/bin/env python3
"""CMD 时间选择：滑窗 / 能量段。只裁时间，不改频谱。"""
from __future__ import annotations

from typing import Any, Iterator

import numpy as np


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))


def iter_sliding_windows(
    wav: np.ndarray,
    sr: int = 16000,
    *,
    win_sec: float = 0.8,
    hop_sec: float = 0.4,
    min_sec: float = 0.35,
) -> list[dict[str, Any]]:
    """在 mix 上切重叠窗。短于一窗则整段作为唯一窗。"""
    w = np.asarray(wav, dtype=np.float32).reshape(-1)
    n = int(w.size)
    win = max(1, int(round(float(win_sec) * sr)))
    hop = max(1, int(round(float(hop_sec) * sr)))
    min_n = max(1, int(round(float(min_sec) * sr)))
    if n <= 0:
        return []
    if n <= win:
        return [{"start": 0, "end": n, "wav": w.copy(), "sec": n / float(sr)}]
    out: list[dict[str, Any]] = []
    i = 0
    while i + min_n <= n:
        j = min(n, i + win)
        if j - i < min_n:
            break
        seg = w[i:j]
        out.append({
            "start": int(i),
            "end": int(j),
            "wav": seg.copy(),
            "sec": (j - i) / float(sr),
        })
        if j >= n:
            break
        i += hop
    if out and out[-1]["end"] < n and (n - out[-1]["start"]) >= min_n:
        # 保证覆盖到末尾
        i = max(0, n - win)
        if not out or int(out[-1]["start"]) != i:
            out.append({
                "start": int(i),
                "end": int(n),
                "wav": w[i:n].copy(),
                "sec": (n - i) / float(sr),
            })
    return out


def energy_speech_segments(
    wav: np.ndarray,
    sr: int = 16000,
    *,
    frame_ms: float = 20.0,
    pad_ms: float = 80.0,
    gap_ms: float = 150.0,
    min_speech_ms: float = 200.0,
    thr_ratio: float = 0.12,
    max_seg_sec: float = 2.5,
) -> list[dict[str, Any]]:
    """能量段：连续语音岛。失败则回退整段。"""
    w = np.asarray(wav, dtype=np.float32).reshape(-1)
    n = int(w.size)
    if n < int(0.05 * sr):
        return [{"start": 0, "end": n, "wav": w.copy(), "sec": n / float(sr) if sr else 0.0}]

    frame = max(1, int(sr * frame_ms / 1000.0))
    n_frames = n // frame
    if n_frames < 3:
        return [{"start": 0, "end": n, "wav": w.copy(), "sec": n / float(sr)}]

    frames = w[: n_frames * frame].reshape(n_frames, frame)
    energy = np.mean(frames.astype(np.float64) ** 2, axis=1)
    noise = float(np.percentile(energy, 20))
    peak = float(np.percentile(energy, 95))
    if peak <= noise * 1.5:
        return [{"start": 0, "end": n, "wav": w.copy(), "sec": n / float(sr)}]

    thr = noise + (peak - noise) * float(thr_ratio)
    speech = energy >= thr
    # 填小间隙
    gap_f = max(0, int(round(gap_ms / frame_ms)))
    if gap_f > 0:
        filled = speech.copy()
        last = -10**9
        for i, on in enumerate(speech):
            if on:
                if last >= 0 and i - last <= gap_f:
                    filled[last : i + 1] = True
                last = i
        speech = filled

    segs: list[tuple[int, int]] = []
    i = 0
    while i < n_frames:
        if not speech[i]:
            i += 1
            continue
        j = i + 1
        while j < n_frames and speech[j]:
            j += 1
        segs.append((i, j))
        i = j

    pad_f = max(0, int(round(pad_ms / frame_ms)))
    min_f = max(1, int(round(min_speech_ms / frame_ms)))
    out: list[dict[str, Any]] = []
    max_n = int(max_seg_sec * sr) if max_seg_sec and max_seg_sec > 0 else n
    for f0, f1 in segs:
        if (f1 - f0) < min_f:
            continue
        i0 = max(0, (f0 - pad_f) * frame)
        i1 = min(n, (f1 + pad_f) * frame)
        if i1 - i0 < int(sr * min_speech_ms / 1000.0):
            continue
        if i1 - i0 > max_n:
            # 段内取能量最高的 max_n
            best_i, best_e = i0, -1.0
            hop = max(1, max_n // 4)
            for k in range(i0, i1 - max_n + 1, hop):
                seg = w[k : k + max_n]
                e = float(np.dot(seg, seg))
                if e > best_e:
                    best_e, best_i = e, k
            i0, i1 = best_i, best_i + max_n
        seg = w[i0:i1]
        out.append({
            "start": int(i0),
            "end": int(i1),
            "wav": seg.copy(),
            "sec": (i1 - i0) / float(sr),
        })
    if not out:
        return [{"start": 0, "end": n, "wav": w.copy(), "sec": n / float(sr)}]
    return out


def crop_with_pad(
    wav: np.ndarray,
    start: int,
    end: int,
    sr: int = 16000,
    *,
    pad_ms: float = 80.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """按窗起止裁切，两侧各留 pad。"""
    w = np.asarray(wav, dtype=np.float32).reshape(-1)
    n = int(w.size)
    pad = max(0, int(round(float(pad_ms) * sr / 1000.0)))
    i0 = max(0, int(start) - pad)
    i1 = min(n, int(end) + pad)
    if i1 <= i0:
        i0, i1 = 0, n
    cropped = w[i0:i1]
    meta = {
        "start": int(i0),
        "end": int(i1),
        "pad_ms": float(pad_ms),
        "win_start": int(start),
        "win_end": int(end),
        "sec": (i1 - i0) / float(sr) if sr else 0.0,
        "full_utt": bool(i0 == 0 and i1 == n),
    }
    return cropped.astype(np.float32), meta


def cmd_windows(
    wav: np.ndarray,
    sr: int = 16000,
    *,
    mode: str = "slide",
    win_sec: float = 0.8,
    hop_sec: float = 0.4,
    pad_ms: float = 80.0,
    min_rms: float = 1e-4,
) -> list[dict[str, Any]]:
    mode = (mode or "slide").lower().strip()
    if mode in ("off", "none", "0", ""):
        w = np.asarray(wav, dtype=np.float32).reshape(-1)
        return [{"start": 0, "end": int(w.size), "wav": w.copy(), "sec": w.size / float(sr)}]
    if mode in ("energy", "vad", "seg"):
        wins = energy_speech_segments(wav, sr, pad_ms=pad_ms)
    else:
        wins = iter_sliding_windows(wav, sr, win_sec=win_sec, hop_sec=hop_sec)
    kept = []
    for w in wins:
        if _rms(w["wav"]) < float(min_rms) and len(wins) > 1:
            continue
        kept.append(w)
    return kept or wins


def iter_windows(*args: Any, **kwargs: Any) -> Iterator[dict[str, Any]]:
    yield from cmd_windows(*args, **kwargs)


if __name__ == "__main__":
    sr = 16000
    t = np.zeros(sr * 3, dtype=np.float32)
    t[sr : sr + sr // 2] = 0.25
    wins = iter_sliding_windows(t, sr, win_sec=0.8, hop_sec=0.4)
    segs = energy_speech_segments(t, sr)
    crop, meta = crop_with_pad(t, sr, int(1.5 * sr), sr, pad_ms=80)
    assert len(wins) >= 5, len(wins)
    assert segs and segs[0]["end"] > segs[0]["start"]
    assert meta["start"] <= sr and meta["end"] >= int(1.5 * sr)
    assert len(crop) > 0
    print("window_geom ok", "n_slide", len(wins), "n_energy", len(segs), "crop_sec", meta["sec"])

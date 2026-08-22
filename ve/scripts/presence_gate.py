#!/usr/bin/env python3
"""PresenceGate：仅判断「enroll 说话人是否出现在 CMD」。

raw = max_k sim(enroll, stream_k)
可选 ScoreNormalizer：
  enroll_znorm / test_znorm / asnorm = 0.5*(z_A+z_B)
reject iff score < thr
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from audio_io import cosine_sim, peak_normalize, save_audio, vad_crop_speech
from paths import default_moss_onnx_path, extract_root, setup_sys_path
from presence_encoder import PresenceEncoder

setup_sys_path()


@dataclass
class PresenceResult:
    score: float
    sim_mix: float
    sim_streams: dict[str, float]
    best_stream: str
    reject: bool
    reason: str
    thr: float
    sep_depth: int = 0
    n_streams: int = 1
    sep_wav_dir: str | None = None
    score_raw: float | None = None
    znorm_mu: float | None = None
    znorm_sigma: float | None = None
    znorm_mu_test: float | None = None
    znorm_sigma_test: float | None = None
    z_enroll: float | None = None
    z_test: float | None = None
    score_norm: str = "raw"
    enroll_vad: dict | None = None
    cmd_window_mode: str = "off"
    best_window: dict | None = None
    second_window: dict | None = None
    n_windows: int = 0
    veto_score: float | None = None
    veto_backend: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "presence_score": round(self.score, 6),
            "sim_enroll_mix": round(self.sim_mix, 6),
            "sim_streams": {k: round(v, 6) for k, v in self.sim_streams.items()},
            "best_stream": self.best_stream,
            "reject_decision": self.reject,
            "reject_reason": self.reason if self.reject else "",
            "presence_thr": self.thr,
            "sep_depth": self.sep_depth,
            "n_streams": self.n_streams,
            "sep_wav_dir": self.sep_wav_dir,
            "score_norm": self.score_norm,
            "cmd_window_mode": self.cmd_window_mode,
            "n_windows": int(self.n_windows),
        }
        if self.best_window is not None:
            d["best_window"] = self.best_window
        if self.second_window is not None:
            d["second_window"] = self.second_window
        if self.veto_score is not None:
            d["veto_score"] = round(float(self.veto_score), 6)
        if self.veto_backend:
            d["veto_backend"] = self.veto_backend
        if self.score_raw is not None:
            d["presence_score_raw"] = round(self.score_raw, 6)
        if self.znorm_mu is not None:
            d["znorm_mu"] = round(self.znorm_mu, 6)
        if self.znorm_sigma is not None:
            d["znorm_sigma"] = round(self.znorm_sigma, 6)
        if self.znorm_mu_test is not None:
            d["znorm_mu_test"] = round(self.znorm_mu_test, 6)
        if self.znorm_sigma_test is not None:
            d["znorm_sigma_test"] = round(self.znorm_sigma_test, 6)
        if self.z_enroll is not None:
            d["z_enroll"] = round(self.z_enroll, 6)
        if self.z_test is not None:
            d["z_test"] = round(self.z_test, 6)
        if self.enroll_vad is not None:
            d["enroll_vad"] = self.enroll_vad
        return d


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return float(np.sqrt(np.mean(x**2) + 1e-12))


def save_sep_streams(
    out_dir: Path,
    streams: dict[str, np.ndarray],
    *,
    sr: int = 16000,
) -> list[str]:
    """写出中间轨 wav，返回相对文件名列表。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for name, w in streams.items():
        if name == "peak":
            continue
        fn = f"{name}.wav"
        save_audio(out_dir / fn, w, sr)
        written.append(fn)
    meta = {
        "sr": sr,
        "streams": {
            k: {"n": int(len(v)), "rms": round(_rms(v), 6)}
            for k, v in streams.items()
            if k != "peak"
        },
    }
    (out_dir / "meta.json").write_text(
        __import__("json").dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return written


class PresenceGate:
    def __init__(
        self,
        encoder: PresenceEncoder,
        *,
        thr: float = 0.25,
        use_sep: bool = False,
        separator: Any | None = None,
        peak: float = 0.95,
        sep_depth: int = 1,
        min_stream_rms: float = 1e-4,
        max_sep_sec: float = 8.0,
        enroll_znorm: Any | None = None,
        test_znorm: Any | None = None,
        score_normalizer: Any | None = None,
        enroll_vad: bool = True,
        enroll_vad_max_sec: float = 4.0,
        cmd_window_mode: str = "off",
        win_sec: float = 0.8,
        hop_sec: float = 0.4,
        win_pad_ms: float = 80.0,
        veto_encoder: PresenceEncoder | None = None,
        veto_margin: float = 0.12,
        veto_gray: float = 0.10,
        veto_windows: bool = False,
    ):
        self.encoder = encoder
        self.thr = float(thr)
        self.separator = separator
        # use_sep 兼容旧开关；实际以 sep_depth 为准
        depth = int(sep_depth)
        if use_sep and depth < 1:
            depth = 1
        if not use_sep and separator is None:
            depth = 0
        if depth > 0 and separator is None:
            depth = 0
        self.sep_depth = max(0, depth)
        self.use_sep = self.sep_depth >= 1
        self.peak = peak
        self.min_stream_rms = float(min_stream_rms)
        self.max_sep_sec = float(max_sep_sec)
        self.enroll_vad = bool(enroll_vad)
        self.enroll_vad_max_sec = float(enroll_vad_max_sec)
        self.cmd_window_mode = (cmd_window_mode or "off").lower().strip()
        if self.cmd_window_mode in ("0", "none", "false", ""):
            self.cmd_window_mode = "off"
        if self.cmd_window_mode in ("1", "true", "on", "yes"):
            self.cmd_window_mode = "slide"
        self.win_sec = float(win_sec)
        self.hop_sec = float(hop_sec)
        self.win_pad_ms = float(win_pad_ms)
        self.veto_encoder = veto_encoder
        self.veto_margin = float(veto_margin)
        self.veto_gray = float(veto_gray)
        self.veto_windows = bool(veto_windows)
        self._enroll_cache: dict[str, np.ndarray] = {}
        self._enroll_vad_meta: dict[str, dict] = {}
        self._veto_enroll_cache: dict[str, np.ndarray] = {}

        # 归一化：优先 score_normalizer；否则由 enroll/test bank 推断
        if score_normalizer is not None:
            self.score_normalizer = score_normalizer
        elif enroll_znorm is not None or test_znorm is not None:
            from cohort_znorm import ScoreNormalizer

            if enroll_znorm is not None and test_znorm is not None:
                mode = "asnorm"
            elif enroll_znorm is not None:
                mode = "enroll_znorm"
            else:
                mode = "test_znorm"
            self.score_normalizer = ScoreNormalizer(
                mode, enroll_bank=enroll_znorm, test_bank=test_znorm
            )
        else:
            self.score_normalizer = None
        # 兼容旧字段
        self.enroll_znorm = enroll_znorm
        self.test_znorm = test_znorm

    def prepare_enroll(
        self, wav: np.ndarray, sr: int = 16000
    ) -> tuple[np.ndarray, dict]:
        """Enroll 预处理：可选能量 VAD 裁剪静音。"""
        w = np.asarray(wav, dtype=np.float32).reshape(-1)
        if not self.enroll_vad:
            return w, {"vad": "off", "cropped": False, "out_sec": round(len(w) / float(sr), 4)}
        return vad_crop_speech(
            w, sr, max_sec=self.enroll_vad_max_sec
        )

    def cache_enroll(self, key: str, wav: np.ndarray, sr: int = 16000) -> np.ndarray:
        if key not in self._enroll_cache:
            cropped, meta = self.prepare_enroll(wav, sr)
            self._enroll_vad_meta[key] = meta
            self._enroll_cache[key] = self.encoder.embed(cropped, sr)
        return self._enroll_cache[key]

    def _clip(self, wav: np.ndarray, sr: int) -> np.ndarray:
        w = np.asarray(wav, dtype=np.float32).reshape(-1)
        if self.max_sep_sec and self.max_sep_sec > 0:
            n = int(self.max_sep_sec * sr)
            if len(w) > n:
                w = w[:n]
        return w

    def _separate_one(self, wav: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray] | None:
        if self.separator is None or not hasattr(self.separator, "separate"):
            return None
        try:
            peak = peak_normalize(self._clip(wav, sr), peak=self.peak)
            s1, s2 = self.separator.separate(peak, sr=sr)
            return (
                np.asarray(s1, dtype=np.float32).reshape(-1),
                np.asarray(s2, dtype=np.float32).reshape(-1),
            )
        except Exception as e:
            print(f"[WARN] presence sep failed: {e}", flush=True)
            return None

    def _sep_streams(self, cmd: np.ndarray, sr: int) -> dict[str, np.ndarray]:
        """按 sep_depth 级联分离，保留所有中间轨（含 mix）。"""
        mix = np.asarray(cmd, dtype=np.float32).reshape(-1)
        streams: dict[str, np.ndarray] = {"mix": mix}
        if self.sep_depth < 1 or self.separator is None:
            return streams

        # depth 1：对 mix 分离
        pair = self._separate_one(mix, sr)
        if pair is None:
            return streams
        s1, s2 = pair
        streams["d1_spk1"] = s1
        streams["d1_spk2"] = s2

        # depth>=2：对上一层叶子再分
        parents = [("d1_spk1", s1), ("d1_spk2", s2)]
        for d in range(2, self.sep_depth + 1):
            next_parents: list[tuple[str, np.ndarray]] = []
            for pname, pw in parents:
                if _rms(pw) < self.min_stream_rms:
                    continue
                pair = self._separate_one(pw, sr)
                if pair is None:
                    continue
                a, b = pair
                # 命名：d2_spk1_a / d2_spk1_b（由 d1_spk1 再分）
                base = pname
                for prefix in ("d1_", "d2_", "d3_"):
                    if base.startswith(prefix):
                        base = base[len(prefix) :]
                        break
                base = base.replace("_a", "").replace("_b", "")
                na, nb = f"d{d}_{base}_a", f"d{d}_{base}_b"
                if na in streams:
                    na, nb = f"d{d}_{pname}_a", f"d{d}_{pname}_b"
                streams[na] = a
                streams[nb] = b
                next_parents.append((na, a))
                next_parents.append((nb, b))
            parents = next_parents
            if not parents:
                break
        return streams

    def score_with_streams(
        self,
        enroll_wav: np.ndarray,
        cmd_wav: np.ndarray,
        *,
        enroll_key: str | None = None,
        sr: int = 16000,
        thr: float | None = None,
        save_dir: Path | None = None,
    ) -> tuple[PresenceResult, dict[str, np.ndarray], np.ndarray]:
        """返回 (PresenceResult, streams, enroll_emb)。"""
        thr = self.thr if thr is None else float(thr)
        enroll_vad_meta: dict | None = None
        if enroll_key:
            e = self.cache_enroll(enroll_key, enroll_wav, sr)
            enroll_vad_meta = self._enroll_vad_meta.get(enroll_key)
        else:
            cropped, enroll_vad_meta = self.prepare_enroll(enroll_wav, sr)
            e = self.encoder.embed(cropped, sr)

        streams = self._sep_streams(cmd_wav, sr)
        sims: dict[str, float] = {}
        embs: dict[str, np.ndarray] = {}
        for name, w in streams.items():
            if name == "peak":
                continue
            if _rms(w) < self.min_stream_rms and name != "mix":
                continue
            emb = self.encoder.embed(w, sr)
            embs[name] = emb
            sims[name] = cosine_sim(e, emb)

        sim_mix = float(sims.get("mix", 0.0))
        best_stream = "mix"
        best = sim_mix
        for name, s in sims.items():
            if s > best:
                best = s
                best_stream = name

        best_window: dict | None = None
        second_window: dict | None = None
        n_windows = 0
        mix_wav = streams.get("mix", np.asarray(cmd_wav, dtype=np.float32).reshape(-1))
        if self.cmd_window_mode != "off":
            from window_geom import cmd_windows

            wins = cmd_windows(
                mix_wav,
                sr,
                mode=self.cmd_window_mode,
                win_sec=self.win_sec,
                hop_sec=self.hop_sec,
                pad_ms=self.win_pad_ms,
                min_rms=self.min_stream_rms,
            )
            ranked: list[dict[str, Any]] = []
            segs = [w["wav"] for w in wins]
            if not segs:
                segs = [mix_wav]
                wins = [{"start": 0, "end": int(mix_wav.size), "wav": mix_wav, "sec": mix_wav.size / float(sr)}]
            embs_w = self.encoder.embed_batch(segs, sr)
            for winfo, emb_w in zip(wins, embs_w):
                sc = cosine_sim(e, emb_w)
                ranked.append({
                    "start": int(winfo["start"]),
                    "end": int(winfo["end"]),
                    "score": float(sc),
                    "sec": float(winfo.get("sec") or 0.0),
                })
            ranked.sort(key=lambda x: -x["score"])
            n_windows = len(ranked)
            if ranked:
                best_window = ranked[0]
                if len(ranked) > 1:
                    second_window = ranked[1]
                # T2：Presence 用 max_window 替换整句/分离 max
                best = float(best_window["score"])
                best_stream = "mix_window"
                sims["mix_window"] = best
                embs["mix_window"] = embs_w[0] if embs_w else e
                for i, winfo in enumerate(wins):
                    if int(winfo["start"]) == int(best_window["start"]) and int(winfo["end"]) == int(best_window["end"]):
                        embs["mix_window"] = embs_w[i]
                        break

        score_raw = float(best)
        t_emb = embs.get(best_stream)
        if t_emb is None:
            t_emb = embs.get("mix")
        if t_emb is None:
            t_emb = e

        z_mu = z_sig = z_mu_t = z_sig_t = z_a = z_b = None
        score_norm = "raw"
        score = score_raw
        if self.score_normalizer is not None:
            test_key = None
            if enroll_key and best_stream:
                test_key = f"{enroll_key}::{best_stream}"
            nout = self.score_normalizer.apply(
                score_raw, e, t_emb, enroll_key=enroll_key, test_key=test_key
            )
            score = float(nout.score)
            score_norm = nout.mode
            z_mu, z_sig = nout.mu_a, nout.sigma_a
            z_mu_t, z_sig_t = nout.mu_b, nout.sigma_b
            z_a, z_b = nout.z_a, nout.z_b
        reject = score < thr
        reason = "speaker_absent" if reject else ""

        veto_score = None
        veto_backend = None
        if (not reject) and self.veto_encoder is not None:
            from lift_common import camp_veto

            v_key = enroll_key or "__anon__"
            if v_key not in self._veto_enroll_cache:
                cropped, _meta = self.prepare_enroll(enroll_wav, sr)
                self._veto_enroll_cache[v_key] = self.veto_encoder.embed(cropped, sr)
            ve = self._veto_enroll_cache[v_key]
            if best_window is not None:
                i0, i1 = int(best_window["start"]), int(best_window["end"])
                tgt = mix_wav[i0:i1]
            else:
                tgt = streams.get(best_stream, mix_wav)
            veto_score = cosine_sim(ve, self.veto_encoder.embed(tgt, sr))
            veto_backend = getattr(self.veto_encoder, "name", "veto")
            if camp_veto(score, veto_score, thr, gray=self.veto_gray, margin=self.veto_margin):
                reject = True
                reason = "camp_veto"

        if (
            (not reject)
            and self.veto_windows
            and best_window is not None
            and second_window is not None
        ):
            from lift_common import window_veto

            if window_veto(
                best_window.get("score"),
                second_window.get("score"),
                thr,
                gray=self.veto_gray,
                margin=self.veto_margin,
            ):
                reject = True
                reason = "window_veto"

        sep_wav_dir = None
        if save_dir is not None:
            save_sep_streams(Path(save_dir), streams, sr=sr)
            sep_wav_dir = str(Path(save_dir).resolve())

        pr = PresenceResult(
            score=score,
            sim_mix=sim_mix,
            sim_streams=sims,
            best_stream=best_stream,
            reject=reject,
            reason=reason,
            thr=thr,
            sep_depth=self.sep_depth,
            n_streams=len(sims),
            sep_wav_dir=sep_wav_dir,
            score_raw=score_raw,
            znorm_mu=z_mu,
            znorm_sigma=z_sig,
            znorm_mu_test=z_mu_t,
            znorm_sigma_test=z_sig_t,
            z_enroll=z_a,
            z_test=z_b,
            score_norm=score_norm,
            enroll_vad=enroll_vad_meta,
            cmd_window_mode=self.cmd_window_mode,
            best_window=best_window,
            second_window=second_window,
            n_windows=n_windows,
            veto_score=veto_score,
            veto_backend=veto_backend,
        )
        return pr, streams, e

    def score(
        self,
        enroll_wav: np.ndarray,
        cmd_wav: np.ndarray,
        *,
        enroll_key: str | None = None,
        sr: int = 16000,
        thr: float | None = None,
        save_dir: Path | None = None,
    ) -> PresenceResult:
        pr, _streams, _emb = self.score_with_streams(
            enroll_wav,
            cmd_wav,
            enroll_key=enroll_key,
            sr=sr,
            thr=thr,
            save_dir=save_dir,
        )
        return pr


def _ensure_moss_env() -> Path | None:
    """设置 MOSS_ONNX_PATH（若未设且文件存在）。"""
    p = default_moss_onnx_path()
    if p.is_file():
        os.environ.setdefault("MOSS_ONNX_PATH", str(p))
        return p
    return None


def try_create_onnx_separator(peak: float = 0.7, device: str = "cuda:0"):
    """创建 MossFormer ONNX 分离器（Presence USE_SEP / sep_route）。"""
    setup_sys_path()
    onnx_path = _ensure_moss_env()
    try:
        from mossformer2_onnx import MossFormer2Separator

        sep = MossFormer2Separator(peak=peak, device=device)
        print(
            f"[INFO] MossFormer separator: mossformer2_onnx"
            + (f" ← {onnx_path}" if onnx_path else ""),
            flush=True,
        )
        return sep
    except Exception as e:
        print(
            "[WARN] MossFormer 不可用，Presence/sep_route 将仅用 mix。"
            f" 原因: mossformer2_onnx:{e}",
            flush=True,
        )
        print(
            "[HINT] AutoDL:\n"
            f"  1) 确认 {extract_root() / 'scripts' / 'mossformer2_onnx.py'}\n"
            "  2) cd ve && ./download_moss_onnx.sh\n"
            "  3) export MOSS_ONNX_PATH=/root/autodl-tmp/checkpoints/MossFormer2_ONNX/simple_model.onnx\n"
            f"  4) {sys.executable} -m pip install -r requirements_moss_ort.txt",
            flush=True,
        )
        return None

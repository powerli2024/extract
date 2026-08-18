#!/usr/bin/env python3
"""声纹分数归一化：enroll Z-Norm / test Z-Norm / AS-Norm。

enroll Z-Norm:  s' = (s - μ_A) / σ_A     （A vs 干净路人 clean_kws）
test   Z-Norm:  s' = (s - μ_B) / σ_B     （获胜轨 B* vs CMD 路人 mix500）
AS-Norm:        s' = 0.5 * (z_A + z_B)
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from audio_io import cosine_sim, load_audio

_SSB = re.compile(r"(SSB\d{4})", re.IGNORECASE)
NormMode = Literal["raw", "enroll_znorm", "test_znorm", "asnorm"]


def speaker_id_from_name(path: Path) -> str:
    stem = path.stem
    m = _SSB.search(stem)
    if m:
        return m.group(1).upper()
    parts = stem.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return stem


def list_cohort_wavs(
    cohort_dir: Path,
    *,
    per_spk: int = 2,
    max_files: int = 400,
    seed: int = 0,
    strategy: str = "auto",
) -> list[Path]:
    """抽样路人 wav。

    strategy:
      per_spk — 每说话人最多 per_spk 条（适合 clean_kws 多说话人）
      flat    — 全局随机抽 max_files（适合 mix500 文件名说话人单一）
      auto    — 说话人很少则 flat，否则 per_spk
    """
    root = Path(cohort_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"cohort 目录不存在: {root}")
    wavs = sorted(root.rglob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"cohort 下无 wav: {root}")

    by_spk: dict[str, list[Path]] = defaultdict(list)
    for p in wavs:
        by_spk[speaker_id_from_name(p)].append(p)

    rng = np.random.default_rng(seed)
    strat = (strategy or "auto").lower()
    if strat == "auto":
        strat = "flat" if len(by_spk) <= 3 else "per_spk"

    if strat == "flat":
        order = list(wavs)
        rng.shuffle(order)
        picked = order[: max(1, int(max_files))]
        print(
            f"[INFO] cohort sample strategy=flat n={len(picked)}/{len(wavs)} "
            f"n_spk_tag={len(by_spk)}",
            flush=True,
        )
        return picked

    picked: list[Path] = []
    spks = sorted(by_spk.keys())
    rng.shuffle(spks)
    for spk in spks:
        files = list(by_spk[spk])
        rng.shuffle(files)
        take = files[: max(1, int(per_spk))]
        picked.extend(take)
        if len(picked) >= max_files:
            break
    picked = picked[:max_files]
    print(
        f"[INFO] cohort sample strategy=per_spk n={len(picked)} "
        f"n_spk={len(by_spk)} per_spk={per_spk}",
        flush=True,
    )
    return picked


def _l2_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return (x / n).astype(np.float64)


class CohortStats:
    """预计算 cohort 嵌入；对任意 probe 估 (μ, σ)。"""

    def __init__(
        self,
        cohort_embs: np.ndarray,
        *,
        eps: float = 1e-3,
        paths: list[Path] | None = None,
        name: str = "cohort",
    ):
        mat = _l2_rows(np.asarray(cohort_embs, dtype=np.float64))
        if mat.shape[0] < 8:
            raise ValueError(f"cohort 太少: n={mat.shape[0]}，建议 >= 50")
        self.cohort = mat
        self.eps = float(eps)
        self.paths = [Path(p) for p in (paths or [])]
        self.name = name
        self._cache: dict[str, tuple[float, float]] = {}

    @property
    def n_cohort(self) -> int:
        return int(self.cohort.shape[0])

    def stats(self, emb: np.ndarray, key: str | None = None) -> tuple[float, float]:
        if key is not None and key in self._cache:
            return self._cache[key]
        e = _l2_rows(emb).reshape(-1)
        sims = self.cohort @ e
        mu = float(np.mean(sims))
        sigma = max(float(np.std(sims)), self.eps)
        if key is not None:
            self._cache[key] = (mu, sigma)
        return mu, sigma

    def zscore(
        self, raw_score: float, emb: np.ndarray, key: str | None = None
    ) -> tuple[float, float, float]:
        mu, sigma = self.stats(emb, key=key)
        return (float(raw_score) - mu) / sigma, mu, sigma

    # 兼容旧名
    def normalize(
        self, raw_score: float, enroll_emb: np.ndarray, key: str | None = None
    ) -> tuple[float, float, float]:
        return self.zscore(raw_score, enroll_emb, key=key)

    def to_meta(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_cohort": self.n_cohort,
            "eps": self.eps,
            "n_cached": len(self._cache),
        }


# 旧名兼容
EnrollZNorm = CohortStats


@dataclass
class NormOut:
    score: float
    mode: str
    mu_a: float | None = None
    sigma_a: float | None = None
    mu_b: float | None = None
    sigma_b: float | None = None
    z_a: float | None = None
    z_b: float | None = None


class ScoreNormalizer:
    """按 mode 把 raw cosine 变成校准分。"""

    def __init__(
        self,
        mode: NormMode = "raw",
        *,
        enroll_bank: CohortStats | None = None,
        test_bank: CohortStats | None = None,
    ):
        mode = (mode or "raw").lower()  # type: ignore[assignment]
        if mode not in ("raw", "enroll_znorm", "test_znorm", "asnorm"):
            raise ValueError(f"未知 score_norm mode={mode}")
        if mode in ("enroll_znorm", "asnorm") and enroll_bank is None:
            raise ValueError(f"{mode} 需要 enroll_bank")
        if mode in ("test_znorm", "asnorm") and test_bank is None:
            raise ValueError(f"{mode} 需要 test_bank")
        self.mode: NormMode = mode  # type: ignore[assignment]
        self.enroll_bank = enroll_bank
        self.test_bank = test_bank

    def apply(
        self,
        raw_score: float,
        enroll_emb: np.ndarray,
        test_emb: np.ndarray,
        *,
        enroll_key: str | None = None,
        test_key: str | None = None,
    ) -> NormOut:
        if self.mode == "raw":
            return NormOut(score=float(raw_score), mode="raw")

        z_a = mu_a = sig_a = None
        z_b = mu_b = sig_b = None
        if self.enroll_bank is not None and self.mode in ("enroll_znorm", "asnorm"):
            z_a, mu_a, sig_a = self.enroll_bank.zscore(
                raw_score, enroll_emb, key=enroll_key
            )
        if self.test_bank is not None and self.mode in ("test_znorm", "asnorm"):
            z_b, mu_b, sig_b = self.test_bank.zscore(
                raw_score, test_emb, key=test_key
            )

        if self.mode == "enroll_znorm":
            score = float(z_a)  # type: ignore[arg-type]
        elif self.mode == "test_znorm":
            score = float(z_b)  # type: ignore[arg-type]
        else:
            score = 0.5 * (float(z_a) + float(z_b))  # type: ignore[arg-type]

        return NormOut(
            score=score,
            mode=self.mode,
            mu_a=mu_a,
            sigma_a=sig_a,
            mu_b=mu_b,
            sigma_b=sig_b,
            z_a=z_a,
            z_b=z_b,
        )

    def to_meta(self) -> dict[str, Any]:
        meta: dict[str, Any] = {"score_norm": self.mode}
        if self.enroll_bank is not None:
            meta["enroll_cohort"] = self.enroll_bank.to_meta()
        if self.test_bank is not None:
            meta["test_cohort"] = self.test_bank.to_meta()
        return meta


def embed_cohort_wavs(
    encoder: Any,
    paths: list[Path],
    *,
    sr: int = 16000,
    desc: str = "cohort_embed",
) -> np.ndarray:
    embs: list[np.ndarray] = []
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore
    it = tqdm(paths, desc=desc, unit="wav", mininterval=0.5) if tqdm else paths
    for p in it:
        wav, _ = load_audio(p, sr=sr)
        embs.append(np.asarray(encoder.embed(wav, sr), dtype=np.float32).reshape(-1))
    return np.stack(embs, axis=0)


def build_cohort_stats(
    encoder: Any,
    cohort_dir: Path,
    *,
    name: str = "cohort",
    per_spk: int = 2,
    max_files: int = 400,
    seed: int = 0,
    eps: float = 1e-3,
    sr: int = 16000,
    strategy: str = "auto",
) -> CohortStats:
    paths = list_cohort_wavs(
        cohort_dir,
        per_spk=per_spk,
        max_files=max_files,
        seed=seed,
        strategy=strategy,
    )
    print(
        f"[INFO] {name} cohort: dir={cohort_dir} n_wav={len(paths)}",
        flush=True,
    )
    mat = embed_cohort_wavs(encoder, paths, sr=sr, desc=f"{name}_embed")
    z = CohortStats(mat, eps=eps, paths=paths, name=name)
    probe = mat[0]
    mu0, sig0 = z.stats(probe, key=None)
    print(
        f"[INFO] {name} ready n={z.n_cohort} dim={mat.shape[1]} "
        f"probe_vs_cohort μ={mu0:.4f} σ={sig0:.4f}",
        flush=True,
    )
    return z


def build_enroll_znorm(
    encoder: Any,
    cohort_dir: Path,
    *,
    per_spk: int = 2,
    max_files: int = 400,
    seed: int = 0,
    eps: float = 1e-3,
    sr: int = 16000,
    strategy: str = "auto",
) -> CohortStats:
    return build_cohort_stats(
        encoder,
        cohort_dir,
        name="enroll_clean",
        per_spk=per_spk,
        max_files=max_files,
        seed=seed,
        eps=eps,
        sr=sr,
        strategy=strategy,
    )


def build_test_znorm(
    encoder: Any,
    cohort_dir: Path,
    *,
    max_files: int = 500,
    seed: int = 0,
    eps: float = 1e-3,
    sr: int = 16000,
    strategy: str = "flat",
) -> CohortStats:
    """CMD 域路人（mix500）：默认 flat 抽样。"""
    return build_cohort_stats(
        encoder,
        cohort_dir,
        name="test_cmd",
        per_spk=1,
        max_files=max_files,
        seed=seed,
        eps=eps,
        sr=sr,
        strategy=strategy,
    )


def build_score_normalizer(
    encoder: Any,
    *,
    mode: NormMode,
    enroll_dir: Path | None = None,
    test_dir: Path | None = None,
    enroll_per_spk: int = 2,
    enroll_max_files: int = 400,
    test_max_files: int = 500,
    seed: int = 0,
    eps: float = 1e-3,
) -> ScoreNormalizer:
    mode = (mode or "raw").lower()  # type: ignore[assignment]
    enroll_bank = test_bank = None
    if mode in ("enroll_znorm", "asnorm"):
        if enroll_dir is None or not Path(enroll_dir).is_dir():
            raise FileNotFoundError(f"enroll cohort 目录无效: {enroll_dir}")
        enroll_bank = build_enroll_znorm(
            encoder,
            Path(enroll_dir),
            per_spk=enroll_per_spk,
            max_files=enroll_max_files,
            seed=seed,
            eps=eps,
        )
    if mode in ("test_znorm", "asnorm"):
        if test_dir is None or not Path(test_dir).is_dir():
            raise FileNotFoundError(f"test cohort 目录无效: {test_dir}")
        test_bank = build_test_znorm(
            encoder,
            Path(test_dir),
            max_files=test_max_files,
            seed=seed,
            eps=eps,
            strategy="flat",
        )
    return ScoreNormalizer(mode, enroll_bank=enroll_bank, test_bank=test_bank)


def apply_cosine_raw(a: np.ndarray, b: np.ndarray) -> float:
    return cosine_sim(a, b)

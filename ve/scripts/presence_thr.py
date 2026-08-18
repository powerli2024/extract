#!/usr/bin/env python3
"""Presence 阈值配置：单 thr 或按 KWS 语言分 thr（zh/en）。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def lang_of(wake: str | None, fallback: str | None = None) -> str:
    """与 build_manifest.lang_of 一致：有汉字→zh，否则偏 en。"""
    if fallback in ("zh", "en"):
        return fallback
    w = (wake or "").strip()
    if not w:
        return "zh"
    if CJK_RE.search(w):
        return "zh"
    compact = w.lower().replace(" ", "")
    if compact in ("hicolmo", "hicolmo.") or re.search(r"[a-zA-Z]", w):
        return "en"
    return "zh"


def load_thr_file(path: Path | None, default: float = 0.25) -> tuple[float, dict[str, Any]]:
    """读 recommended_thr.json。返回 (default_thr, meta)。"""
    if path is None or not Path(path).is_file():
        return float(default), {}
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    thr = default
    if "presence_thr" in obj:
        thr = float(obj["presence_thr"])
    elif "recommended" in obj and isinstance(obj["recommended"], dict):
        if "thr" in obj["recommended"]:
            thr = float(obj["recommended"]["thr"])
    return thr, obj


def thr_for_sample(
    sample: dict[str, Any],
    *,
    default_thr: float,
    thr_meta: dict[str, Any] | None = None,
) -> float:
    """按样本语言取 thr；无分语言表则用 default_thr。"""
    meta = thr_meta or {}
    by = meta.get("thr_by_lang") or meta.get("presence_thr_by_lang")
    if not isinstance(by, dict) or not by:
        return float(default_thr)
    lang = lang_of(sample.get("wake_text"), sample.get("lang"))
    if lang in by and by[lang] is not None:
        return float(by[lang])
    if "default" in by and by["default"] is not None:
        return float(by["default"])
    return float(default_thr)


def build_lang_split_recommendation(
    detail: list[dict[str, Any]],
    *,
    samples: list[dict[str, Any]],
    sweep_fn,
    target_frr: float,
    select_by: str,
) -> dict[str, Any]:
    """对 detail（含 score）按语言分别扫 thr，并汇总池化 contest。

    sweep_fn(scores: list[tuple[label, score]], ...) -> calibration dict with recommended
    """
    uid_lang = {
        str(s["uid"]): lang_of(s.get("wake_text"), s.get("lang")) for s in samples
    }
    buckets: dict[str, list[tuple[str, float]]] = {"zh": [], "en": []}
    for row in detail:
        uid = str(row.get("uid") or "")
        lang = uid_lang.get(uid) or lang_of(row.get("wake_text"), row.get("lang"))
        if lang not in buckets:
            lang = "zh"
        label = str(row.get("label") or "")
        score = float(row.get("presence_score", row.get("score", 0.0)))
        buckets[lang].append((label, score))

    by_lang: dict[str, Any] = {}
    thr_by_lang: dict[str, float] = {}
    n_fr = n_fa = n_pos = n_neg = 0
    for lang, scored in buckets.items():
        n_p = sum(1 for lab, _ in scored if lab == "present")
        n_a = sum(1 for lab, _ in scored if lab == "absent")
        if n_p < 5 or n_a < 3:
            print(
                f"[WARN] lang={lang} 样本不足 pos={n_p} neg={n_a}，跳过分 thr",
                flush=True,
            )
            continue
        cal = sweep_fn(scored, target_frr=target_frr, select_by=select_by)
        rec = cal["recommended"]
        by_lang[lang] = {
            "thr": rec["thr"],
            "rr": rec["rr"],
            "frr": rec["frr"],
            "far": rec["far"],
            "contest_score": rec["contest_score"],
            "n_present": cal["n_present"],
            "n_absent": cal["n_absent"],
        }
        thr_by_lang[lang] = float(rec["thr"])
        # pool errors at lang-specific thr
        for lab, s in scored:
            if lab == "present":
                n_pos += 1
                if s < rec["thr"]:
                    n_fr += 1
            else:
                n_neg += 1
                if s >= rec["thr"]:
                    n_fa += 1

    if not thr_by_lang:
        return {"ok": False, "reason": "no_lang_bucket"}

    frr = n_fr / max(1, n_pos)
    far = n_fa / max(1, n_neg)
    rr = 1.0 - far
    contest = 0.5 * rr + 0.5 * (1.0 - frr)
    # fallback default = zh if present else first
    default_thr = thr_by_lang.get("zh") or next(iter(thr_by_lang.values()))
    thr_by_lang["default"] = float(default_thr)
    return {
        "ok": True,
        "thr_mode": "lang_split",
        "thr_by_lang": thr_by_lang,
        "by_lang": by_lang,
        "pooled": {
            "rr": rr,
            "frr": frr,
            "far": far,
            "cer": frr,
            "contest_score": contest,
            "n_present": n_pos,
            "n_absent": n_neg,
            "n_fr": n_fr,
            "n_fa": n_fa,
        },
        "presence_thr": float(default_thr),
    }

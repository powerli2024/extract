#!/usr/bin/env python3
"""统一拼音/字符 CER（CJK→无调拼音；英文→字符）。自包含，不依赖其他实验包。"""

from __future__ import annotations

import re
import string
import unicodedata
from typing import Any

import editdistance
from pypinyin import Style, lazy_pinyin

_PUNCT = str.maketrans("", "", string.punctuation + "，。！？、；：""''「」『』（）【】《》…·—–-")
_CJK = re.compile(r"[\u4e00-\u9fff]")


def normalize_chars(text: str | None) -> str:
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = "".join(c for c in t if not c.isspace())
    return t.translate(_PUNCT).lower().strip()


def to_pinyin_str(text: str) -> str:
    t = normalize_chars(text)
    if not t:
        return ""
    return "".join(lazy_pinyin(t, style=Style.NORMAL, errors=lambda x: list(x.lower())))


def has_cjk(text: str) -> bool:
    return bool(_CJK.search(text or ""))


def cer_value(hyp: str, ref: str, *, use_pinyin: bool) -> float:
    if use_pinyin:
        h, r = to_pinyin_str(hyp), to_pinyin_str(ref)
    else:
        h, r = normalize_chars(hyp), normalize_chars(ref)
    if not r:
        return 0.0 if not h else 1.0
    return int(editdistance.eval(r, h)) / len(r)


def score_streams(hyps: dict[str, str], wake: str) -> dict[str, dict[str, Any]]:
    use_py = has_cjk(wake)
    out: dict[str, dict[str, Any]] = {}
    for name, hyp in hyps.items():
        hyp = hyp or ""
        out[name] = {
            "hyp": hyp,
            "cer": cer_value(hyp, wake, use_pinyin=use_py),
            "cer_char": cer_value(hyp, wake, use_pinyin=False),
            "cer_py": cer_value(hyp, wake, use_pinyin=True),
            "metric": "pinyin" if use_py else "char",
        }
    return out


def oracle_of(cers: dict[str, dict]) -> tuple[str, float]:
    name = min(
        cers,
        key=lambda k: (cers[k]["cer"], 0 if k == "original" else 1, k),
    )
    return name, float(cers[name]["cer"])


def pack_streams(cers: dict[str, dict]) -> dict[str, dict]:
    return {
        k: {
            "hyp": v["hyp"],
            "cer": round(float(v["cer"]), 4),
            "cer_char": round(float(v["cer_char"]), 4),
            "cer_py": round(float(v["cer_py"]), 4),
        }
        for k, v in cers.items()
    }

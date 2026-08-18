#!/usr/bin/env python3
"""
字符错误率 CER（Character Error Rate）标准计算。

公式:
  CER = (S + I + D) / N
  - S: Substitution 替换
  - I: Insertion 插入
  - D: Deletion 删除
  - N: 参考文本（Ground Truth）字符数

实现: Levenshtein 编辑距离（editdistance 库）
  编辑距离 = S+I+D 的最少步数之和
  CER = editdistance(GT, Hyp) / len(GT)

与 WER 区别:
  - CER 按单个字符（中文 ASR / OCR 通用）
  - WER 按单词（英文）

举例:
  GT  = 我爱中国  (N=4)
  Hyp = 我很国    → 插入「很」+ 删除「中」→ 距离 2
  CER = 2/4 = 0.5
"""

from __future__ import annotations

import string
import unicodedata
from typing import Any

import editdistance

# 去掉空白与常见标点后再按字符计 CER
_PUNCT_TABLE = str.maketrans("", "", string.punctuation + "，。！？、；：""''「」『』（）【】《》…·—–-")


def normalize_for_cer(text: str | None, *, lower: bool = True) -> str:
    """
    归一化文本再计 CER:
      NFKC → 去空白/标点 → 可选小写
    """
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = "".join(ch for ch in t if not ch.isspace())
    t = t.translate(_PUNCT_TABLE)
    if lower:
        t = t.lower()
    return t.strip()


def edit_distance(hyp: str, ref: str) -> int:
    """Levenshtein 编辑距离（字符级）。"""
    return int(editdistance.eval(ref, hyp))


def cer_counts(hyp: str, ref: str, *, normalize: bool = True) -> dict[str, Any]:
    """
    返回单条 CER 明细。
    注: editdistance 只给总距离，不拆分 S/I/D；
        总距离语义上等于最少 (S+I+D)。
    """
    hyp_n = normalize_for_cer(hyp) if normalize else (hyp or "")
    ref_n = normalize_for_cer(ref) if normalize else (ref or "")
    n = len(ref_n)
    if n == 0:
        dist = 0 if not hyp_n else len(hyp_n)
        cer = 0.0 if not hyp_n else 1.0
        return {
            "hyp": hyp_n,
            "ref": ref_n,
            "N": n,
            "edit_distance": dist,
            "S_I_D": dist,
            "cer": cer,
        }
    dist = edit_distance(hyp_n, ref_n)
    return {
        "hyp": hyp_n,
        "ref": ref_n,
        "N": n,
        "edit_distance": dist,
        "S_I_D": dist,
        "cer": float(dist) / float(n),
    }


def compute_cer(hyp: str, ref: str, *, normalize: bool = True) -> float:
    """单条 CER = (S+I+D)/N = edit_distance / len(GT)。"""
    return float(cer_counts(hyp, ref, normalize=normalize)["cer"])


def corpus_cer(pairs: list[tuple[str, str]], *, normalize: bool = True) -> dict[str, Any]:
    """
    语料级 CER: sum(编辑距离) / sum(GT字符数)。
    pairs: [(hyp, ref), ...]
    """
    total_dist = 0
    total_n = 0
    per: list[dict[str, Any]] = []
    for hyp, ref in pairs:
        c = cer_counts(hyp, ref, normalize=normalize)
        per.append(c)
        total_dist += int(c["edit_distance"])
        total_n += int(c["N"])
    overall = 0.0 if total_n == 0 else total_dist / total_n
    return {
        "corpus_cer": overall,
        "total_edit_distance": total_dist,
        "total_ref_chars": total_n,
        "num_utts": len(pairs),
        "mean_utt_cer": (
            sum(p["cer"] for p in per) / len(per) if per else 0.0
        ),
    }

#!/usr/bin/env python3
"""下一刀共享策略：领域 context、时长不匹配、叠话长句加拒、camp 否决。"""
from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

# Qwen3-ASR 官方 context 是热词/背景知识，不是自然语言指令（SFT 明确不 follow instruction）。
DOMAIN_CONTEXT = (
    "智能家居短指令。设备包括空调、灯、窗帘、电视、音箱。"
    "动作包括打开、关闭、调到、暂停、播放。"
    "只转写对设备说的那一句，忽略背景人声、新闻和旁白。"
)
HOTWORD_CONTEXT = (
    "Vocabulary: 科慕, COLMO, 空调, 风速, 智控温, ECO, 无风, 节能, "
    "洗烘, 冲锋衣, 盖世音雄, 防直吹"
)

ZH_THR = 0.29305
EN_THR = 0.357868
TEXT_GRAY = 0.10
TEXT_LEN = 15
CAMP_VETO_MARGIN = 0.12
CAMP_VETO_GRAY = 0.10


def _probe_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "asr_probe" / "scripts"


def classify_hyp(text: Optional[str]) -> dict[str, Any]:
    d = _probe_dir()
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    from refine_linguistic_patterns import classify_v2  # noqa: WPS433

    return classify_v2(text or "")


def thr_of_lang(lang: Optional[str]) -> float:
    return ZH_THR if (lang or "zh") == "zh" else EN_THR


def extra_reject_text(
    score: float,
    thr: float,
    hyp: Optional[str],
    *,
    gray: float = TEXT_GRAY,
    min_len: int = TEXT_LEN,
) -> bool:
    """只加拒、不救回：过门灰区且长非任务句。"""
    if float(score) < float(thr):
        return False
    if not (0.0 <= float(score) - float(thr) <= float(gray)):
        return False
    v2 = classify_hyp(hyp)
    return int(v2.get("len") or 0) >= int(min_len) and (not bool(v2.get("task_oriented")))


def camp_veto(
    eres: float,
    camp: Optional[float],
    thr: float,
    *,
    gray: float = CAMP_VETO_GRAY,
    margin: float = CAMP_VETO_MARGIN,
) -> bool:
    """eres 已接受且灰区，camp 明显更低 → 否决。禁止用 camp 救回 FN。"""
    if camp is None:
        return False
    eres_f, camp_f, thr_f = float(eres), float(camp), float(thr)
    if eres_f < thr_f:
        return False
    if not (0.0 <= eres_f - thr_f <= float(gray)):
        return False
    return camp_f < eres_f - float(margin)


def window_veto(
    best: Optional[float],
    second: Optional[float],
    thr: float,
    *,
    gray: float = CAMP_VETO_GRAY,
    margin: float = CAMP_VETO_MARGIN,
) -> bool:
    """灰区接受且次优窗明显低于最优窗对应的确认（次优 < 最优 - margin）。

    次优接近最优 ⇒ 两个时段都像目标，更像叠话 FA，留给文本加拒。
    次优明显更低 ⇒ 第二路不确认在场，与 camp 否决同构。
    """
    if best is None or second is None:
        return False
    b, s, t = float(best), float(second), float(thr)
    if b < t:
        return False
    if not (0.0 <= b - t <= float(gray)):
        return False
    return s < b - float(margin)


def duration_mismatch(
    hyp: Optional[str],
    dur_sec: Optional[float],
    *,
    min_dur: float = 2.5,
) -> bool:
    """无文本全错代理：长音频却几乎没字，或字数与时长严重不符。"""
    t = "".join(ch for ch in (hyp or "") if not ch.isspace())
    n = len(t)
    if dur_sec is None or dur_sec <= 0:
        return n <= 1
    d = float(dur_sec)
    if d >= min_dur and n <= 2:
        return True
    if d >= 3.5 and n < 6 and (n / d) < 0.8:
        return True
    if n >= 22 and d < 1.2:
        return True
    return False


def cer_hist(cers: list[float]) -> dict[str, int]:
    h = {"=0": 0, "(0,0.25)": 0, "[0.25,0.5)": 0, "[0.5,1)": 0, "=1": 0}
    for c in cers:
        if c <= 0:
            h["=0"] += 1
        elif c < 0.25:
            h["(0,0.25)"] += 1
        elif c < 0.5:
            h["[0.25,0.5)"] += 1
        elif c < 1:
            h["[0.5,1)"] += 1
        else:
            h["=1"] += 1
    return h


def contest_metrics(rows: list[dict[str, Any]], pred: Callable[[dict], bool]) -> dict[str, Any]:
    pos_c: list[float] = []
    n_pos = n_fr = n_neg = n_rej = 0
    n_cer1_accept = 0
    accept_cers: list[float] = []
    for r in rows:
        rej = bool(pred(r))
        split = r.get("split") or ("pos" if r.get("label") in ("present", "pos") else "neg")
        if split == "pos":
            n_pos += 1
            if rej:
                n_fr += 1
                pos_c.append(1.0)
            else:
                c = r.get("cer")
                cf = 1.0 if c is None else float(c)
                pos_c.append(cf)
                accept_cers.append(cf)
                if cf >= 1.0:
                    n_cer1_accept += 1
        else:
            n_neg += 1
            if rej:
                n_rej += 1
    rr = n_rej / n_neg if n_neg else 0.0
    frr = n_fr / n_pos if n_pos else 1.0
    cer = sum(pos_c) / len(pos_c) if pos_c else 1.0
    return {
        "rr": rr,
        "frr": frr,
        "cer": cer,
        "contest": 0.5 * rr + 0.5 * (1.0 - cer),
        "n_fr": n_fr,
        "n_rej_neg": n_rej,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_cer1_accept": n_cer1_accept,
        "cer_accept": (sum(accept_cers) / len(accept_cers)) if accept_cers else None,
        "cer_hist_accept": cer_hist(accept_cers),
    }


def stratified_holdout(
    rows: list[dict[str, Any]],
    frac: float = 0.3,
    seed: int = 7,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by: dict[tuple, list] = defaultdict(list)
    for r in rows:
        by[(r.get("split"), r.get("lang") or "zh")].append(r)
    train, test = [], []
    for g in by.values():
        g = list(g)
        rng.shuffle(g)
        n = max(1, int(round(len(g) * frac)))
        n = min(n, max(0, len(g) - 1)) if len(g) > 1 else 0
        test.extend(g[:n])
        train.extend(g[n:])
    return train, test


def round_metrics(m: dict[str, Any], nd: int = 4) -> dict[str, Any]:
    out = dict(m)
    for k, v in m.items():
        if isinstance(v, float):
            out[k] = round(v, nd)
    return out


if __name__ == "__main__":
    assert duration_mismatch("", 3.0)
    assert duration_mismatch("开", 4.0)
    assert not duration_mismatch("打开空调", 1.5)
    assert extra_reject_text(0.30, 0.29, "势必会挤压产业经济发展空间此时")
    assert not extra_reject_text(0.30, 0.29, "打开空调")
    assert camp_veto(0.30, 0.10, 0.29, margin=0.12)
    assert not camp_veto(0.30, 0.28, 0.29, margin=0.12)
    print("lift_common ok")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VE pos 真实 CER（Qwen3-ASR-1.7B，qwen-asr 后端，与 VM 调用方式一致）。

ASR 调用（对齐 VM/scripts/asr_backend.py）:
    from qwen_asr import Qwen3ASRModel
    model = Qwen3ASRModel.from_pretrained(本地权重目录, dtype=bfloat16, device_map=cuda:0,
                                          max_inference_batch_size=..., max_new_tokens=64)
    results = model.transcribe(audio=[(wav_f32, 16000), ...], language=Chinese/English, context=wake_text)
    text = results[i].text

CER（对齐官方 cer计算方法.txt，正常字符 CER，不用拼音）:
    normalize_for_cer: NFKC → 去空白 → 去标点 → 小写 → strip
    CER = editdistance(ref, hyp) / len(ref)   ==   (S + D + I) / N

pos 总 CER（竞赛口径）：误拒样本(decision=reject)记 CER=1，接受样本用真实 CER。

产物（默认 VE_OUT/reports/asr_cer/）:
    asr_results.jsonl   逐样本：期望文本(cmd_text) vs ASR 实际文本 + S/D/I/N/CER + 对齐串
    summary.json        总 CER、直方图、按语言统计、最差 20 条
    summary.md

用法（AutoDL）:
    source .env_ve
    ./run_asr_cer.sh                                     # 全量（推荐）
    python scripts/asr_cer.py --device cuda:0 --limit 20 # 冒烟
    本地无 GPU 联调: python scripts/asr_cer.py --fake-asr perturb --limit 10
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
import time
import traceback
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Optional

try:
    from paths import default_ve_out, ensure_dir
except Exception:  # paths.py 不在 sys.path 时的兜底
    def default_ve_out() -> Path:
        env = os.environ.get("VE_OUT", "").strip()
        if env:
            return Path(env).expanduser().resolve()
        return Path("/root/autodl-tmp/ve")

    def ensure_dir(p: Path) -> Path:
        p.mkdir(parents=True, exist_ok=True)
        return p

MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
_CJK = re.compile(r"[\u4e00-\u9fff]")

# ------------------------- 文本归一化（对齐 VM cer_metrics.py） -------------------------
_PUNCT_TABLE = str.maketrans(
    "", "", string.punctuation + "，。！？、；：""''「」『』（）【】《》…·—–-"
)


def normalize_for_cer(text: Optional[str], *, lower: bool = True) -> str:
    """NFKC → 去空白 → 去标点 → 小写 → strip。正常字符 CER，不做拼音。"""
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = "".join(ch for ch in t if not ch.isspace())
    t = t.translate(_PUNCT_TABLE)
    if lower:
        t = t.lower()
    return t.strip()


def guess_language(wake_text: Optional[str]) -> Optional[str]:
    """根据唤醒文本粗判语言（Qwen3 支持 language=Chinese/English）。"""
    t = normalize_for_cer(wake_text)
    if not t:
        return None
    cjk = len(_CJK.findall(t))
    if cjk >= max(1, len(t) // 2):
        return "Chinese"
    if re.fullmatch(r"[a-z0-9]+", t):
        return "English"
    return None


# ------------------------- CER 计算（正常字符 CER） -------------------------
def levenshtein_detail(ref: str, hyp: str) -> tuple[int, int, int, str, str]:
    """DP 回溯出 (S, D, I, ref_aligned, hyp_aligned)；〔〕标差异，_ 表空位。
    S+D+I == editdistance(ref, hyp)。"""
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub = dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1)
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, sub)
    s = d = ins = 0
    ra: list[str] = []
    ha: list[str] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (
            0 if ref[i - 1] == hyp[j - 1] else 1
        ):
            if ref[i - 1] == hyp[j - 1]:
                ra.append(ref[i - 1])
                ha.append(hyp[j - 1])
            else:
                s += 1
                ra.append(f"〔{ref[i - 1]}〕")
                ha.append(f"〔{hyp[j - 1]}〕")
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            d += 1
            ra.append(f"〔{ref[i - 1]}〕")
            ha.append("_")
            i -= 1
        else:
            ins += 1
            ra.append("_")
            ha.append(f"〔{hyp[j - 1]}〕")
            j -= 1
    return s, d, ins, "".join(reversed(ra)), "".join(reversed(ha))


def _edit_distance(ref: str, hyp: str) -> int:
    """编辑距离：优先 editdistance（与 VM 一致），缺失时退化为 DP 的 S+D+I（数值相同）。"""
    try:
        import editdistance
        return int(editdistance.eval(ref, hyp))
    except ImportError:
        s, d, i, _, _ = levenshtein_detail(ref, hyp)
        return s + d + i


def compute_cer(ref: str, hyp: str) -> dict[str, Any]:
    """CER = (S+D+I)/N = edit_distance / len(ref)。"""
    n = len(ref)
    if n == 0:
        cer = 0.0 if len(hyp) == 0 else 1.0
        return {"s": 0, "d": 0, "i": len(hyp), "n": 0, "dist": len(hyp),
                "cer": cer, "ref_aligned": "_" * len(hyp), "hyp_aligned": hyp}
    dist = _edit_distance(ref, hyp)
    s, d, ins, ra, ha = levenshtein_detail(ref, hyp)
    # 官方 CER=(S+I+D)/N；插入很多时可大于 1，不能人为截断。
    cer = dist / n
    return {"s": s, "d": d, "i": ins, "n": n, "dist": dist, "cer": round(cer, 6),
            "ref_aligned": ra, "hyp_aligned": ha}

# ------------------------- Qwen3-ASR 后端（对齐 VM asr_backend.py） -------------------------

def _patch_torch_pytree_for_old_torch() -> None:
    """torch<=2.1 只有 _register_pytree_node；transformers 4.5x 需要 register_pytree_node。"""
    try:
        import torch.utils._pytree as pt

        if not hasattr(pt, "register_pytree_node") and hasattr(pt, "_register_pytree_node"):
            pt.register_pytree_node = pt._register_pytree_node  # type: ignore[attr-defined]
    except Exception:
        pass


class Qwen3ASRBackend:
    def __init__(self, model_dir: str, device: str = "cuda:0", dtype: str = "bfloat16",
                 max_new_tokens: int = 64, max_batch: int = 12):
        _patch_torch_pytree_for_old_torch()
        import torch
        from qwen_asr import Qwen3ASRModel

        self.name = Path(model_dir).name
        self.sr = 16000
        torch_dtype = {
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float16": torch.float16, "fp16": torch.float16,
            "float32": torch.float32, "fp32": torch.float32,
        }.get(dtype.lower(), torch.bfloat16)
        # 部分机上 bf16 不稳；失败时由外层捕获。优先尝试 fp16 若 env 指定
        import os

        if os.environ.get("ASR_DTYPE"):
            torch_dtype = {
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                "float16": torch.float16, "fp16": torch.float16,
                "float32": torch.float32, "fp32": torch.float32,
            }.get(os.environ["ASR_DTYPE"].lower(), torch_dtype)
        self.model = Qwen3ASRModel.from_pretrained(
            model_dir,
            dtype=torch_dtype,
            device_map=device,
            max_inference_batch_size=max_batch,
            max_new_tokens=max_new_tokens,
        )
        self._default_chunk = max_batch
        try:
            import transformers
            transformers.logging.set_verbosity_error()
        except Exception:
            pass
        try:
            gen = getattr(self.model, "model", None) or self.model
            for obj in (gen, getattr(gen, "generation_config", None),
                        getattr(self.model, "generation_config", None)):
                if obj is None:
                    continue
                eos = getattr(obj, "eos_token_id", None)
                if eos is not None and hasattr(obj, "pad_token_id"):
                    obj.pad_token_id = eos
        except Exception:
            pass

    def transcribe_many(self, wavs: list[Any], language: Optional[str] = None,
                        wake_text: Optional[str] = None) -> list[str]:
        """批量转写（与 VM 一致：language 由唤醒词推断，context/prompt 回退）。"""
        import numpy as np

        if not wavs:
            return []
        lang = language  # 不再默认按唤醒词猜语言；由调用方决定（默认 None=自动识别）
        ctx = (wake_text or "").strip() or None
        audios = [(np.asarray(w, dtype=np.float32).reshape(-1), 16000) for w in wavs]
        results = None
        last_err: Optional[Exception] = None
        for kwargs in (
            {"audio": audios, "language": lang, "context": ctx},
            {"audio": audios, "language": lang, "prompt": ctx},
            {"audio": audios, "language": lang},
        ):
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            try:
                results = self.model.transcribe(**kwargs)
                break
            except TypeError as e:
                last_err = e
                continue
        if results is None:
            raise last_err or RuntimeError("ASR transcribe failed")
        out: list[str] = []
        for i in range(len(wavs)):
            if results and i < len(results):
                text = getattr(results[i], "text", "") or ""
            else:
                text = ""
            out.append(str(text))
        return out


def fake_hyp(ref: str, mode: str) -> str:
    """冒烟测试：identity=假设转写全对；perturb=人为造 1S+1D+1I。"""
    if mode == "identity":
        return ref
    if len(ref) <= 2:
        return ref + "Q"
    return ref[0] + "X" + ref[2:max(2, len(ref) - 1)] + "Q"


# ------------------------- 主流程辅助 -------------------------
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def resolve_wav(
    ve_out: Path,
    sample: dict[str, Any],
    res: dict[str, Any],
    wav_source: str = "extracted",
) -> Optional[str]:
    """extracted=门控写出的 wav（T2 可能是 0.8s 窗）；mix=原始 CMD 整段。"""
    cands: list[str] = []
    src = (wav_source or "extracted").strip().lower()
    if src == "mix":
        for w in (sample.get("cmd_wav"), res.get("cmd_wav")):
            if w:
                cands.append(str(w))
    else:
        w = res.get("extracted_wav")
        if w:
            cands.append(str(w))
        cands.append(str(ve_out / "extracted" / sample["split"] / f"{sample['uid']}.wav"))
    for c in cands:
        if c and Path(c).is_file():
            return str(Path(c).resolve())
    return None


def resolve_asr_dir(model_dir: Optional[str]) -> Optional[str]:
    """本地 Qwen3-ASR 目录（对齐 VM local_models.resolve_asr_local）。"""
    cands: list[Path] = []
    if model_dir and str(model_dir).strip():
        cands.append(Path(str(model_dir).strip()).expanduser())
    for key in ("ASR_MODEL_DIR", "QWEN3_ASR_DIR"):
        v = os.environ.get(key, "").strip()
        if v:
            cands.append(Path(v).expanduser())
    cands += [
        Path("/root/autodl-tmp/Qwen3-ASR-1.7B"),
        Path("/root/Qwen3-ASR-1.7B"),
    ]
    for c in cands:
        try:
            if c.is_dir() and (
                (c / "config.json").is_file()
                or (c / "model.safetensors").is_file()
                or any(c.glob("*.safetensors"))
                or any(c.glob("*.bin"))
            ):
                return str(c.resolve())
        except OSError:
            continue
    return None


def base_record(sample: dict[str, Any], res: dict[str, Any], norm_ver: str) -> dict[str, Any]:
    r = res or {}
    return {
        "uid": sample["uid"], "split": sample["split"], "id": sample.get("id"),
        "label": sample.get("label"), "lang": sample.get("lang"),
        "wake_text": sample.get("wake_text"),
        "language": guess_language(sample.get("wake_text")),
        "decision": r.get("decision"),
        "presence_score": r.get("presence_score"),
        "presence_thr": r.get("presence_thr"),
        "norm_ver": norm_ver,
        "model": MODEL_ID,
    }


def fixed_record(sample: dict[str, Any], res: dict[str, Any], status: str,
                  note: str, norm_ver: str) -> dict[str, Any]:
    rec = base_record(sample, res, norm_ver)
    ref_norm = normalize_for_cer(sample.get("cmd_text"))
    rec.update({
        "wav": None, "dur_sec": None,
        "cmd_text": sample.get("cmd_text"), "ref_norm": ref_norm,
        "asr_text": None, "hyp_norm": None,
        # 误拒/处理失败按该条 CER=1，等价于错误数=N。
        "s": None, "d": len(ref_norm), "i": 0, "n": len(ref_norm),
        "edit_distance": len(ref_norm),
        "cer": 1.0,
        "ref_aligned": None, "hyp_aligned": None,
        "status": status, "error": note, "asr_ms": None,
    })
    return rec


def _asr_context(args: argparse.Namespace, sample: dict[str, Any]) -> Optional[str]:
    raw = getattr(args, "context", None)
    if raw is not None and str(raw).strip():
        return str(raw).strip()
    if getattr(args, "hotwords", False):
        from lift_common import HOTWORD_CONTEXT
        return HOTWORD_CONTEXT
    if getattr(args, "domain_context", False):
        from lift_common import DOMAIN_CONTEXT
        return DOMAIN_CONTEXT
    if getattr(args, "use_wake_context", False):
        return (sample.get("wake_text") or "").strip() or None
    return None


def _decode_tag(args: argparse.Namespace) -> str:
    lang = args.language or ("guess" if args.guess_language else "auto")
    if getattr(args, "context", None) and str(args.context).strip():
        ctx = "custom"
    elif getattr(args, "hotwords", False):
        ctx = "hotwords"
    elif getattr(args, "domain_context", False):
        ctx = "domain"
    elif getattr(args, "use_wake_context", False):
        ctx = "wake"
    else:
        ctx = "none"
    retry = "1" if getattr(args, "retry_mismatch", False) else "0"
    wav = "mix" if getattr(args, "wav_source", "extracted") == "mix" else "extracted"
    return f"vm-cer-v2|lang={lang}|ctx={ctx}|retry={retry}|wav={wav}"


def _asr_one(asr: Optional["Qwen3ASRBackend"], wav: str, language: Optional[str],
             context: Optional[str]) -> str:
    import librosa
    audio, _sr = librosa.load(wav, sr=16000)
    return asr.transcribe_many([audio], language=language, wake_text=context)[0]


def _init_asr_rec(args: argparse.Namespace, sample: dict[str, Any],
                  res: dict[str, Any], wav: str, norm_ver: str) -> dict[str, Any]:
    rec = base_record(sample, res, norm_ver)
    rec["wav"] = wav
    lang = args.language
    if lang is None and args.guess_language:
        lang = guess_language(sample.get("wake_text"))
    rec["language"] = lang
    rec["asr_context"] = (
        "hotwords" if getattr(args, "hotwords", False) else
        "domain" if getattr(args, "domain_context", False) else
        "wake" if getattr(args, "use_wake_context", False) else
        "custom" if (getattr(args, "context", None) or "").strip() else
        "none"
    )
    return rec


def _retry_mismatch(args: argparse.Namespace, sample: dict[str, Any],
                    res: dict[str, Any], wav: str, asr: "Qwen3ASRBackend",
                    rec: dict[str, Any], hyp: str, lang: Optional[str],
                    ctx: Optional[str]) -> tuple[str, str]:
    from lift_common import DOMAIN_CONTEXT, duration_mismatch

    if not getattr(args, "retry_mismatch", False):
        return hyp, rec.get("asr_pass") or "primary"
    if not duration_mismatch(hyp, rec.get("dur_sec")):
        return hyp, rec.get("asr_pass") or "primary"
    retry_note: dict[str, Any] = {"reason": "duration_mismatch", "hyp0": hyp}
    pass_name = rec.get("asr_pass") or "primary"
    already_domain = (
        (lang == "Chinese")
        and bool(getattr(args, "domain_context", False))
    )
    if not already_domain:
        hyp1 = _asr_one(asr, wav, "Chinese", DOMAIN_CONTEXT)
        retry_note["hyp1"] = hyp1
        if (hyp1 or "").strip():
            hyp = hyp1
            pass_name = "retry_decode"
    mix_wav = sample.get("cmd_wav") or res.get("cmd_wav")
    still = duration_mismatch(hyp, rec.get("dur_sec"))
    if still and mix_wav and Path(str(mix_wav)).is_file():
        mix_p = str(Path(str(mix_wav)).resolve())
        cur_p = str(Path(wav).resolve())
        if mix_p != cur_p:
            hyp2 = _asr_one(asr, mix_p, lang, ctx)
            retry_note["hyp_mix"] = hyp2
            if (hyp2 or "").strip():
                hyp = hyp2
                pass_name = "retry_mix"
    rec["retry"] = retry_note
    return hyp, pass_name


def _fill_cer(rec: dict[str, Any], sample: dict[str, Any], hyp: str,
              t1: float, pass_name: str) -> dict[str, Any]:
    rec["asr_pass"] = pass_name
    rec["asr_ms"] = round((time.time() - t1) * 1000, 1)
    ref = str(sample.get("cmd_text") or "")
    ref_norm = normalize_for_cer(ref)
    hyp_norm = normalize_for_cer(hyp)
    detail = compute_cer(ref_norm, hyp_norm)
    rec.update({
        "cmd_text": ref, "ref_norm": ref_norm,
        "asr_text": hyp, "hyp_norm": hyp_norm,
        "s": detail["s"], "d": detail["d"], "i": detail["i"], "n": detail["n"],
        "edit_distance": detail["dist"],
        "cer": detail["cer"],
        "ref_aligned": detail["ref_aligned"], "hyp_aligned": detail["hyp_aligned"],
        "status": "ok" if hyp_norm else "empty_hyp",
        "error": None,
    })
    return rec


def process_one(args: argparse.Namespace, sample: dict[str, Any],
                res: dict[str, Any], wav: str, asr: Optional[Qwen3ASRBackend],
                norm_ver: str) -> dict[str, Any]:
    rec = _init_asr_rec(args, sample, res, wav, norm_ver)
    try:
        import librosa
        rec["dur_sec"] = round(float(librosa.get_duration(path=wav)), 3)
    except Exception:
        rec["dur_sec"] = None
    t1 = time.time()
    try:
        ctx = _asr_context(args, sample)
        lang = rec.get("language")
        if args.fake_asr:
            hyp = fake_hyp(str(sample.get("cmd_text") or ""), args.fake_asr)
            return _fill_cer(rec, sample, hyp, t1, "fake")
        if asr is None:
            raise RuntimeError("ASR backend is None")
        hyp = _asr_one(asr, wav, lang, ctx)
        rec["asr_pass"] = "primary"
        hyp, pass_name = _retry_mismatch(args, sample, res, wav, asr, rec, hyp, lang, ctx)
        return _fill_cer(rec, sample, hyp, t1, pass_name)
    except Exception as e:  # noqa: BLE001
        rec.update({"status": "asr_error", "error": str(e),
                    "traceback": traceback.format_exc(limit=5),
                    "cer": 1.0,
                    "asr_ms": round((time.time() - t1) * 1000, 1)})
        return rec


def process_chunk(args: argparse.Namespace,
                  chunk: list[tuple[dict[str, Any], dict[str, Any], str]],
                  asr: Optional[Qwen3ASRBackend],
                  norm_ver: str) -> list[dict[str, Any]]:
    """同一 (language, context) 组真正一次 transcribe_many。"""
    if not chunk:
        return []
    if args.fake_asr or asr is None or len(chunk) == 1:
        return [process_one(args, s, r, w, asr, norm_ver) for s, r, w in chunk]

    from collections import defaultdict
    import librosa

    prepared: list[dict[str, Any]] = []
    groups: dict[tuple[Any, Any], list[int]] = defaultdict(list)
    for i, (sample, res, wav) in enumerate(chunk):
        rec = _init_asr_rec(args, sample, res, wav, norm_ver)
        ctx = _asr_context(args, sample)
        audio = None
        try:
            audio, _sr = librosa.load(wav, sr=16000)
            rec["dur_sec"] = round(float(len(audio)) / 16000.0, 3)
        except Exception:
            rec["dur_sec"] = None
        prepared.append({
            "sample": sample, "res": res, "wav": wav, "rec": rec,
            "lang": rec.get("language"), "ctx": ctx, "audio": audio,
        })
        groups[(rec.get("language"), ctx)].append(i)

    t1 = time.time()
    hyps: list[Optional[str]] = [None] * len(chunk)
    errors: list[Optional[str]] = [None] * len(chunk)
    try:
        for (lang, ctx), idxs in groups.items():
            audios = []
            valid: list[int] = []
            for i in idxs:
                audio = prepared[i]["audio"]
                if audio is None:
                    errors[i] = "missing_wav_audio"
                    continue
                audios.append(audio)
                valid.append(i)
            if not audios:
                continue
            texts = list(asr.transcribe_many(audios, language=lang, wake_text=ctx))
            if len(texts) < len(valid):
                texts.extend([""] * (len(valid) - len(texts)))
            for i, text in zip(valid, texts):
                hyps[i] = text
    except Exception:
        return [process_one(args, s, r, w, asr, norm_ver) for s, r, w in chunk]

    out: list[dict[str, Any]] = []
    for i, item in enumerate(prepared):
        rec = item["rec"]
        sample, res, wav = item["sample"], item["res"], item["wav"]
        if errors[i] and hyps[i] is None:
            rec.update({
                "status": "asr_error", "error": errors[i],
                "cer": 1.0,
                "asr_ms": round((time.time() - t1) * 1000, 1),
            })
            out.append(rec)
            continue
        try:
            rec["asr_pass"] = "primary"
            hyp = hyps[i] or ""
            hyp, pass_name = _retry_mismatch(
                args, sample, res, wav, asr, rec, hyp, item["lang"], item["ctx"],
            )
            out.append(_fill_cer(rec, sample, hyp, t1, pass_name))
        except Exception as e:  # noqa: BLE001
            rec.update({"status": "asr_error", "error": str(e),
                        "traceback": traceback.format_exc(limit=5),
                        "cer": 1.0,
                        "asr_ms": round((time.time() - t1) * 1000, 1)})
            out.append(rec)
    return out


# ------------------------- 统计汇总 -------------------------
def _effective_error_and_ref(record: dict[str, Any]) -> tuple[int, int]:
    """为最终字符加权 CER 返回 (errors, N)。失败记录按 CER=1 计 errors=N。"""
    ref = str(record.get("ref_norm") or normalize_for_cer(record.get("cmd_text")))
    n = int(record.get("n") or len(ref))
    if n <= 0:
        return (0, 0)
    if record.get("decision") != "accept" or record.get("status") != "ok":
        return (n, n)
    dist = record.get("edit_distance")
    if dist is not None:
        return (int(dist), n)
    return (round(float(record.get("cer") or 1.0) * n), n)


def build_summary(records: list[dict[str, Any]], rr: Optional[float],
                  meta: dict[str, Any]) -> dict[str, Any]:
    n = len(records)
    accepted = [r for r in records if r.get("decision") == "accept"]
    rejected = [r for r in records if r.get("decision") != "accept"]
    ok = [r for r in accepted if r.get("status") == "ok"]
    err = [r for r in accepted if r.get("status") != "ok"]
    cer_all = [r["cer"] for r in records]
    cer_accept = [r["cer"] for r in accepted]
    cer_ok = [r["cer"] for r in ok]

    # `cer_total` 是正式主指标：所有正样本的总错误数 / 总参考字符数。
    # 对误拒和任何失败，错误数按该条参考字符数计，因而该条 CER=1。
    total_errors, total_chars = (0, 0)
    for r in records:
        errors, chars = _effective_error_and_ref(r)
        total_errors += errors
        total_chars += chars
    total = round(total_errors / total_chars, 6) if total_chars else 0.0
    total_macro = round(sum(cer_all) / n, 6) if n else 0.0
    accepted_mean = round(sum(cer_accept) / len(cer_accept), 6) if cer_accept else None
    ok_mean = round(sum(cer_ok) / len(cer_ok), 6) if cer_ok else None

    cers_sorted = sorted(cer_ok)

    def quantile(p: float) -> Optional[float]:
        if not cers_sorted:
            return None
        idx = min(len(cers_sorted) - 1, int(p * (len(cers_sorted) - 1)))
        return cers_sorted[idx]

    hist: Counter[str] = Counter()
    for r in ok:
        c = r["cer"]
        if c == 0:
            hist["=0"] += 1
        elif c < 0.25:
            hist["(0,0.25)"] += 1
        elif c < 0.5:
            hist["[0.25,0.5)"] += 1
        elif c < 1:
            hist["[0.5,1)"] += 1
        else:
            hist["=1"] += 1

    by_lang: dict[str, dict[str, Any]] = {}
    for lang in sorted({r.get("lang") for r in records}):
        rr_ = [r for r in records if r.get("lang") == lang]
        acc = [r for r in rr_ if r.get("decision") == "accept"]
        by_lang[lang] = {
            "n": len(rr_), "n_accepted": len(acc),
            "cer_total": round(sum(r["cer"] for r in rr_) / len(rr_), 6),
            "cer_accepted": round(sum(r["cer"] for r in acc) / len(acc), 6) if acc else None,
        }

    worst = sorted(accepted, key=lambda r: -(r.get("cer") or 0.0))[:20]

    summary: dict[str, Any] = {
        "meta": meta,
        "n_pos": n,
        "n_accepted": len(accepted),
        "n_rejected_or_other": len(rejected),
        "n_asr_ok": len(ok),
        "n_asr_error_or_empty": len(err),
        "n_cer0_accepted": sum(1 for r in ok if r["cer"] == 0.0),
        "n_cer1_accepted": sum(1 for r in accepted if (r.get("cer") or 0) >= 1.0),
        "cer_total": total,
        "cer_total_macro": total_macro,
        "total_errors": total_errors,
        "total_ref_chars": total_chars,
        "cer_accepted_mean": accepted_mean,
        "cer_ok_mean": ok_mean,
        "cer_p50": quantile(0.5), "cer_p90": quantile(0.9), "cer_p95": quantile(0.95),
        "cer_histogram_accepted": dict(hist),
        "by_lang": by_lang,
        "rr": rr,
        "contest_score_new": round(0.5 * rr + 0.5 * (1 - total), 6) if rr is not None else None,
        "worst_20": [
            {"uid": r["uid"], "cer": r.get("cer"), "status": r.get("status"),
             "ref": r.get("cmd_text"), "hyp": r.get("asr_text")}
            for r in worst
        ],
    }
    return summary


def md_summary(s: dict[str, Any]) -> str:
    lines = [
        "# VE pos 真实 CER（Qwen3-ASR-1.7B，正常字符 CER）",
        "",
        f"- pos 样本: {s['n_pos']}（accept={s['n_accepted']}, 其他={s['n_rejected_or_other']}）",
        f"- ASR 成功: {s['n_asr_ok']}, 失败/空转写: {s['n_asr_error_or_empty']}",
        f"- **CER_total（官方字符加权口径，误拒=1）= {s['cer_total']}**"
        f"（errors={s['total_errors']} / chars={s['total_ref_chars']}）",
        f"- CER_total_macro（逐样本平均，仅诊断）= {s['cer_total_macro']}",
        f"- CER_accepted_mean（仅接受样本真实 CER）= {s['cer_accepted_mean']}",
        f"- CER_ok_mean（ASR 成功且非空转写）= {s['cer_ok_mean']}",
        f"- CER=0 的接受样本数: {s['n_cer0_accepted']}",
        f"- **CER=1 桶（接受样本）: {s.get('n_cer1_accepted')}**",
        f"- 分位: p50={s['cer_p50']} p90={s['cer_p90']} p95={s['cer_p95']}",
        f"- RR={s['rr']} → 新 contest_score = 0.5*RR + 0.5*(1-CER_total) = {s['contest_score_new']}",
        "",
        "## CER 直方图（接受样本）",
        "",
        "```",
        json.dumps(s["cer_histogram_accepted"], ensure_ascii=False),
        "```",
        "",
        "## 按语言",
        "",
        "```",
        json.dumps(s["by_lang"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 最差 20 条",
        "",
    ]
    for w in s["worst_20"]:
        lines.append(f"- {w['uid']} cer={w['cer']} ref={w['ref']!r} hyp={w['hyp']!r}")
    return "\n".join(lines) + "\n"


# ------------------------- 入口 -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VE pos 真实 CER（Qwen3-ASR-1.7B）")
    p.add_argument("--ve-out", type=Path, default=None, help="默认 VE_OUT 或 /root/autodl-tmp/ve")
    p.add_argument("--samples", type=Path, default=None)
    p.add_argument("--results", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None, help="默认 VE_OUT/reports/asr_cer")
    p.add_argument("--model-dir", type=Path, default=None,
                   help="Qwen3-ASR 本地目录；默认探测 ASR_MODEL_DIR/QWEN3_ASR_DIR/…/Qwen3-ASR-1.7B")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"])
    p.add_argument("--batch", type=int, default=12, help="qwen-asr max_inference_batch_size")
    p.add_argument("--max-new-tokens", type=int, default=128, help="生成上限；长命令安全余量")
    p.add_argument("--language", default=None, help="强制 Chinese/English；默认 None=让 Qwen3 自动识别")
    p.add_argument("--guess-language", action="store_true",
                   help="（VM 风格，默认关）按唤醒词推断语言；本数据集唤醒词常与命令语言不一致，默认关闭")
    p.add_argument("--use-wake-context", action="store_true",
                   help="（VM 风格，默认关）把唤醒词作为 context/prompt 传给 ASR；干净评测默认关闭")
    p.add_argument("--context", default=None,
                   help="显式 ASR context 字符串；与 --domain-context / --use-wake-context 互斥优先")
    p.add_argument("--hotwords", action="store_true",
                   help="Qwen3 热词表 context（Vocabulary: …），不是指令句")
    p.add_argument("--wav-source", choices=["extracted", "mix"], default="extracted",
                   help="ASR 音频：extracted=门控产物（T2 窗裁剪）；mix=原始 CMD 整段")
    p.add_argument("--domain-context", action="store_true",
                   help="使用智能家居领域指令 context（T1；官方更推荐 --hotwords）")
    p.add_argument("--retry-mismatch", action="store_true",
                   help="hyp 与时长严重不匹配时二次解码（Chinese+领域），再不行回退 mix")
    p.add_argument(
        "--only-uids",
        default="",
        help="只重跑逗号分隔的 pos UID；默认 resume 保留其余已有 ASR 记录",
    )
    p.add_argument("--neg-fa", action="store_true",
                   help="只转写 Presence 接受的 neg，写出 asr_results.jsonl（叠话加拒用）")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 条待ASR样本（冒烟）")
    p.add_argument("--resume", action="store_true", default=True,
                   help="跳过已存在于 asr_results.jsonl 且同口径记录（默认开启）")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--fake-asr", choices=["identity", "perturb"], default=None,
                   help="冒烟测试：不加载模型。identity=假设全对；perturb=人为造 1S+1D+1I")
    return p.parse_args()


def run_neg_fa(args: argparse.Namespace) -> int:
    """转写被接受的 neg，供叠话长句加拒。"""
    from tqdm import tqdm

    ve_out = (args.ve_out or default_ve_out()).resolve()
    samples = load_jsonl((args.samples or ve_out / "manifest" / "samples.jsonl").resolve())
    all_res = {r["uid"]: r for r in load_jsonl((args.results or ve_out / "results" / "all_results.jsonl").resolve())}
    out_dir = (args.out_dir or ve_out / "reports" / "asr_neg_fa").resolve()
    ensure_dir(out_dir)
    out_path = out_dir / "asr_results.jsonl"
    norm_ver = _decode_tag(args)
    tasks: list[tuple[dict, dict, str]] = []
    for s in samples:
        if s.get("split") != "neg":
            continue
        r = all_res.get(s["uid"]) or {}
        if r.get("decision") != "accept":
            continue
        wav = resolve_wav(ve_out, s, r, getattr(args, "wav_source", "extracted"))
        if not wav:
            cw = s.get("cmd_wav") or r.get("cmd_wav")
            if cw and Path(str(cw)).is_file():
                wav = str(Path(str(cw)).resolve())
        if wav:
            tasks.append((s, r, wav))
    if getattr(args, "limit", 0) and args.limit > 0:
        tasks = tasks[: args.limit]
    print(f"[INFO] neg FA 待 ASR={len(tasks)} → {out_path}", flush=True)
    records: dict[str, dict[str, Any]] = {}
    if args.resume and out_path.is_file():
        for row in load_jsonl(out_path):
            if row.get("uid"):
                records[row["uid"]] = row
        tasks = [t for t in tasks if t[0]["uid"] not in records]
    asr = None
    if tasks and not args.fake_asr:
        model_dir = resolve_asr_dir(str(args.model_dir) if args.model_dir else None)
        if not model_dir:
            raise SystemExit("本地未找到 Qwen3-ASR-1.7B")
        asr = Qwen3ASRBackend(
            model_dir, device=args.device, dtype=args.dtype,
            max_new_tokens=args.max_new_tokens, max_batch=max(1, int(args.batch)),
        )
    pbar = tqdm(total=len(tasks), desc="neg-fa") if tasks else None
    for i in range(0, len(tasks), max(1, int(args.batch))):
        chunk = tasks[i:i + max(1, int(args.batch))]
        for rec in process_chunk(args, chunk, asr, norm_ver):
            records[rec["uid"]] = rec
        if pbar is not None:
            pbar.update(len(chunk))
    if pbar is not None:
        pbar.close()
    ordered = [records[s["uid"]] for s, _, _ in tasks if s["uid"] in records]
    # 含 resume 的全部
    if out_path.is_file() and args.resume:
        ordered = list(records.values())
    write_jsonl(out_path, ordered)
    print(f"[OK] neg FA n={len(ordered)} → {out_path}")
    return 0


def main() -> int:
    args = parse_args()
    if getattr(args, "neg_fa", False):
        return run_neg_fa(args)
    norm_ver = _decode_tag(args)

    ve_out = (args.ve_out or default_ve_out()).resolve()
    samples_path = (args.samples or ve_out / "manifest" / "samples.jsonl").resolve()
    results_path = (args.results or ve_out / "results" / "all_results.jsonl").resolve()
    neg_path = ve_out / "results" / "neg_results.jsonl"
    out_dir = (args.out_dir or ve_out / "reports" / "asr_cer").resolve()
    ensure_dir(out_dir)
    out_path = out_dir / "asr_results.jsonl"

    if not samples_path.is_file():
        raise SystemExit(f"找不到 samples.jsonl: {samples_path}")
    if not results_path.is_file():
        raise SystemExit(f"找不到 results: {results_path}")

    samples = load_jsonl(samples_path)
    all_res = {r["uid"]: r for r in load_jsonl(results_path)}
    pos = [s for s in samples if s.get("split") == "pos"]
    only_uids = {x.strip() for x in str(args.only_uids or "").split(",") if x.strip()}
    known_pos_uids = {str(s["uid"]) for s in pos}
    unknown_only = sorted(only_uids - known_pos_uids)
    if unknown_only:
        raise SystemExit(f"--only-uids 中不存在的 pos UID: {unknown_only}")
    neg_rows = load_jsonl(neg_path) if neg_path.is_file() else []
    n_neg = len(neg_rows)
    rr = round(sum(1 for r in neg_rows if r.get("decision") == "reject") / n_neg, 6) if n_neg else None
    thr = next((r.get("presence_thr") for r in all_res.values()
                if r.get("presence_thr") is not None), None)

    done: dict[str, dict[str, Any]] = {}
    if args.resume and out_path.is_file():
        n_stale = 0
        for r in load_jsonl(out_path):
            uid = r.get("uid")
            if not uid:
                continue
            # 定点修复时，未选 UID 原样保留，避免因 retry 配置变化而全量重跑。
            if only_uids and str(uid) not in only_uids:
                done[str(uid)] = r
                continue
            if r.get("norm_ver") != norm_ver:
                continue
            cur = all_res.get(uid) or {}
            cur_dec = cur.get("decision")
            old_dec = r.get("decision")
            # Presence/thr 重跑后 decision 常变；旧 accept CER 不能复用到新 reject，反之亦然
            if cur_dec != old_dec:
                n_stale += 1
                continue
            if cur_dec == "accept" and getattr(args, "wav_source", "extracted") != "mix":
                # 提取 wav 路径变了也重跑（mix 源不看 extracted 文件名）
                old_wav = r.get("extracted_wav") or r.get("wav")
                new_wav = cur.get("extracted_wav")
                if old_wav and new_wav and Path(str(old_wav)).name != Path(str(new_wav)).name:
                    n_stale += 1
                    continue
                if float(cur.get("presence_thr") or -1) != float(r.get("presence_thr") or -1):
                    # thr 变了但 decision 碰巧相同：仍可复用 CER（同 wav）；仅记录
                    pass
            done[uid] = r
        if done:
            print(
                f"[INFO] resume: 复用 {len(done)} 条（decision 一致）；失效 {n_stale} 条",
                flush=True,
            )

    tasks: list[tuple[dict, dict, str]] = []
    fixed: list[tuple[dict, dict, str, str]] = []
    for s in pos:
        uid = s["uid"]
        r = all_res.get(uid) or {}
        dec = r.get("decision")
        if uid in done and done[uid].get("decision") == dec:
            continue
        if dec == "accept":
            wav = resolve_wav(ve_out, s, r, getattr(args, "wav_source", "extracted"))
            if wav:
                tasks.append((s, r, wav))
                continue
            fixed.append((s, r, "missing_wav", "accept 但找不到提取 wav"))
        elif dec == "reject":
            fixed.append((s, r, "reject", "误拒样本按竞赛口径 CER=1"))
        else:
            fixed.append((s, r, dec or "missing_result", f"decision={dec}"))
    if args.limit and args.limit > 0:
        tasks = tasks[: args.limit]

    scope = f"only_uids={len(only_uids)}" if only_uids else "all_pos"
    print(f"[INFO] scope={scope} pos 总数={len(pos)} 已跳过={len(done)} 待ASR={len(tasks)} 口径固定CER=1={len(fixed)}")
    print(f"[INFO] norm_ver={norm_ver} rr={rr} thr={thr}")

    asr = None
    if tasks and not args.fake_asr:
        model_dir = resolve_asr_dir(str(args.model_dir) if args.model_dir else None)
        if model_dir:
            print(f"[INFO] 本地 Qwen3-ASR 权重: {model_dir}")
        elif os.environ.get("ASR_ALLOW_DOWNLOAD", "0").strip() in ("1", "true", "TRUE", "yes"):
            model_dir = MODEL_ID
            print(f"[WARN] 本地未找到权重，ASR_ALLOW_DOWNLOAD=1 → 在线加载 {MODEL_ID}")
        else:
            raise SystemExit(
                "本地未找到 Qwen3-ASR-1.7B 目录（运行期禁止下载，与 VM 一致）。\n"
                "请先: ./download_qwen3_asr.sh  或  export ASR_MODEL_DIR=/path/to/Qwen3-ASR-1.7B\n"
                "（或 ASR_ALLOW_DOWNLOAD=1 允许在线加载）"
            )
        try:
            asr = Qwen3ASRBackend(
                model_dir, device=args.device, dtype=args.dtype,
                max_new_tokens=args.max_new_tokens, max_batch=max(1, int(args.batch)),
            )
        except Exception as e:  # noqa: BLE001
            raise SystemExit(
                f"Qwen3-ASR 加载失败: {e}\n"
                "请确认: 1) 已运行 ./download_qwen3_asr.sh  2) 依赖: "
                "pip install -U qwen-asr editdistance soundfile librosa numpy"
            )

    records = dict(done)
    t0 = time.time()
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    pbar = tqdm(total=len(tasks), desc="asr", unit="utt") if tqdm and tasks else None
    for i in range(0, len(tasks), max(1, args.batch)):
        chunk = tasks[i:i + args.batch]
        for rec in process_chunk(args, chunk, asr, norm_ver):
            records[rec["uid"]] = rec
        if pbar is not None:
            pbar.update(len(chunk))
        if i + len(chunk) >= len(tasks) or (i + len(chunk)) % 50 == 0:
            write_jsonl(out_path, [records[s["uid"]] for s in pos if s["uid"] in records])
    if pbar is not None:
        pbar.close()

    for sample, res, status, note in fixed:
        records[sample["uid"]] = fixed_record(sample, res, status, note, norm_ver)

    ordered = [records[s["uid"]] for s in pos if s["uid"] in records]
    write_jsonl(out_path, ordered)

    meta = {
        "model": MODEL_ID, "backend": "qwen-asr", "norm_ver": norm_ver,
        "presence_thr": thr, "rr": rr, "limit": args.limit, "fake_asr": args.fake_asr,
        "dtype": args.dtype, "batch": args.batch, "max_new_tokens": args.max_new_tokens,
        "language": args.language, "domain_context": bool(args.domain_context),
        "hotwords": bool(getattr(args, "hotwords", False)),
        "wav_source": getattr(args, "wav_source", "extracted"),
        "retry_mismatch": bool(args.retry_mismatch),
        "elapsed_sec": round(time.time() - t0, 2), "ve_out": str(ve_out),
        "n_written": len(ordered),
    }
    summary = build_summary(ordered, rr, meta)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text(md_summary(summary), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] asr_results.jsonl → {out_path}（{len(ordered)} 条）")
    print(f"[OK] summary → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

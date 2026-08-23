#!/usr/bin/env python3
"""T0/T1：从 best_sep + datasetA 构建 VE 样本契约与 enrollment/cmd manifest。

样本契约:
  uid, split, id, label∈{present,absent}, enroll_wav, cmd_wav, wake_text, cmd_text, lang

约定: pos → present, neg → absent（业务标签；Presence 阈值在此标签上校准）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from paths import (
    VALID_SPLITS,
    default_best_sep,
    default_data_dir,
    default_ve_out,
    ensure_dir,
    require_best_sep,
    setup_sys_path,
)

setup_sys_path()

CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def lang_of(wake: str) -> str:
    return "zh" if CJK_RE.search(wake or "") else "en"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def resolve_best_sep_wav(best_sep: Path, rec: dict[str, Any]) -> str:
    """解析 enroll wav：优先 dest_rel（跨机可移植），再回退 dest_wav（需文件真实存在）。"""
    rel = str(rec.get("dest_rel") or "").replace("\\", "/").lstrip("/")
    if rel:
        cand = (best_sep / rel).resolve()
        if cand.is_file():
            return str(cand)
    dest = str(rec.get("dest_wav") or "")
    if dest:
        p = Path(dest)
        if p.is_file():
            return str(p.resolve())
    # 最后按 uid 约定路径 best_sep/{split}/{uid}.wav
    split = str(rec.get("split") or "")
    uid = str(rec.get("uid") or "")
    if split and uid:
        cand = (best_sep / split / f"{uid}.wav").resolve()
        if cand.is_file():
            return str(cand)
    return str((best_sep / rel).resolve()) if rel else dest


def load_best_sep_index(best_sep: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """uid → best_sep record，并返回 index 内重复 UID。"""
    idx = best_sep / "index.jsonl"
    out: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for r in load_jsonl(idx):
        uid = str(r.get("uid") or "")
        if not uid:
            continue
        if uid in out:
            duplicates.append(uid)
        # 非严格兼容仍取最后一条；严格模式会因 duplicate 失败。
        dest = resolve_best_sep_wav(best_sep, r) if r.get("ok", True) else ""
        out[uid] = {**r, "enroll_wav": dest}
    return out, duplicates


def load_dataset_split(data_dir: Path, split: str) -> list[dict[str, Any]]:
    p = data_dir / f"{split}.jsonl"
    return load_jsonl(p)


def build_items(
    *,
    data_dir: Path,
    best_sep: Path,
    splits: list[str],
    strict_best_sep: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    enroll_map, duplicate_best_sep_uid = load_best_sep_index(best_sep)
    items: list[dict[str, Any]] = []
    qc = {
        "n_dataset": 0,
        "n_ok": 0,
        "missing_enroll": [],
        "missing_cmd": [],
        "missing_best_sep_uid": [],
        "invalid_best_sep_uid": [],
        "duplicate_best_sep_uid": sorted(set(duplicate_best_sep_uid)),
        "duplicate_dataset_uid": [],
        "by_split": {},
    }

    for split in splits:
        if split not in VALID_SPLITS:
            raise SystemExit(f"非法 split={split}")
        label = "present" if split == "pos" else "absent"
        rows = load_dataset_split(data_dir, split)
        n_ok = 0
        seen_dataset_uids: set[str] = set()
        for row in rows:
            qc["n_dataset"] += 1
            sid = row.get("id")
            uid = f"{split}_{sid}"
            if uid in seen_dataset_uids:
                qc["duplicate_dataset_uid"].append(uid)
            seen_dataset_uids.add(uid)
            wake_rel = row.get("唤醒音频") or row.get("wake_audio") or ""
            cmd_rel = row.get("识别音频") or row.get("cmd_audio") or ""
            wake_text = row.get("唤醒文本") or row.get("wake_text") or ""
            cmd_text = row.get("识别文本") or row.get("cmd_text")
            enroll_rec = enroll_map.get(uid)
            enroll_wav = ""
            if enroll_rec and enroll_rec.get("ok", True):
                enroll_wav = str(enroll_rec.get("enroll_wav") or "")
            else:
                if enroll_rec:
                    qc["invalid_best_sep_uid"].append({
                        "uid": uid,
                        "reason": enroll_rec.get("error") or enroll_rec.get("reason") or "best_sep_ok_false",
                    })
                else:
                    qc["missing_best_sep_uid"].append(uid)
                # 兼容旧实验：非严格模式允许回退。正式 KWS 提纯对比必须关闭回退。
                if (not strict_best_sep) and wake_rel:
                    enroll_wav = str((data_dir / wake_rel).resolve())

            cmd_wav = str((data_dir / cmd_rel).resolve()) if cmd_rel else ""
            if not enroll_wav or not Path(enroll_wav).is_file():
                qc["missing_enroll"].append(uid)
                continue
            if not cmd_wav or not Path(cmd_wav).is_file():
                qc["missing_cmd"].append(uid)
                continue

            item = {
                "uid": uid,
                "split": split,
                "id": sid,
                "label": label,
                "kws_rel": wake_rel,
                "cmd_rel": cmd_rel,
                "enroll_wav": str(Path(enroll_wav).resolve()),
                "cmd_wav": str(Path(cmd_wav).resolve()),
                "wake_text": wake_text,
                "cmd_text": cmd_text,
                "lang": lang_of(str(wake_text)),
                "enroll_source": "best_sep" if (enroll_rec and enroll_rec.get("ok", True)) else "dataset_kws",
            }
            if enroll_rec and enroll_rec.get("ok", True):
                item["enroll_meta"] = {
                    "best_stage": enroll_rec.get("best_stage"),
                    "oracle_stream": enroll_rec.get("oracle_stream"),
                    "oracle_cer": enroll_rec.get("oracle_cer"),
                }
            items.append(item)
            n_ok += 1
        qc["by_split"][split] = {
            "n_dataset": len(rows),
            "n_ok": n_ok,
            "label": label,
        }
        qc["n_ok"] += n_ok

    return items, qc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="构建 VE enrollment/cmd manifest")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument(
        "--best-sep",
        type=Path,
        default=None,
        help="干净 KWS enroll 目录（也可用 BEST_SEP_DIR）；须含 index.jsonl 或 pos/*.wav",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="默认 VE_OUT/manifest")
    p.add_argument("--splits", default="pos,neg")
    p.add_argument(
        "--strict-best-sep", action="store_true",
        help="正式评测：每个 dataset UID 都必须有 ok=true 的 best_sep 注册波；禁止回退原始 KWS 或缩小分母。",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = (args.data_dir or default_data_dir()).resolve()
    best_sep = require_best_sep(args.best_sep or default_best_sep())
    out_dir = (args.out_dir or (default_ve_out() / "manifest")).resolve()
    ensure_dir(out_dir)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    print(f"[INFO] data_dir={data_dir}")
    print(f"[INFO] best_sep={best_sep}")
    print(f"[INFO] out_dir={out_dir}")

    items, qc = build_items(
        data_dir=data_dir,
        best_sep=best_sep,
        splits=splits,
        strict_best_sep=bool(args.strict_best_sep),
    )
    write_jsonl(out_dir / "samples.jsonl", items)

    enroll_rows = [
        {
            "uid": it["uid"],
            "split": it["split"],
            "id": it["id"],
            "enroll_path": it["enroll_wav"],
            "wake_text": it["wake_text"],
            "lang": it["lang"],
            "enroll_source": it["enroll_source"],
            "enroll_meta": it.get("enroll_meta"),
        }
        for it in items
    ]
    write_jsonl(out_dir / "enrollment_manifest.jsonl", enroll_rows)

    cmd_rows = [
        {
            "uid": it["uid"],
            "split": it["split"],
            "id": it["id"],
            "cmd_path": it["cmd_wav"],
            "cmd_text": it["cmd_text"],
            "label": it["label"],
        }
        for it in items
    ]
    write_jsonl(out_dir / "cmd_manifest.jsonl", cmd_rows)

    contract = {
        "version": 2,
        "label_rule": {"pos": "present", "neg": "absent"},
        "reject_policy": "speaker_absent_only",
        "fields": {
            "samples.jsonl": [
                "uid",
                "split",
                "id",
                "label",
                "enroll_wav",
                "cmd_wav",
                "wake_text",
                "cmd_text",
                "lang",
            ],
            "metrics": {
                "presence": "EER / FAR@FRR / FRR@FAR；优先低 FRR(present)",
                "tse_quality": "仅 present：可选 sim_enroll_tse / CER（不进拒识）",
                "e2e": "pos_accept_rate / neg_reject_rate / latency",
            },
        },
        "qc": qc,
        "strict_best_sep": bool(args.strict_best_sep),
        "n_samples": len(items),
        "data_dir": str(data_dir),
        "best_sep": str(best_sep),
        "best_sep_index_sha256": (
            hashlib.sha256((best_sep / "index.jsonl").read_bytes()).hexdigest()
            if (best_sep / "index.jsonl").is_file()
            else None
        ),
    }
    (out_dir / "contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "qc.json").write_text(
        json.dumps(qc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[OK] samples={len(items)} → {out_dir / 'samples.jsonl'}")
    print(f"[OK] enrollment_manifest → {out_dir / 'enrollment_manifest.jsonl'}")
    print(f"[OK] missing_enroll={len(qc['missing_enroll'])} missing_cmd={len(qc['missing_cmd'])}")
    strict_failures = (
        qc["missing_enroll"]
        or qc["missing_cmd"]
        or qc["missing_best_sep_uid"]
        or qc["invalid_best_sep_uid"]
        or qc["duplicate_best_sep_uid"]
        or qc["duplicate_dataset_uid"]
        or len(items) != qc["n_dataset"]
    )
    if args.strict_best_sep and strict_failures:
        print("[ERR] strict-best-sep 契约失败；已写出 qc.json，禁止使用不完整/回退注册波跑分")
        return 2
    return 0 if not qc["missing_cmd"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""阶段完成检测 / index 续跑：默认跳过已完成，支持只重跑失败。

完成 = 成功条数达到 catalog（不是“写了多少行”）。
--limit 冒烟写入的 summary.partial / limit>0 一律不算完成。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from paths import THR_NAMES, stage_dir


def _force() -> bool:
    return os.environ.get("VM_FORCE", "").strip() in ("1", "true", "TRUE", "yes")


def _skip_done() -> bool:
    v = os.environ.get("VM_SKIP_DONE", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def retry_failed() -> bool:
    """VM_RETRY_FAILED=1 或调用方显式要求：只补失败/缺失，不整阶段重跑。"""
    return os.environ.get("VM_RETRY_FAILED", "").strip() in ("1", "true", "TRUE", "yes")


def count_index_rows(index_path: Path) -> tuple[int, int]:
    """返回 (总行数, 无 error 的行数)。"""
    if not index_path.is_file():
        return 0, 0
    total = ok = 0
    with index_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            try:
                o = json.loads(line)
            except Exception:
                continue
            if not o.get("error"):
                ok += 1
    return total, ok


def load_index_by_uid(index_path: Path) -> dict[str, dict]:
    """读 index.jsonl；同一 uid 后者覆盖前者。"""
    out: dict[str, dict] = {}
    if not index_path.is_file():
        return out
    with index_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            uid = o.get("uid")
            if uid:
                out[str(uid)] = o
    return out


def is_ok_row(row: dict) -> bool:
    return bool(row) and not row.get("error")


def partition_items(
    items: list[dict],
    existing: dict[str, dict],
    *,
    only_failed: bool,
) -> tuple[list[dict], dict[str, dict]]:
    """
    返回 (待处理 items, 已成功可保留的 uid→row)。
    失败或缺失一律进 todo（不依赖 VM_RETRY_FAILED）。
    """
    keep: dict[str, dict] = {}
    todo: list[dict] = []
    for it in items:
        uid = str(it["uid"])
        prev = existing.get(uid)
        if prev is not None and is_ok_row(prev):
            keep[uid] = prev
            continue
        todo.append(it)
    if only_failed:
        pass
    return todo, keep


def write_index_merged(index_path: Path, rows_by_uid: dict[str, dict], order: list[str]) -> None:
    """按 order 写回完整 index（每 uid 一行）。"""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = index_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fw:
        for uid in order:
            row = rows_by_uid.get(uid)
            if row is None:
                continue
            fw.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(index_path)


def _summary_is_partial(obj: dict) -> bool:
    if obj.get("partial"):
        return True
    try:
        if int(obj.get("limit") or 0) > 0:
            return True
    except (TypeError, ValueError):
        return True
    return False


def _gated_complete(vm_out: Path, stage: str, split: str, summary: dict) -> bool:
    """Gated 完成：按**完整父阶段**重算子集大小，不信任 summary.n_subset（会被 --limit 污染）。"""
    if _summary_is_partial(summary):
        return False
    parent = str(summary.get("parent_stage") or "").strip()
    if not parent:
        return False
    parent_idx = stage_dir(vm_out, parent, split) / "index.jsonl"
    if not parent_idx.is_file():
        return False
    parent_ok: list[dict] = []
    with parent_idx.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get("error") or o.get("oracle_cer") is None:
                continue
            parent_ok.append(o)
    catalog_n = int(summary.get("catalog_n") or 0)
    if catalog_n > 0 and len(parent_ok) < catalog_n:
        return False
    thr_map = summary.get("thr") or {}
    root = stage_dir(vm_out, stage, split)
    by = summary.get("by_thr") or {}
    if not by:
        return False
    for name in THR_NAMES:
        try:
            thr_val = float(thr_map[name])
        except (KeyError, TypeError, ValueError):
            return False
        n_exp = sum(1 for r in parent_ok if float(r["oracle_cer"]) >= thr_val)
        info = by.get(name) or {}
        if n_exp <= 0:
            continue
        duplicate_of = str(info.get("duplicate_of") or "").strip()
        if duplicate_of:
            canonical = by.get(duplicate_of) or {}
            canonical_ip = Path(canonical.get("index") or "") if canonical.get("index") else root / f"thr_{duplicate_of}" / "index.jsonl"
            if not canonical_ip.is_file():
                canonical_ip = root / f"thr_{duplicate_of}" / "index.jsonl"
            if not canonical_ip.is_file():
                return False
            canonical_exp = sum(
                1 for r in parent_ok if float(r["oracle_cer"]) >= float(thr_map[duplicate_of])
            )
            if canonical_exp != n_exp:
                return False
            total, ok = count_index_rows(canonical_ip)
            if ok < n_exp or total > ok:
                return False
            continue
        ip = Path(info.get("index") or "") if info.get("index") else root / f"thr_{name}" / "index.jsonl"
        if not ip.is_file():
            ip = root / f"thr_{name}" / "index.jsonl"
        if not ip.is_file():
            return False
        total, ok = count_index_rows(ip)
        if ok < n_exp or total > ok:
            return False
    return True


def stage_complete(
    vm_out: Path,
    stage: str,
    split: str,
    n_expected: int,
    *,
    gated: bool = False,
) -> bool:
    """判断该 split/stage 是否已完整跑完（可跳过）。

    全量阶段：成功条数 >= n_expected 且无 error 行。
    失败行达标不算完成（不必再设 VM_RETRY_FAILED）。
    """
    if _force() or not _skip_done():
        return False
    root = stage_dir(vm_out, stage, split)
    summary_path = root / "summary.json"
    if gated:
        if not summary_path.is_file():
            return False
        try:
            obj = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return _gated_complete(vm_out, stage, split, obj)

    index_path = root / "index.jsonl"
    if not index_path.is_file() or not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        summary = {}
    if _summary_is_partial(summary):
        return False
    catalog = int(summary.get("catalog_n") or 0)
    need = catalog if catalog > 0 else int(n_expected)
    if need <= 0:
        return False
    total, ok = count_index_rows(index_path)
    if ok < need:
        return False
    if total > ok:
        return False
    return True


def skip_message(stage: str, split: str, detail: str = "") -> None:
    print(
        f"[SKIP] {split}/{stage} 已完成，跳过（不覆盖）。"
        f"强制重跑: --force；失败样本会在未完成时自动补跑。"
        f"{detail}",
        flush=True,
    )

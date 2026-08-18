#!/usr/bin/env python3
"""阶段完成检测 / index 续跑：默认跳过已完成，支持只重跑失败。"""

from __future__ import annotations

import json
import os
from pathlib import Path

from paths import stage_dir


def _force() -> bool:
    return os.environ.get("VM_FORCE", "").strip() in ("1", "true", "TRUE", "yes")


def _skip_done() -> bool:
    # 默认跳过已完成；VM_SKIP_DONE=0 关闭
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
    only_failed=True：只跑有 error / 缺失的；成功行保留。
    only_failed=False 且已有部分结果：同样跳过成功行（断点续跑）。
    """
    keep: dict[str, dict] = {}
    todo: list[dict] = []
    for it in items:
        uid = str(it["uid"])
        prev = existing.get(uid)
        if prev is not None and is_ok_row(prev):
            keep[uid] = prev
            continue
        # 失败或缺失 → 进待办（only_failed 时成功已上面跳过）
        todo.append(it)
    if only_failed:
        # 显式只修失败：与上面逻辑相同
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


def stage_complete(
    vm_out: Path,
    stage: str,
    split: str,
    n_expected: int,
    *,
    gated: bool = False,
) -> bool:
    """判断该 split/stage 是否已完整跑完（可跳过）。有失败且开启 retry 时不算完成。"""
    if _force() or not _skip_done():
        return False
    root = stage_dir(vm_out, stage, split)
    if gated:
        sp = root / "summary.json"
        if not sp.is_file():
            return False
        try:
            obj = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            return False
        by = obj.get("by_thr") or {}
        if not by:
            return False
        for name, info in by.items():
            if int(info.get("n_subset") or 0) <= 0:
                continue
            ip = Path(info.get("index") or "") if info.get("index") else root / f"thr_{name}" / "index.jsonl"
            if not ip.is_file():
                return False
        return True

    index_path = root / "index.jsonl"
    summary_path = root / "summary.json"
    if not index_path.is_file() or not summary_path.is_file():
        return False
    total, ok = count_index_rows(index_path)
    if n_expected <= 0:
        done = total > 0
    else:
        done = total >= n_expected
    if not done:
        return False
    # 有失败样本时：默认仍算「阶段跑过」可 skip；若 VM_RETRY_FAILED=1 则不算完成
    if retry_failed() and ok < total:
        return False
    if retry_failed() and n_expected > 0 and ok < n_expected:
        return False
    return True


def skip_message(stage: str, split: str, detail: str = "") -> None:
    print(
        f"[SKIP] {split}/{stage} 已完成，跳过（不覆盖）。"
        f"强制重跑: --force；只补失败: --retry-failed 或 VM_RETRY_FAILED=1。"
        f"{detail}",
        flush=True,
    )

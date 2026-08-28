"""Dependency-light gate cohort planning for discrete KWS CER values."""

from __future__ import annotations

from typing import Any

THR_NAMES = ("a", "b", "c")


def build_gate_plan(rows: list[dict], thr_map: dict[str, float]) -> dict[str, dict[str, Any]]:
    """Alias thresholds selecting the same UID cohort to the first a/b/c arm."""
    plan: dict[str, dict[str, Any]] = {}
    seen: dict[tuple[str, ...], str] = {}
    for name in THR_NAMES:
        thr = float(thr_map[name])
        subset = [row for row in rows if float(row["oracle_cer"]) >= thr]
        signature = tuple(sorted(str(row["uid"]) for row in subset))
        duplicate_of = seen.get(signature)
        if duplicate_of is None:
            seen[signature] = name
        plan[name] = {
            "thr": thr,
            "rows": subset,
            "uid_signature": signature,
            "duplicate_of": duplicate_of,
        }
    return plan

#!/usr/bin/env python3
"""Strict post-run audit: coverage, pinyin metric, gate dedup and WAV contract."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import soundfile as sf

from gate_policy import build_gate_plan
from paths import STAGE_DIRS, THR_NAMES, VALID_SPLITS, stage_dir

CJK = re.compile(r"[\u4e00-\u9fff]")


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def check_index(path: Path, *, check_wav: bool = True) -> dict[str, Any]:
    rows = load_rows(path)
    seen: set[str] = set()
    errors = metric_errors = missing_wav = duration_errors = 0
    for row in rows:
        uid = str(row.get("uid") or "")
        if not uid or uid in seen:
            errors += 1
        seen.add(uid)
        if row.get("error") or row.get("oracle_cer") is None:
            errors += 1
            continue
        expected_metric = "pinyin" if CJK.search(str(row.get("wake_text") or "")) else "char"
        if row.get("metric") != expected_metric:
            metric_errors += 1
        if check_wav:
            source = Path(str(row.get("kws_path") or ""))
            try:
                source_duration = float(sf.info(str(source)).duration)
            except Exception:
                source_duration = None
            for stream in (row.get("streams") or {}):
                tag = "peak" if stream == "original" else str(stream)
                wav = path.parent / "wav" / f"{uid}_{tag}.wav"
                if not wav.is_file():
                    missing_wav += 1
                elif source_duration is None:
                    duration_errors += 1
                else:
                    try:
                        # Resampling/codec tail rounding is allowed; content loss is not.
                        if abs(float(sf.info(str(wav)).duration) - source_duration) > 0.020:
                            duration_errors += 1
                    except Exception:
                        duration_errors += 1
    return {
        "path": str(path.resolve()), "n_rows": len(rows), "n_uid": len(seen),
        "n_errors": errors, "n_metric_errors": metric_errors,
        "n_missing_wav": missing_wav, "n_duration_errors": duration_errors,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vm-out", type=Path, required=True)
    ap.add_argument("--splits", default="pos,neg")
    ap.add_argument("--expected-uids", type=int, default=1838)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    if any(x not in VALID_SPLITS for x in splits):
        raise SystemExit("[ERR] bad split")
    report: dict[str, Any] = {
        "schema": "extract_sep_audit/v1", "vm_out": str(args.vm_out.resolve()),
        "metric_contract": "zh=pinyin_cer,en=char_cer", "splits": {}, "failures": [],
    }
    for split in splits:
        block: dict[str, Any] = {"stages": {}}
        for key in ("s1", "s2", "s3", "s4"):
            path = stage_dir(args.vm_out, key, split) / "index.jsonl"
            stat = check_index(path)
            block["stages"][key] = stat
            if not path.is_file() or stat["n_errors"] or stat["n_metric_errors"] or stat["n_missing_wav"] or stat["n_duration_errors"]:
                report["failures"].append({"split": split, "stage": key, "stat": stat})
        for key in ("s5", "s6", "s7", "s8"):
            root = stage_dir(args.vm_out, key, split)
            summary_path = root / "summary.json"
            if not summary_path.is_file():
                report["failures"].append({"split": split, "stage": key, "error": "missing_summary"})
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            parent = str(summary.get("parent_stage") or "")
            parent_rows = [r for r in load_rows(stage_dir(args.vm_out, parent, split) / "index.jsonl") if not r.get("error") and r.get("oracle_cer") is not None]
            expected_plan = build_gate_plan(parent_rows, summary.get("thr") or {})
            observed_aliases = (summary.get("gate_dedup") or {}).get("aliases") or {}
            expected_aliases = {name: item["duplicate_of"] for name, item in expected_plan.items() if item["duplicate_of"]}
            gate = {"aliases": observed_aliases, "expected_aliases": expected_aliases, "thresholds": {}}
            if observed_aliases != expected_aliases:
                report["failures"].append({"split": split, "stage": key, "error": "gate_alias_mismatch", "observed": observed_aliases, "expected": expected_aliases})
            for name in THR_NAMES:
                item = expected_plan[name]
                if item["duplicate_of"]:
                    gate["thresholds"][name] = {"duplicate_of": item["duplicate_of"], "n_expected": len(item["rows"])}
                    continue
                stat = check_index(root / f"thr_{name}" / "index.jsonl") if item["rows"] else {"n_rows": 0, "n_uid": 0, "n_errors": 0, "n_metric_errors": 0, "n_missing_wav": 0, "n_duration_errors": 0}
                stat["n_expected"] = len(item["rows"])
                gate["thresholds"][name] = stat
                if stat["n_uid"] != len(item["rows"]) or stat["n_errors"] or stat["n_metric_errors"] or stat["n_missing_wav"] or stat["n_duration_errors"]:
                    report["failures"].append({"split": split, "stage": f"{key}/thr_{name}", "stat": stat})
            block["stages"][key] = gate
        report["splits"][split] = block

    best = load_rows(args.vm_out / "best_sep" / "index.jsonl")
    best_uids = {str(row.get("uid") or "") for row in best}
    report["best_sep"] = {"n_rows": len(best), "n_uid": len(best_uids)}
    if args.expected_uids and (len(best) != args.expected_uids or len(best_uids) != args.expected_uids):
        report["failures"].append({"stage": "best_sep", "expected": args.expected_uids, "rows": len(best), "uids": len(best_uids)})
    for row in best:
        dest = args.vm_out / "best_sep" / str(row.get("dest_rel") or "")
        if not dest.is_file():
            report["failures"].append({"stage": "best_sep", "uid": row.get("uid"), "error": "missing_dest_wav"})
            if len(report["failures"]) > 100:
                break
    report["ok"] = not report["failures"]
    out = args.out or (args.vm_out / "reports" / "strict_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "best_sep": report["best_sep"], "n_failures": len(report["failures"]), "out": str(out)}, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

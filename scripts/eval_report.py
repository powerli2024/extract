#!/usr/bin/env python3
"""充分评估：分 split CER、cascade/gate 增益、pos↔neg 对照，输出 markdown+json。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paths import STAGE_DIRS, VALID_SPLITS, assert_split, default_vm_out, ensure_reports, stage_dir
from report_cer_dist import compute_dist


def _load_index(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        if o.get("error"):
            continue
        rows.append(o)
    return rows


def _cers(rows: list[dict], key: str = "oracle_cer") -> list[float]:
    return [float(r[key]) for r in rows if r.get(key) is not None]


def _pair_delta(
    base_rows: list[dict], new_rows: list[dict]
) -> dict[str, Any]:
    """同 uid 配对：new - base（负=变好）。"""
    bmap = {r["uid"]: r for r in base_rows if r.get("oracle_cer") is not None}
    nmap = {r["uid"]: r for r in new_rows if r.get("oracle_cer") is not None}
    common = sorted(set(bmap) & set(nmap))
    if not common:
        return {"n_paired": 0}
    deltas = [float(nmap[u]["oracle_cer"]) - float(bmap[u]["oracle_cer"]) for u in common]
    improved = sum(1 for d in deltas if d < -1e-9)
    worsened = sum(1 for d in deltas if d > 1e-9)
    same = len(deltas) - improved - worsened
    return {
        "n_paired": len(common),
        "mean_delta": round(sum(deltas) / len(deltas), 4),
        "mean_base": round(sum(float(bmap[u]["oracle_cer"]) for u in common) / len(common), 4),
        "mean_new": round(sum(float(nmap[u]["oracle_cer"]) for u in common) / len(common), 4),
        "n_improved": improved,
        "n_worsened": worsened,
        "n_same": same,
        "improve_rate": round(improved / len(common), 4),
    }


def _stage_stats(vm_out: Path, split: str, stage: str) -> dict[str, Any] | None:
    root = stage_dir(vm_out, stage, split)
    summary_p = root / "summary.json"
    if not summary_p.is_file():
        return None
    summary = json.loads(summary_p.read_text(encoding="utf-8"))
    out: dict[str, Any] = {
        "stage": stage,
        "dir": STAGE_DIRS.get(stage, stage),
        "summary": summary,
    }
    idx = root / "index.jsonl"
    if idx.is_file():
        rows = _load_index(idx)
        cers = _cers(rows)
        out["dist"] = compute_dist(cers) if cers else None
        out["n_scored"] = len(cers)
        out["n_rows"] = len(rows)
    elif "by_thr" in summary:
        thr_stats = {}
        for name, info in summary.get("by_thr", {}).items():
            duplicate_of = info.get("duplicate_of")
            if duplicate_of:
                thr_stats[name] = {
                    "thr": info.get("thr"),
                    "n_subset": info.get("n_subset"),
                    "duplicate_of": duplicate_of,
                    "skipped_duplicate": True,
                    "reason": info.get("reason"),
                }
                continue
            ip = Path(info.get("index", "")) if info.get("index") else root / f"thr_{name}" / "index.jsonl"
            rows = _load_index(ip)
            cers = _cers(rows)
            parent_cers = _cers(rows, "parent_oracle_cer")
            thr_stats[name] = {
                **{k: info.get(k) for k in ("thr", "n_subset", "n_ok", "n_fail", "mean_oracle_cer", "mean_parent_cer")},
                "dist": compute_dist(cers) if cers else None,
                "vs_parent": _pair_delta(
                    [{"uid": r["uid"], "oracle_cer": r["parent_oracle_cer"]} for r in rows if r.get("parent_oracle_cer") is not None],
                    rows,
                )
                if rows
                else {"n_paired": 0},
                "parent_dist": compute_dist(parent_cers) if parent_cers else None,
            }
        out["by_thr"] = thr_stats
    return out


def eval_split(vm_out: Path, split: str) -> dict[str, Any]:
    split = assert_split(split)
    stages = {}
    for key in STAGE_DIRS:
        st = _stage_stats(vm_out, split, key)
        if st:
            stages[key] = st

    deltas = {}
    # cascade vs one-pass
    for a, b, name in (("s1", "s3", "onnx_cascade_vs_full"), ("s2", "s4", "cv_cascade_vs_full")):
        ia = stage_dir(vm_out, a, split) / "index.jsonl"
        ib = stage_dir(vm_out, b, split) / "index.jsonl"
        if ia.is_file() and ib.is_file():
            deltas[name] = _pair_delta(_load_index(ia), _load_index(ib))

    # gate subset vs parent on same uids (from thr indexes)
    for g, parent, name in (
        ("s5", "s1", "s5_cv_peak_vs_s1"),
        ("s6", "s1", "s6_onnx_refine_vs_s1"),
        ("s7", "s2", "s7_onnx_refine_vs_s2"),
        ("s8", "s2", "s8_cv_refine_vs_s2"),
    ):
        if g not in stages or "by_thr" not in stages[g]:
            continue
        deltas[name] = {
            thr: stages[g]["by_thr"][thr].get("vs_parent", {})
            for thr in stages[g]["by_thr"]
        }

    isolation = {
        "split_root": str((vm_out / split).resolve()),
        "uid_prefix_ok": True,
        "note": "中间 wav 仅写在本 split 树下",
    }
    # quick uid prefix check on s1 if present
    s1i = stage_dir(vm_out, "s1", split) / "index.jsonl"
    if s1i.is_file():
        bad = [r["uid"] for r in _load_index(s1i) if not str(r.get("uid", "")).startswith(f"{split}_")]
        isolation["uid_prefix_ok"] = len(bad) == 0
        isolation["bad_uids_sample"] = bad[:5]

    return {
        "split": split,
        "stages": stages,
        "deltas": deltas,
        "isolation": isolation,
    }


def _md_dist(d: dict | None, indent: str = "") -> list[str]:
    if not d:
        return [f"{indent}- (无 CER)"]
    return [
        f"{indent}- n={d['n']} mean={d['mean']} P50={d['p50']} P75={d['p75']} P90={d['p90']}",
        f"{indent}- CER==0 rate={d['cer_eq_0_rate']} hist={d['hist']}",
    ]


def _md_delta(d: dict, indent: str = "") -> list[str]:
    if not d or d.get("n_paired", 0) == 0:
        return [f"{indent}- (无配对)"]
    return [
        f"{indent}- paired={d['n_paired']} mean {d['mean_base']}→{d['mean_new']} Δ={d['mean_delta']}",
        f"{indent}- improved={d['n_improved']} worsened={d['n_worsened']} same={d['n_same']} improve_rate={d['improve_rate']}",
    ]


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# VM 评估报告",
        "",
        f"- vm_out: `{report['vm_out']}`",
        f"- splits: {', '.join(report['splits'].keys())}",
        "",
        "## 评估方案说明",
        "",
        "1. **神谕 CER**：候选流（original / spk* / spk*_r*）最低 CER；CJK=无调拼音，英文=字符。",
        "2. **分树隔离**：`pos/` 与 `neg/` 各自持有 meta、阶段 wav、分报告；uid=`{split}_{id}`。",
        "3. **一阶 vs cascade**：同 uid 配对 ΔCER（负=变好）。",
        "4. **门控子集**：在父阶段 CER≥a/b/c（默认 P50/P75/P90）子集上再分离，报告子集内相对父阶段增益。",
        "5. **跨 split**：pos/neg 分别统计后对照 mean/分布，不合并音频。",
        "",
    ]

    for split, block in report["splits"].items():
        lines += [f"## Split `{split}`", ""]
        iso = block.get("isolation", {})
        lines += [
            f"- root: `{iso.get('split_root')}`",
            f"- uid 前缀检查: {'OK' if iso.get('uid_prefix_ok') else 'FAIL'}",
            "",
            "### 各阶段",
            "",
        ]
        for sk, st in block.get("stages", {}).items():
            lines.append(f"#### {sk} (`{st.get('dir')}`)")
            if st.get("dist"):
                lines += _md_dist(st["dist"])
            elif st.get("by_thr"):
                thr = st.get("summary", {}).get("thr", {})
                lines.append(f"- thr a/b/c = {thr}")
                for tn, ts in st["by_thr"].items():
                    if ts.get("duplicate_of"):
                        lines.append(
                            f"- **thr_{tn}** thr={ts.get('thr')} duplicates "
                            f"thr_{ts.get('duplicate_of')} (same UID cohort; not rerun)"
                        )
                        continue
                    lines.append(f"- **thr_{tn}** thr={ts.get('thr')} n_subset={ts.get('n_subset')}")
                    lines += _md_dist(ts.get("dist"), "  ")
                    lines.append("  - vs parent:")
                    lines += _md_delta(ts.get("vs_parent") or {}, "    ")
            else:
                s = st.get("summary", {})
                lines.append(f"- summary keys={list(s.keys())[:8]}")
            lines.append("")

        lines += ["### 配对增益", ""]
        for name, d in block.get("deltas", {}).items():
            lines.append(f"- **{name}**")
            if isinstance(d, dict) and "n_paired" in d:
                lines += _md_delta(d, "  ")
            else:
                for tn, td in (d or {}).items():
                    lines.append(f"  - thr_{tn}:")
                    lines += _md_delta(td or {}, "    ")
            lines.append("")

    # cross-split
    splits = list(report["splits"].keys())
    if len(splits) >= 2:
        lines += ["## Pos ↔ Neg 对照", ""]
        for stage in ("s1", "s2", "s3", "s4"):
            cells = []
            for sp in splits:
                st = report["splits"][sp]["stages"].get(stage)
                if not st or not st.get("dist"):
                    cells.append(f"{sp}=N/A")
                else:
                    d = st["dist"]
                    cells.append(f"{sp} mean={d['mean']} n={d['n']}")
            lines.append(f"- {stage}: " + " | ".join(cells))
        lines.append("")

    lines += [
        "## 建议读法",
        "",
        "- 先看 s1/s2 全量 mean 与 hist，再看 s3/s4 cascade 是否整体压 CER。",
        "- 门控实验看 **子集内** improve_rate；全量未必变好（易样本二次分离可能变差）。",
        "- pos/neg 分开解读；勿把两树 wav 合并评估。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="VM full evaluation report")
    ap.add_argument("--vm-out", type=str, default="")
    ap.add_argument("--splits", type=str, default="pos,neg")
    args = ap.parse_args()
    vm_out = Path(args.vm_out) if args.vm_out else default_vm_out()
    splits = [assert_split(s.strip()) for s in args.splits.split(",") if s.strip()]
    if not splits:
        splits = list(VALID_SPLITS)

    report = {"vm_out": str(vm_out.resolve()), "splits": {}}
    for split in splits:
        # only include if split root exists or has meta
        if not (vm_out / split).exists():
            print(f"[WARN] 跳过不存在的 split 树: {vm_out / split}")
            continue
        block = eval_split(vm_out, split)
        report["splits"][split] = block
        sp_reports = ensure_reports(vm_out, split)
        (sp_reports / "eval_report.json").write_text(
            json.dumps(block, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (sp_reports / "eval_report.md").write_text(
            render_md({"vm_out": report["vm_out"], "splits": {split: block}}),
            encoding="utf-8",
        )
        print(f"[OK] {split} eval → {sp_reports}")

    reports = ensure_reports(vm_out)
    (reports / "eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md = render_md(report)
    (reports / "eval_report.md").write_text(md, encoding="utf-8")
    print(f"[OK] global eval → {reports / 'eval_report.md'}")


if __name__ == "__main__":
    main()

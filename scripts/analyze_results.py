#!/usr/bin/env python3
"""VM 结果只读分析：完备性、CER 排行、cascade/门控增益、失败清单与补跑建议。

产出:
  $VM_OUT/reports/analysis.json
  $VM_OUT/reports/analysis.md
  终端精简摘要
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paths import (
    STAGE_DIRS,
    THR_NAMES,
    VALID_SPLITS,
    assert_split,
    default_vm_out,
    ensure_reports,
    stage_dir,
)
from report_cer_dist import compute_dist

FULL_STAGES = ("s1", "s2", "s3", "s4")
GATED_STAGES = ("s5", "s6", "s7", "s8")
STAGE_LABEL = {
    "s1": "ONNX 一阶",
    "s2": "CV 一阶",
    "s3": "ONNX cascade",
    "s4": "CV cascade",
    "s5": "s1门控→CV peak",
    "s6": "s1门控→ONNX 二阶",
    "s7": "s2门控→ONNX 二阶",
    "s8": "s2门控→CV 二阶",
}


def _load_index(path: Path, *, include_errors: bool = False) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        if o.get("error") and not include_errors:
            continue
        rows.append(o)
    return rows


def _count_ok_fail(path: Path) -> tuple[int, int, int]:
    """返回 (total, ok, fail)。"""
    rows = _load_index(path, include_errors=True)
    fail = sum(1 for r in rows if r.get("error"))
    ok = len(rows) - fail
    return len(rows), ok, fail


def _cers(rows: list[dict], key: str = "oracle_cer") -> list[float]:
    return [float(r[key]) for r in rows if r.get(key) is not None]


def _pair_delta(base_rows: list[dict], new_rows: list[dict]) -> dict[str, Any]:
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


def _fail_uids(path: Path) -> list[dict[str, str]]:
    out = []
    for r in _load_index(path, include_errors=True):
        if r.get("error"):
            out.append(
                {
                    "uid": str(r.get("uid", "")),
                    "error": str(r.get("error", ""))[:300],
                }
            )
    return out


def analyze_full_stage(vm_out: Path, split: str, stage: str) -> dict[str, Any]:
    root = stage_dir(vm_out, stage, split)
    idx = root / "index.jsonl"
    summary_p = root / "summary.json"
    status = "OK"
    if not summary_p.is_file() and not idx.is_file():
        status = "MISSING"
    total, ok, fail = _count_ok_fail(idx) if idx.is_file() else (0, 0, 0)
    if status != "MISSING" and fail > 0:
        status = "HAS_FAIL"
    rows = _load_index(idx)
    cers = _cers(rows)
    dist = compute_dist(cers) if cers else None
    fails = _fail_uids(idx) if idx.is_file() else []
    return {
        "stage": stage,
        "label": STAGE_LABEL.get(stage, stage),
        "dir": STAGE_DIRS.get(stage, stage),
        "status": status,
        "n_total": total,
        "n_ok": ok,
        "n_fail": fail,
        "dist": dist,
        "fails": fails,
        "summary_exists": summary_p.is_file(),
    }


def analyze_gated_stage(vm_out: Path, split: str, stage: str) -> dict[str, Any]:
    root = stage_dir(vm_out, stage, split)
    summary_p = root / "summary.json"
    if not summary_p.is_file() and not any((root / f"thr_{t}").exists() for t in THR_NAMES):
        return {
            "stage": stage,
            "label": STAGE_LABEL.get(stage, stage),
            "dir": STAGE_DIRS.get(stage, stage),
            "status": "MISSING",
            "by_thr": {},
            "n_fail": 0,
            "fails": [],
        }
    summary = {}
    if summary_p.is_file():
        try:
            summary = json.loads(summary_p.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
    by_thr: dict[str, Any] = {}
    all_fails: list[dict[str, str]] = []
    n_fail_sum = 0
    for name in THR_NAMES:
        sub = root / f"thr_{name}"
        idx = sub / "index.jsonl"
        info = (summary.get("by_thr") or {}).get(name) or {}
        if info.get("index"):
            idx = Path(info["index"])
        if not idx.is_file() and int(info.get("n_subset") or 0) == 0:
            by_thr[name] = {
                "thr": info.get("thr") or (summary.get("thr") or {}).get(name),
                "n_subset": 0,
                "n_ok": 0,
                "n_fail": 0,
                "status": "EMPTY",
                "dist": None,
                "vs_parent": {"n_paired": 0},
                "fails": [],
            }
            continue
        total, ok, fail = _count_ok_fail(idx) if idx.is_file() else (0, 0, 0)
        n_fail_sum += fail
        rows = _load_index(idx)
        cers = _cers(rows)
        parent_rows = [
            {"uid": r["uid"], "oracle_cer": r["parent_oracle_cer"]}
            for r in rows
            if r.get("parent_oracle_cer") is not None
        ]
        fails = _fail_uids(idx) if idx.is_file() else []
        for f in fails:
            f2 = dict(f)
            f2["thr"] = name
            all_fails.append(f2)
        st = "OK"
        if not idx.is_file() and int(info.get("n_subset") or 0) > 0:
            st = "MISSING"
        elif fail > 0:
            st = "HAS_FAIL"
        by_thr[name] = {
            "thr": info.get("thr") or (summary.get("thr") or {}).get(name),
            "n_subset": info.get("n_subset", total),
            "n_ok": ok,
            "n_fail": fail,
            "n_total": total,
            "status": st,
            "dist": compute_dist(cers) if cers else None,
            "vs_parent": _pair_delta(parent_rows, rows) if rows else {"n_paired": 0},
            "fails": fails,
        }
    status = "OK"
    if not summary_p.is_file() and not by_thr:
        status = "MISSING"
    elif n_fail_sum > 0:
        status = "HAS_FAIL"
    return {
        "stage": stage,
        "label": STAGE_LABEL.get(stage, stage),
        "dir": STAGE_DIRS.get(stage, stage),
        "status": status,
        "thr": summary.get("thr"),
        "by_thr": by_thr,
        "n_fail": n_fail_sum,
        "fails": all_fails,
    }


def analyze_split(vm_out: Path, split: str) -> dict[str, Any]:
    split = assert_split(split)
    full = {s: analyze_full_stage(vm_out, split, s) for s in FULL_STAGES}
    gated = {s: analyze_gated_stage(vm_out, split, s) for s in GATED_STAGES}

    # CER 排行 s1-s4
    ranking = []
    for s in FULL_STAGES:
        d = full[s].get("dist")
        if d:
            ranking.append(
                {
                    "stage": s,
                    "label": full[s]["label"],
                    "mean": d["mean"],
                    "p50": d["p50"],
                    "p90": d["p90"],
                    "cer_eq_0_rate": d["cer_eq_0_rate"],
                    "n": d["n"],
                }
            )
    ranking.sort(key=lambda x: (x["mean"], -x["cer_eq_0_rate"]))

    # cascade 增益
    gains: dict[str, Any] = {}
    for a, b, name in (("s1", "s3", "onnx_cascade_vs_full"), ("s2", "s4", "cv_cascade_vs_full")):
        ia = stage_dir(vm_out, a, split) / "index.jsonl"
        ib = stage_dir(vm_out, b, split) / "index.jsonl"
        if ia.is_file() and ib.is_file():
            gains[name] = _pair_delta(_load_index(ia), _load_index(ib))

    # 门控增益
    gate_gains: dict[str, Any] = {}
    best_gate = None
    for s in GATED_STAGES:
        gate_gains[s] = {}
        for tn, info in gated[s].get("by_thr", {}).items():
            vp = info.get("vs_parent") or {}
            gate_gains[s][tn] = vp
            if vp.get("n_paired", 0) > 0:
                cand = {
                    "stage": s,
                    "thr": tn,
                    "improve_rate": vp.get("improve_rate", 0),
                    "mean_delta": vp.get("mean_delta"),
                    "n_paired": vp.get("n_paired"),
                    "label": f"{s}/thr_{tn}",
                }
                if best_gate is None or cand["improve_rate"] > best_gate["improve_rate"]:
                    best_gate = cand
                elif cand["improve_rate"] == best_gate["improve_rate"] and (
                    cand.get("mean_delta") or 0
                ) < (best_gate.get("mean_delta") or 0):
                    best_gate = cand

    # 结论
    best_full = ranking[0] if ranking else None
    best_one = None
    for r in ranking:
        if r["stage"] in ("s1", "s2"):
            best_one = r
            break
    best_cascade = None
    for r in ranking:
        if r["stage"] in ("s3", "s4"):
            best_cascade = r
            break

    conclusions: list[str] = []
    if best_full:
        conclusions.append(
            f"全量最优: {best_full['stage']}({best_full['label']}) "
            f"mean_CER={best_full['mean']} (n={best_full['n']})"
        )
    if best_one and best_cascade:
        if best_cascade["mean"] < best_one["mean"] - 1e-6:
            conclusions.append(
                f"cascade 有收益: {best_cascade['stage']} mean={best_cascade['mean']} "
                f"< 一阶 {best_one['stage']} mean={best_one['mean']}"
            )
        else:
            conclusions.append(
                f"cascade 未压过最优一阶: 一阶 {best_one['stage']} mean={best_one['mean']} "
                f"vs cascade {best_cascade['stage']} mean={best_cascade['mean']}"
            )
    if best_one:
        other = "s2" if best_one["stage"] == "s1" else "s1"
        other_row = next((r for r in ranking if r["stage"] == other), None)
        if other_row:
            conclusions.append(
                f"一阶对比: {best_one['stage']} mean={best_one['mean']} "
                f"vs {other} mean={other_row['mean']}"
            )
    if best_gate and best_gate["improve_rate"] > 0:
        conclusions.append(
            f"门控最高改善率: {best_gate['label']} "
            f"improve_rate={best_gate['improve_rate']} Δ={best_gate['mean_delta']}"
        )
    elif best_gate:
        conclusions.append(
            f"门控改善有限: 最佳 {best_gate['label']} improve_rate={best_gate['improve_rate']}"
        )

    # 失败汇总 + 补跑建议
    fail_stages: list[str] = []
    for s, info in full.items():
        if info.get("n_fail", 0) > 0:
            fail_stages.append(s)
    for s, info in gated.items():
        if info.get("n_fail", 0) > 0:
            fail_stages.append(s)
    retry_cmds = [
        f"./run_stage.sh {s} --splits {split} --retry-failed --max-sep-sec 4"
        for s in fail_stages
    ]

    return {
        "split": split,
        "full": full,
        "gated": gated,
        "ranking": ranking,
        "gains": gains,
        "gate_gains": gate_gains,
        "best_gate": best_gate,
        "conclusions": conclusions,
        "fail_stages": fail_stages,
        "retry_cmds": retry_cmds,
    }


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# VM 结果分析",
        "",
        f"- vm_out: `{report['vm_out']}`",
        f"- splits: {', '.join(report['splits'].keys())}",
        "",
        "## 结论摘要",
        "",
    ]
    for sp, block in report["splits"].items():
        lines.append(f"### `{sp}`")
        for c in block.get("conclusions") or ["(无足够数据)"]:
            lines.append(f"- {c}")
        if block.get("retry_cmds"):
            lines.append("- 建议补跑:")
            for cmd in block["retry_cmds"]:
                lines.append(f"  - `{cmd}`")
        lines.append("")

    for sp, block in report["splits"].items():
        lines += [f"## Split `{sp}` — 完备性", "", "| stage | status | ok | fail | total/notes |", "|---|---|---:|---:|---|"]
        for s in FULL_STAGES:
            info = block["full"][s]
            lines.append(
                f"| {s} {info['label']} | {info['status']} | {info['n_ok']} | "
                f"{info['n_fail']} | n={info['n_total']} |"
            )
        for s in GATED_STAGES:
            info = block["gated"][s]
            notes = []
            for tn, ti in (info.get("by_thr") or {}).items():
                notes.append(f"{tn}:ok={ti.get('n_ok')}/fail={ti.get('n_fail')}")
            lines.append(
                f"| {s} {info['label']} | {info['status']} | — | {info.get('n_fail', 0)} | "
                f"{'; '.join(notes) or '—'} |"
            )
        lines.append("")

        lines += [
            f"## Split `{sp}` — 全量 CER 排行 (s1–s4)",
            "",
            "| rank | stage | mean | P50 | P90 | CER==0 | n |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
        for i, r in enumerate(block.get("ranking") or [], 1):
            lines.append(
                f"| {i} | {r['stage']} {r['label']} | {r['mean']} | {r['p50']} | "
                f"{r['p90']} | {r['cer_eq_0_rate']} | {r['n']} |"
            )
        lines.append("")

        lines += [f"## Split `{sp}` — Cascade 增益", ""]
        for name, d in (block.get("gains") or {}).items():
            if not d or d.get("n_paired", 0) == 0:
                lines.append(f"- **{name}**: (无配对)")
            else:
                lines.append(
                    f"- **{name}**: paired={d['n_paired']} "
                    f"{d['mean_base']}→{d['mean_new']} Δ={d['mean_delta']} "
                    f"improve_rate={d['improve_rate']} "
                    f"(↑{d['n_improved']} ↓{d['n_worsened']} ={d['n_same']})"
                )
        lines.append("")

        lines += [f"## Split `{sp}` — 门控增益 (vs parent)", ""]
        for s in GATED_STAGES:
            lines.append(f"### {s} ({STAGE_LABEL.get(s, s)})")
            for tn in THR_NAMES:
                ti = (block["gated"][s].get("by_thr") or {}).get(tn) or {}
                vp = ti.get("vs_parent") or {}
                if vp.get("n_paired", 0) == 0:
                    lines.append(
                        f"- thr_{tn}: subset={ti.get('n_subset', 0)} (无配对/空)"
                    )
                else:
                    lines.append(
                        f"- thr_{tn} thr={ti.get('thr')}: n={vp['n_paired']} "
                        f"{vp['mean_base']}→{vp['mean_new']} Δ={vp['mean_delta']} "
                        f"improve_rate={vp['improve_rate']}"
                    )
            lines.append("")

        lines += [f"## Split `{sp}` — 失败清单", ""]
        any_fail = False
        for s in FULL_STAGES:
            fails = block["full"][s].get("fails") or []
            if not fails:
                continue
            any_fail = True
            lines.append(f"### {s} (n_fail={len(fails)})")
            for f in fails[:20]:
                lines.append(f"- `{f['uid']}`: {f['error'][:160]}")
            if len(fails) > 20:
                lines.append(f"- … 另有 {len(fails) - 20} 条，见 analysis.json")
            lines.append("")
        for s in GATED_STAGES:
            fails = block["gated"][s].get("fails") or []
            if not fails:
                continue
            any_fail = True
            lines.append(f"### {s} (n_fail={len(fails)})")
            for f in fails[:20]:
                thr = f.get("thr", "?")
                lines.append(f"- thr_{thr} `{f['uid']}`: {f['error'][:160]}")
            if len(fails) > 20:
                lines.append(f"- … 另有 {len(fails) - 20} 条，见 analysis.json")
            lines.append("")
        if not any_fail:
            lines.append("- (无失败样本)")
            lines.append("")

        if block.get("retry_cmds"):
            lines += ["### 补跑命令", ""]
            for cmd in block["retry_cmds"]:
                lines.append(f"```bash\n{cmd}\n```")
            lines.append("")

    lines += [
        "## 说明",
        "",
        "- 神谕 CER：候选流最低 CER；CJK=无调拼音，英文=字符。",
        "- ΔCER 负值表示变好；门控指标只看子集内配对。",
        "- 更细分布见 `eval_report.md`；本报告侧重完备性与决策摘要。",
        "",
    ]
    return "\n".join(lines)


def print_stdout(report: dict[str, Any]) -> None:
    print("=" * 60)
    print(" VM analysis")
    print(f" vm_out={report['vm_out']}")
    print("=" * 60)
    for sp, block in report["splits"].items():
        print(f"\n[{sp}] conclusions:")
        for c in block.get("conclusions") or ["(no data)"]:
            print(f"  - {c}")
        print(f"\n[{sp}] completeness (ok/fail):")
        for s in FULL_STAGES:
            info = block["full"][s]
            print(
                f"  {s:4} {info['status']:10} ok={info['n_ok']:<5} fail={info['n_fail']:<4} "
                f"n={info['n_total']}"
            )
        for s in GATED_STAGES:
            info = block["gated"][s]
            bits = []
            for tn in THR_NAMES:
                ti = (info.get("by_thr") or {}).get(tn) or {}
                bits.append(f"{tn}:{ti.get('n_ok', 0)}/{ti.get('n_fail', 0)}")
            print(f"  {s:4} {info['status']:10} thr[{' '.join(bits)}] fail={info.get('n_fail', 0)}")
        print(f"\n[{sp}] CER ranking s1-s4:")
        for i, r in enumerate(block.get("ranking") or [], 1):
            print(
                f"  #{i} {r['stage']} mean={r['mean']:.4f} "
                f"p50={r['p50']:.4f} p90={r['p90']:.4f} zero={r['cer_eq_0_rate']:.3f}"
            )
        if block.get("retry_cmds"):
            print(f"\n[{sp}] retry:")
            for cmd in block["retry_cmds"]:
                print(f"  {cmd}")
    print(f"\n→ {report.get('out_md', '')}")
    print(f"→ {report.get('out_json', '')}")


def main() -> None:
    ap = argparse.ArgumentParser(description="VM results analysis (read-only)")
    ap.add_argument("--vm-out", type=str, default="")
    ap.add_argument("--splits", type=str, default="pos,neg")
    args = ap.parse_args()
    vm_out = Path(args.vm_out) if args.vm_out else default_vm_out()
    splits = [assert_split(s.strip()) for s in args.splits.split(",") if s.strip()]
    if not splits:
        splits = list(VALID_SPLITS)

    report: dict[str, Any] = {"vm_out": str(vm_out.resolve()), "splits": {}}
    for split in splits:
        if not (vm_out / split).exists():
            print(f"[WARN] skip missing split tree: {vm_out / split}")
            continue
        report["splits"][split] = analyze_split(vm_out, split)

    reports = ensure_reports(vm_out)
    out_json = reports / "analysis.json"
    out_md = reports / "analysis.md"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(render_md(report), encoding="utf-8")
    report["out_json"] = str(out_json)
    report["out_md"] = str(out_md)
    print_stdout(report)


if __name__ == "__main__":
    main()

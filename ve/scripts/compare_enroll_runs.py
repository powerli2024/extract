#!/usr/bin/env python3
"""配对审计两个 VE 输出，确认切换 BEST_SEP 后注册音频/分数/决策是否真的变化。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def enroll_map(root: Path) -> dict[str, str]:
    path = root / "manifest" / "enrollment_manifest.jsonl"
    out: dict[str, str] = {}
    for row in load_jsonl(path):
        uid = str(row["uid"])
        digest = str(row.get("enroll_sha256") or "")
        if not digest:
            digest = sha256_file(Path(row["enroll_path"]))
        out[uid] = digest
    return out


def decision_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pos = [r for r in rows if r.get("split") == "pos"]
    neg = [r for r in rows if r.get("split") == "neg"]
    pos_rej = sum(r.get("decision") == "reject" for r in pos)
    neg_rej = sum(r.get("decision") == "reject" for r in neg)
    return {
        "n_pos": len(pos), "n_pos_false_reject": pos_rej,
        "frr": pos_rej / len(pos) if pos else None,
        "n_neg": len(neg), "n_neg_correct_reject": neg_rej,
        "rr": neg_rej / len(neg) if neg else None,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="比较两个 BEST_SEP 实验是否真正使用了不同注册音频")
    p.add_argument("--left", type=Path, required=True, help="第一个 VE_OUT")
    p.add_argument("--right", type=Path, required=True, help="第二个 VE_OUT")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    left, right = args.left.resolve(), args.right.resolve()
    le, re = enroll_map(left), enroll_map(right)
    lr = {str(r["uid"]): r for r in load_jsonl(left / "results" / "all_results.jsonl")}
    rr = {str(r["uid"]): r for r in load_jsonl(right / "results" / "all_results.jsonl")}
    common = sorted(set(le) & set(re) & set(lr) & set(rr))
    same_audio = sum(le[u] == re[u] for u in common)
    changed_decision = [u for u in common if lr[u].get("decision") != rr[u].get("decision")]
    score_delta = [
        abs(float(lr[u].get("presence_score") or 0.0) - float(rr[u].get("presence_score") or 0.0))
        for u in common
    ]
    report = {
        "left": str(left), "right": str(right), "n_common": len(common),
        "enroll_audio_same": same_audio,
        "enroll_audio_changed": len(common) - same_audio,
        "same_audio_ratio": same_audio / len(common) if common else None,
        "presence_score_delta_mean": sum(score_delta) / len(score_delta) if score_delta else None,
        "presence_score_delta_max": max(score_delta) if score_delta else None,
        "decision_changed": len(changed_decision),
        "decision_changed_pos": sum(lr[u].get("split") == "pos" for u in changed_decision),
        "decision_changed_neg": sum(lr[u].get("split") == "neg" for u in changed_decision),
        "left_metrics": decision_metrics(list(lr.values())),
        "right_metrics": decision_metrics(list(rr.values())),
        "warnings": [],
    }
    if report["enroll_audio_changed"] == 0:
        report["warnings"].append("所有共同 UID 的注册音频字节完全相同；BEST_SEP 切换未真正改变输入。")
    elif report["decision_changed"] == 0:
        report["warnings"].append("注册音频或分数已变化，但没有样本跨越当前冻结阈值；RR/FRR 相同是离散阈值结果。")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

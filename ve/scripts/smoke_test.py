#!/usr/bin/env python3
"""本地冒烟：不加载大模型，只验证契约与报告逻辑。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from calibrate_presence import sweep_thresholds  # noqa: E402
from apply_lift_overlay import official_metrics  # noqa: E402
from asr_cer import compute_cer  # noqa: E402
from report_ve import summarize, write_run_reports  # noqa: E402


def main() -> int:
    # 官方 CER 不截断：1 个参考字符、额外插入 2 个字符 => CER=2。
    assert compute_cer("a", "abc")["cer"] == 2.0
    weighted = official_metrics([
        {"split": "pos", "cmd_text": "甲", "asr_status": "ok", "ref_chars": 1, "edit_distance": 0},
        {"split": "pos", "cmd_text": "乙丙丁戊", "asr_status": "asr_error", "ref_chars": 4},
        {"split": "neg"},
    ], lambda r: r.get("split") == "neg")
    assert weighted["cer"] == 0.8 and weighted["rr"] == 1.0
    print("official cer ok", weighted)

    scores = (
        [("present", 0.9)] * 50
        + [("present", 0.2)] * 1  # 1/51 ≈ 2% FRR at thr=0.25
        + [("absent", 0.1)] * 40
        + [("absent", 0.4)] * 5
    )
    cal = sweep_thresholds(scores, target_frr=0.02)
    assert "recommended" in cal
    print("calib ok", cal["recommended"])

    rows = []
    for i in range(10):
        rows.append(
            {
                "uid": f"pos_{i}",
                "split": "pos",
                "label": "present",
                "decision": "accept" if i < 9 else "reject",
                "presence_score": 0.5,
                "reject_reason": "speaker_absent" if i == 9 else "",
                "elapsed_ms": 100 + i,
            }
        )
    for i in range(5):
        rows.append(
            {
                "uid": f"neg_{i}",
                "split": "neg",
                "label": "absent",
                "decision": "reject" if i < 4 else "accept",
                "presence_score": 0.1 if i < 4 else 0.6,
                "reject_reason": "speaker_absent" if i < 4 else "",
                "elapsed_ms": 80,
            }
        )
    out = ROOT.parent / "ve_out" / "reports" / "smoke"
    s = write_run_reports(out, rows, meta={"smoke": True, "presence_thr": 0.25})
    assert s["audit"]["n_non_absent_reject"] == 0
    print("report ok", json.dumps(s["splits"], ensure_ascii=False))
    print("[OK] smoke_test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

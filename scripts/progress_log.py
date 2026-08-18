#!/usr/bin/env python3
"""阶段循环进度：tqdm 进度条 + 每 N 条/结束时汇总，避免逐条刷屏。"""

from __future__ import annotations

import os
import sys
from typing import Any


def log_every() -> int:
    try:
        return max(1, int(os.environ.get("VM_LOG_EVERY", "200")))
    except Exception:
        return 200


class StageProgress:
    def __init__(self, total: int, desc: str) -> None:
        self.total = max(0, int(total))
        self.desc = desc
        self.every = log_every()
        self.i = 0
        self.n_ok = 0
        self.n_fail = 0
        self.n_warn_print = 0
        self.cers: list[float] = []
        self._last_uid = ""
        try:
            from tqdm import tqdm

            self._pbar = tqdm(
                total=self.total,
                desc=desc,
                dynamic_ncols=True,
                file=sys.stderr,
                mininterval=0.5,
            )
        except Exception:
            self._pbar = None
            print(f"[START] {desc} n={self.total} log_every={self.every}", flush=True)

    def tick(
        self,
        *,
        uid: str = "",
        ok: bool = True,
        cer: float | None = None,
        err: str | None = None,
    ) -> None:
        self.i += 1
        self._last_uid = uid or self._last_uid
        if ok:
            self.n_ok += 1
            if cer is not None:
                self.cers.append(float(cer))
        else:
            self.n_fail += 1
            # 警告限流：前 5 条 + 之后每 every 条打一条
            if err and (self.n_fail <= 5 or self.i % self.every == 0):
                self.n_warn_print += 1
                msg = f"[WARN] {uid}: {err}"
                if self._pbar is not None:
                    self._pbar.write(msg)
                else:
                    print(msg, flush=True)

        mean = (sum(self.cers) / len(self.cers)) if self.cers else None
        postfix = {
            "ok": self.n_ok,
            "fail": self.n_fail,
            "mean_cer": f"{mean:.3f}" if mean is not None else "-",
        }
        if self._pbar is not None:
            self._pbar.set_postfix(postfix, refresh=False)
            self._pbar.update(1)
        elif self.i % 50 == 0 or self.i == self.total:
            print(
                f"[{self.desc}] {self.i}/{self.total} ok={self.n_ok} fail={self.n_fail}",
                flush=True,
            )

        if self.i % self.every == 0 or self.i == self.total:
            self._report_checkpoint()

    def _report_checkpoint(self) -> None:
        mean = (sum(self.cers) / len(self.cers)) if self.cers else None
        mean_s = f"{mean:.4f}" if mean is not None else "n/a"
        line = (
            f"[进度] {self.desc} {self.i}/{self.total} "
            f"ok={self.n_ok} fail={self.n_fail} "
            f"scored={len(self.cers)} mean_cer={mean_s} "
            f"last={self._last_uid}"
        )
        if self._pbar is not None:
            self._pbar.write(line)
        else:
            print(line, flush=True)

    def close(self) -> dict[str, Any]:
        if self._pbar is not None:
            self._pbar.close()
        mean = (sum(self.cers) / len(self.cers)) if self.cers else None
        return {
            "n_done": self.i,
            "n_ok": self.n_ok,
            "n_fail": self.n_fail,
            "n_scored": len(self.cers),
            "mean_oracle_cer": round(mean, 4) if mean is not None else None,
        }

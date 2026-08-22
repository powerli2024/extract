#!/usr/bin/env bash
# 从 extract 根进入 VE 下一刀。用法: ./ve.sh t1|t2|t3|t4
# sep 分支禁用：Presence / mix 只在 main。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$ROOT/.sep-only" ]]; then
  echo "[ERR] 当前是 extract@sep（只做 KWS 分离）。VE / Presence 请用旁边的 main 克隆：" >&2
  echo "  cd /root/extract && ./ve.sh $*" >&2
  echo "本目录入口: ./run_sep.sh" >&2
  exit 1
fi
cd "$ROOT/ve"
chmod +x *.sh 2>/dev/null || true
exec ./run_next_lift.sh "$@"

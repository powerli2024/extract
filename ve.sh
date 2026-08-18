#!/usr/bin/env bash
# 从 extract 根进入 VE 下一刀。用法: ./ve.sh t1|t2|t3|t4
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT/ve"
chmod +x *.sh 2>/dev/null || true
exec ./run_next_lift.sh "$@"

#!/usr/bin/env bash
# 在 ve/ 里也可 ./ve.sh t1（extract 根的 ../ve.sh 同样）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROOT/run_next_lift.sh" "$@"

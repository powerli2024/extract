#!/usr/bin/env bash
# 在 ve/ 里也可 ./ve.sh t1（extract 根的 ../ve.sh 同样）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$ROOT/../.sep-only" ]]; then
  echo "[ERR] extract@sep 不做 Presence/mix。请 git checkout main 或在 extract 根 ./run_sep.sh" >&2
  exit 1
fi
exec "$ROOT/run_next_lift.sh" "$@"

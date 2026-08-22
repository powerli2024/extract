#!/usr/bin/env bash
# sep 分支唯一默认入口：重做全部 KWS 分离实验（s1–s8）并写出 kws 交接。
# 无 MMS-FA。不要在本分支跑 ./ve.sh。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/paths_defaults.sh"
export VM_OUT="${VM_OUT:-/root/autodl-tmp/kws_sep}"
STAGES="${STAGES:-collect,s1,s2,s3,s4,s5,s6,s7,s8,compare,eval,handoff}"
echo "[INFO] extract@sep  VM_OUT=$VM_OUT  STAGES=$STAGES"
exec bash "$ROOT/run_all.sh" --stages "$STAGES" --vm-out "$VM_OUT" --no-pack "$@"

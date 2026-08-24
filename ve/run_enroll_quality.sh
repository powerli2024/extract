#!/usr/bin/env bash
# Label-free KWS enrollment quality audit. Does not read CMD audio or labels.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
[[ -f "$ROOT/.env_ve" ]] && source "$ROOT/.env_ve" || true
# shellcheck disable=SC1091
source "$ROOT/pick_python.sh"
export PYTHONPATH="$ROOT/scripts:$ROOT/../scripts:${PYTHONPATH:-}"

if [[ $# -eq 0 ]]; then
  cat <<'EOF'
用法:
  ./run_enroll_quality.sh --manifest "$VE_OUT/manifest/enrollment_manifest.jsonl" \
      --out-dir "$VE_OUT/reports/enroll_quality"
  ./run_enroll_quality.sh --audio /path/enroll.wav --out-dir /tmp/enroll_quality
  ./run_enroll_quality.sh --audio-dir /path/enrolls --backend none --out-dir /tmp/enroll_quality

默认 backend=eres2netv2，会增加声纹稳定性检查；--backend none 只跑快速信号质检。
EOF
  exit 2
fi

exec "$PYTHON_BIN" "$ROOT/scripts/enroll_quality.py" "$@"

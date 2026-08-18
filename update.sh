#!/usr/bin/env bash
# 以 GitHub origin/main 更新代码。
# 本地改过 setup_env.sh 会挡住 git pull —— 本脚本先 stash 再快进。
# 未跟踪的数据/权重不会动。丢弃本地改动: ./update.sh --hard
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

git fetch origin

if [[ "${1:-}" == "--hard" ]]; then
  echo "[WARN] git reset --hard origin/main （丢弃已跟踪文件的本地改动）"
  git reset --hard origin/main
else
  if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "[INFO] 已跟踪文件有本地改动，stash 后 pull（git stash list 可找回）"
    git stash push -m "autodl-before-pull $(date +%Y%m%d_%H%M%S)" || true
  fi
  git pull --ff-only origin main
fi

echo "[ OK ] $(git log -1 --oneline)"
echo "接着: chmod +x *.sh && ./setup_env.sh"

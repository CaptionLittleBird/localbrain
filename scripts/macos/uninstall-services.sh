#!/usr/bin/env bash
# 停止并移除 install-services.sh 注册的 launchd 服务。
set -euo pipefail

if [ "$(id -u)" -eq 0 ]; then
  echo "不要用 sudo 运行，直接 bash $0 即可。" >&2
  exit 1
fi

AGENTS="$HOME/Library/LaunchAgents"
DOMAIN="gui/$(id -u)"

for cmd in serve watch; do
  label="com.localbrain.$cmd"
  launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
  rm -f "$AGENTS/$label.plist"
  echo "已移除 $label"
done

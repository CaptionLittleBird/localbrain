#!/usr/bin/env bash
# 停止并移除 install-services.sh 注册的 launchd 服务。
set -euo pipefail

AGENTS="$HOME/Library/LaunchAgents"
DOMAIN="gui/$(id -u)"

for cmd in serve watch; do
  label="com.localbrain.$cmd"
  launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
  rm -f "$AGENTS/$label.plist"
  echo "已移除 $label"
done

#!/usr/bin/env bash
# 把 API 服务（serve）和入库监听（watch）注册成 launchd 用户服务：
# 登录后自动启动，进程挂了自动拉起。重复执行会用最新配置重启服务。
set -euo pipefail

DIR="$(cd "$(dirname "$0")/../.." && pwd)"
AGENTS="$HOME/Library/LaunchAgents"
LOGS="$HOME/Library/Logs/localbrain"
TEMPLATE="$DIR/scripts/macos/service.plist.template"
DOMAIN="gui/$(id -u)"

if [ ! -x "$DIR/.venv/bin/localbrain" ]; then
  echo "没找到 $DIR/.venv/bin/localbrain，先在项目目录运行 uv sync" >&2
  exit 1
fi
mkdir -p "$AGENTS" "$LOGS"

for cmd in serve watch; do
  label="com.localbrain.$cmd"
  plist="$AGENTS/$label.plist"
  sed -e "s#__LABEL__#$label#g" -e "s#__DIR__#$DIR#g" -e "s#__CMD__#$cmd#g" -e "s#__LOGS__#$LOGS#g" \
    "$TEMPLATE" > "$plist"
  launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
  launchctl bootstrap "$DOMAIN" "$plist"
  echo "已启动 $label，日志：$LOGS/$cmd.log"
done

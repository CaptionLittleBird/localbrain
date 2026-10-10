#!/usr/bin/env bash
# 把 API 服务（serve）和入库监听（watch）注册成 launchd 用户服务：
# 登录后自动启动，进程挂了自动拉起。重复执行会用最新代码和配置重启服务。
set -euo pipefail

if [ "$(id -u)" -eq 0 ]; then
  echo "不要用 sudo 运行：这是当前用户的登录服务，直接 bash $0 即可。" >&2
  exit 1
fi

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

# 之前用 sudo 跑过的话，日志目录或 plist 可能变成 root 所有，服务会写不进去
for path in "$LOGS" "$AGENTS"/com.localbrain.*.plist; do
  if [ -e "$path" ] && [ ! -w "$path" ]; then
    echo "$path 当前用户没有写权限（可能是之前用 sudo 运行留下的），先执行：" >&2
    echo "  sudo chown -R \"$(whoami)\" \"$LOGS\" \"$AGENTS\"/com.localbrain.*" >&2
    exit 1
  fi
done

failed=0
for cmd in serve watch; do
  label="com.localbrain.$cmd"
  plist="$AGENTS/$label.plist"
  sed -e "s#__LABEL__#$label#g" -e "s#__DIR__#$DIR#g" -e "s#__CMD__#$cmd#g" -e "s#__LOGS__#$LOGS#g" \
    "$TEMPLATE" > "$plist"

  # 已加载就先卸载。bootout 是异步的，要等旧服务真正退出，否则紧接着 bootstrap 会报 "5: Input/output error"
  if launchctl print "$DOMAIN/$label" >/dev/null 2>&1; then
    launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
    for _ in $(seq 1 20); do
      launchctl print "$DOMAIN/$label" >/dev/null 2>&1 || break
      sleep 0.5
    done
  fi

  launchctl enable "$DOMAIN/$label" 2>/dev/null || true # 防止之前被 launchctl disable 过
  if launchctl bootstrap "$DOMAIN" "$plist"; then
    echo "已启动 $label，日志：$LOGS/$cmd.log"
  else
    echo "启动 $label 失败" >&2
    failed=1
  fi
done

if [ "$failed" -ne 0 ]; then
  cat >&2 <<EOF

排查方法：
  1. 系统设置 → 通用 → 登录项与扩展，确认 localbrain 的"允许在后台"开关是打开的
  2. 稍等几秒再执行一次本脚本
  3. 查看服务状态：launchctl print $DOMAIN/com.localbrain.serve | head -30
EOF
  exit 1
fi

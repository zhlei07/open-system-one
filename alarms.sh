#!/bin/bash
# 首发闹钟。macOS 通知 + 语音提醒，到点自动弹出。
# 用法: nohup ./alarms.sh & （已由 Claude 启动）
notify () {
  osascript -e "display notification \"$2\" with title \"🚀 $1\" sound name \"Glass\"" 2>/dev/null
  say -v Ting-Ting "$3" 2>/dev/null &
}
wait_until () { until [ "$(date +%H%M)" -ge "$1" ]; do sleep 20; done; }

wait_until 1950; notify "10 分钟后首发" "打开 LAUNCH.md，确认 HN 已登录" "还有十分钟首发"
wait_until 2000; notify "第 1 步：发 Hacker News" "点 submit，然后立刻贴首评" "现在发布 HN，然后马上贴首评"
wait_until 2010; notify "第 2 步：发 X thread" "7 条，逐条发，最后一条带 HN 链接" "现在发 X 的七条线程"
wait_until 2030; notify "回帖检查" "去 HN 看有没有评论，30 分钟内回复最关键" "去检查 HN 的评论"
wait_until 2200; notify "第 3 步：发 Reddit" "r/LocalLLaMA，内容在 LAUNCH.md" "现在发 Reddit"
notify "首发流程结束" "明早发 LinkedIn 和短视频" "首发流程已完成"

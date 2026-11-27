#!/bin/bash

# --- 请修改这里的配置 ---
TARGET_PROCESS="my_target_app"  # 你要监控的进程名称
PYTHON_SCRIPT="python iii.py"   # 进程结束后要执行的命令
CHECK_INTERVAL=5                # 检查间隔（秒）
# -------------------------

echo "正在监控进程: '$TARGET_PROCESS'..."
echo "当它结束后，将执行: '$PYTHON_SCRIPT'"

# 使用一个 while 循环来检查进程是否存在
# pgrep -x "$TARGET_PROCESS" > /dev/null 会在找到进程时返回0，找不到时返回1
# -x 参数表示精确匹配进程名，防止 "my_app" 匹配到 "my_app_server"
while pgrep -x "$TARGET_PROCESS" > /dev/null; do
    # 进程仍在运行，等待一段时间
    sleep $CHECK_INTERVAL
done

echo "检测到进程 '$TARGET_PROCESS' 已结束。"
echo "正在启动脚本..."

# 执行你的 Python 脚本
$PYTHON_SCRIPT

echo "脚本执行完毕。"


while ps -p 145473 > /dev/null; do sleep 5; done && python iii.py
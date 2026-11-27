#!/bin/bash

# 存储所有后台进程的PID
pids=()

# 捕获 Ctrl+C 信号，触发 kill_all_processes 函数
trap 'kill_all_processes' SIGINT

kill_all_processes() {
    echo "检测到 Ctrl+C，终止所有子进程..."
    # 使用 kill 发送信号，2>/dev/null 忽略不存在进程的错误
    kill "${pids[@]}" 2>/dev/null
    exit 1
}

dataset='mp3d'
experiment_name='exp6'

# --- 新增 ---
# 定义 episode_l 值的列表 (Bash 数组)
# 数组的索引将从 0 开始，与 split_l 的值完美对应
episode_l_values=(0 0 0 0 0 0 0 0 0 0 0)
cuda_values=(0 1 2 2 2 2 3 3 3 3 3)
port_values=(8000 8000 8001 8001 8002 8002 8003 8003 8004 8004 8005)

mkdir -p "exp/$dataset/$experiment_name/log"
mkdir -p "exp/$dataset/$experiment_name/observations"

# 启动多个任务，每隔30秒启动一个
for split_l in $(seq 0 1 10); do
    python -u scripts/main.py \
        experiment.split_l=${split_l} \
        experiment.split_r=$((split_l + 1)) \
        experiment.episode_l=${episode_l_values[split_l]} \
        experiment.port=${port_values[split_l]} \
        experiment.cuda_id=${cuda_values[split_l]} \
        experiment.dataset="$dataset" \
        experiment.experiment_name="$experiment_name" \
        experiment.type="experiment" \
        > "exp/$dataset/$experiment_name/log/${split_l}.log" 2>&1 &
        
    pids+=($!)
    sleep 30
done

# 等待所有后台进程完成
wait
echo "所有任务已完成。"

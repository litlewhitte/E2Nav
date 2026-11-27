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

dataset='hm3d_v2'
experiment_name='exp15'


# --- 参数设置 ---
# 起始端口号
base_port=8000
# 可用的GPU数量（4卡机器）
num_gpus=4
# 任务计数器，用于分配资源
task_index=0 

mkdir -p "exp/$dataset/$experiment_name/log"
mkdir -p "exp/$dataset/$experiment_name/observations"

# --- 启动任务循环 ---
# 第一部分：前6个任务 (split_l: 0, 3, 6, 9, 12, 15)
for split_l in $(seq 0 3 15); do
    # 使用模运算来循环分配CUDA ID (0, 1, 2, 3, 0, 1)
    cuda_id=$((task_index % num_gpus))
    
    # 为每个进程计算一个唯一的端口号
    port=$((base_port + task_index))

    echo "正在启动任务 ${task_index}: split [${split_l}, $(($split_l + 3))), 使用 CUDA:${cuda_id} 在端口 ${port}"

    python -u scripts/main.py \
        experiment.split_l=${split_l} \
        experiment.split_r=$(($split_l + 3)) \
        experiment.episode_l=0 \
        experiment.port=${port} \
        experiment.cuda_id=${cuda_id} \
        experiment.dataset="$dataset" \
        experiment.experiment_name="$experiment_name" \
        experiment.type="experiment" \
        > "exp/$dataset/$experiment_name/log/${task_index}.log" 2>&1 &
        
    # 存储新启动进程的PID
    pids+=($!)
    
    # 任务计数器加一
    task_index=$((task_index + 1))
    
    # 等待30秒再启动下一个，避免瞬间资源占用过高
    sleep 40
done

# 等待所有后台进程完成
echo "所有任务已启动，等待它们完成..."
wait
echo "所有任务已完成。"

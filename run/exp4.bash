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
experiment_name='exp6-plus-plus'
episode_l_values=(69 42 61 53 70 54 75 54 58 56 71 71)

# --- 参数设置 ---
# 起始端口号
base_port=8000
# 可用的GPU数量
num_gpus=8
# 任务计数器，用于分配资源
task_index=0

mkdir -p "exp/$dataset/$experiment_name/log"
mkdir -p "exp/$dataset/$experiment_name/observations"

# --- 启动任务循环 ---
# 循环从0到33，步长为3，以覆盖 [0, 36) 的范围
# 这将生成12个任务 (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33)
for split_l in $(seq 0 3 33); do
    # 使用模运算来循环分配CUDA ID (0, 1, ..., 7, 0, 1, ...)
    cuda_id=$((task_index % num_gpus))
    # episode_id=$((split_l % 3))
    # 为每个进程计算一个唯一的端口号
    port=$((base_port + task_index))

    echo "正在启动任务 ${task_index}: split [${split_l}, $(($split_l + 3))), 使用 CUDA:${cuda_id} 在端口 ${port}"

    python -u scripts/main.py \
        --split_l ${split_l} \
        --split_r $(($split_l + 3)) \
        --episode_l ${episode_l_values[task_index]} \
        --port ${port} \
        --cuda_id ${cuda_id} \
        --dataset "$dataset" \
        --experiment_name "$experiment_name" \
        --type "experiment" \
        > "exp/$dataset/$experiment_name/log/${task_index}.log" 2>&1 &
        
    # 存储新启动进程的PID
    pids+=($!)
    
    # 任务计数器加一
    task_index=$((task_index + 1))
    
    # 等待30秒再启动下一个，避免瞬间资源占用过高
    sleep 20
done

# 等待所有后台进程完成
echo "所有任务已启动，等待它们完成..."
wait
echo "所有任务已完成。"

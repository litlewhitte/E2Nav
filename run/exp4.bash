pids=()

trap 'kill_all_processes' SIGINT

kill_all_processes() {
    echo "Interrupt received; terminating workers..."
    kill "${pids[@]}" 2>/dev/null
    exit 1
}

dataset='hm3d_v2'
experiment_name='exp6-plus-plus'
episode_l_values=(69 42 61 53 70 54 75 54 58 56 71 71)

base_port=8000
num_gpus=8
task_index=0

mkdir -p "exp/$dataset/$experiment_name/log"
mkdir -p "exp/$dataset/$experiment_name/observations"

for split_l in $(seq 0 3 33); do
    cuda_id=$((task_index % num_gpus))
    # episode_id=$((split_l % 3))
    port=$((base_port + task_index))

    echo "Starting task ${task_index}: split [${split_l}, $(($split_l + 3))), CUDA:${cuda_id}, port ${port}"

    python -u -m scripts.main \
        experiment.split_l=${split_l} \
        experiment.split_r=$(($split_l + 3)) \
        experiment.episode_l=${episode_l_values[task_index]} \
        experiment.port=${port} \
        experiment.cuda_id=${cuda_id} \
        experiment.dataset="$dataset" \
        experiment.experiment_name="$experiment_name" \
        experiment.type="experiment" \
        > "exp/$dataset/$experiment_name/log/${task_index}.log" 2>&1 &
        
    pids+=($!)
    
    task_index=$((task_index + 1))
    
    sleep 20
done

echo "All tasks started. Waiting for workers..."
wait
echo "All tasks finished."

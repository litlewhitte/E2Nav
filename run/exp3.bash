#!/bin/bash

pids=()

trap 'kill_all_processes' SIGINT

kill_all_processes() {
    echo "Interrupt received; terminating workers..."
    kill "${pids[@]}" 2>/dev/null
    exit 1
}

dataset='hm3d_v2'
experiment_name='exp2'


base_port=8000
num_gpus=8
task_index=0 

mkdir -p "exp/$dataset/$experiment_name/log"
mkdir -p "exp/$dataset/$experiment_name/observations"

for split_l in $(seq 0 3 33); do
    cuda_id=$((task_index % num_gpus))
    
    port=$((base_port + task_index))

    echo "Starting task ${task_index}: split [${split_l}, $(($split_l + 3))), CUDA:${cuda_id}, port ${port}"

    python -u -m scripts.main \
        experiment.split_l=${split_l} \
        experiment.split_r=$(($split_l + 3)) \
        experiment.episode_l=0 \
        experiment.port=${port} \
        experiment.cuda_id=${cuda_id} \
        experiment.dataset="$dataset" \
        experiment.experiment_name="$experiment_name" \
        experiment.type="experiment" \
        > "exp/$dataset/$experiment_name/log/${task_index}.log" 2>&1 &
        
    pids+=($!)
    
    task_index=$((task_index + 1))
    
    sleep 40
done

echo "All tasks started. Waiting for workers..."
wait
echo "All tasks finished."

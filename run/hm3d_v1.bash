#!/bin/bash

pids=()

trap 'kill_all_processes' SIGINT

kill_all_processes() {
    echo "Interrupt received; terminating workers..."
    kill "${pids[@]}" 2>/dev/null
    exit 1
}

dataset='hm3d_v1'
experiment_name='exp13'

episode_l_values=(0 0 0 0 0 0 0 0 0 0 0)
cuda_values=(0 1 2 2 2 2 3 3 3 3 3)
port_values=(8000 8000 8001 8001 8002 8002 8003 8003 8004 8004 8005)

mkdir -p "exp/$dataset/$experiment_name/log"
mkdir -p "exp/$dataset/$experiment_name/observations"

for split_l in $(seq 0 1 10); do
    python -u -m scripts.main \
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

wait
echo "All tasks finished."

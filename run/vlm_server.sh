#!/usr/bin/env bash
mkdir -p temp
nohup vllm serve "${E2NAV_VLLM_MODEL:-Qwen/Qwen2.5-VL-72B-Instruct}" \
  --port 8000 \
  --host 0.0.0.0 \
  --dtype bfloat16 \
  --tensor-parallel-size 4 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.6 > temp/vllm.log 2>&1 &

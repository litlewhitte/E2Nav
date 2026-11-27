nohup vllm serve /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouqihang-240108100050/HuNav/models/Qwen2.5-VL-72B-Instruct/ \
  --port 8000 \
  --host 0.0.0.0 \
  --dtype bfloat16 \
  --tensor-parallel-size 4 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.6 > /inspire/hdd/global_user/zhouqihang-240108100050/E2Nav/temp/server_log_3/vlm_server.log 2>&1 &
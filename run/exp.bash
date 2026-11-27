# python -u scripts/main.py --port 8000 --cuda_id 0 --dataset mp3d  --split_l 10 > temp/log/mp3d_10.log 2>&1 &
# python -u scripts/main.py --port 8001 --cuda_id 1 --dataset mp3d  --split_l 8 --episode_l 35 > temp/log/mp3d_08.log 2>&1 &
# # python -u scripts/main.py --port 8002 --cuda_id 2 --dataset mp3d  --split_l 2 > temp/log/mp3d_02.log 2>&1 &
# # python -u scripts/main.py --port 8002 --cuda_id 2 --dataset mp3d  --split_l 3 > temp/log/mp3d_03.log 2>&1 &
# python -u scripts/main.py --port 8003 --cuda_id 3 --dataset mp3d  > temp/log/mp3d_00.log 2>&1 &

# # python -u scripts/main.py --port 8002 --cuda_id 2 --dataset hm3d_v2  --split_l 1  > temp/log/hm3d_01.log 2>&1 &
# # python -u scripts/main.py --port 8004 --cuda_id 4 --dataset hm3d_v2  --split_l 2  > temp/log/hm3d_02.log 2>&1 &
# # python -u scripts/main.py --port 8005 --cuda_id 5 --dataset hm3d_v2  --split_l 6  > temp/log/hm3d_06.log 2>&1 &
# # python -u scripts/main.py --port 8006 --cuda_id 6 --dataset hm3d_v2  --split_l 14 > temp/log/hm3d_14.log 2>&1 &
# # python -u scripts/main.py --port 8007 --cuda_id 7 --dataset hm3d_v2  --split_l 10 > temp/log/hm3d_10.log 2>&1 &

python -u scripts/main.py --split_l 0 --split_r 3 --episode_l 0 --port 8001 --cuda_id 1 --dataset hm3d_v2 --experiment_name exp6 --type experiment > exp/hm3d_v2/exp6/log/0.log 2>&1 &
python -u scripts/main.py --split_l 3 --split_r 6 --episode_l 0 --port 8002 --cuda_id 2 --dataset hm3d_v2 --experiment_name exp6 --type experiment > exp/hm3d_v2/exp6/log/3.log 2>&1 &
python -u scripts/main.py --split_l 6 --split_r 9 --episode_l 0 --port 8004 --cuda_id 4 --dataset hm3d_v2 --experiment_name exp6 --type experiment > exp/hm3d_v2/exp6/log/6.log 2>&1 &
python -u scripts/main.py --split_l 9 --split_r 12 --episode_l 0 --port 8006 --cuda_id 6 --dataset hm3d_v2 --experiment_name exp6 --type experiment > exp/hm3d_v2/exp6/log/9.log 2>&1 &
python -u scripts/main.py --split_l 12 --split_r 15 --episode_l 0 --port 8007 --cuda_id 7 --dataset hm3d_v2 --experiment_name exp6 --type experiment > exp/hm3d_v2/exp6/log/12.log 2>&1 &
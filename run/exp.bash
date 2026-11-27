# python -u -m scripts.main experiment.port=8000 experiment.cuda_id=0 experiment.dataset=mp3d  experiment.split_l=10 > temp/log/mp3d_10.log 2>&1 &
# python -u -m scripts.main experiment.port=8001 experiment.cuda_id=1 experiment.dataset=mp3d  experiment.split_l=8 experiment.episode_l=35 > temp/log/mp3d_08.log 2>&1 &
# # python -u -m scripts.main experiment.port=8002 experiment.cuda_id=2 experiment.dataset=mp3d  experiment.split_l=2 > temp/log/mp3d_02.log 2>&1 &
# # python -u -m scripts.main experiment.port=8002 experiment.cuda_id=2 experiment.dataset=mp3d  experiment.split_l=3 > temp/log/mp3d_03.log 2>&1 &
# python -u -m scripts.main experiment.port=8003 experiment.cuda_id=3 experiment.dataset=mp3d  > temp/log/mp3d_00.log 2>&1 &

# # python -u -m scripts.main experiment.port=8002 experiment.cuda_id=2 experiment.dataset=hm3d_v2  experiment.split_l=1  > temp/log/hm3d_01.log 2>&1 &
# # python -u -m scripts.main experiment.port=8004 experiment.cuda_id=4 experiment.dataset=hm3d_v2  experiment.split_l=2  > temp/log/hm3d_02.log 2>&1 &
# # python -u -m scripts.main experiment.port=8005 experiment.cuda_id=5 experiment.dataset=hm3d_v2  experiment.split_l=6  > temp/log/hm3d_06.log 2>&1 &
# # python -u -m scripts.main experiment.port=8006 experiment.cuda_id=6 experiment.dataset=hm3d_v2  experiment.split_l=14 > temp/log/hm3d_14.log 2>&1 &
# # python -u -m scripts.main experiment.port=8007 experiment.cuda_id=7 experiment.dataset=hm3d_v2  experiment.split_l=10 > temp/log/hm3d_10.log 2>&1 &

python -u -m scripts.main experiment.split_l=0 experiment.split_r=3 experiment.episode_l=0 experiment.port=8001 experiment.cuda_id=1 experiment.dataset=hm3d_v2 experiment.experiment_name=exp6 experiment.type=experiment > exp/hm3d_v2/exp6/log/0.log 2>&1 &
python -u -m scripts.main experiment.split_l=3 experiment.split_r=6 experiment.episode_l=0 experiment.port=8002 experiment.cuda_id=2 experiment.dataset=hm3d_v2 experiment.experiment_name=exp6 experiment.type=experiment > exp/hm3d_v2/exp6/log/3.log 2>&1 &
python -u -m scripts.main experiment.split_l=6 experiment.split_r=9 experiment.episode_l=0 experiment.port=8004 experiment.cuda_id=4 experiment.dataset=hm3d_v2 experiment.experiment_name=exp6 experiment.type=experiment > exp/hm3d_v2/exp6/log/6.log 2>&1 &
python -u -m scripts.main experiment.split_l=9 experiment.split_r=12 experiment.episode_l=0 experiment.port=8006 experiment.cuda_id=6 experiment.dataset=hm3d_v2 experiment.experiment_name=exp6 experiment.type=experiment > exp/hm3d_v2/exp6/log/9.log 2>&1 &
python -u -m scripts.main experiment.split_l=12 experiment.split_r=15 experiment.episode_l=0 experiment.port=8007 experiment.cuda_id=7 experiment.dataset=hm3d_v2 experiment.experiment_name=exp6 experiment.type=experiment > exp/hm3d_v2/exp6/log/12.log 2>&1 &
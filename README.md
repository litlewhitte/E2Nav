# E2Nav
E2Nav: Exploration-Exploitation trade-off for autonomous exploration and object navigation


## Setup
### Environment setup

Downlode datesets


Create a python environment with python 3.9
```shell
conda create -n soon python=3.9
conda activate soon
```

Install habitat-sim and habitat-lab
```
conda install habitat-sim==0.2.4 -c conda-forge -c aihabitat
cd third_party/habitat-lab 
pip install -e habitat-lab
pip install -e habitat-baselines  # install habitat_baselines
cd ../..
```

Install torch
```
conda install -c pytorch faiss-gpu=1.8.0
pip install torch==2.6.0+cu124 torchvision -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

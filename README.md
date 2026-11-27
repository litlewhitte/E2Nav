# E2Nav

E2Nav is an RGB-D agent for ObjectNav in Habitat. Given an object category, it builds an occupancy map and a scene graph as it moves. It uses room and object landmarks to score exploration frontiers, searches candidate sequences with Monte Carlo tree search (MCTS), and uses a vision-language model to check potential targets.

## Environment

Run commands from the repository root on Linux with an NVIDIA GPU. The navigation process, GroundedSAM detection service and Ollama can use separate environments. The original experiments targeted Python 3.9, Habitat-Sim 0.2.4 and CUDA-enabled PyTorch 2.6. Adapt the GPU packages to the installed driver.

```bash
git clone --recurse-submodules https://github.com/litlewhitte/E2Nav.git
cd E2Nav
conda create -n e2nav python=3.9
conda activate e2nav
conda install habitat-sim=0.2.4 headless -c conda-forge -c aihabitat
python -m pip install -e third_party/habitat-lab
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

The mapping code also imports GPU FAISS, CuPy, cuML, cuDF, cuGraph and PyTorch3D. Install builds compatible with your Python, PyTorch and CUDA versions. These packages have platform-specific install procedures. See the [RAPIDS installation guide](https://docs.rapids.ai/install/), [FAISS installation guide](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) and [PyTorch3D installation guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).

## Data and models

Obtain ObjectNav episodes and scene assets from their respective dataset providers. Each Habitat YAML defines `DATASET.DATA_PATH` and `DATASET.SCENES_DIR`. The default is HM3D v2 validation:

```text
datasets/hm3d_v2/
  objectnav_hm3d_v2/val/val.json.gz
  objectnav_hm3d_v2/val/content/      # per-scene episode files, when supplied
  scene_datasets/                     # preserve the provider's scene hierarchy
```

Other configurations are [HM3D v1](configs/hm3d_v1.yaml), [MP3D](configs/mp3d.yaml) and [HSSD](configs/hssd.yaml). Scene identifiers in the episodes must resolve under the configured scene directory.

In the detection service environment, install PyTorch and torchvision, FastAPI, uvicorn, supervision, OpenCV and Pillow. Install GroundingDINO, Segment Anything and Recognize Anything (RAM) using the [upstream installation instructions](third_party/Grounded-Segment-Anything/README.md). Place the three pretrained checkpoints under `models/checkpoints/`:

- `groundingdino_swint_ogc.pth`
- `sam_vit_h_4b8939.pth`
- `ram_swin_large_14m.pth`

```bash
mkdir -p models
cp third_party/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py models/
python scripts/grounded_sam_server.py --host 127.0.0.1 --port 8000 --device_id 0
```

Keep the detector running. Its API serves `POST /infer_groundedsam`. With `use_clip_feature=true`, the navigation process also loads OpenCLIP `ViT-H-14` with `laion2b_s32b_b79k` weights on first use.

## Run an episode

Start Ollama and pull the vision-language model with `ollama pull qwen2.5vl:72b`, or point the agent at an existing Ollama-compatible vision endpoint. The default URL is `http://localhost:11434/api/chat`. Override it with `E2NAV_CHAT_URL`, the model with `E2NAV_CHAT_MODEL`, and an optional full authorization header value with `E2NAV_AUTHORIZATION`.

In the navigation environment, run the first episode of the first validation scene:

```bash
python -m scripts.main \
  experiment.dataset=hm3d_v2 experiment.split_l=0 experiment.split_r=1 \
  experiment.episode_l=1 experiment.episode_r=2 \
  experiment.cuda_id=0 experiment.port=8000 experiment.type=debug
```

Scene indices start at zero and episode numbers at one. Both upper bounds are exclusive. The stock Habitat runner orders available scene names before slicing. Debug runs write observations and maps under `observations/`; experiment runs write to `exp/<dataset>/<experiment_name>/observations/`. Configure map resolution, planner weights, model settings and output paths in [configs/e2.yaml](configs/e2.yaml).

## License

E2Nav's original code is distributed under the [MIT License](LICENSE). Third-party code, model weights and datasets are subject to their respective licenses.

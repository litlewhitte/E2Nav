import os
import argparse # 1. 首先导入 argparse 和 os

# --- 命令行参数解析 (最先执行) ---
# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="GroundedSAM FastAPI Server")
parser.add_argument(
    "--device_id",
    type=str, # 使用字符串类型
    default="0",  # 默认值设为您原来的 "3"
    help="CUDA device ID(s) to make visible (e.g., '0', '1', '0,1'). Sets CUDA_VISIBLE_DEVICES."
)
parser.add_argument(
    "--port",
    type=int,
    default=8001,
    help="Port to run the FastAPI server on."
)
parser.add_argument(
    "--host",
    type=str,
    default="0.0.0.0",
    help="Host to bind the FastAPI server to."
)

# 解析命令行参数
# 注意：如果脚本还可能被其他方式调用（非直接执行），
# 并且可能附带其他不相关的参数，使用 parse_known_args() 可能更安全。
# 但对于直接执行的脚本，parse_args() 通常就足够了。
cli_args = parser.parse_args()

# 2. 根据解析到的参数设置 CUDA_VISIBLE_DEVICES 环境变量
# 这必须在 import torch 之前完成
os.environ['CUDA_VISIBLE_DEVICES'] = cli_args.device_id

# --- PyTorch 和其他库的导入 ---
# 3. 现在可以安全地导入 torch 和其他依赖 CUDA 的库
import torch
# (保留您原来的所有其他 import 语句)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import base64
import io
from PIL import Image
import numpy as np
import cv2
import torchvision # For NMS and transforms
import torchvision.transforms as TS

# Import model-specific classes
from ram.models import ram as ram_model_builder
from ram import inference_ram

from groundingdino.util.inference import Model as GroundingDINOModel
from segment_anything import build_sam, SamPredictor
import supervision as sv

# --- Configuration ---
# (这些路径保持不变)
GROUNDING_DINO_CONFIG_PATH = 'models/GroundingDINO_SwinT_OGC.py'
GROUNDING_DINO_CHECKPOINT_PATH = 'models/checkpoints/groundingdino_swint_ogc.pth'
RAM_CHECKPOINT_PATH = 'models/checkpoints/ram_swin_large_14m.pth'
SAM_CHECKPOINT_PATH = 'models/checkpoints/sam_vit_h_4b8939.pth'

# DEVICE 将在 torch 导入后根据 CUDA_VISIBLE_DEVICES 的设置正确判断
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RAM_IMAGE_SIZE = 384

print(f"Using device: {DEVICE}") # 这将反映实际可见的 GPU

# --- Global Variables ---
app = FastAPI()
ram_model: Optional[ram_model_builder] = None
grounding_dino_model: Optional[GroundingDINOModel] = None
sam_predictor: Optional[SamPredictor] = None
ram_image_transform: Optional[TS.Compose] = None

# --- Pydantic Models (保持不变) ---
class GroundedSAMInferRequest(BaseModel):
    image_base64: str
    vocabulary: Optional[List[str]] = None
    use_ram_for_vocabulary: bool = False
    box_threshold: float = 0.3
    text_threshold: float = 0.25
    iou_threshold_nms: float = 0.35

class BoxItem(BaseModel):
    box: List[float]
    class_name: str
    class_id: int
    score: float
    mask_base64: Optional[str] = None

class GroundedSAMInferResponse(BaseModel):
    results: List[BoxItem]
    used_vocabulary: List[str]

def image_to_base64(img_np, format="png"):
    img_pil = Image.fromarray(img_np.astype(np.uint8))
    buffered = io.BytesIO()
    img_pil.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# --- FastAPI Event Handling: Load Models on Startup (保持不变) ---
@app.on_event("startup")
async def load_resources():
    global grounding_dino_model, sam_predictor, ram_model, ram_image_transform

    print("开始加载模型...")
    # (模型加载逻辑保持不变, DEVICE 变量已根据环境变量正确设置)
    print(f"加载 RAM 模型 from: {RAM_CHECKPOINT_PATH}")
    try:
        ram_model = ram_model_builder(
            pretrained=RAM_CHECKPOINT_PATH,
            image_size=RAM_IMAGE_SIZE,
            vit='swin_l'
        )
        ram_model.eval()
        ram_model = ram_model.to(DEVICE)

        normalize = TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ram_image_transform = TS.Compose([
            TS.Resize((RAM_IMAGE_SIZE, RAM_IMAGE_SIZE)),
            TS.ToTensor(),
            normalize
        ])
        print("RAM 模型和图像 Transform 加载成功!")
    except ImportError:
        print("错误：无法导入 'inference_ram'。请确保它在 'ram' 模块中可用，并检查导入路径。")
        raise RuntimeError("无法导入 'inference_ram'")
    except Exception as e:
        print(f"加载 RAM 模型或 Transform 失败: {e}")
        raise RuntimeError(f"无法加载 RAM 模型或 Transform: {e}")

    print(f"加载 GroundingDINO 模型 from: {GROUNDING_DINO_CONFIG_PATH} with config: {GROUNDING_DINO_CONFIG_PATH}")
    try:
        grounding_dino_model = GroundingDINOModel(
            model_config_path=GROUNDING_DINO_CONFIG_PATH,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
            device=DEVICE
        )
        print("GroundingDINO 模型加载成功!")
    except Exception as e:
        print(f"加载 GroundingDINO 模型失败: {e}")
        raise RuntimeError(f"无法加载 GroundingDINO 模型: {e}")

    print(f"加载 SAM 模型 from: {SAM_CHECKPOINT_PATH}")
    try:
        sam = build_sam(checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device=DEVICE)
        sam_predictor = SamPredictor(sam)
        print("SAM 模型加载成功!")
    except Exception as e:
        print(f"加载 SAM 模型失败: {e}")
        raise RuntimeError(f"无法加载 SAM 模型: {e}")
    print("所有模型加载完毕。")


# --- Server-side RAM tag generation (保持不变) ---
def generate_tags_with_ram(pil_image_rgb: Image.Image) -> List[str]:
    if ram_model is None or ram_image_transform is None:
        raise RuntimeError("RAM model or image_transform is not loaded.")
    image_tensor_for_ram = ram_image_transform(pil_image_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        res = inference_ram(image_tensor_for_ram, ram_model)
    raw_tags_string = res[0].split(" | ")
    return raw_tags_string


# --- Server-side segment function (保持不变) ---
def server_segment(sam_predictor_instance: SamPredictor, image_rgb: np.ndarray, xyxy_boxes: np.ndarray) -> np.ndarray:
    sam_predictor_instance.set_image(image_rgb)
    result_masks = []
    if xyxy_boxes.size == 0:
        return np.empty((0, image_rgb.shape[0], image_rgb.shape[1]), dtype=bool) if image_rgb.ndim == 3 else np.array([])
    for box in xyxy_boxes:
        masks, scores, logits = sam_predictor_instance.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks, dtype=bool)

# --- API Endpoint for Inference (保持不变) ---
@app.post("/infer_groundedsam", response_model=GroundedSAMInferResponse)
async def infer_groundedsam(request: GroundedSAMInferRequest):
    if grounding_dino_model is None or sam_predictor is None:
        raise HTTPException(status_code=503, detail="基础模型正在加载或加载失败，请稍后再试。")
    if request.use_ram_for_vocabulary and (ram_model is None or ram_image_transform is None):
        raise HTTPException(status_code=503, detail="RAM 模型或其图像 Transform 未加载，无法用于词汇表生成。")

    print(f"收到 GroundedSAM 推理请求: use_ram_for_vocabulary='{request.use_ram_for_vocabulary}', box_threshold='{request.box_threshold}', nms_iou='{request.iou_threshold_nms}'")

    try:
        try:
            image_data = base64.b64decode(request.image_base64)
            pil_image_rgb = Image.open(io.BytesIO(image_data)).convert("RGB")
            image_np_rgb = np.array(pil_image_rgb)
            image_np_bgr = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"图像解码失败: {e}")
            raise HTTPException(status_code=400, detail=f"无效的 image_base64: {e}")

        effective_vocabulary: List[str]
        if request.use_ram_for_vocabulary:
            print("使用 RAM 生成词汇表...")
            effective_vocabulary = generate_tags_with_ram(pil_image_rgb)
            if not effective_vocabulary:
                print("RAM 未生成任何标签，无法继续进行检测。")
                return GroundedSAMInferResponse(results=[], used_vocabulary=[])
            print(f"RAM 生成的词汇表: {effective_vocabulary}")
        elif request.vocabulary:
            effective_vocabulary = request.vocabulary
            print(f"使用提供的词汇表: {effective_vocabulary}")
        else:
            raise HTTPException(status_code=400, detail="未提供词汇表且未选择使用 RAM 生成词汇表。")

        print("运行 GroundingDINO detection...")
        gd_detections: sv.Detections = grounding_dino_model.predict_with_classes(
            image=image_np_bgr,
            classes=effective_vocabulary,
            box_threshold=request.box_threshold,
            text_threshold=request.text_threshold
        )
        print(f"GroundingDINO 在 NMS 前检测到 {len(gd_detections.xyxy)} 个对象。")

        if len(gd_detections.xyxy) > 0:
            nms_idx = torchvision.ops.nms(
                torch.from_numpy(gd_detections.xyxy).to(DEVICE).float(),
                torch.from_numpy(gd_detections.confidence).to(DEVICE).float(),
                request.iou_threshold_nms
            ).cpu().numpy().tolist()
            gd_detections = gd_detections[nms_idx]
            print(f"GroundingDINO 在 NMS 后检测到 {len(gd_detections.xyxy)} 个对象。")

        if len(gd_detections.xyxy) == 0:
            return GroundedSAMInferResponse(results=[], used_vocabulary=effective_vocabulary)

        print("运行 SAM segmentation...")
        final_masks_np = np.array([])
        if gd_detections.xyxy.shape[0] > 0:
            final_masks_np = server_segment(
                sam_predictor_instance=sam_predictor,
                image_rgb=image_np_rgb,
                xyxy_boxes=gd_detections.xyxy
            )
            print(f"SAM 生成了 {final_masks_np.shape[0]} 个掩码。")
        else:
            print("NMS 后没有检测框传递给 SAM。")

        output_results = []
        for i in range(len(gd_detections.xyxy)):
            box = gd_detections.xyxy[i].tolist()
            class_id = gd_detections.class_id[i]
            class_name = effective_vocabulary[class_id] if 0 <= class_id < len(effective_vocabulary) else "Unknown"
            score = gd_detections.confidence[i].item()
            mask_base64 = None
            if i < len(final_masks_np):
                mask_uint8 = (final_masks_np[i] * 255).astype(np.uint8)
                mask_base64 = image_to_base64(mask_uint8)

            output_results.append(BoxItem(
                box=box,
                class_name=class_name,
                class_id=class_id,
                score=score,
                mask_base64=mask_base64
            ))

        print(f"处理完成，返回 {len(output_results)} 个结果。")
        return GroundedSAMInferResponse(results=output_results, used_vocabulary=effective_vocabulary)

    except RuntimeError as e:
        print(f"推理过程中发生运行时错误: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"推理运行时错误: {str(e)}")
    except Exception as e:
        print(f"推理过程中发生一般错误: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"推理错误: {str(e)}")

# --- Uvicorn 启动 ---
import uvicorn # uvicorn 的 import 可以放在这里，因为它不直接依赖 CUDA_VISIBLE_DEVICES

if __name__ == "__main__": # 确保这部分代码只在直接运行脚本时执行
    # 路径检查逻辑保持不变
    paths_to_check = {
        "GroundingDINO config": GROUNDING_DINO_CONFIG_PATH,
        "GroundingDINO checkpoint": GROUNDING_DINO_CHECKPOINT_PATH,
        "SAM checkpoint": SAM_CHECKPOINT_PATH,
        "RAM checkpoint": RAM_CHECKPOINT_PATH
    }
    all_paths_exist = True
    for name, path in paths_to_check.items():
        if not os.path.exists(path):
            print(f"错误: {name} 未在路径 {path} 找到")
            all_paths_exist = False
    if not all_paths_exist:
        print("一个或多个模型路径无效。请检查配置并重试。")
        exit(1)

    module_name = os.path.splitext(os.path.basename(__file__))[0]
    # 使用从命令行解析得到的 cli_args.host 和 cli_args.port
    uvicorn.run(
        f"{module_name}:app",
        host=cli_args.host,
        port=cli_args.port,
        workers=1,
        reload=False # 在生产环境中通常设为 False
    )
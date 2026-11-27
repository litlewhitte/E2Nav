import os
import argparse

parser = argparse.ArgumentParser(description="GroundedSAM FastAPI Server")
parser.add_argument(
    "--device_id",
    type=str,
    default="0",
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

cli_args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = cli_args.device_id

import torch
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
GROUNDING_DINO_CONFIG_PATH = 'models/GroundingDINO_SwinT_OGC.py'
GROUNDING_DINO_CHECKPOINT_PATH = 'models/checkpoints/groundingdino_swint_ogc.pth'
RAM_CHECKPOINT_PATH = 'models/checkpoints/ram_swin_large_14m.pth'
SAM_CHECKPOINT_PATH = 'models/checkpoints/sam_vit_h_4b8939.pth'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RAM_IMAGE_SIZE = 384

print(f"Using device: {DEVICE}")

# --- Global Variables ---
app = FastAPI()
ram_model: Optional[ram_model_builder] = None
grounding_dino_model: Optional[GroundingDINOModel] = None
sam_predictor: Optional[SamPredictor] = None
ram_image_transform: Optional[TS.Compose] = None

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

@app.on_event("startup")
async def load_resources():
    global grounding_dino_model, sam_predictor, ram_model, ram_image_transform

    print("Loading models...")
    print(f"Loading RAM checkpoint {RAM_CHECKPOINT_PATH}")
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
        print("Loaded RAM model and image transform.")
    except ImportError:
        print("Cannot import 'inference_ram'. Check the installed ram package and import path.")
        raise RuntimeError("Cannot import 'inference_ram'")
    except Exception as e:
        print(f"RAM model or transform failed to load: {e}")
        raise RuntimeError(f"Cannot load RAM model or transform: {e}")

    print(f"Loading GroundingDINO with configuration {GROUNDING_DINO_CONFIG_PATH}")
    try:
        grounding_dino_model = GroundingDINOModel(
            model_config_path=GROUNDING_DINO_CONFIG_PATH,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
            device=DEVICE
        )
        print("Loaded GroundingDINO.")
    except Exception as e:
        print(f"GroundingDINO failed to load: {e}")
        raise RuntimeError(f"Cannot load GroundingDINO: {e}")

    print(f"Loading SAM checkpoint {SAM_CHECKPOINT_PATH}")
    try:
        sam = build_sam(checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device=DEVICE)
        sam_predictor = SamPredictor(sam)
        print("Loaded SAM.")
    except Exception as e:
        print(f"SAM failed to load: {e}")
        raise RuntimeError(f"Cannot load SAM: {e}")
    print("All models loaded.")


def generate_tags_with_ram(pil_image_rgb: Image.Image) -> List[str]:
    if ram_model is None or ram_image_transform is None:
        raise RuntimeError("RAM model or image_transform is not loaded.")
    image_tensor_for_ram = ram_image_transform(pil_image_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        res = inference_ram(image_tensor_for_ram, ram_model)
    raw_tags_string = res[0].split(" | ")
    return raw_tags_string


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

@app.post("/infer_groundedsam", response_model=GroundedSAMInferResponse)
async def infer_groundedsam(request: GroundedSAMInferRequest):
    if grounding_dino_model is None or sam_predictor is None:
        raise HTTPException(status_code=503, detail="Base models are unavailable. Check startup logs and retry.")
    if request.use_ram_for_vocabulary and (ram_model is None or ram_image_transform is None):
        raise HTTPException(status_code=503, detail="RAM model or image transform unavailable for vocabulary generation.")

    print(f"GroundedSAM inference request: use_ram_for_vocabulary='{request.use_ram_for_vocabulary}', box_threshold='{request.box_threshold}', nms_iou='{request.iou_threshold_nms}'")

    try:
        try:
            image_data = base64.b64decode(request.image_base64)
            pil_image_rgb = Image.open(io.BytesIO(image_data)).convert("RGB")
            image_np_rgb = np.array(pil_image_rgb)
            image_np_bgr = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Image decoding failed: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image_base64: {e}")

        effective_vocabulary: List[str]
        if request.use_ram_for_vocabulary:
            print("Generating vocabulary with RAM...")
            effective_vocabulary = generate_tags_with_ram(pil_image_rgb)
            if not effective_vocabulary:
                print("RAM returned no tags for detection.")
                return GroundedSAMInferResponse(results=[], used_vocabulary=[])
            print(f"RAM vocabulary: {effective_vocabulary}")
        elif request.vocabulary:
            effective_vocabulary = request.vocabulary
            print(f"Using supplied vocabulary: {effective_vocabulary}")
        else:
            raise HTTPException(status_code=400, detail="Supply a vocabulary or enable RAM vocabulary generation.")

        print("Running GroundingDINO detection...")
        gd_detections: sv.Detections = grounding_dino_model.predict_with_classes(
            image=image_np_bgr,
            classes=effective_vocabulary,
            box_threshold=request.box_threshold,
            text_threshold=request.text_threshold
        )
        print(f"GroundingDINO detected {len(gd_detections.xyxy)} objects before NMS.")

        if len(gd_detections.xyxy) > 0:
            nms_idx = torchvision.ops.nms(
                torch.from_numpy(gd_detections.xyxy).to(DEVICE).float(),
                torch.from_numpy(gd_detections.confidence).to(DEVICE).float(),
                request.iou_threshold_nms
            ).cpu().numpy().tolist()
            gd_detections = gd_detections[nms_idx]
            print(f"GroundingDINO retained {len(gd_detections.xyxy)} objects after NMS.")

        if len(gd_detections.xyxy) == 0:
            return GroundedSAMInferResponse(results=[], used_vocabulary=effective_vocabulary)

        print("Running SAM segmentation...")
        final_masks_np = np.array([])
        if gd_detections.xyxy.shape[0] > 0:
            final_masks_np = server_segment(
                sam_predictor_instance=sam_predictor,
                image_rgb=image_np_rgb,
                xyxy_boxes=gd_detections.xyxy
            )
            print(f"SAM generated {final_masks_np.shape[0]} masks.")
        else:
            print("No boxes remained after NMS for SAM segmentation.")

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

        print(f"Returning {len(output_results)} results.")
        return GroundedSAMInferResponse(results=output_results, used_vocabulary=effective_vocabulary)

    except RuntimeError as e:
        print(f"Inference runtime error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference runtime error: {str(e)}")
    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

import uvicorn

if __name__ == "__main__":
    paths_to_check = {
        "GroundingDINO config": GROUNDING_DINO_CONFIG_PATH,
        "GroundingDINO checkpoint": GROUNDING_DINO_CHECKPOINT_PATH,
        "SAM checkpoint": SAM_CHECKPOINT_PATH,
        "RAM checkpoint": RAM_CHECKPOINT_PATH
    }
    all_paths_exist = True
    for name, path in paths_to_check.items():
        if not os.path.exists(path):
            print(f"Missing {name} at {path}")
            all_paths_exist = False
    if not all_paths_exist:
        print("Invalid model paths. Check the configuration and retry.")
        exit(1)

    module_name = os.path.splitext(os.path.basename(__file__))[0]
    uvicorn.run(
        f"{module_name}:app",
        host=cli_args.host,
        port=cli_args.port,
        workers=1,
        reload=False
    )
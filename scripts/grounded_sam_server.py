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
from groundingdino.util.utils import get_phrases_from_posmap
import re
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
    objectness_score: float
    mask_base64: Optional[str] = None
    probabilities: Optional[List[float]] = None

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

def phrases2classes(phrases: List[str], classes: List[str]) -> np.ndarray:
    def find_index(string, lst):
        string = string.lower().split()[0]
        for i, s in enumerate(lst):
            if string in s.lower():
                return i
        return -1

    class_ids = []
    for phrase in phrases:
        cleaned_phrase = phrase.replace('.', '')
        class_id = find_index(cleaned_phrase, classes)
        class_ids.append(class_id)
        
    return np.array(class_ids)

def get_class_to_token_map(tokenizer, classes: List[str]):
    """Map each class name to its token indices in the full caption."""
    class_to_token_map = {}
    caption = ". ".join(classes) + "."
    
    try:
        tokenized = tokenizer(caption, return_tensors="pt")
    except Exception as e:
        print(f"Tokenization failed: {e}")
        return None, None

    tokens = [tokenizer.decode(ids) for ids in tokenized.input_ids[0]]
    
    current_pos = 0
    for class_name in classes:
        class_tokenized_ids = tokenizer.encode(class_name, add_special_tokens=False)
        num_class_tokens = len(class_tokenized_ids)
        
        if num_class_tokens == 0:
            continue
            
        found = False
        for i in range(current_pos, len(tokens) - num_class_tokens + 1):
            window = tokenized.input_ids[0, i : i + num_class_tokens]
            if torch.equal(window, torch.tensor(class_tokenized_ids)):
                class_to_token_map[class_name] = list(range(i, i + num_class_tokens))
                current_pos = i + num_class_tokens
                found = True
                break
        if not found:
             print(f"No exact token mapping for class '{class_name}' in caption.")
                
    return class_to_token_map, tokenized


def predict_with_full_details(
    model: GroundingDINOModel,
    image_bgr: np.ndarray,
    classes: List[str],
    box_threshold: float,
    text_threshold: float
) -> List[Dict[str, Any]]:
    h_img, w_img, _ = image_bgr.shape
    if not classes:
        return []

    caption = ". ".join(classes) + "."
    image_tensor = model.preprocess_image(image_bgr=image_bgr).to(model.device)
    tokenizer = model.model.tokenizer
    tokenized = tokenizer(caption)
    class_to_token_map, _ = get_class_to_token_map(tokenizer, classes)

    with torch.no_grad():
        outputs = model.model(image_tensor[None], captions=[caption])

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]
    prediction_boxes = outputs["pred_boxes"].cpu()[0]

    objectness_logits = prediction_logits.max(dim=1)[0]
    box_mask = objectness_logits > box_threshold
    
    logits_filtered = prediction_logits[box_mask]
    boxes_filtered = prediction_boxes[box_mask]
    objectness_scores_filtered = objectness_logits[box_mask]

    if logits_filtered.shape[0] == 0:
        return []

    phrases_from_model = []
    for logit in logits_filtered:
        phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        phrases_from_model.append(phrase)

    class_ids_from_model = phrases2classes(phrases=phrases_from_model, classes=classes)

    detailed_results = []
    for i in range(logits_filtered.shape[0]):
        if class_ids_from_model[i] == -1:
            continue

        final_class_id = class_ids_from_model[i]
        box_coords = boxes_filtered[i]
        objectness_score = objectness_scores_filtered[i].item()
        box_logit = logits_filtered[i]

        class_scores = {}
        for class_name, token_indices in class_to_token_map.items():
            if token_indices:
                class_scores[class_name] = box_logit[token_indices].max().item()
        
        ordered_scores = [class_scores.get(c, 0.0) for c in classes]
        probabilities = torch.softmax(torch.tensor(ordered_scores), dim=-1).numpy().tolist()
        
        final_class_score = ordered_scores[final_class_id]

        cx_norm, cy_norm, w_norm, h_norm = box_coords.numpy()
        box_xyxy_abs = [
            (cx_norm - w_norm / 2) * w_img,
            (cy_norm - h_norm / 2) * h_img,
            (cx_norm + w_norm / 2) * w_img,
            (cy_norm + h_norm / 2) * h_img,
        ]

        detailed_results.append({
            "box_xyxy": box_xyxy_abs,
            "objectness_confidence": objectness_score,
            "class_score": final_class_score,
            "class_id": final_class_id,
            "probabilities": probabilities
        })

    return detailed_results

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
            h_img, w_img, _ = image_np_bgr.shape
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

        print("Running detailed GroundingDINO detection...")
        detailed_predictions = predict_with_full_details(
            model=grounding_dino_model,
            image_bgr=image_np_bgr,
            classes=effective_vocabulary,
            box_threshold=request.box_threshold,
            text_threshold=request.text_threshold
        )
        print(f"GroundingDINO detected {len(detailed_predictions)} objects before NMS.")

        if len(detailed_predictions) > 0:
            boxes_for_nms = torch.from_numpy(np.array([p["box_xyxy"] for p in detailed_predictions])).to(DEVICE).float()
            confidences_for_nms = torch.tensor([p["class_score"] for p in detailed_predictions]).to(DEVICE).float()
            nms_idx = torchvision.ops.nms(
                boxes_for_nms,
                confidences_for_nms,
                request.iou_threshold_nms
            ).cpu().numpy().tolist()
            gd_detections_after_nms = [detailed_predictions[i] for i in nms_idx]
            print(f"GroundingDINO retained {len(gd_detections_after_nms)} objects after NMS.")
        else:
            gd_detections_after_nms = []


        if not gd_detections_after_nms:
            return GroundedSAMInferResponse(results=[], used_vocabulary=effective_vocabulary)

        boxes_for_sam = np.array([p["box_xyxy"] for p in gd_detections_after_nms])

        print("Running SAM segmentation...")
        final_masks_np = server_segment(
            sam_predictor_instance=sam_predictor,
            image_rgb=image_np_rgb,
            xyxy_boxes=boxes_for_sam
        )
        print(f"SAM generated {final_masks_np.shape[0]} masks.")


        output_results = []
        for i, det in enumerate(gd_detections_after_nms):
            class_id = det["class_id"]
            class_name = effective_vocabulary[class_id] if 0 <= class_id < len(effective_vocabulary) else "Unknown"
            
            mask_base64 = None
            if i < len(final_masks_np):
                mask_uint8 = (final_masks_np[i] * 255).astype(np.uint8)
                mask_base64 = image_to_base64(mask_uint8)

            output_results.append(BoxItem(
                box=det["box_xyxy"],
                class_name=class_name,
                class_id=class_id,
                score=det["class_score"],
                objectness_score=det["objectness_confidence"],
                mask_base64=mask_base64,
                probabilities=det["probabilities"]
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

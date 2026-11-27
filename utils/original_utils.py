# import argparse
# import sys
# import os

# script_dir = os.path.dirname(os.path.abspath(__file__))

# sys.path.insert(0, GSA_PATH)

# import numpy as np
# import json
# import supervision as sv
# import torch
# import torchvision
# from PIL import Image
# import litellm

# import GroundingDINO.groundingdino.datasets.transforms as T
# from GroundingDINO.groundingdino.models import build_model
# from GroundingDINO.groundingdino.util.slconfig import SLConfig
# from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
# from GroundingDINO.groundingdino.util.inference import predict

# from segment_anything import (
#     build_sam,
#     build_sam_hq,
#     SamPredictor
# ) 
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# from ram.models import ram
# from ram import inference_ram
# import torchvision.transforms as TS


# def detection_segmentation(observations, obj_goal, grounding_dino_model, ram_model, sam_predictor, vocabulary, device='cuda'):
#     BOX_THRESHOLD = 0.45
#     TEXT_THRESHOLD = 0.45
#     IOU_THRESHOLD = 0.35
#     normalize = TS.Normalize(
#     mean=[0.485, 0.456, 0.406],
#     std=[0.229, 0.224, 0.225]
#     )
#     transform = TS.Compose(
#         [
#             TS.Resize((384, 384)),
#             TS.ToTensor(), 
#             normalize
#         ]
#     )
    
#     # ram_model.eval()
#     # ram_model = ram_model.to('cuda')
#     image = observations['rgb']
#     image = image[:, :, ::-1]
#     # image_pillow = Image.fromarray(image)
#     # image_pillow = image_pillow.resize((384, 384))
#     # image_pillow = transform(image_pillow).unsqueeze(0).to(device)
#     # res = inference_ram(image_pillow , ram_model)
#     # AUTOMATIC_CLASSES=res[0].split(" | ")
    
#     detections = grounding_dino_model.predict_with_classes(
#         image=image,
#         classes=vocabulary,
#         box_threshold=BOX_THRESHOLD,
#         text_threshold=TEXT_THRESHOLD
#     )
#     print(f"check detections len: {len(detections)}")
#     nms_idx = torchvision.ops.nms(
#         torch.from_numpy(detections.xyxy), 
#         torch.from_numpy(detections.confidence), 
#         IOU_THRESHOLD
#     ).numpy().tolist()
    
#     detections.xyxy = detections.xyxy[nms_idx]
#     detections.confidence = detections.confidence[nms_idx]
#     detections.class_id = detections.class_id[nms_idx]
    
#     box_annotator = sv.BoxAnnotator()
#     labels = [
#         f"{vocabulary[class_id]} {confidence:0.2f}" 
#         for _, _, confidence, class_id, _, _
#         in detections]
#     annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
    
#     def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
#         sam_predictor.set_image(image)
#         result_masks = []
#         for box in xyxy:
#             masks, scores, logits = sam_predictor.predict(
#                 box=box,
#                 multimask_output=True
#             )
#             index = np.argmax(scores)
#             result_masks.append(masks[index])
#         return np.array(result_masks)

#     detections.mask = segment(
#         sam_predictor=sam_predictor,
#         image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
#         xyxy=detections.xyxy
#     )
    
#     box_annotator = sv.BoxAnnotator()
#     mask_annotator = sv.MaskAnnotator()
#     labels = [
#         f"{vocabulary[class_id]} {confidence:0.2f}" 
#         for _, _, confidence, class_id, _, _
#         in detections]
#     annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
#     annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    
#     return detections, annotated_frame, annotated_image



# In your client script (e.g., the one containing detection_segmentation_client)

import requests
import base64
import io
import numpy as np
import cv2
import os
from PIL import Image
import supervision as sv
# import torch # Not needed if NMS is fully on server
# import torchvision # Not needed if NMS is fully on server
from typing import Optional, List # For type hinting

# --- Helper functions (pil_image_to_base64, base64_to_mask_array - keep as before) ---
def pil_image_to_base64(pil_img, format="JPEG"):
    buffered = io.BytesIO()
    pil_img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def base64_to_pil_image(base64_string):
    image_bytes = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_bytes))

def base64_to_mask_array(base64_string):
    pil_mask = base64_to_pil_image(base64_string).convert('L') # Convert to grayscale
    mask_array = np.array(pil_mask)
    return mask_array > 127 # Boolean mask

def save_masked_regions(rgb_image, detections, output_dir="masked_output", background_color=(255, 255, 255)):
    if detections.mask is None or len(detections.mask) == 0:
        print("No valid mask data found.")
        return
    
    if rgb_image.shape[:2] != detections.mask[0].shape[:2]:
        print(f"Image dimensions {rgb_image.shape} do not match mask dimensions {detections.mask[0].shape}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    background = np.full_like(rgb_image, background_color)
    
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    
    for idx, mask in enumerate(detections.mask):
        try:
            if mask.ndim == 3:
                mask = mask.squeeze()
            bool_mask = mask.astype(bool)

            masked_image = np.where(
                bool_mask[:, :, None],
                bgr_image,
                background
            )

            filename = f"object_{idx}.jpg"
            save_path = os.path.join(output_dir, filename)
            
            cv2.imwrite(save_path, masked_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            print(f"Saved to {save_path}")

        except Exception as e:
            print(f"Error processing mask {idx}: {str(e)}")

def save_individual_bbox_images(rgb_image, detections, output_dir="individual_bbox_output", 
                               box_color=(0, 255, 0), thickness=2):
    if detections.xyxy is None or len(detections.xyxy) == 0:
        print("No valid bounding boxes found.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    image_height, image_width = bgr_image.shape[:2]
    
    for idx, bbox in enumerate(detections.xyxy):
        x1, y1, x2, y2 = map(int, bbox)
        
        img_copy = bgr_image.copy()
        
        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(x1 + 1, min(x2, image_width))
        y2 = max(y1 + 1, min(y2, image_height))
        
        cv2.rectangle(
            img=img_copy,
            pt1=(x1, y1),
            pt2=(x2, y2),
            color=box_color,
            thickness=thickness
        )
        
        filename = f"scene_bbox_{idx:04d}.png"
        save_path = os.path.join(output_dir, filename)
        
        cv2.imwrite(save_path, img_copy, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print(f"Saved scene with detection box to {save_path}")
            
    
            
# --- Modified detection_segmentation_client function ---
def detection_segmentation(
    observations: dict,
    vocabulary: Optional[List[str]], # Now optional, used if use_ram is False
    port: int,
    use_ram: bool = False,           # New parameter
    box_threshold: float = 0.35,
    text_threshold: float = 0.35,
    iou_threshold: float = 0.35
):
    server_url = f"http://localhost:{port}/infer_groundedsam" # Ensure this matches your server endpoint
    # image_bgr_obs = observations['rgb']
    # image_rgb = cv2.cvtColor(image_bgr_obs, cv2.COLOR_BGR2RGB) 
    image_rgb = observations['rgb']

    pil_image = Image.fromarray(image_rgb) # Expects RGB
    image_base64 = pil_image_to_base64(pil_image)

    payload = {
        "image_base64": image_base64,
        "use_ram_for_vocabulary": use_ram, # Pass the use_ram flag to server
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "iou_threshold_nms": iou_threshold
    }

    if not use_ram:
        if vocabulary is None or not vocabulary: # Check if vocabulary is provided when not using RAM
            print("Supply a vocabulary when use_ram is false.")
            return sv.Detections.empty(), image_rgb.copy(), image_rgb.copy()
        payload["vocabulary"] = vocabulary
    elif vocabulary: # use_ram is True but vocabulary is also provided
        print("Using RAM vocabulary because use_ram is enabled.")


    try:
        response = requests.post(server_url, json=payload, timeout=120)
        response.raise_for_status()
        response_json = response.json()
        results_data = response_json["results"]
        # Get the vocabulary actually used by the server for labeling
        server_used_vocabulary = response_json.get("used_vocabulary", [])
    except requests.exceptions.RequestException as e:
        print(f"Server request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Server response: {e.response.text}")
        return sv.Detections.empty(), image_rgb.copy(), image_rgb.copy()

    if not results_data:
        return sv.Detections.empty(), image_rgb.copy(), image_rgb.copy()

    xyxy_list = []
    confidence_list = []
    class_id_list = []
    mask_list = []

    for item in results_data:
        xyxy_list.append(item['box'])
        confidence_list.append(item['score'])
        class_id_list.append(item['class_id']) # This ID is relative to server_used_vocabulary

        if item.get('mask_base64'):
            mask_array = base64_to_mask_array(item['mask_base64'])
            mask_list.append(mask_array)

    if not xyxy_list:
        return sv.Detections.empty(), image_rgb.copy(), image_rgb.copy()

    detections = sv.Detections(
        xyxy=np.array(xyxy_list),
        confidence=np.array(confidence_list),
        class_id=np.array(class_id_list),
        mask=np.array(mask_list) if mask_list else None
    )

    annotating_image_display = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()

    labels = []
    # Use server_used_vocabulary for consistent labeling
    labeling_vocabulary = server_used_vocabulary
    if not labeling_vocabulary and not use_ram and vocabulary: # Fallback if server didn't send vocab but we have original
        labeling_vocabulary = vocabulary

    if len(detections.class_id) > 0 and labeling_vocabulary:
        for i in range(len(detections.xyxy)):
            class_id = detections.class_id[i]
            conf = detections.confidence[i]
            if 0 <= class_id < len(labeling_vocabulary):
                label = f"{labeling_vocabulary[class_id]} {conf:0.2f}"
            else:
                label = f"UnknownID:{class_id} (VocabSize:{len(labeling_vocabulary)}) {conf:0.2f}"
            labels.append(label)
    elif len(detections.class_id) > 0: # No vocabulary available for labels
         labels = [f"ID:{detections.class_id[i]} C:{detections.confidence[i]:0.2f}" for i in range(len(detections.xyxy))]


    annotated_frame_bbox = box_annotator.annotate(
        scene=annotating_image_display.copy(),
        detections=detections,
        labels=labels if labels else None
    )

    if detections.mask is not None and len(detections.mask) > 0:
        annotated_image_mask = mask_annotator.annotate(
            scene=annotating_image_display.copy(),
            detections=detections
        )
        annotated_image_mask = box_annotator.annotate(
            scene=annotated_image_mask,
            detections=detections,
            labels=labels if labels else None
        )
    else:
        annotated_image_mask = annotated_frame_bbox.copy() # if no masks, show bbox annotations


    return detections, annotated_frame_bbox, annotated_image_mask


# Example usage (ensure your server is running with the updated code)
if __name__ == '__main__':

    observations = dict()
    file_path = "temp/9.png"
    pic = Image.open(file_path)
    pic = np.array(pic)

    observations["rgb"] = pic
    with open('utils/classes/classes.txt', 'r') as file:
        vocabulary = [line.rstrip('\n') for line in file]
    # vocabulary = ["living room", "dining room", "bedroom", "kitchen", "bathroom", "entryway", "study", "home gym", "storage room", "basement", "balcony", "garden", "walk-in closet"]
    # vocabulary = ["living room", "dining room", "bedroom", "kitchen", "bathroom", "entryway", "study", "gym", "storage room", "basement", "balcony", "garden", "walk-in closet"]
    # vocabulary = ["floor"]
    # vocabulary.append("cabinet")
    
    use_ram = False
    detections, annotated_frame_bbox, annotated_image_mask = detection_segmentation(
        observations=observations,
        vocabulary=vocabulary,
        use_ram=use_ram,
        port=8003,
        box_threshold=0.35,  
        text_threshold=0.35, 
        iou_threshold=0.35
    )
    if use_ram == False:
        for i in range(len(detections.xyxy)):
            print(vocabulary[detections.class_id[i]])
    else:
        for i in range(len(detections.xyxy)):
            print(detections.class_id[i])
    # print(dir(detections[0]))
    cv2.imwrite("temp/9-1.png", annotated_image_mask)

    # save_individual_bbox_images(rgb_image=pic, detections=detections, output_dir="temp/bbox")

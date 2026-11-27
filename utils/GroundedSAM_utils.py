# import argparse
# import sys
# import os

# script_dir = os.path.dirname(os.path.abspath(__file__))
# GSA_PATH = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouqihang-240108100050/HuNav/models/Grounded-Segment-Anything'
# sys.path.insert(0, GSA_PATH)

# import numpy as np
# import json
# import supervision as sv
# import torch
# import torchvision
# from PIL import Image
# import litellm
# from skimage.morphology import binary_dilation, binary_closing  # 新增形态学处理库

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
    # 验证输入有效性
    if detections.mask is None or len(detections.mask) == 0:
        print("警告：未找到有效的mask数据")
        return
    
    if rgb_image.shape[:2] != detections.mask[0].shape[:2]:
        print(f"错误：图像尺寸{rgb_image.shape}与mask尺寸{detections.mask[0].shape}不匹配")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成纯色背景图（与原始图像同尺寸）
    background = np.full_like(rgb_image, background_color)
    
    # 转换RGB到BGR格式（因cv2.imwrite使用BGR）
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    
    # 遍历所有mask
    for idx, mask in enumerate(detections.mask):
        try:
            # 确保mask为二维布尔数组
            if mask.ndim == 3:
                mask = mask.squeeze()
            bool_mask = mask.astype(bool)

            # 合成图像：mask区域保留原图，其余背景色
            masked_image = np.where(
                bool_mask[:, :, None],  # 增加通道维度用于广播
                bgr_image,
                background
            )

            # 生成文件名
            filename = f"object_{idx}.jpg"
            save_path = os.path.join(output_dir, filename)
            
            # 保存图像（默认JPEG压缩质量为95）
            cv2.imwrite(save_path, masked_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            print(f"已保存：{save_path}")

        except Exception as e:
            print(f"处理mask {idx} 时出错：{str(e)}")

def save_individual_bbox_images(rgb_image, detections, output_dir="individual_bbox_output", 
                               box_color=(0, 255, 0), thickness=2):
    # 输入验证
    if detections.xyxy is None or len(detections.xyxy) == 0:
        print("警告：未找到有效的bbox数据")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换颜色空间为OpenCV所需的BGR格式
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    image_height, image_width = bgr_image.shape[:2]
    
    # 遍历所有检测框
    for idx, bbox in enumerate(detections.xyxy):
        # 解析并规范坐标
        x1, y1, x2, y2 = map(int, bbox)
        
        # 创建图像副本
        img_copy = bgr_image.copy()
        
        # 坐标边界保护
        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(x1 + 1, min(x2, image_width))
        y2 = max(y1 + 1, min(y2, image_height))
        
        # 绘制当前检测框
        cv2.rectangle(
            img=img_copy,
            pt1=(x1, y1),
            pt2=(x2, y2),
            color=box_color,
            thickness=thickness
        )
        
        # 生成文件名
        filename = f"scene_bbox_{idx:04d}.png"
        save_path = os.path.join(output_dir, filename)
        
        # 保存图像
        cv2.imwrite(save_path, img_copy, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print(f"已保存包含单个检测框的场景图：{save_path}")
            
    
            
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
    server_url = f"http://localhost:{port}/infer_groundedsam"
    # 假设 observations['rgb'] 是 RGB 格式的 numpy 数组
    # image_bgr_obs = observations['rgb']
    # image_rgb = cv2.cvtColor(image_bgr_obs, cv2.COLOR_BGR2RGB) 
    image_rgb = observations['rgb']
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB) 

    pil_image = Image.fromarray(image_rgb)
    image_base64 = pil_image_to_base64(pil_image)

    payload = {
        "image_base64": image_base64,
        "use_ram_for_vocabulary": use_ram,
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "iou_threshold_nms": iou_threshold
    }

    if not use_ram:
        if vocabulary is None or not vocabulary:
            print("错误: 当 'use_ram' 为 False 时，必须提供 'vocabulary'。")
            return sv.Detections.empty(), image_bgr.copy(), image_bgr.copy(), []
        payload["vocabulary"] = vocabulary
    elif vocabulary:
        print("提醒: 'use_ram' 为 True，但同时提供了 'vocabulary'。将使用 RAM 生成的词汇表。")


    try:
        response = requests.post(server_url, json=payload, timeout=120)
        response.raise_for_status()
        response_json = response.json()
        results_data = response_json["results"]
        # 获取服务器实际使用的词汇表
        server_used_vocabulary = response_json.get("used_vocabulary", [])
    except requests.exceptions.RequestException as e:
        print(f"调用服务器出错: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"服务器响应内容: {e.response.text}")
        return sv.Detections.empty(), image_bgr.copy(), image_bgr.copy(), []

    if not results_data:
        return sv.Detections.empty(), image_bgr.copy(), image_bgr.copy(), []

    xyxy_list = []
    confidence_list = []
    class_id_list = []
    mask_list = []
    probabilities_list = []  # <--- (a) 初始化一个列表来存储概率

    for item in results_data:
        xyxy_list.append(item['box'])
        confidence_list.append(item['score'])
        class_id_list.append(item['class_id'])

        if item.get('mask_base64'):
            mask_array = base64_to_mask_array(item['mask_base64'])
            mask_list.append(mask_array)
        
        # <--- (b) 解析并存储概率数据
        # 如果服务器返回了概率，就添加到列表中，否则添加一个空列表
        probabilities_list.append(item.get('probabilities', []))


    if not xyxy_list:
        return sv.Detections.empty(), image_bgr.copy(), image_bgr.copy(), []

    detections = sv.Detections(
        xyxy=np.array(xyxy_list),
        confidence=np.array(confidence_list),
        class_id=np.array(class_id_list),
        mask=np.array(mask_list) if mask_list else None
    )

    annotating_image_display = image_bgr   # display保持bgr格式
    

    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()

    labels = []
    labeling_vocabulary = server_used_vocabulary
    if not labeling_vocabulary and not use_ram and vocabulary:
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
    elif len(detections.class_id) > 0:
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
        annotated_image_mask = annotated_frame_bbox.copy()

    # <--- (c) 修改函数返回值
    return detections, annotated_frame_bbox, annotated_image_mask, probabilities_list


# Example usage (ensure your server is running with the updated code)
# if __name__ == '__main__':

#     observations = dict()
#     file_path = "temp/10.png"
#     pic = Image.open(file_path)
#     pic = np.array(pic)

#     observations["rgb"] = pic
#     with open('utils/classes/classes.txt', 'r') as file:
#         vocabulary = [line.rstrip('\n') for line in file]
#     # vocabulary = ["living room", "dining room", "bedroom", "kitchen", "bathroom", "entryway", "study", "home gym", "storage room", "basement", "balcony", "garden", "walk-in closet"]
#     # vocabulary = ["living room", "dining room", "bedroom", "kitchen", "bathroom", "entryway", "study", "gym", "storage room", "basement", "balcony", "garden", "walk-in closet"]
#     # vocabulary = ["curtain"]
#     # vocabulary.append("cabinet")
    
#     use_ram = True
#     detections, annotated_frame_bbox, annotated_image_mask = detection_segmentation(
#         observations=observations,
#         vocabulary=vocabulary,
#         use_ram=use_ram,
#         port=8003,
#         box_threshold=0.35,  
#         text_threshold=0.35, 
#         iou_threshold=0.35
#     )
#     if use_ram == False:
#         for i in range(len(detections.xyxy)):
#             print(vocabulary[detections.class_id[i]])
#     else:
#         for i in range(len(detections.xyxy)):
#             print(detections.class_id[i])
#     # print(dir(detections[0]))
#     cv2.imwrite("temp/10-2.png", annotated_image_mask)
    
#     # save_masked_regions(rgb_image=pic, detections=detections, output_dir="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouqihang-240108100050/HuNav/utils/masks")
#     # save_individual_bbox_images(rgb_image=pic, detections=detections, output_dir="temp/bbox")


if __name__ == '__main__':

    observations = dict()
    file_path = "/inspire/hdd/global_user/zhouqihang-240108100050/E2Nav/observations/20251014_085411_0_50/episode_3/rgb/step_0082.png"
    pic = Image.open(file_path).convert("RGB") # 确保是RGB
    pic_np = np.array(pic)

    observations["rgb"] = pic_np
    
    # 定义你的词汇表
    with open('utils/classes/classes.txt', 'r') as file:
        vocabulary = [line.strip() for line in file if line.strip()]
    vocabulary = ["sofa"]
    # vocabulary.append("flower")
    # 选择是否使用RAM
    use_ram = False # 设为False来测试你自己的词汇表和概率
    
    # 调用修改后的函数，接收新增的返回值
    detections, annotated_frame_bbox, annotated_image_mask, probabilities = detection_segmentation(
        observations=observations,
        vocabulary=vocabulary, # 传递你的词汇表
        use_ram=use_ram,
        port=8000, # <--- 确保端口号正确
        box_threshold=0.35,  
        text_threshold=0.32, # text_threshold可以设低一点，以捕获更多潜在匹配
        iou_threshold=0.35
    )
    print(len(probabilities), len(detections.xyxy), len(probabilities[0]))
    print(probabilities[0])
    # 确定用于标签和概率分析的词汇表
    # 服务器返回的词汇表是黄金标准


    print("\n" + "="*20 + " 检测结果分析 " + "="*20)
    print(f"共检测到 {len(detections.xyxy)} 个物体。")
    print(f"使用的词汇表 ({len(vocabulary)}个类别): {vocabulary}")
    print("="*55 + "\n")

    # 遍历每个检测结果，并打印其详细信息
    if len(detections.xyxy) > 0:
        for i in range(len(detections.xyxy)):
            class_id = detections.class_id[i]
            confidence = detections.confidence[i]
            class_name = vocabulary[class_id] if 0 <= class_id < len(vocabulary) else "Unknown"
            
            print(f"--- 物体 #{i+1} ---")
            print(f"  - 预测类别: {class_name} (ID: {class_id})")
            print(f"  - 置信度 (原始分数): {confidence:.4f}")

            # 打印概率分布
            if i < len(probabilities) and probabilities[i]:
                probs = np.array(probabilities[i])
                print(f"  - 概率分布总和: {np.sum(probs):.4f}") # 验证总和是否为1
                
                # 获取概率最高的 Top-5 类别
                top_k = 5
                top_indices = np.argsort(probs)[-top_k:][::-1]

                print(f"  - Top-{top_k} 概率分布:")
                for idx in top_indices:
                    prob_class_name = vocabulary[idx]
                    prob_value = probs[idx]
                    print(f"    - {prob_class_name:<20}: {prob_value:.4f}")
            else:
                print("  - 未提供概率分布。")
            print("-" * 20 + "\n")


    # 保存标注后的图像
    # 注意：annotated_image_mask的颜色空间是RGB，cv2.imwrite需要BGR
    output_image_bgr = cv2.cvtColor(annotated_image_mask, cv2.COLOR_RGB2BGR)
    cv2.imwrite("temp/good-1.png", output_image_bgr)
    
    save_masked_regions(rgb_image=pic_np, detections=detections, output_dir="temp/bbox2")
    save_individual_bbox_images(rgb_image=pic_np, detections=detections, output_dir="temp/bbox2")
"""
CLIP特征提取工具函数
参考concept-graphs的实现，为E2Nav提供真正的视觉特征
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def compute_clip_features_batched(image_rgb, detections, clip_model, clip_preprocess, device):
    """
    批量计算检测结果的CLIP图像特征
    
    参考: concept-graphs/conceptgraph/utils/model_utils.py
    
    Args:
        image_rgb: RGB图像 (H, W, 3) numpy数组
        detections: supervision.Detections对象，包含xyxy边界框
        clip_model: CLIP模型
        clip_preprocess: CLIP预处理函数
        device: 设备（'cuda'或'cpu'）
        
    Returns:
        image_feats: CLIP图像特征，shape (N, clip_dim)，如(N, 1024)
    """
    if len(detections.xyxy) == 0:
        return np.empty((0, clip_model.visual.output_dim))
    
    image_pil = Image.fromarray(image_rgb)
    padding = 20  # 边界框padding
    
    preprocessed_images = []
    
    # 准备批量数据
    for idx in range(len(detections.xyxy)):
        x_min, y_min, x_max, y_max = detections.xyxy[idx]
        
        # 检查并调整padding以避免超出图像边界
        image_width, image_height = image_pil.size
        left_padding = min(padding, x_min)
        top_padding = min(padding, y_min)
        right_padding = min(padding, image_width - x_max)
        bottom_padding = min(padding, image_height - y_max)
        
        # 应用padding
        x_min -= left_padding
        y_min -= top_padding
        x_max += right_padding
        y_max += bottom_padding
        
        # 裁剪图像
        cropped_image = image_pil.crop((x_min, y_min, x_max, y_max))
        
        # CLIP预处理
        preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0)
        preprocessed_images.append(preprocessed_image)
    
    # 转换为batch
    preprocessed_images_batch = torch.cat(preprocessed_images, dim=0).to(device)
    
    # 批量推理
    with torch.no_grad():
        image_features = clip_model.encode_image(preprocessed_images_batch)
        # L2归一化（CLIP标准做法）
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # 转换为numpy
    image_feats = image_features.cpu().numpy()
    
    return image_feats


def compute_clip_features_from_masks(image_rgb, masks, bboxes, clip_model, clip_preprocess, device):
    """
    使用mask区域计算CLIP特征（更精确的方法）
    
    Args:
        image_rgb: RGB图像 (H, W, 3)
        masks: 布尔mask数组 (N, H, W)
        bboxes: 边界框 (N, 4)，格式为[x1, y1, x2, y2]
        clip_model: CLIP模型
        clip_preprocess: CLIP预处理
        device: 设备
        
    Returns:
        image_feats: CLIP特征 (N, clip_dim)
    """
    if len(bboxes) == 0:
        return np.empty((0, clip_model.visual.output_dim))
    
    image_pil = Image.fromarray(image_rgb)
    padding = 20
    
    preprocessed_images = []
    
    for idx in range(len(bboxes)):
        x_min, y_min, x_max, y_max = bboxes[idx]
        mask = masks[idx]
        
        # Padding处理
        image_width, image_height = image_pil.size
        left_padding = min(padding, x_min)
        top_padding = min(padding, y_min)
        right_padding = min(padding, image_width - x_max)
        bottom_padding = min(padding, image_height - y_max)
        
        x_min -= left_padding
        y_min -= top_padding
        x_max += right_padding
        y_max += bottom_padding
        
        # 裁剪图像
        cropped_image = image_pil.crop((x_min, y_min, x_max, y_max))
        
        # 可选：使用mask进一步处理（将背景设为白色或黑色）
        # 这里我们直接使用bbox裁剪，与concept-graphs一致
        
        # CLIP预处理
        preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0)
        preprocessed_images.append(preprocessed_image)
    
    # 批量推理
    preprocessed_images_batch = torch.cat(preprocessed_images, dim=0).to(device)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(preprocessed_images_batch)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    image_feats = image_features.cpu().numpy()
    
    return image_feats


"""CLIP image feature extraction for detected objects."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def compute_clip_features_batched(image_rgb, detections, clip_model, clip_preprocess, device):
    """Compute image embeddings for a batch of detections."""
    if len(detections.xyxy) == 0:
        return np.empty((0, clip_model.visual.output_dim))
    
    image_pil = Image.fromarray(image_rgb)
    padding = 20
    
    preprocessed_images = []
    
    for idx in range(len(detections.xyxy)):
        x_min, y_min, x_max, y_max = detections.xyxy[idx]
        
        image_width, image_height = image_pil.size
        left_padding = min(padding, x_min)
        top_padding = min(padding, y_min)
        right_padding = min(padding, image_width - x_max)
        bottom_padding = min(padding, image_height - y_max)
        
        x_min -= left_padding
        y_min -= top_padding
        x_max += right_padding
        y_max += bottom_padding
        
        cropped_image = image_pil.crop((x_min, y_min, x_max, y_max))
        
        preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0)
        preprocessed_images.append(preprocessed_image)
    
    preprocessed_images_batch = torch.cat(preprocessed_images, dim=0).to(device)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(preprocessed_images_batch)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    image_feats = image_features.cpu().numpy()
    
    return image_feats


def compute_clip_features_from_masks(image_rgb, masks, bboxes, clip_model, clip_preprocess, device):
    """Compute CLIP embeddings over masked regions."""
    if len(bboxes) == 0:
        return np.empty((0, clip_model.visual.output_dim))
    
    image_pil = Image.fromarray(image_rgb)
    padding = 20
    
    preprocessed_images = []
    
    for idx in range(len(bboxes)):
        x_min, y_min, x_max, y_max = bboxes[idx]
        mask = masks[idx]
        
        image_width, image_height = image_pil.size
        left_padding = min(padding, x_min)
        top_padding = min(padding, y_min)
        right_padding = min(padding, image_width - x_max)
        bottom_padding = min(padding, image_height - y_max)
        
        x_min -= left_padding
        y_min -= top_padding
        x_max += right_padding
        y_max += bottom_padding
        
        cropped_image = image_pil.crop((x_min, y_min, x_max, y_max))
        
        
        preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0)
        preprocessed_images.append(preprocessed_image)
    
    preprocessed_images_batch = torch.cat(preprocessed_images, dim=0).to(device)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(preprocessed_images_batch)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    image_feats = image_features.cpu().numpy()
    
    return image_feats


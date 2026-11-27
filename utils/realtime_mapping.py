import numpy as np
import open3d as o3d
from scipy.sparse import csr_matrix
from sklearn.cluster import dbscan
from scipy.sparse.csgraph import connected_components
import cv2
import os
import kornia
import faiss
import torch
from collections import Counter
from sklearn.cluster import DBSCAN
from utils.vlm_utils import object_confirmed

import kornia
import cupy as cp
from cuml.cluster import DBSCAN as cuDBSCAN
import cugraph
import cudf
from open3d.core import Tensor as O3DTensor
from open3d.t.geometry import PointCloud as O3DPointCloud


def rgb_mask_gen(rgb, mask):
    rgb_mask = np.full_like(rgb, (255, 255, 255))
    rgb_mask[mask] = rgb[mask]
    return rgb_mask



def get_aggregate_feature(image, depth, extrinsic, detections, visual_feature, vocabulary, obj_goal, mandatory_detection, voxel_size=0.01, device="cuda:0", total_steps=None, pcd_dir=None, clip_model=None, clip_preprocess=None):
    device = torch.device(device)
    if detections is None or len(detections.xyxy) == 0:
        return []
    
    clip_features = None
    if clip_model is not None and clip_preprocess is not None:
        from utils.clip_utils import compute_clip_features_batched
        clip_features = compute_clip_features_batched(
            image_rgb=image,
            detections=detections,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            device=device
        )
    
    H, W = image.shape[:2]
    fx, fy, cx, cy = 388.19, 388.19, 319.5, 239.5
    
    depth_squeezed = np.squeeze(depth)
    depth_smoothed = cv2.bilateralFilter(depth_squeezed.astype(np.float32), d=5, sigmaColor=0.1, sigmaSpace=10)
    kernel = np.ones((5, 5), np.uint8)
    cleaned_masks_np = np.stack([
        cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2) 
        for m in detections.mask
    ])

    image_t = torch.from_numpy(image).to(device)
    depth_t = torch.from_numpy(depth_smoothed).to(device)
    extrinsic_t = torch.from_numpy(np.array(extrinsic).reshape(4, 4)).float().to(device)
    cleaned_masks_t = torch.from_numpy(cleaned_masks_np).to(device)
    
    non_zero_indices = torch.nonzero(cleaned_masks_t, as_tuple=False)
    
    if non_zero_indices.shape[0] == 0:
        return []

    batch_indices = non_zero_indices[:, 0]
    ys = non_zero_indices[:, 1]
    xs = non_zero_indices[:, 2]

    zs = depth_t[ys, xs]
    colors_t = image_t[ys, xs].float() / 255.0

    valid_depth_mask = (zs > 0.5) & (zs < 5.0)
    
    batch_indices = batch_indices[valid_depth_mask]
    ys, xs, zs = ys[valid_depth_mask], xs[valid_depth_mask], zs[valid_depth_mask]
    colors_t = colors_t[valid_depth_mask]

    if batch_indices.shape[0] == 0:
        return []
        
    x_cam = (xs.float() - cx) * zs / fx
    y_cam = (ys.float() - cy) * zs / fy
    
    points_cam_t = torch.stack([x_cam, y_cam, zs, torch.ones_like(zs)], dim=0)
    
    points_world_t = (extrinsic_t @ points_cam_t)[:3, :].T

    feature_list = []
    num_detections = len(detections.xyxy)
    
    for i in range(num_detections):
        point_mask = (batch_indices == i)
        
        if torch.count_nonzero(point_mask) < 100:
            continue
            
        points_world_i = points_world_t[point_mask]
        colors_i = colors_t[point_mask] 
        sampled_points, sampled_colors = voxel_downsample_torch(points_world_i, colors_i, voxel_size)
        
        pcd_down = o3d.geometry.PointCloud()
        pcd_down.points = o3d.utility.Vector3dVector(sampled_points.cpu().numpy())
        pcd_down.colors = o3d.utility.Vector3dVector(sampled_colors.cpu().float().numpy())
        
        valid_points = sampled_points
        valid_colors = sampled_colors
        
        if len(valid_points) < 100:
            continue
        
        filtered_points, mask = remove_outliers_torch(valid_points)
        filtered_colors = valid_colors[mask]

        if len(filtered_points) < 10:
            continue

        pcd_filtered = o3d.geometry.PointCloud()
        pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points.cpu().numpy())
        pcd_filtered.colors = o3d.utility.Vector3dVector(filtered_colors.cpu().numpy())


        if not pcd_filtered.has_points():
            continue

        points_np = filtered_points.cpu().numpy()
        labels = cluster_dbscan_gpu(points_np, eps=0.05, min_samples=10)

        if labels.max() == -1:
            continue
            
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        main_cluster_label_val = unique_labels[counts.argmax()]
        
        cluster_info = {}
        for label_val in unique_labels:
            indices = np.where(labels == label_val)[0]
            cluster_pcd = pcd_filtered.select_by_index(indices)
            aabb = cluster_pcd.get_axis_aligned_bounding_box()
            size = np.linalg.norm(aabb.get_max_bound() - aabb.get_min_bound())
            
            cluster_info[label_val] = {
                'center': aabb.get_center(),
                'indices': indices,
                'size': size
            }
        
        num_clusters = len(unique_labels)
        label_map = {label_val: idx for idx, label_val in enumerate(unique_labels)}
        
        adj_matrix = np.zeros((num_clusters, num_clusters), dtype=bool)
        for i_idx, i_label in enumerate(unique_labels):
            for j_idx in range(i_idx + 1, num_clusters):
                j_label = unique_labels[j_idx]
                info_i = cluster_info[i_label]
                info_j = cluster_info[j_label]
                
                center_dist = np.linalg.norm(info_i['center'] - info_j['center'])
                adaptive_threshold = (info_i['size'] + info_j['size']) / 2.0
                min_threshold = 0.05
                merge_threshold = max(adaptive_threshold, min_threshold)

                if center_dist < merge_threshold:
                    adj_matrix[i_idx, j_idx] = adj_matrix[j_idx, i_idx] = True

        # graph = csr_matrix(adj_matrix)
        # _, component_labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        # main_component_id = component_labels[label_map[main_cluster_label_val]]
        # 
        # final_indices = np.concatenate([
        #     cluster_info[unique_labels[idx]]['indices'] 
        #     for idx, comp_id in enumerate(component_labels) if comp_id == main_component_id
        # ])
        # 
        # if len(final_indices) == 0:
        #     continue
        #     
        # pcd_final = pcd_filtered.select_by_index(final_indices)
        
        cluster_sizes = [len(cluster_info[label]['indices']) for label in unique_labels]
        
        if len(cluster_sizes) == 1:
            largest_label = unique_labels[0]
            final_indices = cluster_info[largest_label]['indices']
        else:
            sorted_indices = np.argsort(cluster_sizes)[::-1]
            largest_size = cluster_sizes[sorted_indices[0]]
            second_largest_size = cluster_sizes[sorted_indices[1]]
            
            size_ratio_threshold = 0.5
            second_to_first_ratio = second_largest_size / largest_size
            
            if second_to_first_ratio > size_ratio_threshold:
                continue
            
            largest_label = unique_labels[sorted_indices[0]]
            final_indices = cluster_info[largest_label]['indices']
        
        if len(final_indices) == 0:
            continue
            
        pcd_final = pcd_filtered.select_by_index(final_indices)

        if not pcd_final.has_points():
            continue

        center = pcd_final.get_center().tolist()
        class_name = obj_goal if mandatory_detection else vocabulary[detections.class_id[i]]
        score = detections.confidence[i]
        bbox = detections.xyxy[i]
        
        clip_embedding = clip_features[i] if clip_features is not None else None
        
        feature_list.append({
            'detections_id': i,
            'embedding': [visual_feature[i]],
            'clip_embedding': clip_embedding,
            'bbox': bbox,
            'pcd': pcd_final,
            'class_name': class_name,
            'score': score,
            'center': center
        })
        
    return feature_list

def project_pcd_to_mask(
    pcd: o3d.geometry.PointCloud,
    extrinsic: np.ndarray,
    fx: float = 388.19,
    fy: float = 388.19,
    cx: float = 319.5,
    cy: float = 239.5,
    device: str = "cuda:0"
):
    image_height = 480
    image_width = 680

    if not pcd.has_points():
        return None

    dev = torch.device(device)
    points_world_t = torch.from_numpy(np.asarray(pcd.points)).float().to(dev)
    extrinsic_t = torch.from_numpy(extrinsic.reshape(4, 4)).float().to(dev)
    cam_from_world_t = torch.inverse(extrinsic_t)
    points_world_h = torch.cat([points_world_t, torch.ones(points_world_t.shape[0], 1, device=dev)], dim=1)
    points_cam_h = points_world_h @ cam_from_world_t.T
    points_cam_t = points_cam_h[:, :3] / points_cam_h[:, 3, None]

    visible_mask = points_cam_t[:, 2] > 0.01
    if not torch.any(visible_mask):
        return None
    points_cam_t = points_cam_t[visible_mask]
    
    x_cam, y_cam, z_cam = points_cam_t[:, 0], points_cam_t[:, 1], points_cam_t[:, 2]
    u = (x_cam * fx / z_cam) + cx
    v = (y_cam * fy / z_cam) + cy

    in_frame_mask = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
    if not torch.any(in_frame_mask):
        return None
        
    u_final = u[in_frame_mask]
    v_final = v[in_frame_mask]

    x_min = int(torch.min(u_final).item())
    y_min = int(torch.min(v_final).item())
    x_max = int(torch.max(u_final).item())
    y_max = int(torch.max(v_final).item())
    bbox = [x_min, y_min, x_max, y_max]

    return bbox

def voxel_downsample_torch(points, colors, voxel_size):
    voxel_size_tensor = torch.tensor(voxel_size, device=points.device)
    voxel_grid = torch.div(points, voxel_size_tensor, rounding_mode='floor').long()
    voxel_hash = voxel_grid[:, 0] + 1000000 * voxel_grid[:, 1] + 1000000000 * voxel_grid[:, 2]
    unique_voxels, inverse_indices = torch.unique(voxel_hash, return_inverse=True)
    sampled_indices = torch.arange(len(voxel_hash), device=points.device)
    sampled_indices = torch.unique(inverse_indices, return_counts=False)
    return points[sampled_indices], colors[sampled_indices]

def remove_outliers_torch(points, k=20, std_ratio=2.0):
    points_np = points.cpu().numpy().astype('float32')
    
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatL2(3)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(points_np)
    
    distances, _ = gpu_index.search(points_np, k + 1)
    
    dists_t = torch.from_numpy(distances).to(points.device)
    
    avg_dists = dists_t[:, 1:].mean(dim=1)
    threshold = avg_dists.mean() + std_ratio * avg_dists.std()
    mask = avg_dists < threshold
    
    return points[mask], mask

def cluster_dbscan_gpu(points_np, eps, min_samples):
    gdf = cudf.DataFrame({
        'x': points_np[:, 0], 
        'y': points_np[:, 1], 
        'z': points_np[:, 2]
    })
    dbscan = cuDBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(gdf)
    return labels.to_numpy()


# def get_aggregate_feature(image, depth, extrinsic, detections, visual_feature, vocabulary, obj_goal, mandatory_detection, voxel_size=0.01, device="cuda:0"):
#     device = torch.device(device)
#     if detections is None or len(detections.xyxy) == 0:
#         return []
#     H, W = image.shape[:2]
#     fx, fy, cx, cy = 388.19, 388.19, 319.5, 239.5
    
#     depth_squeezed = np.squeeze(depth)
#     depth_smoothed = cv2.bilateralFilter(depth_squeezed.astype(np.float32), d=5, sigmaColor=0.1, sigmaSpace=10)
#     kernel = np.ones((5, 5), np.uint8)
#     cleaned_masks_np = np.stack([
#         cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2) 
#         for m in detections.mask
#     ])

#     image_t = torch.from_numpy(image).to(device)
#     depth_t = torch.from_numpy(depth_smoothed).to(device)
#     extrinsic_t = torch.from_numpy(np.array(extrinsic).reshape(4, 4)).float().to(device)
#     cleaned_masks_t = torch.from_numpy(cleaned_masks_np).to(device)
    
#     non_zero_indices = torch.nonzero(cleaned_masks_t, as_tuple=False)
    
#     if non_zero_indices.shape[0] == 0:
#         return []

#     batch_indices = non_zero_indices[:, 0]
#     ys = non_zero_indices[:, 1]
#     xs = non_zero_indices[:, 2]

#     zs = depth_t[ys, xs]

#     valid_depth_mask = (zs > 0.5) & (zs < 5.0)
    
#     batch_indices = batch_indices[valid_depth_mask]
#     ys, xs, zs = ys[valid_depth_mask], xs[valid_depth_mask], zs[valid_depth_mask]
#     colors_t = colors_t[valid_depth_mask]

#     if batch_indices.shape[0] == 0:
#         return []
        
#     x_cam = (xs.float() - cx) * zs / fx
#     y_cam = (ys.float() - cy) * zs / fy
    
#     points_cam_t = torch.stack([x_cam, y_cam, zs, torch.ones_like(zs)], dim=0)
    
#     points_world_t = (extrinsic_t @ points_cam_t)[:3, :].T

#     feature_list = []
#     num_detections = len(detections.xyxy)
    
#     for i in range(num_detections):
#         point_mask = (batch_indices == i)
        
#         if torch.count_nonzero(point_mask) < 100:
#             continue
            
#         points_world_np = points_world_t[point_mask].cpu().numpy()
#         colors_np = colors_t[point_mask].cpu().numpy()

#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(points_world_np)
#         pcd.colors = o3d.utility.Vector3dVector(colors_np)
        
#         pcd_down = pcd.voxel_down_sample(voxel_size)
#         if not pcd_down.has_points():
#             continue

#         pcd_filtered, _ = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
#         if not pcd_filtered.has_points():
#             continue

#         eps = 0.05
#         min_points = 10
#         labels = np.array(pcd_filtered.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

#             continue
            
#         unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
#         main_cluster_label_val = unique_labels[counts.argmax()]
        
#         cluster_info = {}
#         for label_val in unique_labels:
#             indices = np.where(labels == label_val)[0]
#             cluster_pcd = pcd_filtered.select_by_index(indices)
#             aabb = cluster_pcd.get_axis_aligned_bounding_box()
#             size = np.linalg.norm(aabb.get_max_bound() - aabb.get_min_bound())
            
#             cluster_info[label_val] = {
#                 'center': aabb.get_center(),
#                 'indices': indices,
#                 'size': size
#             }
        
#         num_clusters = len(unique_labels)
#         label_map = {label_val: idx for idx, label_val in enumerate(unique_labels)}
        
#         adj_matrix = np.zeros((num_clusters, num_clusters), dtype=bool)
#         for i_idx, i_label in enumerate(unique_labels):
#             for j_idx in range(i_idx + 1, num_clusters):
#                 j_label = unique_labels[j_idx]
#                 info_i = cluster_info[i_label]
#                 info_j = cluster_info[j_label]
                
#                 center_dist = np.linalg.norm(info_i['center'] - info_j['center'])
#                 adaptive_threshold = (info_i['size'] + info_j['size']) / 2.0
#                 min_threshold = 0.05
#                 merge_threshold = max(adaptive_threshold, min_threshold)

#                 if center_dist < merge_threshold:
#                     adj_matrix[i_idx, j_idx] = adj_matrix[j_idx, i_idx] = True

#         graph = csr_matrix(adj_matrix)
#         _, component_labels = connected_components(csgraph=graph, directed=False, return_labels=True)
#         main_component_id = component_labels[label_map[main_cluster_label_val]]
        
#         final_indices = np.concatenate([
#             cluster_info[unique_labels[idx]]['indices'] 
#             for idx, comp_id in enumerate(component_labels) if comp_id == main_component_id
#         ])
        
#         if len(final_indices) == 0:
#             continue
            
#         pcd_final = pcd_filtered.select_by_index(final_indices)

#         if not pcd_final.has_points():
#             continue

#         center = pcd_final.get_center().tolist()
#         class_name = obj_goal if mandatory_detection else vocabulary[detections.class_id[i]]
#         score = detections.confidence[i]
#         bbox = detections.xyxy[i]
        
#         feature_list.append({
#             'embedding': [visual_feature[i]],
#             'bbox': bbox,
#             'pcd': pcd_final,
#             'class_name': class_name,
#             'score': score,
#             'center': center
#         })
        
#     return feature_list




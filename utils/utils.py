import torch
import torch.nn.functional as F
import numpy as np
import faiss
import open3d as o3d
from collections import Counter
from skimage import color

import cv2
import numpy as np
import faiss
import torch
import open3d as o3d

# def compute_overlap_matrix_2set(objects_old, objects_new) -> np.ndarray:
#     '''
#     Compute pairwise overlapping between two sets of objects using nearest neighbor search.
#     Enhanced with empty point cloud handling and optimized sampling.
#     '''
#     m = len(objects_old)
#     n = len(objects_new)
#     overlap_matrix = np.zeros((m, n))

#     # Preprocess old objects with empty cloud handling
#     points_old = []
#     for obj in objects_old:
#         pcd = obj.pcd
#         points = np.asarray(pcd.points, dtype=np.float32) if not pcd.is_empty() else np.zeros((1, 3), dtype=np.float32)
#         if len(points) > 512:
#             points = points[np.random.choice(len(points), 512, replace=False)]
#         points_old.append(points)

#     # Build FAISS indices for old objects
#     indices = []
#     for arr in points_old:
#         index = faiss.IndexFlatL2(arr.shape[1])
#         index.add(arr)
#         indices.append(index)

#     # Preprocess new objects with empty cloud handling
#     points_new = []
#     for obj in objects_new:
#         pcd = obj['pcd']
#         points = np.asarray(pcd.points, dtype=np.float32) if not pcd.is_empty() else np.zeros((1, 3), dtype=np.float32)
#         if len(points) > 512:
#             points = points[np.random.choice(len(points), 512, replace=False)]
#         points_new.append(points)

#     # Generate bounding boxes with empty check
#     def safe_get_bbox(pcd):
#         if pcd.is_empty() or len(pcd.points) == 0:
#             return np.zeros((8, 3), dtype=np.float32)
#         return np.asarray(pcd.get_axis_aligned_bounding_box().get_box_points())

#     # Batch compute IoU
#     bbox_old = torch.from_numpy(np.stack([safe_get_bbox(obj.pcd) for obj in objects_old]))
#     bbox_new = torch.from_numpy(np.stack([safe_get_bbox(obj['pcd']) for obj in objects_new]))
#     iou = compute_iou_batch(bbox_old, bbox_new)  # Assume this function is defined

#     # Parallelized overlap calculation
#     for i in range(m):
#         for j in range(n):
#             if iou[i, j] < 1e-6:
#                 continue
                
#             # Perform nearest neighbor search
#             distances, _ = indices[i].search(points_new[j], 1)
            
#             # Calculate overlap ratio (threshold = 2.5cm)
#             valid_points = np.sum(distances < 0.025**2)
#             total_points = max(len(points_new[j]), 1)  # Avoid division by zero
#             overlap_matrix[i, j] = valid_points / total_points

#     return overlap_matrix
    


# def compute_overlap_matrix_2set(
#     objects_old, 
#     objects_new, 
#     proximity_thresh=0.05,
#     center_dist_thresh=5.0,
#     min_iou_thresh=1e-6
# ) -> np.ndarray:
#     """

#     Args:

#     Returns:
#     """
#     m = len(objects_old)
#     n = len(objects_new)
    
#     if m == 0 or n == 0:
#         return np.zeros((m, n))

#     overlap_matrix = np.zeros((m, n))
    
#     points_old = [np.asarray(obj.pcd.points, dtype=np.float32) for obj in objects_old]
#     centers_old = np.array([obj.center for obj in objects_old])
    
#     points_new = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects_new]
#     centers_new = np.array([obj['center'] for obj in objects_new])

#     # ...

#     center_dists = np.linalg.norm(centers_old[:, np.newaxis, :] - centers_new[np.newaxis, :, :], axis=2)

#     def build_faiss_index(points_list, sample_size=1024):
#         indices = []
#         for arr in points_list:
#                 indices.append(None)
#                 continue
            
#             # from open3d.geometry import PointCloud
#             # pcd = PointCloud()
#             # pcd.points = o3d.utility.Vector3dVector(arr)
#             # sampled_points = np.asarray(sampled_pcd.points, dtype=np.float32)
#             # if sampled_points.shape[0] == 0:
#             #     indices.append(None)
#             #     continue
            
#             current_sample_size = min(arr.shape[0], sample_size)
#             sampled_points = arr[np.random.choice(arr.shape[0], current_sample_size, replace=False)]

#             index = faiss.IndexFlatL2(sampled_points.shape[1])
#             index.add(sampled_points)
#             indices.append(index)
#         return indices

#     indices_old = build_faiss_index(points_old)
#     indices_new = build_faiss_index(points_new)
    
#     proximity_thresh_sq = proximity_thresh**2

#     for i in range(m):
#         if indices_old[i] is None:
#             continue
            
#         for j in range(n):
#             if indices_new[j] is None:
#                 continue
            
#             if center_dists[i, j] > center_dist_thresh:
#                 continue
            
#             pts_to_query_from_i = points_old[i][np.random.choice(points_old[i].shape[0], min(points_old[i].shape[0], 512), replace=False)]
#             pts_to_query_from_j = points_new[j][np.random.choice(points_new[j].shape[0], min(points_new[j].shape[0], 512), replace=False)]

#             distances_j, _ = indices_old[i].search(pts_to_query_from_j, 1)
#             inliers_j = np.sum(distances_j < proximity_thresh_sq)
#             overlap_j_to_i = inliers_j / len(pts_to_query_from_j)

#             distances_i, _ = indices_new[j].search(pts_to_query_from_i, 1)
#             inliers_i = np.sum(distances_i < proximity_thresh_sq)
#             overlap_i_to_j = inliers_i / len(pts_to_query_from_i)
            
#             overlap_matrix[i, j] = max(overlap_i_to_j, overlap_j_to_i)
#     return overlap_matrix


def _prepare_object_data(objects, sample_size=512, voxel_size=0.05, use_voxel_downsampling=False):
    indices = []
    sampled_points_list = []
    
    for obj in objects:
        if isinstance(obj, dict):
            pcd = obj['pcd']
        else:
            pcd = obj.pcd
            
        points = np.asarray(pcd.points, dtype=np.float32)

        if points.shape[0] < 10:
            indices.append(None)
            sampled_points_list.append(np.zeros((0, 3), dtype=np.float32))
            continue
        
        if use_voxel_downsampling:
            o3d_pcd = o3d.geometry.PointCloud()
            o3d_pcd.points = o3d.utility.Vector3dVector(points)
            sampled_pcd = o3d_pcd.voxel_down_sample(voxel_size=voxel_size)
            sampled_points = np.asarray(sampled_pcd.points, dtype=np.float32)
            if sampled_points.shape[0] == 0:
                 sampled_points = points
        else:
            current_sample_size = min(points.shape[0], sample_size)
            sampled_points = points[np.random.choice(points.shape[0], current_sample_size, replace=False)]

        sampled_points_list.append(sampled_points)
        
        index = faiss.IndexFlatL2(points.shape[1])
        index.add(points)
        indices.append(index)
        
    return indices, sampled_points_list


def compute_overlap_matrix_2set(
    objects_old, 
    objects_new, 
    proximity_thresh=0.05,
    center_dist_thresh=5.0,
):
    """Compute overlap and minimum point distances between two object sets."""
    m = len(objects_old)
    n = len(objects_new)
    
    if m == 0 or n == 0:
        return np.zeros((m, n)), np.full((m, n), np.inf)

    overlap_matrix = np.zeros((m, n), dtype=np.float32)
    min_distance_matrix = np.full((m, n), np.inf, dtype=np.float32)
    
    centers_old = np.array([obj.center if hasattr(obj, 'center') else obj['center'] for obj in objects_old])
    centers_new = np.array([obj['center'] if isinstance(obj, dict) else obj.center for obj in objects_new])

    center_dists = np.linalg.norm(centers_old[:, np.newaxis, :] - centers_new[np.newaxis, :, :], axis=2)
    candidate_mask = (center_dists <= center_dist_thresh)

    indices_old, sampled_points_to_query_old = _prepare_object_data(objects_old)
    indices_new, sampled_points_to_query_new = _prepare_object_data(objects_new)
    
    proximity_thresh_sq = proximity_thresh**2

    candidate_indices = np.argwhere(candidate_mask)
    
    for i, j in candidate_indices:
        if indices_old[i] is None or indices_new[j] is None:
            continue
        
        min_distances = []
        
        query_pts_from_j = sampled_points_to_query_new[j]
        if query_pts_from_j.shape[0] == 0:
            overlap_j_to_i = 0.0
        else:
            distances_j, _ = indices_old[i].search(query_pts_from_j, 1)
            inliers_j = np.sum(distances_j < proximity_thresh_sq)
            overlap_j_to_i = inliers_j / query_pts_from_j.shape[0]
            min_distances.append(np.sqrt(distances_j.min()))

        query_pts_from_i = sampled_points_to_query_old[i]
        if query_pts_from_i.shape[0] == 0:
            overlap_i_to_j = 0.0
        else:
            distances_i, _ = indices_new[j].search(query_pts_from_i, 1)
            inliers_i = np.sum(distances_i < proximity_thresh_sq)
            overlap_i_to_j = inliers_i / query_pts_from_i.shape[0]
            min_distances.append(np.sqrt(distances_i.min()))
            
        overlap_matrix[i, j] = max(overlap_i_to_j, overlap_j_to_i)
        
        if min_distances:
            min_distance_matrix[i, j] = min(min_distances)
            
    return overlap_matrix, min_distance_matrix


def _prepare_color_histograms(objects, bins=(16, 8)):
    """Precompute normalized color histograms."""
    histograms = []
    epsilon = 1e-10

    for obj in objects:
        pcd = obj.pcd if hasattr(obj, 'pcd') else obj['pcd']
        
        if not pcd.has_points() or not pcd.has_colors():
            histograms.append(None)
            continue
            
        colors = np.asarray(pcd.colors)
        if colors.shape[0] == 0:
            histograms.append(None)
            continue

        hsv = color.rgb2hsv(colors[:, np.newaxis, :])[:, 0, :]
        h, s = hsv[:, 0], hsv[:, 1]
        
        hist_counts, _, _ = np.histogram2d(h, s, bins=bins, range=[[0, 1], [0, 1]])
        
        total_sum = np.sum(hist_counts)
        prob_dist = hist_counts / (total_sum + epsilon)
        
        histograms.append(prob_dist)
        
    return histograms

def _find_closest_point_in_2d_projection(pcd, target_xy, r_resolution=20):
    """Find the projected cloud point nearest a 2-D target."""
    points_3d = np.asarray(pcd.points)
    if points_3d.shape[0] == 0:
        return None, float('inf'), -1
    points_2d = points_3d[:, :2]
    target_xy = np.asarray(target_xy) / r_resolution
    diffs = points_2d - target_xy
    dist_sq = np.sum(diffs**2, axis=1)
    min_index = np.argmin(dist_sq)
    closest_point_3d = points_3d[min_index]
    closest_2d = (round(closest_point_3d[0] * r_resolution), round(closest_point_3d[1] * r_resolution))
    return closest_2d

def compute_color_similarity_matrix(
    objects_old, 
    objects_new, 
    center_dist_thresh=5.0,
    bins: tuple = (16, 8)
) -> np.ndarray:
    """Compute pairwise color similarity."""
    m = len(objects_old)
    n = len(objects_new)
    
    if m == 0 or n == 0:
        return np.zeros((m, n))

    color_sim_matrix = np.zeros((m, n), dtype=np.float32)
    
    centers_old = np.array([obj.center if hasattr(obj, 'center') else obj['center'] for obj in objects_old])
    centers_new = np.array([obj.center if hasattr(obj, 'center') else obj['center'] for obj in objects_new])

    center_dists = np.linalg.norm(centers_old[:, np.newaxis, :] - centers_new[np.newaxis, :, :], axis=2)
    candidate_mask = (center_dists <= center_dist_thresh)

    histograms_old = _prepare_color_histograms(objects_old, bins=bins)
    histograms_new = _prepare_color_histograms(objects_new, bins=bins)
    
    candidate_indices = np.argwhere(candidate_mask)
    
    for i, j in candidate_indices:
        hist_old = histograms_old[i]
        hist_new = histograms_new[j]
        
        if hist_old is None or hist_new is None:
            continue
            
        similarity = np.sum(np.sqrt(hist_old * hist_new))
        color_sim_matrix[i, j] = similarity
            
    return color_sim_matrix


def expand_3d_box(bbox: torch.Tensor, eps=0.02) -> torch.Tensor:
    '''
    Expand the side of 3D boxes such that each side has at least eps length.
    Assumes the bbox cornder order in open3d convention. 
    
    bbox: (N, 8, D)
    
    returns: (N, 8, D)
    '''
    center = bbox.mean(dim=1)  # shape: (N, D)

    va = bbox[:, 1, :] - bbox[:, 0, :]  # shape: (N, D)
    vb = bbox[:, 2, :] - bbox[:, 0, :]  # shape: (N, D)
    vc = bbox[:, 3, :] - bbox[:, 0, :]  # shape: (N, D)
    
    a = torch.linalg.vector_norm(va, ord=2, dim=1, keepdim=True)  # shape: (N, 1)
    b = torch.linalg.vector_norm(vb, ord=2, dim=1, keepdim=True)  # shape: (N, 1)
    c = torch.linalg.vector_norm(vc, ord=2, dim=1, keepdim=True)  # shape: (N, 1)
    
    va = torch.where(a < eps, va / a * eps, va)  # shape: (N, D)
    vb = torch.where(b < eps, vb / b * eps, vb)  # shape: (N, D)
    vc = torch.where(c < eps, vc / c * eps, vc)  # shape: (N, D)
    
    new_bbox = torch.stack([
        center - va/2.0 - vb/2.0 - vc/2.0,
        center + va/2.0 - vb/2.0 - vc/2.0,
        center - va/2.0 + vb/2.0 - vc/2.0,
        center - va/2.0 - vb/2.0 + vc/2.0,
        center + va/2.0 + vb/2.0 + vc/2.0,
        center - va/2.0 + vb/2.0 + vc/2.0,
        center + va/2.0 - vb/2.0 + vc/2.0,
        center + va/2.0 + vb/2.0 - vc/2.0,
    ], dim=1) # shape: (N, 8, D)
    
    new_bbox = new_bbox.to(bbox.device)
    new_bbox = new_bbox.type(bbox.dtype)
    
    return new_bbox

def compute_3d_iou_accuracte_batch(bbox1, bbox2):
    '''
    Compute IoU between two sets of oriented (or axis-aligned) 3D bounding boxes.
    
    bbox1: (M, 8, D), e.g. (M, 8, 3)
    bbox2: (N, 8, D), e.g. (N, 8, 3)
    
    returns: (M, N)
    '''
    # Must expend the box beforehand, otherwise it may results overestimated results
    bbox1 = expand_3d_box(bbox1, 0.02)
    bbox2 = expand_3d_box(bbox2, 0.02)
    
    import pytorch3d.ops as ops

    bbox1 = bbox1[:, [0, 2, 5, 3, 1, 7, 4, 6]]
    bbox2 = bbox2[:, [0, 2, 5, 3, 1, 7, 4, 6]]
    
    inter_vol, iou = ops.box3d_overlap(bbox1.float(), bbox2.float())
    
    return iou


def compute_iou_batch(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    '''
    Compute IoU between two sets of axis-aligned 3D bounding boxes.
    
    bbox1: (M, V, D), e.g. (M, 8, 3)
    bbox2: (N, V, D), e.g. (N, 8, 3)
    
    returns: (M, N)
    '''
    # Compute min and max for each box
    bbox1_min, _ = bbox1.min(dim=1) # Shape: (M, 3)
    bbox1_max, _ = bbox1.max(dim=1) # Shape: (M, 3)
    bbox2_min, _ = bbox2.min(dim=1) # Shape: (N, 3)
    bbox2_max, _ = bbox2.max(dim=1) # Shape: (N, 3)

    # Expand dimensions for broadcasting
    bbox1_min = bbox1_min.unsqueeze(1)  # Shape: (M, 1, 3)
    bbox1_max = bbox1_max.unsqueeze(1)  # Shape: (M, 1, 3)
    bbox2_min = bbox2_min.unsqueeze(0)  # Shape: (1, N, 3)
    bbox2_max = bbox2_max.unsqueeze(0)  # Shape: (1, N, 3)

    # Compute max of min values and min of max values
    # to obtain the coordinates of intersection box.
    inter_min = torch.max(bbox1_min, bbox2_min)  # Shape: (M, N, 3)
    inter_max = torch.min(bbox1_max, bbox2_max)  # Shape: (M, N, 3)

    # Compute volume of intersection box
    inter_vol = torch.prod(torch.clamp(inter_max - inter_min, min=0), dim=2)  # Shape: (M, N)

    # Compute volumes of the two sets of boxes
    bbox1_vol = torch.prod(bbox1_max - bbox1_min, dim=2)  # Shape: (M, 1)
    bbox2_vol = torch.prod(bbox2_max - bbox2_min, dim=2)  # Shape: (1, N)

    # Compute IoU, handling the special case where there is no intersection
    # by setting the intersection volume to 0.
    iou = inter_vol / (bbox1_vol + bbox2_vol - inter_vol + 1e-10)

    return iou




# def compute_spatial_similarity(pcl1: o3d.geometry.PointCloud, 
#                               pcl2: o3d.geometry.PointCloud,
#                               max_num_points: int = 4096,
#                               distance_threshold: float = 0.1) -> float:
#     """
    
        
#     """
#     points1 = np.asarray(pcl1.points, dtype=np.float32)
#     points2 = np.asarray(pcl2.points, dtype=np.float32)

#     if len(points1) > max_num_points:
#         indices = np.random.choice(len(points1), max_num_points, replace=False)
#         points1 = points1[indices]
#     if len(points2) > max_num_points:
#         indices = np.random.choice(len(points2), max_num_points, replace=False)
#         points2 = points2[indices]

#     if len(points1) == 0 or len(points2) == 0:
#         return 0.0

#     min1, max1 = np.min(points1, axis=0), np.max(points1, axis=0)
#     min2, max2 = np.min(points2, axis=0), np.max(points2, axis=0)
    
#     if np.any((max1 < min2) | (max2 < min1)):
#         return 0.0

#     def create_index(points):
#         index = faiss.IndexFlatL2(points.shape[1])
#         index.add(points)
#         return index

#     threshold_sq = distance_threshold ** 2
    
#     index1 = create_index(points1)
#     dist1, _ = index1.search(points2, 1)
#     ratio_pcl2_to_pcl1 = np.mean(dist1 < threshold_sq)

#     index2 = create_index(points2)
#     dist2, _ = index2.search(points1, 1)
#     ratio_pcl1_to_pcl2 = np.mean(dist2 < threshold_sq)

#     return (ratio_pcl2_to_pcl1 + ratio_pcl1_to_pcl2) / 2.0

# def compute_cosine_similarity(vec1, vec2):
#     t1 = torch.from_numpy(vec1).float() if isinstance(vec1, np.ndarray) else vec1.float()
#     t2 = torch.from_numpy(vec2).float() if isinstance(vec2, np.ndarray) else vec2.float()
    
#     if t1.dim() == 1:
#     if t2.dim() == 1:
#         t2 = t2.unsqueeze(0)
    
#     if t1.device != t2.device:
#         t2 = t2.to(t1.device)
    


# def get_similarity(new_object, old_object_node):
#     'feature of newly detect object'
#     # new_class_id = new_object['class_id']
#     new_center = new_object['center']
#     new_visual_embedding = new_object['embedding'] # (N,)
#     new_pcd = new_object['pcd']
#     # old_class_id = old_object_node.class_id
#     old_center = old_object_node.center
#     old_visual_embedding = old_object_node.siglip_embedding
#     old_pcd = old_object_node.pcd
    
    
#     visual_sim = compute_cosine_similarity(new_visual_embedding, old_visual_embedding)
#     spatial_sim = compute_spatial_similarity(new_pcd, old_pcd)
#     aggregate_sim = spatial_sim + visual_sim
#     return aggregate_sim


def merge_point_clouds(pcd_new, pcd_old, voxel_size=0.01):
    merged_pcd = o3d.geometry.PointCloud()
    
    points_old = np.asarray(pcd_old.points)
    points_new = np.asarray(pcd_new.points)
    merged_points = np.vstack((points_old, points_new))
    merged_pcd.points = o3d.utility.Vector3dVector(merged_points)
    
    if pcd_old.has_colors() and pcd_new.has_colors():
        colors_old = np.asarray(pcd_old.colors)
        colors_new = np.asarray(pcd_new.colors)
        merged_colors = np.vstack((colors_old, colors_new))
        merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors)

    downsampled_pcd = merged_pcd.voxel_down_sample(voxel_size)
    
    if not downsampled_pcd.is_empty():
        new_center = downsampled_pcd.get_center().tolist()
    else:
        center_old = pcd_old.get_center()
        center_new = pcd_new.get_center()
        new_center = ((center_old + center_new) / 2).tolist()
        
    return downsampled_pcd, new_center

def merge_pcd_batch(object_list, voxel_size=0.01, default_color=(1.0, 1.0, 1.0)):
    if not object_list:
        raise ValueError("Expected at least one point cloud")
    
    merged_pcd = o3d.geometry.PointCloud()
    
    all_points = []
    all_colors = []
    
    for obj in object_list:
        pcd = obj.pcd
        points = np.asarray(pcd.points)
        all_points.append(points)
        
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
        else:
            colors = np.tile(default_color, (points.shape[0], 1))
        all_colors.append(colors)
    
    merged_pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    merged_pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
    
    if len(merged_pcd.points) == 0:
        return merged_pcd
    
    downsampled_pcd = merged_pcd.voxel_down_sample(voxel_size)
    
    if len(downsampled_pcd.points) > 0:
        points = np.asarray(downsampled_pcd.points)
        center = np.mean(points, axis=0)
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        x_size = max_coords[0] - min_coords[0]
        y_size = max_coords[1] - min_coords[1]
            
    return downsampled_pcd

def volume_ratio_voxel_gen(pcd_local, pcd_overall, voxel_size=0.01):
    voxel_grid1 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_local, voxel_size=voxel_size)
    num_voxels1 = len(voxel_grid1.get_voxels())

    voxel_grid2 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_overall, voxel_size=voxel_size)
    num_voxels2 = len(voxel_grid2.get_voxels())

    if num_voxels2 == 0:
        return 0.0
    ratio = num_voxels1 / num_voxels2 
    return ratio


def process_pcd(pcd, cfg, run_dbscan=True):
    
    pcd = pcd.voxel_down_sample(voxel_size=cfg.downsample_voxel_size)
        
    if cfg.dbscan_remove_noise and run_dbscan:
        # print("Before dbscan:", len(pcd.points))
        pcd = pcd_denoise_dbscan(
            pcd, 
            eps=cfg.dbscan_eps, 
            min_points=cfg.dbscan_min_points
        )
        # print("After dbscan:", len(pcd.points))
        
    return pcd

def pcd_denoise_dbscan(pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10) -> o3d.geometry.PointCloud:
    ### Remove noise via clustering
    pcd_clusters = pcd.cluster_dbscan(
        eps=eps,
        min_points=min_points,
    )
    
    # Convert to numpy arrays
    obj_points = np.asarray(pcd.points)
    obj_colors = np.asarray(pcd.colors)
    pcd_clusters = np.array(pcd_clusters)

    # Count all labels in the cluster
    counter = Counter(pcd_clusters)

    # Remove the noise label
    if counter and (-1 in counter):
        del counter[-1]

    if counter:
        # Find the label of the largest cluster
        most_common_label, _ = counter.most_common(1)[0]
        
        # Create mask for points in the largest cluster
        largest_mask = pcd_clusters == most_common_label

        # Apply mask
        largest_cluster_points = obj_points[largest_mask]
        largest_cluster_colors = obj_colors[largest_mask]
        
        # If the largest cluster is too small, return the original point cloud
        if len(largest_cluster_points) < 5:
            return pcd

        # Create a new PointCloud object
        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)
        
        pcd = largest_cluster_pcd
        
    return pcd

def get_bounding_box(cfg, pcd):
    if ("accurate" in cfg.spatial_sim_type or "overlap" in cfg.spatial_sim_type) and len(pcd.points) >= 4:
        try:
            return pcd.get_oriented_bounding_box(robust=True)
        except RuntimeError as e:
            print(f"Met {e}, use axis aligned bounding box instead")
            return pcd.get_axis_aligned_bounding_box()
    else:
        return pcd.get_axis_aligned_bounding_box()
    
    
def merge_obj2_into_obj1(cfg, obj1, obj2, run_dbscan=True):
    '''
    Merge the new object to the old object
    This operation is done in-place
    '''
    n_obj1_det = obj1['num_detections']
    n_obj2_det = obj2['num_detections']
    
    for k in obj1.keys():
        if k in ['caption']:
            # Here we need to merge two dictionaries and adjust the key of the second one
            for k2, v2 in obj2['caption'].items():
                obj1['caption'][k2 + n_obj1_det] = v2
        # elif k not in ['pcd', 'bbox', 'clip_ft', "text_ft"]:  # annotated by yinhang
        elif k not in ['pcd', 'bbox', 'clip_ft', "text_ft", "score", "captions", "reason", "id", "node"]:  # added by someone
            if isinstance(obj1[k], list) or isinstance(obj1[k], int):
                obj1[k] += obj2[k]
            elif k == "inst_color":
                obj1[k] = obj1[k] # Keep the initial instance color
            else:
                # TODO: handle other types if needed in the future
                raise NotImplementedError
        else: # pcd, bbox, clip_ft, text_ft are handled below
            continue

    # merge pcd and bbox
    obj1['pcd'] += obj2['pcd']
    obj1['pcd'] = process_pcd(obj1['pcd'], cfg, run_dbscan=run_dbscan)
    obj1['bbox'] = get_bounding_box(cfg, obj1['pcd'])
    obj1['bbox'].color = [0,1,0]
    
    # # merge clip ft  # annotated by someone
    # obj1['clip_ft'] = (obj1['clip_ft'] * n_obj1_det +
    #                    obj2['clip_ft'] * n_obj2_det) / (
    #                    n_obj1_det + n_obj2_det)
    # obj1['clip_ft'] = F.normalize(obj1['clip_ft'], dim=0)

    # # merge text_ft
    # obj2['text_ft'] = to_tensor(obj2['text_ft'], cfg.device)
    # obj1['text_ft'] = to_tensor(obj1['text_ft'], cfg.device)
    # obj1['text_ft'] = (obj1['text_ft'] * n_obj1_det +
    #                    obj2['text_ft'] * n_obj2_det) / (
    #                    n_obj1_det + n_obj2_det)
    # obj1['text_ft'] = F.normalize(obj1['text_ft'], dim=0)
    
    return obj1


def fill_small_holes(occupancy_map, closing_kernel_size, hole_fill_area):
    original_device = occupancy_map.device
    original_dtype = occupancy_map.dtype
    processed_map = occupancy_map.squeeze().cpu().numpy()
    unknown_mask = (processed_map == 0.0).astype(np.uint8) 
    unknown_mask_grey = unknown_mask * 255
    if closing_kernel_size % 2 == 0:
        closing_kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel_size, closing_kernel_size))
    closed_mask = cv2.morphologyEx(unknown_mask_grey.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        closed_mask, connectivity=8
    )
    filled_mask = closed_mask.copy()
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area <= hole_fill_area:
            filled_mask[labels == i] = 0

    non_unexplored_mask = cv2.bitwise_not(filled_mask)
    non_unexplored_mask[non_unexplored_mask==255] = 1
    holes_mask = non_unexplored_mask & unknown_mask
    processed_map[holes_mask == 1] = 1
    return torch.from_numpy(processed_map).to(device=original_device, dtype=original_dtype).unsqueeze(0).unsqueeze(0)


def compute_visual_similarity_from_probs(probs1, probs2):
    """Compare visual features represented by probability distributions."""
    if isinstance(probs1, list):
        probs1 = torch.tensor(probs1)
    if isinstance(probs2, list):
        probs2 = torch.tensor(probs2)
        
    probs1 = torch.softmax(probs1, dim=0)
    probs2 = torch.softmax(probs2, dim=0)
    
    kl_div = torch.sum(probs1 * torch.log(probs1 / (probs2 + 1e-8) + 1e-8))
    
    visual_similarity = torch.exp(-kl_div).item()
    
    return visual_similarity


def compute_visual_similarity_matrix_clip(object_nodes, aggregate_features, debug=False):
    """Compare CLIP embeddings."""
    m = len(object_nodes)
    n = len(aggregate_features)
    
    if m == 0 or n == 0:
        return np.zeros((m, n))
    
    obj_embeddings = torch.stack([
        torch.tensor(obj_node.clip_embedding) if isinstance(obj_node.clip_embedding, np.ndarray) 
        else obj_node.clip_embedding
        for obj_node in object_nodes
    ])  # (M, clip_dim)
    
    det_embeddings = torch.stack([
        torch.tensor(feat['clip_embedding']) if isinstance(feat['clip_embedding'], np.ndarray)
        else feat['clip_embedding']
        for feat in aggregate_features
    ])  # (N, clip_dim)
    
    obj_embeddings = F.normalize(obj_embeddings, p=2, dim=1)
    det_embeddings = F.normalize(det_embeddings, p=2, dim=1)
    
    det_embeddings_t = det_embeddings.unsqueeze(-1)  # (N, clip_dim, 1)
    obj_embeddings_t = obj_embeddings.T.unsqueeze(0)  # (1, clip_dim, M)
    
    visual_sim_matrix = F.cosine_similarity(det_embeddings_t, obj_embeddings_t, dim=1)  # (N, M)
    visual_sim_matrix = visual_sim_matrix.T
    
    if debug:
        print(f"\n=== CLIP visual similarity matrix ===")
        print(f"Objects: {m}, detections: {n}")
        print(f"CLIP embedding dimensions: {obj_embeddings.shape[1]}")
        print(f"Similarity range: [{visual_sim_matrix.min():.3f}, {visual_sim_matrix.max():.3f}]")
        print(f"Mean similarity: {visual_sim_matrix.mean():.3f}")
        
        threshold = 0.7
        high_sim_pairs = []
        for i in range(m):
            for j in range(n):
                if visual_sim_matrix[i, j] > threshold:
                    high_sim_pairs.append((i, j, visual_sim_matrix[i, j].item()))
        
        if high_sim_pairs:
            print(f"High similarity pairs (>{threshold}): {len(high_sim_pairs)}")
            for i, j, sim in high_sim_pairs[:10]:
                obj_class = object_nodes[i].class_name
                det_class = aggregate_features[j]['class_name']
                print(f"  Object {i} [{obj_class}] vs detection {j} [{det_class}]: {sim:.3f}")
        print("=" * 50)
    
    return visual_sim_matrix.numpy()


def compute_visual_similarity_matrix_belief(object_nodes, aggregate_features, probs_list, debug=False):
    """Compare DINO class probability distributions."""
    m = len(object_nodes)
    n = len(aggregate_features)
    
    if m == 0 or n == 0:
        return np.zeros((m, n))
    
    obj_probs_matrix = torch.stack([obj_node.class_candidates for obj_node in object_nodes])
    
    probs_matrix = torch.stack([torch.tensor(probs) if isinstance(probs, list) else probs for probs in probs_list])
    
    obj_entropy = -torch.sum(obj_probs_matrix * torch.log(obj_probs_matrix + 1e-8), dim=1)
    probs_entropy = -torch.sum(probs_matrix * torch.log(probs_matrix + 1e-8), dim=1)
    
    obj_probs_matrix = obj_probs_matrix / (torch.sum(obj_probs_matrix, dim=1, keepdim=True) + 1e-8)
    probs_matrix = probs_matrix / (torch.sum(probs_matrix, dim=1, keepdim=True) + 1e-8)
    
    def sharpen_distribution(probs, sharpening_factor=2.0):
        """Sharpen a probability distribution."""
        sharpened = torch.pow(probs, sharpening_factor)
        return sharpened / (torch.sum(sharpened, dim=1, keepdim=True) + 1e-8)
    
    entropy_threshold = 2.8
    if probs_entropy.mean() > entropy_threshold:
        probs_matrix = sharpen_distribution(probs_matrix, sharpening_factor=2.5)
    
    if obj_entropy.mean() > entropy_threshold:
        obj_probs_matrix = sharpen_distribution(obj_probs_matrix, sharpening_factor=1.8)
    
    obj_probs_norm = torch.nn.functional.normalize(obj_probs_matrix, p=2, dim=1)
    probs_norm = torch.nn.functional.normalize(probs_matrix, p=2, dim=1)
    cosine_sim_matrix = torch.mm(obj_probs_norm, probs_norm.T)  # (M, N)
    
    obj_probs_expanded = obj_probs_matrix.unsqueeze(1)  # (M, 1, vocab_size)
    probs_expanded = probs_matrix.unsqueeze(0)  # (1, N, vocab_size)
    
    kl_pq = torch.sum(obj_probs_expanded * torch.log(obj_probs_expanded / (probs_expanded + 1e-8) + 1e-8), dim=2)
    kl_qp = torch.sum(probs_expanded * torch.log(probs_expanded / (obj_probs_expanded + 1e-8) + 1e-8), dim=2)
    js_div_matrix = 0.5 * (kl_pq + kl_qp)
    
    sensitivity_factor = 5.0
    js_sim_matrix = torch.exp(-sensitivity_factor * js_div_matrix)
    
    obj_max_indices = torch.argmax(obj_probs_matrix, dim=1)
    probs_max_indices = torch.argmax(probs_matrix, dim=1)
    obj_max_probs = torch.max(obj_probs_matrix, dim=1)[0]
    probs_max_probs = torch.max(probs_matrix, dim=1)[0]
    
    class_match_matrix = (obj_max_indices.unsqueeze(1) == probs_max_indices.unsqueeze(0)).float()
    
    prob_weight_matrix = torch.outer(obj_max_probs, probs_max_probs)
    
    class_weighted_sim = class_match_matrix * prob_weight_matrix
    
    
    obj_entropies = -torch.sum(obj_probs_matrix * torch.log(obj_probs_matrix + 1e-8), dim=1)
    probs_entropies = -torch.sum(probs_matrix * torch.log(probs_matrix + 1e-8), dim=1)
    max_entropy = np.log(obj_probs_matrix.shape[1])
    
    
    entropy_threshold = max_entropy * 0.9
    
    if obj_entropies.mean() > entropy_threshold or probs_entropies.mean() > entropy_threshold:
        if obj_entropies.mean() > entropy_threshold:
            obj_probs_matrix = torch.pow(obj_probs_matrix, 2.0)
            obj_probs_matrix = obj_probs_matrix / (torch.sum(obj_probs_matrix, dim=1, keepdim=True) + 1e-8)
            
        if probs_entropies.mean() > entropy_threshold:
            probs_matrix = torch.pow(probs_matrix, 2.0)
            probs_matrix = probs_matrix / (torch.sum(probs_matrix, dim=1, keepdim=True) + 1e-8)
        
        obj_probs_norm = torch.nn.functional.normalize(obj_probs_matrix, p=2, dim=1)
        probs_norm = torch.nn.functional.normalize(probs_matrix, p=2, dim=1)
        cosine_sim_matrix = torch.mm(obj_probs_norm, probs_norm.T)
        
        obj_probs_expanded = obj_probs_matrix.unsqueeze(1)
        probs_expanded = probs_matrix.unsqueeze(0)
        kl_pq = torch.sum(obj_probs_expanded * torch.log(obj_probs_expanded / (probs_expanded + 1e-8) + 1e-8), dim=2)
        kl_qp = torch.sum(probs_expanded * torch.log(probs_expanded / (obj_probs_expanded + 1e-8) + 1e-8), dim=2)
        js_div_matrix = 0.5 * (kl_pq + kl_qp)
        js_sim_matrix = torch.exp(-5.0 * js_div_matrix)
        
    
    visual_sim_matrix = 0.3 * cosine_sim_matrix + 0.4 * js_sim_matrix + 0.3 * class_weighted_sim
    
    
    return visual_sim_matrix.numpy()


def compute_visual_similarity_matrix(object_nodes, aggregate_features, probs_list=None, 
                                    use_clip=True, debug=False):
    """Compute pairwise visual similarity using the configured representation."""
    if use_clip:
        return compute_visual_similarity_matrix_clip(object_nodes, aggregate_features, debug=debug)
    else:
        if probs_list is None:
            raise ValueError("The belief method requires probs_list")
        return compute_visual_similarity_matrix_belief(object_nodes, aggregate_features, probs_list, debug=debug)


def compute_center_distance_similarity(object_nodes, aggregate_features, max_distance=2.0):
    """Compare object center distances for compatibility with older callers."""
    m = len(object_nodes)
    n = len(aggregate_features)
    
    if m == 0 or n == 0:
        return np.zeros((m, n))
    
    distance_sim_matrix = np.zeros((m, n))
    
    for i, obj_node in enumerate(object_nodes):
        for j, agg_feat in enumerate(aggregate_features):
            center_dist = np.linalg.norm(np.array(obj_node.center) - np.array(agg_feat['center']))
            
            distance_sim = np.exp(-center_dist / max_distance)
            distance_sim_matrix[i, j] = distance_sim
    
    return distance_sim_matrix


def compute_min_point_distance_similarity(min_distance_matrix, max_distance=1.0):
    """Compute spatial similarity from minimum point distances."""
    distance_sim_matrix = np.exp(-min_distance_matrix / max_distance)
    
    return distance_sim_matrix


def compute_aggregate_similarities(overlap_sim, distance_sim, visual_sim, 
                                 object_nodes, aggregate_features,
                                 min_distance_matrix=None,
                                 overlap_weight=0.7, visual_weight=0.3,
                                 max_match_distance=2.0,
                                 distance_threshold=0.3,
                                 distance_penalty_strength=5.0,
                                 high_visual_boost_enabled=True,
                                 high_visual_threshold=0.7,
                                 high_visual_distance_threshold=0.3,
                                 high_visual_boost_factor=0.5,
                                 debug=False):
    """Combine similarity measures subject to matching thresholds."""
    m, n = overlap_sim.shape
    
    if min_distance_matrix is not None:
        distance_for_penalty = min_distance_matrix
    else:
        center_distance_matrix = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                center_distance_matrix[i, j] = np.linalg.norm(
                    np.array(object_nodes[i].center) - np.array(aggregate_features[j]['center'])
                )
        distance_for_penalty = center_distance_matrix
    
    base_similarity = overlap_weight * overlap_sim + visual_weight * visual_sim
    
    if high_visual_boost_enabled:
        high_visual_mask = visual_sim >= high_visual_threshold
        close_distance_mask = distance_for_penalty <= high_visual_distance_threshold
        boost_mask = high_visual_mask & close_distance_mask
        base_similarity = np.where(boost_mask, base_similarity + high_visual_boost_factor, base_similarity)
    
    distance_penalty = np.where(
        distance_for_penalty <= distance_threshold,
        1.0,
        np.exp(-distance_penalty_strength * (distance_for_penalty - distance_threshold) / (max_match_distance - distance_threshold))
    )
    
    smart_sim_matrix = base_similarity * distance_penalty
    
    smart_sim_matrix = np.where(
        distance_for_penalty > max_match_distance, 
        0.0, 
        smart_sim_matrix
    )
    
    smart_sim_matrix = np.where(
        (distance_for_penalty > distance_threshold) & (overlap_sim < 0.05) & (visual_sim < 0.3),
        0.0,
        smart_sim_matrix
    )
    
    
    return smart_sim_matrix


def aggregate_similarities(spatial_sim, visual_sim, spatial_weight=0.7, visual_weight=0.3):
    """Linearly combine similarity scores."""
    return spatial_weight * spatial_sim + visual_weight * visual_sim


def filter_contained_pointclouds(feature_list, probs_list=None, proximity_thresh=0.05, 
                                sample_size=512, visual_sim_threshold=0.7, debug=False):
    """Filter smaller point clouds contained within other detections in this frame."""
    n = len(feature_list)
    if n <= 1:
        return feature_list
    
    indices = []
    sampled_points_list = []
    point_counts = []
    
    for obj_feature in feature_list:
        pcd = obj_feature['pcd']
        points = np.asarray(pcd.points, dtype=np.float32)
        
        if points.shape[0] < 10:
            indices.append(None)
            sampled_points_list.append(np.zeros((0, 3), dtype=np.float32))
            point_counts.append(0)
            continue
        
        point_counts.append(points.shape[0])
        
        current_sample_size = min(points.shape[0], sample_size)
        sampled_points = points[np.random.choice(points.shape[0], current_sample_size, replace=False)]
        sampled_points_list.append(sampled_points)
        
        index = faiss.IndexFlatL2(points.shape[1])
        index.add(points)
        indices.append(index)
    
    proximity_thresh_sq = proximity_thresh ** 2
    
    overlap_matrix = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        if indices[i] is None or sampled_points_list[i].shape[0] == 0:
            continue
        
        for j in range(n):
            if i == j or indices[j] is None:
                continue
            
            query_pts = sampled_points_list[i]
            distances, _ = indices[j].search(query_pts, 1)
            inliers = np.sum(distances < proximity_thresh_sq)
            overlap_matrix[i, j] = inliers / query_pts.shape[0]
    
    to_remove = set()
    removal_info = []
    
    for i in range(n):
        if i in to_remove:
            continue
        
        for j in range(n):
            if i == j or j in to_remove:
                continue
            
            if abs(overlap_matrix[i, j] - 1.0) < 1e-6:
                visual_sim = None
                should_remove = False
                skip_reason = ""
                
                has_clip = (feature_list[i].get('clip_embedding') is not None and 
                           feature_list[j].get('clip_embedding') is not None)
                
                if has_clip:
                    clip_emb_i = torch.tensor(feature_list[i]['clip_embedding']) if isinstance(feature_list[i]['clip_embedding'], np.ndarray) else feature_list[i]['clip_embedding']
                    clip_emb_j = torch.tensor(feature_list[j]['clip_embedding']) if isinstance(feature_list[j]['clip_embedding'], np.ndarray) else feature_list[j]['clip_embedding']
                    
                    clip_emb_i = F.normalize(clip_emb_i.unsqueeze(0), p=2, dim=1)
                    clip_emb_j = F.normalize(clip_emb_j.unsqueeze(0), p=2, dim=1)
                    
                    visual_sim = F.cosine_similarity(clip_emb_i, clip_emb_j, dim=1).item()
                    
                    smaller_idx = i if point_counts[i] < point_counts[j] else j
                    if feature_list[smaller_idx]['class_name'] == 'cushion':
                        should_remove = False
                        skip_reason = f"Keep the smaller cushion cloud (CLIP similarity={visual_sim:.3f})"
                    elif visual_sim >= visual_sim_threshold:
                        should_remove = True
                        skip_reason = f"CLIP similarity {visual_sim:.3f} >= {visual_sim_threshold:.3f}: same object"
                    else:
                        should_remove = False
                        skip_reason = f"CLIP similarity {visual_sim:.3f} < {visual_sim_threshold:.3f}: distinct objects"
                else:
                    should_remove = False
                    skip_reason = "No CLIP features; skipping filter"
                    visual_sim = 0.0
                
                if should_remove:
                    if point_counts[i] < point_counts[j]:
                        to_remove.add(i)
                        removal_info.append((i, j, overlap_matrix[i, j], visual_sim, True, skip_reason))
                        break
                    elif point_counts[j] < point_counts[i]:
                        to_remove.add(j)
                        removal_info.append((j, i, overlap_matrix[j, i], visual_sim, True, skip_reason))
                else:
                    removal_info.append((i, j, overlap_matrix[i, j], visual_sim, False, skip_reason))
    
    if debug and len(removal_info) > 0:
        has_clip_global = any(feat.get('clip_embedding') is not None for feat in feature_list)
        method_name = "CLIP embeddings" if has_clip_global else "DINO belief"
        
        print(f"\n{'='*80}")
        print(f"[Frame containment] Checked {n} detections and found {len(removal_info)} containment pairs")
        print(f"Visual similarity method: {method_name}")
        print(f"{'='*80}")
        
        removed_count = 0
        kept_count = 0
        
        for removed_idx, container_idx, overlap_val, visual_sim, was_removed, reason in removal_info:
            removed_obj = feature_list[removed_idx]
            container_obj = feature_list[container_idx]
            removed_center = removed_obj.get('center', [0, 0, 0])
            container_center = container_obj.get('center', [0, 0, 0])
            
            if was_removed:
                removed_count += 1
                status = "Filtered"
            else:
                kept_count += 1
                status = "Retained"
            
            print(f"\n[{status}] Detection {removed_idx} [{removed_obj['class_name']}] is contained in detection {container_idx} [{container_obj['class_name']}]")
            print(f"  Contained cloud {removed_idx}:")
            print(f"    Class: {removed_obj['class_name']}")
            print(f"    Points: {point_counts[removed_idx]}")
            print(f"    Center: ({removed_center[0]:.3f}, {removed_center[1]:.3f}, {removed_center[2]:.3f}) m")
            print(f"  Containing cloud {container_idx}:")
            print(f"    Class: {container_obj['class_name']}")
            print(f"    Points: {point_counts[container_idx]}")
            print(f"    Center: ({container_center[0]:.3f}, {container_center[1]:.3f}, {container_center[2]:.3f}) m")
            print(f"  Directional overlap [{removed_idx} -> {container_idx}]: {overlap_val:.4f} (approx. 1.0)")
            print(f"  Visual similarity: {visual_sim:.3f}")
            print(f"  Decision: {reason}")
        
        print(f"\nSummary: removed {removed_count} clouds, retained {kept_count}")
        print(f"Remaining detections: {len(feature_list) - len(to_remove)}")
        print(f"{'='*80}\n")
    
    filtered_feature_list = [
        feature_list[i] for i in range(n) if i not in to_remove
    ]
    
    return filtered_feature_list


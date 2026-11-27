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
#     计算两组物体之间的重叠矩阵，优先使用类别匹配，然后用几何重叠度验证。

#     Args:
#         objects_old (list): 已存在的对象列表 (包含 pcd, center)。
#         objects_new (list): 新检测到的对象列表 (包含 pcd, center)。
#         class_names_old (list): 已存在对象的类别名称列表。
#         class_names_new (list): 新检测到对象的类别名称列表。
#         proximity_thresh (float): 判断点是否为inlier的距离阈值(米)。
#         center_dist_thresh (float): 中心点最大距离的预过滤阈值(米)。
#         min_iou_thresh (float): 3D BBox IoU的最小预过滤阈值。

#     Returns:
#         np.ndarray: (m, n) 的重叠度矩阵。
#     """
#     m = len(objects_old)
#     n = len(objects_new)
    
#     if m == 0 or n == 0:
#         return np.zeros((m, n))

#     overlap_matrix = np.zeros((m, n))
    
#     # --- 步骤 1: 数据预处理 ---
#     # 提取点云和中心点
#     points_old = [np.asarray(obj.pcd.points, dtype=np.float32) for obj in objects_old]
#     centers_old = np.array([obj.center for obj in objects_old])
    
#     points_new = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects_new]
#     centers_new = np.array([obj['center'] for obj in objects_new])

#     # --- 步骤 2: 高效预过滤 ---
#     # 2a. 3D BBox IoU (如果可用且高效)
#     # 你的BBox IoU计算逻辑可以保留在这里
#     # ...

#     # 2b. 中心点距离过滤
#     center_dists = np.linalg.norm(centers_old[:, np.newaxis, :] - centers_new[np.newaxis, :, :], axis=2)

#     # --- 步骤 3: 优化后的对称重叠度计算 ---
#     # 预先为所有对象构建FAISS索引，避免在循环中重复构建
#     def build_faiss_index(points_list, sample_size=1024):
#         indices = []
#         for arr in points_list:
#             if arr.shape[0] < 10: # 点太少，无法建立有效索引
#                 indices.append(None)
#                 continue
            
#             # 使用Voxel Down-sampling是比随机采样更鲁棒的选择，因为它能更好地保持点云的形状结构
#             # from open3d.geometry import PointCloud
#             # pcd = PointCloud()
#             # pcd.points = o3d.utility.Vector3dVector(arr)
#             # sampled_pcd = pcd.voxel_down_sample(voxel_size=0.05) # 5cm体素下采样
#             # sampled_points = np.asarray(sampled_pcd.points, dtype=np.float32)
#             # if sampled_points.shape[0] == 0:
#             #     indices.append(None)
#             #     continue
            
#             # 或者继续使用随机采样，但可以增加采样点数
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
            
#             # 应用几何预过滤
#             if center_dists[i, j] > center_dist_thresh:
#                 continue
            
#             # 获取采样后的点用于查询 (注意：查询用的点也应该被采样，以保证计算效率)
#             pts_to_query_from_i = points_old[i][np.random.choice(points_old[i].shape[0], min(points_old[i].shape[0], 512), replace=False)]
#             pts_to_query_from_j = points_new[j][np.random.choice(points_new[j].shape[0], min(points_new[j].shape[0], 512), replace=False)]

#             # 计算 new -> old 的重叠率
#             distances_j, _ = indices_old[i].search(pts_to_query_from_j, 1)
#             inliers_j = np.sum(distances_j < proximity_thresh_sq)
#             overlap_j_to_i = inliers_j / len(pts_to_query_from_j)

#             # 计算 old -> new 的重叠率 (使用预先建立的索引)
#             distances_i, _ = indices_new[j].search(pts_to_query_from_i, 1)
#             inliers_i = np.sum(distances_i < proximity_thresh_sq)
#             overlap_i_to_j = inliers_i / len(pts_to_query_from_i)
            
#             # 使用两个重叠率中的最大值，这对于部分观测情况更鲁棒
#             overlap_matrix[i, j] = max(overlap_i_to_j, overlap_j_to_i)
#     return overlap_matrix


def _prepare_object_data(objects, sample_size=512, voxel_size=0.05, use_voxel_downsampling=False):
    indices = []
    sampled_points_list = []
    
    for obj in objects:
        # 在某些情况下，obj可能是字典，而在另一些情况下是对象
        # 我们需要统一处理
        if isinstance(obj, dict):
            pcd = obj['pcd']
        else:
            pcd = obj.pcd
            
        points = np.asarray(pcd.points, dtype=np.float32)

        if points.shape[0] < 10:
            indices.append(None)
            sampled_points_list.append(np.zeros((0, 3), dtype=np.float32))
            continue
        
        # --- 采样 ---
        # 体素下采样通常比随机采样更能保持点云的几何结构，结果更稳定
        if use_voxel_downsampling:
            o3d_pcd = o3d.geometry.PointCloud()
            o3d_pcd.points = o3d.utility.Vector3dVector(points)
            sampled_pcd = o3d_pcd.voxel_down_sample(voxel_size=voxel_size)
            sampled_points = np.asarray(sampled_pcd.points, dtype=np.float32)
            if sampled_points.shape[0] == 0: # 如果下采样后点没了，回退到原始点
                 sampled_points = points
        else:
            # 随机采样作为备选方案
            current_sample_size = min(points.shape[0], sample_size)
            sampled_points = points[np.random.choice(points.shape[0], current_sample_size, replace=False)]

        sampled_points_list.append(sampled_points)
        
        # --- 构建FAISS索引 ---
        # 注意：索引是建立在“被查询”的点云上的
        index = faiss.IndexFlatL2(points.shape[1])
        index.add(points) # 使用原始点云构建索引以获得最高精度
        indices.append(index)
        
    return indices, sampled_points_list


def compute_overlap_matrix_2set(
    objects_old, 
    objects_new, 
    proximity_thresh=0.05,
    center_dist_thresh=5.0,
):
    """
    计算两组对象之间的重叠度矩阵和最小点距离矩阵
    
    Returns:
        overlap_matrix: (m, n) 重叠相似度矩阵
        min_distance_matrix: (m, n) 最小点距离矩阵（米）
    """
    m = len(objects_old)
    n = len(objects_new)
    
    if m == 0 or n == 0:
        return np.zeros((m, n)), np.full((m, n), np.inf)

    overlap_matrix = np.zeros((m, n), dtype=np.float32)
    min_distance_matrix = np.full((m, n), np.inf, dtype=np.float32)  # 初始化为无穷大
    
    # --- 步骤 1: 统一数据提取 ---
    # 假设 object 可以是 dict 或 class instance
    centers_old = np.array([obj.center if hasattr(obj, 'center') else obj['center'] for obj in objects_old])
    centers_new = np.array([obj['center'] if isinstance(obj, dict) else obj.center for obj in objects_new])

    # --- 步骤 2: 高效预过滤 ---
    # 向量化计算所有对之间的中心距离
    center_dists = np.linalg.norm(centers_old[:, np.newaxis, :] - centers_new[np.newaxis, :, :], axis=2)
    # 创建一个布尔掩码，标记所有潜在的匹配对
    candidate_mask = (center_dists <= center_dist_thresh)

    # --- 步骤 3: 预计算FAISS索引和采样点 (核心优化) ---
    # 为两组对象分别准备索引和用于查询的采样点
    # 注意：查询点使用采样点以提高速度，而索引基于原始点以保证精度
    indices_old, sampled_points_to_query_old = _prepare_object_data(objects_old)
    indices_new, sampled_points_to_query_new = _prepare_object_data(objects_new)
    
    proximity_thresh_sq = proximity_thresh**2

    # --- 步骤 4: 优化后的对称重叠度计算 ---
    # 使用 np.argwhere 仅遍历通过了中心距离过滤的 (i, j) 对
    candidate_indices = np.argwhere(candidate_mask)
    
    for i, j in candidate_indices:
        # 检查索引和采样点是否有效
        if indices_old[i] is None or indices_new[j] is None:
            continue
        
        # 初始化最小距离列表
        min_distances = []
        
        query_pts_from_j = sampled_points_to_query_new[j]
        if query_pts_from_j.shape[0] == 0:
            overlap_j_to_i = 0.0
        else:
            # 计算 new -> old 的重叠率
            # 从 new 中采样点，在 old 的索引中查询
            distances_j, _ = indices_old[i].search(query_pts_from_j, 1)
            inliers_j = np.sum(distances_j < proximity_thresh_sq)
            overlap_j_to_i = inliers_j / query_pts_from_j.shape[0]
            # 记录最小距离（转换为实际距离，FAISS返回的是平方距离）
            min_distances.append(np.sqrt(distances_j.min()))

        query_pts_from_i = sampled_points_to_query_old[i]
        if query_pts_from_i.shape[0] == 0:
            overlap_i_to_j = 0.0
        else:
            # 计算 old -> new 的重叠率
            # 从 old 中采样点，在 new 的索引中查询
            distances_i, _ = indices_new[j].search(query_pts_from_i, 1)
            inliers_i = np.sum(distances_i < proximity_thresh_sq)
            overlap_i_to_j = inliers_i / query_pts_from_i.shape[0]
            # 记录最小距离
            min_distances.append(np.sqrt(distances_i.min()))
            
        # 使用两个重叠率中的最大值，这对于部分观测情况更鲁棒
        overlap_matrix[i, j] = max(overlap_i_to_j, overlap_j_to_i)
        
        # 使用两个方向中的最小距离
        if min_distances:
            min_distance_matrix[i, j] = min(min_distances)
            
    return overlap_matrix, min_distance_matrix


def _prepare_color_histograms(objects, bins=(16, 8)):
    """
    为一组对象预先计算并归一化其颜色直方图。
    这是一个辅助函数，用于优化主计算流程。

    Args:
        objects (list): 对象列表，每个元素需要有 `pcd` 属性。
        bins (tuple): 用于(Hue, Saturation)的2D直方图的箱数。

    Returns:
        list: 归一化后的颜色直方图列表。如果某个对象无效，则对应位置为None。
    """
    histograms = []
    epsilon = 1e-10  # 防止除以零

    for obj in objects:
        pcd = obj.pcd if hasattr(obj, 'pcd') else obj['pcd']
        
        # 验证点云和颜色
        if not pcd.has_points() or not pcd.has_colors():
            histograms.append(None)
            continue
            
        colors = np.asarray(pcd.colors)
        if colors.shape[0] == 0:
            histograms.append(None)
            continue

        # 转换到HSV并提取H, S通道
        hsv = color.rgb2hsv(colors[:, np.newaxis, :])[:, 0, :]
        h, s = hsv[:, 0], hsv[:, 1]
        
        # 创建计数直方图
        hist_counts, _, _ = np.histogram2d(h, s, bins=bins, range=[[0, 1], [0, 1]])
        
        # 归一化为概率分布
        total_sum = np.sum(hist_counts)
        prob_dist = hist_counts / (total_sum + epsilon)
        
        histograms.append(prob_dist)
        
    return histograms

def _find_closest_point_in_2d_projection(pcd, target_xy, r_resolution=20):
    """
    找到点云中离目标2D点最近的点
    
    Args:
        pcd: 点云对象
        target_xy: 目标2D坐标 (像素坐标)
        r_resolution: 分辨率 (pixel/meter)，默认20对应5cm/pixel
    
    Returns:
        closest_2d: 最近点的2D像素坐标
    """
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
    """
    计算两组物体之间的颜色相似度矩阵（优化版）。

    该版本模仿几何重叠计算的优化策略：
    1.  **向量化过滤**: 使用NumPy向量化操作预先计算所有对之间的中心距离，并创建一个布尔掩码。
    2.  **预计算**: 在主循环外一次性完成所有点云的颜色直方图计算和归一化。
    3.  **高效迭代**: 使用`np.argwhere`仅迭代那些通过了预过滤的候选对。
    
    Args:
        objects_old (list): 已存在的对象列表 (每个元素应包含 `pcd` 和 `center` 属性/键)。
        objects_new (list): 新检测到的对象列表 (每个元素应包含 `pcd` 和 `center` 属性/键)。
        center_dist_thresh (float): 中心点最大距离的预过滤阈值(米)。
        bins (tuple): 用于颜色直方图的(Hue, Saturation)箱数。

    Returns:
        np.ndarray: (m, n) 的颜色相似度矩阵。
    """
    m = len(objects_old)
    n = len(objects_new)
    
    if m == 0 or n == 0:
        return np.zeros((m, n))

    color_sim_matrix = np.zeros((m, n), dtype=np.float32)
    
    # --- 步骤 1: 统一数据提取 (与原函数相同) ---
    centers_old = np.array([obj.center if hasattr(obj, 'center') else obj['center'] for obj in objects_old])
    centers_new = np.array([obj.center if hasattr(obj, 'center') else obj['center'] for obj in objects_new])

    # --- 步骤 2: 高效预过滤 (与原函数相同) ---
    # 只有空间上接近的物体才需要计算颜色相似度
    center_dists = np.linalg.norm(centers_old[:, np.newaxis, :] - centers_new[np.newaxis, :, :], axis=2)
    candidate_mask = (center_dists <= center_dist_thresh)

    # --- 步骤 3: 预计算颜色直方图 (核心优化) ---
    histograms_old = _prepare_color_histograms(objects_old, bins=bins)
    histograms_new = _prepare_color_histograms(objects_new, bins=bins)
    
    # --- 步骤 4: 优化后的颜色相似度计算 ---
    # 使用 np.argwhere 仅遍历通过了中心距离过滤的 (i, j) 对
    candidate_indices = np.argwhere(candidate_mask)
    
    for i, j in candidate_indices:
        hist_old = histograms_old[i]
        hist_new = histograms_new[j]
        
        # 检查预计算的直方图是否有效
        if hist_old is None or hist_new is None:
            continue
            
        # 计算Bhattacharyya系数，这是一个非常快速的操作
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
#     计算两个点云之间的空间相似度（双向平均覆盖率）
    
#     参数:
#         pcl1 (o3d.geometry.PointCloud): 第一个点云
#         pcl2 (o3d.geometry.PointCloud): 第二个点云
#         max_num_points (int): 最大采样点数（用于降采样）
#         distance_threshold (float): 判定为邻近点的距离阈值（单位：米）
        
#     返回:
#         float: 空间相似度值（0~1之间）
#     """
#     # 转换为numpy数组并降采样
#     points1 = np.asarray(pcl1.points, dtype=np.float32)
#     points2 = np.asarray(pcl2.points, dtype=np.float32)

#     # 降采样处理
#     if len(points1) > max_num_points:
#         indices = np.random.choice(len(points1), max_num_points, replace=False)
#         points1 = points1[indices]
#     if len(points2) > max_num_points:
#         indices = np.random.choice(len(points2), max_num_points, replace=False)
#         points2 = points2[indices]

#     # 空点云检查
#     if len(points1) == 0 or len(points2) == 0:
#         return 0.0

#     # 快速包围盒检查
#     min1, max1 = np.min(points1, axis=0), np.max(points1, axis=0)
#     min2, max2 = np.min(points2, axis=0), np.max(points2, axis=0)
    
#     if np.any((max1 < min2) | (max2 < min1)):
#         return 0.0

#     # 创建Faiss索引
#     def create_index(points):
#         index = faiss.IndexFlatL2(points.shape[1])
#         index.add(points)
#         return index

#     # 双向查询
#     threshold_sq = distance_threshold ** 2
    
#     # 计算pcl2在pcl1中的覆盖率
#     index1 = create_index(points1)
#     dist1, _ = index1.search(points2, 1)
#     ratio_pcl2_to_pcl1 = np.mean(dist1 < threshold_sq)

#     # 计算pcl1在pcl2中的覆盖率
#     index2 = create_index(points2)
#     dist2, _ = index2.search(points1, 1)
#     ratio_pcl1_to_pcl2 = np.mean(dist2 < threshold_sq)

#     # 取双向平均作为最终相似度
#     return (ratio_pcl2_to_pcl1 + ratio_pcl1_to_pcl2) / 2.0

# def compute_cosine_similarity(vec1, vec2):
#     # 类型转换：numpy array -> torch tensor
#     t1 = torch.from_numpy(vec1).float() if isinstance(vec1, np.ndarray) else vec1.float()
#     t2 = torch.from_numpy(vec2).float() if isinstance(vec2, np.ndarray) else vec2.float()
    
#     # 维度处理：确保形状为 (1, dim)
#     if t1.dim() == 1:
#         t1 = t1.unsqueeze(0)  # 添加批次维度
#     if t2.dim() == 1:
#         t2 = t2.unsqueeze(0)
    
#     # 设备检查（可选，如果使用GPU需要统一设备）
#     if t1.device != t2.device:
#         t2 = t2.to(t1.device)
    
#     return F.cosine_similarity(t1, t2, dim=1).item()  # 返回标量值


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

    # 2. 体素下采样以保持点云规模可控
    downsampled_pcd = merged_pcd.voxel_down_sample(voxel_size)
    
    # 3. 重新计算中心点
    if not downsampled_pcd.is_empty():
        new_center = downsampled_pcd.get_center().tolist()
    else:
        # 如果下采样后点云为空，则取两个中心点的中点作为备用策略
        center_old = pcd_old.get_center()
        center_new = pcd_new.get_center()
        new_center = ((center_old + center_new) / 2).tolist()
        
    return downsampled_pcd, new_center

def merge_pcd_batch(object_list, voxel_size=0.01, default_color=(1.0, 1.0, 1.0)):
    # 校验输入有效性
    if not object_list:
        raise ValueError("输入点云列表不能为空")
    
    # 初始化合并点云
    merged_pcd = o3d.geometry.PointCloud()
    
    # 合并所有点云数据
    all_points = []
    all_colors = []
    
    for obj in object_list:
        pcd = obj.pcd
        # 转换点坐标
        points = np.asarray(pcd.points)
        all_points.append(points)
        
        # 处理颜色信息
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
        else:
            colors = np.tile(default_color, (points.shape[0], 1))
        all_colors.append(colors)
    
    # 拼接所有点云数据
    merged_pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    merged_pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
    
    if len(merged_pcd.points) == 0:
        return merged_pcd  # 直接返回空点云，跳过下采样
    
    # 执行体素下采样
    downsampled_pcd = merged_pcd.voxel_down_sample(voxel_size)
    
    # 打印统计信息（新增部分）
    if len(downsampled_pcd.points) > 0:
        points = np.asarray(downsampled_pcd.points)
        # 计算中心坐标
        center = np.mean(points, axis=0)
        # 计算XYZ方向尺寸
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        x_size = max_coords[0] - min_coords[0]
        y_size = max_coords[1] - min_coords[1]
            
    return downsampled_pcd

def volume_ratio_voxel_gen(pcd_local, pcd_overall, voxel_size=0.01):
    voxel_grid1 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_local, voxel_size=voxel_size)
    num_voxels1 = len(voxel_grid1.get_voxels())

    # 将 pcd2 转换为体素
    voxel_grid2 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_overall, voxel_size=voxel_size)
    num_voxels2 = len(voxel_grid2.get_voxels())

    # 计算并返回体积比
    if num_voxels2 == 0:
        return 0.0
    # 可以直接用体素数量比，因为 voxel_size^3 会被约掉
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
    """
    基于概率分布计算视觉相似度
    
    Args:
        probs1, probs2: 两个物体的概率分布 (torch.Tensor)
        
    Returns:
        visual_similarity: 视觉相似度分数 (0-1之间)
    """
    if isinstance(probs1, list):
        probs1 = torch.tensor(probs1)
    if isinstance(probs2, list):
        probs2 = torch.tensor(probs2)
        
    # 确保概率分布是归一化的
    probs1 = torch.softmax(probs1, dim=0)
    probs2 = torch.softmax(probs2, dim=0)
    
    # 使用KL散度的倒数作为相似度度量
    # KL散度越小，相似度越高
    kl_div = torch.sum(probs1 * torch.log(probs1 / (probs2 + 1e-8) + 1e-8))
    
    # 将KL散度转换为相似度分数 (0-1之间)
    # 使用指数函数将KL散度映射到相似度
    visual_similarity = torch.exp(-kl_div).item()
    
    return visual_similarity


def compute_visual_similarity_matrix_clip(object_nodes, aggregate_features, debug=False):
    """
    基于CLIP embeddings计算视觉相似度矩阵（真正的视觉特征）
    
    参考: concept-graphs的实现
    
    Args:
        object_nodes: 现有物体节点列表，每个节点包含clip_embedding属性
        aggregate_features: 新检测到的物体特征列表，每个包含'clip_embedding'
        debug: 是否输出调试信息
        
    Returns:
        visual_sim_matrix: 视觉相似度矩阵 (M x N)，范围[-1, 1]
    """
    m = len(object_nodes)
    n = len(aggregate_features)
    
    if m == 0 or n == 0:
        return np.zeros((m, n))
    
    # 提取CLIP embeddings
    # object_nodes中的embeddings: 已归一化的CLIP特征
    obj_embeddings = torch.stack([
        torch.tensor(obj_node.clip_embedding) if isinstance(obj_node.clip_embedding, np.ndarray) 
        else obj_node.clip_embedding
        for obj_node in object_nodes
    ])  # (M, clip_dim)
    
    # aggregate_features中的embeddings
    det_embeddings = torch.stack([
        torch.tensor(feat['clip_embedding']) if isinstance(feat['clip_embedding'], np.ndarray)
        else feat['clip_embedding']
        for feat in aggregate_features
    ])  # (N, clip_dim)
    
    # 确保归一化（CLIP标准做法）
    obj_embeddings = F.normalize(obj_embeddings, p=2, dim=1)
    det_embeddings = F.normalize(det_embeddings, p=2, dim=1)
    
    # 计算余弦相似度（与concept-graphs一致）
    # 使用矩阵乘法高效计算
    det_embeddings_t = det_embeddings.unsqueeze(-1)  # (N, clip_dim, 1)
    obj_embeddings_t = obj_embeddings.T.unsqueeze(0)  # (1, clip_dim, M)
    
    visual_sim_matrix = F.cosine_similarity(det_embeddings_t, obj_embeddings_t, dim=1)  # (N, M)
    visual_sim_matrix = visual_sim_matrix.T  # 转置为 (M, N)
    
    if debug:
        print(f"\n=== CLIP视觉相似度矩阵调试 ===")
        print(f"物体数量: {m}, 检测数量: {n}")
        print(f"CLIP embedding维度: {obj_embeddings.shape[1]}")
        print(f"相似度范围: [{visual_sim_matrix.min():.3f}, {visual_sim_matrix.max():.3f}]")
        print(f"相似度均值: {visual_sim_matrix.mean():.3f}")
        
        # 显示高相似度对
        threshold = 0.7
        high_sim_pairs = []
        for i in range(m):
            for j in range(n):
                if visual_sim_matrix[i, j] > threshold:
                    high_sim_pairs.append((i, j, visual_sim_matrix[i, j].item()))
        
        if high_sim_pairs:
            print(f"高相似度对 (>{threshold}): {len(high_sim_pairs)}个")
            for i, j, sim in high_sim_pairs[:10]:  # 只显示前10个
                obj_class = object_nodes[i].class_name
                det_class = aggregate_features[j]['class_name']
                print(f"  物体{i}[{obj_class}] ↔ 检测{j}[{det_class}]: {sim:.3f}")
        print("=" * 50)
    
    return visual_sim_matrix.numpy()


# ===== 原来基于DINO belief的方法（注释保留） =====
def compute_visual_similarity_matrix_belief(object_nodes, aggregate_features, probs_list, debug=False):
    """
    基于DINO belief概率分布计算"类别相似度"矩阵
    
    ⚠️ 注意：这不是真正的视觉相似度！
    这是基于检测器的类别判断分数计算的相似度，无法捕获视觉外观、纹理、颜色等信息。
    
    Args:
        object_nodes: 现有物体节点列表
        aggregate_features: 新检测到的物体特征列表
        probs_list: 对应的概率分布列表
        debug: 是否输出调试信息
        
    Returns:
        visual_sim_matrix: "类别相似度"矩阵 (M x N)
    """
    m = len(object_nodes)
    n = len(aggregate_features)
    
    if m == 0 or n == 0:
        return np.zeros((m, n))
    
    # 提取所有物体节点的概率分布，构建矩阵 (M, vocab_size)
    obj_probs_matrix = torch.stack([obj_node.class_candidates for obj_node in object_nodes])
    
    # 转换probs_list为tensor矩阵 (N, vocab_size)
    probs_matrix = torch.stack([torch.tensor(probs) if isinstance(probs, list) else probs for probs in probs_list])
    
    # 计算熵用于后续处理
    obj_entropy = -torch.sum(obj_probs_matrix * torch.log(obj_probs_matrix + 1e-8), dim=1)
    probs_entropy = -torch.sum(probs_matrix * torch.log(probs_matrix + 1e-8), dim=1)
    
    # 归一化处理（保持原始分布的差异性）
    obj_probs_matrix = obj_probs_matrix / (torch.sum(obj_probs_matrix, dim=1, keepdim=True) + 1e-8)
    probs_matrix = probs_matrix / (torch.sum(probs_matrix, dim=1, keepdim=True) + 1e-8)
    
    # 处理概率分布过于平坦的问题
    # 对于熵过高（过于平坦）的分布，进行锐化处理
    def sharpen_distribution(probs, sharpening_factor=2.0):
        """锐化概率分布，减少过于平坦的分布"""
        sharpened = torch.pow(probs, sharpening_factor)
        return sharpened / (torch.sum(sharpened, dim=1, keepdim=True) + 1e-8)
    
    # 如果新检测的分布过于平坦（高熵），进行锐化
    entropy_threshold = 2.8
    if probs_entropy.mean() > entropy_threshold:
        probs_matrix = sharpen_distribution(probs_matrix, sharpening_factor=2.5)
    
    # 同样对现有物体分布进行适度锐化（如果需要）
    if obj_entropy.mean() > entropy_threshold:
        obj_probs_matrix = sharpen_distribution(obj_probs_matrix, sharpening_factor=1.8)
    
    # 方法1：余弦相似度（类似concept-graphs的CLIP特征相似度）
    obj_probs_norm = torch.nn.functional.normalize(obj_probs_matrix, p=2, dim=1)
    probs_norm = torch.nn.functional.normalize(probs_matrix, p=2, dim=1)
    cosine_sim_matrix = torch.mm(obj_probs_norm, probs_norm.T)  # (M, N)
    
    # 方法2：改进的KL散度（对称化Jensen-Shannon散度）
    obj_probs_expanded = obj_probs_matrix.unsqueeze(1)  # (M, 1, vocab_size)
    probs_expanded = probs_matrix.unsqueeze(0)  # (1, N, vocab_size)
    
    # 计算对称KL散度（Jensen-Shannon散度）
    kl_pq = torch.sum(obj_probs_expanded * torch.log(obj_probs_expanded / (probs_expanded + 1e-8) + 1e-8), dim=2)
    kl_qp = torch.sum(probs_expanded * torch.log(probs_expanded / (obj_probs_expanded + 1e-8) + 1e-8), dim=2)
    js_div_matrix = 0.5 * (kl_pq + kl_qp)
    
    # 使用可调的敏感度因子，降低敏感度让相同类别更容易匹配
    sensitivity_factor = 5.0  # 降低敏感度，让JS散度相似度更高
    js_sim_matrix = torch.exp(-sensitivity_factor * js_div_matrix)
    
    # 方法3：基于主要类别匹配的相似度
    obj_max_indices = torch.argmax(obj_probs_matrix, dim=1)  # (M,) 最大概率的类别索引
    probs_max_indices = torch.argmax(probs_matrix, dim=1)  # (N,) 最大概率的类别索引
    obj_max_probs = torch.max(obj_probs_matrix, dim=1)[0]  # (M,) 最大概率值
    probs_max_probs = torch.max(probs_matrix, dim=1)[0]  # (N,) 最大概率值
    
    # 类别匹配矩阵：如果最大概率的类别索引相同则为1，否则为0
    class_match_matrix = (obj_max_indices.unsqueeze(1) == probs_max_indices.unsqueeze(0)).float()
    
    # 基于概率强度的权重：两个物体的最大概率值的乘积
    prob_weight_matrix = torch.outer(obj_max_probs, probs_max_probs)
    
    # 加权类别相似度：类别匹配时为概率乘积，不匹配时为0
    class_weighted_sim = class_match_matrix * prob_weight_matrix
    
    # 基于概率分布的视觉相似度计算，采用锐化和连续计算
    
    # 分析概率分布特性
    obj_entropies = -torch.sum(obj_probs_matrix * torch.log(obj_probs_matrix + 1e-8), dim=1)
    probs_entropies = -torch.sum(probs_matrix * torch.log(probs_matrix + 1e-8), dim=1)
    max_entropy = np.log(obj_probs_matrix.shape[1])
    
    
    # 对于平坦分布，通过锐化增强区分度
    # 检测是否需要额外锐化
    entropy_threshold = max_entropy * 0.9  # 90%的最大熵
    
    if obj_entropies.mean() > entropy_threshold or probs_entropies.mean() > entropy_threshold:
        # 对过于平坦的分布进行额外锐化
        if obj_entropies.mean() > entropy_threshold:
            obj_probs_matrix = torch.pow(obj_probs_matrix, 2.0)
            obj_probs_matrix = obj_probs_matrix / (torch.sum(obj_probs_matrix, dim=1, keepdim=True) + 1e-8)
            
        if probs_entropies.mean() > entropy_threshold:
            probs_matrix = torch.pow(probs_matrix, 2.0)
            probs_matrix = probs_matrix / (torch.sum(probs_matrix, dim=1, keepdim=True) + 1e-8)
        
        # 重新计算相似度矩阵
        obj_probs_norm = torch.nn.functional.normalize(obj_probs_matrix, p=2, dim=1)
        probs_norm = torch.nn.functional.normalize(probs_matrix, p=2, dim=1)
        cosine_sim_matrix = torch.mm(obj_probs_norm, probs_norm.T)
        
        obj_probs_expanded = obj_probs_matrix.unsqueeze(1)
        probs_expanded = probs_matrix.unsqueeze(0)
        kl_pq = torch.sum(obj_probs_expanded * torch.log(obj_probs_expanded / (probs_expanded + 1e-8) + 1e-8), dim=2)
        kl_qp = torch.sum(probs_expanded * torch.log(probs_expanded / (obj_probs_expanded + 1e-8) + 1e-8), dim=2)
        js_div_matrix = 0.5 * (kl_pq + kl_qp)
        js_sim_matrix = torch.exp(-5.0 * js_div_matrix)
        
    
    # 组合余弦相似度、JS散度和类别匹配，保持连续性
    # 权重分配：余弦30%, JS散度40%, 类别匹配30%
    visual_sim_matrix = 0.3 * cosine_sim_matrix + 0.4 * js_sim_matrix + 0.3 * class_weighted_sim
    
    
    return visual_sim_matrix.numpy()


def compute_visual_similarity_matrix(object_nodes, aggregate_features, probs_list=None, 
                                    use_clip=True, debug=False):
    """
    计算视觉相似度矩阵（统一接口）
    
    根据use_clip参数选择使用CLIP embeddings或DINO belief
    
    Args:
        object_nodes: 现有物体节点列表
        aggregate_features: 新检测到的物体特征列表
        probs_list: 概率分布列表（仅用于belief方法）
        use_clip: 是否使用CLIP embeddings（True）还是DINO belief（False）
        debug: 是否输出调试信息
        
    Returns:
        visual_sim_matrix: 视觉相似度矩阵 (M x N)
    """
    if use_clip:
        # 使用CLIP embeddings（真正的视觉特征）
        return compute_visual_similarity_matrix_clip(object_nodes, aggregate_features, debug=debug)
    else:
        # 使用DINO belief（检测器判断分数）
        if probs_list is None:
            raise ValueError("使用belief方法时必须提供probs_list")
        return compute_visual_similarity_matrix_belief(object_nodes, aggregate_features, probs_list, debug=debug)


def compute_center_distance_similarity(object_nodes, aggregate_features, max_distance=2.0):
    """
    计算基于中心距离的空间相似度（已弃用，保留用于向后兼容）
    
    Args:
        object_nodes: 现有物体节点列表
        aggregate_features: 新检测到的物体特征列表
        max_distance: 距离衰减因子（米），用于计算距离相似度
        
    Returns:
        distance_sim_matrix: 距离相似度矩阵 (M x N)
    """
    m = len(object_nodes)
    n = len(aggregate_features)
    
    if m == 0 or n == 0:
        return np.zeros((m, n))
    
    distance_sim_matrix = np.zeros((m, n))
    
    for i, obj_node in enumerate(object_nodes):
        for j, agg_feat in enumerate(aggregate_features):
            # 计算中心点距离
            center_dist = np.linalg.norm(np.array(obj_node.center) - np.array(agg_feat['center']))
            
            # 距离相似度：距离越近相似度越高
            # 使用指数衰减函数
            distance_sim = np.exp(-center_dist / max_distance)
            distance_sim_matrix[i, j] = distance_sim
    
    return distance_sim_matrix


def compute_min_point_distance_similarity(min_distance_matrix, max_distance=1.0):
    """
    基于最小点距离矩阵计算空间相似度（从重叠度计算中零成本获取）
    
    Args:
        min_distance_matrix: 最小点距离矩阵 (M x N)，由compute_overlap_matrix_2set返回
        max_distance: 距离衰减因子（米），用于计算距离相似度
                     从配置文件中读取，通常通过agent.center_distance_decay传入
        
    Returns:
        distance_sim_matrix: 距离相似度矩阵 (M x N)
    """
    # 使用指数衰减函数将距离转换为相似度
    # 距离为0时相似度为1，距离为无穷大时相似度趋近于0
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
    """
    基于数学方程的统一相似度聚合，使用阈值约束而非分情况处理
    
    Args:
        overlap_sim: 点云重叠相似度矩阵
        distance_sim: 最小点距离相似度矩阵（用于展示，不直接参与计算）
        visual_sim: 视觉相似度矩阵
        object_nodes: 现有物体节点列表
        aggregate_features: 新检测物体特征列表
        min_distance_matrix: 最小点距离矩阵 (M x N)，用于距离惩罚和硬约束（推荐）⭐
        overlap_weight, visual_weight: 各种相似度权重
        max_match_distance: 最大匹配距离（米），基于最小点距离
        distance_threshold: 距离惩罚起始阈值（米），基于最小点距离
        distance_penalty_strength: 距离惩罚强度
        high_visual_boost_enabled: 是否启用高视觉boost
        high_visual_threshold: 高视觉阈值（超过此值认为是高视觉相似度）
        high_visual_distance_threshold: 高视觉距离阈值（小于此值认为是近距离）
        high_visual_boost_factor: boost因子（加到基础相似度上）
        debug: 是否输出决策过程
        
    Returns:
        smart_sim_matrix: 智能聚合的相似度矩阵
    """
    m, n = overlap_sim.shape
    
    # 使用最小点距离矩阵进行所有距离相关计算
    # 如果没有提供min_distance_matrix，则计算中心距离作为回退方案
    if min_distance_matrix is not None:
        distance_for_penalty = min_distance_matrix  # ✅ 优先使用最小点距离
    else:
        # 回退方案：计算中心距离
        center_distance_matrix = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                center_distance_matrix[i, j] = np.linalg.norm(
                    np.array(object_nodes[i].center) - np.array(aggregate_features[j]['center'])
                )
        distance_for_penalty = center_distance_matrix
    
    # 统一的相似度聚合方程（简化版）
    # 基础相似度 = 重叠相似度 + 视觉相似度
    base_similarity = overlap_weight * overlap_sim + visual_weight * visual_sim
    
    # 高视觉近距离boost机制
    # 当视觉相似度很高且距离很近时，给予额外的相似度boost，降低对重叠的依赖
    if high_visual_boost_enabled:
        # 条件1: 视觉相似度高于阈值
        high_visual_mask = visual_sim >= high_visual_threshold
        # 条件2: 距离小于阈值
        close_distance_mask = distance_for_penalty <= high_visual_distance_threshold
        # 两个条件都满足时应用boost
        boost_mask = high_visual_mask & close_distance_mask
        # 应用boost：为基础相似度增加一个固定值
        base_similarity = np.where(boost_mask, base_similarity + high_visual_boost_factor, base_similarity)
    
    # 距离惩罚制度：超过distance_threshold后给予指数衰减惩罚
    # 使用最小点距离（如果提供）或中心距离进行惩罚计算
    # distance_threshold从配置文件读取（如0.7米）
    distance_penalty = np.where(
        distance_for_penalty <= distance_threshold,
        1.0,  # 距离≤distance_threshold：无惩罚
        np.exp(-distance_penalty_strength * (distance_for_penalty - distance_threshold) / (max_match_distance - distance_threshold))  # 使用可配置的惩罚强度
    )
    
    # 简化的聚合相似度：基础相似度 × 距离惩罚
    smart_sim_matrix = base_similarity * distance_penalty
    
    # 强制距离约束（使用最小点距离，与距离惩罚一致）：
    # 1. 最小点距>max_match_distance时强制为0（绝对约束）
    smart_sim_matrix = np.where(
        distance_for_penalty > max_match_distance, 
        0.0, 
        smart_sim_matrix
    )
    
    # 2. 中等距离无重叠且视觉相似度低时强制为0（避免误匹配）
    # 只有在视觉相似度也很低时才强制为0，给高视觉相似度的匹配机会
    smart_sim_matrix = np.where(
        (distance_for_penalty > distance_threshold) & (overlap_sim < 0.05) & (visual_sim < 0.3),
        0.0,
        smart_sim_matrix
    )
    
    
    return smart_sim_matrix


def aggregate_similarities(spatial_sim, visual_sim, spatial_weight=0.7, visual_weight=0.3):
    """
    简单的线性聚合（保持向后兼容）
    
    Args:
        spatial_sim: 空间相似度矩阵
        visual_sim: 视觉相似度矩阵
        spatial_weight: 空间相似度权重
        visual_weight: 视觉相似度权重
        
    Returns:
        aggregated_sim: 聚合后的相似度矩阵
    """
    return spatial_weight * spatial_sim + visual_weight * visual_sim


def filter_contained_pointclouds(feature_list, probs_list=None, proximity_thresh=0.05, 
                                sample_size=512, visual_sim_threshold=0.7, debug=False):
    """
    过滤当前帧点云列表中被完全包含的小点云
    
    只在启用CLIP时进行智能过滤：
    - 如果有CLIP embeddings：计算视觉相似度，相似则删除小点云，不相似则保留
    - 如果没有CLIP embeddings：不进行过滤，保留所有点云
    
    Args:
        feature_list: aggregate_feature列表，每个元素包含'pcd'、'center'、'class_name'、可选'clip_embedding'
        probs_list: 保留用于兼容性（当前未使用）
        proximity_thresh: 判断点是否为inlier的距离阈值（米），默认0.05m
        sample_size: 采样点数，默认512
        visual_sim_threshold: CLIP视觉相似度阈值，默认0.5
        debug: 是否输出调试信息，默认False
        
    Returns:
        filtered_feature_list: 过滤后的特征列表
    """
    n = len(feature_list)
    if n <= 1:
        return feature_list
    
    # 预处理：构建FAISS索引和采样点
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
        
        # 随机采样用于查询
        current_sample_size = min(points.shape[0], sample_size)
        sampled_points = points[np.random.choice(points.shape[0], current_sample_size, replace=False)]
        sampled_points_list.append(sampled_points)
        
        # 构建FAISS索引（基于全部点）
        index = faiss.IndexFlatL2(points.shape[1])
        index.add(points)
        indices.append(index)
    
    proximity_thresh_sq = proximity_thresh ** 2
    
    # 计算单向overlap矩阵（仅需计算i→j方向）
    # overlap_matrix[i, j] 表示点云i中有多少比例的点在点云j附近
    overlap_matrix = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        if indices[i] is None or sampled_points_list[i].shape[0] == 0:
            continue
        
        for j in range(n):
            if i == j or indices[j] is None:
                continue
            
            # 计算 i → j 的单向重叠率
            # 从点云i中采样点，在点云j的索引中查询
            query_pts = sampled_points_list[i]
            distances, _ = indices[j].search(query_pts, 1)
            inliers = np.sum(distances < proximity_thresh_sq)
            overlap_matrix[i, j] = inliers / query_pts.shape[0]
    
    # 标记需要删除的点云，并记录删除信息
    to_remove = set()
    removal_info = []  # 存储删除信息：(被删除的索引, 包含它的索引, overlap值, 视觉相似度, 是否删除)
    
    for i in range(n):
        if i in to_remove:
            continue
        
        for j in range(n):
            if i == j or j in to_remove:
                continue
            
            # 检查点云i是否被点云j完全包含
            # 条件：overlap_matrix[i, j] == 1.0（点云i完全在点云j内）
            if abs(overlap_matrix[i, j] - 1.0) < 1e-6:
                # 点云i被点云j完全包含
                visual_sim = None
                should_remove = False
                skip_reason = ""
                
                # 检查是否有CLIP embeddings
                has_clip = (feature_list[i].get('clip_embedding') is not None and 
                           feature_list[j].get('clip_embedding') is not None)
                
                if has_clip:
                    # 只在使用CLIP时才进行视觉相似度判断
                    clip_emb_i = torch.tensor(feature_list[i]['clip_embedding']) if isinstance(feature_list[i]['clip_embedding'], np.ndarray) else feature_list[i]['clip_embedding']
                    clip_emb_j = torch.tensor(feature_list[j]['clip_embedding']) if isinstance(feature_list[j]['clip_embedding'], np.ndarray) else feature_list[j]['clip_embedding']
                    
                    # L2归一化
                    clip_emb_i = F.normalize(clip_emb_i.unsqueeze(0), p=2, dim=1)
                    clip_emb_j = F.normalize(clip_emb_j.unsqueeze(0), p=2, dim=1)
                    
                    # 余弦相似度
                    visual_sim = F.cosine_similarity(clip_emb_i, clip_emb_j, dim=1).item()
                    
                    # 检查小点云是否是cushion（特殊保护）
                    smaller_idx = i if point_counts[i] < point_counts[j] else j
                    if feature_list[smaller_idx]['class_name'] == 'cushion':
                        # cushion不过滤
                        should_remove = False
                        skip_reason = f"小点云是cushion，保留（CLIP相似度={visual_sim:.3f}）"
                    elif visual_sim >= visual_sim_threshold:
                        # 视觉相似度高，说明是同一物品的局部和整体
                        should_remove = True
                        skip_reason = f"CLIP相似度{visual_sim:.3f} ≥ {visual_sim_threshold:.3f}，同一物品"
                    else:
                        # 视觉相似度低，说明是两个不同物体
                        should_remove = False
                        skip_reason = f"CLIP相似度{visual_sim:.3f} < {visual_sim_threshold:.3f}，不同物体"
                else:
                    # 没有CLIP embeddings，不进行过滤（保留所有点云）
                    should_remove = False
                    skip_reason = "无CLIP特征，跳过过滤"
                    visual_sim = 0.0  # 用于调试显示
                
                # 根据判断结果决定是否删除
                if should_remove:
                    if point_counts[i] < point_counts[j]:
                        to_remove.add(i)
                        removal_info.append((i, j, overlap_matrix[i, j], visual_sim, True, skip_reason))
                        break  # i已标记删除，跳出内层循环
                    elif point_counts[j] < point_counts[i]:
                        to_remove.add(j)
                        removal_info.append((j, i, overlap_matrix[j, i], visual_sim, True, skip_reason))
                else:
                    # 不删除，记录保留信息
                    removal_info.append((i, j, overlap_matrix[i, j], visual_sim, False, skip_reason))
    
    # 调试信息输出
    if debug and len(removal_info) > 0:
        # 检查是否使用CLIP
        has_clip_global = any(feat.get('clip_embedding') is not None for feat in feature_list)
        method_name = "CLIP embeddings" if has_clip_global else "DINO belief"
        
        print(f"\n{'='*80}")
        print(f"[当前帧包含关系过滤] 检查了{n}个检测，发现{len(removal_info)}对包含关系")
        print(f"视觉相似度方法: {method_name}")
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
                status = "✓ 已过滤"
            else:
                kept_count += 1
                status = "✗ 保留"
            
            print(f"\n[{status}] 检测{removed_idx} [{removed_obj['class_name']}] 被 检测{container_idx} [{container_obj['class_name']}] 完全包含")
            print(f"  ├─ 被检测点云{removed_idx}:")
            print(f"  │  ├─ 类别: {removed_obj['class_name']}")
            print(f"  │  ├─ 点数: {point_counts[removed_idx]}")
            print(f"  │  └─ 中心: ({removed_center[0]:.3f}, {removed_center[1]:.3f}, {removed_center[2]:.3f})m")
            print(f"  ├─ 包含点云{container_idx}:")
            print(f"  │  ├─ 类别: {container_obj['class_name']}")
            print(f"  │  ├─ 点数: {point_counts[container_idx]}")
            print(f"  │  └─ 中心: ({container_center[0]:.3f}, {container_center[1]:.3f}, {container_center[2]:.3f})m")
            print(f"  ├─ 单向overlap[{removed_idx}→{container_idx}]: {overlap_val:.4f} (≈1.0)")
            print(f"  ├─ 视觉相似度: {visual_sim:.3f}")
            print(f"  └─ 决策: {reason}")
        
        print(f"\n总结: 过滤掉{removed_count}个点云，保留{kept_count}个点云")
        print(f"最终保留检测: {len(feature_list) - len(to_remove)}个")
        print(f"{'='*80}\n")
    
    # 构建过滤后的列表
    filtered_feature_list = [
        feature_list[i] for i in range(n) if i not in to_remove
    ]
    
    return filtered_feature_list


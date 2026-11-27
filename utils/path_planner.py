import cv2
import numpy as np
import sys
import math
import torch
import os
import time
from collections import deque
import heapq # Import heapq for A* priority queue
from typing import Union, List, Dict, Any, Tuple # Import Union, List, Tuple, Dict, Any for type hints

# Provided function (keep it outside the class as requested)
def load_image_as_occupancy_tensor(image_path: str) -> Union[torch.Tensor, None]:
    """
    Loads a grayscale image and converts it to an occupancy tensor.
    Values: 0 (unknown), 1 (free), 2 (obstacle).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    obstacle_threshold_max = 50 # Values <= 50 are obstacles
    free_threshold_min = 200   # Values >= 200 are free space

    occupancy_np = np.zeros_like(img, dtype=np.uint8) # Default is 0 (Unknown)
    occupancy_np[img <= obstacle_threshold_max] = 2 # Obstacle
    occupancy_np[img >= free_threshold_min] = 1   # Free

    print(f"\nLoaded image {image_path}. Map shape: {img.shape}")
    print(f"Occupancy values distribution: 0={np.sum(occupancy_np==0)}, 1={np.sum(occupancy_np==1)}, 2={np.sum(occupancy_np==2)}")

    # Ensure torch is available
    if 'torch' not in sys.modules:
        print("Error: torch not found. Cannot create tensor from image.")
        return None

    try:
        # Convert to torch tensor (add batch and channel dimensions)
        occupancy_tensor = torch.from_numpy(occupancy_np).unsqueeze(0).unsqueeze(0).to(torch.float32)
        print(f"Converted to torch tensor: {occupancy_tensor.shape}, dtype: {occupancy_tensor.dtype}")
        return occupancy_tensor
    except Exception as e:
        print(f"An error occurred during tensor conversion: {e}")
        return None

# Heuristic function for A* (Euclidean distance)
def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Calculates Euclidean distance between two points (a and b)."""
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

class PathPlanner:
    def __init__(self, robot_radius: int, centering_weight: float = 50.0, hole_fill_area: int = 200, safety_margin: int = 5, closing_kernel_size:int = 11, high_risk_penalty: float = 200.0):
        if robot_radius < 0:
            raise ValueError("Robot radius cannot be negative.")
        self.robot_radius = robot_radius
        self.centering_weight = float(centering_weight)
        self.hole_fill_area = hole_fill_area
        self.safety_margin = safety_margin
        self.high_risk_penalty = high_risk_penalty
        self.closing_kernel_size = closing_kernel_size
        self.collision_point_inflation_radius = self.robot_radius - 1
        self.collision_offsets = set()
        self.inflated_collision_points = set()

        # 定义8个方向的移动及其基础移动成本 (dx, dy, base_cost)
        self.neighbors = [
            (0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0),  # 上下左右
            (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),  # 对角线
            (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))
        ]
    
    def reset(self):
        self.collision_offsets = set()
        self.inflated_collision_points = set()

    def _fill_small_holes(self, mask: np.ndarray) -> np.ndarray:
        if self.closing_kernel_size % 2 == 0: 
            self.closing_kernel_size += 1 
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.closing_kernel_size, self.closing_kernel_size))
        closed_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            closed_mask, connectivity=8
        )
        filled_mask = closed_mask.copy()
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area <= self.hole_fill_area:
                filled_mask[labels == i] = 0
                
        return filled_mask

    def _add_collision_point(self, point):
        px, py = point
        # radius = self.collision_point_inflation_radius
        radius = 1
        self.collision_offsets.add(point)
        if radius <= 0:
            self.inflated_collision_points.add(point)
            return

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx**2 + dy**2 <= radius**2:
                    new_point = (px + dx, py + dy)
                    self.inflated_collision_points.add(new_point)

    def _is_valid(self, x: int, y: int, height: int, width: int) -> bool:
        return 0 <= x < width and 0 <= y < height

    # --- 修改后的 find_path 方法 ---
    def find_path(
        self, 
        occupancy_map: torch.Tensor, 
        start: Tuple[int, int], 
        end: Tuple[int, int], 
        start_adjustment_radius: int = 5,
        # # 新增参数
        # safety_margin: int = 5,  # 安全边际，用于定义高风险区
        # high_risk_penalty: float = 200.0 # 高风险区的惩罚值
    ) -> Union[List[Tuple[int, int]], None]:
        
        if not isinstance(occupancy_map, torch.Tensor):
            raise TypeError("occupancy_map must be a torch.Tensor.")
        if occupancy_map.ndim != 4 or occupancy_map.shape[:2] != (1, 1):
             raise ValueError(f"occupancy_map must have shape (1, 1, H, W), but got {occupancy_map.shape}")
        
        occupancy_np = occupancy_map.squeeze().cpu().numpy().astype(np.uint8)
        height, width = occupancy_np.shape

        obstacle_mask = (occupancy_np == 2.0).astype(np.uint8)
        unknown_mask = (occupancy_np == 0.0).astype(np.uint8)
        
        
        # 【新逻辑】创建一个包含所有预膨胀碰撞点的掩码
        # 这个掩码将同时用于致死层和高风险层
        preinflated_offsets_mask = np.zeros_like(occupancy_np, dtype=np.uint8)
        for x, y in self.inflated_collision_points:
            # 确保点在地图范围内
            if 0 <= y < height and 0 <= x < width:
                preinflated_offsets_mask[y, x] = 1

        # 1. 创建“致死层” (Lethal Layer)
        # 致死层 = 膨胀后的常规障碍物 + 预膨胀的特殊碰撞点
        
        # 1.1 膨胀常规障碍物
        if self.robot_radius > 0:
            obstacle_kernel_size = 2 * self.robot_radius + 1
            obstacle_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (obstacle_kernel_size, obstacle_kernel_size))
            inflated_obstacles = cv2.dilate(obstacle_mask, obstacle_kernel)
        else:
            inflated_obstacles = obstacle_mask.copy()
            
        # 1.2 合并得到最终的致死膨胀区
        lethal_inflated_mask = cv2.bitwise_or(inflated_obstacles, preinflated_offsets_mask)
        lethal_inflated_mask_255 = (lethal_inflated_mask * 255).astype(np.uint8)


        # 2. 定义可通行区域 (is_passable)
        # 不可通行区 = 致死膨胀区 + 未知区域
        final_impassable_mask = cv2.bitwise_or(lethal_inflated_mask_255, (unknown_mask * 255).astype(np.uint8))
        is_passable = (final_impassable_mask == 0)


        # 3. 创建“高风险层” (Precautionary Layer)
        additional_cost_map = np.zeros_like(occupancy_np, dtype=np.float32)
        if self.safety_margin > 0 and self.high_risk_penalty > 0:
            # 3.1 准备基础掩码：常规障碍物 + 未知区域 + 预膨胀的特殊碰撞点
            precautionary_base_mask = cv2.bitwise_or(obstacle_mask, unknown_mask)
            precautionary_base_mask = cv2.bitwise_or(precautionary_base_mask, preinflated_offsets_mask)
            precautionary_base_mask_255 = (precautionary_base_mask * 255).astype(np.uint8)

            # 3.2 膨胀这个基础掩码
            precautionary_radius = self.robot_radius + self.safety_margin
            precautionary_kernel_size = 2 * precautionary_radius + 1
            precautionary_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (precautionary_kernel_size, precautionary_kernel_size))
            precautionary_inflated_mask = cv2.dilate(precautionary_base_mask_255, precautionary_kernel)
            
            # 3.3 计算高风险区并施加代价
            high_risk_zone = (precautionary_inflated_mask > 0) & (lethal_inflated_mask_255 == 0)
            high_risk_zone_on_passable_area = high_risk_zone & is_passable
            additional_cost_map[high_risk_zone_on_passable_area] = self.high_risk_penalty
        
        # 4. 计算居中代价 (逻辑不变)
        if self.centering_weight > 0:
            dist_map = cv2.distanceTransform(is_passable.astype(np.uint8), cv2.DIST_L2, 5)
            mask = dist_map > 0
            additional_cost_map[mask] += self.centering_weight / (dist_map[mask] + 1.0)
            
            
        # --- A* 搜索部分 (基本不变) ---
        # ... (后面的 A* 搜索逻辑完全一样, 它会自然地使用我们创建的 additional_cost_map)
        def heuristic(a, b): # 假设的启发函数
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

        start_node = (int(start[0]), int(start[1]))
        end_node = (int(end[0]), int(end[1]))


        if not (self._is_valid(*end_node, height, width) and is_passable[end_node[1], end_node[0]]):
            return None

        if not (self._is_valid(*start_node, height, width) and is_passable[start_node[1], start_node[0]]) or start_node in self.collision_offsets:
            found_new_start = None
            for r in range(1, start_adjustment_radius + 1):
                if found_new_start: break
                for dy in range(-r, r + 1):
                    if found_new_start: break
                    for dx in range(-r, r + 1):
                        if max(abs(dx), abs(dy)) != r: continue
                        
                        px, py = start_node[0] + dx, start_node[1] + dy
                        if self._is_valid(px, py, height, width) and is_passable[py, px] and (px, py) not in self.collision_offsets:
                            found_new_start = (px, py)
                            break
            
            if found_new_start:
                start_node = found_new_start
            else:
                return None
        
        frontier = []
        heapq.heappush(frontier, (heuristic(start_node, end_node), start_node))
        came_from = {start_node: None}
        cost_so_far = {start_node: 0.0}

        while frontier:
            _, current_node = heapq.heappop(frontier)

            if current_node == end_node:
                path = []
                curr = end_node
                while curr is not None:
                    path.append(curr)
                    curr = came_from.get(curr)
                path.reverse()
                return path

            cx, cy = current_node
            for dx, dy, base_move_cost in self.neighbors:
                next_node = (cx + dx, cy + dy)
                nx, ny = next_node

                if not (self._is_valid(nx, ny, height, width) and is_passable[ny, nx]):
                    continue
                
                if dx != 0 and dy != 0 and not (is_passable[cy, nx] and is_passable[ny, cx]):
                    continue

                additional_cost = additional_cost_map[ny, nx]
                new_cost = cost_so_far[current_node] + base_move_cost + additional_cost

                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + heuristic(next_node, end_node)
                    heapq.heappush(frontier, (priority, next_node))
                    came_from[next_node] = current_node
        
        return None    
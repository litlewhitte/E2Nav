"""
路径规划模块：包含A*和Hybrid A*两种路径规划算法
"""
import math
import heapq
import numpy as np
import cv2
import torch
from typing import List, Tuple, Union, Optional


# ========================================
# A* 路径规划器（原始实现）
# ========================================
class AStarPlanner:
    """
    传统A*路径规划器
    - 使用膨胀后的障碍物地图
    - 只检查中心点是否可通行
    - 支持8方向移动
    """
    def __init__(
        self, 
        robot_radius: int, 
        centering_weight: float = 50.0, 
        hole_fill_area: int = 200, 
        safety_margin: int = 5, 
        closing_kernel_size: int = 11, 
        high_risk_penalty: float = 200.0
    ):
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
        """重置碰撞点记录"""
        self.collision_offsets = set()
        self.inflated_collision_points = set()
    
    def add_collision_point(self, point: Tuple[int, int]):
        """
        添加碰撞点（膨胀半径=1）
        
        Args:
            point: 碰撞点坐标 (x, y)
        """
        px, py = point
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
        """检查坐标是否在地图范围内"""
        return 0 <= x < width and 0 <= y < height

    def find_path(
        self, 
        occupancy_map: torch.Tensor, 
        start: Tuple[int, int], 
        end: Tuple[int, int], 
        start_adjustment_radius: int = 5,
    ) -> Union[List[Tuple[int, int]], None]:
        """
        使用A*算法进行路径规划
        
        Args:
            occupancy_map: 占用栅格地图 (1, 1, H, W)
            start: 起点坐标 (x, y)
            end: 终点坐标 (x, y)
            start_adjustment_radius: 起点调整搜索半径
            
        Returns:
            路径列表 [(x, y), ...] 或 None
        """
        if not isinstance(occupancy_map, torch.Tensor):
            raise TypeError("occupancy_map must be a torch.Tensor.")
        if occupancy_map.ndim != 4 or occupancy_map.shape[:2] != (1, 1):
             raise ValueError(f"occupancy_map must have shape (1, 1, H, W), but got {occupancy_map.shape}")
        
        occupancy_np = occupancy_map.squeeze().cpu().numpy().astype(np.uint8)
        height, width = occupancy_np.shape

        obstacle_mask = (occupancy_np == 2.0).astype(np.uint8)
        unknown_mask = (occupancy_np == 0.0).astype(np.uint8)
        
        # 创建预膨胀碰撞点掩码
        preinflated_offsets_mask = np.zeros_like(occupancy_np, dtype=np.uint8)
        for x, y in self.inflated_collision_points:
            if 0 <= y < height and 0 <= x < width:
                preinflated_offsets_mask[y, x] = 1

        # 1. 膨胀常规障碍物
        if self.robot_radius > 0:
            obstacle_kernel_size = 2 * self.robot_radius + 1
            obstacle_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (obstacle_kernel_size, obstacle_kernel_size))
            inflated_obstacles = cv2.dilate(obstacle_mask, obstacle_kernel)
        else:
            inflated_obstacles = obstacle_mask.copy()
            
        # 2. 合并致死层（膨胀后的障碍物 + 预膨胀的特殊碰撞点）
        lethal_inflated_mask = cv2.bitwise_or(inflated_obstacles, preinflated_offsets_mask)
        lethal_inflated_mask_255 = (lethal_inflated_mask * 255).astype(np.uint8)

        # 3. 定义可通行区域（不可通行 = 致死膨胀区 + 未知区域）
        final_impassable_mask = cv2.bitwise_or(lethal_inflated_mask_255, (unknown_mask * 255).astype(np.uint8))
        is_passable = (final_impassable_mask == 0)

        # 4. 创建高风险层
        additional_cost_map = np.zeros_like(occupancy_np, dtype=np.float32)
        if self.safety_margin > 0 and self.high_risk_penalty > 0:
            # 高风险基础：常规障碍物 + 未知区域 + 预膨胀的特殊碰撞点
            precautionary_base_mask = cv2.bitwise_or(obstacle_mask, unknown_mask)
            precautionary_base_mask = cv2.bitwise_or(precautionary_base_mask, preinflated_offsets_mask)
            precautionary_base_mask_255 = (precautionary_base_mask * 255).astype(np.uint8)

            precautionary_radius = self.robot_radius + self.safety_margin
            precautionary_kernel_size = 2 * precautionary_radius + 1
            precautionary_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (precautionary_kernel_size, precautionary_kernel_size))
            precautionary_inflated_mask = cv2.dilate(precautionary_base_mask_255, precautionary_kernel)
            
            high_risk_zone = (precautionary_inflated_mask > 0) & (lethal_inflated_mask_255 == 0)
            high_risk_zone_on_passable_area = high_risk_zone & is_passable
            additional_cost_map[high_risk_zone_on_passable_area] = self.high_risk_penalty
        
        # 5. 计算居中代价
        if self.centering_weight > 0:
            dist_map = cv2.distanceTransform(is_passable.astype(np.uint8), cv2.DIST_L2, 5)
            mask = dist_map > 0
            additional_cost_map[mask] += self.centering_weight / (dist_map[mask] + 1.0)
            
        # ========== A* 搜索 ==========
        def heuristic(a, b):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

        start_node = (int(start[0]), int(start[1]))
        end_node = (int(end[0]), int(end[1]))

        # 检查终点
        if not (self._is_valid(*end_node, height, width) and is_passable[end_node[1], end_node[0]]):
            return None

        # 检查起点（额外检查collision_offsets）
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
        
        # A*搜索
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
                
                # 对角线移动检查
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


# ========================================
# Hybrid A* 路径规划器
# ========================================
class HybridAStarPlanner:
    """
    Hybrid A*路径规划器（考虑朝向和运动学约束）
    - 使用未膨胀的障碍物地图
    - 检查机器人整个足迹是否可通行
    - 搜索空间为 (x, y, θ)
    """
    def __init__(
        self,
        robot_radius: int,
        forward_distance: float = 0.25,
        turn_angle: float = 30.0,
        theta_resolution: int = 12,
        centering_weight: float = 50.0,
        high_risk_penalty: float = 200.0,
        safety_margin: int = 5,
        goal_arrival_factor: float = 1.2,
    ):
        self.robot_radius = robot_radius
        self.forward_distance = forward_distance
        self.turn_angle_rad = math.radians(turn_angle)
        self.theta_resolution = theta_resolution
        self.centering_weight = centering_weight
        self.high_risk_penalty = high_risk_penalty
        self.safety_margin = safety_margin
        self.goal_arrival_factor = goal_arrival_factor
        
        self.theta_step = 2 * math.pi / theta_resolution
        self._robot_footprint_offsets = self._generate_robot_footprint()
        self.high_risk_map = None
        self.centering_map = None
    
    def _generate_robot_footprint(self) -> List[Tuple[int, int]]:
        """生成机器人轮廓的像素偏移（圆形）"""
        offsets = []
        radius = self.robot_radius
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx*dx + dy*dy <= radius*radius:
                    offsets.append((dx, dy))
        return offsets
    
    def _discretize_state(self, x: float, y: float, theta: float) -> Tuple[int, int, int]:
        """将连续状态离散化为网格索引"""
        grid_x = int(round(x))
        grid_y = int(round(y))
        discrete_theta = int(round(theta / self.theta_step)) % self.theta_resolution
        return (grid_x, grid_y, discrete_theta)
    
    def _check_collision(self, x: float, y: float, is_passable: np.ndarray) -> bool:
        """检查机器人足迹是否碰撞"""
        height, width = is_passable.shape
        
        for dx, dy in self._robot_footprint_offsets:
            px = int(round(x + dx))
            py = int(round(y + dy))
            
            if px < 0 or px >= width or py < 0 or py >= height:
                return True
            
            if not is_passable[py, px]:
                return True
        
        return False
    
    def _expand_node(
        self, 
        current: Tuple[float, float, float],
        resolution_scale: float = 20.0,
        debug: bool = False
    ) -> List[Tuple[float, float, float, float, str]]:
        """扩展当前节点，返回后继状态"""
        x, y, theta = current
        successors = []
        
        # 前进
        forward_pixels = self.forward_distance * resolution_scale
        new_x = x + forward_pixels * math.cos(theta)
        new_y = y + forward_pixels * math.sin(theta)
        successors.append((new_x, new_y, theta, forward_pixels, 'forward'))
        
        # 左转（顺时针，θ增大）
        new_theta_left = theta + self.turn_angle_rad
        successors.append((x, y, new_theta_left, 1.0, 'turn_left'))
        
        # 右转（逆时针，θ减小）
        new_theta_right = theta - self.turn_angle_rad
        successors.append((x, y, new_theta_right, 1.0, 'turn_right'))
        
        return successors
    
    def _compute_heuristic(
        self,
        current: Tuple[float, float, float],
        goal: Tuple[float, float, float],
        goal_radius: float = None
    ) -> float:
        """计算启发式函数（只考虑距离）"""
        x1, y1, theta1 = current
        x2, y2, theta2 = goal
        
        euclidean_dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        heuristic_weight = 1.05
        
        return euclidean_dist * heuristic_weight
    
    def find_path(
        self,
        is_passable: np.ndarray,
        start: Tuple[float, float, float],
        goal: Tuple[float, float, float],
        resolution_scale: float = 20.0,
        max_iterations: int = 3000,
        high_risk_map: Optional[np.ndarray] = None,
        centering_map: Optional[np.ndarray] = None,
        start_adjustment_radius: int = 5,
    ) -> Optional[List[Tuple[float, float, float]]]:
        """
        Hybrid A*路径规划
        
        Args:
            is_passable: 可通行地图 (H, W)
            start: 起点 (x, y, theta)
            goal: 终点 (x, y, theta)
            resolution_scale: 像素/米转换比例
            max_iterations: 最大迭代次数
            high_risk_map: 高风险地图
            centering_map: 居中代价地图
            start_adjustment_radius: 起点调整半径
            
        Returns:
            路径 [(x, y, theta), ...] 或 None
        """
        self.high_risk_map = high_risk_map
        self.centering_map = centering_map
        
        height, width = is_passable.shape
        
        # 归一化角度
        def normalize_angle(angle):
            while angle > math.pi:
                angle -= 2 * math.pi
            while angle <= -math.pi:
                angle += 2 * math.pi
            return angle
        
        start_normalized = (start[0], start[1], normalize_angle(start[2]))
        goal_normalized = (goal[0], goal[1], normalize_angle(goal[2]))
        
        start = start_normalized
        goal = goal_normalized
        
        # 检查起点
        start_collision = self._check_collision(start[0], start[1], is_passable)
        if start_collision:
            found_new_start = None
            for r in range(1, start_adjustment_radius + 1):
                if found_new_start:
                    break
                for dy in range(-r, r + 1):
                    if found_new_start:
                        break
                    for dx in range(-r, r + 1):
                        if max(abs(dx), abs(dy)) != r:
                            continue
                        
                        px = start[0] + dx
                        py = start[1] + dy
                        
                        if (0 <= px < width and 0 <= py < height and 
                            not self._check_collision(px, py, is_passable)):
                            found_new_start = (px, py, start[2])
                            break
            
            if found_new_start:
                start = found_new_start
            else:
                return None
        
        # 检查终点
        goal_collision = self._check_collision(goal[0], goal[1], is_passable)
        if goal_collision:
            for angle_offset in [0, math.pi/6, -math.pi/6, math.pi/3, -math.pi/3]:
                adjusted_theta = normalize_angle(goal[2] + angle_offset)
                if not self._check_collision(goal[0], goal[1], is_passable):
                    goal = (goal[0], goal[1], adjusted_theta)
                    break
            else:
                return None
        
        # 初始化
        open_set = []
        closed_set = set()
        g_score = {}
        came_from = {}
        
        start_discrete = self._discretize_state(start[0], start[1], start[2])
        g_score[start_discrete] = 0.0
        h_start = self._compute_heuristic(start, goal)
        counter = 0
        heapq.heappush(open_set, (h_start, counter, start, start_discrete))
        
        best_state = None
        best_distance = float('inf')
        goal_threshold = self.forward_distance * resolution_scale * self.goal_arrival_factor
        
        iterations = 0
        while open_set and iterations < max_iterations:
            iterations += 1
            
            _, _, current_continuous, current_discrete = heapq.heappop(open_set)
            
            if current_discrete in closed_set:
                continue
            
            closed_set.add(current_discrete)
            
            # 检查是否到达目标（只考虑2D距离）
            dist_to_goal = math.sqrt(
                (current_continuous[0] - goal[0])**2 + 
                (current_continuous[1] - goal[1])**2
            )
            
            if dist_to_goal < best_distance:
                best_distance = dist_to_goal
                best_state = (current_continuous, current_discrete)
            
            if dist_to_goal < goal_threshold:
                path = self._reconstruct_path(came_from, current_continuous, current_discrete, start)
                return path
            
            # 扩展节点
            successors = self._expand_node(current_continuous, resolution_scale, debug=False)
            successors_sorted = sorted(successors, key=lambda x: 0 if x[4] == 'forward' else 1)
            
            for new_x, new_y, new_theta, edge_cost, action in successors_sorted:
                if self._check_collision(new_x, new_y, is_passable):
                    continue
                
                if new_x < 0 or new_x >= width or new_y < 0 or new_y >= height:
                    continue
                
                new_discrete = self._discretize_state(new_x, new_y, new_theta)
                
                if new_discrete in closed_set:
                    continue
                
                actual_edge_cost = edge_cost
                ix, iy = int(round(new_x)), int(round(new_y))
                
                if high_risk_map is not None and 0 <= ix < width and 0 <= iy < height:
                    actual_edge_cost += high_risk_map[iy, ix]
                
                if centering_map is not None and 0 <= ix < width and 0 <= iy < height:
                    actual_edge_cost += centering_map[iy, ix]
                
                tentative_g = g_score[current_discrete] + actual_edge_cost
                
                if new_discrete not in g_score or tentative_g < g_score[new_discrete]:
                    g_score[new_discrete] = tentative_g
                    came_from[new_discrete] = (current_continuous, current_discrete)
                    
                    h_score = self._compute_heuristic((new_x, new_y, new_theta), goal)
                    f_score = tentative_g + h_score
                    
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, (new_x, new_y, new_theta), new_discrete))
        
        # Fallback
        fallback_threshold = self.forward_distance * resolution_scale * 3.0
        if best_state is not None and best_distance < fallback_threshold:
            best_continuous, best_discrete = best_state
            path = self._reconstruct_path(came_from, best_continuous, best_discrete, start)
            return path
        
        return None
    
    def _reconstruct_path(
        self,
        came_from: dict,
        current_continuous: Tuple[float, float, float],
        current_discrete: Tuple[int, int, int],
        start: Tuple[float, float, float]
    ) -> List[Tuple[float, float, float]]:
        """重建路径"""
        path = [current_continuous]
        
        while current_discrete in came_from:
            parent_continuous, parent_discrete = came_from[current_discrete]
            path.append(parent_continuous)
            current_discrete = parent_discrete
            
            if parent_continuous == start:
                break
        
        path.reverse()
        return path


# ========================================
# 路径规划器统一接口
# ========================================
class PathPlanner:
    """
    路径规划器统一接口
    - 持有 AStarPlanner 和 HybridAStarPlanner 实例
    - 提供统一的调用接口
    """
    def __init__(
        self,
        robot_radius: int,
        # A* 专用参数
        centering_weight: float = 50.0,
        hole_fill_area: int = 200,
        safety_margin: int = 5,
        closing_kernel_size: int = 11,
        high_risk_penalty: float = 200.0,
        # Hybrid A* 专用参数
        use_hybrid_astar: bool = False,
        forward_distance: float = 0.25,
        turn_angle: float = 30.0,
        theta_resolution: int = 12,
        r_resolution: float = 20.0,
        goal_arrival_factor: float = 1.2,
        hybrid_centering_weight: float = 50.0,
        hybrid_safety_margin: int = 5,
        hybrid_high_risk_penalty: float = 200.0,
    ):
        self.use_hybrid_astar = use_hybrid_astar
        self.r_resolution = r_resolution
        
        # 在PathPlanner层统一管理碰撞点（两个规划器共享）
        self.collision_offsets = set()
        self.inflated_collision_points = set()
        self.robot_radius = robot_radius
        
        # 保存Hybrid A*参数（用于地图预处理）
        self.hybrid_centering_weight = hybrid_centering_weight
        self.hybrid_safety_margin = hybrid_safety_margin
        self.hybrid_high_risk_penalty = hybrid_high_risk_penalty
        
        # 创建A*规划器
        self.astar_planner = AStarPlanner(
            robot_radius=robot_radius,
            centering_weight=centering_weight,
            hole_fill_area=hole_fill_area,
            safety_margin=safety_margin,
            closing_kernel_size=closing_kernel_size,
            high_risk_penalty=high_risk_penalty
        )
        # 共享碰撞点引用
        self.astar_planner.collision_offsets = self.collision_offsets
        self.astar_planner.inflated_collision_points = self.inflated_collision_points
        
        # 创建Hybrid A*规划器
        self.hybrid_planner = HybridAStarPlanner(
            robot_radius=robot_radius,
            forward_distance=forward_distance,
            turn_angle=turn_angle,
            theta_resolution=theta_resolution,
            centering_weight=hybrid_centering_weight,
            high_risk_penalty=hybrid_high_risk_penalty,
            safety_margin=hybrid_safety_margin,
            goal_arrival_factor=goal_arrival_factor,
        )
    
    def reset(self):
        """重置碰撞点记录（两个规划器共享）"""
        self.collision_offsets.clear()
        self.inflated_collision_points.clear()
    
    def _add_collision_point(self, point: Tuple[int, int]):
        """
        添加碰撞点（两个规划器共享）
        - 膨胀半径=1像素
        - A*使用inflated_collision_points（地图中标记）+ collision_offsets（起点检查）
        - Hybrid A*将collision点当作障碍物（通过足迹检测确保距离>robot_radius）
        """
        self.astar_planner.add_collision_point(point)
    
    @staticmethod
    def extract_2d_path(path: Union[List[Tuple[int, int]], List[Tuple[int, int, float]]]) -> List[Tuple[int, int]]:
        """从路径中提取2D坐标"""
        if path is None or len(path) == 0:
            return path
        
        if len(path[0]) == 2:
            return path
        elif len(path[0]) == 3:
            return [(int(round(p[0])), int(round(p[1]))) for p in path]
        else:
            raise ValueError(f"Unknown path format: {path[0]}")
    
    @staticmethod
    def compute_path_length(path: Union[List[Tuple[int, int]], List[Tuple[int, int, float]]]) -> float:
        """计算路径长度"""
        if path is None or len(path) < 2:
            return 0.0
        
        length = 0.0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            length += math.sqrt(dx*dx + dy*dy)
        
        return length
    
    # ========== A* 路径规划（公开接口）==========
    def find_path_astar(
        self,
        occupancy_map: torch.Tensor,
        start: Union[Tuple[int, int], Tuple[int, int, float]],
        end: Union[Tuple[int, int], Tuple[int, int, float]],
        start_adjustment_radius: int = 5,
    ) -> Union[List[Tuple[int, int]], None]:
        """
        使用A*进行路径规划
        
        Args:
            occupancy_map: 占用栅格地图 (1, 1, H, W)
            start: 起点 (x, y) 或 (x, y, theta)
            end: 终点 (x, y) 或 (x, y, theta)
            start_adjustment_radius: 起点调整半径
            
        Returns:
            路径 [(x, y), ...] 或 None
        """
        # 提取2D坐标
        if len(start) == 3:
            start = (start[0], start[1])
        if len(end) == 3:
            end = (end[0], end[1])
        
        return self.astar_planner.find_path(
            occupancy_map=occupancy_map,
            start=start,
            end=end,
            start_adjustment_radius=start_adjustment_radius,
        )
    
    # ========== Hybrid A* 路径规划（公开接口）==========
    def find_path_hybrid_astar(
        self,
        occupancy_map: torch.Tensor,
        start: Union[Tuple[int, int], Tuple[int, int, float]],
        end: Union[Tuple[int, int], Tuple[int, int, float]],
        start_adjustment_radius: int = 5,
        start_theta: Optional[float] = None,
        end_theta: Optional[float] = None,
    ) -> Union[List[Tuple[int, int]], None]:
        """
        使用Hybrid A*进行路径规划
        
        Args:
            occupancy_map: 占用栅格地图 (1, 1, H, W)
            start: 起点 (x, y) 或 (x, y, theta)
            end: 终点 (x, y) 或 (x, y, theta)
            start_adjustment_radius: 起点调整半径
            start_theta: 起点朝向（弧度），如果start是2D则必须提供
            end_theta: 终点朝向（弧度），如果end是2D则必须提供
            
        Returns:
            路径 [(x, y), ...] 或 None（返回2D路径用于可视化）
        """
        # 构建3D起点和终点
        if len(start) == 3:
            start_3d = start
        elif len(start) == 2:
            if start_theta is None:
                raise ValueError("start_theta must be provided when start is 2D")
            start_3d = (start[0], start[1], start_theta)
        else:
            raise ValueError(f"Invalid start format: {start}")
        
        if len(end) == 3:
            end_3d = end
        elif len(end) == 2:
            if end_theta is None:
                end_theta = 0.0
            end_3d = (end[0], end[1], end_theta)
        else:
            raise ValueError(f"Invalid end format: {end}")
        
        # 地图预处理
        if not isinstance(occupancy_map, torch.Tensor):
            raise TypeError("occupancy_map must be a torch.Tensor.")
        if occupancy_map.ndim != 4 or occupancy_map.shape[:2] != (1, 1):
            raise ValueError(f"occupancy_map must have shape (1, 1, H, W), but got {occupancy_map.shape}")
        
        occupancy_np = occupancy_map.squeeze().cpu().numpy().astype(np.uint8)
        height, width = occupancy_np.shape
        
        # 提取基础掩码
        obstacle_mask = (occupancy_np == 2).astype(np.uint8)
        unknown_mask = (occupancy_np == 0).astype(np.uint8)
        
        # 创建碰撞点掩码（将collision点当作障碍物处理）
        collision_mask = np.zeros_like(occupancy_np, dtype=np.uint8)
        for x, y in self.collision_offsets:
            if 0 <= y < height and 0 <= x < width:
                collision_mask[y, x] = 1
        
        # 合并障碍物和碰撞点（Hybrid A*不膨胀，通过足迹检测）
        combined_obstacle_mask = cv2.bitwise_or(obstacle_mask, collision_mask)
        
        # 定义不可达区域（不膨胀障碍物）
        final_impassable_mask = cv2.bitwise_or(combined_obstacle_mask, unknown_mask)
        is_passable = (final_impassable_mask == 0)
        
        # 创建高风险地图（使用Hybrid A*专用参数）
        high_risk_map = np.zeros_like(occupancy_np, dtype=np.float32)
        
        if self.hybrid_safety_margin > 0 and self.hybrid_high_risk_penalty > 0:
            # 高风险基础：原始障碍物 + 原始碰撞点 + 未知区域
            risk_base_mask = cv2.bitwise_or(obstacle_mask, unknown_mask)
            risk_base_mask = cv2.bitwise_or(risk_base_mask, collision_mask)
            risk_base_mask_255 = (risk_base_mask * 255).astype(np.uint8)
            
            risk_kernel_size = 2 * self.hybrid_safety_margin + 1
            risk_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (risk_kernel_size, risk_kernel_size))
            risk_inflated_mask = cv2.dilate(risk_base_mask_255, risk_kernel)
            
            high_risk_zone = (risk_inflated_mask > 0) & (final_impassable_mask == 0)
            high_risk_map[high_risk_zone] = self.hybrid_high_risk_penalty
        
        # 创建居中代价地图（使用Hybrid A*专用参数）
        centering_map = np.zeros_like(occupancy_np, dtype=np.float32)
        
        if self.hybrid_centering_weight > 0:
            dist_map = cv2.distanceTransform(is_passable.astype(np.uint8), cv2.DIST_L2, 5)
            mask = dist_map > 0
            centering_map[mask] = self.hybrid_centering_weight / (dist_map[mask] + 1.0)
        
        # 调用Hybrid A*
        distance = math.sqrt((end_3d[0] - start_3d[0])**2 + (end_3d[1] - start_3d[1])**2)
        max_iterations = min(int(20000 + distance * 200), 50000)
        
        path = self.hybrid_planner.find_path(
            is_passable=is_passable,
            start=start_3d,
            goal=end_3d,
            resolution_scale=self.r_resolution,
            max_iterations=max_iterations,
            high_risk_map=high_risk_map,
            centering_map=centering_map,
            start_adjustment_radius=start_adjustment_radius,
        )
        
        # 返回2D路径
        if path is not None:
            return self.extract_2d_path(path)
        return None

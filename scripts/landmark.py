from PIL import Image
import heapq
import math
import time
import numpy as np
import torch.nn.functional as F
import cv2
import random
import torch
from scipy.ndimage import binary_erosion, binary_dilation, label, center_of_mass, generate_binary_structure, binary_propagation, binary_fill_holes
from scipy.ndimage import sum as ndi_sum
from skimage.draw import line
from utils.llm_utils import llm_auxiliary_frontier
from utils.path_planner import PathPlanner
from utils.skeleton_extract import SkeletonProcessor
from utils.mcts_planner import MCTSPlanner
from utils.utils import _find_closest_point_in_2d_projection
from sklearn.cluster import KMeans
from collections import deque
from scipy.spatial.distance import cdist
from collections import defaultdict

def heuristic(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

class Landmark():
    def __init__(self, id, type, pos):
        self.id = id
        self.type = type
        self.pos = pos
        self.junction_edge = []
        self.endpoint_dege = []
        self.is_fixed = False
        
    def add_edge(self, junctions, endpoints):
        for junction in junctions:
            self.junction_edge.append(junction)
        for endpoint in endpoints:
            self.endpoint_dege.append(endpoints)
        

class RoomNode():
    def __init__(self, id, center, room_name, is_arrive):
        self.id = id
        self.center = center
        self.room_name = room_name  
        self.size = 120  # default size
        self.object_edge = []
        self.is_arrive = is_arrive   # 没有到达的room 用essential obj的center作为room center，到达了的room 用当前的gps作为room center
    def add_object(self, object_node_lst, r_resolution=20):
        for object_node in object_node_lst:
            if heuristic((object_node.center[0] * r_resolution, object_node.center[1] * r_resolution), self.center) < self.size / 2:
                self.object_edge.append(object_node)
    def update_arrived_room(self, agent, observations):
        pixel_x = round(agent.center_pixel + agent.gps[0] * agent.r_resolution)
        pixel_y = round(agent.center_pixel - agent.gps[1] * agent.r_resolution)
        self.is_arrive = True
        self.center = (pixel_x, pixel_y)
    def update_not_arrived_room(self, room_center):
        self.center = room_center

class ucb_counter():
    def __init__(self, eps=0.5, c=1.0, dist_th=30.0):
        self.total_num = 0
        self.eps = eps
        self.c = c
        self.dist_th = dist_th
        self.explored_node_lst = []
    def reset(self):
        self.total_num = 0
        self.explored_node_lst = []
    def update_total_num(self):
        self.total_num += 1
    def add_explored_node(self, coords):
        idx = self.explored_node_detection(coords)
        if idx is not None:
            self.explored_node_lst[idx]["explored_num"] += 1
        else:
            explored_node = {"coords": coords, "explored_num": 1}
            self.explored_node_lst.append(explored_node)
    def explored_node_detection(self, coords):
        for idx, node in enumerate(self.explored_node_lst):
            if heuristic(coords, node["coords"]) < self.dist_th:
                return idx
        return None
    def calculate_score(self, coords):
        idx = self.explored_node_detection(coords)
        if idx is not None:
            explored_num = self.explored_node_lst[idx]["explored_num"]
        else:
            explored_num = 0
        score = self.c * math.sqrt(math.log(self.total_num)/(explored_num + self.eps))
        return score


class LandmarkMap():
    def __init__(self, agent):
        self.agent = agent 
        self.robot_radius = self.agent.robot_radius
        self.landmark_lst = {'junctions': [], "endpoints": []}
        self.room_lst = []
        self.pre_observation = None
        self.similarity_between_pre_and_current = None
        self.pre_occupancy_map = None
        self.escape_point_goal = (self.agent.center_pixel, self.agent.center_pixel)
        self.point_goal = (self.agent.center_pixel, self.agent.center_pixel)
        self.temp_start = (self.agent.center_pixel, self.agent.center_pixel)
        self.turning_cnt = 0
        self.collision_escaping = False
        self.temp_path = []
        self.action_iteration = 0
        self.action_sequence = []
        self.replan = False
        self.collision_escaping_cnt = 0
        self.history_room = []
        self.obj_destination = (self.agent.center_pixel, self.agent.center_pixel)
        self.bad_frontier = set()
        self.approach_new_room = False
        self.closing_kernel_size = self.agent.closing_kernel_size
        self.hole_fill_area = self.agent.hole_fill_area
        self.ucb_counter = ucb_counter()
  
        self.path_planner = PathPlanner(
            robot_radius=self.robot_radius,
            # A* 专用参数
            centering_weight=self.agent.centering_weight,
            safety_margin=self.agent.safety_margin,
            closing_kernel_size=self.closing_kernel_size,
            hole_fill_area=self.hole_fill_area,
            high_risk_penalty=self.agent.high_risk_penalty,
            # Hybrid A* 专用参数
            use_hybrid_astar=self.agent.use_hybrid_astar,
            forward_distance=self.agent.forward_distance,
            turn_angle=self.agent.turn_angle,
            theta_resolution=self.agent.theta_resolution,
            r_resolution=self.agent.r_resolution,
            goal_arrival_factor=self.agent.goal_arrival_factor,
            hybrid_centering_weight=self.agent.hybrid_centering_weight,
            hybrid_safety_margin=self.agent.hybrid_safety_margin,
            hybrid_high_risk_penalty=self.agent.hybrid_high_risk_penalty,
        )
        self.skeleton_processor = SkeletonProcessor(
            dilate_kernel=self.agent.dilate_kernel,
            min_component_size=self.agent.min_component_size,
            pruning_iterations=self.agent.pruning_iterations,
            min_junction_distance=self.agent.min_junction_distance,
            closing_kernel_size=self.closing_kernel_size,
            hole_fill_area=self.hole_fill_area
        )
        self.mcts_planner = MCTSPlanner(
            path_planner=self.path_planner,
            robot_radius=self.robot_radius,
            exploration_weight=self.agent.mcts_exploration_weight
        )
        # self.skeleton_processor = SkeletonProcessor(
        #     dilate_kernel=8,
        #     min_component_size=50,
        #     pruning_iterations=10,
        #     min_junction_distance=15,
        #     closing_kernel_size=self.closing_kernel_size,
        #     hole_fill_area=self.hole_fill_area
        # )


        
    def reset(self):
        self.landmark_lst = {'junctions': [], "endpoints": []}
        self.room_lst = []
        self.pre_observation = None
        self.similarity_between_pre_and_current = None
        self.pre_occupancy_map = None
        self.escape_point_goal = (self.agent.center_pixel, self.agent.center_pixel)
        self.point_goal = (self.agent.center_pixel, self.agent.center_pixel)    
        self.temp_start = (self.agent.center_pixel, self.agent.center_pixel)
        self.obj_destination = (self.agent.center_pixel, self.agent.center_pixel)
        self.turning_cnt = 0
        self.collision_escaping = False
        self.temp_path = []
        self.action_iteration = 0
        self.action_sequence = []
        self.replan = False
        self.collision_escaping_cnt = 0
        self.history_room = []
        self.bad_frontier = set()
        self.approach_new_room = False
        self.ucb_counter.reset()
        self.path_planner.reset()
        # 主动感知相关参数重置
        self.ap_action_index = 0
                                
    def add_room(self, room_center, room_name, is_arrive, observations):
        if is_arrive:
            for room in self.room_lst:
                if heuristic((self.agent.center_pixel + self.agent.gps[0] * self.agent.r_resolution, self.agent.center_pixel - self.agent.gps[1] * self.agent.r_resolution), room.center) < room.size / 2:
                    room.update_arrived_room(self.agent, observations)
                    print(f"update_arrived_room:{room.room_name}")
                    return
            # 如果在房间列表中没有找到一个相似的房间
            id = len(self.room_lst)
            temp_center = (self.agent.center_pixel + self.agent.gps[0] * self.agent.r_resolution, self.agent.center_pixel - self.agent.gps[1] * self.agent.r_resolution)
            new_node = RoomNode(id=id, center=temp_center, room_name=room_name, is_arrive=True)   # if not is_arrive 我标记的room center就是essential object center 不做任何修改
            self.room_lst.append(new_node)
        else:  # 远端更新房间
            for room in self.room_lst:
                # if room.room_name == room_name and heuristic(room_center, room.center) < room.size / 2:
                if heuristic(room_center, room.center) < room.size / 2:
                    room.update_not_arrived_room(room_center)
                    return
            # 如果在房间列表中没有找到一个相似的房间
            id = len(self.room_lst)
            temp_center = room_center
            new_node = RoomNode(id=id, center=temp_center, room_name=room_name, is_arrive=False)   # if not is_arrive 我标记的room center就是essential object center 不做任何修改
            self.room_lst.append(new_node)
    
    # def add_landmark(self, junctions, endpoints):
    #     # 在每一次运行local_policy的时候执行add_landmark，在列表中添加Landmark
    #     for endpoint in endpoints
    
    def siglip_similarity(self, observations, vision_model, vision_processor):
        if self.pre_observation is None:
            pre_observation_temp = observations["rgb"]
        else:
            pre_observation_temp = self.pre_observation   # 读取上一个observation
        self.pre_observation = observations["rgb"]
        # torch.cuda.empty_cache()
        with torch.no_grad():
            pre_input = vision_processor(images=[pre_observation_temp], return_tensors="pt").to(vision_model.device)
            pre_embedding = vision_model.get_image_features(**pre_input)
            pre_embedding = F.normalize(pre_embedding, p=2, dim=-1)
            current_input = vision_processor(images=[observations["rgb"]], return_tensors="pt").to(vision_model.device)
            current_embedding = vision_model.get_image_features(**current_input)
            current_embedding = F.normalize(current_embedding, p=2, dim=-1)
        similarity = F.cosine_similarity(pre_embedding, current_embedding, dim=1)
        return similarity

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
            
    # def unexplored_exploring(self, occupancy_map, candidate, bfs_radius=30):
    #     # processed_map = occupancy_map.squeeze().cpu().numpy()
    #     # H, W = processed_map.shape
    #     # # TODO: cancel the radiation
    #     # # dilation_radius = 5
    #     # # obstacle_mask = (processed_map == 2)
    #     # # y, x = np.ogrid[-dilation_radius:dilation_radius+1, -dilation_radius:dilation_radius+1]
    #     # # kernel = x**2 + y**2 <= dilation_radius**2
    #     # # dilated_obstacles = binary_dilation(obstacle_mask, structure=kernel)
    #     # # processed_map[dilated_obstacles] = 2

    #     # non_unexplored_mask = (processed_map != 0)
    #     # filled_mask = binary_fill_holes(non_unexplored_mask)
    #     # holes_mask = filled_mask & ~non_unexplored_mask
    #     # processed_map[holes_mask] = 1
        
    #     processed_map = occupancy_map.squeeze().cpu().numpy()
    #     H, W = processed_map.shape
    #     # H, W = processed_map.shape
    #     # unknown_mask = (processed_map == 0.0).astype(np.uint8)
    #     # unknown_mask_gray = unknown_mask * 255
    #     # unknown_mask_filled = self._fill_small_holes(unknown_mask_gray)
    #     # non_unexplored_mask = cv2.bitwise_not(unknown_mask_filled)
    #     # non_unexplored_mask[non_unexplored_mask==255] = 1
    #     # holes_mask = non_unexplored_mask & unknown_mask
    #     # processed_map[holes_mask] = 1

    #     start_x, start_y = candidate
    #     if processed_map[start_y, start_x] == 2:
    #         return 0, 0

    #     queue = deque([(start_x, start_y)])  # 队列存储 (x, y) 坐标
    #     visited = set([(start_x, start_y)])  # 使用集合来高效地跟踪已访问的节点
    #     frontier_pixel_count = 0             # 最终要返回的计数器

    #     while queue:
    #         curr_x, curr_y = queue.popleft()
    #         is_free_space = (processed_map[curr_y, curr_x] > 0 and processed_map[curr_y, curr_x] < 2)

    #         if is_free_space:
    #             has_unexplored_neighbor = False
    #             for dy in [-1, 0, 1]:
    #                 for dx in [-1, 0, 1]:
    #                     if dx == 0 and dy == 0:
    #                         continue  

    #                     neighbor_x, neighbor_y = curr_x + dx, curr_y + dy
    #                     if 0 <= neighbor_y < H and 0 <= neighbor_x < W:
    #                         if processed_map[neighbor_y, neighbor_x] == 0:
    #                             has_unexplored_neighbor = True
    #                             break 
    #                 if has_unexplored_neighbor:
    #                     break  
                
    #             if has_unexplored_neighbor:
    #                 frontier_pixel_count += 1

    #         for dy in [-1, 0, 1]:
    #             for dx in [-1, 0, 1]:
    #                 if dx == 0 and dy == 0:
    #                     continue

    #                 next_x, next_y = curr_x + dx, curr_y + dy
    #                 if not (0 <= next_y < H and 0 <= next_x < W):
    #                     continue
    #                 if (next_x, next_y) in visited:
    #                     continue
    #                 if processed_map[next_y, next_x] == 2:
    #                     continue
    #                 distance_from_start = np.sqrt((next_x - start_x)**2 + (next_y - start_y)**2)
    #                 if distance_from_start > bfs_radius:
    #                     continue
    #                 visited.add((next_x, next_y))
    #                 queue.append((next_x, next_y))

    #     return frontier_pixel_count, len(visited)
    
    def unexplored_exploring(self, occupancy_map: torch.Tensor, candidate: tuple, bfs_radius: int = 30, return_points: bool = False):
        """
        计算candidate可以探索到的未探索点
        
        Args:
            occupancy_map: 占用地图
            candidate: 候选点坐标 (x, y)
            bfs_radius: BFS搜索半径（步数）
            return_points: 是否返回未探索点集合
            
        Returns:
            如果return_points=False: 返回未探索点数量(int)
            如果return_points=True: 返回(未探索点数量, 未探索点集合)
        """
        processed_map = occupancy_map.squeeze().cpu().numpy()
        H, W = processed_map.shape
        start_x, start_y = candidate
        
        if not (0 <= start_y < H and 0 <= start_x < W):
            return (0, set()) if return_points else 0
        
        if processed_map[start_y, start_x] == 2:
            return (0, set()) if return_points else 0

        q = deque([((start_x, start_y), 0)])
        visited = { (start_x, start_y) }
        unexplored_area_count = 0
        unexplored_points_set = set()
        
        if processed_map[start_y, start_x] == 0:
            unexplored_area_count += 1
            if return_points:
                unexplored_points_set.add((start_x, start_y))
                
        while q:
            (current_x, current_y), dist = q.popleft()
            if dist >= bfs_radius:
                continue
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_x, next_y = current_x + dx, current_y + dy
        
                if not (0 <= next_y < H and 0 <= next_x < W):
                    continue
                if (next_x, next_y) in visited:
                    continue
                neighbor_value = processed_map[next_y, next_x]
                visited.add((next_x, next_y))
                if neighbor_value == 2:
                    continue
                if neighbor_value == 0:
                    unexplored_area_count += 1
                    if return_points:
                        unexplored_points_set.add((next_x, next_y))
                q.append(((next_x, next_y), dist + 1))
        
        if return_points:
            return unexplored_area_count, unexplored_points_set
        else:
            return unexplored_area_count
        
    
    def calculate_angle(self, now, goal, current_heading):
        # 计算目标朝向与当前朝向的夹角
        delta_x = goal[0] - now[0]
        delta_y_actual = now[1] - goal[1]
        angle_rad = math.atan2(delta_y_actual, delta_x)
        target_angle = math.degrees(angle_rad) % 360
        
        # 计算与当前朝向的角度差
        delta_angle = (target_angle + current_heading) % 360
        if delta_angle > 180:
            delta_angle -= 360 
        return delta_angle   
    
    def calculate_heading(self, now, goal):
        if self.temp_start != now:
            vector_start_now = np.array((now[0]-self.temp_start[0], now[1]-self.temp_start[1]))
            vector_now_goal = np.array((goal[0]-now[0], goal[1]-now[1]))
            cos_angle = np.dot(vector_start_now, vector_now_goal) / (np.linalg.norm(vector_start_now) * np.linalg.norm(vector_now_goal))
            angle_rad = np.degrees(np.arccos(cos_angle))
        else:
            angle_rad = 0.0
    
        return angle_rad  
    
    
    # def _update_temp_start(self, now):
    #     dx = now[0] - self.temp_start[0]
    #     dy = now[1] - self.temp_start[1]
        
    #     if dx == 0 and dy == 0:
    #         raise ValueError("Points A and B must be distinct.")
        
    #     g = math.gcd(abs(dx), abs(dy))
    #     step_x = dx // g
    #     step_y = dy // g
        
    #     self.temp_start = (round(now[0] + step_x), round(now[1] + step_y))
    
    def _update_temp_start(self, now, step_size=10.0):
        dx = now[0] - self.temp_start[0]
        dy = now[1] - self.temp_start[1]
        length = math.sqrt(dx**2 + dy**2)
        step_x = (dx / length) * step_size
        step_y = (dy / length) * step_size
        self.temp_start = (round(now[0] + step_x), round(now[1] + step_y))
   
   
    def _find_escape_point(self, point, search_radius=None, max_distance=None):
        if search_radius is None:
            search_radius = self.agent.escape_point_search_radius
        if max_distance is None:
            max_distance = self.agent.escape_point_max_distance
        ax, ay = point[0], point[1]
        map_array = self.agent.occupancy_map[0,0].detach().cpu().numpy()
        obstacle_mask = (map_array == 2.0).astype(np.uint8)
        # add dilation
        if self.agent.robot_radius % 2 == 0:
            dilation_radius = self.agent.robot_radius + 1
        kernel = np.ones((dilation_radius, dilation_radius), np.uint8)
        obstacle_mask = cv2.dilate(obstacle_mask, kernel, iterations=3)
        # ---
        unexplored_map = (map_array == 0.0).astype(np.uint8)
        H, W = map_array.shape
        force_x, force_y = 0.0, 0.0
        for y in range(max(0, ay-search_radius), min(H, ay+search_radius+1)):
            for x in range(max(0, ax-search_radius), min(W, ax+search_radius+1)):
                if obstacle_mask[y, x] or unexplored_map[y, x]:
                    dx = x - ax
                    dy = y - ay  
                    dist_sq = dx**2 + dy**2
                    if dist_sq == 0:
                        continue
                    weight = 2.0 if obstacle_mask[y, x] else 0.8
                    force_x -= weight * dx / dist_sq
                    force_y -= weight * dy / dist_sq 
                    
        if math.hypot(force_x, force_y) < 1e-5:
            escape_dir = (1, 1)  
        else:
            norm = math.hypot(force_x, force_y)
            escape_dir = (force_x/norm, force_y/norm)
        best_score = -float('inf')
        best_point = None
        
        for angle in np.linspace(-15, 15, 7):
            theta = math.radians(angle)
            rot_dir = (
                escape_dir[0]*math.cos(theta) - escape_dir[1]*math.sin(theta),
                escape_dir[0]*math.sin(theta) + escape_dir[1]*math.cos(theta)
            )

            for dist in np.linspace(max_distance, 2, 10):
                bx = int(round(ax + rot_dir[0] * dist))
                by = int(round(ay + rot_dir[1] * dist))  
                
                if not (0 <= bx < W and 0 <= by < H):
                    continue
                    
                if not (0 < map_array[by, bx] < 2):
                    continue
                    
                path_valid = True
                steps = max(abs(bx-ax), abs(by-ay))
                for t in np.linspace(0, 1, steps+1):
                    px = int(ax + t*(bx-ax))
                    py = int(ay + t*(by-ay)) 
                    if not (0 <= px < W and 0 <= py < H):
                        path_valid = False
                        break
                    if obstacle_mask[py, px]:
                        path_valid = False
                        break
                        
                if path_valid:
                    safe_count = 0
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            cx = bx + dx
                            cy = by + dy
                            if 0 <= cx < W and 0 <= cy < H:
                                if 0 < map_array[cy, cx] < 2:
                                    safe_count += 1

                    if safe_count > best_score:
                        best_score = safe_count
                        best_point = (bx, by)
        
        return best_point if best_point else None
    
    def _find_nearest_valid_point(self, occupancy_map, start, point, max_radius, inflated_radius=None, sampling_step=5):
        map_array = occupancy_map[0,0].detach().cpu().numpy()
        h, w = map_array.shape
        obstacle_mask = (map_array == 2.0).astype(np.uint8)
        
        kernel_size = int(2 * self.robot_radius) + 1 
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        inflated_obstacles = cv2.dilate(obstacle_mask, kernel)
        safe_navigation_space = np.ones_like(map_array, dtype=bool)
        safe_navigation_space[inflated_obstacles == 1] = False
        
        
        px, py = point[0], point[1]
        if 0 <= px < w and 0 <= py < h and safe_navigation_space[py, px]:
            if self.path_planner.find_path_astar(occupancy_map=occupancy_map, start=start, end=point, start_adjustment_radius=int(self.robot_radius)+1):
                return point # 原始 point 有效且可达
        
        # 优化的采样策略：稀疏半径级别，随半径增大增加采样密度
        # 小半径：稀疏采样（step大），大半径：密集采样（step小）
        radius_levels = [5, 10, 20, 30]  # 只在关键半径上搜索
        
        for r in radius_levels:
            if r > max_radius:
                break
            
            # 根据半径动态调整采样步长：半径越大，步长越小（采样越密）
            if r <= 5:
                current_sampling_step = 10   # 小半径：很稀疏
            elif r <= 10:
                current_sampling_step = 8
            elif r <= 20:
                current_sampling_step = 6
            else:
                current_sampling_step = 5    # 大半径：较密集
            
            candidate_points_at_radius_r = []
            for i in range(-r, r + 1, current_sampling_step):
                candidate_points_at_radius_r.append((point[0] + i, point[1] - r)) # 上
                candidate_points_at_radius_r.append((point[0] + i, point[1] + r)) # 下
                candidate_points_at_radius_r.append((point[0] - r, point[1] + i)) # 左
                candidate_points_at_radius_r.append((point[0] + r, point[1] + i)) # 右
            
            candidate_points_at_radius_r = list(set(candidate_points_at_radius_r))
            
            for np_x, np_y in candidate_points_at_radius_r:
                new_point = (np_x, np_y)
                if 0 <= new_point[0] < w and 0 <= new_point[1] < h and safe_navigation_space[new_point[1], new_point[0]]:
                    if self.path_planner.find_path_astar(occupancy_map=occupancy_map, start=start, end=new_point, start_adjustment_radius=int(self.robot_radius)+1):
                        return new_point 
        return None

    def _find_object_destination(self, occupancy_map, start, point, max_radius, inflated_radius=None, sampling_step=5):
        map_array = occupancy_map[0, 0].detach().cpu().numpy()
        h, w = map_array.shape
        obstacle_mask = (map_array == 2.0).astype(np.uint8)

        kernel_size = int(2 * self.robot_radius) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        inflated_obstacles = cv2.dilate(obstacle_mask, kernel)
        safe_navigation_space = np.ones_like(map_array, dtype=bool)
        safe_navigation_space[inflated_obstacles == 1] = False
        
        px, py = point[0], point[1]
        start_x, start_y = start[0], start[1]

        if 0 <= px < w and 0 <= py < h and safe_navigation_space[py, px]:
            if self.path_planner.find_path_astar(occupancy_map=occupancy_map, start=start, end=point, start_adjustment_radius=int(self.robot_radius) + 1):
                return point

        priority_candidates = []  
        other_candidates = []     

        robot_dir_x = start_x - px
        robot_dir_y = start_y - py
        
        for r in range(1, max_radius + 1):
            points_at_radius_r = set()
            for i in range(-r, r + 1, sampling_step):
                points_at_radius_r.add((px + i, py - r)) # 上
                points_at_radius_r.add((px + i, py + r)) # 下
                points_at_radius_r.add((px - r, py + i)) # 左
                points_at_radius_r.add((px + r, py + i)) # 右
            
            for p in points_at_radius_r:
                candidate_dir_x = p[0] - px
                candidate_dir_y = p[1] - py
                
                dot_product = robot_dir_x * candidate_dir_x + robot_dir_y * candidate_dir_y
                
                if dot_product > 0:
                    priority_candidates.append(p)
                else:
                    other_candidates.append(p)
        # print("[819DEBUG-priority_candidates]", priority_candidates)
        # print("[819DEBUG-other_candidates]", other_candidates)
        for new_point in priority_candidates:
            if 0 <= new_point[0] < w and 0 <= new_point[1] < h and safe_navigation_space[new_point[1], new_point[0]]:
                if self.path_planner.find_path_astar(occupancy_map=occupancy_map, start=start, end=new_point, start_adjustment_radius=int(self.robot_radius) + 1):
                    return new_point 
        
        for new_point in other_candidates:
            if 0 <= new_point[0] < w and 0 <= new_point[1] < h and safe_navigation_space[new_point[1], new_point[0]]:
                if self.path_planner.find_path_astar(occupancy_map=occupancy_map, start=start, end=new_point, start_adjustment_radius=int(self.robot_radius) + 1):
                    return new_point

        return None
    
    
    def object_clustering(self, object_center_array, landmark_array, max_radius=None,  max_objects_per_landmark=None):
        if max_radius is None:
            max_radius = self.agent.scene_radius  # 使用scene_radius作为默认值
        if max_objects_per_landmark is None:
            max_objects_per_landmark = self.agent.max_objects_per_landmark
            
        num_landmarks = landmark_array.shape[0]
        num_objects = object_center_array.shape[0]

        if num_landmarks == 0:
            return {}
        if num_objects == 0:
            # return [{i: [] for i in range(num_landmarks)}] # 每个landmark都对应一个空列表
            return {}

        dist_matrix = cdist(landmark_array, object_center_array, 'euclidean')
        matching_dict = {}
        for i in range(num_landmarks):
            distances_to_objects = dist_matrix[i, :]
            candidate_indices_for_landmark = np.where(distances_to_objects <= max_radius)[0]

            if candidate_indices_for_landmark.size == 0:
                matching_dict[i] = []
                continue
            distances_of_candidates = distances_to_objects[candidate_indices_for_landmark]
            objects_with_dist = list(zip(candidate_indices_for_landmark, distances_of_candidates))
            objects_with_dist.sort(key=lambda x: x[1]) 

            selected_object_indices = [
                obj_idx for obj_idx, dist in objects_with_dist[:max_objects_per_landmark]
            ]
            matching_dict[i] = selected_object_indices
        return matching_dict
    
    def _calculate_relative_direction(self, landmark_pos, object_pos):
        """计算物体相对于地标的8个基本方位。"""
        # 注意：这里的坐标系可能需要根据您的系统进行调整
        # 假设 Y 轴向上为前方，X 轴向右为右方
        delta_x = object_pos[0] - landmark_pos[0]
        delta_y = object_pos[1] - landmark_pos[1]
        
        angle = math.degrees(math.atan2(delta_y, delta_x)) # -180 to 180

        if -22.5 <= angle < 22.5:
            return "to the right"
        elif 22.5 <= angle < 67.5:
            return "in front-right"
        elif 67.5 <= angle < 112.5:
            return "in front"
        elif 112.5 <= angle < 157.5:
            return "in front-left"
        elif angle >= 157.5 or angle < -157.5:
            return "to the left"
        elif -157.5 <= angle < -112.5:
            return "behind-left"
        elif -112.5 <= angle < -67.5:
            return "behind"
        elif -67.5 <= angle < -22.5:
            return "behind-right"
        return "nearby"
    
    def choose_frontier_llm(self, candidates):
        NUM_CLUSTERS = 5  # 希望LLM评估的地标数量
        SCENE_RADIUS = 80 # 观察地标周围多大范围内的物体 (单位：栅格)

        if not candidates or len(candidates) < 5:
            return {}

        candidate_coords = np.array([c[1][0] for c in candidates])
        
        kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0, n_init='auto').fit(candidate_coords)
        cluster_labels = kmeans.labels_
            
        selected_landmarks_info = {} # {cluster_id: (original_candidate_index, coord)}
        for i in range(5):
            indices_in_cluster = np.where(cluster_labels == i)[0]
            if len(indices_in_cluster) == 0:
                continue
    
            best_candidate_in_cluster_idx = max(indices_in_cluster, key=lambda j: candidates[j][0][0])
            
            selected_landmarks_info[i] = {
                "original_idx": best_candidate_in_cluster_idx,
                "coord": candidates[best_candidate_in_cluster_idx][1][0]
            }

        landmark_scenes = {}
        for cluster_id, landmark_info in selected_landmarks_info.items():
            landmark_coord = landmark_info["coord"]
            nearby_objects = []
            
            for obj_node in self.agent.scenegraph.object_node_lst:
                obj_coord = (obj_node.center[0] * self.agent.r_resolution, obj_node.center[1] * self.agent.r_resolution)
                distance = heuristic(landmark_coord, obj_coord)
                
                if distance < SCENE_RADIUS:
                    relative_dir = self._calculate_relative_direction(landmark_coord, obj_coord)
                    nearby_objects.append({
                        "object_name": obj_node.class_name,
                        "distance": f"{distance/20.0:.1f}m", # 转换为米
                        "direction": relative_dir
                    })
            
            landmark_scenes[cluster_id] = nearby_objects

        if not landmark_scenes:
            return {}

        raw_scores = llm_auxiliary_frontier(self.agent.client, self.agent.model, self.agent.obj_goal, landmark_scenes, authorization=self.agent.authorization)

        llm_final_scores = {}
        for cluster_id, score in raw_scores.items():
            cluster_id = int(cluster_id)
            if cluster_id in selected_landmarks_info:
                original_idx = selected_landmarks_info[cluster_id]["original_idx"]
                normalized_score = max(0.0, min(1.0, score / 10.0))
                llm_final_scores[original_idx] = normalized_score
                
        return llm_final_scores

    def choose_frontier(self, occupancy_map, current_pos, candidates, max_adjustment_radius=None, w_angle=1.0, w_dist=1.0, w_explore=1.0, w_avoid=1.0, mcts_iterations=None):
        """
        使用MCTS算法选择最优frontier
        
        Args:
            occupancy_map: 占用地图
            current_pos: 当前位置
            candidates: 候选frontier列表
            max_adjustment_radius: 最大调整半径
            w_angle, w_dist, w_explore, w_avoid: 评分权重
            mcts_iterations: MCTS迭代次数（会根据候选点数量自适应）
        
        Returns:
            最优frontier坐标，或None
        """
        if not candidates:
            return None
        
        # 从配置读取参数
        if max_adjustment_radius is None:
            max_adjustment_radius = self.agent.max_adjustment_radius
        if mcts_iterations is None:
            mcts_iterations = self.agent.mcts_iterations
        
        print(f"[MCTS] Starting with {len(candidates)} candidates")
        
        # ========== 阶段1.1: 先计算所有候选点的explore_score（BFS探索）==========
        print(f"[MCTS] Computing exploration potential for {len(candidates)} candidates...")
        candidates_with_exploration = []
        min_unexplored, max_unexplored = float('inf'), 0.0
        bfs_radius = self.agent.bfs_radius
        
        for candidate in candidates:
            # 直接在原始候选点上计算探索潜力（无需先验证可达性）
            unexplored, unexplored_points_set = self.unexplored_exploring(
                occupancy_map, candidate[0], bfs_radius=bfs_radius, return_points=True
            )
            
            candidates_with_exploration.append({
                'original_coord': candidate[0],
                'original_data': candidate[1],
                'unexplored_count': unexplored,
                'unexplored_points': unexplored_points_set
            })
            min_unexplored = min(min_unexplored, unexplored)
            max_unexplored = max(max_unexplored, unexplored)
        
        # 归一化explore_score
        unexplored_range = max_unexplored - min_unexplored
        for cand_data in candidates_with_exploration:
            cand_data['explore_score'] = (
                (cand_data['unexplored_count'] - min_unexplored) / unexplored_range 
                if unexplored_range > 0 else 0.0
            )
        
        # 按explore_score从高到低排序（优先处理高探索价值的点）
        candidates_with_exploration.sort(key=lambda x: x['explore_score'], reverse=True)
        
        # ========== 阶段1.2: 可达性过滤 + 聚类去重（合并执行）==========
        print(f"[MCTS] Performing reachability filter with clustering...")
        filtered_candidates = []
        valid_coords_list = []  # 已经验证可达的点列表
        clustering_dist = self.agent.frontier_clustering_distance
        skipped_count = 0
        unreachable_count = 0
        
        for cand_data in candidates_with_exploration:
            original_coord = cand_data['original_coord']
            
            # 聚类检查：如果与已有的valid点太近，直接跳过（省略可达性验证）
            if any(heuristic(original_coord, valid_coord) < clustering_dist for valid_coord in valid_coords_list):
                skipped_count += 1
                continue
            
            # 可达性验证（这是最耗时的操作）
            valid_coord = self._find_nearest_valid_point(
                occupancy_map, current_pos, original_coord, max_radius=max_adjustment_radius
            )
            
            if not valid_coord:
                unreachable_count += 1
                continue
            
            # 二次聚类检查：验证调整后的valid_coord是否与已有点太近
            if any(heuristic(valid_coord, existing_coord) < clustering_dist for existing_coord in valid_coords_list):
                skipped_count += 1
                continue
            
            # 通过所有检查，添加到filtered列表
            filtered_candidates.append({
                'idx': len(filtered_candidates),
                'coord': valid_coord,
                'explore_score': cand_data['explore_score'],
                'unexplored_points': cand_data['unexplored_points']
            })
            valid_coords_list.append(valid_coord)
        
        if not filtered_candidates:
            print("[MCTS] No valid candidates after reachability filter + clustering")
            return None
        
        print(f"[MCTS] {len(filtered_candidates)} candidates after filter (skipped {skipped_count} by clustering, {unreachable_count} unreachable)")
        # ========== 阶段1.4: 预处理 - 计算距离信息（用于distance reward） ==========        
        num_candidates = len(filtered_candidates)
        bad_frontier_list = list(self.bad_frontier) if self.bad_frontier else []
        
        # 辅助函数：计算path长度
        def calc_path_length(path):
            if path is None:
                return -1.0  # 无法找到路径
            length = self.path_planner.compute_path_length(path)
            return length if length > 0 else -1.0
        
        # 计算current_pos到各candidate的path距离（用于distance reward）
        # 使用 A* 方法（提高MCTS速度）
        current_to_candidates_dist = np.array([
            calc_path_length(
                self.path_planner.find_path_astar(occupancy_map, current_pos, cand['coord'])
            )
            for cand in filtered_candidates
        ])
        current_to_candidates_dist[current_to_candidates_dist == 0] = -1.0
        
        # 计算max_candidate_distance（使用直线距离，用于avoidance reward归一化）
        max_candidate_distance = 0.0
        for i in range(num_candidates):
            for j in range(i+1, num_candidates):
                coord1 = filtered_candidates[i]['coord']
                coord2 = filtered_candidates[j]['coord']
                dist = heuristic(coord1, coord2)
                if dist > max_candidate_distance:
                    max_candidate_distance = dist
                
        # ========== 阶段1.5: 收集未探索点信息（复用1.1的BFS结果） ==========
        occ_map_np = occupancy_map.squeeze().cpu().numpy()
        unexplored_points = set(zip(*np.where(occ_map_np == 0)[::-1]))  # (x,y) format
        candidate_unexplored_points = [cand['unexplored_points'] for cand in filtered_candidates]
        
        
        # # Debug: 每个candidate的探索潜力
        # print(f"\n[MCTS Debug] Candidate exploration potential:")
        # for i, cand in enumerate(filtered_candidates):
        #     print(f"  Candidate #{i} at {cand['coord']}: {len(candidate_unexplored_points[i])} unexplored points (BFS radius 30, 4-dir)")

        
        # ========== 阶段1.6: 判断是否需要turning around ==========
        turning_around = True
        
        if w_angle == 0.0:
            turning_around = False
            print(f"[MCTS] w_angle=0, no turning around")
        elif current_pos == self.temp_start:
            turning_around = False
            print(f"[MCTS] current_pos == temp_start, no turning around")
        else:
            candidates_with_angles = [(abs(self.calculate_heading(now=current_pos, goal=cand['coord'])), cand['coord']) 
                                      for cand in filtered_candidates]
            candidates_with_angles.sort()
            top_k_ratio = self.agent.top_k_forward_ratio
            angle_threshold = self.agent.good_forward_angle_threshold
            dist_threshold = self.agent.good_forward_distance_threshold
            top_k = max(1, math.floor(top_k_ratio * len(candidates_with_angles)))
            
            for theta, coord in candidates_with_angles[:top_k]:
                if theta < angle_threshold and heuristic(current_pos, coord) > dist_threshold:
                    turning_around = False
                    print(f"[MCTS] Good forward option: θ={theta:.1f}°, d={heuristic(current_pos, coord):.1f}, no turning")
                    break
        
        if turning_around:
            print(f"[MCTS] Turning around: update temp_start {self.temp_start} → {current_pos}")
            self._update_temp_start(current_pos)
        
        # ========== 阶段2: MCTS搜索 ==========
        print(f"[MCTS] Running {mcts_iterations} iterations (candidates={len(filtered_candidates)})")
        
        # 输出exploration点数最多的前5名candidates
        candidates_with_exploration = [(i, cand['coord'], len(cand['unexplored_points'])) 
                                       for i, cand in enumerate(filtered_candidates)]
        candidates_with_exploration.sort(key=lambda x: x[2], reverse=True)
        print(f"[MCTS] Top 5 candidates by exploration potential:")
        for rank, (idx, coord, num_points) in enumerate(candidates_with_exploration[:5], 1):
            print(f"  #{rank}: Cand#{idx} at {coord} → {num_points} unexplored points")
        
        mcts_start_time = time.time()
        best_candidate = self.mcts_planner.select_best_frontier(
            occupancy_map=occupancy_map,
            current_pos=current_pos,
            candidates=filtered_candidates,
            iterations=mcts_iterations,
            weights={'angle': w_angle, 'dist': w_dist, 'explore': w_explore, 'avoid': w_avoid},
            temp_start=self.temp_start,
            bad_frontier=self.bad_frontier,
            current_to_candidates_dist=current_to_candidates_dist,
            max_candidate_distance=max_candidate_distance,
            unexplored_points=unexplored_points,
            candidate_unexplored_points=candidate_unexplored_points,
            num_bad_frontier=len(bad_frontier_list),
            turning_mode=turning_around  # 传入外部判断的turning标志
        )
        mcts_end_time = time.time()
        print(f"[MCTS] Search time: {mcts_end_time - mcts_start_time:.3f}s")
        
        if best_candidate:
            self.ucb_counter.add_explored_node(best_candidate)
            print(f"[MCTS] Selected frontier: {best_candidate}")
            return best_candidate
        
        return None
    
    def _check_need_turning(self, current_pos, valid_candidates, calculated_data):
        """检查是否需要调头"""
        if current_pos == self.temp_start:
            return False
        
        candidates_with_angles = [(theta, valid_candidates[idx][0]) for theta, _, _, _, idx in calculated_data]
        candidates_with_angles.sort()
        
        top_k_ratio = self.agent.top_k_forward_ratio
        angle_threshold = self.agent.good_forward_angle_threshold
        dist_threshold = self.agent.good_forward_distance_threshold
        top_k = max(1, int(top_k_ratio * len(candidates_with_angles)))
        for theta, coord in candidates_with_angles[:top_k]:
            if theta < angle_threshold and heuristic(current_pos, coord) > dist_threshold:
                return False
        
        print("Need turning around")
        self._update_temp_start(current_pos)
        return True
    
    
    def record(self, observations, occupacy_map):
        self.pre_occupancy_map = occupacy_map
        self.pre_observation = observations['rgb']
    
    def look_down_up(self):
        self.action = [4,5]

    def min_distance_to_mask(self, point, mask):
        nonzero_coords = torch.nonzero(mask, as_tuple=False)
        if nonzero_coords.size(0) == 0:
            return float('inf')

        nonzero_coords = nonzero_coords.to(dtype=torch.float32)
        point = torch.tensor(point, dtype=torch.float32, device=mask.device)

        deltas = nonzero_coords - point.unsqueeze(0)
        distances_sq = torch.sum(deltas ** 2, dim=1)

        min_sq = torch.min(distances_sq)
        min_distance = torch.sqrt(min_sq)
        
        return min_distance.item()
    
    def get_mask_centroid(self, mask):
        nonzero_coords = torch.nonzero(mask, as_tuple=False)
        if nonzero_coords.size(0) == 0:
            return None
        centroid = torch.mean(nonzero_coords.float(), dim=0)
        return (round(centroid[1].item()), round(centroid[0].item()))
    
    def action_gen(self, current_pos, current_heading, goal_adjustment_en=False):
        # 转换朝向为弧度（用于Hybrid A*）
        current_heading_rad = math.radians(current_heading)
        
        # 归一化当前朝向到 [-π, π] 范围
        current_heading_normalized = current_heading_rad
        while current_heading_normalized > math.pi:
            current_heading_normalized -= 2 * math.pi
        while current_heading_normalized < -math.pi:
            current_heading_normalized += 2 * math.pi
        
        # 根据 FSM 状态选择路径规划方法
        current_state = self.agent.FSM.state
        use_hybrid = current_state in ['AO', 'CO', 'AR'] and self.path_planner.use_hybrid_astar
        
        if use_hybrid:
            # 使用 Hybrid A* 进行精确导航
            temp_path = self.path_planner.find_path_hybrid_astar(
                occupancy_map=self.agent.occupancy_map, 
                start=current_pos, 
                end=self.point_goal, 
                start_adjustment_radius=500,
                start_theta=current_heading_normalized,  # 传递归一化后的弧度
                end_theta=None  # 终点朝向不限制
            )
        else:
            # 使用 A* 进行快速路径规划
            temp_path = self.path_planner.find_path_astar(
                occupancy_map=self.agent.occupancy_map, 
                start=current_pos, 
                end=self.point_goal, 
                start_adjustment_radius=500
            )
        
        temp_destination = self.point_goal
        
        # ========== 安全保障机制：Hybrid A* 失败时回退到 A* ==========
        if temp_path is None and use_hybrid:
            temp_path = self.path_planner.find_path_astar(
                occupancy_map=self.agent.occupancy_map, 
                start=current_pos, 
                end=self.point_goal, 
                start_adjustment_radius=500
            )
        
        if temp_path is None:
            if goal_adjustment_en:
                print(f"adjust goal and edit self.point_goal to {self.point_goal}")
                adjustment_radius = self.agent.max_adjustment_radius
                point_goal = self._find_nearest_valid_point(occupancy_map=self.agent.occupancy_map, start=current_pos, point=self.point_goal, max_radius=adjustment_radius)
                if point_goal is None:
                    self.bad_frontier.add(self.point_goal)  
                    self.point_goal = current_pos
                    self.agent.FSM.set_point_goal(self.point_goal)
                    return None
                else:
                    self.point_goal = point_goal
                    self.agent.FSM.set_point_goal(self.point_goal)
                    # 目标调整后重新规划（根据状态选择方法）
                    if use_hybrid:
                        temp_path = self.path_planner.find_path_hybrid_astar(
                            occupancy_map=self.agent.occupancy_map, 
                            start=current_pos, 
                            end=self.point_goal, 
                            start_adjustment_radius=500,
                            start_theta=current_heading_normalized,
                            end_theta=None
                        )
                    else:
                        temp_path = self.path_planner.find_path_astar(
                            occupancy_map=self.agent.occupancy_map, 
                            start=current_pos, 
                            end=self.point_goal, 
                            start_adjustment_radius=500
                        )
                    temp_destination = self.point_goal
            else:
                self.bad_frontier.add(self.point_goal)  
                self.point_goal = current_pos
                self.agent.FSM.set_point_goal(self.point_goal)
                return None

        if temp_path is None:
            return None
        
        # 检查路径类型
        is_hybrid_path = len(temp_path) > 0 and len(temp_path[0]) == 3
        
        if is_hybrid_path:
            # Hybrid A* 模式：直接从路径推导动作
            if len(temp_path) < 2:
                self.action = 1
                return self.path_planner.extract_2d_path(temp_path)
            
            # 获取当前状态和下一个状态
            current_x, current_y, current_theta = temp_path[0]
            next_x, next_y, next_theta = temp_path[1]
            
            # 计算角度差（弧度）
            angle_diff = next_theta - current_theta
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            
            # 判断动作类型
            if abs(angle_diff) > 0.01:
                if angle_diff > 0:
                    self.action = 2  # 左转
                else:
                    self.action = 3  # 右转
            else:
                self.action = 1  # 前进
            
            return self.path_planner.extract_2d_path(temp_path)
        
        else:
            # A* 模式：使用传统的临时目标点方法
            path_2d = self.path_planner.extract_2d_path(temp_path)
            
            nav_temp_goal_dist = self.agent.navigation_temp_goal_distance
            for i, nav_point in enumerate(path_2d):
                if heuristic(current_pos, nav_point) > nav_temp_goal_dist:
                    temp_destination = nav_point
                    break
                elif i == len(path_2d) - 1:
                    temp_destination = nav_point
                    
            delta_angle = self.calculate_angle(now=current_pos, goal=temp_destination, current_heading=current_heading)
                    
            # 决定动作
            turning_angle_threshold = self.agent.turning_angle_threshold
            if delta_angle > turning_angle_threshold:
                self.action = 3  # 右转
            elif delta_angle < -turning_angle_threshold:
                self.action = 2  # 左转
            else:
                self.action = 1  # 直行
            
            # 返回2D路径用于可视化
            return path_2d


    def filter_intersections(self, intersection_points, distance_threshold=20):
        if not intersection_points:
            return []
        threshold_sq = distance_threshold ** 2
        cell_size = float(distance_threshold)
        grid = {}
        representative_points = []

        for point_data in intersection_points:
            coords, value_a = point_data
            px, py = coords

            # 计算点所在的网格坐标
            cell_x = int(px / cell_size)
            cell_y = int(py / cell_size)

            is_too_close = False
            # 检查当前点所在的 3x3 邻域（包括自己的格子）
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    check_cell_x = cell_x + dx
                    check_cell_y = cell_y + dy
                    
                    # 获取邻近格子中的代表点列表
                    # grid.get(key, []) 可以在key不存在时返回一个空列表，避免错误
                    neighbor_points = grid.get((check_cell_x, check_cell_y), [])
                    
                    for rep_point_data in neighbor_points:
                        rep_coords, _ = rep_point_data
                        rep_x, rep_y = rep_coords
                        
                        # 计算距离的平方
                        dist_sq = (px - rep_x)**2 + (py - rep_y)**2
                        
                        if dist_sq < threshold_sq:
                            is_too_close = True
                            break  # 已经发现过近的点，无需再比较
                    if is_too_close:
                        break
            
            # 如果这个点与所有已知的代表点都足够远
            if not is_too_close:
                # 把它作为新的代表点
                representative_points.append(point_data)
                
                # 将它添加到网格中，以备后续检查
                cell_key = (cell_x, cell_y)
                if cell_key not in grid:
                    grid[cell_key] = []
                grid[cell_key].append(point_data)
                
        return representative_points
    
    # def final_policy(self):
    #     obj_destination = (round(self.agent.FSM.confirmed_essential_obj.center[0] * 20), round(self.agent.FSM.confirmed_essential_obj.center[1] * 20)) 
    #     current_heading = math.degrees(self.agent.observations['compass']) % 360  # 获取当前朝向 
    #     delta_angle = self.calculate_angle(now=(self.agent.pixel_x, self.agent.pixel_y), goal=obj_destination, current_heading=current_heading)
                
    #     # 决定动作
    #     if delta_angle > 16:
    #         self.action = 3  # 右转
    #     elif delta_angle < -16:
    #         self.action = 2  # 左转
    #     else:
    #         self.action = 1
    #         self.agent.final_policy_done = True
        
    #     return self.action

    
    def local_policy(self):
        pixel_x = round(self.agent.center_pixel + self.agent.gps[0] * self.agent.r_resolution)
        pixel_y = round(self.agent.center_pixel - self.agent.gps[1] * self.agent.r_resolution)
        current_heading = math.degrees(self.agent.observations['compass']) % 360  # 获取当前朝向 
        print(f"check offsets: {self.path_planner.collision_offsets}")
        
        intersection_points, vis_img, _, endpoints, junctions = self.skeleton_processor.process_image(occupancy_map=self.agent.occupancy_map)
        if len(intersection_points) == 0:
            if self.agent.action == [5, 4]:
                return 0, self.point_goal, None, vis_img
            else:
                return [5, 4], self.point_goal, None, vis_img

        # 考虑是否要添加房间
        if self.agent.FSM.is_add_room[0]:
            essential_obj_center = (round(self.agent.r_resolution * self.agent.FSM.approaching_essential_obj.center[0]), round(self.agent.r_resolution * self.agent.FSM.approaching_essential_obj.center[1]))
            if junctions != []:
                room_center = min(junctions, key=lambda junction: heuristic(essential_obj_center, junction[0]))[0]
            else:
                room_center = min(endpoints, key=lambda endpoint: heuristic(essential_obj_center, endpoint[0]))[0]

            self.add_room(
                room_center=room_center,
                room_name=self.agent.FSM.is_add_room[1],
                is_arrive=self.agent.FSM.is_add_room[2],
                observations=self.agent.observations,
            )
            
            # self.add_place(observations=observations, room=FSM.is_add_room[1], optional_room_list=optional_room_list, vision_model=vision_model, vision_processor=vision_processor, vlm_model=vlm_model, vlm_processor=vlm_processor)
            self.agent.FSM.is_add_room[0] = False

        # # 处理不同状态对应的local policy
        # if self.agent.FSM.state == 'CE':
        #     temp_destination = self._find_escape_point(point=(pixel_x, pixel_y), search_radius=30, max_distance=7)
        #     if temp_destination is None:
        #         self.action = 1
        #         return self.action, temp_destination, None, vis_img
        #     else:
        #         delta_angle = self.calculate_angle(now=(pixel_x, pixel_y), goal=temp_destination, current_heading=current_heading)
                        
        #         # 决定动作
        #         if delta_angle > 16:
        #             self.action = 3  # 右转
        #         elif delta_angle < -16:
        #             self.action = 2  # 左转
        #         else:
        #             self.action = 1  # 直行
                        
        #         return self.action, temp_destination, None, vis_img
        
        # 处理不同状态对应的local policy
        
        if self.agent.FSM.state == 'CE':
            max_escaping = self.agent.max_collision_escaping_count
            if self.collision_escaping_cnt < max_escaping:
                self.action = 6
                self.collision_escaping_cnt += 1
            else:
                self.action = 1
                self.collision_escaping_cnt = 0
            return self.action, None, None, vis_img
        
        elif self.agent.FSM.state == 'CE-plus':
            if self.action == 1:
                self.bad_frontier.add(self.point_goal)  
                self.bad_frontier.add((pixel_x, pixel_y))
                best_endpoint = self.choose_frontier(occupancy_map=self.agent.occupancy_map, current_pos=(pixel_x, pixel_y), candidates=intersection_points, w_angle=0.1, w_dist=0.1, w_explore=0.1, w_avoid=1.0)
                if best_endpoint is None:
                    return 0, self.point_goal, None, vis_img
                else:
                    self.point_goal = best_endpoint
                    self.agent.FSM.set_point_goal(self.point_goal) 
        
            temp_path = self.action_gen(current_pos=(pixel_x, pixel_y), current_heading=current_heading)
            return self.action, self.point_goal, temp_path, vis_img  
                  
        elif self.agent.FSM.state == 'BS':  
            # 重置obj_destination，确保下次进入AO/CO状态时能正确计算
            self.obj_destination = (self.agent.center_pixel, self.agent.center_pixel)
            replan = False
            bs_replan_threshold = self.agent.bs_replan_distance_threshold  # BS状态专用的重新规划距离阈值
            
            # 检查是否需要重新选择frontier（防卡死机制触发）
            if self.agent.FSM.need_reselect_frontier:
                replan = True
                self.agent.FSM.need_reselect_frontier = False
            
            if heuristic((pixel_x, pixel_y), self.point_goal) >= bs_replan_threshold or (pixel_x, pixel_y) == (self.agent.center_pixel, self.agent.center_pixel):
                temp_path = self.action_gen(current_pos=(pixel_x, pixel_y), current_heading=current_heading, goal_adjustment_en=True)
                if temp_path is None:
                    print("Navigation Failed!!")
                    replan = True            
                        
            if heuristic((pixel_x, pixel_y), self.point_goal) < bs_replan_threshold or replan:                
                # 加一个转圈的逻辑
                max_turning = self.agent.max_turning_count
                if self.turning_cnt < max_turning and (pixel_x, pixel_y) != (self.agent.center_pixel, self.agent.center_pixel) and not replan:
                    self.turning_cnt += 1
                    self.action = 6
                    return self.action, self.point_goal, None, vis_img        
                else:
                    self.turning_cnt = 0
                    self.bad_frontier.add(self.point_goal)
                    if replan:
                        best_endpoint = self.choose_frontier(occupancy_map=self.agent.occupancy_map, current_pos=(pixel_x, pixel_y), candidates=intersection_points, w_angle=1.0, w_dist=0.0, w_explore=0.0, w_avoid=4.5)
                    else:
                        best_endpoint = self.choose_frontier(occupancy_map=self.agent.occupancy_map, current_pos=(pixel_x, pixel_y), candidates=endpoints, w_angle=2.0, w_dist=1.0, w_explore=1.5, w_avoid=1.0)
                    
                    if best_endpoint is None:
                        print("all endpoint is unreachable")
                        # 使用更大的搜索半径重试
                        max_radius_fallback = self.agent.max_adjustment_radius * 3
                        best_endpoint = self.choose_frontier(occupancy_map=self.agent.occupancy_map, current_pos=(pixel_x, pixel_y), candidates=intersection_points, max_adjustment_radius=max_radius_fallback, w_angle=2.0, w_dist=1.0, w_explore=1.5, w_avoid=1.0)
                        if best_endpoint is None:
                            print("all intersections is unreachable")
                            return 0, None, None, vis_img
                        
                    self.point_goal = best_endpoint
                    self.agent.FSM.set_point_goal(self.point_goal) 
                    temp_path = self.action_gen(current_pos=(pixel_x, pixel_y), current_heading=current_heading)
                    if temp_path is None:
                        print("Assert is not None")
                        return 0, None, None, vis_img
            
            return self.action, self.point_goal, temp_path, vis_img
 
        
        elif self.agent.FSM.state == 'AR':
            self.point_goal = self.agent.FSM.point_goal
            temp_path = self.action_gen(current_pos=(pixel_x, pixel_y), current_heading=current_heading, goal_adjustment_en=True)
            return self.action, self.point_goal, temp_path, vis_img
        
        
        elif self.agent.FSM.state == 'AO':
            obj_destination = _find_closest_point_in_2d_projection(self.agent.FSM.confirmed_essential_obj.pcd, (pixel_x, pixel_y), r_resolution=self.agent.r_resolution)
            obj_stable_distance = self.agent.object_close_confirmation_distance
            if heuristic(obj_destination, self.obj_destination) < obj_stable_distance:   # 保持稳定，保持self.point_goal不变
                temp_path = self.action_gen(current_pos=(pixel_x, pixel_y), current_heading=current_heading, goal_adjustment_en=True)
                return self.action, self.point_goal, temp_path, vis_img
            else:
                self.obj_destination = obj_destination
                obj_dest_radius = self.agent.object_destination_radius
                nearest_free_point = self._find_object_destination(occupancy_map=self.agent.occupancy_map, start=(pixel_x, pixel_y), point=self.obj_destination, max_radius=obj_dest_radius)  
                if nearest_free_point is None:     # target object is in obstacles or target object is unreachable
                    print("target object is in obstacles or target object is unreachable")
                    self.agent.FSM.skip_blocked_target(self.agent.scenegraph.object_node_lst)
                    self.action = 2
                    return self.action, None, None, vis_img
                else:
                    self.point_goal = nearest_free_point
                    self.agent.FSM.set_point_goal(self.point_goal)
                    temp_path = self.action_gen(current_pos=(pixel_x, pixel_y), current_heading=current_heading, goal_adjustment_en=True)
                    return self.action, self.point_goal, temp_path, vis_img
        
        elif self.agent.FSM.state == 'AP':  # Active Perception 主动感知状态
            print("[Active Perception] Generating action sequence for active perception")
            
            # 初始化动作序列（第一次进入AP状态时生成）
            if not hasattr(self, 'ap_action_sequence') or self.ap_action_index == 0:
                self.ap_action_index = 0
                
                # 根据bbox与边界的相交情况生成动作序列
                action_sequence = []
                
                # 获取需要转动的方向
                directions = self.agent.FSM.active_perception_directions
                print(f"[Active Perception] BBox intersects boundaries: {directions}")
                
                if directions:
                    # 根据相交的边界生成动作序列
                    # 左右转不需要回正，上下转需要回正
                    
                    # 处理水平方向（左右转）
                    if 'left' in directions:
                        print("[Active Perception] Adding LEFT turn action")
                        action_sequence.append(2)  # 左转
                    if 'right' in directions:
                        print("[Active Perception] Adding RIGHT turn action")
                        action_sequence.append(3)  # 右转
                    
                    # 处理垂直方向（上下转，需要回正）
                    if 'up' in directions:
                        print("[Active Perception] Adding LOOK UP action")
                        action_sequence.append(4)  # 向上看
                        action_sequence.append(5)  # 向下看回来（回正）
                    if 'down' in directions:
                        print("[Active Perception] Adding LOOK DOWN action")
                        action_sequence.append(5)  # 向下看
                        action_sequence.append(4)  # 向上看回来（回正）
                    
                    # 保存生成的动作序列
                    self.ap_action_sequence = action_sequence
                    print(f"[Active Perception] Generated action sequence: {action_sequence} (total {len(action_sequence)} actions)")
                else:
                    # 如果没有方向信息，使用默认序列
                    print("[Active Perception] Warning: No direction info, using default sequence")
                    self.ap_action_sequence = [2, 3, 4, 5]
            
            # 检查是否完成所有动作
            if self.ap_action_index >= len(self.ap_action_sequence) - 1:
                # 动作序列完成，设置标志
                print("[Active Perception] Action sequence completed")
                self.action = self.ap_action_sequence[-1]
                self.agent.FSM.active_perception_action_done = True
                self.ap_action_index = 0  # 重置计数器

                return self.action, self.point_goal, None, vis_img
            else:
                # 执行当前动作
                current_action = self.ap_action_sequence[self.ap_action_index]
                print(f"[Active Perception] Executing action {self.ap_action_index + 1}/{len(self.ap_action_sequence)}: {current_action}")
                
                self.ap_action_index += 1
                self.action = current_action
                return self.action, self.point_goal, None, vis_img
        
        # elif self.agent.FSM.state == 'CO':
        #     obj_destination = _find_closest_point_in_2d_projection(self.agent.FSM.confirmed_essential_obj.pcd, (pixel_x, pixel_y), r_resolution=self.agent.r_resolution)
        #     if heuristic(obj_destination, self.obj_destination) < 3:   # 保持稳定
        #         temp_path = self.action_gen(current_pos=(pixel_x, pixel_y), current_heading=current_heading, goal_adjustment_en=True)
        #         return self.action, self.point_goal, temp_path, vis_img
        #     else:
        #         self.obj_destination = obj_destination
        #         nearest_free_point = self._find_object_destination(occupancy_map=self.agent.occupancy_map, start=(pixel_x, pixel_y), point=self.obj_destination, max_radius=20)  
        #         if nearest_free_point is None:     # target object is in obstacles or target object is unreachable
        #             print("target object is in obstacles or target object is unreachable")
        #             self.agent.FSM.skip_blocked_target(self.agent.scenegraph.object_node_lst)
        #             self.action = 2
        #             return self.action, None, None, vis_img
        #         else:
        #             self.point_goal = nearest_free_point
        #             self.agent.FSM.set_point_goal(self.point_goal)
        #             temp_path = self.action_gen(current_pos=(pixel_x, pixel_y), current_heading=current_heading)
        #             return self.action, self.point_goal, temp_path, vis_img
        
        elif self.agent.FSM.state == 'CO':
            self.obj_destination = _find_closest_point_in_2d_projection(self.agent.FSM.confirmed_essential_obj.pcd, (pixel_x, pixel_y), r_resolution=self.agent.r_resolution)
            obj_dest_radius = self.agent.object_destination_radius
            nearest_free_point = self._find_object_destination(occupancy_map=self.agent.occupancy_map, start=(pixel_x, pixel_y), point=self.obj_destination, max_radius=obj_dest_radius)  
            if nearest_free_point is None:     # target object is in obstacles or target object is unreachable
                print("target object is in obstacles or target object is unreachable")
                self.agent.FSM.skip_blocked_target(self.agent.scenegraph.object_node_lst)
                self.action = 2
                return self.action, None, None, vis_img
            else:
                self.point_goal = nearest_free_point
                self.agent.FSM.set_point_goal(self.point_goal)
                temp_path = self.action_gen(current_pos=(pixel_x, pixel_y), current_heading=current_heading)
                return self.action, self.point_goal, temp_path, vis_img
        
        # Recovery状态：执行回退动作序列
        elif self.agent.FSM.state == 'RC':
            if self.agent.FSM.recovery_step_count < len(self.agent.FSM.recovery_action_sequence):
                action = self.agent.FSM.recovery_action_sequence[self.agent.FSM.recovery_step_count]
                self.agent.FSM.recovery_step_count += 1
                print(f"[Recovery] 执行回退动作 {self.agent.FSM.recovery_step_count}/{len(self.agent.FSM.recovery_action_sequence)}: action={action}")
                self.action = action
                return self.action, None, None, vis_img
            else:
                # 回退序列执行完毕，但状态转换会在FSM中处理
                self.action = 2
                return self.action, None, None, vis_img
        
        
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
from utils.vlm_utils import room_type
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
    def add_object(self, object_node_lst):
        for object_node in object_node_lst:
            if heuristic((object_node.center[0] * 20, object_node.center[1] * 20), self.center) < self.size / 2:
                self.object_edge.append(object_node)
    def update_arrived_room(self, agent, observations):
        pixel_x = round(agent.center_pixel + agent.gps[0] * 20)
        pixel_y = round(agent.center_pixel - agent.gps[1] * 20)
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
            centering_weight=10.0,
            safety_margin=2,
            closing_kernel_size=self.closing_kernel_size,
            hole_fill_area=self.hole_fill_area
        )
        self.skeleton_processor = SkeletonProcessor(
            dilate_kernel=6,
            min_component_size=50,
            pruning_iterations=10,
            min_junction_distance=15,
            closing_kernel_size=self.closing_kernel_size,
            hole_fill_area=self.hole_fill_area
        )
        self.mcts_planner = MCTSPlanner(
            path_planner=self.path_planner,
            robot_radius=self.robot_radius,
            exploration_weight=2.0  # 降低exploration，让高reward的节点被更多访问
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
                if heuristic((self.agent.center_pixel + self.agent.gps[0] * 20, self.agent.center_pixel - self.agent.gps[1] * 20), room.center) < room.size / 2:
                    room.update_arrived_room(self.agent, observations)
                    print(f"update_arrived_room:{room.room_name}")
                    return
            # 如果在房间列表中没有找到一个相似的房间
            id = len(self.room_lst)
            temp_center = (self.agent.center_pixel + self.agent.gps[0] * 20, self.agent.center_pixel - self.agent.gps[1] * 20)
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
    
    def unexplored_exploring(self, occupancy_map: torch.Tensor, candidate: tuple, bfs_radius: int = 30) -> int:
        processed_map = occupancy_map.squeeze().cpu().numpy()
        H, W = processed_map.shape
        start_x, start_y = candidate
        
        if not (0 <= start_y < H and 0 <= start_x < W):
            return 0
        
        if processed_map[start_y, start_x] == 2:
            return 0

        q = deque([((start_x, start_y), 0)])
        visited = { (start_x, start_y) }
        unexplored_area_count = 0
        if processed_map[start_y, start_x] == 0:
            unexplored_area_count += 1
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
                q.append(((next_x, next_y), dist + 1))
                
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
   
   
    def _find_escape_point(self, point, search_radius=10, max_distance=7):
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
            if self.path_planner.find_path(occupancy_map=occupancy_map, start=start, end=point, start_adjustment_radius=int(self.robot_radius)+1):
                return point # 原始 point 有效且可达
        for r in range(1, max_radius + 1):
            candidate_points_at_radius_r = []
            for i in range(-r, r + 1, sampling_step):
                candidate_points_at_radius_r.append((point[0] + i, point[1] - r)) # 上
                candidate_points_at_radius_r.append((point[0] + i, point[1] + r)) # 下
                candidate_points_at_radius_r.append((point[0] - r, point[1] + i)) # 左
                candidate_points_at_radius_r.append((point[0] + r, point[1] + i)) # 右
            
            candidate_points_at_radius_r = list(set(candidate_points_at_radius_r))
            
            for np_x, np_y in candidate_points_at_radius_r:
                new_point = (np_x, np_y)
                if 0 <= new_point[0] < w and 0 <= new_point[1] < h and safe_navigation_space[new_point[1], new_point[0]]:
                    if self.path_planner.find_path(occupancy_map=occupancy_map, start=start, end=new_point, start_adjustment_radius=int(self.robot_radius)+1):
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
            if self.path_planner.find_path(occupancy_map=occupancy_map, start=start, end=point, start_adjustment_radius=int(self.robot_radius) + 1):
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
                if self.path_planner.find_path(occupancy_map=occupancy_map, start=start, end=new_point, start_adjustment_radius=int(self.robot_radius) + 1):
                    return new_point 
        
        for new_point in other_candidates:
            if 0 <= new_point[0] < w and 0 <= new_point[1] < h and safe_navigation_space[new_point[1], new_point[0]]:
                if self.path_planner.find_path(occupancy_map=occupancy_map, start=start, end=new_point, start_adjustment_radius=int(self.robot_radius) + 1):
                    return new_point

        return None
    
    
    def object_clustering(self, object_center_array, landmark_array, max_radius= 50.0,  max_objects_per_landmark= 5):
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
    
    # def choose_frontier_llm(self, candidates):        
    #     object_center_lst = []
    #     landmark_lst = []
    #     landmark_idx_lst = []
    #     for object_node in self.agent.scenegraph.object_node_lst:
    #         object_center_lst.append([object_node.center[0] * 20, object_node.center[1] * 20])
    #     object_center_array = np.array(object_center_lst)
    #     for i, landmark_candidate in enumerate(candidates):
    #         skip = False
    #         for landmark in landmark_lst:
    #             if heuristic(landmark, landmark_candidate[1][0]) < 50:
    #                 skip = True
    #                 break
    #         if not skip:
    #             landmark_lst.append(landmark_candidate[1][0])
    #             landmark_idx_lst.append(i)
    #         if len(landmark_lst) > 7:
    #             break
    #     landmark_array = np.array(landmark_lst)
    #     landmark_object_matching = self.object_clustering(object_center_array, landmark_array)
    #     for landmark_idx in landmark_object_matching.keys():
    #         if landmark_object_matching[landmark_idx] == []:
    #             continue
    #         else:
    #             landmark_object_matching[landmark_idx] = [self.agent.scenegraph.object_node_lst[obj_idx].class_name for obj_idx in landmark_object_matching[landmark_idx]]
       
    #     candidates_scores_dict = llm_auxiliary_frontier(self.agent.client, self.agent.obj_goal, landmark_object_matching)
    #     # sorted_keys = sorted(candidates_scores_dict, key=candidates_scores_dict.get, reverse=True)
    #     sorted_landmark = [(landmark_idx_lst[int(idx)], landmark_lst[int(idx)], candidates_scores_dict[int(idx)]) for idx in candidates_scores_dict]
        
    #     return sorted_landmark
    
    def choose_frontier(self, occupancy_map, current_pos, candidates, max_adjustment_radius=10, w_angle=1.0, w_dist=1.0, w_explore=1.0, w_avoid=1.0):
        if not candidates:
            return []
        self.ucb_counter.update_total_num()
        calculated_data = []
        turning_around = True
        max_theta = 0.0
        min_d = float('inf')
        max_d = 0.0
        min_min_distance = float('inf')  # 所有候选点中最小的“最小距离”
        max_min_distance = 0.0           # 所有候选点中最大的“最小距离”
        min_unexplored_num = float('inf')
        max_unexplored_num = 0.0
        unreachable_destination_lst = []
        
        candidates_with_diffs = []
        for candidate in candidates:
            diff = abs(self.calculate_heading(now=current_pos, goal=candidate[0]))
            candidates_with_diffs.append((diff, candidate))
        candidates_with_diffs.sort()

        top_k = math.floor(0.2 * len(candidates_with_diffs))
        if current_pos == self.temp_start:
            turning_around = False
        
        for theta_diff, candidate in candidates_with_diffs[0: top_k]:
            if theta_diff < 90 and heuristic(current_pos, candidate[0]) > 30:
                turning_around = False
                print("Don't turn around! ", candidate)
                break
            
        if w_angle == 0.0:
            turning_around = False
        
        if turning_around:
            print("Do turn around! ")
            self._update_temp_start(current_pos) 
        
        # 收集所有候选点的基础信息
        for candidate in candidates:
            goal_coord = candidate[0]

            if self.bad_frontier:
                min_distance = min(heuristic(bad, goal_coord) for bad in self.bad_frontier)
            else:
                min_distance = float('inf')
                
            theta = abs(self.calculate_heading(now=current_pos, goal=goal_coord))
            d = heuristic(current_pos, goal_coord)
            # frontier_num, total_num = self.unexplored_exploring(occupancy_map=occupancy_map, candidate=goal_coord, bfs_radius=30)
            unexplored_num = self.unexplored_exploring(occupancy_map=occupancy_map, candidate=goal_coord, bfs_radius=30)
            calculated_data.append((theta, d, unexplored_num, min_distance, candidate))
            
            # 更新极值
            max_theta = max(max_theta, theta)
            min_d = min(min_d, d)
            max_d = max(max_d, d)
            max_unexplored_num = max(max_unexplored_num, unexplored_num)
            min_unexplored_num = min(min_unexplored_num, unexplored_num)
            max_min_distance = max(max_min_distance, min_distance)
            min_min_distance = min(min_min_distance, min_distance)
        
        # 计算中等距离基准和最大偏离
        mid_d = 0.3 * min_d + 0.7 * max_d
        # if turning_around:
        #     mid_d = max_d
        # else:
        #     mid_d = 0.3 * min_d + 0.7 * max_d
        diffs = [abs(d - mid_d) for _, d, _, _, _ in calculated_data]
        max_diff = max(diffs) if diffs else 0.0  # 最大偏离mid_d的距离

        # 计算每个候选点的评分项
        metric_results = []
        print_show = []
        for theta, d, unexplored_num, min_dist, candidate in calculated_data:
            # 角度项（1 - 归一化角度差）
            if turning_around:
                angle_score = 1.0
            else:
                # angle_score = 1 - (theta / max_theta) if max_theta != 0 else 1.0
                angle_threshold_deg = 60.0
                steepness = 0.2 
                angle_score = 1.0 / (1.0 + math.exp(steepness * (theta - angle_threshold_deg)))
            
            # 距离项（中等距离优先：1 - 偏离比例）
            if max_diff != 0:
                diff_norm = abs(d - mid_d) / max_diff
            else:
                diff_norm = 0.0
            distance_score = 1 - diff_norm
            
            # 探索项
            if (max_unexplored_num - min_unexplored_num) != 0:
                explore_score = (unexplored_num - min_unexplored_num) / (max_unexplored_num - min_unexplored_num)
            else:
                explore_score = 0.0
            
            # 避免项（归一化最小距离）
            if (max_min_distance - min_min_distance) != 0:
                avoid_score = (min_dist - min_min_distance) / (max_min_distance - min_min_distance)
            else:
                avoid_score = 0.0
            
            # 综合得分（加权和）
            base_score = (w_angle * angle_score + w_dist * distance_score + w_explore * explore_score + w_avoid * avoid_score) if (distance_score > 0 and avoid_score > 0 and explore_score > 0) else 0.0
            ucb_score = self.ucb_counter.calculate_score(candidate[0])
            total_score = base_score + ucb_score
            
            metric_results.append(((total_score, angle_score, distance_score, explore_score, avoid_score), candidate))
            print_show.append(((total_score, base_score, ucb_score, angle_score, distance_score, explore_score, avoid_score, unexplored_num), candidate))

        # 按得分降序排序
        metric_results.sort(key=lambda x: x[0][0], reverse=True)
        print_show.sort(key=lambda x: x[0][0], reverse=True)
        
        # 打印调试信息
        print("---------Check Score Begin---------")
        for i, item in enumerate(print_show):
            total_score, base_score, ucb_score, angles, dists, explore, avoids, unexplored_num = item[0]
            print(f"No.{i}: Total_score={total_score:.3f}, Base_score={base_score:.3f}, UCB_score={ucb_score:.3f}, Angle={angles:.3f}, Dist={dists:.3f}, Explore={explore:.3f}, Avoid={avoids:.3f}, Coord={item[1][0]}, unexplored_num={unexplored_num}")
        print("----------Check Score End----------")
        
        # if turning_around:
        #     frontier_llm = self.choose_frontier_llm(candidates=metric_results)
        #     total_value_llm = sum(llm_value for _,_,llm_value in frontier_llm)
        #     print(f"test turning around: {frontier_llm}")
        #     metric_results_llm = []
        #     for frontier in frontier_llm:
        #         candidate_idx = frontier[0]
        #         llm_value = frontier[2] / total_value_llm if total_value_llm > 0 else 0.0
        #         explore_value = metric_results[candidate_idx][0][3]
        #         avoid_value = metric_results[candidate_idx][0][4]
        #         # distance_value = metric_results[candidate_idx][0][2]
        #         value = llm_value + explore_value + avoid_value  
        #         print(f"test llm value: llm_value={llm_value}, explore_value*w_explore={explore_value * w_explore}, avoid_value*w_avoid={avoid_value * w_avoid}")
        #         metric_results_llm.append((value, frontier[1]))
        #     metric_results_llm.sort(key=lambda x: x[0], reverse=True)
        #     print("zqh check metric llm", metric_results_llm)
        #     sorted_frontier = [item[1] for item in metric_results_llm]
            
        #     for frontier in sorted_frontier:
        #         for unreachable_destination in unreachable_destination_lst:
        #             if heuristic(frontier, unreachable_destination) < 20:
        #                 unreachable_destination_lst.append(frontier)
        #                 break
        #         if frontier not in unreachable_destination_lst and heuristic(frontier, current_pos) > 20:    
        #             valid_coords = self._find_nearest_valid_point(occupancy_map, current_pos, frontier, max_radius=10)
        #             if valid_coords:
        #                 print(f"[TA] adjust the candidate {frontier} to {valid_coords}")
        #                 return valid_coords
        #             else:
        #                 unreachable_destination_lst.append(frontier)     

        # 如果turning_around==False or turning around llm check failed
        sorted_frontier = [item[1][0] for item in metric_results]
        if sorted_frontier == []:
            return None
            
        for frontier in sorted_frontier:
            # 做一个初筛
            for unreachable_destination in unreachable_destination_lst:
                if heuristic(frontier, unreachable_destination) < 20:
                    unreachable_destination_lst.append(frontier)
                    break
            
            if frontier not in unreachable_destination_lst and heuristic(frontier, current_pos) > 20:
                valid_coords = self._find_nearest_valid_point(occupancy_map, current_pos, frontier, max_radius=max_adjustment_radius)
                if valid_coords:
                    print(f"[DTA] adjust the candidate {frontier} to {valid_coords}")
                    self.ucb_counter.add_explored_node(valid_coords)
                    print("check explored node lst: ", self.ucb_counter.explored_node_lst)
                    return valid_coords
                else:
                    unreachable_destination_lst.append(frontier)    
        return None


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
                obj_coord = (obj_node.center[0] * 20, obj_node.center[1] * 20)
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

        raw_scores = llm_auxiliary_frontier(self.agent.client, self.agent.obj_goal, landmark_scenes)

        llm_final_scores = {}
        for cluster_id, score in raw_scores.items():
            cluster_id = int(cluster_id)
            if cluster_id in selected_landmarks_info:
                original_idx = selected_landmarks_info[cluster_id]["original_idx"]
                normalized_score = max(0.0, min(1.0, score / 10.0))
                llm_final_scores[original_idx] = normalized_score
                
        return llm_final_scores

    
    
    def _check_need_turning(self, current_pos, valid_candidates, calculated_data):
        """检查是否需要调头"""
        if current_pos == self.temp_start:
            return False
        
        candidates_with_angles = [(theta, valid_candidates[idx][0]) for theta, _, _, _, idx in calculated_data]
        candidates_with_angles.sort()
        
        top_k = max(1, int(0.2 * len(candidates_with_angles)))
        for theta, coord in candidates_with_angles[:top_k]:
            if theta < 90 and heuristic(current_pos, coord) > 30:
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
        print(f"check self.point_goal:{self.point_goal}, check pos:{current_pos[0], current_pos[1]}, check temp_start={self.temp_start}")
        temp_path = self.path_planner.find_path(occupancy_map=self.agent.occupancy_map, start=current_pos, end=self.point_goal, start_adjustment_radius=500)
        temp_destination = self.point_goal
        if temp_path is None:
            if goal_adjustment_en:
                print(f"adjust goal and edit self.point_goal to {self.point_goal}")
                point_goal = self._find_nearest_valid_point(occupancy_map=self.agent.occupancy_map, start=current_pos, point=self.point_goal, max_radius=10)
                if point_goal is None:
                    self.bad_frontier.add(self.point_goal)  
                    self.point_goal = current_pos
                    self.agent.FSM.set_point_goal(self.point_goal)
                    return None
                else:
                    self.point_goal = point_goal
                    self.agent.FSM.set_point_goal(self.point_goal)
                    temp_path = self.path_planner.find_path(occupancy_map=self.agent.occupancy_map, start=current_pos, end=self.point_goal, start_adjustment_radius=500)
                    temp_destination = self.point_goal
            else:
                self.bad_frontier.add(self.point_goal)  
                self.point_goal = current_pos
                self.agent.FSM.set_point_goal(self.point_goal)
                return None

        for i, nav_point in enumerate(temp_path):
            if heuristic(current_pos, nav_point) > 3.6:
                temp_destination = nav_point  # 设置一个temp_path上的临时导航点（如果能正确抵达该导航点且不改变终点坐标，那么agent将沿着该temp_path走到终点）
                break
            elif i == len(temp_path) - 1:
                temp_destination = nav_point   # 如果temp path非常短以至于temp_path[-1]的距离都不够达到3.6，那么这个时候就使用temp_path[-1]
                
        delta_angle = self.calculate_angle(now=current_pos, goal=temp_destination, current_heading=current_heading)
                
        # 决定动作
        if delta_angle > 16:
            self.action = 3  # 右转
        elif delta_angle < -16:
            self.action = 2  # 左转
        else:
            self.action = 1  # 直行
        return temp_path


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
        pixel_x = round(self.agent.center_pixel + self.agent.gps[0] * 20)
        pixel_y = round(self.agent.center_pixel - self.agent.gps[1] * 20)
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
            essential_obj_center = (round(20 * self.agent.FSM.approaching_essential_obj.center[0]), round(20 * self.agent.FSM.approaching_essential_obj.center[1]))
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
            if self.collision_escaping_cnt < 3:
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
            navigation_distance_threshold = getattr(self.agent, 'navigation_distance_threshold', 15)
            
            # 检查是否需要重新选择frontier（防卡死机制触发）
            if self.agent.FSM.need_reselect_frontier:
                replan = True
                self.agent.FSM.need_reselect_frontier = False
            
            if heuristic((pixel_x, pixel_y), self.point_goal) >= navigation_distance_threshold or (pixel_x, pixel_y) == (400, 400):
                temp_path = self.action_gen(current_pos=(pixel_x, pixel_y), current_heading=current_heading, goal_adjustment_en=True)
                if temp_path is None:
                    print("Navigation Failed!!")
                    replan = True            
                        
            if heuristic((pixel_x, pixel_y), self.point_goal) < navigation_distance_threshold or replan:                
                # 加一个转圈的逻辑
                if self.turning_cnt < 5 and (pixel_x, pixel_y) != (400, 400) and not replan:
                    self.turning_cnt += 1
                    self.action = 6
                    return self.action, self.point_goal, None, vis_img        
                else:
                    self.turning_cnt = 0
                    self.bad_frontier.add(self.point_goal)
                    if replan:
                        best_endpoint = self.choose_frontier(occupancy_map=self.agent.occupancy_map, current_pos=(pixel_x, pixel_y), candidates=intersection_points, max_adjustment_radius=10, w_angle=1.0, w_dist=0.1, w_explore=0.1, w_avoid=1.0)
                    else:
                        best_endpoint = self.choose_frontier(occupancy_map=self.agent.occupancy_map, current_pos=(pixel_x, pixel_y), candidates=endpoints, max_adjustment_radius=10, w_angle=2.0, w_dist=1.0, w_explore=1.5, w_avoid=1.0)
                    
                    if best_endpoint is None:
                        print("all endpoint is unreachable")
                        best_endpoint = self.choose_frontier(occupancy_map=self.agent.occupancy_map, current_pos=(pixel_x, pixel_y), candidates=intersection_points, max_adjustment_radius=30, w_angle=2.0, w_dist=1.0, w_explore=1.5, w_avoid=1.0)
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
            obj_destination = _find_closest_point_in_2d_projection(self.agent.FSM.confirmed_essential_obj.pcd, (pixel_x, pixel_y))
            if heuristic(obj_destination, self.obj_destination) < 20:   # 保持稳定，保持self.point_goal不变
                temp_path = self.action_gen(current_pos=(pixel_x, pixel_y), current_heading=current_heading, goal_adjustment_en=True)
                return self.action, self.point_goal, temp_path, vis_img
            else:
                self.obj_destination = obj_destination
                nearest_free_point = self._find_object_destination(occupancy_map=self.agent.occupancy_map, start=(pixel_x, pixel_y), point=self.obj_destination, max_radius=20)  
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
        #     obj_destination = _find_closest_point_in_2d_projection(self.agent.FSM.confirmed_essential_obj.pcd, (pixel_x, pixel_y))
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
            self.obj_destination = _find_closest_point_in_2d_projection(self.agent.FSM.confirmed_essential_obj.pcd, (pixel_x, pixel_y))
            nearest_free_point = self._find_object_destination(occupancy_map=self.agent.occupancy_map, start=(pixel_x, pixel_y), point=self.obj_destination, max_radius=20)  
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
        
        
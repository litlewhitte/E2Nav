from utils.vlm_utils import room_type, object_confirmed, object_detection, object_final_confirmed, nearby_confirmed, vlm_belief_distribution, active_perception_confirmation
from utils.realtime_mapping import project_pcd_to_mask
from utils.utils import _find_closest_point_in_2d_projection

import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, label, center_of_mass, generate_binary_structure
from collections import deque, Counter
import math
import torch
import cv2
import os

def heuristic(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

class FSM():
    def __init__(self, agent):
        self.agent = agent
        self.state = 'BS'
        self.vlm_model = None
        self.vlm_processor = None
        self.history_room = []
        self.history_action = []
        self.optional_room_list = self.agent.optional_room_list
        self.approaching_room_list = []
        self.approaching_essential_obj = None
        self.approaching_room_name = None
        self.mandatory_detection = False
        self.confirmed_object_list = []
        self.confirmed_wrong_list = []
        self.confirmed_bbox_dict = {}
        self.BS_point_goal_reg = None
        self.temp_cnt = 0
        self.collision_pos_set = []
        self.permanently_unreachable_targets = []
        self.temporarily_blocked_targets = []  # 专门跟踪被障碍物遮挡的目标
        self.blocked_target_centers = {}  # 记录被遮挡目标的center历史，用于检测变化
        self.original_score = 0.0
        self.confirmed_essential_obj = None
        self.state_before_collision = None
        self.debug_rgb = {}
        self.confirmed_observation_score = {}
        self.is_add_room = [False, None, False]
        self.robot_radius = self.agent.robot_radius
        self.history_agent_state = []
        self.collision_cnt = 0
        self.point_goal = (self.agent.center_pixel, self.agent.center_pixel)
        self.active_perception_target = None
        self.active_perception_object_id = None
        self.active_perception_directions = []  # 新增：需要转动的方向列表
        self.active_perception_images = []
        self.active_perception_action_done = False
        
        # BS阶段防卡死机制
        self.bs_point_goal_collision_count = {}  # 记录BS阶段每个point_goal的碰撞次数 {point_goal: count}
        self.current_bs_point_goal = None  # 当前BS阶段的point_goal
        self.max_bs_collision_threshold = self.agent.max_bs_collision_threshold  # 最大允许碰撞次数
        self.need_reselect_frontier = False  # 标志：需要重新选择frontier
        
        # 长程回退恢复机制
        self.recovery_action_sequence = []  # 回退动作序列
        self.recovery_step_count = 0  # 回退步数计数
        self.max_recovery_steps = self.agent.max_recovery_steps  # 最大回退步数
        
    def reset(self, obj_goal, room_rank):
        self.state = 'BS'
        self.history_room = []
        self.history_action = []
        self.approaching_room_list = []
        self.is_add_room = [False, None, False]
        self.approaching_essential_obj = None
        self.approaching_room_name = None
        self.mandatory_detection = False
        self.confirmed_object_list = []
        self.confirmed_wrong_list = []
        self.permanently_unreachable_targets = []
        self.temporarily_blocked_targets = []
        self.blocked_target_centers = {}
        self.confirmed_bbox_dict = {}
        self.BS_point_goal_reg = None
        self.temp_cnt = 0
        self.collision_pos_set = []
        self.original_score = 0.0
        self.confirmed_essential_obj = None
        self.state_before_collision = None
        self.debug_rgb = {}
        self.confirmed_observation_score = {}
        self.obj_goal = obj_goal
        self.room_rank = room_rank
        self.history_agent_state = []
        self.collision_cnt = 0
        self.point_goal = (self.agent.center_pixel, self.agent.center_pixel)
        # 主动感知相关参数
        self.active_perception_target = None
        self.active_perception_object_id = None
        self.active_perception_directions = []  # 新增：需要转动的方向列表
        self.active_perception_images = []
        self.active_perception_action_done = False
        
        # BS阶段防卡死机制
        self.bs_point_goal_collision_count = {}
        self.current_bs_point_goal = None
        self.need_reselect_frontier = False
        
        # 长程回退恢复机制
        self.recovery_action_sequence = []
        self.recovery_step_count = 0
    
    def calculate_angle(self, now, goal, current_heading):
        delta_x = goal[0] - now[0]
        delta_y_actual = now[1] - goal[1]
        angle_rad = math.atan2(delta_y_actual, delta_x)
        target_angle = math.degrees(angle_rad) % 360
        
        delta_angle = (target_angle + current_heading) % 360
        if delta_angle > 180:
            delta_angle -= 360 
        return delta_angle   
    
    def set_point_goal(self, point_goal):
        self.point_goal = point_goal
        
        # 如果是在BS状态下设置point_goal，记录下来
        if self.state == 'BS':
            self.current_bs_point_goal = point_goal
            # 如果是新的point_goal，初始化碰撞计数
            if point_goal not in self.bs_point_goal_collision_count:
                self.bs_point_goal_collision_count[point_goal] = 0
    
    def get_bbox_boundary_intersection(self, bbox, image_width=640, image_height=480, margin=None):
        """
        检测bbox与图像边界的相交情况，返回需要转动的方向
        
        Args:
            bbox: [x_min, y_min, x_max, y_max] 边界框坐标
            image_width: 图像宽度
            image_height: 图像高度
            margin: 边界容忍度（像素），在margin范围内认为靠近边界
            
        Returns:
            dict: {
                'intersects': bool,  # 是否与边界相交
                'directions': list,  # 需要转动的方向 ['left', 'right', 'up', 'down']
                'bbox_center': tuple  # bbox中心坐标
            }
        """
        if margin is None:
            margin = self.agent.bbox_boundary_margin
        x_min, x_max, y_min, y_max = bbox
        
        # 检查与各边界的相交情况
        intersects_left = x_min <= margin
        intersects_right = x_max >= (image_width - margin)
        intersects_top = y_min <= margin
        intersects_bottom = y_max >= (image_height - margin)
        
        # 计算bbox中心
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        
        # 确定需要转动的方向
        directions = []
        if intersects_left:
            directions.append('left')
        if intersects_right:
            directions.append('right')
        if intersects_top:
            directions.append('up')
        if intersects_bottom:
            directions.append('down')
        
        intersects = len(directions) > 0
        
        if intersects:
            print(f"[Active Perception] BBox intersects boundaries: {directions}")
            print(f"[Active Perception] BBox coords: ({x_min}, {y_min}, {x_max}, {y_max}), Image size: ({image_width}, {image_height})")
            print(f"[Active Perception] BBox center: ({center_x}, {center_y})")
        
        return {
            'intersects': intersects,
            'directions': directions,
            'bbox_center': (int(center_x), int(center_y))
        }
    
    def get_bbox_center(self, bbox):
        """
        计算bbox的中心点（像素坐标）
        
        Args:
            bbox: [x_min, y_min, x_max, y_max]
            
        Returns:
            tuple: (center_x, center_y) bbox中心点坐标
        """
        x_min, y_min, x_max, y_max = bbox
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        return (int(center_x), int(center_y))
    
    def skip_unreachable_target(self, object_node_lst):
        print("skip current target")
        del self.confirmed_object_list[self.confirmed_object_list.index(self.confirmed_essential_obj.id)]
        if self.confirmed_essential_obj.id not in self.permanently_unreachable_targets:
            self.permanently_unreachable_targets.append(self.confirmed_essential_obj.id)
     
        if self.confirmed_object_list == []:
            self.confirmed_essential_obj = None
            self.state = 'BS'    
        else:    # 从现在的confirmed_object_list列表中选择一个分数最大的作为现在的探索目标
            max_score_obj = {"id": None, "score": 0.0}
            for obj_idx in self.confirmed_object_list:
                if object_node_lst[obj_idx].score > max_score_obj['score']:
                    max_score_obj["id"] = obj_idx
                    max_score_obj["score"] = object_node_lst[obj_idx].score
            self.confirmed_essential_obj = object_node_lst[max_score_obj["id"]]
            self.state = 'AO'       

    def skip_blocked_target(self, object_node_lst):
        """专门处理障碍物遮挡的情况，将目标添加到obstacle_blocked_targets列表"""
        print("current target is blocked by obstacles!!")
        target_id = self.confirmed_essential_obj.id
        del self.confirmed_object_list[self.confirmed_object_list.index(target_id)]
        
        # 添加到临时遮挡列表而不是permanently_unreachable_targets
        if target_id not in self.temporarily_blocked_targets:
            self.temporarily_blocked_targets.append(target_id)
            # 记录当前的center位置
            target_obj = None
            for obj in object_node_lst:
                if obj.id == target_id:
                    target_obj = obj
                    break
            
            if target_obj is not None:
                center_to_store = target_obj.center.copy() if target_obj.center is not None else None
                self.blocked_target_centers[target_id] = center_to_store
     
        if self.confirmed_object_list == []:
            self.confirmed_essential_obj = None
            self.state = 'BS'    
        else:    # 从现在的confirmed_object_list列表中选择一个分数最大的作为现在的探索目标
            max_score_obj = {"id": None, "score": 0.0}
            for obj_idx in self.confirmed_object_list:
                if object_node_lst[obj_idx].score > max_score_obj['score']:
                    max_score_obj["id"] = obj_idx
                    max_score_obj["score"] = object_node_lst[obj_idx].score
            self.confirmed_essential_obj = object_node_lst[max_score_obj["id"]]
            self.state = 'AO'

    def check_blocked_targets_recovery(self, object_node_lst):
        """检查被遮挡的目标是否重新变得可达（通过center变化检测）"""
        targets_to_remove = []
        
        for target_id in self.temporarily_blocked_targets:
            # 检查目标是否在object_node_lst中
            target_obj = None
            for obj in object_node_lst:
                if obj.id == target_id:
                    target_obj = obj
                    break
            
            if target_obj is not None:
                current_center = target_obj.center
                stored_center = self.blocked_target_centers.get(target_id)
                
                # 检查center是否发生了变化
                if current_center is not None and stored_center is not None:
                    # 计算center之间的距离变化
                    center_distance = np.linalg.norm(np.array(current_center) - np.array(stored_center))
                    center_change_threshold = self.agent.center_change_threshold
                    if center_distance > center_change_threshold:  # 如果center变化超过阈值，说明有新的检测融合
                        print(f"Target {target_id} center changed, removing from temporarily_blocked_targets")
                        targets_to_remove.append(target_id)
                        # 从temporarily_blocked_targets中移除，但不添加到confirmed_object_list
                        # 让系统重新评估这个目标
                elif current_center is not None and stored_center is None:
                    # 如果之前没有center信息，现在有了，也认为是变化
                    print(f"Target {target_id} center appeared, removing from temporarily_blocked_targets")
                    targets_to_remove.append(target_id)
        
        # 移除检测到变化的目标
        for target_id in targets_to_remove:
            if target_id in self.temporarily_blocked_targets:
                self.temporarily_blocked_targets.remove(target_id)
            if target_id in self.blocked_target_centers:
                del self.blocked_target_centers[target_id]
        
        # 检查被遮挡的目标是否重新变得可达（检查地图navigable）
        blocked_target_check_interval = self.agent.blocked_target_check_interval
        if self.agent.total_steps % blocked_target_check_interval == 0 and len(self.temporarily_blocked_targets) > 0:
            for target_id in list(self.temporarily_blocked_targets):  # 使用list复制，避免迭代时修改
                target_obj = object_node_lst[target_id]
                
                if target_obj is not None and target_obj.pcd is not None:
                    # 获取目标物体上离当前位置最近的点
                    obj_destination = _find_closest_point_in_2d_projection(
                        target_obj.pcd, 
                        (self.pixel_x, self.pixel_y),
                        r_resolution=self.agent.r_resolution
                    )
                    
                    # 使用_find_object_destination找到物体周围最近的可达点
                    if obj_destination is not None:
                        blocked_target_radius = self.agent.blocked_target_reachability_radius
                        nearest_free_point = self.agent.landmarkmap._find_object_destination(
                            occupancy_map=self.agent.occupancy_map,
                            start=(self.pixel_x, self.pixel_y),
                            point=obj_destination,
                            max_radius=blocked_target_radius
                        )
                        
                        # 如果找到了可达点，说明目标重新变得可达
                        if nearest_free_point is not None:
                            print(f"Target {target_id} [{target_obj.class_name}] is now reachable, removing from temporarily_blocked_targets")
                            targets_to_remove.append(target_id)
            
            # 移除重新变得可达的目标
            for target_id in targets_to_remove:
                if target_id in self.temporarily_blocked_targets:
                    self.temporarily_blocked_targets.remove(target_id)
                if target_id in self.blocked_target_centers:
                    del self.blocked_target_centers[target_id]
        
        
        
    
    def add_collision_offsets(self):
        dx, dy = 0, 0
        if 337.5 <= self.current_heading or self.current_heading < 22.5:
            dx = 1
            dy = 0
        elif 22.5 <= self.current_heading < 67.5:
            dx = 1
            dy = 1
        elif 67.5 <= self.current_heading < 112.5:
            dx = 0
            dy = 1
        elif 112.5 <= self.current_heading < 157.5:
            dx = -1
            dy = 1
        elif 157.5 <= self.current_heading < 202.5:
            dx = -1
            dy = 0
        elif 202.5 <= self.current_heading < 247.5:
            dx = -1
            dy = -1
        elif 247.5 <= self.current_heading < 292.5:
            dx = 0
            dy = -1
        elif 292.5 <= self.current_heading < 337.5:
            dx = 1
            dy = -1
            
        next_x = self.pixel_x + dx*2
        next_y = self.pixel_y + dy*2
        self.agent.landmarkmap.path_planner._add_collision_point((next_x, next_y))
    
    def collision_detected(self):
        if len(self.history_agent_state) < 3:
            return False
        else:
            if self.history_action[-1] == 1 and self.history_agent_state[-1] == self.history_agent_state[-2]:    # 即使在上一步用了前进，但是没有前进
                print("collision happened!")
                self.state_before_collision = self.state
                if (self.agent.pixel_x, self.agent.pixel_y) not in self.collision_pos_set:
                    self.collision_pos_set.append((self.agent.pixel_x, self.agent.pixel_y))
                self.add_collision_offsets()
                
                # BS阶段防卡死：记录当前point_goal的碰撞次数
                if self.state == 'BS' and self.current_bs_point_goal is not None:
                    self.bs_point_goal_collision_count[self.current_bs_point_goal] += 1
                    collision_count = self.bs_point_goal_collision_count[self.current_bs_point_goal]
                    
                    # 如果碰撞次数超过阈值，标记需要重新选择frontier
                    if collision_count > self.max_bs_collision_threshold:
                        print(f"[BS collision avoidance]{self.current_bs_point_goal} collision times > {self.max_bs_collision_threshold}, choose new frontier")
                        self.agent.landmarkmap.bad_frontier.add(self.current_bs_point_goal)
                        self.need_reselect_frontier = True
                
                # self.agent.landmarkmap.path_planner._add_collision_point((self.agent.pixel_x, self.agent.pixel_y))
                return True
            else:
                return False
    
    def collision_plus_detected(self):
        collision_plus_radius = self.agent.collision_plus_detection_radius
        for collision_pos in self.collision_pos_set:
            if heuristic((self.pixel_x, self.pixel_y), collision_pos) < collision_plus_radius and (self.pixel_x, self.pixel_y) != collision_pos:
                return True
        return False
    
    def failure_detected(self):
        failure_history_length = self.agent.failure_detection_history_length
        if len(self.history_agent_state) < failure_history_length:
            return False
        else:
            detected_lst = self.history_agent_state[-failure_history_length:-1]
            if len(set(item[:2] for item in detected_lst)) <= 1:
                return True
            else:
                return False
    
    def generate_reverse_action_sequence(self):
        """
        Generate reverse action sequence based on action history
        
        Algorithm:
        1. Reverse action sequence and convert to reverse actions
        2. Insert 180-degree turn before first MOVE_FORWARD
        3. Optimize rotation sequences: merge consecutive turn actions (2,3,6) 
           into minimal rotation commands using only action 2 or 3
        
        Action mapping:
        0: STOP - skip
        1: MOVE_FORWARD - keep as 1
        2: TURN_LEFT(30deg) -> 3 (TURN_RIGHT)
        3: TURN_RIGHT(30deg) -> 2 (TURN_LEFT)
        4: LOOK_UP -> 5 (LOOK_DOWN)
        5: LOOK_DOWN -> 4 (LOOK_UP)
        6: TURN_RIGHT_2(60deg) -> [2, 2] (TURN_LEFT twice)
        """
        current_state = self.history_agent_state[-1]
        
        # Step 1: Reverse the action sequence and convert to reverse actions
        reverse_sequence = []
        for action in reversed(self.history_action):
            if action == 0:  # STOP - skip
                continue
            elif action == 1:  # MOVE_FORWARD - keep as 1
                reverse_sequence.append(1)
            elif action == 2:  # TURN_LEFT -> TURN_RIGHT
                reverse_sequence.append(3)
            elif action == 3:  # TURN_RIGHT -> TURN_LEFT
                reverse_sequence.append(2)
            elif action == 4:  # LOOK_UP -> LOOK_DOWN
                reverse_sequence.append(5)
            elif action == 5:  # LOOK_DOWN -> LOOK_UP
                reverse_sequence.append(4)
            elif action == 6:  # TURN_RIGHT_2 -> TURN_LEFT twice
                reverse_sequence.extend([2, 2])
        
        # Step 2: Find first MOVE_FORWARD and insert 180-degree turn before it
        first_forward_idx = -1
        for i, action in enumerate(reverse_sequence):
            if action == 1:
                first_forward_idx = i
                break
        
        if first_forward_idx >= 0:
            # Insert 180-degree turn (6 x TURN_RIGHT) before first MOVE_FORWARD
            turn_180 = [3, 3, 3, 3, 3, 3]
            reverse_sequence = reverse_sequence[:first_forward_idx] + turn_180 + reverse_sequence[first_forward_idx:]
            print(f"[Rollback] Inserted 180-degree turn before first MOVE_FORWARD at position {first_forward_idx}")
        
        # Step 3: Optimize rotation sequences
        optimized_sequence = []
        cumulative_angle = 0  # Track cumulative rotation in degrees
        
        def flush_rotation():
            """Convert cumulative angle to optimal action sequence"""
            nonlocal cumulative_angle
            if cumulative_angle == 0:
                return []
            
            # Normalize angle to [-180, 180]
            angle = cumulative_angle % 360
            if angle > 180:
                angle -= 360
            
            # Generate optimal sequence using only action 2 (left) or 3 (right)
            actions = []
            if angle > 0:
                # Positive = right turns (action 3)
                num_turns = int(round(angle / 30))
                actions = [3] * num_turns
            elif angle < 0:
                # Negative = left turns (action 2)
                num_turns = int(round(abs(angle) / 30))
                actions = [2] * num_turns
            
            cumulative_angle = 0
            return actions
        
        for action in reverse_sequence:
            if action == 2:  # TURN_LEFT (counter-clockwise, negative)
                cumulative_angle -= 30
            elif action == 3:  # TURN_RIGHT (clockwise, positive)
                cumulative_angle += 30
            elif action in [1, 4, 5]:  # MOVE_FORWARD, LOOK_UP, LOOK_DOWN
                # Flush accumulated rotation before non-rotation action
                optimized_sequence.extend(flush_rotation())
                optimized_sequence.append(action)
            # Note: action 6 should not appear here as it was already converted to [2,2]
        
        # Flush any remaining rotation at the end
        optimized_sequence.extend(flush_rotation())
        
        print(f"[Rollback] Generated reverse sequence: {len(reverse_sequence)} actions")
        print(f"[Rollback] Optimized sequence: {len(optimized_sequence)} actions (saved {len(reverse_sequence) - len(optimized_sequence)} actions)")
        
        return optimized_sequence
    
    def check_escaped_stuck(self):
        """
        Check if robot has escaped from stuck state using path planning
        
        Returns:
            bool: True if escaped (path found), False if still stuck
        """
        # Use A* path planner to check if path exists from current position to target
        path = self.agent.landmarkmap.path_planner.find_path_astar(
            occupancy_map=self.agent.occupancy_map,
            start=(self.pixel_x, self.pixel_y),
            end=(self.agent.center_pixel, self.agent.center_pixel),
            start_adjustment_radius=int(self.robot_radius) + 1
        )
        
        if path is not None:
            print(f"[Recovery] Escape Done: path found (length={len(path)})")
            return True
        else:
            print(f"[Recovery] Still stuck: no path to target")
            return False
    
    def collision_free_detected(self):
        if self.history_action[-1] == 1:
            if self.history_agent_state[-1] != self.history_agent_state[-2]:   # 在CE state直行且避免碰撞，完全脱困
                self.state = self.state_before_collision


    def object_detected(self, object_node_lst):   # 这个方法用object_node_lst更新当前self.confirmed_object_list, 既做增加又做删减
        if self.state in ['BS', 'AR']:
            if self.agent.scenegraph.confirmed_observations["newly_detection"]:
                for i, idx_bbox in enumerate(self.agent.scenegraph.confirmed_observations["confirmed_observations"]):
                    idx, bbox_observations = next(iter(idx_bbox.items()))
                    
                    # vlm_probs = vlm_belief_distribution(
                    #     url=self.agent.client,
                    #     view_rgb=bbox_observations,
                    #     vocabulary=self.agent.vocabulary
                    # )
                    # object_node_lst[idx].class_update(
                    #     class_name=self.agent.obj_goal,
                    #     probs=vlm_probs,
                    #     bbox=self.agent.scenegraph.confirmed_observations["bbox"][i][idx] 
                    # )
                    
                    confirmed_result = object_confirmed(
                        url=self.agent.client,
                        model=self.agent.model,
                        view_rgb=bbox_observations,
                        object_type=self.agent.obj_goal,
                        similar_object=self.agent.class_description[self.agent.obj_goal]["comparison_classes"],
                        vocabulary=self.agent.vocabulary,

                        authorization=self.agent.authorization,
                    )
                    print("check confirmed process: ", confirmed_result, "\ncheck confirmed object id: ", idx)
                    
                    if not confirmed_result.get('object_present'):
                        if torch.argmax(torch.tensor(self.agent.scenegraph.confirmed_observations["probs"][i][idx])).item() == self.agent.obj_goal_id:   # 对于最大值
                            object_node_lst[idx]._rollback_class()   
                    else:
                        if torch.argmax(torch.tensor(self.agent.scenegraph.confirmed_observations["probs"][i][idx])).item() == self.agent.obj_goal_id:   # 对于最大值
                            object_node_lst[idx].class_update(
                                class_name=self.agent.obj_goal,
                                probs=self.agent.scenegraph.confirmed_observations["probs"][i][idx],
                                bbox=self.agent.scenegraph.confirmed_observations["bbox"][i][idx] 
                            )
                        else:  # 非最大值出现了一个vlm确认的物品
                            while torch.argmax(torch.tensor(object_node_lst[idx].class_candidates)).item() != self.agent.obj_goal_id:
                                object_node_lst[idx].topk_class_update(
                                    class_id=self.agent.obj_goal_id
                                )
                        
                        
                        current_bbox = self.agent.scenegraph.confirmed_observations["bbox"][i][idx]
                        # 从config读取图像尺寸
                        image_height = self.agent.config.SIMULATOR.RGB_SENSOR.HEIGHT
                        image_width = self.agent.config.SIMULATOR.RGB_SENSOR.WIDTH
                        
                        # 检测bbox与边界的相交情况
                        boundary_info = self.get_bbox_boundary_intersection(current_bbox, image_width, image_height, margin=10)
                        if boundary_info['intersects'] and idx not in self.confirmed_wrong_list and idx not in self.permanently_unreachable_targets and idx not in self.temporarily_blocked_targets:
                            # bbox与边界相交，标记该物体需要主动感知
                            print(f"[Active Perception] Object {idx} bbox intersects boundaries: {boundary_info['directions']}, and object id is {idx}")
                            
                            # 设置主动感知参数
                            self.active_perception_object_id = idx
                            self.active_perception_target = boundary_info['bbox_center']
                            self.active_perception_directions = boundary_info['directions']  # 新增：保存需要转动的方向
                            self.active_perception_images = []
                            self.active_perception_action_done = False
                            
                            # 设置一个标志，表示需要进入主动感知状态
                            # 这个标志会在state_transition中被检查
                            object_node_lst[idx].need_active_perception = True
                            print(f"[Active Perception] Set object {idx} to need active perception")
                            break
                        
                        else:
                            # VLM确认物体存在，进行class_update
                            print("No need to do active perception")
                            
        
        # elif self.state in ['AP']:
        #     if self.agent.scenegraph.confirmed_observations["newly_detection"]:
        #         for i, idx_bbox in enumerate(self.agent.scenegraph.confirmed_observations["confirmed_observations"]):
        #             idx, bbox_observations = next(iter(idx_bbox.items()))
        #             if idx != self.active_perception_object_id:
        #                 confirmed_result = object_confirmed(
        #                     url=self.agent.client,
        #                     model=self.agent.model,
                        # authorization=self.agent.authorization,
        #                     view_rgb=bbox_observations,
        #                     object_type=self.agent.obj_goal,
        #                     similar_object=self.agent.class_description[self.agent.obj_goal]["comparison_classes"],
        #                     vocabulary=self.agent.vocabulary,
        #                 )
        #                 print("check confirmed process: ", confirmed_result, "\ncheck confirmed object id: ", idx)
                        
        #                 if not confirmed_result.get('object_present'):
        #                     if torch.argmax(torch.tensor(self.agent.scenegraph.confirmed_observations["probs"][i][idx])).item() == self.agent.obj_goal_id:   # 对于最大值
        #                         object_node_lst[idx]._rollback_class()   
        #                 else:
        #                     if torch.argmax(torch.tensor(self.agent.scenegraph.confirmed_observations["probs"][i][idx])).item() == self.agent.obj_goal_id:   # 对于最大值
        #                         object_node_lst[idx].class_update(
        #                             class_name=self.agent.obj_goal,
        #                             probs=self.agent.scenegraph.confirmed_observations["probs"][i][idx],
        #                             bbox=self.agent.scenegraph.confirmed_observations["bbox"][i][idx] 
        #                         )
        #                     else:  # 非最大值出现了一个vlm确认的物品
        #                         while torch.argmax(torch.tensor(object_node_lst[idx].class_candidates)).item() != self.agent.obj_goal_id:
        #                             object_node_lst[idx].topk_class_update(
        #                                 class_id=self.agent.obj_goal_id
        #                             )
                    

        elif self.state in ['AO']:
            if self.agent.scenegraph.confirmed_observations["newly_detection"]:
                for i, idx_bbox in enumerate(self.agent.scenegraph.confirmed_observations["confirmed_observations"]):
                    idx, bbox_observations = next(iter(idx_bbox.items()))
                    
                    if idx in self.confirmed_object_list:   # 正在跟踪的物品或者已经在列表中的物品且在当前时刻检测为obj goal（这样的话 kl divergence不是必然会很小？？？）
                        obj_node = object_node_lst[idx]
                        kl_threshold = self.agent.kl_divergence_recheck_threshold
                        if obj_node.kl_divergence > kl_threshold:
                            print(f"kl divergence is {obj_node.kl_divergence}, so we check again")
                            confirmed_result = object_confirmed(
                                url=self.agent.client,
                                model=self.agent.model,
                        view_rgb=bbox_observations,
                                object_type=self.agent.obj_goal,
                                similar_object=self.agent.class_description[self.agent.obj_goal]["comparison_classes"],
                                vocabulary=self.agent.vocabulary,
                                use_bbox=True
                            ,

                        authorization=self.agent.authorization)
                            
                            if not confirmed_result.get('object_present'):
                                if torch.argmax(torch.tensor(self.agent.scenegraph.confirmed_observations["probs"][i][idx])).item() == self.agent.obj_goal_id:   # 对于最大值
                                    object_node_lst[idx]._rollback_class()   
                            else:
                                if torch.argmax(torch.tensor(self.agent.scenegraph.confirmed_observations["probs"][i][idx])).item() == self.agent.obj_goal_id:   # 对于最大值
                                    object_node_lst[idx].class_update(
                                        class_name=self.agent.obj_goal,
                                        probs=self.agent.scenegraph.confirmed_observations["probs"][i][idx],
                                        bbox=self.agent.scenegraph.confirmed_observations["bbox"][i][idx] 
                                    )
                                else:  # 非最大值出现了一个vlm确认的物品
                                    while torch.argmax(torch.tensor(object_node_lst[idx].class_candidates)).item() != self.agent.obj_goal_id:
                                        object_node_lst[idx].topk_class_update(
                                            class_id=self.agent.obj_goal_id
                                        )
                            
                            
                    else:   # 新检测到的物品，用vlm做一下确认
                        # vlm_probs = vlm_belief_distribution(
                        #     url=self.agent.client,
                        #     view_rgb=bbox_observations,
                        #     vocabulary=self.agent.vocabulary
                        # )
                        # object_node_lst[idx].class_update(
                        #     class_name=self.agent.obj_goal,
                        #     probs=vlm_probs,
                        #     bbox=self.agent.scenegraph.confirmed_observations["bbox"][i][idx] 
                        # )
                        
                        confirmed_result = object_confirmed(
                            url=self.agent.client,
                            model=self.agent.model,
                        view_rgb=bbox_observations,
                            object_type=self.agent.obj_goal,
                            similar_object=self.agent.class_description[self.agent.obj_goal]["comparison_classes"],
                            vocabulary=self.agent.vocabulary,
                            use_bbox=True
                        ,

                        authorization=self.agent.authorization)
                        print("check newly confirmed process AO: ", confirmed_result)   
        
                        if not confirmed_result.get('object_present'):
                            if torch.argmax(torch.tensor(self.agent.scenegraph.confirmed_observations["probs"][i][idx])).item() == self.agent.obj_goal_id:   # 对于最大值
                                object_node_lst[idx]._rollback_class()   
                        else:
                            if torch.argmax(torch.tensor(self.agent.scenegraph.confirmed_observations["probs"][i][idx])).item() == self.agent.obj_goal_id:   # 对于最大值
                                object_node_lst[idx].class_update(
                                    class_name=self.agent.obj_goal,
                                    probs=self.agent.scenegraph.confirmed_observations["probs"][i][idx],
                                    bbox=self.agent.scenegraph.confirmed_observations["bbox"][i][idx] 
                                )
                            else:  # 非最大值出现了一个vlm确认的物品
                                while torch.argmax(torch.tensor(object_node_lst[idx].class_candidates)).item() != self.agent.obj_goal_id:
                                    object_node_lst[idx].topk_class_update(
                                        class_id=self.agent.obj_goal_id
                                    )
                
                            
            print(f"check kl divergence of essential obj: {self.confirmed_essential_obj.kl_divergence}")
            ## 试一试将点云project到当前位置的mask
        
        # 加入新物品 
        for object_node in object_node_lst:        
            if self.obj_goal in object_node.class_name and object_node.id not in self.confirmed_object_list and object_node.id not in self.confirmed_wrong_list and object_node.id not in self.permanently_unreachable_targets and object_node.id not in self.temporarily_blocked_targets:  
                self.confirmed_object_list.append(object_node.id)
        # 删除类别发生变更的旧物品
        for i in reversed(range(len(self.confirmed_object_list))):    
            idx = self.confirmed_object_list[i]
            if self.obj_goal not in object_node_lst[idx].class_name:
                del self.confirmed_object_list[i]
        
        if "CE" not in self.state:
            self.save_confirmed_bbox()
    
    # def check_score(self, object_node_lst):
    #     if self.obj_goal in object_node_lst[self.confirmed_essential_obj.id].class_name:
    #         return object_node_lst[self.confirmed_essential_obj.id].score
    #     else:
    #         return -1.0 * object_node_lst[self.confirmed_essential_obj.id].score
        
    # def object_checking(self, object_node_lst):
    #     if self.obj_goal in object_node_lst[self.confirmed_essential_obj.id].class_name:
    #         scores = list(self.confirmed_essential_obj.class_candidates.values())
    #         if len(scores) > 1:
    #             sorted_scores = sorted(scores, reverse=True)
    #             delta_scores = (sorted_scores[0] - sorted_scores[1])/sorted_scores[0]
    #         else: 
    #             delta_scores = scores[0] / 2
    #         return delta_scores
    #     else:
    #         return -1.0
    
    def object_checking(self, object_node_lst):
        if self.obj_goal in object_node_lst[self.confirmed_essential_obj.id].class_name:
            probs = self.confirmed_essential_obj.class_candidates
            entropy = -torch.sum(probs * torch.log(probs + 1e-6))
            print(f"check entropy when checking object class: {entropy}")
            entropy_ratio = self.agent.entropy_threshold_ratio
            if entropy < math.log(len(self.agent.vocabulary)) * entropy_ratio:     
                return True
            else:
                return False
        else:
            return False
    
    
    def collect_active_perception_image(self, save_bbox=False):
        # """
        # 收集AP状态的带bbox图像
        # 在AP状态的每个步骤中调用，用于收集多视角的带bbox图像
        # """
        # obj_id = self.active_perception_object_id
        # bbox_found = False
        
        # if obj_id is not None:
        #     # 在字典列表中查找该物体的bbox
            # for bbox_dict in self.agent.scenegraph.confirmed_observations["bbox"]:
            #     if obj_id in bbox_dict:
            #         bbox = bbox_dict[obj_id]  # 获取bbox坐标
            #         # 使用scenegraph的方法生成带bbox的图像
            #         bbox_image = self.agent.scenegraph.get_bbox_observations(bbox)
            #         self.active_perception_images.append(bbox_image)
            #         bbox_found = True
            #         break
        
        # if not bbox_found:
        #     # 如果没有找到bbox信息，使用原始图像
        #     self.active_perception_images.append(self.agent.observations['rgb'].copy())
        
        obj_id = self.active_perception_object_id
        if save_bbox:
            for bbox_dict in self.agent.scenegraph.confirmed_observations["bbox"]:
                if obj_id in bbox_dict:
                    bbox = bbox_dict[obj_id] 
                    bbox_image = self.agent.scenegraph.get_bbox_observations(bbox)
                    self.active_perception_images.append(bbox_image)
                    break 
        
        elif not save_bbox:
            # 如果没有找到bbox信息，使用原始图像
            self.active_perception_images.append(self.agent.observations['rgb'].copy())

    def select_uniform_views(self, obj_id, max_views=5):
        """
        从confirmed_bbox_dict中均匀选择指定数量的视角图片
        
        Args:
            obj_id: 物体ID
            max_views: 最大选择数量，默认5张
            
        Returns:
            list: 选择的视角图片列表
        """
        all_views = self.confirmed_bbox_dict[obj_id]
        
        if len(all_views) <= max_views:
            return all_views
        
        # 选择第1张、中间1张、最后1张，再均匀选择剩余张数
        indices = [0, len(all_views)//2, len(all_views)-1]  # 第1张、中间、最后1张
        remaining_count = max_views - 3  # 还需要选择的张数
        
        if remaining_count > 0:
            # 在剩余位置中均匀选择
            remaining_indices = [i for i in range(1, len(all_views)-1) if i != len(all_views)//2]
            
            if len(remaining_indices) >= remaining_count:
                # 均匀分布选择
                step = len(remaining_indices) // remaining_count
                for i in range(remaining_count):
                    indices.append(remaining_indices[i * step])
            else:
                # 如果剩余位置不足，全部选择
                indices.extend(remaining_indices)
        
        # 按索引顺序排序并获取图片
        indices = sorted(indices)
        return [all_views[i] for i in indices]

    def save_confirmed_bbox(self):
        if self.agent.scenegraph.confirmed_observations["newly_detection"]:
            for i, idx_bbox in enumerate(self.agent.scenegraph.confirmed_observations["confirmed_observations"]):
                idx, bbox_observations = next(iter(idx_bbox.items()))
                for obj_id in self.confirmed_object_list:
                    if idx == obj_id:
                        if idx not in self.confirmed_bbox_dict.keys():
                            self.confirmed_bbox_dict[idx] = [bbox_observations]
                            self.debug_rgb[idx] = [self.agent.total_steps]
                        else:
                            self.confirmed_bbox_dict[idx].append(bbox_observations)
                            self.debug_rgb[idx].append(self.agent.total_steps)
        
        
        elif self.agent.scenegraph.final_confirmed_observatinos != {}:
            for obj_id in self.agent.scenegraph.final_confirmed_observatinos.keys():
                if obj_id in self.confirmed_object_list:
                    if obj_id not in self.confirmed_bbox_dict.keys():
                        self.confirmed_bbox_dict[obj_id] = [self.agent.scenegraph.final_confirmed_observatinos[obj_id]]
                        self.debug_rgb[obj_id] = [self.agent.total_steps]
                    else:
                        self.confirmed_bbox_dict[obj_id].append(self.agent.scenegraph.final_confirmed_observatinos[obj_id])
                        self.debug_rgb[obj_id].append(self.agent.total_steps)               
                                 

    def choose_bbox(self):
        confirmed_bbox_list = self.confirmed_bbox_dict[self.confirmed_essential_obj.id]
        confirmed_observation_score_list = self.confirmed_observation_score[self.confirmed_essential_obj.id]
        def min_max_normalize(numbers):
            if not numbers:
                return []
                
            min_val = min(numbers)
            max_val = max(numbers)

            if max_val == min_val:
                return [0.0] * len(numbers)

            normalized_list = [(x - min_val) / (max_val - min_val) for x in numbers]
            return normalized_list
        
        max_views = self.agent.max_confirmed_bbox_views - 1  # >5 means >=6
        if len(confirmed_bbox_list) > max_views:
            angle_list = [item[0] for item in confirmed_observation_score_list]
            normalized_angle_list = min_max_normalize(angle_list)
            dist_list = [item[1] for item in confirmed_observation_score_list]
            normalized_dist_list = min_max_normalize(dist_list)
            score_list = [angle * dist for angle, dist in zip(normalized_angle_list, normalized_dist_list)]
            paired_list = list(zip(score_list, confirmed_bbox_list))
            sorted_pairs = sorted(paired_list, key=lambda item: item[0], reverse=True)
            sorted_pairs = sorted_pairs[:5]
            return [bbox for _, bbox in sorted_pairs]
        
        else:
            return confirmed_bbox_list
        
    def draw_bbox(self, rgb, bbox):
        output_image = rgb.copy()
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(
            output_image,
            (x_min, y_min),      # Top-left corner
            (x_max, y_max),      # Bottom-right corner
            (0, 255, 0),               # BGR color
            2            # Line thickness
        )
        return output_image
    
    def set_confirmed_object(self, object_node_lst):
        if self.confirmed_object_list != []:
            max_score_obj = {"id": None, "score": 0.0}
            for i in range(len(self.confirmed_object_list)):   
                idx = self.confirmed_object_list[i]
                if object_node_lst[idx].score > max_score_obj["score"]:  
                    max_score_obj["id"] = idx
                    max_score_obj["score"] = object_node_lst[idx].score
            self.confirmed_essential_obj = object_node_lst[max_score_obj["id"]]
        else:
            self.confirmed_essential_obj = None

    
    def object_mandatory_detected(self):
        mandatory_interval = self.agent.mandatory_detection_interval
        if self.agent.total_steps % mandatory_interval == 0:
            goal_found, is_likely = object_detection(self.agent.client, self.agent.model, self.agent.observations['rgb'], self.agent.obj_goal, self.agent.room_rank, self.agent.curr_detection_nodes, authorization=self.agent.authorization)
            if goal_found:
                if self.confirmed_object_list == []:
                    self.mandatory_detection = True
            elif is_likely:   # 没有找到物品，但是很可能前面的场景中有物品
                if nearby_confirmed(self.agent.client, self.agent.model, self.agent.observations['rgb'], self.agent.obj_goal, authorization=self.agent.authorization):
                    room, essential_obj_node = room_type(self.agent.client, self.agent.model, self.agent.observations['rgb'], self.agent.optional_room_list, self.agent.curr_detection_nodes, authorization=self.agent.authorization)
                    # cv2.imwrite("temp/good.png", cv2.cvtColor(self.agent.observations['rgb'], cv2.COLOR_RGB2BGR))
                    # print("[DEBUG] essential_obj_node is None", essential_obj_node is None)
                    if essential_obj_node is None:
                        return False
                    else:
                        self.approaching_essential_obj = essential_obj_node
                        self.approaching_room_name = room
                        return True
                else:
                    return False
                # room, essential_obj_node = room_type(self.agent.client, self.agent.observations['rgb'], self.agent.optional_room_list, self.agent.curr_detection_nodes)
                # if room is not None and room != 'None' and room in self.agent.room_rank:
                #     if self.agent.room_rank[room] > 4:
                #         self.approaching_essential_obj = essential_obj_node
                #         self.approaching_room_name = room
                #         return True
                # return nearby_confirmed(self.agent.client, self.agent.model, self.agent.observations['rgb'], self.agent.obj_goal)
        return False
        
    # def room_detected(self):
    #     if self.agent.total_steps % 5 == 0:
    #         room, essential_obj_node, goal_object_found = room_type(self.agent.client, self.agent.model, self.agent.observations['rgb'], self.agent.optional_room_list, self.agent.curr_detection_nodes, self.agent.obj_goal)
    #         print(f"room={room}, object={essential_obj_node.class_name if essential_obj_node is not None else None}, is_found={goal_object_found}")
    #         if goal_object_found and self.confirmed_object_list == []:
    #             # vlm找到了目标，但是GroundedSAM未找到目标或高置信度认为眼前目标错误，这个时候还是选择相信vlm
    #             self.mandatory_detection = True
    #             return False
    #         elif room in self.agent.optional_room_list and essential_obj_node is not None:
    #             print(f"object_center={essential_obj_node.center}")
    #             if self.agent.room_rank[room] > 4:
    #                 self.approaching_essential_obj = essential_obj_node
    #                 self.approaching_room_name = room
    #                 return True
    #     return False
    
    def is_planar(self, pcd, distance_threshold=None, inlier_ratio_threshold=None):
        if not pcd.has_points() or len(pcd.points) < 3:
            return False
        
        if distance_threshold is None:
            distance_threshold = self.agent.planar_detection_threshold
        if inlier_ratio_threshold is None:
            inlier_ratio_threshold = self.agent.planar_inlier_ratio_threshold
        
        try:
            _, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                                    ransac_n=3,
                                                    num_iterations=1000)
        except Exception as e:
            print(f"Open3D plane segmentation failed: {e}")
            return False

        inlier_ratio = len(inliers) / len(pcd.points)
        print(f"planar ratio is {inlier_ratio}")
        if inlier_ratio >= inlier_ratio_threshold:
            print("The object is planar and we reject to confirmed!!")
            return True
        else:
            return False
    
    def is_explored_room(self, pos):
        for room_node in self.agent.landmarkmap.room_lst:
            if heuristic(pos, room_node.center) < room_node.size / 2:
                return True
        return False
    
    def room_center_gen(self):
        obj_destination = (round(self.approaching_essential_obj.center[0] * self.agent.r_resolution), round(self.approaching_essential_obj.center[1] * self.agent.r_resolution))
        room_max_radius = self.agent.room_destination_max_radius
        best_destination = self.agent.landmarkmap._find_nearest_valid_point(occupancy_map=self.agent.occupancy_map, start=(self.agent.pixel_x, self.agent.pixel_y), point=obj_destination, max_radius=room_max_radius)
        
        if best_destination is None:
            print("The room is unreachable!!")
        else:
            room_check_distance = self.agent.room_approach_check_distance
            if self.state == 'BS':
                if heuristic((self.agent.pixel_x, self.agent.pixel_y), best_destination) < room_check_distance or heuristic(self.point_goal, best_destination) < room_check_distance or self.is_explored_room((self.agent.pixel_x, self.agent.pixel_y)):
                    print(f"No need to get close to this room! agent pos = {(self.agent.pixel_x, self.agent.pixel_y)}; best destination = {best_destination}, point_goal = {self.point_goal}, ")
                    return False
                else:
                    print("I should get close to this room!")
                    self.BS_point_goal_reg = self.point_goal
                    self.point_goal = best_destination
                    return True
            elif self.state == 'AR':
                room_fine_tune_distance = self.agent.room_fine_tuning_distance
                if heuristic(self.point_goal, best_destination) > room_fine_tune_distance:
                    print("Fine-tuning the point goal")
                    self.point_goal = best_destination
                    return True
            

    def state_transition(self):
        self.pixel_x = round(self.agent.center_pixel + self.agent.gps[0] * self.agent.r_resolution)
        self.pixel_y = round(self.agent.center_pixel - self.agent.gps[1] * self.agent.r_resolution)
        self.current_heading = math.degrees(self.agent.observations['compass']) % 360
        self.history_agent_state.append((self.pixel_x, self.pixel_y, self.current_heading))   
        self.history_action.append(self.agent.action)      # 这里记录的action是agent在上一个time step的动作
        
        # 从配置读取距离阈值
        navigation_distance_threshold = self.agent.navigation_distance_threshold
        
        if self.failure_detected() and self.state != 'RC':
            failure_history = self.agent.failure_detection_history_length
            print(f"{failure_history} steps without moving, stuck detected!")
            print(f"collision offsets: {self.agent.landmarkmap.path_planner.collision_offsets}")
            # 进入Recovery恢复状态，尝试长程回退
            self.state = 'RC'
            self.agent.landmarkmap.bad_frontier.add((self.pixel_x, self.pixel_y))
            print("check history action sequence", self.history_action)
            self.recovery_action_sequence = self.generate_reverse_action_sequence()
            print("check recovery action sequence", self.recovery_action_sequence)
            self.recovery_step_count = 0
            print(f"[Recovery] 进入恢复状态，生成{len(self.recovery_action_sequence)}步回退序列")
        
        elif self.state == 'RC':  # Recovery恢复状态
            # 检查是否脱困
            if self.check_escaped_stuck():
                print(f"[Recovery] Done! Back to BS")
                self.state = 'BS'
                self.recovery_action_sequence = []
                self.recovery_step_count = 0
                # 清空历史，避免再次检测到failure
                self.history_agent_state = self.history_agent_state[-3:]
                self.history_action = self.history_action[-3:]
            
            # 检查是否超过最大回退步数
            elif self.recovery_step_count >= self.max_recovery_steps:
                print(f"[Recovery] 回退{self.max_recovery_steps}步仍未脱困，放弃")
                self.state = 'ST'
            
            # 检查是否已经回退完所有历史动作
            elif self.recovery_step_count >= len(self.recovery_action_sequence):
                print(f"[Recovery] 已回退到出生点仍卡死，放弃")
                self.state = 'ST'
            
            # 否则继续执行回退动作序列
            # （实际动作在landmark.local_policy中返回）
        
        else:
            if self.state == 'BS':
                # 步骤1: 先检测collision（可能设置need_reselect_frontier=True）
                if self.collision_detected():
                    self.state = 'CE'
                    # 碰撞时不执行后续逻辑，直接进入CE状态
                
                else:
                    # 步骤2: 没有碰撞，进行状态转换（检测物品）
                    self.object_detected(self.agent.scenegraph.object_node_lst)
                    
                    # 在set_confirmed_object之前检查是否有物体需要主动感知
                    if self.confirmed_object_list != []:
                        # 检查confirmed_object_list中的物体是否有需要主动感知的
                        need_active_perception = False
                        ap_target_id = None
                        
                        for obj_id in self.confirmed_object_list:
                            obj_node = self.agent.scenegraph.object_node_lst[obj_id]
                            if hasattr(obj_node, 'need_active_perception') and obj_node.need_active_perception:
                                need_active_perception = True
                                ap_target_id = obj_id
                                break  # 找到第一个需要主动感知的物体
                        
                        if need_active_perception:
                            # 发现需要主动感知的物体，进入AP状态
                            print(f"[Active Perception] Transitioning to AP state for object {ap_target_id}")
                            self.collect_active_perception_image(save_bbox=True)
                            self.BS_point_goal_reg = self.point_goal
                            self.state = 'AP'
                            # 发现了关键物品，取消replan标志（如果之前碰撞设置了）
                            self.need_reselect_frontier = False
                        else:
                            # 发现普通关键物品，进入AO状态
                            self.set_confirmed_object(self.agent.scenegraph.object_node_lst)
                            self.BS_point_goal_reg = self.point_goal
                            self.state = 'AO'
                            # 发现了关键物品，取消replan标志（如果之前碰撞设置了）
                            self.need_reselect_frontier = False
                    
                    else:
                        # 没有发现关键物品，继续BS状态
                        # 检查是否需要强制检测和场景可能性
                        if self.object_mandatory_detected():
                            if self.room_center_gen():
                                self.state = 'AR'
                                # 发现了房间，取消replan标志
                                self.need_reselect_frontier = False
                            else:
                                self.state = 'BS'
                        else:
                            self.state = 'BS'
                        
                        # # 步骤3: 如果还在BS状态且没有发现关键物品，检查探索潜力
                        # if self.state == 'BS':
                        #     # 检查当前point_goal的探索潜力
                        #     if self.current_bs_point_goal is not None and self.current_bs_point_goal != (self.agent.center_pixel, self.agent.center_pixel):
                        #         unexplored_count = self.agent.landmarkmap.unexplored_exploring(
                        #             occupancy_map=self.agent.occupancy_map,
                        #             candidate=self.current_bs_point_goal,
                        #             bfs_radius=self.agent.bfs_radius,
                        #             return_points=False
                        #         )
                                
                        #         if unexplored_count == 0:
                        #             # 探索潜力为0，触发重新规划
                        #             print(f"[BS exploration check] Point goal {self.current_bs_point_goal} has 0 explorable points, triggering replan")
                        #             self.agent.landmarkmap.bad_frontier.add(self.current_bs_point_goal)
                        #             self.need_reselect_frontier = True
                        #             self.current_bs_point_goal = None
            
            elif self.state == 'AO':
                if self.collision_detected():
                    self.state = 'CE'
                # if self.collision_detected():
                #     if self.collision_plus_detected():
                #         self.state = 'CE-plus'
                #     else:
                #         self.state = 'CE'
                else:
                    self.object_detected(self.agent.scenegraph.object_node_lst)
                    if self.confirmed_object_list != []:
                        if self.confirmed_essential_obj.id not in self.confirmed_object_list:
                            self.set_confirmed_object(self.agent.scenegraph.object_node_lst)
                            # self.reset_confirmed_rgb_list()
                            print("change approaching goal!!")   
                        elif heuristic((self.pixel_x, self.pixel_y), self.point_goal) < navigation_distance_threshold:
                            # self.save_confirmed_bbox()
                            self.state = 'CO'    
                        # else:
                        #     self.save_confirmed_bbox()
                    else:
                        self.confirmed_essential_obj = None
                        # self.reset_confirmed_rgb_list()
                        self.point_goal = self.BS_point_goal_reg
                        self.agent.landmarkmap.point_goal = self.point_goal
                        self.state = 'BS'
            
            elif self.state == 'CO':
                if self.collision_detected():   # 这里的collision detection不是为了做一个远距离的停止，而是为了在非常靠近的时候做一个提前停止的操作
                    # obj_destination = (round(self.confirmed_essential_obj.center[0] * self.agent.r_resolution), round(self.confirmed_essential_obj.center[1] * self.agent.r_resolution))
                    obj_destination = _find_closest_point_in_2d_projection(self.confirmed_essential_obj.pcd, (self.agent.pixel_x, self.agent.pixel_y), r_resolution=self.agent.r_resolution)
                    
                    obj_close_distance = self.agent.object_close_confirmation_distance
                    if heuristic((self.pixel_x, self.pixel_y), obj_destination) < obj_close_distance:
                        similar_object_index = torch.topk(self.confirmed_essential_obj.class_candidates.flatten(), 3).indices
                        similar_object_name = [self.agent.vocabulary[i] for i in similar_object_index]
                        for obj_name in self.agent.class_description[self.agent.obj_goal]["comparison_classes"]:
                            if obj_name not in similar_object_name:
                                similar_object_name.append(obj_name)
                                
                        # 均匀采集5张图片
                        # view_rgbs = self.select_uniform_views(self.confirmed_essential_obj.id, max_views=5)
                        max_views = self.agent.max_confirmed_bbox_views
                        view_rgbs = self.confirmed_bbox_dict[self.confirmed_essential_obj.id] if len(self.confirmed_bbox_dict[self.confirmed_essential_obj.id]) < max_views else self.confirmed_bbox_dict[self.confirmed_essential_obj.id][0:max_views-1]
                        for i, rgb in enumerate(view_rgbs):
                            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(os.path.join(self.agent.pcd_dir, f"final_confirmed_rgb_{self.agent.total_steps}_{i}.png"), rgb)
                        

                        # 获取当前物体的belief信息
                        object_belief = self.confirmed_essential_obj.class_candidates
                        
                        # 获取目标物体的详细描述
                        object_description = None
                        if self.agent.obj_goal in self.agent.class_description:
                            object_description = self.agent.class_description[self.agent.obj_goal].get("description", None)
                        
                        # 使用多数投票法：调用5次VLM最终确认
                        num_votes = 5
                        vote_results = []
                        print(f"[VLM Final] Starting {num_votes} voting rounds...")
                        for vote_idx in range(num_votes):
                            vote_result = object_final_confirmed(
                                url=self.agent.client,
                                model=self.agent.model,
                                view_rgbs=view_rgbs,
                                object_type=self.agent.obj_goal,
                                similar_object=similar_object_name,
                                vocabulary=self.agent.vocabulary,
                                object_belief=object_belief,
                                object_description=object_description,

                                authorization=self.agent.authorization,
                            )
                            vote_results.append(vote_result)
                            print(f"[VLM Final] Vote {vote_idx + 1}/{num_votes}: {vote_result}")
                        
                        # 多数投票法决定最终结果
                        positive_votes = sum(vote_results)
                        final_confirmed_result = positive_votes > (num_votes / 2)
                        print(f"[VLM Final] Voting complete: {positive_votes}/{num_votes} positive votes")
                        print(f"[VLM Final] Final decision: {'CONFIRMED' if final_confirmed_result else 'REJECTED'}")
                        
                        print(f"check final confirmed({len(self.confirmed_bbox_dict[self.confirmed_essential_obj.id])} and idx is {self.debug_rgb[self.confirmed_essential_obj.id]})")
                        print(f"entropy of confirmed object is {self.confirmed_essential_obj._entropy_gen()}")
                        is_planar = self.is_planar(self.confirmed_essential_obj.pcd)
                        # is_planar = False

                        min_views = self.agent.min_confirmed_bbox_views
                        if len(self.confirmed_bbox_dict[self.confirmed_essential_obj.id]) > min_views - 1 and final_confirmed_result and not is_planar:
                            self.state = 'ST'
                        else:
                            self.skip_unreachable_target(self.agent.scenegraph.object_node_lst)
                            # self.reset_confirmed_rgb_list()
                    else:
                        self.skip_unreachable_target(self.agent.scenegraph.object_node_lst)
                        # self.reset_confirmed_rgb_list()
                        
                else:
                    self.object_detected(self.agent.scenegraph.object_node_lst)
                    if self.confirmed_essential_obj.id not in self.confirmed_object_list:
                        self.set_confirmed_object(self.agent.scenegraph.object_node_lst)
                        if self.confirmed_essential_obj is None:
                            self.state = 'BS'
                            # self.reset_confirmed_rgb_list()
                            print("go to BS")
                        else:
                            self.state = 'AO'
                            # self.reset_confirmed_rgb_list()
                            print("change approaching goal CO!!") 
                    
                    else: 
                        if heuristic((self.pixel_x, self.pixel_y), self.point_goal) >= navigation_distance_threshold:    
                            self.state = 'AO'
                        elif heuristic((self.pixel_x, self.pixel_y), self.point_goal) < self.agent.object_very_close_distance:
                            # obj_destination = (round(self.confirmed_essential_obj.center[0] * self.agent.r_resolution), round(self.confirmed_essential_obj.center[1] * self.agent.r_resolution))
                            obj_destination = _find_closest_point_in_2d_projection(self.confirmed_essential_obj.pcd, (self.agent.pixel_x, self.agent.pixel_y), r_resolution=self.agent.r_resolution)

                            obj_close_check_dist = self.agent.object_close_confirmation_distance
                            if heuristic((self.pixel_x, self.pixel_y), obj_destination) > obj_close_check_dist:   # 如果当前位置距离物品实际上还比较远
                                self.skip_unreachable_target(self.agent.scenegraph.object_node_lst)
                            else:
                                similar_object_index = torch.topk(self.confirmed_essential_obj.class_candidates.flatten(), 3).indices
                                similar_object_name = [self.agent.vocabulary[i] for i in similar_object_index]
                                for obj_name in self.agent.class_description[self.agent.obj_goal]["comparison_classes"]:
                                    if obj_name not in similar_object_name:
                                        similar_object_name.append(obj_name)
                                

                                # 均匀采集5张图片
                                # view_rgbs = self.select_uniform_views(self.confirmed_essential_obj.id, max_views=5)
                                max_views = self.agent.max_confirmed_bbox_views
                                view_rgbs = self.confirmed_bbox_dict[self.confirmed_essential_obj.id] if len(self.confirmed_bbox_dict[self.confirmed_essential_obj.id]) < max_views else self.confirmed_bbox_dict[self.confirmed_essential_obj.id][0:max_views-1]
                                for i, rgb in enumerate(view_rgbs):
                                    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                                    cv2.imwrite(os.path.join(self.agent.pcd_dir, f"final_confirmed_rgb_{self.agent.total_steps}_{i}.png"), rgb)

                                
                                object_belief = self.confirmed_essential_obj.class_candidates
                        
                                # 获取目标物体的详细描述
                                object_description = None
                                if self.agent.obj_goal in self.agent.class_description:
                                    object_description = self.agent.class_description[self.agent.obj_goal].get("description", None)
                        
                                # 使用多数投票法：调用5次VLM最终确认
                                num_votes = 5
                                vote_results = []
                                print(f"[VLM Final] Starting {num_votes} voting rounds...")
                                for vote_idx in range(num_votes):
                                    vote_result = object_final_confirmed(
                                        url=self.agent.client,
                                        model=self.agent.model,
                                        view_rgbs=view_rgbs,
                                        object_type=self.agent.obj_goal,
                                        similar_object=similar_object_name,
                                        vocabulary=self.agent.vocabulary,
                                        object_belief=object_belief,
                                        object_description=object_description,
                                        authorization=self.agent.authorization,
                                    )
                                    vote_results.append(vote_result)
                                    print(f"[VLM Final] Vote {vote_idx + 1}/{num_votes}: {vote_result}")
                                
                                # 多数投票法决定最终结果
                                positive_votes = sum(vote_results)
                                final_confirmed_result = positive_votes > (num_votes / 2)
                                print(f"[VLM Final] Voting complete: {positive_votes}/{num_votes} positive votes")
                                print(f"[VLM Final] Final decision: {'CONFIRMED' if final_confirmed_result else 'REJECTED'}")
                         
                                print(f"check final confirmed({len(self.confirmed_bbox_dict[self.confirmed_essential_obj.id])} and idx is {self.debug_rgb[self.confirmed_essential_obj.id]})")
                                print(f"entropy of confirmed object is {self.confirmed_essential_obj._entropy_gen()}")
                                is_planar = self.is_planar(self.confirmed_essential_obj.pcd)
                                # is_planar = False
                                
                                min_views = self.agent.min_confirmed_bbox_views
                                if len(self.confirmed_bbox_dict[self.confirmed_essential_obj.id]) > min_views - 1 and final_confirmed_result and not is_planar:
                                    self.state = 'ST'
                                else:
                                    # self.reset_confirmed_rgb_list()
                                    if self.confirmed_essential_obj.id in self.confirmed_object_list:
                                        del self.confirmed_object_list[self.confirmed_object_list.index(self.confirmed_essential_obj.id)]
                                        self.confirmed_wrong_list.append(self.confirmed_essential_obj.id)
                                    if self.confirmed_object_list != []:
                                        self.set_confirmed_object(self.agent.scenegraph.object_node_lst)
                                        self.state = 'AO'
                                    else:
                                        self.state = 'BS' 

                                                    
            elif self.state == 'AR':
                if self.collision_detected():
                    self.state = 'CE'
                # if self.collision_detected():
                #     if self.collision_plus_detected():
                #         self.state = 'CE-plus'
                #     else:
                #         self.state = 'CE'
                elif heuristic((self.pixel_x, self.pixel_y), self.point_goal) < navigation_distance_threshold:
                    self.is_add_room = [True, self.approaching_room_name, True]
                    self.object_detected(self.agent.scenegraph.object_node_lst)
                    self.set_confirmed_object(self.agent.scenegraph.object_node_lst)
                    if self.confirmed_object_list != []:
                        self.state = 'AO'
                    else:
                        self.point_goal = self.BS_point_goal_reg
                        self.agent.landmarkmap.point_goal = self.point_goal
                        self.state = 'BS'
                else:
                    self.object_detected(self.agent.scenegraph.object_node_lst)
                    self.set_confirmed_object(self.agent.scenegraph.object_node_lst)
                    if self.confirmed_object_list != []:
                        self.state = 'AO'
                    else:
                        room_update_interval = self.agent.blocked_target_check_interval
                        if self.agent.total_steps % room_update_interval == 0:
                            self.room_center_gen()
            
            elif self.state == 'CE':      
                if self.history_action[-1] == 1:
                    if self.history_agent_state[-1] != self.history_agent_state[-2]:   # 在CE state直行且避免碰撞，完全脱困
                        self.state = self.state_before_collision
                    elif self.history_agent_state[-1] == self.history_agent_state[-2]:   # 在CE state直行但是没有脱困
                        self.state = 'CE-plus'
                        self.agent.landmarkmap.bad_frontier.add(self.agent.landmarkmap.point_goal)
            
            elif self.state == 'CE-plus':   
                if self.history_action[-1] == 1 and self.history_agent_state[-1] != self.history_agent_state[-2]:    # 在CE-plus state直行且避免碰撞，完全脱困
                    self.state = self.state_before_collision
            
            elif self.state == 'AP':  # Active Perception 主动感知状态
                # self.object_detected(self.agent.scenegraph.object_node_lst)
                self.collect_active_perception_image()
                # 如果主动感知动作序列已完成
                if self.active_perception_action_done:
                    print(f"[Active Perception] Action sequence completed, collecting {len(self.active_perception_images)} images")
                    obj_id = self.active_perception_object_id
                    assert obj_id is not None, f"active_perception_object_id should not be None in AP state"
                    
                    self.confirmed_bbox_dict[obj_id] += self.active_perception_images                    
                    obj_node = self.agent.scenegraph.object_node_lst[obj_id]
                    similar_object_index = torch.topk(obj_node.class_candidates.flatten(), 3).indices
                    similar_object_name = [self.agent.vocabulary[i] for i in similar_object_index]
                    for obj_name in self.agent.class_description[self.agent.obj_goal]["comparison_classes"]:
                        if obj_name not in similar_object_name:
                            similar_object_name.append(obj_name)
                    
                    # 调试：保存AP图片
                    if self.agent.type == 'debug':
                        for i, img in enumerate(self.active_perception_images):
                            cv2.imwrite(os.path.join(self.agent.pcd_dir, f"ap_idx{i}_obj_id{self.active_perception_object_id}_step{self.agent.total_steps}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    
                    # 调用VLM确认函数
                    active_perception_confirmed = active_perception_confirmation(
                        url=self.agent.client,
                        model=self.agent.model,
                        view_rgbs=self.active_perception_images,
                        object_type=self.agent.obj_goal,
                        similar_object=similar_object_name,
                        vocabulary=self.agent.vocabulary,
                        authorization=self.agent.authorization
                    )
                    
                    if active_perception_confirmed:
                        # VLM确认成功，转换到AO状态
                        print(f"[Active Perception] VLM confirmed object {self.active_perception_object_id}")
                        while torch.argmax(torch.tensor(self.agent.scenegraph.object_node_lst[obj_id].class_candidates)).item() != self.agent.obj_goal_id:
                            self.agent.scenegraph.object_node_lst[obj_id].topk_class_update(
                                class_id=self.agent.obj_goal_id
                            )
                        # 加入新物品（检查是否已存在，避免重复）
                        if obj_id not in self.confirmed_object_list:
                            self.confirmed_object_list.append(obj_id)
                        self.set_confirmed_object(self.agent.scenegraph.object_node_lst)
                        self.state = 'AO'
                    else:
                        # VLM确认失败，物体不是目标，从confirmed_object_list中移除
                        print(f"[Active Perception] VLM rejected object {self.active_perception_object_id}")
                        if self.active_perception_object_id in self.confirmed_object_list:
                            del self.confirmed_object_list[self.confirmed_object_list.index(self.active_perception_object_id)]
                            self.confirmed_wrong_list.append(self.active_perception_object_id)
                        
                        # 重新检查confirmed_object_list
                        if self.confirmed_object_list != []:
                            self.set_confirmed_object(self.agent.scenegraph.object_node_lst)
                            self.state = 'AO'
                        else:
                            self.confirmed_essential_obj = None
                            self.point_goal = self.BS_point_goal_reg
                            self.agent.landmarkmap.point_goal = self.point_goal
                            self.state = 'BS'
                    
                    # 重置主动感知相关变量
                    self.agent.scenegraph.object_node_lst[self.active_perception_object_id].need_active_perception = False         
                    self.active_perception_target = None
                    self.active_perception_object_id = None
                    self.active_perception_directions = []
                    self.active_perception_images = []
                    self.active_perception_action_done = False           
                    
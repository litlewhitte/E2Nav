import os

import habitat
import json
import cv2
import torch
import traceback
import math
import open_clip 
import numpy as np
import time
import open3d as o3d
from datetime import datetime
from scripts.visualization import NavigationVisualization
from scripts.habitat_runner import evaluate
from scripts.scene_graph import SceneGraph
from scripts.landmark_planner import LandmarkPlanner
from scripts.navigation_state_machine import NavigationStateMachine

import hydra
from omegaconf import DictConfig, OmegaConf

from utils.occupancy_mapping import SemanticMapper, ElevationMapper, ElevationGradientCalculator
from utils.utils import merge_point_clouds, merge_pcd_batch, fill_small_holes
from utils.llm_utils import room_rank


def draw_points_on_image(image_path, points_list, output_path, point_color=(255, 0, 0), point_radius=4, point_thickness=-1):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    for point in points_list:
        center_x = int(round(point[0]))
        center_y = int(round(point[1]))

        height, width = img.shape[:2]
        cv2.circle(img, (center_x, center_y), point_radius, point_color, point_thickness)
    success = cv2.imwrite(output_path, img)
    return success


class ObjectNavigationAgent(NavigationVisualization):
    def __init__(self, task_config, cfg):
        self._POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS
        self.total_steps = 0
        self.episode_id = 0
        # self.obj_goal = self.simulator._env.current_episode.object_category
        self.obj_goal = None
        
        self.episode_l = cfg.experiment.episode_l
        self.episode_r = cfg.experiment.episode_r
        self.split_l = cfg.experiment.split_l
        self.split_r = cfg.experiment.split_r
        self.dataset = cfg.experiment.dataset
        self.experiment_name = cfg.experiment.experiment_name
        self.metrics = {'distance_to_goal': 0., 'spl': 0., 'softspl': 0.}
        self.if_upstairs = False
        self.type = cfg.experiment.type
        self.device = f'cuda:{cfg.experiment.cuda_id}'
        self.port = cfg.experiment.port
        self.config = task_config
        self.min_height = cfg.height.min_height
        self.max_height = cfg.height.max_height
        self.gps = [0.0, 0.0, 0.0]
        
        self.detection_segmentation = cfg.perception.detection_segmentation
        self.curr_detection_nodes = []
        
        if self.type == 'debug':
            self.save_dir = f"{cfg.paths.save_dir_base}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.split_l}_{self.split_r}"
        else:
            self.save_dir = f"{cfg.paths.exp_dir_base}/{self.dataset}/{self.experiment_name}/observations/{self.dataset}_{self.split_l}_to_{self.split_r}"
        os.makedirs(self.save_dir, exist_ok=True)
    
        self.map_size_cm = cfg.mapping.map_size_cm
        self.resolution = self.map_resolution = cfg.mapping.resolution
        self.r_resolution = 100 / self.resolution
        self.center_pixel = cfg.mapping.center_pixel
        self.camera_horizon = cfg.mapping.camera_horizon
        self.dilation_deg = cfg.mapping.dilation_deg
        self.collision_threshold = cfg.mapping.collision_threshold
        self.closing_kernel_size = cfg.mapping.closing_kernel_size
        self.hole_fill_area = cfg.mapping.hole_fill_area
        self.episode_view_points = []

        self.explored_module = SemanticMapper(
            self, 
            max_height=cfg.semantic_mapping.explored.max_height, 
            min_height=cfg.semantic_mapping.explored.min_height, 
            pcd_threshold=cfg.semantic_mapping.explored.pcd_threshold
        ).to(self.device)
        
        self.obstacle_module = SemanticMapper(
            self, 
            max_height=cfg.semantic_mapping.obstacle.max_height, 
            min_height=cfg.semantic_mapping.obstacle.min_height, 
            pcd_threshold=cfg.semantic_mapping.obstacle.pcd_threshold
        ).to(self.device)
        
        self.high_obstacle_module = SemanticMapper(
            self, 
            max_height=cfg.semantic_mapping.high_obstacle.max_height,
            min_height=cfg.semantic_mapping.high_obstacle.min_height
        ).to(self.device)
        
        self.extra_area_module = SemanticMapper(
            self, 
            max_height=cfg.semantic_mapping.extra_area.max_height, 
            min_height=cfg.semantic_mapping.extra_area.min_height, 
            pcd_threshold=cfg.semantic_mapping.extra_area.pcd_threshold
        ).to(self.device)
        
        self.elevation_module = ElevationMapper(
            self, 
            max_height=self.max_height, 
            min_height=self.min_height, 
            slice_point_threshold=cfg.semantic_mapping.elevation.slice_point_threshold
        ).to(self.device)
        
        self.elevation_gradiant_module = ElevationGradientCalculator(
            self, 
            gradient_threshold=cfg.semantic_mapping.elevation_gradient.gradient_threshold
        )
        
        self.upstairs_obstacle_module = SemanticMapper(
            self, 
            max_height=cfg.semantic_mapping.upstairs_obstacle.max_height,
            min_height=cfg.semantic_mapping.upstairs_obstacle.min_height
        ).to(self.device)
        
        self.upstairs_free_module = SemanticMapper(
            self, 
            max_height=cfg.semantic_mapping.upstairs_free.max_height,
            min_height=cfg.semantic_mapping.upstairs_free.min_height
        ).to(self.device) 
        
        self.explored_module.eval()
        self.obstacle_module.eval()
        self.high_obstacle_module.eval()
        self.extra_area_module .eval()
        self.upstairs_obstacle_module.eval()
        self.upstairs_free_module.eval()
        self.elevation_module.eval()
        
        self.explored_module.set_view_angles(0)
        self.obstacle_module.set_view_angles(0)
        self.high_obstacle_module.set_view_angles(0)
        self.extra_area_module.set_view_angles(0)
        self.upstairs_obstacle_module.set_view_angles(0)
        self.upstairs_free_module.set_view_angles(0)
        self.elevation_module.set_view_angles(0)
        
        self.init_map()
        self.observations = None
        self.action = None
        self.action_cnt = cfg.robot.action_cnt
        self.robot_radius = cfg.robot.radius
        self.radius = cfg.robot.radius
        self.pitch = cfg.perception.pitch
        
        self.mask_area_threshold = cfg.detection_filter.mask_area_threshold
        self.conf_threshold = cfg.detection_filter.conf_threshold
        self.max_bbox_area_ratio = cfg.detection_filter.max_bbox_area_ratio
        
        self.overlap_weight = cfg.similarity.overlap_weight
        self.visual_sim_weight = cfg.similarity.visual_sim_weight
        self.max_match_distance = cfg.similarity.max_match_distance
        self.distance_threshold = cfg.similarity.distance_threshold
        self.center_distance_decay = cfg.similarity.center_distance_decay
        self.distance_penalty_strength = cfg.similarity.distance_penalty_strength
        self.min_similarity_thresh = cfg.similarity.min_similarity_thresh
        self.containment_visual_sim_threshold = cfg.similarity.containment_visual_sim_threshold
        
        self.turning_angle_threshold = cfg.actions.turning_angle_threshold
        self.max_steps = cfg.actions.max_steps
        self.depth_min = cfg.actions.depth_min
        self.depth_max = cfg.actions.depth_max
        
        self.max_bs_collision_threshold = cfg.fsm.max_bs_collision_threshold
        self.collision_plus_detection_radius = cfg.fsm.collision_plus_detection_radius
        self.max_recovery_steps = cfg.fsm.max_recovery_steps
        self.failure_detection_history_length = cfg.fsm.failure_detection_history_length
        self.kl_divergence_recheck_threshold = cfg.fsm.kl_divergence_recheck_threshold
        self.entropy_threshold_ratio = cfg.fsm.entropy_threshold_ratio
        self.max_confirmed_bbox_views = cfg.fsm.max_confirmed_bbox_views
        self.min_confirmed_bbox_views = cfg.fsm.min_confirmed_bbox_views
        
        self.bs_replan_distance_threshold = cfg.fsm.bs_replan_distance_threshold
        self.navigation_distance_threshold = cfg.fsm.navigation_distance_threshold
        self.object_close_confirmation_distance = cfg.fsm.object_close_confirmation_distance
        self.object_very_close_distance = cfg.fsm.object_very_close_distance
        self.room_approach_check_distance = cfg.fsm.room_approach_check_distance
        self.room_fine_tuning_distance = cfg.fsm.room_fine_tuning_distance
        self.center_change_threshold = cfg.fsm.center_change_threshold
        self.blocked_target_check_interval = cfg.fsm.blocked_target_check_interval
        self.blocked_target_reachability_radius = cfg.fsm.blocked_target_reachability_radius
        self.room_destination_max_radius = cfg.fsm.room_destination_max_radius
        self.mandatory_detection_interval = cfg.fsm.mandatory_detection_interval
        self.bbox_boundary_margin = cfg.fsm.bbox_boundary_margin
        self.planar_detection_threshold = cfg.fsm.planar_detection_threshold
        self.planar_inlier_ratio_threshold = cfg.fsm.planar_inlier_ratio_threshold
        self.max_collision_escaping_count = cfg.fsm.max_collision_escaping_count
        
        self.min_valid_candidate_distance = cfg.exploration.min_valid_candidate_distance
        self.frontier_clustering_distance = cfg.exploration.frontier_clustering_distance
        self.good_forward_angle_threshold = cfg.exploration.good_forward_angle_threshold
        self.good_forward_distance_threshold = cfg.exploration.good_forward_distance_threshold
        self.top_k_forward_ratio = cfg.exploration.top_k_forward_ratio
        self.navigation_temp_goal_distance = cfg.exploration.navigation_temp_goal_distance
        self.max_turning_count = cfg.exploration.max_turning_count
        self.object_destination_radius = cfg.exploration.object_destination_radius
        self.escape_point_search_radius = cfg.exploration.escape_point_search_radius
        self.escape_point_max_distance = cfg.exploration.escape_point_max_distance
        self.mcts_iterations = cfg.exploration.mcts_iterations
        self.mcts_exploration_weight = cfg.exploration.mcts_exploration_weight
        self.mcts_rollout_visit_ratio = cfg.exploration.mcts_rollout_visit_ratio
        self.max_objects_per_landmark = cfg.exploration.max_objects_per_landmark
        self.scene_radius = cfg.exploration.scene_radius
        self.num_clusters = cfg.exploration.num_clusters
        self.w_angle = cfg.exploration.w_angle
        self.w_dist = cfg.exploration.w_dist
        self.w_explore = cfg.exploration.w_explore
        self.w_avoid = cfg.exploration.w_avoid
        
        self.hard_constraint_overlap_threshold = cfg.similarity.hard_constraint_overlap_threshold
        self.hard_constraint_visual_threshold = cfg.similarity.hard_constraint_visual_threshold
        self.high_visual_boost_enabled = cfg.similarity.high_visual_boost_enabled
        self.high_visual_threshold = cfg.similarity.high_visual_threshold
        self.high_visual_distance_threshold = cfg.similarity.high_visual_distance_threshold
        self.high_visual_boost_factor = cfg.similarity.high_visual_boost_factor
        
        self.use_hybrid_astar = cfg.path_planning.use_hybrid_astar
        self.max_adjustment_radius = cfg.path_planning.max_adjustment_radius
        self.start_adjustment_radius = cfg.path_planning.start_adjustment_radius
        
        self.centering_weight = cfg.path_planning.centering_weight
        self.safety_margin = cfg.path_planning.safety_margin
        self.high_risk_penalty = cfg.path_planning.high_risk_penalty
        
        self.forward_distance = cfg.path_planning.forward_distance
        self.turn_angle = cfg.path_planning.turn_angle
        self.theta_resolution = cfg.path_planning.theta_resolution
        self.goal_arrival_factor = cfg.path_planning.goal_arrival_factor
        self.hybrid_centering_weight = cfg.path_planning.hybrid_centering_weight
        self.hybrid_safety_margin = cfg.path_planning.hybrid_safety_margin
        self.hybrid_high_risk_penalty = cfg.path_planning.hybrid_high_risk_penalty
        
        self.dilate_kernel = cfg.skeleton.dilate_kernel
        self.min_component_size = cfg.skeleton.min_component_size
        self.pruning_iterations = cfg.skeleton.pruning_iterations
        self.min_junction_distance = cfg.skeleton.min_junction_distance
        
        # BFS radius
        self.bfs_radius = cfg.exploration.bfs_radius
        
        self.merged_pcd = o3d.geometry.PointCloud()
        
        self.point_goal = tuple(cfg.navigation.point_goal)
        self.temp_path = []
        
        self.next_goal = tuple(cfg.navigation.next_goal)
        with open(cfg.paths.classes_txt, 'r') as file:
            self.vocabulary = [line.rstrip('\n') for line in file]
        with open(cfg.paths.classes_json) as file:
            self.class_description = json.load(file)

        
        self.state = cfg.navigation.state   # Broad Search
        
        self.optional_room_list = cfg.rooms.optional_room_list
        self.room_segmentation_delta = cfg.rooms.room_segmentation_delta
        self.min_room_area = cfg.rooms.min_room_area

        self.llm_model = None
        self.llm_tokenizer = None
        
        # self.openai_api_key = "EMPTY"
        # self.openai_api_base = "http://localhost:8000/v1"
        # self.client = OpenAI(
        #                 api_key=self.openai_api_key,
        #                 base_url=self.openai_api_base,
        #             )
        
        client_cfg = cfg.client
        self.client = getattr(client_cfg, "ollama_url", client_cfg.url)
        self.model = getattr(client_cfg, "ollama_model", client_cfg.model)
        self.authorization = getattr(client_cfg, 'authorization', "")
        
        self.use_clip_feature = getattr(cfg, 'use_clip_feature', False)
        if self.use_clip_feature:
            clip_cache_dir = os.path.join(os.getcwd(), "models", "checkpoints", "clip")
            os.makedirs(clip_cache_dir, exist_ok=True)
            
            os.environ['TORCH_HOME'] = os.path.join(os.getcwd(), "models", "checkpoints")
            
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                "ViT-H-14", 
                pretrained="laion2b_s32b_b79k",
                cache_dir=clip_cache_dir
            )
            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.eval()
        else:
            self.clip_model = None
            self.clip_preprocess = None
        
        self.scene_graph = SceneGraph(agent=self)
        self.landmark_planner = LandmarkPlanner(agent=self)
        self.state_machine = NavigationStateMachine(agent=self)
        
        
        
    def reset(self):
        # print("=======ATTENTION=======")
        # print("env test", self.simulator._env)
        # print("get metrics", self.simulator._env.get_metrics())
        # print("measurement", self.simulator._env._task.measurements.measures)
        # print("task", self.simulator._env._task)
        # # print(type(self.simulator._env._task.measurements.measures['distance_to_goal']._sim.get_agent_state().position))
        # distance_to_goal = self.simulator._env._task.measurements.measures['distance_to_goal']
        # print("agent pos", distance_to_goal._sim.get_agent_state().position)
        # episode=self.simulator._env.current_episode
        # for i, goal in enumerate(episode.goals):
        #     print(f"goal_{i}--goal pos--{goal.position}")
        #     zqh_episode_view_point = [view_point.agent_state.position for view_point in goal.view_points]
        #     min_dist = distance_to_goal._sim.geodesic_distance(
        #         np.array([3.3561, 0.068823, 2.2482]),
        #         zqh_episode_view_point,
        #         episode)
        #     print("dist", min_dist)
        # print("=======ATTENTION=======")
        
        
        self.episode_id += 1
        self.total_steps = 0
        self.gps = [0.0,0.0,0.0]
        self.obj_goal = self.simulator._env.current_episode.object_category
        self.normalize_goal_category()
        self.obj_goal_id = self.vocabulary.index(self.obj_goal)
        # self.obj_goal = 'weeder'

        # episode=self.simulator._env.current_episode
        # self.episode_view_points = []
        # distance_to_goal = self.simulator._env._task.measurements.measures['distance_to_goal']
        # agent_pos = distance_to_goal._sim.get_agent_state().position
        # # for goal in episode.goals:
        # #     if abs(goal.position[1] - agent_pos[1]) < 1.0:
        # #         for view_point in goal.view_points:
        # #             view_point_x = 400 + 20 * (view_point.agent_state.position[0] - agent_pos[0])
        # #             view_point_y = 400 - 20 * (view_point.agent_state.position[2] - agent_pos[2])
        # #             self.episode_view_points.append((view_point_x, view_point_y))
        # for goal in episode.goals:
        #     if abs(goal.position[1] - agent_pos[1]) < 1.0:
        #         for view_point in goal.view_points:
        #             view_point_x = self.center_pixel + 20 * view_point.agent_state.position[0]
        #             view_point_y = self.center_pixel + 20 * view_point.agent_state.position[2]
        #             self.episode_view_points.append((view_point_x, view_point_y))
        
        if self.episode_id >= self.episode_l and self.episode_id < self.episode_r:
            self.room_rank = room_rank(self.client, self.model, self.optional_room_list, self.obj_goal, authorization=self.authorization)
            self.curr_detection_nodes = []
            self.metrics = {'distance_to_goal': 0., 'spl': 0., 'softspl': 0.}
            self.action_cnt = 0
            self.scene_graph.reset()
            self.landmark_planner.reset()
            self.state_machine.reset(obj_goal=self.obj_goal, room_rank=self.room_rank)
            # if self.episode_id > 5:
            print(f"============================episode_{self.episode_id}============================")
            print(f"my object goal is {self.obj_goal} and the room rank is {self.room_rank}")
            self.episode_dir = os.path.join(self.save_dir, f"episode_{self.episode_id}")
            self.pcd_dir = os.path.join(self.episode_dir, "pcd")
            self.map_dir = os.path.join(self.episode_dir, "maps")
            self.room_dir = os.path.join(self.episode_dir, "room")
            self.original_map_dir = os.path.join(self.episode_dir, "origin")
            self.place_dir = os.path.join(self.episode_dir, "places")
            self.skeleton_dir = os.path.join(self.episode_dir, "skeleton")
            self.elevation_dir = os.path.join(self.episode_dir, "elevation")
            self.gradient_dir = os.path.join(self.episode_dir, "gradient")
            self.rgb_dir = os.path.join(self.episode_dir, "rgb")
            self.depth_dir = os.path.join(self.episode_dir, "depth")
            self.bbox_dir = os.path.join(self.episode_dir, "bbox")
            self.e2_dir = os.path.join(self.episode_dir, "e2")
            
            os.makedirs(self.episode_dir, exist_ok=True)
            os.makedirs(self.pcd_dir, exist_ok=True)
            os.makedirs(self.map_dir, exist_ok=True)
            os.makedirs(self.room_dir, exist_ok=True)
            os.makedirs(self.rgb_dir, exist_ok=True)
            os.makedirs(self.bbox_dir, exist_ok=True)
            os.makedirs(self.e2_dir, exist_ok=True)
            
            if self.type == "debug":
                os.makedirs(self.original_map_dir, exist_ok=True)
                os.makedirs(self.skeleton_dir, exist_ok=True)
                os.makedirs(self.place_dir, exist_ok=True)
                os.makedirs(self.elevation_dir, exist_ok=True)
                os.makedirs(self.gradient_dir, exist_ok=True)
                os.makedirs(self.depth_dir, exist_ok=True)
                        
            
            
            self.pose_file = open(os.path.join(self.episode_dir, "pose.txt"), "w")
            self.pose_file.write("step\tx\ty\tz\tcompass\tpitch\n")
            
            self.init_map()
            self.point_goal = (self.center_pixel, self.center_pixel)
            self.temp_path = []
            self.action = None
            self.pitch = 0.0
            
            self.state = 'BS'
        
    def _save_pose(self, observations):
        """Save the pose for either 2-D or 3-D GPS readings."""
        try:
            gps = observations.get("gps", np.zeros(2))
            compass = observations.get("compass", [0.0])
            
            x = gps[0] if len(gps) > 0 else 0.0
            y = gps[1] if len(gps) > 1 else 0.0
            z = gps[2] if len(gps) > 2 else 0.0
            
            pose_str = "{:04d}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\n".format(
                self.total_steps, x, y, z, compass[0], self.pitch
            )
            self.pose_file.write(pose_str)
        except Exception as e:
            print(f"Could not save pose: {str(e)}\nRaw data: gps={gps}, compass={compass}, pitch={self.pitch}")
    
    def update_position_from_gps(self, observations):
        if len(observations['gps']) == 2:
            self.gps[0] = observations['gps'][0]
            self.gps[1] = observations['gps'][1]
        else:
            self.gps[0] = - observations['gps'][2]
            self.gps[1] = observations['gps'][0]
            self.gps[2] = observations['gps'][1]        

    def _process_depth(self, depth_obs):
        """Process the depth sensor observation."""
        if isinstance(depth_obs, torch.Tensor):
            depth_obs = depth_obs.cpu().numpy()
        
        min_depth = 0.0
        max_depth = 5.0
        depth_normalized = np.clip((depth_obs - min_depth)/(max_depth - min_depth), 0, 1)
        return (depth_normalized * 65535).astype(np.uint16)
    
    def normalize_goal_category(self):
        if self.obj_goal == "tv_monitor" or self.obj_goal == "tv":
            self.obj_goal = "television"
        elif self.obj_goal == "potted_plant":
            self.obj_goal = "plant"
        elif self.obj_goal == "couch":
            self.obj_goal = "sofa"
    
    def act(self, observations):
        # return self._act(observations)
        try:
            return self._act(observations)
        except Exception as e:
            print(f"Err={e}, so we stop!")
            print(f"\n{traceback.format_exc()}")
            return {"action": 0}
    
    def _act(self, observations):
        if self.total_steps >= 500:
            return {"action": 0}
        
        if self.total_steps == 499:
            print("[DEBUG] 499 steps fail case")
            print(f"object goal is {self.obj_goal}, collision set is {self.landmark_planner.path_planner.collision_offsets}")
            print(f"temporarily_blocked_targets is {self.state_machine.temporarily_blocked_targets}, permanently_unreachable_targets is {self.state_machine.permanently_unreachable_targets}, confirmed_wrong_target is {self.state_machine.confirmed_wrong_list}")
            print("[DEBUG] check wrong object class")
            for obj_id in self.state_machine.temporarily_blocked_targets + self.state_machine.permanently_unreachable_targets + self.state_machine.confirmed_wrong_list:
                try:
                    obj = self.scene_graph.object_node_lst[obj_id]
                    o3d.io.write_point_cloud(os.path.join(self.pcd_dir, f"wrong_obj{obj.id}_class_{obj.class_name}.ply"), obj.pcd)
                except:
                    print("can't write pcd!")
                
                try:
                    obj = self.scene_graph.object_node_lst[obj_id]
                    print(f"The object {obj_id} is {obj.class_name}")
                    if obj.class_name in self.vocabulary:
                        print(f"Its probs for {obj.class_name} is {obj.class_candidates[self.vocabulary.index(obj.class_name)]} and for object goal is {obj.class_candidates[self.vocabulary.index(obj.class_name)]}")
                        print(f"Its entropy is {obj.class_entropy()}")
                except:
                    print("can't write pcd!")
        
        # if self.type == 'debug':
        #     if self.total_steps == 52:
        #         for obj in self.scene_graph.object_node_lst:
        #             o3d.io.write_point_cloud(os.path.join(self.pcd_dir, f"step_{self.total_steps}_id_{obj.id}_class_{obj.class_name}.ply"), obj.pcd)
            
        if self.episode_id < self.episode_l or self.episode_id >= self.episode_r:
            return {"action": 0}

           
        self.total_steps += 1
        print(f"=======================episode{self.episode_id}---step{self.total_steps}=======================")
        distance_to_goal = self.simulator._env._task.measurements.measures['distance_to_goal']
        episode = self.simulator._env.current_episode
        distance_to_goal_points = distance_to_goal._sim.geodesic_distance(distance_to_goal._sim.get_agent_state().position, [goal.position for goal in episode.goals], episode)
        distance_to_view_points = distance_to_goal._sim.geodesic_distance(distance_to_goal._sim.get_agent_state().position, distance_to_goal._episode_view_points, episode)
        print("distance_to_goal_point=", distance_to_goal_points)
        print("distance_to_view_points=", distance_to_view_points)

        observations["depth"][observations["depth"]==0.5] = 100 # don't construct unprecise map with distance less than 0.5 m

        self.observations = observations
        self.update_position_from_gps(observations)
        self.pixel_x = round(self.center_pixel + self.gps[0] * 100 / self.resolution)
        self.pixel_y = round(self.center_pixel - self.gps[1] * 100 / self.resolution)
        
        center_lst = []
        draw_room_node = []
        room_node_lst = []
        # for i, object_node in enumerate(self.scene_graph.object_node_lst):
        #     class_name = object_node.class_name
        #     center = object_node.center
        #     center_lst.append((center, class_name))
        self.update_map(observations) 
        
        if self.total_steps > 16:
            start_time = time.time()
            if 'CE' in self.state_machine.state or self.state_machine.state == 'AP':
                bbox_img = cv2.cvtColor(observations["rgb"], cv2.COLOR_RGB2BGR)
                self.curr_detection_nodes = []
            else:
                detections, _, bbox_img, probs = self.scene_graph.detect_objects()
                self.curr_detection_nodes, _ = self.scene_graph.integrate_detections(detections, probs)
                self.state_machine.mandatory_detection = False
                # print("[DEBUG] blocked target:", self.state_machine.temporarily_blocked_targets)
                self.state_machine.recheck_blocked_targets(self.scene_graph.object_node_lst)
            self.scene_graph.segment_rooms(delta=self.room_segmentation_delta)
            self.draw_room_regions(self.scene_graph.room_node_lst)
            self.draw_room_dilation_debug(self.scene_graph.room_debug)
            
    
            print(f"TIME for scene_graph generation:{time.time()-start_time}")
            
            start_time = time.time()
            self.state_machine.update_navigation_state()
            print(f"TIME for NavigationStateMachine:{time.time()-start_time}")
                
            self.save_observations_bbox(bbox_img)
            
        self.save_observations(observations)
        self._save_pose(observations)
        
        start_time = time.time()
        
        print(f"TIME for update map:{time.time()-start_time}")
        
        
        # episode=self.simulator._env.current_episode
        # self.episode_view_points = []
        # distance_to_goal = self.simulator._env._task.measurements.measures['distance_to_goal']
        # agent_pos = distance_to_goal._sim.get_agent_state().position
        # print("agent pos: ", agent_pos)
        # for goal in episode.goals:
        #     # if abs(goal.position[1] - agent_pos[1]) < 0.5:
        #     #     view_point_x = 400 + 20 * (goal.position[0] - agent_pos[0])
        #     #     view_point_y = 400 + 20 * (goal.position[2] - agent_pos[2])
        #     #     self.episode_view_points.append((view_point_y, view_point_x))
            
        #     if abs(goal.position[1] - agent_pos[1]) < 1.0:
        #         for view_point in goal.view_points:
        #             view_point_x = 400 + 20 * (view_point.agent_state.position[0] - agent_pos[0])
        #             view_point_y = 400 - 20 * (view_point.agent_state.position[2] - agent_pos[2])
        #             self.episode_view_points.append((view_point_x, view_point_y))
        
        
        if self.state_machine.state == 'ST':
            self.draw_final_map([], self.occupancy_map, observations, "total", [], [])
            print(f"find!!! x={self.pixel_x}, y={self.pixel_y} and object goal is {self.obj_goal}")
            
            try:
                o3d.io.write_point_cloud(os.path.join(self.pcd_dir, f"episode_{self.episode_id}_class_{self.obj_goal}.ply"), self.state_machine.confirmed_essential_obj.pcd)
                if self.type == 'debug':
                    for obj in self.scene_graph.object_node_lst:
                        o3d.io.write_point_cloud(os.path.join(self.pcd_dir, f"episode_{self.episode_id}_id_{obj.id}_class_{obj.class_name}.ply"), obj.pcd)
                    # for i, picture in enumerate(self.state_machine.confirmed_bbox_dict[self.state_machine.confirmed_essential_obj.id]):
                    #     cv2.imwrite(os.path.join(self.pcd_dir, f"step_{self.total_steps}_id_{self.state_machine.confirmed_essential_obj.id}_picture_{i}.png"), picture)
            
            
            except:
                print("can't save point cloud! ")

            # for i, pic in enumerate(self.state_machine.confirmed_bbox_dict[self.state_machine.confirmed_essential_obj.id]):
            #     pic = cv2.cvtColor(pic, cv2.COLOR_RGB2BGR)
            #     cv2.imwrite(os.path.join(self.pcd_dir, f"pic_{i}.png"), pic)
            return {"action": 0}
       
        if self.total_steps < 17:
            self.draw_final_map([], self.occupancy_map, observations, "total", [], [])
            # self.draw_final_map([], self.elevation_gradiant_map, observations, "gradient", [], [])
            # self.visualize_elevation_map(self.elevation_map, min_h=self.min_height, max_h=self.max_height)

        if self.total_steps == 1:
            self.action = 5
        elif self.total_steps <= 7:
            self.action = 6
        elif self.total_steps == 8:
            self.action = 5
        elif self.total_steps <= 14:
            self.action = 6
        elif self.total_steps <= 15:
            self.action = 4
        elif self.total_steps <= 16:
            self.action = 4
        
        elif self.action_cnt == 0:
            # if self.state_machine.state != 'CE':
            # if self.state_machine.confirmed_essential_obj is not None:
            #     self.zqhid = self.state_machine.confirmed_essential_obj.id
            #     o3d.io.write_point_cloud(os.path.join(self.pcd_dir, f"step_{self.total_steps}_id_{self.state_machine.confirmed_essential_obj.id}.ply"), self.state_machine.confirmed_essential_obj.pcd)
            # else:
            #     for obj in self.scene_graph.object_node_lst:
            #         try:
            #             if obj.id == self.zqhid:
            #                 o3d.io.write_point_cloud(os.path.join(self.pcd_dir, f"step_{self.total_steps}_id_{self.state_machine.confirmed_essential_obj.id}.ply"), obj.pcd)
            #         except:
            #             pass
            start_time = time.time()
            self.action, self.point_goal, self.temp_path, vis_img = self.landmark_planner.plan_local_action()
            # if self.total_steps > 87 and self.total_steps < 95:
            #     print(self.landmark_planner.path_planner.collision_offsets)
            #     print(self.temp_path)
            print(f"TIME for local policy:{time.time()-start_time}")

            start_time = time.time()
            
            for room_node in self.landmark_planner.room_lst:
                draw_room_node.append((room_node.center[0], room_node.center[1]))
                room_node_lst.append(room_node.room_name)
            
            # visualization
            self.draw_final_map(center_lst, self.occupancy_map, observations, "total", draw_room_node, room_node_lst)
            if self.type == "debug":
                self.draw_final_map(center_lst, self.occupancy_map, observations, "places", draw_room_node, room_node_lst)
                self.draw_final_map(center_lst, self.original_occupancy_map, observations, "origin", draw_room_node, room_node_lst)
                self.draw_final_map(center_lst, self.elevation_gradiant_map, observations, "gradient", draw_room_node, room_node_lst)
                self.visualize_elevation_map(self.elevation_map, min_h=self.min_height, max_h=self.max_height)
                cv2.imwrite(os.path.join(self.skeleton_dir, f"skeleton_{self.total_steps:04d}.png"), vis_img)
                     
            
            savepath = os.path.join(self.map_dir, f'map_total_{self.total_steps:04d}.png')
            self.visualize_path(savepath, self.pixel_x, self.pixel_y, savepath)
            self.draw_landmark_overview()
            
        
            print(f"check state:{self.state_machine.state}")
            print("-------------Object Node Begin-------------")
            print([(obj.class_name, obj.id, obj.score) for obj in self.scene_graph.object_node_lst])
            print("--------------Object Node End--------------")
            print("**************NavigationStateMachine Test Begin**************")
            if self.state_machine.approaching_essential_obj is not None:
                print(f"essential_obj_center is {self.state_machine.approaching_essential_obj.center[0] * self.r_resolution}, {self.state_machine.approaching_essential_obj.center[1] * self.r_resolution}")
            if self.state_machine.confirmed_essential_obj is not None:
                print(f"confirmed_obj_center is {self.state_machine.confirmed_essential_obj.center[0] * self.r_resolution}, {self.state_machine.confirmed_essential_obj.center[1] * self.r_resolution}")
                print(f"confirmed obj_id is {self.state_machine.confirmed_essential_obj.id}, and its score dict is {self.state_machine.confirmed_essential_obj.class_candidates}")
            print(f"confirmed_object_list is {self.state_machine.confirmed_object_list}")
            print("***************NavigationStateMachine Test End***************")
            print("***************Locomotion Test Begin***************")
            print(f"agent pos is {self.pixel_x, self.pixel_y}")
            if self.landmark_planner.point_goal is not None:
                print(f"point goal is {self.landmark_planner.point_goal}")
            print("***************Locomotion Test End***************")
            print(f"action = {self.action}")
            print(f"TIME for data saving:{time.time()-start_time}")
        
        
        if isinstance(self.action, list):
            if self.action_cnt < len(self.action) - 1:
                self.action_cnt += 1
                self.update_pitch(self.action[self.action_cnt-1])
                return {"action": self.action[self.action_cnt-1]}
            else:
                self.action_cnt = 0
                self.update_pitch(self.action[-1])
                return {"action": self.action[-1]}
        else:
            self.update_pitch(self.action)
            return {"action": self.action}
    
    def init_map(self):
        self.map_size = self.map_size_cm // self.map_resolution
        full_w, full_h = int(self.map_size), int(self.map_size)
        self.full_map = torch.zeros(1,1 ,full_w, full_h).float().to(self.device)
        self.room_map = torch.zeros(1,9 ,full_w, full_h).float().to(self.device)
        self.visited = self.full_map[0,0].cpu().numpy()
        self.collision_map = self.full_map[0,0].cpu().numpy()
        self.occupancy_map = torch.zeros(1,1 ,full_w, full_h).float().to(self.device)
        self.original_occupancy_map = torch.zeros(1,1 ,full_w, full_h).float().to(self.device)
        self.one_step_obstacle_map = torch.zeros(1,1 ,full_w, full_h).float().to(self.device)
        self.explored_map = torch.zeros(1,1 ,full_w, full_h).float().to(self.device)
        self.obstacle_map = 2.0 * torch.zeros(1,1 ,full_w, full_h).float().to(self.device) # 0 is unknown, 1 is free
        self.elevation_map = torch.full((1, 1, full_h, full_w), fill_value=-300, dtype=torch.float).to(self.device)
        self.elevation_gradiant_map = torch.zeros(1,1 ,full_w, full_h).float().to(self.device)
        self.one_step_obstacle_map = 2.0 * torch.zeros(1,1 ,full_w, full_h).float().to(self.device)
        self.one_step_explored_map = torch.zeros(1,1 ,full_w, full_h).float().to(self.device)
        self.stairs_map = torch.zeros(1,1 ,full_w, full_h).float().to(self.device)
        self.upstairs_free_map = torch.zeros(1,1 ,full_w, full_h).float().to(self.device)
        self.upstairs_obstacle_map = torch.zeros(1,1 ,full_w, full_h).float().to(self.device)
        self.upstairs_map = 2.0 * torch.zeros(1,1 ,full_w, full_h).float().to(self.device)
        self.floor_map = torch.zeros(1,1 ,full_w, full_h).float().to(self.device)
        self.is_stairs_area = torch.zeros(full_w, full_h).int().to(self.device)
        self.is_floor_area = torch.zeros(full_w, full_h).int().to(self.device)
        self.high_occupancy_map = 2.0 * torch.zeros(1,1 ,full_w, full_h).float().to(self.device) # 0 is unknown, 1 is free
        self.high_obstacle_map = 2.0 * torch.zeros(1,1 ,full_w, full_h).float().to(self.device) # 0 is unknown, 1 is free
        self.room_segmentation_map = torch.zeros(1,1 ,full_w, full_h).float().to(self.device)
        self.full_pose = torch.zeros(3).float().to(self.device)
        self.goal_gps_map = self.full_map[0,0].cpu().numpy()
        self.origins = np.zeros((2))
        
        def init_map_and_pose():
            self.full_map.fill_(0.)
            self.full_pose.fill_(0.)
            self.full_pose[:2] = self.map_size_cm / 100.0 / 2.0  # put the agent in the middle of the map

        init_map_and_pose()
        
    def update_map(self, observations):
        self.full_pose[0] = self.map_size_cm / 100.0 / 2.0+self.gps[0]
        self.full_pose[1] = self.map_size_cm / 100.0 / 2.0-self.gps[1]
        self.full_pose[2:] = torch.from_numpy(observations['compass'] * 57.29577951308232).to(self.device) # input degrees and meters
        self.agent_z_pos = self.gps[2] 
        depth_data = torch.from_numpy(observations['depth'])
        valid_depth_data = depth_data * ((depth_data > 0.5) & (depth_data < 5.0))
        self.obstacle_map = self.obstacle_module(torch.squeeze(valid_depth_data, dim=-1).to(self.device), self.full_pose, self.agent_z_pos, self.obstacle_map)
        self.explored_map = self.explored_module(torch.squeeze(valid_depth_data, dim=-1).to(self.device), self.full_pose, self.agent_z_pos, self.explored_map)
        self.high_obstacle_map = self.high_obstacle_module(torch.squeeze(valid_depth_data, dim=-1).to(self.device), self.full_pose, self.agent_z_pos, self.high_obstacle_map)
        
        # self.obstacle_map[(self.occupancy_map > 0) & (self.occupancy_map < 2)] = 0
          
        self.occupancy_map = self.obstacle_map + self.explored_map
        self.original_occupancy_map = self.obstacle_map + self.explored_map
        self.room_segmentation_map = self.high_obstacle_map + self.explored_map
        
        self.elevation_map = self.elevation_module(torch.squeeze(valid_depth_data, dim=-1).to(self.device), self.full_pose, self.agent_z_pos, self.elevation_map)
        self.elevation_gradiant_map = self.elevation_gradiant_module(self.elevation_map)
        
        self.occupancy_map[self.elevation_gradiant_map == 1] = 1
        # self.original_occupancy_map = self.occupancy_map
        self.occupancy_map = fill_small_holes(
            occupancy_map=self.occupancy_map,
            closing_kernel_size=self.closing_kernel_size,
            hole_fill_area=self.hole_fill_area
        )   
    
    def merge_pcd(self):
        merged_pcd = merge_pcd_batch(object_list=self.scene_graph.object_node_lst)
        return merged_pcd

    def update_pitch(self, action):
        if action == 4: 
            self.pitch += math.pi/6
        elif action == 5:
            self.pitch -= math.pi/6
        
        self.obstacle_module.set_view_angles(-1 * math.degrees(self.pitch))
        self.explored_module.set_view_angles(-1 * math.degrees(self.pitch))
        self.high_obstacle_module.set_view_angles(-1 * math.degrees(self.pitch))
        self.extra_area_module.set_view_angles(-1 * math.degrees(self.pitch))
        self.upstairs_obstacle_module.set_view_angles(-1 * math.degrees(self.pitch))
        self.upstairs_free_module.set_view_angles(-1 * math.degrees(self.pitch))
        self.elevation_module.set_view_angles(-1 * math.degrees(self.pitch))
        
    def update_metrics(self, metrics):
        self.metrics['distance_to_goal'] = metrics['distance_to_goal']
        self.metrics['spl'] = metrics['spl']
        self.metrics['softspl'] = metrics['softspl']



@hydra.main(version_base=None, config_path="../configs", config_name="e2")
def main(cfg: DictConfig) -> None:
    os.environ["CHALLENGE_CONFIG_FILE"] = f"configs/{cfg.experiment.dataset}.yaml"
    config = habitat.get_config(os.environ["CHALLENGE_CONFIG_FILE"])
    
    agent = ObjectNavigationAgent(task_config=config, cfg=cfg)
    evaluate(agent, config, cfg.experiment)

if __name__ == "__main__":
    main()

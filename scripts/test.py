import argparse
import sys
import os

import habitat
import copy
import json
import cv2
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import Normalize
from openai import OpenAI
import open3d as o3d
from datetime import datetime
from scripts.scene_graph import SceneGraph
from scripts.landmark_planner import LandmarkPlanner
from scripts.navigation_state_machine import NavigationStateMachine
from scripts.video_gen import pic_to_video
from PIL import Image, ImageDraw


from utils.occupancy_mapping import SemanticMapper, ElevationMapper, ElevationGradientCalculator
from utils.utils import merge_point_clouds, merge_pcd_batch
from utils.llm_utils import room_rank

import warnings


warnings.filterwarnings('ignore', message='`BoxAnnotator` is deprecated and will be removed in `supervision-0.22.0`')

class DiagnosticNavigationAgent():
    def __init__(self, task_config, args=None):
        self._POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS
        self.total_steps = 0
        self.episode_id = 0
        # self.obj_goal = self.simulator._env.current_episode.object_category
        self.obj_goal = None
        self.episode_l = args.episode_l
        self.episode_r = args.episode_r
        self.split_l = args.split_l
        self.split_r = args.split_r
        self.dataset = args.dataset
        self.experiment_name = args.experiment_name
        self.metrics = {'distance_to_goal': 0., 'spl': 0., 'softspl': 0.}
        self.if_upstairs = False
        self.device = f'cuda:{args.cuda_id}'
        self.port = args.port
        self.config = task_config
        self.min_height = -150
        self.max_height = 100
        self.gps = [0.0, 0.0, 0.0]
        
        
        self.detection_segmentation = True
        self.curr_detection_nodes = []
        
        if args.type == 'debug':
            self.save_dir = f"observations/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.split_l}_{self.split_r}"
        else:
            self.save_dir = f"exp/{self.dataset}/{self.experiment_name}/observations/{args.dataset}_{self.split_l}_to_{self.split_r}"
        os.makedirs(self.save_dir, exist_ok=True)
    
        self.map_size_cm = 4000
        self.resolution = self.map_resolution = 5
        self.r_resolution = 100 / self.resolution
        self.center_pixel = round(self.map_size_cm / self.resolution / 2)    
        self.camera_horizon = 0
        self.dilation_deg = 0
        self.collision_threshold = 0.08

        self.explored_module = SemanticMapper(self, max_height=150, min_height=-10, pcd_threshold=0.0).to(self.device)
        self.obstacle_module = SemanticMapper(self, max_height=100, min_height=10, pcd_threshold=0.5).to(self.device)
        self.high_obstacle_module = SemanticMapper(self, max_height=100,min_height=90).to(self.device)
        self.extra_area_module = SemanticMapper(self, max_height=100, min_height=-100, pcd_threshold=0.5).to(self.device)
        
        self.elevation_module = ElevationMapper(self, max_height=self.max_height, min_height=self.min_height, slice_point_threshold=0).to(self.device)
        self.elevation_gradiant_module = ElevationGradientCalculator(self, gradient_threshold=80)
        
        self.upstairs_obstacle_module = SemanticMapper(self, max_height=100,min_height=90).to(self.device)
        self.upstairs_free_module = SemanticMapper(self, max_height=100,min_height=-10).to(self.device)
        
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
        self.action_cnt = 0
        self.robot_radius = 3
        self.pitch = 0.0
        self.merged_pcd = o3d.geometry.PointCloud()
        
        self.point_goal = (self.center_pixel, self.center_pixel)
        self.temp_path = []
        self.look_down = False
        
        self.next_goal = (self.center_pixel, self.center_pixel)
        with open('utils/classes/classes.txt', 'r') as file:
            self.vocabulary = [line.rstrip('\n') for line in file]
        with open('utils/classes/classes.json') as file:
            self.class_description = json.load(file)

        
        self.state = 'BS'   # Broad Search
        
        self.optional_room_list = ["living room", "dining room", "bedroom", "kitchen", "bathroom", "entryway", "study", "home gym", "storage room", "basement", "balcony", "garden", "walk-in closet"]
        # self.optional_room_list = ["living room", "dining room", "bedroom", "kitchen", "bathroom", "study", "home gym", "balcony"]
        
        
        # """
        # Grounded-SAM
        # """




        # self.grounding_dino_model = Model(model_config_path=self.config_file, model_checkpoint_path=self.grounding_dino_checkpoint)
        # self.ram_model = ram(pretrained=self.ram_checkpoint,
        #                                 image_size=384,
        #                                 vit='swin_l').to(self.device)
        # self.sam_predictor = SamPredictor(build_sam(checkpoint=self.sam_checkpoint).to(self.device))
        
        """
        YOLO
        """


        
        # self.detection_model = YOLO(self.yolo_checkpoint)
        # self.detection_model.set_classes(self.vocabulary)
        # self.segmentation_model = SAM(self.sam_checkpoint)
        # self.class_dict = self.detection_model.names
        # assert "stairs" in self.class_dict.values()
        # self.stairs_idx = [k for k, v in self.class_dict.items() if v == "stairs"]
        
        
        # self.vision_model = AutoModel.from_pretrained("google/siglip2-base-patch16-224", cache_dir="models/siglip2", device_map="auto").eval()
        # self.vision_processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224", cache_dir="models/siglip2")



        # self.vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct", torch_dtype="auto", device_map="auto", cache_dir="models/qwenVL_72B")
        # self.vlm_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct", cache_dir="models/qwenVL_72B")


        self.llm_model = None
        self.llm_tokenizer = None
        
        # self.openai_api_key = "EMPTY"
        # self.openai_api_base = "http://localhost:8000/v1"
        # self.client = OpenAI(
        #                 api_key=self.openai_api_key,
        #                 base_url=self.openai_api_base,
        #             )
        
        self.client = os.environ.get("E2NAV_CHAT_URL", "http://localhost:11434/api/chat")
        self.model = "qwen25-vl"
        self.authorization = ""
        
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
            self.original_map_dir = os.path.join(self.episode_dir, "origin")
            self.place_dir = os.path.join(self.episode_dir, "places")
            self.skeleton_dir = os.path.join(self.episode_dir, "skeleton")
            # self.stairs_dir = os.path.join(self.episode_dir, "stairs")
            self.elevation_dir = os.path.join(self.episode_dir, "elevation")
            self.gradient_dir = os.path.join(self.episode_dir, "gradient")
            os.makedirs(self.episode_dir, exist_ok=True)
            os.makedirs(self.pcd_dir, exist_ok=True)
            os.makedirs(self.map_dir, exist_ok=True)
            os.makedirs(self.original_map_dir, exist_ok=True)
            os.makedirs(self.skeleton_dir, exist_ok=True)
            os.makedirs(self.place_dir, exist_ok=True)
            # os.makedirs(self.stairs_dir, exist_ok=True)
            os.makedirs(self.elevation_dir, exist_ok=True)
            os.makedirs(self.gradient_dir, exist_ok=True)
        
            self.rgb_dir = os.path.join(self.episode_dir, "rgb")
            self.depth_dir = os.path.join(self.episode_dir, "depth")
            self.bbox_dir = os.path.join(self.episode_dir, "bbox")
            os.makedirs(self.rgb_dir, exist_ok=True)
            os.makedirs(self.depth_dir, exist_ok=True)
            os.makedirs(self.bbox_dir, exist_ok=True)
            
            self.pose_file = open(os.path.join(self.episode_dir, "pose.txt"), "w")
            self.pose_file.write("step\tx\ty\tz\tcompass\tpitch\n")
            
            self.init_map()
            self.point_goal = (self.center_pixel, self.center_pixel)
            self.temp_path = []
            self.action = None
            self.pitch = 0.0
            self.look_down = False
            
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
            self.obj_goal = "monitor"
        elif self.obj_goal == "potted_plant":
            self.obj_goal = "plant"
        elif self.obj_goal == "couch":
            self.obj_goal = "sofa"
    
    def act(self, observations):
        try:
            return self.act_(observations)
        except Exception as e:
            print(f"Err={e}, so we stop!")
            return {"action": 0}

    def act_(self, observations):
        if self.total_steps >= 500:
            return {"action": 0}
        # if self.total_steps == 50:
        #     for i in range(len(self.scene_graph.object_node_lst)):
        #         class_name = self.scene_graph.object_node_lst[i].class_name

        # if self.total_steps == 26:

        
        # if self.episode_id in [1]:
        #     return {"action": 0}
        # if self.episode_id not in [125, 146]:
        #     return {"action": 0}
        if self.episode_id < self.episode_l or self.episode_id >= self.episode_r:
            return {"action": 0}

            
        self.total_steps += 1
        print(f"=======================episode{self.episode_id}---step{self.total_steps}=======================")
        
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
        stairs_mask = None
        
        if self.total_steps > 16:
            start_time = time.time()
            if 'CE' in self.state_machine.state:
                bbox_img = cv2.cvtColor(observations["rgb"], cv2.COLOR_RGB2BGR)
                self.curr_detection_nodes = []
            else:
                detections, _, bbox_img = self.scene_graph.detect_objects()
                self.curr_detection_nodes, stairs_mask = self.scene_graph.integrate_detections(detections)
                self.state_machine.mandatory_detection = False
    
            print(f"TIME for scene_graph generation:{time.time()-start_time}")
            
            start_time = time.time()
            self.state_machine.update_navigation_state()
            print(f"TIME for NavigationStateMachine:{time.time()-start_time}")
                
            self.save_observations_bbox(bbox_img)

        self.save_observations(observations)
        self._save_pose(observations)
        
        start_time = time.time()
        self.update_map(observations, stairs_mask) 
        print(f"TIME for update map:{time.time()-start_time}")
        
        if self.state_machine.state == 'ST':
            # print("=======FIND=======")
            # print("env test", self.simulator._env)
            # print("get metrics", self.simulator._env.get_metrics())
            # print("measurement", self.simulator._env._task.measurements.measures)
            # print("task", self.simulator._env._task)
            # distance_to_goal = self.simulator._env._task.measurements.measures['distance_to_goal']
            # print("agent pos", distance_to_goal._sim.get_agent_state().position)
            # episode=self.simulator._env.current_episode
            # for i, goal in enumerate(episode.goals):
            #     print(f"episode{i}--goal pos--{goal.position}")
            # print("=======FIND=======")
            
            print(f"find!!! x={self.pixel_x}, y={self.pixel_y}")
            try:
                o3d.io.write_point_cloud(os.path.join(self.pcd_dir, f"episode_{self.episode_id}_class_{self.obj_goal}.ply"), self.state_machine.confirmed_essential_obj.pcd)
                # for obj in self.scene_graph.object_node_lst:
                #     o3d.io.write_point_cloud(os.path.join(self.pcd_dir, f"episode_{self.episode_id}_id_{obj.id}_class_{obj.class_name}.ply"), obj.pcd)
            except:
                print("can't save point cloud! ")
            return {"action": 0}
        
        if self.total_steps < 17:
            self.draw_final_map([], self.occupancy_map, observations, "total", [], [])
            self.draw_final_map([], self.elevation_gradiant_map, observations, "gradient", [], [])
            self.visualize_elevation_map(self.elevation_map, min_h=self.min_height, max_h=self.max_height)

        if self.total_steps == 1:
            self.action = 5
        elif self.total_steps == 2:
            self.action = 5
        elif self.total_steps <= 14:
            self.action = 3
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
            # self.action, self.point_goal, self.temp_path, vis_img = self.landmark_planner.plan_local_action(
            #     occupancy_map=self.occupancy_map,
            #     high_occupancy_map=self.high_occupancy_map,
            #     is_stairs_area=self.is_stairs_area,
            #     observations=observations,
            #     curr_detection_nodes=curr_detection_nodes,
            #     pre_action=self.action,
            #     object_node_lst=self.scene_graph.object_node_lst,
            #     vision_model=None,
            #     vision_processor=None,
            #     vlm_model=None,
            #     vlm_processor=None,
            #     optional_room_list=self.optional_room_list,
            #     NavigationStateMachine=self.state_machine
            # )
            print(f"TIME for local policy:{time.time()-start_time}")

            start_time = time.time()
            
            for room_node in self.landmark_planner.room_lst:
                draw_room_node.append((room_node.center[0], room_node.center[1]))
                room_node_lst.append(room_node.room_name)
            
            self.draw_final_map(center_lst, self.occupancy_map, observations, "total", draw_room_node, room_node_lst)
            self.draw_final_map(center_lst, self.occupancy_map, observations, "places", draw_room_node, room_node_lst)
            self.draw_final_map(center_lst, self.original_occupancy_map, observations, "origin", draw_room_node, room_node_lst)
            # self.draw_final_map(center_lst, self.occupancy_map + self.stairs_map, observations, "stairs", draw_room_node, room_node_lst)
            self.draw_final_map(center_lst, self.elevation_gradiant_map, observations, "gradient", draw_room_node, room_node_lst)
            self.visualize_elevation_map(self.elevation_map, min_h=self.min_height, max_h=self.max_height)
                     
            cv2.imwrite(os.path.join(self.skeleton_dir, f"skeleton_{self.total_steps:04d}.png"), vis_img)
            savepath = os.path.join(self.map_dir, f'map_total_{self.total_steps:04d}.png')
            self.visualize_path(savepath, self.pixel_x, self.pixel_y, savepath)
            # else:
            #     draw_room_node = []
            #     room_node_lst = []
            #     for room_node in self.landmark_planner.room_lst:
            #         draw_room_node.append((room_node.center[0], room_node.center[1]))
            #         room_node_lst.append(room_node.room_name)
            #     self.draw_final_map(center_lst, self.occupancy_map, observations, "total", draw_room_node, room_node_lst)
            #     self.draw_final_map(center_lst, self.occupancy_map, observations, "high", draw_room_node, room_node_lst)
            #     savepath = os.path.join(self.map_dir, f'map_total_{self.total_steps:04d}.png')
            #     self.visualize_path(savepath, pixel_x, pixel_y, savepath)
        
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
    
    def visualize_path(self, image_path, start, end, save_path):
        if self.temp_path != None:
            img = Image.open(image_path)

            draw = ImageDraw.Draw(img)
            if len(self.temp_path) > 1:
                draw.line(self.temp_path, fill=(255,0,0), width=3)
            
            img.save(save_path)
            print(f"Results saved to {save_path}")
          
    def init_map(self):
        self.map_size = self.map_size_cm // self.map_resolution
        full_w, full_h = self.map_size, self.map_size
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
        self.full_pose = torch.zeros(3).float().to(self.device)
        self.goal_gps_map = self.full_map[0,0].cpu().numpy()
        self.origins = np.zeros((2))
        
        def init_map_and_pose():
            self.full_map.fill_(0.)
            self.full_pose.fill_(0.)
            self.full_pose[:2] = self.map_size_cm / 100.0 / 2.0  # put the agent in the middle of the map

        init_map_and_pose()
        
    def update_map(self, observations, stairs_mask):
        self.full_pose[0] = self.map_size_cm / 100.0 / 2.0+self.gps[0]
        self.full_pose[1] = self.map_size_cm / 100.0 / 2.0-self.gps[1]
        self.full_pose[2:] = torch.from_numpy(observations['compass'] * 57.29577951308232).to(self.device) # input degrees and meters
        self.agent_z_pos = self.gps[2] 
        depth_data = torch.from_numpy(observations['depth'])
        valid_depth_data = depth_data * ((depth_data > 0.5) & (depth_data < 5.0))
        self.obstacle_map = self.obstacle_module(torch.squeeze(valid_depth_data, dim=-1).to(self.device), self.full_pose, self.obstacle_map)
        self.explored_map = self.explored_module(torch.squeeze(valid_depth_data, dim=-1).to(self.device), self.full_pose, self.explored_map)
        
        # self.obstacle_map[(self.occupancy_map > 0) & (self.occupancy_map < 2)] = 0
          
        self.occupancy_map = self.obstacle_map + self.explored_map
        self.original_occupancy_map = self.obstacle_map + self.explored_map
        
        self.elevation_map = self.elevation_module(torch.squeeze(valid_depth_data, dim=-1).to(self.device), self.full_pose, self.agent_z_pos, self.elevation_map)
        self.elevation_gradiant_map = self.elevation_gradiant_module(self.elevation_map)
        
        self.occupancy_map[self.elevation_gradiant_map == 1] = 1
        
        # if stairs_mask is not None:  
        #     stairs_mask_tensor = torch.from_numpy(stairs_mask).float().to(self.device)
        #     stairs_mask_tensor = stairs_mask_tensor.unsqueeze(0)
        #     type_prob = [1]
        #     self.stairs_map = self.extra_area_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), self.full_pose, self.stairs_map, stairs_mask_tensor, type_prob)
        #     self.is_stairs_area = self.fill_small_gaps(self.is_stairs_area)
            
        # self.occupancy_map[0,0][self.is_stairs_area == 1] = 1.35
        
        
    # def update_map(self, observations, stairs_mask):
    #     self.full_pose[0] = self.map_size_cm / 100.0 / 2.0+torch.from_numpy(observations['gps']).to(self.device)[0]
    #     self.full_pose[1] = self.map_size_cm / 100.0 / 2.0-torch.from_numpy(observations['gps']).to(self.device)[1]
    #     self.full_pose[2:] = torch.from_numpy(observations['compass'] * 57.29577951308232).to(self.device) # input degrees and meters
    #     self.obstacle_map = self.obstacle_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), self.full_pose, self.obstacle_map)
    #     self.explored_map = self.explored_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), self.full_pose, self.explored_map)
        
    #     if stairs_mask is not None:  
    #         stairs_mask_tensor = torch.from_numpy(stairs_mask).float().to(self.device)
    #         stairs_mask_tensor = stairs_mask_tensor.unsqueeze(0)
    #         type_prob = [1]
    #         self.stairs_map = self.extra_area_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), self.full_pose, self.stairs_map, stairs_mask_tensor, type_prob)
    #         self.is_stairs_area = self.fill_small_gaps(self.is_stairs_area)

        
    #     # self.high_obstacle_map = self.high_obstacle_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), self.full_pose, self.high_obstacle_map)
    #     if self.state_machine.state == 'US':
    #         self.upstairs_free_map = self.upstairs_free_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), self.full_pose, self.upstairs_free_map)
    #         self.upstairs_obstacle_map = self.upstairs_obstacle_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), self.full_pose, torch.zeros_like(self.upstairs_obstacle_map))
    #         self.upstairs_map = self.upstairs_free_map + self.upstairs_obstacle_map

            
    #     self.occupancy_map = self.obstacle_map + self.explored_map
    #     # self.occupancy_map[0,0][(self.is_stairs_area == 1) | (self.is_floor_area == 1)] = 1.35
    #     self.occupancy_map[0,0][self.is_stairs_area == 1] = 1.35
    #     # self.high_occupancy_map = self.high_obstacle_map + self.explored_map
    
    def fill_small_gaps(self, is_area, kernel_size=11):
        img = is_area.float().unsqueeze(0).unsqueeze(0)
        padding = (kernel_size - 1) // 2
        dilated = F.max_pool2d(img, kernel_size=kernel_size, stride=1, padding=padding)
        closed = -F.max_pool2d(-dilated, kernel_size=kernel_size, stride=1, padding=padding)
        result = closed.squeeze().int()
        return result
    
    # def fill_small_gaps(self, is_area, kernel_size=11):
    #     img = is_area.float().unsqueeze(0).unsqueeze(0)
    #     padding = (kernel_size - 1) // 2
    #     dilated = F.max_pool2d(img, kernel_size=kernel_size, stride=1, padding=padding)
    #     closed = -F.max_pool2d(-dilated, kernel_size=kernel_size, stride=1, padding=padding)
    #     result = closed.squeeze().int()
        
    #     mask = result > 0
    #     if not mask.any():
    #         return result, None 
        
    #     coords = torch.nonzero(mask)  
    #     coords_xy = coords[:, [1, 0]].float()  
    #     center = coords_xy.mean(dim=0)         
    #     center = (int(center[0].item()), int(center[1].item()))
    #     return result, center
    
    def merge_pcd(self):
        merged_pcd = merge_pcd_batch(object_list=self.scene_graph.object_node_lst)
        return merged_pcd

    def save_observations(self, observations):
        if "rgb" in observations:
            img = observations["rgb"]
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                os.path.join(self.rgb_dir, f"step_{self.total_steps:04d}.png"),
                img
            )
        if "depth" in observations:
            depth_img = self._process_depth(observations["depth"])
            cv2.imwrite(
                os.path.join(self.depth_dir, f"depth_{self.total_steps:04d}.png"),
                depth_img
            )    
    
    def save_observations_bbox(self, bbox_img):
        cv2.imwrite(os.path.join(self.bbox_dir, f"bbox_{self.total_steps:04d}.jpg"),bbox_img)
    
    # def save_observations_bbox(self, observations, detections):
    #     bbox_lst = []
    #     name_lst = []
    #     score_lst = []
        
    #     if len(detections) > 0:
    #         for idx in range(len(detections.xyxy)):
    #             bbox_lst.append(detections.xyxy[idx])
    #             name_lst.append(self.class_dict[detections.class_id[idx]])
    #             score_lst.append(detections.confidence[idx])
        
    #     if "rgb" in observations:
    #         img = observations["rgb"]
    #         if isinstance(img, torch.Tensor):
    #             img = img.cpu().numpy()
    #         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
    #         for bbox, name, score in zip(bbox_lst, name_lst, score_lst):
                
    #             cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
    #             text = f"{name}: {score:.2f}"
                
                
    #             (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    #             cv2.rectangle(img, (x1, text_y - text_height), (x1 + text_width, text_y), color, -1)
                
    #             cv2.putText(img,
    #                         text,
    #                         (x1, text_y),
    #                         cv2.FONT_HERSHEY_SIMPLEX,
    #                         0.5,
    #                         1,
    #                         cv2.LINE_AA)
            
    #         cv2.imwrite(
    #             os.path.join(self.rgb_dir, f"step_{self.total_steps:04d}.jpg"),
    #             img
    #         )
            
    #     if "depth" in observations:
    #         depth_img = self._process_depth(observations["depth"])
    #         cv2.imwrite(
    #             os.path.join(self.depth_dir, f"depth_{self.total_steps:04d}.png"),
    #             depth_img
    #         )
            
    def draw_final_map(self, center_lst, map, observations, map_name, candidates, candidate_name):
        map_width = 800
        map_data = map[0,0].detach().cpu().numpy() 
        compass = observations.get("compass", [0.0])
            
        pixel_x = round(self.center_pixel + self.gps[0] * 100 / self.resolution)
        pixel_y = round(self.center_pixel - self.gps[1] * 100 / self.resolution)

        color_map = np.zeros((*map_data.shape, 3), dtype=np.uint8)
        free_color = (255, 255, 255)
        obstacle_color = (0, 0, 0)
        unexplored_color = (128, 128, 128)
        stairs_color = (225, 127, 80)
        color_map[map_data == 0.0] = unexplored_color
        color_map[(map_data > 0.0) & (map_data < 2.0) & (map_data != 1.35)] = free_color
        # color_map[(map_data > 0.0) & (map_data < 2.0)] = free_color
        color_map[map_data == 1.35] = stairs_color
        color_map[map_data == 2.0] = obstacle_color  

        if map_name in ['total', 'stairs', 'high', 'gradient']:
            cv2.circle(color_map,
                    (pixel_x, pixel_y),
                    radius=4,
                    color=(255, 0, 255),
                    thickness=-1)
                        
            if self.point_goal is not None:
                cv2.circle(color_map,
                        (round(self.point_goal[0]), round(self.point_goal[1])),
                        radius=4,
                        color=(12, 131, 205),  
                        thickness=-1)
            
            arrow_length = 10
            angle = compass[0]
            dx = np.cos(angle) * arrow_length
            dy = np.sin(angle) * arrow_length  
            end_x = round(pixel_x + dx)
            end_y = round(pixel_y + dy) 
            cv2.arrowedLine(
                color_map,
                (pixel_x, pixel_y), 
                (end_x, end_y),
                color=(128, 255, 0),  
                thickness=1,
                tipLength=0.3
            )
            for i, (cx, cy) in enumerate(candidates):
                cv2.circle(
                    color_map,
                    (round(cx), round(cy)),
                    radius=3,
                    color=(0, 255, 0),
                    thickness=-1
                )
                cv2.putText(
                    color_map,
                    candidate_name[i],
                    (round(cx) + 10, round(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (123, 25, 190),
                    1,
                    cv2.LINE_AA
                )


        if map_name == 'total':
            map_path = os.path.join(self.map_dir, f"map_{map_name}_{self.total_steps:04d}.png")
        elif map_name == 'places':
            map_path = os.path.join(self.place_dir, f"map_{map_name}_{self.total_steps:04d}.png")
        elif map_name == 'origin':
            map_path = os.path.join(self.original_map_dir, f"map_{map_name}_{self.total_steps:04d}.png")
        # elif map_name == 'stairs':
        #     map_path = os.path.join(self.stairs_dir, f"map_{map_name}_{self.total_steps:04d}.png")
        elif map_name == 'gradient':
            map_path = os.path.join(self.gradient_dir, f"map_{map_name}_{self.total_steps:04d}.png")
        cv2.imwrite(map_path, color_map)    
    
    
    # def visualize_elevation_map(self, elevation_map, min_h, max_h):
    #     if not isinstance(elevation_map, torch.Tensor):
    #         raise TypeError("elevation_map must be a PyTorch tensor.")
        
    #     # 1. Prepare data
    #     # Squeeze batch and channel dimensions, move to CPU, convert to numpy
    #     map_data = elevation_map.squeeze().cpu().numpy()
        
    #     # Unobserved areas will have the initial low value. We create a mask for them.
    #     unobserved_value = min_h - 1.0 
    #     observed_mask = map_data > unobserved_value
        
    #     # 2. Create plot
    #     fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        
    #     # Set the background color for unobserved areas
    #     ax.set_facecolor('black')
        
    #     # Display the elevation data using a suitable colormap ('terrain' or 'viridis' are good)
    #     # Only plot the observed data.
    #     # We use vmin and vmax to keep the color scale consistent across different frames.
    #     im = ax.imshow(np.where(observed_mask, map_data, np.nan), cmap='terrain', vmin=min_h, vmax=max_h)
        
    #     # 3. Add colorbar and labels
    #     cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    #     cbar.set_label('Elevation (cm)')
    #     ax.set_title('Top-Down Elevation Map')
    #     ax.set_xlabel('Map X Coordinate')
    #     ax.set_ylabel('Map Y Coordinate')
        
    #     # 4. Save and close
    #     plt.savefig(os.path.join(self.elevation_dir, f"elevation_{self.total_steps}"), bbox_inches='tight')
    #     plt.close(fig)


    
    def visualize_elevation_map(
        self, 
        elevation_map_tensor, 
        min_h=-100, 
        max_h=200, 
        z_resolution=5, 
        min_voxel_idx_offset=-30,
        unobserved_value=-300,
        cmap_name='gray_r',
        bad_value_color='darkcyan'
    ):
        map_numpy = elevation_map_tensor.squeeze().cpu().numpy()

        observed_mask = map_numpy != unobserved_value
        
        height_map_cm = (map_numpy + min_voxel_idx_offset) * z_resolution
        height_map_cm[~observed_mask] = np.nan

        import copy
        cmap = copy.copy(plt.get_cmap(cmap_name)) 
        
        cmap.set_bad(color=bad_value_color)

        fig, ax = plt.subplots(figsize=(10, 10))
        
        im = ax.imshow(
            height_map_cm, 
            cmap=cmap, 
            vmin=min_h, 
            vmax=max_h
        )
        
        ax.set_title("Elevation Map (Best Practice)")
        ax.axis('off')

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Height (cm)')
        
        save_path = os.path.join(self.elevation_dir, f"elevation_{self.total_steps}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
    
    def visualize_gradient_map(
        self, 
        gradient_map_tensor,
        # Sensible defaults for gradient visualization
        max_grad_val=20, # Corresponds to a steep slope. Tune this value.
        unobserved_value=-300,
        cmap_name='inferno', # A perceptually uniform map like 'inferno' or 'viridis' is good
        bad_value_color='darkcyan'
    ):
        map_numpy = gradient_map_tensor.squeeze().cpu().numpy()

        observed_mask = map_numpy != unobserved_value
        
        # The values are already gradient magnitudes (cm/cell), no conversion needed.
        # Set unobserved areas to NaN for plotting.
        gradient_map = np.copy(map_numpy)
        gradient_map[~observed_mask] = np.nan

        import copy
        cmap = copy.copy(plt.get_cmap(cmap_name)) 
        cmap.set_bad(color=bad_value_color)

        fig, ax = plt.subplots(figsize=(10, 10))
        
        im = ax.imshow(
            gradient_map, 
            cmap=cmap, 
            vmin=0, # Gradient magnitude is always non-negative
            vmax=max_grad_val 
        )
        
        ax.set_title("Elevation Gradient Map (Steepness)")
        ax.axis('off')

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Gradient Magnitude (cm height change per grid cell)')
            
        save_path = os.path.join(self.gradient_dir, f"gradient_{self.total_steps}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_l", type=int, default=0)
    parser.add_argument("--split_r", type=int, default=20)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--type", type=str, default="debug")
    parser.add_argument("--dataset", type=str, default="mp3d")
    parser.add_argument("--experiment_name", type=str, default="exp1")
    parser.add_argument("--episode_l", type=int, default=1)
    parser.add_argument("--episode_r", type=int, default=2200)
    args = parser.parse_args()
    
    os.environ["CHALLENGE_CONFIG_FILE"] = f"configs/{args.dataset}_t.yaml"
    config = habitat.get_config(os.environ["CHALLENGE_CONFIG_FILE"])
    
    agent = DiagnosticNavigationAgent(task_config=config, args=args)

    challenge = habitat.Challenge(
        eval_remote=False
    )
    challenge.submit(agent)

if __name__ == "__main__":
    main()

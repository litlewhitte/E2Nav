import argparse
import sys
import os

import habitat
import copy
import json
import cv2
import torch
import traceback
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
import open_clip 
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import Normalize
from openai import OpenAI
import open3d as o3d
from datetime import datetime
from scripts.scenegraph import SceneGraph
from scripts.landmark import LandmarkMap
from scripts.FSM import FSM
from scripts.video_gen import pic_to_video
from PIL import Image, ImageDraw

import hydra
from omegaconf import DictConfig, OmegaConf

from utils.occupacy_mapping import Semantic_Mapping, Elevation_Mapping, ElevationGradientCalculator
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


class LeftTurnAgent():
    def __init__(self, task_config, cfg):
        self._POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS
        self.total_steps = 0
        self.episode_id = 0
        # self.obj_goal = self.simulator._env.current_episode.object_category
        self.obj_goal = None
        
        # 从配置文件加载参数
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
        
        # 是否进行2D分割检测
        self.detection_segmentation = cfg.perception.detection_segmentation
        self.curr_detection_nodes = []
        
        # 初始化存储路径
        if self.type == 'debug':
            self.save_dir = f"{cfg.paths.save_dir_base}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.split_l}_{self.split_r}"
        else:
            self.save_dir = f"{cfg.paths.exp_dir_base}/{self.dataset}/{self.experiment_name}/observations/{self.dataset}_{self.split_l}_to_{self.split_r}"
        os.makedirs(self.save_dir, exist_ok=True)
    
        # 地图初始化
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

        # 语义映射模块初始化
        self.explored_module = Semantic_Mapping(
            self, 
            max_height=cfg.semantic_mapping.explored.max_height, 
            min_height=cfg.semantic_mapping.explored.min_height, 
            pcd_thereshold=cfg.semantic_mapping.explored.pcd_threshold
        ).to(self.device)
        
        self.obstacle_module = Semantic_Mapping(
            self, 
            max_height=cfg.semantic_mapping.obstacle.max_height, 
            min_height=cfg.semantic_mapping.obstacle.min_height, 
            pcd_thereshold=cfg.semantic_mapping.obstacle.pcd_threshold
        ).to(self.device)
        
        self.high_obstacle_module = Semantic_Mapping(
            self, 
            max_height=cfg.semantic_mapping.high_obstacle.max_height,
            min_height=cfg.semantic_mapping.high_obstacle.min_height
        ).to(self.device)
        
        self.extra_area_module = Semantic_Mapping(
            self, 
            max_height=cfg.semantic_mapping.extra_area.max_height, 
            min_height=cfg.semantic_mapping.extra_area.min_height, 
            pcd_thereshold=cfg.semantic_mapping.extra_area.pcd_threshold
        ).to(self.device)
        
        self.elevation_module = Elevation_Mapping(
            self, 
            max_height=self.max_height, 
            min_height=self.min_height, 
            slice_point_threshold=cfg.semantic_mapping.elevation.slice_point_threshold
        ).to(self.device)
        
        self.elevation_gradiant_module = ElevationGradientCalculator(
            self, 
            gradient_threshold=cfg.semantic_mapping.elevation_gradient.gradient_threshold
        )
        
        self.upstairs_obstacle_module = Semantic_Mapping(
            self, 
            max_height=cfg.semantic_mapping.upstairs_obstacle.max_height,
            min_height=cfg.semantic_mapping.upstairs_obstacle.min_height
        ).to(self.device)
        
        self.upstairs_free_module = Semantic_Mapping(
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
        # 感知初始化
        self.observations = None
        # 动作初始化
        self.action = None
        self.action_cnt = cfg.robot.action_cnt
        self.robot_radius = cfg.robot.radius
        # 俯仰角初始化
        self.pitch = cfg.perception.pitch
        
        # 检测过滤参数初始化
        self.mask_area_threshold = cfg.detection_filter.mask_area_threshold
        self.conf_threshold = cfg.detection_filter.conf_threshold
        self.max_bbox_area_ratio = cfg.detection_filter.max_bbox_area_ratio
        
        # 相似度计算参数初始化
        self.overlap_weight = cfg.similarity.overlap_weight
        self.visual_sim_weight = cfg.similarity.visual_sim_weight
        self.max_match_distance = cfg.similarity.max_match_distance
        self.distance_threshold = cfg.similarity.distance_threshold
        self.center_distance_decay = cfg.similarity.center_distance_decay
        self.distance_penalty_strength = cfg.similarity.distance_penalty_strength
        self.min_similarity_thresh = cfg.similarity.min_similarity_thresh
        self.containment_visual_sim_threshold = cfg.similarity.containment_visual_sim_threshold
        
        # 导航参数初始化
        self.navigation_distance_threshold = cfg.actions.navigation_distance_threshold
        
        
        # 点云初始化
        self.merged_pcd = o3d.geometry.PointCloud()
        
        # 初始化目标点
        self.point_goal = tuple(cfg.navigation.point_goal)
        self.temp_path = []
        
        self.next_goal = tuple(cfg.navigation.next_goal)
        # 初始化vocabulary
        with open(cfg.paths.classes_txt, 'r') as file:
            self.vocabulary = [line.rstrip('\n') for line in file]
        with open(cfg.paths.classes_json) as file:
            self.class_description = json.load(file)

        
        # 初始化状态
        self.state = cfg.navigation.state   # Broad Search
        
        # 房间清单
        self.optional_room_list = cfg.rooms.optional_room_list
        
        
        # """
        # Grounded-SAM
        # """
        # self.config_file = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouqihang-240108100050/HuNav/models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
        # self.grounding_dino_checkpoint = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouqihang-240108100050/HuNav/models/checkpoints/groundingdino_swint_ogc.pth'
        # self.ram_checkpoint = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouqihang-240108100050/HuNav/models/checkpoints/ram_swin_large_14m.pth'
        # self.sam_checkpoint = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouqihang-240108100050/HuNav/models/checkpoints/sam_vit_h_4b8939.pth'
        # self.grounding_dino_model = Model(model_config_path=self.config_file, model_checkpoint_path=self.grounding_dino_checkpoint)
        # self.ram_model = ram(pretrained=self.ram_checkpoint,
        #                                 image_size=384,
        #                                 vit='swin_l').to(self.device)
        # self.sam_predictor = SamPredictor(build_sam(checkpoint=self.sam_checkpoint).to(self.device))
        
        """
        YOLO
        """
        # self.yolo_checkpoint = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouqihang-240108100050/HuNav/models/checkpoints/yolov8x-world.pt'
        # self.sam_checkpoint = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouqihang-240108100050/HuNav/models/checkpoints/mobile_sam.pt'
        
        # self.detection_model = YOLO(self.yolo_checkpoint)
        # self.detection_model.set_classes(self.vocabulary)
        # self.segmentation_model = SAM(self.sam_checkpoint)
        # self.class_dict = self.detection_model.names
        # assert "stairs" in self.class_dict.values()
        # self.stairs_idx = [k for k, v in self.class_dict.items() if v == "stairs"]
        
        
        # self.vision_model = AutoModel.from_pretrained("google/siglip2-base-patch16-224", cache_dir="models/siglip2", device_map="auto").eval()
        # self.vision_processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224", cache_dir="models/siglip2")
        # self.pcd_detect_tokenizer = AutoTokenizer.from_pretrained("/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouqihang-240108100050/HuNav/models/SpatialLM-Llama-1B")
        # self.pcd_detect_model = AutoModelForCausalLM.from_pretrained("/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouqihang-240108100050/HuNav/models/SpatialLM-Llama-1B")

        # # qwen VLM导入
        # self.vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct", torch_dtype="auto", device_map="auto", cache_dir="models/qwenVL_72B")
        # self.vlm_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct", cache_dir="models/qwenVL_72B")
        
        # # qwen LLM导入
        # self.llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouqihang-240108100050/HuNav/models/qwen_72B", torch_dtype="auto", device_map="auto")
        # self.llm_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouqihang-240108100050/HuNav/models/qwen_72B")
        self.llm_model = None
        self.llm_tokenizer = None
        
        # self.openai_api_key = "EMPTY"
        # self.openai_api_base = "http://localhost:8000/v1"
        # self.client = OpenAI(
        #                 api_key=self.openai_api_key,
        #                 base_url=self.openai_api_base,
        #             )
        
        # 客户端配置
        self.client = cfg.client.url
        
        # CLIP模型初始化（用于真正的视觉特征提取）
        self.use_clip_feature = getattr(cfg, 'use_clip_feature', False)
        if self.use_clip_feature:
            clip_cache_dir = os.path.join(os.getcwd(), "models", "checkpoints", "clip")
            os.makedirs(clip_cache_dir, exist_ok=True)
            
            # 设置环境变量，让torch使用自定义缓存路径
            os.environ['TORCH_HOME'] = os.path.join(os.getcwd(), "models", "checkpoints")
            
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                "ViT-H-14", 
                pretrained="laion2b_s32b_b79k",
                cache_dir=clip_cache_dir  # 指定缓存目录
            )
            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.eval()
        else:
            self.clip_model = None
            self.clip_preprocess = None
        
        # 初始化场景描述器
        self.scenegraph = SceneGraph(agent=self)
        self.landmarkmap = LandmarkMap(agent=self)
        self.FSM = FSM(agent=self)  
        
        
        
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
        self.category_revision()
        self.obj_goal = 'television'
        self.obj_goal_id = self.vocabulary.index(self.obj_goal)
        # # self.obj_goal = 'weeder'

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
            self.room_rank = room_rank(self.client, self.optional_room_list, self.obj_goal)
            self.curr_detection_nodes = []
            self.metrics = {'distance_to_goal': 0., 'spl': 0., 'softspl': 0.}
            self.action_cnt = 0
            self.scenegraph.reset()
            self.landmarkmap.reset()
            self.FSM.reset(obj_goal=self.obj_goal, room_rank=self.room_rank)
            # if self.episode_id > 5:
            #     raise StopIteration("这里其实不是报错了，只是因为我不知道怎么让程序停下来")
            # 创建episode目录结构
            print(f"============================episode_{self.episode_id}============================")
            print(f"my object goal is {self.obj_goal} and the room rank is {self.room_rank}")
            self.episode_dir = os.path.join(self.save_dir, f"episode_{self.episode_id}")
            self.pcd_dir = os.path.join(self.episode_dir, "pcd")
            self.map_dir = os.path.join(self.episode_dir, "maps")
            self.original_map_dir = os.path.join(self.episode_dir, "origin")
            self.place_dir = os.path.join(self.episode_dir, "places")
            self.skeleton_dir = os.path.join(self.episode_dir, "skeleton")
            self.elevation_dir = os.path.join(self.episode_dir, "elevation")
            self.gradient_dir = os.path.join(self.episode_dir, "gradient")
            self.rgb_dir = os.path.join(self.episode_dir, "rgb")
            self.depth_dir = os.path.join(self.episode_dir, "depth")
            self.bbox_dir = os.path.join(self.episode_dir, "bbox")
            
            os.makedirs(self.episode_dir, exist_ok=True)
            os.makedirs(self.pcd_dir, exist_ok=True)
            os.makedirs(self.map_dir, exist_ok=True)
            os.makedirs(self.rgb_dir, exist_ok=True)
            os.makedirs(self.bbox_dir, exist_ok=True)
            
            if self.type == "debug":
                os.makedirs(self.original_map_dir, exist_ok=True)
                os.makedirs(self.skeleton_dir, exist_ok=True)
                os.makedirs(self.place_dir, exist_ok=True)
                os.makedirs(self.elevation_dir, exist_ok=True)
                os.makedirs(self.gradient_dir, exist_ok=True)
                os.makedirs(self.depth_dir, exist_ok=True)
                        
            
            
            # 初始化pose文件
            self.pose_file = open(os.path.join(self.episode_dir, "pose.txt"), "w")
            self.pose_file.write("step\tx\ty\tz\tcompass\tpitch\n")
            
            # 初始化地图
            self.init_map()
            self.point_goal = (self.center_pixel, self.center_pixel)
            self.temp_path = []
            # 动作初始化
            self.action = None
            # 初始化俯仰角
            self.pitch = 0.0
            
            # 初始化状态
            self.state = 'BS'
        
    def _save_pose(self, observations):
        """保存位姿信息（兼容2D/3D GPS数据）"""
        try:
            gps = observations.get("gps", np.zeros(2))  # 默认二维
            compass = observations.get("compass", [0.0])
            
            # 解析坐标
            x = gps[0] if len(gps) > 0 else 0.0
            y = gps[1] if len(gps) > 1 else 0.0
            z = gps[2] if len(gps) > 2 else 0.0  # 处理可能的3D数据
            
            # 写入文件
            pose_str = "{:04d}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\n".format(
                self.total_steps, x, y, z, compass[0], self.pitch
            )
            self.pose_file.write(pose_str)
        except Exception as e:
            print(f"保存位姿失败: {str(e)}\n原始数据: gps={gps}, compass={compass}, pitch={self.pitch}")
    
    def get_pose(self, observations):
        if len(observations['gps']) == 2:
            self.gps[0] = observations['gps'][0]
            self.gps[1] = observations['gps'][1]
        else:
            self.gps[0] = - observations['gps'][2]
            self.gps[1] = observations['gps'][0]
            self.gps[2] = observations['gps'][1]        

    def _process_depth(self, depth_obs):
        """处理深度传感器数据"""
        if isinstance(depth_obs, torch.Tensor):
            depth_obs = depth_obs.cpu().numpy()
        
        min_depth = 0.0  # 从配置中读取
        max_depth = 5.0  # 从配置中读取
        depth_normalized = np.clip((depth_obs - min_depth)/(max_depth - min_depth), 0, 1)
        return (depth_normalized * 65535).astype(np.uint16)
    
    def category_revision(self):
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
        if self.total_steps >= 500:  # 从配置中读取
            return {"action": 0}
        
        if self.total_steps == 499:
            print("[DEBUG] 499 steps fail case")
            print(f"object goal is {self.obj_goal}, collision set is {self.landmarkmap.path_planner.collision_offsets}")
            print(f"temporarily_blocked_targets is {self.FSM.temporarily_blocked_targets}, permanently_unreachable_targets is {self.FSM.permanently_unreachable_targets}, confirmed_wrong_target is {self.FSM.confirmed_wrong_list}")
            print("[DEBUG] check wrong object class")
            for obj_id in self.FSM.temporarily_blocked_targets + self.FSM.permanently_unreachable_targets + self.FSM.confirmed_wrong_list:
                try:
                    o3d.io.write_point_cloud(os.path.join(self.pcd_dir, f"wrong_obj{obj.id}_class_{obj.class_name}.ply"), obj.pcd)
                except:
                    print("can't write pcd!")
                
                try:
                    obj = self.scenegraph.object_node_lst[obj_id]
                    print(f"The object {obj_id} is {obj.class_name}")
                    if obj.class_name in self.vocabulary:
                        print(f"Its probs for {obj.class_name} is {obj.class_candidates[self.vocabulary.index(obj.class_name)]} and for object goal is {obj.class_candidates[self.vocabulary.index(obj.class_name)]}")
                        print(f"Its entropy is {obj._entropy_gen()}")
                except:
                    print("can't write pcd!")
        
        # if self.type == 'debug':
        #     if self.total_steps == 52:
        #         for obj in self.scenegraph.object_node_lst:
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

        # 感知部分
        observations["depth"][observations["depth"]==0.5] = 100 # don't construct unprecise map with distance less than 0.5 m

        self.observations = observations
        self.get_pose(observations)
        self.pixel_x = round(self.center_pixel + self.gps[0] * 100 / self.resolution)
        self.pixel_y = round(self.center_pixel - self.gps[1] * 100 / self.resolution)
        
        # 地图可视化
        center_lst = []
        draw_room_node = []
        room_node_lst = []
        # for i, object_node in enumerate(self.scenegraph.object_node_lst):
        #     class_name = object_node.class_name
        #     center = object_node.center
        #     center_lst.append((center, class_name))
        stairs_mask = None
        
        if self.total_steps > 16:   # 前16步是地图初始化阶段
            start_time = time.time()
            # Grounded-SAM版的realtime_mapping
            if 'CE' in self.FSM.state or self.FSM.state == 'AP':
                bbox_img = cv2.cvtColor(observations["rgb"], cv2.COLOR_RGB2BGR)
                self.curr_detection_nodes = []
            else:
                detections, _, bbox_img, probs = self.scenegraph.detection_utils()
                self.curr_detection_nodes, stairs_mask = self.scenegraph.realtime_mapping(detections, probs)
                self.FSM.mandatory_detection = False
                # 检查被遮挡的目标是否重新变得可达
                # print("[DEBUG] blocked target:", self.FSM.temporarily_blocked_targets)
                self.FSM.check_blocked_targets_recovery(self.scenegraph.object_node_lst)
    
            print(f"TIME for scenegraph generation:{time.time()-start_time}")
            
            start_time = time.time()
            self.FSM.state_transition()
            print(f"TIME for FSM:{time.time()-start_time}")
                
            self.save_observations_bbox(bbox_img)
            
        self.save_observations(observations)
        self._save_pose(observations)
        
        start_time = time.time()
        self.update_map(observations, stairs_mask) 
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
        
        
        # 找到物品
        if self.FSM.state == 'ST': 
            self.draw_final_map([], self.occupancy_map, observations, "total", [], [])
            print(f"find!!! x={self.pixel_x}, y={self.pixel_y} and object goal is {self.obj_goal}")
            
            try:
                o3d.io.write_point_cloud(os.path.join(self.pcd_dir, f"episode_{self.episode_id}_class_{self.obj_goal}.ply"), self.FSM.confirmed_essential_obj.pcd)
                if self.type == 'debug':
                    for obj in self.scenegraph.object_node_lst:
                        o3d.io.write_point_cloud(os.path.join(self.pcd_dir, f"episode_{self.episode_id}_id_{obj.id}_class_{obj.class_name}.ply"), obj.pcd)
                    # for i, picture in enumerate(self.FSM.confirmed_bbox_dict[self.FSM.confirmed_essential_obj.id]):
                    #     cv2.imwrite(os.path.join(self.pcd_dir, f"step_{self.total_steps}_id_{self.FSM.confirmed_essential_obj.id}_picture_{i}.png"), picture)
            
            
            except:
                print("can't save point cloud! ")

            # for i, pic in enumerate(self.FSM.confirmed_bbox_dict[self.FSM.confirmed_essential_obj.id]):
            #     pic = cv2.cvtColor(pic, cv2.COLOR_RGB2BGR)
            #     cv2.imwrite(os.path.join(self.pcd_dir, f"pic_{i}.png"), pic)
            return {"action": 0}
       
        if self.total_steps < 17:
            self.draw_final_map([], self.occupancy_map, observations, "total", [], [])
            # self.draw_final_map([], self.elevation_gradiant_map, observations, "gradient", [], [])
            # self.visualize_elevation_map(self.elevation_map, min_h=self.min_height, max_h=self.max_height)

        # 运动部分
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
            # if self.FSM.state != 'CE':
            # if self.FSM.confirmed_essential_obj is not None:
            #     self.zqhid = self.FSM.confirmed_essential_obj.id
            #     o3d.io.write_point_cloud(os.path.join(self.pcd_dir, f"step_{self.total_steps}_id_{self.FSM.confirmed_essential_obj.id}.ply"), self.FSM.confirmed_essential_obj.pcd)
            # else:
            #     for obj in self.scenegraph.object_node_lst:
            #         try:
            #             if obj.id == self.zqhid:
            #                 o3d.io.write_point_cloud(os.path.join(self.pcd_dir, f"step_{self.total_steps}_id_{self.FSM.confirmed_essential_obj.id}.ply"), obj.pcd)
            #         except:
            #             pass
            start_time = time.time()
            self.action, self.point_goal, self.temp_path, vis_img = self.landmarkmap.local_policy()
            # if self.total_steps > 87 and self.total_steps < 95:
            #     print(self.landmarkmap.path_planner.collision_offsets)
            #     print(self.temp_path)
            print(f"TIME for local policy:{time.time()-start_time}")

            start_time = time.time()
            
            for room_node in self.landmarkmap.room_lst:
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
                     
            
            # 可视化路径
            savepath = os.path.join(self.map_dir, f'map_total_{self.total_steps:04d}.png')
            self.visualize_path(savepath, self.pixel_x, self.pixel_y, savepath)
            
            
        
            print(f"check state:{self.FSM.state}")
            print("-------------Object Node Begin-------------")
            print([(obj.class_name, obj.id, obj.score) for obj in self.scenegraph.object_node_lst])
            print("--------------Object Node End--------------")
            print("**************FSM Test Begin**************")
            if self.FSM.approaching_essential_obj is not None:
                print(f"essential_obj_center is {self.FSM.approaching_essential_obj.center[0] * self.r_resolution}, {self.FSM.approaching_essential_obj.center[1] * self.r_resolution}")
            if self.FSM.confirmed_essential_obj is not None:
                print(f"confirmed_obj_center is {self.FSM.confirmed_essential_obj.center[0] * self.r_resolution}, {self.FSM.confirmed_essential_obj.center[1] * self.r_resolution}")
                print(f"confirmed obj_id is {self.FSM.confirmed_essential_obj.id}, and its score dict is {self.FSM.confirmed_essential_obj.class_candidates}")
            print(f"confirmed_object_list is {self.FSM.confirmed_object_list}")
            print("***************FSM Test End***************")
            print("***************Locomotion Test Begin***************")
            print(f"agent pos is {self.pixel_x, self.pixel_y}")
            if self.landmarkmap.point_goal is not None:
                print(f"point goal is {self.landmarkmap.point_goal}")
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
    
    def visualize_path(self, image_path, start, end, save_path):
        if self.temp_path != None:
            img = Image.open(image_path)

            draw = ImageDraw.Draw(img)
            # 绘制路径（使用抗锯齿的红色线条）
            if len(self.temp_path) > 1:
                draw.line(self.temp_path, fill=(255,0,0), width=3)
            
            img.save(save_path)
            print(f"结果已保存至 {save_path}")    
          
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
        self.obstacle_map = self.obstacle_module(torch.squeeze(valid_depth_data, dim=-1).to(self.device), self.full_pose, self.agent_z_pos, self.obstacle_map)
        self.explored_map = self.explored_module(torch.squeeze(valid_depth_data, dim=-1).to(self.device), self.full_pose, self.agent_z_pos, self.explored_map)
        
        # self.obstacle_map[(self.occupancy_map > 0) & (self.occupancy_map < 2)] = 0
          
        self.occupancy_map = self.obstacle_map + self.explored_map
        self.original_occupancy_map = self.obstacle_map + self.explored_map
        
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
        merged_pcd = merge_pcd_batch(object_list=self.scenegraph.object_node_lst)
        return merged_pcd

    def save_observations(self, observations):
        # 保存RGB图像
        if "rgb" in observations:
            img = observations["rgb"]
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                os.path.join(self.rgb_dir, f"step_{self.total_steps:04d}.png"),
                img
            )
        # 保存深度图像
        if self.type == 'debug':
            if "depth" in observations:
                depth_img = self._process_depth(observations["depth"])
                cv2.imwrite(
                    os.path.join(self.depth_dir, f"depth_{self.total_steps:04d}.png"),
                    depth_img
                )    
    
    def save_observations_bbox(self, bbox_img):
        cv2.imwrite(os.path.join(self.bbox_dir, f"bbox_{self.total_steps:04d}.jpg"),bbox_img)
        
    def draw_final_map(self, center_lst, map, observations, map_name, candidates, candidate_name):
        # 1. 获取地图数据
        map_width = 800
        map_data = map[0,0].detach().cpu().numpy() 
        compass = observations.get("compass", [0.0])
            
        pixel_x = round(self.center_pixel + self.gps[0] * 100 / self.resolution)
        pixel_y = round(self.center_pixel - self.gps[1] * 100 / self.resolution)

        # 2. 创建彩色地图画布
        # 将二值图转为三通道并转换为BGR颜色空间
        color_map = np.zeros((*map_data.shape, 3), dtype=np.uint8)
        free_color = (255, 255, 255)   # 自由空间：白色
        obstacle_color = (0, 0, 0)   # 障碍物：黑色
        unexplored_color = (128, 128, 128)  # 未探索：灰色
        stairs_color = (225, 127, 80)
        # 阈值定义（根据实际数据调整）
        color_map[map_data == 0.0] = unexplored_color
        color_map[(map_data > 0.0) & (map_data < 2.0) & (map_data != 1.35)] = free_color
        # color_map[(map_data > 0.0) & (map_data < 2.0)] = free_color
        color_map[map_data == 1.35] = stairs_color
        color_map[map_data == 2.0] = obstacle_color  

        if map_name in ['total', 'stairs', 'high', 'gradient']:
            # 绘制实心圆点
            cv2.circle(color_map,
                    (pixel_x, pixel_y),
                    radius=4,
                    color=(255, 0, 255),  # 粉色填充
                    thickness=-1)
            # agent_pos = self.simulator._env._task.measurements.measures['distance_to_goal']._sim.get_agent_state().position
            # cv2.circle(color_map,
            #         (300+int(20*agent_pos[2]), 300+int(20*agent_pos[0])),
            #         radius=5,
            #         color=(0, 0, 0),  # 粉色填充
            #         thickness=-1)
                        
            # 绘制当前的长程目标点
            if self.point_goal is not None:
                cv2.circle(color_map,
                        (round(self.point_goal[0]), round(self.point_goal[1])),
                        radius=4,
                        color=(12, 131, 205),  
                        thickness=-1)
            
            # 绘制方向箭头
            arrow_length = 10  # 箭头长度（像素）
            angle = compass[0]  # 当前朝向角度（弧度）
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
                    color=(0, 255, 0),  # BGR格式的绿色
                    thickness=-1
                )
                cv2.putText(
                    color_map,
                    candidate_name[i],  # 显示坐标值，保留1位小数
                    (round(cx) + 10, round(cy) - 10),  # 文字位置（右侧偏移）
                    cv2.FONT_HERSHEY_SIMPLEX,  # 字体类型
                    0.5,                       # 字体大小
                    (123, 25, 190),            # BGR绿色文字
                    1,                         # 线宽
                    cv2.LINE_AA                # 抗锯齿渲染
                )


        # 6. 保存地图
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
        if self.total_steps == 17:
            self.draw_points_on_image(map_path, self.episode_view_points, map_path)  


    def draw_points_on_image(self, image_path, points_list, output_path, point_color=(255, 0, 0), point_radius=2, point_thickness=-1):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        for point in points_list:
            center_x = int(round(point[0]))
            center_y = int(round(point[1]))

            height, width = img.shape[:2]
            cv2.circle(img, (center_x, center_y), point_radius, point_color, point_thickness)
        success = cv2.imwrite(output_path, img)
        return success
    
    def visualize_elevation_map(
        self, 
        elevation_map_tensor, 
        min_h=-100, 
        max_h=200, 
        z_resolution=5, 
        min_voxel_idx_offset=-30,
        unobserved_value=-300,
        cmap_name='gray_r',
        bad_value_color='darkcyan'  # 为未探索区域指定一个独特的颜色
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



@hydra.main(version_base=None, config_path="../configs", config_name="e2")
def main(cfg: DictConfig) -> None:
    # 设置habitat配置文件
    os.environ["CHALLENGE_CONFIG_FILE"] = f"configs/{cfg.experiment.dataset}.yaml"
    config = habitat.get_config(os.environ["CHALLENGE_CONFIG_FILE"])
    
    agent = LeftTurnAgent(task_config=config, cfg=cfg)
    challenge = habitat.Challenge(
        eval_remote=False,
        split_l=cfg.experiment.split_l,
        split_r=cfg.experiment.split_r
    )
    challenge.submit(agent)

if __name__ == "__main__":
    main()
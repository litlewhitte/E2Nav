import math
import cv2
import torch
import supervision as sv
from ultralytics import YOLO, SAM
import time
import os
import re
import open3d as o3d
import supervision as sv
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

# from HuNav.models.detection import yolo_detection
# from models.segmentation import sam_segmentation

from utils.realtime_mapping import get_aggregate_feature
from utils.utils import compute_overlap_matrix_2set, merge_obj2_into_obj1, merge_point_clouds, volume_ratio_voxel_gen, compute_color_similarity_matrix, compute_visual_similarity_matrix, compute_center_distance_similarity, compute_min_point_distance_similarity, compute_aggregate_similarities, filter_contained_pointclouds
from utils.GroundedSAM_utils import detection_segmentation
from utils.scene_filter import object_filter, close_vocabulary_filter
from utils.vlm_utils import object_confirmed, object_comparison, stairs_confirmed
# from utils.SpatialLM_utils import SpatialLM_detection



class ObjectNode():
    def __init__(self, agent):
        self.id = None
        self.pcd = None
        self.class_name = None
        self.agent = agent
        # self.class_candidates = dict()
        self.class_candidates = torch.zeros(len(self.agent.vocabulary))
        self.last_post_probs = torch.zeros(len(self.agent.vocabulary))
        self.kl_divergence = None
        self.class_id = None
        self.siglip_embedding = None
        self.clip_embedding = None  # CLIP图像特征（真正的视觉特征）
        self.center = None
        self.place = None
        self.score = 0.0
        self.edge = []
        # 主动感知标志
        self.need_active_perception = False
    
    def _init_class(self, class_name, class_id, probs):
        self.last_post_probs = torch.zeros(len(self.agent.vocabulary))
        self.class_candidates = torch.tensor(probs)
        self.class_name = class_name
        self.class_id = class_id
        self.score = max(self.class_candidates)
    
    def _bayes_update(self, probs):
        obs = torch.tensor(probs)
        evidence = obs @ self.class_candidates
        post_probs = (obs * self.class_candidates) / evidence
        self._KL_divergence_gen(self.class_candidates, post_probs)
        self.last_post_probs = self.class_candidates  # 保留一下上次的probs
        self.class_candidates = post_probs
        self.class_id = torch.argmax(self.class_candidates)
        self.class_name = self.agent.vocabulary[self.class_id]
        self.score = self.class_candidates[self.class_id]
    
    # def _bayes_update(self, probs):
    #     """基于KL散度和观测熵的贝叶斯更新方法"""
    #     obs = torch.tensor(probs)
    #     prior = self.class_candidates
        
    #     # 计算KL散度：差异越大，越需要谨慎对待新观测
    #     kl_div = torch.sum(obs * torch.log((obs + 1e-8) / (prior + 1e-8)))
        
    #     # 计算新观测的熵：熵越高，说明新观测更有价值（信息量更大）
    #     obs_entropy = -torch.sum(obs * torch.log(obs + 1e-8))
    #     max_entropy = torch.log(torch.tensor(len(obs), dtype=torch.float32))  # 最大可能熵
    #     normalized_entropy = obs_entropy / max_entropy  # 归一化熵 [0,1]
        
    #     # 计算综合权重：
    #     # - KL散度越高，权重越低（更谨慎）
    #     # - 熵越高，权重越高（更有价值）
    #     # 使用KL散度的倒数来降低权重，用熵来提升权重
    #     kl_weight = torch.exp(-kl_div * 2.0)  # KL散度越高，权重越低
    #     entropy_weight = normalized_entropy  # 熵越高，权重越高
        
    #     # 综合权重：平衡KL散度和熵的影响
    #     weight = kl_weight * entropy_weight * 0.8  # 可调参数：整体权重系数
        
    #     # 限制权重范围
    #     weight = torch.clamp(weight, 0.05, 0.7)  # 可调参数：最小和最大权重
        
    #     # 加权更新
    #     post_probs = weight * obs + (1 - weight) * prior
        
    #     # 归一化
    #     post_probs = post_probs / torch.sum(post_probs)
        
    #     # 更新KL散度
    #     self._KL_divergence_gen(self.class_candidates, post_probs)
    #     self.last_post_probs = self.class_candidates  # 保留一下上次的probs
    #     self.class_candidates = post_probs
    #     self.class_id = torch.argmax(self.class_candidates)
    #     self.class_name = self.agent.vocabulary[self.class_id]
    #     self.score = self.class_candidates[self.class_id]
    
    def _rollback_class(self):      # 当vlm double check失败的时候，回退class candidates
        self.class_candidates = self.last_post_probs    
        self.class_id = torch.argmax(self.class_candidates)
        self.class_name = self.agent.vocabulary[self.class_id]
        self.score = self.class_candidates[self.class_id]
    
    def _KL_divergence_gen(self, prior_probs, post_probs):
        self.kl_divergence = torch.sum(post_probs * (torch.log(post_probs) - torch.log(prior_probs))).item()
    
    def _entropy_gen(self):
        entropy = -torch.sum(self.class_candidates * torch.log(self.class_candidates + 1e-6))
        return entropy
        
    def class_update(self, class_name, probs, bbox):
        class_id = torch.argmax(torch.tensor(probs))
        class_name = self.agent.vocabulary[class_id]
        if class_name not in self.agent.vocabulary:
            return
        else:
            class_id = self.agent.vocabulary.index(class_name)
            if max(self.class_candidates).item() == 0.0:    
                self._init_class(class_name=class_name, class_id=class_id, probs=probs)
            else:
                self._bayes_update(probs=probs)
    
    def topk_class_update(self, class_id):
        assert max(self.class_candidates).item() > 0
        vocab_size = len(self.agent.vocabulary)
        goal_prob = 1.0 / (1.0 + 0.6 * (vocab_size - 1))
        non_goal_prob = 0.6 * goal_prob
        topk_probs = torch.ones(vocab_size) * non_goal_prob
        topk_probs[class_id] = goal_prob
        self._bayes_update(probs=topk_probs)

        
                
    
class SceneGraph():
    def __init__(self, agent):   # 每一个场景中会进行一个scenegraph update
        self.agent = agent
        self.map_size_cm = self.agent.map_size_cm
        self.confirmed_observations = {"newly_detection": False, "confirmed_observations": [], "score": [], "bbox": [], "probs": []}
        self.final_confirmed_observatinos = {}
        self.object_node_lst = []
        self.bbox_lst = []
        self.place_node_lst = []
        
    
    def reset(self):
        self.confirmed_observations = {"newly_detection": False, "confirmed_observations": [], "score": [], "bbox": [], "probs": []}
        self.final_confirmed_observatinos = {}
        self.object_node_lst = []
        self.bbox_lst = []
        self.place_node_lst = []

    def get_extrinsic_matrix(self):
        # 获取传感器数据
        x = self.map_size_cm / 100.0 / 2.0 + self.agent.gps[0]
        y = self.map_size_cm / 100.0 / 2.0 - self.agent.gps[1]
        z = 0.88
        yaw = (self.observations['compass'] - np.pi / 2)[0]
        pitch = self.pitch

        # 绕相机X轴的俯仰
        cos_p, sin_p = np.cos(pitch), np.sin(pitch)
        Rx = np.array([
            [1, 0,      0     ],
            [0, cos_p, -sin_p],
            [0, sin_p,  cos_p]
        ])

        # 绕世界Z轴的偏航
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        Rz = np.array([
            [cos_y, -sin_y, 0],
            [sin_y,  cos_y, 0],
            [0,      0,     1]
        ])

        # 相机坐标系 → 世界坐标系的轴变换
        R_cam_to_world = np.array([
            [1, 0,  0],
            [0, 0,  1],  # Y_cam → -Z_world
            [0, -1,  0]   # Z_cam → Y_world
        ])

        # 组合旋转：Rz @ R_cam_to_world @ Rx
        R = Rz @ R_cam_to_world @ Rx

        # 构造外参矩阵
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = R
        pose_matrix[:3, 3] = [x, y, z]
        return pose_matrix

    def get_bbox_observations(self, bbox, box_color=(0, 255, 0), thickness=2):
        bgr_image = cv2.cvtColor(self.observations["rgb"], cv2.COLOR_RGB2BGR)
        H, W = bgr_image.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        img_copy = np.copy(bgr_image)
        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(x1 + 1, min(x2, W))
        y2 = max(y1 + 1, min(y2, H))
        cv2.rectangle(
            img=img_copy,
            pt1=(x1, y1),
            pt2=(x2, y2),
            color=box_color,  
            thickness=thickness
        )
        bbox_observations = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        return bbox_observations
    
    
    
    def pcd_filter(self, feature_list):
        for i in reversed(range(len(feature_list))):
            obj_feature = feature_list[i]
            if obj_feature['class_name'] in ['door', 'stairs']:
                del feature_list[i]
            else:
                obj_center = obj_feature['center']
                if obj_center[2] < 0.1:
                    del feature_list[i]
        return feature_list

    
    def detection_utils(self):
        if self.agent.FSM.mandatory_detection:
            vocabulay = [self.agent.obj_goal]
            detections, bbox_img, mask_img, probs = detection_segmentation(
                observations=self.agent.observations,
                vocabulary=vocabulay,
                port=self.agent.port,
                use_ram=False,
                box_threshold=0.35,  
                text_threshold=0.35, 
                iou_threshold=0.35
            )
            if len(probs) > 0:
                vocab_size = len(self.agent.vocabulary)
                goal_prob = 1.0 / (1.0 + 0.6 * (vocab_size - 1))
                non_goal_prob = 0.6 * goal_prob
                mandatory_probs = torch.ones(vocab_size) * non_goal_prob
                goal_index = self.agent.vocabulary.index(self.agent.obj_goal)
                mandatory_probs[goal_index] = goal_prob
                for i in range(len(probs)):
                    probs[i] = mandatory_probs
                    
            return detections, bbox_img, mask_img, probs
        else:
            vocabulay = self.agent.vocabulary
            detections, bbox_img, mask_img, probs = detection_segmentation(
                observations=self.agent.observations,
                vocabulary=vocabulay,
                port=self.agent.port,
                use_ram=False,
                box_threshold=0.35,  
                text_threshold=0.35, 
                iou_threshold=0.35
            )        
            
            return detections, bbox_img, mask_img, probs
    
    def filter_detections(self, detections, probs, mask_area_threshold=25, conf_threshold=0.35, max_bbox_area_ratio=0.7):
        """
        在检测阶段过滤掉低质量检测
        
        Args:
            detections: 检测结果
            probs: 概率分布
            mask_area_threshold: 掩码面积阈值（像素）
            conf_threshold: 置信度阈值
            max_bbox_area_ratio: 最大边界框面积比例
            
        Returns:
            filtered_detections, filtered_probs: 过滤后的检测结果和概率
        """
        if len(detections.xyxy) == 0:
            return detections, probs
            
        # 获取图像尺寸
        image_height, image_width = self.observations["rgb"].shape[:2]
        total_image_area = image_height * image_width
        
        # 创建过滤掩码
        keep_mask = []
        
        for i in range(len(detections.xyxy)):
            # 计算掩码面积
            mask_area = detections.mask[i].sum() if hasattr(detections, 'mask') and detections.mask is not None else 0
            
            # 计算边界框面积
            x1, y1, x2, y2 = detections.xyxy[i]
            bbox_area = (x2 - x1) * (y2 - y1)
            bbox_area_ratio = bbox_area / total_image_area
            
            # 获取置信度
            confidence = detections.confidence[i] if hasattr(detections, 'confidence') else 1.0
            
            # 应用过滤条件
            if (mask_area >= mask_area_threshold and 
                confidence >= conf_threshold and 
                bbox_area_ratio <= max_bbox_area_ratio):
                keep_mask.append(True)
            else:
                keep_mask.append(False)
        
        keep_mask = np.array(keep_mask)
        
        if not np.any(keep_mask):
            # 如果没有检测通过过滤，返回空结果
            # 使用与detection_segmentation相同的空检测格式
            empty_detections = sv.Detections.empty()
            return empty_detections, []
        
        # 过滤检测结果
        filtered_detections = type(detections)(
            xyxy=detections.xyxy[keep_mask],
            confidence=detections.confidence[keep_mask] if hasattr(detections, 'confidence') else None,
            class_id=detections.class_id[keep_mask] if hasattr(detections, 'class_id') else None,
            mask=detections.mask[keep_mask] if hasattr(detections, 'mask') and detections.mask is not None else None
        )
        
        # 过滤概率分布
        filtered_probs = [probs[i] for i in range(len(probs)) if keep_mask[i]]
        
        return filtered_detections, filtered_probs
    
    
    
    # Grounded-SAM版的realtime mapping
    def realtime_mapping(self, detections, probs):
        self.observations = self.agent.observations
        self.pitch = self.agent.pitch
        self.extrinstic = self.get_extrinsic_matrix()
        curr_detection_nodes = []
        self.confirmed_observations = {"newly_detection": False, "confirmed_observations": [], "score": [], "bbox": [], "probs": []}
        self.final_confirmed_observatinos = {}
        if self.agent.FSM.mandatory_detection:
            vocabulay = [self.agent.obj_goal]
        else:
            vocabulay = self.agent.vocabulary
        
        # 在检测阶段就过滤掉低质量检测
        detections, probs = self.filter_detections(
            detections, probs, 
            mask_area_threshold=getattr(self.agent, 'mask_area_threshold', 25),
            conf_threshold=getattr(self.agent, 'conf_threshold', 0.25),
            max_bbox_area_ratio=getattr(self.agent, 'max_bbox_area_ratio', 0.5)
        )
        
        stairs_mask = None
                
        if len(detections.xyxy) != 0:        
            # visual_feature = get_mask_embeddings(self.observations["rgb"], detections, vision_model, processor) # embeddings[0][0].len = 768 embeddings.len = mask.len
            visual_feature = [i for i in range(len(detections.xyxy))]   # 调试中暂不考虑siglip，用一个全0列表代替
            
            aggregate_feature = get_aggregate_feature(
                image=self.observations["rgb"],
                depth=self.observations["depth"],
                extrinsic=self.extrinstic,
                detections=detections,
                visual_feature=visual_feature,
                vocabulary=vocabulay,
                obj_goal=self.agent.obj_goal,
                mandatory_detection=self.agent.FSM.mandatory_detection,
                voxel_size=0.01,
                device=self.agent.device,
                total_steps=self.agent.total_steps,
                pcd_dir=self.agent.pcd_dir,
                clip_model=getattr(self.agent, 'clip_model', None),
                clip_preprocess=getattr(self.agent, 'clip_preprocess', None)
            )
            
            # background filter
            aggregate_feature = self.pcd_filter(aggregate_feature)
            
            # 过滤当前帧中被完全包含的小点云
            # 只在使用CLIP时才基于视觉相似度过滤，否则不过滤
            aggregate_feature = filter_contained_pointclouds(
                aggregate_feature, 
                visual_sim_threshold=getattr(self.agent, 'containment_visual_sim_threshold', 0.7),
                debug=(self.agent.type == "debug")
            )
            
            if len(aggregate_feature) != 0:
                if self.object_node_lst == []:
                    for idx, new_object in enumerate(aggregate_feature):
                        new_object_node = ObjectNode(self.agent)
                        new_object_node.id = idx
                        new_object_node.class_update(
                            class_name=new_object['class_name'],
                            probs=probs[new_object['detections_id']],
                            bbox=new_object['bbox']
                        )
                        new_object_node.pcd = new_object['pcd']
                        new_object_node.siglip_embedding = new_object['embedding']
                        new_object_node.clip_embedding = new_object.get('clip_embedding', None)  # CLIP特征
                        new_object_node.center = new_object['center']
                        
                        self.object_node_lst.append(new_object_node)
                        curr_detection_nodes.append(new_object_node)

                        topk_indicies = torch.topk(new_object_node.class_candidates, 3).indices.tolist()  # 如果当前的检测形成的pcd对应的object node的probs的topk包含goal
                        if self.agent.obj_goal_id in topk_indicies:
                            self.confirmed_observations["newly_detection"] = True
                            self.confirmed_observations["confirmed_observations"].append({idx: self.get_bbox_observations(bbox=new_object["bbox"])})
                            self.confirmed_observations["score"].append({idx: new_object_node.score})
                            self.confirmed_observations["bbox"].append({idx: new_object["bbox"]})
                            self.confirmed_observations["probs"].append({idx: probs[new_object['detections_id']]})
                        
                        # if new_object['class_name'] == self.agent.obj_goal:
                        #     self.confirmed_observations["newly_detection"] = True
                        #     self.confirmed_observations["confirmed_observations"].append({idx: self.get_bbox_observations(bbox=new_object["bbox"])})
                        #     self.confirmed_observations["score"].append({idx: new_object_node.score})
                        #     self.confirmed_observations["bbox"].append({idx: new_object["bbox"]})
                        #     self.confirmed_observations["probs"].append({idx: probs[new_object['detections_id']]})
                        
                        
                else:
                    # 计算多种空间相似度
                    # compute_overlap_matrix_2set 同时返回重叠度矩阵和最小点距离矩阵（零额外成本）
                    overlap_similarity_matrix, min_distance_matrix = compute_overlap_matrix_2set(
                        self.object_node_lst, aggregate_feature
                    )
                    
                    # 使用最小点距离计算距离相似度（比中心距离更准确）
                    # 距离衰减因子从配置文件中读取（agent.center_distance_decay）
                    distance_decay_factor = getattr(self.agent, 'center_distance_decay', 0.3)
                    distance_similarity_matrix = compute_min_point_distance_similarity(
                        min_distance_matrix, max_distance=distance_decay_factor
                    )
                    
                    # 提取与aggregate_feature对应的probs子集
                    aggregate_probs = [probs[obj['detections_id']] for obj in aggregate_feature]
                    
                    # 计算视觉相似度矩阵
                    # 根据配置选择使用CLIP embeddings或DINO belief
                    use_clip_feature = getattr(self.agent, 'use_clip_feature', False)
                    visual_similarity_matrix = compute_visual_similarity_matrix(
                        self.object_node_lst, 
                        aggregate_feature, 
                        probs_list=aggregate_probs,
                        use_clip=use_clip_feature,
                        debug=(self.agent.type == "debug")
                    )
                    
                    
                    # 使用智能聚合函数
                    overlap_weight = getattr(self.agent, 'overlap_weight', 0.7)
                    visual_sim_weight = getattr(self.agent, 'visual_sim_weight', 0.3)
                    max_match_distance = getattr(self.agent, 'max_match_distance', 2.0)
                    distance_threshold = getattr(self.agent, 'distance_threshold', 0.3)
                    distance_penalty_strength = getattr(self.agent, 'distance_penalty_strength', 5.0)
                    
                    # 高视觉boost参数
                    high_visual_boost_enabled = getattr(self.agent, 'high_visual_boost_enabled', True)
                    high_visual_threshold = getattr(self.agent, 'high_visual_threshold', 0.7)
                    high_visual_distance_threshold = getattr(self.agent, 'high_visual_distance_threshold', 0.3)
                    high_visual_boost_factor = getattr(self.agent, 'high_visual_boost_factor', 0.5)
                    
                    aggregated_similarity_matrix = compute_aggregate_similarities(
                        overlap_similarity_matrix, distance_similarity_matrix, visual_similarity_matrix,
                        self.object_node_lst, aggregate_feature,
                        min_distance_matrix=min_distance_matrix,  # ⭐传入最小点距离矩阵
                        overlap_weight=overlap_weight,
                        visual_weight=visual_sim_weight,
                        max_match_distance=max_match_distance,
                        distance_threshold=distance_threshold,
                        distance_penalty_strength=distance_penalty_strength,
                        high_visual_boost_enabled=high_visual_boost_enabled,
                        high_visual_threshold=high_visual_threshold,
                        high_visual_distance_threshold=high_visual_distance_threshold,
                        high_visual_boost_factor=high_visual_boost_factor,
                        debug=(self.agent.type == "debug")
                    )
                    
                    
                    # 使用匈牙利算法进行最优二分图匹配
                    min_similarity_thresh = getattr(self.agent, 'min_similarity_thresh', 0.7)
                    
                    
                    # 创建过滤后的相似度矩阵
                    filtered_sim_matrix = np.where(
                        aggregated_similarity_matrix >= min_similarity_thresh,
                        aggregated_similarity_matrix, 0
                    )
                    
                    
                    # 使用匈牙利算法
                    cost_matrix = -filtered_sim_matrix
                    cost_matrix[filtered_sim_matrix == 0] = 1e6
                    old_indices, new_indices = linear_sum_assignment(cost_matrix)
                    
                    # 筛选有效匹配
                    matched_pairs = []
                    for i, j in zip(old_indices, new_indices):
                        if filtered_sim_matrix[i, j] > 0:
                            matched_pairs.append((i, j))
                    
                    matched_new_indices = {j for i, j in matched_pairs}
                    
                    # 调试信息：匈牙利匹配结果
                    if self.agent.type == "debug":
                        # 检查没有匹配到的检测
                        all_new_indices = set(range(len(aggregate_feature)))
                        unmatched_new_indices = all_new_indices - matched_new_indices
                        
                        if unmatched_new_indices:
                            print(f"\n{'='*80}")
                            print(f"未匹配检测分析 (共{len(unmatched_new_indices)}个)")
                            print(f"{'='*80}")
                            print(f"当前场景中已有物体数量: {len(self.object_node_lst)}")
                            for j in sorted(unmatched_new_indices):
                                new_obj = aggregate_feature[j]
                                # 找到与这个检测最相似的现有物体
                                max_sim = -1  # 改为-1以捕获所有情况，包括相似度为0的
                                best_match_idx = -1
                                for i in range(len(self.object_node_lst)):
                                    if aggregated_similarity_matrix[i, j] > max_sim:
                                        max_sim = aggregated_similarity_matrix[i, j]
                                        best_match_idx = i
                                
                                # 获取最佳匹配的详细相似度信息
                                if best_match_idx >= 0 and len(self.object_node_lst) > 0:
                                    best_overlap = overlap_similarity_matrix[best_match_idx, j]
                                    best_distance_sim = distance_similarity_matrix[best_match_idx, j]
                                    best_visual = visual_similarity_matrix[best_match_idx, j]
                                    best_obj = self.object_node_lst[best_match_idx]
                                    
                                    # 获取最小点距离（米）
                                    min_point_dist = min_distance_matrix[best_match_idx, j]
                                    
                                    # 计算聚合相似度的详细过程（简化版）
                                    base_sim = overlap_weight * best_overlap + visual_sim_weight * best_visual
                                    
                                    # 高视觉近距离boost
                                    boost_applied = False
                                    boost_value = 0.0
                                    if high_visual_boost_enabled:
                                        if best_visual >= high_visual_threshold and min_point_dist <= high_visual_distance_threshold:
                                            boost_value = high_visual_boost_factor
                                            base_sim += boost_value
                                            boost_applied = True
                                    
                                    # 距离惩罚（基于最小点距离）
                                    if min_point_dist <= distance_threshold:
                                        dist_penalty = 1.0
                                    else:
                                        dist_penalty = np.exp(-distance_penalty_strength * (min_point_dist - distance_threshold) / (max_match_distance - distance_threshold))
                                    
                                    # 简化的聚合相似度
                                    theoretical_sim = base_sim * dist_penalty
                                    
                                    # 检查硬约束（基于最小点距离）
                                    hard_constraint_triggered = False
                                    constraint_reason = ""
                                    if min_point_dist > max_match_distance:
                                        hard_constraint_triggered = True
                                        constraint_reason = f"最小点距{min_point_dist:.3f}m > 最大距离{max_match_distance:.2f}m"
                                    elif min_point_dist > distance_threshold and best_overlap < 0.05 and best_visual < 0.3:
                                        hard_constraint_triggered = True
                                        constraint_reason = f"最小点距{min_point_dist:.3f}m > {distance_threshold:.2f}m 且无重叠({best_overlap:.3f}) 且低视觉({best_visual:.3f}<0.3)"
                                    
                                    print(f"\n[检测{j}] 类别: {new_obj['class_name']} → 将创建为物体{len(self.object_node_lst) + list(sorted(unmatched_new_indices)).index(j)}")
                                    print(f"  最佳候选: 物体{best_match_idx} [{best_obj.class_name}]")
                                    print(f"  ┌─ 原始相似度:")
                                    print(f"  │  ├─ 重叠: {best_overlap:.3f}")
                                    print(f"  │  ├─ 视觉: {best_visual:.3f}")
                                    print(f"  │  └─ 最小点距: {min_point_dist:.3f}m (距离相似度={best_distance_sim:.3f})")
                                    print(f"  ├─ 聚合计算:")
                                    base_sim_before_boost = overlap_weight * best_overlap + visual_sim_weight * best_visual
                                    print(f"  │  ├─ 基础相似度 = {overlap_weight:.2f}×{best_overlap:.3f} + {visual_sim_weight:.2f}×{best_visual:.3f} = {base_sim_before_boost:.3f}")
                                    if boost_applied:
                                        print(f"  │  ├─ 🚀 高视觉boost = +{boost_value:.3f} (视觉={best_visual:.3f}≥{high_visual_threshold:.2f}, 距离={min_point_dist:.3f}m≤{high_visual_distance_threshold:.2f}m)")
                                        print(f"  │  ├─ Boost后基础 = {base_sim_before_boost:.3f} + {boost_value:.3f} = {base_sim:.3f}")
                                    print(f"  │  ├─ 距离惩罚 = {dist_penalty:.3f} (最小点距={min_point_dist:.3f}m, 阈值={distance_threshold:.2f}m)")
                                    print(f"  │  └─ 理论聚合 = {base_sim:.3f} × {dist_penalty:.3f} = {theoretical_sim:.3f}")
                                    if hard_constraint_triggered:
                                        print(f"  ├─ ⚠️  硬约束触发: {constraint_reason} → 强制设为0.000")
                                    print(f"  └─ 实际矩阵值 = {max_sim:.3f} (匹配阈值={min_similarity_thresh:.3f})")
                                    
                                    # 判断未匹配原因
                                    if hard_constraint_triggered:
                                        print(f"  ➜ 决策: ❌ 硬约束阻止匹配")
                                    elif max_sim < min_similarity_thresh:
                                        print(f"  ➜ 决策: ❌ 相似度不足")
                                    else:
                                        print(f"  ➜ 决策: ⚠️  匈牙利算法未选择 (被其他检测抢先)")
                                        # 检查是否被其他物体匹配
                                        for matched_i, matched_j in matched_pairs:
                                            if matched_i == best_match_idx:
                                                other_sim = aggregated_similarity_matrix[matched_i, matched_j]
                                                other_new = aggregate_feature[matched_j]
                                                print(f"       → 物体{matched_i}已与检测{matched_j}[{other_new['class_name']}]匹配 (相似度={other_sim:.3f} > {max_sim:.3f})")
                                                break
                                else:
                                    # 场景中没有现有物体，或所有相似度都是0
                                    print(f"\n[检测{j}] 类别: {new_obj['class_name']} → 将创建为物体{len(self.object_node_lst) + list(sorted(unmatched_new_indices)).index(j)}")
                                    if len(self.object_node_lst) == 0:
                                        print(f"  └─ 决策: ✓ 首次检测（场景中无现有物体）")
                                    else:
                                        # 显示与所有物体的相似度（找出为什么都是0）
                                        print(f"  ├─ 与所有{len(self.object_node_lst)}个现有物体的相似度:")
                                        for i in range(len(self.object_node_lst)):
                                            obj = self.object_node_lst[i]
                                            agg_sim = aggregated_similarity_matrix[i, j]
                                            overlap = overlap_similarity_matrix[i, j]
                                            visual = visual_similarity_matrix[i, j]
                                            dist_sim = distance_similarity_matrix[i, j]
                                            min_d = min_distance_matrix[i, j]
                                            
                                            print(f"  │  物体{i}[{obj.class_name}]: 聚合={agg_sim:.3f}, 重叠={overlap:.3f}, 视觉={visual:.3f}, 最小点距={min_d:.3f}m")
                                        print(f"  └─ 决策: ✓ 所有相似度均为0或过低，创建新物体")
                        
                        # 显示成功匹配的详情
                        if matched_pairs:
                            print(f"\n{'='*80}")
                            print(f"成功匹配详情 (共{len(matched_pairs)}对)")
                            print(f"{'='*80}")
                            for idx, (i, j) in enumerate(matched_pairs):
                                old_obj = self.object_node_lst[i]
                                new_obj = aggregate_feature[j]
                                similarity = filtered_sim_matrix[i, j]
                                overlap_sim = overlap_similarity_matrix[i, j]
                                distance_sim = distance_similarity_matrix[i, j]
                                visual_sim = visual_similarity_matrix[i, j]
                                
                                # 获取最小点距离
                                min_point_dist = min_distance_matrix[i, j]
                                
                                # 计算聚合过程（简化版，基于最小点距离）
                                base_sim_before = overlap_weight * overlap_sim + visual_sim_weight * visual_sim
                                base_sim = base_sim_before
                                boost_applied_match = False
                                if high_visual_boost_enabled:
                                    if visual_sim >= high_visual_threshold and min_point_dist <= high_visual_distance_threshold:
                                        base_sim += high_visual_boost_factor
                                        boost_applied_match = True
                                
                                if min_point_dist <= distance_threshold:
                                    dist_penalty = 1.0
                                else:
                                    dist_penalty = np.exp(-distance_penalty_strength * (min_point_dist - distance_threshold) / (max_match_distance - distance_threshold))
                                theoretical_sim = base_sim * dist_penalty
                                
                                print(f"\n✓ 匹配#{idx+1}: 物体{i} [{old_obj.class_name}] ← 检测{j} [{new_obj['class_name']}]")
                                print(f"  ├─ 原始相似度: 重叠={overlap_sim:.3f}, 视觉={visual_sim:.3f}")
                                print(f"  ├─ 最小点距: {min_point_dist:.3f}m (距离相似度={distance_sim:.3f})")
                                if boost_applied_match:
                                    print(f"  ├─ 聚合计算: 基础={base_sim_before:.3f} + boost({high_visual_boost_factor:.2f}) = {base_sim:.3f}, 距离惩罚={dist_penalty:.3f}")
                                else:
                                    print(f"  ├─ 聚合计算: 基础={base_sim:.3f}, 距离惩罚={dist_penalty:.3f}")
                                print(f"  ├─ 理论聚合 = {base_sim:.3f} × {dist_penalty:.3f} = {theoretical_sim:.3f}")
                                print(f"  └─ 实际相似度: {similarity:.3f}")
                            print(f"{'='*80}\n")
                    
                    for i, j in matched_pairs:
                        matched_det_pcd = aggregate_feature[j]['pcd']
                        matched_obj_pcd = self.object_node_lst[i].pcd
                        
                        self.object_node_lst[i].pcd, self.object_node_lst[i].center = merge_point_clouds(
                            pcd_new=matched_det_pcd, pcd_old=matched_obj_pcd, voxel_size=0.01
                        )
                            
                        self.object_node_lst[i].class_update(
                            class_name=aggregate_feature[j]['class_name'],
                            probs=probs[aggregate_feature[j]['detections_id']],
                            bbox=aggregate_feature[j]['bbox']
                        )
                        curr_detection_nodes.append(self.object_node_lst[i])
                        
                        if self.agent.FSM.mandatory_detection:
                            topk_indicies = [self.agent.obj_goal_id]
                        else:
                            topk_indicies = torch.topk(self.object_node_lst[i].class_candidates, 3).indices.tolist()
                        topk_classes = [self.agent.vocabulary[i] for i in topk_indicies]
                        
                        if self.agent.obj_goal_id in topk_indicies:
                            self.confirmed_observations["newly_detection"] = True 
                            self.confirmed_observations["confirmed_observations"].append({i: self.get_bbox_observations(bbox=aggregate_feature[j]["bbox"])})
                            self.confirmed_observations["score"].append({i: self.object_node_lst[i].score})
                            self.confirmed_observations["bbox"].append({i: aggregate_feature[j]["bbox"]})
                            self.confirmed_observations["probs"].append({i: probs[aggregate_feature[j]['detections_id']]})
                            
                        # if aggregate_feature[j]['class_name'] == self.agent.obj_goal:
                        #     self.confirmed_observations["newly_detection"] = True 
                        #     self.confirmed_observations["confirmed_observations"].append({i: self.get_bbox_observations(bbox=aggregate_feature[j]["bbox"])})
                        #     self.confirmed_observations["score"].append({i: self.object_node_lst[i].score})
                        #     self.confirmed_observations["bbox"].append({i: aggregate_feature[j]["bbox"]})
                        #     self.confirmed_observations["probs"].append({i: probs[aggregate_feature[j]['detections_id']]})
                        
                        
                        if self.agent.obj_goal in topk_classes:
                            self.final_confirmed_observatinos[i] = self.get_bbox_observations(bbox=aggregate_feature[j]["bbox"])
                        

                    # 处理未匹配的检测
                    all_new_indices = set(range(len(aggregate_feature)))
                    unmatched_new_indices = all_new_indices - matched_new_indices
                    
                    # 检查一对多冲突并丢弃冲突检测
                    valid_new_indices = set()
                    
                    for j in unmatched_new_indices:
                        max_similarity = aggregated_similarity_matrix[:, j].max()
                        max_obj_idx = aggregated_similarity_matrix[:, j].argmax()
                        
                        # 检查是否与已匹配的物体有冲突
                        is_conflict = False
                        if max_similarity >= min_similarity_thresh:
                            for matched_i, matched_j in matched_pairs:
                                if matched_i == max_obj_idx:
                                    is_conflict = True
                                    break
                        
                        if not is_conflict:
                            valid_new_indices.add(j)
                    
                    unmatched_new_indices = valid_new_indices
                    
                    # 调试信息：冲突检查后确认创建的物体
                    if self.agent.type == "debug" and unmatched_new_indices:
                        print(f"\n[冲突检查完成] 确认创建 {len(unmatched_new_indices)} 个新物体:")
                        for j in sorted(unmatched_new_indices):
                            new_id = len(self.object_node_lst) + list(sorted(unmatched_new_indices)).index(j)
                            print(f"  ✓ 检测{j} [{aggregate_feature[j]['class_name']}] → 新物体ID={new_id}")

                    for j in unmatched_new_indices:
                        new_object_node = ObjectNode(self.agent)
                        new_object_node.id = len(self.object_node_lst)
                        
                        if self.agent.type == "debug":
                            print(f"[创建] 新物体ID={new_object_node.id}, 类别={aggregate_feature[j]['class_name']}, 中心=({aggregate_feature[j]['center'][0]:.3f}, {aggregate_feature[j]['center'][1]:.3f}, {aggregate_feature[j]['center'][2]:.3f})m") 

                        new_object_node.class_update(
                            class_name=aggregate_feature[j]['class_name'],
                            probs=probs[aggregate_feature[j]['detections_id']],
                            bbox=aggregate_feature[j]['bbox']
                        )
                        new_object_node.pcd = aggregate_feature[j]['pcd']
                        new_object_node.siglip_embedding = aggregate_feature[j]['embedding']
                        new_object_node.clip_embedding = aggregate_feature[j].get('clip_embedding', None)  # CLIP特征
                        new_object_node.center = aggregate_feature[j]['center']
                        
                        self.object_node_lst.append(new_object_node)
                        curr_detection_nodes.append(new_object_node)
                        
                        topk_indicies = torch.topk(torch.tensor(probs[aggregate_feature[j]['detections_id']]), 3).indices.tolist()
                        if self.agent.obj_goal_id in topk_indicies:
                            self.confirmed_observations["newly_detection"] = True
                            self.confirmed_observations["confirmed_observations"].append({new_object_node.id: self.get_bbox_observations(bbox=aggregate_feature[j]["bbox"])})
                            self.confirmed_observations["score"].append({new_object_node.id: new_object_node.score})
                            self.confirmed_observations["bbox"].append({new_object_node.id: aggregate_feature[j]["bbox"]})
                            self.confirmed_observations["probs"].append({new_object_node.id: probs[aggregate_feature[j]['detections_id']]})
                        
                        # if aggregate_feature[j]['class_name'] == self.agent.obj_goal:
                        #     self.confirmed_observations["newly_detection"] = True
                        #     self.confirmed_observations["confirmed_observations"].append({new_object_node.id: self.get_bbox_observations(bbox=aggregate_feature[j]["bbox"])})
                        #     self.confirmed_observations["score"].append({new_object_node.id: new_object_node.score})
                        #     self.confirmed_observations["bbox"].append({new_object_node.id: aggregate_feature[j]["bbox"]})
                        #     self.confirmed_observations["probs"].append({new_object_node.id: probs[aggregate_feature[j]['detections_id']]})
                    
                             
        return curr_detection_nodes, stairs_mask
                    
if __name__ == "__main__":
    scenegraph = SceneGraph()

import math
import cv2
import torch
import supervision as sv
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import binary_dilation, binary_erosion, label

# from HuNav.models.detection import yolo_detection
# from models.segmentation import sam_segmentation

from utils.realtime_mapping import get_aggregate_feature
from utils.utils import compute_overlap_matrix_2set, merge_obj2_into_obj1, merge_point_clouds, volume_ratio_voxel_gen, compute_color_similarity_matrix, compute_visual_similarity_matrix, compute_center_distance_similarity, compute_min_point_distance_similarity, compute_aggregate_similarities, filter_contained_pointclouds
from utils.grounded_sam_client import detection_segmentation
from utils.skeleton_extract import SkeletonProcessor
# from utils.SpatialLM_utils import SpatialLM_detection

ROOM_COLOR_PALETTE = [
    (230, 57, 70),
    (29, 53, 87),
    (69, 123, 157),
    (168, 218, 220),
    (42, 157, 143),
    (233, 196, 106),
    (244, 162, 97),
    (231, 111, 81),
    (142, 202, 230),
    (34, 87, 122),
    (255, 111, 97),
    (149, 213, 178),
    (144, 190, 109),
    (67, 170, 139),
    (87, 117, 144),
    (255, 183, 3)
]



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
        self.clip_embedding = None
        self.center = None
        self.place = None
        self.score = 0.0
        self.edge = []
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
        self._compute_kl_divergence(self.class_candidates, post_probs)
        self.last_post_probs = self.class_candidates
        self.class_candidates = post_probs
        self.class_id = torch.argmax(self.class_candidates)
        self.class_name = self.agent.vocabulary[self.class_id]
        self.score = self.class_candidates[self.class_id]
    
    # def _bayes_update(self, probs):
    #     obs = torch.tensor(probs)
    #     prior = self.class_candidates
        
    #     kl_div = torch.sum(obs * torch.log((obs + 1e-8) / (prior + 1e-8)))
        
    #     obs_entropy = -torch.sum(obs * torch.log(obs + 1e-8))
        
        
        
        
    #     post_probs = weight * obs + (1 - weight) * prior
        
    #     post_probs = post_probs / torch.sum(post_probs)
        
    #     self._compute_kl_divergence(self.class_candidates, post_probs)
    #     self.class_candidates = post_probs
    #     self.class_id = torch.argmax(self.class_candidates)
    #     self.class_name = self.agent.vocabulary[self.class_id]
    #     self.score = self.class_candidates[self.class_id]
    
    def _rollback_class(self):
        self.class_candidates = self.last_post_probs    
        self.class_id = torch.argmax(self.class_candidates)
        self.class_name = self.agent.vocabulary[self.class_id]
        self.score = self.class_candidates[self.class_id]
    
    def _compute_kl_divergence(self, prior_probs, post_probs):
        self.kl_divergence = torch.sum(post_probs * (torch.log(post_probs) - torch.log(prior_probs))).item()
    
    def class_entropy(self):
        entropy = -torch.sum(self.class_candidates * torch.log(self.class_candidates + 1e-6))
        return entropy
        
    def update_class_belief(self, class_name, probs, bbox):
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

        
class RoomNode():
    def __init__(self, room_id, region_points, color=None):
        self.id = room_id
        self.color = self._sample_color() if color is None else tuple(color)
        self.region_points = np.zeros((0, 2), dtype=np.int32)
        self.region_set = set()
        self.center = (0.0, 0.0)
        self.area = 0
        self.room_name = None
        self.center_locked = False
        self.update_region(region_points, union=False)

    def _sample_color(self):
        if ROOM_COLOR_PALETTE:
            return ROOM_COLOR_PALETTE[self.id % len(ROOM_COLOR_PALETTE)]
        color = np.random.randint(32, 200, size=3, dtype=np.uint8)
        return int(color[0]), int(color[1]), int(color[2])

    def update_region(self, region_points, union=True):
        if region_points is None or len(region_points) == 0:
            return
        if isinstance(region_points, list):
            region_points = np.array(region_points, dtype=np.int32)
        if not isinstance(region_points, np.ndarray):
            region_points = np.array(region_points, dtype=np.int32)
        if region_points.size == 0:
            return
        coords_set = set(map(lambda pt: (int(pt[0]), int(pt[1])), region_points.tolist()))
        if union:
            self.region_set |= coords_set
        else:
            self.region_set = coords_set
        self.region_points = np.array(list(self.region_set), dtype=np.int32)
        self.area = len(self.region_points)
        if self.area > 0 and not self.center_locked:
            self.center = (
                float(self.region_points[:, 0].mean()),
                float(self.region_points[:, 1].mean())
            )

    def lock_center(self, center=None):
        if center is not None:
            self.center = (float(center[0]), float(center[1]))
        self.center_locked = True

    def coords_set(self):
        return self.region_set


class SceneGraph():
    def __init__(self, agent):
        self.agent = agent
        self.map_size_cm = self.agent.map_size_cm
        self.confirmed_observations = {"newly_detection": False, "confirmed_observations": [], "score": [], "bbox": [], "probs": []}
        self.final_confirmed_observatinos = {}
        self.object_node_lst = []
        self.bbox_lst = []
        self.place_node_lst = []
        self.room_node_lst = []
        self.room_debug = None
        self.room_id_counter = 0
        self.skeleton_processor = SkeletonProcessor(
            dilate_kernel=self.agent.dilate_kernel,
            min_component_size=self.agent.min_component_size,
            pruning_iterations=self.agent.pruning_iterations,
            min_junction_distance=self.agent.min_junction_distance,
            closing_kernel_size=self.agent.closing_kernel_size,
            hole_fill_area=self.agent.hole_fill_area
        )
        
    
    def reset(self):
        self.confirmed_observations = {"newly_detection": False, "confirmed_observations": [], "score": [], "bbox": [], "probs": []}
        self.final_confirmed_observatinos = {}
        self.object_node_lst = []
        self.bbox_lst = []
        self.place_node_lst = []
        self.room_node_lst = []
        self.room_debug = None
        self.room_id_counter = 0

    def get_extrinsic_matrix(self):
        x = self.map_size_cm / 100.0 / 2.0 + self.agent.gps[0]
        y = self.map_size_cm / 100.0 / 2.0 - self.agent.gps[1]
        z = 0.88
        yaw = (self.observations['compass'] - np.pi / 2)[0]
        pitch = self.pitch

        cos_p, sin_p = np.cos(pitch), np.sin(pitch)
        Rx = np.array([
            [1, 0,      0     ],
            [0, cos_p, -sin_p],
            [0, sin_p,  cos_p]
        ])

        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        Rz = np.array([
            [cos_y, -sin_y, 0],
            [sin_y,  cos_y, 0],
            [0,      0,     1]
        ])

        R_cam_to_world = np.array([
            [1, 0,  0],
            [0, 0,  1],  # Y_cam → -Z_world
            [0, -1,  0]   # Z_cam → Y_world
        ])

        R = Rz @ R_cam_to_world @ Rx

        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = R
        pose_matrix[:3, 3] = [x, y, z]
        return pose_matrix

    def draw_box_on_observation(self, bbox, box_color=(0, 255, 0), thickness=2):
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
    
    
    
    def filter_object_point_clouds(self, feature_list):
        for i in reversed(range(len(feature_list))):
            obj_feature = feature_list[i]
            if obj_feature['class_name'] in ['door', 'stairs']:
                del feature_list[i]
            else:
                obj_center = obj_feature['center']
                if obj_center[2] < 0.1:
                    del feature_list[i]
        return feature_list

    
    def detect_objects(self):
        if self.agent.state_machine.mandatory_detection:
            detection_vocabulary = [self.agent.obj_goal]
            detections, bbox_img, mask_img, probs = detection_segmentation(
                observations=self.agent.observations,
                vocabulary=detection_vocabulary,
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
            detection_vocabulary = self.agent.vocabulary
            detections, bbox_img, mask_img, probs = detection_segmentation(
                observations=self.agent.observations,
                vocabulary=detection_vocabulary,
                port=self.agent.port,
                use_ram=False,
                box_threshold=0.35,  
                text_threshold=0.35, 
                iou_threshold=0.35
            )        
            
            return detections, bbox_img, mask_img, probs
    
    def filter_detections(self, detections, probs, mask_area_threshold=25, conf_threshold=0.35, max_bbox_area_ratio=0.7):
        """Filter low-quality detections before matching."""
        if len(detections.xyxy) == 0:
            return detections, probs
            
        image_height, image_width = self.observations["rgb"].shape[:2]
        total_image_area = image_height * image_width
        
        keep_mask = []
        
        for i in range(len(detections.xyxy)):
            mask_area = detections.mask[i].sum() if hasattr(detections, 'mask') and detections.mask is not None else 0
            
            x1, y1, x2, y2 = detections.xyxy[i]
            bbox_area = (x2 - x1) * (y2 - y1)
            bbox_area_ratio = bbox_area / total_image_area
            
            confidence = detections.confidence[i] if hasattr(detections, 'confidence') else 1.0
            
            if (mask_area >= mask_area_threshold and 
                confidence >= conf_threshold and 
                bbox_area_ratio <= max_bbox_area_ratio):
                keep_mask.append(True)
            else:
                keep_mask.append(False)
        
        keep_mask = np.array(keep_mask)
        
        if not np.any(keep_mask):
            empty_detections = sv.Detections.empty()
            return empty_detections, []
        
        filtered_detections = type(detections)(
            xyxy=detections.xyxy[keep_mask],
            confidence=detections.confidence[keep_mask] if hasattr(detections, 'confidence') else None,
            class_id=detections.class_id[keep_mask] if hasattr(detections, 'class_id') else None,
            mask=detections.mask[keep_mask] if hasattr(detections, 'mask') and detections.mask is not None else None
        )
        
        filtered_probs = [probs[i] for i in range(len(probs)) if keep_mask[i]]
        
        return filtered_detections, filtered_probs
    
    
    
    def integrate_detections(self, detections, probs):
        self.observations = self.agent.observations
        self.pitch = self.agent.pitch
        self.extrinstic = self.get_extrinsic_matrix()
        curr_detection_nodes = []
        self.confirmed_observations = {"newly_detection": False, "confirmed_observations": [], "score": [], "bbox": [], "probs": []}
        self.final_confirmed_observatinos = {}
        if self.agent.state_machine.mandatory_detection:
            detection_vocabulary = [self.agent.obj_goal]
        else:
            detection_vocabulary = self.agent.vocabulary
        
        detections, probs = self.filter_detections(
            detections, probs, 
            mask_area_threshold=self.agent.mask_area_threshold,
            conf_threshold=self.agent.conf_threshold,
            max_bbox_area_ratio=self.agent.max_bbox_area_ratio
        )
        
        stairs_mask = None
                
        if len(detections.xyxy) != 0:        
            # visual_feature = get_mask_embeddings(self.observations["rgb"], detections, vision_model, processor) # embeddings[0][0].len = 768 embeddings.len = mask.len
            visual_feature = [i for i in range(len(detections.xyxy))]
            
            aggregate_feature = get_aggregate_feature(
                image=self.observations["rgb"],
                depth=self.observations["depth"],
                extrinsic=self.extrinstic,
                detections=detections,
                visual_feature=visual_feature,
                vocabulary=detection_vocabulary,
                obj_goal=self.agent.obj_goal,
                mandatory_detection=self.agent.state_machine.mandatory_detection,
                voxel_size=0.01,
                device=self.agent.device,
                total_steps=self.agent.total_steps,
                pcd_dir=self.agent.pcd_dir,
                clip_model=self.agent.clip_model,
                clip_preprocess=self.agent.clip_preprocess
            )
            
            # background filter
            aggregate_feature = self.filter_object_point_clouds(aggregate_feature)
            
            aggregate_feature = filter_contained_pointclouds(
                aggregate_feature, 
                visual_sim_threshold=self.agent.containment_visual_sim_threshold,
                debug=(self.agent.type == "debug")
            )
            
            if len(aggregate_feature) != 0:
                if self.object_node_lst == []:
                    for idx, new_object in enumerate(aggregate_feature):
                        new_object_node = ObjectNode(self.agent)
                        new_object_node.id = idx
                        new_object_node.update_class_belief(
                            class_name=new_object['class_name'],
                            probs=probs[new_object['detections_id']],
                            bbox=new_object['bbox']
                        )
                        new_object_node.pcd = new_object['pcd']
                        new_object_node.siglip_embedding = new_object['embedding']
                        new_object_node.clip_embedding = new_object.get('clip_embedding', None)
                        new_object_node.center = new_object['center']
                        
                        self.object_node_lst.append(new_object_node)
                        curr_detection_nodes.append(new_object_node)

                        topk_indicies = torch.topk(new_object_node.class_candidates, 3).indices.tolist()
                        if self.agent.obj_goal_id in topk_indicies:
                            self.confirmed_observations["newly_detection"] = True
                            self.confirmed_observations["confirmed_observations"].append({idx: self.draw_box_on_observation(bbox=new_object["bbox"])})
                            self.confirmed_observations["score"].append({idx: new_object_node.score})
                            self.confirmed_observations["bbox"].append({idx: new_object["bbox"]})
                            self.confirmed_observations["probs"].append({idx: probs[new_object['detections_id']]})
                        
                        # if new_object['class_name'] == self.agent.obj_goal:
                        #     self.confirmed_observations["newly_detection"] = True
                        #     self.confirmed_observations["confirmed_observations"].append({idx: self.draw_box_on_observation(bbox=new_object["bbox"])})
                        #     self.confirmed_observations["score"].append({idx: new_object_node.score})
                        #     self.confirmed_observations["bbox"].append({idx: new_object["bbox"]})
                        #     self.confirmed_observations["probs"].append({idx: probs[new_object['detections_id']]})
                        
                        
                else:
                    overlap_similarity_matrix, min_distance_matrix = compute_overlap_matrix_2set(
                        self.object_node_lst, aggregate_feature
                    )
                    
                    distance_decay_factor = self.agent.center_distance_decay
                    distance_similarity_matrix = compute_min_point_distance_similarity(
                        min_distance_matrix, max_distance=distance_decay_factor
                    )
                    
                    aggregate_probs = [probs[obj['detections_id']] for obj in aggregate_feature]
                    
                    use_clip_feature = self.agent.use_clip_feature
                    visual_similarity_matrix = compute_visual_similarity_matrix(
                        self.object_node_lst, 
                        aggregate_feature, 
                        probs_list=aggregate_probs,
                        use_clip=use_clip_feature,
                        debug=(self.agent.type == "debug")
                    )
                    
                    
                    overlap_weight = self.agent.overlap_weight
                    visual_sim_weight = self.agent.visual_sim_weight
                    max_match_distance = self.agent.max_match_distance
                    distance_threshold = self.agent.distance_threshold
                    distance_penalty_strength = self.agent.distance_penalty_strength
                    
                    high_visual_boost_enabled = self.agent.high_visual_boost_enabled
                    high_visual_threshold = self.agent.high_visual_threshold
                    high_visual_distance_threshold = self.agent.high_visual_distance_threshold
                    high_visual_boost_factor = self.agent.high_visual_boost_factor
                    
                    aggregated_similarity_matrix = compute_aggregate_similarities(
                        overlap_similarity_matrix, distance_similarity_matrix, visual_similarity_matrix,
                        self.object_node_lst, aggregate_feature,
                        min_distance_matrix=min_distance_matrix,
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
                    
                    
                    min_similarity_thresh = self.agent.min_similarity_thresh
                    
                    
                    filtered_sim_matrix = np.where(
                        aggregated_similarity_matrix >= min_similarity_thresh,
                        aggregated_similarity_matrix, 0
                    )
                    
                    
                    cost_matrix = -filtered_sim_matrix
                    cost_matrix[filtered_sim_matrix == 0] = 1e6
                    old_indices, new_indices = linear_sum_assignment(cost_matrix)
                    
                    matched_pairs = []
                    for i, j in zip(old_indices, new_indices):
                        if filtered_sim_matrix[i, j] > 0:
                            matched_pairs.append((i, j))
                    
                    matched_new_indices = {j for i, j in matched_pairs}
                    
                    if self.agent.type == "debug":
                        all_new_indices = set(range(len(aggregate_feature)))
                        unmatched_new_indices = all_new_indices - matched_new_indices
                        
                        if unmatched_new_indices:
                            print(f"\n{'='*80}")
                            print(f"Unmatched detections ({len(unmatched_new_indices)})")
                            print(f"{'='*80}")
                            print(f"Existing objects in scene: {len(self.object_node_lst)}")
                            for j in sorted(unmatched_new_indices):
                                new_obj = aggregate_feature[j]
                                max_sim = -1
                                best_match_idx = -1
                                for i in range(len(self.object_node_lst)):
                                    if aggregated_similarity_matrix[i, j] > max_sim:
                                        max_sim = aggregated_similarity_matrix[i, j]
                                        best_match_idx = i
                                
                                if best_match_idx >= 0 and len(self.object_node_lst) > 0:
                                    best_overlap = overlap_similarity_matrix[best_match_idx, j]
                                    best_distance_sim = distance_similarity_matrix[best_match_idx, j]
                                    best_visual = visual_similarity_matrix[best_match_idx, j]
                                    best_obj = self.object_node_lst[best_match_idx]
                                    
                                    min_point_dist = min_distance_matrix[best_match_idx, j]
                                    
                                    base_sim = overlap_weight * best_overlap + visual_sim_weight * best_visual
                                    
                                    boost_applied = False
                                    boost_value = 0.0
                                    if high_visual_boost_enabled:
                                        if best_visual >= high_visual_threshold and min_point_dist <= high_visual_distance_threshold:
                                            boost_value = high_visual_boost_factor
                                            base_sim += boost_value
                                            boost_applied = True
                                    
                                    if min_point_dist <= distance_threshold:
                                        dist_penalty = 1.0
                                    else:
                                        dist_penalty = np.exp(-distance_penalty_strength * (min_point_dist - distance_threshold) / (max_match_distance - distance_threshold))
                                    
                                    theoretical_sim = base_sim * dist_penalty
                                    
                                    hard_constraint_triggered = False
                                    constraint_reason = ""
                                    hard_overlap_thresh = self.agent.hard_constraint_overlap_threshold
                                    hard_visual_thresh = self.agent.hard_constraint_visual_threshold
                                    if min_point_dist > max_match_distance:
                                        hard_constraint_triggered = True
                                        constraint_reason = f"Minimum point distance {min_point_dist:.3f} m exceeds maximum {max_match_distance:.2f} m"
                                    elif min_point_dist > distance_threshold and best_overlap < hard_overlap_thresh and best_visual < hard_visual_thresh:
                                        hard_constraint_triggered = True
                                        constraint_reason = f"Minimum point distance {min_point_dist:.3f} m > {distance_threshold:.2f} m with low overlap ({best_overlap:.3f}) and visual similarity ({best_visual:.3f} < {hard_visual_thresh})"
                                    
                                    print(f"\n[Detection {j}] Class: {new_obj['class_name']}; creating object {len(self.object_node_lst) + list(sorted(unmatched_new_indices)).index(j)}")
                                    print(f"  Best candidate: object {best_match_idx} [{best_obj.class_name}]")
                                    print(f"  Raw similarity:")
                                    print(f"    Overlap: {best_overlap:.3f}")
                                    print(f"    Visual: {best_visual:.3f}")
                                    print(f"    Minimum point distance: {min_point_dist:.3f} m (distance similarity={best_distance_sim:.3f})")
                                    print(f"  Aggregate calculation:")
                                    base_sim_before_boost = overlap_weight * best_overlap + visual_sim_weight * best_visual
                                    print(f"    Base similarity = {overlap_weight:.2f} * {best_overlap:.3f} + {visual_sim_weight:.2f} * {best_visual:.3f} = {base_sim_before_boost:.3f}")
                                    if boost_applied:
                                        print(f"    Visual boost = +{boost_value:.3f} (visual={best_visual:.3f} >= {high_visual_threshold:.2f}, distance={min_point_dist:.3f} m <= {high_visual_distance_threshold:.2f} m)")
                                        print(f"    Boosted base = {base_sim_before_boost:.3f} + {boost_value:.3f} = {base_sim:.3f}")
                                    print(f"    Distance penalty = {dist_penalty:.3f} (minimum point distance={min_point_dist:.3f} m, threshold={distance_threshold:.2f} m)")
                                    print(f"    Expected aggregate = {base_sim:.3f} * {dist_penalty:.3f} = {theoretical_sim:.3f}")
                                    if hard_constraint_triggered:
                                        print(f"  Hard constraint: {constraint_reason}; forcing score to zero")
                                    print(f"  Matrix value = {max_sim:.3f} (match threshold={min_similarity_thresh:.3f})")
                                    
                                    if hard_constraint_triggered:
                                        print(f"  Decision: blocked by hard constraint")
                                    elif max_sim < min_similarity_thresh:
                                        print(f"  Decision: insufficient similarity")
                                    else:
                                        print(f"  Decision: another detection won the assignment")
                                        for matched_i, matched_j in matched_pairs:
                                            if matched_i == best_match_idx:
                                                other_sim = aggregated_similarity_matrix[matched_i, matched_j]
                                                other_new = aggregate_feature[matched_j]
                                                print(f"    Object {matched_i} matched detection {matched_j} [{other_new['class_name']}] (similarity {other_sim:.3f} > {max_sim:.3f})")
                                                break
                                else:
                                    print(f"\n[Detection {j}] Class: {new_obj['class_name']}; creating object {len(self.object_node_lst) + list(sorted(unmatched_new_indices)).index(j)}")
                                    if len(self.object_node_lst) == 0:
                                        print(f"  Decision: first detection in an empty scene")
                                    else:
                                        print(f"  Similarity to {len(self.object_node_lst)} existing objects:")
                                        for i in range(len(self.object_node_lst)):
                                            obj = self.object_node_lst[i]
                                            agg_sim = aggregated_similarity_matrix[i, j]
                                            overlap = overlap_similarity_matrix[i, j]
                                            visual = visual_similarity_matrix[i, j]
                                            dist_sim = distance_similarity_matrix[i, j]
                                            min_d = min_distance_matrix[i, j]
                                            
                                            print(f"    Object {i} [{obj.class_name}]: aggregate={agg_sim:.3f}, overlap={overlap:.3f}, visual={visual:.3f}, minimum distance={min_d:.3f} m")
                                        print(f"  Decision: create an object because all scores are zero or below threshold")
                        
                        if matched_pairs:
                            print(f"\n{'='*80}")
                            print(f"Matched pairs ({len(matched_pairs)})")
                            print(f"{'='*80}")
                            for idx, (i, j) in enumerate(matched_pairs):
                                old_obj = self.object_node_lst[i]
                                new_obj = aggregate_feature[j]
                                similarity = filtered_sim_matrix[i, j]
                                overlap_sim = overlap_similarity_matrix[i, j]
                                distance_sim = distance_similarity_matrix[i, j]
                                visual_sim = visual_similarity_matrix[i, j]
                                
                                min_point_dist = min_distance_matrix[i, j]
                                
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
                                
                                print(f"\nMatch #{idx+1}: object {i} [{old_obj.class_name}] from detection {j} [{new_obj['class_name']}]")
                                print(f"  Raw similarity: overlap={overlap_sim:.3f}, visual={visual_sim:.3f}")
                                print(f"  Minimum point distance: {min_point_dist:.3f} m (distance similarity={distance_sim:.3f})")
                                if boost_applied_match:
                                    print(f"  Aggregate: base={base_sim_before:.3f} + boost({high_visual_boost_factor:.2f}) = {base_sim:.3f}, distance penalty={dist_penalty:.3f}")
                                else:
                                    print(f"  Aggregate: base={base_sim:.3f}, distance penalty={dist_penalty:.3f}")
                                print(f"  Expected aggregate = {base_sim:.3f} * {dist_penalty:.3f} = {theoretical_sim:.3f}")
                                print(f"  Actual similarity: {similarity:.3f}")
                            print(f"{'='*80}\n")
                    
                    for i, j in matched_pairs:
                        matched_det_pcd = aggregate_feature[j]['pcd']
                        matched_obj_pcd = self.object_node_lst[i].pcd
                        
                        self.object_node_lst[i].pcd, self.object_node_lst[i].center = merge_point_clouds(
                            pcd_new=matched_det_pcd, pcd_old=matched_obj_pcd, voxel_size=0.01
                        )
                            
                        self.object_node_lst[i].update_class_belief(
                            class_name=aggregate_feature[j]['class_name'],
                            probs=probs[aggregate_feature[j]['detections_id']],
                            bbox=aggregate_feature[j]['bbox']
                        )
                        curr_detection_nodes.append(self.object_node_lst[i])
                        
                        if self.agent.state_machine.mandatory_detection:
                            topk_indicies = [self.agent.obj_goal_id]
                        else:
                            topk_indicies = torch.topk(self.object_node_lst[i].class_candidates, 3).indices.tolist()
                        topk_classes = [self.agent.vocabulary[i] for i in topk_indicies]
                        
                        if self.agent.obj_goal_id in topk_indicies:
                            self.confirmed_observations["newly_detection"] = True 
                            self.confirmed_observations["confirmed_observations"].append({i: self.draw_box_on_observation(bbox=aggregate_feature[j]["bbox"])})
                            self.confirmed_observations["score"].append({i: self.object_node_lst[i].score})
                            self.confirmed_observations["bbox"].append({i: aggregate_feature[j]["bbox"]})
                            self.confirmed_observations["probs"].append({i: probs[aggregate_feature[j]['detections_id']]})
                            
                        # if aggregate_feature[j]['class_name'] == self.agent.obj_goal:
                        #     self.confirmed_observations["newly_detection"] = True 
                        #     self.confirmed_observations["confirmed_observations"].append({i: self.draw_box_on_observation(bbox=aggregate_feature[j]["bbox"])})
                        #     self.confirmed_observations["score"].append({i: self.object_node_lst[i].score})
                        #     self.confirmed_observations["bbox"].append({i: aggregate_feature[j]["bbox"]})
                        #     self.confirmed_observations["probs"].append({i: probs[aggregate_feature[j]['detections_id']]})
                        
                        
                        if self.agent.obj_goal in topk_classes:
                            self.final_confirmed_observatinos[i] = self.draw_box_on_observation(bbox=aggregate_feature[j]["bbox"])
                        

                    all_new_indices = set(range(len(aggregate_feature)))
                    unmatched_new_indices = all_new_indices - matched_new_indices
                    
                    valid_new_indices = set()
                    
                    for j in unmatched_new_indices:
                        max_similarity = aggregated_similarity_matrix[:, j].max()
                        max_obj_idx = aggregated_similarity_matrix[:, j].argmax()
                        
                        is_conflict = False
                        if max_similarity >= min_similarity_thresh:
                            for matched_i, matched_j in matched_pairs:
                                if matched_i == max_obj_idx:
                                    is_conflict = True
                                    break
                        
                        if not is_conflict:
                            valid_new_indices.add(j)
                    
                    unmatched_new_indices = valid_new_indices
                    
                    if self.agent.type == "debug" and unmatched_new_indices:
                        print(f"\n[Conflict check] Creating {len(unmatched_new_indices)} new objects:")
                        for j in sorted(unmatched_new_indices):
                            new_id = len(self.object_node_lst) + list(sorted(unmatched_new_indices)).index(j)
                            print(f"  Detection {j} [{aggregate_feature[j]['class_name']}] -> new object ID={new_id}")

                    for j in unmatched_new_indices:
                        new_object_node = ObjectNode(self.agent)
                        new_object_node.id = len(self.object_node_lst)
                        
                        if self.agent.type == "debug":
                            print(f"[Create] Object ID={new_object_node.id}, class={aggregate_feature[j]['class_name']}, center=({aggregate_feature[j]['center'][0]:.3f}, {aggregate_feature[j]['center'][1]:.3f}, {aggregate_feature[j]['center'][2]:.3f}) m")

                        new_object_node.update_class_belief(
                            class_name=aggregate_feature[j]['class_name'],
                            probs=probs[aggregate_feature[j]['detections_id']],
                            bbox=aggregate_feature[j]['bbox']
                        )
                        new_object_node.pcd = aggregate_feature[j]['pcd']
                        new_object_node.siglip_embedding = aggregate_feature[j]['embedding']
                        new_object_node.clip_embedding = aggregate_feature[j].get('clip_embedding', None)
                        new_object_node.center = aggregate_feature[j]['center']
                        
                        self.object_node_lst.append(new_object_node)
                        curr_detection_nodes.append(new_object_node)
                        
                        topk_indicies = torch.topk(torch.tensor(probs[aggregate_feature[j]['detections_id']]), 3).indices.tolist()
                        if self.agent.obj_goal_id in topk_indicies:
                            self.confirmed_observations["newly_detection"] = True
                            self.confirmed_observations["confirmed_observations"].append({new_object_node.id: self.draw_box_on_observation(bbox=aggregate_feature[j]["bbox"])})
                            self.confirmed_observations["score"].append({new_object_node.id: new_object_node.score})
                            self.confirmed_observations["bbox"].append({new_object_node.id: aggregate_feature[j]["bbox"]})
                            self.confirmed_observations["probs"].append({new_object_node.id: probs[aggregate_feature[j]['detections_id']]})
                        
                        # if aggregate_feature[j]['class_name'] == self.agent.obj_goal:
                        #     self.confirmed_observations["newly_detection"] = True
                        #     self.confirmed_observations["confirmed_observations"].append({new_object_node.id: self.draw_box_on_observation(bbox=aggregate_feature[j]["bbox"])})
                        #     self.confirmed_observations["score"].append({new_object_node.id: new_object_node.score})
                        #     self.confirmed_observations["bbox"].append({new_object_node.id: aggregate_feature[j]["bbox"]})
                        #     self.confirmed_observations["probs"].append({new_object_node.id: probs[aggregate_feature[j]['detections_id']]})
                    
                             
        return curr_detection_nodes, stairs_mask

    def segment_rooms(self, delta=None):
        """Segment rooms from the occupancy map."""
        occupancy_map = getattr(self.agent, "room_segmentation_map", None)
        if occupancy_map is None or occupancy_map.numel() == 0:
            self.room_debug = None
            return self.room_node_lst

        map_slice = occupancy_map[0, 0].detach().cpu().numpy()
        free_space_mask = (map_slice > 0.0) & (map_slice < 2.0)
        obstacle_mask = np.isclose(map_slice, 2.0)

        dilation_radius = delta if delta is not None else getattr(self.agent, "room_segmentation_delta", 30)
        dilation_radius = max(0, int(dilation_radius))
        structure = np.ones((3, 3), dtype=bool)

        if dilation_radius > 0:
            eroded_free_space = binary_erosion(free_space_mask, structure=structure, iterations=dilation_radius)
        else:
            eroded_free_space = free_space_mask

        separated_free = eroded_free_space
        self.room_debug = {
            "free_space_mask": free_space_mask,
            "eroded_free_space": eroded_free_space,
            "separated_free": separated_free
        }
        if not separated_free.any():
            return self.room_node_lst

        labeled_map, room_cnt = label(separated_free.astype(np.uint8), structure=structure.astype(np.uint8))
        if room_cnt == 0:
            return self.room_node_lst

        room_candidates = []
        for room_idx in range(1, room_cnt + 1):
            component_mask = labeled_map == room_idx
            if not component_mask.any():
                continue

            if dilation_radius > 0:
                expanded_mask = binary_dilation(component_mask, structure=structure, iterations=dilation_radius)
            else:
                expanded_mask = component_mask

            expanded_mask = expanded_mask & free_space_mask
            if not expanded_mask.any():
                expanded_mask = component_mask

            coords_rc = np.column_stack(np.where(expanded_mask))
            if coords_rc.size == 0:
                continue

            center_rc = coords_rc.mean(axis=0)
            center = (float(center_rc[1]), float(center_rc[0]))
            region_points = np.stack((coords_rc[:, 1], coords_rc[:, 0]), axis=1).astype(np.int32)

            coords_set = set(map(lambda pt: (int(pt[0]), int(pt[1])), region_points.tolist()))
            room_candidates.append({
                "coords": region_points,
                "coords_set": coords_set,
                "center": center,
                "area": len(region_points)
            })

        min_area = self.agent.min_room_area
        merged_rooms = self._merge_small_rooms(room_candidates, min_area=min_area)
        updated_rooms = self._update_room_nodes(merged_rooms)
        self._snap_rooms_to_skeleton(updated_rooms)
        return updated_rooms

    def _merge_small_rooms(self, room_regions, min_area=30):
        if not room_regions:
            return []
        large_regions = [r for r in room_regions if r["area"] >= min_area]
        small_regions = [r for r in room_regions if r["area"] < min_area]

        if not large_regions and small_regions:
            small_regions = sorted(small_regions, key=lambda r: r["area"], reverse=True)
            large_regions.append(small_regions.pop(0))

        def merge_region(target, source):
            target["coords_set"].update(source["coords_set"])
            target["coords"] = np.array(list(target["coords_set"]), dtype=np.int32)
            target["area"] = len(target["coords"])
            if target["area"] > 0:
                target["center"] = (
                    float(target["coords"][:, 0].mean()),
                    float(target["coords"][:, 1].mean())
                )

        for region in small_regions:
            if not large_regions:
                break
            dists = [
                np.linalg.norm(
                    np.array(region["center"]) - np.array(target["center"])
                ) for target in large_regions
            ]
            target_idx = int(np.argmin(dists))
            merge_region(large_regions[target_idx], region)

        return large_regions

    def _update_room_nodes(self, room_regions, iou_thresh=0.3):
        if room_regions is None:
            return self.room_node_lst

        prev_rooms = self.room_node_lst
        updated_rooms = []
        matched_prev = set()

        for region in room_regions:
            coords = region["coords"]
            coords_set = region["coords_set"]
            best_node = None
            best_iou = 0.0
            best_overlap = 0
            for node in prev_rooms:
                iou, overlap = self._compute_iou_and_overlap(node.coords_set(), coords_set)
                if iou > best_iou:
                    best_iou = iou
                    best_overlap = overlap
                    best_node = node
            if best_node is not None and (best_iou >= iou_thresh or best_overlap > 0):
                best_node.update_region(coords, union=True)
                matched_prev.add(best_node.id)
                updated_rooms.append(best_node)
            else:
                new_node = RoomNode(
                    room_id=self.room_id_counter,
                    region_points=coords
                )
                self.room_id_counter += 1
                updated_rooms.append(new_node)

        for node in prev_rooms:
            if node.id not in matched_prev:
                updated_rooms.append(node)

        self.room_node_lst = self._merge_small_room_nodes_by_distance(
            updated_rooms,
            min_area=self.agent.min_room_area
        )
        return self.room_node_lst

    def _merge_small_room_nodes_by_distance(self, nodes, min_area):
        if not nodes:
            return []
        large_nodes = [node for node in nodes if node.area >= min_area]
        small_nodes = [node for node in nodes if node.area < min_area]

        if not large_nodes and small_nodes:
            large_nodes.append(small_nodes.pop(0))

        for node in small_nodes:
            if not large_nodes:
                break
            distances = [
                self._euclidean_distance(node.center, target.center)
                for target in large_nodes
            ]
            target_idx = int(np.argmin(distances))
            target_node = large_nodes[target_idx]
            target_node.update_region(node.region_points, union=True)

        return large_nodes

    def _euclidean_distance(self, a, b):
        ax, ay = a
        bx, by = b
        return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

    def _compute_skeleton_intersections(self):
        if not hasattr(self, "skeleton_processor"):
            return []
        occupancy_map = getattr(self.agent, "occupancy_map", None)
        if occupancy_map is None:
            return []
        if hasattr(occupancy_map, "numel") and occupancy_map.numel() == 0:
            return []
        try:
            key_points, _, _, endpoints, junctions = self.skeleton_processor.process_image(
                occupancy_map=occupancy_map
            )
        except Exception as exc:
            print(f"[SceneGraph] Skeleton processing failed: {exc}")
            return []

        candidate_points = []
        if junctions:
            candidate_points.extend(junctions)
        if not candidate_points and endpoints:
            candidate_points.extend(endpoints)
        if not candidate_points and key_points:
            candidate_points.extend(key_points)

        intersections = []
        for point in candidate_points:
            if not point:
                continue
            coords = point[0]
            if isinstance(coords, (list, tuple, np.ndarray)) and len(coords) >= 2:
                intersections.append((float(coords[0]), float(coords[1])))
        return intersections

    def _select_skeleton_point_for_room(self, room, intersections):
        if not intersections:
            return None
        best_point = None
        best_dist = float("inf")
        region = getattr(room, "region_set", None)

        if region:
            for point in intersections:
                key = (int(round(point[0])), int(round(point[1])))
                if key in region:
                    dist = self._euclidean_distance(point, room.center)
                    if dist < best_dist:
                        best_dist = dist
                        best_point = point
            if best_point is not None:
                return best_point

        for point in intersections:
            dist = self._euclidean_distance(point, room.center)
            if dist < best_dist:
                best_dist = dist
                best_point = point
        return best_point

    def _snap_rooms_to_skeleton(self, rooms):
        if not rooms:
            return
        intersections = self._compute_skeleton_intersections()
        if not intersections:
            return
        for room in rooms:
            if getattr(room, "center_locked", False):
                continue
            snapped = self._select_skeleton_point_for_room(room, intersections)
            if snapped is not None:
                room.lock_center(snapped)

    def _compute_iou_and_overlap(self, set_a, set_b):
        if not set_a or not set_b:
            return 0.0, 0
        intersection = len(set_a & set_b)
        if intersection == 0:
            return 0.0, 0
        union = len(set_a | set_b)
        if union == 0:
            return 0.0, intersection
        return intersection / union, intersection
                    
if __name__ == "__main__":
    scene_graph = SceneGraph()

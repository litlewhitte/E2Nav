import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
import numpy as np
import kornia

import utils.depth_utils as du

import cv2
import time
import math

class SemanticMapper(nn.Module):

    """
    SemanticMapper
    """

    def __init__(self, agent, max_height=None, min_height=None, num_cats=1, pcd_threshold=0.5):
        super(SemanticMapper, self).__init__()

        self.device = agent.device
        self.screen_h = agent.config.SIMULATOR.DEPTH_SENSOR.HEIGHT
        self.screen_w = agent.config.SIMULATOR.DEPTH_SENSOR.WIDTH
        self.agent_height = agent.config.SIMULATOR.AGENT_0.HEIGHT*100.
        self.resolution = agent.resolution
        self.z_resolution = 5
        self.map_size_cm = agent.map_size_cm
        self.vision_range = 100
        self.dropout = 0.5
        self.fov = agent.config.SIMULATOR.DEPTH_SENSOR.HFOV
        self.du_scale = 1
        self.exp_pred_threshold = 10
        self.map_pred_threshold = 10
        self.max_z_consider = self.agent_height + 1 + 50
        self.min_z_consider = 20
        if max_height is not None:
            self.max_z_consider = max_height
            self.min_z_consider = min_height
        self.pcd_threshold = pcd_threshold
            
        self.view_angles = [0.0] #TODO: what is the view angle? 
        
        self.max_height = int(250 / self.z_resolution)
        self.min_height = int(-150 / self.z_resolution)
        self.shift_loc = [self.vision_range * self.resolution // 2, 0, np.pi/2.0]
        self.camera_matrix = du.get_camera_matrix(self.screen_w, self.screen_h, self.fov)

        vr = 100

        if num_cats is None:
            num_cats = 1
        self.init_grid = torch.zeros(1, num_cats, vr, vr,
                                self.max_height - self.min_height).float().to(self.device)
        self.feat = torch.ones(1,num_cats,
                          int(self.screen_h//self.du_scale) * int(self.screen_w//self.du_scale)
                         ).float().to(self.device)
        
    def set_view_angles(self, view_angle):
        self.view_angles[0] = -view_angle


    def forward(self, depth, pose_obs, agent_z_pos, maps_last, type_mask=None, type_prob=None):
        if type_mask is not None:
            return self.forward_(depth, pose_obs, maps_last, type_mask, type_prob)
        
        depth = torch.unsqueeze(depth, 0) * 100
        pose_obs = torch.unsqueeze(pose_obs, 0)

        # point_cloud_t = du.get_point_cloud_from_z_t(depth, self.camera_matrix, self.device, scale=self.du_scale)
        # #Multiprocessing
        # agent_view_t = du.transform_camera_view_t_multiple(point_cloud_t, self.agent_height, self.view_angles, self.device)

        # agent_view_centered_t = du.transform_pose_t(agent_view_t, self.shift_loc, self.device)
        
        point_cloud_t = du.get_point_cloud_from_z_t(depth, self.camera_matrix, self.device, scale=self.du_scale)
        
        current_camera_height = self.agent_height + agent_z_pos * 100.

        #Multiprocessing
        agent_view_t = du.transform_camera_view_t_multiple(point_cloud_t, current_camera_height, self.view_angles, self.device)

        agent_view_centered_t = du.transform_pose_t(agent_view_t, self.shift_loc, self.device)

        def pose_transform(pose_obs, agent_view_centered_t):
            pose_obs = pose_obs.clone()
            agent_view_centered_t = agent_view_centered_t.clone()
            pose_obs[0, 0] = pose_obs[0, 0] * 100
            pose_obs[0, 1] = pose_obs[0, 1] * 100
            pose_obs[0, 2] = pose_obs[0, 2] - 90
            pose_obs[0, 2] = pose_obs[0, 2] / 57.29577951308232
            x = pose_obs[0, 0].item()
            y = pose_obs[0, 1].item()
            t = pose_obs[0, 2].item()
            pose_matrix = torch.tensor([
                [np.cos(t), -np.sin(t), 0, x],
                [np.sin(t), np.cos(t), 0, y],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ], dtype=agent_view_centered_t.dtype, device=agent_view_centered_t.device)
            agent_view_centered_t = agent_view_centered_t.permute(3, 0, 1, 2)
            ones = torch.ones(agent_view_centered_t.shape[1:], device=agent_view_centered_t.device).unsqueeze(0)
            agent_view_centered_t = torch.cat([agent_view_centered_t, ones], dim=0)
            shape = agent_view_centered_t.shape
            agent_view_centered_t = agent_view_centered_t.reshape(4, -1)
            point_cloud_world = pose_matrix @ agent_view_centered_t
            point_cloud_world = point_cloud_world.reshape(shape)
            point_cloud_world = point_cloud_world[:3]
            point_cloud_world = point_cloud_world.permute(1, 2, 3, 0)
            return point_cloud_world
        
        point_cloud_world = pose_transform(pose_obs, agent_view_centered_t)

        max_h = self.max_height
        min_h = self.min_height
        xy_resolution = self.resolution
        z_resolution = self.z_resolution
        vision_range = self.vision_range
        XYZ_cm_std = agent_view_centered_t.float()
        XYZ_cm_std[...,:2] = (XYZ_cm_std[...,:2] / xy_resolution)
        XYZ_cm_std[...,:2] = (XYZ_cm_std[...,:2] - vision_range//2.)/vision_range*2.
        XYZ_cm_std[...,2] = XYZ_cm_std[...,2] / z_resolution
        XYZ_cm_std[...,2] = (XYZ_cm_std[...,2] -
                             (max_h+min_h)//2.)/(max_h-min_h)*2.


        XYZ_cm_std = XYZ_cm_std.permute(0,3,1,2)
        XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0],
                                     XYZ_cm_std.shape[1],
                                     XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3])

        voxels = du.splat_feat_nd(self.init_grid*0., self.feat, XYZ_cm_std).transpose(2,3)

        min_z = int(self.min_z_consider/z_resolution - min_h)
        max_z = int(self.max_z_consider/z_resolution - min_h) 

        agent_height_proj = voxels[...,min_z:max_z].sum(4)
        all_height_proj = voxels.sum(4)

        fp_map_pred = agent_height_proj[:,0:1,:,:]
        fp_exp_pred = all_height_proj[:,0:1,:,:]
        fp_map_pred = fp_map_pred/self.map_pred_threshold
        fp_exp_pred = fp_exp_pred/self.exp_pred_threshold
        fp_map_pred = torch.clamp(fp_map_pred, min=0.0, max=1.0)
        fp_exp_pred = torch.clamp(fp_exp_pred, min=0.0, max=1.0)

        agent_view = torch.zeros(1, 1,
                                 int(self.map_size_cm//self.resolution),
                                 int(self.map_size_cm//self.resolution)
                                 ).to(self.device)

        x1 = int(self.map_size_cm//(self.resolution * 2) - self.vision_range//2)
        x2 = int(x1 + self.vision_range)
        y1 = int(self.map_size_cm//(self.resolution * 2))
        y2 = int(y1 + self.vision_range)
        agent_view[:,0, y1:y2, x1:x2] = fp_map_pred
        # agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred

        def get_new_pose_batch(pose, rel_pose_change):

            pose[:,1] += rel_pose_change[:,0] * \
                            torch.sin(pose[:,2]/57.29577951308232) \
                        + rel_pose_change[:,1] * \
                            torch.cos(pose[:,2]/57.29577951308232)
            pose[:,0] += rel_pose_change[:,0] * \
                            torch.cos(pose[:,2]/57.29577951308232) \
                        - rel_pose_change[:,1] * \
                            torch.sin(pose[:,2]/57.29577951308232)
            pose[:,2] += rel_pose_change[:,2]*57.29577951308232

            pose[:,2] = torch.fmod(pose[:,2]-180.0, 360.0)+180.0
            pose[:,2] = torch.fmod(pose[:,2]+180.0, 360.0)-180.0

            return pose

        current_poses = pose_obs  # get_new_pose_batch(poses_last, pose_obs)
        st_pose = current_poses.clone().detach()

        st_pose[:, :2] = - (st_pose[:, :2]
                                * 100.0/self.resolution
                                - self.map_size_cm//(self.resolution*2)) /\
                                (self.map_size_cm//(self.resolution*2))
        st_pose[:, 2] = 90. - (st_pose[:, 2])

        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(),
                                        self.device)

        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)

        maps2 = torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1)
        map_pred, _ = torch.max(maps2,1)

        map_pred[map_pred > self.pcd_threshold] = 1
        map_pred[map_pred <= self.pcd_threshold] = 0
        # map_pred[map_pred > 0] = 1

        return map_pred
    
    def forward_(self, depth, pose_obs, maps_last, type_mask=None, type_prob=None):
        depth = torch.unsqueeze(depth, 0) * 100
        type_mask = torch.unsqueeze(type_mask, 0)
        pose_obs = torch.unsqueeze(pose_obs, 0)

        bs, c, h, w = type_mask.size()
        point_cloud_t = du.get_point_cloud_from_z_t(depth, self.camera_matrix, self.device, scale=self.du_scale)
        
        agent_view_t = du.transform_camera_view_t_multiple(point_cloud_t, self.agent_height, self.view_angles, self.device)
        agent_view_centered_t = du.transform_pose_t(agent_view_t, self.shift_loc, self.device)

        self.feat = nn.MaxPool2d(self.du_scale)(type_mask.float())
        self.feat = self.feat.view(bs, c, int(h//self.du_scale) * int(w//self.du_scale))
        
        XYZ_cm_std = agent_view_centered_t.float()
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] / self.resolution - self.vision_range//2.) / self.vision_range * 2.
        XYZ_cm_std[..., 2] = (XYZ_cm_std[..., 2]/self.z_resolution - (self.max_height+self.min_height)//2.) / (self.max_height-self.min_height) * 2.
        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2).view(XYZ_cm_std.shape[0], XYZ_cm_std.shape[3], -1)
        
        voxels = du.splat_feat_nd(self.init_grid*0., (self.feat > 0).float(), XYZ_cm_std).transpose(2, 3)
        voxels = (voxels > 0).float()

        min_z = int(self.min_z_consider/self.z_resolution - self.min_height)
        max_z = int(self.max_z_consider/self.z_resolution - self.min_height)
        agent_height_proj = (voxels[..., min_z:max_z].sum(4) > 0).float()

        agent_view = torch.zeros(bs, c, int(self.map_size_cm//self.resolution), 
                                int(self.map_size_cm//self.resolution), device=self.device)
        x1 = int(self.map_size_cm//(self.resolution * 2) - self.vision_range//2)
        x2 = int(x1 + self.vision_range)
        y1 = int(self.map_size_cm//(self.resolution * 2))
        y2 = int(y1 + self.vision_range)
        agent_view[:, :, y1:y2, x1:x2] = agent_height_proj

        current_poses = pose_obs
        st_pose = current_poses.clone().detach()
        st_pose[:, :2] = - (st_pose[:, :2]*100/self.resolution - self.map_size_cm//(self.resolution*2)) / (self.map_size_cm//(self.resolution*2))
        st_pose[:, 2] = 90. - st_pose[:, 2]
        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(), self.device)
        
        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)
        
        translated = (translated > 0).float()
        maps2 = torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1)
        map_pred, _ = torch.max(maps2, 1)
        
        map_pred[map_pred > 0.5] = 1

        return map_pred




def get_grid(pose, grid_size, device):
    """
    Input:
        `pose` FloatTensor(bs, 3)
        `grid_size` 4-tuple (bs, _, grid_h, grid_w)
        `device` torch.device (cpu or gpu)
    Output:
        `rot_grid` FloatTensor(bs, grid_h, grid_w, 2)
        `trans_grid` FloatTensor(bs, grid_h, grid_w, 2)
    """
    pose = pose.float()
    x = pose[:, 0]
    y = pose[:, 1]
    t = pose[:, 2]

    bs = x.size(0)
    t = t * np.pi / 180.
    cos_t = t.cos()
    sin_t = t.sin()

    theta11 = torch.stack([cos_t, -sin_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta12 = torch.stack([sin_t, cos_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta1 = torch.stack([theta11, theta12], 1)

    theta21 = torch.stack([torch.ones(x.shape).to(device),
                           -torch.zeros(x.shape).to(device), x], 1)
    theta22 = torch.stack([torch.zeros(x.shape).to(device),
                           torch.ones(x.shape).to(device), y], 1)
    theta2 = torch.stack([theta21, theta22], 1)

    rot_grid = F.affine_grid(theta1, torch.Size(grid_size))  # annotated by someone
    trans_grid = F.affine_grid(theta2, torch.Size(grid_size))  # annotated by someone
    # rot_grid = F.affine_grid(theta1, torch.Size(grid_size), align_corners=False)
    # trans_grid = F.affine_grid(theta2, torch.Size(grid_size), align_corners=False)

    return rot_grid, trans_grid


def splat_feat_max_nd(init_grid, feat, coords):
    """Pool the maximum feature value in each voxel."""
    b, c, w, h, d = init_grid.shape
    b, c, n = feat.shape
    
    coords = (coords + 1) / 2.
    coords[..., 0] *= (w - 1)
    coords[..., 1] *= (h - 1)
    coords[..., 2] *= (d - 1)
    coords = coords.round().long()

    idx_x = coords[..., 0].unsqueeze(1).expand(-1, c, -1)
    idx_y = coords[..., 1].unsqueeze(1).expand(-1, c, -1)
    idx_z = coords[..., 2].unsqueeze(1).expand(-1, c, -1)

    flat_index = idx_x * h * d + idx_y * d + idx_z
    
    grid_flat = init_grid.view(b, c, -1)
    
    grid_flat.scatter_reduce_(2, flat_index, feat, reduce="amax", include_self=True)
    
    return grid_flat.view(b, c, w, h, d)



class ElevationMapper(nn.Module):
    """
    ElevationMapper
    Generates and updates a 2D elevation map from depth images.
    The value of each grid cell represents the height of the obstacle.
    """

    def __init__(self, agent, max_height=None, min_height=None, slice_point_threshold=3):
        super(ElevationMapper, self).__init__()

        self.device = agent.device
        self.screen_h = agent.config.SIMULATOR.DEPTH_SENSOR.HEIGHT
        self.screen_w = agent.config.SIMULATOR.DEPTH_SENSOR.WIDTH
        self.agent_height = agent.config.SIMULATOR.AGENT_0.HEIGHT * 100.
        self.resolution = agent.resolution
        self.z_resolution = 5
        self.map_size_cm = agent.map_size_cm
        self.vision_range = 100
        self.fov = agent.config.SIMULATOR.DEPTH_SENSOR.HFOV
        self.du_scale = 1

        self.max_z_consider = self.agent_height + 1 + 50
        self.min_z_consider = -100
        if max_height is not None:
            self.max_z_consider = max_height
        if min_height is not None:
            self.min_z_consider = min_height

        self.slice_point_threshold = slice_point_threshold
            
        self.view_angles = [0.0]
        
        # Voxel grid Z dimension range
        self.max_height_voxel_idx = int(250 / self.z_resolution)
        self.min_height_voxel_idx = int(-150 / self.z_resolution)
        self.shift_loc = [self.vision_range * self.resolution // 2, 0, np.pi / 2.0]
        self.camera_matrix = du.get_camera_matrix(self.screen_w, self.screen_h, self.fov)

        vr = self.vision_range
        self.init_grid = torch.zeros(1, 1, vr, vr,
                                self.max_height_voxel_idx - self.min_height_voxel_idx).float().to(self.device)
        self.feat = torch.ones(1, 1,
                          int(self.screen_h // self.du_scale) * int(self.screen_w // self.du_scale)
                         ).float().to(self.device)
        
    def set_view_angles(self, view_angle):
        self.view_angles[0] = -view_angle

    def forward(self, depth, pose_obs, agent_z_pos, maps_last):
        if depth.dim() == 2:
            depth = torch.unsqueeze(depth, 0)
        if pose_obs.dim() == 1:
            pose_obs = torch.unsqueeze(pose_obs, 0)

        depth_cm = depth * 100
        
        point_cloud_t = du.get_point_cloud_from_z_t(depth_cm, self.camera_matrix, self.device, scale=self.du_scale)
        agent_view_t = du.transform_camera_view_t_multiple(point_cloud_t, self.agent_height + agent_z_pos*100, self.view_angles, self.device)
        agent_view_centered_t = du.transform_pose_t(agent_view_t, self.shift_loc, self.device)
        

        max_h, min_h = self.max_height_voxel_idx, self.min_height_voxel_idx
        
        XYZ_cm_std = agent_view_centered_t.float()
        XYZ_cm_std[...,:2] = (XYZ_cm_std[...,:2] / self.resolution)
        XYZ_cm_std[...,:2] = (XYZ_cm_std[...,:2] - self.vision_range//2.) / self.vision_range * 2.
        XYZ_cm_std[...,2] = XYZ_cm_std[...,2] / self.z_resolution
        XYZ_cm_std[...,2] = (XYZ_cm_std[...,2] - (max_h+min_h)//2.) / (max_h-min_h) * 2.

        XYZ_cm_std = XYZ_cm_std.permute(0,3,1,2)
        XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0], XYZ_cm_std.shape[1], -1)

        voxels = du.splat_feat_nd(self.init_grid * 0., self.feat, XYZ_cm_std).transpose(2,3)
        
        
        B, C, W, H, D = voxels.shape

        # is_obstacle_voxel: (B, C, W, H, D) -> boolean tensor
        is_obstacle_voxel = voxels > self.slice_point_threshold

        # z_indices: (D,) -> (1, 1, 1, 1, D) -> (B, C, W, H, D)
        z_indices = torch.arange(D, device=self.device, dtype=torch.float32)
        z_indices = z_indices.view(1, 1, 1, 1, D).expand(B, C, W, H, D)

        min_z_idx = int(self.min_z_consider / self.z_resolution) - self.min_height_voxel_idx
        max_z_idx = int(self.max_z_consider / self.z_resolution) - self.min_height_voxel_idx

        # z_indices_mask: (B, C, W, H, D) -> boolean tensor
        z_indices_mask = (z_indices >= min_z_idx) & (z_indices <= max_z_idx)

        # combined_mask: (B, C, W, H, D) -> boolean tensor
        combined_mask = is_obstacle_voxel & z_indices_mask

        valid_z_indices = torch.where(combined_mask, z_indices, 0)

        # local_elevation_map: (B, C, W, H)
        local_elevation_map, _ = torch.max(valid_z_indices, dim=4)
        
        
        # B, C, W, H, D = voxels.shape
        
        # # is_obstacle_voxel: (B, C, W, H, D) -> boolean tensor
        # is_obstacle_voxel = voxels > self.slice_point_threshold

        # # z_indices: (D,) -> (1, 1, 1, 1, D) -> (B, C, W, H, D)
        # z_indices = torch.arange(D, device=self.device, dtype=torch.float32)
        # z_indices = z_indices.view(1, 1, 1, 1, D).expand(B, C, W, H, D)

        # valid_z_indices = torch.where(is_obstacle_voxel, z_indices, 0)
        
        # # local_elevation_map: (B, C, W, H)
        # local_elevation_map, _ = torch.max(valid_z_indices, dim=4)
        
        # min_z_idx = int(self.min_z_consider / self.z_resolution) - self.min_height_voxel_idx
        # max_z_idx = int(self.max_z_consider / self.z_resolution) - self.min_height_voxel_idx
        
        # # local_elevation_map[local_elevation_map < min_z_idx] = 0
        # # local_elevation_map.clamp_(max=max_z_idx)
        
        # valid_range_mask = (local_elevation_map >= min_z_idx) & (local_elevation_map <= max_z_idx)
        # local_elevation_map = local_elevation_map * valid_range_mask.float()

        agent_view = torch.zeros(1, 1,
                                 int(self.map_size_cm // self.resolution),
                                 int(self.map_size_cm // self.resolution),
                                 dtype=torch.float32).to(self.device)

        x1 = int(self.map_size_cm // (self.resolution * 2) - self.vision_range // 2)
        x2 = int(x1 + self.vision_range)
        y1 = int(self.map_size_cm // (self.resolution * 2))
        y2 = int(y1 + self.vision_range)
        
        agent_view[:, :, y1:y2, x1:x2] = local_elevation_map

        st_pose = pose_obs.clone().detach()
        st_pose[:, :2] = - (st_pose[:, :2] * 100.0 / self.resolution - self.map_size_cm // (self.resolution * 2)) / (self.map_size_cm // (self.resolution * 2))
        st_pose[:, 2] = 90. - st_pose[:, 2]
        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(), self.device)

        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)

        
        maps2 = torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1)
        map_pred, _ = torch.max(maps2, 1)
        map_pred[(translated==0) & (maps_last==-300)] = -300

        return map_pred


def get_gaussian_kernel(kernel_size=5, sigma=1.5, channels=1, device='cpu'):
    x_cord = torch.arange(kernel_size, device=device)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2.
    variance = sigma**2.

    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) / \
                          (2 * variance)
                      )
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    
    return gaussian_kernel




class ElevationGradientCalculator(nn.Module):
    """Classify terrain by elevation gradient."""
    def __init__(self, 
                 agent,
                 gradient_threshold=15,
                 z_resolution=5, 
                 min_voxel_idx_offset=-30,
                 unobserved_value=-300,
                 morph_kernel_size=5,
                 bilateral_kernel_size=7,
                 sigma_color=10.0,
                 sigma_space=3.0):
        """Initialize the terrain gradient classifier."""
        super(ElevationGradientCalculator, self).__init__()
        self.device = agent.device
        self.gradient_threshold = gradient_threshold
        self.z_resolution = z_resolution
        self.min_voxel_idx_offset = min_voxel_idx_offset
        self.unobserved_value = unobserved_value
        
        # self.bilateral_kernel_size = bilateral_kernel_size
        # self.sigma_color = sigma_color
        # self.sigma_space = sigma_space
        
        self.morph_kernel = torch.ones(morph_kernel_size, morph_kernel_size, device=self.device)
        # self.median_kernel_size = median_kernel_size
        # self.gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=1.5, device=self.device)
        
        sobel_x = torch.tensor([[-1., 0., 1.], 
                                [-2., 0., 2.], 
                                [-1., 0., 1.]], device=self.device)
        sobel_y = torch.tensor([[-1., -2., -1.], 
                                [ 0.,  0.,  0.], 
                                [ 1.,  2.,  1.]], device=self.device)

        self.sobel_x_kernel = sobel_x.view(1, 1, 3, 3)
        self.sobel_y_kernel = sobel_y.view(1, 1, 3, 3)
        
        # self.neighbor_kernel = torch.ones(1, 1, 5, 5, device=self.device)

    def forward(self, elevation_map_tensor):
        """Convert an elevation map to a steepness map."""
        input_map = elevation_map_tensor.clone()
        if input_map.dim() == 2:
            input_map = input_map.unsqueeze(0)
        if input_map.dim() == 4 and input_map.shape[1] == 1:
            input_map = input_map.squeeze(1)

        observed_mask = (input_map != self.unobserved_value)

        height_map_cm = (input_map + self.min_voxel_idx_offset) * self.z_resolution
        # height_map_cm[~observed_mask] = 0.0
        
        # height_map_cm_4d = height_map_cm.unsqueeze(1)
        # denoised_height_map_4d = kornia.filters.bilateral_blur(
        #     height_map_cm_4d,
        #     kernel_size=(self.bilateral_kernel_size, self.bilateral_kernel_size),
        #     sigma_color=self.sigma_color,
        #     sigma_space=(self.sigma_space, self.sigma_space)
        # )
        # denoised_height_map = denoised_height_map_4d.squeeze(1)
        # denoised_height_map[~observed_mask] = 0.0
        # grad_input_4d = denoised_height_map.unsqueeze(1)
        # grad_x = F.conv2d(grad_input_4d, self.sobel_x_kernel, padding='same')
        # grad_y = F.conv2d(grad_input_4d, self.sobel_y_kernel, padding='same')
        
        height_map_cm_4d = height_map_cm.unsqueeze(1)
        dilated_map = kornia.morphology.dilation(height_map_cm_4d, self.morph_kernel)
        closed_map = kornia.morphology.erosion(dilated_map, self.morph_kernel)
        denoised_height_map = closed_map.squeeze(1)
        # denoised_height_map[~observed_mask] = 0.0
        grad_input_4d = denoised_height_map.unsqueeze(1)
        grad_x = F.conv2d(grad_input_4d, self.sobel_x_kernel, padding='same')
        grad_y = F.conv2d(grad_input_4d, self.sobel_y_kernel, padding='same')

        # height_map_cm_4d = height_map_cm.unsqueeze(1)
        # grad_x = F.conv2d(height_map_cm_4d, self.sobel_x_kernel, padding='same')
        # grad_y = F.conv2d(height_map_cm_4d, self.sobel_y_kernel, padding='same')
        
        # height_map_cm_4d_for_filter = height_map_cm.unsqueeze(1)
        # denoised_height_map_4d = kf.median_blur(height_map_cm_4d_for_filter, (self.median_kernel_size, self.median_kernel_size))
        # denoised_height_map = denoised_height_map_4d.squeeze(1)
        # denoised_height_map[~observed_mask] = 0.0
        # grad_input_4d = denoised_height_map.unsqueeze(1)
        # grad_x = F.conv2d(grad_input_4d, self.sobel_x_kernel, padding='same')
        # grad_y = F.conv2d(grad_input_4d, self.sobel_y_kernel, padding='same')

        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        grad_magnitude = grad_magnitude.squeeze(1)

        
        classified_map = torch.zeros_like(grad_magnitude, dtype=torch.long, device=self.device)
        
        flat_mask = (grad_magnitude < self.gradient_threshold) & observed_mask
        classified_map[flat_mask] = 1
        
        steep_mask = (grad_magnitude >= self.gradient_threshold) & observed_mask
        classified_map[steep_mask] = 2
        

        return classified_map.unsqueeze(1)
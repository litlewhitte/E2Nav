"""Map rendering and episode observation output for the navigation agent."""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw


class NavigationVisualization:
    """Render maps and observations using the current agent state."""

    def _draw_path_on_image(self, image, path=None, color=(255, 0, 0), width=3):
        if path is None:
            path = self.temp_path
        if not path or len(path) < 2:
            return
        draw = ImageDraw.Draw(image)
        draw.line(path, fill=color, width=width)

    def visualize_path(self, image_path, start, end, save_path, color=(255, 0, 0)):
        if self.temp_path is not None and len(self.temp_path) > 1:
            img = Image.open(image_path)
            self._draw_path_on_image(img, self.temp_path, color=(255, 0, 0), width=3)
            img.save(save_path)
            print(f"Results saved to {save_path}")

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
        map_width = 800
        map_data = map[0,0].detach().cpu().numpy()
        compass = observations.get("compass", [0.0])

        pixel_x = round(self.center_pixel + self.gps[0] * 100 / self.resolution)
        pixel_y = round(self.center_pixel - self.gps[1] * 100 / self.resolution)

        color_map = np.zeros((*map_data.shape, 3), dtype=np.uint8)
        free_color = (255, 255, 255)
        obstacle_color = (0, 0, 0)
        unexplored_color = (200, 200, 200)
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
            # agent_pos = self.simulator._env._task.measurements.measures['distance_to_goal']._sim.get_agent_state().position
            # cv2.circle(color_map,
            #         (300+int(20*agent_pos[2]), 300+int(20*agent_pos[0])),
            #         radius=5,
            #         thickness=-1)

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
        if self.total_steps == 17:
            self.draw_points_on_image(map_path, self.episode_view_points, map_path)


    def _build_room_map_canvas(self):
        room_map_tensor = getattr(self, "room_segmentation_map", None)
        if room_map_tensor is None or room_map_tensor.numel() == 0:
            return None
        map_data = room_map_tensor[0, 0].detach().cpu().numpy()
        color_map = np.zeros((*map_data.shape, 3), dtype=np.uint8)
        free_color = (255, 255, 255)
        obstacle_color = (0, 0, 0)
        unexplored_color = (200, 200, 200)
        stairs_color = (225, 127, 80)
        color_map[map_data == 0.0] = unexplored_color
        color_map[(map_data > 0.0) & (map_data < 2.0) & (map_data != 1.35)] = free_color
        color_map[map_data == 1.35] = stairs_color
        color_map[map_data == 2.0] = obstacle_color
        return color_map

    def draw_room_regions(self, room_nodes):
        if not hasattr(self, "room_dir") or self.room_dir is None:
            return
        base_map = self._build_room_map_canvas()
        if base_map is None:
            return
        room_map = base_map.copy()
        height, width = room_map.shape[:2]

        if room_nodes is None:
            room_nodes = []

        def _color_from_id(idx):
            np.random.seed(idx * 7919)
            color = np.random.randint(64, 255, size=3, dtype=np.uint8)
            return int(color[0]), int(color[1]), int(color[2])

        for room in room_nodes:
            if getattr(room, "region_points", None) is None or len(room.region_points) == 0:
                continue
            coords = room.region_points
            valid_mask = (
                (coords[:, 0] >= 0) & (coords[:, 0] < width) &
                (coords[:, 1] >= 0) & (coords[:, 1] < height)
            )
            coords = coords[valid_mask]
            if coords.size == 0:
                continue
            color = getattr(room, "color", None)
            if color is None:
                color = _color_from_id(room.id)
            room_map[coords[:, 1], coords[:, 0]] = color
            center_x = int(np.clip(round(room.center[0]), 0, width - 1))
            center_y = int(np.clip(round(room.center[1]), 0, height - 1))
            cv2.circle(room_map, (center_x, center_y), 5, (0, 0, 255), -1)

            room_label = room.room_name if getattr(room, "room_name", None) is not None else f"R{room.id}"
            font_scale = 0.5
            thickness = 1
            text_size, _ = cv2.getTextSize(room_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_x = center_x + 8
            text_y = center_y - 5

            cv2.putText(
                room_map,
                room_label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness + 2,
                cv2.LINE_AA
            )
            cv2.putText(
                room_map,
                room_label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

        filename = os.path.join(self.room_dir, f"rooms_{self.total_steps:04d}.png")
        cv2.imwrite(filename, room_map)

    def draw_landmark_overview(self):
        if not hasattr(self, "e2_dir") or self.e2_dir is None:
            return
        if not hasattr(self, "occupancy_map") or self.occupancy_map is None:
            return
        if hasattr(self, "landmark_planner"):
            self.landmark_planner.update_object_landmarks()
            self.landmark_planner.update_room_landmarks()
        map_data = self.occupancy_map[0,0].detach().cpu().numpy()
        color_map = np.zeros((*map_data.shape, 3), dtype=np.uint8)
        unexplored_color = (200, 200, 200)
        free_color = (255, 255, 255)
        obstacle_color = (0, 0, 0)
        stairs_color = (225, 127, 80)
        color_map[map_data == 0.0] = unexplored_color
        color_map[(map_data > 0.0) & (map_data < 2.0) & (map_data != 1.35)] = free_color
        color_map[map_data == 1.35] = stairs_color
        color_map[map_data == 2.0] = obstacle_color

        pixel_x = round(self.center_pixel + self.gps[0] * 100 / self.resolution)
        pixel_y = round(self.center_pixel - self.gps[1] * 100 / self.resolution)
        cv2.circle(color_map, (pixel_x, pixel_y), 5, (255, 0, 255), -1)
        compass_val = self.observations.get("compass", [0.0])[0]
        arrow_length = 10
        dx = np.cos(compass_val) * arrow_length
        dy = np.sin(compass_val) * arrow_length
        end_x = round(pixel_x + dx)
        end_y = round(pixel_y + dy)
        cv2.arrowedLine(
            color_map,
            (pixel_x, pixel_y),
            (end_x, end_y),
            color=(0, 128, 0),
            thickness=1,
            tipLength=0.3
        )

        type_colors = {
            'O': (159, 134, 244),   # #F4869F in BGR
            'R': (0, 212, 0),       # #00D400 in BGR
            'F': (185, 66, 64)      # #4042B9 in BGR
        }

        if hasattr(self.landmark_planner, "landmark_lst"):
            top_landmark = None
            top_pos = None
            top_score = float("-inf")
            for idx, landmark in enumerate(self.landmark_planner.landmark_lst):
                if getattr(landmark, "pos", None) is None:
                    continue
                lx = int(np.clip(round(landmark.pos[0]), 0, color_map.shape[1]-1))
                ly = int(np.clip(round(landmark.pos[1]), 0, color_map.shape[0]-1))
                color = type_colors.get(getattr(landmark, "type", None), (255, 255, 255))
                cv2.circle(color_map, (lx, ly), 3, color, -1)
                score = getattr(landmark, "score", 0.0) or 0.0
                if score > top_score:
                    top_score = score
                    top_landmark = landmark
                    top_pos = (lx, ly)
                if landmark.type == 'R' and getattr(landmark, "name", None):
                    label = landmark.name
                    cv2.putText(
                        color_map,
                        label,
                        (lx + 5, ly - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (0, 0, 0),
                        3,
                        cv2.LINE_AA
                    )
                    cv2.putText(
                        color_map,
                        label,
                        (lx + 5, ly - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA
                    )
            if top_landmark is not None and top_pos is not None:
                cv2.circle(
                    color_map,
                    top_pos,
                    radius=8,
                    color=(0, 0, 255),
                    thickness=2
                )
                score_text = f"{top_score:.2f}"
                sx = top_pos[0] + 6
                sy = top_pos[1] + 14
                cv2.putText(
                    color_map,
                    score_text,
                    (sx, sy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA
                )

        filename = os.path.join(self.e2_dir, f"landmarks_{self.total_steps:04d}.png")
        cv2.imwrite(filename, color_map)

    def draw_room_dilation_debug(self, room_debug):
        if not hasattr(self, "room_dir") or self.room_dir is None:
            return
        if room_debug is None:
            return
        base_map = self._build_room_map_canvas()
        if base_map is None:
            return

        free_space_mask = room_debug.get("free_space_mask")
        eroded_free_space = room_debug.get("eroded_free_space")
        if free_space_mask is None or eroded_free_space is None:
            return
        eroded_difference = np.logical_and(free_space_mask, np.logical_not(eroded_free_space))
        debug_map = base_map.copy()
        debug_map[free_space_mask] = (255, 255, 255)
        debug_map[~free_space_mask] = (0, 0, 0)
        debug_map[eroded_difference] = (0, 255, 255)

        filename = os.path.join(self.room_dir, f"dilated_{self.total_steps:04d}.png")
        cv2.imwrite(filename, debug_map)

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

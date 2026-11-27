import cv2
import numpy as np
import sys
import math
import torch
import os
import time
from collections import deque

class SkeletonProcessor():
    def __init__(self, dilate_kernel=21, min_component_size=50, pruning_iterations=10, min_junction_distance=10, closing_kernel_size=11, hole_fill_area=200):
        self.dilate_kernel = dilate_kernel
        self.min_component_size = min_component_size
        self.pruning_iterations = pruning_iterations
        self.min_junction_distance = min_junction_distance
        self.hole_fill_area = hole_fill_area
        self.closing_kernel_size = closing_kernel_size
        self.pruned_skeleton = None
        self.endpoints = None
        self.merged_junctions = None

    def _calculate_crossing_number(self, neighborhood):
        # Ensure the center pixel is part of the skeleton (value 255)
        if neighborhood[1, 1] != 255:
            return 0
        # Define the 8 neighbors in order (clockwise or counter-clockwise)
        p_indices = [(0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0), (0, 0)]
        # Convert neighbor values to 0 or 1
        p = [1 if neighborhood[idx] == 255 else 0 for idx in p_indices]
        # Calculate crossing number: sum of absolute differences between adjacent neighbors, divided by 2
        cn = 0
        for i in range(8):
             cn += abs(p[i] - p[(i + 1) % 8])
        return cn // 2

    def _find_skeleton_keypoints(self, skeleton_img):
        if skeleton_img is None or skeleton_img.size == 0 or np.sum(skeleton_img) == 0:
             # Return empty lists if the input image is empty or all zero
             return [], []

        # Ensure the input is a binary grayscale image
        if len(skeleton_img.shape) > 2:
            skeleton_img = cv2.cvtColor(skeleton_img, cv2.COLOR_BGR2GRAY)
        # Threshold to ensure it's strictly binary 0 or 255
        _, skeleton_img_binary = cv2.threshold(skeleton_img, 127, 255, cv2.THRESH_BINARY)

        height, width = skeleton_img_binary.shape
        endpoints = []
        raw_junctions = []

        # Iterate only over potential skeleton pixels (excluding 1-pixel border)
        # to safely check 3x3 neighborhood
        rows, cols = np.where(skeleton_img_binary[1:height-1, 1:width-1] == 255)
        # Adjust indices back to the original image coordinates
        rows += 1
        cols += 1

        for r, c in zip(rows, cols):
            neighborhood = skeleton_img_binary[r - 1 : r + 2, c - 1 : c + 2]
            cn = self._calculate_crossing_number(neighborhood)
            if cn == 1:
                endpoints.append(((c, r), 1)) # Store as (x, y) coordinate tuple, with CN
            elif cn >= 3:
                raw_junctions.append(((c, r), cn)) # Store as (x, y) coordinate tuple, with CN

        return endpoints, raw_junctions

    def _merge_nearby_points_with_branches(self, points_with_cn):
        """
        Merges nearby points (typically junctions) within a specified distance,
        calculating the centroid and summing the branch counts.
        """
        if not points_with_cn:
            return []

        n = len(points_with_cn)
        # Disjoint Set Union (DSU) data structure
        parent = list(range(n))

        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i]) # Path compression
            return parent[i]

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_j] = root_i # Union by rank/size can be added for optimization, but simple union is fine for this scale

        min_distance_sq = self.min_junction_distance * self.min_junction_distance

        # Iterate through all pairs of points
        for i in range(n):
            for j in range(i + 1, n):
                p1_coords = points_with_cn[i][0] # (x, y)
                p2_coords = points_with_cn[j][0] # (x, y)
                dist_sq = (p1_coords[0] - p2_coords[0])**2 + (p1_coords[1] - p2_coords[1])**2
                if dist_sq < min_distance_sq:
                    union(i, j)

        # Group points by their root parent
        merged_groups = {}
        for i in range(n):
            root = find(i)
            if root not in merged_groups:
                merged_groups[root] = []
            merged_groups[root].append(i) # Store the index of the original point

        # Calculate centroid and sum branch counts for each merged group
        merged_points_with_cn = []
        for root, original_indices in merged_groups.items():
            sum_x = 0
            sum_y = 0
            total_branches_in_group = 0
            n_group = len(original_indices) # Number of original points in this group

            for idx in original_indices:
                coords, cn = points_with_cn[idx]
                sum_x += coords[0]
                sum_y += coords[1]
                # Note: cn for endpoints is 1, for junctions is >= 3.
                # When merging junctions, summing Cns gives a rough idea of the "total connectivity"
                # entering the merged region. It's not strictly the number of branches *from* the centroid,
                # but represents the sum of Cns of the original points.
                # If you only pass junctions to this function, it sums junction Cns.
                total_branches_in_group += cn

            # Calculate centroid - round to nearest integer pixel
            centroid_x = round(sum_x / n_group)
            centroid_y = round(sum_y / n_group)

            # Use the summed branch count
            merged_branch_count = total_branches_in_group

            merged_points_with_cn.append(((centroid_x, centroid_y), merged_branch_count))

        return merged_points_with_cn

    def _prune_skeleton(self, skeleton_img):
        """
        Prunes the skeleton by removing small components and short branches
        that don't lead to a junction or endpoint, avoiding junction points.
        """
        if skeleton_img is None or skeleton_img.size == 0 or np.sum(skeleton_img) == 0:
             print("Warning: Input skeleton image for pruning is empty.")
             # Return an empty image with the same shape if input is empty
             if skeleton_img is not None:
                 return np.zeros_like(skeleton_img)
             else:
                 return None # Cannot return zeros_like if input is None

        if len(skeleton_img.shape) > 2:
            skeleton_img = cv2.cvtColor(skeleton_img, cv2.COLOR_BGR2GRAY)
        _, skeleton_img_binary = cv2.threshold(skeleton_img, 127, 255, cv2.THRESH_BINARY)

        pruned_skeleton = skeleton_img_binary.copy()
        height, width = pruned_skeleton.shape

        # # 1. Remove small connected components
        # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pruned_skeleton, 8, cv2.CV_32S)
        # # Iterate from 1 to num_labels (label 0 is background)
        # for i in range(1, num_labels):
        #     area = stats[i, cv2.CC_STAT_AREA]
        #     if area < self.min_component_size:
        #         pruned_skeleton[labels == i] = 0 # Set pixels of small components to 0

        # # Re-get dimensions and handle empty skeleton after component removal
        # # This check is important if removing small components made the entire skeleton disappear
        # if np.sum(pruned_skeleton) == 0:
        #      print("Warning: Skeleton became empty after removing small components.")
        #      return pruned_skeleton # Return the now empty skeleton

        # 1. Remove all small connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skeleton_img, 8, cv2.CV_32S)
        if num_labels <= 1:
            return np.zeros_like(skeleton_img)

        areas = stats[1:, cv2.CC_STAT_AREA]
        longest_component_label = np.argmax(areas) + 1
        longest_skeleton = np.zeros(skeleton_img.shape, dtype=np.uint8)
        longest_skeleton[labels == longest_component_label] = 255
        pruned_skeleton = longest_skeleton

        # 2. Identify and protect junction points
        junction_mask = np.zeros_like(pruned_skeleton, dtype=np.uint8)
        # Check points within the inner area to have a valid 3x3 neighborhood
        rows, cols = np.where(pruned_skeleton[1:height-1, 1:width-1] == 255)
        rows += 1 # Adjust indices
        cols += 1
        for r, c in zip(rows, cols):
             neighborhood = pruned_skeleton[r - 1 : r + 2, c - 1 : c + 2]
             cn = self._calculate_crossing_number(neighborhood)
             if cn >= 3:
                 junction_mask[r, c] = 255 # Mark junction pixels

        # 3. Iteratively prune end points that are not junctions
        # Create a mask of pixels that can potentially be removed (skeleton pixels NOT near a junction)
        # We dilate the junction mask slightly to protect points immediately adjacent to junctions
        # A small kernel is sufficient, e.g., 3x3
        kernel_junction_protection = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_junction_mask = cv2.dilate(junction_mask, kernel_junction_protection, iterations=1)

        for i in range(self.pruning_iterations):
            # Identify skeleton pixels that are *not* protected near junctions
            pixels_to_check = cv2.bitwise_and(pruned_skeleton, cv2.bitwise_not(dilated_junction_mask))

            # Find potential endpoints among these unprotected pixels (excluding border)
            rows, cols = np.where(pixels_to_check[1:height-1, 1:width-1] == 255)
            rows += 1
            cols += 1

            if len(rows) == 0:
                # No unprotected skeleton pixels left to check
                break

            points_to_remove_this_iter = []
            for r, c in zip(rows, cols):
                 # Check the crossing number *in the current state of pruned_skeleton*
                 # A pixel might have been a midpoint but becomes an endpoint as neighbors are removed
                 neighborhood = pruned_skeleton[r - 1 : r + 2, c - 1 : c + 2]
                 cn = self._calculate_crossing_number(neighborhood)

                 # A pixel is a removable endpoint if it's a CN=1 and not part of the protected junction area
                 # The pixels_to_check mask already handles the 'not protected' part.
                 # So we just need to check if it's a CN=1 endpoint *within the current pruned_skeleton state*.
                 # We also need to ensure its removal doesn't break connectivity of essential parts,
                 # but iterative removal of simple endpoints (CN=1) generally preserves essential topology
                 # when junctions are protected.
                 if cn == 1:
                    points_to_remove_this_iter.append((r, c))

            if not points_to_remove_this_iter:
                # No removable endpoints found in this iteration
                break

            # Remove the marked points
            for r, c in points_to_remove_this_iter:
                 pruned_skeleton[r, c] = 0 # Set pixel to background

            # Important: After removing pixels, the crossing numbers of neighbors might change.
            # The loop iterates a fixed number of times or until no points are removed.

        return pruned_skeleton
    
    # def _fill_small_holes(self, mask: np.ndarray) -> np.ndarray:
    #     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    #         mask.astype(np.uint8), connectivity=8
    #     )

    #     filled_mask = mask.copy()
    #     for i in range(1, num_labels):
    #         area = stats[i, cv2.CC_STAT_AREA]
    #         if area <= self.hole_fill_area:
    #             filled_mask[labels == i] = 0        
    #     return filled_mask
    
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
        # cv2.imwrite("temp/test1.png", filled_mask)
        return filled_mask
        

    def _generate_initial_skeleton_and_base_image(self, occupancy_map):

        occ_np = occupancy_map[0, 0].cpu().numpy().astype(np.float32)
        height, width = occ_np.shape

        obstacle_mask = (occ_np == 2.0).astype(np.uint8) * 255

        # unexplored_mask_unfilled = (occ_np == 0.0).astype(np.uint8) * 255
        # unexplored_mask = self._fill_small_holes(unexplored_mask_unfilled)
        unexplored_mask = (occ_np == 0.0).astype(np.uint8) * 255

        # free_space_mask = ((occ_np > 0.0) & (occ_np < 2.0)).astype(np.uint8) * 255
        
        kernel_size = self.dilate_kernel
        if kernel_size % 2 == 0: # Ensure kernel size is odd
            kernel_size += 1

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        dilated_obstacles = cv2.dilate(obstacle_mask, kernel, iterations=1)
        # TODO: cancle the dislation of unexplored areas
        # dilated_unexplored = cv2.dilate(unexplored_mask, kernel, iterations=1) # Avoid areas still marked as unexplored
        dilated_unexplored = unexplored_mask

        area_to_avoid = cv2.bitwise_or(dilated_obstacles, dilated_unexplored)   
        
        # TODO: use cv2.bitwise_not(area_to_avoid) instead of cv2.bitwise_and(free_space_mask, cv2.bitwise_not(area_to_avoid))
        # thinning_target = cv2.bitwise_and(free_space_mask, cv2.bitwise_not(area_to_avoid))
        thinning_target = cv2.bitwise_not(area_to_avoid)

        initial_skeleton = np.zeros((height, width), dtype=np.uint8) 
        if np.sum(thinning_target) > 0: 
            _, thinning_target_binary = cv2.threshold(thinning_target, 1, 255, cv2.THRESH_BINARY)
            try:
                skeleton_zhang = cv2.ximgproc.thinning(thinning_target_binary, cv2.ximgproc.THINNING_ZHANGSUEN)
                try: 
                   skeleton_guo = cv2.ximgproc.thinning(thinning_target_binary, cv2.ximgproc.THINNING_GUOHALL)
                   initial_skeleton = cv2.bitwise_and(skeleton_zhang, skeleton_guo)
                except AttributeError:
                   print("Info: cv2.ximgproc.THINNING_GUOHALL not found. Using only Zhang-Suen thinning.")
                   initial_skeleton = skeleton_zhang
            except AttributeError:
                 print("Error: cv2.ximgproc.thinning not found. Cannot generate initial skeleton.")
                 return None, None 

        base_img_bgr = np.zeros((height, width, 3), dtype=np.uint8)
        base_img_bgr[occ_np == 0.0] = (128, 128, 128) # Unexplored (Gray)
        base_img_bgr[(occ_np > 0.0) & (occ_np < 2.0)] = (255, 255, 255) # Free Space (White)
        base_img_bgr[occ_np == 2.0] = (0, 0, 0) # Obstacle (Black)
        base_img_bgr = np.ascontiguousarray(base_img_bgr) # Ensure contiguous memory for OpenCV

        return base_img_bgr, initial_skeleton # Return base image and the initial (unpruned) skeleton


    def process_image(self, occupancy_map):
        """
        Main method to process the occupancy map: generates skeleton, prunes it,
        finds key points (endpoints and merged junctions), and creates an
        annotated visualization image. Does NOT fill unexplored holes.
        """
        # Step 1: Generate initial skeleton and base image (without hole filling)
        base_img_bgr, initial_skeleton = self._generate_initial_skeleton_and_base_image(occupancy_map)

        if base_img_bgr is None or initial_skeleton is None:
            print("Error during initial skeleton generation (possibly cv2.ximgproc not found).")
            # Reset internal state and return None results
            self.pruned_skeleton, self.endpoints, self.merged_junctions = None, None, None
            return None, None, None, None, None # all_key_points, annotated_image, pruned_skeleton, endpoints, merged_junctions

        # Step 2: Prune the initial skeleton
        pruned_skeleton = self._prune_skeleton(initial_skeleton)
        self.pruned_skeleton = pruned_skeleton # Store the result

        # Prepare annotated image by copying the base image
        annotated_image = base_img_bgr.copy()

        # Overlay the pruned skeleton onto the base image
        # Create a boolean mask for skeleton pixels
        pruned_skeleton_pixels_mask = (pruned_skeleton == 255)
        skeleton_color_bgr = (255, 0, 0) # Blue color for skeleton

        # Apply the color to the skeleton pixels in the annotated image
        # Need a 3-channel mask for the BGR image
        pruned_skeleton_pixels_mask_3_channel = np.stack([pruned_skeleton_pixels_mask]*3, axis=-1)
        annotated_image = np.where(pruned_skeleton_pixels_mask_3_channel, skeleton_color_bgr, annotated_image).astype(np.uint8)
        annotated_image = np.ascontiguousarray(annotated_image) # Ensure contiguous memory

        # Step 3: Find key points (endpoints and junctions) on the PRUNED skeleton
        endpoints, raw_junctions = self._find_skeleton_keypoints(pruned_skeleton)
        self.endpoints = endpoints # Store the result

        # Step 4: Merge nearby raw junctions
        merged_junctions = self._merge_nearby_points_with_branches(raw_junctions)
        self.merged_junctions = merged_junctions # Store the result

        # Combine endpoints and merged junctions for annotation and the return list
        all_key_points = endpoints + merged_junctions

        # Step 5: Annotate key points on the image
        if all_key_points:
            for (point_x, point_y), branch_count in all_key_points:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 1

                # Choose color and circle size based on point type (branch_count)
                if branch_count == 1: # Endpoint
                    cv2.circle(annotated_image, (point_x, point_y), 5, (0, 0, 255), -1) # Red color, filled circle
                    text = str(branch_count)
                    text_color = (0, 0, 0) # Black text
                elif branch_count >= 3: # Junction
                    cv2.circle(annotated_image, (point_x, point_y), 7, (0, 255, 255), -1) # Yellow color, filled circle
                    text = str(branch_count)
                    text_color = (0, 0, 0) # Black text
                else:
                     # Pixels with CN=2 are mid-branch points, we don't annotate them explicitly
                     continue # Skip annotation for non-endpoint/non-junctions

                # # Add text annotation for branch count near the point
                # (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
                # # Position text slightly offset from the circle
                # text_origin_x = point_x + 10
                # text_origin_y = point_y + text_height // 2 # Vertically center text roughly

                # # Adjust text position if it goes out of image bounds
                # img_h, img_w = annotated_image.shape[:2]
                # if text_origin_x + text_width > img_w:
                #     text_origin_x = point_x - text_width - 5 # Place text to the left
                # if text_origin_y + text_height > img_h:
                #      text_origin_y = point_y - baseline - 5 # Place text above
                # # Ensure text is not completely outside top/left boundary
                # text_origin_x = max(0, text_origin_x)
                # text_origin_y = max(text_height, text_origin_y) # Ensure top of text is at least text_height

                # cv2.putText(annotated_image, text, (text_origin_x, text_origin_y),
                #             font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        return all_key_points, annotated_image, pruned_skeleton, endpoints, merged_junctions

    def _find_nearest_skeleton_pixel(self, target_rc, skeleton_img, max_radius=20):
        """
        Finds the nearest skeleton pixel (value 255) to a target coordinate (row, col)
        using BFS within a maximum radius.
        """
        if skeleton_img is None or skeleton_img.size == 0 or np.sum(skeleton_img) == 0:
             print("Warning: Skeleton image for nearest pixel search is empty.")
             return None

        height, width = skeleton_img.shape
        start_r, start_c = target_rc

        # Check if the target point is within image bounds
        if not (0 <= start_r < height and 0 <= start_c < width):
             print(f"Warning: Target point ({start_c},{start_r}) is outside image bounds ({width}x{height}).")
             return None # Target is out of bounds

        # If the target itself is a skeleton pixel, return it
        if skeleton_img[start_r, start_c] == 255:
            return target_rc # (row, col)

        # Use BFS to search for the nearest skeleton pixel
        q = deque([(start_r, start_c)])
        visited = np.zeros_like(skeleton_img, dtype=bool)
        visited[start_r, start_c] = True # Mark start as visited

        # 8-connectivity neighbors
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        max_radius_sq = max_radius * max_radius

        while q:
            curr_r, curr_c = q.popleft()

            # Explore neighbors
            for dr, dc in neighbors:
                neighbor_r, neighbor_c = curr_r + dr, curr_c + dc
                neighbor_rc = (neighbor_r, neighbor_c)

                # Check bounds
                if 0 <= neighbor_r < height and 0 <= neighbor_c < width:
                    # Check if visited
                    if not visited[neighbor_r, neighbor_c]:
                        neighbor_dist_sq = (neighbor_r - start_r)**2 + (neighbor_c - start_c)**2
                        if neighbor_dist_sq <= max_radius_sq:
                            visited[neighbor_r, neighbor_c] = True # Mark as visited

                            # Check if this neighbor is a skeleton pixel
                            if skeleton_img[neighbor_r, neighbor_c] == 255:
                                 # Found the nearest skeleton pixel
                                 return neighbor_rc # Return as (row, col)

                            # If not a skeleton pixel, add to queue for further search
                            q.append(neighbor_rc) # Add to queue

        # If the queue is empty and no skeleton pixel was found within radius
        print(f"Warning: No skeleton pixel found within {max_radius} pixels of target ({start_c},{start_r}).")
        return None # No skeleton pixel found within the maximum radius


    def find_junction_from_endpoint(self, start_point_xy):
        """
        Finds the first junction encountered when traversing the skeleton from a given endpoint.
        Returns a list containing the coordinate(s) of the found junction (at most one).
        """
        # Check if skeleton processing results are available
        if self.pruned_skeleton is None or self.endpoints is None or self.merged_junctions is None:
            print("Error: Skeleton processing results not available. Call process_image first.")
            return [] # Return empty list indicating failure

        # Optional: Verify if the start_point_xy is close to a listed endpoint
        # This makes the function more robust if the input isn't exactly a detected endpoint pixel
        # But the current logic assumes the input *is* one of the detected endpoints for validation.
        # Let's stick to validating against the found endpoints list for clarity.
        start_point_is_listed_endpoint = False
        start_endpoint_info = None
        for ep_xy, ep_cn in self.endpoints:
             if ep_xy == start_point_xy:
                 start_point_is_listed_endpoint = True
                 start_endpoint_info = (ep_xy, ep_cn)
                 break

        if not start_point_is_listed_endpoint:
             print(f"Warning: Nominal start point {start_point_xy} is not listed as an endpoint.")
             # Optionally, could try finding the nearest endpoint instead, but for now, fail.
             return []

        # Find the actual skeleton pixel closest to the nominal starting point
        # This handles cases where the input XY might be slightly off the skeleton pixel
        start_point_rc_nominal = (start_point_xy[1], start_point_xy[0]) # Convert (x, y) to (row, col)
        actual_start_rc = self._find_nearest_skeleton_pixel(
            target_rc=start_point_rc_nominal,
            skeleton_img=self.pruned_skeleton,
            # Search radius for nearest skeleton pixel - can be a bit larger
            # than junction distance to find endpoint even if pruning shifted it slightly.
            max_radius=self.min_junction_distance * 2
        )

        if actual_start_rc is None:
             print(f"Warning: Listed endpoint {start_point_xy} is too far from the skeleton.")
             return [] # Could not find a nearby skeleton pixel

        height, width = self.pruned_skeleton.shape

        # Use BFS to traverse the skeleton from the actual start pixel
        # We need to find the first junction pixel encountered.
        q = deque([actual_start_rc])
        visited = {actual_start_rc} # Use a set for faster lookup of visited pixels

        # Get set of merged junction coordinates for quick lookup during traversal
        merged_junction_coords_set = {j[0] for j in self.merged_junctions} # (x, y) coordinates

        neighbors = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)] # 8-connectivity

        while q:
            curr_r, curr_c = q.popleft()
            current_rc = (curr_r, curr_c)
            current_xy = (curr_c, curr_r) # Convert back to (x, y) for junction set lookup

            # Check if the current pixel is a listed junction (and not the very start pixel itself, if start happened to be a junction)
            if current_rc != actual_start_rc and current_xy in merged_junction_coords_set:
                # Found the first junction pixel
                return [current_xy] # Return as a list containing the single junction coordinate

            # Explore neighbors
            for dr, dc in neighbors:
                neighbor_r, neighbor_c = curr_r + dr, curr_c + dc
                neighbor_rc = (neighbor_r, neighbor_c)

                # Check bounds, if it's a skeleton pixel, and if not visited
                if (0 <= neighbor_r < height and 0 <= neighbor_c < width and
                    self.pruned_skeleton[neighbor_r, neighbor_c] == 255 and
                    neighbor_rc not in visited):

                     visited.add(neighbor_rc) # Mark as visited
                     q.append(neighbor_rc) # Add to queue for exploration

        # If the loop finishes and no junction pixel was found
        print(f"Warning: BFS from {actual_start_rc} (nearest to listed endpoint {start_point_xy}) finished without reaching a listed junction.")
        return [] # Return empty list if no junction was found

    def find_endpoints_from_junction(self, start_point_xy):
        """
        Finds all endpoints reachable from a given junction by traversing along the skeleton branches.
        Returns a list of reachable endpoint coordinates (x, y).
        """
        # Check if skeleton processing results are available
        if self.pruned_skeleton is None or self.endpoints is None or self.merged_junctions is None:
            print("Error: Skeleton processing results not available. Call process_image first.")
            return [] # Return empty list indicating failure

        # Optional: Verify if the start_point_xy is close to a listed junction
        start_point_is_listed_junction = False
        for junc_xy, junc_cn in self.merged_junctions:
             if junc_xy == start_point_xy:
                 start_point_is_listed_junction = True
                 break

        if not start_point_is_listed_junction:
             print(f"Warning: Nominal start point {start_point_xy} is not listed as a junction.")
             # Optionally, could find the nearest junction instead. For now, fail.
             return []

        # Find the actual skeleton pixel closest to the nominal starting point (junction)
        start_point_rc_nominal = (start_point_xy[1], start_point_xy[0]) # Convert (x, y) to (row, col)
        actual_start_rc = self._find_nearest_skeleton_pixel(
            target_rc=start_point_rc_nominal,
            skeleton_img=self.pruned_skeleton,
            max_radius=self.min_junction_distance * 2 # Search radius
        )

        if actual_start_rc is None:
             print(f"Warning: Listed junction {start_point_xy} is too far from the skeleton.")
             return [] # Could not find a nearby skeleton pixel

        height, width = self.pruned_skeleton.shape
        neighbors = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)] # 8-connectivity

        found_endpoints_xy = [] # List to store the coordinates of found endpoints

        # Get sets of keypoint coordinates for quick lookup
        merged_junction_coords_set = {j[0] for j in self.merged_junctions} # (x, y) coordinates
        endpoint_coords_set = {e[0] for e in self.endpoints} # (x, y) coordinates

        # Find the initial skeleton neighbors of the actual starting junction pixel.
        # Each neighbor typically represents the start of a different branch emanating from the junction.
        initial_branch_starts_rc = []
        for dr, dc in neighbors:
            neighbor_r, neighbor_c = actual_start_rc[0] + dr, actual_start_rc[1] + dc
            neighbor_rc = (neighbor_r, neighbor_c)
            # Check bounds and if the neighbor is a skeleton pixel
            if 0 <= neighbor_r < height and 0 <= neighbor_c < width and \
               self.pruned_skeleton[neighbor_r, neighbor_c] == 255:
               initial_branch_starts_rc.append(neighbor_rc)

        if not initial_branch_starts_rc:
             print(f"Warning: Actual start point ({actual_start_rc[1]},{actual_start_rc[0]}) has no skeleton neighbors.")
             return [] # The junction pixel is isolated or at the very end of the skeleton (shouldn't happen for a junction)

        # For each branch starting from a neighbor, perform a separate BFS
        for branch_start_rc in initial_branch_starts_rc:
            q = deque([branch_start_rc])
            # Visited set for THIS branch search.
            # Initialize visited with the start of the branch and the actual junction pixel itself,
            # so we don't immediately go back to the junction or revisit pixels within this branch's path.
            visited = {branch_start_rc, actual_start_rc}
            branch_found_endpoint = False # Flag to stop search on this branch once an endpoint is found

            while q and not branch_found_endpoint:
                 curr_rc = q.popleft()
                 curr_r, curr_c = curr_rc
                 current_xy = (curr_c, curr_r) # Convert to (x, y)

                 # Check if the current pixel is a listed endpoint
                 if current_xy in endpoint_coords_set:
                      found_endpoints_xy.append(current_xy) # Add the endpoint coordinate
                      branch_found_endpoint = True # Stop searching this specific branch
                      # print(f"Found endpoint {current_xy} from branch starting at {branch_start_rc}") # Debug print
                      break # Exit the inner while loop for this branch

                 # Check if the current pixel is another junction (and not the starting junction)
                 # If we hit another junction, this branch segment ends there.
                 if current_rc != actual_start_rc and current_xy in merged_junction_coords_set:
                      # print(f"Branch from {branch_start_rc} hit another junction at {current_xy}. Stopping search on this branch.") # Debug print
                      break # Stop searching this branch

                 # Explore neighbors (for pixels that are not endpoints or other junctions)
                 for dr, dc in neighbors:
                      neighbor_r, neighbor_c = curr_r + dr, curr_c + dc
                      neighbor_rc = (neighbor_r, neighbor_c)

                      # Check bounds, if it's a skeleton pixel, and if not visited within this branch's search
                      if (0 <= neighbor_r < height and 0 <= neighbor_c < width and \
                          self.pruned_skeleton[neighbor_r, neighbor_c] == 255):

                            if neighbor_rc not in visited:
                                visited.add(neighbor_rc) # Mark as visited
                                q.append(neighbor_rc) # Add to queue

        # Post-processing: Remove duplicate endpoint coordinates if any were found via different paths
        unique_endpoints_coords_set = set()
        final_endpoints = []
        for ep_xy in found_endpoints_xy:
            if ep_xy not in unique_endpoints_coords_set:
                unique_endpoints_coords_set.add(ep_xy)
                final_endpoints.append(ep_xy)

        return final_endpoints # Return list of unique endpoint coordinates

    def find_connected_endpoint(self, start_point_xy, return_path=False):
        """
        Finds the other endpoint connected to the given start_point_xy
        by traversing the skeleton. Assumes the start point is an endpoint.
        Returns a list containing the single connected endpoint's coordinate.
        If return_path is True, also returns the list of pixels forming the path.
        """
        # Check if skeleton processing results are available
        if self.pruned_skeleton is None or self.endpoints is None or self.merged_junctions is None:
            print("Error: Skeleton processing results not available. Call process_image first.")
            if return_path: return [], [] # Return empty lists
            return []

        # Check if the input point is actually a listed junction (should not be for this function)
        if start_point_xy in [junc[0] for junc in self.merged_junctions]:
            print(f"Warning: Input point {start_point_xy} is listed as a junction. This function is intended for endpoints.")
            if return_path: return [], []
            return []

        # Check if the input point is a listed endpoint
        start_point_is_listed_endpoint = False
        for ep_xy, ep_cn in self.endpoints:
             if ep_xy == start_point_xy:
                 start_point_is_listed_endpoint = True
                 break

        if not start_point_is_listed_endpoint:
             print(f"Warning: Nominal start point {start_point_xy} is not listed as an endpoint.")
             if return_path: return [], []
             return []

        start_point_rc_nominal = (start_point_xy[1], start_point_xy[0]) # Convert (x, y) to (row, col)

        # Find the actual skeleton pixel nearest the nominal start endpoint
        actual_start_rc = self._find_nearest_skeleton_pixel(
            target_rc=start_point_rc_nominal,
            skeleton_img=self.pruned_skeleton,
            # Search radius for nearest skeleton pixel - slightly larger than junction distance is reasonable
            max_radius=self.min_junction_distance * 2
        )

        if actual_start_rc is None:
             print(f"Warning: Listed endpoint {start_point_xy} is too far from the skeleton.")
             if return_path: return [], []
             return []

        height, width = self.pruned_skeleton.shape
        neighbors = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)] # 8-connectivity

        # Find the single skeleton neighbor of the actual starting endpoint pixel.
        # A proper endpoint pixel (CN=1) should only have one skeleton neighbor.
        # We start the BFS from this neighbor to move *away* from the endpoint tip.
        neighbor_of_start_rc = None
        # Iterate through potential neighbors
        for dr, dc in neighbors:
             r, c = actual_start_rc[0] + dr, actual_start_rc[1] + dc
             # Check bounds and if the neighbor is a skeleton pixel
             if 0 <= r < height and 0 <= c < width and self.pruned_skeleton[r, c] == 255:
                  neighbor_of_start_rc = (r, c)
                  break # Found the (should be single) neighbor

        if neighbor_of_start_rc is None:
             # This is an edge case for a supposed endpoint on a pruned skeleton.
             # It implies the point is isolated or pruning was too aggressive.
             print(f"Warning: Actual start point ({actual_start_rc[1]},{actual_start_rc[0]}) has no skeleton neighbors. Cannot traverse.")
             if return_path: return [], []
             return []

        # --- BFS from the neighbor of the start endpoint ---
        # The search starts *after* the tip pixel itself, moving into the branch.
        # We look for the *first* visited pixel that is an endpoint and *not* the original one.

        q = deque([neighbor_of_start_rc])

        # Use a dictionary to track visited pixels AND store the parent for path reconstruction
        # Initialize visited set/parent dict. The actual start point is considered 'visited'
        # in this context because we don't want to go back to the tip immediately.
        parent = {neighbor_of_start_rc: actual_start_rc, actual_start_rc: None} # parent[child] = parent_node
        visited = {neighbor_of_start_rc, actual_start_rc} # Set of (r, c) tuples

        # Get set of keypoint coordinates for quick lookup
        endpoint_coords_set = {e[0] for e in self.endpoints} # (x, y) coordinates
        merged_junction_coords_set = {j[0] for j in self.merged_junctions} # (x, y) coordinates


        found_endpoint_rc = None # Will store the (r, c) of the found endpoint

        while q:
            curr_rc = q.popleft()
            curr_r, curr_c = curr_rc
            curr_xy = (curr_c, curr_r) # Convert to (x, y) for endpoint/junction set lookup

            # Check if the current pixel is another listed endpoint (and not the starting one)
            # We compare curr_rc against actual_start_rc as they are pixel coordinates.
            if curr_rc != actual_start_rc and curr_xy in endpoint_coords_set:
                 found_endpoint_rc = curr_rc # Found the target endpoint pixel
                 break # Found the target endpoint, stop the search

            # Check if the current pixel is a listed junction
            # If we hit a junction, this path segment ends here when searching from an endpoint.
            if current_rc != actual_start_rc and curr_xy in merged_junction_coords_set:
                 # print(f"Search from endpoint hit junction at {curr_xy}. Stopping search.") # Debug print
                 break # Stop search along this path if it reaches a junction

            # Explore neighbors (only if current pixel is not the target endpoint or a junction)
            for dr, dc in neighbors:
                neighbor_r, neighbor_c = curr_r + dr, curr_c + dc
                neighbor_rc = (neighbor_r, neighbor_c)

                # Check bounds, if it's a skeleton pixel, and if not visited
                if (0 <= neighbor_r < height and 0 <= neighbor_c < width and
                    self.pruned_skeleton[neighbor_r, neighbor_c] == 255 and
                    neighbor_rc not in visited):

                     visited.add(neighbor_rc) # Mark as visited
                     parent[neighbor_rc] = curr_rc # Record parent for path reconstruction
                     q.append(neighbor_rc) # Add to queue

        # If the loop finished, check if an endpoint was found
        if found_endpoint_rc is None:
            print(f"Warning: Search from endpoint {start_point_xy} finished without finding another reachable endpoint before hitting a junction or end of path.")
            if return_path: return [], []
            return [] # Return empty list if no other endpoint found

        # Endpoint found. Return its coordinate(s) as a list.
        found_endpoint_xy = (found_endpoint_rc[1], found_endpoint_rc[0]) # Convert (r, c) to (x, y)

        if return_path:
             # Reconstruct Path
             # Trace back from the found endpoint pixel to the actual start endpoint pixel.
             path_rc = []
             node_rc = found_endpoint_rc
             # Trace back using parent links until we reach the actual_start_rc or a pixel with no parent
             # The parent of actual_start_rc is None, so the loop stops there.
             while node_rc is not None:
                 path_rc.append(node_rc)
                 # Use .get() for parent lookup in case of unexpected start point or bug
                 node_rc = parent.get(node_rc)
             path_rc.reverse() # Path is now from actual_start_rc to found_endpoint_rc

             # Convert path pixels from (row, col) to (x, y)
             path_xy = [(c, r) for r, c in path_rc]

             return [found_endpoint_xy], path_xy # Return list with the single endpoint + path

        else:
            # Just return the list containing the single found endpoint
            return [found_endpoint_xy]


# Note: The find_junction_from_endpoint and find_endpoints_from_junction methods are kept
# but are not demonstrated in the main block below, as find_connected_endpoint
# now handles a common endpoint-to-endpoint search pattern.

if __name__ == "__main__":
    # Example Usage:
    # Ensure you have an image file like 'E.png' in the same directory or provide a full path.
    # For testing, we'll create a dummy image with clear sections.

    dummy_img_path = "test_map_no_hole_filling.png"
    # Create a 100x100 image with distinct values for states
    dummy_img_h, dummy_img_w = 100, 100
    dummy_img = np.full((dummy_img_h, dummy_img_w), 220, dtype=np.uint8) # Default Free Space (value ~1.5)
    cv2.rectangle(dummy_img, (0,0), (dummy_img_w-1, dummy_img_h-1), 10, 5) # Border Obstacle (value ~0.04)
    cv2.rectangle(dummy_img, (20,20), (80,80), 10, -1) # Inner Obstacle (value ~0.04)
    cv2.rectangle(dummy_img, (30,30), (70,70), 140, -1) # Inner Unexplored Area (value ~0.55 - using mid-range value)
    cv2.line(dummy_img, (dummy_img_w//2, 0), (dummy_img_w//2, dummy_img_h//2 - 15), 220, 3) # Free space corridor from top
    cv2.line(dummy_img, (dummy_img_w//2, dummy_img_h//2 + 15), (dummy_img_w//2, dummy_img_h-1), 220, 3) # Free space corridor to bottom

    cv2.imwrite(dummy_img_path, dummy_img)

    # Use the dummy image path
    input_image_path = dummy_img_path

    skelen_dir = "." # Directory to save output images
    os.makedirs(skelen_dir, exist_ok=True)

    def load_image_as_occupancy_tensor(image_path):
        """Loads an image and converts it to the specified occupancy map tensor format."""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Could not load image from {image_path}")
            return None

        # Define thresholds based on typical grayscale values for different states
        # Mapping used for the dummy image above:
        # 0-50 (low values): Obstacle -> 2.0
        # 51-199 (mid values): Unexplored -> 0.0
        # 200-255 (high values): Free Space -> 1.0

        obstacle_threshold_max = 50
        unexplored_threshold_min = 51
        unexplored_threshold_max = 199
        free_threshold_min = 200

        # Initialize occupancy map numpy array
        occupancy_np = np.zeros_like(img, dtype=np.float32) # Use float32 for values 0.0, 1.0, 2.0

        # Assign values based on thresholds
        occupancy_np[img <= obstacle_threshold_max] = 2.0 # Obstacles
        occupancy_np[(img >= unexplored_threshold_min) & (img <= unexplored_threshold_max)] = 0.0 # Unexplored
        occupancy_np[img >= free_threshold_min] = 1.0 # Free Space (use 1.0 as a representative free space value)

        print(f"\nLoaded image {image_path}. Map shape: {img.shape}")
        print(f"Mapping grayscale ranges to occupancy values:")
        print(f"  <= {obstacle_threshold_max} (e.g., Black): 2.0 (Obstacle)")
        print(f"  {unexplored_threshold_min}-{unexplored_threshold_max} (e.g., Gray): 0.0 (Unexplored)")
        print(f"  >= {free_threshold_min} (e.g., White): 1.0 (Free Space)")
        print(f"Occupancy values distribution in loaded map: 0.0={np.sum(occupancy_np==0.0)}, 1.0={np.sum(occupancy_np==1.0)}, 2.0={np.sum(occupancy_np==2.0)}")


        try:
             # Convert NumPy array to PyTorch tensor (shape (1, 1, H, W))
             occupancy_tensor = torch.from_numpy(occupancy_np).unsqueeze(0).unsqueeze(0).to(torch.float32)
             print(f"Converted to torch tensor: {occupancy_tensor.shape}, dtype: {occupancy_tensor.dtype}")
             return occupancy_tensor
        except ImportError:
             print("Error: torch not found. Cannot create tensor from image.")
             return None
        except Exception as e:
             print(f"An error occurred during tensor conversion: {e}")
             return None


    # Load the occupancy map tensor
    input_occupancy_tensor = load_image_as_occupancy_tensor(input_image_path)

    if input_occupancy_tensor is None:
        sys.exit(1) # Exit if image loading or tensor conversion failed

    # Define processor parameters
    processor_params = {
        "dilate_kernel": 15,        # Increased kernel size for potentially larger maps/features
        "min_component_size": 100,  # Increased minimum size for components
        "pruning_iterations": 20,   # Increased pruning iterations
        "min_junction_distance": 20 # Increased junction merging distance
    }

    processor = SkeletonProcessor(**processor_params)

    print(f"\nStarting skeleton processing for {input_image_path} without hole filling...")
    start_time = time.time()

    # Process the occupancy map
    all_key_points, annotated_output_image, pruned_skeleton_img, endpoints_list, merged_junctions_list = processor.process_image(occupancy_map=input_occupancy_tensor)

    end_time = time.time()
    processing_duration = end_time - start_time
    print(f"Finished processing in {processing_duration:.4f} seconds.")

    # Save results
    if annotated_output_image is not None:
        # Add a timestamp or hash to the filename to avoid overwriting if run multiple times quickly
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"skelen_annotated_pruned_skeleton_{timestamp_str}.png"
        full_output_path = os.path.join(skelen_dir, output_filename)
        cv2.imwrite(full_output_path, annotated_output_image)
        print(f"\nAnnotated image saved to: {full_output_path}")
        # Update description to reflect no hole filling
        print("This image shows the occupancy map (original mapping), the pruned skeleton (blue), endpoints (red, '1'), and merged junctions (yellow, branch count).")

    else:
         print("\nAnnotated image could not be generated.")

    if pruned_skeleton_img is not None and np.sum(pruned_skeleton_img) > 0:
        skeleton_output_filename = f"skelen_pruned_skeleton_{timestamp_str}.png"
        full_skeleton_path = os.path.join(skelen_dir, skeleton_output_filename)
        cv2.imwrite(full_skeleton_path, pruned_skeleton_img)
        print(f"Pruned skeleton image saved to: {full_skeleton_path}")
    else:
         print("\nPruned skeleton image is empty or could not be generated.")


    print(f"\nDetected Endpoints ({len(endpoints_list)}): {endpoints_list}")
    print(f"Detected Merged Junctions ({len(merged_junctions_list)}): {merged_junctions_list}")
    print(f"All Key Points ({len(all_key_points)}): {all_key_points}")


    print("\n--- Testing find_connected_endpoint Function ---")
    if processor.pruned_skeleton is None or not processor.endpoints:
         print("Skipping find_connected_connected test: Skeleton or endpoints are missing.")
         if processor.pruned_skeleton is None: print("  - processor.pruned_skeleton is None")
         if not processor.endpoints: print("  - processor.endpoints is empty")
    else:
        print(f"Endpoints found: {len(processor.endpoints)}")
        if processor.endpoints and len(processor.endpoints) >= 1: # Need at least one endpoint to start from
            # Try the first endpoint found as the starting point for the search
            if len(processor.endpoints) > 0:
                 start_endpoint_xy_example = processor.endpoints[0][0] # Example: Use the first detected endpoint
                 print(f"\nSearching for a connected endpoint starting from endpoint: {start_endpoint_xy_example}")

                 # Call the function with return_path=True for visualization demo
                 result_list, path_xy = processor.find_connected_endpoint(start_point_xy=start_endpoint_xy_example, return_path=True)

                 if result_list: # Check if the list is not empty (i.e., a connected endpoint was found)
                     found_endpoint_xy = result_list[0] # Get the single found endpoint coordinate
                     print(f"Successfully found a connected endpoint at: {found_endpoint_xy}")
                     print(f"Path length for visualization: {len(path_xy)} pixels.")

                     # --- VISUALIZATION PART ---
                     if annotated_output_image is not None:
                         # Create a temporary copy to draw the path
                         temp_img = annotated_output_image.copy()

                         # Draw the path pixels (e.g., in bright green)
                         path_color_bgr = (0, 255, 0) # Green
                         for p_x, p_y in path_xy:
                              # Check bounds before drawing
                              if 0 <= p_y < temp_img.shape[0] and 0 <= p_x < temp_img.shape[1]:
                                  temp_img[p_y, p_x] = path_color_bgr

                         # Mark the starting endpoint (green circle outline)
                         cv2.circle(temp_img, start_endpoint_xy_example, 8, (0, 255, 0), 2)
                         # Mark the found endpoint (orange circle outline)
                         cv2.circle(temp_img, found_endpoint_xy, 8, (0, 255, 255), 2)

                         # Save the visualization image
                         path_output_filename = f"skelen_endpoint_{start_endpoint_xy_example}_to_endpoint_{found_endpoint_xy}_path_{timestamp_str}.png"
                         full_output_path_vis = os.path.join(skelen_dir, path_output_filename)
                         cv2.imwrite(full_output_path_vis, temp_img)
                         print(f"Endpoint-to-endpoint path visualization saved to {full_output_path_vis}")
                     # --- END VISUALIZATION PART ---

                 else:
                     print(f"No other connected endpoint found starting from {start_endpoint_xy_example}.")

            else:
                 print("No endpoints detected to test `find_connected_endpoint`.")


            print("\nTesting find_connected_endpoint with a junction input (should warn):")
            if processor.merged_junctions:
                junction_xy_example = processor.merged_junctions[0][0] # Example: Use the first detected junction
                print(f"Calling find_connected_endpoint with a junction input: {junction_xy_example}")
                result_list_junction = processor.find_connected_endpoint(start_point_xy=junction_xy_example)
                print(f"Result for junction input {junction_xy_example}: {result_list_junction}")
            else:
                print("No junctions found to test junction input warning.")

        else:
            print("Not enough endpoints found on the skeleton (less than 1) to test find_connected_endpoint.")


    print("\n--- Testing find_junction_from_endpoint Function ---")
    if processor.pruned_skeleton is None or not processor.endpoints or not processor.merged_junctions:
         print("Skipping find_junction_from_endpoint test: Skeleton, endpoints, or junctions are missing.")
    else:
        if len(processor.endpoints) > 0:
            start_endpoint_xy_example = processor.endpoints[0][0] # Example: Use the first detected endpoint
            print(f"\nSearching for junction from endpoint: {start_endpoint_xy_example}")
            found_junctions = processor.find_junction_from_endpoint(start_point_xy=start_endpoint_xy_example)
            print(f"Found junction(s): {found_junctions}")
        else:
            print("No endpoints found to test `find_junction_from_endpoint`.")

    print("\n--- Testing find_endpoints_from_junction Function ---")
    if processor.pruned_skeleton is None or not processor.endpoints or not processor.merged_junctions:
         print("Skipping find_endpoints_from_junction test: Skeleton, endpoints, or junctions are missing.")
    else:
        if len(processor.merged_junctions) > 0:
            start_junction_xy_example = processor.merged_junctions[0][0] # Example: Use the first detected junction
            print(f"\nSearching for endpoints from junction: {start_junction_xy_example}")
            found_endpoints = processor.find_endpoints_from_junction(start_point_xy=start_junction_xy_example)
            print(f"Found endpoint(s): {found_endpoints}")
        else:
            print("No junctions found to test `find_endpoints_from_junction`.")


    print("\nProcessing and search demonstration complete.")
    print(f"Look for output images in the '{skelen_dir}' directory.")

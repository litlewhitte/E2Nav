from PIL import Image
import heapq
import math
import torch
import numpy as np
import cv2
from scipy.ndimage import binary_erosion, binary_dilation, binary_closing, label, center_of_mass, generate_binary_structure
from scipy.ndimage import sum as ndi_sum
from sklearn.cluster import KMeans
from collections import deque


def filter_small_regions(mask, min_area):
    """过滤小连通区域"""
    labeled, num = label(mask)
    valid_mask = np.zeros_like(mask)
    
    for i in range(1, num+1):
        region = (labeled == i)
        if ndi_sum(region) >= min_area:
            valid_mask |= region
    return valid_mask

# 计算质心坐标
def get_valid_centroids(mask, labeled, num_regions):
    """获取有效质心坐标"""
    centroids = []
    if num_regions == 0:
        return centroids
    
    # 计算所有质心
    all_centroids = center_of_mass(mask, labeled, index=range(1, num_regions+1))
    
    # 过滤无效坐标
    for ct in all_centroids:
        # 检查NaN和坐标有效性
        if not np.isnan(ct).any() and \
           0 <= ct[0] < mask.shape[0] and \
           0 <= ct[1] < mask.shape[1]:
            # 转换为图像坐标系 (x,y)
            centroids.append((int(round(ct[1])), int(round(ct[0]))))
    return centroids

def heuristic(a, b):
    """欧几里得距离启发函数"""
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# def heuristic(a, b):
#     return abs(a[0] - b[0]) + abs(a[1] - b[1])

# def heuristic(a, b):
#     # 使用对角线距离优化启发函数
#     dx = abs(a[0] - b[0])
#     dy = abs(a[1] - b[1])
#     return 1.414 * min(dx, dy) + abs(dx - dy)

# def close_small_obstacles(map_array, threshold=10):
#     # 创建副本避免修改原始数据
#     processed_map = map_array.copy()
    
#     # 标记所有2.0的连通区域
#     binary_2 = (processed_map == 2.0).astype(int)
#     struct = generate_binary_structure(2, 2)  # 8邻域连通
#     labeled_array, num_features = label(binary_2, structure=struct)
    
#     # 计算组件尺寸并过滤
#     component_sizes = np.bincount(labeled_array.ravel())
#     for label_num in range(1, num_features + 1):
#         if component_sizes[label_num] <= threshold:
#             processed_map[labeled_array == label_num] = 1.0
            
#     return processed_map

def find_nearest_valid_point(target, inflated_map, max_radius=20):
    """BFS搜索最近的可行点"""
    directions = [(-1,0),(1,0),(0,-1),(0,1),
                (-1,-1),(-1,1),(1,-1),(1,1)]
    
    visited = set()
    queue = deque([(target[0], target[1], 0)])
    
    while queue:
        x, y, depth = queue.popleft()
        if depth > max_radius:
            return None
        
        # 边界检查
        if 0 <= x < inflated_map.shape[1] and 0 <= y < inflated_map.shape[0]:
            if not inflated_map[y, x]:
                return (x, y)
            
            # 扩展相邻节点
            if depth < max_radius:
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny, depth + 1))
    return None


def astar(start, end, occupancy_map):
    """
    A*路径规划算法
    :param start: 起点坐标(x, y)
    :param end: 终点坐标(x, y)
    :param occupancy_map: torch张量(1,1,w,h)，地图数据
    :return: 规划路径列表[(x1,y1), (x2,y2), ...]
    """
    # 输入验证
    if occupancy_map.dim() != 4 or occupancy_map.shape[0] != 1 or occupancy_map.shape[1] != 1:
        raise ValueError("输入地图格式应为(1,1,w,h)")
    
    # 转换并预处理地图
    map_array = occupancy_map[0,0].detach().cpu().numpy()
    obstacle_map = (map_array == 2.0).astype(np.uint8)  # 障碍物区域
    
    # # 填补小缺口（闭运算）
    # closed_map = binary_closing(obstacle_map, structure=np.ones((3,3)))
    
    # 生成膨胀地图
    robot_radius = 5
    y, x = np.ogrid[-robot_radius:robot_radius+1, -robot_radius:robot_radius+1]
    kernel = (x**2 + y**2) <= robot_radius**2
    inflated_map = binary_dilation(obstacle_map, structure=kernel)
    
    # 终点调整
    end = (int(end[0]), int(end[1]))
    if (0 <= end[0] < inflated_map.shape[1] and 0 <= end[1] < inflated_map.shape[0]):
        if inflated_map[end[1], end[0]]:
            end = find_nearest_valid_point(end, inflated_map) or end
    
    # 起点调整
    start = (int(start[0]), int(start[1]))
    if (0 <= start[0] < inflated_map.shape[1] and 0 <= start[1] < inflated_map.shape[0]):
        if inflated_map[start[1], start[0]]:
            start = find_nearest_valid_point(start, inflated_map) or start

    # A*算法初始化
    neighbors = [(0,1),(0,-1),(1,0),(-1,0),
               (1,1),(1,-1),(-1,1),(-1,-1)]
    
    frontier = []
    heapq.heappush(frontier, (0, start[0], start[1]))
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    found = False
    
    while frontier:
        current_cost, x, y = heapq.heappop(frontier)
        current = (x, y)
        
        if current == end:
            found = True
            break
        
        for dx, dy in neighbors:
            next_p = (x + dx, y + dy)
            
            # 边界检查
            if not (0 <= next_p[0] < inflated_map.shape[1] and 
                    0 <= next_p[1] < inflated_map.shape[0]):
                continue
                
            # 障碍物检查
            if inflated_map[next_p[1], next_p[0]]:
                continue
                
            # 对角线移动检查
            if dx != 0 and dy != 0:
                if inflated_map[y, x+dx] or inflated_map[y+dy, x]:
                    continue
            
            # 计算移动代价
            move_cost = 1.414 if (dx and dy) else 1
            new_cost = cost_so_far[current] + move_cost
            
            if next_p not in cost_so_far or new_cost < cost_so_far.get(next_p, float('inf')):
                cost_so_far[next_p] = new_cost
                priority = new_cost + heuristic(end, next_p)
                heapq.heappush(frontier, (priority, next_p[0], next_p[1]))
                came_from[next_p] = current
    
    # 路径重构
    if not found:
        return None
    
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = came_from.get(current)
    path.reverse()
    
    return path if len(path) > 1 else None

def getEdgeFrontier(explored_map, obstacles_map):
    explored_array = explored_map[0,0].detach().cpu().numpy() 
    obstacles_array = obstacles_map[0,0].detach().cpu().numpy()
    
    explored = explored_array == 1.0
    unexplored = (explored_array > 0.0) & (explored_array < 1.0)
    obstacles = obstacles_array == 1.0

    # 使用八邻域检测边界
    dilation_kernel = np.ones((3, 3), dtype=bool)
    erosion_kernel = np.ones((4, 4), dtype=bool)
    explored_border = binary_dilation(explored, structure=dilation_kernel) & unexplored
    unexplored_border = binary_dilation(unexplored, structure=dilation_kernel) & explored
    border = explored_border | unexplored_border

    # 扩展边界到3像素宽
    expanded_border = binary_dilation(border, structure=dilation_kernel, iterations=1)
    border_without_obstacles = expanded_border ^ (expanded_border & obstacles)
    eroded = binary_erosion(border_without_obstacles, structure=erosion_kernel, iterations=1)
    final_border = binary_dilation(eroded, structure=erosion_kernel, iterations=1)

    # 应用区域过滤
    MIN_AREA = 50  # 最小保留区域的像素数
    filtered_border = filter_small_regions(final_border, MIN_AREA)    # 过滤后的区域边界

    # 重新标记过滤后的区域
    labeled, num_regions = label(filtered_border)

    # 获取有效坐标
    centroid_coords = get_valid_centroids(filtered_border, labeled, num_regions)
    return centroid_coords


def getFrontier(pos, occupancy_map):
    h, w = occupancy_map.shape[2], occupancy_map.shape[3]
    device = occupancy_map.device
    
    # === 新增：坐标系参数 ===
    resolution = 0.05  # 米/像素 (修改点1：添加物理单位转换)
    map_center = (400, 400)  # 假设地图中心在(400,400)像素
    
    # === 修改点2：世界坐标转像素坐标 ===
    x_world, y_world = pos  # 输入为世界坐标（米）
    x_pixel = int(map_center[0] + x_world / resolution)
    y_pixel = int(map_center[1] - y_world / resolution)  # y轴反转！

    # 步骤1: 创建自由空间掩码
    free_mask = (occupancy_map > 0) & (occupancy_map < 2)
    coords = torch.nonzero(free_mask.squeeze())
    print(f"[Debug] 自由空间点数: {coords.size(0)}")
    
    if coords.size(0) == 0:
        return []

    # 步骤2: 计算距离场（物理距离）
    # === 修改点3：使用像素坐标计算 ===
    candidate_y = coords[:, 0].float()  # 图像坐标y（行）
    candidate_x = coords[:, 1].float()  # 图像坐标x（列）
    
    dx_pixel = candidate_x - x_pixel
    dy_pixel = y_pixel - candidate_y  # 关键修改点：y轴反转！
    
    # 转换为物理距离（米）
    distances = torch.sqrt(dx_pixel**2 + dy_pixel**2) * resolution  # 修改点4：添加单位转换

    # 动态距离范围（保持物理单位）
    q_low = torch.quantile(distances, 0.25)
    q_high = torch.quantile(distances, 0.75)

    valid_dist = (distances >= q_low) & (distances <= q_high)
    candidate_coords = coords[valid_dist]
    
    if candidate_coords.size(0) == 0:
        candidate_coords = coords
        distances = torch.sqrt(dx_pixel**2 + dy_pixel**2) * resolution

    # 步骤3: 角度分区（笛卡尔坐标系）
    # === 修改点5：统一使用笛卡尔坐标系计算角度 ===
    dx_valid = candidate_coords[:, 1].float() - x_pixel  # 列差（x方向）
    dy_valid = y_pixel - candidate_coords[:, 0].float()  # 行差（y方向反转！）
    
    angles = torch.atan2(dy_valid, dx_valid)  # 现在使用正确的坐标系
    angles = (angles + 2 * math.pi) % (2 * math.pi)  # 0~2π

    num_sectors = 8
    sector_size = 2 * math.pi / num_sectors
    sectors = torch.floor(angles / sector_size).long() % num_sectors  # 修改点6：精确分区

    # 步骤4: 各分区选择（保持不变）
    selected_indices = []
    for sector in range(num_sectors):
        sector_mask = (sectors == sector)
        sector_indices = torch.nonzero(sector_mask).view(-1)
        print(f"  分区{sector}: {sector_indices.size(0)}点", end=" | ")
        
        if sector_indices.size(0) > 0:
            sector_distances = distances[sector_indices]
            sorted_order = torch.argsort(sector_distances)
            median_idx = sorted_order[len(sorted_order)//2]
            selected_global_idx = sector_indices[median_idx]
            selected_indices.append(selected_global_idx.unsqueeze(0))
            print(f"选择索引{median_idx}")
        else:
            print("无候选点")

    # 合并候选点
    if selected_indices:
        final_indices = torch.cat(selected_indices)
        final_coords = candidate_coords[final_indices]
    else:
        final_coords = candidate_coords
    print(f"[Debug] 最终候选点数: {final_coords.size(0)}")

    # 步骤5: 邻域障碍物检查 (5x5领域)
    obstacle_mask = (occupancy_map == 2).float()  # 障碍物标记为2的掩码

    # 创建5x5卷积核 (检测5x5邻域)
    kernel = torch.ones(1, 1, 5, 5, device=device)  # [in_ch, out_ch, H, W]
    obstacle_counts = torch.nn.functional.conv2d(
        obstacle_mask, 
        kernel, 
        padding=2  # 关键修改：5x5核需要padding=2才能保持输出尺寸
    )

    valid_points = []
    for coord in final_coords:
        y, x = coord[0].item(), coord[1].item()
        
        # 双重保险检查 (坐标合法性和邻域障碍物)
        if 0 <= x < w and 0 <= y < h:  # 确保不越界
            # 检查5x5邻域内是否有障碍物（obstacle_counts > 0表示存在）
            if obstacle_counts[0, 0, y, x] == 0:  # 5x5领域全为自由空间
                valid_points.append((x, y))
        else:
            print(f"[Warning] 候选点坐标越界: ({x}, {y})")

    print(f"[Debug] 通过5x5障碍检查的有效点: {len(valid_points)}")
    
    # 备用点处理
    if not valid_points and candidate_coords.size(0) > 0:
        print("[Debug] 使用备用候选点！")
        backup_coord = candidate_coords[0]
        valid_points = [(int(backup_coord[1]), int(backup_coord[0]))]
    
    print(f"[Debug] 最终返回点数: {len(valid_points)}\n")
    return valid_points

def local_policy(map, observations, point_goal):
    gps = observations.get("gps", np.zeros(2))  # 默认二维
    x = gps[0] if len(gps) > 0 else 0.0
    y = gps[1] if len(gps) > 1 else 0.0
    pixel_x = int(400 + x * 20)
    pixel_y = int(400 - y * 20)
    current_heading = math.degrees(observations['compass']) % 360  # 获取当前朝向
    
    temp_path = astar((pixel_x, pixel_y), point_goal, map)  # 从起点到终点的路径规划temp_path 
    if temp_path == None or len(temp_path) > 100:
        print(f'temp path is none')
        return 3, None, None, True
    else:   
        for nav_point in temp_path:
            if heuristic((pixel_x, pixel_y), nav_point) > 3:
                temp_destination = nav_point  # 设置一个temp_path上的临时导航点（如果能正确抵达该导航点且不改变终点坐标，那么agent将沿着该temp_path走到终点）
                break
            
    # 计算目标方向
    delta_x = temp_destination[0] - pixel_x
    delta_y_actual = pixel_y - temp_destination[1]
    angle_rad = math.atan2(delta_y_actual, delta_x)
    target_angle = math.degrees(angle_rad) % 360
    
    # 计算与当前朝向的角度差
    delta_angle = (target_angle + current_heading) % 360
    if delta_angle > 180:
        delta_angle -= 360
    print(f'Here is the debug info: the temp_destination is {temp_destination}, the point_goal is {point_goal}')
    # 决定动作
    if delta_angle > 16:
        action = 3  # 右转
    elif delta_angle < -16:
        action = 2  # 左转
    else:
        action = 1  # 直行
    
    return action, temp_destination, temp_path, False


if __name__ == '__main__':
    explored_map = Image.open('/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouqihang-240108100050/HuNav/temp/explored.png').convert('RGB')
    explored_array = np.array(explored_map)
    obstacles_map = Image.open('/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouqihang-240108100050/HuNav/temp/obstacles.png').convert('RGB')
    obstacles_array = np.array(obstacles_map)
    merged_map = Image.open('/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouqihang-240108100050/HuNav/temp/map.png').convert('RGB')
    merged_array = np.array(merged_map)
    getFrontier(explored_array, obstacles_array)
    i = astar((400, 400), (499, 369), merged_array)
    
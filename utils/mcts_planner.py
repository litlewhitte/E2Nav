"""
MCTS (Monte Carlo Tree Search) Planner for Frontier Selection

使用MCTS算法在多个候选frontier点中选择全局最优的探索目标。
"""

import math
import random
import numpy as np

def heuristic(a, b):
    """计算两点间的欧氏距离"""
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

class MCTSNode:
    """MCTS树节点"""
    def __init__(self, visited_indices, current_position, parent=None, temp_start=None):
        self.visited = set(visited_indices)  # 已访问的候选点索引
        self.position = current_position      # 当前位置
        self.parent = parent                  # 父节点
        self.temp_start = temp_start          # 临时起点（用于计算angle）
        self.children = {}                    # {candidate_idx: child_node}
        self.visits = 0                       # 访问次数
        self.value = 0.0                      # 累积价值
        self.unvisited = []                   # 未访问的候选点索引（在初始化后设置）
    
    def set_unvisited(self, total_candidates):
        """设置未访问的候选点列表"""
        self.unvisited = [i for i in range(total_candidates) if i not in self.visited]

class MCTSPlanner:
    """
    MCTS规划器，用于选择最优frontier
    """
    
    def __init__(self, path_planner, robot_radius=5, exploration_weight=1.414):
        """
        初始化MCTS规划器
        
        Args:
            path_planner: PathPlanner实例（保留以备未来使用）
            robot_radius: 机器人半径
            exploration_weight: UCT探索权重
        """
        self.path_planner = path_planner
        self.robot_radius = robot_radius
        self.exploration_weight = exploration_weight
    
    def select_best_frontier(self, occupancy_map, current_pos, candidates, 
                            iterations, weights, temp_start, bad_frontier,
                            current_to_candidates_dist, max_candidate_distance,
                            unexplored_points, candidate_unexplored_points, num_bad_frontier,
                            turning_mode=False):
        """
        使用MCTS选择最优frontier
        
        Args:
            occupancy_map: 占用地图
            current_pos: 当前位置
            candidates: 候选点列表（已预处理过的字典列表）
            iterations: MCTS迭代次数
            weights: 评分权重字典 {'angle', 'dist', 'explore', 'avoid'}（仅用于heuristic）
            temp_start: 临时起点（用于动态判断是否需要turning around）
            bad_frontier: 已知的失败点集合
            current_to_candidates_dist: current_pos到每个candidate的path距离数组
            max_candidate_distance: candidates之间的最大直线距离（用于avoidance reward归一化）
            unexplored_points: 所有未探索点的集合
            candidate_unexplored_points: 每个candidate可以探索到的未探索点列表
            num_bad_frontier: bad_frontier点的数量
            turning_mode: 是否处于turning around模式（由外部判断并传入）
        
        Returns:
            最优frontier坐标，或None
        
        Note:
            - Heuristic函数：用于simulation中的单步更新（选择下一个frontier）
            - Reward函数：rollout结束后计算总reward，包含exploration, distance, avoidance三项
            - 如果turning_mode=True，first_step_angle_reward统一为1.0
        """
        if not candidates:
            return None
        
        # 初始化根节点（root的temp_start就是传入的temp_start）
        root = MCTSNode([], current_pos, parent=None, temp_start=temp_start)
        root.set_unvisited(len(candidates))
        
        # 计算simulation的停止条件（访问50%的candidates）
        # max_visits_in_simulation = max(1, int(len(candidates) * 0.5))
        max_visits_in_simulation = min(3, len(candidates))
        
        turning_mode_str = "TURNING MODE ⤵️" if turning_mode else "NORMAL MODE ➡️"
        print(f"[MCTS] 🔍 Starting search ({turning_mode_str}): {iterations} iterations, {len(candidates)} candidates, rollout limit={max_visits_in_simulation}")
        
        # MCTS主循环
        for iteration in range(iterations):
            # Debug控制：只在第一次迭代时输出详细reward计算
            is_first_iteration = (iteration == -1)
            
            # 1. Selection: 从root开始选择到叶节点
            node = self._select(root, candidates)
            
            # 2. Expansion: 标准MCTS逻辑
            # - Root节点：第一次访问就expansion
            # - 其他节点：只有visits > 0时才expansion
            # - visits == 0的非root节点：直接simulation
            # - 终止节点：len(visited) >= max_visits_in_simulation，不expansion
            if node.unvisited and len(node.visited) < max_visits_in_simulation:
                if node.parent is None or node.visits > 0:
                    # 扩展所有未访问的子节点，并为每个子节点计算temp_start
                    self._expand(node, candidates, weights)
                    
                    # 从新扩展的子节点中随机选择一个进行simulation
                    if node.children:
                        newly_expanded = [idx for idx in node.children.keys() 
                                         if node.children[idx].visits == 0]
                        if newly_expanded:
                            selected_idx = random.choice(newly_expanded)
                            node = node.children[selected_idx]
            
            # 3. Simulation: 从当前节点rollout到50%的candidates
            reward = self._simulate(
                node, candidates, occupancy_map, weights, temp_start, bad_frontier, max_visits_in_simulation,
                current_to_candidates_dist, max_candidate_distance,
                unexplored_points, candidate_unexplored_points, num_bad_frontier,
                is_first_iteration, turning_mode, current_pos
            )
            
            # 4. Backpropagation: 更新路径上所有节点
            self._backpropagate(node, reward)
        
        # 计算树的最大深度
        max_tree_depth = self._get_max_depth(root)
        print(f"[MCTS] ✅ Search completed: tree depth={max_tree_depth}")
        
        # 选择最优子节点（基于avg_value，即exploitation）
        if not root.children:
            return None
        
        # 计算所有子节点的UCB值和avg_value（用于调试输出）
        child_stats = []
        for idx, child in root.children.items():
            avg_value = child.value / max(1, child.visits)
            # 计算UCB值
            if child.visits > 0:
                exploitation = child.value / child.visits
                exploration = self.exploration_weight * math.sqrt(
                    math.log(root.visits) / child.visits
                )
                ucb = exploitation + exploration
            else:
                exploitation = 0
                exploration = 0
                ucb = float('inf')
            
            child_stats.append({
                'idx': idx,
                'coord': candidates[idx]['coord'],
                'visits': child.visits,
                'value': child.value,
                'avg_value': avg_value,
                'exploitation': exploitation,
                'exploration': exploration,
                'ucb': ucb
            })
        
        # 按avg_value降序排序（NaN排在最后）
        child_stats.sort(key=lambda x: (np.isnan(x['avg_value']), -x['avg_value'] if not np.isnan(x['avg_value']) else 0))
        
        # 输出Top 5
        top_k = min(5, len(child_stats))
        print(f"[MCTS] 📊 Top {top_k} candidates by avg_value:")
        for i in range(top_k):
            stat = child_stats[i]
            print(f"  #{i+1}: idx={stat['idx']}, coord={stat['coord']}, "
                  f"avg_value={stat['avg_value']:.4f}, UCB={stat['ucb']:.4f} "
                  f"(exploit={stat['exploitation']:.4f}, explore={stat['exploration']:.4f}), "
                  f"visits={stat['visits']}")
        
        # 选择avg_value最高的（Max Child策略）
        # 跳过NaN值，选择第一个非NaN的候选点
        best_candidate = None
        for stat in child_stats:
            if not np.isnan(stat['avg_value']):
                best_candidate = stat
                break
        
        if best_candidate is None:
            # 所有候选点都是NaN，选择visits最多的（静默处理）
            best_candidate = max(child_stats, key=lambda x: x['visits'])
        
        best_idx = best_candidate['idx']
        best_child = root.children[best_idx]
        best_coord = best_candidate['coord']
        
        # 输出最优UCT路径（从选中的节点开始的长程规划）
        print(f"[MCTS] 🗺️  Long-term Planning - Best UCT path from root:")
        # 先输出选中的根节点（Step 1）
        print(f"  Step 1: Candidate #{best_idx} {best_coord} | "
              f"avg_value={best_candidate['avg_value']:.4f}, UCB={best_candidate['ucb']:.4f} "
              f"(exploit={best_candidate['exploitation']:.4f}, explore={best_candidate['exploration']:.4f}) | "
              f"visits={best_child.visits}")
        # 然后递归输出其子树的最优路径
        self._print_best_uct_path(best_child, candidates, depth=1, max_depth=5, temp_start=temp_start, current_pos=current_pos)
        
        # 添加最优路径的reward计算调试
        print(f"\n[MCTS] 🔬 Debug - Evaluating Best UCT path reward:")
        self._debug_evaluate_best_path(root, best_child, candidates, current_to_candidates_dist, 
                                      max_candidate_distance, unexplored_points, candidate_unexplored_points,
                                      num_bad_frontier, bad_frontier, max_visits_in_simulation,
                                      temp_start, current_pos, turning_mode, weights)
        
        print(f"[MCTS] 🎯 Selected: idx={best_idx}, coord={best_coord}, avg_value={best_candidate['avg_value']:.4f}, visits={best_child.visits}")
        
        return best_coord
    
    def _select(self, node, candidates):
        """
        Selection阶段: 使用UCT公式选择节点
        
        严格按照UCT公式，一层层选择UCT值最大的子节点，直到叶子节点
        
        UCT = exploitation + exploration
        exploitation = node.value / node.visits
        exploration = c * sqrt(log(parent.visits) / node.visits)
        """
        while node.children and not node.unvisited:
            best_score = -float('inf')
            best_child = None
            
            for idx, child in node.children.items():
                # UCT公式：exploit + explore
                exploit = child.value / (child.visits + 1e-6)
                explore = self.exploration_weight * math.sqrt(
                    math.log(node.visits + 1) / (child.visits + 1e-6)
                )
                uct_score = exploit + explore
                
                if uct_score > best_score:
                    best_score = uct_score
                    best_child = child
            
            node = best_child if best_child else node
        
        return node
    
    def _expand(self, parent, candidates, weights):
        """
        Expansion阶段: 扩展所有未访问的子节点（标准MCTS）
        
        在叶子节点处，一次性创建所有可能的子节点（对应所有未访问的candidates）
        同时为每个子节点计算temp_start：
        - 基于parent的temp_start判断是否需要turning around
        - 如果不需要turning：child.temp_start = parent.temp_start（继承）
        - 如果需要turning：使用_compute_updated_temp_start计算新的temp_start
          （从parent.temp_start到parent.position方向延伸到child.position前方）
        """
        for candidate_idx in list(parent.unvisited):  # 使用list复制，避免迭代时修改
            new_visited = parent.visited | {candidate_idx}
            new_position = candidates[candidate_idx]['coord']
            
            # 判断从parent到这个child是否需要turning around
            # 这个判断基于parent的temp_start和parent→child的移动
            need_turning = self._check_if_need_turning(
                parent.position, parent.temp_start, new_position, 
                candidates, parent.visited, weights.get('angle', 1.0)
            )
            
            # 确定child的temp_start
            if need_turning:
                # turning around，使用_compute_updated_temp_start计算新的temp_start
                # 从parent.temp_start到parent.position的方向延伸到child位置前方
                child_temp_start = self._compute_updated_temp_start(
                    parent.temp_start, parent.position, step_size=10.0
                )
            else:
                # 不需要turning，继承parent的temp_start
                child_temp_start = parent.temp_start
            
            child = MCTSNode(new_visited, new_position, parent, temp_start=child_temp_start)
            child.set_unvisited(len(candidates))
            
            parent.children[candidate_idx] = child
        
        # 清空unvisited列表（所有子节点都已创建）
        parent.unvisited = []
    
    def _simulate(self, node, candidates, occupancy_map, weights, temp_start, bad_frontier, 
                  max_visits_in_simulation,
                  current_to_candidates_dist, max_candidate_distance,
                  unexplored_points, candidate_unexplored_points, num_bad_frontier,
                  is_first_iteration, turning_mode, real_current_pos):
        """
        Simulation阶段: 使用greedy策略rollout + 新的reward函数
        
        从当前节点开始，贪心地选择未访问的候选点（使用heuristic函数）。
        Rollout结束后，使用新的reward函数计算总reward（exploration + distance + avoidance）。
        
        Heuristic函数：用于选择下一个frontier（angle, distance, explore, avoid）
        Reward函数：rollout结束后计算（exploration_reward, distance_reward, avoidance_reward）
        
        关键：
        1. 在每一步到达新位置后，都要动态判断是否需要turning around
        2. 基于simulation的current_pos动态计算所有候选点的距离和评分
        3. 动态计算avoid_score的归一化范围（基于simulation的visited + bad_frontier）
        4. 只有explore_score使用预计算值（因为地图不变）
        
        Args:
            temp_start: 真实的initial temp_start（用于first step angle计算）
            real_current_pos: 真实的agent当前位置（等于root.position，用于first step angle计算）
            turning_mode: 整个搜索是否处于turning around模式（若是，angle reward统一为1.0）
        
        Note:
            real_current_pos理论上等于root节点的position坐标
        """
        visited = set(node.visited)
        current_pos = node.position
        sim_temp_start = node.temp_start
        
        # Rollout: 从node继续，使用greedy策略选择
        step_num = 0
        
        
        while len(visited) < max_visits_in_simulation:
            step_num += 1
            
            # 每一步都重新判断是否需要turning around
            sim_turning, new_temp_start = self._check_turning_with_update(
                current_pos, sim_temp_start, candidates, visited, 
                w_angle=weights.get('angle', 1.0)
            )
            
            # 如果需要更新temp_start，立即更新
            if new_temp_start is not None:
                sim_temp_start = new_temp_start
            
            # 第一遍：计算所有未访问候选点的基础数据
            # 构建simulation中的避免点集合
            avoid_points = []
            visited_coords = [candidates[v]['coord'] for v in visited]
            avoid_points.extend(visited_coords)
            if bad_frontier:
                avoid_points.extend(list(bad_frontier))
            
            # 计算所有未访问候选点的距离和避让信息，找出动态归一化范围
            unvisited_candidates_data = []
            min_avoid_dist = float('inf')
            max_avoid_dist = 0.0
            min_dist = float('inf')
            max_dist = 0.0
            
            for i, cand in enumerate(candidates):
                if i in visited:
                    continue
                
                coord = cand['coord']
                
                # 计算从simulation的current_pos到候选点的距离
                d_sim = heuristic(current_pos, coord)
                
                # 计算到避免点的最小距离
                if avoid_points:
                    min_dist_to_avoid = min(heuristic(coord, ap) for ap in avoid_points)
                else:
                    min_dist_to_avoid = float('inf')
                
                unvisited_candidates_data.append((i, coord, d_sim, min_dist_to_avoid))
                
                # 更新距离范围（用于动态计算mid_d和max_diff）
                min_dist = min(min_dist, d_sim)
                max_dist = max(max_dist, d_sim)
                
                # 更新避让距离范围
                if avoid_points and min_dist_to_avoid != float('inf'):
                    min_avoid_dist = min(min_avoid_dist, min_dist_to_avoid)
                    max_avoid_dist = max(max_avoid_dist, min_dist_to_avoid)
            
            # 动态计算mid_d和max_diff（基于simulation的current_pos，与原始函数一致）
            mid_d_sim = 0.3 * min_dist + 0.7 * max_dist
            distances_to_mid = [abs(d - mid_d_sim) for _, _, d, _ in unvisited_candidates_data]
            max_diff_sim = max(distances_to_mid) if distances_to_mid else 0.0  # 与原始函数一致
            
            # 第二遍：使用动态归一化范围计算评分
            best_score = -float('inf')
            best_idx = None
            best_coord = None
            
            # 存储所有候选点的评分用于debug
            all_scores = []
            
            for i, coord, d_sim, min_dist_to_avoid in unvisited_candidates_data:
                cand = candidates[i]
                
                # 动态归一化avoid_score（与原始函数完全一致）
                if avoid_points and min_dist_to_avoid != float('inf'):
                    if (max_avoid_dist - min_avoid_dist) != 0:
                        avoid_score = (min_dist_to_avoid - min_avoid_dist) / (max_avoid_dist - min_avoid_dist)
                    else:
                        avoid_score = 0.0  # 与原始函数一致
                else:
                    # 如果没有avoid_points，说明没有需要避让的点，给予满分
                    avoid_score = 1.0
                
                # 动态计算angle_score（与原始函数一致）
                if sim_turning:
                    # TURNING MODE: angle_score固定为0.5
                    angle_score = 0.5
                else:
                    theta = abs(self._calculate_heading(sim_temp_start, current_pos, coord))
                    # 使用更陡峭的sigmoid函数（k=0.4）
                    angle_score = 1.0 / (1.0 + math.exp(0.4 * (theta - 60.0)))
                
                # 动态计算distance_score（与原始函数完全一致）
                if max_diff_sim != 0:
                    diff_norm = abs(d_sim - mid_d_sim) / max_diff_sim
                else:
                    diff_norm = 0.0
                distance_score = 1 - diff_norm
                
                # 使用预计算的explore_score（地图不变，可复用）
                explore_score = cand['explore_score']
                
                # 总分（greedy heuristic）- 这个分数同时作为该步的reward
                # 与原始函数保持一致：需要检查distance_score, avoid_score, explore_score都>0
                if distance_score > 0 and avoid_score > 0 and explore_score > 0:
                    # value_score = (weights['angle'] * angle_score + 
                    #               weights['dist'] * distance_score +
                    #               weights['explore'] * explore_score + 
                    #               weights['avoid'] * avoid_score)
                    value_score = (weights['dist'] * distance_score +
                                  weights['explore'] * explore_score + 
                                  weights['avoid'] * avoid_score)
                else:
                    value_score = 0.0  # 任何一项为0，总分为0
                
                if value_score > best_score:
                    best_score = value_score
                    best_idx = i
                    best_coord = coord
            
            if best_idx is None:
                break
            
            visited.add(best_idx)
            current_pos = best_coord
        
        # 计算新的reward（exploration + distance + avoidance，1:1:1加权）
        # 获取完整路径：从root到node的selection路径 + rollout路径
        
        # selection_indices: 从root到node路径中访问的所有candidates（按顺序）
        # 通过回溯parent链并比较visited集合来获取正确的访问顺序
        selection_indices = []
        current_node = node
        temp_path = []
        
        while current_node.parent is not None:
            temp_path.append(current_node)
            current_node = current_node.parent
        temp_path.reverse()  # root方向到node方向
        
        # 从每个节点提取相对于其parent新增的candidate
        for path_node in temp_path:
            if path_node.parent:
                parent_visited = path_node.parent.visited
                node_visited = path_node.visited
                # 找到新增的candidate
                new_candidates = node_visited - parent_visited
                if new_candidates:
                    # 应该只有一个新增的candidate
                    selection_indices.extend(list(new_candidates))
        
        # rollout_indices: rollout中新访问的candidates索引
        rollout_indices = [idx for idx in visited if idx not in node.visited]
        
        # 计算新的reward函数
        # 关键：first step angle应该用真实的initial temp_start和root位置
        # temp_start: 真实的initial temp_start（函数参数传入）
        # real_current_pos: 真实的agent当前位置（root节点位置，函数参数传入）
        
        reward = self._calculate_reward(
            selection_indices, rollout_indices,
            candidates, current_to_candidates_dist, max_candidate_distance,
            unexplored_points, candidate_unexplored_points,
            num_bad_frontier, bad_frontier, max_visits_in_simulation,
            is_first_iteration, 
            temp_start=temp_start,          # 使用真实的initial temp_start
            current_pos=real_current_pos,   # 使用真实的root位置
            turning_mode=turning_mode,
            weights=weights                 # 传入权重参数
        )
        
        return reward
    
    def _calculate_reward(self, selection_indices, rollout_indices,
                              candidates, current_to_candidates_dist, max_candidate_distance,
                              unexplored_points, candidate_unexplored_points,
                              num_bad_frontier, bad_frontier, max_visits_in_simulation,
                              is_first_iteration, temp_start=None, current_pos=None, turning_mode=False, weights=None):
        """
        计算reward函数（exploration + distance + avoidance + first_step_angle，调和平均）
        
        Reward = 4 / (1/exploration + 1/distance + 1/avoidance + 1/angle)
        
        1. Exploration Reward: min(visited candidates的探索点数) / max(所有candidates的探索点数)
        2. Distance Reward: 1 - (总path长度) / (理论最大长度)
        3. Avoidance Reward: min(candidates到bad_frontier的距离) / max(candidates到bad_frontier的距离)
        4. First Step Angle Reward: 第一步的角度保持奖励（使用真实的initial temp_start和root位置）
           - 如果turning_mode=True，angle reward统一为1.0
        
        Args:
            selection_indices: selection路径中访问的candidates索引
            rollout_indices: rollout中新访问的candidates索引
            candidates: 所有候选点列表
            current_to_candidates_dist: current_pos到每个candidate的path距离数组
            max_candidate_distance: candidates之间的最大直线距离（用于avoidance reward归一化）
            unexplored_points: 所有未探索点的集合
            candidate_unexplored_points: 每个candidate可以探索到的未探索点列表
            num_bad_frontier: bad_frontier点的数量
            max_visits_in_simulation: 最大访问数量
            is_first_iteration: 是否是第一次迭代（用于控制debug输出）
            temp_start: 真实的initial temp_start（用于计算首步角度）
            current_pos: 真实的agent当前位置（等于root.position，用于计算首步角度）
            turning_mode: 整个搜索是否处于turning around模式（若是，angle reward统一为1.0）
        
        Returns:
            总reward（调和平均，理论上0-1之间）
        
        Note:
            first_step_angle_reward使用真实的initial状态（temp_start和current_pos）来计算
            整条路径的第一步相对于agent当前运动方向的角度，而不是使用树中节点的temp_start
        """
        all_visited_indices = selection_indices + rollout_indices
        num_candidates = len(candidates)
        
        if is_first_iteration:
            print(f"\n{'='*80}")
            print(f"[MCTS] 🧮 Reward Calculation Debug")
            print(f"{'='*80}")
            print(f"\n📋 Path Information:")
            print(f"  • Selection path: {selection_indices} (from root to leaf)")
            if len(selection_indices) != len(set(selection_indices)):
                print(f"  ⚠️  WARNING: Selection path contains duplicates!")
            print(f"  • Rollout path: {rollout_indices} (greedy simulation)")
            print(f"  • Total visited (unique): {list(set(all_visited_indices))}")
            print(f"  • Num candidates: {num_candidates}, Num bad_frontier: {num_bad_frontier}")
            print(f"\n📊 Normalization Constants:")
            print(f"  • max_visits_in_simulation: {max_visits_in_simulation}")
            print(f"  • max_candidate_distance: {max_candidate_distance:.1f}")
            print(f"  • total_unexplored_points: {len(unexplored_points)}")
        
        # ========== 第一项：Exploration Reward（时间折扣策略）==========
        # 策略：对路径上的点应用时间折扣，越早访问的点权重越高
        # gamma = 0.8，第1步权重1.0，第2步权重0.8，第3步权重0.64...
        
        gamma_exploration = 0.3  # 探索折扣因子
        first_step_explore_ratio = None
        
        # 计算理论最大值：最好的candidate的探索点数
        max_exploration_points = 0
        for idx in range(len(candidate_unexplored_points)):
            num_points = len(candidate_unexplored_points[idx])
            max_exploration_points = max(max_exploration_points, num_points)
        
        if max_exploration_points > 0:
            # 按访问顺序（selection + rollout）计算折扣后的exploration
            discounted_exploration = 0.0
            for step_idx, candidate_idx in enumerate(all_visited_indices):
                if candidate_idx < len(candidate_unexplored_points):
                    exploration_points = len(candidate_unexplored_points[candidate_idx])
                    discount = gamma_exploration ** step_idx
                    discounted_exploration += discount * exploration_points
                    if step_idx == 0 and max_exploration_points > 0:
                        first_step_explore_ratio = exploration_points / max_exploration_points
                        first_step_explore_ratio = max(0.0, min(1.0, first_step_explore_ratio))
            
            # 计算理论最大折扣值（假设每步都访问最好的candidate）
            max_discounted = sum(gamma_exploration ** i for i in range(len(all_visited_indices))) * max_exploration_points
            
            # exploration_reward = 实际折扣探索值 / 理论最大折扣值
            exploration_reward = discounted_exploration / max_discounted if max_discounted > 0 else 0.0
            
            if is_first_iteration:
                print(f"\n[1] 🔍 Exploration Reward (Temporal Discount γ={gamma_exploration}):")
                print(f"  • Best candidate can explore: {max_exploration_points} points")
                print(f"  • Path order (selection → rollout): {all_visited_indices}")
                print(f"  • Discounted exploration calculation:")
                for step_idx, candidate_idx in enumerate(all_visited_indices):
                    if candidate_idx < len(candidate_unexplored_points):
                        exploration_points = len(candidate_unexplored_points[candidate_idx])
                        discount = gamma_exploration ** step_idx
                        weighted = discount * exploration_points
                        coord = candidates[candidate_idx]['coord']
                        print(f"    Step {step_idx+1}: Cand#{candidate_idx} {coord}")
                        print(f"      → {discount:.3f} × {exploration_points} = {weighted:.1f}")
                print(f"  • Total discounted exploration: {discounted_exploration:.1f}")
                print(f"  • Max discounted (if all steps visit best): {max_discounted:.1f}")
                print(f"  → exploration_reward = {discounted_exploration:.1f} / {max_discounted:.1f} = {exploration_reward:.4f}")
                print(f"  💡 Key: First step gets full weight (1.0), later steps decay by {gamma_exploration}")
        else:
            exploration_reward = 0.0
            if first_step_explore_ratio is None:
                first_step_explore_ratio = 0.0
            if is_first_iteration:
                print(f"\n[1] Exploration Reward:")
                print(f"  No unexplored points accessible by candidates")
                print(f"  → exploration_reward = 0.0")
        
        # ========== 第二项：Distance Reward ==========
        # 计算整个路径的总path长度（包括从current_pos到第一个candidate）
        # 使用去重的路径来避免重复计数相同的candidate
        unique_path = []
        for idx in all_visited_indices:
            if not unique_path or idx != unique_path[-1]:
                unique_path.append(idx)
        
        total_path_length = 0.0
        path_segments = []
        
        # 添加从current_pos到第一个candidate的距离
        dist_to_first = None
        first_step_distance_ratio = None
        if len(unique_path) > 0:
            first_idx = unique_path[0]
            if first_idx < len(current_to_candidates_dist):
                dist_to_first = current_to_candidates_dist[first_idx]
                if dist_to_first > 0:
                    total_path_length += dist_to_first
                    if is_first_iteration:
                        coord_first = candidates[first_idx]['coord']
                        path_segments.append((-1, 'current_pos', first_idx, coord_first, dist_to_first))
        
        # 添加candidates之间的距离（使用直线距离估计）
        if len(unique_path) > 1:
            for i in range(len(unique_path) - 1):
                idx1 = unique_path[i]
                idx2 = unique_path[i + 1]
                if idx1 < num_candidates and idx2 < num_candidates:
                    coord1 = candidates[idx1]['coord']
                    coord2 = candidates[idx2]['coord']
                    dist = heuristic(coord1, coord2)
                    if dist > 0:
                        total_path_length += dist
                        if is_first_iteration:
                            path_segments.append((idx1, coord1, idx2, coord2, dist))
        
        # 理论最大长度 = max_visits_in_simulation * 最大两点距离
        max_theoretical_length = max_visits_in_simulation * max_candidate_distance
        
        # 使用反比例函数：distance_reward = 1 - (total_path_length / max_theoretical_length)
        if max_candidate_distance > 0 and dist_to_first is not None and dist_to_first >= 0:
            ratio_first = 1.0 - (dist_to_first / max_candidate_distance)
            first_step_distance_ratio = max(0.0, min(1.0, ratio_first))

        if max_theoretical_length > 0:
            distance_reward = 1.0 - (total_path_length / max_theoretical_length)
            distance_reward = max(0.0, min(1.0, distance_reward))
        else:
            distance_reward = 0.0
        
        if is_first_iteration:
            print(f"\n[2] 📏 Distance Reward:")
            print(f"  • Path segments (current_pos → selection → rollout):")
            for segment in path_segments:
                if segment[0] == -1:
                    # current_pos到第一个candidate
                    idx2, coord2, dist = segment[2], segment[3], segment[4]
                    print(f"    - current_pos → Cand#{idx2} {coord2}: {dist:.1f}")
                else:
                    idx1, coord1, idx2, coord2, dist = segment
                    print(f"    - Cand#{idx1} {coord1} → Cand#{idx2} {coord2}: {dist:.1f}")
            print(f"  • Total path length: {total_path_length:.1f}")
            print(f"  • Max theoretical length: {max_visits_in_simulation} × {max_candidate_distance:.1f} = {max_theoretical_length:.1f}")
            if max_theoretical_length > 0:
                ratio = total_path_length / max_theoretical_length
                print(f"  • Path ratio: {total_path_length:.1f} / {max_theoretical_length:.1f} = {ratio:.4f}")
                print(f"  → distance_reward = 1 - {ratio:.4f} = {distance_reward:.4f}")
            else:
                print(f"  → distance_reward = 0.0 (invalid range)")
        
        # ========== 第三项：Avoidance Reward（时间折扣策略）==========
        # 策略：对路径上每个点的避障距离应用时间折扣，优先考虑第一步的avoidance
        # gamma = 0.9（比exploration稍高，因为avoidance也很重要）
        
        gamma_avoidance = 0.3  # 避障折扣因子
        first_step_avoid_ratio = None
        
        # 将bad_frontier转换为列表
        bad_frontier_list = list(bad_frontier) if bad_frontier else []
        
        max_possible_distance = 0.0
        if len(bad_frontier_list) > 0:
            # 按访问顺序计算每个candidate到bad_frontier的最小距离
            # 然后应用时间折扣
            discounted_avoidance = 0.0
            max_possible_distance = 0.0
            
            # 先计算所有candidates到bad_frontier的最大可能距离（用于归一化）
            for idx in range(len(candidates)):
                coord = candidates[idx]['coord']
                for bad_coord in bad_frontier_list:
                    dist = heuristic(coord, bad_coord)
                    max_possible_distance = max(max_possible_distance, dist)
            
            # 按访问顺序计算折扣后的avoidance
            for step_idx, candidate_idx in enumerate(all_visited_indices):
                if candidate_idx < len(candidates):
                    coord = candidates[candidate_idx]['coord']
                    
                    # 找到这个candidate到所有bad_frontier的最小距离
                    min_dist_to_bad = float('inf')
                    closest_bad_idx = None
                    for bad_idx, bad_coord in enumerate(bad_frontier_list):
                        dist = heuristic(coord, bad_coord)
                        if dist < min_dist_to_bad:
                            min_dist_to_bad = dist
                            closest_bad_idx = bad_idx
                    
                    # 应用折扣（不在这里归一化）
                    if min_dist_to_bad != float('inf'):
                        discount = gamma_avoidance ** step_idx
                        discounted_avoidance += discount * min_dist_to_bad
                        if step_idx == 0 and max_possible_distance > 0:
                            first_step_avoid_ratio = min_dist_to_bad / max_possible_distance
                            first_step_avoid_ratio = max(0.0, min(1.0, first_step_avoid_ratio))
            
            # 计算理论最大折扣值（每步都是最大距离）
            max_discounted_avoidance = sum(gamma_avoidance ** i for i in range(len(all_visited_indices))) * max_possible_distance
            
            # avoidance_reward = 实际折扣避障值 / 理论最大折扣值
            avoidance_reward = discounted_avoidance / max_discounted_avoidance if max_discounted_avoidance > 0 else 0.0
            
            if is_first_iteration:
                print(f"\n[3] 🚫 Avoidance Reward (Temporal Discount γ={gamma_avoidance}):")
                print(f"  • Total bad frontiers: {len(bad_frontier_list)}")
                print(f"  • Max possible distance to bad frontier: {max_possible_distance:.1f}")
                print(f"  • Path order (selection → rollout): {all_visited_indices}")
                print(f"  • Discounted avoidance calculation:")
                
                for step_idx, candidate_idx in enumerate(all_visited_indices):
                    if candidate_idx < len(candidates):
                        coord = candidates[candidate_idx]['coord']
                        
                        # 重新计算这个点的最小距离（用于显示）
                        min_dist_to_bad = float('inf')
                        closest_bad_idx = None
                        for bad_idx, bad_coord in enumerate(bad_frontier_list):
                            dist = heuristic(coord, bad_coord)
                            if dist < min_dist_to_bad:
                                min_dist_to_bad = dist
                                closest_bad_idx = bad_idx
                        
                        discount = gamma_avoidance ** step_idx
                        weighted = discount * min_dist_to_bad
                        
                        print(f"    Step {step_idx+1}: Cand#{candidate_idx} {coord}")
                        print(f"      → min_dist={min_dist_to_bad:.1f}")
                        print(f"      → {discount:.3f} × {min_dist_to_bad:.1f} = {weighted:.1f}")
                
                print(f"  • Total discounted avoidance: {discounted_avoidance:.1f}")
                print(f"  • Max discounted (if all at max distance): (1+γ+γ²+...)×{max_possible_distance:.1f} = {max_discounted_avoidance:.1f}")
                print(f"  → avoidance_reward = {discounted_avoidance:.1f} / {max_discounted_avoidance:.1f} = {avoidance_reward:.4f}")
                print(f"  💡 Key: First step's avoidance gets full weight (1.0), later steps decay")
        else:
            # 没有bad_frontier时，设为1.0（最佳）
            avoidance_reward = 1.0
            first_step_avoid_ratio = 1.0
            if is_first_iteration:
                print(f"\n[3] 🚫 Avoidance Reward:")
                print(f"  • No bad frontiers → avoidance_reward = 1.0")

        if first_step_avoid_ratio is None:
            first_step_avoid_ratio = 0.0 if max_possible_distance > 0 else 1.0
        
        # ========== 第四项：First Step Angle Reward ==========
        # 计算第一步的角度保持奖励
        
        # 如果整个搜索处于turning around模式，angle reward直接为1.0，不做任何计算
        if turning_mode:
            first_step_angle_reward = 1.0
        elif temp_start is not None and current_pos is not None and len(all_visited_indices) > 0:
            first_step_idx = all_visited_indices[0]
            if first_step_idx < len(candidates):
                first_step_coord = candidates[first_step_idx]['coord']
                
                # 计算角度，使用与choose_frontier相同的sigmoid函数（steepness=0.2, threshold=60°）
                theta = abs(self._calculate_heading(temp_start, current_pos, first_step_coord))
                angle_threshold_deg = 60.0
                steepness = 0.2
                first_step_angle_reward = 1.0 / (1.0 + math.exp(steepness * (theta - angle_threshold_deg)))
            else:
                first_step_angle_reward = 0.0
        else:
            first_step_angle_reward = 0.0
        
        if is_first_iteration:
            print(f"\n[4] 🎯 First Step Angle Reward:")
            if turning_mode:
                print(f"  • 🔄 TURNING MODE: angle reward直接设为 1.0（不计算角度）")
            elif temp_start is not None and current_pos is not None and len(all_visited_indices) > 0:
                first_step_idx = all_visited_indices[0]
                if first_step_idx < len(candidates):
                    first_step_coord = candidates[first_step_idx]['coord']
                    print(f"  • First step: Candidate #{first_step_idx} {first_step_coord}")
                    print(f"  • From current_pos {current_pos} to temp_start {temp_start}")
                    if temp_start != current_pos:
                        theta = abs(self._calculate_heading(temp_start, current_pos, first_step_coord))
                        print(f"  • Angle theta = {theta:.1f}°")
                        print(f"  • Angle score = 1.0 / (1.0 + exp(0.2×({theta:.1f} - 60.0))) = {first_step_angle_reward:.4f}")
                    else:
                        print(f"  • temp_start == current_pos, angle = 0°")
                        print(f"  • Angle score = {first_step_angle_reward:.4f}")
                else:
                    print(f"  • Invalid first step index: {first_step_idx}")
            else:
                print(f"  • Missing temp_start or current_pos or no visited indices")
                print(f"  • temp_start: {temp_start}, current_pos: {current_pos}, visited: {len(all_visited_indices)}")
            print(f"  → first_step_angle_reward = {first_step_angle_reward:.4f}")
        
        # ========== 总Reward计算 ==========
        epsilon = 1e-10
        w_exploration = weights['explore']
        w_distance = weights['dist']
        w_avoidance = weights['avoid']
        w_angle = weights['angle']

        # 直接使用安全值（无软阈值）
        exploration_safe = max(exploration_reward, epsilon)
        distance_safe = max(distance_reward, epsilon)
        avoidance_safe = max(avoidance_reward, epsilon)
        angle_safe = max(first_step_angle_reward, epsilon)

        total_weight = w_exploration + w_distance + w_avoidance + w_angle
        total_reward = (
            w_exploration * exploration_safe +
            w_distance * distance_safe +
            w_avoidance * avoidance_safe +
            w_angle * angle_safe
        ) / max(total_weight, epsilon)

        # 首步阈值守卫
        first_step_explore_ratio = min(1.0, max(0.0, first_step_explore_ratio or 0.0))
        first_step_distance_ratio = min(1.0, max(0.0, first_step_distance_ratio or 0.0))
        first_step_avoid_ratio = min(1.0, max(0.0, first_step_avoid_ratio or 0.0))
        first_step_angle_clamped = min(1.0, max(0.0, first_step_angle_reward))

        first_step_guard_failed = any(
            ratio < 0.1 for ratio in (
                first_step_explore_ratio,
                first_step_avoid_ratio,
                # first_step_angle_clamped
            )
        )

        if first_step_guard_failed:
            total_reward = 0.0
            if is_first_iteration:
                print("\n⚠️  First-step guard triggered: one or more first-step ratios ≤ 0.1 → total_reward forced to 0.0")
        elif is_first_iteration:
            print("\n✅ First-step guard passed:")
            print(f"  • explore_ratio = {first_step_explore_ratio:.4f}")
            print(f"  • avoid_ratio = {first_step_avoid_ratio:.4f}")
            print(f"  • angle = {first_step_angle_clamped:.4f}")
        
        # NaN检查和修复（静默处理）
        if np.isnan(total_reward):
            total_reward = 0.0
        
        if is_first_iteration:
            print(f"\n{'='*80}")
            print(f"[MCTS] 🎯 Total Reward Summary:")
            
            print(f"\n📊 Core reward components:")
            print(f"  • exploration: {exploration_safe:.4f}")
            print(f"  • distance:    {distance_safe:.4f}")
            print(f"  • avoidance:   {avoidance_safe:.4f}")
            print(f"  • angle:       {angle_safe:.4f}")
            
            print(f"\n🔧 Weighted Arithmetic Mean (w={w_exploration:.1f},{w_distance:.1f},{w_avoidance:.1f},{w_angle:.1f}):")
            print(f"  → total = ({w_exploration:.1f}×{exploration_safe:.4f} + {w_distance:.1f}×{distance_safe:.4f} + {w_avoidance:.1f}×{avoidance_safe:.4f} + {w_angle:.1f}×{angle_safe:.4f}) / {total_weight:.1f}")
            print(f"  → total_reward = {total_reward:.4f}")
            print(f"{'='*80}\n")
        
        return total_reward
    
    def _compute_updated_temp_start(self, old_temp_start, now, step_size=10.0):
        """
        计算新的temp_start（模拟landmark.py中的_update_temp_start函数）
        
        不是简单地将now设为temp_start，而是：
        1. 计算从old_temp_start到now的方向向量
        2. 沿这个方向，从now往前延伸step_size（默认10像素）
        3. 将延伸后的点作为新的temp_start
        
        这样可以保持"方向记忆"，temp_start始终在now的前方
        
        Args:
            old_temp_start: 旧的temp_start
            now: 当前位置
            step_size: 延伸距离（像素）
            
        Returns:
            新的temp_start坐标
        """
        dx = now[0] - old_temp_start[0]
        dy = now[1] - old_temp_start[1]
        length = math.sqrt(dx**2 + dy**2)
        
        if length == 0:
            # 如果距离为0，无法确定方向，直接返回now
            return now
        
        step_x = (dx / length) * step_size
        step_y = (dy / length) * step_size
        new_temp_start = (round(now[0] + step_x), round(now[1] + step_y))
        
        return new_temp_start
    
    def _check_turning_with_update(self, current_pos, temp_start, candidates, visited, w_angle=1.0):
        """
        在simulation中动态判断是否需要turning around，并返回是否需要更新temp_start
        
        判断逻辑（参考原choose_frontier函数）：
        1. 如果w_angle == 0.0，不需要turning（特殊情况）
        2. 如果current_pos == temp_start，不需要turning
        3. 计算所有未访问候选点的角度差
        4. 取前20%角度最小的候选点
        5. 如果其中有角度<90度且距离>30的，不需要turning
        6. 否则需要turning，使用_compute_updated_temp_start计算新的temp_start
        
        Args:
        
        Returns:

        """
        # 特殊情况：如果w_angle == 0.0，不需要turning
        if w_angle == 0.0:
            return False, None
        
        # 如果当前位置等于temp_start，不需要turning
        if current_pos == temp_start:
            return False, None
        
        # 获取未访问的候选点
        unvisited_candidates = [(i, cand) for i, cand in enumerate(candidates) if i not in visited]
        
        if not unvisited_candidates:
            return False, None
        
        # 计算所有未访问候选点的角度差
        candidates_with_angles = []
        for i, cand in unvisited_candidates:
            coord = cand['coord']
            theta = abs(self._calculate_heading(temp_start, current_pos, coord))
            candidates_with_angles.append((theta, coord, i))
        
        candidates_with_angles.sort()
        
        # 取前20%角度最小的（使用math.floor与原始函数一致）
        top_k = max(1, math.floor(0.2 * len(candidates_with_angles)))
        
        # 检查前20%中是否有角度<90度且距离>30的点
        good_candidates = []
        for theta, coord, idx in candidates_with_angles[:top_k]:
            dist = heuristic(current_pos, coord)
            if theta < 90 and dist > 30:
                good_candidates.append((theta, coord, idx, dist))
        
        if good_candidates:
            # 不需要turning，找到了好的前进方向
            return False, None
        
        # 前方没有好的frontier，需要turning
        # 使用_compute_updated_temp_start计算新的temp_start（从temp_start到current_pos方向延伸）
        new_temp_start = self._compute_updated_temp_start(temp_start, current_pos, step_size=10.0)
        
        return True, new_temp_start
    
    def _check_if_need_turning(self, parent_pos, parent_temp_start, child_pos, 
                                candidates, parent_visited, w_angle=1.0):
        """
        判断从parent到child是否需要turning around（用于expansion阶段）
        
        这个函数用于确定子节点的temp_start：
        - 如果不需要turning：child.temp_start = parent.temp_start
        - 如果需要turning：child.temp_start = child_pos
        
        Args:
            parent_pos: 父节点位置
            parent_temp_start: 父节点的temp_start
            child_pos: 子节点位置
            candidates: 候选点列表
            parent_visited: 父节点已访问的candidates
            w_angle: angle权重
        
        Returns:
            bool: True表示需要turning，False表示不需要
        """
        # 特殊情况：如果w_angle == 0.0，不需要turning
        if w_angle == 0.0:
            return False
        
        # 如果parent_pos == parent_temp_start，不需要turning
        if parent_pos == parent_temp_start:
            return False
        
        # 获取未访问的候选点（从parent的角度）
        unvisited_candidates = [(i, cand) for i, cand in enumerate(candidates) 
                               if i not in parent_visited]
        
        if not unvisited_candidates:
            return False
        
        # 计算所有未访问候选点相对于parent的角度差
        candidates_with_angles = []
        for i, cand in unvisited_candidates:
            coord = cand['coord']
            theta = abs(self._calculate_heading(parent_temp_start, parent_pos, coord))
            candidates_with_angles.append((theta, coord, i))
        
        candidates_with_angles.sort()
        
        # 取前20%角度最小的（使用math.floor与原始函数一致）
        top_k = max(1, math.floor(0.2 * len(candidates_with_angles)))
        
        # 检查前20%中是否有角度<90度且距离>30的点
        for theta, coord, idx in candidates_with_angles[:top_k]:
            dist = heuristic(parent_pos, coord)
            if theta < 90 and dist > 30:
                return False  # 不需要turning
        
        # 前方没有好的frontier，需要turning
        return True
    
    def _calculate_reaching_value(self, parent_pos, node_pos, parent_visited, 
                                   candidates, weights, temp_start, bad_frontier):
        """
        计算从parent到node的value
        
        虽然这一步不是greedy选择（而是UCB/随机选择），但它仍然有一个value
        这个value需要计入reward中
        
        Args:
            parent_pos: 父节点位置
            node_pos: 当前节点位置
            parent_visited: 父节点已访问的candidates索引集合
            candidates: 候选点列表
            weights: 权重字典
            temp_start: 临时起点（parent的temp_start）
            bad_frontier: 失败点集合
        
        Returns:

        """
        # 找到node_pos对应的candidate索引
        target_idx = None
        for i, cand in enumerate(candidates):
            if cand['coord'] == node_pos:
                target_idx = i
                break
        
        if target_idx is None:
            return 0.0
        
        # 判断是否需要turning（基于parent的状态）
        sim_turning, _ = self._check_turning_with_update(
            parent_pos, temp_start, candidates, parent_visited,
            w_angle=weights.get('angle', 1.0)
        )
        
        # 构建avoid点集合
        avoid_points = []
        if parent_visited:
            visited_coords = [candidates[v]['coord'] for v in parent_visited]
            avoid_points.extend(visited_coords)
        if bad_frontier:
            avoid_points.extend(list(bad_frontier))
        
        # 计算所有未访问候选点的距离和避让信息
        unvisited_data = []
        min_avoid_dist = float('inf')
        max_avoid_dist = 0.0
        min_dist = float('inf')
        max_dist = 0.0
        
        for i, cand in enumerate(candidates):
            if i in parent_visited:
                continue
            coord = cand['coord']
            d = heuristic(parent_pos, coord)
            
            if avoid_points:
                min_dist_to_avoid = min(heuristic(coord, ap) for ap in avoid_points)
            else:
                min_dist_to_avoid = float('inf')
            
            unvisited_data.append((i, coord, d, min_dist_to_avoid))
            min_dist = min(min_dist, d)
            max_dist = max(max_dist, d)
            if avoid_points and min_dist_to_avoid != float('inf'):
                min_avoid_dist = min(min_avoid_dist, min_dist_to_avoid)
                max_avoid_dist = max(max_avoid_dist, min_dist_to_avoid)
        
        if not unvisited_data:
            return 0.0
        
        # 计算mid_d和max_diff
        mid_d = 0.3 * min_dist + 0.7 * max_dist
        distances_to_mid = [abs(d - mid_d) for _, _, d, _ in unvisited_data]
        max_diff = max(distances_to_mid) if distances_to_mid else 0.0
        
        # 计算目标candidate的value
        target_value = 0.0
        details = None
        
        for i, coord, d, min_dist_to_avoid in unvisited_data:
            if i != target_idx:
                continue
            
            # angle_score
            if sim_turning:
                # TURNING MODE: angle_score固定为0.5
                angle_score = 0.5
                theta = 0.0
            else:
                theta = abs(self._calculate_heading(temp_start, parent_pos, coord))
                # 使用更陡峭的sigmoid函数（k=0.4）
                angle_score = 1.0 / (1.0 + math.exp(0.4 * (theta - 60.0)))
            
            # distance_score
            if max_diff != 0:
                diff_norm = abs(d - mid_d) / max_diff
            else:
                diff_norm = 0.0
            distance_score = 1 - diff_norm
            
            # explore_score
            explore_score = candidates[i]['explore_score']
            
            # avoid_score
            if avoid_points and min_dist_to_avoid != float('inf'):
                if (max_avoid_dist - min_avoid_dist) != 0:
                    avoid_score = (min_dist_to_avoid - min_avoid_dist) / (max_avoid_dist - min_avoid_dist)
                else:
                    avoid_score = 0.0
            else:
                avoid_score = 1.0
            
            # 总分
            if distance_score > 0 and avoid_score > 0 and explore_score > 0:
                target_value = (weights['angle'] * angle_score +
                               weights['dist'] * distance_score +
                               weights['explore'] * explore_score +
                               weights['avoid'] * avoid_score)
            else:
                target_value = 0.0
            
            # 构建详细信息
            break
        
        return target_value
    
    def _backpropagate(self, node, reward):
        """
        Backpropagation阶段: 回传reward更新路径上所有节点
        """
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    def _get_max_depth(self, node):
        """
        递归计算树的最大深度（以访问的candidates数量衡量）
        
        Args:
            node: 当前节点
        
        Returns:
            最大深度（访问的candidates数量）
        """
        if not node.children:
            # 叶子节点，返回其访问的candidates数量
            return len(node.visited)
        
        # 递归计算所有子节点的深度，取最大值
        max_child_depth = 0
        for child in node.children.values():
            child_depth = self._get_max_depth(child)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
    
    def _print_best_uct_path(self, node, candidates, depth=0, max_depth=5, temp_start=None, current_pos=None):
        """
        输出从当前节点开始的最优UCT路径（用于长程规划可视化）
        
        Args:
            node: 当前节点
            candidates: 候选点列表
            depth: 当前深度
            max_depth: 最大输出深度
            temp_start: 临时起点（用于计算首步角度）
            current_pos: 当前位置（用于计算首步角度）
        """
        if depth >= max_depth or not node.children:
            return
        
        # 找到UCT值最高的子节点
        best_ucb = -float('inf')
        best_child = None
        best_idx = None
        
        for idx, child in node.children.items():
            if child.visits == 0:
                continue  # 跳过未访问的节点
            
            # 计算UCT值
            exploitation = child.value / child.visits
            exploration = self.exploration_weight * math.sqrt(
                math.log(node.visits) / child.visits
            )
            uct = exploitation + exploration
            
            if uct > best_ucb:
                best_ucb = uct
                best_child = child
                best_idx = idx
        
        if best_child is None:
            return
        
        # 输出当前节点信息
        avg_value = best_child.value / max(1, best_child.visits)
        exploitation = best_child.value / best_child.visits
        exploration = self.exploration_weight * math.sqrt(
            math.log(node.visits) / best_child.visits
        )
        
        # 计算首步角度信息（仅对第一步显示）
        angle_info = ""
        if depth == 0 and temp_start is not None and current_pos is not None:
            coord = candidates[best_idx]['coord']
            if temp_start != current_pos:
                theta = abs(self._calculate_heading(temp_start, current_pos, coord))
                # 使用更陡峭的sigmoid函数（k=0.4）
                angle_score = 1.0 / (1.0 + math.exp(0.4 * (theta - 60.0)))
                angle_info = f" | first_step_angle={theta:.1f}° (score={angle_score:.3f})"
            else:
                angle_info = " | first_step_angle=0° (score=1.000)"
        
        indent = "  " * depth
        coord = candidates[best_idx]['coord']
        print(f"{indent}  Step {depth+1}: Candidate #{best_idx} {coord} | "
              f"avg_value={avg_value:.4f}, UCT={best_ucb:.4f} "
              f"(exploit={exploitation:.4f}, explore={exploration:.4f}) | "
              f"visits={best_child.visits}{angle_info}")
        
        # 递归输出下一层
        self._print_best_uct_path(best_child, candidates, depth + 1, max_depth, temp_start, current_pos)
    
    def _extract_best_uct_path(self, root, first_idx, candidates, max_visits):
        """
        提取从根节点开始的最优UCT路径（包括树内部分和greedy rollout）
        
        返回完整路径的候选点索引列表
        """
        path_indices = [first_idx]
        current_node = root.children.get(first_idx)
        
        if current_node is None:
            return path_indices
        
        # 沿着UCT路径追踪（树内部分）
        while current_node.children and len(path_indices) < max_visits:
            best_ucb = -float('inf')
            best_idx = None
            best_child = None
            
            for idx, child in current_node.children.items():
                if child.visits == 0:
                    continue
                
                exploitation = child.value / child.visits
                exploration = self.exploration_weight * math.sqrt(
                    math.log(current_node.visits) / child.visits
                )
                uct = exploitation + exploration
                
                if uct > best_ucb:
                    best_ucb = uct
                    best_idx = idx
                    best_child = child
            
            if best_child is None:
                break
            
            path_indices.append(best_idx)
            current_node = best_child
        
        return path_indices
    
    def _evaluate_path_reward(self, path_indices, candidates, occupancy_map, current_pos,
                              current_to_candidates_dist, max_candidate_distance,
                              unexplored_points, candidate_unexplored_points,
                              num_bad_frontier, bad_frontier, max_visits_in_simulation, weights, 
                              temp_start=None, turning_mode=False, return_reward=False):
        """
        评估给定路径的reward（强制输出详细调试信息）
        
        注意：这个函数是用于debug的，它应该模拟真实的reward计算，
        但是它无法准确跟踪每个节点的temp_start更新（需要遍历树）。
        因此，这里显示的reward可能与实际simulation的reward不同。
        
        Args:
            turning_mode: 整个搜索是否处于turning around模式
            return_reward: 是否返回reward值（用于对比）
        
        Returns:
            float: 如果return_reward=True，返回total_reward，否则返回None
        """
        print(f"  Path (tree exploration): {path_indices}")
        print(f"  ⚠️  Note: This is a debug evaluation that may differ from actual simulation rewards")
        print(f"      because it uses root's temp_start instead of each node's updated temp_start.")
        
        # Selection部分就是path_indices（树内搜索得到的）
        selection_indices = path_indices.copy()
        
        # 如果路径还没达到max_visits，继续greedy rollout
        rollout_indices = []
        if len(path_indices) < max_visits_in_simulation:
            print(f"  Continuing with greedy rollout from path end...")
            visited = set(path_indices)
            current_rollout_pos = candidates[path_indices[-1]]['coord']
            
            # 简化版greedy rollout（不需要完整的turning逻辑）
            while len(visited) < max_visits_in_simulation:
                best_score = -float('inf')
                best_idx = None
                
                for i, cand in enumerate(candidates):
                    if i in visited:
                        continue
                    
                    coord = cand['coord']
                    dist = heuristic(current_rollout_pos, coord)
                    explore_points = len(candidate_unexplored_points[i])
                    
                    # 简单评分：探索点数 - 距离惩罚
                    score = explore_points - dist * 0.1
                    
                    if score > best_score:
                        best_score = score
                        best_idx = i
                
                if best_idx is None:
                    break
                
                visited.add(best_idx)
                rollout_indices.append(best_idx)
                current_rollout_pos = candidates[best_idx]['coord']
            
            if rollout_indices:
                print(f"  Rollout extended: {rollout_indices}")
            else:
                print(f"  Rollout: no extension (reached max visits)")
        
        # 调用_calculate_reward（强制输出详细信息）
        # 注意：这里使用root的temp_start，可能与实际不同
        total_reward = self._calculate_reward(
            selection_indices, rollout_indices,
            candidates, current_to_candidates_dist, max_candidate_distance,
            unexplored_points, candidate_unexplored_points,
            num_bad_frontier, bad_frontier, max_visits_in_simulation,
            is_first_iteration=True,  # 强制输出详细信息
            temp_start=temp_start, current_pos=current_pos, turning_mode=turning_mode,
            weights=weights  # 传入权重参数
        )
        
        print(f"  Total reward for this path (debug evaluation): {total_reward:.4f}\n")
        
        # 如果需要返回reward值（用于对比）
        if return_reward:
            return total_reward
        return None
    
    def _calculate_heading(self, temp_start, now, goal):
        """
        计算从temp_start到now的运动方向，与从now到goal的方向之间的夹角
        
        这个函数与landmark.py中的calculate_heading完全一致
        
        Args:
            temp_start: 临时起点（用于确定当前运动方向）
            now: 当前位置
            goal: 目标位置
        
        Returns:
            角度（度数，0-180）
        """
        # 特殊情况1: 如果now == goal，说明已经到达目标，角度为0
        if now == goal or (now[0] == goal[0] and now[1] == goal[1]):
            return 0.0
        
        # 特殊情况2: 如果temp_start == now，说明刚开始或刚turning，角度为0
        if temp_start == now or (temp_start[0] == now[0] and temp_start[1] == now[1]):
            return 0.0
        
        # 向量1: 从temp_start到now（当前运动方向）
        vector_start_now = np.array((now[0] - temp_start[0], now[1] - temp_start[1]))
        # 向量2: 从now到goal（目标方向）
        vector_now_goal = np.array((goal[0] - now[0], goal[1] - now[1]))
        
        # 计算向量的模长
        norm_start_now = np.linalg.norm(vector_start_now)
        norm_now_goal = np.linalg.norm(vector_now_goal)
        
        # 防止除以0
        if norm_start_now < 1e-6 or norm_now_goal < 1e-6:
            return 0.0
        
        # 计算两个向量的夹角
        cos_angle = np.dot(vector_start_now, vector_now_goal) / (norm_start_now * norm_now_goal)
        # 限制cos_angle在[-1, 1]范围内，避免数值误差
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        
        return angle_deg
    
    def _debug_evaluate_best_path(self, root, best_child, candidates, current_to_candidates_dist, 
                                  max_candidate_distance, unexplored_points, candidate_unexplored_points,
                                  num_bad_frontier, bad_frontier, max_visits_in_simulation,
                                  temp_start, current_pos, turning_mode, weights):
        """
        调试函数：评估最优UCT路径的reward计算过程
        
        从root到best_child的路径，模拟完整的reward计算
        
        Args:
            temp_start: 真实的initial temp_start（等于root.temp_start）
            current_pos: 真实的agent当前位置（等于root.position）
            turning_mode: 整个搜索是否处于turning around模式
        
        Note:
            使用真实的initial状态来计算first_step_angle_reward，确保与实际MCTS计算一致
        """
        # 构建从root到best_child的完整路径
        path_indices = []
        current_node = best_child
        while current_node != root:
            # 找到当前节点在父节点中的索引
            for idx, child in current_node.parent.children.items():
                if child == current_node:
                    path_indices.append(idx)
                    break
            current_node = current_node.parent
        
        # 路径是从root到best_child，需要反转
        path_indices.reverse()
        
        print(f"  Path (selection from root to best child): {path_indices}")
        
        # 模拟rollout：从best_child继续greedy选择
        rollout_indices = []
        visited = set(best_child.visited)
        current_pos_sim = best_child.position
        
        # 使用与_simulate相同的heuristic进行rollout
        for _ in range(max_visits_in_simulation - len(visited)):
            # 计算所有未访问候选点的评分
            best_score = -float('inf')
            best_idx = None
            
            for i, cand in enumerate(candidates):
                if i in visited:
                    continue
                
                coord = cand['coord']
                d_sim = heuristic(current_pos_sim, coord)
                
                # 简化的heuristic评分（与_simulate一致，使用weights）
                explore_score = cand['explore_score']
                distance_score = 1.0  # 简化
                avoid_score = 1.0     # 简化
                
                value_score = (weights['dist'] * distance_score +
                              weights['explore'] * explore_score + 
                              weights['avoid'] * avoid_score)
                
                if value_score > best_score:
                    best_score = value_score
                    best_idx = i
            
            if best_idx is None:
                break
                
            rollout_indices.append(best_idx)
            visited.add(best_idx)
            current_pos_sim = candidates[best_idx]['coord']
        
        if rollout_indices:
            print(f"  Rollout (greedy simulation from best child): {rollout_indices}")
        else:
            print(f"  Rollout: (none - already at max depth)")
        
        # 计算完整路径的reward（启用详细调试输出）
        all_visited_indices = path_indices + rollout_indices
        print(f"\n{'='*80}")
        print(f"[MCTS] 🧮 Reward Calculation for Best Path")
        print(f"{'='*80}\n")
        
        # 调用_calculate_reward进行完整计算，设置is_first_iteration=True以显示详细信息
        reward = self._calculate_reward(
            path_indices, rollout_indices,
            candidates, current_to_candidates_dist, max_candidate_distance,
            unexplored_points, candidate_unexplored_points,
            num_bad_frontier, bad_frontier, max_visits_in_simulation,
            is_first_iteration=True,  # 开启详细调试输出
            temp_start=temp_start,
            current_pos=current_pos,
            turning_mode=turning_mode,
            weights=weights  # 传入权重参数
        )
        
        print(f"\n[MCTS] 🎯 Best path total reward: {reward:.4f}")
        print()
    

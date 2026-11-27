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
                            iterations, weights, temp_start, bad_frontier, debug_mode=False):
        """
        使用MCTS选择最优frontier
        
        Args:
            occupancy_map: 占用地图
            current_pos: 当前位置
            candidates: 候选点列表（已预处理过的字典列表）
            iterations: MCTS迭代次数
            weights: 评分权重字典 {'angle', 'dist', 'explore', 'avoid'}
            temp_start: 临时起点（用于动态判断是否需要turning around）
            bad_frontier: 已知的失败点集合（用于计算avoid_score）
            debug_mode: 是否输出详细的debug信息
        
        Returns:
            最优frontier坐标，或None
        
        Note:
            simulation中的所有评分（angle, distance, avoid）都基于当前simulation状态动态计算，
            只有explore_score使用预计算值（因为地图不变）
        """
        if not candidates:
            return None
        
        # 初始化根节点（root的temp_start就是传入的temp_start）
        root = MCTSNode([], current_pos, parent=None, temp_start=temp_start)
        root.set_unvisited(len(candidates))
        
        # 计算simulation的停止条件（访问40%的candidates）
        max_visits_in_simulation = max(1, int(len(candidates) * 0.4))
        
        print(f"[MCTS] Simulation will visit {max_visits_in_simulation} out of {len(candidates)} candidates (40%)")
        
        # MCTS主循环
        all_rewards = []  # 收集所有simulation的reward用于统计
        
        for iteration in range(iterations):
            # 1. Selection: 从root开始选择到叶节点
            # Debug模式下只在前10次迭代输出Selection详情
            node, selection_path = self._select(root, candidates, debug=(debug_mode and iteration < 10))
            
            # 2. Expansion: 标准MCTS逻辑
            # - Root节点：第一次访问就expansion
            # - 其他节点：只有visits > 0时才expansion
            # - visits == 0的非root节点：直接simulation
            # - 终止节点：len(visited) >= max_visits_in_simulation，不expansion
            if node.unvisited and len(node.visited) < max_visits_in_simulation:
                # Root节点（parent is None）或者visits > 0的节点才expansion
                if node.parent is None or node.visits > 0:
                    # 扩展所有未访问的子节点，并为每个子节点计算temp_start
                    self._expand(node, candidates, weights)
                    
                    # 从新扩展的子节点中随机选择一个进行simulation
                    if node.children:
                        # 选择刚扩展的子节点中的一个（它们的visits都是0）
                        newly_expanded = [idx for idx in node.children.keys() 
                                         if node.children[idx].visits == 0]
                        if newly_expanded:
                            selected_idx = random.choice(newly_expanded)
                            node = node.children[selected_idx]
            
            # 3. Simulation: 从当前节点rollout到40%的candidates
            # Debug模式下只在前10次迭代输出Simulation详情
            reward = self._simulate(
                node, candidates, occupancy_map, weights, temp_start, bad_frontier, max_visits_in_simulation,
                debug_mode=debug_mode,
                debug_iteration=(iteration + 1) if debug_mode else None,  # 只在debug模式下传入迭代号
                selection_path=selection_path  # 传入selection路径
            )
            all_rewards.append(reward)
            
            # 4. Backpropagation: 更新路径上所有节点
            self._backpropagate(node, reward)
        
        # 输出reward统计信息
        if all_rewards:
            print(f"[MCTS] Reward stats: min={min(all_rewards):.3f}, max={max(all_rewards):.3f}, "
                  f"avg={np.mean(all_rewards):.3f}, std={np.std(all_rewards):.3f}")
            spread_pct = (max(all_rewards) - min(all_rewards)) / np.mean(all_rewards) * 100 if np.mean(all_rewards) > 0 else 0
            print(f"[MCTS] Reward spread: {max(all_rewards) - min(all_rewards):.3f} ({spread_pct:.1f}%)")
        
        # 计算并输出树的最大深度
        max_tree_depth = self._get_max_depth(root)
        print(f"[MCTS] Tree max depth: {max_tree_depth} (rollout limit: {max_visits_in_simulation}/{len(candidates)})")
        
        # 选择最优子节点（基于avg_value，即exploitation）
        if not root.children:
            return None
        
        # 计算所有子节点的UCB值和avg_value
        child_stats = []
        for idx, child in root.children.items():
            avg_value = child.value / max(1, child.visits)
            # 计算UCB值（用于调试输出）
            if child.visits > 0:
                exploitation = child.value / child.visits
                exploration = self.exploration_weight * math.sqrt(
                    math.log(root.visits) / child.visits
                )
                ucb = exploitation + exploration
            else:
                exploitation = 0
                exploration = 0
                ucb = float('inf')  # 未访问的节点UCB为无穷大
            
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
        
        # 按avg_value降序排序（最终选择策略）
        child_stats.sort(key=lambda x: x['avg_value'], reverse=True)
        
        # 输出前5名（或全部如果少于5个）
        top_k = min(5, len(child_stats))
        print(f"[MCTS] Top {top_k} candidates by avg_value:")
        for i in range(top_k):
            stat = child_stats[i]
            print(f"  #{i+1}: idx={stat['idx']}, coord={stat['coord']}, "
                  f"avg_value={stat['avg_value']:.4f}, UCB={stat['ucb']:.4f} "
                  f"(exploit={stat['exploitation']:.4f}, explore={stat['exploration']:.4f}), "
                  f"visits={stat['visits']}")
        
        # 输出visits分布统计
        all_visits = [stat['visits'] for stat in child_stats]
        all_values = [stat['avg_value'] for stat in child_stats]
        print(f"[MCTS] Visits distribution: min={min(all_visits)}, max={max(all_visits)}, avg={sum(all_visits)/len(all_visits):.1f}")
        print(f"[MCTS] Avg_value range: [{min(all_values):.4f}, {max(all_values):.4f}], spread={max(all_values)-min(all_values):.4f}")
        
        # 返回avg_value最高的（Max Child策略）
        best_candidate = child_stats[0]
        best_coord = best_candidate['coord']
        
        # 调试输出：输出最优UCT路径（长程规划结果）
        print(f"\n[MCTS Long-term Planning] Best UCT path from root:")
        self._print_best_path(root, candidates, depth=0, max_depth=5)
        print()
        
        return best_coord
    
    def _select(self, node, candidates, debug=False):
        """
        Selection阶段: 使用UCT公式选择节点
        
        严格按照UCT公式，一层层选择UCT值最大的子节点，直到叶子节点
        
        UCT = exploitation + exploration
        exploitation = node.value / node.visits
        exploration = c * sqrt(log(parent.visits) / node.visits)
        """
        selection_path = []  # 记录selection路径
        
        if debug:
            print(f"\n{'='*80}")
            print(f"[MCTS Selection Debug] Selection path from root:")
            print(f"{'='*80}")
        
        # 添加root节点到路径
        selection_path.append({
            'node': node,
            'cand_idx': None,
            'position': node.position,
            'temp_start': node.temp_start,
            'visited': list(node.visited),
            'visits': node.visits,
            'avg_value': node.value / max(1, node.visits),
            'children_count': len(node.children)
        })
        
        if debug:
            print(f"\nDepth 0 (ROOT): pos={node.position}, temp_start={node.temp_start}")
            print(f"  visits={node.visits}, avg_value={node.value / max(1, node.visits):.4f}")
            print(f"  visited={sorted(list(node.visited))}, children={len(node.children)}")
        
        depth = 0
        while node.children and not node.unvisited:
            depth += 1
            best_score = -float('inf')
            best_child = None
            best_child_idx = None
            
            # 收集所有子节点的UCB信息用于debug
            children_ucb_info = []
            
            for idx, child in node.children.items():
                # UCT公式：exploit + explore
                # 当visits=0时，exploration项会非常大，自然优先选择
                exploit = child.value / (child.visits + 1e-6)
                explore = self.exploration_weight * math.sqrt(
                    math.log(node.visits + 1) / (child.visits + 1e-6)
                )
                uct_score = exploit + explore
                
                if debug:
                    children_ucb_info.append({
                        'idx': idx,
                        'coord': child.position,
                        'temp_start': child.temp_start,
                        'ucb': uct_score,
                        'exploit': exploit,
                        'explore': explore,
                        'visits': child.visits,
                        'avg_value': child.value / max(1, child.visits),
                        'visited': list(child.visited)
                    })
                
                if uct_score > best_score:
                    best_score = uct_score
                    best_child = child
                    best_child_idx = idx
            
            if debug and children_ucb_info:
                # 按UCB排序
                children_ucb_info.sort(key=lambda x: x['ucb'], reverse=True)
                print(f"\nDepth {depth} - Evaluating {len(children_ucb_info)} children:")
                for rank, info in enumerate(children_ucb_info[:5]):  # 只显示前5个
                    marker = "★ " if info['idx'] == best_child_idx else "  "
                    print(f"  {marker}#{rank+1} Cand#{info['idx']} {info['coord']}, temp_start={info['temp_start']}")
                    print(f"      UCB={info['ucb']:.4f} (exploit={info['exploit']:.4f} + explore={info['explore']:.4f})")
                    print(f"      visits={info['visits']}, avg_value={info['avg_value']:.4f}, visited={sorted(info['visited'])}")
                if len(children_ucb_info) > 5:
                    print(f"  ... and {len(children_ucb_info) - 5} more children")
                print(f"  => Selected: Cand#{best_child_idx} with UCB={best_score:.4f}")
            
            node = best_child if best_child else node
            
            # 添加选中的节点到路径
            if best_child:
                selection_path.append({
                    'node': node,
                    'cand_idx': best_child_idx,
                    'position': node.position,
                    'temp_start': node.temp_start,
                    'visited': list(node.visited),
                    'visits': node.visits,
                    'avg_value': node.value / max(1, node.visits),
                    'children_count': len(node.children)
                })
        
        if debug:
            print(f"\n{'='*80}")
            print(f"Selection completed at depth {depth}")
            print(f"Final node: pos={node.position}, temp_start={node.temp_start}")
            print(f"  visits={node.visits}, len(visited)={len(node.visited)}, children={len(node.children)}")
            print(f"{'='*80}\n")
        
        return node, selection_path
    
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
                  max_visits_in_simulation, debug_mode=False, debug_iteration=None, selection_path=None):
        """
        Simulation阶段: 使用greedy策略rollout到40%的candidates
        
        从当前节点开始，贪心地选择未访问的候选点，每选择一个就获得其value_score作为奖励。
        累积value_score作为总reward，访问达到40%的candidates时停止。
        
        关键（完全独立于真实探索状态）：
        1. 在每一步到达新位置后，都要动态判断是否需要turning around
        2. 基于simulation的current_pos动态计算所有候选点的距离，并动态计算mid_d和max_diff
        3. 动态计算avoid_score的归一化范围（基于simulation的visited + bad_frontier）
        4. 只有explore_score使用预计算值（因为地图不变）
        
        Note:
            simulation与真实探索状态完全独立，所有评分都基于simulation的当前状态动态计算
        """
        visited = set(node.visited)
        current_pos = node.position
        total_reward = 0.0  # 累积的value_score
        # 使用node的temp_start作为simulation的初始temp_start
        sim_temp_start = node.temp_start
        
        # Debug输出控制：debug模式下前10次迭代输出详细rollout信息
        show_debug = (debug_mode and debug_iteration is not None and debug_iteration <= 10)
        
        if show_debug:
            # 找到当前node在parent中的索引
            node_cand_idx = None
            if node.parent:
                for idx, child in node.parent.children.items():
                    if child == node:
                        node_cand_idx = idx
                        break
            
            print(f"\n{'='*80}")
            print(f"[MCTS Rollout Debug] Iteration {debug_iteration} - Starting rollout:")
            print(f"{'='*80}")
            print(f"\nSelection Path Summary:")
            print(f"  Root → ... → Current Node (depth={len(selection_path) - 1 if selection_path else 0})")
            if selection_path:
                for i, step in enumerate(selection_path):
                    if i == 0:
                        print(f"  Depth {i} (ROOT): pos={step['position']}, temp_start={step['temp_start']}")
                    else:
                        print(f"  Depth {i}: Cand#{step['cand_idx']} → pos={step['position']}, temp_start={step['temp_start']}")
                    print(f"           visits={step['visits']}, avg_value={step['avg_value']:.4f}, visited={sorted(step['visited'])}")
            
            print(f"\nCurrent Node Info:")
            print(f"  Position: {node.position}")
            print(f"  temp_start: {sim_temp_start}")
            print(f"  Already visited (by this node): {sorted(list(node.visited))}")
            if node.parent:
                print(f"  Parent: {node.parent.position}")
            print(f"  Target visits: {max_visits_in_simulation}/{len(candidates)} (40%)")
            print(f"  Simulation uses GREEDY heuristic to select next frontiers")
        
        # 步骤0：如果node不是root，计算从parent到node的value
        # 虽然这一步不是greedy选择（而是UCB选择），但它仍然有一个value需要计入reward
        # 注意：这里使用parent的temp_start来计算angle_score
        if node.parent is not None and len(node.visited) > len(node.parent.visited):
            # 计算"到达node"这一步的value（使用parent的temp_start）
            if show_debug:
                reaching_value, reaching_details = self._calculate_reaching_value(
                    node.parent.position, current_pos, node.parent.visited,
                    candidates, weights, node.parent.temp_start, bad_frontier,
                    return_details=True
                )
            else:
                reaching_value = self._calculate_reaching_value(
                    node.parent.position, current_pos, node.parent.visited,
                    candidates, weights, node.parent.temp_start, bad_frontier,
                    return_details=False
                )
                reaching_details = None
            
            total_reward += reaching_value
            
            if show_debug:
                print(f"\n  [Step 0] Reaching current node from parent (UCB-selected step):")
                print(f"    From parent_pos: {node.parent.position} to current_pos: {current_pos}")
                print(f"    Using parent's temp_start: {node.parent.temp_start}")
                if reaching_details:
                    print(f"    Turning judgment: {'TURNING AROUND' if reaching_details['turning'] else 'No turning needed'}")
                    print(f"    Heuristic breakdown:")
                    print(f"      angle_score  = {reaching_details['angle']:.3f} (θ={reaching_details['theta']:.1f}°)")
                    print(f"      dist_score   = {reaching_details['dist']:.3f} (d={reaching_details['d']:.1f})")
                    print(f"      explore_score= {reaching_details['explore']:.3f}")
                    print(f"      avoid_score  = {reaching_details['avoid']:.3f}")
                    print(f"    Reaching value: {reaching_value:.4f}")
        
        # Rollout: 从node继续，使用greedy策略选择直到访问40%的candidates
        step_num = 0
        
        # 即使到达limit，也显示当前状态下的candidates评分（用于debug）
        if show_debug and len(visited) >= max_visits_in_simulation:
            print(f"\n  [Rollout Status] Node has reached rollout limit:")
            print(f"    len(visited)={len(visited)} >= max_visits_in_simulation={max_visits_in_simulation}")
            print(f"    No greedy steps will be taken in this rollout.")
            print(f"\n  [Current State Analysis] If we were to continue, what would be the best choice?")
            
            # 计算当前状态下的最佳选择（即使不执行）
            sim_turning, _ = self._check_turning_with_update(
                current_pos, sim_temp_start, candidates, visited, w_angle=weights.get('angle', 1.0)
            )
            
            avoid_points = []
            visited_coords = [candidates[v]['coord'] for v in visited]
            avoid_points.extend(visited_coords)
            if bad_frontier:
                avoid_points.extend(list(bad_frontier))
            
            unvisited_candidates_data = []
            min_avoid_dist = float('inf')
            max_avoid_dist = 0.0
            min_dist = float('inf')
            max_dist = 0.0
            
            for i, cand in enumerate(candidates):
                if i in visited:
                    continue
                
                coord = cand['coord']
                d_sim = heuristic(current_pos, coord)
                
                if avoid_points:
                    min_dist_to_avoid = min(heuristic(coord, ap) for ap in avoid_points)
                else:
                    min_dist_to_avoid = float('inf')
                
                unvisited_candidates_data.append((i, coord, d_sim, min_dist_to_avoid))
                min_dist = min(min_dist, d_sim)
                max_dist = max(max_dist, d_sim)
                
                if avoid_points and min_dist_to_avoid != float('inf'):
                    min_avoid_dist = min(min_avoid_dist, min_dist_to_avoid)
                    max_avoid_dist = max(max_avoid_dist, min_dist_to_avoid)
            
            mid_d_sim = 0.3 * min_dist + 0.7 * max_dist
            distances_to_mid = [abs(d - mid_d_sim) for _, _, d, _ in unvisited_candidates_data]
            max_diff_sim = max(distances_to_mid) if distances_to_mid else 0.0
            
            print(f"    Unvisited candidates: {len(unvisited_candidates_data)}")
            print(f"    Avoid points: visited={len(visited_coords)}, bad_frontier={len(bad_frontier) if bad_frontier else 0}")
            print(f"    Distance range: [{min_dist:.1f}, {max_dist:.1f}], mid_d={mid_d_sim:.1f}, max_diff={max_diff_sim:.1f}")
            if avoid_points and min_avoid_dist != float('inf'):
                print(f"    Avoid distance range: [{min_avoid_dist:.1f}, {max_avoid_dist:.1f}]")
            print(f"    Turning around: {sim_turning}")
            
            all_scores = []
            for i, coord, d_sim, min_dist_to_avoid in unvisited_candidates_data:
                cand = candidates[i]
                
                if avoid_points and min_dist_to_avoid != float('inf'):
                    if (max_avoid_dist - min_avoid_dist) != 0:
                        avoid_score = (min_dist_to_avoid - min_avoid_dist) / (max_avoid_dist - min_avoid_dist)
                    else:
                        avoid_score = 0.0
                else:
                    avoid_score = 1.0
                
                if sim_turning:
                    angle_score = -1.0
                    theta = 0.0
                else:
                    theta = abs(self._calculate_heading(sim_temp_start, current_pos, coord))
                    angle_score = 1.0 / (1.0 + math.exp(0.2 * (theta - 60.0)))
                
                if max_diff_sim != 0:
                    diff_norm = abs(d_sim - mid_d_sim) / max_diff_sim
                else:
                    diff_norm = 0.0
                distance_score = 1 - diff_norm
                
                explore_score = cand['explore_score']
                
                if distance_score > 0 and avoid_score > 0 and explore_score > 0:
                    value_score = (weights['angle'] * angle_score + 
                                  weights['dist'] * distance_score +
                                  weights['explore'] * explore_score + 
                                  weights['avoid'] * avoid_score)
                else:
                    value_score = 0.0
                
                all_scores.append({
                    'idx': i,
                    'coord': coord,
                    'value': value_score,
                    'angle': angle_score,
                    'dist': distance_score,
                    'explore': explore_score,
                    'avoid': avoid_score,
                    'theta': theta,
                    'd': d_sim
                })
            
            all_scores.sort(key=lambda x: x['value'], reverse=True)
            print(f"\n    Hypothetical next step (if rollout continued):")
            for rank, score_info in enumerate(all_scores[:5]):
                marker = "★ " if rank == 0 else "  "
                print(f"      {marker}#{rank+1} Cand#{score_info['idx']} {score_info['coord']}: "
                      f"value={score_info['value']:.4f} "
                      f"(angle={score_info['angle']:.3f}, dist={score_info['dist']:.3f}, "
                      f"explore={score_info['explore']:.3f}, avoid={score_info['avoid']:.3f}) "
                      f"[θ={score_info['theta']:.1f}°, d={score_info['d']:.1f}]")
        
        while len(visited) < max_visits_in_simulation:
            step_num += 1
            
            if show_debug:
                print(f"\n  [Step {step_num}] Current pos: {current_pos}, sim_temp_start: {sim_temp_start}")
            
            # 每一步都重新判断是否需要turning around
            # 如果需要turning，立即更新sim_temp_start（在计算angle_score之前）
            if show_debug:
                sim_turning, new_temp_start, reason = self._check_turning_with_update(
                    current_pos, sim_temp_start, candidates, visited, 
                    w_angle=weights.get('angle', 1.0), return_reason=True
                )
            else:
                sim_turning, new_temp_start = self._check_turning_with_update(
                    current_pos, sim_temp_start, candidates, visited, 
                    w_angle=weights.get('angle', 1.0), return_reason=False
                )
                reason = None
            
            if show_debug:
                print(f"    Turning judgment:")
                if reason['type'] == 'w_angle_zero':
                    print(f"      Reason: w_angle=0.0 → No turning needed")
                elif reason['type'] == 'pos_eq_temp_start':
                    print(f"      Reason: current_pos == sim_temp_start → No turning needed")
                elif reason['type'] == 'no_unvisited':
                    print(f"      Reason: No unvisited candidates → No turning needed")
                elif reason['type'] == 'found_good_forward':
                    print(f"      Checking top {reason['top_k']} candidates (20% of {reason['total_unvisited']} unvisited)")
                    print(f"      Found {len(reason['good_candidates'])} good forward option(s) (θ<90° and dist>30):")
                    for theta, coord, idx, dist in reason['good_candidates']:
                        print(f"        Cand#{idx} {coord}: θ={theta:.1f}°, dist={dist:.1f}px ✓")
                    print(f"      Result: No turning needed")
                elif reason['type'] == 'need_turning':
                    print(f"      Checking top {reason['top_k']} candidates (20% of {reason['total_unvisited']} unvisited)")
                    print(f"      Top {reason['top_k']} forward candidates (by angle):")
                    for cand_info in reason['top_candidates']:
                        print(f"        Cand#{cand_info['idx']} {cand_info['coord']}: θ={cand_info['theta']:.1f}°, dist={cand_info['dist']:.1f}px → {cand_info['reason']}")
                    print(f"      Result: NEED TURNING AROUND (no good forward options)")
            
            # 如果需要更新temp_start，立即更新
            if new_temp_start is not None:
                if show_debug:
                    print(f"      Updated sim_temp_start: {sim_temp_start} → {new_temp_start}")
                sim_temp_start = new_temp_start
            
            # 第一遍：计算所有未访问候选点的基础数据
            # 构建simulation中的避免点集合
            avoid_points = []
            visited_coords = [candidates[v]['coord'] for v in visited]
            avoid_points.extend(visited_coords)
            if bad_frontier:
                avoid_points.extend(list(bad_frontier))
            
            if show_debug:
                print(f"    Current state: {'TURNING MODE (all angles get score=-1.0)' if sim_turning else 'NORMAL MODE (angles matter)'}")
                print(f"    Unvisited candidates: {len(candidates) - len(visited)}")
                print(f"    Avoid points: visited={len(visited_coords)}, bad_frontier={len(bad_frontier) if bad_frontier else 0}")
            
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
            
            if show_debug:
                print(f"    Distance range: [{min_dist:.1f}, {max_dist:.1f}], mid_d={mid_d_sim:.1f}, max_diff={max_diff_sim:.1f}")
                if avoid_points and min_avoid_dist != float('inf'):
                    print(f"    Avoid distance range: [{min_avoid_dist:.1f}, {max_avoid_dist:.1f}]")
            
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
                    angle_score = -1.0
                else:
                    theta = abs(self._calculate_heading(sim_temp_start, current_pos, coord))
                    angle_score = 1.0 / (1.0 + math.exp(0.2 * (theta - 60.0)))
                
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
                    value_score = (weights['angle'] * angle_score + 
                                  weights['dist'] * distance_score +
                                  weights['explore'] * explore_score + 
                                  weights['avoid'] * avoid_score)
                else:
                    value_score = 0.0  # 任何一项为0，总分为0
                
                if show_debug:
                    # 存储所有候选点的评分信息
                    theta = abs(self._calculate_heading(sim_temp_start, current_pos, coord)) if not sim_turning else 0.0
                    all_scores.append({
                        'idx': i,
                        'coord': coord,
                        'value': value_score,
                        'angle': angle_score,
                        'dist': distance_score,
                        'explore': explore_score,
                        'avoid': avoid_score,
                        'theta': theta,
                        'd': d_sim
                    })
                
                if value_score > best_score:
                    best_score = value_score
                    best_idx = i
                    best_coord = coord
            
            if show_debug:
                # 按value排序并输出
                all_scores.sort(key=lambda x: x['value'], reverse=True)
                print(f"    All candidates scores (sorted by value, showing all {len(all_scores)} candidates):")
                for rank, score_info in enumerate(all_scores):
                    marker = "★ " if score_info['idx'] == best_idx else "  "
                    print(f"      {marker}#{rank+1} Cand#{score_info['idx']} {score_info['coord']}: "
                          f"value={score_info['value']:.4f} "
                          f"(angle={score_info['angle']:.3f}, dist={score_info['dist']:.3f}, "
                          f"explore={score_info['explore']:.3f}, avoid={score_info['avoid']:.3f}) "
                          f"[θ={score_info['theta']:.1f}°, d={score_info['d']:.1f}]")
            
            if best_idx is None:
                if show_debug:
                    print(f"    => No valid candidate found, stopping rollout")
                break
            
            if show_debug:
                print(f"\n    => Selected: Candidate #{best_idx}, coord={best_coord}, value_score={best_score:.4f}")
            
            # 累加该步的value_score作为reward
            total_reward += best_score
            visited.add(best_idx)
            current_pos = best_coord
        
        # Reward标准化：按访问的节点数量平均
        # 这样可以使exploitation的范围更加稳定，避免因candidates数量不同导致的波动
        # num_visited包括：Step 0（从parent到node，如果有）+ greedy rollout步数
        if node.parent is not None:
            # 从parent开始计数（包括到达node这一步）
            num_visited = len(visited) - len(node.parent.visited)
        else:
            # root节点，只计算greedy rollout
            num_visited = len(visited) - len(node.visited)
        
        reward = total_reward / max(1, num_visited)
        
        if show_debug:
            print(f"\n[MCTS Rollout Debug] Rollout completed:")
            print(f"  Started from node with {len(node.visited)} already visited")
            print(f"  Total visited after rollout: {len(visited)}")
            print(f"  Greedy steps taken in simulation: {step_num}")
            print(f"  Total reward (sum of all steps): {total_reward:.4f}")
            print(f"  Average reward per step: {reward:.4f}")
            print(f"  Final position: {current_pos}")
            print(f"{'='*80}\n")
        
        return reward
    
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
    
    def _check_turning_with_update(self, current_pos, temp_start, candidates, visited, w_angle=1.0, return_reason=False):
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
            return_reason: 如果为True，返回详细的判断原因
        
        Returns:
            如果return_reason=False: (turning, new_temp_start)
            如果return_reason=True: (turning, new_temp_start, reason_dict)
        """
        # 特殊情况：如果w_angle == 0.0，不需要turning
        if w_angle == 0.0:
            if return_reason:
                return False, None, {'type': 'w_angle_zero'}
            return False, None
        
        # 如果当前位置等于temp_start，不需要turning
        if current_pos == temp_start:
            if return_reason:
                return False, None, {'type': 'pos_eq_temp_start'}
            return False, None
        
        # 获取未访问的候选点
        unvisited_candidates = [(i, cand) for i, cand in enumerate(candidates) if i not in visited]
        
        if not unvisited_candidates:
            if return_reason:
                return False, None, {'type': 'no_unvisited'}
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
            if return_reason:
                return False, None, {
                    'type': 'found_good_forward',
                    'good_candidates': good_candidates,
                    'top_k': top_k,
                    'total_unvisited': len(unvisited_candidates)
                }
            return False, None
        
        # 前方没有好的frontier，需要turning
        # 使用_compute_updated_temp_start计算新的temp_start（从temp_start到current_pos方向延伸）
        new_temp_start = self._compute_updated_temp_start(temp_start, current_pos, step_size=10.0)
        
        if return_reason:
            # 记录前20%的所有候选点信息，说明为什么它们不满足条件
            top_candidates_info = []
            for theta, coord, idx in candidates_with_angles[:top_k]:
                dist = heuristic(current_pos, coord)
                top_candidates_info.append({
                    'idx': idx,
                    'coord': coord,
                    'theta': theta,
                    'dist': dist,
                    'reason': 'θ>=90' if theta >= 90 else 'dist<=30'
                })
            return True, new_temp_start, {
                'type': 'need_turning',
                'top_candidates': top_candidates_info,
                'top_k': top_k,
                'total_unvisited': len(unvisited_candidates)
            }
        
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
                                   candidates, weights, temp_start, bad_frontier,
                                   return_details=False):
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
            return_details: 是否返回详细的heuristic分解
        
        Returns:
            如果return_details=False: 返回value
            如果return_details=True: 返回(value, details_dict)
        """
        # 找到node_pos对应的candidate索引
        target_idx = None
        for i, cand in enumerate(candidates):
            if cand['coord'] == node_pos:
                target_idx = i
                break
        
        if target_idx is None:
            if return_details:
                return 0.0, None
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
            if return_details:
                return 0.0, None
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
                angle_score = -1.0
                theta = 0.0
            else:
                theta = abs(self._calculate_heading(temp_start, parent_pos, coord))
                angle_score = 1.0 / (1.0 + math.exp(0.2 * (theta - 60.0)))
            
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
            if return_details:
                details = {
                    'angle': angle_score,
                    'dist': distance_score,
                    'explore': explore_score,
                    'avoid': avoid_score,
                    'theta': theta,
                    'd': d,
                    'turning': sim_turning
                }
            
            break
        
        if return_details:
            return target_value, details
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
        递归计算树的最大深度
        
        Args:
            node: 当前节点
        
        Returns:
            最大深度（root的深度为0）
        """
        if not node.children:
            return 0  # 叶子节点，深度为0
        
        # 递归计算所有子节点的深度，取最大值
        max_child_depth = 0
        for child in node.children.values():
            child_depth = self._get_max_depth(child)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth + 1  # 当前节点深度 = 最深子节点 + 1
    
    def _print_tree_structure(self, node, candidates, depth=0, max_depth=5, prefix="", is_last=True):
        """
        递归打印MCTS树的结构
        
        Args:
            node: 当前节点
            candidates: 候选点列表
            depth: 当前深度
            max_depth: 最大递归深度
            prefix: 打印前缀（用于显示树结构）
            is_last: 是否是最后一个子节点
        """
        if depth > max_depth:
            return
        
        # 计算节点信息
        avg_value = node.value / node.visits if node.visits > 0 else 0.0
        
        # 找到节点对应的candidate索引（如果不是root）
        node_cand_idx = None
        if node.parent:
            for idx, child in node.parent.children.items():
                if child == node:
                    node_cand_idx = idx
                    break
        
        # 打印当前节点信息
        if depth == 0:
            # Root节点
            print(f"ROOT: pos={node.position}, temp_start={node.temp_start}")
            print(f"      visits={node.visits}, total_reward={node.value:.4f}, avg_value={avg_value:.4f}")
            print(f"      visited_candidates={sorted(list(node.visited))}")
            print(f"      children_count={len(node.children)}")
        else:
            # 子节点 - 计算UCB值
            if node.visits > 0 and node.parent.visits > 0:
                exploitation = node.value / node.visits
                exploration = self.exploration_weight * math.sqrt(
                    math.log(node.parent.visits + 1) / (node.visits + 1e-6)
                )
                ucb = exploitation + exploration
            elif node.visits == 0:
                exploitation = 0.0
                exploration = self.exploration_weight * math.sqrt(
                    math.log(node.parent.visits + 1) / 1e-6
                )
                ucb = exploitation + exploration
            else:
                exploitation = 0.0
                exploration = 0.0
                ucb = 0.0
            
            connector = "└── " if is_last else "├── "
            print(f"{prefix}{connector}Cand#{node_cand_idx}: pos={node.position}, temp_start={node.temp_start}")
            print(f"{prefix}{'    ' if is_last else '│   '}    visits={node.visits}, total_reward={node.value:.4f}, avg_value={avg_value:.4f}")
            print(f"{prefix}{'    ' if is_last else '│   '}    UCB={ucb:.4f} (exploitation={exploitation:.4f} + exploration={exploration:.4f})")
            print(f"{prefix}{'    ' if is_last else '│   '}    visited={sorted(list(node.visited))}, children={len(node.children)}")
        
        # 递归打印子节点
        if node.children and depth < max_depth:
            # 按照candidate索引排序子节点
            sorted_children = sorted(node.children.items(), key=lambda x: x[0])
            
            for i, (idx, child) in enumerate(sorted_children):
                is_last_child = (i == len(sorted_children) - 1)
                
                if depth == 0:
                    new_prefix = ""
                else:
                    new_prefix = prefix + ("    " if is_last else "│   ")
                
                self._print_tree_structure(child, candidates, depth + 1, max_depth, new_prefix, is_last_child)
    
    def _print_best_path(self, node, candidates, depth=0, max_depth=5):
        """
        递归打印从当前节点开始的最优Value路径（按avg_value选择）
        
        Args:
            node: 当前节点
            candidates: 候选点列表
            depth: 当前深度
            max_depth: 最大递归深度
        """
        if depth >= max_depth or not node.children:
            return
        
        # 计算所有子节点的avg_value，选择最优的
        best_child = None
        best_child_idx = None
        best_avg_value = -float('inf')
        
        for idx, child in node.children.items():
            if child.visits > 0:
                avg_value = child.value / child.visits
                
                if avg_value > best_avg_value:
                    best_avg_value = avg_value
                    best_child = child
                    best_child_idx = idx
        
        if best_child is None:
            return
        
        # 打印当前层的最优选择
        coord = candidates[best_child_idx]['coord']
        exploitation = best_child.value / best_child.visits
        exploration = self.exploration_weight * math.sqrt(
            math.log(node.visits) / best_child.visits
        )
        uct = exploitation + exploration
        
        indent = "  " * depth
        print(f"{indent}Step {depth + 1}: Candidate #{best_child_idx} {coord} | "
              f"avg_value={exploitation:.4f}, UCT={uct:.4f} "
              f"(exploit={exploitation:.4f}, explore={exploration:.4f}) | "
              f"visits={best_child.visits}")
        
        # 递归打印下一层
        self._print_best_path(best_child, candidates, depth + 1, max_depth)
    
    @staticmethod
    def _calculate_heading(temp_start, now, goal):
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
        if temp_start != now:
            # 向量1: 从temp_start到now（当前运动方向）
            vector_start_now = np.array((now[0] - temp_start[0], now[1] - temp_start[1]))
            # 向量2: 从now到goal（目标方向）
            vector_now_goal = np.array((goal[0] - now[0], goal[1] - now[1]))
            
            # 计算两个向量的夹角
            cos_angle = np.dot(vector_start_now, vector_now_goal) / (
                np.linalg.norm(vector_start_now) * np.linalg.norm(vector_now_goal)
            )
            # 限制cos_angle在[-1, 1]范围内，避免数值误差
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_angle))
        else:
            # 如果temp_start == now，说明刚开始或刚turning，角度为0
            angle_deg = 0.0
        
        return angle_deg


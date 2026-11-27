"""Monte Carlo tree search for frontier selection."""

import math
import random
import numpy as np

def heuristic(a, b):
    """Return Euclidean distance between two points."""
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

class MCTSNode:
    """A node in the frontier search tree."""
    def __init__(self, visited_indices, current_position, parent=None, temp_start=None):
        self.visited = set(visited_indices)
        self.position = current_position
        self.parent = parent
        self.temp_start = temp_start
        self.children = {}                    # {candidate_idx: child_node}
        self.visits = 0
        self.value = 0.0
        self.unvisited = []
    
    def set_unvisited(self, total_candidates):
        """Set the candidates that have not yet been visited."""
        self.unvisited = [i for i in range(total_candidates) if i not in self.visited]

class MCTSPlanner:
    """Select a frontier using Monte Carlo tree search."""
    
    def __init__(self, path_planner, robot_radius=5, exploration_weight=1.414):
        """Initialize search parameters."""
        self.path_planner = path_planner
        self.robot_radius = robot_radius
        self.exploration_weight = exploration_weight
    
    def select_best_frontier(self, occupancy_map, current_pos, candidates, 
                            iterations, weights, temp_start, bad_frontier,
                            current_to_candidates_dist, max_candidate_distance,
                            unexplored_points, candidate_unexplored_points, num_bad_frontier,
                            turning_mode=False):
        """Select the best frontier by searching candidate sequences."""
        if not candidates:
            return None, None
        
        root = MCTSNode([], current_pos, parent=None, temp_start=temp_start)
        root.set_unvisited(len(candidates))
        
        # max_visits_in_simulation = max(1, int(len(candidates) * 0.5))
        max_visits_in_simulation = min(3, len(candidates))
        
        turning_mode_str = "TURNING MODE ⤵️" if turning_mode else "NORMAL MODE ➡️"
        print(f"[MCTS] 🔍 Starting search ({turning_mode_str}): {iterations} iterations, {len(candidates)} candidates, rollout limit={max_visits_in_simulation}")
        
        for iteration in range(iterations):
            is_first_iteration = (iteration == -1)
            
            node = self._select(root, candidates)
            
            if node.unvisited and len(node.visited) < max_visits_in_simulation:
                if node.parent is None or node.visits > 0:
                    self._expand(node, candidates, weights)
                    
                    if node.children:
                        newly_expanded = [idx for idx in node.children.keys() 
                                         if node.children[idx].visits == 0]
                        if newly_expanded:
                            selected_idx = random.choice(newly_expanded)
                            node = node.children[selected_idx]
            
            reward = self._simulate(
                node, candidates, occupancy_map, weights, temp_start, bad_frontier, max_visits_in_simulation,
                current_to_candidates_dist, max_candidate_distance,
                unexplored_points, candidate_unexplored_points, num_bad_frontier,
                is_first_iteration, turning_mode, current_pos
            )
            
            self._backpropagate(node, reward)
        
        max_tree_depth = self._get_max_depth(root)
        print(f"[MCTS] ✅ Search completed: tree depth={max_tree_depth}")
        
        if not root.children:
            return None, None
        
        child_stats = []
        for idx, child in root.children.items():
            avg_value = child.value / max(1, child.visits)
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
        
        child_stats.sort(key=lambda x: (np.isnan(x['avg_value']), -x['avg_value'] if not np.isnan(x['avg_value']) else 0))
        
        top_k = min(5, len(child_stats))
        print(f"[MCTS] 📊 Top {top_k} candidates by avg_value:")
        for i in range(top_k):
            stat = child_stats[i]
            print(f"  #{i+1}: idx={stat['idx']}, coord={stat['coord']}, "
                  f"avg_value={stat['avg_value']:.4f}, UCB={stat['ucb']:.4f} "
                  f"(exploit={stat['exploitation']:.4f}, explore={stat['exploration']:.4f}), "
                  f"visits={stat['visits']}")
        
        best_candidate = None
        for stat in child_stats:
            if not np.isnan(stat['avg_value']):
                best_candidate = stat
                break
        
        if best_candidate is None:
            best_candidate = max(child_stats, key=lambda x: x['visits'])
        
        best_idx = best_candidate['idx']
        best_child = root.children[best_idx]
        best_coord = best_candidate['coord']
        
        print(f"[MCTS] 🗺️  Long-term Planning - Best UCT path from root:")
        print(f"  Step 1: Candidate #{best_idx} {best_coord} | "
              f"avg_value={best_candidate['avg_value']:.4f}, UCB={best_candidate['ucb']:.4f} "
              f"(exploit={best_candidate['exploitation']:.4f}, explore={best_candidate['exploration']:.4f}) | "
              f"visits={best_child.visits}")
        self._print_best_uct_path(best_child, candidates, depth=1, max_depth=5, temp_start=temp_start, current_pos=current_pos)
        
        print(f"\n[MCTS] 🔬 Debug - Evaluating Best UCT path reward:")
        self._debug_evaluate_best_path(root, best_child, candidates, current_to_candidates_dist, 
                                      max_candidate_distance, unexplored_points, candidate_unexplored_points,
                                      num_bad_frontier, bad_frontier, max_visits_in_simulation,
                                      temp_start, current_pos, turning_mode, weights)
        
        print(f"[MCTS] 🎯 Selected: idx={best_idx}, coord={best_coord}, avg_value={best_candidate['avg_value']:.4f}, visits={best_child.visits}")
        
        value_dict = {stat['idx']: stat['avg_value'] for stat in child_stats}
        return best_coord, value_dict
    
    def _select(self, node, candidates):
        """Select a node using the UCT score."""
        while node.children and not node.unvisited:
            best_score = -float('inf')
            best_child = None
            
            for idx, child in node.children.items():
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
        """Expand unvisited children."""
        for candidate_idx in list(parent.unvisited):
            new_visited = parent.visited | {candidate_idx}
            new_position = candidates[candidate_idx]['coord']
            
            need_turning = self._check_if_need_turning(
                parent.position, parent.temp_start, new_position, 
                candidates, parent.visited, weights.get('angle', 1.0)
            )
            
            if need_turning:
                child_temp_start = self._compute_updated_temp_start(
                    parent.temp_start, parent.position, step_size=10.0
                )
            else:
                child_temp_start = parent.temp_start
            
            child = MCTSNode(new_visited, new_position, parent, temp_start=child_temp_start)
            child.set_unvisited(len(candidates))
            
            parent.children[candidate_idx] = child
        
        parent.unvisited = []
    
    def _simulate(self, node, candidates, occupancy_map, weights, temp_start, bad_frontier, 
                  max_visits_in_simulation,
                  current_to_candidates_dist, max_candidate_distance,
                  unexplored_points, candidate_unexplored_points, num_bad_frontier,
                  is_first_iteration, turning_mode, real_current_pos):
        """Estimate reward with a greedy rollout."""
        visited = set(node.visited)
        current_pos = node.position
        sim_temp_start = node.temp_start
        
        step_num = 0
        
        
        while len(visited) < max_visits_in_simulation:
            step_num += 1
            
            sim_turning, new_temp_start = self._check_turning_with_update(
                current_pos, sim_temp_start, candidates, visited, 
                w_angle=weights.get('angle', 1.0)
            )
            
            if new_temp_start is not None:
                sim_temp_start = new_temp_start
            
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
            
            best_score = -float('inf')
            best_idx = None
            best_coord = None
            
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
                    angle_score = 0.5
                else:
                    theta = abs(self._calculate_heading(sim_temp_start, current_pos, coord))
                    angle_score = 1.0 / (1.0 + math.exp(0.4 * (theta - 60.0)))
                
                if max_diff_sim != 0:
                    diff_norm = abs(d_sim - mid_d_sim) / max_diff_sim
                else:
                    diff_norm = 0.0
                distance_score = 1 - diff_norm
                
                explore_score = cand['explore_score']
                
                if distance_score > 0 and avoid_score > 0 and explore_score > 0:
                    # value_score = (weights['angle'] * angle_score + 
                    #               weights['dist'] * distance_score +
                    #               weights['explore'] * explore_score + 
                    #               weights['avoid'] * avoid_score)
                    value_score = (weights['dist'] * distance_score +
                                  weights['explore'] * explore_score + 
                                  weights['avoid'] * avoid_score)
                else:
                    value_score = 0.0
                
                if value_score > best_score:
                    best_score = value_score
                    best_idx = i
                    best_coord = coord
            
            if best_idx is None:
                break
            
            visited.add(best_idx)
            current_pos = best_coord
        
        
        selection_indices = []
        current_node = node
        temp_path = []
        
        while current_node.parent is not None:
            temp_path.append(current_node)
            current_node = current_node.parent
        temp_path.reverse()
        
        for path_node in temp_path:
            if path_node.parent:
                parent_visited = path_node.parent.visited
                node_visited = path_node.visited
                new_candidates = node_visited - parent_visited
                if new_candidates:
                    selection_indices.extend(list(new_candidates))
        
        rollout_indices = [idx for idx in visited if idx not in node.visited]
        
        
        reward = self._calculate_reward(
            selection_indices, rollout_indices,
            candidates, current_to_candidates_dist, max_candidate_distance,
            unexplored_points, candidate_unexplored_points,
            num_bad_frontier, bad_frontier, max_visits_in_simulation,
            is_first_iteration, 
            temp_start=temp_start,
            current_pos=real_current_pos,
            turning_mode=turning_mode,
            weights=weights
        )
        
        return reward
    
    def _calculate_reward(self, selection_indices, rollout_indices,
                              candidates, current_to_candidates_dist, max_candidate_distance,
                              unexplored_points, candidate_unexplored_points,
                              num_bad_frontier, bad_frontier, max_visits_in_simulation,
                              is_first_iteration, temp_start=None, current_pos=None, turning_mode=False, weights=None):
        """Combine exploration, distance and avoidance rewards."""
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
        
        
        gamma_exploration = 0.3
        first_step_explore_ratio = None
        
        max_exploration_points = 0
        for idx in range(len(candidate_unexplored_points)):
            num_points = len(candidate_unexplored_points[idx])
            max_exploration_points = max(max_exploration_points, num_points)
        
        if max_exploration_points > 0:
            discounted_exploration = 0.0
            for step_idx, candidate_idx in enumerate(all_visited_indices):
                if candidate_idx < len(candidate_unexplored_points):
                    exploration_points = len(candidate_unexplored_points[candidate_idx])
                    discount = gamma_exploration ** step_idx
                    discounted_exploration += discount * exploration_points
                    if step_idx == 0 and max_exploration_points > 0:
                        first_step_explore_ratio = exploration_points / max_exploration_points
                        first_step_explore_ratio = max(0.0, min(1.0, first_step_explore_ratio))
            
            max_discounted = sum(gamma_exploration ** i for i in range(len(all_visited_indices))) * max_exploration_points
            
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
        
        unique_path = []
        for idx in all_visited_indices:
            if not unique_path or idx != unique_path[-1]:
                unique_path.append(idx)
        
        total_path_length = 0.0
        path_segments = []
        
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
        
        max_theoretical_length = max_visits_in_simulation * max_candidate_distance
        
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
        
        
        gamma_avoidance = 0.3
        first_step_avoid_ratio = None
        
        bad_frontier_list = list(bad_frontier) if bad_frontier else []
        
        max_possible_distance = 0.0
        if len(bad_frontier_list) > 0:
            discounted_avoidance = 0.0
            max_possible_distance = 0.0
            
            for idx in range(len(candidates)):
                coord = candidates[idx]['coord']
                for bad_coord in bad_frontier_list:
                    dist = heuristic(coord, bad_coord)
                    max_possible_distance = max(max_possible_distance, dist)
            
            for step_idx, candidate_idx in enumerate(all_visited_indices):
                if candidate_idx < len(candidates):
                    coord = candidates[candidate_idx]['coord']
                    
                    min_dist_to_bad = float('inf')
                    closest_bad_idx = None
                    for bad_idx, bad_coord in enumerate(bad_frontier_list):
                        dist = heuristic(coord, bad_coord)
                        if dist < min_dist_to_bad:
                            min_dist_to_bad = dist
                            closest_bad_idx = bad_idx
                    
                    if min_dist_to_bad != float('inf'):
                        discount = gamma_avoidance ** step_idx
                        discounted_avoidance += discount * min_dist_to_bad
                        if step_idx == 0 and max_possible_distance > 0:
                            first_step_avoid_ratio = min_dist_to_bad / max_possible_distance
                            first_step_avoid_ratio = max(0.0, min(1.0, first_step_avoid_ratio))
            
            max_discounted_avoidance = sum(gamma_avoidance ** i for i in range(len(all_visited_indices))) * max_possible_distance
            
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
            avoidance_reward = 1.0
            first_step_avoid_ratio = 1.0
            if is_first_iteration:
                print(f"\n[3] 🚫 Avoidance Reward:")
                print(f"  • No bad frontiers → avoidance_reward = 1.0")

        if first_step_avoid_ratio is None:
            first_step_avoid_ratio = 0.0 if max_possible_distance > 0 else 1.0
        
        
        if turning_mode:
            first_step_angle_reward = 1.0
        elif temp_start is not None and current_pos is not None and len(all_visited_indices) > 0:
            first_step_idx = all_visited_indices[0]
            if first_step_idx < len(candidates):
                first_step_coord = candidates[first_step_idx]['coord']
                
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
                print(f"  Turning mode: angle reward fixed at 1.0")
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
        
        epsilon = 1e-10
        w_exploration = weights['explore']
        w_distance = weights['dist']
        w_avoidance = weights['avoid']
        w_angle = weights['angle']

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
        """Update the simulated reference position for heading estimates."""
        dx = now[0] - old_temp_start[0]
        dy = now[1] - old_temp_start[1]
        length = math.sqrt(dx**2 + dy**2)
        
        if length == 0:
            return now
        
        step_x = (dx / length) * step_size
        step_y = (dy / length) * step_size
        new_temp_start = (round(now[0] + step_x), round(now[1] + step_y))
        
        return new_temp_start
    
    def _check_turning_with_update(self, current_pos, temp_start, candidates, visited, w_angle=1.0):
        """Update the reference position when the rollout requires a turn."""
        if w_angle == 0.0:
            return False, None
        
        if current_pos == temp_start:
            return False, None
        
        unvisited_candidates = [(i, cand) for i, cand in enumerate(candidates) if i not in visited]
        
        if not unvisited_candidates:
            return False, None
        
        candidates_with_angles = []
        for i, cand in unvisited_candidates:
            coord = cand['coord']
            theta = abs(self._calculate_heading(temp_start, current_pos, coord))
            candidates_with_angles.append((theta, coord, i))
        
        candidates_with_angles.sort()
        
        top_k = max(1, math.floor(0.2 * len(candidates_with_angles)))
        
        good_candidates = []
        for theta, coord, idx in candidates_with_angles[:top_k]:
            dist = heuristic(current_pos, coord)
            if theta < 90 and dist > 30:
                good_candidates.append((theta, coord, idx, dist))
        
        if good_candidates:
            return False, None
        
        new_temp_start = self._compute_updated_temp_start(temp_start, current_pos, step_size=10.0)
        
        return True, new_temp_start
    
    def _check_if_need_turning(self, parent_pos, parent_temp_start, child_pos, 
                                candidates, parent_visited, w_angle=1.0):
        """Determine whether expanding a child requires a turnaround."""
        if w_angle == 0.0:
            return False
        
        if parent_pos == parent_temp_start:
            return False
        
        unvisited_candidates = [(i, cand) for i, cand in enumerate(candidates) 
                               if i not in parent_visited]
        
        if not unvisited_candidates:
            return False
        
        candidates_with_angles = []
        for i, cand in unvisited_candidates:
            coord = cand['coord']
            theta = abs(self._calculate_heading(parent_temp_start, parent_pos, coord))
            candidates_with_angles.append((theta, coord, i))
        
        candidates_with_angles.sort()
        
        top_k = max(1, math.floor(0.2 * len(candidates_with_angles)))
        
        for theta, coord, idx in candidates_with_angles[:top_k]:
            dist = heuristic(parent_pos, coord)
            if theta < 90 and dist > 30:
                return False
        
        return True
    
    def _calculate_reaching_value(self, parent_pos, node_pos, parent_visited, 
                                   candidates, weights, temp_start, bad_frontier):
        """Score the transition from the parent to this node."""
        target_idx = None
        for i, cand in enumerate(candidates):
            if cand['coord'] == node_pos:
                target_idx = i
                break
        
        if target_idx is None:
            return 0.0
        
        sim_turning, _ = self._check_turning_with_update(
            parent_pos, temp_start, candidates, parent_visited,
            w_angle=weights.get('angle', 1.0)
        )
        
        avoid_points = []
        if parent_visited:
            visited_coords = [candidates[v]['coord'] for v in parent_visited]
            avoid_points.extend(visited_coords)
        if bad_frontier:
            avoid_points.extend(list(bad_frontier))
        
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
        
        mid_d = 0.3 * min_dist + 0.7 * max_dist
        distances_to_mid = [abs(d - mid_d) for _, _, d, _ in unvisited_data]
        max_diff = max(distances_to_mid) if distances_to_mid else 0.0
        
        target_value = 0.0
        details = None
        
        for i, coord, d, min_dist_to_avoid in unvisited_data:
            if i != target_idx:
                continue
            
            # angle_score
            if sim_turning:
                angle_score = 0.5
                theta = 0.0
            else:
                theta = abs(self._calculate_heading(temp_start, parent_pos, coord))
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
            
            if distance_score > 0 and avoid_score > 0 and explore_score > 0:
                target_value = (weights['angle'] * angle_score +
                               weights['dist'] * distance_score +
                               weights['explore'] * explore_score +
                               weights['avoid'] * avoid_score)
            else:
                target_value = 0.0
            
            break
        
        return target_value
    
    def _backpropagate(self, node, reward):
        """Propagate the rollout reward along the search path."""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    def _get_max_depth(self, node):
        """Find the maximum depth in visited candidates."""
        if not node.children:
            return len(node.visited)
        
        max_child_depth = 0
        for child in node.children.values():
            child_depth = self._get_max_depth(child)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
    
    def _print_best_uct_path(self, node, candidates, depth=0, max_depth=5, temp_start=None, current_pos=None):
        """Print the highest-UCT path from the current node."""
        if depth >= max_depth or not node.children:
            return
        
        best_ucb = -float('inf')
        best_child = None
        best_idx = None
        
        for idx, child in node.children.items():
            if child.visits == 0:
                continue
            
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
        
        avg_value = best_child.value / max(1, best_child.visits)
        exploitation = best_child.value / best_child.visits
        exploration = self.exploration_weight * math.sqrt(
            math.log(node.visits) / best_child.visits
        )
        
        angle_info = ""
        if depth == 0 and temp_start is not None and current_pos is not None:
            coord = candidates[best_idx]['coord']
            if temp_start != current_pos:
                theta = abs(self._calculate_heading(temp_start, current_pos, coord))
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
        
        self._print_best_uct_path(best_child, candidates, depth + 1, max_depth, temp_start, current_pos)
    
    def _extract_best_uct_path(self, root, first_idx, candidates, max_visits):
        """Extract the UCT path and its greedy rollout."""
        path_indices = [first_idx]
        current_node = root.children.get(first_idx)
        
        if current_node is None:
            return path_indices
        
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
        """Evaluate the reward for a specified path."""
        print(f"  Path (tree exploration): {path_indices}")
        print(f"  ⚠️  Note: This is a debug evaluation that may differ from actual simulation rewards")
        print(f"      because it uses root's temp_start instead of each node's updated temp_start.")
        
        selection_indices = path_indices.copy()
        
        rollout_indices = []
        if len(path_indices) < max_visits_in_simulation:
            print(f"  Continuing with greedy rollout from path end...")
            visited = set(path_indices)
            current_rollout_pos = candidates[path_indices[-1]]['coord']
            
            while len(visited) < max_visits_in_simulation:
                best_score = -float('inf')
                best_idx = None
                
                for i, cand in enumerate(candidates):
                    if i in visited:
                        continue
                    
                    coord = cand['coord']
                    dist = heuristic(current_rollout_pos, coord)
                    explore_points = len(candidate_unexplored_points[i])
                    
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
        
        total_reward = self._calculate_reward(
            selection_indices, rollout_indices,
            candidates, current_to_candidates_dist, max_candidate_distance,
            unexplored_points, candidate_unexplored_points,
            num_bad_frontier, bad_frontier, max_visits_in_simulation,
            is_first_iteration=True,
            temp_start=temp_start, current_pos=current_pos, turning_mode=turning_mode,
            weights=weights
        )
        
        print(f"  Total reward for this path (debug evaluation): {total_reward:.4f}\n")
        
        if return_reward:
            return total_reward
        return None
    
    def _calculate_heading(self, temp_start, now, goal):
        """Measure the change from the previous heading to the goal heading."""
        if now == goal or (now[0] == goal[0] and now[1] == goal[1]):
            return 0.0
        
        if temp_start == now or (temp_start[0] == now[0] and temp_start[1] == now[1]):
            return 0.0
        
        vector_start_now = np.array((now[0] - temp_start[0], now[1] - temp_start[1]))
        vector_now_goal = np.array((goal[0] - now[0], goal[1] - now[1]))
        
        norm_start_now = np.linalg.norm(vector_start_now)
        norm_now_goal = np.linalg.norm(vector_now_goal)
        
        if norm_start_now < 1e-6 or norm_now_goal < 1e-6:
            return 0.0
        
        cos_angle = np.dot(vector_start_now, vector_now_goal) / (norm_start_now * norm_now_goal)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        
        return angle_deg
    
    def _debug_evaluate_best_path(self, root, best_child, candidates, current_to_candidates_dist, 
                                  max_candidate_distance, unexplored_points, candidate_unexplored_points,
                                  num_bad_frontier, bad_frontier, max_visits_in_simulation,
                                  temp_start, current_pos, turning_mode, weights):
        """Report reward components along the highest-UCT path."""
        path_indices = []
        current_node = best_child
        while current_node != root:
            for idx, child in current_node.parent.children.items():
                if child == current_node:
                    path_indices.append(idx)
                    break
            current_node = current_node.parent
        
        path_indices.reverse()
        
        print(f"  Path (selection from root to best child): {path_indices}")
        
        rollout_indices = []
        visited = set(best_child.visited)
        current_pos_sim = best_child.position
        
        for _ in range(max_visits_in_simulation - len(visited)):
            best_score = -float('inf')
            best_idx = None
            
            for i, cand in enumerate(candidates):
                if i in visited:
                    continue
                
                coord = cand['coord']
                d_sim = heuristic(current_pos_sim, coord)
                
                explore_score = cand['explore_score']
                distance_score = 1.0
                avoid_score = 1.0
                
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
        
        all_visited_indices = path_indices + rollout_indices
        print(f"\n{'='*80}")
        print(f"[MCTS] 🧮 Reward Calculation for Best Path")
        print(f"{'='*80}\n")
        
        reward = self._calculate_reward(
            path_indices, rollout_indices,
            candidates, current_to_candidates_dist, max_candidate_distance,
            unexplored_points, candidate_unexplored_points,
            num_bad_frontier, bad_frontier, max_visits_in_simulation,
            is_first_iteration=True,
            temp_start=temp_start,
            current_pos=current_pos,
            turning_mode=turning_mode,
            weights=weights
        )
        
        print(f"\n[MCTS] 🎯 Best path total reward: {reward:.4f}")
        print()
    

"""
Reward Calculation functions for the Graph Routing project.

This module contains reward calculation functions for the Graph Routing project.
"""
import numpy as np

def calculate_improved_reward_v2(cur_node_id, next_node_id, goal_node_id, graph_cache,
                                is_done=False, reward_goal=10.0, step_penalty=-0.01,
                                is_failure=False, failure_type=None,
                                prev_edge_id=None, recent_visited_nodes=None,
                                backtrack_penalty=-0.06, repeat_visit_penalty=-0.03):
    """
    🔥 势能型奖励（Potential-based Reward Shaping）- V2版本：基于node_id的跨地图泛化版本

    使用GraphCache进行统一距离计算，确保preprocess/reward/feature/evaluation口径一致。

    Args:
        cur_node_id: 当前节点ID
        next_node_id: 下一节点ID
        goal_node_id: 目标节点ID
        graph_cache: GraphCache实例
        is_done: 是否到达终止状态
        reward_goal: 目标奖励值
        step_penalty: 步惩罚
        is_failure: 是否失败
        failure_type: 失败类型 ('loop_stuck', 'timeout', etc.)
        prev_edge_id: 上一步的边ID（用于回退检测）
        recent_visited_nodes: 最近访问的节点列表（用于循环检测）
        backtrack_penalty: 回退惩罚
        repeat_visit_penalty: 重复访问惩罚

    Returns:
        reward: 奖励值
        reward_components: 奖励组件字典
    """
    reward_components = {}

    # ---------- 1) 终止奖励/惩罚 ----------
    terminal_reward = 0.0
    if is_done:
        terminal_reward = reward_goal
        reward_components['terminal_success'] = terminal_reward
    elif is_failure:
        if failure_type == 'loop_stuck':
            terminal_reward = -reward_goal
        elif failure_type == 'timeout':
            terminal_reward = -0.4 * reward_goal
        else:
            terminal_reward = -0.3 * reward_goal
        reward_components['terminal_failure'] = terminal_reward

    # ---------- 2) 势能型进展奖励 + 固定步惩罚 ----------
    progress_reward = step_penalty
    reward_components['step_penalty'] = float(step_penalty)

    # 使用GraphCache计算统一距离
    d_cur = graph_cache.get_dist_to_goal(cur_node_id, goal_node_id)
    d_next = graph_cache.get_dist_to_goal(next_node_id, goal_node_id)
    delta_d = d_next - d_cur

    # 进展奖励：距离减少多少就奖励多少
    progress_reward += -delta_d  # 负号因为距离减少是好的
    reward_components['progress'] = -delta_d
    reward_components['d_cur'] = d_cur
    reward_components['d_next'] = d_next
    reward_components['delta_d'] = delta_d

    # ---------- 3) 回退惩罚 ----------
    backtrack_reward = 0.0
    if prev_edge_id is not None:
        # 检查是否严格反向边：从当前边到达的节点是否是上一步边的起始节点
        try:
            prev_u, prev_v = graph_cache.get_edge_nodes(prev_edge_id)
            # 如果下一步节点等于上一步边的起始节点，说明在回退
            if next_node_id == prev_u:
                backtrack_reward = backtrack_penalty
                reward_components['backtrack'] = backtrack_reward
        except:
            pass

    # ---------- 4) 循环惩罚 ----------
    loop_reward = 0.0
    if recent_visited_nodes and next_node_id in recent_visited_nodes:
        loop_reward = repeat_visit_penalty
        reward_components['loop'] = loop_reward

    total_reward = terminal_reward + progress_reward + backtrack_reward + loop_reward
    reward_components['total'] = total_reward

    return total_reward, reward_components

def calculate_improved_reward(action_id, is_done, reward_goal, 
                             step_idx=0, state_history=None, path_info=None, 
                             prev_action=None, edge_id_map=None, step_penalty=-0.01, backtrack_penalty=-0.06,
                             is_failure=False, failure_type=None,
                             recent_visited_nodes=None, repeat_visit_penalty=-0.03,
                             G=None, item_num=None):
    """
    🔥 势能型奖励（Potential-based Reward Shaping）
    
    核心原理：
    1. 终止奖励：成功 = +reward_goal；loop = -reward_goal；timeout = -0.4×reward_goal
    2. 进展奖励（势能差）：Φ(s) = -d(s, goal) → r_shaping = Φ(s') - Φ(s) = d_prev - d_current
    3. 步惩罚：从专家路径长度推导 step_penalty = -reward_goal / (expert_length × tolerance_factor)
    4. 回退惩罚：仅保留严格反向边检测（A→B then B→A）
    
    优点：
    - 距离缩短多少就奖励多少，与目标对齐
    - 步惩罚可解释，避免"拍脑袋"常数
    - 消除复杂的阈值判断，减少抖动源
    """
    reward_components = {}

    # ---------- 1) 终止奖励/惩罚（与 reward_goal 同量纲）----------
    terminal_reward = 0.0
    if is_done:
        terminal_reward = reward_goal
        reward_components['terminal_success'] = terminal_reward
    elif is_failure:
        if failure_type == 'loop_stuck':
            terminal_reward = -reward_goal  # 🔥 与成功奖励对称
        elif failure_type == 'timeout':
            terminal_reward = -0.4 * reward_goal  # 🔥 与 reward_goal 成比例
        else:
            terminal_reward = -0.3 * reward_goal
        reward_components['terminal_failure'] = terminal_reward

    # ---------- 2) 势能型进展奖励 + 固定步惩罚 ----------
    # 🔥 使用固定步惩罚，不再根据target_length自适应
    progress_reward = step_penalty
    reward_components['step_penalty'] = float(step_penalty)
    
    try:
        if state_history is not None and path_info is not None and 'end_pos' in path_info:
            # 🔥 提取当前位置和目标位置
            current_pos = None
            if isinstance(state_history, np.ndarray) and state_history.ndim == 2 and len(state_history) >= 1:
                for i in range(len(state_history) - 1, -1, -1):
                    if not np.all(state_history[i] == 0) and len(state_history[i]) >= 15:
                        # 🔥 修复：使用固定索引13, 14，而不是负索引
                        x, y = float(state_history[i][13]), float(state_history[i][14])
                        if not (np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y)):
                            current_pos = (x, y)
                            break
            target_pos = path_info['end_pos']

            prev_pos = None
            if isinstance(state_history, np.ndarray) and state_history.ndim == 2 and len(state_history) >= 2:
                non_zero_count = 0
                for i in range(len(state_history) - 1, -1, -1):
                    if not np.all(state_history[i] == 0) and len(state_history[i]) >= 15:
                        non_zero_count += 1
                        if non_zero_count == 2:  # 倒数第二个非全零状态
                            # 🔥 修复：使用固定索引13, 14，而不是负索引
                            x, y = float(state_history[i][13]), float(state_history[i][14])
                            if not (np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y)):
                                prev_pos = (x, y)
                            break

            if current_pos is not None and target_pos is not None:
                cur_dist = np.hypot(current_pos[0] - target_pos[0], current_pos[1] - target_pos[1])

                if 'distance_improvement' in path_info:
                    distance_improvement = float(path_info['distance_improvement'])  # prev_dist - cur_dist
                    prev_dist = cur_dist + distance_improvement
                else:
                    if prev_pos is not None:
                        prev_dist = np.hypot(prev_pos[0] - target_pos[0], prev_pos[1] - target_pos[1])
                        distance_improvement = prev_dist - cur_dist
                    else:
                        prev_dist = cur_dist
                        distance_improvement = 0.0
                
                # 🔥 势能型奖励：直接用距离变化作为奖励（Φ(s') - Φ(s) = d_prev - d_current）
                # 距离缩短 → 正奖励；距离增加 → 负奖励
                potential_reward = distance_improvement
                
                # 合并步惩罚和势能奖励
                progress_reward = step_penalty + potential_reward
                
                # 文本标注（便于调试和日志分析）
                if distance_improvement > 0.01:
                    reward_components['progress_type'] = 'approach'
                elif distance_improvement < -0.01:
                    reward_components['progress_type'] = 'retreat'
                else:
                    reward_components['progress_type'] = 'stagnant'
                
                reward_components['distance_improvement'] = float(distance_improvement)
                reward_components['potential_reward'] = float(potential_reward)
                reward_components['current_distance'] = float(cur_dist)
                reward_components['previous_distance'] = float(prev_dist)
        else:
            reward_components['progress_type'] = 'default_penalty'
    except Exception as e:
        print(f"奖励计算异常: {e}")
        progress_reward = step_penalty
        reward_components['progress_type'] = 'error_fallback'

    # ---------- 3) 回退惩罚（仅保留严格反向边检测）----------
    backtrack_penalty_value = 0.0
    if prev_action is not None and edge_id_map is not None:
        if _is_backtrack_action(action_id, prev_action, edge_id_map):
            backtrack_penalty_value = backtrack_penalty
            reward_components['backtrack_penalty'] = float(backtrack_penalty_value)

    # ---------- 4) 重复访问惩罚（防止循环探索）----------
    repeat_visit_penalty_value = 0.0
    repeat_node_penalty = 0.0
    repeat_edge_penalty = 0.0
    cycle_pattern_penalty = 0.0
    
    # 提取当前动作会到达的节点和使用的边
    current_action_node = None
    current_action_edge_id = action_id
    
    try:
        # 从state_history中提取当前位置（执行动作前的位置）
        # 支持多种格式：numpy数组、列表等
        state_history_array = None
        if state_history is not None:
            # 转换为numpy数组以便处理
            if isinstance(state_history, list):
                state_history_array = np.array(state_history)
            elif isinstance(state_history, np.ndarray):
                state_history_array = state_history
        
        if state_history_array is not None and state_history_array.ndim == 2:
            # 获取最后一个非零状态作为当前位置
            current_pos = None
            for i in range(len(state_history_array) - 1, -1, -1):
                if not np.all(state_history_array[i] == 0) and len(state_history_array[i]) >= 15:
                    # 🔥 使用固定索引13, 14
                    # 🔥 修复：统一使用 state_history_array，而不是 state_history
                    x, y = float(state_history_array[i][13]), float(state_history_array[i][14])
                    if not (np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y)):
                        current_pos = (x, y)
                        break
            
            # 如果找到了当前位置，尝试获取下一个位置（执行动作后的位置）
            if current_pos is not None and edge_id_map is not None:
                try:
                    # 查找当前动作对应的边
                    target_edge = None
                    for edge_key, edge_idx in edge_id_map.items():
                        if edge_idx == action_id:
                            target_edge = edge_key
                            break
                    
                    if target_edge is not None:
                        # 确定下一个节点（边的另一端）
                        if target_edge[0] == current_pos:
                            current_action_node = target_edge[1]
                        elif target_edge[1] == current_pos:
                            current_action_node = target_edge[0]
                except Exception:
                    pass
        
        # 4.1) 重复节点惩罚：如果下一个节点在最近访问列表中
        if current_action_node is not None and recent_visited_nodes is not None and len(recent_visited_nodes) > 0:
            # 检查下一个节点是否在最近访问的节点列表中
            node_tolerance = 1e-6  # 节点坐标比较的容差
            for visited_node in recent_visited_nodes:
                if visited_node is not None and len(visited_node) == 2:
                    dist = np.hypot(current_action_node[0] - visited_node[0], 
                                   current_action_node[1] - visited_node[1])
                    if dist < node_tolerance:
                        # 找到重复节点，应用惩罚
                        # 惩罚强度：距离越近的重复访问惩罚越大（最近访问的节点惩罚最大）
                        repeat_node_penalty = repeat_visit_penalty * 1.5  # 节点重复惩罚比边重复更严重
                        reward_components['repeat_node_penalty'] = float(repeat_node_penalty)
                        break
        
        # 4.2) 重复边惩罚：如果当前边在最近访问的边列表中
        # 🔥 排除已经被回退惩罚（3）检测到的情况，避免重复惩罚
        if state_history_array is not None and edge_id_map is not None:
            # 检查是否已经被回退惩罚检测到（严格反向边：A→B then B→A）
            is_backtrack = False
            if prev_action is not None:
                is_backtrack = _is_backtrack_action(action_id, prev_action, edge_id_map)
            
            # 只有不是回退动作时，才应用重复边惩罚
            if not is_backtrack:
                # 🔥 修复：统一使用 state_history_array
                # 从state_history_array中提取最近访问的边ID（最近5步）
                recent_edge_ids = extract_recent_edge_ids_from_state(state_history_array, edge_id_map, K=5)
                if current_action_edge_id in recent_edge_ids:
                    # 找到重复边，应用惩罚
                    # 惩罚强度：根据重复次数递增
                    repeat_count = recent_edge_ids.count(current_action_edge_id)
                    repeat_edge_penalty = repeat_visit_penalty * (1.0 + 0.3 * repeat_count)  # 重复次数越多，惩罚越大
                    reward_components['repeat_edge_penalty'] = float(repeat_edge_penalty)
                    reward_components['repeat_edge_count'] = int(repeat_count)
        
        # 4.3) 循环模式检测：检测短循环（如A-B-A或A-B-C-A）
        # 🔥 排除已经被回退惩罚（3）检测到的情况，避免重复惩罚
        if state_history_array is not None and recent_visited_nodes is not None and len(recent_visited_nodes) >= 2:
            # 检查是否已经被回退惩罚检测到（严格反向边：A→B then B→A）
            is_backtrack = False
            if prev_action is not None:
                is_backtrack = _is_backtrack_action(action_id, prev_action, edge_id_map)
            
            # 只有不是回退动作时，才应用循环模式惩罚
            if not is_backtrack:
                # 🔥 修复：统一使用 state_history_array
                # 提取最近访问的节点序列（从state_history_array中提取，更准确）
                recent_nodes_from_history = extract_recent_nodes_from_state(state_history_array, K=4)
                
                if current_action_node is not None and len(recent_nodes_from_history) >= 2:
                    # 检测2步循环：A-B-A
                    if len(recent_nodes_from_history) >= 2:
                        if (abs(current_action_node[0] - recent_nodes_from_history[-2][0]) < 1e-6 and
                            abs(current_action_node[1] - recent_nodes_from_history[-2][1]) < 1e-6):
                            # 形成了A-B-A循环
                            cycle_pattern_penalty = repeat_visit_penalty * 2.0  # 循环模式惩罚更严重
                            reward_components['cycle_pattern_penalty'] = float(cycle_pattern_penalty)
                            reward_components['cycle_type'] = '2-step'
                    
                    # 检测3步循环：A-B-C-A
                    if len(recent_nodes_from_history) >= 3:
                        if (abs(current_action_node[0] - recent_nodes_from_history[-3][0]) < 1e-6 and
                            abs(current_action_node[1] - recent_nodes_from_history[-3][1]) < 1e-6):
                            # 形成了A-B-C-A循环
                            cycle_pattern_penalty = repeat_visit_penalty * 1.5
                            reward_components['cycle_pattern_penalty'] = float(cycle_pattern_penalty)
                            reward_components['cycle_type'] = '3-step'
        
        # 4.4) 未探索边惩罚：如果当前节点还有未尝试的边，但对已尝试的边重复使用
        unexplored_edge_penalty = 0.0
        if current_pos is not None and G is not None and edge_id_map is not None and item_num is not None:
            try:
                # 获取当前节点的所有有效边
                if current_pos in G:
                    neighbors = list(G.neighbors(current_pos))
                    all_valid_edge_ids = set()
                    for neighbor in neighbors:
                        edge_key1 = (current_pos, neighbor)
                        edge_key2 = (neighbor, current_pos)
                        edge_id = edge_id_map.get(edge_key1, edge_id_map.get(edge_key2, -1))
                        if 0 <= edge_id < item_num:
                            all_valid_edge_ids.add(edge_id)
                    
                    # 从state_history中提取从当前节点出发已尝试的边（最近10步）
                    if state_history_array is not None and len(state_history_array) > 0:
                        # 提取所有最近访问的边ID
                        recent_edge_ids_all = extract_recent_edge_ids_from_state(state_history_array, edge_id_map, K=10)
                        tried_edges_from_current = []
                        for edge_id in recent_edge_ids_all:
                            # 检查这条边是否从当前节点出发
                            for edge_key, edge_idx in edge_id_map.items():
                                if edge_idx == edge_id:
                                    if edge_key[0] == current_pos or edge_key[1] == current_pos:
                                        if edge_id in all_valid_edge_ids:
                                            tried_edges_from_current.append(edge_id)
                                        break
                        
                        # 去重
                        tried_edges_from_current = list(set(tried_edges_from_current))
                        
                        # 计算未尝试的边数量
                        unexplored_edges = all_valid_edge_ids - set(tried_edges_from_current)
                        num_unexplored = len(unexplored_edges)
                        num_total = len(all_valid_edge_ids)
                        
                        # 如果还有未尝试的边，但当前动作使用的是已尝试的边，给予惩罚
                        # 🔥 排除已经被回退惩罚（3）检测到的情况，避免重复惩罚
                        is_backtrack = False
                        if prev_action is not None:
                            is_backtrack = _is_backtrack_action(action_id, prev_action, edge_id_map)
                        
                        if num_unexplored > 0 and current_action_edge_id in tried_edges_from_current and not is_backtrack:
                            # 惩罚强度：未尝试边越多，惩罚越大
                            # 公式：penalty = base_penalty * (unexplored_ratio) * (1 + repeat_count_factor)
                            unexplored_ratio = num_unexplored / max(num_total, 1)
                            repeat_count = tried_edges_from_current.count(current_action_edge_id)
                            unexplored_edge_penalty = repeat_visit_penalty * unexplored_ratio * (1.0 + 0.5 * repeat_count) * 2.0  # 2.0倍基础惩罚
                            reward_components['unexplored_edge_penalty'] = float(unexplored_edge_penalty)
                            reward_components['unexplored_edges_count'] = int(num_unexplored)
                            reward_components['total_edges_count'] = int(num_total)
                            reward_components['unexplored_ratio'] = float(unexplored_ratio)
            except Exception:
                # 如果计算失败，不影响其他奖励
                pass
        
        # 合并所有重复访问惩罚
        repeat_visit_penalty_value = repeat_node_penalty + repeat_edge_penalty + cycle_pattern_penalty + unexplored_edge_penalty
        if repeat_visit_penalty_value != 0.0:
            reward_components['repeat_visit_penalty'] = float(repeat_visit_penalty_value)
    
    except Exception as e:
        # 如果计算重复访问惩罚时出错，不影响其他奖励计算
        if 'repeat_visit_debug' in globals():
            print(f"⚠️ 重复访问惩罚计算异常: {e}")

    # ---------- 5) 合成最终奖励 ----------
    final_reward = terminal_reward + progress_reward + backtrack_penalty_value + repeat_visit_penalty_value
    # 🔥 裁剪范围：[-2×reward_goal, reward_goal]，避免异常峰值
    final_reward = float(np.clip(final_reward, -2.0 * reward_goal, reward_goal))
    reward_components['progress_reward'] = float(progress_reward)
    reward_components['final_reward'] = final_reward
    return final_reward, reward_components

def _is_backtrack_action(current_action_id, prev_action_id, edge_id_map):
    """
    🔥 简化回退检测：仅保留严格反向边检测（A→B then B→A）
    
    Args:
        current_action_id: 当前动作ID
        prev_action_id: 上一步动作ID
        edge_id_map: 边ID映射
        
    Returns:
        bool: 是否为回退动作
    """
    if prev_action_id is None:
        return False
    
 
    if current_action_id == prev_action_id:
        return True
    
    return False

def infer_prev_action_from_state_history(state, edge_id_map):
    """
    从状态历史中推断上一步动作
    
    Args:
        state: 状态向量 [state_size, feature_dim]
        edge_id_map: 边ID映射
        
    Returns:
        prev_action_id: 上一步动作的ID，无法推断返回None
    """
    if not isinstance(state, (list, np.ndarray)) or len(state) == 0:
        return None
    
    # 找到最后两个非零状态
    non_zero_states = []
    for i in range(len(state) - 1, -1, -1):
        if not np.all(state[i] == 0):
            non_zero_states.append(state[i])
            if len(non_zero_states) >= 2:
                break
    
    if len(non_zero_states) < 2:
        return None
    
    last_state = non_zero_states[0]
    
    if len(last_state) < 4:
        return None
    
    try:
        # 从最后状态的边信息推断动作
        last_edge = last_state[:4]  # [start_x, start_y, end_x, end_y]
        last_edge_key = ((last_edge[0], last_edge[1]), (last_edge[2], last_edge[3]))
        
        # 查找边ID
        if last_edge_key in edge_id_map:
            return edge_id_map[last_edge_key]
        elif (last_edge_key[1], last_edge_key[0]) in edge_id_map:
            return edge_id_map[(last_edge_key[1], last_edge_key[0])]
        
    except (IndexError, TypeError, ValueError):
        pass
    
    return None

"""
State Processing functions for the Graph Routing project.

This module contains state processing functions for the Graph Routing project.
"""
import numpy as np

def extract_target_position_from_state(state):
    """从状态中提取目标位置"""
    # 🔥 修复：使用固定索引11, 12，而不是负索引
    # 状态格式（前15维固定）：[start_x, start_y, end_x, end_y, crossing, path_type, length, width, curb, origin_x, origin_y, dest_x, dest_y, cur_x, cur_y]
    # 索引：              0        1        2      3      4         5          6       7      8     9         10         11     12      13     14
    # dest_x, dest_y 在索引 11, 12
    # state是嵌套列表结构 [state_size, feature_dim]
    if isinstance(state, list) and len(state) > 0:
        # 找到最后一个非零状态
        for i in range(len(state) - 1, -1, -1):
            if isinstance(state[i], list) and len(state[i]) >= 15:
                # 检查是否全为零
                if not all(x == 0 for x in state[i]):
                    return (float(state[i][11]), float(state[i][12]))  # dest_x, dest_y 在索引 11, 12
    elif hasattr(state, 'shape') and len(state.shape) > 1:
        # 多维数组情况：找到最后一个非零状态
        for i in range(state.shape[0] - 1, -1, -1):
            if not np.all(state[i] == 0):
                if len(state[i]) >= 15:
                    return (float(state[i][11]), float(state[i][12]))  # dest_x, dest_y 在索引 11, 12
                break
    elif isinstance(state, (list, np.ndarray)) and len(state) >= 15:
        # 一维列表情况
        if isinstance(state[0], (list, np.ndarray)):
            # 如果state是列表的列表，取最后一个非零元素
            for i in range(len(state) - 1, -1, -1):
                if isinstance(state[i], (list, np.ndarray)) and len(state[i]) >= 15:
                    if not np.all(np.array(state[i]) == 0):
                        return (float(state[i][11]), float(state[i][12]))
        else:
            # 直接是一维数组
            return (float(state[11]), float(state[12]))
    
    return None # 默认值

def extract_next_node_from_state(next_state):
    """
    从下一状态中提取节点位置（用于回访检测）
    
    Args:
        next_state: 下一状态 [state_size, feature_dim]
        
    Returns:
        (x, y) 或 None
    """
    try:
        # 找到最后一个非零状态
        last_idx = -1
        for i in range(len(next_state) - 1, -1, -1):
            if not np.all(next_state[i] == 0):
                last_idx = i
                break
        
        if last_idx >= 0:
            # 🔥 新格式检测：9维 = 历史序列，无坐标
            if len(next_state[last_idx]) == 9:
                return None  # 新格式无坐标信息
            elif len(next_state[last_idx]) >= 15:
                # ⚠️ 旧格式（DEPRECATED）：15维包含坐标
                x = float(next_state[last_idx][13])
                y = float(next_state[last_idx][14])
                if not (np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y)):
                    return (x, y)
    except Exception:
        pass
    
    return None

def extract_recent_nodes_from_state(state, K=3):
    """
    从状态历史中提取最近K步访问的节点（用于回访检测）
    
    ⚠️ DEPRECATED: 此函数仅用于旧格式数据（15维坐标）
    新格式数据（9维历史序列）不包含坐标信息，返回空列表
    
    Args:
        state: 状态数组 [state_size, feature_dim]
        K: 最近K步
        
    Returns:
        list of (x, y) tuples
    """
    recent_nodes = []
    count = 0
    
    try:
        # 从后往前遍历，提取最近K个非零状态的位置
        for i in range(len(state) - 1, -1, -1):
            if not np.all(state[i] == 0):
                # 🔥 新格式检测：9维 = 历史序列，无坐标
                if len(state[i]) == 9:
                    return []  # 新格式无坐标信息
                elif len(state[i]) >= 15:
                    # ⚠️ 旧格式（DEPRECATED）：15维包含坐标
                    x = float(state[i][13])
                    y = float(state[i][14])
                    if not (np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y)):
                        recent_nodes.append((x, y))
                        count += 1
                        if count >= K:
                            break
    except Exception:
        pass
    
    return recent_nodes

def extract_recent_edge_ids_from_state(state, edge_id_map, K=3):
    """
    从状态历史中提取最近K步的edge_id列表（用于回环检测）
    
    注意：edge_id_map存储的是无向边，每条边可以双向通行。
    我们只需要记录最近K步走过的edge_id列表即可。
    
    Args:
        state: 状态数组 [state_size, feature_dim]
        edge_id_map: 边ID映射 {(start, end): edge_id}
        K: 最近K步
        
    Returns:
        recent_edge_ids: 最近K步的edge_id列表
    """
    recent_edge_ids = []
    count = 0
    
    try:
        # 从后往前遍历，提取最近K个非零状态的边ID
        for i in range(len(state) - 1, -1, -1):
            if not np.all(state[i] == 0):
                if len(state[i]) >= 4:
                    # 从状态中提取边信息：[start_x, start_y, end_x, end_y, ...]
                    edge_start = (float(state[i][0]), float(state[i][1]))
                    edge_end = (float(state[i][2]), float(state[i][3]))
                    
                    # 无向边：尝试两个方向查找edge_id
                    edge_key_forward = (edge_start, edge_end)
                    edge_key_reverse = (edge_end, edge_start)
                    
                    edge_id = edge_id_map.get(edge_key_forward, edge_id_map.get(edge_key_reverse, -1))
                    
                    if edge_id >= 0:
                        recent_edge_ids.append(edge_id)
                        count += 1
                        if count >= K:
                            break
    except Exception:
        pass
    
    return recent_edge_ids

def extract_position_from_state_for_shaping(state):
    """
    从状态中提取位置信息
    支持：1. 单个状态向量 [feature_dim]  2. 状态历史列表 [[state_size, feature_dim]]
    """
    try:
        if not isinstance(state, (list, np.ndarray)) or len(state) == 0:
            return None
        
        # 检查是否是单个状态向量
        is_single_vector = (len(state.shape) == 1 if hasattr(state, 'shape') 
                           else not isinstance(state[0], (list, np.ndarray)))
        
        if is_single_vector:
            if len(state) >= 15:
                # 🔥 使用固定索引13, 14获取当前位置（前15维的位置固定）
                x, y = float(state[13]), float(state[14])
                if not (np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y)):
                    return (x, y)
            return None
        
        # 状态历史列表：找到最后一个非零状态
        # 🔥 修复：使用固定索引13, 14，而不是负索引，因为text embeddings会改变总维度
        for i in range(len(state) - 1, -1, -1):
            if (isinstance(state[i], (list, np.ndarray)) and 
                len(state[i]) >= 15 and 
                not np.all(state[i] == 0)):
                # 状态格式（前15维固定）：[start_x, start_y, end_x, end_y, crossing, path_type, length, width, curb, origin_x, origin_y, dest_x, dest_y, cur_x, cur_y]
                # 索引：              0        1        2      3      4         5          6       7      8     9         10         11     12      13     14
                # cur_x, cur_y 在索引 13, 14
                x, y = float(state[i][13]), float(state[i][14])
                if not (np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y)):
                    return (x, y)
        
        return None
    except Exception:
        return None

def enhance_states_with_text_embeddings(
    states, G, edge_id_map, 
    text_embedding_loader=None, embedding_proj_weights=None, embedding_proj_biases=None,
    original_feature_dim=15, user_embedding=None, user_embeddings_dict=None, user_ids=None
):
    """
    增强状态数据，添加文本 embeddings 和用户 embeddings（批量优化版本）
    
    Args:
        states: 状态数组 [batch_size, state_size, original_feature_dim]
        G: 图对象
        edge_id_map: 边ID映射
        text_embedding_loader: TextEmbeddingLoader 实例（可选）
        embedding_proj_weights: 投影权重字典（可选）
        embedding_proj_biases: 投影偏置字典（可选）
        original_feature_dim: 原始特征维度（默认15）
        user_embedding: 单个用户统计 embedding（可选，单用户模式）
        user_embeddings_dict: 所有用户的embeddings字典（可选，多用户模式）
        user_ids: 每个样本对应的user_id列表 [batch_size]（多用户模式时必需）
    
    Returns:
        enhanced_states: 增强后的状态 [batch_size, state_size, enhanced_feature_dim]
    """
    if text_embedding_loader is None or embedding_proj_weights is None:
        # 如果没有启用文本 embeddings，只添加用户 embedding
        if user_embedding is not None:
            return add_user_embedding_to_states(states, user_embedding)
        elif user_embeddings_dict is not None and user_ids is not None:
            return add_user_embeddings_to_states_batch(states, user_embeddings_dict, user_ids)
        return states
    
    # 🔥 保持与输入数据一致的dtype，避免精度损失
    input_dtype = states.dtype if hasattr(states, 'dtype') else np.float64
    
    batch_size = states.shape[0]
    state_size = states.shape[1]
    
    # 🔥 性能优化：预先计算配置，避免重复计算
    use_edge = 'edge' in (embedding_proj_weights or {})
    use_node = 'node' in (embedding_proj_weights or {})
    use_goal = 'goal' in (embedding_proj_weights or {})
    # 获取投影维度（从任意一个存在的embedding类型获取，默认64）
    if embedding_proj_weights:
        if 'edge' in embedding_proj_weights:
            proj_dim = embedding_proj_weights['edge'].shape[1]
        elif 'node' in embedding_proj_weights:
            proj_dim = embedding_proj_weights['node'].shape[1]
        elif 'goal' in embedding_proj_weights:
            proj_dim = embedding_proj_weights['goal'].shape[1]
        else:
            proj_dim = 64
    else:
        proj_dim = 64
    
    text_dim = 0
    if use_edge:
        text_dim += proj_dim
    if use_node:
        text_dim += proj_dim
    if use_goal:
        text_dim += proj_dim
    
    enhanced_feature_dim = original_feature_dim + text_dim
    
    # 🔥 性能优化：预分配numpy数组，避免列表追加
    enhanced_states = np.zeros((batch_size, state_size, enhanced_feature_dim), dtype=input_dtype)
    
    # 🔥 性能优化：批量加载embeddings
    # 第一步：收集所有需要的keys
    edge_ids_to_load = []
    node_keys_to_load = []
    goal_keys_to_load = []
    frame_info = []  # 存储每个frame的信息，用于后续处理
    
    # 🔥 性能优化：缓存edge_id查找结果（避免重复图查找）
    edge_id_cache = {}
    
    for i in range(batch_size):
        for t in range(state_size):
            state_frame = states[i, t]  # [original_feature_dim]
            
            # 检查是否为零向量（padding）
            if np.all(state_frame == 0):
                frame_info.append((i, t, None))  # None表示padding
                continue
            
            # 从状态帧中提取信息
            # 状态格式（前15维）：[start_x, start_y, end_x, end_y, crossing, path_type, length, width, curb, origin_x, origin_y, dest_x, dest_y, cur_x, cur_y]
            # 索引：              0        1        2      3      4         5          6       7      8     9         10         11     12      13     14
            cur_node = (float(state_frame[13]), float(state_frame[14]))  # 当前节点 (索引 13, 14)
            dest_pos = (float(state_frame[11]), float(state_frame[12]))  # 目标位置 (索引 11, 12)
            
            # 提取 static_feature: [length, width, curb, crossing, path_type]
            if len(state_frame) >= 15:
                static_feature = [
                    float(state_frame[6]),   # length
                    float(state_frame[7]),   # width (obstacle_free_width)
                    float(state_frame[8]),   # curb (curb_height)
                    float(state_frame[4]),   # crossing
                    float(state_frame[5])    # path_type
                ]
            else:
                static_feature = [0.0, 0.0, 0.0, 0, 0]
            
            # 提取 target_edge
            target_edge = (
                (float(state_frame[0]), float(state_frame[1])),  # start
                (float(state_frame[2]), float(state_frame[3]))   # end
            )
            
            # 🔥 性能优化：使用缓存避免重复图查找
            edge_id = None
            if cur_node in edge_id_cache:
                edge_id = edge_id_cache[cur_node]
            elif cur_node in G:
                neighbors = list(G.neighbors(cur_node))
                if len(neighbors) > 0:
                    first_neighbor = neighbors[0]
                    if (cur_node, first_neighbor) in edge_id_map:
                        edge_id = edge_id_map[(cur_node, first_neighbor)]
                    elif (first_neighbor, cur_node) in edge_id_map:
                        edge_id = edge_id_map[(first_neighbor, cur_node)]
                edge_id_cache[cur_node] = edge_id  # 缓存结果
            
            # 收集需要加载的keys
            frame_idx = len(frame_info)
            if use_edge and edge_id is not None:
                edge_ids_to_load.append((frame_idx, edge_id))
            if use_node:
                coord_key = f"{cur_node[0]:.16f}_{cur_node[1]:.16f}"
                node_keys_to_load.append((frame_idx, coord_key))
            if use_goal and discretize_goal_context is not None:
                goal_key, _, _ = discretize_goal_context(cur_node, dest_pos)
                goal_keys_to_load.append((frame_idx, goal_key))
            
            frame_info.append((i, t, {
                'cur_node': cur_node,
                'dest_pos': dest_pos,
                'static_feature': static_feature,
                'target_edge': target_edge,
                'edge_id': edge_id,
                'state_frame': state_frame
            }))
    
    # 第二步：批量加载embeddings
    embedding_cache = {}  # 缓存已加载的embeddings
    
    if use_edge and edge_ids_to_load:
        unique_edge_ids = list(set([edge_id for _, edge_id in edge_ids_to_load]))
        edge_embeddings = text_embedding_loader.get_edge_embeddings_batch(unique_edge_ids, fallback_zero=True)
        for idx, edge_id in enumerate(unique_edge_ids):
            embedding_cache[f'edge:{edge_id}'] = edge_embeddings[idx]
    
    if use_node and node_keys_to_load:
        unique_node_keys = list(set([key for _, key in node_keys_to_load]))
        node_embeddings = text_embedding_loader.get_node_embeddings_batch(unique_node_keys, fallback_zero=True)
        for idx, key in enumerate(unique_node_keys):
            embedding_cache[f'node:{key}'] = node_embeddings[idx]
    
    if use_goal and goal_keys_to_load:
        unique_goal_keys = list(set([key for _, key in goal_keys_to_load]))
        goal_embeddings = text_embedding_loader.get_goal_embeddings_batch(unique_goal_keys, fallback_zero=True)
        for idx, key in enumerate(unique_goal_keys):
            embedding_cache[f'goal:{key}'] = goal_embeddings[idx]
    
    # 第三步：使用缓存的embeddings构建增强特征（向量化优化版本）
    # 🔥 性能优化：批量收集embeddings，然后批量投影
    valid_frames = [(frame_idx, i, t, info) for frame_idx, (i, t, info) in enumerate(frame_info) if info is not None]
    
    if not valid_frames:
        return enhanced_states
    
    # 收集所有需要投影的embeddings
    edge_embs_list = []
    node_embs_list = []
    goal_embs_list = []
    valid_indices = []
    
    for frame_idx, i, t, info in valid_frames:
        valid_indices.append((frame_idx, i, t))
        
        # 从缓存获取embeddings
        edge_emb = None
        node_emb = None
        goal_emb = None
        
        if use_edge and info['edge_id'] is not None:
            cache_key = f"edge:{info['edge_id']}"
            edge_emb = embedding_cache.get(cache_key)
        
        if use_node:
            coord_key = f"{info['cur_node'][0]:.16f}_{info['cur_node'][1]:.16f}"
            cache_key = f"node:{coord_key}"
            node_emb = embedding_cache.get(cache_key)
        
        if use_goal and discretize_goal_context is not None:
            goal_key, _, _ = discretize_goal_context(info['cur_node'], info['dest_pos'])
            cache_key = f"goal:{goal_key}"
            goal_emb = embedding_cache.get(cache_key)
        
        edge_embs_list.append(edge_emb if edge_emb is not None else np.zeros(text_embedding_loader.hidden_dim, dtype=input_dtype))
        node_embs_list.append(node_emb if node_emb is not None else np.zeros(text_embedding_loader.hidden_dim, dtype=input_dtype))
        goal_embs_list.append(goal_emb if goal_emb is not None else np.zeros(text_embedding_loader.hidden_dim, dtype=input_dtype))
    
    # 🔥 批量投影计算（向量化）
    num_valid = len(valid_frames)
    
    if use_edge and 'edge' in embedding_proj_weights:
        edge_embs_array = np.stack(edge_embs_list, axis=0)  # [num_valid, hidden_dim]
        edge_projs = np.dot(edge_embs_array, embedding_proj_weights['edge'])  # [num_valid, proj_dim]
        if embedding_proj_biases and 'edge' in embedding_proj_biases:
            edge_projs += embedding_proj_biases['edge']
        edge_projs = np.tanh(edge_projs)  # [num_valid, proj_dim]
    else:
        edge_projs = None
    
    if use_node and 'node' in embedding_proj_weights:
        node_embs_array = np.stack(node_embs_list, axis=0)  # [num_valid, hidden_dim]
        node_projs = np.dot(node_embs_array, embedding_proj_weights['node'])  # [num_valid, proj_dim]
        if embedding_proj_biases and 'node' in embedding_proj_biases:
            node_projs += embedding_proj_biases['node']
        node_projs = np.tanh(node_projs)  # [num_valid, proj_dim]
    else:
        node_projs = None
    
    if use_goal and 'goal' in embedding_proj_weights:
        goal_embs_array = np.stack(goal_embs_list, axis=0)  # [num_valid, hidden_dim]
        goal_projs = np.dot(goal_embs_array, embedding_proj_weights['goal'])  # [num_valid, proj_dim]
        if embedding_proj_biases and 'goal' in embedding_proj_biases:
            goal_projs += embedding_proj_biases['goal']
        goal_projs = np.tanh(goal_projs)  # [num_valid, proj_dim]
    else:
        goal_projs = None
    
    # 构建增强特征
    for idx, (frame_idx, i, t, info) in enumerate(valid_frames):
        state_frame = info['state_frame']
        
        # 基础特征（15维）
        base_feature = np.array([
            state_frame[0], state_frame[1],  # start_x, start_y
            state_frame[2], state_frame[3],  # end_x, end_y
            state_frame[4],  # crossing
            state_frame[5],  # path_type
            state_frame[6],  # length
            state_frame[7],  # width
            state_frame[8],  # curb
            state_frame[9], state_frame[10],  # origin_x, origin_y
            state_frame[11], state_frame[12],  # dest_x, dest_y
            state_frame[13], state_frame[14]  # cur_x, cur_y
        ], dtype=input_dtype)
        
        # 拼接投影后的embeddings
        parts = [base_feature]
        
        if edge_projs is not None:
            parts.append(edge_projs[idx])
        elif use_edge:
            parts.append(np.zeros(proj_dim, dtype=input_dtype))
        
        if node_projs is not None:
            parts.append(node_projs[idx])
        elif use_node:
            parts.append(np.zeros(proj_dim, dtype=input_dtype))
        
        if goal_projs is not None:
            parts.append(goal_projs[idx])
        elif use_goal:
            parts.append(np.zeros(proj_dim, dtype=input_dtype))
        
        enhanced_states[i, t] = np.concatenate(parts)
    
    # Add user embedding if provided
    if user_embedding is not None:
        # Single-user mode: use the same embedding for all samples
        enhanced_states = add_user_embedding_to_states(enhanced_states, user_embedding)
    elif user_embeddings_dict is not None and user_ids is not None:
        # Multi-user mode: use user-specific embeddings
        enhanced_states = add_user_embeddings_to_states_batch(enhanced_states, user_embeddings_dict, user_ids)
    
    return enhanced_states

def add_user_embedding_to_states(states, user_embedding):
    """
    Add user embedding to states by concatenating it to each state frame (single-user mode).
    
    Args:
        states: State array [batch_size, state_size, feature_dim]
        user_embedding: User statistics embedding [user_dim] or None
        
    Returns:
        enhanced_states: States with user embedding [batch_size, state_size, feature_dim + user_dim]
                        or original states if user_embedding is None
    """
    if user_embedding is None:
        return states
    
    batch_size, state_size, feature_dim = states.shape
    user_dim = len(user_embedding)
    
    # Preserve input dtype
    input_dtype = states.dtype if hasattr(states, 'dtype') else np.float64
    
    # Preallocate enhanced states array
    enhanced_feature_dim = feature_dim + user_dim
    enhanced_states = np.zeros((batch_size, state_size, enhanced_feature_dim), dtype=input_dtype)
    
    # Broadcast user embedding to all state frames
    user_emb_broadcast = user_embedding.astype(input_dtype)
    
    for i in range(batch_size):
        for t in range(state_size):
            state_frame = states[i, t]
            
            # Check if this is a padding frame (all zeros)
            if np.all(state_frame == 0):
                # Keep as zero (including user embedding part)
                continue
            
            # Concatenate base state with user embedding
            enhanced_states[i, t] = np.concatenate([state_frame, user_emb_broadcast])
    
    return enhanced_states

def add_user_embeddings_to_states_batch(states, user_embeddings_dict, user_ids):
    """
    Add user-specific embeddings to states based on user_ids (multi-user mode).
    
    Args:
        states: State array [batch_size, state_size, feature_dim]
        user_embeddings_dict: Dictionary of {user_id_str: user_embedding}
        user_ids: List of user_id strings for each sample [batch_size]
        
    Returns:
        enhanced_states: States with user-specific embeddings [batch_size, state_size, feature_dim + user_dim]
    """
    if user_embeddings_dict is None or user_ids is None:
        return states
    
    batch_size, state_size, feature_dim = states.shape
    
    # Get user_dim from first user's embedding
    first_user_emb = next(iter(user_embeddings_dict.values()))
    user_dim = len(first_user_emb)
    
    # Preserve input dtype
    input_dtype = states.dtype if hasattr(states, 'dtype') else np.float64
    
    # Preallocate enhanced states array
    enhanced_feature_dim = feature_dim + user_dim
    enhanced_states = np.zeros((batch_size, state_size, enhanced_feature_dim), dtype=input_dtype)
    
    # Process each sample with its corresponding user embedding
    for i in range(batch_size):
        user_id = user_ids[i] if i < len(user_ids) else None
        
        # 🔥 Convert user_id to string format if needed (e.g., 0 -> 'user_001')
        # Note: user_id in data is 0-indexed, but folder names are 1-indexed
        if user_id is not None:
            if isinstance(user_id, (int, np.integer)):
                # Convert 0-indexed to 1-indexed: 0 -> user_001, 1 -> user_002, etc.
                user_id_str = f'user_{user_id + 1:03d}'
            elif isinstance(user_id, str):
                # If already in format 'user_XXX', use as is
                if user_id.startswith('user_'):
                    user_id_str = user_id
                else:
                    # Try to parse as number
                    try:
                        user_num = int(user_id)
                        # Convert 0-indexed to 1-indexed
                        user_id_str = f'user_{user_num + 1:03d}'
                    except:
                        user_id_str = user_id
            else:
                user_id_str = None
        else:
            user_id_str = None
        
        # Get user embedding for this sample
        if user_id_str and user_id_str in user_embeddings_dict:
            user_emb = user_embeddings_dict[user_id_str].astype(input_dtype)
        else:
            # Use zero embedding if user_id not found
            user_emb = np.zeros(user_dim, dtype=input_dtype)
            if user_id_str:  # Only warn if user_id was provided but not found
                print(f"⚠️  Warning: User ID '{user_id_str}' not found in embeddings dict, using zero embedding")
        
        for t in range(state_size):
            state_frame = states[i, t]
            
            # Check if this is a padding frame (all zeros)
            if np.all(state_frame == 0):
                # Keep as zero (including user embedding part)
                continue
            
            # Concatenate base state with user-specific embedding
            enhanced_states[i, t] = np.concatenate([state_frame, user_emb])
    
    return enhanced_states

def compute_next_states_for_negative_samples(
    states, negative_actions, edge_id_map, G, state_size=15,
    text_embedding_loader=None, embedding_proj_weights=None, embedding_proj_biases=None,
    user_embedding=None
):
    """
    为负样本计算转移后的状态（简化版：不加载 embeddings）
    
    ⚠️ 新格式（9维历史序列）：直接返回当前状态的副本，不进行状态转移
    
    Args:
        states: 当前状态 [batch_size, state_size, feature_dim]
        negative_actions: 负样本动作 [batch_size, num_neg]
        edge_id_map: 边ID映射
        G: 图对象
        state_size: 状态大小
        text_embedding_loader: TextEmbeddingLoader 实例（可选）
        embedding_proj_weights: 投影权重字典（可选）
        embedding_proj_biases: 投影偏置字典（可选）
        
    Returns:
        next_states: 转移后的状态 [batch_size, num_neg, state_size, feature_dim]
    """
    batch_size = states.shape[0]
    num_neg = negative_actions.shape[1]
    input_feature_dim = states.shape[2]
    
    # 🔥 新格式检测：9维 = 历史序列，不进行状态转移
    if input_feature_dim == 9:
        # 新格式：直接返回当前状态的副本（负样本不需要状态转移）
        # [batch_size, num_neg, state_size, feature_dim]
        next_states = np.repeat(states[:, np.newaxis, :, :], num_neg, axis=1)
        return next_states
    
    # ⚠️ 旧格式（DEPRECATED）：使用坐标进行状态转移
    # 🔥 性能优化：不加载 embeddings 到 neighbor states，用零向量填充
    # Neighbor states 主要用于 Q 值计算，embeddings 增益有限但需要维度一致
    use_text_embeddings = (text_embedding_loader is not None and embedding_proj_weights is not None)

    # 计算预期特征维度
    expected_feature_dim = input_feature_dim
    if user_embedding is not None:
        expected_feature_dim += len(user_embedding)

    # 提取原始状态（前15维）
    if use_text_embeddings and input_feature_dim > 15:
        states_raw = states[:, :, :15]  # [batch_size, state_size, 15]
    else:
        states_raw = states
    
    next_states = []
    for i in range(batch_size):
        sample_next_states = []
        for j in range(num_neg):
            action_id = negative_actions[i, j]
            next_state_raw = compute_state_transition(
                states_raw[i:i+1], [action_id], edge_id_map, G,
                text_embedding_loader=None,  # 不加载 embeddings
                embedding_proj_weights=None,
                embedding_proj_biases=None,
                user_embedding=user_embedding
            )  # [1, state_size, feature_dim]

            # 如果计算出的next_state维度与预期不匹配，需要填充
            if next_state_raw.shape[-1] != expected_feature_dim:
                # 需要填充维度
                missing_dims = expected_feature_dim - next_state_raw.shape[-1]
                padding = np.zeros((1, state_size, missing_dims), dtype=next_state_raw.dtype)
                next_state = np.concatenate([next_state_raw, padding], axis=2)
            else:
                next_state = next_state_raw
            
            sample_next_states.append(next_state[0])
        
        next_states.append(sample_next_states)
    return np.array(next_states)

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import argparse
import pickle
import time
from utils.utility import *
from utils.logger import *
from utils.path_planning import *
from utils.rewards import *
from utils.evaluation import *
from utils.state_processing import *
from SASRecModules import *
from copy import deepcopy

# 🔥 文本 Embedding 集成支持
try:
    from text_embedding_loader import TextEmbeddingLoader
    from text_context_generator import discretize_goal_context
    TEXT_EMBEDDING_AVAILABLE = True
except ImportError as e:
    TextEmbeddingLoader = None
    discretize_goal_context = None
    TEXT_EMBEDDING_AVAILABLE = False
    _TEXT_EMBEDDING_IMPORT_ERROR = str(e)

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

def generate_neighbor_actions(positive_action_id, valid_edge_ids, max_neighbors=20):
    """
    生成邻居动作覆盖（路线A：邻居覆盖TD）
    
    策略：优先包含专家动作，然后尽可能多地包含其他有效邻居
    - 如果邻居数 ≤ max_neighbors：全部包含
    - 如果邻居数 > max_neighbors：优先采样（回退边、随机邻居）
    
    Args:
        positive_action_id: 专家动作ID（正样本）
        valid_edge_ids: 有效动作ID列表（所有合法邻居）
        max_neighbors: 最大邻居数量（防止计算爆炸）
        
    Returns:
        neighbor_ids: 邻居动作ID列表（包含专家动作）
    """
    # 确保专家动作在列表中
    if positive_action_id in valid_edge_ids:
        other_actions = [eid for eid in valid_edge_ids if eid != positive_action_id]
    else:
        # 专家动作可能无效（极少情况），只用有效邻居
        other_actions = list(valid_edge_ids)
        positive_action_id = None
    
    # 如果邻居总数不多，全部包含
    total_neighbors = len(other_actions) + (1 if positive_action_id is not None else 0)
    if total_neighbors <= max_neighbors:
        if positive_action_id is not None:
            return [positive_action_id] + other_actions
        else:
            return other_actions
    
    # 邻居太多，需要采样
    # 策略：保留专家动作 + 采样其他邻居
    num_to_sample = max_neighbors - (1 if positive_action_id is not None else 0)
    
    if len(other_actions) <= num_to_sample:
        sampled_others = other_actions
    else:
        # 随机采样（未来可优先采样回退边等）
        sampled_others = np.random.choice(other_actions, num_to_sample, replace=False).tolist()
    
    if positive_action_id is not None:
        return [positive_action_id] + sampled_others
    else:
        return sampled_others

def extract_action_ids(actions, edge_id_map):
    """
    Extract action_ids from action list
    
    Args:
        actions: Action list, may contain int or feature vector
        edge_id_map: Edge ID映射
        
    Returns:
        action_ids: Extracted action ID list
    """
    action_ids = []
    for act in actions:
        if isinstance(act, (int, np.integer)):
            action_ids.append(act)
        elif isinstance(act, list) and len(act) == 15:
            edge_id = get_edge_id(act, edge_id_map, True)
            action_ids.append(edge_id if edge_id >= 0 else 0)
        else:
            action_ids.append(0)
    return action_ids

def parse_args():
    parser = argparse.ArgumentParser(description="Run improved LLM4Rec with better reward design.")

    parser.add_argument('--epoch', type=int, default=120,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='../data',
                        help='data directory')
    parser.add_argument('--graph_id', type=str, default='default_graph',
                        help='Graph identifier for multi-map support (default: default_graph)')
    # MOE (Mixture of Experts) support
    parser.add_argument('--user_id', type=str, default=None,
                        help='User ID for MOE training (e.g., user_001). If None, use standard single-user training.')
    parser.add_argument('--moe_data_dir', type=str, default='examples/moe_data/processed',
                        help='MOE data root directory (default: examples/moe_data/processed)')
    parser.add_argument('--use_user_embedding', action='store_true',
                        help='Use user embedding in state representation. If --user_id is specified, load that user; otherwise load all users.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=128,
                        help='Number of hidden factors, i.e., embedding size.')
    
    # Reward parameters - Enhanced for better path planning
    parser.add_argument('--r_goal', type=float, default=3.0,
                        help='Reward for reaching the goal.')
    parser.add_argument('--r_step', type=float, default=-0.005,
                        help='penalty per step (time cost).')
    parser.add_argument('--r_backtrack', type=float, default=-0.08,
                        help='penalty for backtracking actions (reduced by 50% for balance).')
    parser.add_argument('--r_repeat_visit', type=float, default=-0.03,
                        help='penalty for revisiting recently visited nodes/edges (default: -0.03).')
    parser.add_argument('--neg', type=int, default=1,
                        help='Number of negative samples per positive sample.')
    
    # Learning parameters
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--discount', type=float, default=0.99,
                        help='Discount factor for RL (balanced for termination reward).')
    
    # RL parameters
    # parser.add_argument('--weight', type=float, default=1.0, 
    #                     help='weight for Q-learning loss in the total loss function.')
    parser.add_argument('--lr_2', type=float, default=0.0001,   
                        help='Learning rate for the second optimizer, which is used for the second loss function.')
    
    # Model architecture parameters
    parser.add_argument('--num_heads', default=8, type=int, help='number of heads (for SASRec)')
    parser.add_argument('--num_blocks', default=6, type=int, help='Number of blocks (for SASRec)')
    parser.add_argument('--dropout_rate', default=0.15, type=float, help='Dropout rate for regularization.')
    parser.add_argument('--max_candidates', type=int, default=20, help='Maximum candidates per step for candidate-based action selection') 
    
    # Training phase parameters - 两阶段训练策略
    parser.add_argument('--phase1_epochs', type=int, default=10,  # 第一阶段：纯SL训练的epoch数
                        help='Number of epochs for Phase 1: SL-only training.')
    parser.add_argument('--rl_weight_phase2', type=float, default=0.16,  # 🔥 急救：大幅降低RL权重
                        help='RL weight for Phase 2 (default: 0.08, emergency cooldown from 0.3 for stability).')
    parser.add_argument('--actor_rl_weight', type=float, default=0.1,  # Actor RL辅损权重
                        help='Actor RL auxiliary loss weight (λ_RL, default: 0.05 for Phase 2).')

    parser.add_argument('--rl_update_ratio', type=int, default=3,
                        help='SL:RL update ratio (e.g., 3 means update SL 3 times per 1 RL update for stability).')
    parser.add_argument('--resume_ckpt', type=str, default=None,
                        help='Path to checkpoint to resume training from (e.g. ../data/saved_model/v4_sl_only_10k_steps.ckpt)')
    parser.add_argument('--preload_datasets', action='store_true',
                        help='Preload and cache all datasets before training (recommended for faster evaluation)')
    
    # Epoch-based evaluation and logging (recommended for consistency across different data ratios)
    parser.add_argument('--log_frequency_epoch', type=float, default=0.5,
                        help='Log metrics every N epochs (e.g., 0.1, 0.5, 1). Will be converted to steps automatically based on dataset size.')
    parser.add_argument('--eval_frequency_epoch', type=float, default=1.0,
                        help='Evaluate every N epochs (e.g., 0.5, 1, 2). Will be converted to steps automatically based on dataset size.')
    
    # Early stopping parameters
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping based on validation metrics.')
    parser.add_argument('--early_stopping_metric', type=str, default='step_level_accuracy',
                        choices=['path_success_rate', 'step_level_accuracy', 'path_level_accuracy'],
                        help='Metric to monitor for early stopping (default: path_success_rate, which is exact match rate).')
    parser.add_argument('--early_stopping_patience', type=int, default=7,
                        help='Number of evaluations with no improvement before stopping (default: 7, which is ~7 epochs if eval_frequency_epoch=1).')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0001,
                        help='Minimum change in monitored metric to qualify as an improvement (default: 0.0001).')
    parser.add_argument('--early_stopping_start_epoch', type=int, default=10,
                        help='Start early stopping check after this many epochs (default: 10, to avoid stopping too early).')
    
    # DAgger-lite on-policy collection parameters
    parser.add_argument('--onpolicy_buffer_size', type=int, default=240000,
                        help='Size of on-policy ring buffer (O-bucket) for RL training. Set to ~200K to match Expert dataset size.')
    parser.add_argument('--onpolicy_collect_frequency', type=int, default=4000,
                        help='Frequency of on-policy data collection (steps). Reduced for "slow drip" strategy: collect more frequently with fewer samples per collection.')
    parser.add_argument('--onpolicy_collect_trajectories', type=int, default=4,
                        help='Number of trajectories to collect per on-policy collection. Reduced for "slow drip" strategy: collect 3-4 trajectories more frequently instead of 20 at once.')
    parser.add_argument('--onpolicy_mix_ratio', type=float, default=0.2,
                        help='Ratio of on-policy data in RL training batch (0.0-1.0). Fixed at 0.2, no gradual increase for stability.')
    parser.add_argument('--onpolicy_max_mix_ratio', type=float, default=0.2,
                        help='Maximum ratio of on-policy data in RL training batch (0.0-1.0). Set equal to initial for emergency cooldown.')
    parser.add_argument('--onpolicy_max_steps', type=int, default=40,
                        help='Maximum steps per on-policy rollout. Reduced to avoid excessive "wandering" samples that pollute the buffer.')
    parser.add_argument('--onpolicy_epsilon', type=float, default=0.3,
                        help='Epsilon for ε-greedy exploration during on-policy collection (0.0-1.0).')
    parser.add_argument('--precollect_buffer_ratio', type=float, default=1.0,
                        help='Target ratio of O-buffer to pre-fill with SL head before Phase-2 training (1.0 = 100%%).')
    parser.add_argument('--precollect_trajectories_per_batch', type=int, default=100,
                        help='Number of trajectories to collect per pre-collection batch.')
    parser.add_argument('--precollect_batch_size', type=int, default=84,
                        help='Number of pre-collection batches (total: batch_size * trajectories_per_batch ≈ 8400 trajectories ≈ 200K transitions).')
    parser.add_argument('--precollected_obuffer_path', type=str, default='precollected_obuffer_sl.df',
                        help='Path to save/load pre-collected O-buffer (relative to data_directory). If file exists, load it; otherwise collect and save.')
    
    # Training data sampling ratio
    parser.add_argument('--train_data_ratio', type=float, default=1.0,
                        help='Ratio of training data to use (0.1=10%%, 0.25=25%%, 0.5=50%%, 0.75=75%%, 1.0=100%%). Default: 1.0 (use all data).')
    
    # Two-Stage Training Configuration
    parser.add_argument('--stage1_save_model', action='store_true', default=True,
                        help='Save Stage-1 model')
    parser.add_argument('--stage1_model_path', type=str, default='sl_only.ckpt',
                        help='Path to save/load Stage-1 SL-only model.')
    
    parser.add_argument('--eval_only', action='store_true', default=False,
                        help='If set, skip training and only run final evaluation on test dataset. Requires --resume_ckpt or existing checkpoint.')
    parser.add_argument('--seed_phase2', type=int, default=42,
                        help='Random seed for Phase-2 reproducibility.')
    
    # Evaluation control
    parser.add_argument('--skip_initial_eval', action='store_true', default=False,
                        help='Skip initial evaluation (before training) to save time. Useful when initial evaluation data is already available.')
    
    # Top-T Sampling Evaluation Parameters
    parser.add_argument('--eval_num_trials', type=int, default=5,
                        help='Number of sampling trials (T) for evaluation. Each path is sampled T times with temperature sampling.')
    parser.add_argument('--eval_temperature', type=float, default=1.0,
                        help='Temperature for evaluation sampling. 0.0=greedy, 1.0=sample from distribution, >1.0=more random.')
    
    # 🔥 Text Embedding Integration Parameters
    parser.add_argument('--use_text_embeddings', action='store_true', default=False,
                        help='Enable text embeddings to enhance features (requires text_embedding_cache_dir).')
    parser.add_argument('--text_embedding_cache_dir', type=str, default='./data/text_embeddings_test',
                        help='Directory containing text embedding LMDB cache.')
    parser.add_argument('--embedding_proj_dim', type=int, default=64,
                        help='Projection dimension for text embeddings (default: 64).')
    parser.add_argument('--use_edge_embedding', action='store_true', default=False,  # 🔥 临时禁用edge embedding以提升性能
                        help='Use edge text embeddings (requires --use_text_embeddings).')
    parser.add_argument('--use_node_embedding', action='store_true', default=True,
                        help='Use node text embeddings (requires --use_text_embeddings).')
    parser.add_argument('--use_goal_embedding', action='store_true', default=True,
                        help='Use goal text embeddings (requires --use_text_embeddings).')
    

    return parser.parse_args()

class OnPolicyRingBuffer:
    """
    DAgger-lite on-policy ring buffer (O-bucket) for RL training.
    Stores transitions collected from the current RL policy.
    Enhanced with labeling (success/loop/timeout/stuck) and stratified sampling.
    """
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.buffer = []
        self.position = 0
        # Track edge usage for duplicate limiting
        self.edge_usage_count = {}  # {(state_hash, action_id): count}
        self.max_duplicate_per_edge = 3  # Max samples per (state, action) pair (reduced from 10 for better diversity in "slow drip" strategy)
        # Track collection batches for overlap calculation
        self.collection_history = []  # List of sets of transition hashes from recent collections
        # 🔥 Performance optimization: Index task_key -> list of indices for fast sampling
        self.task_key_index = {}  # {task_key: [indices]}
        
    def _hash_transition(self, transition):
        """Create a hash for a transition to detect duplicates (supports both schemas)."""
        try:
            # NEW SCHEMA: use cur_node_id + taken_edge_id
            if 'cur_node_id' in transition and 'taken_edge_id' in transition:
                return hash((transition['cur_node_id'], transition['taken_edge_id']))

            # OLD SCHEMA: use state + action
            state = transition.get('state', None)
            action = transition.get('action', None)
            if state is not None and action is not None:
                # Create a simple hash from state and action
                state_str = str(np.array(state).flatten()[:10])  # Use first 10 elements
                return hash((state_str, action))
        except:
            pass
        return hash(str(transition))
    
    def _is_edge_duplicate_exceeded(self, transition):
        """Check if this edge (state, action) pair has exceeded duplicate limit (supports both schemas)."""
        # NEW SCHEMA: use cur_node_id + taken_edge_id
        if 'cur_node_id' in transition and 'taken_edge_id' in transition:
            edge_key = (transition['cur_node_id'], transition['taken_edge_id'])
        else:
            # OLD SCHEMA: use state + action
            edge_key = (id(transition.get('state')), transition.get('action'))

        count = self.edge_usage_count.get(edge_key, 0)
        return count >= self.max_duplicate_per_edge
    
    def add(self, cur_node_id=None, goal_node_id=None, cand_edge_ids=None, taken_edge_id=None,
            bc_action_idx=None, next_node_id=None, reward=None, done=None, state=None,
            next_state=None, len_state=None, len_next_state=None, d_cur=None, d_next=None,
            delta_d=None, task_key=None, termination_type=None, edge_action=None):
        """
        Add a transition to the ring buffer (V2 schema for cross-map generalization).

        Supports both old and new transition schemas for backward compatibility.

        NEW SCHEMA (preferred):
            cur_node_id, goal_node_id, cand_edge_ids, taken_edge_id, bc_action_idx,
            next_node_id, reward, done, state, next_state, len_state, len_next_state,
            d_cur, d_next, delta_d, task_key, termination_type

        OLD SCHEMA (deprecated):
            state, action, reward, next_state, is_done, len_state, len_next_state,
            task_key, termination_type, edge_action
        """
        # Support both old and new schemas
        if cur_node_id is not None and taken_edge_id is not None:
            # NEW SCHEMA (V2)
            transition = {
                'cur_node_id': cur_node_id,
                'goal_node_id': goal_node_id,
                'cand_edge_ids': cand_edge_ids,
                'taken_edge_id': taken_edge_id,
                'bc_action_idx': bc_action_idx,  # Behavioral cloning action index
                'next_node_id': next_node_id,
                'reward': reward,
                'done': done,
                'state': state,
                'next_state': next_state,
                'len_state': len_state,
                'len_next_state': len_next_state,
                'd_cur': d_cur,
                'd_next': d_next,
                'delta_d': delta_d,
                'task_key': task_key,
                'termination_type': termination_type
            }

            # Use taken_edge_id for duplicate detection in new schema
            if edge_action is None and taken_edge_id is not None:
                edge_action = taken_edge_id

        else:
            # OLD SCHEMA (backward compatibility)
            transition = {
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'is_done': done if done is not None else is_done,
                'len_state': len_state,
                'len_next_state': len_next_state,
                'task_key': task_key,
                'termination_type': termination_type
            }

        # Check duplicate limit for same edge
        if edge_action is not None:
            edge_key = edge_action
            count = self.edge_usage_count.get(edge_key, 0)
            if count >= self.max_duplicate_per_edge:
                return False  # Skip if exceeded limit
            self.edge_usage_count[edge_key] = count + 1
        
        if len(self.buffer) < self.max_size:
            # Add new transition
            new_index = len(self.buffer)
            self.buffer.append(transition)
            # Update task_key index
            if task_key is not None:
                if task_key not in self.task_key_index:
                    self.task_key_index[task_key] = []
                self.task_key_index[task_key].append(new_index)
        else:
            # Remove old edge count when overwriting
            old_trans = self.buffer[self.position]
            if 'edge_action' in old_trans:
                old_key = old_trans['edge_action']
                if old_key in self.edge_usage_count:
                    self.edge_usage_count[old_key] = max(0, self.edge_usage_count[old_key] - 1)
            
            # Remove old transition from task_key index
            old_task_key = old_trans.get('task_key')
            if old_task_key is not None and old_task_key in self.task_key_index:
                try:
                    self.task_key_index[old_task_key].remove(self.position)
                    if len(self.task_key_index[old_task_key]) == 0:
                        del self.task_key_index[old_task_key]
                except ValueError:
                    pass  # Index might have been removed already
            
            # Add new transition at overwritten position
            self.buffer[self.position] = transition
            # Update task_key index for new transition
            if task_key is not None:
                if task_key not in self.task_key_index:
                    self.task_key_index[task_key] = []
                if self.position not in self.task_key_index[task_key]:
                    self.task_key_index[task_key].append(self.position)
            
            self.position = (self.position + 1) % self.max_size
        
        return True
    
    def sample(self, n, stratified=False, success_ratio=0.35):
        """
        🔥 智能三层分层采样（急救策略）
        
        Args:
            stratified: If True, use 3-tier stratified sampling
            success_ratio: Ratio of success/near-goal samples (default 0.35 for 35%)
        
        三层分层策略：
        - 第1层（成功/近终点）：30-40% （优质学习目标）
        - 第2层（正进展）：30% （学习正向探索）
        - 第3层（失败/loop）：30-40% （学习边界）
        
        保底约束：确保每batch至少有K_succ成功样本和K_pos正进展样本
        """
        if len(self.buffer) == 0:
            return []
        
        n = min(n, len(self.buffer))
        
        if not stratified:
            indices = np.random.choice(len(self.buffer), n, replace=False)
            return [self.buffer[i] for i in indices]
        
        # 🔥 三层分类
        def _safe_get_reward(t):
            """Safely get reward value, handling tuples and other types."""
            reward = t.get('reward', 0)
            if isinstance(reward, (tuple, list)):
                reward = reward[0] if len(reward) > 0 else 0
            try:
                return float(reward)
            except (ValueError, TypeError):
                return 0.0
        
        # 第1层：成功/近终点（到终点阈值内）
        tier1_success = [i for i, t in enumerate(self.buffer) 
                        if t.get('termination_type') == 'success' or 
                        (t.get('is_done') and _safe_get_reward(t) > 1.0)]
        
        # 第2层：正进展（distance_improvement > 0，但未到终点）
        tier2_progress = [i for i, t in enumerate(self.buffer) 
                         if i not in tier1_success and 
                         _safe_get_reward(t) > 0.01]  # 正奖励表示有进展
        
        # 第3层：其余（loop/timeout/无进展）
        tier3_failure = [i for i in range(len(self.buffer)) 
                        if i not in tier1_success and i not in tier2_progress]
        
        # 🔥 三层采样比例
        n_tier1 = int(n * success_ratio)  # 35%成功
        n_tier2 = int(n * 0.30)  # 30%正进展
        n_tier3 = n - n_tier1 - n_tier2  # 35%失败/边界
        
        # 保底约束：至少K_succ成功和K_pos正进展
        K_succ = max(2, int(n * 0.1))  # 至少10%或2个
        K_pos = max(3, int(n * 0.15))  # 至少15%或3个
        
        n_tier1 = max(K_succ, min(n_tier1, len(tier1_success)))
        n_tier2 = max(K_pos, min(n_tier2, len(tier2_progress)))
        n_tier3 = min(n_tier3, len(tier3_failure))
        
        # 如果某层不足，重新分配到其他层
        if n_tier1 < K_succ and len(tier1_success) < K_succ:
            deficit = K_succ - len(tier1_success)
            n_tier1 = len(tier1_success)
            n_tier2 += deficit // 2
            n_tier3 += deficit - deficit // 2
        
        if n_tier2 < K_pos and len(tier2_progress) < K_pos:
            deficit = K_pos - len(tier2_progress)
            n_tier2 = len(tier2_progress)
            n_tier3 += deficit
        
        # 确保总和为n
        total = n_tier1 + n_tier2 + n_tier3
        if total < n:
            n_tier3 += (n - total)
        elif total > n:
            n_tier3 = max(0, n_tier3 - (total - n))
        
        # Sample from each tier
        sampled_indices = []
        if n_tier1 > 0 and len(tier1_success) > 0:
            tier1_indices = np.random.choice(tier1_success, min(n_tier1, len(tier1_success)), replace=False)
            sampled_indices.extend(tier1_indices)
        
        if n_tier2 > 0 and len(tier2_progress) > 0:
            tier2_indices = np.random.choice(tier2_progress, min(n_tier2, len(tier2_progress)), replace=False)
            sampled_indices.extend(tier2_indices)
        
        if n_tier3 > 0 and len(tier3_failure) > 0:
            tier3_indices = np.random.choice(tier3_failure, min(n_tier3, len(tier3_failure)), replace=False)
            sampled_indices.extend(tier3_indices)
        
        # If we need more samples, fill from remaining
        remaining = n - len(sampled_indices)
        if remaining > 0:
            remaining_indices = [i for i in range(len(self.buffer)) if i not in sampled_indices]
            if len(remaining_indices) > 0:
                additional = np.random.choice(remaining_indices, min(remaining, len(remaining_indices)), replace=False)
                sampled_indices.extend(additional)
        
        return [self.buffer[i] for i in sampled_indices]
    
    def sample_by_task(self, n, task_key, stratified=False, success_ratio=0.6):
        """
        Sample n transitions from the buffer filtered by task_key.
        
        🔥 Performance optimized: Uses task_key_index instead of full buffer scan.
        
        Args:
            stratified: If True, use stratified sampling
            success_ratio: Ratio of success samples in stratified mode
        """
        if len(self.buffer) == 0:
            return []
        
        # 🔥 Performance optimization: Use index instead of scanning entire buffer
        if task_key in self.task_key_index:
            indices = self.task_key_index[task_key]
            # Filter out invalid indices (in case buffer was overwritten in ring buffer mode)
            valid_indices = [i for i in indices if i < len(self.buffer) and self.buffer[i].get('task_key') == task_key]
        else:
            # Fallback: scan buffer if index is missing (shouldn't happen, but safe)
            valid_indices = [i for i, t in enumerate(self.buffer) if t.get('task_key') == task_key]
        
        if len(valid_indices) == 0:
            return []
        
        n = min(n, len(valid_indices))
        
        # Get transitions for filtered indices
        filtered_transitions = [(i, self.buffer[i]) for i in valid_indices]
        
        if not stratified:
            sampled_indices = np.random.choice(valid_indices, n, replace=False)
            return [self.buffer[i] for i in sampled_indices]
        
        def _safe_get_reward(t):
            """Safely get reward value, handling tuples and other types."""
            reward = t.get('reward', 0)
            if isinstance(reward, (tuple, list)):
                reward = reward[0] if len(reward) > 0 else 0
            try:
                return float(reward)
            except (ValueError, TypeError):
                return 0.0
        
        # Stratified sampling
        success_transitions = [(i, t) for i, t in filtered_transitions
                              if t.get('termination_type') == 'success' or 
                              (t.get('is_done') and _safe_get_reward(t) > 1.0)]
        non_success_transitions = [(i, t) for i, t in filtered_transitions 
                                   if (i, t) not in success_transitions]
        
        n_success = int(n * success_ratio)
        n_non_success = n - n_success
        
        n_success = min(n_success, len(success_transitions))
        n_non_success = min(n_non_success, len(non_success_transitions))
        
        sampled_indices = []
        if n_success > 0 and len(success_transitions) > 0:
            success_indices = np.random.choice([i for i, t in success_transitions], n_success, replace=False)
            sampled_indices.extend(success_indices)
        
        if n_non_success > 0 and len(non_success_transitions) > 0:
            non_success_indices = np.random.choice([i for i, t in non_success_transitions], n_non_success, replace=False)
            sampled_indices.extend(non_success_indices)
        
        # Fill remaining
        remaining = n - len(sampled_indices)
        if remaining > 0:
            remaining_indices = [i for i in valid_indices if i not in sampled_indices]
            if len(remaining_indices) > 0:
                additional = np.random.choice(remaining_indices, min(remaining, len(remaining_indices)), replace=False)
                sampled_indices.extend(additional)
        
        return [self.buffer[i] for i in sampled_indices]
    
    def get_composition_snapshot(self):
        """Get a snapshot of buffer composition."""
        if len(self.buffer) == 0:
            return {
                'total': 0,
                'success': 0,
                'non_success': 0,
                'success_ratio': 0.0,
                'termination_types': {}
            }
        
        def _safe_get_reward(t):
            """Safely get reward value, handling tuples and other types."""
            reward = t.get('reward', 0)
            if isinstance(reward, (tuple, list)):
                reward = reward[0] if len(reward) > 0 else 0
            try:
                return float(reward)
            except (ValueError, TypeError):
                return 0.0
        
        success_count = sum(1 for t in self.buffer 
                           if t.get('termination_type') == 'success' or 
                           (t.get('is_done') and _safe_get_reward(t) > 1.0))
        non_success_count = len(self.buffer) - success_count
        
        termination_types = {}
        for t in self.buffer:
            term_type = t.get('termination_type', 'unknown')
            termination_types[term_type] = termination_types.get(term_type, 0) + 1
        
        return {
            'total': len(self.buffer),
            'success': success_count,
            'non_success': non_success_count,
            'success_ratio': success_count / len(self.buffer) if len(self.buffer) > 0 else 0.0,
            'termination_types': termination_types
        }
    
    def record_collection_batch(self, transitions):
        """Record a batch of collected transitions for overlap calculation."""
        batch_hashes = set(self._hash_transition(t) for t in transitions)
        self.collection_history.append(batch_hashes)
        # Keep only last 3 collections
        if len(self.collection_history) > 3:
            self.collection_history.pop(0)
    
    def calculate_collection_overlap(self):
        """Calculate overlap rate among recent 3 collections."""
        if len(self.collection_history) < 2:
            return 0.0
        
        # Calculate pairwise overlap
        overlaps = []
        for i in range(len(self.collection_history)):
            for j in range(i + 1, len(self.collection_history)):
                set1 = self.collection_history[i]
                set2 = self.collection_history[j]
                if len(set1) > 0 and len(set2) > 0:
                    overlap = len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0.0
                    overlaps.append(overlap)
        
        return np.mean(overlaps) if len(overlaps) > 0 else 0.0
    
    def size(self):
        """Return current buffer size."""
        return len(self.buffer)
    
    def is_empty(self):
        """Check if buffer is empty."""
        return len(self.buffer) == 0
    
    def to_dataframe(self):
        """
        Convert buffer to pandas DataFrame for saving.
        Returns DataFrame with columns: state, action, reward, next_state, is_done, 
                                       len_state, len_next_state, task_key, termination_type
        """
        if len(self.buffer) == 0:
            return pd.DataFrame(columns=['state', 'action', 'reward', 'next_state', 'is_done', 
                                        'len_state', 'len_next_state', 'task_key', 'termination_type'])
        
        data_list = []
        for trans in self.buffer:
            data_list.append({
                'state': trans['state'],
                'action': trans['action'],
                'reward': trans['reward'],
                'next_state': trans['next_state'],
                'is_done': trans['is_done'],
                'len_state': trans['len_state'],
                'len_next_state': trans['len_next_state'],
                'task_key': trans.get('task_key', None),
                'termination_type': trans.get('termination_type', None)
            })
        
        return pd.DataFrame(data_list)
    
    def save_to_file(self, file_path):
        """Save buffer to pickle file (DataFrame format, compatible with replay_buffer.df)."""
        df = self.to_dataframe()
        df.to_pickle(file_path)
        print(f"✅ O-buffer saved to {file_path} ({len(df)} transitions)")
        return df
    
    def load_from_file(self, file_path, max_size=None):
        """
        Load buffer from pickle file.
        
        Args:
            file_path: Path to pickle file
            max_size: Maximum buffer size (if None, uses current max_size)
        """
        if not os.path.exists(file_path):
            print(f"⚠️  O-buffer file not found: {file_path}")
            return False
        
        try:
            df = pd.read_pickle(file_path)
            
            if max_size is not None:
                self.max_size = max_size
            
            # Clear current buffer
            self.buffer = []
            self.position = 0
            self.edge_usage_count = {}
            self.task_key_index = {}  # Clear task_key index
            
            # Load transitions from DataFrame
            loaded_count = 0
            for _, row in df.iterrows():
                transition = {
                    'state': row['state'],
                    'action': row['action'],
                    'reward': row['reward'],
                    'next_state': row['next_state'],
                    'is_done': row['is_done'],
                    'len_state': row['len_state'],
                    'len_next_state': row['len_next_state'],
                    'task_key': row.get('task_key', None),
                    'termination_type': row.get('termination_type', None)
                }
                
                # Add to buffer (respect max_size)
                if len(self.buffer) < self.max_size:
                    # Add to buffer
                    self.buffer.append(transition)
                    # 🔥 Rebuild task_key index for fast sampling
                    task_key = transition.get('task_key')
                    if task_key is not None:
                        if task_key not in self.task_key_index:
                            self.task_key_index[task_key] = []
                        self.task_key_index[task_key].append(len(self.buffer) - 1)
                    loaded_count += 1
                else:
                    break  # Stop if buffer is full
            
            print(f"✅ O-buffer loaded from {file_path} ({loaded_count}/{len(df)} transitions loaded, buffer_size={self.max_size})")
            print(f"   🔥 Task_key index rebuilt: {len(self.task_key_index)} unique task_keys")
            
            # Rebuild edge_usage_count from loaded data (simplified, will rebuild during training)
            # Note: edge_usage_count will be rebuilt naturally as we add new samples
            
            return True
        except Exception as e:
            print(f"❌ Failed to load O-buffer from {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return False

class ImprovedQNetwork:
    def __init__(self, hidden_size, learning_rate, feature_dim, item_num, state_size, dropout_rate, num_heads, num_blocks, lr_2, neg=2, max_candidates=20, cand_feature_dim=12, name='ImprovedDQNetwork'):
        """
        ImprovedQNetwork V2: Support candidate-based action selection.

        Args:
            max_candidates: Maximum number of candidates per step (for padding)
            cand_feature_dim: Dimension of candidate edge features (default: 12)
        """
        tf.compat.v1.disable_eager_execution()
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.feature_dim = int(feature_dim)
        self.cand_feature_dim = int(cand_feature_dim)  # 🔥 候选边特征维度

        # self.weight = weight
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        self.item_num = item_num  # Keep for backward compatibility
        self.max_candidates = max_candidates  # NEW: Max candidates per step
        self.neg = neg
        self.is_training = tf.compat.v1.placeholder(tf.bool, shape=())
        self.name = name
        self.lr_2 = lr_2


        with tf.compat.v1.variable_scope(self.name):
            self.all_embeddings=self.initialize_embeddings()
            self.inputs = tf.compat.v1.placeholder(tf.float32, [None, state_size, feature_dim])
            self.len_state = tf.compat.v1.placeholder(tf.int32, [None])

            # Feature preprocessing layer
            with tf.compat.v1.variable_scope('input_dense'):
                weights = tf.compat.v1.get_variable('weights', 
                    [self.feature_dim, self.hidden_size],
                    initializer=tf.compat.v1.glorot_normal_initializer())
                biases = tf.compat.v1.get_variable('biases', 
                    [self.hidden_size],
                    initializer=tf.compat.v1.zeros_initializer())
                self.input_emb = tf.nn.relu(tf.matmul(tf.reshape(self.inputs, [-1, self.feature_dim]), weights) + biases)
                self.input_emb = tf.reshape(self.input_emb, [-1, self.state_size, self.hidden_size])

            # SASRec model implementation
            pos_emb = tf.nn.embedding_lookup(self.all_embeddings['pos_embeddings'],
                                             tf.tile(tf.expand_dims(tf.range(self.state_size), 0),
                                                     [tf.shape(self.inputs)[0], 1]))
            self.seq = self.input_emb + pos_emb
            
            mask = tf.cast(tf.reduce_any(tf.not_equal(self.inputs, 0), axis=-1), tf.float32)
            mask = tf.expand_dims(mask, -1)
            
            self.seq = tf.cond(self.is_training,
                lambda: tf.nn.dropout(self.seq, rate=self.dropout_rate),
                lambda: self.seq)
            self.seq *= mask

            for i in range(self.num_blocks):
                with tf.compat.v1.variable_scope("num_blocks_%d" % i):
                    self.seq = multihead_attention(queries=normalize(self.seq),
                                                   keys=self.seq,
                                                   num_units=self.hidden_size,
                                                   num_heads=self.num_heads,
                                                   dropout_rate=self.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")

                    self.seq = feedforward(normalize(self.seq), num_units=[self.hidden_size, self.hidden_size],
                                           dropout_rate=self.dropout_rate,
                                           is_training=self.is_training,
                                           block_id=i)

                    self.seq *= mask

            self.seq = normalize(self.seq)
            self.states_hidden = extract_axis_1(self.seq, self.len_state - 1)
            
            # 🔥 RL去耦：创建两个版本的编码器输出
            # Q head使用stop_gradient版本，RL梯度不会反传到编码器
            # SL head使用正常版本，SL梯度正常反传
            self.states_hidden_for_q = tf.stop_gradient(self.states_hidden)   # RL不反传到编码器
            self.states_hidden_for_sl = self.states_hidden                     # SL正常反传

            # Output layers - Q头输出Q值（线性，无激活函数）
            with tf.compat.v1.variable_scope('output1'):
                weights1 = tf.compat.v1.get_variable('weights', 
                    [self.hidden_size, self.item_num],
                    initializer=tf.compat.v1.glorot_normal_initializer())
                biases1 = tf.compat.v1.get_variable('biases', 
                    [self.item_num],
                    initializer=tf.compat.v1.zeros_initializer())
                # 🔥 修复：Q值直接输出，不用sigmoid
                # 原因：TRFL需要Q值可以是任意实数（支持负reward）
                # 🔥 RL去耦：使用stop_gradient版本，RL梯度不反传到编码器
                self.output1 = tf.matmul(self.states_hidden_for_q, weights1) + biases1
            
            with tf.compat.v1.variable_scope('output2'):
                weights2 = tf.compat.v1.get_variable('weights', 
                    [self.hidden_size, self.item_num],
                    initializer=tf.compat.v1.glorot_normal_initializer())
                biases2 = tf.compat.v1.get_variable('biases', 
                    [self.item_num],
                    initializer=tf.compat.v1.zeros_initializer())
                # 🔥 SL head使用正常版本，梯度正常反传
                self.output2 = tf.matmul(self.states_hidden_for_sl, weights2) + biases2

            # NEW: Candidate-based action selection placeholders (must be defined before use)
            self.cand_features = tf.compat.v1.placeholder(tf.float32, [None, self.max_candidates, self.cand_feature_dim], name='cand_features')
            self.cand_mask = tf.compat.v1.placeholder(tf.float32, [None, self.max_candidates], name='cand_mask')

            # NEW: Candidate-based action selection outputs
            with tf.compat.v1.variable_scope('candidate_output'):
                # Candidate feature processing (simple linear transformation)
                cand_weights = tf.compat.v1.get_variable('cand_weights',
                    [self.cand_feature_dim, self.hidden_size],
                    initializer=tf.compat.v1.glorot_normal_initializer())
                cand_biases = tf.compat.v1.get_variable('cand_biases',
                    [self.hidden_size],
                    initializer=tf.compat.v1.zeros_initializer())

                # Process candidate features: [batch, max_candidates, cand_feature_dim] -> [batch, max_candidates, hidden_size]
                cand_processed = tf.nn.relu(tf.matmul(
                    tf.reshape(self.cand_features, [-1, self.cand_feature_dim]), cand_weights) + cand_biases)
                cand_processed = tf.reshape(cand_processed, [-1, self.max_candidates, self.hidden_size])

                # Combine state hidden with candidate features (simple concatenation approach)
                # state_hidden: [batch, hidden_size] -> [batch, 1, hidden_size] -> [batch, max_candidates, hidden_size]
                state_expanded = tf.expand_dims(self.states_hidden_for_q, 1)
                state_tiled = tf.tile(state_expanded, [1, self.max_candidates, 1])

                # Combine: [batch, max_candidates, 2*hidden_size]
                combined_features = tf.concat([state_tiled, cand_processed], axis=-1)

                # Final candidate scoring: [batch, max_candidates, 2*hidden_size] -> [batch, max_candidates]
                final_weights = tf.compat.v1.get_variable('final_weights',
                    [2 * self.hidden_size, 1],
                    initializer=tf.compat.v1.glorot_normal_initializer())
                final_biases = tf.compat.v1.get_variable('final_biases',
                    [1],
                    initializer=tf.compat.v1.zeros_initializer())

                self.output1_candidates = tf.squeeze(tf.matmul(
                    tf.reshape(combined_features, [-1, 2 * self.hidden_size]), final_weights) + final_biases)
                self.output1_candidates = tf.reshape(self.output1_candidates, [-1, self.max_candidates])

                # Apply candidate mask
                self.output1_masked = self.output1_candidates + (1.0 - self.cand_mask) * -1e9

            # Action mask placeholder (backward compatibility)
            self.action_mask = tf.compat.v1.placeholder(tf.float32, [None, item_num], name='action_mask')
            
            # Training phase placeholder
            self.training_phase = tf.compat.v1.placeholder(tf.int32, name='training_phase')
            
            # 🔥 强化Mask penalty：确保无效动作Q值足够低
            q_value_max = tf.reduce_max(self.output1, axis=1, keepdims=True)
            q_value_min = tf.reduce_min(self.output1, axis=1, keepdims=True)
            q_value_range = q_value_max - q_value_min
            
            # 🔥 修复：使用更强的mask_penalty，确保无效动作Q值远低于有效动作
            # 对于Q值范围[0,1]，使用-2.0确保无效动作Q值足够低
            pen_by_range = -2.0 * tf.stop_gradient(q_value_range)  # 更强惩罚
            pen_below_min = tf.stop_gradient(q_value_min) - 2.0     # 更强惩罚
            
            pen = tf.minimum(pen_by_range, pen_below_min)
            pen = tf.clip_by_value(pen, clip_value_min=-10.0, clip_value_max=-1.0)  # 更强范围
            mask_penalty = tf.stop_gradient(pen)
            
            self.mask_penalty_value = mask_penalty

            # Use action mask directly (no SL prior pruning)
            self.candidate_mask = tf.cast(self.action_mask, tf.bool)
            
            # Training and inference logits (simplified without SL prior)
            self.output1_for_training = self.output1 + mask_penalty * (1.0 - tf.cast(self.action_mask, tf.float32))
            self.output2_for_training = self.output2 + mask_penalty * (1.0 - tf.cast(self.action_mask, tf.float32))
            
            # 🔥 路A改造：添加SL先验融合
            # self.sl_prior_beta = tf.compat.v1.placeholder(tf.float32, name='sl_prior_beta')
            # self.use_sl_prior = tf.compat.v1.placeholder(tf.bool, name='use_sl_prior')
            
            # SL策略的对数概率
            # self.sl_log_probs = tf.nn.log_softmax(self.output2)
            
            # SL先验增强的Q值：Q' = Q + β*log(π_SL)
            # self.sl_enhanced_q = self.output1 + self.sl_prior_beta * self.sl_log_probs
            
            # 根据是否使用SL先验选择Q值
            self.q_for_action_selection = self.output1
            
            # For inference, use action mask directly
            self.output1_masked = self.output1 * tf.cast(self.action_mask, tf.float32) + mask_penalty * (1.0 - tf.cast(self.action_mask, tf.float32))
            self.output2_masked = self.output2 * tf.cast(self.action_mask, tf.float32) + mask_penalty * (1.0 - tf.cast(self.action_mask, tf.float32))
            
            # SL先验增强的Q值（用于动作选择）
            # self.q_for_action_selection_masked = self.q_for_action_selection * tf.cast(self.action_mask, tf.float32) + mask_penalty * (1.0 - tf.cast(self.action_mask, tf.float32))
            
            # Action inputs
            self.actions = tf.compat.v1.placeholder(tf.int32, [None])
            # Placeholders
            self.reward = tf.compat.v1.placeholder(tf.float32, [None])
            self.discount = tf.compat.v1.placeholder(tf.float32, [None])
            self.targetQs_ = tf.compat.v1.placeholder(tf.float32, [None, item_num])
            self.targetQs_selector = tf.compat.v1.placeholder(tf.float32, [None, item_num])
            # 🔥 Actor-Critic DDPG-style: Actor target probs placeholder
            self.actor_target_probs = tf.compat.v1.placeholder(tf.float32, [None, item_num], name='actor_target_probs')
            # 🔥 清理：删除未使用的target_Q_current相关placeholders
            
            # 🔥 Actor-Critic DDPG-style: Soft expectation target for Critic
            # V_tgt(s') = Σ π_tgt(a'|s') · Q_target(s', a')
            # y = r + γ * (1 - done) * V_tgt(s')
            
            # Compute soft expectation target: weighted sum over all valid actions
            # self.actor_target_probs: [batch, item_num] - π_tgt(a'|s')
            # self.targetQs_: [batch, item_num] - Q_target(s', a')
            soft_v_target = tf.reduce_sum(self.actor_target_probs * self.targetQs_, axis=1)  # [batch]
            
            # TD target: y = r + γ * V_tgt(s')
            td_target = self.reward + self.discount * soft_v_target
            td_target = tf.stop_gradient(td_target)
            
            # Current Q value: Q(s, a)
            batch_indices = tf.range(tf.shape(self.actions)[0])
            action_indices = tf.stack([batch_indices, self.actions], axis=1)
            current_q = tf.gather_nd(self.output1_for_training, action_indices)
            
            # Huber loss (same as TRFL for stability)
            td_error = td_target - current_q
            huber_loss = tf.compat.v1.losses.huber_loss(
                labels=td_target,
                predictions=current_q,
                reduction=tf.compat.v1.losses.Reduction.NONE
            )
            qloss_positive = huber_loss
            # 负样本TD学习：使用相同的软期望目标
            self.negative_actions = tf.compat.v1.placeholder(tf.int32, [None, None], name='negative_actions')
            self.negative_rewards = tf.compat.v1.placeholder(tf.float32, [None, None], name='negative_rewards')
            self.negative_target_Qs = tf.compat.v1.placeholder(tf.float32, [None, None, item_num], name='negative_target_Qs')
            self.negative_actor_target_probs = tf.compat.v1.placeholder(tf.float32, [None, None, item_num], name='negative_actor_target_probs')
            self.negative_loss_weight = tf.compat.v1.placeholder(tf.float32, name='negative_loss_weight')
            # 🔥 添加负样本损失掩码，用于处理填充位
            self.negative_loss_mask = tf.compat.v1.placeholder(tf.float32, [None, None], name='negative_loss_mask')
            
            # 使用固定负样本数量计算平均值
            qloss_negative = 0
            
            # 使用固定的负样本数量，在训练循环中处理动态数量
            for i in range(self.neg):
                negative = tf.gather(self.negative_actions, i, axis=1)
                neg_reward = tf.gather(self.negative_rewards, i, axis=1)
                neg_mask = tf.gather(self.negative_loss_mask, i, axis=1)
                
                # 🔥 Actor-Critic DDPG-style: Soft expectation target for negative samples
                neg_target_Q = tf.gather(self.negative_target_Qs, i, axis=1)  # [batch, item_num]
                neg_actor_probs = tf.gather(self.negative_actor_target_probs, i, axis=1)  # [batch, item_num]
                
                # V_tgt(s') = Σ π_tgt(a'|s') · Q_target(s', a')
                neg_soft_v_target = tf.reduce_sum(neg_actor_probs * neg_target_Q, axis=1)
                
                # TD target: y = r + γ * V_tgt(s')
                neg_td_target = neg_reward + self.discount * neg_soft_v_target
                neg_td_target = tf.stop_gradient(neg_td_target)
                
                # Current Q value
                neg_batch_indices = tf.range(tf.shape(negative)[0])
                neg_action_indices = tf.stack([neg_batch_indices, negative], axis=1)
                neg_current_q = tf.gather_nd(self.output1_for_training, neg_action_indices)
                
                # Huber loss
                neg_huber_loss = tf.compat.v1.losses.huber_loss(
                    labels=neg_td_target,
                    predictions=neg_current_q,
                    reduction=tf.compat.v1.losses.Reduction.NONE
                )
                
                # 🔥 应用损失掩码，对填充位置零损失
                masked_neg_loss = neg_huber_loss * neg_mask
                qloss_negative += masked_neg_loss
            
            # 🔥 使用掩码归一化，避免填充位影响平均值
            total_mask = tf.reduce_sum(self.negative_loss_mask)
            if self.neg > 0:
                self.negative_q_loss = qloss_negative / tf.maximum(total_mask, 1.0)
            else:
                self.negative_q_loss = 0.0
            
            # 🔥 Actor SL Loss (主损：监督学习，模仿专家)
            ce_loss_pre = tf.compat.v1.nn.sparse_softmax_cross_entropy_with_logits(labels=self.actions, logits=self.output2_for_training)
            self.ce_loss = tf.reduce_mean(ce_loss_pre)


            # 🔥 Actor-Critic DDPG-style: Actor RL auxiliary loss
            # L_RL = -E[Σ π(a|s) · Q(s,a)] (maximize expected Q, minimize negative expected Q)
            # Only computed on valid actions (action mask applied)
            
            # Compute actor policy: π(a|s)
            actor_probs = tf.nn.softmax(self.output2_for_training)  # [batch, item_num]
            
            # Expected Q under current actor policy: Σ π(a|s) · Q(s,a)
            # Only sum over valid actions (action_mask already applied to output)
            expected_q = tf.reduce_sum(actor_probs * self.output1_for_training, axis=1)  # [batch]
            
            # Actor RL loss: minimize negative expected Q (i.e., maximize expected Q)
            # Note: gradient only flows through actor (π), not critic (Q)
            expected_q_for_actor = tf.reduce_sum(
                actor_probs * tf.stop_gradient(self.output1_for_training), 
                axis=1
            )
            self.actor_rl_loss = -tf.reduce_mean(expected_q_for_actor)
            
            # 🔥 Critic Loss组件
            # 分离base loss和total loss以便监控
            self.q_loss_base = tf.reduce_mean(qloss_positive)
            # 🔥 总Q损失 = 主损失 + 负样本惩罚
            self.q_loss = self.q_loss_base + self.negative_loss_weight * self.negative_q_loss
            
            # 🔥 权重参数
            self.rl_weight = tf.compat.v1.placeholder(tf.float32, name='rl_weight')
            self.actor_rl_weight = tf.compat.v1.placeholder(tf.float32, name='actor_rl_weight')  # λ_RL for actor
            
            self.q_loss_weighted = self.rl_weight * self.q_loss
            self.actor_rl_loss_weighted = self.actor_rl_weight * self.actor_rl_loss

            # 🔥 Actor 总损失 = λ_SL · L_SL + λ_RL · L_RL
            self.actor_total_loss = self.ce_loss + self.actor_rl_loss_weighted
            
            # 三阶段训练损失定义
            phase1_sl_only_loss = self.ce_loss  # Phase 1: SL only
            phase2_sl_rl_loss = self.actor_total_loss + self.q_loss_weighted  # Phase 2+: Actor (SL+RL) + Critic
            
            self.balanced_loss = tf.case([
                (tf.equal(self.training_phase, 1), lambda: phase1_sl_only_loss),
                (tf.equal(self.training_phase, 2), lambda: phase2_sl_rl_loss),
                (tf.equal(self.training_phase, 3), lambda: phase2_sl_rl_loss)
            ], default=lambda: phase2_sl_rl_loss)
            
            self.phase1_loss = phase1_sl_only_loss
            self.phase2_loss = phase2_sl_rl_loss
            self.phase3_loss = phase2_sl_rl_loss
            
            # 分层梯度管理：分离变量组
            self.q_head_vars = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                scope=f"{self.name}/output1"
            ) + tf.compat.v1.get_collection(  # Include candidate output variables
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                scope=f"{self.name}/candidate_output"
            )
            self.sl_head_vars = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                scope=f"{self.name}/output2"
            )
            
            all_vars = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, 
                scope=self.name
            )
            self.shared_encoder_vars = [var for var in all_vars 
                                      if var not in self.q_head_vars and var not in self.sl_head_vars]
                       
            # 损失组件监控
            self.loss_components = {
                'total_loss': self.balanced_loss,
                'balanced_loss': self.balanced_loss,
                'phase1_sl_only_loss': phase1_sl_only_loss,
                'phase2_sl_rl_loss': phase2_sl_rl_loss,
                # 🔥 Actor losses
                'ce_loss': self.ce_loss,
                'actor_rl_loss': self.actor_rl_loss,
                'actor_rl_loss_weighted': self.actor_rl_loss_weighted,
                'actor_total_loss': self.actor_total_loss,
                'actor_rl_weight': self.actor_rl_weight,
                'expected_q': tf.reduce_mean(expected_q),
                # 🔥 Critic losses
                'q_loss': self.q_loss,
                'q_loss_base': self.q_loss_base,
                'negative_q_loss': self.negative_q_loss,
                'negative_loss_weight': self.negative_loss_weight,
                'q_loss_weighted': self.q_loss_weighted,
                'qloss_positive': tf.reduce_mean(qloss_positive),
                'td_error_mean': tf.reduce_mean(tf.abs(td_error)),
                'soft_v_target_mean': tf.reduce_mean(soft_v_target),
                # 🔥 General
                'training_phase': self.training_phase,
                'rl_weight': self.rl_weight,
                'mask_penalty': self.mask_penalty_value,
                'q_value_range': q_value_range,
                'q_value_min': q_value_min,
                'q_value_max': tf.reduce_max(self.output1_masked),
                'q_value_mean': tf.reduce_mean(self.output1_masked),
                'q_value_std': tf.math.reduce_std(self.output1_masked),
                'q_value_unmasked_max': tf.reduce_max(self.output1),
                'q_value_unmasked_min': tf.reduce_min(self.output1),
                'mask_penalty_gap': q_value_min - self.mask_penalty_value,
                'q_head_var_count': tf.constant(len(self.q_head_vars), dtype=tf.int32),
                'sl_head_var_count': tf.constant(len(self.sl_head_vars), dtype=tf.int32),
                'shared_encoder_var_count': tf.constant(len(self.shared_encoder_vars), dtype=tf.int32),
            }

            # 任务头专用优化器
            self.q_head_optimizer = tf.compat.v1.train.AdamOptimizer(self.lr_2).minimize(
                self.q_loss, var_list=self.q_head_vars
            )
            # 🔥 Actor optimizer: minimize actor_total_loss (SL + RL auxiliary)
            self.sl_head_optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(
                self.actor_total_loss, var_list=self.sl_head_vars
            )
            
            # 🔥 RL去耦：共享编码器只使用SL梯度更新
            # Q head已经使用了stop_gradient，所以Q梯度不会流回编码器
            # 因此不需要梯度合成，只使用SL梯度
            # 🔥 Actor gradient: use actor_total_loss (SL + RL auxiliary)
            self.sl_grads = tf.gradients(self.actor_total_loss, self.shared_encoder_vars)
            
            # 不再需要Q梯度和梯度合成（因为Q head已经被stop_gradient隔离）
            # self.q_grads = tf.gradients(self.q_loss, self.shared_encoder_vars)  # 已被stop_gradient阻断
            # self.combined_grads = ...  # 不再需要梯度合成
            
            # 共享编码器优化器：只使用SL梯度
            valid_sl_grads = [grad for grad in self.sl_grads if grad is not None]
            valid_sl_vars = [var for grad, var in zip(self.sl_grads, self.shared_encoder_vars) if grad is not None]
            
            if valid_sl_grads:
                clipped_sl_grads, _ = tf.clip_by_global_norm(valid_sl_grads, 5.0)
                self.shared_encoder_optimizer = tf.compat.v1.train.AdamOptimizer(self.lr_2).apply_gradients(
                    zip(clipped_sl_grads, valid_sl_vars)
                )
            else:
                self.shared_encoder_optimizer = tf.no_op()
            
            # 阶段性训练优化器
            sl_only_grads = tf.gradients(self.ce_loss, self.shared_encoder_vars)
            valid_sl_grads = [grad for grad in sl_only_grads if grad is not None]
            valid_sl_vars = [var for grad, var in zip(sl_only_grads, self.shared_encoder_vars) if grad is not None]
            
            if valid_sl_grads:
                clipped_sl_grads, _ = tf.clip_by_global_norm(valid_sl_grads, 5.0)
                self.phase1_shared_optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate).apply_gradients(
                    zip(clipped_sl_grads, valid_sl_vars)
                )
            else:
                self.phase1_shared_optimizer = tf.no_op()
            
            q_only_grads = tf.gradients(self.q_loss, self.shared_encoder_vars)
            valid_q_grads = [grad for grad in q_only_grads if grad is not None]
            valid_q_vars = [var for grad, var in zip(q_only_grads, self.shared_encoder_vars) if grad is not None]
            
            if valid_q_grads:
                clipped_q_grads, _ = tf.clip_by_global_norm(valid_q_grads, 5.0)
                self.phase2_shared_optimizer = tf.compat.v1.train.AdamOptimizer(self.lr_2).apply_gradients(
                    zip(clipped_q_grads, valid_q_vars)
                )
            else:
                self.phase2_shared_optimizer = tf.no_op()
            
            # 训练操作组合
            self.train_phase1 = tf.group(self.sl_head_optimizer, self.phase1_shared_optimizer)
            self.train_phase2_joint = tf.group(self.q_head_optimizer, self.sl_head_optimizer, self.shared_encoder_optimizer)
            
            # Probability output for evaluation
            self.probs = tf.nn.softmax(self.output2_masked)
            self.train_vars = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
            
            # 🔥 Actor-Critic DDPG-style: Create Actor Target network
            # Used to compute soft expectation target for Critic: V_tgt(s') = Σ π_tgt(a'|s') · Q_target(s', a')
            with tf.compat.v1.variable_scope('actor_target'):
                # Reuse the same architecture as output2 (Actor/SL head)
                actor_target_weights = tf.compat.v1.get_variable('weights', 
                    [self.hidden_size, self.item_num],
                    initializer=tf.compat.v1.glorot_normal_initializer())
                actor_target_biases = tf.compat.v1.get_variable('biases', 
                    [self.item_num],
                    initializer=tf.compat.v1.zeros_initializer())
                
                # Input: next state hidden features (will be fed during training)
                self.next_states_hidden = tf.compat.v1.placeholder(tf.float32, [None, self.hidden_size], name='next_states_hidden')
                self.actor_target_logits = tf.matmul(self.next_states_hidden, actor_target_weights) + actor_target_biases
                
                # Apply action mask to actor target
                self.next_action_mask = tf.compat.v1.placeholder(tf.float32, [None, item_num], name='next_action_mask')
                # 🔥 Use fixed penalty to avoid dependency on mask_penalty_value (which depends on whole graph)
                mask_penalty_target = -10.0  # Fixed large negative value for invalid actions
                self.actor_target_logits_masked = self.actor_target_logits + mask_penalty_target * (1.0 - tf.cast(self.next_action_mask, tf.float32))
                
                # Compute actor target policy output: π_tgt(a'|s')
                # Note: self.actor_target_probs is a placeholder (defined earlier) for training
                # This output is used to compute the values that will be fed to that placeholder
                self.actor_target_output = tf.nn.softmax(self.actor_target_logits_masked)
            
            # Get actor target variables for Polyak update
            self.actor_target_vars = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, 
                scope=f"{self.name}/actor_target"
            )
            self.actor_main_vars = self.sl_head_vars  # Main actor is output2 (SL head)
                
    def initialize_embeddings(self):
        all_embeddings = dict()
        pos_embeddings = tf.Variable(tf.random.normal([self.state_size, self.hidden_size], 0.0, 0.01),
                                        name='pos_embeddings')
        all_embeddings['pos_embeddings'] = pos_embeddings
        return all_embeddings

def predict_path_with_model(sess, model, start_state, target_pos, max_steps, G, edge_id_map, item_num, 
                            use_rl_head=False, arrival_threshold=1e-6,
                            text_embedding_loader=None, embedding_proj_weights=None, embedding_proj_biases=None,
                            user_id_for_model=None, user_embedding=None):
    """
    使用模型进行确定性贪心rollout预测路径
    
    Args:
        sess: TensorFlow session
        model: 模型
        start_state: 起始状态
        target_pos: 目标位置 (x, y)
        max_steps: 最大步数预算
        G: 图结构
        edge_id_map: 边ID映射
        item_num: 动作数量
        use_rl_head: 是否使用RL head（False时使用SL head）
        arrival_threshold: 到达阈值
        text_embedding_loader: TextEmbeddingLoader 实例（可选）
        embedding_proj_weights: 投影权重字典（可选）
        embedding_proj_biases: 投影偏置字典（可选）
        user_id_for_model: 用户ID（用于PersonalizedQNetwork）
        user_embedding: 用户embedding向量（可选）
    
    Returns:
        predicted_path: 预测的动作序列（动作ID列表）
        csv_origin: 起点坐标（用于可视化）
        csv_destination: 终点坐标（用于可视化）
        success: 是否成功到达
        trajectory: 状态轨迹
    """
    current_state = np.array(start_state).copy()
    predicted_path = []
    trajectory = [current_state.copy()]
    
    # 从start_state提取起点和终点
    csv_origin = None
    csv_destination = None
    try:
        if isinstance(current_state, np.ndarray) and current_state.ndim == 2:
            first_frame = current_state[0]
            if len(first_frame) >= 15:
                # 🔥 修复：使用固定索引（前15维），而不是负索引，因为text embeddings会改变总维度
                # 状态格式（前15维固定）：[start_x, start_y, end_x, end_y, crossing, path_type, length, width, curb, origin_x, origin_y, dest_x, dest_y, cur_x, cur_y]
                # 索引：              0        1        2      3      4         5          6       7      8     9         10         11     12      13     14
                # origin_x, origin_y 在索引 9, 10
                # dest_x, dest_y 在索引 11, 12
                csv_origin = (float(first_frame[9]), float(first_frame[10]))
                csv_destination = (float(first_frame[11]), float(first_frame[12]))
    except:
        pass
    
    current_len_state = effective_len(current_state)
    success = False
    
    for step in range(max_steps):
        # 生成动作掩码
        action_mask = generate_action_mask_batch([current_state], G, edge_id_map, item_num)[0]
        valid_actions = np.where(action_mask > 0)[0]
        
        if len(valid_actions) == 0:
            break
        
        # 创建feed_dict
        feed_dict = {
            model.inputs: [current_state],
            model.len_state: [current_len_state],
            model.action_mask: [action_mask],
            model.is_training: False,
            model.training_phase: 2 if use_rl_head else 1,
            model.rl_weight: 0.3 if use_rl_head else 0.0  # 🔥 使用降低的RL权重（默认0.3，可通过参数调整）
        }
        
        # 🔥 如果模型有user_id_ph占位符（PersonalizedQNetwork且为lora模式），需要提供user_id
        if hasattr(model, 'user_id_ph') and model.user_id_ph is not None:
            if user_id_for_model is not None:
                feed_dict[model.user_id_ph] = np.array([int(user_id_for_model)], dtype=np.int32)
            else:
                # 默认使用0
                feed_dict[model.user_id_ph] = np.array([0], dtype=np.int32)
        
        # 根据use_rl_head选择head
        if use_rl_head:
            # 使用RL head (Q值，贪心选择)
            q_values = sess.run(model.output1_masked, feed_dict=feed_dict)[0]
            action_id = int(np.argmax(q_values[valid_actions]))
            action_id = valid_actions[action_id]
        else:
            # 使用SL head (概率，贪心选择)
            probs = sess.run(model.probs, feed_dict=feed_dict)[0]
            action_id = int(np.argmax(probs[valid_actions]))
            action_id = valid_actions[action_id]
        
        predicted_path.append(action_id)
        
        # 执行动作，更新状态
        try:
            next_state = compute_state_transition([current_state], [action_id], edge_id_map, G,
                                                 text_embedding_loader=text_embedding_loader,
                                                 embedding_proj_weights=embedding_proj_weights,
                                                 embedding_proj_biases=embedding_proj_biases,
                                                 user_embedding=user_embedding)[0]
            current_state = next_state
            current_len_state = min(current_len_state + 1, model.state_size)
            trajectory.append(current_state.copy())
            
            # 检查是否到达目标
            current_pos = extract_position_from_state_for_shaping(current_state)
            if current_pos and target_pos:
                distance = np.sqrt((current_pos[0] - target_pos[0])**2 + 
                                 (current_pos[1] - target_pos[1])**2)
                if distance < arrival_threshold:
                    success = True
                    break
        except Exception as e:
            if len(predicted_path) <= 2:
                print(f"   ⚠️  Exception in rollout step {step}: {e}")
            break
    
    return predicted_path, csv_origin, csv_destination, success, trajectory

def determine_training_phase(global_step, args, load_stage1_model=False):
    """确定当前训练阶段和相关参数"""
    
    # 🔥 确保phase边界值是数字，避免tuple/float比较错误
    phase1_end = args.phase1_sl_only_steps
    
    # 🔥 如果已经加载了Stage-1模型，直接进入Phase-2
    if load_stage1_model:
        return 2, "Phase2-SL+RL-Joint", args.rl_weight_phase2, args.lr

    
    if global_step < phase1_end:
        # 第一阶段：纯SL训练
        current_phase = 1
        phase_name = "Phase1-SL-Only"
        rl_weight = 0.0
        current_lr = args.lr
        
    else:
        # 超出第一阶段，使用第二阶段设置
        current_phase = 2
        phase_name = "Phase2-SL+RL-Joint"
        rl_weight = args.rl_weight_phase2
        current_lr = args.lr_2
    
    return current_phase, phase_name, rl_weight, current_lr

def build_feed_dict(model, state, len_state, target_Qs, reward,
                   discount, action_ids, target_Qs_selector, action_masks,
                   current_phase, rl_weight, actor_target_probs=None,
                   negative_actions=None, negative_rewards=None,
                   negative_target_Qs=None, negative_actor_target_probs=None, negative_loss_weight=0.0,
                   negative_loss_mask=None, actor_rl_weight=0.1, is_training=True,
                   cand_features=None, cand_mask=None):
    """
    Build universal feed_dict
    
    Args:
        model: Model object
        state, len_state, target_Qs, reward, discount, action_ids, target_Qs_selector, action_masks: 训练数据
        current_phase: 训练阶段
        rl_weight: RL权重
        actor_target_probs: Actor target概率 π_tgt(a'|s') [batch_size, item_num] (🔥 Actor-Critic DDPG)
        negative_actions: 负样本动作 [batch_size, num_neg]
        negative_rewards: 负样本奖励 [batch_size, num_neg]
        negative_target_Qs: 负样本target Q值 [batch_size, num_neg, item_num]
        negative_actor_target_probs: 负样本Actor target概率 [batch_size, num_neg, item_num] (🔥 Actor-Critic DDPG)
        negative_loss_weight: 负样本损失权重
        actor_rl_weight: Actor RL辅损权重 λ_RL
        is_training: 是否训练模式
        user_labels: (deprecated) 用户标签（不再使用）
        
    Returns:
        feed_dict: Built feed_dict
    """
    feed_dict = {
        model.inputs: state,
        model.len_state: len_state,
        model.targetQs_: target_Qs,
        model.reward: reward,
        model.discount: discount,
        model.actions: action_ids,
        model.targetQs_selector: target_Qs_selector,
        model.action_mask: action_masks,
        model.is_training: is_training,
        model.training_phase: current_phase,
        model.rl_weight: rl_weight,
        model.actor_rl_weight: actor_rl_weight,  # 🔥 Actor-Critic DDPG
        model.negative_loss_weight: negative_loss_weight,
    }

    # NEW: Add candidate-based training support
    if cand_features is not None and hasattr(model, 'cand_features'):
        feed_dict[model.cand_features] = cand_features
        feed_dict[model.cand_mask] = cand_mask if cand_mask is not None else np.ones_like(cand_features[:, :, 0])
    
    # 🔥 Actor-Critic DDPG: Add actor_target_probs
    if actor_target_probs is not None:
        feed_dict[model.actor_target_probs] = actor_target_probs
    else:
        # Provide zero probs as placeholder
        batch_size = len(state)
        feed_dict[model.actor_target_probs] = np.zeros((batch_size, model.item_num), dtype=np.float32)
    
    # 添加负样本数据（如果提供）
    if negative_actions is not None and negative_rewards is not None and negative_target_Qs is not None:
        feed_dict[model.negative_actions] = negative_actions
        feed_dict[model.negative_rewards] = negative_rewards
        feed_dict[model.negative_target_Qs] = negative_target_Qs
        # 🔥 Actor-Critic DDPG: Add negative_actor_target_probs
        if negative_actor_target_probs is not None:
            feed_dict[model.negative_actor_target_probs] = negative_actor_target_probs
        else:
            # Provide zero probs as placeholder
            batch_size = negative_actions.shape[0] if len(negative_actions.shape) > 0 else 0
            num_neg = negative_actions.shape[1] if len(negative_actions.shape) > 1 else 0
            feed_dict[model.negative_actor_target_probs] = np.zeros((batch_size, num_neg, model.item_num), dtype=np.float32)
        
        # 🔥 添加损失掩码
        if negative_loss_mask is not None:
            feed_dict[model.negative_loss_mask] = negative_loss_mask
        else:
            # 如果没有提供掩码，创建全1掩码（向后兼容）
            batch_size = negative_actions.shape[0] if len(negative_actions.shape) > 0 else 0
            num_neg = negative_actions.shape[1] if len(negative_actions.shape) > 1 else 0
            feed_dict[model.negative_loss_mask] = np.ones((batch_size, num_neg), dtype=np.float32)
    else:
        # 提供空的占位符
        batch_size = len(state)
        feed_dict[model.negative_actions] = np.zeros((batch_size, 0), dtype=np.int32)
        feed_dict[model.negative_rewards] = np.zeros((batch_size, 0), dtype=np.float32)
        feed_dict[model.negative_target_Qs] = np.zeros((batch_size, 0, model.item_num), dtype=np.float32)
        feed_dict[model.negative_actor_target_probs] = np.zeros((batch_size, 0, model.item_num), dtype=np.float32)
        feed_dict[model.negative_loss_mask] = np.zeros((batch_size, 0), dtype=np.float32)


    return feed_dict

def preload_all_datasets(data_directory='../data', user_id=None, moe_data_dir='examples/moe_data/processed', edge_id_map=None, state_size=10, graph_id='default_graph'):
    """
    预加载所有数据集并缓存，用于提升后续评估性能
    
    Args:
        data_directory: Data directory for map/graph data
        user_id: User ID for MOE mode (default: None)
        moe_data_dir: MOE data root directory (default: 'examples/moe_data/processed')
        edge_id_map: Edge ID mapping (optional)
        state_size: State size (history length, default: 10)
        graph_id: Graph identifier (default: 'default_graph')
    """
    print("Preloading all datasets for caching...")
    
    # 如果没有提供edge_id_map，尝试加载
    if edge_id_map is None:
        try:
            raw_data_dir = os.path.join(data_directory, 'graph_data', graph_id, 'raw_data')
            data = load_map_data(raw_data_dir)
            df, meta_data = data["df"], data["meta_data"]
            G, edge_id_map, edge_feature_list = load_or_create_graph(df.copy(), data_directory, graph_id)
        except Exception as e:
            print(f"⚠️  Warning: Could not load edge_id_map: {e}")
            edge_id_map = None
    
    datasets = ['train', 'val', 'test']
    cache_dir = os.path.join(data_directory, 'evaluation_cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    for dataset in datasets:
        print(f"\nProcessing {dataset} dataset...")
        
        # 检查是否已有缓存
        # 🔥 修复：在cache文件名中包含user_id，避免不同用户数据混淆
        if user_id is not None:
            cache_file = os.path.join(cache_dir, f'{dataset}_batch_data_{user_id}.pkl')
        else:
            cache_file = os.path.join(cache_dir, f'{dataset}_batch_data.pkl')
        if os.path.exists(cache_file):
            continue
        
        # 构建数据
        dataset_files = {
            'train': 'sampled_train.df',
            'val': 'sampled_val.df', 
            'test': 'sampled_test.df'
        }
        
        data_file = dataset_files[dataset]
        # 支持MOE模式
        eval_data_path = get_moe_data_path(data_directory, user_id, moe_data_dir, data_file)
        if not os.path.exists(eval_data_path):
            print(f"⚠️  Skipping {dataset} dataset (file not found: {eval_data_path})")
            continue
        eval_sessions = pd.read_pickle(eval_data_path)
        eval_ids = eval_sessions.route_id.unique()
        
        print(f'Building data for {len(eval_ids)} paths...')
        
        # 加载 graph_cache 和 feature_builder（用于新格式数据）
        graph_cache = None
        feature_builder = None
        try:
            from utils.graph_cache import MultiGraphCache
            multi_cache = MultiGraphCache(os.path.join(data_directory, 'graph_data'))
            graph_cache = multi_cache.get_cache(graph_id)
            feature_builder = multi_cache.get_feature_builder(graph_id)
        except Exception as e:
            print(f'⚠️  Could not load GraphCache/FeatureBuilder: {e}')
        
        all_path_data = []
        all_path_actions = []
        all_path_len_states = []
        all_path_lengths = []
        all_path_user_ids = []
        all_cand_edges = []
        all_bc_indices = []
        
        for route_id in eval_ids:
            group = eval_sessions[eval_sessions['route_id'] == route_id]
            if len(group) == 0:
                continue
            
            path_states, path_actions, path_len_states, path_length, path_user_ids, path_cand_edges, path_bc_indices = build_path_data_for_batch(
                group, state_size=state_size, edge_id_map=edge_id_map,
                graph_cache=graph_cache, feature_builder=feature_builder
            )
            all_path_data.extend(path_states)
            all_path_actions.extend(path_actions)
            all_path_user_ids.extend(path_user_ids)
            all_cand_edges.extend(path_cand_edges)
            all_bc_indices.extend(path_bc_indices)
            all_path_len_states.extend(path_len_states)
            all_path_lengths.append(path_length)
        
        # 保存缓存
        cached_data = {
            'path_data': all_path_data,
            'path_actions': all_path_actions,
            'path_len_states': all_path_len_states,
            'path_lengths': all_path_lengths,
            'path_user_ids': all_path_user_ids,
            'cand_edges': all_cand_edges,
            'bc_indices': all_bc_indices,
            'eval_ids': eval_ids,
            'state_size': state_size
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cached_data, f)
        
        print(f'{dataset} dataset cached: {len(all_path_data)} states')
    
    print(f"\nAll datasets preloaded")
    print(f"Cache directory: {cache_dir}")

def precollect_obuffer_at_phase2_transition(sess, model, onpolicy_buffer, data_directory, args, user_id, moe_data_dir,
                                            G, edge_id_map, item_num, reward_goal, state_size, feature_dim, 
                                            model_feature_dim, text_embedding_loader=None, 
                                            embedding_proj_weights=None, embedding_proj_biases=None,
                                            precollected_obuffer_file=None, replay_buffer=None, user_id_for_model=None,
                                            user_embedding=None, user_embeddings_dict=None):
    """
    在Phase-2切换时执行O-buffer预收集
    
    Args:
        sess: TensorFlow session
        model: 模型对象
        onpolicy_buffer: On-policy buffer对象
        data_directory: 数据目录
        args: 参数对象
        user_id: 用户ID（MOE模式）
        moe_data_dir: MOE数据目录
        G: 图对象
        edge_id_map: 边ID映射
        item_num: 动作数量
        reward_goal: 奖励目标
        state_size: 状态大小
        feature_dim: 特征维度
        model_feature_dim: 模型特征维度
        text_embedding_loader: 文本embedding加载器（可选）
        embedding_proj_weights: Embedding投影权重（可选）
        embedding_proj_biases: Embedding投影偏置（可选）
        precollected_obuffer_file: 预收集文件保存路径（可选，如果为None则自动生成）
        replay_buffer: 已加载的replay buffer（可选，如果提供则使用它而不是从文件读取）
        user_embedding: 用户embedding（单用户模式，可选）
        user_embeddings_dict: 所有用户的embeddings字典（多用户模式，可选）
    
    Returns:
        bool: 是否成功执行预收集
    """
    try:
        print(f"\n🔥 Starting O-buffer pre-collection at Phase-2 transition...")
        print(f"   This may take a while, but will improve Phase-2 training quality")
        
        # 🔥 判断是否为多用户模式
        is_multi_user = (user_embeddings_dict is not None and len(user_embeddings_dict) > 0 and user_embedding is None)
        if is_multi_user:
            user_keys = list(user_embeddings_dict.keys())
            print(f"   📊 Multi-user mode detected: {len(user_embeddings_dict)} users")
            print(f"      Available user_ids: {user_keys}")
        elif user_embedding is not None:
            print(f"   📊 Single-user mode detected: using user embedding (dim={len(user_embedding)})")
        
        # 计算预收集目标大小
        onpolicy_buffer_size = args.onpolicy_buffer_size
        if isinstance(args.precollect_buffer_ratio, (tuple, list)):
            precollect_buffer_ratio = float(args.precollect_buffer_ratio[0]) if len(args.precollect_buffer_ratio) > 0 else 0.5
        else:
            try:
                precollect_buffer_ratio = float(args.precollect_buffer_ratio)
            except (ValueError, TypeError):
                precollect_buffer_ratio = 0.5  # 小样本时降低目标
        precollect_target_size = int(onpolicy_buffer_size * precollect_buffer_ratio)
        
        # 🔥 从训练数据中提取起终点对（优先使用已加载的replay_buffer，如果提供了的话）
        print(f"   📊 Extracting unique start-end pairs from training data...")
        
        if replay_buffer is not None:
            # 使用已加载的replay_buffer（可能已经采样过）
            train_replay_buffer = replay_buffer
            print(f"   📊 Using provided replay_buffer ({len(train_replay_buffer)} transitions)")
        else:
            # 从文件读取（兼容旧代码）
            train_replay_buffer_path = get_moe_data_path(data_directory, user_id, moe_data_dir, 'replay_buffer.df')
            if not os.path.exists(train_replay_buffer_path):
                print(f"   ⚠️  Training replay buffer not found: {train_replay_buffer_path}")
                print(f"   Will skip pre-collection and collect during training")
                return False
            train_replay_buffer = pd.read_pickle(train_replay_buffer_path)
            print(f"   📊 Loaded replay_buffer from file ({len(train_replay_buffer)} transitions)")
        
        # 🔥 辅助函数：将整数user_id转换为字符串格式
        def normalize_user_id(uid):
            """Convert user_id to string format: 0 -> 'user_001', 1 -> 'user_002', etc."""
            if isinstance(uid, int):
                return f'user_{uid+1:03d}'
            elif isinstance(uid, str):
                # If already in string format, return as is
                return uid
            else:
                return 'user_001'
        
        # 🔥 提取start-end pairs，并在多用户模式下保留user_id
        if is_multi_user:
            # 多用户模式：保存 (start, end, user_id) 元组
            start_end_pairs_with_user = []
            seen_pairs = set()
            for _, row in train_replay_buffer.iterrows():
                action = row['action']
                if isinstance(action, (list, np.ndarray)) and len(action) >= 15:
                    start_x, start_y = float(action[9]), float(action[10])
                    end_x, end_y = float(action[11]), float(action[12])
                    # 检查replay_buffer中是否有user_id列
                    if 'user_id' in row:
                        # 🔥 转换整数user_id为字符串格式
                        pair_user_id = normalize_user_id(row['user_id'])
                    else:
                        # 如果没有user_id列，使用传入的user_id参数
                        pair_user_id = normalize_user_id(user_id) if user_id is not None else 'user_001'
                    
                    pair_key = (start_x, start_y, end_x, end_y, pair_user_id)
                    if pair_key not in seen_pairs:
                        start_end_pairs_with_user.append(((start_x, start_y), (end_x, end_y), pair_user_id))
                        seen_pairs.add(pair_key)
            
            start_end_pairs = start_end_pairs_with_user
            print(f"   ✅ Found {len(start_end_pairs)} unique start-end-user pairs")
        else:
            # 单用户模式：只保存 (start, end)
            start_end_pairs = set()
            for _, row in train_replay_buffer.iterrows():
                action = row['action']
                if isinstance(action, (list, np.ndarray)) and len(action) >= 15:
                    start_x, start_y = float(action[9]), float(action[10])
                    end_x, end_y = float(action[11]), float(action[12])
                    start_end_pairs.add(((start_x, start_y), (end_x, end_y)))
            
            start_end_pairs = list(start_end_pairs)
            print(f"   ✅ Found {len(start_end_pairs)} unique start-end pairs")
        
        # 执行预收集（简化版：只收集部分数据，避免耗时过长）
        original_max_duplicate = onpolicy_buffer.max_duplicate_per_edge
        onpolicy_buffer.max_duplicate_per_edge = 1000
        
        budget_B = 60
        arrival_threshold_eval = 1e-6
        total_collected = 0
        total_added = 0
        successful_paths = 0
        failed_paths = 0
        
        # 对每个起终点对生成一条路径（收集所有）
        for pair_idx, pair_data in enumerate(start_end_pairs):
            if onpolicy_buffer.size() >= precollect_target_size:
                break
            
            try:
                # 🔥 解包pair数据（支持多用户和单用户模式）
                if is_multi_user:
                    start_pos, end_pos, pair_user_id = pair_data
                    # 获取该用户的embedding
                    current_user_embedding = user_embeddings_dict.get(pair_user_id, None)
                    if current_user_embedding is None:
                        # 如果找不到该用户的embedding，使用第一个用户的
                        first_key = next(iter(user_embeddings_dict.keys()))
                        current_user_embedding = user_embeddings_dict[first_key]
                        if pair_idx == 0:
                            print(f"   ⚠️  User '{pair_user_id}' not found in embeddings, using '{first_key}'")
                else:
                    start_pos, end_pos = pair_data
                    current_user_embedding = user_embedding
                
                # 构建初始状态
                from utils.utility import pad_history
                cur_x, cur_y = float(start_pos[0]), float(start_pos[1])
                initial_state_frame = [0.0] * 9 + [float(start_pos[0]), float(start_pos[1]), float(end_pos[0]), float(end_pos[1]), cur_x, cur_y]
                start_state = np.array(pad_history([initial_state_frame], state_size, [0.0] * 15))
                
                # 处理文本embeddings和用户embeddings
                # 🔥 Debug: Print state shape before enhancement
                if pair_idx == 0:
                    print(f"   🔍 Debug info for first pair:")
                    print(f"      - start_state.shape: {start_state.shape}")
                    print(f"      - feature_dim: {feature_dim}")
                    print(f"      - model_feature_dim: {model_feature_dim}")
                    print(f"      - args.use_text_embeddings: {args.use_text_embeddings}")
                    print(f"      - args.use_user_embedding: {args.use_user_embedding}")
                    print(f"      - is_multi_user: {is_multi_user}")
                    if is_multi_user:
                        print(f"      - pair_user_id: '{pair_user_id}' (converted from replay_buffer)")
                        print(f"      - current_user_embedding: {current_user_embedding is not None} (shape: {current_user_embedding.shape if current_user_embedding is not None else 'None'})")
                    else:
                        print(f"      - user_embedding: {current_user_embedding is not None} (shape: {current_user_embedding.shape if current_user_embedding is not None else 'None'})")
                
                if args.use_text_embeddings and text_embedding_loader is not None:
                    if start_state.shape[1] == feature_dim:
                        start_state = enhance_states_with_text_embeddings(
                            np.array([start_state]), G, edge_id_map,
                            text_embedding_loader=text_embedding_loader,
                            embedding_proj_weights=embedding_proj_weights,
                            embedding_proj_biases=embedding_proj_biases,
                            original_feature_dim=feature_dim,
                            user_embedding=current_user_embedding if args.use_user_embedding else None,
                            user_embeddings_dict=None  # 🔥 单个pair只用当前用户的embedding
                        )[0]
                        if pair_idx == 0:
                            print(f"      - After text enhancement: {start_state.shape}")
                elif args.use_user_embedding and current_user_embedding is not None:
                    # 如果只启用了用户embedding，没有文本embedding
                    start_state = add_user_embedding_to_states(np.array([start_state]), current_user_embedding)[0]
                    if pair_idx == 0:
                        print(f"      - After user embedding: {start_state.shape}")
                
                # 🔥 Validate state shape matches model expectation
                if pair_idx == 0 and start_state.shape[1] != model_feature_dim:
                    error_msg = (
                        f"\n❌ STATE SHAPE MISMATCH DETECTED!\n"
                        f"   Model expects: {model_feature_dim} features\n"
                        f"   State provides: {start_state.shape[1]} features\n"
                        f"   Difference: {model_feature_dim - start_state.shape[1]} features missing\n"
                        f"\n💡 Possible causes:\n"
                    )
                    if model_feature_dim - start_state.shape[1] == 11:
                        error_msg += (
                            f"   1. Model was trained WITH user embeddings (11 dims)\n"
                            f"   2. Current run is WITHOUT --use_user_embedding flag\n"
                            f"\n🔧 Solution: Add --use_user_embedding flag to your command\n"
                        )
                    else:
                        error_msg += (
                            f"   1. Model was trained with different feature configuration\n"
                            f"   2. Text embeddings or user embeddings mismatch\n"
                            f"\n🔧 Solution: Use same flags as when model was trained\n"
                        )
                    print(error_msg)
                    raise ValueError(f"State shape mismatch: {start_state.shape[1]} != {model_feature_dim}")
                
                # 生成路径
                predicted_path, _, _, path_success, trajectory = predict_path_with_model(
                    sess, model, start_state, end_pos,
                    max_steps=budget_B, G=G, edge_id_map=edge_id_map, item_num=item_num,
                    use_rl_head=False, arrival_threshold=arrival_threshold_eval,
                    text_embedding_loader=text_embedding_loader,
                    embedding_proj_weights=embedding_proj_weights,
                    embedding_proj_biases=embedding_proj_biases,
                    user_id_for_model=user_id_for_model,  # 🔥 传递user_id_for_model（在precollect函数中）
                    user_embedding=current_user_embedding  # 🔥 传递当前用户的embedding
                )
                
                if len(predicted_path) < 1:
                    failed_paths += 1
                    continue
                
                # 转换为transitions
                current_state = start_state.copy()
                current_len_state = effective_len(current_state)
                
                for step_idx, action_id in enumerate(predicted_path):
                    if step_idx >= len(trajectory) - 1:
                        break
                    
                    next_state = trajectory[step_idx + 1].copy()
                    next_len_state = effective_len(next_state)
                    is_done = (step_idx == len(predicted_path) - 1) or path_success
                    
                    reward = calculate_improved_reward(
                        action_id, is_done, reward_goal,
                        step_idx=step_idx, state_history=None, path_info=None,
                        prev_action=None, edge_id_map=edge_id_map,
                        step_penalty=-0.01, backtrack_penalty=-0.06,
                        is_failure=False, failure_type=None,
                        recent_visited_nodes=[], repeat_visit_penalty=-0.03
                    )
                    
                    termination_type = 'success' if (is_done and path_success) else ('timeout' if (step_idx == len(predicted_path) - 1 and not path_success) else 'normal')
                    task_key = f"{start_pos[0]}_{start_pos[1]}_{end_pos[0]}_{end_pos[1]}"
                    
                    added = onpolicy_buffer.add(
                        current_state.copy(), action_id, reward,
                        next_state.copy(), is_done,
                        current_len_state, next_len_state,
                        task_key=task_key, termination_type=termination_type,
                        edge_action=None
                    )
                    
                    if added:
                        total_added += 1
                    total_collected += 1
                    
                    current_state = next_state.copy()
                    current_len_state = next_len_state
                
                if path_success:
                    successful_paths += 1
                else:
                    failed_paths += 1
                
                if (pair_idx + 1) % 100 == 0 or onpolicy_buffer.size() >= precollect_target_size:
                    print(f"   Progress: {pair_idx + 1}/{len(start_end_pairs)} pairs, "
                          f"buffer: {onpolicy_buffer.size()}/{precollect_target_size} transitions, "
                          f"success: {successful_paths}, failed: {failed_paths}, "
                          f"added: {total_added}/{total_collected} transitions")
                    
                    # 每1000对保存一次checkpoint，避免丢失进度
                    if (pair_idx + 1) % 1000 == 0:
                        try:
                            if precollected_obuffer_file is None:
                                precollected_obuffer_file = get_moe_data_path(data_directory, user_id, moe_data_dir, args.precollected_obuffer_path)
                            checkpoint_file = precollected_obuffer_file.replace('.df', '_checkpoint.df')
                            onpolicy_buffer.save_to_file(checkpoint_file)
                            print(f"   💾 Checkpoint saved: {checkpoint_file} ({onpolicy_buffer.size()} transitions)")
                        except Exception as e:
                            print(f"   ⚠️  Failed to save checkpoint: {e}")
                
            except Exception as e:
                print(f"   ⚠️  Failed to generate path for pair {pair_idx}: {e}")
                failed_paths += 1
                continue
        
        onpolicy_buffer.max_duplicate_per_edge = original_max_duplicate
        
        # Print statistics
        snapshot = onpolicy_buffer.get_composition_snapshot()
        total_processed = successful_paths + failed_paths
        
        print(f"\n✅ Pre-collection complete!")
        print(f"   Processed: {total_processed}/{len(start_end_pairs)} start-end pairs")
        print(f"   Total collected: {total_collected} transitions")
        if total_collected > 0:
            print(f"   Total added: {total_added} transitions ({total_added/total_collected*100:.1f}% acceptance rate)")
        else:
            print(f"   Total added: {total_added} transitions")
        print(f"   Final buffer size: {onpolicy_buffer.size()}/{onpolicy_buffer_size}")
        print(f"   Successful paths: {successful_paths}, Failed paths: {failed_paths}")
        print(f"   Success ratio in buffer: {snapshot['success_ratio']:.2%}")
        print(f"   Termination types: {snapshot['termination_types']}")
        print(f"   🔥 Parameters (aligned with initial evaluation):")
        print(f"      - max_steps (budget_B): {budget_B}")
        print(f"      - arrival_threshold: {arrival_threshold_eval}")
        print(f"      - use_rl_head: False (SL head only)")
        
        # Save pre-collected O-buffer for future use
        try:
            if precollected_obuffer_file is None:
                # 使用新的路径组织方式（按阶段和用户，带时间戳）
                new_precollected_path = get_precollected_path(data_directory, user_id)
            else:
                new_precollected_path = precollected_obuffer_file
            onpolicy_buffer.save_to_file(new_precollected_path)
            print(f"   💾 Saved to {new_precollected_path} for future use")
            
            # 删除checkpoint文件（如果存在）
            checkpoint_file = new_precollected_path.replace('.df', '_checkpoint.df')
            if os.path.exists(checkpoint_file):
                try:
                    os.remove(checkpoint_file)
                    print(f"   🗑️  Removed checkpoint file: {checkpoint_file}")
                except:
                    pass
        except Exception as e:
            print(f"   ⚠️  Failed to save pre-collected O-buffer: {e}")
            import traceback
            traceback.print_exc()
            
            # 尝试保存checkpoint作为备份
            try:
                if precollected_obuffer_file is None:
                    precollected_obuffer_file = get_moe_data_path(data_directory, user_id, moe_data_dir, args.precollected_obuffer_path)
                checkpoint_file = precollected_obuffer_file.replace('.df', '_checkpoint.df')
                onpolicy_buffer.save_to_file(checkpoint_file)
                print(f"   💾 Saved checkpoint as backup: {checkpoint_file}")
            except:
                pass
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to pre-collect O-buffer: {e}")
        import traceback
        traceback.print_exc()
        return False

def collect_onpolicy_transitions(sess, model, replay_buffer, G, edge_id_map, item_num, 
                                 num_trajectories, max_steps, reward_goal, args,
                                 text_embedding_loader=None, embedding_proj_weights=None, embedding_proj_biases=None,
                                 user_embedding=None, user_embeddings_dict=None):
    """
    Collect on-policy transitions by rolling out the current RL policy.
    Returns a list of transitions for the on-policy buffer (O-bucket).
    
    Enhanced with stuck exploration mode:
    - No early termination for loop_stuck
    - Adaptive epsilon boost during stuck periods
    - Lightweight reward shaping for backtrack actions
    
    Args:
        sess: TensorFlow session
        model: Current RL policy model
        replay_buffer: Expert dataset to sample initial states from
        G: Graph structure
        edge_id_map: Edge ID mapping
        item_num: Number of actions
        num_trajectories: Number of trajectories to collect
        max_steps: Maximum steps per trajectory
        reward_goal: Goal reward
        args: Training arguments
        
    Returns:
        transitions: List of collected transitions
    """
    transitions = []
    
    # Sample starting points from expert dataset
    sampled_routes = replay_buffer.sample(n=num_trajectories)
    
    for route_idx, (_, route_data) in enumerate(sampled_routes.iterrows()):
        try:
            # Extract initial state from expert trajectory
            state = route_data['state']
            
            # 🔥 Extract user_id for multi-user mode
            current_user_id_raw = route_data.get('user_id', 0)
            # 🔥 Convert integer user_id to string format: 0 -> 'user_001', 1 -> 'user_002', etc.
            if isinstance(current_user_id_raw, int):
                current_user_id = f'user_{current_user_id_raw+1:03d}'
            else:
                current_user_id = current_user_id_raw if isinstance(current_user_id_raw, str) else 'user_001'
            
            # 🔥 Get user embedding for this trajectory
            if user_embeddings_dict is not None and len(user_embeddings_dict) > 0:
                # Multi-user mode: get the embedding for this user
                current_user_embedding = user_embeddings_dict.get(current_user_id, None)
                if current_user_embedding is None:
                    # Fallback to first user if not found
                    first_key = next(iter(user_embeddings_dict.keys()))
                    current_user_embedding = user_embeddings_dict[first_key]
            elif user_embedding is not None:
                # Single-user mode
                current_user_embedding = user_embedding
            else:
                current_user_embedding = None
            
            # Extract target position
            target_pos = extract_target_position_from_state(state)
            if target_pos is None:
                continue
            
            # 🔥 提取起点终点坐标，用于生成 task_key
            start_x, start_y, end_x, end_y = None, None, None, None
            
            # 🔥 统一状态初始化：从len_state=1开始（与固定模式一致）
            # 从state中提取起点坐标，构建干净的初始状态
            if isinstance(state, list) and len(state) > 0:
                # 找到最后一个非零状态帧
                last_non_zero_idx = -1
                for i in range(len(state) - 1, -1, -1):
                    if isinstance(state[i], (list, np.ndarray)) and len(state[i]) >= 15:
                        if not np.all(np.array(state[i]) == 0):
                            last_non_zero_idx = i
                            break
                
                if last_non_zero_idx >= 0:
                    last_frame = state[last_non_zero_idx]
                    if len(last_frame) >= 15:
                        # 🔥 修复：使用固定索引，而不是负索引
                        # origin_x, origin_y 在索引 9, 10
                        # dest_x, dest_y 在索引 11, 12
                        start_x, start_y = float(last_frame[9]), float(last_frame[10])
                        end_x, end_y = float(last_frame[11]), float(last_frame[12])
                        # 构建干净的初始状态（与固定模式一致）
                        initial_state = [0.0] * 9 + [start_x, start_y, end_x, end_y, start_x, start_y]
                        # Pad到state_size
                        from utils.utility import pad_history
                        state_for_pad = [initial_state]
                        current_state = np.array(pad_history(state_for_pad, len(state), np.zeros(15)))
                        current_len_state = 1  # 🔥 统一从1开始
                    else:
                        current_state = np.array(state).copy()
                        current_len_state = 1
                else:
                    current_state = np.array(state).copy()
                    current_len_state = 1
            else:
                current_state = np.array(state).copy()
                current_len_state = 1
            
            # 🔥 生成 task_key（如果成功提取了坐标）
            if start_x is not None and start_y is not None and end_x is not None and end_y is not None:
                task_key = f"{start_x}_{start_y}_{end_x}_{end_y}"
            else:
                # 如果无法提取坐标，尝试从 target_pos 生成（使用当前状态的位置作为起点）
                current_pos = extract_position_from_state_for_shaping(current_state)
                if current_pos and target_pos:
                    task_key = f"{current_pos[0]}_{current_pos[1]}_{target_pos[0]}_{target_pos[1]}"
                else:
                    task_key = None  # 如果完全无法生成，设为 None
            
            # 🔥 初始化距离缓存用于准确的距离改善计算
            last_dist_to_goal = None
            
            # 🔥 统一状态跟踪：最小化跟踪（与固定模式一致）
            prev_action = None
            
            for step in range(max_steps):
                # 🔥 修复：如果启用了文本 embeddings，需要增强 current_state 以匹配模型期望的维度
                state_for_model = current_state
                if text_embedding_loader is not None and embedding_proj_weights is not None:
                    # Check current feature dimension without modifying current_state
                    if isinstance(current_state, np.ndarray):
                        if current_state.ndim == 2:
                            feature_dim = current_state.shape[1]
                            state_size = current_state.shape[0]
                            state_to_enhance = current_state
                        elif current_state.ndim == 1:
                            # If 1D, it's likely a single frame - reshape for processing (copy to avoid modifying original)
                            state_to_enhance = current_state.reshape(1, -1)
                            feature_dim = state_to_enhance.shape[1]
                            state_size = 1
                        else:
                            # Fallback: convert to array
                            state_to_enhance = np.array(current_state)
                            if state_to_enhance.ndim == 2:
                                feature_dim = state_to_enhance.shape[1]
                                state_size = state_to_enhance.shape[0]
                            else:
                                feature_dim = 15
                                state_size = len(state_to_enhance) if state_to_enhance.ndim == 1 else 1
                                state_to_enhance = state_to_enhance.reshape(state_size, feature_dim)
                    else:
                        # Fallback: assume list of frames
                        state_size = len(current_state)
                        feature_dim = len(current_state[0]) if len(current_state) > 0 else 15
                        state_to_enhance = np.array(current_state)
                    
                    # Only enhance if feature_dim is 15 (original dimension)
                    # If it's already enhanced (e.g., from previous iteration), use it directly
                    if feature_dim == 15:
                        # Reshape to [1, state_size, feature_dim] for batch processing
                        current_state_batch = state_to_enhance.reshape(1, state_size, feature_dim)
                        
                        # Enhance with text embeddings and user embeddings
                        enhanced_state_batch = enhance_states_with_text_embeddings(
                            current_state_batch, G, edge_id_map,
                            text_embedding_loader=text_embedding_loader,
                            embedding_proj_weights=embedding_proj_weights,
                            embedding_proj_biases=embedding_proj_biases,
                            original_feature_dim=feature_dim,
                            user_embedding=user_embedding,
                            user_embeddings_dict=user_embeddings_dict,
                            user_ids=[current_user_id]
                        )
                        
                        # Reshape back to [state_size, enhanced_feature_dim]
                        state_for_model = enhanced_state_batch[0]  # Remove batch dimension
                    else:
                        # Already enhanced, use directly
                        state_for_model = current_state
                
                # Generate action mask (use current_state - compute_state_transition handles both original and enhanced)
                action_mask = generate_action_mask_batch([current_state], G, edge_id_map, item_num)[0]
                
                # 🔥 修复：使用SL head进行采集（与predict_path_with_model一致）
                feed_dict = {
                    model.inputs: [state_for_model],
                    model.len_state: [current_len_state],
                    model.action_mask: [action_mask],
                    model.is_training: False,
                    model.training_phase: 1,  # 🔥 使用SL head采集，与predict_path_with_model一致
                    model.rl_weight: 0.0      # 🔥 使用纯SL head，不用RL权重
                }
                
                # 🔥 统一探索策略：简单ε-greedy（与固定模式一致）
                # Use SL head for action selection: π_SL(a|s) (与predict一致)
                actor_probs = sess.run(model.probs, feed_dict=feed_dict)[0]
                
                # 🔥 简化：使用与固定模式相同的简单探索策略
                valid_actions = np.where(action_mask > 0)[0]
                
                # 🔥 Phase-2: Use simple ε-greedy with Actor policy (与固定模式一致)
                epsilon = args.onpolicy_epsilon
                
                # Action selection
                if np.random.random() < epsilon:
                    # Random exploration from valid actions only
                    action_id = np.random.choice(valid_actions)
                else:
                    # Greedy selection from Actor policy: argmax π(a|s)
                    action_id = int(np.argmax(actor_probs))
                
                # Execute action and get next state
                try:
                    # 🔥 修复：传递text embedding和user embedding参数以保持状态维度一致
                    next_state = compute_state_transition(
                        [current_state], [action_id], edge_id_map, G,
                        text_embedding_loader=text_embedding_loader,
                        embedding_proj_weights=embedding_proj_weights,
                        embedding_proj_biases=embedding_proj_biases,
                        user_embedding=current_user_embedding  # 🔥 使用当前用户的embedding
                    )[0]
                    next_len_state = min(current_len_state + 1, model.state_size)
                    
                    # Check if done
                    current_pos = extract_position_from_state_for_shaping(next_state)
                    is_done = False
                    if current_pos and target_pos:
                        distance = np.sqrt((current_pos[0] - target_pos[0])**2 + 
                                         (current_pos[1] - target_pos[1])**2)
                        is_done = (distance < 1e-6)
                    
                    # 🔥 Calculate reward (no additional immediate penalties)
                    
                    # 🔥 计算当前距离和距离改善
                    current_pos = extract_position_from_state_for_shaping(current_state)
                    if current_pos and target_pos:
                        current_dist = np.sqrt((current_pos[0] - target_pos[0])**2 + 
                                               (current_pos[1] - target_pos[1])**2)
                        
                        # 计算距离改善
                        if last_dist_to_goal is None:
                            distance_improvement = 0.0  # 首步不奖励进展
                        else:
                            distance_improvement = last_dist_to_goal - current_dist
                        
                        last_dist_to_goal = current_dist
                    else:
                        distance_improvement = 0.0
                    

                    # 🔥 统一奖励计算：使用与固定模式相同的参数（与固定模式一致）
                    path_info = {
                        'end_pos': target_pos,
                        'distance_improvement': distance_improvement
                    }
                    
                    reward, _ = calculate_improved_reward(
                        action_id, is_done, reward_goal, step,
                        current_state, path_info,
                        prev_action=prev_action, edge_id_map=edge_id_map,
                        step_penalty=args.r_step, backtrack_penalty=args.r_backtrack,
                        recent_visited_nodes=[],  # 🔥 统一：与固定模式一致（空列表）
                        repeat_visit_penalty=-0.03
                    )
                    
                    # 🔥 统一终止类型判断：简化逻辑（与固定模式一致）
                    termination_type = None
                    if is_done:
                        termination_type = 'success'
                    elif step >= max_steps - 1:
                        termination_type = 'timeout'
                    else:
                        termination_type = 'normal'
                    
                    # Add transition with termination type and edge_action for duplicate detection
                    transition = {
                        'state': current_state.copy(),
                        'action': action_id,
                        'reward': reward,
                        'next_state': next_state.copy(),
                        'is_done': is_done,
                        'len_state': current_len_state,
                        'len_next_state': next_len_state,
                        'task_key': task_key,
                        'termination_type': termination_type,
                        'edge_action': (prev_action, action_id) if prev_action is not None else None
                    }
                    transitions.append(transition)
                    
                    # Update state
                    current_state = next_state
                    current_len_state = next_len_state
                    prev_action = action_id
                    
                    # 🔥 统一：与固定模式一致，无额外状态跟踪
                    # Stop only if done
                    if is_done:
                        break
                        
                except Exception as e:
                    # Invalid action, stop this trajectory
                    break
                    
        except Exception as e:
            # Skip this trajectory if any error occurs
            continue
    

    if len(transitions) <= 0:
        print(f"   ⚠️  WARNING: No transitions collected! Check rollout configuration.")
        print(f"   💡 Debug info:")
        print(f"      - Attempted trajectories: {num_trajectories}")
        print(f"      - Max steps per trajectory: {max_steps}")
        print(f"      - Possible issues:")
        print(f"        1. All trajectories failed due to exceptions")
        print(f"        2. All starting states had invalid target_pos")
        print(f"        3. State transition failures (check text embedding compatibility)")
        print()
    
    return transitions

if __name__ == '__main__':
    args = parse_args()

    # Initialize data directory first
    data_directory = args.data
    if not os.path.exists(data_directory):
        data_directory = os.path.join(os.path.dirname(__file__), '..', 'data')
        if not os.path.exists(data_directory):
            raise FileNotFoundError(f"Data directory not found: {args.data} or {data_directory}")
    
    # MOE模式检查
    if args.user_id is not None:
        # 自动处理用户ID格式：如果输入的是 "001"，转换为 "user_001"
        if not args.user_id.startswith('user_'):
            if args.user_id.isdigit():
                args.user_id = f'user_{args.user_id:0>3}'  # 确保3位数字，如 001 -> user_001
            else:
                args.user_id = f'user_{args.user_id}'
        
        print(f"🔀 MOE Mode: Training for user {args.user_id}")
        print(f"   MOE data directory: {args.moe_data_dir}")
        user_data_dir = os.path.join(args.moe_data_dir, args.user_id)
        if not os.path.exists(user_data_dir):
            raise FileNotFoundError(f"MOE user data directory not found: {user_data_dir}")
        print(f"   User data directory: {user_data_dir}")
    else:
        print("📊 Standard Mode: Single-user training")
    
    # Two-Stage Training Logic
    # 优先使用用户指定的路径，否则自动查找
    if args.stage1_model_path and args.stage1_model_path != 'sl_only.ckpt':
        # 用户指定了自定义路径
        if os.path.isabs(args.stage1_model_path):
            stage1_model_path = args.stage1_model_path
        else:
            stage1_model_path = os.path.join(data_directory, 'saved_model', args.stage1_model_path)
    else:
        # 自动查找Stage-1模型
        stage1_model_path = find_stage1_model(data_directory, args.user_id)
        if stage1_model_path is None:
            # 回退到旧路径（向后兼容）
            stage1_model_path = os.path.join(data_directory, 'saved_model', args.stage1_model_path)
    
    # Check if Stage-1 model exists
    if os.path.exists(stage1_model_path + '.meta'):
        print(f"🚀 Loading Stage-1 SL-only model from {stage1_model_path}")
        load_stage1_model = True
    else:
        print(f"⚠️ Stage-1 model not found at {stage1_model_path}")
        print(f"   Will run Stage-1 SL-only training first...")
        load_stage1_model = False

    # Set seeds for reproducibility
    np.random.seed(args.seed_phase2)
    tf.compat.v1.set_random_seed(args.seed_phase2)

    # 加载数据统计文件（支持MOE）
    data_statis_path = get_moe_data_path(data_directory, args.user_id, args.moe_data_dir, 'data_statis.df', args.graph_id)
    if not os.path.exists(data_statis_path):
        raise FileNotFoundError(f"Data statistics file not found: {data_statis_path}")
    data_statis = pd.read_pickle(data_statis_path)
    state_size = data_statis['state_size'][0]
    feature_dim = data_statis['feature_dim'][0]

    reward_goal = args.r_goal

    # 🔥 初始化文本 Embedding 集成器（如果启用）
    text_embedding_loader = None
    embedding_proj_weights = None
    embedding_proj_biases = None
    enhanced_feature_dim = feature_dim
    
    if args.use_text_embeddings:
        if not TEXT_EMBEDDING_AVAILABLE:
            print("⚠️  WARNING: Text embedding modules not available. Disabling text embeddings.")
            if '_TEXT_EMBEDDING_IMPORT_ERROR' in globals():
                print(f"   Import error: {_TEXT_EMBEDDING_IMPORT_ERROR}")
                if 'lmdb' in _TEXT_EMBEDDING_IMPORT_ERROR.lower():
                    print("   💡 Hint: Install lmdb with: pip install lmdb")
            args.use_text_embeddings = False
        else:
            try:
                text_embedding_loader = TextEmbeddingLoader(
                    args.text_embedding_cache_dir,
                    readonly=True,
                    preload_all=True  # 🔥 预加载所有 embeddings 到内存
                )
                print(f"✅ Text embedding loader initialized from {args.text_embedding_cache_dir}")
                print(f"   Embedding dim: {text_embedding_loader.hidden_dim}")
                
                # 计算增强后的 feature_dim
                text_dim = 0
                if args.use_edge_embedding:
                    text_dim += args.embedding_proj_dim
                if args.use_node_embedding:
                    text_dim += args.embedding_proj_dim
                if args.use_goal_embedding:
                    text_dim += args.embedding_proj_dim
                
                enhanced_feature_dim = feature_dim + text_dim
                print(f"   Original feature_dim: {feature_dim}")
                print(f"   Enhanced feature_dim: {enhanced_feature_dim} (+{text_dim} from text embeddings)")
                
                # 初始化投影权重（使用 Xavier/Glorot 初始化）
                # 注意：这些权重是 numpy 版本，用于预处理
                # 实际的投影层会在 TensorFlow 图中创建和训练，但这里先用固定权重
                embedding_proj_weights = {}
                embedding_proj_biases = {}
                
                if args.use_edge_embedding:
                    # Xavier/Glorot 初始化
                    limit = np.sqrt(6.0 / (text_embedding_loader.hidden_dim + args.embedding_proj_dim))
                    embedding_proj_weights['edge'] = np.random.uniform(
                        -limit, limit, (text_embedding_loader.hidden_dim, args.embedding_proj_dim)
                    ).astype(np.float32)
                    embedding_proj_biases['edge'] = np.zeros(args.embedding_proj_dim, dtype=np.float32)
                
                if args.use_node_embedding:
                    limit = np.sqrt(6.0 / (text_embedding_loader.hidden_dim + args.embedding_proj_dim))
                    embedding_proj_weights['node'] = np.random.uniform(
                        -limit, limit, (text_embedding_loader.hidden_dim, args.embedding_proj_dim)
                    ).astype(np.float32)
                    embedding_proj_biases['node'] = np.zeros(args.embedding_proj_dim, dtype=np.float32)
                
                if args.use_goal_embedding:
                    limit = np.sqrt(6.0 / (text_embedding_loader.hidden_dim + args.embedding_proj_dim))
                    embedding_proj_weights['goal'] = np.random.uniform(
                        -limit, limit, (text_embedding_loader.hidden_dim, args.embedding_proj_dim)
                    ).astype(np.float32)
                    embedding_proj_biases['goal'] = np.zeros(args.embedding_proj_dim, dtype=np.float32)
                
                print(f"   Initialized projection weights (numpy, will be replaced by TF trained weights)")
                print(f"   🔥 Using embeddings: edge={args.use_edge_embedding}, node={args.use_node_embedding}, goal={args.use_goal_embedding}")
                
            except Exception as e:
                print(f"❌ ERROR: Failed to initialize text embedding loader: {e}")
                print("   Disabling text embeddings.")
                args.use_text_embeddings = False
                text_embedding_loader = None

    # 🔥 加载用户 embedding（如果启用）
    user_embedding = None
    user_embeddings_dict = None
    user_dim = 0
    
    if args.use_user_embedding:
        print(f"\n🧑 User Embedding Mode: ENABLED")
        
        if args.user_id is not None:
            # 单用户模式：加载指定用户的embedding
            print(f"   Mode: Single-user (loading {args.user_id})")
            user_embedding = load_user_embedding(args.user_id, args.moe_data_dir)
            if user_embedding is not None:
                user_dim = len(user_embedding)
                enhanced_feature_dim += user_dim
                print(f"✅ User embedding loaded: dim={user_dim}")
                print(f"   Final enhanced feature_dim: {enhanced_feature_dim} (base={feature_dim}, text={text_dim if args.use_text_embeddings else 0}, user={user_dim})")
            else:
                print(f"⚠️  User embedding not found for {args.user_id}, proceeding without it")
                args.use_user_embedding = False
        else:
            # 多用户模式：加载所有用户的embeddings
            print(f"   Mode: Multi-user (loading all users)")
            user_embeddings_dict = load_all_user_embeddings(args.moe_data_dir)
            if user_embeddings_dict is not None and len(user_embeddings_dict) > 0:
                # 从第一个用户的embedding获取维度
                first_user_emb = next(iter(user_embeddings_dict.values()))
                user_dim = len(first_user_emb)
                enhanced_feature_dim += user_dim
                print(f"✅ Loaded {len(user_embeddings_dict)} user embeddings")
                print(f"   User embedding dim: {user_dim}")
                print(f"   Final enhanced feature_dim: {enhanced_feature_dim} (base={feature_dim}, text={text_dim if args.use_text_embeddings else 0}, user={user_dim})")
            else:
                print(f"⚠️  No user embeddings found in {args.moe_data_dir}, proceeding without them")
                args.use_user_embedding = False
                user_embeddings_dict = None
    else:
        print(f"\n🧑 User Embedding Mode: DISABLED (use --use_user_embedding to enable)")

    tf.compat.v1.reset_default_graph()

    # Load map data from graph-specific raw_data directory
    raw_data_dir = os.path.join(data_directory, 'graph_data', args.graph_id, 'raw_data')
    data = load_map_data(raw_data_dir)
    df, meta_data = data["df"], data["meta_data"]
    df_copy = deepcopy(df)

    G, edge_id_map, edge_feature_list = load_or_create_graph(df_copy, data_directory, args.graph_id)

    item_num = G.number_of_edges()

    # 🔥 使用增强后的 feature_dim（包含text embeddings和user embeddings）
    # enhanced_feature_dim已经根据是否启用text/user embeddings正确更新
    model_feature_dim = enhanced_feature_dim
    
    QN_1 = ImprovedQNetwork(name='QN_1', hidden_size=args.hidden_factor, learning_rate=args.lr,
                           feature_dim=model_feature_dim, item_num=item_num, state_size=state_size,
                           dropout_rate=args.dropout_rate,
                           num_heads=args.num_heads, num_blocks=args.num_blocks, lr_2=args.lr_2, neg=args.neg,
                           max_candidates=args.max_candidates if hasattr(args, 'max_candidates') else 20,
                           cand_feature_dim=12,  # 🔥 候选边特征维度（12维）
                           )
    QN_2 = ImprovedQNetwork(name='QN_2', hidden_size=args.hidden_factor, learning_rate=args.lr,
                           feature_dim=model_feature_dim, item_num=item_num, state_size=state_size,
                           dropout_rate=args.dropout_rate,
                           num_heads=args.num_heads, num_blocks=args.num_blocks, lr_2=args.lr_2, neg=args.neg,
                           max_candidates=args.max_candidates if hasattr(args, 'max_candidates') else 20,
                           cand_feature_dim=12,  # 🔥 候选边特征维度（12维）
                           )
    hard_update_ops = build_hard_update_ops(QN_1.train_vars, QN_2.train_vars)
    soft_update_ops = build_soft_update_ops(QN_1.train_vars, QN_2.train_vars, tau=0.001)  # 🔥 使用降低的tau
    
    # 🔥 Actor-Critic DDPG-style: Build Actor target update operations
    actor_target_soft_update_ops = build_soft_update_ops(QN_1.actor_main_vars, QN_1.actor_target_vars, tau=0.001)  # 🔥 使用降低的tau
    actor_target_hard_update_ops = build_hard_update_ops(QN_1.actor_main_vars, QN_1.actor_target_vars)

    # Initialize global_step based on whether we're loading Stage-1 model
    if load_stage1_model:
        global_step = args.phase1_sl_only_steps  # 🔥 从Stage-1结束的步数开始
        # 🔥 确保global_step是整数，避免tuple/float比较错误
        print(f"🎯 Starting from global_step={global_step} (Stage-1 completed)")
    else:
        global_step = 0
        print(f"🎯 Starting from global_step={global_step} (fresh training)")

    # 加载replay buffer（支持MOE）
    replay_buffer_path = get_moe_data_path(data_directory, args.user_id, args.moe_data_dir, 'replay_buffer.df', args.graph_id)
    if not os.path.exists(replay_buffer_path):
        raise FileNotFoundError(f"Replay buffer file not found: {replay_buffer_path}")
    replay_buffer = pd.read_pickle(replay_buffer_path)
    original_size = len(replay_buffer)
    print(f"✅ Loaded replay buffer from: {replay_buffer_path} ({original_size} transitions)")
    
    # 🔥 根据 train_data_ratio 参数采样训练集
    if args.train_data_ratio < 1.0:
        if args.train_data_ratio <= 0 or args.train_data_ratio > 1.0:
            raise ValueError(f"train_data_ratio must be between 0 and 1.0, got {args.train_data_ratio}")
        sample_size = int(original_size * args.train_data_ratio)
        if sample_size < 1:
            raise ValueError(f"train_data_ratio {args.train_data_ratio} results in sample_size < 1 (original_size={original_size})")
        replay_buffer = replay_buffer.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"📊 Sampled {len(replay_buffer)} transitions ({args.train_data_ratio*100:.1f}% of original {original_size} transitions)")
    else:
        print(f"📊 Using full training set ({original_size} transitions, 100%)")

    saver = tf.compat.v1.train.Saver()
    # Initialize training logger (修复logger未定义问题)
    logger = init_logger(os.path.join(data_directory, 'training_logs'))
    
    # Initialize on-policy ring buffer (O-bucket) for DAgger-lite
    # 🔥 确保buffer_size是整数，避免tuple错误
    normalized_buffer_size = args.onpolicy_buffer_size
    onpolicy_buffer = OnPolicyRingBuffer(max_size=normalized_buffer_size)
    print(f"Initialized on-policy ring buffer (O-bucket) with max_size={normalized_buffer_size}")

    # 🔥 初始化 graph_cache 和 feature_builder（用于新格式数据和候选边特征重建）
    graph_cache = None
    feature_builder = None
    try:
        from utils.graph_cache import MultiGraphCache
        multi_cache = MultiGraphCache(os.path.join(data_directory, 'graph_data'))
        graph_cache = multi_cache.get_cache(args.graph_id)
        feature_builder = multi_cache.get_feature_builder(args.graph_id)
        print(f'✅ Loaded GraphCache and FeatureBuilder for graph "{args.graph_id}"')
    except Exception as e:
        print(f'⚠️  Could not load GraphCache/FeatureBuilder: {e}')
        print(f'   Training will use old format data without candidate features')

    # Preload datasets if requested
    if args.preload_datasets:
        preload_all_datasets(data_directory, args.user_id, args.moe_data_dir, graph_id=args.graph_id)

    # If resume_ckpt is provided, try to load and set global_step
    resume_ckpt = args.resume_ckpt
    if resume_ckpt is not None:
        # Handle path - if not absolute path, assume it's in saved_model directory
        if not os.path.isabs(resume_ckpt):
            resume_ckpt = os.path.join(data_directory, 'saved_model', resume_ckpt)
        
        # Remove .index if user copy-pasted the full file name
        if not resume_ckpt.endswith('.index'):
            resume_ckpt = resume_ckpt.replace('.index', '')
    
    # 🔥 Eval-only mode: Check if we need to load a checkpoint
    if args.eval_only:
        if resume_ckpt is not None and os.path.exists(resume_ckpt + ".index"):
            # User specified a valid checkpoint, use it
            print(f"🔍 Eval-only mode: Using specified checkpoint: {resume_ckpt}")
        elif resume_ckpt is None or not os.path.exists(resume_ckpt + ".index"):
            # Try to use stage1_model_path as fallback
            if os.path.exists(stage1_model_path + '.meta'):
                resume_ckpt = stage1_model_path
                print(f"🔍 Eval-only mode: Using Stage-1 model as checkpoint: {resume_ckpt}")
            else:
                # Try to find the latest checkpoint in saved_model directory
                saved_model_dir = os.path.join(data_directory, 'saved_model')
                if os.path.exists(saved_model_dir):
                    import glob
                    checkpoints = glob.glob(os.path.join(saved_model_dir, '*.ckpt.index'))
                    if checkpoints:
                        # Sort by modification time, get the latest
                        checkpoints.sort(key=os.path.getmtime, reverse=True)
                        latest_ckpt = checkpoints[0].replace('.index', '')
                        resume_ckpt = latest_ckpt
                        print(f"🔍 Eval-only mode: Found latest checkpoint: {resume_ckpt}")
                    else:
                        raise FileNotFoundError(
                            f"❌ Eval-only mode requires a checkpoint. Please provide --resume_ckpt or ensure a checkpoint exists in {saved_model_dir}"
                        )
                else:
                    raise FileNotFoundError(
                        f"❌ Eval-only mode requires a checkpoint. Please provide --resume_ckpt or ensure saved_model directory exists."
                    )
    
    # 🔥 配置 GPU 内存增长，避免一次性分配所有内存
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(hard_update_ops)
        # 🔥 Actor-Critic DDPG-style: Initialize Actor target
        sess.run(actor_target_hard_update_ops)
        
        # 🔥 Load Stage-1 model if available (优先级高于resume_ckpt)
        # Skip Stage-1 loading in eval-only mode to allow loading specified checkpoint
        if load_stage1_model and not args.eval_only:
            print(f"📥 Loading Stage-1 SL-only model...")
            try:
                saver.restore(sess, stage1_model_path)
                print(f"✅ Successfully loaded Stage-1 model from {stage1_model_path}")
                print(f"🚀 Will start training from Phase-2 (SL+RL joint training)")
                
                # 🔥 Pre-collect O-buffer with SL head (performance optimization: save/load)
                # 优先使用用户指定的路径，否则自动查找
                if args.precollected_obuffer_path and args.precollected_obuffer_path != 'precollected_obuffer_sl.df':
                    # 用户指定了自定义路径
                    if os.path.isabs(args.precollected_obuffer_path):
                        precollected_obuffer_file = args.precollected_obuffer_path
                    else:
                        precollected_obuffer_file = os.path.join(data_directory, args.precollected_obuffer_path)
                else:
                    # 自动查找预收集数据
                    precollected_obuffer_file = find_precollected_data(data_directory, args.user_id)
                    if precollected_obuffer_file is None:
                        # 回退到旧路径（向后兼容）
                        precollected_obuffer_file = os.path.join(data_directory, args.precollected_obuffer_path)
                # 🔥 确保参数是数字，避免tuple/float比较错误
                onpolicy_buffer_size = args.onpolicy_buffer_size
                # 处理precollect_buffer_ratio（可能是tuple或float）
                if isinstance(args.precollect_buffer_ratio, (tuple, list)):
                    precollect_buffer_ratio = float(args.precollect_buffer_ratio[0]) if len(args.precollect_buffer_ratio) > 0 else 0.89
                else:
                    try:
                        precollect_buffer_ratio = float(args.precollect_buffer_ratio)
                    except (ValueError, TypeError):
                        precollect_buffer_ratio = 0.89
                precollect_target_size = int(onpolicy_buffer_size * precollect_buffer_ratio)
                
                # Check if pre-collected O-buffer exists (try main file first, then checkpoint)
                load_success = False
                checkpoint_file = precollected_obuffer_file.replace('.df', '_checkpoint.df')
                
                if os.path.exists(precollected_obuffer_file):
                    print(f"\n🔥 Loading pre-collected O-buffer from {precollected_obuffer_file}...")
                    load_success = onpolicy_buffer.load_from_file(precollected_obuffer_file, max_size=onpolicy_buffer_size)
                    
                    if load_success and onpolicy_buffer.size() >= precollect_target_size * 0.9:  # Allow 10% tolerance
                        print(f"✅ Pre-collected O-buffer loaded: {onpolicy_buffer.size()}/{onpolicy_buffer_size} transitions")
                        snapshot = onpolicy_buffer.get_composition_snapshot()
                        print(f"   Success ratio: {snapshot['success_ratio']:.2%}")
                        print(f"   Termination types: {snapshot['termination_types']}")
                    else:
                        print(f"⚠️  Pre-collected file exists but size insufficient ({onpolicy_buffer.size()}/{precollect_target_size}), trying checkpoint...")
                        load_success = False
                elif os.path.exists(checkpoint_file):
                    print(f"\n🔥 Pre-collected O-buffer not found, but checkpoint exists: {checkpoint_file}")
                    print(f"   Loading checkpoint...")
                    load_success = onpolicy_buffer.load_from_file(checkpoint_file, max_size=onpolicy_buffer_size)
                    
                    if load_success and onpolicy_buffer.size() >= precollect_target_size * 0.5:  # Checkpoint only needs 50%
                        print(f"✅ Checkpoint loaded: {onpolicy_buffer.size()}/{onpolicy_buffer_size} transitions")
                        print(f"   Will continue from where it left off...")
                        snapshot = onpolicy_buffer.get_composition_snapshot()
                        print(f"   Success ratio: {snapshot['success_ratio']:.2%}")
                        print(f"   Termination types: {snapshot['termination_types']}")
                    else:
                        print(f"⚠️  Checkpoint size insufficient ({onpolicy_buffer.size()}/{precollect_target_size}), re-collecting...")
                        onpolicy_buffer.buffer = []
                        onpolicy_buffer.position = 0
                        onpolicy_buffer.edge_usage_count = {}
                        load_success = False
                else:
                    load_success = False
                    print(f"\n🔥 Pre-collected O-buffer not found at {precollected_obuffer_file}")
                    print(f"   Checkpoint not found at {checkpoint_file}")
                    print(f"   Will collect new O-buffer with SL head...")
                
                if not load_success:
                    onpolicy_buffer.buffer = []
                    onpolicy_buffer.position = 0
                    onpolicy_buffer.edge_usage_count = {}
                
                # If loading failed or file doesn't exist, perform pre-collection
                if not load_success:
                    # 🔥 使用统一的预收集函数，避免代码重复
                    precollect_success = precollect_obuffer_at_phase2_transition(
                        sess, QN_1, onpolicy_buffer, data_directory, args, args.user_id, args.moe_data_dir,
                        G, edge_id_map, item_num, reward_goal, state_size, feature_dim, model_feature_dim,
                        text_embedding_loader, embedding_proj_weights, embedding_proj_biases,
                        precollected_obuffer_file=precollected_obuffer_file,
                        replay_buffer=replay_buffer,  # 🔥 传递已采样的replay_buffer
                        user_id_for_model=None,
                        user_embedding=user_embedding if args.use_user_embedding else None,
                        user_embeddings_dict=user_embeddings_dict if args.use_user_embedding else None
                    )
                    if not precollect_success:
                        print(f"   ⚠️  Pre-collection failed, will collect during training")
                
                # 🔥 加载模型后进行初始评估
                print(f"\n🔍 Evaluating loaded Stage-1 model...")
                # 🔥 确保global_step是整数，避免tuple/float比较错误
                current_phase, phase_name, rl_weight, current_lr = determine_training_phase(global_step, args, load_stage1_model)
                print(f"📊 Current Phase: {phase_name} (Step {global_step})")
                
            except Exception as e:
                print(f"❌ Failed to load Stage-1 model: {e}")
                print(f"   Will start training from scratch...")
                load_stage1_model = False
                global_step = 0  # 🔥 重置global_step
                
        elif resume_ckpt is not None and os.path.exists(resume_ckpt + ".index"):
            print(f"Restoring model from checkpoint: {resume_ckpt}")
            saver.restore(sess, resume_ckpt)
            # 自动推断global_step
            import re
            match = re.search(r'step[_\-]?(\d+)', resume_ckpt)
            if match:
                global_step = int(match.group(1))
                print(f"Resuming from global_step={global_step}")
            else:
                # 兼容10k等特殊命名
                match = re.search(r'(\d+)[kK]', resume_ckpt)
                if match:
                    global_step = int(match.group(1)) * 1000
                    print(f"Resuming from global_step={global_step}")
                else:
                    print("Could not infer global_step from checkpoint name, starting from 0.")
            
            # 🔥 修复：加载checkpoint后，检查模型期望的特征维度
            # 如果模型期望增强特征（143维），但text_embedding_loader未初始化，强制初始化
            model_expects_enhanced = QN_1.feature_dim > feature_dim
            if model_expects_enhanced and text_embedding_loader is None:
                print(f"\n⚠️  WARNING: Loaded model expects feature_dim={QN_1.feature_dim} (enhanced), but text_embedding_loader is None.")
                print(f"   Attempting to initialize text embedding loader...")
                if args.text_embedding_cache_dir and os.path.exists(args.text_embedding_cache_dir):
                    try:
                        from text_embedding_loader import TextEmbeddingLoader
                        text_embedding_loader = TextEmbeddingLoader(
                            args.text_embedding_cache_dir,
                            readonly=True,
                            preload_all=True
                        )
                        print(f"✅ Text embedding loader initialized from {args.text_embedding_cache_dir}")
                        
                        # 初始化投影权重（使用随机初始化，实际权重应该从checkpoint加载）
                        if embedding_proj_weights is None:
                            embedding_proj_weights = {}
                            embedding_proj_biases = {}
                        if args.use_edge_embedding or True:  # 默认启用edge
                            if 'edge' not in embedding_proj_weights:
                                limit = np.sqrt(6.0 / (text_embedding_loader.hidden_dim + args.embedding_proj_dim))
                                embedding_proj_weights['edge'] = np.random.uniform(
                                    -limit, limit, (text_embedding_loader.hidden_dim, args.embedding_proj_dim)
                                ).astype(np.float32)
                                embedding_proj_biases['edge'] = np.zeros(args.embedding_proj_dim, dtype=np.float32)
                        if args.use_node_embedding or True:  # 默认启用node
                            if 'node' not in embedding_proj_weights:
                                limit = np.sqrt(6.0 / (text_embedding_loader.hidden_dim + args.embedding_proj_dim))
                                embedding_proj_weights['node'] = np.random.uniform(
                                    -limit, limit, (text_embedding_loader.hidden_dim, args.embedding_proj_dim)
                                ).astype(np.float32)
                                embedding_proj_biases['node'] = np.zeros(args.embedding_proj_dim, dtype=np.float32)
                        if args.use_goal_embedding or True:  # 默认启用goal
                            if 'goal' not in embedding_proj_weights:
                                limit = np.sqrt(6.0 / (text_embedding_loader.hidden_dim + args.embedding_proj_dim))
                                embedding_proj_weights['goal'] = np.random.uniform(
                                    -limit, limit, (text_embedding_loader.hidden_dim, args.embedding_proj_dim)
                                ).astype(np.float32)
                                embedding_proj_biases['goal'] = np.zeros(args.embedding_proj_dim, dtype=np.float32)
                        
                        # 强制启用text embeddings
                        args.use_text_embeddings = True
                        args.use_edge_embedding = True
                        args.use_node_embedding = True
                        args.use_goal_embedding = True
                        print(f"   ✅ Forced enable text embeddings for evaluation")
                    except Exception as e:
                        print(f"❌ Failed to initialize text embedding loader: {e}")
                        raise RuntimeError(
                            f"Model expects feature_dim={QN_1.feature_dim} (enhanced), but cannot initialize text_embedding_loader. "
                            f"Please provide --text_embedding_cache_dir or ensure the model was trained with text embeddings."
                        )
                else:
                    raise RuntimeError(
                        f"Model expects feature_dim={QN_1.feature_dim} (enhanced), but text_embedding_loader is None and "
                        f"--text_embedding_cache_dir is not provided or does not exist. "
                        f"Please provide --text_embedding_cache_dir=<path_to_text_embeddings>."
                    )
            elif model_expects_enhanced and text_embedding_loader is not None:
                print(f"✅ Model expects enhanced features (dim={QN_1.feature_dim}), text_embedding_loader is available")
        elif args.eval_only and resume_ckpt is not None and os.path.exists(resume_ckpt + ".index"):
            # Eval-only mode: Load the checkpoint we determined earlier
            print(f"🔍 Eval-only mode: Loading checkpoint: {resume_ckpt}")
            saver.restore(sess, resume_ckpt)
            # Try to infer global_step from checkpoint name
            import re
            match = re.search(r'step[_\-]?(\d+)', resume_ckpt)
            if match:
                global_step = int(match.group(1))
            else:
                match = re.search(r'(\d+)[kK]', resume_ckpt)
                if match:
                    global_step = int(match.group(1)) * 1000
                else:
                    # Default to phase2 start step if can't infer
                    global_step = args.phase1_sl_only_steps
            print(f"✅ Checkpoint loaded, inferred global_step={global_step}")
            
            # 🔥 修复：加载checkpoint后，检查模型期望的特征维度
            model_expects_enhanced = QN_1.feature_dim > feature_dim
            if model_expects_enhanced and text_embedding_loader is None:
                print(f"\n⚠️  WARNING: Loaded model expects feature_dim={QN_1.feature_dim} (enhanced), but text_embedding_loader is None.")
                print(f"   Attempting to initialize text embedding loader...")
                if args.text_embedding_cache_dir and os.path.exists(args.text_embedding_cache_dir):
                    try:
                        from text_embedding_loader import TextEmbeddingLoader
                        text_embedding_loader = TextEmbeddingLoader(
                            args.text_embedding_cache_dir,
                            readonly=True,
                            preload_all=True
                        )
                        print(f"✅ Text embedding loader initialized from {args.text_embedding_cache_dir}")
                        
                        # 初始化投影权重
                        if embedding_proj_weights is None:
                            embedding_proj_weights = {}
                            embedding_proj_biases = {}
                        if args.use_edge_embedding or True:
                            if 'edge' not in embedding_proj_weights:
                                limit = np.sqrt(6.0 / (text_embedding_loader.hidden_dim + args.embedding_proj_dim))
                                embedding_proj_weights['edge'] = np.random.uniform(
                                    -limit, limit, (text_embedding_loader.hidden_dim, args.embedding_proj_dim)
                                ).astype(np.float32)
                                embedding_proj_biases['edge'] = np.zeros(args.embedding_proj_dim, dtype=np.float32)
                        if args.use_node_embedding or True:
                            if 'node' not in embedding_proj_weights:
                                limit = np.sqrt(6.0 / (text_embedding_loader.hidden_dim + args.embedding_proj_dim))
                                embedding_proj_weights['node'] = np.random.uniform(
                                    -limit, limit, (text_embedding_loader.hidden_dim, args.embedding_proj_dim)
                                ).astype(np.float32)
                                embedding_proj_biases['node'] = np.zeros(args.embedding_proj_dim, dtype=np.float32)
                        if args.use_goal_embedding or True:
                            if 'goal' not in embedding_proj_weights:
                                limit = np.sqrt(6.0 / (text_embedding_loader.hidden_dim + args.embedding_proj_dim))
                                embedding_proj_weights['goal'] = np.random.uniform(
                                    -limit, limit, (text_embedding_loader.hidden_dim, args.embedding_proj_dim)
                                ).astype(np.float32)
                                embedding_proj_biases['goal'] = np.zeros(args.embedding_proj_dim, dtype=np.float32)
                        
                        args.use_text_embeddings = True
                        args.use_edge_embedding = True
                        args.use_node_embedding = True
                        args.use_goal_embedding = True
                        print(f"   ✅ Forced enable text embeddings for evaluation")
                    except Exception as e:
                        print(f"❌ Failed to initialize text embedding loader: {e}")
                        raise RuntimeError(
                            f"Model expects feature_dim={QN_1.feature_dim} (enhanced), but cannot initialize text_embedding_loader. "
                            f"Please provide --text_embedding_cache_dir or ensure the model was trained with text embeddings."
                        )
                else:
                    raise RuntimeError(
                        f"Model expects feature_dim={QN_1.feature_dim} (enhanced), but text_embedding_loader is None and "
                        f"--text_embedding_cache_dir is not provided or does not exist. "
                        f"Please provide --text_embedding_cache_dir=<path_to_text_embeddings>."
                    )
            elif model_expects_enhanced and text_embedding_loader is not None:
                print(f"✅ Model expects enhanced features (dim={QN_1.feature_dim}), text_embedding_loader is available")
        elif args.eval_only:
            raise RuntimeError(f"❌ Eval-only mode: No valid checkpoint to load: {resume_ckpt}")
        else:
            print("No checkpoint loaded, training from scratch.")

        num_rows=replay_buffer.shape[0]
        num_batches=int(num_rows/args.batch_size)
        
        # 🔥 计算 Phase-1 的总步数（基于epoch数）
        min_sl_steps = 1000
        steps_per_epoch = num_batches
        args.phase1_sl_only_steps = max(min_sl_steps, args.phase1_epochs * steps_per_epoch)
        print(f"📊 Calculated phase1_sl_only_steps = max({min_sl_steps}, {args.phase1_epochs} epochs × {steps_per_epoch} steps/epoch) = {args.phase1_sl_only_steps} steps")
        
        # 🔥 根据 epoch-based 参数自动计算 step-based 的 log_frequency 和 eval_frequency
        # 这样不同 data ratio 下，用户只需设置相同的 epoch 参数，就能保证一致的评估频率
        args.log_frequency = max(1, int(args.log_frequency_epoch * steps_per_epoch))
        args.eval_frequency = max(1, int(args.eval_frequency_epoch * steps_per_epoch))
        print(f"📊 Calculated log_frequency = {args.log_frequency_epoch} epochs x {steps_per_epoch} steps/epoch = {args.log_frequency} steps")
        print(f"📊 Calculated eval_frequency = {args.eval_frequency_epoch} epochs x {steps_per_epoch} steps/epoch = {args.eval_frequency} steps")
        
        # Initial evaluation
        if not args.skip_initial_eval:
            print("\n" + "="*60)
            print("INITIAL EVALUATION (Before Training)")
            print("="*60)
            
            # 🔥 修复：检查模型期望的特征维度，如果模型期望增强特征，必须提供text_embedding_loader
            model_expects_enhanced = QN_1.feature_dim > feature_dim
            if model_expects_enhanced:
                if text_embedding_loader is None:
                    print(f"⚠️  WARNING: Model expects feature_dim={QN_1.feature_dim} (enhanced), but text_embedding_loader is None.")
                    print(f"   Attempting to initialize text embedding loader from checkpoint or args...")
                    # 尝试从checkpoint加载投影权重，或者使用默认配置初始化
                    if args.text_embedding_cache_dir and os.path.exists(args.text_embedding_cache_dir):
                        try:
                            from text_embedding_loader import TextEmbeddingLoader
                            text_embedding_loader = TextEmbeddingLoader(
                                cache_dir=args.text_embedding_cache_dir,
                                preload_all=True
                            )
                            print(f"✅ Text embedding loader initialized from {args.text_embedding_cache_dir}")
                            
                            # 初始化投影权重（使用随机初始化，实际权重应该从checkpoint加载）
                            embedding_proj_weights = {}
                            embedding_proj_biases = {}
                            if args.use_edge_embedding or True:  # 默认启用edge
                                limit = np.sqrt(6.0 / (text_embedding_loader.hidden_dim + args.embedding_proj_dim))
                                embedding_proj_weights['edge'] = np.random.uniform(
                                    -limit, limit, (text_embedding_loader.hidden_dim, args.embedding_proj_dim)
                                ).astype(np.float32)
                                embedding_proj_biases['edge'] = np.zeros(args.embedding_proj_dim, dtype=np.float32)
                            if args.use_node_embedding or True:  # 默认启用node
                                limit = np.sqrt(6.0 / (text_embedding_loader.hidden_dim + args.embedding_proj_dim))
                                embedding_proj_weights['node'] = np.random.uniform(
                                    -limit, limit, (text_embedding_loader.hidden_dim, args.embedding_proj_dim)
                                ).astype(np.float32)
                                embedding_proj_biases['node'] = np.zeros(args.embedding_proj_dim, dtype=np.float32)
                            if args.use_goal_embedding or True:  # 默认启用goal
                                limit = np.sqrt(6.0 / (text_embedding_loader.hidden_dim + args.embedding_proj_dim))
                                embedding_proj_weights['goal'] = np.random.uniform(
                                    -limit, limit, (text_embedding_loader.hidden_dim, args.embedding_proj_dim)
                                ).astype(np.float32)
                                embedding_proj_biases['goal'] = np.zeros(args.embedding_proj_dim, dtype=np.float32)
                            
                            # 强制启用text embeddings
                            args.use_text_embeddings = True
                            args.use_edge_embedding = True
                            args.use_node_embedding = True
                            args.use_goal_embedding = True
                            print(f"   ✅ Forced enable text embeddings for evaluation")
                        except Exception as e:
                            print(f"❌ Failed to initialize text embedding loader: {e}")
                            raise RuntimeError(
                                f"Model expects feature_dim={QN_1.feature_dim} (enhanced), but cannot initialize text_embedding_loader. "
                                f"Please provide --text_embedding_cache_dir or ensure the model was trained with text embeddings."
                            )
                    else:
                        raise RuntimeError(
                            f"Model expects feature_dim={QN_1.feature_dim} (enhanced), but text_embedding_loader is None and "
                            f"--text_embedding_cache_dir is not provided. Please provide --text_embedding_cache_dir."
                        )
                else:
                    print(f"✅ Model expects enhanced features (dim={QN_1.feature_dim}), text_embedding_loader is available")
            
            # 🔥 根据是否加载Stage-1模型确定评估参数
            if load_stage1_model:
                # 加载了Stage-1模型，使用Phase-2参数进行评估
                current_phase, phase_name, rl_weight, current_lr = determine_training_phase(global_step, args, load_stage1_model)
                print(f"📊 Evaluating loaded Stage-1 model with Phase-2 parameters:")
                print(f"   Phase: {phase_name}, RL Weight: {rl_weight}, LR: {current_lr}")
                # 🔥 使用test集进行初始评估，显示完整的新评估指标（A.单步准确率, B.Reach@B, C.路径准确率/覆盖度）
                print(f"\n🔥 Using TEST dataset for initial evaluation to show complete metrics...")
                batch_evaluate_improved(sess, QN_1, dataset='test', logger=logger, step=global_step, 
                                      training_phase=current_phase, rl_weight=rl_weight, 
                                      batch_size=args.batch_size, sample_ratio=1.0,  # 使用全量test集
                                      rl_rollout_sample_size=50, G=G, edge_id_map=edge_id_map, 
                                      item_num=item_num, reward_goal=reward_goal, target_model=QN_2,
                                      eval_num_trials=args.eval_num_trials, eval_temperature=args.eval_temperature,
                                      use_text_embeddings=args.use_text_embeddings, text_embedding_loader=text_embedding_loader, 
                                      embedding_proj_weights=embedding_proj_weights, embedding_proj_biases=embedding_proj_biases, feature_dim=feature_dim,
                                      data_directory=data_directory, user_id=args.user_id, moe_data_dir=args.moe_data_dir, graph_id=args.graph_id)
            else:
                # 从头开始训练，使用Phase-1参数进行评估
                print(f"📊 Evaluating fresh model with Phase-1 parameters:")
                # 🔥 使用test集进行初始评估
                print(f"\n🔥 Using TEST dataset for initial evaluation to show complete metrics...")
                batch_evaluate_improved(sess, QN_1, dataset='test', logger=logger, step=global_step, 
                                      training_phase=1, rl_weight=0.0, 
                                      batch_size=args.batch_size, sample_ratio=1.0,  # 使用全量test集
                                      rl_rollout_sample_size=50, G=G, edge_id_map=edge_id_map, 
                                      item_num=item_num, reward_goal=reward_goal, target_model=QN_2,
                                      eval_num_trials=args.eval_num_trials, eval_temperature=args.eval_temperature,
                                      use_text_embeddings=args.use_text_embeddings, text_embedding_loader=text_embedding_loader, 
                                      embedding_proj_weights=embedding_proj_weights, embedding_proj_biases=embedding_proj_biases, feature_dim=feature_dim,
                                      data_directory=data_directory, user_id=args.user_id, moe_data_dir=args.moe_data_dir, graph_id=args.graph_id)
            print("="*60 + "\n")
        else:
            print("\n" + "="*60)
            print("SKIPPING INITIAL EVALUATION (--skip_initial_eval flag set)")
            print("="*60 + "\n")
            # 🔥 即使跳过评估，也需要确定训练阶段的参数（用于后续训练）
            if load_stage1_model:
                current_phase, phase_name, rl_weight, current_lr = determine_training_phase(global_step, args, load_stage1_model)
            else:
                current_phase, phase_name, rl_weight, current_lr = determine_training_phase(global_step, args, False)
            print(f"📊 Starting training from: {phase_name} (Step {global_step})")
            print(f"⚙️ RL Weight: {rl_weight}, Learning Rate: {current_lr}")
            
            # 🔥 如果已经进入 Phase-2 但没有执行预收集（从 checkpoint 恢复或 Stage-1 模型不存在），也需要执行预收集
            if current_phase >= 2 and not load_stage1_model:
                print(f"\n🔥 Detected Phase-2 training (step {global_step} >= {args.phase1_sl_only_steps})")
                print(f"   Checking pre-collected O-buffer...")
                
                # 执行预收集逻辑（复用 Stage-1 加载时的逻辑）
                precollected_obuffer_file = os.path.join(data_directory, args.precollected_obuffer_path)
                onpolicy_buffer_size = args.onpolicy_buffer_size
                if isinstance(args.precollect_buffer_ratio, (tuple, list)):
                    precollect_buffer_ratio = float(args.precollect_buffer_ratio[0]) if len(args.precollect_buffer_ratio) > 0 else 0.89
                else:
                    try:
                        precollect_buffer_ratio = float(args.precollect_buffer_ratio)
                    except (ValueError, TypeError):
                        precollect_buffer_ratio = 0.89
                precollect_target_size = int(onpolicy_buffer_size * precollect_buffer_ratio)
                
                checkpoint_file = precollected_obuffer_file.replace('.df', '_checkpoint.df')
                load_success = False
                
                if os.path.exists(precollected_obuffer_file):
                    print(f"   🔥 Loading pre-collected O-buffer from {precollected_obuffer_file}...")
                    load_success = onpolicy_buffer.load_from_file(precollected_obuffer_file, max_size=onpolicy_buffer_size)
                    if load_success and onpolicy_buffer.size() >= precollect_target_size * 0.9:
                        print(f"   ✅ Pre-collected O-buffer loaded: {onpolicy_buffer.size()}/{onpolicy_buffer_size} transitions")
                    else:
                        print(f"   ⚠️  File exists but size insufficient ({onpolicy_buffer.size()}/{precollect_target_size}), will re-collect")
                        load_success = False
                elif os.path.exists(checkpoint_file):
                    print(f"   🔥 Loading checkpoint: {checkpoint_file}")
                    load_success = onpolicy_buffer.load_from_file(checkpoint_file, max_size=onpolicy_buffer_size)
                    if load_success and onpolicy_buffer.size() >= precollect_target_size * 0.5:
                        print(f"   ✅ Checkpoint loaded: {onpolicy_buffer.size()}/{onpolicy_buffer_size} transitions")
                    else:
                        print(f"   ⚠️  Checkpoint size insufficient ({onpolicy_buffer.size()}/{precollect_target_size}), will re-collect")
                        load_success = False
                
                if not load_success:
                    print(f"   🔥 Pre-collected O-buffer not found or insufficient")
                    print(f"   Will collect new O-buffer with SL head...")
                    print(f"   Target: {precollect_target_size} transitions ({precollect_buffer_ratio*100:.0f}% of buffer = {onpolicy_buffer_size})")
                    print(f"   Current: {onpolicy_buffer.size()} transitions")
                    
                    # 🔥 执行预收集（复用 Stage-1 加载时的逻辑）
                    # 注意：这里假设模型已经从 checkpoint 恢复，可以用于生成路径
                    try:
                        # 🔥 从training data中提取所有唯一的起终点对（优先使用已加载的replay_buffer）
                        print(f"   📊 Extracting unique start-end pairs from training data...")
                        
                        if replay_buffer is not None:
                            # 使用已加载的replay_buffer（可能已经采样过）
                            train_replay_buffer = replay_buffer
                            print(f"   📊 Using provided replay_buffer ({len(train_replay_buffer)} transitions)")
                        else:
                            # 从文件读取（兼容旧代码）
                            train_replay_buffer_path = get_moe_data_path(data_directory, args.user_id, args.moe_data_dir, 'replay_buffer.df', args.graph_id)
                            if os.path.exists(train_replay_buffer_path):
                                train_replay_buffer = pd.read_pickle(train_replay_buffer_path)
                                print(f"   📊 Loaded replay_buffer from file ({len(train_replay_buffer)} transitions)")
                            else:
                                train_replay_buffer = pd.read_pickle(os.path.join(data_directory, 'replay_buffer.df'))
                                print(f"   📊 Loaded replay_buffer from standard path ({len(train_replay_buffer)} transitions)")
                        
                        start_end_pairs = set()
                        for _, row in train_replay_buffer.iterrows():
                            action = row['action']
                            if len(action) >= 15:
                                # 🔥 修复：使用固定索引，而不是负索引
                                # origin_x, origin_y 在索引 9, 10
                                # dest_x, dest_y 在索引 11, 12
                                start_x, start_y = float(action[9]), float(action[10])
                                end_x, end_y = float(action[11]), float(action[12])
                                start_end_pairs.add(((start_x, start_y), (end_x, end_y)))
                        
                        start_end_pairs = list(start_end_pairs)
                        print(f"   ✅ Found {len(start_end_pairs)} unique start-end pairs from training data")
                        
                        # 临时放宽duplicate限制
                        original_max_duplicate = onpolicy_buffer.max_duplicate_per_edge
                        onpolicy_buffer.max_duplicate_per_edge = 1000
                        print(f"   🔧 Temporarily relaxed max_duplicate_per_edge: {original_max_duplicate} → 1000")
                        
                        # 预收集参数
                        budget_B = 60
                        arrival_threshold_eval = 1e-6
                        total_collected = 0
                        total_added = 0
                        successful_paths = 0
                        failed_paths = 0
                        
                        # 对每个起终点对生成路径
                        for pair_idx, (start_pos, end_pos) in enumerate(start_end_pairs):
                            if onpolicy_buffer.size() >= precollect_target_size:
                                break
                            
                            try:
                                # 🔥 修复：ALWAYS构建正确的初始状态，不要从replay buffer中取
                                # replay buffer中的state可能是mid-trajectory状态，不是initial state
                                from utils.utility import pad_history
                                
                                # 构建初始状态（与build_path_data_for_batch格式一致）
                                cur_x, cur_y = start_pos[0], start_pos[1]  # 🔥 当前位置 = 起始位置
                                initial_state_frame = [0.0] * 9 + [start_pos[0], start_pos[1], end_pos[0], end_pos[1], cur_x, cur_y]
                                start_state = np.array(pad_history([initial_state_frame], state_size, [0.0] * 15))  # 🔥 Convert entire padded history
                                
                                # 🔥 如果启用了文本 embeddings，需要增强 start_state
                                if args.use_text_embeddings and text_embedding_loader is not None:
                                    # 检查 start_state 的维度
                                    if isinstance(start_state, np.ndarray) and start_state.ndim == 2:
                                        state_feature_dim = start_state.shape[1]
                                        # 如果是原始维度（15），需要增强
                                        if state_feature_dim == feature_dim:
                                            start_state = enhance_states_with_text_embeddings(
                                                np.array([start_state]), G, edge_id_map,
                                                text_embedding_loader=text_embedding_loader,
                                                embedding_proj_weights=embedding_proj_weights,
                                                embedding_proj_biases=embedding_proj_biases,
                                                original_feature_dim=feature_dim,
                                                user_embedding=user_embedding if args.use_user_embedding else None,
                                                user_embeddings_dict=user_embeddings_dict if args.use_user_embedding else None
                                            )[0]
                                        elif state_feature_dim != model_feature_dim:
                                            print(f"   ⚠️  [Pair {pair_idx}] State feature_dim mismatch: {state_feature_dim} != {model_feature_dim}")
                                elif args.use_user_embedding and user_embedding is not None:
                                    # 如果只启用了用户embedding，没有文本embedding
                                    start_state = add_user_embedding_to_states(np.array([start_state]), user_embedding)[0]
                                elif model_feature_dim != feature_dim:
                                    # 🔥 模型期望增强后的特征，但text embeddings未启用
                                    if pair_idx < 3:
                                        print(f"   ❌ [Pair {pair_idx}] Model expects feature_dim={model_feature_dim}, but text embeddings disabled (feature_dim={feature_dim})")
                                    raise ValueError(f"Model expects feature_dim={model_feature_dim}, but text embeddings are disabled (feature_dim={feature_dim}). "
                                                    f"Please enable text embeddings with --use_text_embeddings or retrain the model without text embeddings.")
                                
                                # 生成路径
                                predicted_path, _, _, path_success, trajectory = predict_path_with_model(
                                    sess, QN_1, start_state, end_pos, 
                                    max_steps=budget_B, G=G, edge_id_map=edge_id_map, item_num=item_num,
                                    use_rl_head=False, arrival_threshold=arrival_threshold_eval,
                                    text_embedding_loader=text_embedding_loader,
                                    embedding_proj_weights=embedding_proj_weights,
                                    embedding_proj_biases=embedding_proj_biases,
                                    user_id_for_model=None,  # 🔥 在训练阶段，使用None（非个性化模型）
                                    user_embedding=user_embedding  # 🔥 传递user_embedding
                                )
                                
                                if len(predicted_path) < 1:
                                    failed_paths += 1
                                    continue
                                
                                # 转换为transitions并添加到buffer
                                current_state = start_state.copy()
                                current_len_state = effective_len(current_state)
                                added_for_path = 0
                                
                                for step_idx, action_id in enumerate(predicted_path):
                                    if step_idx >= len(trajectory) - 1:
                                        break
                                    
                                    next_state = trajectory[step_idx + 1].copy()
                                    next_len_state = effective_len(next_state)
                                    is_done = (step_idx == len(predicted_path) - 1) or path_success
                                    
                                    reward = calculate_improved_reward(
                                        action_id, is_done, reward_goal,
                                        step_idx=step_idx, state_history=None, path_info=None,
                                        prev_action=None, edge_id_map=edge_id_map,
                                        step_penalty=-0.01, backtrack_penalty=-0.06,
                                        is_failure=False, failure_type=None,
                                        recent_visited_nodes=[], repeat_visit_penalty=-0.03
                                    )
                                    
                                    termination_type = 'success' if (is_done and path_success) else ('timeout' if (step_idx == len(predicted_path) - 1) else 'normal')
                                    task_key = f"{start_pos[0]}_{start_pos[1]}_{end_pos[0]}_{end_pos[1]}"
                                    
                                    added = onpolicy_buffer.add(
                                        current_state.copy(), action_id, reward,
                                        next_state.copy(), is_done,
                                        current_len_state, next_len_state,
                                        task_key=task_key, termination_type=termination_type, edge_action=None
                                    )
                                    if added:
                                        added_for_path += 1
                                    
                                    current_state = next_state.copy()
                                    current_len_state = next_len_state
                                
                                total_collected += len(predicted_path)
                                total_added += added_for_path
                                if path_success:
                                    successful_paths += 1
                                else:
                                    failed_paths += 1
                                
                                if (pair_idx + 1) % 100 == 0 or onpolicy_buffer.size() >= precollect_target_size:
                                    print(f"   Progress: {pair_idx + 1}/{len(start_end_pairs)} pairs, "
                                          f"buffer: {onpolicy_buffer.size()}/{precollect_target_size} transitions, ")
                                
                            except Exception as e:
                                failed_paths += 1
                                if (pair_idx + 1) % 100 == 0:
                                    print(f"   ⚠️  Error at pair {pair_idx}: {e}")
                                continue
                        
                        # 恢复duplicate限制并保存
                        onpolicy_buffer.max_duplicate_per_edge = original_max_duplicate
                        print(f"\n   ✅ Pre-collection complete!")
                        print(f"      Processed: {successful_paths + failed_paths}/{len(start_end_pairs)} pairs")
                        print(f"      Buffer size: {onpolicy_buffer.size()}/{precollect_target_size} transitions")
                        print(f"      Success: {successful_paths}, Failed: {failed_paths}")
                        
                        # 保存
                        try:
                            # 使用新的路径组织方式（按阶段和用户，带时间戳）
                            new_precollected_path = get_precollected_path(data_directory, args.user_id)
                            onpolicy_buffer.save_to_file(new_precollected_path)
                            print(f"      💾 Saved to {new_precollected_path}")
                        except Exception as e:
                            print(f"      ⚠️  Failed to save: {e}")
                            try:
                                checkpoint_file = precollected_obuffer_file.replace('.df', '_checkpoint.df')
                                onpolicy_buffer.save_to_file(checkpoint_file)
                                print(f"      💾 Saved checkpoint: {checkpoint_file}")
                            except:
                                pass
                        print()
                    except Exception as e:
                        print(f"   ❌ Pre-collection failed: {e}")
                        import traceback
                        traceback.print_exc()
                        print(f"   ⚠️  Will continue training without pre-collected buffer")
                        print()

        # 🔥 Eval-only mode: Skip training and go directly to final evaluation
        if args.eval_only:
            print("\n" + "="*60)
            print("EVAL-ONLY MODE: Skipping training, proceeding to final evaluation")
            print("="*60)
            
            # 🔥 修复：检查模型期望的特征维度，如果模型期望增强特征，必须提供text_embedding_loader
            model_expects_enhanced = QN_1.feature_dim > feature_dim
            if model_expects_enhanced:
                if text_embedding_loader is None:
                    print(f"⚠️  WARNING: Model expects feature_dim={QN_1.feature_dim} (enhanced), but text_embedding_loader is None.")
                    print(f"   Attempting to initialize text embedding loader from checkpoint or args...")
                    # 尝试从checkpoint加载投影权重，或者使用默认配置初始化
                    if args.text_embedding_cache_dir and os.path.exists(args.text_embedding_cache_dir):
                        try:
                            from text_embedding_loader import TextEmbeddingLoader
                            text_embedding_loader = TextEmbeddingLoader(
                                cache_dir=args.text_embedding_cache_dir,
                                preload_all=True
                            )
                            print(f"✅ Text embedding loader initialized from {args.text_embedding_cache_dir}")
                            
                            # 初始化投影权重（使用随机初始化，实际权重应该从checkpoint加载）
                            embedding_proj_weights = {}
                            embedding_proj_biases = {}
                            if args.use_edge_embedding or True:  # 默认启用edge
                                limit = np.sqrt(6.0 / (text_embedding_loader.hidden_dim + args.embedding_proj_dim))
                                embedding_proj_weights['edge'] = np.random.uniform(
                                    -limit, limit, (text_embedding_loader.hidden_dim, args.embedding_proj_dim)
                                ).astype(np.float32)
                                embedding_proj_biases['edge'] = np.zeros(args.embedding_proj_dim, dtype=np.float32)
                            if args.use_node_embedding or True:  # 默认启用node
                                limit = np.sqrt(6.0 / (text_embedding_loader.hidden_dim + args.embedding_proj_dim))
                                embedding_proj_weights['node'] = np.random.uniform(
                                    -limit, limit, (text_embedding_loader.hidden_dim, args.embedding_proj_dim)
                                ).astype(np.float32)
                                embedding_proj_biases['node'] = np.zeros(args.embedding_proj_dim, dtype=np.float32)
                            if args.use_goal_embedding or True:  # 默认启用goal
                                limit = np.sqrt(6.0 / (text_embedding_loader.hidden_dim + args.embedding_proj_dim))
                                embedding_proj_weights['goal'] = np.random.uniform(
                                    -limit, limit, (text_embedding_loader.hidden_dim, args.embedding_proj_dim)
                                ).astype(np.float32)
                                embedding_proj_biases['goal'] = np.zeros(args.embedding_proj_dim, dtype=np.float32)
                            
                            # 强制启用text embeddings
                            args.use_text_embeddings = True
                            args.use_edge_embedding = True
                            args.use_node_embedding = True
                            args.use_goal_embedding = True
                            print(f"   ✅ Forced enable text embeddings for evaluation")
                        except Exception as e:
                            print(f"❌ Failed to initialize text embedding loader: {e}")
                            raise RuntimeError(
                                f"Model expects feature_dim={QN_1.feature_dim} (enhanced), but cannot initialize text_embedding_loader. "
                                f"Please provide --text_embedding_cache_dir or ensure the model was trained with text embeddings."
                            )
                    else:
                        raise RuntimeError(
                            f"Model expects feature_dim={QN_1.feature_dim} (enhanced), but text_embedding_loader is None and "
                            f"--text_embedding_cache_dir is not provided. Please provide --text_embedding_cache_dir."
                        )
                else:
                    print(f"✅ Model expects enhanced features (dim={QN_1.feature_dim}), text_embedding_loader is available")
            
            # Ensure model is loaded
            if resume_ckpt is not None and os.path.exists(resume_ckpt + ".index"):
                if not (load_stage1_model and os.path.exists(stage1_model_path + '.meta')):
                    # Only restore if not already loaded via stage1_model
                    print(f"📥 Loading checkpoint for evaluation: {resume_ckpt}")
                    saver.restore(sess, resume_ckpt)
                    # Try to infer global_step from checkpoint name
                    import re
                    match = re.search(r'step[_\-]?(\d+)', resume_ckpt)
                    if match:
                        global_step = int(match.group(1))
                    else:
                        match = re.search(r'(\d+)[kK]', resume_ckpt)
                        if match:
                            global_step = int(match.group(1)) * 1000
                        else:
                            # Default to phase2 start step if can't infer
                            global_step = args.phase1_sl_only_steps
                    print(f"✅ Checkpoint loaded, inferred global_step={global_step}")
            elif load_stage1_model and os.path.exists(stage1_model_path + '.meta'):
                print(f"✅ Using already loaded Stage-1 model for evaluation")
            else:
                raise RuntimeError("❌ Eval-only mode requires a loaded checkpoint. Please provide --resume_ckpt or ensure checkpoint exists.")
            
            # 🔥 修复：在加载checkpoint后，检查模型期望的特征维度
            model_expects_enhanced = QN_1.feature_dim > feature_dim
            if model_expects_enhanced:
                if text_embedding_loader is None:
                    print(f"⚠️  WARNING: Model expects feature_dim={QN_1.feature_dim} (enhanced), but text_embedding_loader is None.")
                    print(f"   Attempting to initialize text embedding loader from checkpoint or args...")
                    # 尝试从checkpoint加载投影权重，或者使用默认配置初始化
                    if args.text_embedding_cache_dir and os.path.exists(args.text_embedding_cache_dir):
                        try:
                            from text_embedding_loader import TextEmbeddingLoader
                            text_embedding_loader = TextEmbeddingLoader(
                                cache_dir=args.text_embedding_cache_dir,
                                preload_all=True
                            )
                            print(f"✅ Text embedding loader initialized from {args.text_embedding_cache_dir}")
                            
                            # 初始化投影权重（使用随机初始化，实际权重应该从checkpoint加载）
                            embedding_proj_weights = {}
                            embedding_proj_biases = {}
                            if args.use_edge_embedding or True:  # 默认启用edge
                                limit = np.sqrt(6.0 / (text_embedding_loader.hidden_dim + args.embedding_proj_dim))
                                embedding_proj_weights['edge'] = np.random.uniform(
                                    -limit, limit, (text_embedding_loader.hidden_dim, args.embedding_proj_dim)
                                ).astype(np.float32)
                                embedding_proj_biases['edge'] = np.zeros(args.embedding_proj_dim, dtype=np.float32)
                            if args.use_node_embedding or True:  # 默认启用node
                                limit = np.sqrt(6.0 / (text_embedding_loader.hidden_dim + args.embedding_proj_dim))
                                embedding_proj_weights['node'] = np.random.uniform(
                                    -limit, limit, (text_embedding_loader.hidden_dim, args.embedding_proj_dim)
                                ).astype(np.float32)
                                embedding_proj_biases['node'] = np.zeros(args.embedding_proj_dim, dtype=np.float32)
                            if args.use_goal_embedding or True:  # 默认启用goal
                                limit = np.sqrt(6.0 / (text_embedding_loader.hidden_dim + args.embedding_proj_dim))
                                embedding_proj_weights['goal'] = np.random.uniform(
                                    -limit, limit, (text_embedding_loader.hidden_dim, args.embedding_proj_dim)
                                ).astype(np.float32)
                                embedding_proj_biases['goal'] = np.zeros(args.embedding_proj_dim, dtype=np.float32)
                            
                            # 强制启用text embeddings
                            args.use_text_embeddings = True
                            args.use_edge_embedding = True
                            args.use_node_embedding = True
                            args.use_goal_embedding = True
                            print(f"   ✅ Forced enable text embeddings for evaluation")
                        except Exception as e:
                            print(f"❌ Failed to initialize text embedding loader: {e}")
                            raise RuntimeError(
                                f"Model expects feature_dim={QN_1.feature_dim} (enhanced), but cannot initialize text_embedding_loader. "
                                f"Please provide --text_embedding_cache_dir or ensure the model was trained with text embeddings."
                            )
                    else:
                        raise RuntimeError(
                            f"Model expects feature_dim={QN_1.feature_dim} (enhanced), but text_embedding_loader is None and "
                            f"--text_embedding_cache_dir is not provided. Please provide --text_embedding_cache_dir."
                        )
                else:
                    print(f"✅ Model expects enhanced features (dim={QN_1.feature_dim}), text_embedding_loader is available")
            
            # Determine current phase for evaluation
            current_phase, phase_name, rl_weight, current_lr = determine_training_phase(global_step, args, load_stage1_model)
            print(f"📊 Evaluation will use: Phase={phase_name}, RL Weight={rl_weight}, LR={current_lr}")
            print("="*60 + "\n")
            
            # Skip to final evaluation (will be executed after the training loop block)
            pass
        else:
            # Normal training mode
            # Initialize training phase variables
            current_phase, phase_name, rl_weight, current_lr = determine_training_phase(global_step, args, load_stage1_model)
            
            # 🔥 标志：确保Stage-1消息只打印一次
            stage1_message_printed = False
            
            # 🔥 初始化：用于追踪step之间的时间间隔
            t_step_end_prev = None
            step_total_time_prev = None
            
            # 🔥 记录训练开始时的 global_step，用于计算训练循环内的实际步数
            training_start_step = global_step
            print(f"📊 Training will start from step {training_start_step:,}")
            print("="*60 + "\n")
            
            # 🔥 Early stopping initialization
            best_val_metric = -float('inf')  # Best validation metric value
            patience_counter = 0  # Number of evaluations without improvement
            best_model_step = 0  # Step where best model was found
            best_model_path = None  # Path to best model checkpoint
            early_stop_triggered = False  # Flag to indicate if early stopping was triggered
            
            if args.early_stopping:
                print(f"🛑 Early Stopping Enabled:")
                print(f"   Metric: {args.early_stopping_metric}")
                print(f"   Patience: {args.early_stopping_patience} evaluations (~{args.early_stopping_patience * args.eval_frequency_epoch:.1f} epochs)")
                print(f"   Min Delta: {args.early_stopping_min_delta}")
                print(f"   Start after: {args.early_stopping_start_epoch} epochs")
                print("="*60 + "\n")
            
            for i in range(args.epoch):
                for j in range(num_batches):
                    # 🔥 Stage-1: Save SL-only model at specified steps (only if not already loaded)
                    if (args.stage1_save_model and 
                        global_step == args.phase1_sl_only_steps and
                        not load_stage1_model and
                        not stage1_message_printed):  # 🔥 如果已经加载了Stage-1模型，跳过保存
                        print(f"\n💾 Saving Stage-1 SL-only model at step {global_step}...")
                        # 使用新的路径组织方式（按阶段和用户）
                        new_stage1_path = get_stage1_model_path(data_directory, args.user_id, global_step)
                        saver.save(sess, new_stage1_path)
                        print(f"✅ Stage-1 model saved to {new_stage1_path}")
                        print(f"🎯 Stage-1 SL-only training completed!")
                        # 🔥 Phase-2训练：随机起终点模式
                        print(f"🚀 Ready for Phase-2 Random Start-End RL training...")
                        stage1_message_printed = True
                        
                        # 🔥 修复：在Phase-2开始时检查并触发预收集O-buffer
                        if onpolicy_buffer.size() < args.onpolicy_buffer_size * 0.1:  # 如果O-buffer几乎为空
                            print(f"\n🔥 Phase-2 transition detected: Checking O-buffer pre-collection...")
                            print(f"   Current O-buffer size: {onpolicy_buffer.size()}/{args.onpolicy_buffer_size}")
                            
                            # 尝试加载预收集文件
                            precollected_obuffer_file = get_moe_data_path(data_directory, args.user_id, args.moe_data_dir, args.precollected_obuffer_path)
                            load_success = False
                            
                            if os.path.exists(precollected_obuffer_file):
                                print(f"   📂 Found pre-collected O-buffer: {precollected_obuffer_file}")
                                load_success = onpolicy_buffer.load_from_file(precollected_obuffer_file, max_size=args.onpolicy_buffer_size)
                                if load_success:
                                    print(f"   ✅ Loaded {onpolicy_buffer.size()}/{args.onpolicy_buffer_size} transitions")
                                else:
                                    print(f"   ⚠️  Failed to load pre-collected O-buffer")
                            
                            # 如果加载失败或文件不存在，执行预收集
                            if not load_success:
                                precollect_success = precollect_obuffer_at_phase2_transition(
                                    sess, QN_1, onpolicy_buffer, data_directory, args, args.user_id, args.moe_data_dir,
                                    G, edge_id_map, item_num, reward_goal, state_size, feature_dim, model_feature_dim,
                                    text_embedding_loader, embedding_proj_weights, embedding_proj_biases,
                                    precollected_obuffer_file=precollected_obuffer_file,
                                    replay_buffer=replay_buffer,  # 🔥 传递已采样的replay_buffer
                                    user_id_for_model=None,
                                    user_embedding=user_embedding if args.use_user_embedding else None,
                                    user_embeddings_dict=user_embeddings_dict if args.use_user_embedding else None
                                )
                                if not precollect_success:
                                    print(f"   ⚠️  Pre-collection failed, will collect during training")
                        
                        # 继续训练，不退出
                    elif (args.stage1_save_model and 
                          global_step == args.phase1_sl_only_steps and
                          load_stage1_model and
                          not stage1_message_printed):  # 🔥 如果已经加载了Stage-1模型，只打印一次
                        print(f"\n🎯 Stage-1 model already loaded, skipping save and continuing to Phase-2...")
                        # 🔥 Phase-2训练：随机起终点模式
                        print(f"🚀 Ready for Phase-2 Random Start-End RL training...")
                        stage1_message_printed = True
                        
                        # 🔥 修复：在Phase-2开始时检查并触发预收集O-buffer（与上面相同的逻辑）
                        if onpolicy_buffer.size() < args.onpolicy_buffer_size * 0.1:  # 如果O-buffer几乎为空
                            print(f"\n🔥 Phase-2 transition detected: Checking O-buffer pre-collection...")
                            print(f"   Current O-buffer size: {onpolicy_buffer.size()}/{args.onpolicy_buffer_size}")
                            
                            # 尝试加载预收集文件
                            precollected_obuffer_file = get_moe_data_path(data_directory, args.user_id, args.moe_data_dir, args.precollected_obuffer_path)
                            load_success = False
                            
                            if os.path.exists(precollected_obuffer_file):
                                print(f"   📂 Found pre-collected O-buffer: {precollected_obuffer_file}")
                                load_success = onpolicy_buffer.load_from_file(precollected_obuffer_file, max_size=args.onpolicy_buffer_size)
                                if load_success:
                                    print(f"   ✅ Loaded {onpolicy_buffer.size()}/{args.onpolicy_buffer_size} transitions")
                                else:
                                    print(f"   ⚠️  Failed to load pre-collected O-buffer")
                            
                            # 如果加载失败或文件不存在，执行预收集
                            if not load_success:
                                precollect_success = precollect_obuffer_at_phase2_transition(
                                    sess, QN_1, onpolicy_buffer, data_directory, args, args.user_id, args.moe_data_dir,
                                    G, edge_id_map, item_num, reward_goal, state_size, feature_dim, model_feature_dim,
                                    text_embedding_loader, embedding_proj_weights, embedding_proj_biases,
                                    precollected_obuffer_file=precollected_obuffer_file,
                                    replay_buffer=replay_buffer,  # 🔥 传递已采样的replay_buffer
                                    user_id_for_model=None,
                                    user_embedding=user_embedding if args.use_user_embedding else None,
                                    user_embeddings_dict=user_embeddings_dict if args.use_user_embedding else None
                                )
                                if not precollect_success:
                                    print(f"   ⚠️  Pre-collection failed, will collect during training")
                        
                        # 继续训练，不退出
                    # 🔥 DAgger-lite: Periodic on-policy collection (only in Phase 2+)
                    if (current_phase >= 2 and
                        global_step % args.onpolicy_collect_frequency == 0):
                        print(f"\n🔄 [Step {global_step}] Starting on-policy data collection...")
                        new_transitions = collect_onpolicy_transitions(
                            sess, QN_1, replay_buffer, G, edge_id_map, item_num,
                            args.onpolicy_collect_trajectories, args.onpolicy_max_steps,
                            reward_goal, args,
                            text_embedding_loader=text_embedding_loader if args.use_text_embeddings else None,
                            embedding_proj_weights=embedding_proj_weights if args.use_text_embeddings else None,
                            embedding_proj_biases=embedding_proj_biases if args.use_text_embeddings else None,
                            user_embedding=user_embedding if args.use_user_embedding else None,
                            user_embeddings_dict=user_embeddings_dict if args.use_user_embedding else None
                        )
                        print(f"  ✅ Collected {len(new_transitions)} transitions")
                        
                        # Add all transitions with labels (no hard filtering)
                        added_count = 0
                        skipped_duplicate = 0
                        for trans in new_transitions:
                            added = onpolicy_buffer.add(
                                trans['state'], trans['action'], trans['reward'],
                                trans['next_state'], trans['is_done'],
                                trans['len_state'], trans['len_next_state'],
                                task_key=trans.get('task_key', None),
                                termination_type=trans.get('termination_type', None),
                                edge_action=trans.get('edge_action', None)
                            )
                            if added:
                                added_count += 1
                            else:
                                skipped_duplicate += 1
                        
                        # Record collection batch for overlap calculation
                        onpolicy_buffer.record_collection_batch(new_transitions)
                    # 🔥 Strict separation: SL must always use FULL expert data (unfiltered E-bucket)
                    # Minimal change: regardless of Phase-2 fixed task, always sample globally for SL
                    expert_batch = replay_buffer.sample(n=args.batch_size).to_dict()
                    
                    # 🔥 For RL training: Mix expert (E-bucket) + on-policy (O-bucket)
                    # Determine mix ratio based on phase
                    if current_phase >= 2 and not onpolicy_buffer.is_empty():
                        # Calculate how many samples from each bucket
                        n_onpolicy = int(args.batch_size * args.onpolicy_mix_ratio)
                        n_expert = args.batch_size - n_onpolicy
                        
                        # Sample from expert (E-bucket)
                        # Random start-end mode: sample from all expert data
                        expert_rl_batch = replay_buffer.sample(n=n_expert).to_dict()
                        
                        # Sample from on-policy (O-bucket) with stratified sampling (6:4 success:non-success)
                        # Random start-end mode: sample from all tasks
                        onpolicy_samples = onpolicy_buffer.sample(n=n_onpolicy)                    
                        # Combine for RL training
                        # Handle both dict and list formats for expert_rl_batch
                        # 🔥 修复：新格式使用'taken_edge_id'，旧格式使用'action'
                        action_key = 'taken_edge_id' if 'taken_edge_id' in expert_rl_batch else 'action'
                        
                        if isinstance(expert_rl_batch['state'], dict):
                            expert_states = list(expert_rl_batch['state'].values())
                            expert_actions = list(expert_rl_batch[action_key].values())
                            expert_next_states = list(expert_rl_batch['next_state'].values())
                            expert_len_states = list(expert_rl_batch['len_state'].values())
                            expert_len_next_states = list(expert_rl_batch['len_next_state'].values())
                            expert_is_dones = list(expert_rl_batch.get('done', expert_rl_batch.get('is_done', {})).values())
                        else:
                            expert_states = expert_rl_batch['state']
                            expert_actions = expert_rl_batch[action_key]
                            expert_next_states = expert_rl_batch['next_state']
                            expert_len_states = expert_rl_batch['len_state']
                            expert_len_next_states = expert_rl_batch['len_next_state']
                            expert_is_dones = expert_rl_batch.get('done', expert_rl_batch.get('is_done', [False] * len(expert_states)))
                        
                        # 🔥 修复：处理新格式（taken_edge_id）和旧格式（action）
                        onpolicy_actions = []
                        for t in onpolicy_samples:
                            if 'taken_edge_id' in t:
                                onpolicy_actions.append(t['taken_edge_id'])
                            elif 'action' in t:
                                onpolicy_actions.append(t['action'])
                            else:
                                # 如果都没有，使用0作为默认值
                                onpolicy_actions.append(0)
                        
                        rl_batch = {
                            'state': expert_states + [t['state'] for t in onpolicy_samples],
                            action_key: expert_actions + onpolicy_actions,  # 使用统一的键
                            'next_state': expert_next_states + [t['next_state'] for t in onpolicy_samples],
                            'len_state': expert_len_states + [t['len_state'] for t in onpolicy_samples],
                            'len_next_state': expert_len_next_states + [t['len_next_state'] for t in onpolicy_samples],
                            'is_done': expert_is_dones + [t.get('done', t.get('is_done', False)) for t in onpolicy_samples]
                        }
                    else:
                        # Phase 1 or empty O-bucket: RL uses only expert data
                        rl_batch = expert_batch
                    
                    # Extract data for training
                    # For SL: use expert_batch (E-bucket only, ALWAYS unfiltered)
                    # For RL: use rl_batch (E-bucket + O-bucket mix)
                    
                    # 🔥 根据训练阶段选择batch（仅用于RL训练计算，SL训练单独使用expert_batch）
                    if current_phase >= 2:
                        # Phase-2+: batch用于RL训练（包含过滤后的E+O混合数据）
                        # 注意：SL训练在后续代码中单独使用expert_batch，不受此batch影响
                        batch = rl_batch
                        # 🔥 检查RL batch是否为空（Phase-2固定任务模式可能没有expert数据）
                        if len(batch['state']) == 0:
                            print(f"⚠️  Empty RL batch at step {global_step}")
                            print(f"   Expert batch size: {len(expert_batch['state'])}")
                            print(f"   Onpolicy buffer size: {onpolicy_buffer.size()}")
                            print(f"   Current phase: {current_phase}")
                            print(f"   Skipping training, but incrementing global_step")
                            # 🔥 即使batch为空，也要递增global_step，避免无限循环
                            global_step += 1
                            continue
                    else:
                        # Phase-1: batch用于SL训练（全部都是expert数据，rl_batch = expert_batch）
                        batch = expert_batch
                        # 🔥 检查SL batch是否为空
                        if len(batch['state']) == 0:
                            print(f"⚠️  Empty SL batch at step {global_step}, skipping training, but incrementing global_step")
                            # 🔥 即使batch为空，也要递增global_step，避免无限循环
                            global_step += 1
                            continue
                    
                    # 🔥 修复：处理batch数据格式，支持list、dict、pandas Series和numpy array
                    def extract_batch_field(field_data):
                        """从batch中提取字段，支持list、dict、pandas Series和numpy array格式"""
                        if isinstance(field_data, list):
                            return field_data
                        elif isinstance(field_data, dict):
                            return list(field_data.values())
                        elif isinstance(field_data, pd.Series):
                            # pandas Series：使用 .tolist() 转换
                            return field_data.tolist()
                        elif isinstance(field_data, np.ndarray):
                            # 如果是numpy数组，转换为list（每个元素是一个样本）
                            if field_data.ndim > 1:
                                return [field_data[i] for i in range(len(field_data))]
                            else:
                                return field_data.tolist()
                        else:
                            # 其他类型，尝试转换为list
                            return [field_data] if not isinstance(field_data, (list, tuple)) else list(field_data)
                    
                    # 🔥 统一规范化函数：安全地将state转换为numpy数组，避免警告
                    def normalize_state_batch(state_list, state_size, feature_dim, name="state"):
                        """
                        规范化state批次数据，统一格式为numpy数组
                        处理list、tuple、numpy array等不同格式，避免不规则嵌套序列警告
                        """
                        if isinstance(state_list, np.ndarray):
                            # 已经是numpy数组，检查是否需要reshape
                            if state_list.ndim == 1:
                                if len(state_list) == state_size * feature_dim:
                                    return state_list.reshape(1, state_size, feature_dim)
                                elif len(state_list) == state_size:
                                    original_state = state_list.copy()
                                    normalized = np.zeros((1, state_size, feature_dim))
                                    normalized[0, 0, :min(len(original_state), feature_dim)] = original_state[:min(len(original_state), feature_dim)]
                                    return normalized
                                else:
                                    raise ValueError(f"Cannot handle 1D {name} array with shape {state_list.shape}, expected length {state_size * feature_dim} or {state_size}")
                            elif state_list.ndim == 2:
                                if state_list.shape[1] == state_size * feature_dim:
                                    return state_list.reshape(-1, state_size, feature_dim)
                                elif state_list.shape == (state_size, feature_dim):
                                    return state_list.reshape(1, state_size, feature_dim)
                                else:
                                    raise ValueError(f"Cannot handle 2D {name} array with shape {state_list.shape}, expected ({state_size}, {feature_dim}) or (batch_size, {state_size * feature_dim})")
                            # 如果已经是3D数组，假设格式正确
                            return state_list
                        
                        if not isinstance(state_list, (list, tuple)):
                            # 单个元素，转换为list
                            state_list = [state_list]
                        
                        # 检查是否包含不规则嵌套序列（可能导致警告）
                        # 先尝试逐个规范化，避免直接使用np.array()导致警告
                        normalized_states = []
                        for s in state_list:
                            # 统一转换为numpy array：处理list、tuple、numpy array
                            if isinstance(s, (list, tuple)):
                                # 先转换为numpy array（单个元素，不会产生警告）
                                s_arr = np.array(s, dtype=np.float32)
                            elif isinstance(s, np.ndarray):
                                s_arr = s.astype(np.float32) if s.dtype != np.float32 else s
                            else:
                                raise TypeError(f"Unexpected {name} element type: {type(s)}, expected list, tuple, or numpy array")
                            
                            # 规范化形状
                            if s_arr.ndim == 1:
                                if len(s_arr) == state_size * feature_dim:
                                    s_arr = s_arr.reshape(state_size, feature_dim)
                                elif len(s_arr) == state_size:
                                    original_vals = s_arr.copy()
                                    s_arr = np.zeros((state_size, feature_dim), dtype=np.float32)
                                    s_arr[0, :min(len(original_vals), feature_dim)] = original_vals[:min(len(original_vals), feature_dim)]
                                else:
                                    raise ValueError(f"Cannot normalize {name} with shape {s_arr.shape} to ({state_size}, {feature_dim})")
                            elif s_arr.ndim == 2:
                                if s_arr.shape != (state_size, feature_dim):
                                    if s_arr.shape[0] == state_size and s_arr.shape[1] != feature_dim:
                                        if s_arr.shape[1] < feature_dim:
                                            padding = np.zeros((state_size, feature_dim - s_arr.shape[1]), dtype=np.float32)
                                            s_arr = np.concatenate([s_arr, padding], axis=1)
                                        else:
                                            s_arr = s_arr[:, :feature_dim]
                                    elif s_arr.shape[0] != state_size:
                                        raise ValueError(f"Cannot normalize {name} with shape {s_arr.shape} to ({state_size}, {feature_dim})")
                            else:
                                raise ValueError(f"Unexpected {name} dimension: {s_arr.ndim}, shape: {s_arr.shape}")
                            
                            normalized_states.append(s_arr)
                        
                        # 现在所有states都有相同的形状，可以安全地stack
                        if len(normalized_states) == 0:
                            raise ValueError(f"Empty {name} list")
                        return np.stack(normalized_states)
                    
                    next_state = extract_batch_field(batch['next_state'])
                    len_next_state = extract_batch_field(batch['len_next_state'])
                    state = extract_batch_field(batch['state'])
                    len_state = extract_batch_field(batch['len_state'])
                    # 修复：replay_buffer 使用 'done' 而不是 'is_done'
                    is_done = extract_batch_field(batch.get('done', batch.get('is_done', [False] * len(state))))
                    # 🔥 提取user_id用于数据标识
                    user_ids = extract_batch_field(batch.get('user_id', [0] * len(state)))  # 默认user_id=0如果没有
                    
                    # 🔥 统一规范化：使用统一的规范化函数，避免警告
                    state = normalize_state_batch(state, state_size, feature_dim, name="state")
                    next_state = normalize_state_batch(next_state, state_size, feature_dim, name="next_state")

                    # 🔥 新增：重建候选边特征用于候选边排序训练
                    # 从batch数据中提取候选边信息和图上下文
                    # 🔥 修复：使用args.max_candidates而不是mainQN.max_candidates（mainQN此时还未定义）
                    max_candidates = args.max_candidates if hasattr(args, 'max_candidates') else 20
                    
                    graph_ids_batch = extract_batch_field(batch['graph_id'])  # 多图支持
                    cand_edge_ids_batch = extract_batch_field(batch['cand_edge_ids'])  # 每个元素是list
                    cur_node_ids_batch = extract_batch_field(batch['cur_node_id'])
                    goal_node_ids_batch = extract_batch_field(batch['goal_node_id'])
                    d_start_batch = extract_batch_field(batch.get('d_start', [100.0] * len(cand_edge_ids_batch)))
                    prev_node_ids_batch = extract_batch_field(batch.get('prev_node_id', [None] * len(cand_edge_ids_batch)))
                    recent_visited_batch = extract_batch_field(batch.get('recent_visited_nodes', [None] * len(cand_edge_ids_batch)))

                    # 为每个batch项重建候选边特征 [batch_size, max_candidates, feature_dim]
                    cand_features_batch = []
                    cand_masks_batch = []

                    for b in range(len(cand_edge_ids_batch)):
                        graph_id = graph_ids_batch[b] if b < len(graph_ids_batch) else 'default_graph'
                        cand_edge_ids = cand_edge_ids_batch[b]
                        cur_node_id = cur_node_ids_batch[b]
                        goal_node_id = goal_node_ids_batch[b]
                        d_start = d_start_batch[b]
                        prev_node_id = prev_node_ids_batch[b]
                        recent_visited = recent_visited_batch[b]

                        # TODO: 根据graph_id选择对应的GraphCache和FeatureBuilder
                        # 目前所有数据都来自同一个图，所以使用同一个cache
                        current_cache = graph_cache
                        current_builder = feature_builder

                        # 重建候选边特征（传入d_start和历史信息）
                        cand_features = current_builder.build_candidate_features(
                            cand_edge_ids=cand_edge_ids,
                            cur_node_id=cur_node_id,
                            goal_node_id=goal_node_id,
                            d_start=d_start,
                            prev_node_id=prev_node_id,
                            recent_visited_nodes=recent_visited
                        )

                        # 填充到固定大小 [max_candidates, feature_dim]
                        cand_features_padded = current_builder.pad_candidate_features(
                            cand_features, max_candidates
                        )
                        cand_features_batch.append(cand_features_padded)

                        # 创建mask [max_candidates]
                        cand_mask = current_builder.create_candidate_mask(
                            len(cand_edge_ids), max_candidates
                        )
                        cand_masks_batch.append(cand_mask)

                    # 转换为numpy数组
                    cand_features_batch = np.array(cand_features_batch, dtype=np.float32)
                    cand_masks_batch = np.array(cand_masks_batch, dtype=np.float32)
                    
                    # 🔥 如果启用了文本 embeddings，增强状态数据
                    if args.use_text_embeddings and text_embedding_loader is not None:
                        state = enhance_states_with_text_embeddings(
                        state, G, edge_id_map,
                        text_embedding_loader=text_embedding_loader,
                        embedding_proj_weights=embedding_proj_weights,
                        embedding_proj_biases=embedding_proj_biases,
                        original_feature_dim=feature_dim,
                        user_embedding=user_embedding,
                        user_embeddings_dict=user_embeddings_dict,
                        user_ids=user_ids
                    )
                    next_state = enhance_states_with_text_embeddings(
                        next_state, G, edge_id_map,
                        text_embedding_loader=text_embedding_loader,
                        embedding_proj_weights=embedding_proj_weights,
                        embedding_proj_biases=embedding_proj_biases,
                        original_feature_dim=feature_dim,
                        user_embedding=user_embedding,
                        user_embeddings_dict=user_embeddings_dict,
                        user_ids=user_ids
                    )

                    mainQN = QN_1
                    target_QN = QN_2
                    if global_step % 100 == 0:
                        sess.run(soft_update_ops)
                        # 🔥 Actor-Critic DDPG-style: Update Actor target with Polyak averaging
                        sess.run(actor_target_soft_update_ops)

                    current_phase, phase_name, rl_weight, current_lr = determine_training_phase(global_step, args, load_stage1_model)
                    
                    # 生成next state的action masks (for RL training)
                    next_action_masks = generate_action_mask_batch(next_state, G, edge_id_map, item_num)
                    
                    # 🔥 修复：计算target Q时不使用dropout（is_training=False）
                    # 避免bootstrap目标中引入噪声
                    next_state_feed_dict = {
                        target_QN.inputs: next_state,
                        target_QN.len_state: len_next_state, 
                        target_QN.action_mask: next_action_masks,
                        target_QN.is_training: False,  # 🔥 修复：target不用dropout
                        target_QN.training_phase: current_phase,
                        target_QN.rl_weight: rl_weight,
                        mainQN.inputs: next_state,
                        mainQN.len_state: len_next_state,
                        mainQN.action_mask: next_action_masks,
                        mainQN.is_training: False,  # 🔥 修复：target不用dropout
                        mainQN.training_phase: current_phase,
                        mainQN.rl_weight: rl_weight
                    }
                    # 🔥 Actor-Critic DDPG-style: Get next_states_hidden and target Qs
                    next_states_hidden, target_Qs, target_Qs_selector = sess.run(
                        [mainQN.states_hidden, target_QN.output1_for_training, mainQN.output1_for_training],
                        feed_dict=next_state_feed_dict
                    )
                    
                    # 🔥 Actor-Critic DDPG-style: Compute Actor target probs π_tgt(a'|s')
                    # Only compute in Phase 2+ (RL training), use dummy values in Phase 1 (SL-only)
                    if current_phase >= 2:
                        actor_target_feed_dict = {
                            mainQN.next_states_hidden: next_states_hidden,
                            mainQN.next_action_mask: next_action_masks,
                            mainQN.is_training: False,  # Actor target doesn't use dropout
                            mainQN.training_phase: current_phase,
                            mainQN.rl_weight: rl_weight
                        }
                        actor_target_probs = sess.run(mainQN.actor_target_output, feed_dict=actor_target_feed_dict)
                    else:
                        # Phase 1 (SL-only): Use dummy uniform distribution
                        batch_size = target_Qs.shape[0]
                        actor_target_probs = np.ones((batch_size, item_num), dtype=np.float32) / item_num

                    # 🔥 修复：终止状态已经通过discount=0处理，不需要再清零target_Qs
                    # 删除冗余的target_Qs[terminal_mask]=0

                    # 生成current state的action masks
                    current_action_masks = generate_action_mask_batch(state, G, edge_id_map, item_num, debug_output=True)
                    
                    # 提取动作信息
                    # 🔥 修复：新格式使用'taken_edge_id'，旧格式使用'action'
                    if 'taken_edge_id' in batch:
                        # 新格式：直接使用taken_edge_id（已经是edge ID）
                        if isinstance(batch['taken_edge_id'], list):
                            action_ids = batch['taken_edge_id']
                        else:
                            action_ids = list(batch['taken_edge_id'].values())
                    elif 'action' in batch:
                        # 旧格式：从action中提取edge ID
                        if isinstance(batch['action'], list):
                            action = batch['action']
                        else:
                            action = list(batch['action'].values())
                        action_ids = extract_action_ids(action, edge_id_map)
                    else:
                        raise KeyError("Batch must contain either 'taken_edge_id' (new format) or 'action' (old format)")
                    

                    batch_size = len(state)
                    
                    # 🔥 路线A：邻居覆盖TD（替代原有的随机负样本）
                    # 🔥 修复：固定邻居列数 = args.neg，保持和TRFL循环一致
                    max_neighbors_per_sample = args.neg  # 固定列数，避免维度不匹配
                    neighbor_samples = []
                    neighbor_counts = []  # 记录每个样本的实际邻居数
                    
                    for idx in range(len(state)):
                        action_id = action_ids[idx]
                        current_mask = current_action_masks[idx]
                        valid_edge_ids = np.where(current_mask == 1)[0].tolist()
                        
                        # 使用邻居覆盖策略（包含专家动作）
                        neighbor_ids = generate_neighbor_actions(action_id, valid_edge_ids, max_neighbors_per_sample)
                        
                        # 🔥 修复：始终填充/截断到固定长度 args.neg
                        if len(neighbor_ids) < args.neg:
                            # 使用item_num作为越界ID（而不是0，避免与真实动作混淆）
                            padded_neighbors = neighbor_ids + [item_num] * (args.neg - len(neighbor_ids))
                        else:
                            # 截断到args.neg
                            padded_neighbors = neighbor_ids[:args.neg]
                        
                        neighbor_samples.append(padded_neighbors)
                        neighbor_counts.append(len(neighbor_ids))
                    
                    # 🔥 修复：max_neighbors_in_batch 现在是固定的 args.neg
                    max_neighbors_in_batch = args.neg
                    padded_neighbor_samples = neighbor_samples
                    
                    # 计算邻居的下一状态（基于已知图转移）
                    neighbor_next_states = compute_next_states_for_negative_samples(
                        np.array(state), np.array(padded_neighbor_samples), edge_id_map, G, state_size,
                        text_embedding_loader=text_embedding_loader if args.use_text_embeddings else None,
                        embedding_proj_weights=embedding_proj_weights if args.use_text_embeddings else None,
                        embedding_proj_biases=embedding_proj_biases if args.use_text_embeddings else None,
                        user_embedding=user_embedding
                    )
                    neighbor_next_len_states = np.array([[min(len_state[i] + 1, state_size)] * max_neighbors_in_batch 
                                                        for i in range(len(len_state))])
                    
                    # 计算邻居的target Q值
                    neighbor_target_Qs = []
                    all_neighbor_states = []
                    all_neighbor_len_states = []
                    for i in range(max_neighbors_in_batch):
                        neighbor_next_state = neighbor_next_states[:, i, :, :]
                        neighbor_next_len_state = neighbor_next_len_states[:, i]
                        all_neighbor_states.append(neighbor_next_state)
                        all_neighbor_len_states.append(neighbor_next_len_state)
                    
                    if all_neighbor_states:
                        combined_neighbor_states = np.concatenate(all_neighbor_states, axis=0)
                        combined_neighbor_len_states = np.concatenate(all_neighbor_len_states, axis=0)
                    else:
                        combined_neighbor_states = np.zeros((0, state_size, feature_dim), dtype=np.float32)
                        combined_neighbor_len_states = np.zeros(0, dtype=np.int32)
                    
                    combined_neighbor_masks = generate_action_mask_batch(combined_neighbor_states, G, edge_id_map, item_num, debug_output=False)
                    
                    # 🔥 修复：Double Q - selector用main Q，价值用target Q
                    # Selector Q: 用main Q选择最优动作
                    combined_neighbor_selector_Q = sess.run(mainQN.output1_for_training,
                                                feed_dict={
                                                    mainQN.inputs: combined_neighbor_states,
                                                    mainQN.len_state: combined_neighbor_len_states,
                                                    mainQN.action_mask: combined_neighbor_masks,
                                                    mainQN.is_training: False,
                                                    mainQN.training_phase: current_phase,
                                                    mainQN.rl_weight: rl_weight
                                                })
                    
                    # Target Q: 用target Q估值
                    combined_neighbor_target_Q = sess.run(target_QN.output1_for_training,
                                                feed_dict={
                                                    target_QN.inputs: combined_neighbor_states,
                                                    target_QN.len_state: combined_neighbor_len_states,
                                                    target_QN.action_mask: combined_neighbor_masks,
                                                    target_QN.is_training: False,
                                                    target_QN.training_phase: current_phase,
                                                    target_QN.rl_weight: rl_weight
                                                })
                    
                    # 🔥 Actor-Critic DDPG-style: Compute Actor target probs for neighbors
                    # Only compute in Phase 2+ (RL training)
                    if current_phase >= 2:
                        # 🔥 Actor-Critic DDPG-style: Get hidden features for Actor target (neighbors)
                        combined_neighbor_hidden = sess.run(mainQN.states_hidden,
                                                    feed_dict={
                                                        mainQN.inputs: combined_neighbor_states,
                                                        mainQN.len_state: combined_neighbor_len_states,
                                                        mainQN.action_mask: combined_neighbor_masks,
                                                        mainQN.is_training: False,
                                                        mainQN.training_phase: current_phase,
                                                        mainQN.rl_weight: rl_weight
                                                    })
                        
                        # 🔥 Actor-Critic DDPG-style: Compute Actor target probs for neighbors
                        combined_neighbor_actor_target_probs = sess.run(mainQN.actor_target_output,
                                                                    feed_dict={
                                                                        mainQN.next_states_hidden: combined_neighbor_hidden,
                                                                        mainQN.next_action_mask: combined_neighbor_masks,
                                                                        mainQN.is_training: False,
                                                                        mainQN.training_phase: current_phase,
                                                                        mainQN.rl_weight: rl_weight
                                                                    })
                        
                        # 拆分邻居Q值和Actor概率（selector, target, actor_probs）
                        neighbor_selector_Qs = []
                        neighbor_actor_target_probs_list = []
                        for neighbor_idx in range(max_neighbors_in_batch):
                            start_idx = neighbor_idx * batch_size
                            end_idx = (neighbor_idx + 1) * batch_size
                            neighbor_target_Q = combined_neighbor_target_Q[start_idx:end_idx]
                            neighbor_selector_Q = combined_neighbor_selector_Q[start_idx:end_idx]
                            neighbor_actor_probs = combined_neighbor_actor_target_probs[start_idx:end_idx]
                            neighbor_target_Qs.append(neighbor_target_Q)
                            neighbor_selector_Qs.append(neighbor_selector_Q)
                            neighbor_actor_target_probs_list.append(neighbor_actor_probs)
                    else:
                        # Phase 1 (SL-only): Use dummy values for neighbors
                        neighbor_target_Qs = []
                        neighbor_selector_Qs = []
                        neighbor_actor_target_probs_list = []
                        for neighbor_idx in range(max_neighbors_in_batch):
                            start_idx = neighbor_idx * batch_size
                            end_idx = (neighbor_idx + 1) * batch_size
                            neighbor_target_Q = combined_neighbor_target_Q[start_idx:end_idx]
                            neighbor_selector_Q = combined_neighbor_selector_Q[start_idx:end_idx]
                            # Dummy uniform distribution for Phase 1
                            dummy_actor_probs = np.ones((batch_size, item_num), dtype=np.float32) / item_num
                            neighbor_target_Qs.append(neighbor_target_Q)
                            neighbor_selector_Qs.append(neighbor_selector_Q)
                            neighbor_actor_target_probs_list.append(dummy_actor_probs)
                    
                    # 🔥 处理无效邻居（填充的越界ID）和邻居终止状态
                    padded_neighbor_array = np.array(padded_neighbor_samples)
                    neighbor_is_done_list = []
                    
                    # 🔥 预先计算path_infos，用于邻居终止状态检测
                    path_infos = []
                    for k in range(len(is_done)):
                        path_info = {}
                        if k < len(state) and len_state[k] > 0:
                            # 获取第一个非零状态（初始状态）
                            initial_state = None
                            for i in range(len(state[k])):
                                if not np.all(state[k][i] == 0):
                                    initial_state = state[k][i]
                                    break
                            
                            if initial_state is not None and len(initial_state) >= 15:
                                # 🔥 修复：使用固定索引，而不是负索引
                                # origin_x, origin_y 在索引 9, 10
                                # dest_x, dest_y 在索引 11, 12
                                start_x, start_y = initial_state[9], initial_state[10]
                                end_x, end_y = initial_state[11], initial_state[12]
                                path_info['start_pos'] = (start_x, start_y)
                                path_info['end_pos'] = (end_x, end_y)
                                dx, dy = end_x - start_x, end_y - start_y
                                target_length = max(1, int(np.sqrt(dx*dx + dy*dy) / 0.1))
                                path_info['target_length'] = target_length
                            else:
                                # 如果无法从初始状态提取路径信息，使用默认值
                                path_info['start_pos'] = (0.0, 0.0)
                                path_info['end_pos'] = (1.0, 1.0)
                                path_info['target_length'] = 10
                        else:
                            # 如果状态为空，使用默认值
                            path_info['start_pos'] = (0.0, 0.0)
                            path_info['end_pos'] = (1.0, 1.0)
                            path_info['target_length'] = 10
                        path_infos.append(path_info)
                    
                    for neighbor_idx in range(max_neighbors_in_batch):
                        # 🔥 计算邻居的终止状态（基于邻居的next_state）
                        neighbor_is_done = np.zeros(batch_size, dtype=bool)
                        for k in range(batch_size):
                            neighbor_action_id = padded_neighbor_array[k, neighbor_idx]
                            if neighbor_action_id < item_num:  # 非填充ID
                                # 检查邻居动作是否到达目标
                                neighbor_next_state = neighbor_next_states[k, neighbor_idx]
                                neighbor_next_pos = extract_position_from_state_for_shaping(neighbor_next_state)
                                if neighbor_next_pos is not None and k < len(path_infos):
                                    target_pos = path_infos[k].get('end_pos')
                                    if target_pos is not None:
                                        dist_to_target = np.hypot(neighbor_next_pos[0] - target_pos[0], 
                                                                neighbor_next_pos[1] - target_pos[1])
                                        neighbor_is_done[k] = dist_to_target < 0.1  # 到达目标阈值
                        
                        neighbor_is_done_list.append(neighbor_is_done)
                        
                        # 🔥 越界ID（填充）或邻居终止状态的Q值设为0
                        invalid_mask = (padded_neighbor_array[:, neighbor_idx] >= item_num) | neighbor_is_done
                        neighbor_target_Qs[neighbor_idx][invalid_mask] = 0.0
                        neighbor_selector_Qs[neighbor_idx][invalid_mask] = 0.0

                    predictions = sess.run(
                        mainQN.probs,
                        feed_dict={
                            mainQN.inputs: state,
                            mainQN.len_state: len_state,
                            mainQN.action_mask: current_action_masks,
                            mainQN.is_training: False,
                            mainQN.training_phase: current_phase,
                            mainQN.rl_weight: rl_weight,
                        }
                    )
                    top1_preds = np.argmax(predictions, axis=1)
                    
                    reward = []
                    
                    step_indices = list(batch.get('step_idx', {}).values()) if 'step_idx' in batch else [0] * batch_size
                    step_indices_array = np.array(step_indices[:batch_size] if len(step_indices) >= batch_size else [0] * batch_size)
                    
                    
                    # 计算奖励
                    # 训练策略：SL学习模仿，RL学习可达性
                    for k in range(batch_size):
                        current_step_idx = step_indices_array[k]
                        current_path_info = path_infos[k] if k < len(path_infos) else {}
                        
                        # 推断上一步动作用于回退检测
                        prev_action_k = None
                        if k < len(state) and len_state[k] > 1:
                            prev_action_k = infer_prev_action_from_state_history(state[k], edge_id_map)
                        
                        # 计算正样本奖励（用户动作）
                        reward_value, reward_comps = calculate_improved_reward(
                            action_ids[k], is_done[k],
                            reward_goal,
                            current_step_idx, next_state[k], current_path_info,  # 🔥 修复：使用next_state而不是state
                            prev_action=prev_action_k, edge_id_map=edge_id_map,
                            step_penalty=args.r_step,
                            backtrack_penalty=args.r_backtrack
                        )
                        reward.append(reward_value)
                    
                    # 🔥 步骤2&3：检测坏动作并应用终止强负
                    # 参数设置
                    K_RECENT = 10  # 最近K步回环检测窗口
                    C_LOOP = 0.5  # 终止强负幅度
                    
                    # 🔥 提取最近K步的edge_id列表（用于回环检测）
                    # 由于是无向图，直接记录最近K步走过的edge_id
                    recent_edge_ids_batch = []
                    for k in range(batch_size):
                        if k < len(state):
                            edge_ids = extract_recent_edge_ids_from_state(state[k], edge_id_map, K=K_RECENT)
                            recent_edge_ids_batch.append(edge_ids)
                        else:
                            recent_edge_ids_batch.append([])
                    
                    # 计算邻居动作的奖励（基于已知图转移）
                    neighbor_rewards_list = []
                    bad_action_counts = {'reverse': 0, 'revisit': 0, 'total': 0}  # 统计坏动作数量
                    
                    for neighbor_idx in range(max_neighbors_in_batch):
                        neighbor_rewards_for_this_idx = []
                        for k in range(batch_size):
                            current_step_idx = step_indices_array[k]
                            neighbor_action_id = padded_neighbor_array[k, neighbor_idx]
                            current_path_info = path_infos[k] if k < len(path_infos) else {}
                            
                            # 如果是越界ID（填充），奖励设为0
                            if neighbor_action_id >= item_num:
                                neighbor_rewards_for_this_idx.append(0.0)
                            else:
                                # 🔥 检测是否为坏动作（回环）
                                # 策略：检查当前edge_id是否在最近K步中已经走过
                                # 由于是无向图，重复走同一条边就是回环
                                is_bad_action = False
                                
                                if len(recent_edge_ids_batch[k]) > 0:
                                    if neighbor_action_id in recent_edge_ids_batch[k]:
                                        is_bad_action = True
                                        bad_action_counts['revisit'] += 1
                                        bad_action_counts['total'] += 1
                                
                                # 🔥 根据是否为坏动作设置奖励
                                if is_bad_action:
                                    # 坏动作：终止强负（reward=-C_LOOP）
                                    neighbor_reward_value = -C_LOOP
                                else:
                                    # 🔥 正常动作：使用邻居自己的next_state计算奖励
                                    neighbor_next_state = neighbor_next_states[k, neighbor_idx]
                                    neighbor_next_pos = extract_position_from_state_for_shaping(neighbor_next_state)
                                    
                                    # 计算邻居的距离改善
                                    neighbor_distance_improvement = 0.0
                                    if neighbor_next_pos is not None and k < len(path_infos):
                                        target_pos = path_infos[k].get('end_pos')
                                        if target_pos is not None:
                                            # 当前状态到目标的距离
                                            current_pos = extract_position_from_state_for_shaping(state[k])
                                            if current_pos is not None:
                                                current_dist = np.hypot(current_pos[0] - target_pos[0], current_pos[1] - target_pos[1])
                                                neighbor_dist = np.hypot(neighbor_next_pos[0] - target_pos[0], neighbor_next_pos[1] - target_pos[1])
                                                neighbor_distance_improvement = current_dist - neighbor_dist
                                    
                                    # 使用邻居的距离改善计算奖励
                                    neighbor_path_info = current_path_info.copy()
                                    neighbor_path_info['distance_improvement'] = neighbor_distance_improvement
                                    # 推断上一步动作用于回退检测
                                    prev_action_k = None
                                    if k < len(state) and len_state[k] > 1:
                                        prev_action_k = infer_prev_action_from_state_history(state[k], edge_id_map)
                                    neighbor_reward_value, _ = calculate_improved_reward(
                                        action_id=neighbor_action_id, 
                                        is_done=False,  # 邻居动作不会导致终止
                                        reward_goal=reward_goal,
                                        step_idx=current_step_idx,
                                        state_history=neighbor_next_state,  # 🔥 修复：使用邻居的下一状态
                                        path_info=neighbor_path_info,
                                        prev_action=prev_action_k,  # 🔥 修复：使用相同的前一个动作
                                        edge_id_map=edge_id_map,
                                        step_penalty=args.r_step,
                                        backtrack_penalty=args.r_backtrack
                                    )
                                
                                neighbor_rewards_for_this_idx.append(neighbor_reward_value)
                        neighbor_rewards_list.append(np.array(neighbor_rewards_for_this_idx))
                    
                    # 坏动作统计（只在log_frequency时打印）
                    if global_step % args.log_frequency == 0 and bad_action_counts['total'] > 0:
                        print(f"  ⚠️  Bad Actions: {bad_action_counts['total']} (revisit={bad_action_counts['revisit']})")
                    
                    # 计算discount
                    discount = []
                    for k in range(len(action_ids)):
                        discount.append(0.0 if is_done[k] else args.discount)
                    discount = np.array(discount)
                    
                    # 准备邻居数据（替代原有的负样本）
                    neighbor_actions_array = padded_neighbor_array  # [batch_size, max_neighbors]
                    neighbor_rewards_array = np.array(neighbor_rewards_list).T  # [batch_size, max_neighbors]
                    neighbor_target_Qs_array = np.array(neighbor_target_Qs).transpose(1, 0, 2)  # [batch_size, max_neighbors, item_num]
                    # 🔥 Actor-Critic DDPG: Actor target probs for neighbors
                    neighbor_actor_target_probs_array = np.array(neighbor_actor_target_probs_list).transpose(1, 0, 2)  # [batch_size, max_neighbors, item_num]
                    
                    # 🔥 GPT建议: 负样本终止处理 - 回环样本设为终止状态，无bootstrap
                    # 对识别为回环的邻居样本：
                    # 1. discount = 0.0 (终止)
                    # 2. target_Q = 0.0, selector_Q = 0.0 (无bootstrap)
                    # 3. loss_mask = 1.0 (参与训练，非padding)
                    neighbor_discounts = np.full((batch_size, max_neighbors_in_batch), args.discount, dtype=np.float32)
                    
                    # 🔥 性能优化：复用已计算的action masks（在第5893行已经计算过）
                    # 避免重复计算，直接从current_action_masks提取valid edge IDs
                    batch_action_masks_cache = {}
                    for k in range(batch_size):
                        # 直接从current_action_masks获取valid edge IDs，无需重新计算
                        batch_action_masks_cache[k] = np.where(current_action_masks[k] == 1)[0]
                    
                    for neighbor_idx in range(max_neighbors_in_batch):
                        for k in range(batch_size):
                            neighbor_action_id = padded_neighbor_array[k, neighbor_idx]
                            
                            # 检测坏动作（回环）- 无向图：检查edge_id是否在最近K步中出现
                            if neighbor_action_id < item_num:  # 非填充ID
                                if len(recent_edge_ids_batch[k]) > 0:
                                    # 如果当前edge_id在最近K步中出现过 → 回环 → 设为终止状态
                                    if neighbor_action_id in recent_edge_ids_batch[k]:
                                        # 🔥 死胡同豁免：检查是否只有这一条合法边（使用预计算的mask）
                                        valid_edge_ids_at_k = batch_action_masks_cache.get(k, np.array([]))
                                        # 如果只有1条合法边且就是这条"回头边"，则豁免
                                        if len(valid_edge_ids_at_k) == 1 and valid_edge_ids_at_k[0] == neighbor_action_id:
                                            continue  # 跳过设为终止状态，允许这条边
                                        
                                        # 1. 设置discount=0.0 (终止状态)
                                        neighbor_discounts[k, neighbor_idx] = 0.0
                                        # 2. 将Q值和Actor概率设为0 (无bootstrap)
                                        neighbor_target_Qs_array[k, neighbor_idx, :] = 0.0
                                        neighbor_actor_target_probs_array[k, neighbor_idx, :] = 0.0
                                        # 3. 奖励应该是负值 (在前面的reward计算中已经设置)
                    
                    # 设置邻居覆盖TD的损失权重（Phase2才启用）
                    neighbor_loss_weight = 0.5 if current_phase >= 2 else 0.0
                    
                    # 构造feed_dict（使用邻居数据）
                    # 🔥 构造损失掩码：填充位置零损失
                    negative_loss_mask_array = np.zeros((batch_size, max_neighbors_in_batch), dtype=np.float32)
                    for k in range(batch_size):
                        for neighbor_idx in range(max_neighbors_in_batch):
                            neighbor_action_id = padded_neighbor_array[k, neighbor_idx]
                            if neighbor_action_id < item_num:  # 非填充ID
                                negative_loss_mask_array[k, neighbor_idx] = 1.0
                    
                    # 🔥 新增：为负样本重建候选边特征用于训练
                    # 使用之前已经提取的候选边信息（在8408-8411行提取）
                    # 注意：这里重用已经提取的变量，而不是重新提取
                    batch_cand_features = []
                    batch_cand_masks = []

                    for b in range(len(cand_edge_ids_batch)):
                        cand_edge_ids = cand_edge_ids_batch[b]
                        cur_node_id = cur_node_ids_batch[b]
                        goal_node_id = goal_node_ids_batch[b]
                        d_start = d_start_batch[b] if b < len(d_start_batch) else 100.0
                        prev_node_id = prev_node_ids_batch[b] if b < len(prev_node_ids_batch) else None
                        recent_visited = recent_visited_batch[b] if b < len(recent_visited_batch) else None

                        # 重建候选边特征
                        cand_features = feature_builder.build_candidate_features(
                            cand_edge_ids=cand_edge_ids,
                            cur_node_id=cur_node_id,
                            goal_node_id=goal_node_id,
                            d_start=d_start,
                            prev_node_id=prev_node_id,
                            recent_visited_nodes=recent_visited
                        )

                        # 填充到固定大小 [max_candidates, feature_dim]
                        cand_features_padded = feature_builder.pad_candidate_features(
                            cand_features, mainQN.max_candidates
                        )
                        batch_cand_features.append(cand_features_padded)

                        # 创建mask
                        cand_mask = feature_builder.create_candidate_mask(
                            len(cand_edge_ids), mainQN.max_candidates
                        )
                        batch_cand_masks.append(cand_mask)

                    # 转换为numpy数组
                    batch_cand_features = np.array(batch_cand_features)  # [batch_size, max_candidates, feature_dim]
                    batch_cand_masks = np.array(batch_cand_masks)      # [batch_size, max_candidates]

                    feed_dict = build_feed_dict(mainQN, state, len_state, target_Qs, reward,
                                            discount, action_ids, target_Qs_selector, current_action_masks,
                                            current_phase, rl_weight,
                                            actor_target_probs=actor_target_probs,  # 🔥 Actor-Critic DDPG
                                            negative_actions=neighbor_actions_array,
                                            negative_rewards=neighbor_rewards_array,
                                            negative_target_Qs=neighbor_target_Qs_array,
                                            negative_actor_target_probs=neighbor_actor_target_probs_array,  # 🔥 Actor-Critic DDPG
                                            negative_loss_weight=neighbor_loss_weight,
                                            negative_loss_mask=negative_loss_mask_array,  # 🔥 传递损失掩码
                                            actor_rl_weight=args.actor_rl_weight,  # 🔥 Actor RL辅损权重 λ_RL
                                            cand_features=cand_features_batch,  # 🔥 新增：重建的候选边特征
                                            cand_mask=cand_masks_batch,         # 🔥 新增：候选边mask
                                            is_training=True)
                
                
                    # 🔥 Execute training steps with strict SL/RL separation
                    loss_components = sess.run(mainQN.loss_components, feed_dict=feed_dict)
                    
                    # 🔥 Separate training for SL and RL heads
                    if current_phase == 1:
                        # Phase 1: SL-only training on expert data
                        sess.run(mainQN.train_phase1, feed_dict=feed_dict)
                        
                    elif current_phase >= 2:
                        # Phase 2+: Separate training with different data for different heads
                        
                        # 🔥 Step 1: Train SL head + shared encoder on expert data
                        # Extract expert data for SL training
                        # 🔥 修复：新格式使用'taken_edge_id'，旧格式使用'action'
                        action_key = 'taken_edge_id' if 'taken_edge_id' in expert_batch else 'action'
                        
                        if isinstance(expert_batch['state'], dict):
                            expert_state = list(expert_batch['state'].values())
                            expert_action = list(expert_batch[action_key].values())
                            expert_len_state = list(expert_batch['len_state'].values())
                            expert_user_ids = list(expert_batch.get('user_id', {}).values()) if 'user_id' in expert_batch else [0] * len(expert_state)
                        else:
                            expert_state = expert_batch['state']
                            expert_action = expert_batch[action_key]
                            expert_len_state = expert_batch['len_state']
                            expert_user_ids = expert_batch.get('user_id', [0] * len(expert_state))
                        
                        # 🔥 确保expert_state是numpy数组
                        if isinstance(expert_state, list):
                            expert_state = np.array(expert_state)
                        
                        # 🔥 如果启用了文本 embeddings，增强专家状态数据
                        if args.use_text_embeddings and text_embedding_loader is not None:
                            expert_state = enhance_states_with_text_embeddings(
                                expert_state, G, edge_id_map,
                                text_embedding_loader=text_embedding_loader,
                                embedding_proj_weights=embedding_proj_weights,
                                embedding_proj_biases=embedding_proj_biases,
                                original_feature_dim=feature_dim,
                                user_embedding=user_embedding,
                                user_embeddings_dict=user_embeddings_dict,
                                user_ids=expert_user_ids
                            )
                        elif args.use_user_embedding:
                            # 只启用用户embeddings（没有文本embeddings）
                            if user_embeddings_dict is not None and expert_user_ids is not None:
                                expert_state = add_user_embeddings_to_states_batch(expert_state, user_embeddings_dict, expert_user_ids)
                            else:
                                expert_state = add_user_embedding_to_states(expert_state, user_embedding)
                        
                        expert_action_masks = generate_action_mask_batch(expert_state, G, edge_id_map, item_num)
                        expert_action_ids = extract_action_ids(expert_action, edge_id_map)
                        
                        expert_feed_dict = {
                            mainQN.inputs: expert_state,
                            mainQN.len_state: expert_len_state,
                            mainQN.action_mask: expert_action_masks,
                            mainQN.actions: expert_action_ids,
                            mainQN.is_training: True,
                            mainQN.training_phase: current_phase,
                            mainQN.rl_weight: rl_weight,
                            mainQN.actor_rl_weight: args.actor_rl_weight,  # 🔥 Actor RL辅损权重 λ_RL
                            # Dummy values for RL-related placeholders (not used in SL training)
                            mainQN.targetQs_: np.zeros((len(expert_state), item_num), dtype=np.float32),
                            mainQN.reward: np.zeros(len(expert_state), dtype=np.float32),
                            mainQN.discount: np.zeros(len(expert_state), dtype=np.float32),
                            mainQN.targetQs_selector: np.zeros((len(expert_state), item_num), dtype=np.float32),
                            mainQN.actor_target_probs: np.zeros((len(expert_state), item_num), dtype=np.float32),  # 🔥 Actor-Critic
                            mainQN.negative_actions: np.zeros((len(expert_state), 0), dtype=np.int32),
                            mainQN.negative_rewards: np.zeros((len(expert_state), 0), dtype=np.float32),
                            mainQN.negative_target_Qs: np.zeros((len(expert_state), 0, item_num), dtype=np.float32),
                            mainQN.negative_actor_target_probs: np.zeros((len(expert_state), 0, item_num), dtype=np.float32),  # 🔥 Actor-Critic
                            mainQN.negative_loss_mask: np.zeros((len(expert_state), 0), dtype=np.float32),
                            mainQN.negative_loss_weight: 0.0
                        }
                        
                        # Train SL head on expert data
                        sess.run(mainQN.sl_head_optimizer, feed_dict=expert_feed_dict)
                        
                        # 🔥 Step 2: Train RL head on mixed data
                        # Use the previously computed feed_dict which contains mixed RL batch
                        sess.run(mainQN.q_head_optimizer, feed_dict=feed_dict)
                        
                        # 🔥 Step 3: Update shared encoder (RL去耦版本)
                        # 🔥 RL去耦：由于Q head使用了stop_gradient，RL梯度已经被阻断
                        # 共享编码器只会被SL梯度更新，因此始终使用expert数据即可
                        # 不再需要复杂的冻结和梯度合成逻辑
                        sess.run(mainQN.shared_encoder_optimizer, feed_dict=expert_feed_dict)

                    global_step += 1

                    def tensor_to_scalar(tensor):
                        if hasattr(tensor, 'numpy'):
                            # TensorFlow tensor
                            numpy_array = tensor.numpy()
                            if numpy_array.size == 1:
                                return float(numpy_array.item())
                            else:
                                # Multi-dimensional tensor, take the mean
                                return float(np.mean(numpy_array))
                        elif hasattr(tensor, 'item'):
                            # NumPy array
                            if tensor.size == 1:
                                return float(tensor.item())
                            else:
                                # Multi-dimensional array, take the mean
                                return float(np.mean(tensor))
                        else:
                            # Already a scalar
                            return float(tensor)

                    total_loss = tensor_to_scalar(loss_components['total_loss'])
                    
                    # 🔥 详细统计信息：只在log_frequency时打印
                    if global_step % args.log_frequency == 0:
                        # 🔥 Log on-policy buffer statistics
                        onpolicy_size = onpolicy_buffer.size()
                        onpolicy_ratio_actual = 0.0
                        if current_phase >= 2 and onpolicy_size > 0:
                            n_onpolicy_samples = int(args.batch_size * args.onpolicy_mix_ratio)
                            onpolicy_ratio_actual = n_onpolicy_samples / args.batch_size if args.batch_size > 0 else 0.0
                        
                        # 在线策略缓冲区统计（只在log_frequency时打印）
                        if current_phase >= 2:
                            print(f"  📊 O-bucket: {onpolicy_size}/{args.onpolicy_buffer_size} "
                                f"(mix={onpolicy_ratio_actual:.2f}, {int(args.batch_size * args.onpolicy_mix_ratio)}/{args.batch_size} samples)")
                        
                        # 提取Q值统计信息
                        q_value_range = tensor_to_scalar(loss_components['q_value_range'])
                        q_value_min = tensor_to_scalar(loss_components['q_value_min'])
                        q_value_max = tensor_to_scalar(loss_components['q_value_max'])
                        q_value_mean = tensor_to_scalar(loss_components['q_value_mean'])
                        q_value_std = tensor_to_scalar(loss_components['q_value_std'])
                        mask_penalty = tensor_to_scalar(loss_components['mask_penalty'])
                        mask_penalty_gap = tensor_to_scalar(loss_components['mask_penalty_gap'])
                        q_unmasked_min = tensor_to_scalar(loss_components['q_value_unmasked_min'])
                        q_unmasked_max = tensor_to_scalar(loss_components['q_value_unmasked_max'])
                        
                        # 提取QLoss components
                        qloss_positive = tensor_to_scalar(loss_components['qloss_positive'])
                        
                        # Reward统计 - 只在RL阶段收集（phase >= 2）
                        if current_phase >= 2:
                            if not hasattr(calculate_improved_reward, 'reward_stats'):
                                calculate_improved_reward.reward_stats = {
                                    'rewards': [],
                                    'positive_rewards': [],
                                    'negative_rewards': [],
                                    'reward_components': {}
                                }

                            # 收集当前batch的reward统计
                            all_rewards = np.array(reward)  # 这是正样本的reward
                            all_positive_rewards = all_rewards[all_rewards > 0]
                            all_negative_rewards = all_rewards[all_rewards < 0]

                            calculate_improved_reward.reward_stats['rewards'].extend(all_rewards.tolist())
                            calculate_improved_reward.reward_stats['positive_rewards'].extend(all_positive_rewards.tolist())
                            calculate_improved_reward.reward_stats['negative_rewards'].extend(all_negative_rewards.tolist())

                            # 保持最近10000个reward用于统计
                            max_history = 10000
                            if len(calculate_improved_reward.reward_stats['rewards']) > max_history:
                                calculate_improved_reward.reward_stats['rewards'] = calculate_improved_reward.reward_stats['rewards'][-max_history:]
                                calculate_improved_reward.reward_stats['positive_rewards'] = calculate_improved_reward.reward_stats['positive_rewards'][-max_history:]
                                calculate_improved_reward.reward_stats['negative_rewards'] = calculate_improved_reward.reward_stats['negative_rewards'][-max_history:]

                            # 计算reward统计
                            if calculate_improved_reward.reward_stats['rewards']:
                                all_rewards = np.array(calculate_improved_reward.reward_stats['rewards'])
                                positive_rewards = np.array(calculate_improved_reward.reward_stats['positive_rewards']) if calculate_improved_reward.reward_stats['positive_rewards'] else np.array([])
                                negative_rewards = np.array(calculate_improved_reward.reward_stats['negative_rewards']) if calculate_improved_reward.reward_stats['negative_rewards'] else np.array([])

                                reward_stats = {
                                    'all_rewards': {
                                        'sum': float(np.sum(all_rewards)),
                                        'count': len(all_rewards),
                                        'mean': float(np.mean(all_rewards)),
                                        'std': float(np.std(all_rewards)),
                                        'min': float(np.min(all_rewards)),
                                        'max': float(np.max(all_rewards)),
                                        'range': float(np.max(all_rewards) - np.min(all_rewards))
                                    },
                                    'positive_rewards': {
                                        'count': len(positive_rewards),
                                        'mean': float(np.mean(positive_rewards)) if len(positive_rewards) > 0 else 0.0,
                                        'std': float(np.std(positive_rewards)) if len(positive_rewards) > 0 else 0.0,
                                        'min': float(np.min(positive_rewards)) if len(positive_rewards) > 0 else 0.0,
                                        'max': float(np.max(positive_rewards)) if len(positive_rewards) > 0 else 0.0
                                    },
                                    'negative_rewards': {
                                        'count': len(negative_rewards),
                                        'mean': float(np.mean(negative_rewards)) if len(negative_rewards) > 0 else 0.0,
                                        'std': float(np.std(negative_rewards)) if len(negative_rewards) > 0 else 0.0,
                                        'min': float(np.min(negative_rewards)) if len(negative_rewards) > 0 else 0.0,
                                        'max': float(np.max(negative_rewards)) if len(negative_rewards) > 0 else 0.0
                                    }
                                }
                            else:
                                reward_stats = None
                        else:
                            # SL阶段不收集reward统计
                            reward_stats = None
                        
                        # RL不稳定性检测
                        if current_phase > 1:
                            instability_detected = False
                            instability_warnings = []
                            
                            if q_value_range > 100.0:
                                instability_warnings.append(f"Q-value range too large ({q_value_range:.2f})")
                                instability_detected = True
                            if q_value_std > 15.0:
                                instability_warnings.append(f"Q-value std too large ({q_value_std:.2f})")
                                instability_detected = True
                            if abs(q_value_mean) > 30.0:
                                instability_warnings.append(f"Q-value mean too extreme ({q_value_mean:.2f})")
                                instability_detected = True
                            if total_loss > 0.5:
                                instability_warnings.append(f"Total loss too high ({total_loss:.4f})")
                                instability_detected = True
                            
                            
                            if instability_detected:
                                print(f"  ⚠️  RL Instability Detected:")
                                for warning in instability_warnings:
                                    print(f"      - {warning}")
                        
                        # 计算correct_predictions和total_samples用于logger
                        if current_phase >= 2:
                            # Phase-2: SL准确率只计算expert部分
                            # 🔥 修复：基于expert数据单独计算predictions，而不是从混合batch中切片
                            # 因为混合batch中expert数据的位置不确定，且expert_state已经在训练时处理好了
                            if 'expert_state' in locals() and expert_state is not None and len(expert_state) > 0 and 'expert_action_ids' in locals():
                                # expert_state已经在训练时处理好了（包括文本embeddings增强），直接使用
                                # expert_action_masks也已经计算好了，可以直接使用
                                if 'expert_action_masks' in locals() and expert_action_masks is not None:
                                    expert_action_masks_for_acc = expert_action_masks
                                else:
                                    # 如果expert_action_masks不可用，重新计算
                                    expert_action_masks_for_acc = generate_action_mask_batch(expert_state, G, edge_id_map, item_num)
                                
                                # 计算expert数据的predictions
                                expert_predictions = sess.run(
                                    mainQN.probs,
                                    feed_dict={
                                        mainQN.inputs: expert_state,
                                        mainQN.len_state: expert_len_state if 'expert_len_state' in locals() else [1] * len(expert_state),
                                        mainQN.action_mask: expert_action_masks_for_acc,
                                        mainQN.is_training: False,
                                        mainQN.training_phase: current_phase,
                                        mainQN.rl_weight: rl_weight,
                                    }
                                )
                                expert_top1_preds = np.argmax(expert_predictions, axis=1)
                                correct_predictions = np.sum(expert_top1_preds == expert_action_ids)
                                total_samples = len(expert_action_ids)
                            else:
                                # 如果expert_state不可用，回退到旧方法（可能不准确）
                                correct_predictions = np.sum(top1_preds[:len(expert_action_ids)] == expert_action_ids) if len(top1_preds) >= len(expert_action_ids) else 0
                                total_samples = len(expert_action_ids) if 'expert_action_ids' in locals() else 0
                        else:
                            # Phase-1: 使用全部数据（都是expert数据）
                            correct_predictions = np.sum(top1_preds == action_ids)
                            total_samples = len(action_ids)
                        
                        # 计算accuracy
                        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
                        avg_reward = np.mean(reward) if reward else 0.0
                        
                        # Log all loss components (always log to file, but only print detailed stats at log_frequency)
                        logger.log_loss_components(global_step, loss_components, current_lr, phase_name)
                        logger.log_loss(global_step, total_loss, current_lr, phase_name)
                        logger.log_accuracy(global_step, accuracy, correct_predictions, total_samples, phase_name, avg_reward)
                        
                        # 记录Reward统计到logger（总是记录到文件）
                        if reward_stats:
                            # 创建reward统计日志数据
                            reward_log_data = {
                            'step': global_step,
                            'phase': phase_name,
                            'rl_weight': rl_weight,
                            'all_history_reward_sum': reward_stats['all_rewards']['sum'],
                            'all_history_reward_mean': reward_stats['all_rewards']['mean'],
                            'all_history_reward_std': reward_stats['all_rewards']['std'],
                            'all_history_reward_min': reward_stats['all_rewards']['min'],
                            'all_history_reward_max': reward_stats['all_rewards']['max'],
                            'all_history_reward_count': reward_stats['all_rewards']['count'],
                            'positive_reward_count': reward_stats['positive_rewards']['count'],
                            'positive_reward_mean': reward_stats['positive_rewards']['mean'],
                            'positive_reward_min': reward_stats['positive_rewards']['min'],
                            'positive_reward_max': reward_stats['positive_rewards']['max'],
                            'negative_reward_count': reward_stats['negative_rewards']['count'],
                            'negative_reward_mean': reward_stats['negative_rewards']['mean'],
                            'negative_reward_min': reward_stats['negative_rewards']['min'],
                            'negative_reward_max': reward_stats['negative_rewards']['max'],
                            'reward_range': reward_stats['all_rewards']['range'],
                            'positive_negative_ratio': (reward_stats['positive_rewards']['count'] /
                                                    max(reward_stats['all_rewards']['count'], 1))
                            }

                            # 记录reward统计到logger
                            logger.log_reward_stats(global_step, reward_log_data)
                        
                        # 打印训练统计信息（只在log_frequency时打印）
                        if global_step % args.log_frequency == 0:
                            print(f"\n📊 Training Stats (Step {global_step}):")
                            print(f" Current Time is {time.strftime('%Y-%m-%d %H:%M:%S')}")
                            print(f"  Loss: {total_loss:.4f} | Accuracy: {accuracy:.4f} ({correct_predictions}/{total_samples})")
                            if reward_stats:
                                print(f"  Reward: mean={reward_stats['all_rewards']['mean']:.4f}, "
                                    f"std={reward_stats['all_rewards']['std']:.4f}, "
                                    f"+{reward_stats['positive_rewards']['count']}/-{reward_stats['negative_rewards']['count']}")
                            if current_phase >= 2:
                                print(f"  Q-value: mean={q_value_mean:.2f}, std={q_value_std:.2f}, range=[{q_value_min:.2f}, {q_value_max:.2f}]")
                            print()  # 空行分隔
                        
                    # 阶段切换提示
                    if global_step == args.phase1_sl_only_steps:
                        print(f"\nPHASE TRANSITION: Phase1-SL-Only → Phase2-SL+RL (Step {global_step})")
                                    
                    # 🔥 标准评估 (SL/RL通用评估)
                    if global_step % args.eval_frequency == 0:                  
                        print(f"\n{'='*60}")
                        print(f"🔍 EVALUATION at step {global_step}")
                        print(f"{'='*60}")
                        
                        # 🔥 Stage-1: Skip RL evaluations to reduce computational cost
                        is_stage1 = (global_step <= args.phase1_sl_only_steps)
                        
                        if is_stage1:
                            print(f"🎯 Stage-1: SL-only evaluation (skipping RL metrics)")
                            # Only run basic SL evaluation
                            train_metrics = batch_evaluate_improved(sess, QN_1, dataset='train', logger=logger, step=global_step, 
                                                                training_phase=1, rl_weight=0.0, batch_size=args.batch_size, rl_rollout_sample_size=0, G=G, edge_id_map=edge_id_map, item_num=item_num, reward_goal=reward_goal, target_model=QN_2,
                                                                eval_num_trials=args.eval_num_trials, eval_temperature=args.eval_temperature,
                                                                use_text_embeddings=args.use_text_embeddings, text_embedding_loader=text_embedding_loader, 
                                                                embedding_proj_weights=embedding_proj_weights, embedding_proj_biases=embedding_proj_biases, feature_dim=feature_dim,
                                                                data_directory=data_directory, user_id=args.user_id, moe_data_dir=args.moe_data_dir, graph_id=args.graph_id)
                            val_metrics = batch_evaluate_improved(sess, QN_1, dataset='val', logger=logger, step=global_step,
                                                            training_phase=1, rl_weight=0.0, batch_size=args.batch_size, rl_rollout_sample_size=0, G=G, edge_id_map=edge_id_map, item_num=item_num, reward_goal=reward_goal, target_model=QN_2,
                                                            eval_num_trials=args.eval_num_trials, eval_temperature=args.eval_temperature,
                                                            use_text_embeddings=args.use_text_embeddings, text_embedding_loader=text_embedding_loader, 
                                                            embedding_proj_weights=embedding_proj_weights, embedding_proj_biases=embedding_proj_biases, feature_dim=feature_dim,
                                                            data_directory=data_directory, user_id=args.user_id, moe_data_dir=args.moe_data_dir, graph_id=args.graph_id)
                            # 🔥 修复：如果val数据集不存在，跳过val评估
                            if val_metrics is None:
                                print(f"⚠️  Skipping VAL evaluation (dataset not found)")
                        else:
                            train_metrics = batch_evaluate_improved(sess, QN_1, dataset='train', logger=logger, step=global_step, 
                                                                training_phase=current_phase, rl_weight=rl_weight, batch_size=args.batch_size, rl_rollout_sample_size=50, G=G, edge_id_map=edge_id_map, item_num=item_num, reward_goal=reward_goal, target_model=QN_2,
                                                                eval_num_trials=args.eval_num_trials, eval_temperature=args.eval_temperature,
                                                                use_text_embeddings=args.use_text_embeddings, text_embedding_loader=text_embedding_loader, 
                                                                embedding_proj_weights=embedding_proj_weights, embedding_proj_biases=embedding_proj_biases, feature_dim=feature_dim,
                                                                data_directory=data_directory, user_id=args.user_id, moe_data_dir=args.moe_data_dir, graph_id=args.graph_id)
                            val_metrics = batch_evaluate_improved(sess, QN_1, dataset='val', logger=logger, step=global_step,
                                                            training_phase=current_phase, rl_weight=rl_weight, batch_size=args.batch_size, rl_rollout_sample_size=50, G=G, edge_id_map=edge_id_map, item_num=item_num, reward_goal=reward_goal, target_model=QN_2,
                                                            eval_num_trials=args.eval_num_trials, eval_temperature=args.eval_temperature,
                                                            use_text_embeddings=args.use_text_embeddings, text_embedding_loader=text_embedding_loader, 
                                                            embedding_proj_weights=embedding_proj_weights, embedding_proj_biases=embedding_proj_biases, feature_dim=feature_dim,
                                                            data_directory=data_directory, user_id=args.user_id, moe_data_dir=args.moe_data_dir, graph_id=args.graph_id)
                            # 🔥 修复：如果val数据集不存在，跳过val评估
                            if val_metrics is None:
                                print(f"⚠️  Skipping VAL evaluation (dataset not found)")
                        
                        # 🔥 Early stopping check (only if enabled and val_metrics exists)
                        if args.early_stopping and val_metrics is not None:
                            current_epoch = (global_step - training_start_step) / num_batches
                            
                            # Only start checking after early_stopping_start_epoch
                            if current_epoch >= args.early_stopping_start_epoch:
                                # Get the monitored metric
                                current_val_metric = val_metrics.get(args.early_stopping_metric, 0.0)
                                
                                # Check if this is an improvement
                                improvement = current_val_metric - best_val_metric
                                
                                if improvement > args.early_stopping_min_delta:
                                    # Improvement detected
                                    print(f"✅ Early Stopping: Improvement detected!")
                                    print(f"   Metric ({args.early_stopping_metric}): {best_val_metric:.6f} → {current_val_metric:.6f} (+{improvement:.6f})")
                                    
                                    best_val_metric = current_val_metric
                                    patience_counter = 0
                                    best_model_step = global_step
                                    
                                    # Save best model
                                    best_model_dir = os.path.join(data_directory, 'saved_model')
                                    os.makedirs(best_model_dir, exist_ok=True)
                                    best_model_path = os.path.join(best_model_dir, f'best_model_early_stopping.ckpt')
                                    saver.save(sess, best_model_path)
                                    print(f"   💾 Best model saved: {best_model_path}")
                                    print(f"   Patience counter reset: {patience_counter}/{args.early_stopping_patience}")
                                else:
                                    # No improvement
                                    patience_counter += 1
                                    print(f"⚠️  Early Stopping: No improvement")
                                    print(f"   Metric ({args.early_stopping_metric}): {current_val_metric:.6f} (best: {best_val_metric:.6f})")
                                    print(f"   Patience: {patience_counter}/{args.early_stopping_patience}")
                                    
                                    if patience_counter >= args.early_stopping_patience:
                                        print(f"\n{'='*60}")
                                        print(f"🛑 EARLY STOPPING TRIGGERED!")
                                        print(f"{'='*60}")
                                        print(f"   No improvement for {args.early_stopping_patience} evaluations")
                                        print(f"   Best {args.early_stopping_metric}: {best_val_metric:.6f} at step {best_model_step}")
                                        print(f"   Best model saved at: {best_model_path}")
                                        print(f"{'='*60}\n")
                                        early_stop_triggered = True
                                        break  # Exit the inner loop (batches)
                            else:
                                print(f"ℹ️  Early Stopping: Waiting until epoch {args.early_stopping_start_epoch} (current: {current_epoch:.1f})")
                        
                        print(f"{'='*60}\n")
                
                # 🔥 Check if early stopping was triggered (exit outer epoch loop)
                if early_stop_triggered:
                    break
                
        # 🔥 Load best model if early stopping was triggered
        if args.early_stopping and early_stop_triggered and best_model_path:
            print(f"\n{'='*60}")
            print(f"🔄 Loading best model for final evaluation")
            print(f"{'='*60}")
            print(f"   Best model step: {best_model_step}")
            print(f"   Best {args.early_stopping_metric}: {best_val_metric:.6f}")
            print(f"   Loading from: {best_model_path}")
            try:
                saver.restore(sess, best_model_path)
                print(f"   ✅ Best model loaded successfully")
            except Exception as e:
                print(f"   ⚠️  Failed to load best model: {e}")
                print(f"   Will use current model for final evaluation")
            print(f"{'='*60}\n")
        
        # Final evaluation
        print("\n" + "="*60)
        print("FINAL EVALUATION (After Training)")
        if args.early_stopping and early_stop_triggered:
            print("(Using best model from early stopping)")
        print("="*60)
        print("\nTest set evaluation:")
        # 🔥 始终执行test集的完整评估（显示A.单步准确率, B.Reach@B, C.路径准确率/覆盖度）
        print(f"\n🔥 Final TEST dataset evaluation with complete metrics...")
        test_metrics = batch_evaluate_improved(sess, QN_1, dataset='test', logger=logger, step=global_step,
                                            training_phase=current_phase, rl_weight=rl_weight, batch_size=args.batch_size, 
                                            sample_ratio=1.0,  # 使用全量test集
                                            rl_rollout_sample_size=50, G=G, edge_id_map=edge_id_map, 
                                            item_num=item_num, reward_goal=reward_goal, target_model=QN_2,
                                            eval_num_trials=args.eval_num_trials, eval_temperature=args.eval_temperature,
                                            use_text_embeddings=args.use_text_embeddings, text_embedding_loader=text_embedding_loader, 
                                            embedding_proj_weights=embedding_proj_weights, embedding_proj_biases=embedding_proj_biases, feature_dim=feature_dim,
                                            data_directory=data_directory, user_id=args.user_id, moe_data_dir=args.moe_data_dir,
                                            save_per_episode_info=True, graph_id=args.graph_id)
    
        # Save logs and model (only in normal training mode, not eval-only)
        if not args.eval_only:
            logger.save_all_logs()

            # Save final model
            final_checkpoint_path = os.path.join(data_directory, f'saved_model/llm4rec_improved_{time.strftime("%Y%m%d_%H%M%S")}.ckpt')
            saver.save(sess, final_checkpoint_path)
            print(f"\nFinal model saved: {final_checkpoint_path}")
            
            # 🔥 Print early stopping summary
            if args.early_stopping:
                print(f"\n{'='*60}")
                print(f"📊 Early Stopping Summary")
                print(f"{'='*60}")
                if early_stop_triggered:
                    print(f"   Status: ✅ Triggered at step {global_step}")
                    print(f"   Best model: {best_model_path}")
                    print(f"   Best {args.early_stopping_metric}: {best_val_metric:.6f}")
                    print(f"   Best model step: {best_model_step}")
                    print(f"   Epochs saved: ~{(global_step - best_model_step) / num_batches:.1f}")
                else:
                    print(f"   Status: ❌ Not triggered (training completed normally)")
                    print(f"   Best {args.early_stopping_metric}: {best_val_metric:.6f}")
                    print(f"   Best model step: {best_model_step}")
                print(f"{'='*60}")
        else:
            print("\n🔍 Eval-only mode: Skipping log and model saving (evaluation complete)")
        
        # 🔥 清理 evaluation_cache 缓存（只删除当前进程的缓存文件）
        print(f"\n{'='*60}")
        print("CLEANING UP EVALUATION CACHE")
        print(f"{'='*60}")
        cache_dir = os.path.join(data_directory, 'evaluation_cache')
        if os.path.exists(cache_dir):
            try:
                import os
                current_pid = os.getpid()
                current_pid_suffix = f'_p{current_pid}'
                # 只删除当前进程的缓存文件
                deleted_count = 0
                deleted_size = 0
                for filename in os.listdir(cache_dir):
                    if filename.endswith('.pkl') and current_pid_suffix in filename:
                        file_path = os.path.join(cache_dir, filename)
                        try:
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            deleted_count += 1
                            deleted_size += file_size
                        except Exception as e:
                            print(f"⚠️  Failed to delete {filename}: {e}")

                if deleted_count > 0:
                    print(f"✅ Cleaned {deleted_count} cache files for current process ({deleted_size / (1024*1024):.2f} MB)")
                else:
                    print("ℹ️  No cache files found for current process")
            except Exception as e:
                print(f"⚠️  Failed to clean evaluation cache: {e}")
        else:
            print(f"ℹ️  No evaluation cache found at: {cache_dir}")
        print(f"{'='*60}\n") 

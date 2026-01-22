"""
Utility functions for the Graph Routing project.

This module contains utility functions for loading map data, creating graphs,
and other common operations.
"""

import os
import pandas as pd
import geopandas as gpd
import json
import pickle
from typing import Dict, Any, Tuple
import tensorflow as tf
import numpy as np

def load_map_data(data_directory: str) -> Dict[str, Any]:
    """
    Load map data from the specified directory.

    Args:
        data_directory: Path to the data directory containing network_map and metadata.json

    Returns:
        Dictionary containing 'df' (GeoDataFrame) and 'meta_data' (metadata dict)
    """
    # Load network map - try different formats
    network_file_path = os.path.join(data_directory, 'network_map.gpkg')
    if not os.path.exists(network_file_path):
        # Try .shp format as fallback
        network_file_path = os.path.join(data_directory, 'network_map.shp')
        if not os.path.exists(network_file_path):
            # Try without extension
            network_file_path = os.path.join(data_directory, 'network_map')
            if not os.path.exists(network_file_path):
                raise FileNotFoundError(f"Network map not found at {os.path.join(data_directory, 'network_map')} (tried .gpkg, .shp, and no extension)")

    df = gpd.read_file(network_file_path)

    # Load metadata
    metadata_path = os.path.join(data_directory, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    with open(metadata_path, 'r') as f:
        meta_data = json.load(f)

    return {
        'df': df,
        'meta_data': meta_data
    }


def load_or_create_graph(df: gpd.GeoDataFrame, data_directory: str, graph_id: str = 'default_graph') -> Tuple[Any, Dict, list]:
    """
    Load existing graph cache or create a new one.

    Args:
        df: Road network GeoDataFrame
        data_directory: Base data directory
        graph_id: Graph identifier

    Returns:
        Tuple of (G, edge_id_map, edge_feature_list)
    """
    from .graph_cache import GraphCache

    # Create cache directory path
    cache_dir = os.path.join(data_directory, 'graph_data', graph_id, 'cache')
    cache_file = os.path.join(cache_dir, 'graph_cache.pkl')

    # Try to load existing cache
    try:
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Graph cache file not found: {cache_file}")

        # Load GraphCache
        graph_cache = GraphCache()
        graph_cache.load_from_file(cache_file)
        print(f"Loaded existing graph cache from {cache_file}")

        # Rebuild the NetworkX graph
        graph_cache.rebuild_graph()
        G = graph_cache.G

        # Load edge_id_map and edge_feature_list from separate cache files
        edge_id_map_file = os.path.join(cache_dir, 'edge_id_map.pkl')
        edge_feature_list_file = os.path.join(cache_dir, 'edge_feature_list.pkl')

        if os.path.exists(edge_id_map_file):
            with open(edge_id_map_file, 'rb') as f:
                edge_id_map = pickle.load(f)
        else:
            edge_id_map = {}
            print(f"Warning: edge_id_map.pkl not found at {edge_id_map_file}")

        if os.path.exists(edge_feature_list_file):
            with open(edge_feature_list_file, 'rb') as f:
                edge_feature_list = pickle.load(f)
        else:
            edge_feature_list = []
            print(f"Warning: edge_feature_list.pkl not found at {edge_feature_list_file}")

        return G, edge_id_map, edge_feature_list

    except FileNotFoundError as e:
        print(f"No existing cache found: {e}")
        print("Please ensure the graph cache has been built first.")
        print(f"Run: python src/build_graph_cache.py --graph-id {graph_id} --data-dir {data_directory}")
        raise

    except Exception as e:
        print(f"Error loading cache: {e}")
        raise


def pad_history(history_list, max_length, pad_value=0):
    """
    Pad history list to maximum length with FIFO (keep most recent).

    Args:
        history_list: List of history items
        max_length: Maximum length to pad to
        pad_value: Value to use for padding

    Returns:
        Padded list (most recent items preserved)
    """
    if len(history_list) >= max_length:
        # FIFO: 保留最新的 max_length 个元素
        return history_list[-max_length:]
    else:
        # 不足长度时，在前面填充零（最新的在后面）
        return [pad_value] * (max_length - len(history_list)) + history_list


def get_edge_id(edge, edge_id_map):
    """
    Get edge ID from edge tuple.

    Args:
        edge: Edge tuple (u, v)
        edge_id_map: Dictionary mapping edges to IDs

    Returns:
        Edge ID or None if not found
    """
    return edge_id_map.get(edge, edge_id_map.get((edge[1], edge[0]), None))


def to_pickled_df(data_directory: str, **kwargs):
    """
    Save DataFrames to pickle files in the specified directory.
    
    Args:
        data_directory: Directory to save files
        **kwargs: Named DataFrames to save (e.g., replay_buffer=df, data_statis=df)
    """
    os.makedirs(data_directory, exist_ok=True)
    
    for name, df in kwargs.items():
        if df is not None:
            file_path = os.path.join(data_directory, f'{name}.df')
            df.to_pickle(file_path)
            print(f"Saved {name} to {file_path}")


"""MOE Deprecated functions"""
def load_user_embedding(user_id, moe_data_dir='examples/moe_data/processed'):
    """
    Load user statistics embedding for a specific user.
    
    Args:
        user_id: User ID (e.g., 'user_001' or 1)
        moe_data_dir: MOE data directory path
        
    Returns:
        user_embedding: numpy array of user statistics [user_dim]
        or None if user embedding not available
    """
    if user_id is None:
        return None
    
    # Normalize user_id to string format
    if isinstance(user_id, int):
        user_id_str = f'user_{user_id:03d}'
    elif isinstance(user_id, str):
        # If already in format 'user_XXX', use as is
        if user_id.startswith('user_'):
            user_id_str = user_id
        else:
            # Try to parse as number
            try:
                user_num = int(user_id)
                user_id_str = f'user_{user_num:03d}'
            except:
                user_id_str = user_id
    else:
        return None
    
    # Construct path to user stats file
    user_stats_path = os.path.join(moe_data_dir, user_id_str, f'user_stats_{user_id_str.split("_")[1]}.npy')
    
    if not os.path.exists(user_stats_path):
        print(f"Warning: User embedding file not found: {user_stats_path}")
        return None
    
    try:
        user_embedding = np.load(user_stats_path)
        print(f"✅ Loaded user embedding from: {user_stats_path} (dim={len(user_embedding)})")
        return user_embedding
    except Exception as e:
        print(f"Error loading user embedding: {e}")
        return None

def load_all_user_embeddings(moe_data_dir='examples/moe_data/processed'):
    """
    Load all available user statistics embeddings.
    
    Args:
        moe_data_dir: MOE data directory path
        
    Returns:
        user_embeddings_dict: dict of {user_id_str: user_embedding_array}
        or None if no embeddings found
    """
    import glob
    
    if not os.path.exists(moe_data_dir):
        print(f"Warning: MOE data directory not found: {moe_data_dir}")
        return None
    
    # Find all user folders (exclude files like user_stats_scaler.json)
    user_folders = glob.glob(os.path.join(moe_data_dir, 'user_*'))
    # Filter to only keep directories
    user_folders = [f for f in user_folders if os.path.isdir(f)]
    
    if not user_folders:
        print(f"Warning: No user folders found in {moe_data_dir}")
        return None
    
    user_embeddings_dict = {}
    
    for user_folder in sorted(user_folders):
        user_id_str = os.path.basename(user_folder)
        
        # Skip non-folder items (e.g., user_stats_scaler.json)
        if not os.path.isdir(user_folder):
            continue
        
        # Try to load user stats
        try:
            user_num = user_id_str.split('_')[1]
            user_stats_path = os.path.join(user_folder, f'user_stats_{user_num}.npy')
            
            if os.path.exists(user_stats_path):
                user_embedding = np.load(user_stats_path)
                user_embeddings_dict[user_id_str] = user_embedding
                print(f"✅ Loaded {user_id_str}: dim={len(user_embedding)}")
            else:
                print(f"⚠️  User stats file not found: {user_stats_path}")
        except Exception as e:
            print(f"❌ Error loading {user_id_str}: {e}")
            continue
    
    if not user_embeddings_dict:
        print(f"Warning: No user embeddings loaded from {moe_data_dir}")
        return None
    
    print(f"\n📊 Loaded {len(user_embeddings_dict)} user embeddings total")
    return user_embeddings_dict

def get_moe_data_path(data_directory, user_id, moe_data_dir, filename, graph_id='default_graph'):
    """
    获取MOE数据文件路径（如果指定了user_id）或标准路径

    Args:
        data_directory: 标准数据目录（用于地图、图数据等）
        user_id: 用户ID（如果为None，使用标准路径）
        moe_data_dir: MOE数据根目录
        filename: 文件名（如 'replay_buffer.df'）
        graph_id: 图ID（默认为 'default_graph'）

    Returns:
        完整文件路径
    """
    if user_id is not None and moe_data_dir is not None:
        # MOE模式：使用用户特定的数据目录和文件名
        # 🔥 修复：将user_id转换为字符串
        user_id_str = str(user_id) if not isinstance(user_id, str) else user_id
        user_data_dir = os.path.join(moe_data_dir, user_id_str)
        # 将文件名转换为用户特定的文件名（如 replay_buffer.df -> replay_buffer_user_001.df）
        if filename.endswith('.df'):
            base_name = filename[:-3]  # 移除 .df
            user_filename = f'{base_name}_{user_id_str}.df'
        else:
            user_filename = f'{filename}_{user_id_str}'
        return os.path.join(user_data_dir, user_filename)
    else:
        # 标准模式：使用原有路径
        # 对于训练数据文件（.df），将其保存到图特定的 trajectories 目录
        trajectory_files = ['data_statis.df', 'replay_buffer.df', 'sampled_train.df', 'sampled_val.df', 'sampled_test.df']
        if filename.endswith('.df') and filename in trajectory_files:
            return os.path.join(data_directory, 'graph_data', graph_id, 'trajectories', filename)
        else:
            return os.path.join(data_directory, filename)

def effective_len(state):
    """
    计算状态的有效历史长度（非零状态数量）
    
    Args:
        state: 状态数组 [state_size, feature_dim]
        
    Returns:
        有效长度（非零状态的数量）
    """
    return int(sum(1 for s in state if not np.all(np.array(s) == 0)))

def normalize_edge(edge):
    """
    规范化边表示，处理有向/无向一致口径
    将边统一为 (min_node, max_node) 的元组形式
    """
    if isinstance(edge, tuple) and len(edge) == 2:
        node1, node2 = edge
        # 按坐标排序（如果是坐标元组）
        if isinstance(node1, (tuple, list)) and isinstance(node2, (tuple, list)):
            if len(node1) == 2 and len(node2) == 2:
                # 坐标格式，转换为字符串或保持元组
                return tuple(sorted([tuple(node1), tuple(node2)]))
        # 直接按值排序
        return tuple(sorted([node1, node2]))
    return edge

def build_soft_update_ops(main_vars, target_vars, tau=0.001):  # 🔥 降低tau从0.005到0.001（更慢更稳）
    return [tf.compat.v1.assign(t, (1.0 - tau) * t + tau * m)
            for t, m in zip(target_vars, main_vars)]

def build_hard_update_ops(main_vars, target_vars):
    return [tf.compat.v1.assign(t, m) for t, m in zip(target_vars, main_vars)]

def find_stage1_model(data_directory, user_id):
    """
    查找Stage-1模型（自动查找最新版本）
    
    Args:
        data_directory: 数据目录
        user_id: 用户ID（如果为None，使用default）
    
    Returns:
        模型路径（不含.ckpt扩展名），如果未找到返回None
    """
    user_folder = user_id if user_id is not None else 'default'
    stage1_dir = os.path.join(data_directory, 'saved_model', 'stage1', user_folder)
    
    if not os.path.exists(stage1_dir):
        return None
    
    # 查找所有.ckpt文件
    import glob
    pattern = os.path.join(stage1_dir, 'sl_only_*.ckpt.meta')
    matches = glob.glob(pattern)
    
    if not matches:
        return None
    
    # 按修改时间排序，返回最新的
    matches.sort(key=os.path.getmtime, reverse=True)
    latest = matches[0].replace('.meta', '')
    return latest


def get_stage1_model_path(data_directory, user_id, step, timestamp=None):
    """
    获取Stage-1模型保存路径（按阶段和用户组织）
    
    Args:
        data_directory: 数据目录
        user_id: 用户ID（如果为None，使用default）
        step: 训练步数
        timestamp: 时间戳（如果为None，自动生成）
    
    Returns:
        完整模型路径（不含.ckpt扩展名）
    """
    import time
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    user_folder = user_id if user_id is not None else 'default'
    stage1_dir = os.path.join(data_directory, 'saved_model', 'stage1', user_folder)
    os.makedirs(stage1_dir, exist_ok=True)
    
    if user_id is not None:
        model_name = f'sl_only_{user_id}_step{step}_{timestamp}'
    else:
        model_name = f'sl_only_step{step}_{timestamp}'
    
    return os.path.join(stage1_dir, model_name)

def get_precollected_path(data_directory, user_id, timestamp=None):
    """
    获取预收集数据保存路径（按阶段和用户组织）
    
    Args:
        data_directory: 数据目录
        user_id: 用户ID（如果为None，使用default）
        timestamp: 时间戳（如果为None，自动生成）
    
    Returns:
        完整文件路径
    """
    import time
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    user_folder = user_id if user_id is not None else 'default'
    precollected_dir = os.path.join(data_directory, 'precollected', 'stage1', user_folder)
    os.makedirs(precollected_dir, exist_ok=True)
    
    if user_id is not None:
        filename = f'precollected_obuffer_sl_{user_id}_{timestamp}.df'
    else:
        filename = f'precollected_obuffer_sl_{timestamp}.df'
    
    return os.path.join(precollected_dir, filename)

def find_precollected_data(data_directory, user_id):
    """
    查找预收集数据（自动查找最新版本）
    
    Args:
        data_directory: 数据目录
        user_id: 用户ID（如果为None，使用default）
    
    Returns:
        文件路径，如果未找到返回None
    """
    user_folder = user_id if user_id is not None else 'default'
    precollected_dir = os.path.join(data_directory, 'precollected', 'stage1', user_folder)
    
    if not os.path.exists(precollected_dir):
        return None
    
    # 查找所有.df文件
    import glob
    if user_id is not None:
        pattern = os.path.join(precollected_dir, f'precollected_obuffer_sl_{user_id}_*.df')
    else:
        pattern = os.path.join(precollected_dir, 'precollected_obuffer_sl_*.df')
    matches = glob.glob(pattern)
    
    if not matches:
        return None
    
    # 按修改时间排序，返回最新的
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]

def generate_action_mask_batch(states, G, edge_id_map, item_num, debug_output=False):
    """
    批量生成动作掩码
    
    🔥 新格式（9维历史序列）：
    - 不包含坐标信息，无法从图中查找邻居
    - 返回全1掩码，实际使用候选边掩码
    
    注意：此函数主要用于向后兼容，新格式数据应使用候选边机制
    """
    batch_size = len(states)
    # 🔥 新格式数据：直接返回全1掩码（候选边已在数据中指定）
    masks = np.ones((batch_size, item_num), dtype=np.float32)
    return masks
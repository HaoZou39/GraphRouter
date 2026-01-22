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

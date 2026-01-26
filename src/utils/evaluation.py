"""
Evaluation functions for the Graph Routing project.

This module contains evaluation functions for the Graph Routing project.
"""
from utils.utility import *
import numpy as np
from utils.state_processing import *

def batch_evaluate_v2(sess, model, graph_cache, feature_builder, eval_data, max_steps=50,
                     use_rl_head=False, temperature=1.0, num_trials=5):
    """
    批量评估V2 - 使用候选集决策和edge_id路径

    Args:
        sess: TensorFlow session
        model: 模型实例
        graph_cache: GraphCache实例
        feature_builder: FeatureBuilder实例
        eval_data: 评估数据 (DataFrame with route_id, start_node_id, goal_node_id, expert_path)
        max_steps: 最大步数
        use_rl_head: 是否使用RL head
        temperature: 采样温度
        num_trials: 采样次数

    Returns:
        results: 评估结果字典
    """
    results = {
        'reach_budget': [],
        'success_rate': [],
        'avg_steps': [],
        'avg_path_length': [],
        'route_ids': []
    }

    for _, row in eval_data.iterrows():
        route_id = row['route_id']
        start_node_id = row['start_node_id']
        goal_node_id = row['goal_node_id']

        trial_successes = []
        trial_steps = []
        trial_lengths = []

        for trial in range(num_trials):
            # 执行rollout
            path, edge_path, success, steps = rollout_with_candidates(
                sess=sess,
                model=model,
                start_node_id=start_node_id,
                goal_node_id=goal_node_id,
                graph_cache=graph_cache,
                feature_builder=feature_builder,
                max_steps=max_steps,
                use_rl_head=use_rl_head,
                temperature=temperature
            )

            trial_successes.append(success)
            trial_steps.append(steps)
            trial_lengths.append(len(edge_path))

        # 计算该route的指标
        success_rate = np.mean(trial_successes)
        avg_steps = np.mean(trial_steps)
        avg_length = np.mean(trial_lengths)

        # Reach@Budget (假设budget = max_steps)
        reach_budget = 1.0 if success_rate > 0 else 0.0

        results['reach_budget'].append(reach_budget)
        results['success_rate'].append(success_rate)
        results['avg_steps'].append(avg_steps)
        results['avg_path_length'].append(avg_length)
        results['route_ids'].append(route_id)

    # 计算整体指标
    results['overall_reach_budget'] = np.mean(results['reach_budget'])
    results['overall_success_rate'] = np.mean(results['success_rate'])
    results['overall_avg_steps'] = np.mean(results['avg_steps'])
    results['overall_avg_path_length'] = np.mean(results['avg_path_length'])

    return results

def calculate_path_accuracy(true_path, predicted_path):
    """
    计算路径准确率（序列级严格匹配）
    
    Args:
        true_path: GT路径（动作ID列表或边列表）
        predicted_path: 预测路径（动作ID列表或边列表）
    
    Returns:
        accuracy: 逐步严格匹配的准确率
        prefix_accuracies: Prefix@k 准确率列表
        lcs_ratio: 最长公共子序列比率
    """
    if not true_path or not predicted_path:
        return 0.0, [], 0.0
    
    # 计算逐步匹配
    min_len = min(len(true_path), len(predicted_path))
    correct_steps = 0
    prefix_accuracies = []
    
    # 🔥 辅助函数：安全比较两个值（处理数组、元组等）
    def safe_equal(a, b):
        """安全比较两个值，处理数组、元组等情况"""
        try:
            # 首先检查是否是数组（优先处理）
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                # 如果一个是数组，另一个不是，转换为数组再比较
                if isinstance(a, np.ndarray) and not isinstance(b, np.ndarray):
                    try:
                        b = np.array(b)
                    except:
                        return False
                elif isinstance(b, np.ndarray) and not isinstance(a, np.ndarray):
                    try:
                        a = np.array(a)
                    except:
                        return False
                # 现在两个都是数组，使用 np.array_equal
                return np.array_equal(a, b)
            
            # 如果是元组或列表，逐个比较
            if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
                if len(a) != len(b):
                    return False
                # 🔥 使用列表推导式而不是生成器，避免在 all() 中遇到数组比较问题
                results = []
                for i in range(len(a)):
                    try:
                        results.append(safe_equal(a[i], b[i]))
                    except (ValueError, TypeError):
                        results.append(False)
                return all(results)
            
            # 普通值直接比较（但需要处理可能的数组比较问题）
            try:
                result = a == b
                # 如果结果是数组，使用 all() 或 any()
                if isinstance(result, np.ndarray):
                    return result.all() if result.size > 0 else False
                return bool(result)
            except (ValueError, TypeError):
                return False
        except (ValueError, TypeError, AttributeError) as e:
            # 如果比较失败，返回 False
            return False
    
    for i in range(min_len):
        if safe_equal(true_path[i], predicted_path[i]):
            correct_steps += 1
        prefix_accuracies.append(correct_steps / (i + 1))
    
    accuracy = correct_steps / max(len(true_path), len(predicted_path))
    
    # 计算LCS（最长公共子序列）
    def lcs(seq1, seq2):
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # 🔥 使用安全比较函数
                if safe_equal(seq1[i-1], seq2[j-1]):
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    
    lcs_length = lcs(true_path, predicted_path)
    max_len = max(len(true_path), len(predicted_path))
    lcs_ratio = lcs_length / max_len if max_len > 0 else 0.0
    
    return accuracy, prefix_accuracies, lcs_ratio

def calculate_edge_set_metrics(true_edges, predicted_edges):
    """
    计算边的集合级指标（IoU, Precision, Recall, F1）
    
    Args:
        true_edges: GT边集合（列表）
        predicted_edges: 预测边集合（列表）
    
    Returns:
        dict: 包含 IoU, Precision, Recall, F1 的字典
    """
    # 规范化边
    true_edges_set = set(normalize_edge(e) for e in true_edges)
    predicted_edges_set = set(normalize_edge(e) for e in predicted_edges)
    
    # 计算交集和并集
    intersection = true_edges_set & predicted_edges_set
    union = true_edges_set | predicted_edges_set
    
    # 计算指标
    iou = len(intersection) / len(union) if len(union) > 0 else 0.0
    precision = len(intersection) / len(predicted_edges_set) if len(predicted_edges_set) > 0 else 0.0
    recall = len(intersection) / len(true_edges_set) if len(true_edges_set) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'intersection_size': len(intersection),
        'true_size': len(true_edges_set),
        'predicted_size': len(predicted_edges_set)
    }

def build_path_data_for_batch(group, state_size=10, edge_id_map=None, graph_cache=None, feature_builder=None):
    """
    为批量评估构建路径数据
    
    Args:
        group: 路径数据组（DataFrame）
        state_size: 状态大小（历史长度，默认10）
        edge_id_map: 边ID映射（可选，如果为None则从row中提取）
        graph_cache: GraphCache实例（新格式需要）
        feature_builder: FeatureBuilder实例（新格式需要）
    
    Returns:
        path_states: 状态序列
        path_actions: 动作序列（edge IDs）
        path_len_states: 状态长度序列
        path_length: 路径长度
        path_user_ids: 用户ID序列
        path_cand_edges: 每步的候选边列表（新格式）
        path_bc_indices: 每步的ground truth候选索引（新格式）
    """
    # 🔥 确保state_size在作用域中（防御性编程）
    _state_size = state_size if state_size is not None else 10
    
    # 导入pad_history和get_edge_id
    try:
        from utils.utility import pad_history, get_edge_id
    except ImportError:
        # 如果导入失败，定义简化版本
        def pad_history(itemlist, length, pad_item):
            """简化版pad_history"""
            if len(itemlist) < length:
                return itemlist + [pad_item] * (length - len(itemlist))
            elif len(itemlist) > length:
                return itemlist[-length:]
            return itemlist
        
        def get_edge_id(feature_vec, edge_id_map):
            """简化版get_edge_id"""
            if edge_id_map is None:
                return -1
            try:
                start_x, start_y, end_x, end_y = feature_vec[:4]
                target_coords = ((start_x, start_y), (end_x, end_y))
                return edge_id_map.get(target_coords, -1)
            except:
                return -1
    
    path_states = []
    path_actions = []
    path_len_states = []
    path_user_ids = []  # 🔥 新增：收集user_id标签
    path_cand_edges = []  # 🔥 新增：收集每步的候选边
    path_bc_indices = []  # 🔥 新增：收集每步的BC索引
    path_cur_node_ids = []  # 🔥 新增：收集每步的当前节点ID
    path_goal_node_ids = []  # 🔥 新增：收集每步的目标节点ID
    path_d_starts = []  # 🔥 新增：收集每步的起始距离
    path_prev_node_ids = []  # 🔥 新增：收集每步的前一个节点ID
    path_recent_visited = []  # 🔥 新增：收集每步的最近访问节点
    history = []
    history_edge_ids = []  # 新格式：跟踪历史edge_ids
    initial_state = None
    
    # 检查是否为新格式（有 action_edge_id 但没有 feature_vec）
    is_new_format = 'action_edge_id' in group.columns and 'feature_vec' not in group.columns
    
    if is_new_format and (graph_cache is None or feature_builder is None):
        raise ValueError("New format requires graph_cache and feature_builder")
    
    for index, row in group.iterrows():
        # 检查数据格式：新格式使用 node_id/edge_id，旧格式使用 'feature_vec'
        if is_new_format:
            # 新格式：需要构建历史序列
            action_edge_id = row.get('action_edge_id', 0)
            cur_node_id_raw = row.get('cur_node_id')
            # 处理可能的nan值
            cur_node_id = None if (cur_node_id_raw is None or
                                 (isinstance(cur_node_id_raw, float) and np.isnan(cur_node_id_raw))) else int(cur_node_id_raw)
            
            # 🔥 获取当前步的候选边
            cand_edge_ids = graph_cache.get_candidate_edges(cur_node_id)
            path_cand_edges.append(cand_edge_ids)

            # 🔥 找到action在候选边中的索引（BC index）
            bc_action_idx = None
            if action_edge_id in cand_edge_ids:
                bc_action_idx = cand_edge_ids.index(action_edge_id)
            path_bc_indices.append(bc_action_idx)

            # 🔥 收集节点信息用于候选边特征重建
            goal_node_id_raw = row.get('goal_node_id')
            # 处理可能的nan值
            goal_node_id = None if (goal_node_id_raw is None or
                                  (isinstance(goal_node_id_raw, float) and np.isnan(goal_node_id_raw))) else int(goal_node_id_raw)
            d_start = row.get('d_start', 100.0)
            prev_node_id_raw = row.get('prev_node_id')
            # 将nan值转换为None
            prev_node_id = None if (prev_node_id_raw is None or
                                  (isinstance(prev_node_id_raw, float) and np.isnan(prev_node_id_raw))) else prev_node_id_raw
            recent_visited_nodes_raw = row.get('recent_visited_nodes', [])
            # 处理可能的nan值和None值
            if recent_visited_nodes_raw is None or (isinstance(recent_visited_nodes_raw, float) and np.isnan(recent_visited_nodes_raw)):
                recent_visited_nodes = []
            else:
                recent_visited_nodes = recent_visited_nodes_raw

            path_cur_node_ids.append(cur_node_id)
            path_goal_node_ids.append(goal_node_id)
            path_d_starts.append(d_start)
            path_prev_node_ids.append(prev_node_id)
            path_recent_visited.append(recent_visited_nodes)
            
            # 构建历史序列（使用 feature_builder）
            # 对于历史边，使用静态特征（prev_node_id和recent_visited_nodes设为None）
            history_tokens = []
            for hist_edge_id in history_edge_ids:
                token = feature_builder.build_history_token(
                    chosen_edge_id=hist_edge_id,
                    prev_node_id=None,  # 历史步骤使用静态特征
                    recent_visited_nodes=None
                )
                history_tokens.append(token)
            
            # Pad到state_size
            history_dim = feature_builder._get_history_dim()
            pad_token = np.zeros(history_dim, dtype=np.float32)
            history_tokens = pad_history(history_tokens, _state_size, pad_token)
            
            # 转换为numpy数组
            state_sequence = np.array(history_tokens, dtype=np.float32)  # [state_size, history_dim]
            path_states.append(state_sequence)
            path_len_states.append(min(len(history_edge_ids), _state_size))
            
            path_actions.append(action_edge_id)

            user_id = row.get('user_id', 0)
            if not isinstance(user_id, (int, np.integer)):
                try:
                    user_id = int(user_id) if user_id is not None else 0
                except (ValueError, TypeError):
                    user_id = 0
            path_user_ids.append(user_id)
            # 更新历史
            history_edge_ids.append(action_edge_id)
            continue  # 跳过旧格式的处理逻辑
        
        # 旧格式处理逻辑
        if len(history) == 0:
            # 创建初始状态
            # 🔥 修复：使用固定索引，而不是负索引
            # feature_vec格式（前15维固定）：[start_x, start_y, end_x, end_y, crossing, path_type, length, width, curb, origin_x, origin_y, dest_x, dest_y, cur_x, cur_y]
            # 索引：              0        1        2      3      4         5          6       7      8     9         10         11     12      13     14
            # origin_x, origin_y 在索引 9, 10
            # dest_x, dest_y 在索引 11, 12
            if 'feature_vec' in row and len(row['feature_vec']) >= 15:
                start_x, start_y = row['feature_vec'][9], row['feature_vec'][10]
                end_x, end_y = row['feature_vec'][11], row['feature_vec'][12]
            elif 'feature_vec' in row:
                # 兼容旧格式（如果feature_vec只有15维，使用负索引）
                start_x, start_y = row['feature_vec'][-6], row['feature_vec'][-5]
                end_x, end_y = row['feature_vec'][-4], row['feature_vec'][-3]
            else:
                # 如果没有feature_vec，使用默认值
                start_x, start_y = 0.0, 0.0
                end_x, end_y = 1.0, 1.0
            cur_x, cur_y = start_x, start_y
            
            initial_state = [0.0] * 9 + [start_x, start_y, end_x, end_y, cur_x, cur_y]
            state_history = [initial_state]
            actual_history_length = 1
        else:
            state_history = [initial_state] + list(history)
            actual_history_length = min(len(history) + 1, _state_size)
        
        # 创建状态
        state_for_pad = state_history.copy()
        pad_item = np.zeros_like(row['feature_vec']) if 'feature_vec' in row else np.zeros(15)
        state = pad_history(state_for_pad, _state_size, pad_item)
        
        path_states.append(state)
        path_len_states.append(actual_history_length)
        
        # 获取动作
        if 'action_edge_id' in row:
            # 新格式：直接使用 action_edge_id
            action_id = row['action_edge_id']
        elif 'feature_vec' in row and edge_id_map is not None:
            # 旧格式：从 feature_vec 提取
            action = get_edge_id(row['feature_vec'], edge_id_map)
            action_id = action if action >= 0 else 0
        else:
            action_id = 0  # 默认值
        path_actions.append(action_id)
        
        # 🔥 新增：提取user_id（如果存在）
        user_id = row.get('user_id', 0)  # 默认值为0（如果没有user_id字段）
        if not isinstance(user_id, (int, np.integer)):
            # 如果是字符串或其他类型，尝试转换为整数
            try:
                user_id = int(user_id) if user_id is not None else 0
            except (ValueError, TypeError):
                user_id = 0
        path_user_ids.append(user_id)
        
        if 'feature_vec' in row:
            history.append(row['feature_vec'])
    
    return path_states, path_actions, path_len_states, len(path_actions), path_user_ids, path_cand_edges, path_bc_indices, path_cur_node_ids, path_goal_node_ids, path_d_starts, path_prev_node_ids, path_recent_visited

def batch_evaluate_improved(sess, model, dataset='val', logger=None, step=None, training_phase=1, rl_weight=0.0, batch_size=64, sample_ratio=1.0, rl_rollout_sample_size=100, G=None, edge_id_map=None, item_num=None, reward_goal=None, target_model=None, eval_num_trials=5, eval_temperature=1.0,
                            use_text_embeddings=False, text_embedding_loader=None, embedding_proj_weights=None, embedding_proj_biases=None, feature_dim=15,
                            data_directory='../data', user_id=None, moe_data_dir='examples/moe_data/processed', state_size=10, user_id_for_model=None, model_config_id=None, save_per_episode_info=False, graph_id='default_graph'):
    """
    Batch evaluation function for significantly improved performance with caching

    Args:
        eval_num_trials: Number of sampling trials (T) for path-level evaluation (default: 5)
        eval_temperature: Temperature for sampling (default: 1.0, 0.0=greedy)
        use_text_embeddings: Whether to use text embeddings (default: False)
        text_embedding_loader: TextEmbeddingLoader instance (optional)
        embedding_proj_weights: Projection weights dict (optional)
        embedding_proj_biases: Projection biases dict (optional)
        feature_dim: Original feature dimension (default: 15)
        data_directory: Data directory for map/graph data (default: '../data')
        user_id: User ID for MOE mode (default: None)
        moe_data_dir: MOE data root directory (default: 'examples/moe_data/processed')
        save_per_episode_info: Whether to save per-episode information (episode_id, success, pred_path, gt_path) to file (default: False)
    """
    import os
    import pickle
    import time
    import numpy as np
    import pandas as pd
    
    # 🔥 处理reward_goal默认值
    if reward_goal is None:
        reward_goal = 1.0  # 默认奖励值
    
    dataset_files = {
        'train': 'sampled_train.df',
        'val': 'sampled_val.df', 
        'test': 'sampled_test.df'
    }
    
    if dataset not in dataset_files:
        raise ValueError(f"Unsupported dataset: {dataset}. Supported datasets: {list(dataset_files.keys())}")
    
    # 缓存文件路径
    cache_dir = os.path.join(data_directory, 'evaluation_cache')
    os.makedirs(cache_dir, exist_ok=True)
    # 🔥 修复：在cache文件名中包含user_id、state_size和model_config_id，避免不同配置数据混淆
    # 🔥 新增：当使用用户embeddings时，在cache文件名中加入标记，强制重新构建
    user_embedding_suffix = '_ue' if (user_id is not None or hasattr(model, 'feature_dim') and model.feature_dim > feature_dim + (128 if use_text_embeddings else 0)) else ''
    cache_suffix = f'_{user_id}' if user_id is not None else ''
    config_suffix = f'_c{model_config_id}' if model_config_id is not None else ''
    # 直接用进程ID避免同时运行多个任务时的缓存冲突
    import os
    task_suffix = f'_p{os.getpid()}'
    graph_suffix = f'_g{graph_id}' if graph_id != 'default_graph' else ''  # 只在非默认图时添加图ID
    cache_file = os.path.join(cache_dir, f'{dataset}_batch_data{cache_suffix}_s{state_size}{config_suffix}{user_embedding_suffix}{graph_suffix}{task_suffix}.pkl')

    # 尝试加载缓存数据
    cache_valid = False
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                # 🔥 检查缓存中的state_size是否匹配
                cached_state_size = cached_data.get('state_size', None)
                if cached_state_size == state_size:
                    # 🔥 新增：检查缓存数据的维度是否与模型期望一致
                    if cached_data['path_data']:
                        sample_state = cached_data['path_data'][0]
                        cached_state_dim = sample_state.shape[-1] if hasattr(sample_state, 'shape') else len(sample_state[0]) if isinstance(sample_state, list) and sample_state else 15
                        expected_dim = model.feature_dim if hasattr(model, 'feature_dim') else feature_dim

                        if cached_state_dim == expected_dim:
                            all_path_data = cached_data['path_data']
                            all_path_actions = cached_data['path_actions']
                            all_path_len_states = cached_data['path_len_states']
                            all_path_lengths = cached_data['path_lengths']
                            eval_ids = cached_data['eval_ids']
                            # 🔥 新增：加载user_id（如果存在，兼容旧缓存）
                            all_path_user_ids = cached_data.get('path_user_ids', [0] * len(all_path_data))
                            # 🔥 新增：加载候选边和BC索引（如果存在，兼容旧缓存）
                            all_cand_edges = cached_data.get('cand_edges', [[]] * len(all_path_data))
                            all_bc_indices = cached_data.get('bc_indices', [None] * len(all_path_data))
                            # 🔥 新增：加载节点信息（如果存在，兼容旧缓存）
                            all_cur_node_ids = cached_data.get('cur_node_ids', [None] * len(all_path_data))
                            all_goal_node_ids = cached_data.get('goal_node_ids', [None] * len(all_path_data))
                            all_d_starts = cached_data.get('d_starts', [100.0] * len(all_path_data))
                            all_prev_node_ids = cached_data.get('prev_node_ids', [None] * len(all_path_data))
                            all_recent_visited = cached_data.get('recent_visited', [[]] * len(all_path_data))
                            cache_valid = True
                            print(f'✅ Loaded cached evaluation data (state_size={state_size}, dim={cached_state_dim})')
                        else:
                            print(f'⚠️  Cache state dimension mismatch (cached: {cached_state_dim}, expected: {expected_dim}), rebuilding...')
                    else:
                        print(f'⚠️  Cache data is empty, rebuilding...')
                else:
                    print(f'⚠️  Cache state_size mismatch (cached: {cached_state_size}, required: {state_size}), rebuilding...')
        except Exception as e:
            print(f'Failed to load cache: {e}, rebuilding data...')
    else:
        print(f'No cache found for {dataset} dataset, building data...')

    # 🔥 加载 graph_cache 和 feature_builder（用于新格式数据）
    # 即使命中缓存也需要加载，因为后续评估逻辑需要这些对象
    graph_cache = None
    global_feature_builder = None  # 使用不同的变量名避免作用域冲突
    try:
        from utils.graph_cache import MultiGraphCache
        multi_cache = MultiGraphCache(os.path.join(data_directory, 'graph_data'))
        graph_cache = multi_cache.get_cache(graph_id)
        global_feature_builder = multi_cache.get_feature_builder(graph_id)
        print(f'✅ Loaded GraphCache and FeatureBuilder for graph "{graph_id}"')
    except Exception as e:
        print(f'⚠️  Could not load GraphCache/FeatureBuilder: {e}')
        print(f'   Assuming old format data (with feature_vec)')
        raise Exception(f'Could not load GraphCache/FeatureBuilder: {e}')

    # 为后续使用设置别名
    feature_builder = global_feature_builder

    # 初始化节点信息变量（用于缓存兼容性）
    if not cache_valid:
        # 缓存无效时，这些变量会在数据重建过程中设置
        all_cur_node_ids = []
        all_goal_node_ids = []
        all_d_starts = []
        all_prev_node_ids = []
        all_recent_visited = []

    # 如果没有缓存或加载失败，重新构建数据
    if not cache_valid:
        # 初始化all_path_user_ids（如果从缓存加载失败）
        if 'all_path_user_ids' not in locals():
            all_path_user_ids = []
        data_file = dataset_files[dataset]
        # 支持MOE模式
        eval_data_path = get_moe_data_path(data_directory, user_id, moe_data_dir, data_file)
        # 🔥 打印使用的数据文件路径和统计信息（用于调试）
        print(f'📊 Using {dataset.upper()} dataset: {eval_data_path}')
        if user_id is not None:
            print(f'   MOE Mode: user_id={user_id}, moe_data_dir={moe_data_dir}')
        if os.path.exists(eval_data_path):
            eval_sessions_temp = pd.read_pickle(eval_data_path)
            num_paths = len(eval_sessions_temp.route_id.unique())
            num_steps = len(eval_sessions_temp)
            print(f'   ✅ Dataset loaded: {num_paths} paths, {num_steps} total steps')
            # 🔥 调试：检查数据中是否有user_id列
            if 'user_id' in eval_sessions_temp.columns:
                unique_users = eval_sessions_temp['user_id'].unique()
                print(f'   📊 Data contains user_ids: {unique_users[:10]}... (showing first 10)')
            else:
                print(f'   ⚠️  Data does not contain user_id column')
        if not os.path.exists(eval_data_path):
            # 🔥 修复：如果val数据集不存在，返回None而不是抛出异常（支持train:test 2:8分割）
            if dataset == 'val':
                print(f'⚠️  VAL dataset not found: {eval_data_path}')
                print(f'   Skipping VAL evaluation (dataset may be split as train:test only)')
                return None
            else:
                raise FileNotFoundError(f"Evaluation data file not found: {eval_data_path}")
        eval_sessions = pd.read_pickle(eval_data_path)
        eval_ids = eval_sessions.route_id.unique()
        
        print(f'Start batch evaluating {dataset.upper()} dataset...')
        print(f'Dataset size: {len(eval_ids)} paths, Batch size: {batch_size}')
        
        # 预计算所有路径数据
        all_path_data = []
        all_path_actions = []
        all_path_len_states = []
        all_path_lengths = []
        all_path_user_ids = []  # 🔥 新增：收集所有user_id标签
        all_cand_edges = []  # 🔥 新增：收集所有候选边
        all_bc_indices = []  # 🔥 新增：收集所有BC索引
        all_cur_node_ids = []  # 🔥 新增：收集所有当前节点ID
        all_goal_node_ids = []  # 🔥 新增：收集所有目标节点ID
        all_d_starts = []  # 🔥 新增：收集所有起始距离
        all_prev_node_ids = []  # 🔥 新增：收集所有前一个节点ID
        all_recent_visited = []  # 🔥 新增：收集所有最近访问节点
        
        for route_id in eval_ids:
            group = eval_sessions[eval_sessions['route_id'] == route_id]
            if len(group) == 0:
                continue
            
            # 构建路径数据
            # 🔥 修复：需要传递state_size和edge_id_map参数
            # state_size应该与模型的state_size匹配，使用传入的state_size参数
            path_states, path_actions, path_len_states, path_length, path_user_ids, path_cand_edges, path_bc_indices, path_cur_node_ids, path_goal_node_ids, path_d_starts, path_prev_node_ids, path_recent_visited = build_path_data_for_batch(
                group, state_size=state_size, edge_id_map=edge_id_map,
                graph_cache=graph_cache, feature_builder=global_feature_builder
            )
            all_path_data.extend(path_states)
            all_path_actions.extend(path_actions)
            all_path_len_states.extend(path_len_states)
            all_path_lengths.append(path_length)
            all_path_user_ids.extend(path_user_ids)  # 🔥 新增：收集user_id
            all_cand_edges.extend(path_cand_edges)  # 🔥 新增：收集候选边
            all_bc_indices.extend(path_bc_indices)  # 🔥 新增：收集BC索引
            all_cur_node_ids.extend(path_cur_node_ids)  # 🔥 新增：收集当前节点ID
            all_goal_node_ids.extend(path_goal_node_ids)  # 🔥 新增：收集目标节点ID
            all_d_starts.extend(path_d_starts)  # 🔥 新增：收集起始距离
            all_prev_node_ids.extend(path_prev_node_ids)  # 🔥 新增：收集前一个节点ID
            all_recent_visited.extend(path_recent_visited)  # 🔥 新增：收集最近访问节点
        
        print(f'Precomputed {len(all_path_data)} states')
        
        # 保存缓存数据
        try:
            cached_data = {
                'path_data': all_path_data,
                'path_actions': all_path_actions,
                'path_len_states': all_path_len_states,
                'path_lengths': all_path_lengths,
                'path_user_ids': all_path_user_ids,  # 🔥 新增：保存user_id
                'cand_edges': all_cand_edges,  # 🔥 新增：保存候选边
                'bc_indices': all_bc_indices,  # 🔥 新增：保存BC索引
                'cur_node_ids': all_cur_node_ids,  # 🔥 新增：保存当前节点ID
                'goal_node_ids': all_goal_node_ids,  # 🔥 新增：保存目标节点ID
                'd_starts': all_d_starts,  # 🔥 新增：保存起始距离
                'prev_node_ids': all_prev_node_ids,  # 🔥 新增：保存前一个节点ID
                'recent_visited': all_recent_visited,  # 🔥 新增：保存最近访问节点
                'eval_ids': eval_ids,
                'state_size': state_size  # 🔥 新增：保存state_size用于验证
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
            print(f'Cached data saved to: {cache_file}')
        except Exception as e:
            print(f'Failed to save cache: {e}')
    
    # 采样训练集评估
    if dataset == 'train' and sample_ratio < 1.0:
        total_paths = len(all_path_lengths)
        sample_size = int(total_paths * sample_ratio)
        if sample_size < 1:
            sample_size = 1
        sampled_indices = np.random.choice(total_paths, sample_size, replace=False)
        # 采样路径索引对应的步数范围
        sampled_path_lengths = [all_path_lengths[i] for i in sampled_indices]
        # 计算每条路径的起止步数
        path_start_ends = []
        start = 0
        for l in all_path_lengths:
            end = start + l
            path_start_ends.append((start, end))
            start = end
        # 采样所有步的索引
        sampled_steps = []
        sampled_actions = []
        sampled_len_states = []
        sampled_user_ids = []  # 🔥 新增：采样user_id
        sampled_cand_edges = []  # 🔥 新增：采样候选边
        sampled_bc_indices = []  # 🔥 新增：采样BC索引
        for idx in sampled_indices:
            s, e = path_start_ends[idx]
            sampled_steps.extend(range(s, e))
            sampled_actions.extend(all_path_actions[s:e])
            sampled_len_states.extend(all_path_len_states[s:e])
            sampled_user_ids.extend(all_path_user_ids[s:e])  # 🔥 新增：采样user_id
            sampled_cand_edges.extend(all_cand_edges[s:e])  # 🔥 新增：采样候选边
            sampled_bc_indices.extend(all_bc_indices[s:e])  # 🔥 新增：采样BC索引
        all_path_data = [all_path_data[i] for i in sampled_steps]
        all_path_actions = sampled_actions
        all_path_len_states = sampled_len_states
        all_path_user_ids = sampled_user_ids  # 🔥 新增：更新user_id列表
        all_cand_edges = sampled_cand_edges  # 🔥 新增：更新候选边列表
        all_bc_indices = sampled_bc_indices  # 🔥 新增：更新BC索引列表
        all_path_lengths = sampled_path_lengths
        print(f"Sampled {sample_size} paths for train evaluation, total {len(all_path_data)} steps.")
    
    # 🔥 加载用户 embedding（如果启用）
    user_embedding = None
    user_embeddings_dict = None

    # 检查是否需要加载用户embeddings（基于模型的feature_dim）
    # 计算当前配置下的预期特征维度（不含用户embedding）
    text_dim = 0
    if use_text_embeddings and text_embedding_loader is not None:
        # 从embedding_proj_weights推断实际使用了哪些embedding
        if embedding_proj_weights:
            if 'edge' in embedding_proj_weights:
                text_dim += embedding_proj_weights['edge'].shape[1]
            if 'node' in embedding_proj_weights:
                text_dim += embedding_proj_weights['node'].shape[1]
            if 'goal' in embedding_proj_weights:
                text_dim += embedding_proj_weights['goal'].shape[1]

    expected_feature_dim_without_user = feature_dim + text_dim
    model_expects_user_embedding = (hasattr(model, 'feature_dim') and
                                   model.feature_dim > expected_feature_dim_without_user)

    if model_expects_user_embedding or user_id is not None:
        if user_id is not None:
            # 单用户模式：加载指定用户的embedding
            print(f"🧑 Loading user embedding for {user_id}...")
            user_embedding = load_user_embedding(user_id, moe_data_dir)
            if user_embedding is not None:
                user_dim = len(user_embedding)
                print(f"✅ User embedding loaded: dim={user_dim}")
            else:
                print(f"⚠️  User embedding not found for {user_id}, will pad with zeros if needed")
        else:
            # 多用户模式：加载所有用户的embeddings
            print(f"🧑 Loading all user embeddings for multi-user evaluation...")
            user_embeddings_dict = load_all_user_embeddings(moe_data_dir)
            if user_embeddings_dict is not None and len(user_embeddings_dict) > 0:
                first_user_emb = next(iter(user_embeddings_dict.values()))
                user_dim = len(first_user_emb)
                print(f"✅ Loaded {len(user_embeddings_dict)} user embeddings (dim={user_dim})")
            else:
                print(f"⚠️  No user embeddings found in {moe_data_dir}, will pad with zeros if needed")
                user_embeddings_dict = None
    
    # 批量预测
    all_predictions = []
    # 新增：RL head预测
    all_rl_predictions = []
    
    # 🔥 新增：用户分类预测
    
    # 新增：有效动作Q-values监控
    all_valid_q_values = []
    all_valid_q_counts = []

    for i in range(0, len(all_path_data), batch_size):
        batch_states = all_path_data[i:i+batch_size]
        batch_len_states = all_path_len_states[i:i+batch_size]
        # 🔥 提取当前批次的user_ids
        batch_user_ids = all_path_user_ids[i:i+batch_size] if all_path_user_ids else [0] * len(batch_states)
        # 🔥 提取当前批次的候选边和BC索引
        batch_cand_edges = all_cand_edges[i:i+batch_size] if all_cand_edges else [None] * len(batch_states)
        batch_bc_indices = all_bc_indices[i:i+batch_size] if all_bc_indices else [None] * len(batch_states)
        # 🔥 提取当前批次的节点信息（现在这些变量始终存在）
        batch_cur_node_ids = all_cur_node_ids[i:i+batch_size] if all_cur_node_ids else [None] * len(batch_states)
        batch_goal_node_ids = all_goal_node_ids[i:i+batch_size] if all_goal_node_ids else [None] * len(batch_states)
        batch_d_starts = all_d_starts[i:i+batch_size] if all_d_starts else [100.0] * len(batch_states)
        batch_prev_node_ids = all_prev_node_ids[i:i+batch_size] if all_prev_node_ids else [None] * len(batch_states)
        batch_recent_visited = all_recent_visited[i:i+batch_size] if all_recent_visited else [[]] * len(batch_states)
        
        # 🔥 修复：检查模型期望的特征维度，如果模型期望增强特征（model.feature_dim > feature_dim），
        # 则必须增强数据，即使 use_text_embeddings=False
        # 这确保加载的模型（用143维训练）能正确评估
        model_expects_enhanced = model.feature_dim > feature_dim
        
        if model_expects_enhanced:
            # 模型期望增强特征，必须增强数据
            if text_embedding_loader is None:
                raise ValueError(
                    f"❌ Model expects feature_dim={model.feature_dim} (enhanced), but text_embedding_loader is None. "
                    f"Please provide text_embedding_loader and related parameters."
                )
            # 确保batch_states是numpy数组
            if isinstance(batch_states, list):
                batch_states = np.array(batch_states)
            batch_states = enhance_states_with_text_embeddings(
                batch_states, G, edge_id_map,
                text_embedding_loader=text_embedding_loader,
                embedding_proj_weights=embedding_proj_weights,
                embedding_proj_biases=embedding_proj_biases,
                original_feature_dim=feature_dim,
                user_embedding=user_embedding,
                user_embeddings_dict=user_embeddings_dict,
                user_ids=batch_user_ids
            )
        elif use_text_embeddings and text_embedding_loader is not None:
            # 模型不期望增强特征，但如果use_text_embeddings=True，也增强（用于兼容）
            if isinstance(batch_states, list):
                batch_states = np.array(batch_states)
            batch_states = enhance_states_with_text_embeddings(
                batch_states, G, edge_id_map,
                text_embedding_loader=text_embedding_loader,
                embedding_proj_weights=embedding_proj_weights,
                embedding_proj_biases=embedding_proj_biases,
                original_feature_dim=feature_dim,
                user_embedding=user_embedding,
                user_embeddings_dict=user_embeddings_dict,
                user_ids=batch_user_ids
            )
        
        # 🔥 生成动作掩码
        # 新格式（feature_dim=9，历史token）不需要从坐标生成mask，直接允许所有动作
        if feature_dim == 9:
            # 新格式：状态是 [batch, state_size, 9] 历史序列，无坐标信息
            # 候选边信息已经在数据中，不需要mask（模型只在候选边中选择）
            batch_masks = np.ones((len(batch_states), item_num), dtype=np.float32)
        else:
            # 旧格式：从状态中提取坐标生成mask
            if model_expects_enhanced:
                # 提取原始特征用于动作掩码生成（动作掩码生成需要原始15维特征）
                batch_states_for_mask = batch_states[:, :, :feature_dim] if batch_states.ndim == 3 else batch_states
            else:
                batch_states_for_mask = batch_states
            
            batch_masks = generate_action_mask_batch(batch_states_for_mask, G, edge_id_map, item_num, debug_output=True)

        # 🔥 准备候选边特征数据（用于新格式SL推理）
        batch_cand_features = None
        batch_cand_masks = None
        # 使用全局变量，避免作用域问题
        if global_feature_builder is not None and batch_cand_edges:
            # 为当前批次准备候选边特征
            batch_cand_features_list = []
            batch_cand_masks_list = []

            for step_idx, step_cand_edges in enumerate(batch_cand_edges):
                if step_cand_edges and len(step_cand_edges) > 0:
                    # 使用真实的节点信息重建候选边特征
                    cur_node_id = batch_cur_node_ids[step_idx]
                    goal_node_id = batch_goal_node_ids[step_idx]
                    d_start = batch_d_starts[step_idx]
                    prev_node_id = batch_prev_node_ids[step_idx]
                    recent_visited_nodes = batch_recent_visited[step_idx]

                    if cur_node_id is not None and goal_node_id is not None and global_feature_builder is not None:
                        # 调试：检查节点是否存在于图中
                        if cur_node_id not in graph_cache.node_coords:
                            raise ValueError(f"Current node {cur_node_id} does not exist in graph {graph_id} for step {step_idx}. "
                                           f"Available nodes: {len(graph_cache.node_coords)} total, "
                                           f"max ID: {max(graph_cache.node_coords.keys()) if graph_cache.node_coords else 'N/A'}")
                        if goal_node_id not in graph_cache.node_coords:
                            raise ValueError(f"Goal node {goal_node_id} does not exist in graph for step {step_idx}. "
                                           f"Available nodes: {len(graph_cache.node_coords)} total, "
                                           f"max ID: {max(graph_cache.node_coords.keys()) if graph_cache.node_coords else 'N/A'}")

                        # 调试：检查候选边是否都存在
                        missing_edges = []
                        for edge_id in step_cand_edges:
                            # 这里需要检查edge_id是否存在，取决于graph_cache的API
                            pass  # 暂时跳过，后面再检查

                        # 重建真实的候选边特征（必须成功，不能使用默认值）
                        try:
                            cand_features = global_feature_builder.build_candidate_features(
                                cand_edge_ids=step_cand_edges,
                                cur_node_id=cur_node_id,
                                goal_node_id=goal_node_id,
                                d_start=d_start,
                                prev_node_id=prev_node_id,
                                recent_visited_nodes=recent_visited_nodes
                            )
                        except Exception as e:
                            # 提供详细的错误信息
                            raise ValueError(f"Failed to build candidate features for step {step_idx}: {str(e)}. "
                                           f"cur_node_id={cur_node_id}, goal_node_id={goal_node_id}, "
                                           f"cand_edge_ids={step_cand_edges[:5]}..., d_start={d_start}")

                        # 检查是否有nan值
                        if np.any(np.isnan(cand_features)):
                            raise ValueError(f"Candidate features contain NaN values for step {step_idx}. "
                                           f"Feature stats: min={np.nanmin(cand_features):.3f}, "
                                           f"max={np.nanmax(cand_features):.3f}, "
                                           f"mean={np.nanmean(cand_features):.3f}")
                    else:
                        # 如果没有节点信息，必须报错，不能使用默认特征
                        raise ValueError(f"Missing required data for candidate feature reconstruction at step {step_idx}: "
                                       f"cur_node_id={cur_node_id}, goal_node_id={goal_node_id}, "
                                       f"feature_builder={global_feature_builder is not None}")

                    cand_mask = np.ones(len(step_cand_edges), dtype=np.float32)
                else:
                    raise 
                    # 没有候选边信息，创建默认特征
                    cand_features = np.zeros((model.max_candidates, 12), dtype=np.float32)
                    # 为默认候选边创建基本特征
                    for i in range(model.max_candidates):
                        cand_features[i, 0] = float(i) / float(model.max_candidates)  # 位置编码
                        cand_features[i, 2] = 1.0  # 默认长度
                        cand_features[i, 3] = 0.5  # 默认宽度
                    cand_mask = np.zeros(model.max_candidates, dtype=np.float32)

                batch_cand_features_list.append(cand_features)
                batch_cand_masks_list.append(cand_mask)

            if batch_cand_features_list:
                # 填充到模型期望的固定大小 [batch_size, max_candidates, feature_dim]
                batch_cand_features = np.zeros((len(batch_states), model.max_candidates, 12), dtype=np.float32)
                batch_cand_masks = np.zeros((len(batch_states), model.max_candidates), dtype=np.float32)

                for j, (cf, cm) in enumerate(zip(batch_cand_features_list, batch_cand_masks_list)):
                    # 截断到模型的最大候选边数量
                    actual_len = min(len(cf), model.max_candidates)
                    batch_cand_features[j, :actual_len] = cf[:actual_len]
                    batch_cand_masks[j, :actual_len] = cm[:actual_len]

        # 为评估创建虚拟的potential_labels（评估时不需要真实标签）
        dummy_potential_labels = np.zeros((len(batch_states), 1), dtype=np.float32)

        # SL head prediction (softmax probability)
        # 🔥 构建基础feed_dict
        base_feed_dict = {
            model.inputs: batch_states,
            model.len_state: batch_len_states,
            model.action_mask: batch_masks,
            model.is_training: False,
            model.training_phase: training_phase,
            model.rl_weight: rl_weight
        }

        # 🔥 添加候选边特征（如果有的话）
        if batch_cand_features is not None and hasattr(model, 'cand_features'):
            base_feed_dict[model.cand_features] = batch_cand_features
            base_feed_dict[model.cand_mask] = batch_cand_masks
        
        # 🔥 如果模型有user_id_ph占位符（PersonalizedQNetwork且为lora模式），需要提供user_id
        if hasattr(model, 'user_id_ph') and model.user_id_ph is not None:
            # 从batch_states中提取user_id，或者使用默认值
            # 如果batch_states来自评估数据，可能没有user_id信息，使用默认值
            batch_size = batch_states.shape[0]
            # 尝试从数据中获取user_id，如果没有则使用传入的user_id或默认值0
            # 🔥 优先使用user_id_for_model（整数），如果没有则从user_id（可能是字符串）中提取
            if user_id_for_model is not None:
                # 直接使用提供的整数user_id
                batch_user_ids_eval = np.full(batch_size, int(user_id_for_model), dtype=np.int32)
            elif user_id is not None:
                # 🔥 修复：如果user_id是字符串（如'user_001'），需要转换为整数
                # 提取数字部分（如'user_001' -> 1）
                if isinstance(user_id, str):
                    # 尝试提取数字部分
                    import re
                    numbers = re.findall(r'\d+', user_id)
                    if numbers:
                        user_id_int = int(numbers[0])  # 取第一个数字
                    else:
                        user_id_int = 0  # 如果无法提取，使用0
                else:
                    user_id_int = int(user_id)
                batch_user_ids_eval = np.full(batch_size, user_id_int, dtype=np.int32)
            else:
                # 如果没有提供user_id，使用0作为默认值
                batch_user_ids_eval = np.zeros(batch_size, dtype=np.int32)
            base_feed_dict[model.user_id_ph] = batch_user_ids_eval
        
        # 🔥 SL head prediction
        # 现在模型直接输出候选边空间的预测
        if hasattr(model, 'probs_personalized'):
            # 个性化模型：使用个性化输出
            batch_preds = sess.run(model.probs_personalized, feed_dict=base_feed_dict)
        else:
            # 基础模型：直接输出候选边空间的预测
            batch_preds = sess.run(model.probs, feed_dict=base_feed_dict)

        # 🔥 后处理：确保预测与候选边数量匹配
        processed_preds = []
        for j, step_cand_edges in enumerate(batch_cand_edges):
            if step_cand_edges and len(step_cand_edges) > 0:
                # 模型输出已经是候选边空间，直接使用对应数量的预测
                num_cand = min(len(step_cand_edges), model.max_candidates)
                cand_scores = batch_preds[j][:num_cand]
                processed_preds.append(cand_scores)
            else:
                # 如果没有候选边信息，使用预测的前N个分数
                max_cand = getattr(model, 'max_candidates', 20)
                processed_preds.append(batch_preds[j][:max_cand])
        # 🔥 修复警告：使用 dtype=object 处理不规则嵌套序列
        batch_preds = np.array(processed_preds, dtype=object)
        
        # RL head prediction (Q values)
        if hasattr(model, 'output1_personalized_masked'):
            # 个性化模型：使用个性化输出
            batch_rl_q = sess.run(model.output1_personalized_masked, feed_dict=base_feed_dict)
        else:
            # 基础模型：使用标准输出
            batch_rl_q = sess.run(model.output1_masked, feed_dict=base_feed_dict)
        

        # 新增：监控有效动作的Q-values（现在使用候选边空间）
        for j in range(len(batch_rl_q)):
            if batch_cand_masks is not None:
                valid_mask = batch_cand_masks[j] > 0
            else:
                raise ValueError("No candidate masks found for evaluation")
                # 如果没有候选边mask，假设所有候选边都有效
                valid_mask = np.ones(model.max_candidates, dtype=bool)
            valid_q_values = batch_rl_q[j][valid_mask]
            if len(valid_q_values) > 0:
                all_valid_q_values.extend(valid_q_values)
                all_valid_q_counts.append(len(valid_q_values))

        # 对于每个样本，我们需要找到在候选边中的最佳选择
        batch_start_idx = len(all_predictions)
        for j in range(len(batch_preds)):
            step_idx = batch_start_idx + j
            # 获取当前步的候选边
            step_cand_edges = all_cand_edges[step_idx]
            
            # 现在SL输出已经是候选边空间，直接使用argmax得到候选索引
            if step_cand_edges and len(step_cand_edges) > 0:
                # SL输出已经是候选边空间，直接找到最高分的索引
                pred_cand_idx = np.argmax(batch_preds[j])  # batch_preds[j] 是 [max_candidates] 的概率分布
                rl_pred_cand_idx = np.argmax(batch_rl_q[j])  # batch_rl_q[j] 也是 [max_candidates]

                # 保存预测的候选索引（用于与bc_action_idx比较）
                all_predictions.append(pred_cand_idx)
                all_rl_predictions.append(rl_pred_cand_idx)
            else:
                raise ValueError("No candidate edges found for evaluation")

    # 🔥 计算SL head指标：对于新格式，比较候选索引；对于旧格式，比较edge_id
    total_steps = len(all_predictions)
    total_correct_steps = 0
    total_correct_steps_rl = 0
    
    for i in range(total_steps):
        step_bc_idx = all_bc_indices[i]
        step_cand_edges = all_cand_edges[i]
        
        if step_bc_idx is not None and step_cand_edges and len(step_cand_edges) > 0:
            # 新格式：比较候选索引
            if all_predictions[i] == step_bc_idx:
                total_correct_steps += 1
            if all_rl_predictions[i] == step_bc_idx:
                total_correct_steps_rl += 1
        else:
            # 旧格式：比较edge_id
            if all_predictions[i] == all_path_actions[i]:
                total_correct_steps += 1
            if all_rl_predictions[i] == all_path_actions[i]:
                total_correct_steps_rl += 1
    
    avg_step_correct = total_correct_steps / total_steps if total_steps > 0 else 0
    avg_step_correct_rl = total_correct_steps_rl / total_steps if total_steps > 0 else 0
    

    # 计算路径级指标和reward
    path_correctness_list = []
    path_correctness_list_rl = []
    successful_paths = 0
    successful_paths_rl = 0
    path_idx = 0
    total_reward = 0.0  # 添加reward计算

    for path_length in all_path_lengths:
        path_predictions = all_predictions[path_idx:path_idx + path_length]
        path_rl_predictions = all_rl_predictions[path_idx:path_idx + path_length]
        path_targets = all_path_actions[path_idx:path_idx + path_length]
        path_bc_idxs = all_bc_indices[path_idx:path_idx + path_length]
        path_cands = all_cand_edges[path_idx:path_idx + path_length]

        # 🔥 计算路径准确率：根据是否有BC索引决定比较方式
        path_correct = 0
        path_correct_rl = 0
        for i in range(path_length):
            if path_bc_idxs[i] is not None and path_cands[i] and len(path_cands[i]) > 0:
                # 新格式：比较候选索引
                if path_predictions[i] == path_bc_idxs[i]:
                    path_correct += 1
                if path_rl_predictions[i] == path_bc_idxs[i]:
                    path_correct_rl += 1
            else:
                # 旧格式：比较edge_id
                if path_predictions[i] == path_targets[i]:
                    path_correct += 1
                if path_rl_predictions[i] == path_targets[i]:
                    path_correct_rl += 1
        
        path_correctness = path_correct / path_length if path_length > 0 else 0
        path_correctness_list.append(path_correctness)
        if path_correct == path_length:
            successful_paths += 1

        # RL head
        path_correctness_rl = path_correct_rl / path_length if path_length > 0 else 0
        path_correctness_list_rl.append(path_correctness_rl)
        if path_correct_rl == path_length:
            successful_paths_rl += 1

        # 🔥 计算路径的reward（包含步时惩罚和回退惩罚）
        # 注意：在batch_evaluate_improved中，我们只计算准确率等指标，不计算详细奖励
        # 因为缺少状态信息，无法准确计算距离变化
        # 奖励计算留给其他函数（如collect_onpolicy_transitions）处理
        
        # 简化处理：只计算基本的步时惩罚
        for step_idx in range(path_length):
            # 🔥 修复：is_done应该基于是否真正到达目标，使用候选索引比较
            step_correct = False
            if path_bc_idxs[step_idx] is not None and path_cands[step_idx] and len(path_cands[step_idx]) > 0:
                # 新格式：比较候选索引
                step_correct = (path_predictions[step_idx] == path_bc_idxs[step_idx])
            else:
                # 旧格式：比较edge_id
                step_correct = (path_predictions[step_idx] == path_targets[step_idx])
            
            # 这里简化处理：假设最后一步且预测正确才算到达
            is_done = (step_idx == len(path_predictions) - 1) and step_correct
            
            # 简化奖励：只使用步时惩罚，避免复杂的距离计算
            reward = -0.01 if not is_done else reward_goal  # 简单的步时惩罚 + 目标奖励
            
            total_reward += reward
        path_idx += path_length

    total_paths = len(all_path_lengths)
    avg_path_correctness = np.mean(path_correctness_list) if path_correctness_list else 0
    avg_path_correctness_rl = np.mean(path_correctness_list_rl) if path_correctness_list_rl else 0
    path_success_rate = successful_paths / total_paths if total_paths > 0 else 0
    path_success_rate_rl = successful_paths_rl / total_paths if total_paths > 0 else 0
    avg_path_length = np.mean(all_path_lengths) if all_path_lengths else 0

    # 新增：计算有效动作Q-values统计
    valid_q_stats = {}
    if all_valid_q_values:
        valid_q_array = np.array(all_valid_q_values)
        valid_q_stats = {
            'mean': float(np.mean(valid_q_array)),
            'std': float(np.std(valid_q_array)),
            'min': float(np.min(valid_q_array)),
            'max': float(np.max(valid_q_array)),
            'range': float(np.max(valid_q_array) - np.min(valid_q_array)),
            'count': len(all_valid_q_values),
            'avg_valid_actions_per_step': float(np.mean(all_valid_q_counts)) if all_valid_q_counts else 0.0
        }
    else:
        valid_q_stats = {
            'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'range': 0.0,
            'count': 0, 'avg_valid_actions_per_step': 0.0
        }

    # ============================================================================
    # 🔥 新评估流程：A. 单步准确率（已在上面计算）
    # ============================================================================
    print(f'\n{dataset.upper()} DATASET EVALUATION RESULTS')
    print(f'Total Steps: {total_steps}, Total Paths: {total_paths}')
    print(f'SL Head - Step-level Accuracy: {avg_step_correct:.4f}')
    
    # ============================================================================
    # 🔥 新评估流程：B. 全测试集的到达率（Reach@B）和 C. 路径准确率/覆盖度
    # ============================================================================
    rollout_results = []  # 初始化为空列表，避免作用域问题
    path_accuracy_metrics = {}  # 初始化为空字典
    reach_at_budget = 0.0  # 初始值
    budget_B = 60
    arrival_threshold = 1e-6
    
    if dataset == 'test':  # 只在test集上做rollout评估
        print(f'\n🔄 Performing Top-{eval_num_trials} sampling rollouts with SL head (temperature={eval_temperature})...')
        arrival_threshold = 1e-6  # 固定的到达阈值
        budget_B = 60  # 预算步数
        
        rollout_results = []
        path_accuracy_metrics = {
            'seq_acc': [],
            'prefix_k': [],  # Prefix@k列表（取k=5）
            'lcs_ratio': [],
            'iou': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'successful_paths': []  # 成功到达的路径的指标
        }

        # 🔥 Per-episode information for saving
        per_episode_info = [] if save_per_episode_info else None
        
        path_start_idx = 0
        for path_idx, path_length in enumerate(all_path_lengths):
            if path_idx % 50 == 0:
                print(f'  Rolling out path {path_idx+1}/{len(all_path_lengths)}...')
            
            # 获取起始状态和目标位置
            start_state = all_path_data[path_start_idx]
            target_pos = extract_target_position_from_state(start_state)

            if target_pos is None:
                path_start_idx += path_length
                continue

            # 🔥 对start_state进行增强，确保维度与模型期望一致
            if not isinstance(start_state, np.ndarray):
                start_state = np.array(start_state)

            # 🔥 获取该路径对应的用户embedding
            path_user_id = all_path_user_ids[path_start_idx] if all_path_user_ids else 0
            user_embedding_for_path = None
            if user_embeddings_dict is not None and path_user_id is not None:
                if isinstance(path_user_id, (int, np.integer)):
                    user_id_str = f'user_{path_user_id + 1:03d}'
                else:
                    user_id_str = str(path_user_id)
                user_embedding_for_path = user_embeddings_dict.get(user_id_str, None)

            # 检查模型期望的特征维度
            model_expects_enhanced = hasattr(model, 'feature_dim') and model.feature_dim > feature_dim

            if model_expects_enhanced:
                # 模型期望增强特征，必须增强start_state
                if text_embedding_loader is None:
                    raise ValueError(
                        f"❌ Model expects feature_dim={model.feature_dim} (enhanced), but text_embedding_loader is None. "
                        f"Please provide text_embedding_loader and related parameters."
                    )

                # start_state已经是(state_size, feature_dim)的形状，直接增强
                start_state = enhance_states_with_text_embeddings(
                    start_state[np.newaxis, :, :], G, edge_id_map,  # 添加batch维度
                    text_embedding_loader=text_embedding_loader,
                    embedding_proj_weights=embedding_proj_weights,
                    embedding_proj_biases=embedding_proj_biases,
                    original_feature_dim=feature_dim,
                    user_embedding=user_embedding_for_path
                )
                start_state = start_state[0]  # 移除batch维度，变回(state_size, enhanced_feature_dim)
            elif use_text_embeddings and text_embedding_loader is not None:
                # 模型不期望增强特征，但如果use_text_embeddings=True，也增强（用于兼容）
                start_state = enhance_states_with_text_embeddings(
                    start_state[np.newaxis, :, :], G, edge_id_map,  # 添加batch维度
                    text_embedding_loader=text_embedding_loader,
                    embedding_proj_weights=embedding_proj_weights,
                    embedding_proj_biases=embedding_proj_biases,
                    original_feature_dim=feature_dim,
                    user_embedding=user_embedding_for_path
                )
                start_state = start_state[0]  # 移除batch维度，变回(state_size, enhanced_feature_dim)
            else:
                # 如果不需要增强，但模型期望的用户embeddings维度仍然缺失，进行zero padding
                if hasattr(model, 'feature_dim') and start_state.shape[-1] < model.feature_dim:
                    missing_dims = model.feature_dim - start_state.shape[-1]
                    padding = np.zeros((start_state.shape[0], missing_dims))  # start_state是(state_size, feature_dim)
                    start_state = np.concatenate([start_state, padding], axis=-1)
            
            # 🔥 Top-T Sampling: 执行T次rollout，记录成功率和最佳路径
            try:
                trial_results = []  # 存储T次试验的结果
                gt_path = all_path_actions[path_start_idx:path_start_idx + path_length]
                
                for trial_idx in range(eval_num_trials):
                    predicted_path, csv_origin, csv_destination, success, trajectory = predict_path_with_temperature_sampling(
                        sess, model, start_state, target_pos, budget_B, G, edge_id_map, item_num,
                        use_rl_head=False, arrival_threshold=arrival_threshold, temperature=eval_temperature,
                        use_text_embeddings=use_text_embeddings, text_embedding_loader=text_embedding_loader,
                        embedding_proj_weights=embedding_proj_weights,
                        embedding_proj_biases=embedding_proj_biases,
                        feature_dim=feature_dim,
                        user_id_for_model=user_id_for_model,  # 🔥 传递user_id_for_model
                        user_embedding=user_embedding_for_path  # 🔥 传递该路径对应的用户embedding
                    )
                    
                    # 计算此次试验的路径准确率（用于选择最佳）
                    seq_acc, prefix_accs, lcs_ratio = calculate_path_accuracy(gt_path, predicted_path)
                    
                    trial_results.append({
                        'predicted_path': predicted_path,
                        'success': success,
                        'steps': len(predicted_path),
                        'seq_acc': seq_acc,
                        'lcs_ratio': lcs_ratio,
                        'prefix_k': prefix_accs[4] if len(prefix_accs) >= 5 else (prefix_accs[-1] if prefix_accs else 0.0)
                    })
                
                # B. Reach@B: Top-T成功率（T次中至少成功一次）
                any_success = any(trial['success'] for trial in trial_results)
                success_count = sum(trial['success'] for trial in trial_results)
                
                rollout_results.append({
                    'success': any_success,  # 至少一次成功
                    'success_count': success_count,  # 成功次数
                    'success_rate': success_count / eval_num_trials,  # 成功率
                    'steps': trial_results[0]['steps'],  # 第一次试验的步数（用于兼容）
                    'path_idx': path_idx
                })
                
                # C. 路径准确率/覆盖度：选择Top-T中模仿最好的（lcs_ratio最高）
                best_trial = max(trial_results, key=lambda x: x['lcs_ratio'])
                predicted_path = best_trial['predicted_path']
                
                # C1. 序列级指标（使用最佳试验）
                path_accuracy_metrics['seq_acc'].append(best_trial['seq_acc'])
                path_accuracy_metrics['lcs_ratio'].append(best_trial['lcs_ratio'])
                path_accuracy_metrics['prefix_k'].append(best_trial['prefix_k'])
                
                # C2. 集合级指标
                # 将动作ID转换为边表示（简化：使用动作ID本身，因为edge_id_map可能不易反向查找）
                # 如果需要真正的边，需要从状态中提取，这里先用动作ID
                gt_edges = gt_path  # 简化：使用动作ID作为边标识
                pred_edges = predicted_path
                
                edge_metrics = calculate_edge_set_metrics(gt_edges, pred_edges)
                path_accuracy_metrics['iou'].append(edge_metrics['iou'])
                path_accuracy_metrics['precision'].append(edge_metrics['precision'])
                path_accuracy_metrics['recall'].append(edge_metrics['recall'])
                path_accuracy_metrics['f1'].append(edge_metrics['f1'])
                
                # 🔥 修复：如果best_trial成功到达，单独记录（使用best_trial的指标）
                if best_trial['success']:
                    path_accuracy_metrics['successful_paths'].append({
                        'seq_acc': best_trial['seq_acc'],
                        'lcs_ratio': best_trial['lcs_ratio'],
                        'prefix_k': best_trial['prefix_k'],
                        'iou': edge_metrics['iou'],
                        'precision': edge_metrics['precision'],
                        'recall': edge_metrics['recall'],
                        'f1': edge_metrics['f1']
                    })

                # 🔥 Save per-episode information if requested
                if save_per_episode_info and per_episode_info is not None:
                    # Check if any trial succeeded within budget_B steps
                    success_within_budget = any(trial['success'] and trial['steps'] <= budget_B for trial in trial_results)

                    # Get best LCS path (overall best)
                    best_lcs_path = [int(x) for x in predicted_path]
                    best_lcs_metrics = {
                        'seq_acc': float(best_trial['seq_acc']),
                        'lcs_ratio': float(best_trial['lcs_ratio']),
                        'prefix_k': float(best_trial['prefix_k']),
                        'steps': int(best_trial['steps']),
                        'success': bool(best_trial['success'])
                    }

                    # Get best LCS path among successful trials
                    successful_trials = [t for t in trial_results if t['success'] and t['steps'] <= budget_B]
                    best_success_lcs_path = None
                    best_success_lcs_metrics = None

                    if successful_trials:
                        best_success_trial = max(successful_trials, key=lambda x: x['lcs_ratio'])
                        best_success_lcs_path = [int(x) for x in best_success_trial['predicted_path']]
                        best_success_lcs_metrics = {
                            'seq_acc': float(best_success_trial['seq_acc']),
                            'lcs_ratio': float(best_success_trial['lcs_ratio']),
                            'prefix_k': float(best_success_trial['prefix_k']),
                            'steps': int(best_success_trial['steps']),
                            'success': True
                        }

                    # 获取稳定的route_id主键（来自数据集的route_id列）
                    route_id = eval_ids[path_idx] if eval_ids is not None and path_idx < len(eval_ids) else path_idx

                    # 整理所有trial的详细信息
                    trial_details = []
                    for trial_idx, trial in enumerate(trial_results):
                        trial_details.append({
                            'trial_idx': int(trial_idx),
                            'predicted_path': [int(x) for x in trial['predicted_path']],
                            'success': bool(trial['success']),
                            'steps': int(trial['steps']),
                            'seq_acc': float(trial['seq_acc']),
                            'lcs_ratio': float(trial['lcs_ratio']),
                            'prefix_k': float(trial['prefix_k'])
                        })

                    per_episode_info.append({
                        'episode_id': int(path_idx),
                        'route_id': str(route_id),  # 稳定的数据语义ID（来自数据集route_id列）
                        'success': bool(success_within_budget),
                        'success_count': int(success_count),  # 0-K successful rollouts
                        'gt_path': [int(x) for x in gt_path],  # Convert to list of ints
                        'best_lcs_path': best_lcs_path,
                        'best_lcs_metrics': best_lcs_metrics,
                        'best_success_lcs_path': best_success_lcs_path,
                        'best_success_lcs_metrics': best_success_lcs_metrics,
                        'trial_details': trial_details  # 所有5次trial的明细结果
                    })
            except Exception as e:
                print(f'  ⚠️  Error rolling out path {path_idx}: {e}')
            
            path_start_idx += path_length
        
        # B. 计算Reach@B (Top-T版本)
        successful_rollouts = sum(1 for r in rollout_results if r['success'])
        reach_at_budget = successful_rollouts / len(rollout_results) if rollout_results else 0.0
        
        # 计算Top-T成功率统计
        avg_success_rate = np.mean([r['success_rate'] for r in rollout_results]) if rollout_results else 0.0
        avg_success_count = np.mean([r['success_count'] for r in rollout_results]) if rollout_results else 0.0
        
        print(f'\n📊 B. Reach@Budget with Top-{eval_num_trials} Sampling (≤{budget_B} steps, threshold={arrival_threshold}):')
        print(f'   Reach@B (at least once): {reach_at_budget:.4f} ({successful_rollouts}/{len(rollout_results)} paths)')
        print(f'   Avg success count per path: {avg_success_count:.2f}/{eval_num_trials}')
        print(f'   Avg success rate per path: {avg_success_rate:.2%}')
        
        # C. 路径准确率/覆盖度汇总
        if path_accuracy_metrics and path_accuracy_metrics.get('seq_acc'):
            print(f'\n📊 C. Rollout Path Accuracy & Coverage:')
            
            # 全体样本统计
            all_metrics = ['seq_acc', 'lcs_ratio', 'prefix_k', 'iou', 'precision', 'recall', 'f1']
            print(f'   --- All Samples ({len(path_accuracy_metrics["seq_acc"])} paths) ---')
            for metric in all_metrics:
                values = path_accuracy_metrics[metric]
                if values:
                    print(f'   {metric.upper()}: mean={np.mean(values):.4f}, median={np.median(values):.4f}, p90={np.percentile(values, 90):.4f}')
            
            # 到达子集统计（成功路径中best-by-LCS的evaluation）
            if path_accuracy_metrics['successful_paths']:
                print(f'   --- Subset where (best-by-LCS trial is also successful): {len(path_accuracy_metrics["successful_paths"])} paths ---')
                for metric in ['seq_acc', 'lcs_ratio', 'prefix_k', 'iou', 'precision', 'recall', 'f1']:
                    values = [p[metric] for p in path_accuracy_metrics['successful_paths']]
                    if values:
                        print(f'   {metric.upper()}: mean={np.mean(values):.4f}, median={np.median(values):.4f}, p90={np.percentile(values, 90):.4f}')

            # 🔥 新增：Best Success LCS Trial统计（所有成功路径中LCS最高的trial）
            if per_episode_info:
                success_lcs_trials = [ep['best_success_lcs_metrics'] for ep in per_episode_info if ep.get('best_success_lcs_metrics')]
                if success_lcs_trials:
                    print(f'   --- Success-conditioned: best-by-GT-LCS among successful trials ({len(success_lcs_trials)} successful episodes) ---')
                    for metric in ['seq_acc', 'lcs_ratio', 'prefix_k']:
                        values = [t[metric] for t in success_lcs_trials if metric in t]
                        if values:
                            print(f'   {metric.upper()}: mean={np.mean(values):.4f}, median={np.median(values):.4f}, p90={np.percentile(values, 90):.4f}')
    
    # 原有的路径级指标（保留用于兼容性）
    print(f'\nTeacher-forcing Mean Step Accuracy (path-avg): {avg_path_correctness:.4f}')
    print(f'Teacher-forcing Exact-Match Rate (all steps correct): {path_success_rate:.4f} ({successful_paths}/{total_paths})')
    print(f'Average Path Length: {avg_path_length:.2f} steps')
    
    # 准备新评估指标的返回值（初始化，避免未定义错误）
    new_eval_metrics = {}
    if dataset == 'test' and 'rollout_results' in locals() and rollout_results and 'path_accuracy_metrics' in locals() and path_accuracy_metrics:
        new_eval_metrics['reach_at_budget'] = float(reach_at_budget) if 'reach_at_budget' in locals() else 0.0
        new_eval_metrics['rollout_budget'] = budget_B if 'budget_B' in locals() else 60
        new_eval_metrics['arrival_threshold'] = arrival_threshold if 'arrival_threshold' in locals() else 1e-6
        
        # 路径准确率/覆盖度指标
        if path_accuracy_metrics and path_accuracy_metrics.get('seq_acc'):
            new_eval_metrics['rollout_seq_acc_mean'] = float(np.mean(path_accuracy_metrics['seq_acc']))
            new_eval_metrics['rollout_seq_acc_median'] = float(np.median(path_accuracy_metrics['seq_acc']))
            new_eval_metrics['rollout_seq_acc_p90'] = float(np.percentile(path_accuracy_metrics['seq_acc'], 90))
            new_eval_metrics['rollout_lcs_ratio_mean'] = float(np.mean(path_accuracy_metrics['lcs_ratio']))
            new_eval_metrics['rollout_prefix_k_mean'] = float(np.mean(path_accuracy_metrics['prefix_k']))
            new_eval_metrics['rollout_iou_mean'] = float(np.mean(path_accuracy_metrics['iou']))
            new_eval_metrics['rollout_precision_mean'] = float(np.mean(path_accuracy_metrics['precision']))
            new_eval_metrics['rollout_recall_mean'] = float(np.mean(path_accuracy_metrics['recall']))
            new_eval_metrics['rollout_f1_mean'] = float(np.mean(path_accuracy_metrics['f1']))
            
            # 成功路径的指标
            if path_accuracy_metrics.get('successful_paths'):
                success_metrics = path_accuracy_metrics['successful_paths']
                new_eval_metrics['rollout_success_seq_acc_mean'] = float(np.mean([m['seq_acc'] for m in success_metrics]))
                new_eval_metrics['rollout_success_f1_mean'] = float(np.mean([m['f1'] for m in success_metrics]))
                new_eval_metrics['rollout_success_iou_mean'] = float(np.mean([m['iou'] for m in success_metrics]))
    
    # 🔥 Phase-1: 对于test集，仍然执行完整评估（包括B和C），只是跳过RL专用指标
    if training_phase == 1 and dataset != 'test':
        print(f'\n🎯 Phase-1: SL-only evaluation (skipping RL metrics)')
        return {
            'step_level_accuracy': avg_step_correct,  # A. 单步准确率
            'path_level_accuracy': avg_path_correctness,
            'path_success_rate': path_success_rate,
            'avg_path_length': avg_path_length,
            'total_steps': total_steps,
            'total_paths': total_paths,
            **new_eval_metrics  # 包含新评估指标（如果是test集）
        }
    
    # 🔥 Phase-1 + test集：显示A、B、C指标，但跳过RL专用指标
    if training_phase == 1 and dataset == 'test':
        print(f'\n🎯 Phase-1: SL-only evaluation on TEST set (showing A/B/C metrics, skipping RL-specific metrics)')
        return {
            'step_level_accuracy': avg_step_correct,  # A. 单步准确率
            'path_level_accuracy': avg_path_correctness,
            'path_success_rate': path_success_rate,
            'avg_path_length': avg_path_length,
            'total_steps': total_steps,
            'total_paths': total_paths,
            **new_eval_metrics  # B. Reach@B 和 C. 路径准确率/覆盖度
        }
    



    # 准备新评估指标的返回值（如果之前未定义，在这里初始化）
    if 'new_eval_metrics' not in locals():
        new_eval_metrics = {}
    if dataset == 'test' and rollout_results and path_accuracy_metrics:
        new_eval_metrics['reach_at_budget'] = float(reach_at_budget)
        new_eval_metrics['rollout_budget'] = budget_B
        new_eval_metrics['arrival_threshold'] = arrival_threshold
        
        # 路径准确率/覆盖度指标
        if path_accuracy_metrics['seq_acc']:
            new_eval_metrics['rollout_seq_acc_mean'] = float(np.mean(path_accuracy_metrics['seq_acc']))
            new_eval_metrics['rollout_seq_acc_median'] = float(np.median(path_accuracy_metrics['seq_acc']))
            new_eval_metrics['rollout_seq_acc_p90'] = float(np.percentile(path_accuracy_metrics['seq_acc'], 90))
            new_eval_metrics['rollout_lcs_ratio_mean'] = float(np.mean(path_accuracy_metrics['lcs_ratio']))
            new_eval_metrics['rollout_prefix_k_mean'] = float(np.mean(path_accuracy_metrics['prefix_k']))
            new_eval_metrics['rollout_iou_mean'] = float(np.mean(path_accuracy_metrics['iou']))
            new_eval_metrics['rollout_precision_mean'] = float(np.mean(path_accuracy_metrics['precision']))
            new_eval_metrics['rollout_recall_mean'] = float(np.mean(path_accuracy_metrics['recall']))
            new_eval_metrics['rollout_f1_mean'] = float(np.mean(path_accuracy_metrics['f1']))
            
            # 成功路径的指标
            if path_accuracy_metrics['successful_paths']:
                success_metrics = path_accuracy_metrics['successful_paths']
                new_eval_metrics['rollout_success_seq_acc_mean'] = float(np.mean([m['seq_acc'] for m in success_metrics]))
                new_eval_metrics['rollout_success_f1_mean'] = float(np.mean([m['f1'] for m in success_metrics]))
                new_eval_metrics['rollout_success_iou_mean'] = float(np.mean([m['iou'] for m in success_metrics]))

    eval_metrics = {
        'dataset': dataset,
        'total_steps': int(total_steps),
        'total_correct_steps': int(total_correct_steps),
        'total_paths': int(total_paths),
        'step_level_accuracy': float(avg_step_correct),  # A. 单步准确率
        'path_level_accuracy': float(avg_path_correctness),
        'path_success_rate': float(path_success_rate),
        'rl_step_level_accuracy': float(avg_step_correct_rl),
        'rl_path_level_accuracy': float(avg_path_correctness_rl),
        'rl_path_success_rate': float(path_success_rate_rl),
        'avg_path_length': float(avg_path_length),
        'total_cumulative_reward': float(total_reward),
        # 🔥 新增：用户分类准确率
        # 新增：有效动作Q-values统计
        'valid_q_mean': valid_q_stats['mean'],
        'valid_q_std': valid_q_stats['std'],
        'valid_q_min': valid_q_stats['min'],
        'valid_q_max': valid_q_stats['max'],
        'valid_q_range': valid_q_stats['range'],
        'valid_q_count': valid_q_stats['count'],
        'avg_valid_actions_per_step': valid_q_stats['avg_valid_actions_per_step'],
        # 🔥 新评估指标：B. Reach@B 和 C. 路径准确率/覆盖度
        **new_eval_metrics,
        # # 🔥 新增：RL rollout奖励统计
        # 'rl_reach_at_budget': float(reach_at_budget),
        # 'rl_loop_stuck_rate': float(loop_stuck_rate),
        # 'rl_invalid_action_rate': float(invalid_action_rate),
        # 'rl_comprehensive_score': float(rl_score),
        # 'rl_rollout_count': len(rl_rollout_results)
    }
    
    # 添加奖励统计到eval_metrics
    # if overall_reward_stats:
    #     eval_metrics.update({
    #         'reward_total_mean': overall_reward_stats['total_rewards_mean'],
    #         'reward_total_std': overall_reward_stats['total_rewards_std'],
    #         'reward_total_min': overall_reward_stats['total_rewards_min'],
    #         'reward_total_max': overall_reward_stats['total_rewards_max'],
    #         'reward_positive_count': overall_reward_stats['positive_total_rewards'],
    #         'reward_negative_count': overall_reward_stats['negative_total_rewards'],
    #         'reward_zero_count': overall_reward_stats['zero_total_rewards']
    #     })
    
    # 🔥 Save per-episode information to file if requested
    if save_per_episode_info and per_episode_info is not None and len(per_episode_info) > 0:
        import json
        import os
        from datetime import datetime

        # Create output directory if it doesn't exist
        per_episode_dir = os.path.join(data_directory, 'per_episode_results')
        os.makedirs(per_episode_dir, exist_ok=True)

        # Generate filename with timestamp and parameters
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_suffix = f'_model{model_config_id}' if model_config_id is not None else ''
        user_suffix = f'_user{user_id}' if user_id is not None else ''
        filename = f'per_episode_{dataset}_{timestamp}{model_suffix}{user_suffix}.json'
        filepath = os.path.join(per_episode_dir, filename)

        # Save to JSON file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'dataset': dataset,
                        'eval_num_trials': eval_num_trials,
                        'budget_steps': budget_B,
                        'arrival_threshold': arrival_threshold,
                        'eval_temperature': eval_temperature,
                        'total_episodes': len(per_episode_info),
                        'successful_episodes': sum(1 for ep in per_episode_info if ep['success']),
                        'timestamp': timestamp,
                        'path_selection': {
                            'best_lcs_path': 'Path with highest LCS ratio among all trials',
                            'best_success_lcs_path': 'Path with highest LCS ratio among successful trials (<= budget_steps)'
                        }
                    },
                    'episodes': per_episode_info
                }, f, indent=2, ensure_ascii=False)

            print(f'💾 Per-episode information saved to: {filepath}')
            print(f'   Total episodes: {len(per_episode_info)}')
            successful_episodes = sum(1 for ep in per_episode_info if ep['success'])
            episodes_with_success_path = sum(1 for ep in per_episode_info if ep['best_success_lcs_path'] is not None)
            print(f'   Successful episodes (any trial): {successful_episodes}')
            print(f'   Episodes with success path data: {episodes_with_success_path}')
        except Exception as e:
            print(f'❌ Failed to save per-episode information: {e}')

    if logger is not None and step is not None:
        logger.log_evaluation(step, eval_metrics)

    return eval_metrics

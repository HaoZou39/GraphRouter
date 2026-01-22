
# 添加缓存来避免重复查找
_route_task_key_cache = {}

def get_task_key_from_route_id(route_id, data_directory=None):
    """从route_id获取对应的task_key（基于坐标比较）"""
    # 检查缓存（在标准化路径后）
    if data_directory is None:
        data_directory = os.path.join(os.path.dirname(__file__), '..', 'data')
        if not os.path.exists(data_directory):
            data_directory = '../data'
    
    normalized_data_dir = os.path.abspath(data_directory)
    cache_key = f"{route_id}_{normalized_data_dir}"
    if cache_key in _route_task_key_cache:
        # print(f"   📋 Using cached task_key for route {route_id}")
        return _route_task_key_cache[cache_key]
    
    try:
        # 🔥 获取route_id对应的起点终点坐标
        
        # 查找coordinate_stats.json文件（可能在多个位置）
        coord_stats_paths = [
            os.path.join(data_directory, 'coordinate_stats.json'),
            os.path.join(data_directory, 'graph_data', 'coordinate_stats.json'),
            os.path.join(data_directory, '..', 'data', 'graph_data', 'coordinate_stats.json')
        ]
        
        coord_stats_file = None
        for path in coord_stats_paths:
            if os.path.exists(path):
                coord_stats_file = path
                break
        
        if coord_stats_file and os.path.exists(coord_stats_file):
            print(f"   📁 Found coordinate_stats.json at: {coord_stats_file}")
            with open(coord_stats_file, 'r') as f:
                coord_stats = json.load(f)
            
            def normalize_coordinate(coord, coord_stats):
                # 处理不同的JSON格式
                if 'mean' in coord_stats and 'std' in coord_stats:
                    return (coord - coord_stats['mean']) / coord_stats['std']
                elif 'x_mean' in coord_stats and 'x_std' in coord_stats:
                    # 分别处理x和y坐标
                    x_norm = (coord[0] - coord_stats['x_mean']) / coord_stats['x_std']
                    y_norm = (coord[1] - coord_stats['y_mean']) / coord_stats['y_std']
                    return np.array([x_norm, y_norm])
                else:
                    print(f"   ⚠️  Unknown coordinate_stats format: {coord_stats.keys()}")
                    return coord
            
            # 查找对应的CSV文件（检查多个可能的目录）
            possible_paths = [
                os.path.join(data_directory, f'route_demo_walk_{route_id}_start_end.csv'),
                os.path.join(data_directory, 'train', f'route_demo_walk_{route_id}_start_end.csv'),
                os.path.join(data_directory, 'val', f'route_demo_walk_{route_id}_start_end.csv'),
                os.path.join(data_directory, 'test', f'route_demo_walk_{route_id}_start_end.csv')
            ]
            
            csv_file = None
            for path in possible_paths:
                print(f"   🔍 Checking: {path}")
                if os.path.exists(path):
                    csv_file = path
                    print(f"   ✅ Found CSV file: {path}")
                    break
            
            if csv_file:
                import pandas as pd
                # 尝试不同的分隔符
                try:
                    df = pd.read_csv(csv_file, sep=';')
                except:
                    df = pd.read_csv(csv_file)
                if len(df) > 0:
                    print(f"   📊 CSV columns: {df.columns.tolist()}")
                    print(f"   📊 CSV shape: {df.shape}")
                    # 解析WKT几何数据获取起点终点
                    from shapely import wkt
                    
                    # 检查不同的CSV格式
                    if 'coordinates' in df.columns:
                        # 新格式：有coordinates列，需要解析origin_node和destination_node
                        print(f"   🔍 Parsing coordinates format...")
                        origin_row = df[df['coordinates'] == 'origin_node']
                        dest_row = df[df['coordinates'] == 'destination_node']
                        
                        print(f"   📍 Origin rows: {len(origin_row)}, Dest rows: {len(dest_row)}")
                        
                        if len(origin_row) > 0 and len(dest_row) > 0:
                            print(f"   📍 Origin geometry: {origin_row.iloc[0]['geometry']}")
                            print(f"   📍 Dest geometry: {dest_row.iloc[0]['geometry']}")
                            
                            origin_geom = wkt.loads(origin_row.iloc[0]['geometry'])
                            dest_geom = wkt.loads(dest_row.iloc[0]['geometry'])
                            origin = origin_geom.coords[0]
                            destination = dest_geom.coords[0]
                            
                            print(f"   ✅ Parsed origin: {origin}")
                            print(f"   ✅ Parsed destination: {destination}")
                        else:
                            print(f"   ⚠️  Could not find origin_node/destination_node rows")
                            print(f"   📊 Available coordinates values: {df['coordinates'].unique()}")
                            return None
                    elif 'geometry' in df.columns:
                        # 标准格式：有geometry列
                        geometry = wkt.loads(df.iloc[0]['geometry'])
                        coords = list(geometry.coords)
                        if len(coords) >= 2:
                            origin = coords[0]
                            destination = coords[-1]
                        else:
                            print(f"   ⚠️  Insufficient coordinates in geometry: {len(coords)}")
                            return None
                    else:
                        print(f"   ⚠️  Unknown CSV format. Columns: {df.columns.tolist()}")
                        return None
                    
                    # 归一化坐标
                    csv_origin = normalize_coordinate(np.array(origin), coord_stats)
                    csv_destination = normalize_coordinate(np.array(destination), coord_stats)
                    
                    # 🔥 生成坐标字符串作为task_key（与extract_task_key_from_expert_transition保持一致）
                    coord_str = f"{csv_origin[0]}_{csv_origin[1]}_{csv_destination[0]}_{csv_destination[1]}"
                    
                    print(f"   🔍 Route {route_id} -> Task_key {coord_str}")
                    print(f"   📍 Coordinates: start=({csv_origin[0]:.3f}, {csv_origin[1]:.3f}), end=({csv_destination[0]:.3f}, {csv_destination[1]:.3f})")
                    
                    # 缓存结果
                    _route_task_key_cache[cache_key] = coord_str
                    return coord_str
        
        print(f"   ⚠️  Could not find coordinates for route {route_id}")
        # 缓存None结果以避免重复查找
        _route_task_key_cache[cache_key] = None
        return None
        
    except Exception as e:
        print(f"   ❌ Error getting task_key from route_id {route_id}: {e}")
        return None
def filter_expert_data_by_task(expert_data, task_key, phase2_mode=False):
    """Filter expert data by task_key with caching for performance."""
    if not phase2_mode:
        return expert_data
    
    # 🔥 性能优化：使用缓存避免重复过滤
    cache_key = f"expert_filter_cache_{task_key}"
    
    # 检查是否已有缓存
    if not hasattr(filter_expert_data_by_task, 'cache'):
        filter_expert_data_by_task.cache = {}
    
    if cache_key in filter_expert_data_by_task.cache:
        cached_indices = filter_expert_data_by_task.cache[cache_key]
        return expert_data.iloc[cached_indices] if cached_indices else expert_data.iloc[0:0]
    
    # 🔥 性能优化：预计算所有task_key映射（一次性计算，多次使用）
    global_cache_key = "expert_task_key_mapping"
    if global_cache_key not in filter_expert_data_by_task.cache:
        print(f"   🔍 Pre-computing task_key mapping for all expert data (one-time cost)...")
        task_keys_series = expert_data.apply(lambda row: extract_task_key_from_expert_transition(row), axis=1)
        filter_expert_data_by_task.cache[global_cache_key] = task_keys_series
        
        # 🔥 调试信息：显示task_key分布
        valid_task_keys = task_keys_series.dropna()
        unique_task_keys = valid_task_keys.unique()
        print(f"   ✅ Task_key mapping computed for {len(expert_data)} expert samples")
        print(f"   📊 Found {len(unique_task_keys)} unique task_keys")
        # print(f"   📊 Top 10 task_keys: {sorted(unique_task_keys, key=str)[:10]}")
        # print(f"   📊 Task_key distribution: {valid_task_keys.value_counts().head(10).to_dict()}")
    else:
        task_keys_series = filter_expert_data_by_task.cache[global_cache_key]
    
    # 🔥 性能优化：使用pandas的布尔索引进行快速过滤
    print(f"   🔍 Filtering expert data for task {task_key} (using cached mapping)...")
    # 过滤掉None值，只保留有效的task_key
    valid_mask = task_keys_series.notna()
    task_mask = task_keys_series == task_key
    mask = valid_mask & task_mask
    filtered_indices = expert_data[mask].index.tolist()
    
    # 统计信息
    total_valid = valid_mask.sum()
    total_none = (~valid_mask).sum()
    if total_none > 0:
        print(f"   ⚠️  {total_none} transitions could not be classified (excluded from filtering)")
    
    # 缓存结果
    filter_expert_data_by_task.cache[cache_key] = filtered_indices
    
    # print(f"   ✅ Found {len(filtered_indices)} expert samples for task {task_key}")
    
    return expert_data.iloc[filtered_indices] if filtered_indices else expert_data.iloc[0:0]

def extract_task_key_from_expert_transition(transition):
    """从expert transition中提取route_id，基于起点终点坐标"""
    # 🔥 调试：打印transition的基本信息
    # print(f"   🔍 extract_task_key_from_expert_transition debug:")
    # print(f"   📊 transition keys: {list(transition.keys())}")
    
    # 方法1: 直接从transition中获取route_id（但转换为坐标字符串格式）
    if 'route_id' in transition:
        route_id = transition['route_id']

    
    # 方法2: 从state中提取起点终点坐标，生成唯一的task_key
    state = transition['state']
    # print(f"   📊 state type: {type(state)}")
    # print(f"   📊 state shape: {np.array(state).shape if hasattr(state, '__len__') else 'scalar'}")
    
    if isinstance(state, np.ndarray) and state.ndim == 2:
        first_frame = state[0]
        # print(f"   📊 Using first frame from 2D array")
    elif isinstance(state, list) and len(state) > 0:
        # 🔥 修复：处理state是list的情况
        if isinstance(state[0], list) and len(state[0]) > 0:
            # state是[[frame1], [frame2], ...]的格式
            first_frame = state[0]  # 取第一个frame
            # print(f"   📊 Using first frame from list of lists")
        else:
            # state是[frame1, frame2, ...]的格式
            first_frame = state
            # print(f"   📊 Using state directly as frame")
    else:
        first_frame = state
        # print(f"   📊 Using state directly")
    
    # print(f"   📊 first_frame type: {type(first_frame)}")
    # print(f"   📊 first_frame length: {len(first_frame) if hasattr(first_frame, '__len__') else 'scalar'}")
    
    # 🔥 根据起点终点坐标生成task_key
    if len(first_frame) >= 15:  # 确保有足够的坐标信息
        try:
            # 🔥 修复：使用固定索引，而不是负索引
            # 状态格式（前15维固定）：[start_x, start_y, end_x, end_y, crossing, path_type, length, width, curb, origin_x, origin_y, dest_x, dest_y, cur_x, cur_y]
            # 索引：              0        1        2      3      4         5          6       7      8     9         10         11     12      13     14
            # origin_x, origin_y 在索引 9, 10
            # dest_x, dest_y 在索引 11, 12
            start_x, start_y = float(first_frame[9]), float(first_frame[10])
            end_x, end_y = float(first_frame[11]), float(first_frame[12])
            
            # print(f"   📊 Extracted coordinates: start=({start_x:.3f}, {start_y:.3f}), end=({end_x:.3f}, {end_y:.3f})")
            
            coord_str = f"{start_x}_{start_y}_{end_x}_{end_y}"
            task_key = coord_str
            
            # print(f"   ✅ Generated task_key from coordinates: {task_key}")
            return task_key
        except (ValueError, TypeError, IndexError) as e:
            print(f"   ❌ Error extracting coordinates: {e}")
            print(f"   📊 first_frame[-6:]: {first_frame[-6:] if len(first_frame) >= 6 else 'insufficient length'}")
            pass
    else:
        print(f"   ❌ Insufficient frame length: {len(first_frame)} < 6")
    
    # 方法3: 尝试从state的第一个元素获取route_id（仅用于调试）
    if len(first_frame) > 0:
        try:
            route_id = int(first_frame[0])
            # if route_id >= 0:
                # print(f"   📋 Found route_id in first element: {route_id} (for debugging)")
        except (ValueError, TypeError) as e:
            print(f"   ❌ Error extracting route_id from first element: {e}")
            print(f"   📊 first_frame[0]: {first_frame[0]}")
            pass
    
    # 方法4: 从其他字段推断
    for key in ['task_id', 'path_id', 'trajectory_id']:
        if key in transition:
            try:
                route_id = int(transition[key])
                if route_id >= 0:
                    # print(f"   ✅ Found route_id in {key}: {route_id}")
                    return route_id
            except (ValueError, TypeError) as e:
                print(f"   ❌ Error extracting route_id from {key}: {e}")
                continue
    
    # 🔥 如果无法提取，返回None而不是默认值0
    # 这样可以在过滤时排除这些无法分类的transition
    print(f"   ❌ Could not extract task_key from transition")
    return None
def count_unique_task_keys_in_batch(batch, batch_name="batch"):
    """统计batch中不同task_key的数量"""
    try:
        states = batch.get('state', [])
        if isinstance(states, dict):
            states = list(states.values())
        elif not isinstance(states, list):
            states = [states]
        
        task_keys = set()
        failed_count = 0
        
        for state in states:
            # 从state中提取task_key
            if isinstance(state, np.ndarray):
                if state.ndim == 2:
                    first_frame = state[0]
                else:
                    first_frame = state
            elif isinstance(state, list):
                if len(state) > 0 and isinstance(state[0], list):
                    first_frame = state[0]
                else:
                    first_frame = state
            else:
                first_frame = state
            
            # 尝试从坐标提取task_key
            if hasattr(first_frame, '__len__') and len(first_frame) >= 15:
                try:
                    # 🔥 修复：使用固定索引，而不是负索引
                    # origin_x, origin_y 在索引 9, 10
                    # dest_x, dest_y 在索引 11, 12
                    start_x, start_y = float(first_frame[9]), float(first_frame[10])
                    end_x, end_y = float(first_frame[11]), float(first_frame[12])
                    task_key = f"{start_x}_{start_y}_{end_x}_{end_y}"
                    task_keys.add(task_key)
                except (ValueError, TypeError, IndexError):
                    failed_count += 1
            else:
                failed_count += 1
        
        return len(task_keys), failed_count
    except Exception as e:
        return -1, 0  # 返回-1表示统计失败
def calculate_loop_stuck_rate_from_rollouts(rl_rollout_results):
    """计算循环/卡住率"""
    if not rl_rollout_results:
        return 0.0
    
    loop_stuck_count = sum(1 for result in rl_rollout_results if result['failure_reason'] == 'loop_stuck')
    return loop_stuck_count / len(rl_rollout_results)

def calculate_rl_comprehensive_score(reach_at_budget, loop_stuck_rate, invalid_action_rate):
    """
    🔥 改进的RL综合评分 - 专注可达性，移除效率项
    
    Args:
        reach_at_budget: 在预算内到达的比例
        loop_stuck_rate: 循环/卡住率
        invalid_action_rate: 无效动作率
    
    Returns:
        rl_score: 综合评分 [0, 1]
    """
    # 🔥 改进：调整权重，专注可达性
    w_reach = 0.7      # 到达率权重（主要目标）
    w_quality = 0.3    # 质量权重（基于循环和无效动作）
    
    # 到达分数
    reach_score = reach_at_budget
    
    # 质量分数（循环和无效动作越少越好）
    quality_score = max(0.0, 1.0 - loop_stuck_rate - invalid_action_rate)
    
    # 综合评分（移除效率项）
    rl_score = (w_reach * reach_score + 
                w_quality * quality_score)
    
    return min(1.0, max(0.0, rl_score))
def _calculate_reward_stats(rewards):
    """计算奖励统计信息"""
    if not rewards:
        return {
            'total_reward': 0.0,
            'avg_reward': 0.0,
            'max_reward': 0.0,
            'min_reward': 0.0,
            'positive_rewards': 0,
            'negative_rewards': 0,
            'zero_rewards': 0
        }
    
    return {
        'total_reward': sum(rewards),
        'avg_reward': sum(rewards) / len(rewards),
        'max_reward': max(rewards),
        'min_reward': min(rewards),
        'positive_rewards': sum(1 for r in rewards if r > 0),
        'negative_rewards': sum(1 for r in rewards if r < 0),
        'zero_rewards': sum(1 for r in rewards if r == 0)
    }
def create_stay_in_place_feature(
    cur_node, origin_x, origin_y, dest_x, dest_y,
    text_embedding_loader=None, embedding_proj_weights=None, embedding_proj_biases=None
):
    """
    创建原地踏步的特征向量（用于无效动作）
    
    Args:
        cur_node: 当前节点坐标
        origin_x, origin_y: 起始位置
        dest_x, dest_y: 目标位置
        text_embedding_loader: TextEmbeddingLoader 实例（可选）
        embedding_proj_weights: 投影权重字典（可选）
        embedding_proj_biases: 投影偏置字典（可选）
    """
    cur_node_x, cur_node_y = cur_node
    
    # 创建原地踏步的特征向量
    static_feature = [0.0, 0.0, 0.0, 0, 0]  # [length, width, curb, crossing, path_type]
    target_edge = ((cur_node_x, cur_node_y), (cur_node_x, cur_node_y))  # 起点和终点都是当前位置
    
    # 使用 create_action_feature 来确保格式一致（包括文本 embeddings）
    # 推断哪些 embeddings 被启用
    use_edge = 'edge' in (embedding_proj_weights or {})
    use_node = 'node' in (embedding_proj_weights or {})
    use_goal = 'goal' in (embedding_proj_weights or {})
    # 🔥 修复：从任意存在的embedding类型获取proj_dim
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
    
    feature_vec = create_action_feature(
        cur_node,  # new_pos
        origin_x, origin_y,  # origin
        dest_x, dest_y,  # dest
        static_feature,
        target_edge,
        text_embedding_loader=text_embedding_loader,
        edge_id=None,  # 原地踏步没有 edge_id
        embedding_proj_weights=embedding_proj_weights,
        embedding_proj_biases=embedding_proj_biases,
        use_edge_embedding=use_edge,
        use_node_embedding=use_node,
        use_goal_embedding=use_goal,
        embedding_proj_dim=proj_dim
    )
    
    return feature_vec

def enhance_states_full(states, G, edge_id_map, 
                       text_embedding_loader=None, embedding_proj_weights=None, embedding_proj_biases=None,
                       user_embedding=None, original_feature_dim=15):
    """
    Full state enhancement with both text embeddings and user embeddings.
    
    Args:
        states: State array [batch_size, state_size, original_feature_dim]
        G: Graph object
        edge_id_map: Edge ID mapping
        text_embedding_loader: TextEmbeddingLoader instance (optional)
        embedding_proj_weights: Projection weights dict (optional)
        embedding_proj_biases: Projection biases dict (optional)
        user_embedding: User statistics embedding [user_dim] or None
        original_feature_dim: Original feature dimension (default 15)
        
    Returns:
        enhanced_states: Enhanced states [batch_size, state_size, enhanced_feature_dim]
    """
    # Step 1: Add text embeddings
    states = enhance_states_with_text_embeddings(
        states, G, edge_id_map,
        text_embedding_loader=text_embedding_loader,
        embedding_proj_weights=embedding_proj_weights,
        embedding_proj_biases=embedding_proj_biases,
        original_feature_dim=original_feature_dim
    )
    
    # Step 2: Add user embedding
    states = add_user_embedding_to_states(states, user_embedding)
    
    return states

def _normalize_to_int(value, default=0):
    """Normalize a value to an integer, handling tuples, lists, and other types."""
    if isinstance(value, (tuple, list)):
        value = value[0] if len(value) > 0 else default
    try:
        return int(float(value)) if value is not None else default
    except (ValueError, TypeError):
        return default


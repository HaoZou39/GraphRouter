"""
Path Planning functions for the Graph Routing project.

This module contains path planning functions for the Graph Routing project.
"""

def create_action_feature(
    new_pos, origin_x, origin_y, dest_x, dest_y, static_feature, target_edge,
    text_embedding_loader=None, edge_id=None, embedding_proj_weights=None, embedding_proj_biases=None,
    use_edge_embedding=True, use_node_embedding=True, use_goal_embedding=True, embedding_proj_dim=64
):
    """
    创建动作特征向量，格式与原始数据一致
    
    Args:
        new_pos: 新位置 (cur_x, cur_y)
        origin_x, origin_y: 起始位置
        dest_x, dest_y: 目标位置
        static_feature: 边信息，包含静态特征
        target_edge: 目标边 ((start_x, start_y), (end_x, end_y))
        text_embedding_loader: TextEmbeddingLoader 实例（可选）
        edge_id: 边ID（用于获取 edge embedding）
        embedding_proj_weights: 投影权重字典（可选）
        embedding_proj_biases: 投影偏置字典（可选）
        use_edge_embedding: 是否使用 edge embedding（默认 True）
        use_node_embedding: 是否使用 node embedding（默认 True）
        use_goal_embedding: 是否使用 goal embedding（默认 True）
        embedding_proj_dim: 投影维度（默认 64）
        
    Returns:
        action_feature: 动作特征向量 [15维 或 15+投影维度]
    """
    # 创建15维的特征向量
    # Get edge endpoint coordinates
    start_x, start_y = target_edge[0]
    end_x, end_y = target_edge[1]    
    length = static_feature[0]
    obstacle_free_width = static_feature[1]
    curb_height = static_feature[2]
    crossing = static_feature[3]
    path_type = static_feature[4]
    cur_node_x, cur_node_y = new_pos

    # Build feature vector, consistent with training data format
    feature_vec = [
        start_x, start_y,  # Line segment start coordinates
        end_x, end_y,      # Line segment end coordinates
        crossing,          # Categorical feature 1
        path_type,         # Categorical feature 2
        length,            # Numerical feature 1
        obstacle_free_width,  # Numerical feature 2
        curb_height,       # Numerical feature 3
        origin_x, origin_y,  # Entire path start coordinates
        dest_x, dest_y,      # Entire path end coordinates
        cur_node_x, cur_node_y  # Current node coordinates (position after this step)
    ]
    
    # 🔥 如果提供了文本 embedding loader，增强特征
    # 重要：无论数据是否缺失，都要添加所有启用的 embeddings 维度，保持一致性
    if text_embedding_loader is not None and embedding_proj_weights is not None:
        # 获取 edge embedding
        if use_edge_embedding and 'edge' in embedding_proj_weights:
            if edge_id is not None:
                edge_emb = text_embedding_loader.get_edge_embedding(edge_id, fallback_zero=True)
                if edge_emb is not None:
                    # 投影 edge embedding
                    edge_proj = np.dot(edge_emb, embedding_proj_weights['edge'])
                    if embedding_proj_biases and 'edge' in embedding_proj_biases:
                        edge_proj += embedding_proj_biases['edge']
                    edge_proj = np.tanh(edge_proj)  # 激活函数
                    feature_vec.extend(edge_proj.tolist())
                else:
                    # 如果获取失败，使用零向量
                    feature_vec.extend([0.0] * embedding_proj_dim)
            else:
                # 如果 edge_id 为 None，使用零向量
                feature_vec.extend([0.0] * embedding_proj_dim)
        
        # 获取 node embedding
        if use_node_embedding and 'node' in embedding_proj_weights:
            coord_key = f"{cur_node_x:.16f}_{cur_node_y:.16f}"
            node_emb = text_embedding_loader.get_node_embedding(coord_key, fallback_zero=True)
            if node_emb is not None:
                node_proj = np.dot(node_emb, embedding_proj_weights['node'])
                if embedding_proj_biases and 'node' in embedding_proj_biases:
                    node_proj += embedding_proj_biases['node']
                node_proj = np.tanh(node_proj)
                feature_vec.extend(node_proj.tolist())
            else:
                # 如果获取失败，使用零向量
                feature_vec.extend([0.0] * embedding_proj_dim)
        
        # 获取 goal embedding
        if use_goal_embedding and 'goal' in embedding_proj_weights:
            if discretize_goal_context is not None:
                goal_key, _, _ = discretize_goal_context(new_pos, (dest_x, dest_y))
                goal_emb = text_embedding_loader.get_goal_embedding(goal_key, fallback_zero=True)
                if goal_emb is not None:
                    goal_proj = np.dot(goal_emb, embedding_proj_weights['goal'])
                    if embedding_proj_biases and 'goal' in embedding_proj_biases:
                        goal_proj += embedding_proj_biases['goal']
                    goal_proj = np.tanh(goal_proj)
                    feature_vec.extend(goal_proj.tolist())
                else:
                    # 如果获取失败，使用零向量
                    feature_vec.extend([0.0] * embedding_proj_dim)
            else:
                # 如果 discretize_goal_context 不可用，使用零向量
                feature_vec.extend([0.0] * embedding_proj_dim)
    
    return feature_vec

def compute_state_transition(
    current_state, action_id, edge_id_map, G,
    text_embedding_loader=None, embedding_proj_weights=None, embedding_proj_biases=None,
    user_embedding=None
):
    """
    计算执行动作后的状态转移
    
    ⚠️ DEPRECATED: 此函数仅用于旧格式数据（15维坐标）
    新格式数据（9维历史序列）应使用rollout_with_candidates等新函数
    
    Args:
        current_state: 当前状态向量 [batch_size, state_size, feature_dim]
        action_id: 动作ID [batch_size]
        edge_id_map: 边ID映射
        G: 图对象
        text_embedding_loader: TextEmbeddingLoader 实例（可选）
        embedding_proj_weights: 投影权重字典（可选）
        embedding_proj_biases: 投影偏置字典（可选）
        user_embedding: 用户embedding向量 [user_embedding_dim]（可选）
        
    Returns:
        next_state: 转移后的状态
    """
    next_states = []
    
    state = current_state[0]  # [state_size, feature_dim]
    action = action_id[0]
    
    # Find the last non-zero state
    last_non_zero_idx = -1
    for i in range(len(state)-1, -1, -1):
        if not np.all(state[i] == 0):
            last_non_zero_idx = i
            break

    if last_non_zero_idx == -1:
        raise ValueError(f"Invalid state transition: last_non_zero_idx is not found")
    
    last_state = state[last_non_zero_idx]
    
    # 🔥 新格式检测：9维 = 历史序列，无坐标
    if len(last_state) == 9:
        raise ValueError(
            "compute_state_transition() does not support new format (9-dim history). "
            "Use rollout_with_candidates() for new format data."
        )
    
    # ⚠️ 旧格式（DEPRECATED）：15维包含坐标
    if len(last_state) < 15:
        raise ValueError(f"Invalid state format: expected 15+ dimensions, got {len(last_state)}")
    
    cur_node = (float(last_state[13]), float(last_state[14]))  # 索引 13, 14: cur_x, cur_y
    origin_x, origin_y = float(last_state[9]), float(last_state[10])  # 索引 9, 10: origin_x, origin_y
    dest_x, dest_y = float(last_state[11]), float(last_state[12])  # 索引 11, 12: dest_x, dest_y
    
    # 尝试根据动作ID找到对应的边
    try:
        new_pos, static_feature, target_edge, edge_id = find_new_position_after_action(action, cur_node, edge_id_map, G)
        
        # 创建新的动作特征向量（🔥 支持文本 embeddings）
        # 注意：这里需要传递 args 来确定哪些 embeddings 被启用
        # 但由于 args 不在作用域内，我们需要从 embedding_proj_weights 推断
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
        
        action_feature = create_action_feature(
            new_pos, origin_x, origin_y, dest_x, dest_y, static_feature, target_edge,
            text_embedding_loader=text_embedding_loader,
            edge_id=edge_id,
            embedding_proj_weights=embedding_proj_weights,
            embedding_proj_biases=embedding_proj_biases,
            use_edge_embedding=use_edge,
            use_node_embedding=use_node,
            use_goal_embedding=use_goal,
            embedding_proj_dim=proj_dim
        )
        
        # 🔥 添加user embedding（如果提供）
        if user_embedding is not None:
            action_feature = np.concatenate([action_feature, user_embedding])
        
        # 🔥 修复：根据action_feature的维度来初始化next_state
        # 如果action_feature是增强的（143维），next_state也需要是增强的
        action_feature_dim = len(action_feature) if isinstance(action_feature, (list, np.ndarray)) else 15
        state_feature_dim = state.shape[1] if state.ndim == 2 else len(state[0]) if len(state) > 0 else 15
        
        # 如果action_feature维度与state不同，需要调整next_state的维度
        if action_feature_dim != state_feature_dim:
            # 需要创建新的next_state，维度匹配action_feature
            state_size = state.shape[0] if state.ndim == 2 else len(state)
            next_state = np.zeros((state_size, action_feature_dim), dtype=state.dtype if hasattr(state, 'dtype') else np.float32)
            # 复制原始状态的前state_feature_dim维（如果state是增强的，只复制原始部分）
            # 如果state是原始的，直接复制
            copy_dim = min(state_feature_dim, action_feature_dim)
            if state.ndim == 2:
                next_state[:, :copy_dim] = state[:, :copy_dim]
            else:
                for i in range(len(state)):
                    if len(state[i]) >= copy_dim:
                        next_state[i, :copy_dim] = state[i][:copy_dim]
        else:
            # 维度匹配，直接复制
            next_state = state.copy()
        
        # 更新状态历史
        first_zero_idx = -1
        for i in range(len(next_state)):
            if np.all(next_state[i] == 0):
                first_zero_idx = i
                break
        
        if first_zero_idx != -1:
            next_state[first_zero_idx] = action_feature
        else:
            next_state = np.roll(next_state, -1, axis=0)
            next_state[-1] = action_feature
            
    except (ValueError, KeyError) as e:
        # 动作无效就直接报错
        raise RuntimeError(f"Invalid action: cannot find valid edge for (cur_node={cur_node}, action_id/action/edge mapping, etc).")

    next_states.append(next_state)
    return np.array(next_states)

def find_new_position_after_action(action_id, current_pos, edge_id_map, G):
    """
    根据动作ID找到新的位置
    
    Args:
        action_id: 动作ID
        current_pos: 当前位置
        edge_id_map: 边ID映射
        G: 图对象
        
    Returns:
        new_pos: 新位置
        static_feature: 边的静态特征
        target_edge: 目标边
        edge_id: 边ID（用于文本 embedding 查找）
    """
    # Check if current node is in the graph, if not, try to find nearest node
    if current_pos not in G:
        raise ValueError(f"Current node {current_pos} not found in graph and graph is empty")

    # 查找对应的边
    target_edge = None
    edge_id = action_id  # 🔥 边ID就是action_id
    
    for edge_key, edge_idx in edge_id_map.items():
        if edge_idx == action_id:
            target_edge = edge_key
            break

    if target_edge is None:
        raise ValueError(f"No edge found for action_id {action_id}")

    # 关键修复：验证选择的边是否与当前节点相连
    if current_pos not in target_edge:
        raise ValueError(f"Action {action_id} (edge {target_edge}) is not connected to current node {current_pos}")

    # 根据target_edge在G中查找边的特征
    edge_data = G.get_edge_data(target_edge[0], target_edge[1])

    if edge_data is None:
        # 如果正向没找到，尝试反向
        edge_data = G.get_edge_data(target_edge[1], target_edge[0])
    
    if edge_data is None:
        raise ValueError(f"Edge {target_edge} not found in graph")
    
    # 获取边的静态特征
    static_feature = edge_data.get('static_feature', [0, 0, 0, 0, 0])

    next_node = target_edge[1] if target_edge[0] == current_pos else target_edge[0]

    return next_node, static_feature, target_edge, edge_id

def rollout_with_candidates(sess, model, start_node_id, goal_node_id, graph_cache, feature_builder,
                           max_steps=50, use_rl_head=False, temperature=1.0):
    """
    使用候选集决策进行rollout - V2版本：跨地图泛化

    Args:
        sess: TensorFlow session
        model: 模型实例
        start_node_id: 起始节点ID
        goal_node_id: 目标节点ID
        graph_cache: GraphCache实例
        feature_builder: FeatureBuilder实例
        max_steps: 最大步数
        use_rl_head: 是否使用RL head
        temperature: 采样温度

    Returns:
        path: 节点ID列表
        edge_path: 边ID列表
        success: 是否到达目标
        steps_taken: 实际步数
    """
    current_node_id = start_node_id
    path = [current_node_id]
    edge_path = []
    visited_nodes = set([current_node_id])
    recent_nodes = []  # 用于循环检测
    state_size = 10  # 默认历史长度

    for step in range(max_steps):
        # 获取候选边
        cand_edge_ids = graph_cache.get_candidate_edges(current_node_id)
        if not cand_edge_ids:
            break  # 没有候选边

        # 🔥 构建历史序列（使用最近state_size步的边ID）
        history_edge_ids = edge_path[-state_size:] if edge_path else []
        
        # 构建状态特征（历史序列）
        history_tokens = []
        for hist_edge_id in history_edge_ids:
            token = feature_builder.build_history_token(
                chosen_edge_id=hist_edge_id,
                prev_node_id=None,  # rollout时使用静态特征
                recent_visited_nodes=None
            )
            history_tokens.append(token)
        
        # Pad到state_size
        history_dim = feature_builder._get_history_dim()
        pad_token = np.zeros(history_dim, dtype=np.float32)
        while len(history_tokens) < state_size:
            history_tokens.insert(0, pad_token)  # 前面补零
        
        # 转换为numpy数组 [state_size, history_dim]
        state_sequence = np.array(history_tokens, dtype=np.float32)
        
        # 🔥 模型推理：使用旧的全局action space方式（不使用候选边特征）
        # 因为模型可能没有cand_features/cand_mask placeholders
        feed_dict = {
            model.inputs: state_sequence.reshape(1, state_size, history_dim),
            model.len_state: np.array([min(len(history_edge_ids), state_size)], dtype=np.int32),
            model.is_training: False
        }
        
        # 添加action_mask（只允许候选边）
        if hasattr(model, 'action_mask'):
            # 获取total edges数量
            num_edges = len(graph_cache.edge_id_to_nodes)
            action_mask = np.zeros((1, num_edges), dtype=np.float32)
            action_mask[0, cand_edge_ids] = 1.0
            feed_dict[model.action_mask] = action_mask

        # 获取预测
        if use_rl_head:
            # RL head: Q-values
            if hasattr(model, 'output1_masked'):
                q_values = sess.run(model.output1_masked, feed_dict=feed_dict)[0]
            else:
                q_values = sess.run(model.output1, feed_dict=feed_dict)[0]
            # 提取候选边的Q值
            cand_q_values = q_values[cand_edge_ids]
            logits = cand_q_values
        else:
            # SL head: probabilities
            if hasattr(model, 'probs'):
                probs = sess.run(model.probs, feed_dict=feed_dict)[0]
            else:
                # 如果没有probs，使用output2
                output = sess.run(model.output2, feed_dict=feed_dict)[0]
                # Softmax
                probs = np.exp(output - np.max(output))
                probs = probs / np.sum(probs)
            # 提取候选边的概率
            cand_probs = probs[cand_edge_ids]
            logits = np.log(cand_probs + 1e-10)  # 转换为logits

        # 温度采样或贪心选择
        if temperature > 0:
            # 温度采样
            scaled_logits = logits / temperature
            probs = np.exp(scaled_logits - np.max(scaled_logits))
            probs = probs / np.sum(probs)
            action_idx = np.random.choice(len(cand_edge_ids), p=probs)
        else:
            # 贪心选择
            action_idx = np.argmax(logits)

        # 执行动作
        chosen_edge_id = cand_edge_ids[action_idx]
        u_id, v_id = graph_cache.get_edge_nodes(chosen_edge_id)

        # 确定下一节点
        next_node_id = v_id if u_id == current_node_id else u_id

        # 记录路径
        edge_path.append(chosen_edge_id)
        path.append(next_node_id)

        # 更新状态
        current_node_id = next_node_id
        visited_nodes.add(current_node_id)
        recent_nodes.append(current_node_id)

        # 检查是否到达目标
        dist_to_goal = graph_cache.get_dist_to_goal(current_node_id, goal_node_id)
        if dist_to_goal < 0.1:  # 到达阈值
            return path, edge_path, True, step + 1

        # 检查循环
        if len(recent_nodes) >= 5 and next_node_id in recent_nodes[-5:]:
            break  # 检测到循环

    return path, edge_path, False, len(edge_path)

def predict_path_with_temperature_sampling(sess, model, start_state, target_pos, max_steps, G, edge_id_map, item_num,
                                          use_rl_head=False, arrival_threshold=1e-6, temperature=1.0,
                                          use_text_embeddings=False, text_embedding_loader=None, embedding_proj_weights=None, embedding_proj_biases=None,
                                          feature_dim=15, user_id_for_model=None, user_embedding=None):
    """
    使用模型进行temperature sampling rollout预测路径
    
    ⚠️ DEPRECATED: 此函数仅用于旧格式数据（15维坐标）
    新格式数据（9维历史序列）应使用rollout_with_candidates
    
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
        temperature: 采样温度 (0.0=greedy, 1.0=按概率采样, >1.0=更随机)
        text_embedding_loader: TextEmbeddingLoader 实例（可选）
        embedding_proj_weights: 投影权重字典（可选）
        embedding_proj_biases: 投影偏置字典（可选）
        feature_dim: 原始特征维度（默认15）
        user_id_for_model: 用户ID（用于PersonalizedQNetwork）
        user_embedding: 用户embedding向量（可选）
    
    Returns:
        predicted_path: 预测的动作序列（动作ID列表）
        csv_origin: 起点坐标（用于可视化）
        csv_destination: 终点坐标（用于可视化）
        success: 是否成功到达
        trajectory: 状态轨迹
    """
    # 🔥 修复：检查输入状态是否已经是增强状态
    # 如果start_state已经是增强状态（维度 > feature_dim），则不需要再次增强
    if not isinstance(start_state, np.ndarray):
        start_state = np.array(start_state)
    
    # 🔥 新格式检测：9维 = 历史序列，无坐标，无法使用此函数
    if start_state.shape[-1] == 9 or (target_pos is None):
        # 新格式无坐标信息，返回失败
        return [], None, None, False, []

    input_is_enhanced = start_state.shape[-1] > feature_dim if start_state.ndim >= 1 else False

    # 使用传入的参数，而不是自己计算

    # 如果启用了文本embeddings，或者需要添加用户embeddings，增强起始状态
    if (use_text_embeddings and text_embedding_loader is not None) or (user_embedding is not None and hasattr(model, 'feature_dim') and start_state.shape[-1] < model.feature_dim):
        # start_state should be 2D: (state_size, feature_dim)
        if start_state.ndim == 2:
            # Add batch dimension: (1, state_size, feature_dim)
            start_state_array = start_state[np.newaxis, :, :]
        elif start_state.ndim == 1:
            # If 1D, reshape to 2D first
            if len(start_state) == feature_dim:
                # Single frame, pad to state_size
                state_size = model.state_size
                start_state_array = np.zeros((1, state_size, feature_dim))
                start_state_array[0, 0, :] = start_state
            else:
                raise ValueError(f"Unexpected start_state shape: {start_state.shape}")
        else:
            raise ValueError(f"Unexpected start_state ndim: {start_state.ndim}")

        enhanced_start_state = enhance_states_with_text_embeddings(
            start_state_array, G, edge_id_map,
            text_embedding_loader=text_embedding_loader,
            embedding_proj_weights=embedding_proj_weights,
            embedding_proj_biases=embedding_proj_biases,
            original_feature_dim=feature_dim,
            user_embedding=user_embedding
        )
        current_state = enhanced_start_state[0]  # Remove batch dimension: (state_size, enhanced_feature_dim)
    else:
        # 即使没有文本embeddings，也可能需要添加用户embeddings
        current_state = np.array(start_state).copy()

        # 检查是否需要添加用户embeddings或填充维度
        if hasattr(model, 'feature_dim') and current_state.shape[-1] < model.feature_dim:
            # 如果模型期望的维度大于当前状态维度，需要添加缺失的维度
            missing_dims = model.feature_dim - current_state.shape[-1]
            print(f"🔧 Model expects {model.feature_dim} dims, current state has {current_state.shape[-1]}, adding {missing_dims} dims")

            if current_state.ndim == 2:
                # 已经是2D状态，扩展最后一维
                padding = np.zeros((current_state.shape[0], missing_dims))
                current_state = np.concatenate([current_state, padding], axis=-1)
            elif current_state.ndim == 1:
                # 1D状态，扩展维度
                padding = np.zeros(missing_dims)
                current_state = np.concatenate([current_state, padding], axis=-1)
    
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
            model.len_state: [effective_len(current_state)],
            model.action_mask: [action_mask],
            model.is_training: False,
            model.training_phase: 2 ,
            model.rl_weight: 0.3
        }
        
        # 🔥 如果模型有user_id_ph占位符（PersonalizedQNetwork且为lora模式），需要提供user_id
        if hasattr(model, 'user_id_ph') and model.user_id_ph is not None:
            if user_id_for_model is not None:
                feed_dict[model.user_id_ph] = np.array([int(user_id_for_model)], dtype=np.int32)
            else:
                # 默认使用0
                feed_dict[model.user_id_ph] = np.array([0], dtype=np.int32)
        
        # 根据use_rl_head选择head，并进行temperature sampling
        if use_rl_head:
            # 使用RL head (Q值)
            q_values = sess.run(model.output1_masked, feed_dict=feed_dict)[0]
            valid_values = q_values[valid_actions]
            
            # Temperature sampling for Q-values (need to convert to probabilities)
            if temperature == 0.0:
                # Greedy
                action_idx = int(np.argmax(valid_values))
            else:
                # Temperature scaling on Q-values
                scaled_values = valid_values / temperature
                # Softmax to get probabilities
                exp_values = np.exp(scaled_values - np.max(scaled_values))  # 数值稳定性
                action_probs = exp_values / np.sum(exp_values)
                # Sample
                action_idx = int(np.random.choice(len(valid_actions), p=action_probs))
        else:
            # 使用SL head (概率) - 已经是softmax后的概率
            probs = sess.run(model.probs, feed_dict=feed_dict)[0]
            valid_probs = probs[valid_actions]
            
            # 🔥 修复：对概率进行正确的temperature调整
            if temperature == 0.0:
                # Greedy
                action_idx = int(np.argmax(valid_probs))
            else:
                # 方法：对概率的幂进行调整（更适合已经是概率的情况）
                # temperature < 1.0 → 更确定性（增强高概率）
                # temperature > 1.0 → 更随机（平滑概率分布）
                adjusted_probs = np.power(valid_probs, 1.0/temperature)
                adjusted_probs = adjusted_probs / np.sum(adjusted_probs)  # 重新归一化
                # Sample
                action_idx = int(np.random.choice(len(valid_actions), p=adjusted_probs))
        
        action_id = valid_actions[action_idx]
        predicted_path.append(action_id)

        # 执行动作，更新状态
        try:
            # 🔥 如果启用了文本embeddings，需要从增强状态中提取原始状态用于状态转移计算
            if use_text_embeddings:
                # 提取原始状态用于状态转移计算
                current_state_original = current_state[:, :feature_dim] if current_state.ndim == 2 else current_state[:feature_dim]
                state_batch = current_state_original  # 直接传递数组，不是列表
            else:
                state_batch = current_state  # 直接传递数组，不是列表

            next_state = compute_state_transition([state_batch], [action_id], edge_id_map, G,
                                                 user_embedding=user_embedding)
            next_state_raw = next_state[0]  # 提取单个状态

            # 🔥 如果启用了文本embeddings，增强next_state
            if use_text_embeddings and text_embedding_loader is not None:
                # 确保next_state_raw是numpy数组格式，并添加batch维度
                if not isinstance(next_state_raw, np.ndarray):
                    next_state_raw = np.array(next_state_raw)

                # next_state_raw should be 2D: (state_size, feature_dim)
                if next_state_raw.ndim == 2:
                    # Add batch dimension: (1, state_size, feature_dim)
                    next_state_array = next_state_raw[np.newaxis, :, :]
                elif next_state_raw.ndim == 1:
                    # If 1D, reshape to 2D first
                    if len(next_state_raw) == feature_dim:
                        # Single frame, pad to state_size
                        state_size = model.state_size
                        next_state_array = np.zeros((1, state_size, feature_dim))
                        next_state_array[0, 0, :] = next_state_raw
                    else:
                        raise ValueError(f"Unexpected next_state_raw shape: {next_state_raw.shape}")
                else:
                    raise ValueError(f"Unexpected next_state_raw ndim: {next_state_raw.ndim}")

                enhanced_next_state = enhance_states_with_text_embeddings(
                    next_state_array, G, edge_id_map,
                    text_embedding_loader=text_embedding_loader,
                    embedding_proj_weights=embedding_proj_weights,
                    embedding_proj_biases=embedding_proj_biases,
                    original_feature_dim=feature_dim,
                    user_embedding=user_embedding
                )
                current_state = enhanced_next_state[0]  # Remove batch dimension: (state_size, enhanced_feature_dim)
            else:
                current_state = next_state_raw if isinstance(next_state_raw, np.ndarray) else np.array(next_state_raw)

            trajectory.append(current_state.copy())
            # 检查是否到达目标
            # 🔥 使用与rollout_rl_trajectory相同的成功判断逻辑
            if len(current_state) >= 2 and len(target_pos) >= 2:
                # 找到最后一个非零状态
                last_non_zero_idx = -1
                for i in range(len(current_state)-1, -1, -1):
                    if not np.all(current_state[i] == 0):
                        last_non_zero_idx = i
                        break

                if last_non_zero_idx != -1:
                    last_state = current_state[last_non_zero_idx]
                    # 🔥 使用固定索引13, 14获取当前位置（前15维的位置固定）
                    if len(last_state) >= 15:
                        cur_x = float(last_state[13])
                        cur_y = float(last_state[14])
                    else:
                        cur_x, cur_y = 0.0, 0.0
                else:
                    cur_x, cur_y = 0.0, 0.0

                distance = np.sqrt((cur_x - target_pos[0])**2 + (cur_y - target_pos[1])**2)

                if distance < arrival_threshold:  # 到达目标（使用容差）
                    success = True
                    break
        except Exception as e:
            # 发生异常时记录并退出
            break
    
    return predicted_path, csv_origin, csv_destination, success, trajectory

def rollout_rl_trajectory(sess, model, start_state, target_pos, max_steps, G, edge_id_map, item_num, reward_goal,
                          use_text_embeddings=False, text_embedding_loader=None, embedding_proj_weights=None, 
                          embedding_proj_biases=None, feature_dim=15, user_embedding=None, user_embeddings_dict=None, user_id=None):
    """
    🔥 RL真滚动：从起始状态按RL策略推进，看是否到达目标
    
    ⚠️ DEPRECATED: 此函数仅用于旧格式数据（15维坐标）
    新格式数据（9维历史序列）应使用rollout_with_candidates
    
    Args:
        sess: TensorFlow session
        model: RL模型
        start_state: 起始状态
        target_pos: 目标位置
        max_steps: 最大步数
        G: 图结构
        edge_id_map: 边ID映射
        item_num: 物品数量
        use_text_embeddings: 是否使用文本embeddings
        text_embedding_loader: TextEmbeddingLoader实例
        embedding_proj_weights: 投影权重字典
        embedding_proj_biases: 投影偏置字典
        feature_dim: 原始特征维度
    
    Returns:
        result: {
            'success': bool,  # 是否成功到达
            'steps': int,     # 实际步数
            'failure_reason': str,  # 失败原因
            'trajectory': list,  # 状态轨迹
            'actions': list,     # 动作序列
            'rewards': list,     # 奖励序列
            'total_reward': float, # 总奖励
            'reward_stats': dict  # 奖励统计
        }
    """
    # 🔥 定义到达阈值
    ARRIVAL_THRESHOLD = 1e-6
    
    # 🔥 提取state_size从start_state的形状
    if not isinstance(start_state, np.ndarray):
        start_state = np.array(start_state)
    
    # 🔥 新格式检测：9维 = 历史序列，无坐标，无法使用此函数
    if start_state.shape[-1] == 9 or (target_pos is None):
        # 新格式无坐标信息，返回失败
        return {
            'success': False,
            'steps': 0,
            'failure_reason': 'unsupported_format',
            'trajectory': [],
            'actions': [],
            'rewards': [],
            'total_reward': 0.0,
            'reward_stats': {}
        }
    
    if start_state.ndim == 2:
        state_size = start_state.shape[0]
    elif start_state.ndim == 1:
        # If 1D, assume it's a single frame and we need to infer state_size
        # This shouldn't happen normally, but handle it gracefully
        state_size = 10  # Default fallback
        if len(start_state) == feature_dim:
            # Single frame case
            pass
        else:
            raise ValueError(f"Cannot infer state_size from 1D start_state with shape {start_state.shape}")
    else:
        raise ValueError(f"Unexpected start_state ndim: {start_state.ndim}")
    
    # 🔥 如果启用了文本embeddings，增强起始状态
    if use_text_embeddings and text_embedding_loader is not None:
        # start_state should be 2D: (state_size, feature_dim)
        if start_state.ndim == 2:
            # Add batch dimension: (1, state_size, feature_dim)
            start_state_array = start_state[np.newaxis, :, :]
        elif start_state.ndim == 1:
            # If 1D, reshape to 2D first
            if len(start_state) == feature_dim:
                # Single frame, pad to state_size
                start_state_array = np.zeros((1, state_size, feature_dim))
                start_state_array[0, 0, :] = start_state
            else:
                raise ValueError(f"Unexpected start_state shape: {start_state.shape}")
        else:
            raise ValueError(f"Unexpected start_state ndim: {start_state.ndim}")
        
        enhanced_start_state = enhance_states_with_text_embeddings(
            start_state_array, G, edge_id_map,
            text_embedding_loader=text_embedding_loader,
            embedding_proj_weights=embedding_proj_weights,
            embedding_proj_biases=embedding_proj_biases,
            original_feature_dim=feature_dim,
            user_embedding=user_embedding,
            user_embeddings_dict=user_embeddings_dict,
            user_ids=[user_id] if user_id is not None else [0]
        )
        current_state = enhanced_start_state[0]  # Remove batch dimension: (state_size, enhanced_feature_dim)
    else:
        current_state = start_state.copy() if isinstance(start_state, np.ndarray) else np.array(start_state)

        # 检查是否需要添加用户embeddings或填充维度
        if hasattr(model, 'feature_dim') and current_state.shape[-1] < model.feature_dim:
            # 如果模型期望的维度大于当前状态维度，需要添加缺失的维度
            missing_dims = model.feature_dim - current_state.shape[-1]
            print(f"🔧 Model expects {model.feature_dim} dims, current state has {current_state.shape[-1]}, adding {missing_dims} dims")

            if current_state.ndim == 2:
                # 已经是2D状态，扩展最后一维
                padding = np.zeros((current_state.shape[0], missing_dims))
                current_state = np.concatenate([current_state, padding], axis=-1)
            elif current_state.ndim == 1:
                # 1D状态，扩展维度
                padding = np.zeros(missing_dims)
                current_state = np.concatenate([current_state, padding], axis=-1)
    
    trajectory = [current_state.copy()]
    actions = []
    rewards = []  # 🔥 新增：记录每步奖励
    prev_action = None  # 跟踪上一步动作
    action_history = None  # 跟踪最近3步动作历史
    recent_visited_nodes = []  # 🔥 GPT建议：跟踪最近K步访问的节点位置
    
    for step in range(max_steps):
        # 🔥 如果启用了文本embeddings，需要从增强状态中提取原始状态用于动作掩码生成
        # 动作掩码生成需要原始状态（15维），但模型需要增强状态
        if use_text_embeddings and text_embedding_loader is not None:
            # 提取原始状态（前feature_dim维）
            current_state_original = current_state[:, :feature_dim] if current_state.ndim == 2 else current_state[:feature_dim]
            action_mask = generate_action_mask_batch([current_state_original], G, edge_id_map, item_num)[0]
        else:
            action_mask = generate_action_mask_batch([current_state], G, edge_id_map, item_num)[0]
        
        valid_actions = [i for i, mask in enumerate(action_mask) if mask > 0]
        
        # 🔥 Actor-Critic DDPG-style: Use Actor for action selection (not Q-greedy)
        # During evaluation, use deterministic policy (argmax), not sampling
        feed_dict = {
            model.inputs: [current_state],
            model.len_state: [effective_len(current_state)],
            model.action_mask: [action_mask],
            model.is_training: False,
            model.training_phase: 2,
            model.rl_weight: 0.3,  # 🔥 使用降低的RL权重（默认0.3，与训练保持一致）
        }
        
        # 🔥 Use Actor (output2) for action selection: π(a|s)
        actor_probs = sess.run(model.probs, feed_dict=feed_dict)[0]
        
        # 🔥 Deterministic policy for evaluation: argmax π(a|s)
        # Only consider valid actions (action_mask already applied in model.probs)
        action_id = int(np.argmax(actor_probs))
        actions.append(action_id)
        
        # 更新动作历史：维护最近3步的动作
        if action_history is None:
            action_history = [action_id, action_id, action_id]  # 初始化：前两步也用当前动作
        else:
            action_history = [action_history[1], action_history[2], action_id]  # 滑动窗口
        
        # 🔥 修复：先保存上一步动作用于奖励计算，再更新prev_action
        prev_action_for_reward = prev_action
        
        # 执行动作，更新状态
        try:
            # 将current_state转换为compute_state_transition期望的格式
            # compute_state_transition需要原始状态（15维），所以需要提取原始部分
            if use_text_embeddings and text_embedding_loader is not None:
                # 提取原始状态用于状态转移计算
                current_state_original = current_state[:, :feature_dim] if current_state.ndim == 2 else current_state[:feature_dim]
                state_batch = current_state_original  # 直接传递数组，不是列表
            else:
                state_batch = current_state  # 直接传递数组，不是列表

            action_batch = [action_id]
            next_state = compute_state_transition([state_batch], action_batch, edge_id_map, G,
                                                 user_embedding=user_embedding)
            next_state_raw = next_state[0]  # 提取单个状态
            
            # 🔥 如果启用了文本embeddings，增强next_state
            if use_text_embeddings and text_embedding_loader is not None:
                # 确保next_state_raw是numpy数组格式，并添加batch维度
                if not isinstance(next_state_raw, np.ndarray):
                    next_state_raw = np.array(next_state_raw)
                
                # next_state_raw should be 2D: (state_size, feature_dim)
                if next_state_raw.ndim == 2:
                    # Add batch dimension: (1, state_size, feature_dim)
                    next_state_array = next_state_raw[np.newaxis, :, :]
                elif next_state_raw.ndim == 1:
                    # If 1D, reshape to 2D first
                    if len(next_state_raw) == feature_dim:
                        # Single frame, pad to state_size
                        next_state_array = np.zeros((1, state_size, feature_dim))
                        next_state_array[0, 0, :] = next_state_raw
                    else:
                        raise ValueError(f"Unexpected next_state_raw shape: {next_state_raw.shape}")
                else:
                    raise ValueError(f"Unexpected next_state_raw ndim: {next_state_raw.ndim}")
                
                enhanced_next_state = enhance_states_with_text_embeddings(
                    next_state_array, G, edge_id_map,
                    text_embedding_loader=text_embedding_loader,
                    embedding_proj_weights=embedding_proj_weights,
                    embedding_proj_biases=embedding_proj_biases,
                    original_feature_dim=feature_dim,
                    user_embedding=user_embedding,
                    user_embeddings_dict=user_embeddings_dict,
                    user_ids=[user_id] if user_id is not None else [0]
                )
                current_state = enhanced_next_state[0]  # Remove batch dimension: (state_size, enhanced_feature_dim)
            else:
                current_state = next_state_raw if isinstance(next_state_raw, np.ndarray) else np.array(next_state_raw)

                # 检查是否需要添加用户embeddings或填充维度
                if hasattr(model, 'feature_dim') and current_state.shape[-1] < model.feature_dim:
                    # 如果模型期望的维度大于当前状态维度，需要添加缺失的维度
                    missing_dims = model.feature_dim - current_state.shape[-1]

                    if current_state.ndim == 2:
                        # 已经是2D状态，扩展最后一维
                        padding = np.zeros((current_state.shape[0], missing_dims))
                        current_state = np.concatenate([current_state, padding], axis=-1)
                    elif current_state.ndim == 1:
                        # 1D状态，扩展维度
                        padding = np.zeros(missing_dims)
                        current_state = np.concatenate([current_state, padding], axis=-1)
            
            trajectory.append(current_state.copy())
            
            # 🔥 新增：计算当前步的奖励
            is_done = False
            if len(current_state) >= 2 and len(target_pos) >= 2:
                # 检查是否到达目标
                last_non_zero_idx = -1
                for i in range(len(current_state)-1, -1, -1):
                    if not np.all(current_state[i] == 0):
                        last_non_zero_idx = i
                        break
                
                if last_non_zero_idx != -1:
                    last_state = current_state[last_non_zero_idx]
                    # 🔥 修复：使用固定索引13, 14，而不是负索引
                    if len(last_state) >= 15:
                        cur_x = float(last_state[13])
                        cur_y = float(last_state[14])
                    else:
                        cur_x, cur_y = 0.0, 0.0
                else:
                    cur_x, cur_y = 0.0, 0.0
                
                distance = np.sqrt((cur_x - target_pos[0])**2 + (cur_y - target_pos[1])**2)
                is_done = (distance < ARRIVAL_THRESHOLD)
            
            # 🔥 势能型奖励：计算进展奖励
            path_info = {'end_pos': target_pos}
            
            reward, _ = calculate_improved_reward(
                action_id=action_id,
                is_done=is_done,
                reward_goal=reward_goal,
                step_idx=step,
                state_history=trajectory,
                path_info=path_info,
                prev_action=prev_action_for_reward,  # 🔥 使用真正的上一步动作
                edge_id_map=edge_id_map,
                step_penalty=-0.01,
                backtrack_penalty=-0.06,  # 回退惩罚
                recent_visited_nodes=recent_visited_nodes,
                repeat_visit_penalty=-0.03,  # 🔥 重复访问惩罚（可通过--r_repeat_visit调整）
                G=G,  # 🔥 传递图结构用于未探索边检测
                item_num=item_num  # 🔥 传递item_num用于边ID验证
            )
            rewards.append(reward)
            
            # 🔥 奖励计算完成后，更新prev_action用于下一步
            prev_action = action_id
            
            # 🔥 GPT建议：更新最近访问节点列表（维护最近K=3步）
            if len(current_state) >= 2:
                # 获取当前位置
                last_non_zero_idx = -1
                for i in range(len(current_state)-1, -1, -1):
                    if not np.all(current_state[i] == 0):
                        last_non_zero_idx = i
                        break
                
                if last_non_zero_idx != -1:
                    last_state = current_state[last_non_zero_idx]
                    # 🔥 使用固定索引13, 14获取当前位置（前15维的位置固定）
                    if hasattr(last_state[13], '__len__') and len(last_state[13]) > 0:
                        cur_x = float(last_state[13][0]) if len(last_state[13]) > 0 else 0.0
                    else:
                        cur_x = float(last_state[13])
                    
                    if hasattr(last_state[14], '__len__') and len(last_state[14]) > 0:
                        cur_y = float(last_state[14][0]) if len(last_state[14]) > 0 else 0.0
                    else:
                        cur_y = float(last_state[14])
                    
                    current_pos = (cur_x, cur_y)
                    recent_visited_nodes.append(current_pos)
                    
                    # 保持最近K=3步的访问记录
                    if len(recent_visited_nodes) > 3:
                        recent_visited_nodes = recent_visited_nodes[-3:]
            
            # 检查是否到达目标
            if len(current_state) >= 2 and len(target_pos) >= 2:
                # 🔥 修复：找到最后一个非零状态，而不是直接取最后两个元素
                last_non_zero_idx = -1
                for i in range(len(current_state)-1, -1, -1):
                    if not np.all(current_state[i] == 0):
                        last_non_zero_idx = i
                        break
                
                if last_non_zero_idx != -1:
                    # 从最后一个非零状态获取当前位置坐标
                    last_state = current_state[last_non_zero_idx]
                    # 🔥 使用固定索引13, 14获取当前位置（前15维的位置固定）
                    if hasattr(last_state[13], '__len__') and len(last_state[13]) > 0:
                        cur_x = float(last_state[13][0]) if len(last_state[13]) > 0 else 0.0
                    else:
                        cur_x = float(last_state[13])
                    
                    if hasattr(last_state[14], '__len__') and len(last_state[14]) > 0:
                        cur_y = float(last_state[14][0]) if len(last_state[14]) > 0 else 0.0
                    else:
                        cur_y = float(last_state[14])
                else:
                    # 如果没有找到非零状态，使用默认值
                    cur_x, cur_y = 0.0, 0.0
                
                distance = np.sqrt((cur_x - target_pos[0])**2 + (cur_y - target_pos[1])**2)
                
                if distance < ARRIVAL_THRESHOLD:  # 到达目标（使用容差）
                    # print(f"✅ SUCCESS at step {step}! Distance: {distance}")
                    reward_stats = _calculate_reward_stats(rewards)
                    return {
                        'success': True,
                        'steps': step + 1,
                        'failure_reason': 'success',
                        'trajectory': trajectory,
                        'actions': actions,
                        'rewards': rewards,
                        'total_reward': reward_stats['total_reward'],
                        'reward_stats': reward_stats
                    }

            # 检查循环（K步回环）- 检查是否回到了更早的位置
            if len(trajectory) >= 10:  # 至少需要10步才开始检查循环
                # 🔥 修复：找到最后一个非零状态获取当前位置坐标
                last_non_zero_idx = -1
                for i in range(len(current_state)-1, -1, -1):
                    if not np.all(current_state[i] == 0):
                        last_non_zero_idx = i
                        break
                
                if last_non_zero_idx != -1:
                    last_state = current_state[last_non_zero_idx]
                    # 🔥 使用固定索引13, 14获取当前位置（前15维的位置固定）
                    if hasattr(last_state[13], '__len__') and len(last_state[13]) > 0:
                        cur_x = float(last_state[13][0]) if len(last_state[13]) > 0 else 0.0
                    else:
                        cur_x = float(last_state[13])
                    
                    if hasattr(last_state[14], '__len__') and len(last_state[14]) > 0:
                        cur_y = float(last_state[14][0]) if len(last_state[14]) > 0 else 0.0
                    else:
                        cur_y = float(last_state[14])
                    
                    current_pos = [cur_x, cur_y]
                else:
                    current_pos = [0, 0]
                
                # 检查是否回到了5步之前的位置（避免立即检测到循环）
                for i in range(max(0, len(trajectory)-10), len(trajectory)-5):
                    # 🔥 修复：从历史状态中找到最后一个非零状态获取位置
                    prev_state = trajectory[i]
                    prev_last_non_zero_idx = -1
                    for j in range(len(prev_state)-1, -1, -1):
                        if not np.all(prev_state[j] == 0):
                            prev_last_non_zero_idx = j
                            break
                    
                    if prev_last_non_zero_idx != -1:
                        prev_last_state = prev_state[prev_last_non_zero_idx]
                        # 🔥 使用固定索引13, 14获取当前位置（前15维的位置固定）
                        if hasattr(prev_last_state[13], '__len__') and len(prev_last_state[13]) > 0:
                            prev_x = float(prev_last_state[13][0]) if len(prev_last_state[13]) > 0 else 0.0
                        else:
                            prev_x = float(prev_last_state[13])
                        
                        if hasattr(prev_last_state[14], '__len__') and len(prev_last_state[14]) > 0:
                            prev_y = float(prev_last_state[14][0]) if len(prev_last_state[14]) > 0 else 0.0
                        else:
                            prev_y = float(prev_last_state[14])
                        
                        prev_pos = [prev_x, prev_y]
                    else:
                        prev_pos = [0, 0]
                    
                    # 使用更宽松的容差，并且要求位置确实相同
                    if np.allclose(current_pos, prev_pos, atol=0.00001):
                        # print(f"🔄 LOOP DETECTED at step {step}: current={current_pos}, prev={prev_pos} (step {i})")
                        # 🔥 优化2: 添加失败终奖
                        loop_path_info = {'end_pos': target_pos}
                        
                        final_reward, _ = calculate_improved_reward(
                            action_id=action_id,
                            is_done=False,
                            reward_goal=reward_goal,
                            step_idx=step,
                            state_history=trajectory,
                            path_info=loop_path_info,
                            prev_action=prev_action,
                            edge_id_map=edge_id_map,
                            step_penalty=-0.01,
                            backtrack_penalty=-0.05,
                            is_failure=True,
                            failure_type='loop_stuck'
                        )
                        rewards.append(final_reward)
                        
                        reward_stats = _calculate_reward_stats(rewards)
                        return {
                            'success': False,
                            'steps': step + 1,
                            'failure_reason': 'loop_stuck',
                            'trajectory': trajectory,
                            'actions': actions,
                            'rewards': rewards,
                            'total_reward': reward_stats['total_reward'],
                            'reward_stats': reward_stats
                        }
                        
        except Exception as e:
            # 调试信息：只在第一步打印异常
            if step == 0:
                print(f"❌ State transition error at step {step}: {e}")
                print(f"   Action ID: {action_id}")
                print(f"   Current state length: {len(current_state)}")
                print(f"   Current state type: {type(current_state)}")
                print(f"   Current state[0] type: {type(current_state[0])}")
                import traceback
                print(f"   Traceback: {traceback.format_exc()}")
            reward_stats = _calculate_reward_stats(rewards)
            return {
                'success': False,
                'steps': step + 1,
                'failure_reason': 'invalid_action',
                'trajectory': trajectory,
                'actions': actions,
                'rewards': rewards,
                'total_reward': reward_stats['total_reward'],
                'reward_stats': reward_stats
            }
    
    # 超时
    print(f"⏰ TIMEOUT after {max_steps} steps")
    # 🔥 优化2: 添加失败终奖
    timeout_path_info = {'end_pos': target_pos}
    
    final_reward, _ = calculate_improved_reward(
        action_id=actions[-1] if actions else 0,
        is_done=False,
        reward_goal=reward_goal,
        step_idx=max_steps,
        state_history=trajectory,
        path_info=timeout_path_info,
        prev_action=actions[-2] if len(actions) >= 2 else None,
        edge_id_map=edge_id_map,
        step_penalty=-0.01,
        backtrack_penalty=-0.05,
        is_failure=True,
        failure_type='timeout',
        G=G, item_num=item_num  # 🔥 传递图结构和item_num用于未探索边检测
    )
    rewards.append(final_reward)
    
    reward_stats = _calculate_reward_stats(rewards)
    return {
        'success': False,
        'steps': max_steps,
        'failure_reason': 'timeout',
        'trajectory': trajectory,
        'actions': actions,
        'rewards': rewards,
        'total_reward': reward_stats['total_reward'],
        'reward_stats': reward_stats
    }

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import argparse
import pickle
import time
from utils.utility import *
from utils.logger import *
from SASRecModules import *
from copy import deepcopy
import trfl

def compute_state_transition(current_state, action_id, edge_id_map, G):
    """
    计算执行动作后的状态转移 - 增强版，处理无效动作
    
    Args:
        current_state: 当前状态向量 [batch_size, state_size, feature_dim]
        action_id: 动作ID [batch_size]
        edge_id_map: 边ID映射
        G: 图对象
        state_size: 状态大小
        
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
    else:
        # Get the last two numbers of the last non-zero state as current node
        last_state = state[last_non_zero_idx]
        cur_node = (float(last_state[-2]), float(last_state[-1]))
        # Get start and end coordinates
        origin_x, origin_y = float(last_state[-6]), float(last_state[-5])  # Start point
        dest_x, dest_y = float(last_state[-4]), float(last_state[-3])  # End point
        
        # 尝试根据动作ID找到对应的边
        try:
            new_pos, static_feature, target_edge = find_new_position_after_action(action, cur_node, edge_id_map, G)
            
            # 构建新的状态
            next_state = state.copy()
            
            # 创建新的动作特征向量
            action_feature = create_action_feature(new_pos, origin_x, origin_y, dest_x, dest_y, static_feature, target_edge)
            
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
            # 动作无效，创建"原地踏步"特征
            
            # 创建原地踏步的特征向量
            next_state = state.copy()
            action_feature = create_stay_in_place_feature(cur_node, origin_x, origin_y, dest_x, dest_y)
            
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

        next_states.append(next_state)
    return np.array(next_states)

def create_stay_in_place_feature(cur_node, origin_x, origin_y, dest_x, dest_y):
    """
    创建原地踏步的特征向量（用于无效动作）
    """
    cur_node_x, cur_node_y = cur_node
    
    # 创建原地踏步的特征向量
    feature_vec = [
        cur_node_x, cur_node_y,  # 起点（当前位置）
        cur_node_x, cur_node_y,  # 终点（当前位置）
        0,  # crossing
        0,  # path_type
        0.0,  # length
        0.0,  # obstacle_free_width
        0.0,  # curb_height
        origin_x, origin_y,  # 路径起始点
        dest_x, dest_y,      # 路径目标点
        cur_node_x, cur_node_y  # 当前位置
    ]
    
    return feature_vec

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
    """
    # Check if current node is in the graph
    if current_pos not in G:
        raise ValueError(f"Current node {current_pos} not found in graph")

    # 查找对应的边
    target_edge = None
    
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

    return next_node, static_feature, target_edge

def create_action_feature(new_pos, origin_x, origin_y, dest_x, dest_y, static_feature, target_edge):
    """
    创建动作特征向量，格式与原始数据一致
    
    Args:
        new_pos: 新位置 (cur_x, cur_y)
        start_pos: 起始位置 (origin_x, origin_y)
        end_pos: 目标位置 (dest_x, dest_y)
        static_feature: 边信息，包含静态特征
        
    Returns:
        action_feature: 动作特征向量 [15维]
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
    
    return feature_vec

def extract_action_ids(actions, edge_id_map):
    """
    Extract action_ids from action list
    
    Args:
        actions: Action list, may contain int or feature vector
        edge_id_map: Edge ID mapping
        
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

    parser.add_argument('--epoch', type=int, default=60,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='../data',
                        help='data directory')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=128,
                        help='Number of hidden factors, i.e., embedding size.')
    
    # Reward parameters - Enhanced for better path planning
    parser.add_argument('--r_goal', type=float, default=1.0,
                        help='reward for reaching the goal.')
    
    # Learning parameters
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--discount', type=float, default=0.999,
                        help='Discount factor for RL.')
    
    # RL parameters
    parser.add_argument('--weight', type=float, default=1.0, 
                        help='weight for Q-learning loss in the total loss function.')
    parser.add_argument('--lr_2', type=float, default=0.0001,
                        help='Learning rate for the second optimizer, which is used for the second loss function.')
    
    # Model architecture parameters
    parser.add_argument('--num_heads', default=8, type=int, help='number of heads (for SASRec)')
    parser.add_argument('--num_blocks', default=6, type=int, help='Number of blocks (for SASRec)')
    parser.add_argument('--dropout_rate', default=0.15, type=float, help='Dropout rate for regularization.') 
    
    # Training phase parameters - 两阶段训练策略
    parser.add_argument('--phase1_sl_only_steps', type=int, default=8000,  # 第一阶段：纯SL训练
                        help='Number of steps for Phase 1: SL-only training.')
    parser.add_argument('--phase2_sl_rl_joint_steps', type=int, default=40000,  # 第二阶段：SL+RL联合训练
                        help='Number of steps for Phase 2: SL+RL joint training.')
    parser.add_argument('--rl_weight_phase2', type=float, default=1.0,  # 第二阶段RL权重
                        help='RL weight for Phase 2 (RL-only).')
    parser.add_argument('--resume_ckpt', type=str, default=None,
                        help='Path to checkpoint to resume training from (e.g. ../data/saved_model/v4_sl_only_10k_steps.ckpt)')
    parser.add_argument('--preload_datasets', action='store_true',
                        help='Preload and cache all datasets before training (recommended for faster evaluation)')
    parser.add_argument('--log_frequency', type=int, default=2000,
                        help='Frequency of logging metrics (steps). Higher values = faster training.')
    parser.add_argument('--eval_frequency', type=int, default=4000,
                        help='Frequency of evaluation (steps). Higher values = faster training.')

    return parser.parse_args()

class ImprovedQNetwork:
    def __init__(self, hidden_size, learning_rate, feature_dim, item_num, state_size, weight, dropout_rate, num_heads, num_blocks, lr_2, name='ImprovedDQNetwork'):
        tf.compat.v1.disable_eager_execution()
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.feature_dim = int(feature_dim)

        self.weight = weight
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        self.item_num = item_num
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

            # Output layers
            with tf.compat.v1.variable_scope('output1'):
                weights1 = tf.compat.v1.get_variable('weights', 
                    [self.hidden_size, self.item_num],
                    initializer=tf.compat.v1.glorot_normal_initializer())
                biases1 = tf.compat.v1.get_variable('biases', 
                    [self.item_num],
                    initializer=tf.compat.v1.zeros_initializer())
                self.output1 = tf.matmul(self.states_hidden, weights1) + biases1
            
            with tf.compat.v1.variable_scope('output2'):
                weights2 = tf.compat.v1.get_variable('weights', 
                    [self.hidden_size, self.item_num],
                    initializer=tf.compat.v1.glorot_normal_initializer())
                biases2 = tf.compat.v1.get_variable('biases', 
                    [self.item_num],
                    initializer=tf.compat.v1.zeros_initializer())
                self.output2 = tf.matmul(self.states_hidden, weights2) + biases2
            
            # Action mask placeholder
            self.action_mask = tf.compat.v1.placeholder(tf.float32, [None, item_num], name='action_mask')
            
            # SL先验候选集剪枝相关
            self.use_sl_prior = tf.compat.v1.placeholder(tf.bool, name='use_sl_prior')
            self.sl_prior_top_p = tf.compat.v1.placeholder(tf.float32, name='sl_prior_top_p')
            self.prev_action = tf.compat.v1.placeholder(tf.int32, [None], name='prev_action')
            self.action_history = tf.compat.v1.placeholder(tf.int32, [None, 3], name='action_history')
            self.training_phase = tf.compat.v1.placeholder(tf.int32, name='training_phase')
            
            # Mask penalty计算：为无效动作提供负奖励
            q_value_max = tf.reduce_max(self.output1, axis=1, keepdims=True)
            q_value_min = tf.reduce_min(self.output1, axis=1, keepdims=True)
            q_value_range = q_value_max - q_value_min
            
            pen_by_range = -0.5 * tf.stop_gradient(q_value_range)
            pen_below_min = tf.stop_gradient(q_value_min) - 1.0
            
            pen = tf.minimum(pen_by_range, pen_below_min)
            pen = tf.clip_by_value(pen, clip_value_min=-8.0, clip_value_max=-1e-3)
            mask_penalty = tf.stop_gradient(pen)
            
            self.mask_penalty_value = mask_penalty

            # SL先验候选集剪枝
            self.sl_probs = tf.nn.softmax(self.output2)
            self.degree = tf.reduce_sum(tf.cast(self.action_mask, tf.int32), axis=1)
            
            self.candidate_mask = tf.cond(
                self.use_sl_prior,
                lambda: build_candidate_mask(self.sl_probs, self.action_mask, self.degree, self.sl_prior_top_p, self.prev_action, self.action_history),
                lambda: tf.cast(self.action_mask, tf.bool)
            )
            
            # 分离训练用和推断用logits
            self.output1_for_training = self.output1 + mask_penalty * (1.0 - tf.cast(self.action_mask, tf.float32))
            self.output2_for_training = self.output2 + mask_penalty * (1.0 - tf.cast(self.action_mask, tf.float32))
            
            self.output1_masked = self.output1 * tf.cast(self.candidate_mask, tf.float32) + mask_penalty * (1.0 - tf.cast(self.candidate_mask, tf.float32))
            self.output2_masked = self.output2 * tf.cast(self.candidate_mask, tf.float32) + mask_penalty * (1.0 - tf.cast(self.candidate_mask, tf.float32))
            
            # Action inputs
            self.actions = tf.compat.v1.placeholder(tf.int32, [None])
            # Placeholders
            self.reward = tf.compat.v1.placeholder(tf.float32, [None])
            self.discount = tf.compat.v1.placeholder(tf.float32, [None])
            self.targetQs_ = tf.compat.v1.placeholder(tf.float32, [None, item_num])
            self.targetQs_selector = tf.compat.v1.placeholder(tf.float32, [None, item_num])
            # 🔥 清理：删除未使用的target_Q_current相关placeholders
            
            # TRFL double Q-learning
            qloss_positive, _ = trfl.double_qlearning(self.output1_for_training, self.actions, self.reward, self.discount,
                                                      self.targetQs_, self.targetQs_selector)
            
            ce_loss_pre = tf.compat.v1.nn.sparse_softmax_cross_entropy_with_logits(labels=self.actions, logits=self.output2_for_training)
            self.ce_loss = tf.reduce_mean(ce_loss_pre)
            self.q_loss = tf.reduce_mean(qloss_positive)
            
            self.rl_weight = tf.compat.v1.placeholder(tf.float32, name='rl_weight')
            self.q_loss_weighted = self.rl_weight * (self.q_loss)

            # 三阶段训练损失定义
            phase1_sl_only_loss = self.ce_loss  
            phase2_sl_rl_loss = self.ce_loss + self.q_loss_weighted
            
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
                'ce_loss': self.ce_loss,
                'q_loss': self.q_loss,
                'q_loss_original': self.q_loss,
                'q_loss_weighted': self.q_loss_weighted,
                'q_loss_scaled': self.q_loss_weighted * self.rl_weight,
                'qloss_positive': tf.reduce_mean(qloss_positive),
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
            self.sl_head_optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(
                self.ce_loss, var_list=self.sl_head_vars
            )
            
            # 共享编码器的梯度合成
            self.q_grads = tf.gradients(self.q_loss, self.shared_encoder_vars)
            self.sl_grads = tf.gradients(self.ce_loss, self.shared_encoder_vars)
            
            def pareto_gradient_synthesis(q_grads, sl_grads, alpha=0.5):
                """加权梯度融合"""
                combined_grads = []
                for q_g, sl_g in zip(q_grads, sl_grads):
                    if q_g is not None and sl_g is not None:
                        combined_grads.append(alpha * q_g + (1.0 - alpha) * sl_g)
                    elif q_g is not None:
                        combined_grads.append(q_g)
                    elif sl_g is not None:
                        combined_grads.append(sl_g)
                    else:
                        combined_grads.append(None)
                return combined_grads
            
            self.combined_grads = pareto_gradient_synthesis(
                self.q_grads, self.sl_grads, alpha=self.rl_weight
            )
            
            valid_combined_grads = [grad for grad in self.combined_grads if grad is not None]
            valid_combined_vars = [var for grad, var in zip(self.combined_grads, self.shared_encoder_vars) if grad is not None]
            
            if valid_combined_grads:
                clipped_combined_grads, _ = tf.clip_by_global_norm(valid_combined_grads, 5.0)
                self.shared_encoder_optimizer = tf.compat.v1.train.AdamOptimizer(self.lr_2).apply_gradients(
                    zip(clipped_combined_grads, valid_combined_vars)
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
                
    def initialize_embeddings(self):
        all_embeddings = dict()
        pos_embeddings = tf.Variable(tf.random.normal([self.state_size, self.hidden_size], 0.0, 0.01),
                                        name='pos_embeddings')
        all_embeddings['pos_embeddings'] = pos_embeddings
        return all_embeddings

def calculate_loop_stuck_rate_from_rollouts(rl_rollout_results):
    """计算循环/卡住率"""
    if not rl_rollout_results:
        return 0.0
    
    loop_stuck_count = sum(1 for result in rl_rollout_results if result['failure_reason'] == 'loop_stuck')
    return loop_stuck_count / len(rl_rollout_results)

def states_equal(state1, state2, tolerance=1e-6):
    """
    判断两个状态是否相等（考虑浮点精度）
    """
    if len(state1) != len(state2):
        return False
    
    for i in range(len(state1)):
        if abs(state1[i] - state2[i]) > tolerance:
            return False
    
    return True

def calculate_rl_comprehensive_score(reach_at_budget, loop_stuck_rate, invalid_action_rate, success_steps_median):
    """
    🔥 改进的RL综合评分 - 专注可达性，移除效率项
    
    Args:
        reach_at_budget: 在预算内到达的比例
        loop_stuck_rate: 循环/卡住率
        invalid_action_rate: 无效动作率
        success_steps_median: 成功步数中位数（仅用于诊断）
    
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

def rollout_rl_trajectory(sess, model, start_state, target_pos, max_steps, G, edge_id_map, item_num, reward_goal):
    """
    🔥 RL真滚动：从起始状态按RL策略推进，看是否到达目标
    
    Args:
        sess: TensorFlow session
        model: RL模型
        start_state: 起始状态
        target_pos: 目标位置
        max_steps: 最大步数
        G: 图结构
        edge_id_map: 边ID映射
        item_num: 物品数量
    
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
    ARRIVAL_THRESHOLD = 1e-3
    current_state = start_state.copy()
    trajectory = [current_state.copy()]
    actions = []
    rewards = []  # 🔥 新增：记录每步奖励
    prev_action = None  # 跟踪上一步动作
    action_history = None  # 跟踪最近3步动作历史
    
    for step in range(max_steps):
        # 生成动作掩码
        action_mask = generate_action_mask_batch([current_state], G, edge_id_map, item_num)[0]
        valid_actions = [i for i, mask in enumerate(action_mask) if mask > 0]
        
        if not valid_actions:
            reward_stats = _calculate_reward_stats(rewards)
            return {
                'success': False,
                'steps': step,
                'failure_reason': 'invalid_action',
                'trajectory': trajectory,
                'actions': actions,
                'rewards': rewards,
                'total_reward': reward_stats['total_reward'],
                'reward_stats': reward_stats
            }
        
        # 使用RL head选择动作
        feed_dict = {
            model.inputs: [current_state],
            model.len_state: [len(current_state)],
            model.action_mask: [action_mask],
            model.is_training: False,
            # 🔥 SL先验候选集剪枝：rollout时也启用
            model.use_sl_prior: False,
            model.sl_prior_top_p: 0.80,
            model.prev_action: [prev_action] if prev_action is not None else [0],
            model.action_history: [action_history] if action_history is not None else [[0, 0, 0]]
        }
        
        # 获取Q值并选择动作
        q_values = sess.run(model.output1_masked, feed_dict=feed_dict)[0]
        
        # 只考虑有效动作
        valid_q_values = [q_values[i] for i in valid_actions]
        if not valid_q_values:
            reward_stats = _calculate_reward_stats(rewards)
            return {
                'success': False,
                'steps': step,
                'failure_reason': 'invalid_action',
                'trajectory': trajectory,
                'actions': actions,
                'rewards': rewards,
                'total_reward': reward_stats['total_reward'],
                'reward_stats': reward_stats
            }
        
        # 选择Q值最大的有效动作
        best_valid_idx = np.argmax(valid_q_values)
        action_id = valid_actions[best_valid_idx]
        actions.append(action_id)
        
        # 更新动作历史：维护最近3步的动作
        if action_history is None:
            action_history = [action_id, action_id, action_id]  # 初始化：前两步也用当前动作
        else:
            action_history = [action_history[1], action_history[2], action_id]  # 滑动窗口
        
        # 更新prev_action用于下一步的no backtrack约束
        prev_action = action_id
        
        # 执行动作，更新状态
        try:
            # 将current_state转换为compute_state_transition期望的格式
            state_batch = [current_state]
            action_batch = [action_id]
            next_state = compute_state_transition(state_batch, action_batch, edge_id_map, G)
            current_state = next_state[0]  # 提取单个状态
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
                    if hasattr(last_state[-2], '__len__') and len(last_state[-2]) > 0:
                        cur_x = float(last_state[-2][0]) if len(last_state[-2]) > 0 else 0.0
                    else:
                        cur_x = float(last_state[-2])
                    
                    if hasattr(last_state[-1], '__len__') and len(last_state[-1]) > 0:
                        cur_y = float(last_state[-1][0]) if len(last_state[-1]) > 0 else 0.0
                    else:
                        cur_y = float(last_state[-1])
                else:
                    cur_x, cur_y = 0.0, 0.0
                
                distance = np.sqrt((cur_x - target_pos[0])**2 + (cur_y - target_pos[1])**2)
                is_done = (distance < ARRIVAL_THRESHOLD)
            
            # 计算奖励（使用简化的奖励函数）
            reward, reward_components = calculate_improved_reward(
                action_id=action_id,
                target_action_id=None,  # rollout时不关心目标动作
                is_done=is_done,
                reward_goal=reward_goal,
                step_idx=step,
                state_history=trajectory,
                is_valid_action=True,
                path_info={'end_pos': target_pos}
            )
            rewards.append(reward)
            
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
                    if hasattr(last_state[-2], '__len__') and len(last_state[-2]) > 0:
                        cur_x = float(last_state[-2][0]) if len(last_state[-2]) > 0 else 0.0
                    else:
                        cur_x = float(last_state[-2])
                    
                    if hasattr(last_state[-1], '__len__') and len(last_state[-1]) > 0:
                        cur_y = float(last_state[-1][0]) if len(last_state[-1]) > 0 else 0.0
                    else:
                        cur_y = float(last_state[-1])
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
                    if hasattr(last_state[-2], '__len__') and len(last_state[-2]) > 0:
                        cur_x = float(last_state[-2][0]) if len(last_state[-2]) > 0 else 0.0
                    else:
                        cur_x = float(last_state[-2])
                    
                    if hasattr(last_state[-1], '__len__') and len(last_state[-1]) > 0:
                        cur_y = float(last_state[-1][0]) if len(last_state[-1]) > 0 else 0.0
                    else:
                        cur_y = float(last_state[-1])
                    
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
                        if hasattr(prev_last_state[-2], '__len__') and len(prev_last_state[-2]) > 0:
                            prev_x = float(prev_last_state[-2][0]) if len(prev_last_state[-2]) > 0 else 0.0
                        else:
                            prev_x = float(prev_last_state[-2])
                        
                        if hasattr(prev_last_state[-1], '__len__') and len(prev_last_state[-1]) > 0:
                            prev_y = float(prev_last_state[-1][0]) if len(prev_last_state[-1]) > 0 else 0.0
                        else:
                            prev_y = float(prev_last_state[-1])
                        
                        prev_pos = [prev_x, prev_y]
                    else:
                        prev_pos = [0, 0]
                    
                    # 使用更宽松的容差，并且要求位置确实相同
                    if np.allclose(current_pos, prev_pos, atol=0.001):
                        # print(f"🔄 LOOP DETECTED at step {step}: current={current_pos}, prev={prev_pos} (step {i})")
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

def calculate_improved_reward(action_id, target_action_id, is_done, reward_goal, 
                             step_idx=0, state_history=None, is_valid_action=True, path_info=None):
    """RL奖励函数：纯到达导向"""
    reward_components = {}
    base_reward = 0.0
    
    reachability_reward = 0.0
    if is_done:
        reachability_reward = reward_goal
        reward_components['reachability'] = reachability_reward
    
    base_reward += reachability_reward
    final_reward = base_reward
    reward_components['final_reward'] = final_reward
    
    return final_reward, reward_components

def determine_training_phase(global_step, args):
    """确定当前训练阶段和相关参数"""
    phase1_end = args.phase1_sl_only_steps
    phase2_end = phase1_end + args.phase2_sl_rl_joint_steps
    
    if global_step < phase1_end:
        # 第一阶段：纯SL训练
        current_phase = 1
        phase_name = "Phase1-SL-Only"
        rl_weight = 0.0
        current_lr = args.lr
        
    elif global_step < phase2_end:
        # 第二阶段：SL + RL joint 训练
        current_phase = 2
        phase_name = "Phase2-SL+RL-Joint"
        rl_weight = args.rl_weight_phase2
        current_lr = args.lr_2
        
    else:
        # 超出预定义阶段，继续使用第二阶段设置
        current_phase = 2
        phase_name = "Phase2-SL+RL-Joint-Extended"
        rl_weight = args.rl_weight_phase2
        current_lr = args.lr_2
    
    return current_phase, phase_name, rl_weight, current_lr

def extract_target_position_from_state(state):
    """从状态中提取目标位置"""
    # state是嵌套列表结构 [state_size, feature_dim]
    if isinstance(state, list) and len(state) > 0:
        # 找到最后一个非零状态
        for i in range(len(state) - 1, -1, -1):
            if isinstance(state[i], list) and len(state[i]) >= 4:
                # 检查是否全为零
                if not all(x == 0 for x in state[i]):
                    return (float(state[i][-4]), float(state[i][-3]))  # 倒数第4、第3位是目标位置
    elif hasattr(state, 'shape') and len(state.shape) > 1:
        # 多维数组情况：找到最后一个非零状态
        for i in range(state.shape[0] - 1, -1, -1):
            if not np.all(state[i] == 0):
                if len(state[i]) >= 4:
                    return (float(state[i][-4]), float(state[i][-3]))  # 倒数第4、第3位是目标位置
                break
    elif isinstance(state, (list, np.ndarray)) and len(state) >= 4:
        # 一维列表情况
        if isinstance(state[0], (list, np.ndarray)):
            # 如果state是列表的列表，取最后一个非零元素
            for i in range(len(state) - 1, -1, -1):
                if isinstance(state[i], (list, np.ndarray)) and len(state[i]) >= 4:
                    if not np.all(np.array(state[i]) == 0):
                        return (float(state[i][-4]), float(state[i][-3]))
        else:
            # 直接是一维数组
            return (float(state[-4]), float(state[-3]))
    
    return None # 默认值

def build_feed_dict(model, state, len_state, target_Qs, reward, 
                   discount, action_ids, target_Qs_selector, action_masks,
                   current_phase, rl_weight, is_training=True, 
                   use_sl_prior=False, sl_prior_top_p=0.80, prev_action=None, action_history=None):
    """
    Build universal feed_dict
    
    Args:
        model: Model object
        ...: Various input parameters
        behavior_prob: Behavior probability
        
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
        # 🔥 清理：删除未使用的target_Q_current相关字段
        model.action_mask: action_masks,
        model.is_training: is_training,
        model.training_phase: current_phase,
        model.rl_weight: rl_weight,
        # 🔥 SL先验候选集剪枝参数
        model.use_sl_prior: use_sl_prior,
        model.sl_prior_top_p: sl_prior_top_p,
        model.prev_action: prev_action if prev_action is not None else np.zeros(len(state), dtype=np.int32),
        model.action_history: action_history if action_history is not None else np.zeros((len(state), 3), dtype=np.int32),
    }
    
    return feed_dict

def batch_evaluate_improved(sess, model, dataset='val', logger=None, step=None, training_phase=1, rl_weight=0.0, batch_size=64, sample_ratio=1.0, rl_rollout_sample_size=100, G=None, edge_id_map=None, item_num=None, reward_goal=None):
    """
    Batch evaluation function for significantly improved performance with caching
    """
    import os
    import pickle
    import time
    import numpy as np
    import pandas as pd
    
    # 记录评估开始时间
    eval_start_time = time.time()
    
    # 🔥 处理reward_goal默认值
    if reward_goal is None:
        reward_goal = 1.0  # 默认奖励值
    
    # 设置数据目录
    data_directory = '../data'  # 默认数据目录
    
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
    cache_file = os.path.join(cache_dir, f'{dataset}_batch_data.pkl')
    
    # 尝试加载缓存数据
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                all_path_data = cached_data['path_data']
                all_path_actions = cached_data['path_actions']
                all_path_len_states = cached_data['path_len_states']
                all_path_lengths = cached_data['path_lengths']
                eval_ids = cached_data['eval_ids']
        except Exception as e:
            print(f'Failed to load cache: {e}, rebuilding data...')
    else:
        print(f'No cache found for {dataset} dataset, building data...')
    
    # 如果没有缓存或加载失败，重新构建数据
    if not os.path.exists(cache_file) or 'all_path_data' not in locals():
        data_file = dataset_files[dataset]
        eval_sessions = pd.read_pickle(os.path.join(data_directory, data_file))
        eval_ids = eval_sessions.route_id.unique()
        
        print(f'Start batch evaluating {dataset.upper()} dataset...')
        print(f'Dataset size: {len(eval_ids)} paths, Batch size: {batch_size}')
        
        # 预计算所有路径数据
        all_path_data = []
        all_path_actions = []
        all_path_len_states = []
        all_path_lengths = []
        
        for route_id in eval_ids:
            group = eval_sessions[eval_sessions['route_id'] == route_id]
            if len(group) == 0:
                continue
            
            # 构建路径数据
            path_states, path_actions, path_len_states, path_length = build_path_data_for_batch(group)
            all_path_data.extend(path_states)
            all_path_actions.extend(path_actions)
            all_path_len_states.extend(path_len_states)
            all_path_lengths.append(path_length)
        
        print(f'Precomputed {len(all_path_data)} states')
        
        # 保存缓存数据
        try:
            cached_data = {
                'path_data': all_path_data,
                'path_actions': all_path_actions,
                'path_len_states': all_path_len_states,
                'path_lengths': all_path_lengths,
                'eval_ids': eval_ids
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
        for idx in sampled_indices:
            s, e = path_start_ends[idx]
            sampled_steps.extend(range(s, e))
            sampled_actions.extend(all_path_actions[s:e])
            sampled_len_states.extend(all_path_len_states[s:e])
        all_path_data = [all_path_data[i] for i in sampled_steps]
        all_path_actions = sampled_actions
        all_path_len_states = sampled_len_states
        all_path_lengths = sampled_path_lengths
        print(f"Sampled {sample_size} paths for train evaluation, total {len(all_path_data)} steps.")
    
    # 批量预测
    all_predictions = []
    # 新增：RL head预测
    all_rl_predictions = []
    
    # 新增：有效动作Q-values监控
    all_valid_q_values = []
    all_valid_q_counts = []

    for i in range(0, len(all_path_data), batch_size):
        batch_states = all_path_data[i:i+batch_size]
        batch_len_states = all_path_len_states[i:i+batch_size]
        batch_masks = generate_action_mask_batch(batch_states, G, edge_id_map, item_num, debug_output=True)

        # 为评估创建虚拟的potential_labels（评估时不需要真实标签）
        dummy_potential_labels = np.zeros((len(batch_states), 1), dtype=np.float32)
        
        # SL head预测 (softmax概率)
        batch_preds = sess.run(model.probs, feed_dict={
            model.inputs: batch_states,
            model.len_state: batch_len_states,
            model.action_mask: batch_masks,
            model.is_training: False,
            model.training_phase: training_phase,
            model.rl_weight: rl_weight,
            # 🔥 SL先验候选集剪枝：评估时也启用
            model.use_sl_prior: False,
            model.sl_prior_top_p: 0.80,
            model.prev_action: np.zeros(len(batch_states), dtype=np.int32),  # 评估时无prev_action
            model.action_history: np.zeros((len(batch_states), 3), dtype=np.int32),  # 评估时无action_history
        })
        # RL head预测 (Q值)
        batch_rl_q = sess.run(model.output1_masked, feed_dict={
            model.inputs: batch_states,
            model.len_state: batch_len_states,
            model.action_mask: batch_masks,
            model.is_training: False,
            model.training_phase: training_phase,
            model.rl_weight: rl_weight,
            # 🔥 SL先验候选集剪枝：评估时也启用
            model.use_sl_prior: False,
            model.sl_prior_top_p: 0.80,
            model.prev_action: np.zeros(len(batch_states), dtype=np.int32),  # 评估时无prev_action
            model.action_history: np.zeros((len(batch_states), 3), dtype=np.int32),  # 评估时无action_history
        })

        # 新增：监控有效动作的Q-values
        for j in range(len(batch_rl_q)):
            valid_mask = batch_masks[j] > 0
            valid_q_values = batch_rl_q[j][valid_mask]
            if len(valid_q_values) > 0:
                all_valid_q_values.extend(valid_q_values)
                all_valid_q_counts.append(len(valid_q_values))

        # 获取top1预测
        batch_top1 = np.argmax(batch_preds, axis=1)
        batch_rl_top1 = np.argmax(batch_rl_q, axis=1)
        all_predictions.extend(batch_top1)
        all_rl_predictions.extend(batch_rl_top1)

    # 计算SL head指标
    total_steps = len(all_predictions)
    total_correct_steps = sum(1 for pred, target in zip(all_predictions, all_path_actions) if pred == target)
    avg_step_correct = total_correct_steps / total_steps if total_steps > 0 else 0
    # 计算RL head指标
    total_correct_steps_rl = sum(1 for pred, target in zip(all_rl_predictions, all_path_actions) if pred == target)
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

        path_correct = sum(1 for pred, target in zip(path_predictions, path_targets) if pred == target)
        path_correctness = path_correct / path_length if path_length > 0 else 0
        path_correctness_list.append(path_correctness)
        if path_correct == path_length:
            successful_paths += 1

        # RL head
        path_correct_rl = sum(1 for pred, target in zip(path_rl_predictions, path_targets) if pred == target)
        path_correctness_rl = path_correct_rl / path_length if path_length > 0 else 0
        path_correctness_list_rl.append(path_correctness_rl)
        if path_correct_rl == path_length:
            successful_paths_rl += 1

        # 计算路径的reward（仍用SL head结果）
        for step_idx, (pred, target) in enumerate(zip(path_predictions, path_targets)):
            # 🔥 修复：is_done应该基于是否真正到达目标，而不是步数位置
            # 这里简化处理：假设最后一步且预测正确才算到达
            is_done = (step_idx == len(path_predictions) - 1) and (pred == target)
            reward, _ = calculate_improved_reward(
                pred, target, is_done,
                reward_goal,
                step_idx, None, True, {}
            )
            total_reward += reward
        path_idx += path_length

    total_paths = len(all_path_lengths)
    avg_path_correctness = np.mean(path_correctness_list) if path_correctness_list else 0
    avg_path_correctness_rl = np.mean(path_correctness_list_rl) if path_correctness_list_rl else 0
    path_success_rate = successful_paths / total_paths if total_paths > 0 else 0
    path_success_rate_rl = successful_paths_rl / total_paths if total_paths > 0 else 0
    avg_path_length = np.mean(all_path_lengths) if all_path_lengths else 0

    # 计算评估耗时
    eval_time = time.time() - eval_start_time
    current_time = time.strftime("%H:%M:%S", time.localtime())

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

    # 打印结果
    print(f'\n{dataset.upper()} DATASET EVALUATION RESULTS')
    print(f'Total Steps: {total_steps}, Total Paths: {total_paths}')
    print(f'SL Head - Step-level Accuracy: {avg_step_correct:.4f}')
    print(f'SL Head - Path-level Accuracy: {avg_path_correctness:.4f}')
    print(f'SL Head - Path Success Rate: {path_success_rate:.4f} ({successful_paths}/{total_paths})')
    # 🔥 删除RL模仿指标：RL不评估模仿，只评估可达性
    print(f'Average Path Length: {avg_path_length:.2f} steps')
    # 🔥 移除误导性的总奖励：评估阶段不需要，RL只关注可达性
    
    # 🔥 新增：RL专用评判标准
    print(f'\n🎯 RL-Specific Evaluation Metrics:')
    
    # 🔥 使用RL真滚动计算指标
    print(f'   🔄 Performing RL rollouts...')
    rl_rollout_results = []
    budget_steps = 50  # 预算步数（减少以加速）
    
    # 对部分路径进行RL滚动（采样以加速）
    sample_size = min(rl_rollout_sample_size, len(all_path_lengths))  # 可配置的采样数量
    sampled_indices = np.random.choice(len(all_path_lengths), sample_size, replace=False)
    
    path_start_idx = 0
    for i, path_length in enumerate(all_path_lengths):
        if i in sampled_indices:
            if len(rl_rollout_results) % 20 == 0:
                print(f'   Rolling out path {len(rl_rollout_results)+1}/{sample_size}...')
            
            # 获取起始状态和目标位置
            start_state = all_path_data[path_start_idx]
            target_pos = extract_target_position_from_state(start_state)
            
            # 执行RL滚动
            rollout_result = rollout_rl_trajectory(sess, model, start_state, target_pos, budget_steps, G, edge_id_map, item_num, reward_goal)
            rl_rollout_results.append(rollout_result)
        
        path_start_idx += path_length
    
    # 1. 🔥 Reach@预算：基于真滚动结果
    successful_rollouts = [r for r in rl_rollout_results if r['success']]
    reach_at_budget = len(successful_rollouts) / len(rl_rollout_results) if rl_rollout_results else 0.0
    print(f'   Reach@Budget (≤{budget_steps}): {reach_at_budget:.4f} (严格在预算内到达的比例)')
    
    # 2. 🔥 失败原因分解：基于真滚动结果
    failure_counts = {'timeout': 0, 'loop_stuck': 0, 'out_of_bounds': 0, 'invalid_action': 0, 'other': 0}
    for result in rl_rollout_results:
        if not result['success']:
            failure_reason = result['failure_reason']
            if failure_reason in failure_counts:
                failure_counts[failure_reason] += 1
            else:
                failure_counts['other'] += 1
    
    print(f'   📊 Failure Analysis:')
    for reason, count in failure_counts.items():
        percentage = count / len(rl_rollout_results) * 100 if rl_rollout_results else 0
        print(f'      {reason}: {count} ({percentage:.1f}%)')
    
    # 3. Loop/Stuck率：基于真滚动结果
    loop_stuck_rate = calculate_loop_stuck_rate_from_rollouts(rl_rollout_results)
    print(f'   Loop/Stuck Rate: {loop_stuck_rate:.4f} (循环/卡住的比例)')
    
    # 4. 无效动作率：基于真滚动结果
    invalid_action_count = failure_counts['invalid_action']
    invalid_action_rate = invalid_action_count / len(rl_rollout_results) if rl_rollout_results else 0.0
    print(f'   Invalid Action Rate: {invalid_action_rate:.4f} (无效动作的比例)')
    
    # 5. 成功步数中位数：基于真滚动结果（仅用于诊断）
    success_steps = [r['steps'] for r in successful_rollouts]
    success_steps_median = np.median(success_steps) if success_steps else 0.0
    print(f'   Success Steps Median: {success_steps_median:.1f} (成功路径步数中位数)')
    
    # 6. 🔥 综合RL评分（专注可达性）
    rl_score = calculate_rl_comprehensive_score(reach_at_budget, loop_stuck_rate, invalid_action_rate, success_steps_median)
    print(f'   🔥 RL Comprehensive Score: {rl_score:.4f}/1.0 (专注可达性)')
    
    # 🔥 新增：奖励统计和分析
    all_rewards = []
    all_total_rewards = []
    reward_stats_by_failure = {}
    
    for result in rl_rollout_results:
        if 'rewards' in result and 'reward_stats' in result:
            all_rewards.extend(result['rewards'])
            all_total_rewards.append(result['reward_stats']['total_reward'])
            
            failure_reason = result['failure_reason']
            if failure_reason not in reward_stats_by_failure:
                reward_stats_by_failure[failure_reason] = []
            reward_stats_by_failure[failure_reason].append(result['reward_stats'])
    
    # 计算整体奖励统计
    if all_total_rewards:
        overall_reward_stats = {
            'total_rewards_mean': np.mean(all_total_rewards),
            'total_rewards_std': np.std(all_total_rewards),
            'total_rewards_min': np.min(all_total_rewards),
            'total_rewards_max': np.max(all_total_rewards),
            'positive_total_rewards': sum(1 for r in all_total_rewards if r > 0),
            'negative_total_rewards': sum(1 for r in all_total_rewards if r < 0),
            'zero_total_rewards': sum(1 for r in all_total_rewards if r == 0)
        }
        
        print(f'   📊 Reward Statistics:')
        print(f'      Total Rewards: mean={overall_reward_stats["total_rewards_mean"]:.4f}, std={overall_reward_stats["total_rewards_std"]:.4f}')
        print(f'      Total Rewards Range: [{overall_reward_stats["total_rewards_min"]:.4f}, {overall_reward_stats["total_rewards_max"]:.4f}]')
        print(f'      Positive/Negative/Zero: {overall_reward_stats["positive_total_rewards"]}/{overall_reward_stats["negative_total_rewards"]}/{overall_reward_stats["zero_total_rewards"]}')
        
        # 按失败原因分析奖励
        print(f'   📊 Reward by Failure Reason:')
        for reason, stats_list in reward_stats_by_failure.items():
            if stats_list:
                avg_total = np.mean([s['total_reward'] for s in stats_list])
                avg_avg = np.mean([s['avg_reward'] for s in stats_list])
                print(f'      {reason}: avg_total={avg_total:.4f}, avg_step={avg_avg:.4f} (n={len(stats_list)})')
    else:
        overall_reward_stats = {}
        print(f'   📊 Reward Statistics: No reward data available')
    
    # 新增：打印有效动作Q-values统计
    print(f'📊 Valid Action Q-Values: mean={valid_q_stats["mean"]:.4f}, std={valid_q_stats["std"]:.4f}')
    print(f'   Range=[{valid_q_stats["min"]:.4f}, {valid_q_stats["max"]:.4f}], span={valid_q_stats["range"]:.4f}')
    print(f'   Total valid Q-values: {valid_q_stats["count"]}, Avg valid actions per step: {valid_q_stats["avg_valid_actions_per_step"]:.2f}')
    
    print(f'Evaluation Time: {eval_time:.2f} seconds, Current Time: {current_time}')  # 添加耗时信息和当前时间

    eval_metrics = {
        'dataset': dataset,
        'total_steps': int(total_steps),
        'total_correct_steps': int(total_correct_steps),
        'total_paths': int(total_paths),
        'step_level_accuracy': float(avg_step_correct),
        'path_level_accuracy': float(avg_path_correctness),
        'path_success_rate': float(path_success_rate),
        'rl_step_level_accuracy': float(avg_step_correct_rl),
        'rl_path_level_accuracy': float(avg_path_correctness_rl),
        'rl_path_success_rate': float(path_success_rate_rl),
        'avg_path_length': float(avg_path_length),
        'total_cumulative_reward': float(total_reward),
        'evaluation_time': float(eval_time),
        # 新增：有效动作Q-values统计
        'valid_q_mean': valid_q_stats['mean'],
        'valid_q_std': valid_q_stats['std'],
        'valid_q_min': valid_q_stats['min'],
        'valid_q_max': valid_q_stats['max'],
        'valid_q_range': valid_q_stats['range'],
        'valid_q_count': valid_q_stats['count'],
        'avg_valid_actions_per_step': valid_q_stats['avg_valid_actions_per_step'],
        # 🔥 新增：RL rollout奖励统计
        'rl_reach_at_budget': float(reach_at_budget),
        'rl_loop_stuck_rate': float(loop_stuck_rate),
        'rl_invalid_action_rate': float(invalid_action_rate),
        'rl_success_steps_median': float(success_steps_median),
        'rl_comprehensive_score': float(rl_score),
        'rl_rollout_count': len(rl_rollout_results)
    }
    
    # 添加奖励统计到eval_metrics
    if overall_reward_stats:
        eval_metrics.update({
            'reward_total_mean': overall_reward_stats['total_rewards_mean'],
            'reward_total_std': overall_reward_stats['total_rewards_std'],
            'reward_total_min': overall_reward_stats['total_rewards_min'],
            'reward_total_max': overall_reward_stats['total_rewards_max'],
            'reward_positive_count': overall_reward_stats['positive_total_rewards'],
            'reward_negative_count': overall_reward_stats['negative_total_rewards'],
            'reward_zero_count': overall_reward_stats['zero_total_rewards']
        })
    
    # 🔥 新增：保存详细的奖励数据到文件
    if step is not None:
        import json
        import os
        
        # 创建奖励数据目录
        reward_dir = '../data/training_logs'
        os.makedirs(reward_dir, exist_ok=True)
        
        # 准备详细的奖励数据
        detailed_reward_data = {
            'step': step,
            'dataset': dataset,
            'timestamp': current_time,
            'overall_stats': overall_reward_stats,
            'reward_stats_by_failure': reward_stats_by_failure,
            'rl_rollout_results': [
                {
                    'success': r['success'],
                    'steps': r['steps'],
                    'failure_reason': r['failure_reason'],
                    'total_reward': r['reward_stats']['total_reward'],
                    'avg_reward': r['reward_stats']['avg_reward'],
                    'max_reward': r['reward_stats']['max_reward'],
                    'min_reward': r['reward_stats']['min_reward'],
                    'positive_rewards': r['reward_stats']['positive_rewards'],
                    'negative_rewards': r['reward_stats']['negative_rewards'],
                    'zero_rewards': r['reward_stats']['zero_rewards']
                } for r in rl_rollout_results if 'reward_stats' in r
            ]
        }
        
        # 保存到文件
        reward_filename = f'reward_rollout_analysis_{dataset}_{step}.json'
        reward_filepath = os.path.join(reward_dir, reward_filename)
        
        try:
            with open(reward_filepath, 'w') as f:
                json.dump(detailed_reward_data, f, indent=2)
            print(f'   💾 Reward analysis saved to: {reward_filepath}')
        except Exception as e:
            print(f'   ❌ Failed to save reward analysis: {e}')

    if logger is not None and step is not None:
        logger.log_evaluation(step, eval_metrics)

    return eval_metrics

def build_path_data_for_batch(group):
    """
    为批量评估构建路径数据
    """
    path_states = []
    path_actions = []
    path_len_states = []
    history = []
    initial_state = None
    
    for index, row in group.iterrows():
        if len(history) == 0:
            # 创建初始状态
            start_x, start_y = row['feature_vec'][-6], row['feature_vec'][-5]
            end_x, end_y = row['feature_vec'][-4], row['feature_vec'][-3]
            cur_x, cur_y = start_x, start_y
            
            initial_state = [0.0] * 9 + [start_x, start_y, end_x, end_y, cur_x, cur_y]
            state_history = [initial_state]
            actual_history_length = 1
        else:
            state_history = [initial_state] + list(history)
            actual_history_length = min(len(history) + 1, state_size)
        
        # 创建状态
        state_for_pad = state_history.copy()
        state = pad_history(state_for_pad, state_size, np.zeros_like(row['feature_vec']))
        
        path_states.append(state)
        path_len_states.append(actual_history_length)
        
        # 获取动作
        action = get_edge_id(row['feature_vec'], edge_id_map)
        action_id = action if action >= 0 else 0
        path_actions.append(action_id)
        
        history.append(row['feature_vec'])
    
    return path_states, path_actions, path_len_states, len(path_actions)

def extract_position_from_state_eval(state):
    """从状态中提取当前位置 - 评估版本"""
    if isinstance(state, list) and len(state) > 0:
        for i in range(len(state) - 1, -1, -1):
            if isinstance(state[i], list) and len(state[i]) >= 2:
                if not all(x == 0 for x in state[i]):
                    return (float(state[i][-2]), float(state[i][-1]))
    elif hasattr(state, 'shape') and len(state.shape) > 1:
        for i in range(state.shape[0] - 1, -1, -1):
            if not np.all(state[i] == 0):
                if len(state[i]) >= 2:
                    return (float(state[i][-2]), float(state[i][-1]))
                break
    return (0.0, 0.0)

def preload_all_datasets():
    """
    预加载所有数据集并缓存，用于提升后续评估性能
    """
    print("Preloading all datasets for caching...")
    
    datasets = ['train', 'val', 'test']
    cache_dir = os.path.join(data_directory, 'evaluation_cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    for dataset in datasets:
        print(f"\nProcessing {dataset} dataset...")
        
        # 检查是否已有缓存
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
        eval_sessions = pd.read_pickle(os.path.join(data_directory, data_file))
        eval_ids = eval_sessions.route_id.unique()
        
        print(f'Building data for {len(eval_ids)} paths...')
        
        all_path_data = []
        all_path_actions = []
        all_path_len_states = []
        all_path_lengths = []
        
        for route_id in eval_ids:
            group = eval_sessions[eval_sessions['route_id'] == route_id]
            if len(group) == 0:
                continue
            
            path_states, path_actions, path_len_states, path_length = build_path_data_for_batch(group)
            all_path_data.extend(path_states)
            all_path_actions.extend(path_actions)
            all_path_len_states.extend(path_len_states)
            all_path_lengths.append(path_length)
        
        # 保存缓存
        cached_data = {
            'path_data': all_path_data,
            'path_actions': all_path_actions,
            'path_len_states': all_path_len_states,
            'path_lengths': all_path_lengths,
            'eval_ids': eval_ids
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cached_data, f)
        
        print(f'{dataset} dataset cached: {len(all_path_data)} states')
    
    print(f"\nAll datasets preloaded")
    print(f"Cache directory: {cache_dir}")

def build_candidate_mask(sl_probs, action_mask, degree, top_p=0.80, prev_action=None, action_history=None):
    """
    🔥 SL先验候选集剪枝 - 低度图专用版本
    
    Args:
        sl_probs: SL head输出的概率分布 [B, A]
        action_mask: 动作掩码 [B, A] (1=合法, 0=非法)
        degree: 度数 (2或3)
        top_p: Top-p兜底阈值 (默认0.80)
    
    Returns:
        candidate_mask: 候选动作掩码 [B, A] (1=候选, 0=非候选)
    """
    # 1) detach + 归一到合法集合
    action_mask_bool = tf.cast(action_mask, tf.bool)  # 转换为bool类型
    p_masked = tf.where(action_mask_bool, tf.stop_gradient(sl_probs), 0.0)
    p = p_masked / (tf.reduce_sum(p_masked, axis=1, keepdims=True) + 1e-8)

    # 2) 取 p_(1), p_(2), p_(3) 与熵
    vals3, idx3 = tf.math.top_k(p, k=3)       # [B,3], [B,3]
    p1, p2, p3 = vals3[:,0], vals3[:,1], vals3[:,2]
    gap = p1 - p2
    
    # H over valid set: 先把 0 位置换成极小再算
    p_safe = tf.clip_by_value(p, 1e-8, 1.0)
    H = -tf.reduce_sum(p_safe * tf.math.log(p_safe), axis=1) / tf.math.log(tf.cast(degree, tf.float32))

    # 3) 决定K（矢量化可分 d=2/3 两路判定）
    # degree是[B]形状的tensor，需要使用tf.where进行条件判断
    degree_is_2 = tf.equal(degree, 2)  # [B] bool
    
    # 🔥 更激进的剪枝策略：高确信度时只保留1个选择
    # 对于degree=2的情况
    K_for_2 = tf.where((p1 >= 0.8) & (gap >= 0.4), 1,  # 更高阈值：0.8概率+0.4差距
                       tf.where((p1 >= 0.6) & (gap >= 0.25), 2,  # 中等确信度保留2个
                               2))  # 默认保留2个
    
    # 对于degree=3的情况  
    K_for_3 = tf.where((p1 >= 0.75) & (gap >= 0.35), 1,  # 高确信度：0.75概率+0.35差距
                       tf.where((p1 >= 0.5) & (gap >= 0.2), 2,  # 中等确信度保留2个
                                tf.where(H >= 0.9, 3, 2)))  # 高熵时保留3个
    
    # 根据degree选择对应的K值
    K = tf.where(degree_is_2, K_for_2, K_for_3)

    # 4) top-K mask（按每行的K）
    #   简洁做法：始终取 top-3，再依据 K 生成前缀
    K_prefix = tf.sequence_mask(K, maxlen=3)    # [B,3] True 表示取前缀
    
    # 过滤有效索引，避免-1值
    valid_k = tf.where(K_prefix, idx3, 0)  # 用0替代-1
    k_mask = tf.sequence_mask(tf.cast(K, tf.int32), maxlen=tf.shape(idx3)[1])
    valid_k = tf.where(k_mask, valid_k, -1)  # 重新设置无效位置为-1
    
    # 使用scatter_nd生成掩码，避免one_hot的-1问题
    batch_size = tf.shape(action_mask)[0]
    action_dim = tf.shape(action_mask)[1]
    
    # 创建行索引
    row_indices = tf.reshape(tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, 3]), [-1])
    # 创建列索引（过滤-1）
    col_indices = tf.reshape(valid_k, [-1])
    # 创建有效位置掩码
    valid_positions = tf.not_equal(col_indices, -1)
    row_indices = tf.boolean_mask(row_indices, valid_positions)
    col_indices = tf.boolean_mask(col_indices, valid_positions)
    
    # 生成topk_mask
    indices = tf.stack([row_indices, col_indices], axis=1)
    updates = tf.ones(tf.shape(row_indices)[0], dtype=tf.bool)
    topk_mask = tf.scatter_nd(indices, updates, [batch_size, action_dim])

    # 5) Top-p 兜底
    cumsum = tf.cumsum(vals3, axis=1)
    first_true = tf.argmax(tf.cast(cumsum >= top_p, tf.int32), axis=1)
    p_prefix = tf.sequence_mask(first_true + 1, maxlen=3)
    
    # 同样处理p_idx
    valid_p = tf.where(p_prefix, idx3, -1)
    valid_positions_p = tf.not_equal(valid_p, -1)
    row_indices_p = tf.reshape(tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, 3]), [-1])
    col_indices_p = tf.reshape(valid_p, [-1])
    valid_positions_p = tf.reshape(valid_positions_p, [-1])
    row_indices_p = tf.boolean_mask(row_indices_p, valid_positions_p)
    col_indices_p = tf.boolean_mask(col_indices_p, valid_positions_p)
    
    # 生成topp_mask
    indices_p = tf.stack([row_indices_p, col_indices_p], axis=1)
    updates_p = tf.ones(tf.shape(row_indices_p)[0], dtype=tf.bool)
    topp_mask = tf.scatter_nd(indices_p, updates_p, [batch_size, action_dim])

    # 6) 合并并与 action_mask 相交
    cand = tf.logical_and(action_mask_bool, tf.logical_or(topk_mask, topp_mask))
    
    # 7) 🔥 强化No Backtrack约束：禁止回退和反复横跳
    batch_size = tf.shape(action_mask)[0]
    action_dim = tf.shape(action_mask)[1]
    
    # 检查prev_action是否非零（0表示没有previous action）
    has_prev_action = tf.not_equal(prev_action, 0)  # [B] bool
    
    # 将prev_action转换为one-hot掩码
    prev_action_onehot = tf.cast(tf.one_hot(prev_action, depth=action_dim), tf.bool)
    # 取反：True表示"不是回退动作"
    no_backtrack_mask = tf.logical_not(prev_action_onehot)
    
    # 🔥 额外约束：如果SL确信度很高，进一步限制选择
    # 当p1很高且gap很大时，强制选择top-1，避免犹豫
    high_confidence = tf.logical_and(p1 >= 0.8, gap >= 0.4)  # [B] bool
    high_confidence_expanded = tf.expand_dims(high_confidence, axis=1)  # [B, 1]
    
    # 高确信度时，只保留top-1动作
    top1_mask = tf.sequence_mask(tf.ones(batch_size, dtype=tf.int32), maxlen=action_dim)
    top1_indices = tf.one_hot(idx3[:, 0], depth=action_dim, dtype=tf.bool, on_value=True, off_value=False)
    
    # 应用约束：高确信度时强制top-1，否则应用no backtrack
    cand_high_conf = tf.logical_and(cand, top1_indices)
    cand_with_constraint = tf.logical_and(cand, no_backtrack_mask)
    
    # 🔥 反复横跳检测：如果最近3步有重复模式，进一步限制选择
    oscillation_penalty = tf.zeros_like(cand, dtype=tf.bool)
    if action_history is not None:
        # 检测A->B->A模式（反复横跳）
        # action_history shape: [B, 3] 表示最近3步的动作
        # [step_0, step_1, step_2] = [3步前, 2步前, 1步前]
        step_0 = action_history[:, 0]  # 3步前
        step_1 = action_history[:, 1]  # 2步前  
        step_2 = action_history[:, 2]  # 1步前
        
        # 检测A->B->A模式：step_0 == step_2 且 step_0 != step_1
        # 即：3步前和1步前相同，但2步前不同
        oscillation_pattern = tf.logical_and(
            tf.equal(step_0, step_2),  # 3步前和1步前相同
            tf.not_equal(step_0, step_1)  # 但2步前不同
        )  # [B] bool
        
        # 如果检测到反复横跳，禁止选择3步前的动作
        # 因为A->B->A模式中，3步前是A，选择A会导致A->B->A->A的循环
        oscillation_expanded = tf.expand_dims(oscillation_pattern, axis=1)  # [B, 1]
        step_0_onehot = tf.cast(tf.one_hot(step_0, depth=action_dim), tf.bool)
        oscillation_penalty = tf.logical_and(oscillation_expanded, step_0_onehot)
    
    # 应用所有约束
    # 1. 高确信度时强制top-1
    # 2. 否则应用no backtrack约束（如果有prev_action）
    # 3. 最后应用反复横跳惩罚
    cand = tf.where(high_confidence_expanded, cand_high_conf, 
                   tf.where(tf.expand_dims(has_prev_action, axis=1), cand_with_constraint, cand))
    
    # 应用反复横跳惩罚（无论是否有prev_action都要应用）
    cand = tf.logical_and(cand, tf.logical_not(oscillation_penalty))
    
    # 8) 🔥 死胡同例外：度数为1时允许回退（避免卡死）
    # 如果当前候选集为空（所有动作都被过滤），则恢复原始action_mask
    cand_sum = tf.reduce_sum(tf.cast(cand, tf.float32), axis=1)  # [B]
    empty_cand = tf.equal(cand_sum, 0.0)  # [B] bool
    
    # 对于空候选集，使用原始action_mask
    # 需要将empty_cand扩展到[B, A]形状以匹配cand和action_mask_bool
    empty_cand_expanded = tf.expand_dims(empty_cand, axis=1)  # [B, 1]
    cand = tf.where(empty_cand_expanded, action_mask_bool, cand)
    
    return cand  # [B, A] bool

def generate_action_mask_batch(states, G, edge_id_map, item_num, debug_output=False):
    """
    批量生成动作掩码 - 优化版本，支持缓存和向量化操作
    """
    batch_size = len(states)
    masks = np.zeros((batch_size, item_num), dtype=np.float32)
    
    # 向量化处理：找到所有状态的最后一个非零索引
    last_non_zero_indices = []
    for i, state in enumerate(states):
        last_idx = -1
        for j in range(len(state)-1, -1, -1):
            if not np.all(state[j] == 0):
                last_idx = j
                break
        last_non_zero_indices.append(last_idx)
    
    # 批量处理每个状态
    for i, (state, last_idx) in enumerate(zip(states, last_non_zero_indices)):
        if last_idx == -1:
            masks[i, :] = 1  # 如果没有有效状态，允许所有动作
            continue
            
        # 从最后一个非零状态获取当前节点
        last_state = state[last_idx]
        cur_node = (float(last_state[-2]), float(last_state[-1]))
        
        # 检查当前节点是否在图中
        if cur_node in G:
            try:
                # 使用图的邻接列表快速获取有效边
                neighbors = list(G.neighbors(cur_node))
                for neighbor in neighbors:
                    # 尝试两个方向的边
                    edge_key1 = (cur_node, neighbor)
                    edge_key2 = (neighbor, cur_node)
                    
                    edge_id = edge_id_map.get(edge_key1, edge_id_map.get(edge_key2, -1))
                    if 0 <= edge_id < item_num:
                        masks[i, edge_id] = 1
                
                # 确保至少有一个动作有效
                if np.sum(masks[i]) == 0:
                    masks[i, :] = 1
            except Exception as e:
                if debug_output:
                    print(f"Error processing node {cur_node}: {e}")
                masks[i, :] = 1
        else:
            masks[i, :] = 1  # 如果当前节点不在图中，允许所有动作
    
    return masks

def build_soft_update_ops(main_vars, target_vars, tau=0.005):
    return [tf.compat.v1.assign(t, (1.0 - tau) * t + tau * m)
            for t, m in zip(target_vars, main_vars)]

def build_hard_update_ops(main_vars, target_vars):
    return [tf.compat.v1.assign(t, m) for t, m in zip(target_vars, main_vars)]

if __name__ == '__main__':
    args = parse_args()

    data_directory = args.data
    if not os.path.exists(data_directory):
        data_directory = os.path.join(os.path.dirname(__file__), '..', 'data')
        if not os.path.exists(data_directory):
            raise FileNotFoundError(f"Data directory not found: {args.data} or {data_directory}")
    
    data_statis = pd.read_pickle(os.path.join(data_directory, 'data_statis.df'))
    state_size = data_statis['state_size'][0]
    feature_dim = data_statis['feature_dim'][0]

    reward_goal = args.r_goal

    tf.compat.v1.reset_default_graph()

    data = load_map_data(data_directory)
    df, meta_data = data["df"], data["meta_data"]
    df_copy = deepcopy(df)

    G, edge_id_map, edge_feature_list = load_or_create_graph(df_copy, data_directory)

    item_num = G.number_of_edges()

    QN_1 = ImprovedQNetwork(name='QN_1', hidden_size=args.hidden_factor, learning_rate=args.lr, 
                           feature_dim=feature_dim, item_num=item_num, state_size=state_size,
                           weight=args.weight,
                           dropout_rate=args.dropout_rate,
                           num_heads=args.num_heads, num_blocks=args.num_blocks, lr_2=args.lr_2)
    QN_2 = ImprovedQNetwork(name='QN_2', hidden_size=args.hidden_factor, learning_rate=args.lr, 
                           feature_dim=feature_dim, item_num=item_num, state_size=state_size,
                           weight=args.weight,
                           dropout_rate=args.dropout_rate,
                           num_heads=args.num_heads, num_blocks=args.num_blocks, lr_2=args.lr_2)
    hard_update_ops = build_hard_update_ops(QN_1.train_vars, QN_2.train_vars)
    soft_update_ops = build_soft_update_ops(QN_1.train_vars, QN_2.train_vars, tau=0.005)

    replay_buffer = pd.read_pickle(os.path.join(data_directory, 'replay_buffer.df'))

    saver = tf.compat.v1.train.Saver()
    global_step = 0

    # Initialize training logger (修复logger未定义问题)
    logger = init_logger(os.path.join(data_directory, 'training_logs'))

    # Preload datasets if requested
    if args.preload_datasets:
        preload_all_datasets()

    # If resume_ckpt is provided, try to load and set global_step
    resume_ckpt = args.resume_ckpt
    if resume_ckpt is not None:
        # Handle path - if not absolute path, assume it's in saved_model directory
        if not os.path.isabs(resume_ckpt):
            resume_ckpt = os.path.join(data_directory, 'saved_model', resume_ckpt)
        
        # Remove .index if user copy-pasted the full file name
        if not resume_ckpt.endswith('.index'):
            resume_ckpt = resume_ckpt.replace('.index', '')
    
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(hard_update_ops)
        if resume_ckpt is not None and os.path.exists(resume_ckpt + ".index"):
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
        else:
            print("No checkpoint loaded, training from scratch.")

        num_rows=replay_buffer.shape[0]
        num_batches=int(num_rows/args.batch_size)
        
        # Initial evaluation
        print("\n" + "="*60)
        print("INITIAL EVALUATION (Before Training)")
        print("="*60)
        # 使用批量评估提升性能
        batch_evaluate_improved(sess, QN_1, dataset='train', logger=logger, step=global_step, training_phase=1, rl_weight=0.0, batch_size=args.batch_size, sample_ratio=0.1, rl_rollout_sample_size=50, G=G, edge_id_map=edge_id_map, item_num=item_num, reward_goal=reward_goal)
        print("="*60 + "\n")

        for i in range(args.epoch):
            for j in range(num_batches):
                batch = replay_buffer.sample(n=args.batch_size).to_dict()
                next_state = list(batch['next_state'].values())
                len_next_state = list(batch['len_next_state'].values())
                state = list(batch['state'].values())
                len_state = list(batch['len_state'].values())
                is_done = list(batch['is_done'].values())

                mainQN = QN_1
                target_QN = QN_2
                if global_step % 100 == 0:
                    sess.run(soft_update_ops)
                # 1. 确定训练阶段和参数
                current_phase, phase_name, rl_weight, current_lr = determine_training_phase(global_step, args)
                
                # SL先验候选集剪枝：Phase 2开始启用，前1k步warm-up
                use_sl_prior = (current_phase >= 2) and (global_step >= 1000)
                sl_prior_top_p = 0.80
                
                # 生成next state的action masks
                next_action_masks = generate_action_mask_batch(next_state, G, edge_id_map, item_num)
                
                # 计算next state的target Q值
                next_state_feed_dict = {
                    target_QN.inputs: next_state,
                    target_QN.len_state: len_next_state, 
                    target_QN.action_mask: next_action_masks,
                    target_QN.is_training: True,
                    target_QN.training_phase: current_phase,
                    target_QN.rl_weight: rl_weight,
                    target_QN.use_sl_prior: False,
                    target_QN.sl_prior_top_p: sl_prior_top_p,
                    target_QN.prev_action: np.zeros(len(next_state), dtype=np.int32),
                    target_QN.action_history: np.zeros((len(next_state), 3), dtype=np.int32),
                    mainQN.inputs: next_state,
                    mainQN.len_state: len_next_state,
                    mainQN.action_mask: next_action_masks,
                    mainQN.is_training: True,
                    mainQN.training_phase: current_phase,
                    mainQN.rl_weight: rl_weight,
                    mainQN.use_sl_prior: False,
                    mainQN.sl_prior_top_p: sl_prior_top_p,
                    mainQN.prev_action: np.zeros(len(next_state), dtype=np.int32),
                    mainQN.action_history: np.zeros((len(next_state), 3), dtype=np.int32)
                }
                target_Qs, target_Qs_selector = sess.run(
                    [target_QN.output1_for_training, mainQN.output1_for_training],
                    feed_dict=next_state_feed_dict
                )

                # 处理终止状态
                terminal_mask = np.array(is_done, dtype=bool)
                target_Qs[terminal_mask] = 0

                # 生成current state的action masks
                current_state_array = np.array(state)
                current_action_masks = generate_action_mask_batch(state, G, edge_id_map, item_num, debug_output=True)
                
                # 提取动作信息
                action = list(batch['action'].values())
                action_ids = extract_action_ids(action, edge_id_map)
                
                # 8. 记录mask统计信息
                if global_step % args.log_frequency == 0:
                    mask_mean = np.mean(current_action_masks)
                    mask_sum = np.sum(current_action_masks)
                    print(f"  🎭 Mask Stats: mean={mask_mean:.3f}, total_valid={mask_sum}, batch_size={len(state)}")

                batch_size = len(state)

                predictions = sess.run(
                    mainQN.probs,
                    feed_dict={
                        mainQN.inputs: state,
                        mainQN.len_state: len_state,
                        mainQN.action_mask: current_action_masks,
                        mainQN.is_training: False,
                        mainQN.training_phase: current_phase,
                        mainQN.rl_weight: rl_weight,
                        mainQN.use_sl_prior: use_sl_prior,
                        mainQN.sl_prior_top_p: sl_prior_top_p,
                        mainQN.prev_action: np.zeros(len(state), dtype=np.int32),  # 添加prev_action
                        mainQN.action_history: np.zeros((len(state), 3), dtype=np.int32)  # 添加action_history
                    }
                )
                top1_preds = np.argmax(predictions, axis=1)
                
                reward = []
                
                step_indices = list(batch.get('step_idx', {}).values()) if 'step_idx' in batch else [0] * batch_size
                state_histories = list(batch.get('state_history', {}).values()) if 'state_history' in batch else [None] * batch_size
                
                path_infos = []
                for k in range(len(is_done)):
                    path_info = {}
                    if k < len(state_histories) and state_histories[k] and len(state_histories[k]) > 0:
                        initial_state = state_histories[k][0]
                        if len(initial_state) >= 15:
                            start_x, start_y = initial_state[-6], initial_state[-5]
                            end_x, end_y = initial_state[-4], initial_state[-3]
                            path_info['start_pos'] = (start_x, start_y)
                            path_info['end_pos'] = (end_x, end_y)
                            dx, dy = end_x - start_x, end_y - start_y
                            target_length = max(1, int(np.sqrt(dx*dx + dy*dy) / 0.1))
                            path_info['target_length'] = target_length
                    path_infos.append(path_info)
                
                step_indices_array = np.array(step_indices[:batch_size] if len(step_indices) >= batch_size else [0] * batch_size)
                
                for k in range(batch_size):
                    current_step_idx = step_indices_array[k]
                    current_state_history = state_histories[k] if k < len(state_histories) else None
                    current_path_info = path_infos[k] if k < len(path_infos) else {}
                    
                    # 正样本reward使用真实动作
                    reward_value, _ = calculate_improved_reward(
                        action_ids[k], action_ids[k], is_done[k],
                        reward_goal,
                        current_step_idx, current_state_history, True, current_path_info
                    )
                    reward.append(reward_value)
                                        
                    sample_negative_rewards = []
                    current_mask = current_action_masks[k]
                
                discount = [args.discount] * len(action_ids)

                feed_dict = build_feed_dict(mainQN, state, len_state, target_Qs, reward, 
                                          discount, action_ids, target_Qs_selector, current_action_masks,
                                          current_phase, rl_weight, is_training=True, 
                                          use_sl_prior=use_sl_prior, sl_prior_top_p=sl_prior_top_p,
                                          prev_action=None, action_history=None)  # 训练时暂时不跟踪历史，但可以启用
                
                # 执行训练步骤
                loss_components = sess.run(mainQN.loss_components, feed_dict=feed_dict)
                
                # 分层梯度管理训练
                if current_phase == 1:
                    sess.run(mainQN.train_phase1, feed_dict=feed_dict)
                elif current_phase == 2:
                    sess.run(mainQN.train_phase2_joint, feed_dict=feed_dict)
                else:
                    sess.run(mainQN.train_phase2_joint, feed_dict=feed_dict)
                
                global_step += 1
                
                # 16. 记录训练日志
                if global_step % args.log_frequency == 0:
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
                    
                    # 🔧 新增：RL学习质量监控
                    rl_learning_quality = {
                        'q_value_stability': q_value_std / max(abs(q_value_mean), 0.1),  # Q值稳定性
                        'mask_effectiveness': mask_penalty_gap,  # Mask效果
                        'loss_consistency': total_loss  # Loss一致性
                    }
                    
                    # 🔧 新增：过拟合检测
                    overfitting_indicators = {
                        'q_value_explosion': q_value_range > 20.0,  # Q值范围过大
                        'loss_instability': total_loss > 1.0,  # Loss不稳定
                        'mask_ineffectiveness': mask_penalty_gap < 1.0  # Mask效果差
                    }
                    
                    # 🔧 新增：RL学习进度监控
                    if not hasattr(calculate_improved_reward, 'rl_progress_tracker'):
                        calculate_improved_reward.rl_progress_tracker = {
                            'best_rl_performance': 0.0,
                            'performance_plateau_count': 0,
                            'last_improvement_step': 0
                        }
                    
                    # 🔧 新增：Reward统计
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
                    
                    # 🔧 新增：增强的RL不稳定性检测
                    if current_phase > 1:  # Only check during RL phases
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
                            print(f"⚠️  RL INSTABILITY DETECTED:")
                            for warning in instability_warnings:
                                print(f"   - {warning}")
                            print("🛡️  Consider reducing learning rate or RL weight if instability persists")
                    
                    # 🔧 新增：RL学习质量评估
                    if current_phase > 1:
                        rl_quality_score = 0.0
                        quality_factors = []
                        
                        # Q值稳定性评分
                        if q_value_std < 5.0:
                            rl_quality_score += 0.3
                            quality_factors.append("Q-value stability: GOOD")
                        elif q_value_std < 10.0:
                            rl_quality_score += 0.2
                            quality_factors.append("Q-value stability: FAIR")
                        else:
                            quality_factors.append("Q-value stability: POOR")
                        
                        # Loss一致性评分
                        if total_loss < 0.3:
                            rl_quality_score += 0.4
                            quality_factors.append("Loss consistency: GOOD")
                        elif total_loss < 0.5:
                            rl_quality_score += 0.2
                            quality_factors.append("Loss consistency: FAIR")
                        else:
                            quality_factors.append("Loss consistency: POOR")
                        
                        print(f"📊 RL Learning Quality Score: {rl_quality_score:.2f}/1.0")
                        for factor in quality_factors:
                            print(f"   - {factor}")
                    
                    correct_predictions = np.sum(top1_preds == action_ids)
                    accuracy = correct_predictions / len(action_ids)
                    
                    avg_reward = np.mean(reward) if reward else 0.0
                    
                    # Log all loss components
                    logger.log_loss_components(global_step, loss_components, current_lr, phase_name)
                    logger.log_loss(global_step, total_loss, current_lr, phase_name)
                    logger.log_accuracy(global_step, accuracy, correct_predictions, len(action_ids), phase_name, avg_reward)
                    
                    # 🔧 新增：记录Reward统计到logger
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
                    
                    current_epoch = (global_step - 1) // num_batches + 1
                    batches_in_current_epoch = (global_step - 1) % num_batches + 1
                    
                    current_time = time.strftime("%H:%M:%S", time.localtime())
                    
                    # 基础训练信息
                    base_info = (f"Epoch {current_epoch}/{args.epoch} Batch {batches_in_current_epoch}/{num_batches} "
                               f"Step {global_step} [{phase_name}]: Loss={total_loss:.4f}, Acc={accuracy:.4f}, "
                               f"RL_weight={rl_weight:.3f}, Time: {current_time}")
                    
                    print(base_info)
                    
                    # 🔧 新增：打印Reward统计
                    if reward_stats:
                        print(f"  🎯 Reward Stats (All History):")
                        print(f"     All: count={reward_stats['all_rewards']['count']}, "
                              f"mean={reward_stats['all_rewards']['mean']:.4f}, "
                              f"std={reward_stats['all_rewards']['std']:.4f}, "
                              f"range=[{reward_stats['all_rewards']['min']:.4f}, {reward_stats['all_rewards']['max']:.4f}]")
                        print(f"     Positive: count={reward_stats['positive_rewards']['count']}, "
                              f"mean={reward_stats['positive_rewards']['mean']:.4f}, "
                              f"range=[{reward_stats['positive_rewards']['min']:.4f}, {reward_stats['positive_rewards']['max']:.4f}]")
                        print(f"     Negative: count={reward_stats['negative_rewards']['count']}, "
                              f"mean={reward_stats['negative_rewards']['mean']:.4f}, "
                              f"range=[{reward_stats['negative_rewards']['min']:.4f}, {reward_stats['negative_rewards']['max']:.4f}]")
                        
                        # 🔧 新增：调试信息
                        correct_predictions = np.sum(top1_preds == action_ids)
                        incorrect_predictions = len(action_ids) - correct_predictions
                        print(f"  🔍 Debug: correct_predictions={correct_predictions}, incorrect_predictions={incorrect_predictions}")
                        print(f"  🔍 Debug: accuracy={correct_predictions/len(action_ids):.4f}")

                
                # 三阶段训练的阶段切换提示
                if global_step == args.phase1_sl_only_steps:
                    print(f"\n🔄 PHASE TRANSITION: Phase1-SL-Only → Phase2-SL+RL (Step {global_step})")
                    print(f"   从纯监督学习切换到强化学习+监督学习")
                                
                if global_step % args.eval_frequency == 0:                  
                    print(f"\nEvaluating at step {global_step}...")
                    train_metrics = batch_evaluate_improved(sess, QN_1, dataset='train', logger=logger, step=global_step, 
                                                         training_phase=current_phase, rl_weight=rl_weight, batch_size=args.batch_size, rl_rollout_sample_size=50, G=G, edge_id_map=edge_id_map, item_num=item_num, reward_goal=reward_goal)
                    val_metrics = batch_evaluate_improved(sess, QN_1, dataset='val', logger=logger, step=global_step,
                                                       training_phase=current_phase, rl_weight=rl_weight, batch_size=args.batch_size, rl_rollout_sample_size=50, G=G, edge_id_map=edge_id_map, item_num=item_num, reward_goal=reward_goal)
        
        # Final evaluation
        print("\nTest set evaluation:")
        test_metrics = batch_evaluate_improved(sess, QN_1, dataset='test', logger=logger, step=global_step,
                                            training_phase=current_phase, rl_weight=rl_weight, batch_size=args.batch_size, rl_rollout_sample_size=50, G=G, edge_id_map=edge_id_map, item_num=item_num, reward_goal=reward_goal)
 
        # Save logs
        logger.save_all_logs()
        
        # Save final model
        final_checkpoint_path = os.path.join(data_directory, f'saved_model/llm4rec_improved_{time.strftime("%Y%m%d_%H%M%S")}.ckpt')
        saver.save(sess, final_checkpoint_path)
        print(f"\nFinal model saved: {final_checkpoint_path}") 

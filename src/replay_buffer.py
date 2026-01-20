"""
Replay Buffer V2: Cross-map generalization replay buffer.

This module constructs training transitions from node_id/edge_id sequences,
building cross-map invariant state representations using FeatureBuilder.

Key Changes from V1:
- Input: node_id/edge_id sequences (not coordinate feature_vec)
- State tokens: Built by FeatureBuilder (cross-map invariant)
- Unified E/O bucket schema with bc_action_idx
- No coordinate dependencies in state construction
"""

import os
import pandas as pd
import numpy as np
from utils.utility import to_pickled_df, pad_history
from utils.graph_cache import GraphCache, MultiGraphCache
from utils.feature_builder import FeatureBuilder
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('history_length', 10, 'uniform history length')
flags.DEFINE_integer('max_candidates', 20, 'maximum candidates per step for buffer storage')


def create_initial_state_from_node_ids(start_node_id: int, goal_node_id: int, graph_cache: GraphCache, feature_builder: FeatureBuilder):
    """
    Create initial state from start/goal node IDs.

    Uses FeatureBuilder to create cross-map invariant state representation.

    Args:
        start_node_id: Starting node ID
        goal_node_id: Goal node ID
        graph_cache: GraphCache instance
        feature_builder: FeatureBuilder instance

    Returns:
        Initial state feature vector (cross-map invariant)
    """
    # Build initial state features using FeatureBuilder
    # At start, current node = start node, no history
    initial_state = feature_builder.build_state_features(
        cur_node_id=start_node_id,
        goal_node_id=goal_node_id,
        history_edge_ids=[]
    )

    return initial_state


def build_transition_from_trajectory_v2(group: pd.DataFrame, graph_cache: GraphCache, feature_builder: FeatureBuilder,
                                       history_length: int, max_candidates: int):
    """
    Build transitions from node_id/edge_id trajectory.

    NEW SCHEMA: E/O bucket compatible transition format

    Args:
        group: Trajectory DataFrame with columns:
               [route_id, step_id, start_node_id, goal_node_id, cur_node_id, action_edge_id, next_node_id, dist_to_goal]
        graph_cache: GraphCache instance
        feature_builder: FeatureBuilder instance
        history_length: Maximum history length for state sequences
        max_candidates: Maximum candidates for buffer storage

    Returns:
        Dictionary with transition data
    """
    transitions = {
        'graph_id': [],        # Graph context for multi-map support
        'cur_node_id': [],
        'goal_node_id': [],
        'cand_edge_ids': [],  # Candidate edges (for E-bucket, can be reconstructed)
        'taken_edge_id': [],
        'bc_action_idx': [],  # Behavioral cloning action index (E-bucket only)
        'next_node_id': [],
        'reward': [],
        'done': [],
        'state': [],  # Cross-map invariant state features
        'next_state': [],
        'len_state': [],
        'len_next_state': [],
        'd_cur': [],   # Current distance to goal
        'd_next': [],  # Next distance to goal
        'delta_d': [], # Distance improvement
    }

    # Group by route_id to process each trajectory
    route_groups = group.groupby('route_id')

    for route_id, trajectory in route_groups:
        trajectory = trajectory.sort_values('step_id').reset_index(drop=True)

        # Initialize history tracking
        history_edge_ids = []
        current_node_id = trajectory.iloc[0]['start_node_id']

        # Get graph_id from trajectory (assume all rows in trajectory have same graph_id)
        graph_id = trajectory.iloc[0].get('graph_id', 'default_graph')

        for i, row in trajectory.iterrows():
            cur_node_id = row['cur_node_id']
            goal_node_id = row['goal_node_id']
            action_edge_id = row['action_edge_id']
            next_node_id = row['next_node_id']
            dist_to_goal = row['dist_to_goal']

            # Build current state (with history)
            state_features = feature_builder.build_state_features(
                cur_node_id=cur_node_id,
                goal_node_id=goal_node_id,
                history_edge_ids=history_edge_ids[-history_length:] if history_edge_ids else []
            )

            # Get candidate edges for current node (for E-bucket BC training)
            cand_edge_ids = graph_cache.get_candidate_edges(cur_node_id)

            # Find action index in candidates (for BC supervision)
            bc_action_idx = None
            if action_edge_id in cand_edge_ids:
                bc_action_idx = cand_edge_ids.index(action_edge_id)

            # Calculate next state
            next_history = history_edge_ids + [action_edge_id]
            next_state_features = feature_builder.build_state_features(
                cur_node_id=next_node_id,
                goal_node_id=goal_node_id,
                history_edge_ids=next_history[-history_length:]
            )

            # Calculate distances
            next_dist_to_goal = graph_cache.get_dist_to_goal(next_node_id, goal_node_id)
            delta_d = next_dist_to_goal - dist_to_goal

            # Calculate reward (simplified version)
            reward = -delta_d  # Distance improvement reward
            if i == len(trajectory) - 1:  # Last step
                done = True
                # Add goal reaching reward
                if next_dist_to_goal < 0.1:  # Close to goal
                    reward += 10.0
            else:
                done = False

            # Store transition
            transitions['graph_id'].append(graph_id)
            transitions['cur_node_id'].append(cur_node_id)
            transitions['goal_node_id'].append(goal_node_id)
            transitions['cand_edge_ids'].append(cand_edge_ids)  # Store actual candidates
            transitions['taken_edge_id'].append(action_edge_id)
            transitions['bc_action_idx'].append(bc_action_idx)
            transitions['next_node_id'].append(next_node_id)
            transitions['reward'].append(reward)
            transitions['done'].append(done)
            transitions['state'].append(state_features)
            transitions['next_state'].append(next_state_features)
            transitions['len_state'].append(min(len(history_edge_ids) + 1, history_length))
            transitions['len_next_state'].append(min(len(next_history) + 1, history_length))
            transitions['d_cur'].append(dist_to_goal)
            transitions['d_next'].append(next_dist_to_goal)
            transitions['delta_d'].append(delta_d)

            # Update history for next step
            history_edge_ids = next_history
            current_node_id = next_node_id

    return transitions


def main(argv):
    import argparse

    parser = argparse.ArgumentParser(description='Build replay buffer from trajectories')
    parser.add_argument('--data_dir', type=str, default='../data',
                       help='Data directory path (default: ../data)')
    parser.add_argument('--graph_id', type=str, default='default_graph',
                       help='Graph identifier for multi-graph support (default: default_graph)')

    args = parser.parse_args()
    data_directory = args.data_dir
    graph_id = args.graph_id

    length = FLAGS.history_length
    max_candidates = FLAGS.max_candidates

    # Load GraphCache and FeatureBuilder using MultiGraphCache
    multi_cache = MultiGraphCache(os.path.join(data_directory, 'graph_data'))

    try:
        graph_cache = multi_cache.get_cache(graph_id)
        feature_builder = multi_cache.get_feature_builder(graph_id)
        print(f"✅ Loaded GraphCache and FeatureBuilder for graph '{graph_id}'")
        print(f"   Nodes: {graph_cache.get_node_count()}, Edges: {graph_cache.get_edge_count()}")
        print(f"   State feature dim: {feature_builder._get_feature_dim()}")
    except FileNotFoundError:
        raise FileNotFoundError(f"GraphCache not found for graph '{graph_id}'. Run graph construction first.")

    # Load new format trajectory data from graph-specific directory
    train_sessions_path = os.path.join(data_directory, 'graph_data', graph_id, 'trajectories', 'sampled_train.df')
    if not os.path.exists(train_sessions_path):
        raise FileNotFoundError(f"Trajectory data not found: {train_sessions_path}. Run preprocess_v2.py first.")

    train_sessions = pd.read_pickle(train_sessions_path)
    print(f"✅ Loaded trajectory data: {train_sessions.shape}")

    # Build transitions
    print("🔄 Building transitions...")
    transitions = build_transition_from_trajectory_v2(
        train_sessions, graph_cache, feature_builder, length, max_candidates
    )

    # Convert to DataFrame
    replay_buffer = pd.DataFrame(transitions)

    print(f"✅ Built replay buffer with {len(replay_buffer)} transitions")
    print(f"   E-bucket transitions (with bc_action_idx): {replay_buffer['bc_action_idx'].notna().sum()}")
    print(f"   State shape: {replay_buffer['state'].iloc[0].shape if len(replay_buffer) > 0 else 'N/A'}")

    # Save replay buffer to graph-specific trajectories directory
    trajectories_dir = os.path.join(data_directory, 'graph_data', graph_id, 'trajectories')
    os.makedirs(trajectories_dir, exist_ok=True)
    to_pickled_df(trajectories_dir, replay_buffer=replay_buffer)

    # Save data statistics
    state_size = length
    feature_dim = feature_builder._get_feature_dim()  # State feature dimension
    data_statis = pd.DataFrame({
        'state_size': [state_size],
        'feature_dim': [feature_dim],
        'max_candidates': [max_candidates]
    })
    to_pickled_df(trajectories_dir, data_statis=data_statis)

    print(f"✅ Saved replay buffer to {os.path.join(trajectories_dir, 'replay_buffer.df')}")
    print(f"✅ Saved data statistics to {os.path.join(trajectories_dir, 'data_statis.df')}")


if __name__ == '__main__':
    app.run(main)
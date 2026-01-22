"""
FeatureBuilder: Cross-map invariant feature construction for path planning.

This module provides MVP (Minimum Viable Product) feature construction functions
that are invariant across different maps, enabling cross-map generalization.

Key Components:
- Candidate edge features: For action selection within candidate sets
- History tokens: For sequence modeling of past actions
- State features: For global state representation

All features avoid absolute coordinates and use only relative, topological, and
progress-related information.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from .graph_cache import GraphCache


class FeatureBuilder:
    """
    Builder for cross-map invariant features.

    Constructs features that are independent of absolute coordinates
    and enable generalization across different maps.
    """

    def __init__(self, graph_cache: GraphCache):
        """
        Initialize FeatureBuilder with GraphCache.

        Args:
            graph_cache: GraphCache instance with map data
        """
        self.graph_cache = graph_cache

    def build_candidate_features(self,
                                cand_edge_ids: List[int],
                                cur_node_id: int,
                                goal_node_id: int,
                                prev_node_id: Optional[int] = None,
                                recent_visited_nodes: Optional[List[int]] = None) -> np.ndarray:
        """
        Build candidate edge features matrix.

        MVP features (strictly enforced):
        - edge_attr: length_norm, width, curb_norm, crossing, path_type
        - progress: d_cur, d_next_i, delta_d_i, rank_delta_d_i
        - local: out_degree, backtrack_flag_i, loop_recent_flag_i

        Args:
            cand_edge_ids: List of candidate edge IDs
            cur_node_id: Current node ID
            goal_node_id: Goal node ID
            prev_node_id: Previous node ID (for backtrack detection)
            recent_visited_nodes: Recently visited node IDs (for loop detection)

        Returns:
            [K, Dc] feature matrix where K=len(cand_edge_ids)
        """
        if not cand_edge_ids:
            return np.array([]).reshape(0, self._get_feature_dim())

        K = len(cand_edge_ids)
        features = []

        # Pre-compute common values
        d_cur = self.graph_cache.get_dist_to_goal(cur_node_id, goal_node_id)
        out_degree = self.graph_cache.get_node_attr(cur_node_id).get('out_degree', 0)

        # Get distances to goal for all candidate target nodes
        cand_distances = []
        for edge_id in cand_edge_ids:
            u_id, v_id = self.graph_cache.get_edge_nodes(edge_id)
            d_next = self.graph_cache.get_dist_to_goal(v_id, goal_node_id)
            cand_distances.append(d_next)

        # Compute rank of delta_d among candidates
        delta_ds = [d - d_cur for d in cand_distances]
        ranks = np.argsort(np.argsort(delta_ds))  # rank from 0 to K-1

        for i, edge_id in enumerate(cand_edge_ids):
            # Get edge attributes
            edge_attr = self.graph_cache.get_edge_attr(edge_id)
            length_norm = edge_attr.get('length_norm', 0)
            width = edge_attr.get('width', 0)
            curb_norm = edge_attr.get('curb_norm', 0)
            crossing = edge_attr.get('crossing', 0)
            path_type = edge_attr.get('path_type', 0)

            # Progress features
            d_next = cand_distances[i]
            delta_d = delta_ds[i]
            rank_delta_d = ranks[i]

            # Local features
            backtrack_flag = 0
            if prev_node_id is not None:
                u_id, v_id = self.graph_cache.get_edge_nodes(edge_id)
                backtrack_flag = 1 if v_id == prev_node_id else 0

            loop_recent_flag = 0
            if recent_visited_nodes:
                u_id, v_id = self.graph_cache.get_edge_nodes(edge_id)
                loop_recent_flag = 1 if v_id in recent_visited_nodes else 0

            # Build feature vector
            feature_vec = [
                # Edge attributes
                length_norm, width, curb_norm, crossing, path_type,
                # Progress
                d_cur, d_next, delta_d, rank_delta_d,
                # Local
                out_degree, backtrack_flag, loop_recent_flag
            ]

            features.append(feature_vec)

        return np.array(features, dtype=np.float32)

    def build_history_token(self,
                           chosen_edge_id: int,
                           transition_context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Build history token for sequence modeling.

        MVP implementation: Use edge attributes as history token.
        Can be extended to include transition context.

        Args:
            chosen_edge_id: ID of the chosen edge
            transition_context: Optional context about the transition

        Returns:
            [Dh] history token vector
        """
        edge_attr = self.graph_cache.get_edge_attr(chosen_edge_id)

        # Basic implementation: use edge attributes
        token = [
            edge_attr.get('length_norm', 0),
            edge_attr.get('width', 0),
            edge_attr.get('curb_norm', 0),
            edge_attr.get('crossing', 0),
            edge_attr.get('path_type', 0)
        ]

        # Optional: add transition context (reward, success, etc.)
        if transition_context:
            # Could add normalized reward, success flag, etc.
            pass

        return np.array(token, dtype=np.float32)

    def build_state_features(self,
                           cur_node_id: int,
                           goal_node_id: int,
                           history_edge_ids: Optional[List[int]] = None) -> np.ndarray:
        """
        Build global state features for sequence encoder input.

        MVP implementation: Basic node and goal information.

        Args:
            cur_node_id: Current node ID
            goal_node_id: Goal node ID
            history_edge_ids: Recent edge IDs in history

        Returns:
            [Ds] state feature vector
        """
        # Node attributes
        node_attr = self.graph_cache.get_node_attr(cur_node_id)
        out_degree = node_attr.get('out_degree', 0)

        # Progress to goal
        dist_to_goal = self.graph_cache.get_dist_to_goal(cur_node_id, goal_node_id)

        # History statistics (optional)
        avg_edge_length = 0
        if history_edge_ids:
            edge_lengths = []
            for edge_id in history_edge_ids[-5:]:  # Last 5 edges
                edge_attr = self.graph_cache.get_edge_attr(edge_id)
                edge_lengths.append(edge_attr.get('length_norm', 0))
            if edge_lengths:
                avg_edge_length = np.mean(edge_lengths)

        state_features = [
            out_degree,
            dist_to_goal,
            avg_edge_length
        ]

        return np.array(state_features, dtype=np.float32)

    def _get_feature_dim(self) -> int:
        """
        Get candidate feature dimension.

        Hard-coded based on MVP feature specification.
        """
        # edge_attr: 5, progress: 4, local: 3 = 12 total
        return 5 + 4 + 3

    def pad_candidate_features(self,
                             features: np.ndarray,
                             max_candidates: int) -> np.ndarray:
        """
        Pad candidate features to fixed size.

        Args:
            features: [K, Dc] feature matrix
            max_candidates: Maximum number of candidates

        Returns:
            [max_candidates, Dc] padded matrix
        """
        if features.shape[0] >= max_candidates:
            return features[:max_candidates]

        # Pad with zeros
        padding = np.zeros((max_candidates - features.shape[0], features.shape[1]),
                          dtype=features.dtype)
        return np.vstack([features, padding])

    def create_candidate_mask(self,
                             num_candidates: int,
                             max_candidates: int) -> np.ndarray:
        """
        Create mask for valid candidates.

        Args:
            num_candidates: Number of actual candidates
            max_candidates: Maximum number of candidates

        Returns:
            [max_candidates] boolean mask
        """
        mask = np.zeros(max_candidates, dtype=bool)
        mask[:num_candidates] = True
        return mask


def build_candidate_features(cand_edge_ids: List[int],
                           cur_node_id: int,
                           goal_node_id: int,
                           graph_cache: GraphCache,
                           prev_node_id: Optional[int] = None,
                           recent_visited_nodes: Optional[List[int]] = None) -> np.ndarray:
    """
    Convenience function for building candidate features.

    Args:
        cand_edge_ids: List of candidate edge IDs
        cur_node_id: Current node ID
        goal_node_id: Goal node ID
        graph_cache: GraphCache instance
        prev_node_id: Previous node ID (for backtrack detection)
        recent_visited_nodes: Recently visited node IDs (for loop detection)

    Returns:
        [K, Dc] feature matrix
    """
    builder = FeatureBuilder(graph_cache)
    return builder.build_candidate_features(
        cand_edge_ids, cur_node_id, goal_node_id, prev_node_id, recent_visited_nodes
    )


def build_history_token(chosen_edge_id: int,
                       graph_cache: GraphCache,
                       transition_context: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Convenience function for building history tokens.

    Args:
        chosen_edge_id: ID of the chosen edge
        graph_cache: GraphCache instance
        transition_context: Optional transition context

    Returns:
        [Dh] history token vector
    """
    builder = FeatureBuilder(graph_cache)
    return builder.build_history_token(chosen_edge_id, transition_context)
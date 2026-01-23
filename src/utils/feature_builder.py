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
        # Path type embedding: 3 types → 3 dim (简单lookup)
        self.path_type_embeddings = np.array([
            [1.0, 0.0, 0.0],  # walk
            [0.0, 1.0, 0.0],  # bike
            [0.0, 0.0, 1.0],  # connection
        ], dtype=np.float32)

    def _embed_path_type(self, path_type: int) -> np.ndarray:
        """将 path_type ∈ {0,1,2} 映射到 3维 embedding"""
        return self.path_type_embeddings[int(path_type)]

    def build_candidate_features(self,
                                cand_edge_ids: List[int],
                                cur_node_id: int,
                                goal_node_id: int,
                                d_start: float,
                                prev_node_id: Optional[int] = None,
                                recent_visited_nodes: Optional[List[int]] = None) -> np.ndarray:
        """
        Build candidate features: [K, 10维]

        Args:
            d_start: Episode初始距离（用于归一化）
        """
        if not cand_edge_ids:
            return np.array([]).reshape(0, 10)

        features = []
        d_cur = self.graph_cache.get_dist_to_goal(cur_node_id, goal_node_id)

        # 获取坐标（用于方向计算）
        cur_coord = self.graph_cache.node_coords[cur_node_id]
        goal_coord = self.graph_cache.node_coords[goal_node_id]

        # 计算参考方向
        if prev_node_id is not None and not (isinstance(prev_node_id, float) and np.isnan(prev_node_id)):
            prev_coord = self.graph_cache.node_coords[prev_node_id]
            ref_vec = np.array([cur_coord[0] - prev_coord[0],
                               cur_coord[1] - prev_coord[1]])
        else:
            ref_vec = np.array([goal_coord[0] - cur_coord[0],
                               goal_coord[1] - cur_coord[1]])
        ref_norm = np.linalg.norm(ref_vec)

        for edge_id in cand_edge_ids:
            edge_attr = self.graph_cache.get_edge_attr(edge_id)
            u_id, v_id = self.graph_cache.get_edge_nodes(edge_id)
            next_coord = self.graph_cache.node_coords[v_id]

            # 边属性
            length_norm = edge_attr.get('length_norm', 0)
            width = edge_attr.get('width', 0)
            curb_norm = edge_attr.get('curb_norm', 0)
            crossing = float(edge_attr.get('crossing', 0))
            path_type = int(edge_attr.get('path_type', 0))
            path_type_emb = self._embed_path_type(path_type)

            # 相对方向
            edge_vec = np.array([next_coord[0] - cur_coord[0],
                                next_coord[1] - cur_coord[1]])
            edge_norm = np.linalg.norm(edge_vec)

            if ref_norm > 1e-6 and edge_norm > 1e-6:
                cos_rel = np.dot(ref_vec, edge_vec) / (ref_norm * edge_norm)
                sin_rel = (ref_vec[0] * edge_vec[1] - ref_vec[1] * edge_vec[0]) / (ref_norm * edge_norm)
            else:
                cos_rel, sin_rel = 0.0, 0.0

            # 进度
            d_next = self.graph_cache.get_dist_to_goal(v_id, goal_node_id)
            delta_frac = (d_next - d_cur) / d_start if d_start > 0 else 0.0

            # 约束
            backtrack_flag = 1.0 if (prev_node_id is not None and v_id == prev_node_id) else 0.0
            loop_flag = 1.0 if (recent_visited_nodes and v_id in recent_visited_nodes) else 0.0

            feature_vec = np.concatenate([
                [length_norm, width, curb_norm, crossing],
                path_type_emb,
                [cos_rel, sin_rel, delta_frac, backtrack_flag, loop_flag]
            ])
            features.append(feature_vec)

        return np.array(features, dtype=np.float32)

    def build_history_token(self,
                           chosen_edge_id: int,
                           prev_node_id: Optional[int] = None,
                           recent_visited_nodes: Optional[List[int]] = None) -> np.ndarray:
        """
        Build history token: [7维]

        包含：边属性(5) + 记忆标记(2)

        Args:
            chosen_edge_id: ID of the chosen edge
            prev_node_id: Previous node ID (for backtrack detection)
            recent_visited_nodes: Recently visited nodes (for loop detection)
        """
        edge_attr = self.graph_cache.get_edge_attr(chosen_edge_id)
        u_id, v_id = self.graph_cache.get_edge_nodes(chosen_edge_id)

        # 边属性
        length_norm = edge_attr.get('length_norm', 0)
        width = edge_attr.get('width', 0)
        curb_norm = edge_attr.get('curb_norm', 0)
        crossing = float(edge_attr.get('crossing', 0))
        path_type = int(edge_attr.get('path_type', 0))
        path_type_emb = self._embed_path_type(path_type)  # [3]

        # 记忆标记
        backtrack_flag = 0.0
        if prev_node_id is not None:
            backtrack_flag = 1.0 if v_id == prev_node_id else 0.0

        visited_recent_flag = 0.0
        if recent_visited_nodes:
            visited_recent_flag = 1.0 if v_id in recent_visited_nodes else 0.0

        token = np.concatenate([
            [length_norm, width, curb_norm, crossing],
            path_type_emb,
            [backtrack_flag, visited_recent_flag]
        ])

        return token.astype(np.float32)

    def build_state_features(self,
                            cur_node_id: int,
                            goal_node_id: int,
                            start_node_id: int,
                            d_start: float) -> np.ndarray:
        """
        Build global state: [2维]

        Args:
            d_start: Episode初始距离（用于归一化）
        """
        node_attr = self.graph_cache.get_node_attr(cur_node_id)
        out_degree = float(node_attr.get('out_degree', 0))

        d_cur = self.graph_cache.get_dist_to_goal(cur_node_id, goal_node_id)
        dist_frac = d_cur / d_start if d_start > 0 else 0.0

        return np.array([out_degree, dist_frac], dtype=np.float32)

    def _get_history_dim(self) -> int:
        """History token维度"""
        return 9  # 4 + 3(path_type_emb) + 2(memory)

    def _get_global_dim(self) -> int:
        """Global state维度"""
        return 2

    def _get_candidate_dim(self) -> int:
        """Candidate feature维度"""
        return 12  # 4 + 3(path_type_emb) + 2(direction) + 1(progress) + 2(constraints)

    def _get_feature_dim(self) -> int:
        """
        Get candidate feature dimension (backward compatibility).
        """
        return self._get_candidate_dim()

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
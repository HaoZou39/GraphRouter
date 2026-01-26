"""
GraphCache: Cross-map generalization infrastructure for path planning.

This module provides a unified cache for graph structure and attributes,
enabling cross-map generalization by using node_id/edge_id as primary keys
instead of coordinate tuples.

Key Features:
- Node/edge ID mapping (coordinate-independent)
- Candidate edge retrieval with stable ordering
- Edge/node attribute storage
- Unified distance calculation with goal-level caching
- Serialization for reproducibility
"""

import pickle
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import os


class GraphCache:
    """
    Cache for graph structure and attributes to enable cross-map generalization.

    Uses node_id/edge_id as primary keys instead of coordinate tuples.
    """

    def __init__(self):
        # Core mappings (coordinate-independent)
        self.node_id_map: Dict[Tuple[float, float], int] = {}  # (norm_x, norm_y) -> node_id
        self.node_coords: Dict[int, Tuple[float, float]] = {}  # node_id -> (norm_x, norm_y)
        self.edge_id_to_nodes: Dict[int, Tuple[int, int]] = {}  # edge_id -> (u_node_id, v_node_id)
        self.node_out_edges: Dict[int, List[int]] = {}  # node_id -> [edge_id1, edge_id2, ...] (stable sorted)

        # Attributes
        self.edge_attrs: Dict[int, Dict[str, Any]] = {}  # edge_id -> {'length_norm': float, 'width': float, ...}
        self.node_attrs: Dict[int, Dict[str, Any]] = {}  # node_id -> {'out_degree': int, ...}

        # Distance cache (goal-level granularity)
        self.dist_cache: Dict[Tuple[int, int], float] = {}  # (goal_node_id, node_id) -> distance

        # Graph reference (for distance computation)
        self.G: Optional[nx.Graph] = None

        # Metadata
        self.coord_stats: Optional[Dict[str, float]] = None
        self.physical_stats: Optional[Dict[str, float]] = None
        self.max_out_degree: int = 1
        self.node_feature_cache: Dict[int, np.ndarray] = {}
        self.node_neighbor_feature_cache: Dict[int, np.ndarray] = {}

    def build_from_networkx(self, G: nx.Graph, edge_id_map: Dict[Tuple, int],
                          edge_feature_list: List[List[float]],
                          coord_stats: Dict[str, float],
                          physical_stats: Dict[str, float]):
        """
        Build GraphCache from NetworkX graph and existing mappings.

        Args:
            G: NetworkX graph with normalized coordinates as nodes
            edge_id_map: {(start_coord, end_coord): edge_id}
            edge_feature_list: List of edge feature vectors
            coord_stats: Coordinate normalization statistics
            physical_stats: Physical feature normalization statistics
        """
        self.G = G
        self.coord_stats = coord_stats
        self.physical_stats = physical_stats

        # Build node ID mapping (coordinate -> node_id)
        node_coords = list(G.nodes())
        for node_id, coord in enumerate(sorted(node_coords)):  # Stable ordering
            self.node_id_map[coord] = node_id
            self.node_coords[node_id] = coord

        # Build edge mappings
        for (start_coord, end_coord), edge_id in edge_id_map.items():
            u_node_id = self.node_id_map[start_coord]
            v_node_id = self.node_id_map[end_coord]
            self.edge_id_to_nodes[edge_id] = (u_node_id, v_node_id)

            # Build reverse mapping (node -> outgoing edges)
            # For undirected graphs, add edge to both nodes' outgoing edge lists
            if u_node_id not in self.node_out_edges:
                self.node_out_edges[u_node_id] = []
            self.node_out_edges[u_node_id].append(edge_id)
            
            # Add edge to v_node as well (for undirected graphs)
            if v_node_id not in self.node_out_edges:
                self.node_out_edges[v_node_id] = []
            self.node_out_edges[v_node_id].append(edge_id)

        # Sort outgoing edges for stable ordering
        for node_id in self.node_out_edges:
            self.node_out_edges[node_id].sort()

        # Store edge attributes
        edge_attr_keys = ['length_norm', 'width', 'curb_norm', 'crossing', 'path_type']
        for edge_id, features in enumerate(edge_feature_list):
            self.edge_attrs[edge_id] = dict(zip(edge_attr_keys, features))

        # Compute node attributes
        max_out_degree = 1
        for node_id in self.node_coords.keys():
            out_degree = len(self.node_out_edges.get(node_id, []))
            self.node_attrs[node_id] = {'out_degree': out_degree}
            max_out_degree = max(max_out_degree, out_degree)
        self.max_out_degree = max_out_degree

    def get_candidate_edges(self, node_id: int) -> List[int]:
        """
        Get candidate outgoing edges for a node (stable sorted).

        Args:
            node_id: Node identifier

        Returns:
            List of edge_ids (stable sorted)
        """
        return self.node_out_edges.get(node_id, []).copy()

    def get_edge_attr(self, edge_id: int) -> Dict[str, Any]:
        """
        Get edge attributes.

        Args:
            edge_id: Edge identifier

        Returns:
            Dictionary of edge attributes
        """
        return self.edge_attrs.get(edge_id, {}).copy()

    def get_node_attr(self, node_id: int) -> Dict[str, Any]:
        """
        Get node attributes.

        Args:
            node_id: Node identifier

        Returns:
            Dictionary of node attributes
        """
        return self.node_attrs.get(node_id, {}).copy()

    def get_node_feature_vector(self, node_id: int) -> np.ndarray:
        """
        Get a node feature vector for GNN-style encoders.

        Features are structural/statistical only (no IDs):
        [out_degree_norm, avg_length_norm, avg_width, avg_curb_norm, avg_crossing]
        """
        if node_id in self.node_feature_cache:
            return self.node_feature_cache[node_id].copy()

        node_attr = self.node_attrs.get(node_id, {})
        out_degree = float(node_attr.get('out_degree', 0.0))
        out_degree_norm = out_degree / max(self.max_out_degree, 1)

        edge_ids = self.node_out_edges.get(node_id, [])
        if not edge_ids:
            features = np.array([out_degree_norm, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            self.node_feature_cache[node_id] = features
            return features.copy()

        length_norms = []
        widths = []
        curb_norms = []
        crossings = []
        for edge_id in edge_ids:
            edge_attr = self.edge_attrs.get(edge_id, {})
            length_norms.append(edge_attr.get('length_norm', 0.0))
            widths.append(edge_attr.get('width', 0.0))
            curb_norms.append(edge_attr.get('curb_norm', 0.0))
            crossings.append(float(edge_attr.get('crossing', 0.0)))

        features = np.array([
            out_degree_norm,
            float(np.mean(length_norms)),
            float(np.mean(widths)),
            float(np.mean(curb_norms)),
            float(np.mean(crossings)),
        ], dtype=np.float32)
        self.node_feature_cache[node_id] = features
        return features.copy()

    def get_neighbor_feature_mean(self, node_id: int) -> np.ndarray:
        """
        Mean feature vector over neighboring nodes (1-hop), for GNN-style aggregation.
        """
        if node_id in self.node_neighbor_feature_cache:
            return self.node_neighbor_feature_cache[node_id].copy()

        edge_ids = self.node_out_edges.get(node_id, [])
        if not edge_ids:
            neighbor_features = np.zeros(5, dtype=np.float32)
            self.node_neighbor_feature_cache[node_id] = neighbor_features
            return neighbor_features.copy()

        neighbor_ids = set()
        for edge_id in edge_ids:
            u_id, v_id = self.edge_id_to_nodes[edge_id]
            neighbor_ids.add(v_id if u_id == node_id else u_id)

        if not neighbor_ids:
            neighbor_features = np.zeros(5, dtype=np.float32)
            self.node_neighbor_feature_cache[node_id] = neighbor_features
            return neighbor_features.copy()

        stacked = np.stack([self.get_node_feature_vector(n_id) for n_id in neighbor_ids], axis=0)
        neighbor_features = np.mean(stacked, axis=0).astype(np.float32)
        self.node_neighbor_feature_cache[node_id] = neighbor_features
        return neighbor_features.copy()

    def get_edge_nodes(self, edge_id: int) -> Tuple[int, int]:
        """
        Get source and target node IDs for an edge.

        Args:
            edge_id: Edge identifier

        Returns:
            (u_node_id, v_node_id)
        """
        return self.edge_id_to_nodes[edge_id]

    def get_dist_to_goal(self, node_id: int, goal_node_id: int) -> float:
        """
        Get distance from node to goal (unified interface).

        Uses graph shortest path with Euclidean fallback.
        Caches results at goal granularity.

        Args:
            node_id: Source node identifier
            goal_node_id: Goal node identifier

        Returns:
            Distance (float)
        """
        cache_key = (goal_node_id, node_id)

        # Check cache first
        if cache_key in self.dist_cache:
            return self.dist_cache[cache_key]

        # Precompute all distances to this goal if not cached
        if not any(k[0] == goal_node_id for k in self.dist_cache.keys()):
            self.precompute_distances_to_goal(goal_node_id)

        # Return cached result (should exist after precomputation)
        return self.dist_cache.get(cache_key, float('inf'))

    def precompute_distances_to_goal(self, goal_node_id: int):
        """
        Precompute distances from all nodes to a specific goal node.

        Args:
            goal_node_id: Goal node identifier
        """
        if self.G is None:
            raise ValueError("Graph not available for distance computation")

        goal_coord = self.node_coords[goal_node_id]

        # Try graph shortest path first
        try:
            # Compute shortest paths from all nodes to goal
            lengths = nx.shortest_path_length(self.G, target=goal_coord, weight='length')

            # Cache results
            for coord, dist in lengths.items():
                if coord in self.node_id_map:
                    node_id = self.node_id_map[coord]
                    self.dist_cache[(goal_node_id, node_id)] = dist

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # Fallback to Euclidean distance
            print(f"Warning: Graph shortest path failed for goal {goal_node_id}, using Euclidean fallback")
            goal_coord = self.node_coords[goal_node_id]

            for node_id, coord in self.node_coords.items():
                euclidean_dist = np.hypot(coord[0] - goal_coord[0], coord[1] - goal_coord[1])
                self.dist_cache[(goal_node_id, node_id)] = euclidean_dist

    def save_to_file(self, path: str):
        """
        Save GraphCache to file.

        Args:
            path: File path
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'node_id_map': self.node_id_map,
                'node_coords': self.node_coords,
                'edge_id_to_nodes': self.edge_id_to_nodes,
                'node_out_edges': self.node_out_edges,
                'edge_attrs': self.edge_attrs,
                'node_attrs': self.node_attrs,
                'dist_cache': self.dist_cache,
                'coord_stats': self.coord_stats,
                'physical_stats': self.physical_stats,
                # Note: self.G is not serialized (can be rebuilt from coordinates)
            }, f)

    def load_from_file(self, path: str):
        """
        Load GraphCache from file.

        Args:
            path: File path
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.node_id_map = data['node_id_map']
        self.node_coords = data['node_coords']
        self.edge_id_to_nodes = data['edge_id_to_nodes']
        self.node_out_edges = data['node_out_edges']
        self.edge_attrs = data['edge_attrs']
        self.node_attrs = data['node_attrs']
        self.dist_cache = data['dist_cache']
        self.coord_stats = data.get('coord_stats')
        self.physical_stats = data.get('physical_stats')

    def rebuild_graph(self) -> nx.Graph:
        """
        Rebuild NetworkX graph from cached data.

        Returns:
            NetworkX graph
        """
        G = nx.Graph()

        # Add nodes
        for coord in self.node_id_map.keys():
            G.add_node(coord)

        # Add edges with attributes
        for edge_id, (u_id, v_id) in self.edge_id_to_nodes.items():
            u_coord = self.node_coords[u_id]
            v_coord = self.node_coords[v_id]

            # Reconstruct static_feature from edge_attrs
            edge_attrs = self.edge_attrs.get(edge_id, {})
            static_feature = [
                edge_attrs.get('length_norm', 0),
                edge_attrs.get('width', 0),
                edge_attrs.get('curb_norm', 0),
                edge_attrs.get('crossing', 0),
                edge_attrs.get('path_type', 0)
            ]

            G.add_edge(u_coord, v_coord,
                      static_feature=static_feature,
                      edge_index=edge_id,
                      length=abs(edge_attrs.get('length_norm', 1.0)))

        self.G = G
        return G

    def get_node_count(self) -> int:
        """Get total number of nodes."""
        return len(self.node_coords)

    def get_edge_count(self) -> int:
        """Get total number of edges."""
        return len(self.edge_id_to_nodes)


class MultiGraphCache:
    """
    Multi-Graph Cache Manager for handling multiple graphs with isolated ID spaces.

    Supports graph-scoped caching with automatic graph context resolution.
    """

    def __init__(self, base_cache_dir: str = '../data/graph_data'):
        """
        Initialize MultiGraphCache.

        Args:
            base_cache_dir: Base directory for graph-specific cache folders
        """
        self.base_cache_dir = Path(base_cache_dir)
        self.cache_dir = self.base_cache_dir
        self.loaded_caches: Dict[str, GraphCache] = {}
        self.loaded_builders: Dict[str, 'FeatureBuilder'] = {}  # Lazy import to avoid circular dependency

    def get_cache(self, graph_id: str) -> GraphCache:
        """
        Get GraphCache for a specific graph, loading it if necessary.

        Args:
            graph_id: Graph identifier (e.g., 'default_graph', 'map_a', 'city_center')

        Returns:
            GraphCache instance for the specified graph
        """
        if graph_id not in self.loaded_caches:
            self._load_graph_cache(graph_id)

        return self.loaded_caches[graph_id]

    def get_feature_builder(self, graph_id: str):
        """
        Get FeatureBuilder for a specific graph, creating it if necessary.

        Args:
            graph_id: Graph identifier

        Returns:
            FeatureBuilder instance for the specified graph
        """
        if graph_id not in self.loaded_builders:
            cache = self.get_cache(graph_id)
            # Lazy import to avoid circular dependency
            from .feature_builder import FeatureBuilder
            self.loaded_builders[graph_id] = FeatureBuilder(cache)

        return self.loaded_builders[graph_id]

    def _load_graph_cache(self, graph_id: str):
        """
        Load GraphCache for a specific graph from disk.

        Args:
            graph_id: Graph identifier
        """
        graph_cache_path = self.cache_dir / graph_id / 'cache' / 'graph_cache.pkl'

        if not graph_cache_path.exists():
            raise FileNotFoundError(f"Graph cache not found for graph '{graph_id}' at {graph_cache_path}")

        cache = GraphCache()
        cache.load_from_file(str(graph_cache_path))

        # Rebuild the NetworkX graph for distance computations
        cache.rebuild_graph()

        self.loaded_caches[graph_id] = cache
        print(f"Loaded graph cache for '{graph_id}': {cache.get_node_count()} nodes, {cache.get_edge_count()} edges")

    def list_available_graphs(self) -> List[str]:
        """
        List all available graphs in the cache directory.

        Returns:
            List of graph IDs that have cache directories
        """
        if not self.cache_dir.exists():
            return []

        return [d.name for d in self.cache_dir.iterdir()
                if d.is_dir() and (d / 'cache' / 'graph_cache.pkl').exists()]

    def save_graph_cache(self, graph_id: str, cache: GraphCache):
        """
        Save a GraphCache for a specific graph.

        Args:
            graph_id: Graph identifier
            cache: GraphCache instance to save
        """
        cache_dir = self.cache_dir / graph_id / 'cache'
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_path = cache_dir / 'graph_cache.pkl'
        cache.save_to_file(str(cache_path))

        # Also save other cache components if they exist
        # (This would be extended based on what other components need to be saved)

        print(f"Saved graph cache for '{graph_id}' to {cache_dir}")

    def preload_graphs(self, graph_ids: List[str]):
        """
        Preload multiple graphs into memory.

        Args:
            graph_ids: List of graph IDs to preload
        """
        for graph_id in graph_ids:
            try:
                self.get_cache(graph_id)
            except Exception as e:
                print(f"Failed to preload graph '{graph_id}': {e}")

    def get_cache_info(self, graph_id: str) -> Dict[str, Any]:
        """
        Get information about a cached graph without loading it.

        Args:
            graph_id: Graph identifier

        Returns:
            Dictionary with cache information
        """
        cache_dir = self.cache_dir / graph_id / 'cache'
        cache_path = cache_dir / 'graph_cache.pkl'

        if not cache_path.exists():
            return {'exists': False}

        # Get file size and modification time
        stat = cache_path.stat()

        return {
            'exists': True,
            'path': str(cache_path),
            'size_mb': stat.st_size / (1024 * 1024),
            'modified': stat.st_mtime
        }

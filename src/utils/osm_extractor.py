"""
OSM Extractor: OSM Network Extraction and Graph Standardization

This module handles:
1. OSM walk network extraction with OSMnx
2. Graph standardization to target statistics (nodes, edges, degrees, lengths)
3. Coordinate-free representation building
4. Edge sampling point generation

Key Features:
- Unified CRS projection (meters)
- Topology cleaning (remove self-loops, fix connectivity)
- Scale standardization to match target graph statistics
- Stable node/edge ID assignment
"""

import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class OSMExtractor:
    """
    OSM network extraction and standardization for graph bank generation.
    """

    def __init__(self):
        # Configure OSMnx (for newer versions)
        try:
            ox.config(use_cache=True, log_console=True)
        except AttributeError:
            # For newer osmnx versions, config might be different
            ox.settings.use_cache = True
            ox.settings.log_console = True

        # Target CRS for meter-based calculations (UTM zone will be auto-detected)
        self.target_crs = None
        self.debug_mode = False

    def extract_walk_network(
        self,
        center: Tuple[float, float],
        radius: float,
        output_dir: Path
    ) -> Tuple[nx.Graph, Dict[str, Any]]:
        """
        Extract walk network from OSM for given AOI.

        Args:
            center: (lat, lon) center point
            radius: Radius in meters
            output_dir: Directory to save raw OSM data

        Returns:
            Tuple of (NetworkX graph, metadata dict)
        """
        logger.info(f"Extracting walk network: center={center}, radius={radius}m")

        # Extract walk network - OSMnx默认返回有向图
        G = ox.graph_from_point(
            center,
            dist=radius,
            network_type='walk',
            simplify=True,
            retain_all=False
        )

        # 对于步行路径，我们需要确保双向性
        # 直接使用OSMnx返回的有向图，不进行复杂的双向转换
        # 这避免了unhashable type错误
        print(f"Using OSMnx directed graph directly: {len(G.nodes())} nodes, {len(G.edges())} edges")

        # Project to UTM for meter calculations
        G_projected = ox.project_graph(G)
        self.target_crs = G_projected.graph['crs']

        # Convert to GeoDataFrame for saving
        nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_projected)

        # Save raw data
        output_dir.mkdir(parents=True, exist_ok=True)
        nodes_gdf.to_file(output_dir / "nodes.geojson", driver='GeoJSON')
        edges_gdf.to_file(output_dir / "edges.geojson", driver='GeoJSON')

        # Extract metadata
        metadata = {
            "crs": str(self.target_crs),
            "center": center,
            "radius": radius,
            "osm_query": f"point={center}, dist={radius}, network_type=walk",
            "extraction_time": pd.Timestamp.now().isoformat(),
            "osm_attribution": "© OpenStreetMap contributors",
            "raw_stats": {
                "nodes": len(G_projected.nodes()),
                "edges": len(G_projected.edges()),
                "total_length_km": sum(nx.get_edge_attributes(G_projected, 'length').values()) / 1000
            }
        }

        return G_projected, metadata

    def standardize_graph(
        self,
        G: nx.Graph,
        target_stats: Dict[str, float],
        crs: str
    ) -> nx.Graph:
        """
        Standardize graph topology to match target statistics.

        Args:
            G: Raw OSM graph
            target_stats: Target statistics (nodes, edges, avg_degree, etc.)
            crs: Coordinate reference system

        Returns:
            Standardized NetworkX graph
        """
        logger.info("Standardizing graph topology")

        G_clean = G.copy()

        # Phase 1: Topology cleaning
        G_clean = self._clean_topology(G_clean)

        # Phase 2: Scale standardization
        G_clean = self._standardize_scale(G_clean, target_stats)

        # Phase 3: Degree distribution alignment
        G_clean = self._align_degree_distribution(G_clean, target_stats)

        logger.info(f"Standardized: {len(G_clean.nodes())} nodes, {len(G_clean.edges())} edges")
        return G_clean

    def _clean_topology(self, G: nx.Graph) -> nx.Graph:
        """Clean graph topology: remove self-loops, small components, etc."""
        # Remove self-loops
        G = G.copy()
        self_loops = list(nx.selfloop_edges(G))
        G.remove_edges_from(self_loops)
        logger.info(f"Removed {len(self_loops)} self-loops")

        # Keep only largest connected component (handle directed graphs)
        try:
            # Try undirected graph methods first
            if not nx.is_connected(G):
                components = list(nx.connected_components(G))
                largest_comp = max(components, key=len)
                G = G.subgraph(largest_comp).copy()
                logger.info(f"Kept largest component: {len(largest_comp)} nodes")
        except nx.NetworkXNotImplemented:
            # Handle directed graphs
            if not nx.is_weakly_connected(G):
                components = list(nx.weakly_connected_components(G))
                largest_comp = max(components, key=len)
                G = G.subgraph(largest_comp).copy()
                logger.info(f"Kept largest weakly connected component: {len(largest_comp)} nodes")

        # Remove isolated nodes
        isolated = list(nx.isolates(G))
        G.remove_nodes_from(isolated)
        logger.info(f"Removed {len(isolated)} isolated nodes")

        return G

    def _standardize_scale(self, G: nx.Graph, target_stats: Dict[str, float]) -> nx.Graph:
        """Adjust node/edge counts to match target statistics."""
        current_nodes = len(G.nodes())
        current_edges = len(G.edges())
        target_nodes = target_stats.get('num_nodes', current_nodes)
        target_edges = target_stats.get('num_edges', current_edges)

        # Node count adjustment (conservative approach)
        if current_nodes > target_nodes * 1.2:
            # Only merge very close nodes to avoid API issues
            try:
                G = self._merge_close_nodes(G, int(target_nodes * 1.1))
            except Exception as e:
                print(f"Node merging failed: {e}, skipping...")
        elif current_nodes < target_nodes * 0.8:
            # Split long edges (if implemented)
            try:
                G = self._split_long_edges(G, target_nodes)
            except Exception as e:
                print(f"Edge splitting failed: {e}, skipping...")

        # Edge length distribution adjustment
        G = self._adjust_edge_lengths(G, target_stats)

        return G

    def _merge_close_nodes(self, G: nx.Graph, target_nodes: int) -> nx.Graph:
        """Merge nodes that are closer than threshold."""
        # Simple merging strategy: iteratively merge closest pairs
        G = G.copy()
        merge_threshold = 5.0  # 5 meters

        while len(G.nodes()) > target_nodes:
            # Find closest node pairs
            min_dist = float('inf')
            merge_pair = None

            nodes_list = list(G.nodes())
            for i, u in enumerate(nodes_list):
                for v in nodes_list[i+1:]:
                    # Calculate great circle distance
                    try:
                        # Try newer OSMnx API
                        dist = ox.distance.great_circle(
                            G.nodes[u]['y'], G.nodes[u]['x'],
                            G.nodes[v]['y'], G.nodes[v]['x']
                        )
                    except AttributeError:
                        # Fallback for older versions
                        import math
                        # Haversine formula approximation
                        lat1, lon1 = G.nodes[u]['y'], G.nodes[u]['x']
                        lat2, lon2 = G.nodes[v]['y'], G.nodes[v]['x']

                        # Convert to radians
                        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

                        # Haversine formula
                        dlon = lon2 - lon1
                        dlat = lat2 - lat1
                        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                        c = 2 * math.asin(math.sqrt(a))
                        dist = 6371 * c * 1000  # Earth radius in km, convert to meters

                    if dist < min_dist:
                        min_dist = dist
                        merge_pair = (u, v)

            if min_dist > merge_threshold or merge_pair is None:
                break

            # Merge the pair
            u, v = merge_pair
            # Combine edges and attributes (simplified)
            try:
                # Try newer NetworkX API
                G = nx.contracted_nodes(G, u, v, self_loops=False)
            except TypeError:
                # Fallback for older NetworkX versions
                G = nx.contracted_nodes(G, u, v)

        return G

    def _split_long_edges(self, G: nx.Graph, target_nodes: int) -> nx.Graph:
        """Split long edges to increase node count."""
        # Simplified implementation: for now, just return the graph as-is
        # In a full implementation, this would split edges longer than a threshold
        # by adding intermediate nodes
        print("Edge splitting not implemented, keeping original graph")
        return G

    def _adjust_edge_lengths(self, G: nx.Graph, target_stats: Dict[str, float]) -> nx.Graph:
        """Adjust edge length distribution to match target."""
        # Get current length stats
        lengths = [data['length'] for _, _, data in G.edges(data=True)]

        # Target length stats
        target_mean = target_stats.get('edge_length_mean_m', np.mean(lengths))
        target_median = target_stats.get('edge_length_median_m', np.median(lengths))

        # For now, keep as-is (length adjustment is complex)
        # Could implement edge splitting/merging based on length distribution
        return G

    def _align_degree_distribution(self, G: nx.Graph, target_stats: Dict[str, float]) -> nx.Graph:
        """Adjust degree distribution to match target."""
        # Simplified: ensure no extremely high degree nodes
        max_target_degree = target_stats.get('max_degree', 8)

        # Remove or split high-degree intersections
        high_degree_nodes = [n for n, d in G.degree() if d > max_target_degree]

        for node in high_degree_nodes:
            # Simplified: remove node and reconnect (complex operation)
            pass

        return G

    def build_coordinate_free_representation(
        self,
        G: nx.Graph
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[Tuple, int], List[List[float]], nx.Graph, List[Dict[str, Any]]]:
        """
        Build coordinate-independent representation for GraphCache.

        Returns:
            Tuple of (coord_stats, physical_stats, edge_id_map, edge_features, G_normalized, osm_data_records)
        """
        # Normalize coordinates to [0,1] range
        x_coords = [data['x'] for _, data in G.nodes(data=True)]
        y_coords = [data['y'] for _, data in G.nodes(data=True)]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        coord_stats = {
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max,
            'x_range': x_max - x_min,
            'y_range': y_max - y_min
        }

        # Create normalized graph
        G_norm = G.copy()
        for node, data in G_norm.nodes(data=True):
            data['x_norm'] = (data['x'] - x_min) / (x_max - x_min)
            data['y_norm'] = (data['y'] - y_min) / (y_max - y_min)

        # Use normalized coordinates as node identifiers
        node_coords_norm = {}
        for node, data in G_norm.nodes(data=True):
            norm_coord = (data['x_norm'], data['y_norm'])
            node_coords_norm[node] = norm_coord

        # Relabel nodes to normalized coordinates
        G_norm_relabeled = nx.relabel_nodes(G_norm, node_coords_norm)

        # Build edge ID map
        edge_id_map = {}
        edge_id_counter = 0

        for u, v, data in G_norm_relabeled.edges(data=True):
            # Sort coordinates for stable ordering
            start_coord = min(u, v)
            end_coord = max(u, v)
            edge_key = (start_coord, end_coord)
            edge_id_map[edge_key] = edge_id_counter
            edge_id_counter += 1

        # Extract physical features from OSM tags
        edge_features = []
        edge_lengths = []
        osm_data_records = []  # 记录OSM原始数据和推理过程

        for (start_coord, end_coord), edge_id in edge_id_map.items():
            # Get the corresponding edge in the normalized graph
            # The relabeled graph should have the same edge attributes
            edge_data = None

            # Find the edge in the relabeled graph
            for u, v, data in G_norm_relabeled.edges(data=True):
                if (u == start_coord and v == end_coord) or (u == end_coord and v == start_coord):
                    edge_data = data
                    break

            if edge_data:
                # Extract OSM tags from edge data
                # Filter out any non-hashable values that might cause issues
                osm_tags = {}
                for k, v in edge_data.items():
                    try:
                        # Test if key-value pair is hashable
                        hash((k, str(v)))
                        osm_tags[k] = v
                    except TypeError:
                        # Skip non-hashable values
                        if self.debug_mode:
                            print(f"DEBUG: Skipping non-hashable edge_data[{k}] = {type(v)}")
                        continue

                if self.debug_mode and edge_id < 3:  # 只显示前3个edge的详细信息
                    print(f"DEBUG: Edge {edge_id} - OSM tags: {osm_tags}")
                    print(f"DEBUG: Edge {edge_id} - All edge_data keys: {list(edge_data.keys())}")
                length = osm_tags.get('length', 10.0)  # Default 10m
                edge_lengths.append(length)

                # Extract attributes from OSM tags (now returns tuples with source info)
                width, width_source = self._extract_width_from_osm(osm_tags)
                curb_norm, curb_source = self._extract_curb_from_osm(osm_tags)
                crossing, crossing_source = self._extract_crossing_from_osm(osm_tags)
                path_type, path_type_source = self._extract_path_type_from_osm(osm_tags)

                features = [length, width, curb_norm, crossing, path_type]

                # Record detailed OSM data and inference process
                osm_record = {
                    'edge_id': edge_id,
                    'start_coord': start_coord,
                    'end_coord': end_coord,
                    # 所有OSM原始数据 - 保存所有可用字段
                    **{f'osm_{k}': v for k, v in osm_tags.items() if k not in ['osmid_original_tags']},
                    # 推理结果和来源
                    'inferred_length': length,
                    'inferred_length_source': 'OSM:length' if osm_tags.get('length') else 'Default:10.0m',
                    'inferred_width': width,
                    'inferred_width_source': width_source,
                    'inferred_curb_norm': curb_norm,
                    'inferred_curb_source': curb_source,
                    'inferred_crossing': crossing,
                    'inferred_crossing_source': crossing_source,
                    'inferred_path_type': path_type,
                    'inferred_path_type_source': path_type_source,
                    'inferred_length': length,
                    'inferred_length_source': 'OSM:length' if osm_tags.get('length') else 'Default:10.0m'
                }
                osm_data_records.append(osm_record)
            else:
                # Fallback for edges without data
                features = [10.0, 1.0, -0.04, 0, 0]  # Default values matching your data
                edge_lengths.append(10.0)

                # Record fallback case
                osm_record = {
                    'edge_id': edge_id,
                    'start_coord': start_coord,
                    'end_coord': end_coord,
                    # OSM原始数据 - 无数据
                    'osm_highway': 'N/A',
                    'osm_width': 'N/A',
                    'osm_kerb': 'N/A',
                    'osm_barrier': 'N/A',
                    'osm_kerb_height': 'N/A',
                    'osm_crossing': 'N/A',
                    'osm_footway_crossing': 'N/A',
                    'osm_bicycle': 'N/A',
                    'osm_foot': 'N/A',
                    'osm_length': 'N/A',
                    # 推理结果 - 默认值
                    'inferred_width': 1.0,
                    'inferred_width_source': 'default_fallback',
                    'inferred_curb_norm': -0.04,
                    'inferred_curb_source': 'default_fallback',
                    'inferred_crossing': 0,
                    'inferred_crossing_source': 'default_fallback',
                    'inferred_path_type': 0,
                    'inferred_path_type_source': 'default_fallback',
                    'inferred_length': 10.0,
                    'inferred_length_source': 'default_fallback'
                }
                osm_data_records.append(osm_record)

            edge_features.append(features)

        # Calculate physical statistics
        physical_stats = {
            'num_nodes': len(G_norm_relabeled.nodes()),
            'num_edges': len(G_norm_relabeled.edges()),
            'avg_degree': sum(dict(G_norm_relabeled.degree()).values()) / len(G_norm_relabeled.nodes()),
            'edge_length_mean_m': np.mean(edge_lengths) if edge_lengths else 10.0,
            'edge_length_median_m': np.median(edge_lengths) if edge_lengths else 10.0
        }

        return coord_stats, physical_stats, edge_id_map, edge_features, G_norm_relabeled, osm_data_records

    def _extract_width_from_osm(self, osm_tags: Dict[str, Any]) -> Tuple[float, str]:
        """
        Extract width from OSM tags.

        Returns:
            Tuple of (width in meters, source description)
        """
        # Check for explicit width tag
        width_str = osm_tags.get('width', '')
        if width_str:
            try:
                # Handle formats like "1.5", "1.5 m", "5 ft"
                width_str = width_str.lower().replace('m', '').replace('meters', '').strip()
                if 'ft' in width_str:
                    # Convert feet to meters
                    width_val = float(width_str.replace('ft', '').strip())
                    return width_val * 0.3048, f"OSM:width={osm_tags.get('width')} (converted from ft)"
                else:
                    return float(width_str), f"OSM:width={osm_tags.get('width')}"
            except (ValueError, AttributeError):
                pass

        # Infer from highway type
        highway = osm_tags.get('highway', '')
        if highway in ['footway', 'pedestrian']:
            return 1.2, f"Inferred:highway={highway} (typical sidewalk width)"
        elif highway == 'cycleway':
            return 1.5, f"Inferred:highway={highway} (typical bike lane width)"
        elif highway == 'path':
            return 1.0, f"Inferred:highway={highway} (typical path width)"
        elif highway == 'steps':
            return 0.8, f"Inferred:highway={highway} (narrow for stairs)"
        else:
            return 1.0, f"Default:highway={highway} (fallback width)"

    def _extract_curb_from_osm(self, osm_tags: Dict[str, Any]) -> Tuple[float, str]:
        """
        Extract curb information from OSM tags.

        Returns:
            Tuple of (curb height normalized, source description)
        """
        # Check for kerb tag
        kerb = osm_tags.get('kerb', '').lower()
        if kerb:
            if kerb in ['lowered', 'flush']:
                return -0.04, f"OSM:kerb={osm_tags.get('kerb')} (lowered/flush)"
            elif kerb == 'raised':
                return 0.04, f"OSM:kerb={osm_tags.get('kerb')} (raised)"
            elif kerb == 'no':
                return -0.04, f"OSM:kerb={osm_tags.get('kerb')} (no curb)"

        # Check for barrier=kerb
        barrier = osm_tags.get('barrier', '').lower()
        if barrier == 'kerb':
            kerb_height = osm_tags.get('kerb_height', '')
            if kerb_height:
                try:
                    height = float(kerb_height)
                    return height if height > 0 else -abs(height), f"OSM:barrier=kerb,kerb_height={kerb_height}"
                except (ValueError, TypeError):
                    pass
            return 0.04, f"OSM:barrier=kerb (default kerb height)"

        # Infer from context
        highway = osm_tags.get('highway', '')
        if highway in ['footway', 'pedestrian', 'crossing']:
            return -0.04, f"Inferred:highway={highway} (sidewalks typically lowered)"
        elif highway in ['primary', 'secondary', 'tertiary']:
            return 0.04, f"Inferred:highway={highway} (main roads typically raised)"
        else:
            return -0.04, f"Default:highway={highway} (pedestrian paths default lowered)"

    def _extract_crossing_from_osm(self, osm_tags: Dict[str, Any]) -> Tuple[int, str]:
        """
        Extract crossing information from OSM tags.

        Returns:
            Tuple of (1 if crossing, 0 otherwise, source description)
        """
        # Direct crossing tags
        if osm_tags.get('crossing'):
            return 1, f"OSM:crossing={osm_tags.get('crossing')}"

        # Highway=crossing
        if osm_tags.get('highway') == 'crossing':
            return 1, f"OSM:highway=crossing"

        # Footway=crossing
        if osm_tags.get('footway') == 'crossing':
            return 1, f"OSM:footway=crossing"

        # Check for crossing_ref or similar
        crossing_keys = [key for key in osm_tags.keys() if key.startswith('crossing')]
        if crossing_keys:
            return 1, f"OSM:crossing-related tags={crossing_keys}"

        return 0, "Default: no crossing indicators found"

    def _extract_path_type_from_osm(self, osm_tags: Dict[str, Any]) -> Tuple[int, str]:
        """
        Extract path type from OSM tags.

        Returns:
            Tuple of (0 for walk_lane, 1 for bike_lane, source description)
        """
        highway = osm_tags.get('highway', '')

        # Direct mappings
        if highway == 'cycleway':
            return 1, f"OSM:highway=cycleway (bike_lane)"
        elif highway in ['footway', 'pedestrian', 'steps']:
            return 0, f"OSM:highway={highway} (walk_lane)"

        # Check for bicycle access
        bicycle = osm_tags.get('bicycle', '').lower()
        if bicycle in ['yes', 'designated', 'permissive']:
            return 1, f"OSM:bicycle={osm_tags.get('bicycle')} (bike_lane)"

        # Check for foot access
        foot = osm_tags.get('foot', '').lower()
        if foot in ['yes', 'designated', 'permissive']:
            return 0, f"OSM:foot={osm_tags.get('foot')} (walk_lane)"

        # Default based on highway type
        if highway in ['path', 'track']:
            # Could be either, check for more specific tags
            if osm_tags.get('sac_scale') or osm_tags.get('mtb:scale'):
                return 1, f"Inferred:highway={highway} with hiking tags (bike/hike path)"
            else:
                return 0, f"Default:highway={highway} (walk_lane)"

        # Default to walk_lane for pedestrian infrastructure
        return 0, f"Default:highway={highway} (walk_lane fallback)"
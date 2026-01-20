"""
Preprocess: Cross-map generalization preprocessing pipeline.

This module processes expert trajectory data into node_id/edge_id sequences
for cross-map generalization, eliminating coordinate dependencies.

- No 15-dim coordinate feature vectors
- Output node_id/edge_id sequences
- Unified distance calculation using GraphCache
- Cross-map invariant preprocessing
"""

import os
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import json
from shapely import wkt
from utils.utility import *
from utils.graph_cache import GraphCache, MultiGraphCache
from copy import deepcopy
import argparse
import networkx as nx

def load_route_from_file(file_path: str, meta_map: dict) -> gpd.GeoDataFrame:
    """
    Load the route (GeoDataFrame) from a file in GeoPackage format.
    """
    loaded_route = gpd.read_file(file_path)
    loaded_route = loaded_route.set_crs(meta_map["CRS"], allow_override=True)
    if loaded_route.empty or loaded_route.geometry.isnull().all():
        raise ValueError(f"Loaded route data is empty or has invalid geometry: {file_path}")
    return loaded_route


def load_normalization_stats(data_directory, graph_id='default_graph'):
    """Load normalization parameters"""
    coord_stats_path = os.path.join(data_directory, 'graph_data', graph_id, 'cache', 'coordinate_stats.json')
    physical_stats_path = os.path.join(data_directory, 'graph_data', graph_id, 'cache', 'physical_stats.json')

    coord_stats = None
    physical_stats = None

    if os.path.exists(coord_stats_path):
        with open(coord_stats_path, 'r') as f:
            coord_stats = json.load(f)

    if os.path.exists(physical_stats_path):
        with open(physical_stats_path, 'r') as f:
            physical_stats = json.load(f)

    return coord_stats, physical_stats


def normalize_coordinate(coord, coord_stats):
    """Normalize a single coordinate"""
    if coord_stats is None:
        return coord
    x, y = coord
    x_norm = (x - coord_stats['x_mean']) / coord_stats['x_std']
    y_norm = (y - coord_stats['y_mean']) / coord_stats['y_std']
    return (x_norm, y_norm)


def process_gpkg_from_graph(file_path, route_id, meta_data, data_directory, graph_cache: GraphCache, graph_id: str = 'default_graph'):
    """
    Process trajectory data into node_id/edge_id sequences for cross-map generalization.

    NEW OUTPUT FORMAT (no coordinate feature vectors):
    {
        'route_id': route_id,
        'step_id': i,
        'start_node_id': node_id,      # Start node ID
        'goal_node_id': goal_node_id,  # Goal node ID
        'cur_node_id': cur_node_id,    # Current node ID
        'action_edge_id': edge_id,     # Action taken (edge ID)
        'next_node_id': next_node_id,  # Next node ID
        'dist_to_goal': dist,          # Distance to goal (unified metric)
    }

    Args:
        file_path: Path to trajectory file (.gpkg)
        route_id: Route identifier
        meta_data: Map metadata
        data_directory: Data directory
        graph_cache: GraphCache instance with node/edge mappings

    Returns:
        pd.DataFrame with new schema
    """
    gdf = load_route_from_file(file_path, meta_data['map'])
    base_name = os.path.basename(file_path)
    csv_name = base_name.replace('p_', '').replace('.gpkg', '_start_end.csv')
    start_end_csv_path = os.path.join(os.path.dirname(file_path), csv_name)

    # Load start/end coordinates from GPKG file
    start_coord = None
    end_coord = None

    try:
        # Load the route geometry from GPKG
        route_gdf = gpd.read_file(file_path)

        if not route_gdf.empty and len(route_gdf) > 0:
            # Get the first and last edges to find start and end points
            first_edge = route_gdf.iloc[0].geometry
            last_edge = route_gdf.iloc[-1].geometry

            if hasattr(first_edge, 'coords') and hasattr(last_edge, 'coords'):
                first_coords = list(first_edge.coords)
                last_coords = list(last_edge.coords)

                if first_coords and last_coords:
                    # Start point is the first coordinate of the first edge
                    start_coord = tuple(first_coords[0])
                    # End point is the last coordinate of the last edge
                    end_coord = tuple(last_coords[-1])

    except Exception as e:
        print(f"Error loading coordinates from GPKG: {e}")

    if start_coord is None or end_coord is None:
        print(f"Warning: Could not load start/end coordinates for {file_path}")
        return pd.DataFrame()

    # Convert coordinates to normalized space and get node IDs
    coord_stats = graph_cache.coord_stats
    start_norm = normalize_coordinate(start_coord, coord_stats)
    end_norm = normalize_coordinate(end_coord, coord_stats)

    # Find closest nodes with tolerance for floating point precision issues
    def find_closest_node(coord, tolerance=1e-6):
        """Find the closest node to a coordinate within tolerance."""
        # First try exact match
        node_id = graph_cache.node_id_map.get(coord)
        if node_id is not None:
            return node_id

        # If no exact match, find closest within tolerance
        min_dist = float('inf')
        closest_node_id = None

        for graph_coord, node_id in graph_cache.node_id_map.items():
            dist = ((graph_coord[0] - coord[0])**2 + (graph_coord[1] - coord[1])**2)**0.5
            if dist < min_dist and dist < tolerance:
                min_dist = dist
                closest_node_id = node_id

        return closest_node_id

    start_node_id = find_closest_node(start_norm)
    goal_node_id = find_closest_node(end_norm)

    if start_node_id is None or goal_node_id is None:
        print(f"Warning: Start or goal node not found in graph for route {route_id}")
        print(f"  Start coord: {start_coord} -> normalized: {start_norm}")
        print(f"  Goal coord: {end_coord} -> normalized: {end_norm}")
        return pd.DataFrame()

    # Check if this is a loop route (start == goal)
    is_loop = (start_node_id == goal_node_id)
    
    # Process trajectory using actual edge sequence from GPKG
    trajectory_rows = []
    cur_node_id = start_node_id
    
    for step_idx, row in gdf.iterrows():
        geom = row['geometry']
        if geom is None or geom.is_empty:
            continue
        
        # Normalize edge coordinates
        edge_start = normalize_coordinate((geom.coords[0][0], geom.coords[0][1]), coord_stats)
        edge_end = normalize_coordinate((geom.coords[-1][0], geom.coords[-1][1]), coord_stats)
        
        # Find node IDs for this edge
        edge_start_node = find_closest_node(edge_start)
        edge_end_node = find_closest_node(edge_end)
        
        if edge_start_node is None or edge_end_node is None:
            continue
        
        # Find edge_id in graph
        edge_id = None
        for eid, (u_id, v_id) in graph_cache.edge_id_to_nodes.items():
            if (u_id == edge_start_node and v_id == edge_end_node) or \
               (u_id == edge_end_node and v_id == edge_start_node):
                edge_id = eid
                break
        
        if edge_id is None:
            continue
        
        # Determine next node based on current position and edge direction
        if edge_start_node == cur_node_id:
            next_node_id = edge_end_node
        elif edge_end_node == cur_node_id:
            next_node_id = edge_start_node
        else:
            # Edge not connected to current node - try to fix by using edge start
            cur_node_id = edge_start_node
            next_node_id = edge_end_node
        
        # For loop routes, calculate dist_to_goal as remaining path length
        if is_loop:
            # For loops, use a small non-zero distance to avoid division by zero
            dist_to_goal = max(0.001, len(gdf) - step_idx)
        else:
            dist_to_goal = graph_cache.get_dist_to_goal(cur_node_id, goal_node_id)
        
        # Build trajectory row
        trajectory_row = {
            'graph_id': graph_id,
            'route_id': route_id,
            'step_id': step_idx,
            'start_node_id': start_node_id,
            'goal_node_id': goal_node_id,
            'cur_node_id': cur_node_id,
            'action_edge_id': edge_id,
            'next_node_id': next_node_id,
            'dist_to_goal': dist_to_goal,
        }
        trajectory_rows.append(trajectory_row)
        
        # Move to next node
        cur_node_id = next_node_id

    return pd.DataFrame(trajectory_rows)


def process_gpkg(file_path, route_id, meta_data, data_directory):
    """Maintain backward compatibility with old function"""
    return process_gpkg_from_graph(file_path, route_id, meta_data, data_directory, None)


def process_split(split_name, data_directory, graph_id, meta_data, graph_cache: GraphCache):
    """
    Process a data split into node_id/edge_id sequences.

    Args:
        split_name: 'train', 'val', or 'test'
        data_directory: Data directory path
        graph_id: Graph identifier
        meta_data: Map metadata
        graph_cache: GraphCache instance
    """
    # Read from graph-specific raw_data directory
    split_dir = os.path.join(data_directory, 'graph_data', graph_id, 'raw_data', split_name)
    all_trajectory_rows = []

    if not os.path.exists(split_dir):
        print(f"Warning: Split directory {split_dir} does not exist")
        return

    processed_count = 0
    error_count = 0
    skipped_count = 0

    for fname in os.listdir(split_dir):
        if fname.endswith('.gpkg'):
            route_id = os.path.splitext(fname)[0].split('_')[-1]
            file_path = os.path.join(split_dir, fname)

            try:
                df = process_gpkg_from_graph(file_path, route_id, meta_data, data_directory, graph_cache, graph_id)
                if not df.empty:
                    all_trajectory_rows.append(df)
                    processed_count += 1
                else:
                    print(f"Warning: Empty trajectory for {file_path}")
                    skipped_count += 1
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                error_count += 1
                continue

    print(f"Processed {processed_count} routes, {skipped_count} skipped, {error_count} errors for {split_name}")

    if all_trajectory_rows:
        all_trajectories = pd.concat(all_trajectory_rows, ignore_index=True)
        # Save to graph-specific trajectories directory
        trajectories_dir = os.path.join(data_directory, 'graph_data', graph_id, 'trajectories')
        os.makedirs(trajectories_dir, exist_ok=True)
        output_path = os.path.join(trajectories_dir, f'sampled_{split_name}.df')
        all_trajectories.to_pickle(output_path)
        print(f"SUCCESS: Saved {split_name} trajectories to {output_path}")
        print(f"   Shape: {all_trajectories.shape}")
        print(f"   Unique routes: {all_trajectories['route_id'].nunique()}")
        print(f"   Total steps: {len(all_trajectories)}")
    else:
        print(f"ERROR: No valid trajectories found in {split_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess trajectory data')
    parser.add_argument('--data_dir', type=str, default='../data',
                       help='Data directory path (default: ../data)')
    parser.add_argument('--graph_id', type=str, default='default_graph',
                       help='Graph identifier for multi-graph support (default: default_graph)')

    args = parser.parse_args()
    data_directory = args.data_dir
    graph_id = args.graph_id

    # Convert relative path to absolute path relative to script location
    if not os.path.isabs(data_directory):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_directory = os.path.join(script_dir, '..', data_directory.lstrip('./').lstrip('../'))

    # Load or create graph cache using MultiGraphCache
    multi_cache = MultiGraphCache(os.path.join(data_directory, 'graph_data'))

    cache_info = multi_cache.get_cache_info(graph_id)
    if not cache_info['exists']:
        print(f"WARNING: GraphCache not found for graph '{graph_id}'. Building graph first...")

        # Load map data

        # Load from graph-specific raw_data directory
        raw_data_dir = os.path.join(data_directory, 'graph_data', graph_id, 'raw_data')
        data = load_map_data(raw_data_dir)
        df, meta_data = data["df"], data["meta_data"]
        df_copy = deepcopy(df)

        # Build graph (this will create graph_cache.pkl automatically)
        print("Building graph and GraphCache...")
        G, edge_id_map, edge_feature_list = load_or_create_graph(df_copy, data_directory, graph_id)

        # The cache should now be saved in the graph directory
        cache_info = multi_cache.get_cache_info(graph_id)
        if not cache_info['exists']:
            raise FileNotFoundError(
                f"GraphCache was not created for graph '{graph_id}'. "
                f"Please check that create_network_graph_with_features() is saving the cache."
            )
        print("SUCCESS: Graph and GraphCache built successfully")
    else:
        # Load metadata for verification from graph-specific location
        meta_data_path = os.path.join(data_directory, 'graph_data', graph_id, 'raw_data', 'metadata.json')
        if not os.path.exists(meta_data_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_data_path}")
        with open(meta_data_path, 'r') as f:
            meta_data = json.load(f)

    # Get graph cache for specified graph
    graph_cache = multi_cache.get_cache(graph_id)
    # Rebuild the NetworkX graph from cached data
    graph_cache.rebuild_graph()
    print(f"SUCCESS: Loaded GraphCache for graph '{graph_id}'")
    print(f"   Nodes: {len(graph_cache.G.nodes)}, Edges: {len(graph_cache.G.edges)}")

    # Load metadata from graph-specific location
    meta_data_path = os.path.join(data_directory, 'graph_data', graph_id, 'raw_data', 'metadata.json')
    if not os.path.exists(meta_data_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_data_path}")
    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)

    # Process all splits
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} split...")
        process_split(split, data_directory, graph_id, meta_data, graph_cache)
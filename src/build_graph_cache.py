"""
Build Graph Cache from raw network data.

This script builds the GraphCache independently from route generation.
Use this when you need to rebuild the cache after code changes (e.g., bug fixes).

Usage:
    python build_graph_cache.py [--graph-id GRAPH_ID] [--data-dir DATA_DIR]

Example:
    python build_graph_cache.py --graph-id default_graph --data-dir ../data/graph_data
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.dataparser import create_network_graph
from utils.graph_cache import GraphCache


def build_cache_for_graph(graph_id: str, data_directory: str):
    """
    Build GraphCache for a specific graph from raw data.
    
    Args:
        graph_id: Graph identifier (e.g., 'default_graph')
        data_directory: Base data directory containing graph_data/
    """
    print(f"\n{'='*60}")
    print(f"Building Graph Cache for: {graph_id}")
    print(f"{'='*60}\n")
    
    # Locate raw data
    graph_dir = os.path.join(data_directory, 'graph_data', graph_id)
    raw_data_dir = os.path.join(graph_dir, 'raw_data')
    
    # Find network_map file
    network_map_path = None
    for filename in ['network_map.gpkg', 'network_map']:
        candidate = os.path.join(raw_data_dir, filename)
        if os.path.exists(candidate):
            network_map_path = candidate
            break
    
    if network_map_path is None:
        raise FileNotFoundError(
            f"Network map file not found in {raw_data_dir}\n"
            f"Expected: network_map.gpkg or network_map"
        )
    
    print(f"📂 Loading network data from: {network_map_path}")
    
    # Load the network data
    df = gpd.read_file(network_map_path)
    print(f"   Loaded {len(df)} edges from network file")
    
    # Create cache directory
    cache_dir = os.path.join(graph_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    print(f"📁 Cache directory: {cache_dir}")
    
    # Build graph from original data WITHOUT applying user constraints
    # This ensures we cache all edges (data-driven, not rule-driven)
    print("\n🔨 Building graph structure...")
    G_con, _ = create_network_graph(df, use_directed=False)
    G = G_con
    
    print(f"   Graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
    
    # Get coordinate bounds for normalization
    print("\n📏 Computing normalization statistics...")
    coords = list(G.nodes())
    x_coords = [coord[0] for coord in coords]
    y_coords = [coord[1] for coord in coords]
    
    coord_stats = {
        'x_min': min(x_coords),
        'x_max': max(x_coords),
        'y_min': min(y_coords),
        'y_max': max(y_coords),
        'x_range': max(x_coords) - min(x_coords),
        'y_range': max(y_coords) - min(y_coords)
    }
    
    print(f"   X range: [{coord_stats['x_min']:.2f}, {coord_stats['x_max']:.2f}]")
    print(f"   Y range: [{coord_stats['y_min']:.2f}, {coord_stats['y_max']:.2f}]")
    
    # Normalize coordinates
    def normalize_coord(coord):
        x_norm = (coord[0] - coord_stats['x_min']) / coord_stats['x_range'] if coord_stats['x_range'] > 0 else 0.5
        y_norm = (coord[1] - coord_stats['y_min']) / coord_stats['y_range'] if coord_stats['y_range'] > 0 else 0.5
        return (x_norm, y_norm)
    
    # Create normalized graph
    print("\n🔄 Creating normalized graph...")
    G_norm = nx.Graph()
    for u, v, edge_data in G.edges(data=True):
        u_norm = normalize_coord(u)
        v_norm = normalize_coord(v)
        G_norm.add_edge(u_norm, v_norm, **edge_data)
    
    # First pass: collect raw data for statistics
    print("\n📊 Collecting edge statistics...")
    edge_lengths = []
    curb_heights = []
    
    # Create mapping from original edge coordinates to dataframe indices
    edge_to_df_idx = {}
    for idx, row in df.iterrows():
        geom = row.geometry
        if geom is not None and hasattr(geom, 'coords'):
            coords_geom = list(geom.coords)
            if len(coords_geom) >= 2:
                start_coord = (coords_geom[0][0], coords_geom[0][1])
                end_coord = (coords_geom[-1][0], coords_geom[-1][1])
                edge_to_df_idx[(start_coord, end_coord)] = idx
    
    for u, v, edge_data in G.edges(data=True):
        # Extract edge attributes for statistics
        length = edge_data.get('length', 1.0)
        edge_lengths.append(length)
        
        # Try to find matching row in dataframe
        df_idx = edge_to_df_idx.get((u, v))
        if df_idx is not None:
            row = df.loc[df_idx]
            curb_height = row.get('curb_height_max', 0.0)
            if pd.isna(curb_height):
                curb_height = 0.0
            curb_heights.append(curb_height)
        else:
            curb_heights.append(0.0)
    
    # Calculate normalization parameters
    max_length = max(edge_lengths) if edge_lengths else 1.0
    curb_max = max(curb_heights) if curb_heights else 0.04
    
    print(f"   Max edge length: {max_length:.2f} m")
    print(f"   Max curb height: {curb_max:.3f} m")
    
    # Second pass: build edge mappings and normalized features
    print("\n🏗️  Building edge features...")
    edge_id_map = {}
    edge_feature_list = []
    
    edge_id_counter = 0
    for u, v, edge_data in G.edges(data=True):
        # Create edge ID mapping with normalized coordinates
        u_norm = normalize_coord(u)
        v_norm = normalize_coord(v)
        edge_id_map[(u_norm, v_norm)] = edge_id_counter
        
        # Extract and normalize features
        length = edge_data.get('length', 1.0)
        length_norm = length / max_length
        
        # Get other features
        df_idx = edge_to_df_idx.get((u, v))
        if df_idx is not None:
            row = df.loc[df_idx]
            width = row.get('obstacle_free_width_float', 1.0)
            curb_height = row.get('curb_height_max', 0.0)
            if pd.isna(curb_height):
                curb_height = 0.0
            curb_norm = curb_height / curb_max if curb_max > 0 else 0.0
            crossing = 1 if row.get('crossing', 'No') == 'Yes' else 0
            # Encode path_type: walk=0, bike=1, walk_bike_connection=2
            path_type_str = str(row.get('path_type', ''))
            if path_type_str == 'walk':
                path_type = 0
            elif path_type_str == 'bike':
                path_type = 1
            elif path_type_str == 'walk_bike_connection':
                path_type = 2
            else:
                path_type = 0
        else:
            # Default values if no matching row found
            width = 1.0
            curb_norm = 0.0
            crossing = 0
            path_type = 0
        
        edge_feature_list.append([
            length_norm,  # length_norm
            width,        # width
            curb_norm,    # curb_norm
            crossing,     # crossing
            path_type     # path_type
        ])
        
        edge_id_counter += 1
    
    # Filter out NaN values from curb_heights
    curb_heights_clean = [h for h in curb_heights if not np.isnan(h)] if curb_heights else []
    
    # Calculate physical statistics
    physical_stats = {
        'num_nodes': len(G_norm.nodes()),
        'num_edges': len(G_norm.edges()),
        'avg_degree': sum(dict(G_norm.degree()).values()) / len(G_norm.nodes()) if G_norm.nodes() else 0,
        'edge_length_mean_m': np.mean(edge_lengths) if edge_lengths else 0,
        'edge_length_std_m': np.std(edge_lengths) if edge_lengths else 0,
        'edge_length_max_m': max(edge_lengths) if edge_lengths else 1.0,
        'curb_height_mean_m': np.mean(curb_heights_clean) if curb_heights_clean else 0,
        'curb_height_std_m': np.std(curb_heights_clean) if curb_heights_clean else 0,
        'curb_height_max_m': max(curb_heights_clean) if curb_heights_clean else 0.04,
        'total_length_m': sum(edge_lengths) if edge_lengths else 0
    }
    
    print(f"   Built {len(edge_feature_list)} edge features")
    
    # Save individual cache files
    print("\n💾 Saving cache files...")
    
    print("   Saving edge_id_map.pkl...")
    with open(os.path.join(cache_dir, 'edge_id_map.pkl'), 'wb') as f:
        pickle.dump(edge_id_map, f)
    
    print("   Saving edge_feature_list.pkl...")
    with open(os.path.join(cache_dir, 'edge_feature_list.pkl'), 'wb') as f:
        pickle.dump(edge_feature_list, f)
    
    print("   Saving coordinate_stats.json...")
    with open(os.path.join(cache_dir, 'coordinate_stats.json'), 'w') as f:
        json.dump(coord_stats, f, indent=2)
    
    print("   Saving physical_stats.json...")
    with open(os.path.join(cache_dir, 'physical_stats.json'), 'w') as f:
        json.dump(physical_stats, f, indent=2)
    
    # Create and save GraphCache
    print("   Creating GraphCache...")
    graph_cache = GraphCache()
    graph_cache.build_from_networkx(G_norm, edge_id_map, edge_feature_list, coord_stats, physical_stats)
    
    print("   Saving graph_cache.pkl...")
    graph_cache.save_to_file(os.path.join(cache_dir, 'graph_cache.pkl'))
    
    print(f"\n✅ Cache build complete!")
    print(f"   Nodes: {len(G_norm.nodes())}")
    print(f"   Edges: {len(G_norm.edges())}")
    print(f"   Cache directory: {cache_dir}")
    
    # Print summary statistics
    print(f"\n📈 Graph Statistics:")
    print(f"   Average degree: {physical_stats['avg_degree']:.2f}")
    print(f"   Total length: {physical_stats['total_length_m']:.2f} m")
    print(f"   Average edge length: {physical_stats['edge_length_mean_m']:.2f} ± {physical_stats['edge_length_std_m']:.2f} m")
    print(f"   Average curb height: {physical_stats['curb_height_mean_m']:.3f} ± {physical_stats['curb_height_std_m']:.3f} m")
    
    return graph_cache


def main():
    parser = argparse.ArgumentParser(
        description='Build GraphCache from raw network data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build cache for default graph
  python build_graph_cache.py
  
  # Build cache for a specific graph
  python build_graph_cache.py --graph-id my_custom_graph
  
  # Specify custom data directory
  python build_graph_cache.py --data-dir /path/to/data
        """
    )
    
    parser.add_argument(
        '--graph-id',
        type=str,
        default='default_graph',
        help='Graph identifier (default: default_graph)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Base data directory (default: ../data)'
    )
    
    args = parser.parse_args()
    
    # Determine data directory
    if args.data_dir:
        data_directory = args.data_dir
    else:
        # Default to ../data relative to this script
        data_directory = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    data_directory = os.path.abspath(data_directory)
    
    if not os.path.exists(data_directory):
        print(f"❌ Error: Data directory not found: {data_directory}")
        sys.exit(1)
    
    print(f"📍 Data directory: {data_directory}")
    
    try:
        build_cache_for_graph(args.graph_id, data_directory)
    except Exception as e:
        print(f"\n❌ Error building cache: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

"""
Utility functions for the LLM4Rec project.

This module contains utility functions for loading map data, creating graphs,
and other common operations.
"""

import os
import pandas as pd
import geopandas as gpd
import json
import pickle
from typing import Dict, Any, Tuple


def load_map_data(data_directory: str) -> Dict[str, Any]:
    """
    Load map data from the specified directory.

    Args:
        data_directory: Path to the data directory containing network_map and metadata.json

    Returns:
        Dictionary containing 'df' (GeoDataFrame) and 'meta_data' (metadata dict)
    """
    # Load network map - try different formats
    network_file_path = os.path.join(data_directory, 'network_map.gpkg')
    if not os.path.exists(network_file_path):
        # Try .shp format as fallback
        network_file_path = os.path.join(data_directory, 'network_map.shp')
        if not os.path.exists(network_file_path):
            # Try without extension
            network_file_path = os.path.join(data_directory, 'network_map')
            if not os.path.exists(network_file_path):
                raise FileNotFoundError(f"Network map not found at {os.path.join(data_directory, 'network_map')} (tried .gpkg, .shp, and no extension)")

    df = gpd.read_file(network_file_path)

    # Load metadata
    metadata_path = os.path.join(data_directory, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    with open(metadata_path, 'r') as f:
        meta_data = json.load(f)

    return {
        'df': df,
        'meta_data': meta_data
    }


def load_or_create_graph(df: gpd.GeoDataFrame, data_directory: str, graph_id: str = 'default_graph') -> Tuple[Any, Dict, list]:
    """
    Load existing graph cache or create a new one.

    Args:
        df: Road network GeoDataFrame
        data_directory: Base data directory
        graph_id: Graph identifier

    Returns:
        Tuple of (G, edge_id_map, edge_feature_list)
    """
    from .graph_cache import GraphCache

    # Create cache directory path
    cache_dir = os.path.join(data_directory, 'graph_data', graph_id, 'cache')
    cache_file = os.path.join(cache_dir, 'graph_cache.pkl')

    # Try to load existing cache
    try:
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Graph cache file not found: {cache_file}")

        # Load GraphCache
        graph_cache = GraphCache()
        graph_cache.load_from_file(cache_file)
        print(f"Loaded existing graph cache from {cache_file}")

        # Rebuild the NetworkX graph
        graph_cache.rebuild_graph()
        G = graph_cache.G

        # Load edge_id_map and edge_feature_list from separate cache files
        edge_id_map_file = os.path.join(cache_dir, 'edge_id_map.pkl')
        edge_feature_list_file = os.path.join(cache_dir, 'edge_feature_list.pkl')

        if os.path.exists(edge_id_map_file):
            with open(edge_id_map_file, 'rb') as f:
                edge_id_map = pickle.load(f)
        else:
            edge_id_map = {}
            print(f"Warning: edge_id_map.pkl not found at {edge_id_map_file}")

        if os.path.exists(edge_feature_list_file):
            with open(edge_feature_list_file, 'rb') as f:
                edge_feature_list = pickle.load(f)
        else:
            edge_feature_list = []
            print(f"Warning: edge_feature_list.pkl not found at {edge_feature_list_file}")

        return G, edge_id_map, edge_feature_list

    except FileNotFoundError as e:
        print(f"No existing cache found: {e}")
        print("Please ensure the graph cache has been built first.")
        print(f"Run: python src/build_graph_cache.py --graph-id {graph_id} --data-dir {data_directory}")
        raise

    except Exception as e:
        print(f"Error loading cache: {e}")
        raise


def pad_history(history_list, max_length, pad_value=0):
    """
    Pad history list to maximum length.

    Args:
        history_list: List of history items
        max_length: Maximum length to pad to
        pad_value: Value to use for padding

    Returns:
        Padded list
    """
    if len(history_list) >= max_length:
        return history_list[:max_length]
    else:
        return history_list + [pad_value] * (max_length - len(history_list))


def get_edge_id(edge, edge_id_map):
    """
    Get edge ID from edge tuple.

    Args:
        edge: Edge tuple (u, v)
        edge_id_map: Dictionary mapping edges to IDs

    Returns:
        Edge ID or None if not found
    """
    return edge_id_map.get(edge, edge_id_map.get((edge[1], edge[0]), None))


def to_pickled_df(data_directory: str, **kwargs):
    """
    Save DataFrames to pickle files in the specified directory.
    
    Args:
        data_directory: Directory to save files
        **kwargs: Named DataFrames to save (e.g., replay_buffer=df, data_statis=df)
    """
    os.makedirs(data_directory, exist_ok=True)
    
    for name, df in kwargs.items():
        if df is not None:
            file_path = os.path.join(data_directory, f'{name}.df')
            df.to_pickle(file_path)
            print(f"Saved {name} to {file_path}")
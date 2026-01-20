"""
Utility functions for the LLM4Rec project.

This module contains utility functions for loading map data, creating graphs,
and other common operations.
"""

import os
import pandas as pd
import geopandas as gpd
import json
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

    # Try to load existing cache
    try:
        graph_cache = GraphCache.load_from_file(cache_dir)
        print(f"Loaded existing graph cache from {cache_dir}")

        # Rebuild the NetworkX graph
        graph_cache.rebuild_graph()
        G = graph_cache.G

        # Get edge mappings (this might need to be implemented in GraphCache)
        edge_id_map = getattr(graph_cache, 'edge_id_map', {})
        edge_feature_list = getattr(graph_cache, 'edge_feature_list', [])

        return G, edge_id_map, edge_feature_list

    except FileNotFoundError:
        print(f"No existing cache found at {cache_dir}, this should not happen for preprocess.py")
        print("Please ensure generate_routes.py has been run first to create the cache.")
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
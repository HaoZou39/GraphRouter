import os
import json
import pandas as pd
import geopandas as gpd
import networkx as nx
import shapely.ops as so
import shapely.geometry as sg
from utils.router import Router
from shapely import wkt
from utils.dataparser import create_network_graph, handle_weight
from copy import deepcopy
import matplotlib.pyplot as plt
import contextily as cx
from typing import Dict, Any
from shapely.geometry import Point
from random import uniform
from collections import Counter
import shutil
import random
import pickle
import numpy as np

def save_route_to_file(gdf_route: gpd.GeoDataFrame, store_path: str, route_id: int, crs: str = None) -> None:
    """Save route to file with simplified naming and CRS."""
    file_path = os.path.join(store_path, f'p_route_{route_id}.gpkg')

    # Set CRS if provided and not already set
    if crs and gdf_route.crs is None:
        gdf_route = gdf_route.set_crs(crs, allow_override=True)

    gdf_route.to_file(file_path, driver='GPKG')
    print(f"Route saved to {file_path}")

def load_route_from_file(store_path: str, route_id: int, meta_map: dict) -> gpd.GeoDataFrame:
    """Load route from file with simplified naming."""
    file_path = os.path.join(store_path, f'p_route_{route_id}.gpkg')
    loaded_route = gpd.read_file(file_path)
    loaded_route = loaded_route.set_crs(meta_map["CRS"], allow_override=True)
    if loaded_route.empty or loaded_route.geometry.isnull().all():
        raise ValueError(f"Loaded route data is empty or has invalid geometry: {file_path}")  
    print(f"Route loaded from {file_path}")
    return loaded_route

def plot_route_with_network(
    df: gpd.GeoDataFrame,
    df_path_fact: gpd.GeoDataFrame,
    gdf_coords: gpd.GeoDataFrame,
    origin_node_loc: Point,
    dest_node_loc: Point,
    meta_map: Dict[str, Any],
    buffer_radius: float,
    figsize: tuple = (12, 12),
) -> None:
    gdf_coords["buffer"] = gdf_coords["geometry"].buffer(buffer_radius, cap_style=3)
    plot_area = gpd.GeoDataFrame(
        geometry=[gdf_coords["buffer"][0].union(gdf_coords["buffer"][1])],
        crs=meta_map["CRS"],
    )
    df_sub = gpd.sjoin(df, plot_area, how='inner').reset_index()

    fig, ax = plt.subplots(figsize=figsize)
    df_sub.plot(ax=ax, color='lightgrey', linewidth=1, label='Network')
    df_path_fact.plot(ax=ax, color='grey', linewidth=4, label='Route')

    gdf_coords.head(1).plot(ax=ax, color='blue', markersize=50, label='Origin Location')
    gdf_coords.tail(1).plot(ax=ax, color='red', markersize=50, label='Destination Location')

    gpd.GeoSeries([origin_node_loc], crs=meta_map["CRS"]).plot(ax=ax, color='green', markersize=20, label='Origin Node')
    gpd.GeoSeries([dest_node_loc], crs=meta_map["CRS"]).plot(ax=ax, color='yellow', markersize=20, label='Destination Node')

    cx.add_basemap(ax=ax, 
                  source=cx.providers.CartoDB.Voyager,
                  crs=meta_map["CRS"]) 

    plt.legend(loc="lower right")
    plt.axis('off')
    plt.show()

def load_data(data_directory: str, graph_id: str = 'default_graph') -> Dict[str, Any]:
    """Load map data from graph-specific raw_data directory."""
    from utils.utility import load_map_data
    # For route generation, we load from graph-specific raw_data directory
    graph_raw_data_dir = os.path.join(data_directory, 'graph_data', graph_id, 'raw_data')
    return load_map_data(graph_raw_data_dir)

def generate_route_with_retry(
    df: gpd.GeoDataFrame,
    meta_data: Dict[str, Any],
    max_retries: int = 5,
    heuristic: str = "dijkstra",
    heuristic_f: str = "my_weight",
) -> Dict[str, Any]:
    """
    Generate a route with connectivity checking and retry logic.

    Args:
        df: Road network GeoDataFrame
        meta_data: Map metadata
        max_retries: Maximum number of OD pair generation attempts
        heuristic: Path finding algorithm
        heuristic_f: Edge weight function

    Returns:
        Route data dictionary
    """
    user_model = meta_data["user_model"]
    meta_map = meta_data["map"]
    df_copy = deepcopy(df)
    df_copy = handle_weight(df_copy, user_model)
    _, G = create_network_graph(df_copy, use_directed=False)

    # Pre-compute connected components for efficiency
    if G.is_directed():
        # For directed graphs, use weakly connected components
        connected_components = list(nx.weakly_connected_components(G))
    else:
        connected_components = list(nx.connected_components(G))

    # Find the largest connected component (most useful for routing)
    largest_component = max(connected_components, key=len)
    print(f"Graph: {len(G.nodes)} nodes, {len(connected_components)} components, largest: {len(largest_component)} nodes")

    router_h = Router(heuristic=heuristic, CRS=meta_map["CRS"], CRS_map=meta_map["CRS_map"])

    for attempt in range(max_retries):
        try:
            # Generate random OD pair within the largest component with minimum distance
            gdf_random_points = generate_random_points_in_component(df, meta_map["CRS"], largest_component, G)

            origin_node, dest_node, origin_node_loc, dest_node_loc, gdf_coords = router_h.set_o_d_coords(G, gdf_random_points)

            # Double-check that both nodes are in the same component
            origin_component = None
            dest_component = None
            for comp in connected_components:
                if origin_node in comp:
                    origin_component = comp
                if dest_node in comp:
                    dest_component = comp

            if origin_component != dest_component:
                continue  # Different components, retry

            # Try to find the route
            path_fact, G_path_fact, df_path_fact = router_h.get_route(G, origin_node, dest_node, heuristic_f)

            # Check if the generated path forms a loop (start == end)
            if len(df_path_fact) > 0:
                first_edge = df_path_fact.iloc[0].geometry
                last_edge = df_path_fact.iloc[-1].geometry
                path_start = (first_edge.coords[0][0], first_edge.coords[0][1])
                path_end = (last_edge.coords[-1][0], last_edge.coords[-1][1])
                
                if path_start == path_end:
                    # Path forms a loop, retry with new OD pair
                    continue

            return {
                "path_fact": path_fact,
                "G_path_fact": G_path_fact,
                "df_path_fact": df_path_fact,
                "origin_node_loc": origin_node_loc,
                "dest_node_loc": dest_node_loc,
                "gdf_coords": gdf_coords,
                "meta_map": meta_map,
            }

        except ValueError as e:
            error_msg = str(e)
            # Route construction or path finding failed, retry with new OD pair
            continue

    # If all retries failed, raise an error
    raise ValueError(f"Failed to generate valid route after {max_retries} attempts")


def generate_random_points_in_component(df: gpd.GeoDataFrame, crs: str, component_nodes, G) -> gpd.GeoDataFrame:
    """
    Generate random OD points that are guaranteed to be in the same connected component.

    Args:
        df: Road network GeoDataFrame
        crs: Coordinate reference system
        component_nodes: Set of nodes in the target component
        G: NetworkX graph

    Returns:
        GeoDataFrame with origin and destination points
    """
    max_attempts = 100  # Maximum attempts to find valid OD pair

    for attempt in range(max_attempts):
        # Generate two random points
        bounds = df.total_bounds
        minx, miny, maxx, maxy = bounds

        origin_point = Point(uniform(minx, maxx), uniform(miny, maxy))
        dest_point = Point(uniform(minx, maxx), uniform(miny, maxy))

        # Find nearest nodes in the graph
        origin_node_loc = so.nearest_points(origin_point, sg.MultiPoint(list(G.nodes)))[1]
        dest_node_loc = so.nearest_points(dest_point, sg.MultiPoint(list(G.nodes)))[1]

        origin_node = (origin_node_loc.x, origin_node_loc.y)
        dest_node = (dest_node_loc.x, dest_node_loc.y)

        # Check if both nodes are in the target component
        if (origin_node in component_nodes and dest_node in component_nodes and
            origin_node != dest_node):
            # Create GeoDataFrame
            gdf_points = gpd.GeoDataFrame(
                {'coordinates': ['origin', 'destination']},
                geometry=[origin_node_loc, dest_node_loc],
                crs=crs,
            )
            return gdf_points

        # Continue searching for valid OD pair

    raise ValueError(f"Could not find valid OD pair in the connected component after {max_attempts} attempts")


def generate_route(
    df: gpd.GeoDataFrame,
    gdf_coords_loaded: gpd.GeoDataFrame,
    meta_data: Dict[str, Any],
    heuristic: str = "dijkstra",
    heuristic_f: str = "my_weight",
) -> Dict[str, Any]:
    """Legacy function for backward compatibility."""
    return generate_route_with_retry(df, meta_data, heuristic=heuristic, heuristic_f=heuristic_f)

    
def batch_generate_routes(
    num_routes: int,
    df: gpd.GeoDataFrame,
    meta_data: Dict[str, Any],
    output_dir: str,
) -> None:
    """Generate routes and save to output directory with connectivity checking."""
    os.makedirs(output_dir, exist_ok=True)

    # Track generated OD pairs to avoid duplicates
    generated_od_pairs = set()
    unique_routes_saved = 0

    for i in range(num_routes):
        try:
            print(f"  Generating route {i + 1}/{num_routes} (unique: {unique_routes_saved}/{num_routes})...")

            # Generate route with OD pair deduplication
            max_od_attempts = 100  # Maximum attempts to find unique OD pair
            for od_attempt in range(max_od_attempts):
                # Use the new connectivity-aware route generation
                route_data = generate_route_with_retry(df, meta_data)
                origin_node_loc, dest_node_loc = route_data["origin_node_loc"], route_data["dest_node_loc"]

                # Create OD pair identifier
                origin_coords = (origin_node_loc.x, origin_node_loc.y)
                dest_coords = (dest_node_loc.x, dest_node_loc.y)
                od_pair = tuple(sorted([origin_coords, dest_coords]))  # Sort to handle direction independence

                # Check if this OD pair was already generated
                if od_pair in generated_od_pairs:
                    if od_attempt == 0:  # Only print for first attempt
                        print(f"    OD pair already exists, trying new one...")
                    continue  # Try again with new random OD pair

                # OD pair is unique, proceed with route generation
                df_path_fact = route_data["df_path_fact"]
                gdf_coords = route_data["gdf_coords"]

                # Add to generated set
                generated_od_pairs.add(od_pair)

                # Save route file
                save_route_to_file(df_path_fact, output_dir, unique_routes_saved, crs=meta_data["map"]["CRS"])

                # Save CSV file
                csv_file_name = os.path.join(output_dir, f'route_{unique_routes_saved}_start_end.csv')
                gdf_combined = gpd.GeoDataFrame(
                    {
                        'coordinates': gdf_coords['coordinates'].tolist() + ['origin_node', 'destination_node'],
                        'geometry': gdf_coords['geometry'].tolist() + [origin_node_loc, dest_node_loc],
                    },
                    crs=gdf_coords.crs
                )
                gdf_combined.to_csv(csv_file_name, sep=';', index=False)

                unique_routes_saved += 1
                print(f"  ✓ Route {unique_routes_saved - 1} saved successfully (OD: {origin_coords} -> {dest_coords})")
                break  # Success, exit OD attempt loop

            else:
                # Failed to find unique OD pair after max attempts
                print(f"  ✗ Could not find unique OD pair after {max_od_attempts} attempts")
                continue

        except Exception as e:
            print(f"  ✗ Error generating route {i}: {e}")
            continue

def plot_all_routes(output_dir: str, data_directory: str, graph_id: str = 'default_graph', max_files: int = 2) -> None:
    """Plot routes with simplified naming."""
    data = load_data(data_directory, graph_id)
    df, meta_data = data["df"], data["meta_data"]
    meta_map = meta_data["map"]

    files = os.listdir(output_dir)
    route_files = [f for f in files if f.endswith('.gpkg')]

    if not route_files:
        print("No valid route files found in the folder.")
        return

    processed_count = 0

    for route_file in route_files:
        if processed_count >= max_files:
            print(f"Processed {max_files} files. Stopping further processing.")
            break
        try:
            file_path = os.path.join(output_dir, route_file)

            # Extract route_id from filename: p_route_{route_id}.gpkg
            route_id = int(route_file.split('_')[-1].split('.')[0])

            route_data = load_route_from_file(output_dir, route_id, meta_map)

            if route_data.empty or route_data.geometry.isnull().all():
                print(f"Invalid geometry in file: {file_path}")
                continue

            gdf_coords_path = os.path.join(output_dir, f'route_{route_id}_start_end.csv')
            if not os.path.exists(gdf_coords_path):
                print(f"Corresponding CSV file not found for route: {route_file}")
                continue

            gdf_coords_loaded = pd.read_csv(gdf_coords_path, sep=';')

            gdf_coords = gdf_coords_loaded[['coordinates', 'geometry']]
            gdf_coords['geometry'] = gdf_coords['geometry'].apply(wkt.loads)
            gdf_coords = gpd.GeoDataFrame(gdf_coords, geometry='geometry')

            origin_point = gdf_coords.loc[gdf_coords['coordinates'] == 'origin', 'geometry'].values[0]
            dest_point = gdf_coords.loc[gdf_coords['coordinates'] == 'destination', 'geometry'].values[0]
            origin_node_loc = gdf_coords.loc[gdf_coords['coordinates'] == 'origin_node', 'geometry'].values[0]
            dest_node_loc = gdf_coords.loc[gdf_coords['coordinates'] == 'destination_node', 'geometry'].values[0]
            gdf_coords = gdf_coords[gdf_coords['coordinates'].isin(['origin', 'destination'])]

            print(f"Plotting route from file: {route_file}")
            plot_route_with_network(
                df=df,
                df_path_fact=route_data,
                gdf_coords=gdf_coords,
                origin_node_loc=origin_node_loc,
                dest_node_loc=dest_node_loc,
                meta_map=meta_map,
                buffer_radius=70,
            )

            processed_count += 1

        except Exception as e:
            print(f"Error processing route file {route_file}: {e}")
            continue

def count_routes_and_remove_duplicates(output_dir: str) -> int:
    """Count unique routes and remove duplicates with simplified naming."""
    files = os.listdir(output_dir)
    route_files = [f for f in files if f.endswith('.gpkg')]

    if not route_files:
        print("No valid route files found in the folder.")
        return 0

    route_counter = Counter()
    geometry_to_files = {}

    for route_file in route_files:
        try:
            file_path = os.path.join(output_dir, route_file)
            route_data = gpd.read_file(file_path)

            if route_data.empty or route_data.geometry.isnull().all():
                print(f"Invalid geometry in file: {file_path}")
                continue

            route_geometry = route_data.geometry.apply(lambda geom: geom.wkt).tolist()
            route_geometry_tuple = tuple(sorted(route_geometry))

            route_counter[route_geometry_tuple] += 1
            geometry_to_files.setdefault(route_geometry_tuple, []).append(route_file)
        except Exception as e:
            print(f"Error processing route file {route_file}: {e}")
            continue

    for geometry, files in geometry_to_files.items():
        if len(files) > 1:
            for duplicate_file in files[1:]:
                duplicate_gpkg_path = os.path.join(output_dir, duplicate_file)
                # Extract route_id from filename: p_route_{route_id}.gpkg
                route_id = duplicate_file.split('_')[-1].split('.')[0]
                duplicate_csv_path = os.path.join(output_dir, f'route_{route_id}_start_end.csv')

                try:
                    if os.path.exists(duplicate_gpkg_path):
                        print(f"Deleting duplicate GeoPackage file: {duplicate_gpkg_path}")
                        os.remove(duplicate_gpkg_path)

                    if os.path.exists(duplicate_csv_path):
                        print(f"Deleting corresponding CSV file: {duplicate_csv_path}")
                        os.remove(duplicate_csv_path)
                except Exception as delete_error:
                    print(f"Error deleting files: {delete_error}")

    return len(route_counter)

# Splitting dataset into train, validation and test
def split_dataset_with_matching_files(input_dir, data_directory, graph_id, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset files (.gpkg and .csv) into train, validation, and test categories
    into graph-specific raw_data directories.

    Parameters:
        input_dir (str): Path to the input directory containing .gpkg and .csv files.
        data_directory (str): Path to the data directory.
        graph_id (str): Graph identifier for multi-graph support.
        train_ratio (float): Proportion of files to be used for training.
        val_ratio (float): Proportion of files to be used for validation.
        test_ratio (float): Proportion of files to be used for testing.
    """
    # Ensure ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1."

    # Get list of .gpkg files
    gpkg_files = [f for f in os.listdir(input_dir) if f.endswith('.gpkg')]

    # Match corresponding .csv files for each .gpkg file
    matched_files = []
    for gpkg_file in gpkg_files:
        # Extract route_id from filename: p_route_{route_id}.gpkg
        route_id = gpkg_file.split('_')[-1].split('.')[0]
        csv_file = f'route_{route_id}_start_end.csv'
        if os.path.exists(os.path.join(input_dir, csv_file)):
            matched_files.append((gpkg_file, csv_file))

    # Shuffle matched files randomly
    random.shuffle(matched_files)

    # Compute split indices
    total_files = len(matched_files)
    train_end_idx = int(total_files * train_ratio)
    val_end_idx = train_end_idx + int(total_files * val_ratio)

    # Split files into categories
    train_files = matched_files[:train_end_idx]
    val_files = matched_files[train_end_idx:val_end_idx]
    test_files = matched_files[val_end_idx:]

    # Create output directories in graph-specific raw_data folder
    raw_data_dir = os.path.join(data_directory, 'graph_data', graph_id, 'raw_data')
    for category in ['train', 'val', 'test']:
        category_dir = os.path.join(raw_data_dir, category)
        os.makedirs(category_dir, exist_ok=True)

    # Copy files into respective directories
    for category, file_pairs in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
        for gpkg_file, csv_file in file_pairs:
            shutil.copy(os.path.join(input_dir, gpkg_file), os.path.join(raw_data_dir, category, gpkg_file))
            shutil.copy(os.path.join(input_dir, csv_file), os.path.join(raw_data_dir, category, csv_file))

    print(f"Dataset successfully split into categories:")
    raw_data_dir = os.path.join(data_directory, 'graph_data', 'default_graph', 'raw_data')
    print(f"  Train: {len(train_files)} file pairs -> {os.path.join(raw_data_dir, 'train')}")
    print(f"  Validation: {len(val_files)} file pairs -> {os.path.join(raw_data_dir, 'val')}")
    print(f"  Test: {len(test_files)} file pairs -> {os.path.join(raw_data_dir, 'test')}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate routes and split into train/val/test')
    parser.add_argument('--data_dir', type=str, default='../data',
                       help='Data directory path (default: ../data)')
    parser.add_argument('--graph_id', type=str, default='default_graph',
                       help='Graph identifier for multi-graph support (default: default_graph)')
    parser.add_argument('--num_routes', type=int, default=100,
                       help='Number of routes to generate (default: 10000)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Training set ratio (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation set ratio (default: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Test set ratio (default: 0.1)')
    
    args = parser.parse_args()

    data_directory = args.data_dir
    graph_id = args.graph_id
    num_routes = args.num_routes
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    test_ratio = args.test_ratio
    
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    print("=" * 80)
    print("ROUTE GENERATION AND SPLITTING")
    print("=" * 80)
    print(f"Data directory: {data_directory}")
    print(f"Number of routes: {num_routes}")
    print(f"Split ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
    print()

    # Create temporary directory for initial route generation
    temp_dir = os.path.join(data_directory, 'temp_routes')
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Load data from graph-specific directory
        print("📂 Loading map data...")
        data = load_data(data_directory, graph_id)
        df, meta_data = data["df"], data["meta_data"]
        print(f"✅ Loaded map data: {len(df)} edges")

        # Check if cache exists, generate if not
        cache_dir = os.path.join(data_directory, 'graph_data', graph_id, 'cache')
        cache_exists = os.path.exists(cache_dir) and os.path.exists(os.path.join(cache_dir, 'graph_cache.pkl'))

        if not cache_exists:
            print(f"📦 Cache not found for graph '{graph_id}', generating cache files...")
            generate_graph_cache(df, meta_data, data_directory, graph_id)
            print("✅ Cache files generated!")
        else:
            print(f"📦 Using existing cache for graph '{graph_id}'")

        # Batch generate unique routes
        print(f"\n🔄 Generating {num_routes} unique routes...")
        batch_generate_routes(num_routes, df, meta_data, temp_dir)

        # Split dataset into graph-specific directories
        print(f"\n📊 Splitting dataset into train/val/test for graph '{graph_id}'...")
        split_dataset_with_matching_files(temp_dir, data_directory, graph_id, train_ratio, val_ratio, test_ratio)

        print("\n" + "=" * 80)
        print("✅ Route generation and splitting completed!")
        print("=" * 80)

    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            print(f"\n🧹 Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)


def generate_graph_cache(df: gpd.GeoDataFrame, meta_data: Dict[str, Any], data_directory: str, graph_id: str):
    """
    Generate all necessary cache files for graph preprocessing.

    Args:
        df: Road network GeoDataFrame
        meta_data: Map metadata
        data_directory: Base data directory
        graph_id: Graph identifier
    """
    from utils.graph_cache import GraphCache
    import momepy

    # Create cache directory
    cache_dir = os.path.join(data_directory, 'graph_data', graph_id, 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    print(f"   Creating cache files in: {cache_dir}")

    # Build graph from original data WITHOUT applying user constraints
    # This ensures we cache all edges (data-driven, not rule-driven)
    G_con, _ = create_network_graph(df, use_directed=False)
    
    # Use the full undirected graph for caching (no filtering by user constraints)
    G = G_con

    # Get coordinate bounds for normalization
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

    # Normalize coordinates
    def normalize_coord(coord):
        x_norm = (coord[0] - coord_stats['x_min']) / coord_stats['x_range'] if coord_stats['x_range'] > 0 else 0.5
        y_norm = (coord[1] - coord_stats['y_min']) / coord_stats['y_range'] if coord_stats['y_range'] > 0 else 0.5
        return (x_norm, y_norm)

    # Create normalized graph
    G_norm = nx.Graph()
    for u, v, edge_data in G.edges(data=True):
        u_norm = normalize_coord(u)
        v_norm = normalize_coord(v)
        G_norm.add_edge(u_norm, v_norm, **edge_data)

    # First pass: collect raw data for statistics
    edge_lengths = []
    curb_heights = []

    # Create mapping from original edge coordinates to dataframe indices for faster lookup
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
            # Ensure NaN values are converted to 0.0
            if pd.isna(curb_height):
                curb_height = 0.0
            curb_heights.append(curb_height)
        else:
            curb_heights.append(0.0)

    # Calculate normalization parameters from collected data
    max_length = max(edge_lengths) if edge_lengths else 1.0
    curb_max = max(curb_heights) if curb_heights else 0.04  # Use 0.04 as fallback if no curb data

    # Second pass: build edge mappings and normalized features
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
            # Ensure NaN values are converted to 0.0
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
                path_type = 0  # default to walk
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

    # Save individual cache files
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

    print(f"   ✅ Cache files created: {len(G_norm.nodes())} nodes, {len(G_norm.edges())} edges")


if __name__ == "__main__":
    main()
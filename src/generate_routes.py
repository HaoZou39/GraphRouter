import os
import json
import pandas as pd
import geopandas as gpd
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

def save_route_to_file(gdf_route: gpd.GeoDataFrame, store_path: str, route_name: str, route_id: int) -> None:
    file_path = f'{store_path}p_route_{route_name}_{route_id}.gpkg'
    gdf_route.to_file(file_path, driver='GPKG')
    print(f"Route saved to {file_path}")

def load_route_from_file(store_path: str, route_name: str, route_id: int, meta_map: dict) -> gpd.GeoDataFrame:
    file_path = f'{store_path}p_route_{route_name}_{route_id}.gpkg'
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

def load_data(map_name: str, map_id: int) -> Dict[str, Any]:
    basic_network_path = f'./examples/{map_name}/network_demo_walk_{map_id}'
    df = gpd.read_file(basic_network_path)

    meta_data_path = f'./examples/{map_name}/metadata_{map_name}_{map_id}.json'
    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)

    return {
        "df": df,
        "meta_data": meta_data,
    }

def generate_route(
    df: gpd.GeoDataFrame,
    gdf_coords_loaded: gpd.GeoDataFrame,
    meta_data: Dict[str, Any],
    heuristic: str = "dijkstra",
    heuristic_f: str = "my_weight",
) -> Dict[str, Any]:
    user_model = meta_data["user_model"]
    meta_map = meta_data["map"]
    df_copy = deepcopy(df)
    df_copy = handle_weight(df_copy, user_model)
    _, G = create_network_graph(df_copy)
    router_h = Router(heuristic=heuristic, CRS=meta_map["CRS"], CRS_map=meta_map["CRS_map"])
    origin_node, dest_node, origin_node_loc, dest_node_loc, gdf_coords = router_h.set_o_d_coords(G, gdf_coords_loaded)
    path_fact, G_path_fact, df_path_fact = router_h.get_route(G, origin_node, dest_node, heuristic_f)
    return {
        "path_fact": path_fact,
        "G_path_fact": G_path_fact,
        "df_path_fact": df_path_fact,
        "origin_node_loc": origin_node_loc,
        "dest_node_loc": dest_node_loc,
        "gdf_coords": gdf_coords,
        "meta_map": meta_map,
    }

def generate_random_points(df: gpd.GeoDataFrame, crs: str, num_routes: int) -> gpd.GeoDataFrame:
    bounds = df.total_bounds  
    minx, miny, maxx, maxy = bounds

    random_points = [
        Point(uniform(minx, maxx), uniform(miny, maxy)) for _ in range(2 * num_routes)
    ]

    gdf_random_points = gpd.GeoDataFrame(
        {'coordinates': ['origin', 'destination'] * num_routes},  
        geometry=random_points,
        crs=crs,
    )
    return gdf_random_points
    
def batch_generate_routes(
    num_routes: int,
    df: gpd.GeoDataFrame,
    meta_data: Dict[str, Any],
    output_dir: str,
    route_name: str,
) -> None:
    crs = meta_data["map"]["CRS"]
    gdf_random_points = generate_random_points(df, crs, num_routes)

    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_routes):
        try:
            gdf_coords_loaded = gdf_random_points.iloc[2 * i:2 * i + 2]

            route_data = generate_route(df, gdf_coords_loaded, meta_data)
            origin_node_loc, dest_node_loc = route_data["origin_node_loc"], route_data["dest_node_loc"]
            df_path_fact = route_data["df_path_fact"]

            save_route_to_file(df_path_fact, output_dir, route_name, i)
            csv_file_name = os.path.join(output_dir, f'route_{route_name}_{i}_start_end.csv')
            gdf_combined = gpd.GeoDataFrame(
                {
                    'coordinates': gdf_coords_loaded['coordinates'].tolist() + ['origin_node', 'destination_node'],
                    'geometry': gdf_coords_loaded['geometry'].tolist() + [origin_node_loc, dest_node_loc],
                },
                crs=gdf_coords_loaded.crs
            )
            gdf_combined.to_csv(csv_file_name, sep=';', index=False)
        except Exception as e:
            print(f"Error generating route {i}: {e}")
            continue

def plot_all_routes(route_name: str, output_dir: str, map_name: str, map_id: int) -> None:
    data = load_data(map_name, map_id)
    df, meta_data = data["df"], data["meta_data"]
    meta_map = meta_data["map"]

    files = os.listdir(output_dir)
    route_files = [f for f in files if f.endswith('.gpkg')]

    if not route_files:
        print("No valid route files found in the folder.")
        return
    
    max_files_to_process = 2

    processed_count = 0

    for route_file in route_files:
        if processed_count >= max_files_to_process:
            print(f"Processed {max_files_to_process} files. Stopping further processing.")
            break
        try:
            file_path = os.path.join(output_dir, route_file)

            route_id = int(route_file.split('_')[-1].split('.')[0])

            route_data = load_route_from_file(output_dir, route_name, route_id, meta_map)

            if route_data.empty or route_data.geometry.isnull().all():
                print(f"Invalid geometry in file: {file_path}")
                continue

            gdf_coords_path = os.path.join(output_dir, f'route_{route_name}_{route_id}_start_end.csv')
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

def count_routes_and_remove_duplicates(route_name: str, output_dir: str, map_name: str, map_id: int) -> int:
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
                duplicate_csv_path = os.path.join(output_dir, f'route_{route_name}_{duplicate_file.split("_")[-1].split(".")[0]}_start_end.csv')

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
def split_dataset_with_matching_files(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset files (.gpkg and .csv) into train, validation, and test categories

    while ensuring that corresponding .gpkg and .csv files are in the same dataset.

    Parameters:
        input_dir (str): Path to the input directory containing .gpkg and .csv files.
        output_dir (str): Path to the output directory where files will be categorized.
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
        route_id = gpkg_file.split('_')[-1].split('.')[0]  # Extract route_id from filename

        csv_file = f'route_demo_walk_{route_id}_start_end.csv'
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

    # Create output directories

    for category in ['train', 'val', 'test']:
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)

    # Move files into respective directories

    for category, file_pairs in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
        for gpkg_file, csv_file in file_pairs:
            shutil.copy(os.path.join(input_dir, gpkg_file), os.path.join(output_dir, category, gpkg_file))
            shutil.copy(os.path.join(input_dir, csv_file), os.path.join(output_dir, category, csv_file))

    print(f"Dataset successfully split into categories:")
    print(f"  Train: {len(train_files)} file pairs")
    print(f"  Validation: {len(val_files)} file pairs")
    print(f"  Test: {len(test_files)} file pairs")

def main():
    map_name = "demo_walk"
    map_id = 0
    route_name = "demo_walk"

    num_routes = 10  # Change this to 1000 or 10000 as needed

    output_dir = f'./examples/{route_name}/test/'
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    data = load_data(map_name, map_id)
    df, meta_data = data["df"], data["meta_data"]

    # Batch generate routes
    batch_generate_routes(num_routes, df, meta_data, output_dir, route_name)

    unique_route_count = count_routes_and_remove_duplicates(route_name, output_dir, map_name, map_id)
    
    print(f"Total number of different routes: {unique_route_count}")
    
    input_dir = './examples/demo_walk/test/'

    output_dir = './examples/demo_walk/split/'

    split_dataset_with_matching_files(input_dir, output_dir, train_ratio=0.8, val_ratio=0.10, test_ratio=0.10)

if __name__ == "__main__":
    main()
import os
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import json
from shapely import wkt

def load_route_from_file(file_path: str, meta_map: dict) -> gpd.GeoDataFrame:
    """
    Load the route (GeoDataFrame) from a file in GeoPackage format.
    """
    loaded_route = gpd.read_file(file_path)
    loaded_route = loaded_route.set_crs(meta_map["CRS"], allow_override=True)
    if loaded_route.empty or loaded_route.geometry.isnull().all():
        raise ValueError(f"Loaded route data is empty or has invalid geometry: {file_path}")  
    return loaded_route

def process_gpkg(file_path, route_id, meta_data):
    gdf = load_route_from_file(file_path, meta_data)
    base_name = os.path.basename(file_path)
    csv_name = base_name.replace('p_', '').replace('.gpkg', '_start_end.csv')
    start_end_csv_path = os.path.join(os.path.dirname(file_path), csv_name)

    if os.path.exists(start_end_csv_path):
        gdf_coords_loaded = pd.read_csv(start_end_csv_path, sep=';')

        # Process origin/destination points and nodes
        gdf_coords = gdf_coords_loaded[['coordinates', 'geometry']]
        gdf_coords['geometry'] = gdf_coords['geometry'].apply(wkt.loads)
        gdf_coords = gpd.GeoDataFrame(gdf_coords, geometry='geometry')

        origin_node_loc = gdf_coords.loc[gdf_coords['coordinates'] == 'origin_node', 'geometry'].values[0]
        dest_node_loc = gdf_coords.loc[gdf_coords['coordinates'] == 'destination_node', 'geometry'].values[0]
        gdf['origin_node_x'] = origin_node_loc.x
        gdf['origin_node_y'] = origin_node_loc.y
        gdf['destination_node_x'] = dest_node_loc.x
        gdf['destination_node_y'] = dest_node_loc.y
    else:
        gdf['origin_node_x'], gdf['origin_node_y'] = None, None
        gdf['destination_node_x'], gdf['destination_node_y'] = None, None

    # process features
    gdf['crossing'] = gdf['crossing'].apply(lambda x: 1 if x == 'Yes' else 0)
    gdf['path_type'] = gdf['path_type'].apply(lambda x: 1 if x == 'walk' else 0)

    num_cols = ['length', 'obstacle_free_width_float', 'curb_height_max']
    gdf[num_cols] = gdf[num_cols].astype(float).fillna(0)
    # scaler = MinMaxScaler()
    # gdf[num_cols] = scaler.fit_transform(gdf[num_cols])

    def linestring_to_vec(geom):
        coords = list(geom.coords)
        return np.array([coords[0][0], coords[0][1], coords[-1][0], coords[-1][1]])
    
    gdf['geom_vec'] = pd.Series(gdf['geometry']).astype(object).apply(linestring_to_vec)

    cur_node_x = [gdf['origin_node_x'].iloc[0]]
    cur_node_y = [gdf['origin_node_y'].iloc[0]]
    for i, row in gdf.iterrows():
        prev_cur_x = cur_node_x[-1]
        prev_cur_y = cur_node_y[-1]
        x1, y1, x2, y2 = row['geom_vec']
        if np.isclose([x1, y1], [prev_cur_x, prev_cur_y]).all():
            cur_x, cur_y = x2, y2
        else:
            cur_x, cur_y = x1, y1
        cur_node_x.append(cur_x)
        cur_node_y.append(cur_y)
    gdf['cur_node_x'] = cur_node_x[1:]
    gdf['cur_node_y'] = cur_node_y[1:]

    def row_to_feature(row):
        # feature_vec structure:
        # [
        #   start_x, start_y, end_x, end_y,         # coordinates of the linestring endpoints
        #   crossing, path_type,                    # categorical features (already label-encoded)
        #   length, obstacle_free_width_float, curb_height_max  # normalized numerical features
        #   origin_node_x, origin_node_y,           # coordinates of the origin node
        #   destination_node_x, destination_node_y, # coordinates of the destination node
        #   cur_node_x, cur_node_y                   # coordinates of the current node
        # ]
        return np.concatenate([
            row['geom_vec'], [row['crossing'], row['path_type']], row[num_cols].values,
            [row['origin_node_x'], row['origin_node_y'], row['destination_node_x'], row['destination_node_y']],
            [row['cur_node_x'], row['cur_node_y']]
        ])

    gdf['feature_vec'] = gdf.apply(row_to_feature, axis=1)

    gdf['route_id'] = route_id
    gdf['step_id'] = np.arange(len(gdf))
    return gdf[['route_id', 'step_id', 'feature_vec']]

def process_split(split_name, data_directory, meta_data):
    split_dir = os.path.join(data_directory, split_name)
    all_feature_rows = []
    for fname in os.listdir(split_dir):
        if fname.endswith('.gpkg'):
            route_id = os.path.splitext(fname)[0].split('_')[-1]
            file_path = os.path.join(split_dir, fname)
            df = process_gpkg(file_path, route_id, meta_data)
            all_feature_rows.append(df)
    if all_feature_rows:
        all_features = pd.concat(all_feature_rows, ignore_index=True)
        all_features.to_pickle(os.path.join(data_directory, f'sampled_{split_name}.df'))
        print(f"Saved {split_name} features to sampled_{split_name}.df")
    else:
        print(f"No .gpkg files found in {split_dir}")

if __name__ == '__main__':
    data_directory = '../data/'
    if not os.path.exists(data_directory):
        raise FileNotFoundError(f"Data directory does not exist: {data_directory}")
    meta_data_path = os.path.join(data_directory, 'metadata.json')
    if not os.path.exists(meta_data_path):
        raise FileNotFoundError(f"Metadata file does not exist: {meta_data_path}")
    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)

    for split in ['train', 'val', 'test']:
        process_split(split, data_directory, meta_data)
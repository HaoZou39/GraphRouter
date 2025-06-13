import os
import numpy as np
import tensorflow as tf
import warnings
import momepy
import json
import geopandas as gpd
from typing import Dict, Any
import pickle
import networkx as nx
import pandas as pd

def to_pickled_df(data_directory, **kwargs):
    for name, df in kwargs.items():
        df.to_pickle(os.path.join(data_directory, name + '.df'))

def pad_history(itemlist,length,pad_item):
    if len(itemlist)>=length:
        return itemlist[-length:]
    if len(itemlist)<length:
        temp = [pad_item] * (length-len(itemlist))
        itemlist.extend(temp)
        return itemlist

def extract_axis_1(data, ind):
    """
    Get specified elements along the first axis of tensor.
    :param data: Tensorflow tensor that will be subsetted.
    :param ind: Indices to take (one for each element along axis 0 of data).
    :return: Subsetted tensor.
    """

    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)

    return res

def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs

def calculate_hit(sorted_list,topk,true_items,rewards,r_click,total_reward,hit_click,ndcg_click,hit_purchase,ndcg_purchase):
    for i in range(len(topk)):
        rec_list = sorted_list[:, -topk[i]:]
        for j in range(len(true_items)):
            if true_items[j] in rec_list[j]:
                rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])
                total_reward[i] += rewards[j]
                if rewards[j] == r_click:
                    hit_click[i] += 1.0
                    ndcg_click[i] += 1.0 / np.log2(rank + 1)
                else:
                    hit_purchase[i] += 1.0
                    ndcg_purchase[i] += 1.0 / np.log2(rank + 1)

def create_network_graph(G_df):
    # Make bi-directionality of sidewalks (and not of bike paths) explicit
    G_df['oneway'] = np.where(G_df['bikepath_id'].isna(), False, True)

    # Create graphs that take directionality into account
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        G = momepy.gdf_to_nx(G_df, approach="primal", multigraph=True, directed=True, length="length", oneway_column="oneway")
    
    # Generate nodes position data
    for node in G.nodes:
        G.nodes[node]['pos'] = node

    return G

def load_map_data(data_directory: str) -> Dict[str, Any]:
    """
    Load the required map data files.
    """
    # Load network data
    network_file_path = os.path.join(data_directory, 'network_map')
    df = gpd.read_file(network_file_path)

    # Load metadata
    meta_data_path = os.path.join(data_directory, 'metadata.json')
    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)

    return {
        "df": df,
        "meta_data": meta_data,
    }

def round_node(node, ndigits=5):
    return (round(node[0], ndigits), round(node[1], ndigits))

def get_valid_actions(state_seq, G):
    actions_list = []
    n = len(state_seq)
    for i, state in enumerate(state_seq):
        arr = np.array(state, dtype=float)
        if np.all(arr == 0):
            actions_list.append([])
            continue
        node1 = round_node((arr[0], arr[1]))
        node2 = round_node((arr[2], arr[3]))
        if n == 1:
            curr_node = node1
        elif i == 0:
            next_arr = np.array(state_seq[i+1], dtype=float)
            next_nodes = [round_node((next_arr[0], next_arr[1])), round_node((next_arr[2], next_arr[3]))]
            if node1 in next_nodes:
                curr_node = node2
            else:
                curr_node = node1
        else:
            prev_arr = np.array(state_seq[i-1], dtype=float)
            prev_nodes = [round_node((prev_arr[0], prev_arr[1])), round_node((prev_arr[2], prev_arr[3]))]
            if node1 in prev_nodes:
                curr_node = node2
            else:
                curr_node = node1
        if curr_node in G:
            neighbors = list(G.neighbors(curr_node))
            actions_list.append(neighbors)
        else:
            actions_list.append([])
    return actions_list
    
def create_network_graph_with_features(gdf):
    G = nx.Graph()
    edge_id_map = {}
    edge_feature_list = []
    
    for idx, row in gdf.iterrows():
        geom = row['geometry']
        if geom is None or geom.is_empty:
            continue
            
        start_point = (geom.coords[0][0], geom.coords[0][1])
        end_point = (geom.coords[-1][0], geom.coords[-1][1])
        
        static_feature = [
            row['length'] if not pd.isna(row['length']) else 0,
            row['obstacle_free_width_float'] if not pd.isna(row['obstacle_free_width_float']) else 0,
            row['curb_height_max'] if not pd.isna(row['curb_height_max']) else 0,
            1 if row['crossing'] == 'Yes' else 0,
            1 if row['path_type'] == 'walk' else 0
        ]
        
        G.add_edge(start_point, end_point, 
                  static_feature=static_feature,
                  edge_index=idx,
                  start_point=start_point,
                  end_point=end_point)
        
        edge_id_map[(start_point, end_point)] = idx
        edge_feature_list.append(static_feature)

    return G, edge_id_map, edge_feature_list

def get_dynamic_feature(state, edge, G):
    cur_node = (state[-2], state[-1])  # 当前节点坐标
    edge_data = G.edges[edge]
    
    feature_vec = [
        edge_data['start_point'][0], edge_data['start_point'][1],  # start_x, start_y
        edge_data['end_point'][0], edge_data['end_point'][1],      # end_x, end_y
        *edge_data['static_feature'],                              # 静态特征
        edge_data['start_point'][0], edge_data['start_point'][1],  # origin_node_x, origin_node_y
        edge_data['end_point'][0], edge_data['end_point'][1],      # destination_node_x, destination_node_y
        cur_node[0], cur_node[1]                                   # cur_node_x, cur_node_y
    ]
    
    return feature_vec
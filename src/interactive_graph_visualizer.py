import os
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import argparse

class InteractiveGraphVisualizer:
    def __init__(self, graph_id="default_graph", base_data_dir="../data/graph_data"):
        self.graph_id = graph_id
        self.base_data_dir = base_data_dir
        self.graph_dir = Path(base_data_dir) / graph_id
        self.G = None
        self.graph_cache = None
        self.fig = None
        self.ax = None
        self.annotation = None
        self.load_data()
        
    def load_data(self):
        """从指定graph_id加载图数据"""
        cache_dir = self.graph_dir / "cache"
        
        # 初始化edge_feature_list为None（确保属性存在）
        self.edge_feature_list = None

        print(f"Loading graph data from {self.graph_dir} ...")

        # 加载graph_cache.pkl（统一格式）
        cache_file = cache_dir / "graph_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.graph_cache = pickle.load(f)
                print(f"Loaded graph_cache format: {self.graph_cache.get_node_count()} nodes, {self.graph_cache.get_edge_count()} edges")

                # 同时加载edge_feature_list（归一化数据，用于特殊边检测）
                edge_feature_file = cache_dir / "edge_feature_list.pkl"
                if edge_feature_file.exists():
                    try:
                        with open(edge_feature_file, 'rb') as f:
                            self.edge_feature_list = pickle.load(f)
                        print(f"✓ Also loaded edge_feature_list with {len(self.edge_feature_list)} features")
                    except Exception as e:
                        print(f"✗ Failed to load edge_feature_list: {e}")
                        print("Continuing without edge_feature_list (special edge highlighting will be disabled)")
                        self.edge_feature_list = None
                else:
                    print(f"✗ edge_feature_list.pkl not found at {edge_feature_file}")
                    print("Will use graph_cache.edge_attrs instead")

                return
            except Exception as e:
                print(f"Failed to load graph_cache: {e}")
                raise
        else:
            print(f"Cache directory: {cache_dir}")
            print(f"Cache directory exists: {cache_dir.exists()}")
            if cache_dir.exists():
                available_files = [f.name for f in cache_dir.glob("*") if f.is_file()]
                print(f"Available files: {available_files}")
            else:
                available_files = []
            raise FileNotFoundError(f"Graph cache not found in {cache_dir}. Please run generate_routes.py first. Available files: {available_files}")
    
    def get_edge_features(self, edge_id, debug=False):
        """获取边的特征信息"""
        # 优先使用edge_feature_list（与create_visualization中的特殊边判断逻辑保持一致）
        if hasattr(self, 'edge_feature_list') and self.edge_feature_list and edge_id < len(self.edge_feature_list):
            features = self.edge_feature_list[edge_id]
            result = {
                'edge_id': edge_id,
                'length_norm': features[0] if len(features) > 0 else 'N/A',
                'width': features[1] if len(features) > 1 else 'N/A',
                'curb_norm': features[2] if len(features) > 2 else 'N/A',
                'crossing': features[3] if len(features) > 3 else 'N/A',
                'path_type': features[4] if len(features) > 4 else 'N/A',
                'all_features': features,
                'data_source': 'edge_feature_list'  # 标记数据来源
            }
            
            if debug:
                print(f"[DEBUG] Edge {edge_id} from edge_feature_list: curb_norm={features[2]}")
            
            # 尝试从graph_cache获取增强属性（如果有）
            if self.graph_cache and hasattr(self.graph_cache, 'edge_attrs') and edge_id in self.graph_cache.edge_attrs:
                attrs = self.graph_cache.edge_attrs[edge_id]
                result['width_bin'] = attrs.get('width_bin', 'N/A')
                result['curb_level'] = attrs.get('curb_level', 'N/A')
                result['confidence'] = attrs.get('confidence', 'N/A')
                result['evidence_type'] = attrs.get('evidence_type', 'N/A')
            
            return result
        
        # 回退到graph_cache.edge_attrs（如果edge_feature_list不可用）
        if self.graph_cache and hasattr(self.graph_cache, 'edge_attrs') and edge_id in self.graph_cache.edge_attrs:
            attrs = self.graph_cache.edge_attrs[edge_id]
            result = {
                'edge_id': edge_id,
                'length_norm': attrs.get('length_norm', 'N/A'),
                'width': attrs.get('width', 'N/A'),
                'curb_norm': attrs.get('curb_norm', 'N/A'),
                'crossing': attrs.get('crossing', 'N/A'),
                'path_type': attrs.get('path_type', 'N/A'),
                # 增强属性（如果有）
                'width_bin': attrs.get('width_bin', 'N/A'),
                'curb_level': attrs.get('curb_level', 'N/A'),
                'confidence': attrs.get('confidence', 'N/A'),
                'evidence_type': attrs.get('evidence_type', 'N/A'),
                'all_attrs': attrs,
                'data_source': 'graph_cache.edge_attrs'  # 标记数据来源
            }
            
            if debug:
                print(f"[DEBUG] Edge {edge_id} from graph_cache.edge_attrs: curb_norm={attrs.get('curb_norm', 'N/A')}")
            
            return result

        if debug:
            print(f"[DEBUG] Edge {edge_id} not found in any data source!")
        return None
    
    def on_edge_hover(self, event):
        """鼠标悬停在边上时显示信息"""
        if event.inaxes != self.ax:
            return
            
        if self.annotation:
            self.annotation.remove()
            self.annotation = None
            
        x, y = event.xdata, event.ydata
        min_dist = float('inf')
        closest_edge = None
        
        # 获取图对象（使用与create_visualization相同的逻辑）
        if self.graph_cache and hasattr(self.graph_cache, 'G') and self.graph_cache.G is not None:
            graph = self.graph_cache.G
        elif self.graph_cache and hasattr(self.graph_cache, 'node_coords') and self.graph_cache.node_coords:
            # 构建临时图用于悬停检测（只在第一次调用时构建）
            if not hasattr(self, '_hover_graph'):
                print("Building hover detection graph...")
                self._hover_graph = nx.Graph()
                for node_id, coord in self.graph_cache.node_coords.items():
                    self._hover_graph.add_node(coord)
                for edge_id, (u_id, v_id) in self.graph_cache.edge_id_to_nodes.items():
                    if u_id in self.graph_cache.node_coords and v_id in self.graph_cache.node_coords:
                        u_coord = self.graph_cache.node_coords[u_id]
                        v_coord = self.graph_cache.node_coords[v_id]
                        self._hover_graph.add_edge(u_coord, v_coord)
                print("Hover graph ready")
            graph = self._hover_graph
        else:
            graph = self.G

        for edge in graph.edges():
            start_node = edge[0]
            end_node = edge[1]
            edge_mid_x = (start_node[0] + end_node[0]) / 2
            edge_mid_y = (start_node[1] + end_node[1]) / 2
            dist = np.sqrt((x - edge_mid_x)**2 + (y - edge_mid_y)**2)

            if dist < min_dist and dist < 0.1:
                min_dist = dist
                closest_edge = edge
        
        if closest_edge:
            # 在edge_id_to_nodes中查找对应的edge_id
            edge_id = None
            if self.graph_cache and hasattr(self.graph_cache, 'edge_id_to_nodes') and self.graph_cache.edge_id_to_nodes:
                # 新格式：通过edge_id_to_nodes反向查找
                # closest_edge是坐标元组，需要先转换为节点ID
                start_coord, end_coord = closest_edge
                # 从node_coords反向查找节点ID
                coord_to_id = {coord: nid for nid, coord in self.graph_cache.node_coords.items()}

                if start_coord in coord_to_id and end_coord in coord_to_id:
                    start_id = coord_to_id[start_coord]
                    end_id = coord_to_id[end_coord]
                    node_pair = (start_id, end_id)

                    # 在edge_id_to_nodes中查找匹配的边
                    # 对于无向图，可能需要检查正向和反向
                    for eid, nodes in self.graph_cache.edge_id_to_nodes.items():
                        if nodes == node_pair or nodes == (node_pair[1], node_pair[0]):
                            edge_id = eid
                            break
            else:
                # 如果没有graph_cache，说明数据加载有问题
                print("Warning: No graph_cache available for edge lookup")

            if edge_id is not None:
                features = self.get_edge_features(edge_id)
                if features:
                    # 计算边的中点作为注释位置
                    edge_mid_x = (closest_edge[0][0] + closest_edge[1][0]) / 2
                    edge_mid_y = (closest_edge[0][1] + closest_edge[1][1]) / 2

                    # 构建信息文本
                    info_text = f"Edge ID: {features['edge_id']}\n"
                    info_text += f"Length: {features.get('length_norm', features.get('normalized_length', 'N/A'))}\n"
                    info_text += f"Width: {features.get('width', features.get('obstacle_free_width', 'N/A'))}\n"
                    info_text += f"Curb: {features.get('curb_norm', features.get('normalized_curb_height', 'N/A'))}\n"
                    info_text += f"Crossing: {features.get('crossing', 'N/A')}\n"
                    info_text += f"Path Type: {features.get('path_type', 'N/A')}"

                    # 添加增强属性（如果有）
                    if 'width_bin' in features and features['width_bin'] != 'N/A':
                        info_text += f"\nWidth Bin: {features['width_bin']}"
                    if 'curb_level' in features and features['curb_level'] != 'N/A':
                        info_text += f"\nCurb Level: {features['curb_level']}"
                    if 'confidence' in features and features['confidence'] != 'N/A':
                        info_text += f"\nConfidence: {features['confidence']:.2f}"
                    
                    # 显示数据来源（用于调试）
                    if 'data_source' in features:
                        info_text += f"\n[Source: {features['data_source']}]"
                    
                    self.annotation = self.ax.annotate(
                        info_text,
                        xy=(edge_mid_x, edge_mid_y),  # 箭头指向边的中点
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                        fontsize=8,
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                    )
                    self.fig.canvas.draw_idle()
    
    def create_visualization(self):
        """创建交互式可视化"""
        print("Creating visualization...")
        start_time = time.time()

        self.fig, self.ax = plt.subplots(figsize=(15, 10))

        # 获取节点位置和图对象
        if self.graph_cache and hasattr(self.graph_cache, 'G') and self.graph_cache.G is not None:
            # 新格式：有G属性的图
            pos = {node: node for node in self.graph_cache.G.nodes()}
            graph = self.graph_cache.G
            num_nodes = len(self.graph_cache.node_coords)
            num_edges = len(self.graph_cache.edge_attrs)
            graph_type = "Attributed Graph (New Format)"
        elif self.graph_cache and hasattr(self.graph_cache, 'node_coords') and self.graph_cache.node_coords:
            # 旧格式：只有node_coords的图（像default_graph）
            # 构建NetworkX图
            graph = nx.Graph()
            pos = {}

            # 添加节点
            for node_id, coord in self.graph_cache.node_coords.items():
                graph.add_node(coord)
                pos[coord] = coord

            # 添加边（基于edge_id_to_nodes）
            for edge_id, (u_id, v_id) in self.graph_cache.edge_id_to_nodes.items():
                if u_id in self.graph_cache.node_coords and v_id in self.graph_cache.node_coords:
                    u_coord = self.graph_cache.node_coords[u_id]
                    v_coord = self.graph_cache.node_coords[v_id]
                    graph.add_edge(u_coord, v_coord)

            num_nodes = len(self.graph_cache.node_coords)
            num_edges = len(self.graph_cache.edge_attrs)
            graph_type = "Graph Cache (Legacy Format)"
        else:
            # 旧格式NetworkX图
            pos = {node: node for node in self.G.nodes()}
            graph = self.G
            num_nodes = self.G.number_of_nodes()
            num_edges = self.G.number_of_edges()
            graph_type = "Graph (Legacy Format)"

        # 绘制节点（根据图大小调整）
        node_size = 10 if num_nodes > 1000 else 20  # 大图使用更小的节点
        alpha = 0.5 if num_nodes > 1000 else 0.7   # 大图降低透明度
        nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color='lightblue', alpha=alpha, ax=self.ax)

        # 绘制边（确保不显示箭头，根据图大小调整）
        edge_width = 0.5 if num_edges > 2000 else 1  # 大图使用更细的边
        edge_alpha = 0.3 if num_edges > 2000 else 0.5  # 大图降低边透明度

        # 首先绘制所有普通边（灰色）
        nx.draw_networkx_edges(graph, pos, alpha=edge_alpha, edge_color='gray',
                             width=edge_width, ax=self.ax, arrows=False)

        # 绘制特殊边：curb != 0 或 crossing != 0 或 path_type != 0（红色突出显示）
        special_edges = []
        special_edge_labels = []

        # 统一使用归一化后的数据（edge_feature_list），因为：
        # 1. 数据已经过适当预处理和归一化
        # 2. 与模型训练时使用的数据格式一致
        # 3. 避免原始数据scale不同的问题
        if self.edge_feature_list:
            for edge_id, features in enumerate(self.edge_feature_list):
                if len(features) >= 5:
                    width_norm = features[1]
                    curb_norm = features[2]  # curb_norm (归一化值)
                    crossing = features[3]   # crossing (0或1)
                    path_type = features[4]  # path_type (0, 1, 或 2)

                    # 检查是否为特殊边
                    is_special = (width_norm < 1.0)

                    if is_special:
                        # 找到对应的坐标
                        if self.graph_cache and hasattr(self.graph_cache, 'edge_id_to_nodes') and edge_id in self.graph_cache.edge_id_to_nodes:
                            # Graph cache格式：通过edge_id_to_nodes和node_coords查找坐标
                            u_id, v_id = self.graph_cache.edge_id_to_nodes[edge_id]
                            if u_id in self.graph_cache.node_coords and v_id in self.graph_cache.node_coords:
                                u_coord = self.graph_cache.node_coords[u_id]
                                v_coord = self.graph_cache.node_coords[v_id]
                                special_edges.append((u_coord, v_coord))

                                # 创建标签
                                label_parts = []
                                if curb_norm > 0:
                                    label_parts.append(f"Curb:{curb_norm:.2f}")
                                if crossing > 0:
                                    label_parts.append("Crossing:Yes")
                                if path_type > 0:
                                    path_names = {1: 'bike', 2: 'connection'}
                                    label_parts.append(f"Type:{path_names.get(path_type, path_type)}")
                                special_edge_labels.append(" | ".join(label_parts))
                        else:
                            print(f"Warning: Could not find coordinates for edge_id {edge_id} in graph_cache")

        # 绘制特殊边（红色，较粗）
        if special_edges:
            special_edge_width = edge_width * 2  # 特殊边更粗
            special_edge_alpha = min(edge_alpha * 2, 0.8)  # 特殊边更明显

            # 创建特殊边的子图
            special_graph = nx.Graph()
            special_graph.add_edges_from(special_edges)

            nx.draw_networkx_edges(special_graph, pos, alpha=special_edge_alpha,
                                 edge_color='red', width=special_edge_width,
                                 ax=self.ax, arrows=False)

            print(f"Highlighted {len(special_edges)} special edges (curb≠0 or crossing≠0 or type≠0)")

        # 设置标题和标签
        self.ax.set_title(f'Interactive {graph_type} Visualization\n'
                         f'Graph: {self.graph_id}\n'
                         f'Nodes: {num_nodes}, Edges: {num_edges}',
                         fontsize=14, fontweight='bold')
        self.ax.set_xlabel('X Coordinate (normalized)')
        self.ax.set_ylabel('Y Coordinate (normalized)')

        # 添加说明
        instruction_text = "Instructions:\n" \
                          "• Hover over edges to see feature information\n" \
                          "• Blue nodes = graph intersections\n" \
                          "• Gray edges = regular walkable paths\n" \
                          "• Red edges = special paths (curb>0, crossing=Yes, or type≠walk)\n" \
                          "• Use mouse wheel to zoom, drag to pan"

        self.ax.text(0.02, 0.02, instruction_text, transform=self.ax.transAxes,
                    verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                    fontsize=9)

        # 添加图例
        legend_text = "Edge Features:\n" \
                     "• Length: normalized path length\n" \
                     "• Width: path width (meters)\n" \
                     "• Curb: curb height (0=flat, >0=raised curb)\n" \
                     "• Crossing: pedestrian crossing (0=No, 1=Yes)\n" \
                     "• Path Type: walk=0, bike=1, connection=2\n" \
                     "• Red edges: special accessibility features"

        self.ax.text(0.98, 0.02, legend_text, transform=self.ax.transAxes,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    fontsize=8)

        # 连接事件
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_edge_hover)

        plt.tight_layout()

        end_time = time.time()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Interactive Graph Visualizer")
    parser.add_argument("--graph_id", type=str, default="default_graph",
                       help="Graph ID to visualize (e.g., default_graph, amsterdam_center)")
    parser.add_argument("--data_dir", type=str, default="../data/graph_data",
                       help="Base data directory")

    args = parser.parse_args()

    print(f"Visualizing graph: {args.graph_id}")
    print(f"Data directory: {args.data_dir}")

    visualizer = InteractiveGraphVisualizer(args.graph_id, args.data_dir)
    visualizer.create_visualization()

if __name__ == "__main__":
    main()

# 使用示例：
# python src/interactive_graph_visualizer.py --graph_id default_graph --data_dir ../data/graph_data
#
# 功能说明：
# - 灰色边：普通步行路径
# - 红色边：特殊路径（curb高度不为0，或有交叉点，或不是纯步行路径）
# - 鼠标悬停在边上可查看详细信息
# - 支持缩放和平移操作 
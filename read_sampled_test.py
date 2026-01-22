#!/usr/bin/env python3
"""
脚本：读取sampled_test.df文件的前十行内容
"""

import pandas as pd
import os

def read_sampled_test_head():
    # 文件路径
    file_path = "data/graph_data/default_graph/trajectories/replay_buffer.df"

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件不存在 {file_path}")
        return

    try:
        # 读取pickle格式的DataFrame
        print(f"正在读取文件：{file_path}")
        df = pd.read_pickle(file_path)

        print(f"DataFrame形状：{df.shape}")
        print(f"列名：{list(df.columns)}")
        print("\n按route_id排序后的前十行内容：")
        print("=" * 50)

        # 按route_id和step_id排序
        # df_sorted = df.sort_values(['route_id', 'step_id'])

        print("前十行（按route_id和step_id排序）:")
        # 设置 pandas 显示选项，不省略中间的列
        with pd.option_context('display.max_columns', None, 'display.max_colwidth', None):
            print(df.head(30))

        print("\n" + "=" * 80)
        print("数据分析：")

        # 检查是否有到达终点的记录
        completed_routes = df_sorted[df_sorted['dist_to_goal'] < 0.1]
        print(f"到达终点的记录数: {len(completed_routes)}")
        print(f"总记录数: {len(df_sorted)}")
        print(f"到达终点比例: {len(completed_routes)/len(df_sorted)*100:.2f}%")
        if len(completed_routes) > 0:
            print(f"\n到达终点的路径示例 (前5个):")
            print(completed_routes[['route_id', 'step_id', 'cur_node_id', 'goal_node_id', 'dist_to_goal']].head().to_string())
        else:
            print("\n该数据集中没有到达终点的完整路径记录。")
            print("这些可能是路径规划过程中的中间状态或训练样本。")

        print("\n" + "=" * 60)
        print("完整路径轨迹示例 - 显示到达终点的路径：")

        # 选择几个到达终点的route_id来显示完整轨迹
        completed_route_ids = completed_routes['route_id'].unique()[:3]  # 显示前3个到达终点的路径

        for route_id in completed_route_ids:
            print(f"\nRoute ID: {route_id} (完整轨迹)")
            route_data = df_sorted[df_sorted['route_id'] == route_id].sort_values('step_id')
            print(route_data[['step_id', 'start_node_id', 'goal_node_id', 'cur_node_id', 'next_node_id', 'dist_to_goal']].to_string())

            # 显示路径统计
            start_node = route_data['start_node_id'].iloc[0]
            goal_node = route_data['goal_node_id'].iloc[0]
            steps = len(route_data)
            final_dist = route_data['dist_to_goal'].iloc[-1]
            print(f"起始节点: {start_node} -> 目标节点: {goal_node} | 总步数: {steps} | 最终距离: {final_dist:.6f}")

        print("\n" + "=" * 60)
        print("对比：未到达终点的路径示例")

        # 显示一个未到达终点的路径作为对比
        incomplete_routes = df_sorted[df_sorted['dist_to_goal'] >= 0.1]
        if len(incomplete_routes) > 0:
            incomplete_route_ids = incomplete_routes['route_id'].unique()[:1]
            for route_id in incomplete_route_ids:
                print(f"\nRoute ID: {route_id} (未完成轨迹)")
                route_data = df_sorted[df_sorted['route_id'] == route_id].sort_values('step_id')
                print(route_data[['step_id', 'start_node_id', 'goal_node_id', 'cur_node_id', 'next_node_id', 'dist_to_goal']].to_string())

                start_node = route_data['start_node_id'].iloc[0]
                goal_node = route_data['goal_node_id'].iloc[0]
                steps = len(route_data)
                final_dist = route_data['dist_to_goal'].iloc[-1]
                print(f"起始节点: {start_node} -> 目标节点: {goal_node} | 总步数: {steps} | 最终距离: {final_dist:.6f}")

        print("=" * 50)
        print(f"数据类型信息：")
        print(df.dtypes)

    except Exception as e:
        print(f"读取文件时出错：{e}")

if __name__ == "__main__":
    read_sampled_test_head()
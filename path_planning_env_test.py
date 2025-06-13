import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np
from typing import Optional
from collections import defaultdict

class PathPlanningEnv(gym.Env):
    """ 
    Reinforcement Learning Environment for Path Planning in a Graph Network

    State: 
        - Current node coordinates (x, y)
        - Target coordinates relative to current position (dx, dy)
        - Normalized progress (steps / max_steps)
  
    Action: 
        - Discrete action: choose next node from current node's neighbors

    Reward: 
        - Negative distance penalty for each step
        - -10 penalty for invalid moves
        - +1000 reward for reaching the target
        - Additional penalty for revisiting edges
    """
    DISTANCE_REWARD_SCALE = 10.0
    STEP_PENALTY = 0.1
    LOOP_PENALTY = 100.0
    GOAL_REWARD = 1000.0
    INVALID_PENALTY = 5.0

    def __init__(self, G: nx.Graph, start_node, end_node, max_steps=200):
        super().__init__()
        
        # Graph and path parameters
        self.G = G
        self.start_node = start_node
        self.end_node = end_node
        self.max_steps = max_steps
        
        # Action space: discrete selection of next node from neighbors
        max_neighbors = max(len(list(G.neighbors(node))) for node in G.nodes)
        self.action_space = spaces.Discrete(max_neighbors)

        # Observation space: [current_x, current_y, delta_x, delta_y, progress]
        obs_high = np.array([1e5, 1e5, 1e5, 1e5, 1.0])
        obs_low = -obs_high.copy()
        obs_low[-1] = 0.0  # Progress can't be negative
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float64)

        # Internal state variables
        self.current_node = None
        self.steps = 0
        self.invalid_action_count = 0  # Counter for invalid actions
        self.target_coord = np.array(end_node)
        self.visited_edges = set()   # Track visited edges to penalize loops
        self.visited_nodes = set()   # Optionally track visited nodes
        self.node_visit_count = defaultdict(int)  # 新增：记录每个节点访问次数
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_node = self.start_node
        self.steps = 0
        self.invalid_action_count = 0
        self.visited_edges = set()   # Track visited edges to penalize loops
        self.visited_nodes = set()   # Optionally track visited nodes
        self.node_visit_count = defaultdict(int)  # 新增：记录每个节点访问次数
        return self._get_observation(), {}
    
    def step(self, action):
        reward = 0
        done = False
        truncated = False
        info = {}

        neighbors_raw = list(self.G.neighbors(self.current_node))
        neighbors = [n for n in neighbors_raw if n != self.current_node]

        # If there are no neighbors, terminate the episode
        if len(neighbors) == 0:
            reward = -50  # 更大惩罚
            done = True
            truncated = True
            info['failure'] = 'No neighbors'
            return self._get_observation(), reward, done, truncated, info

        # Clip action to valid range
        action = int(np.clip(action, 0, len(neighbors) - 1))

        if not self.G.has_node(self.current_node):
            reward = -50
            done = True
            return self._get_observation(), reward, done, False, {}

        if neighbors[action] == self.current_node:
            reward = -20  # 更大惩罚
            self.invalid_action_count += 1
            return self._get_observation(), reward, done, truncated, info

        prev_dist = np.linalg.norm(np.array(self.current_node) - self.target_coord)
        next_node = neighbors[action]
        edge = (self.current_node, next_node)
        self.current_node = next_node
        self.steps += 1
        self.invalid_action_count = 0

        self.node_visit_count[next_node] += 1
        self.visited_edges.add(edge)
        self.visited_nodes.add(next_node)

        if self.G.is_multigraph():
            # MultiGraph: 取第一条边的 'length' 属性
            edge_data = self.G.get_edge_data(self.current_node, next_node)
            if edge_data:
                edge_length = list(edge_data.values())[0].get('length', 1.0)
            else:
                edge_length = 1.0
        else:
            # 普通 Graph
            edge_length = self.G[self.current_node][next_node].get('length', 1.0)
        
        reward -= edge_length

        new_dist = np.linalg.norm(np.array(self.current_node) - self.target_coord)
        # 距离终点变小奖励
        distance_reward = (prev_dist - new_dist) * self.DISTANCE_REWARD_SCALE
        reward = distance_reward - self.STEP_PENALTY

        # 节点循环惩罚
        if self.node_visit_count[next_node] > 1:
            reward -= self.LOOP_PENALTY

        # 边循环惩罚
        if list(self.visited_edges).count(edge) > 1:
            reward -= self.LOOP_PENALTY / 2

        # 终点奖励
        if self.current_node == self.end_node:
            reward += self.GOAL_REWARD
            done = True
            info['success'] = True
        elif self.steps >= self.max_steps:
            reward -= new_dist * 0.1
            done = True
            truncated = True
            info['timeout'] = True

        return self._get_observation(), reward, done, truncated, info

    def _get_observation(self):
        """Get current state observation"""
        progress = min(max(self.steps / self.max_steps, 0), 1) if self.max_steps != 0 else 0
        current_coord = np.array(self.current_node)
        delta = self.target_coord - current_coord
        return np.concatenate([
            current_coord,
            delta,
            [progress]
        ]).astype(np.float64)
# GraphRouter: LLM-informed Graph-constrained Routing

A unified pipeline for LLM-informed graph-constrained routing that discovers implicit user preferences through cross-map invariant navigation policies. Transforms coordinate-dependent path planning into candidate edge ranking, enabling preference-aware routing that transfers across different graph topologies while respecting structural constraints.

## Core Innovation: LLM-informed Graph-constrained Routing under Implicit User Preferences

**Problem**: Traditional routing systems memorize map-specific coordinate patterns, failing to capture implicit user preferences that could generalize across different graphs. 15-dimensional coordinate vectors create "coordinate fingerprints" that don't transfer between maps.

**Solution**: Learn implicit user routing preferences through LLM-informed, graph-constrained navigation. Transform coordinate-dependent routing into candidate edge ranking that discovers transferable preference patterns under graph topology constraints.

## Key Transformations

### 1. Data Representation
- **Before**: Coordinate fingerprints (start/end/origin/dest/cur positions)
- **After**: Graph topology + implicit preference signals (edge sequences, progress metrics, user preference indicators)

### 2. Routing Paradigm
- **Before**: Map-specific coordinate matching
- **After**: LLM-informed preference learning under graph constraints

### 3. Action Space
- **Before**: Global edge classification with masking
- **After**: Candidate edge ranking within local graph neighborhoods

### 4. Learning Objective
- **Before**: Memorize coordinate patterns
- **After**: Discover transferable routing preferences (length preferences, turn patterns, path characteristics)

### 5. Training Pipeline
- **Phase 1 (SL)**: Expert route imitation to learn baseline preferences
- **Phase 2 (RL)**: Preference optimization with reachability constraints (progress rewards, loop penalties, arrival bonuses)

## Project Structure

```
GraphRouter/
├── data/
│   └── graph_data/          # Multi-graph data directory
│       └── [graph_id]/      # Graph-specific workspace (e.g., default_graph, city_center)
│           ├── cache/       # Graph cache files
│           │   ├── graph_cache.pkl
│           │   ├── coordinate_stats.json
│           │   └── ...
│           ├── trajectories/# Preprocessed and training data
│           │   ├── sampled_train.df
│           │   ├── sampled_val.df
│           │   ├── sampled_test.df
│           │   ├── replay_buffer.df         # Training transitions
│           │   └── data_statis.df           # Training statistics
│           └── raw_data/    # Original/raw data
│               ├── metadata.json
│               ├── network_map/
│               ├── train/   # Raw trajectory data
│               ├── val/
│               └── test/
├── src/
│   ├── generate_routes.py   # Route generation and E-bucket creation
│   ├── preprocess.py         # Cross-map invariant preprocessing
│   ├── replay_buffer.py     # Candidate-based buffer management
│   └── utils/
│       ├── graph_cache.py   # Cross-map invariant graph utilities
│       └── ...             # Other utilities
└── README.md
```

## LLM-informed Routing Pipeline

### Phase 1: Preference-rich Route Generation

Generate diverse expert routes that capture implicit user preferences under graph constraints.

```bash
# Generate preference-informed routes (E-bucket)
python src/generate_routes.py --graph_id city_center --num_routes 20000 --data_dir ../data
```

**Generates**: Expert trajectories (E-bucket) for the specified graph that encode routing preferences through diverse path characteristics.

### Phase 2: Graph-invariant Preference Extraction

Transform coordinate-dependent routes into preference signals that can transfer across different graphs.

```bash
# Extract cross-map invariant routing preferences
python src/preprocess.py --graph_id city_center --data_dir ../data
```

**Transforms**:
- Coordinate sequences → Graph topology sequences + Graph context
- Absolute positions → Relative progress signals
- Map-specific patterns → Graph-scoped transferable preferences
- **Multi-map ready**: Each transition tagged with `graph_id` for conflict-free mixing

### Phase 3: Training Data Preparation

Convert preprocessed trajectories into training-ready replay buffer with candidate edge features.

```bash
# Build replay buffer for training
python src/replay_buffer.py --graph_id city_center --data_dir ../data
```

**Creates**: Training transitions with dynamically reconstructed candidate edge features for the specified graph.

### Phase 3: Two-Stage Preference Learning

#### Stage 1: Supervised Preference Imitation
```python
# Learn routing preferences from expert demonstrations
# - Input: E-bucket with preference-rich routes
# - Output: Preference-aware routing policy
# - Method: Imitate expert routing decisions under graph constraints
```

#### Stage 2: RL Preference Optimization
```python
# Optimize preferences with graph-aware rewards
# - Input: Mixed expert + learned trajectories
# - Output: Cross-map generalizable routing preferences
# - Rewards: Route efficiency, user preference alignment, graph constraint satisfaction
```

## Data Format Evolution

### Raw Routes (Preference-rich but Map-specific)
- **Trajectory**: `p_route_{id}.gpkg` - Geometric paths with implicit preferences
- **Coordinates**: `route_{id}_start_end.csv` - Absolute start/end positions

### Preprocessed Routing Decisions (Multi-Graph Invariant Preferences)
Each routing decision captures preference learning under graph constraints with explicit multi-map context:
- `graph_id`: **NEW** - Graph identifier for multi-map disambiguation
- `route_id`: Route identifier within graph (preference context)
- `step_id`: Decision step within route
- `start_node_id`: Route origin (local to graph)
- `goal_node_id`: Route destination (local to graph)
- `cur_node_id`: Current decision point (local to graph)
- `action_edge_id`: Chosen routing action (reveals preferences, local to graph)
- `next_node_id`: Next decision point (local to graph)
- `dist_to_goal`: Progress signal (constraint satisfaction, graph-specific)

**Key Transformation**: From coordinate-locked routes to graph-scoped preference signals with explicit multi-map context, enabling safe mixing of data from different graphs while preserving local ID semantics.

#### Multi-Graph Data Structure
The system now supports mixing data from multiple graphs through explicit graph context:

| Field | Type | Dimension | Description |
|-------|------|-----------|-------------|
| `graph_id` | str | Scalar | **NEW**: Graph identifier for multi-map disambiguation |
| `cur_node_id` | int | Scalar | Current node ID (local to graph) |
| `goal_node_id` | int | Scalar | Goal node ID (local to graph) |
| `cand_edge_ids` | List[int] | Variable(3-10) | Candidate edge IDs (local to graph) |
| `taken_edge_id` | int | Scalar | Chosen edge ID (local to graph) |
| `bc_action_idx` | int/None | Scalar | Index in candidate list for BC |
| `next_node_id` | int | Scalar | Next node ID (local to graph) |
| Other fields | ... | ... | Rewards, states, distances, etc. |

**Multi-map Safety**: Local IDs are only meaningful within their graph context, preventing ID conflicts when mixing data from different maps.

#### Cache Organization
```
data/graph_data/
├── default_graph/           # Current graph (ID: 'default_graph')
│   ├── graph_cache.pkl      # GraphCache with local IDs
│   ├── coordinate_stats.json
│   ├── physical_stats.json
│   └── ...
├── graph_A/                 # Future graphs
└── graph_B/
```

Each graph maintains its own ID space and cache, enabling true multi-map generalization.

## Dependencies

Required packages for graph-constrained routing pipeline:
- geopandas, shapely: Geometric route processing
- pandas, numpy: Data manipulation and preprocessing
- networkx: Graph topology and routing algorithms
- scikit-learn: Feature preprocessing
- matplotlib, contextily: Route visualization
- LLM integration libraries (as needed for preference modeling)

## Notes

- Ensure map data exists in `data/network_map` before running route generation
- Graph cache is automatically built during preprocessing if not present
- All coordinate transformations use normalized space for cross-map generalization
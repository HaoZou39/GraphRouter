# Graph Bank Generator: Multi-city Accessibility Graph Bank

This module generates diverse city-scale accessibility graphs with realistic topology and inferred attributes for training preference-aware routing models.

## Architecture Overview

The graph bank generator follows a 6-phase pipeline:

1. **OSM Extraction & Standardization**: Extract walk networks from OpenStreetMap and standardize to target statistics
2. **Sampling Points Generation**: Generate edge sampling points for attribute inference
3. **Attribute Perception**: Infer accessibility attributes using VLM, street imagery, or rules
4. **World Model**: Apply probabilistic reasoning to combine priors with observations
5. **NetworkX Export**: Export Router-compatible graphs with edge weights
6. **Quality Control**: Comprehensive validation and statistical comparison

## Quick Start

### 1. Install Dependencies

```bash
pip install osmnx geopandas shapely pandas numpy networkx
pip install openai  # For VLM inference (optional)
```

### 2. Configure AOIs

Edit `config/graph_bank_config.json` to specify areas of interest:

```json
{
  "aoi_list": [
    {
      "name": "amsterdam_center",
      "center": [52.3676, 4.9041],
      "radius": 1000,
      "description": "Amsterdam city center"
    }
  ],
  "target_stats_path": "data/graph_data/default_graph/cache/physical_stats.json"
}
```

### 3. Generate Graph Bank

```bash
# Generate all configured AOIs
python scripts/generate_graph_bank.py

# Generate specific AOI with rule-based attributes (faster)
python scripts/generate_graph_bank.py --aoi amsterdam_center --method rules

# Skip expensive phases for testing
python scripts/generate_graph_bank.py --skip_phases 3,4
```

## File Structure

Following the existing project conventions:

```
data/graph_data/
├── [graph_id]/                    # Each city/subregion
│   ├── cache/                     # Core graph files
│   │   ├── graph_cache.pkl        # GraphCache object
│   │   ├── coordinate_stats.json  # Coordinate normalization
│   │   ├── physical_stats.json    # Physical statistics
│   │   ├── edge_feature_list.pkl  # Edge attributes
│   │   └── edge_id_map.pkl        # Edge mappings
│   ├── trajectories/              # Generated training data
│   │   ├── sampled_train.df       # Training trajectories
│   │   └── data_statis.df         # Statistics
│   └── raw_data/                  # Raw data and metadata
│       ├── metadata.json          # OSM metadata
│       ├── edge_samples.parquet   # Sampling points
│       ├── attrs_inferred.parquet # Inferred attributes
│       └── qc_report.json         # Quality control
```

## Configuration Options

### AOI Specification
- `name`: Unique identifier for the graph
- `center`: [latitude, longitude] coordinates
- `radius`: Extraction radius in meters

### Attribute Inference Methods

#### VLM Inference (Recommended)
Uses GPT-4V to analyze street imagery:
```bash
python scripts/generate_graph_bank.py --method vlm
```
Requires OpenAI API key in config.

#### Rule-based Inference (Fast)
Uses OSM tags and heuristics:
```bash
python scripts/generate_graph_bank.py --method rules
```

### API Keys Setup
Add your API keys to `config/graph_bank_config.json`:

```json
{
  "attribute_settings": {
    "api_keys": {
      "openai": "sk-...",
      "mapillary": "MLY|...",
      "google": "AIza..."
    }
  }
}
```

## Quality Control

Each generated graph includes a comprehensive QC report:

```python
# Load QC report
import json
with open("data/graph_data/amsterdam_center/raw_data/qc_report.json", 'r') as f:
    qc = json.load(f)

print(f"Overall score: {qc['summary']['overall_score']:.2f}")
print(f"Grade: {qc['summary']['grade']}")
print("Warnings:", qc['warnings'])
```

QC checks include:
- **Topology**: Connectivity, degree distribution, shortest paths
- **Geometry**: Edge lengths, node spacing
- **Attributes**: Distribution diversity, uncertainty levels
- **Statistics**: Alignment with target Armstrong graph

## Integration with Existing Pipeline

Generated graphs are fully compatible with your existing Router and training pipeline:

```python
# Use generated graph for route generation
from src.generate_routes import generate_route

# Load graph (Router will automatically find it in data/graph_data/)
route_result = generate_route(
    data_directory="data/graph_data/amsterdam_center",
    # ... other parameters
)
```

## Advanced Usage

### Custom Standardization

Modify target statistics in `config/graph_bank_config.json`:

```json
{
  "standardization_settings": {
    "target_node_range": [200, 400],
    "target_edge_range": [300, 600],
    "target_degree_range": [2.5, 4.0]
  }
}
```

### Parallel Generation

For multiple AOIs, the generator processes them sequentially. For parallel processing:

```bash
# Generate different AOIs in parallel terminals
python scripts/generate_graph_bank.py --aoi amsterdam_center &
python scripts/generate_graph_bank.py --aoi rotterdam_business &
```

### Custom Attribute Schemas

Extend attributes in `src/utils/attribute_world_model.py`:

```python
# Add new attribute
self.attribute_schemas['surface_type'] = ['asphalt', 'concrete', 'brick', 'unknown']
self.priors['surface_type'] = {
    'default': np.array([0.5, 0.3, 0.1, 0.1])
}
```

## Troubleshooting

### Common Issues

1. **OSM Extraction Fails**
   - Check internet connection
   - Verify coordinates are valid lat/lon
   - Try smaller radius

2. **VLM Inference Fails**
   - Check OpenAI API key and credits
   - Falls back to rule-based inference
   - Check API rate limits

3. **Low QC Scores**
   - Review target statistics alignment
   - Check OSM data quality for AOI
   - Adjust standardization parameters

4. **Memory Issues**
   - Reduce AOI radius
   - Skip detailed QC (`detailed_reports: false`)
   - Process one AOI at a time

### Debugging

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check intermediate files in `raw_data/` and `cache/` directories.

## Performance Notes

- **VLM Inference**: ~30-60 seconds per edge (expensive but accurate)
- **Rule-based**: ~1 second per edge (fast but less accurate)
- **OSM Extraction**: 10-30 seconds per AOI depending on size
- **Total Time**: 5-30 minutes per AOI depending on method

## Contributing

To extend the graph bank generator:

1. Add new attribute types in `AttributeWorldModel`
2. Implement new perception methods in `AttributePerceiver`
3. Add QC checks in `QCChecker`
4. Update configuration schema as needed

## License & Attribution

- Generated graphs include OSM attribution metadata
- Respect API terms of service for imagery providers
- Consider data licensing for distribution
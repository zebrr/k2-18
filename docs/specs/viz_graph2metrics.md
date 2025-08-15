# viz_graph2metrics.md

## Status: READY (VIZ-METRICS-02 IMPLEMENTED)

Module for computing NetworkX metrics on K2-18 knowledge graph for visualization.

## Module Purpose

The `graph2metrics.py` utility enriches the LearningChunkGraph with various network metrics computed using NetworkX. These metrics provide insights into node importance, clustering, and connectivity patterns that enhance the interactive visualization experience.

## CLI Interface

```bash
# Run on production data
python -m viz.graph2metrics

# Run on test data  
python -m viz.graph2metrics --test-mode
```

### Command-line Arguments

- `--test-mode` - Use test data from `/viz/data/test/` instead of `/viz/data/in/`

## Input/Output Files

### Input Files
- **Production mode**: 
  - `/viz/data/in/LearningChunkGraph.json` - Knowledge graph structure
  - `/viz/data/in/ConceptDictionary.json` - Concept definitions
- **Test mode**:
  - `/viz/data/test/LearningChunkGraph.json` - Test graph (10-20 nodes)
  - `/viz/data/test/ConceptDictionary.json` - Test concepts (5-10 concepts)

### Output Files
Always saved to `/viz/data/out/` regardless of mode:
- `LearningChunkGraph_wow.json` - Graph enriched with metrics
- `ConceptDictionary_wow.json` - Concepts with mention index

### Log Files
- `/viz/logs/graph2metrics.log` - Execution log with timestamps

## Algorithm

### 1. Initialization
- Parse command-line arguments
- Setup UTF-8 console encoding
- Initialize logging to file and console
- Load configuration from `/viz/config.toml`

### 2. Data Loading
- Determine input directory based on mode
- Load JSON files
- Validate against schemas:
  - `LearningChunkGraph` schema
  - `ConceptDictionary` schema
- Validate graph invariants

### 3. NetworkX Conversion
- Create `nx.DiGraph()` (directed graph)
- Add nodes with all attributes from JSON
- Add edges with type, weight, conditions
- Verify graph is directed
- Log basic statistics:
  - Number of nodes
  - Number of edges
  - Weakly connected components

### 4. Metrics Computation
- `compute_centrality_metrics()` - **IMPLEMENTED** - Computes six centrality metrics for each node:
  - `degree_in` (int) - Number of incoming edges
  - `degree_out` (int) - Number of outgoing edges
  - `degree_centrality` (float) - Normalized degree centrality
  - `pagerank` (float) - PageRank importance score
  - `betweenness_centrality` (float) - Measure of bridge nodes
  - `closeness_centrality` (float) - Average distance to other nodes
- `compute_clustering()` - Placeholder for Louvain communities (VIZ-METRICS-03)
- `generate_demo_path()` - Placeholder for tour mode path (VIZ-METRICS-04)

### 5. Output Generation
- Copy input data to output (metrics will be added in future versions)
- Save enriched JSON files with UTF-8 encoding
- Log completion status

## Dependencies

### External Libraries
- `networkx>=3.0` - Graph algorithms and metrics
- `python-louvain>=0.16` - Community detection (future use)

### Internal Modules
- `src.utils.config` - Configuration loading
- `src.utils.console_encoding` - UTF-8 output support
- `src.utils.exit_codes` - Standardized exit codes
- `src.utils.validation` - JSON schema validation

## Configuration

Uses `/viz/config.toml` with sections:
- `[graph2metrics]` - Metrics computation parameters
  - `pagerank_damping` (float, 0-1, default=0.85) - PageRank damping factor
  - `pagerank_max_iter` (int, >0, default=100) - Max iterations for PageRank
  - `betweenness_normalized` (bool, default=true) - Normalize betweenness values
  - `closeness_harmonic` (bool, default=true) - Use harmonic mean for disconnected graphs
  - `louvain_resolution` (float, default=1.0) - Community detection resolution (VIZ-METRICS-03)
  - `demo_strategy` (int, default=1) - Path generation strategy (VIZ-METRICS-04)
  - `demo_max_nodes` (int, default=15) - Maximum nodes in demo path (VIZ-METRICS-04)

## Exit Codes

- `0` (EXIT_SUCCESS) - Successful completion
- `1` (EXIT_CONFIG_ERROR) - Configuration loading/validation error
- `2` (EXIT_INPUT_ERROR) - Input file not found or validation failed
- `3` (EXIT_RUNTIME_ERROR) - Unexpected runtime error
- `5` (EXIT_IO_ERROR) - File I/O error

## Console Output

### Production Mode
```
Using production data
Graph loaded: 300 nodes, 500 edges
✓ Graph metrics computed successfully
```

### Test Mode
```
[TEST MODE] Using test data
[TEST MODE] Graph loaded: 16 nodes, 22 edges
✓ Graph metrics computed successfully (TEST MODE)
```

## Logging Format

```
2024-01-15 10:30:45 - INFO - === START graph2metrics ===
2024-01-15 10:30:45 - INFO - Loading configuration from /viz/config.toml
2024-01-15 10:30:45 - INFO - Config loaded, demo_strategy: 1
2024-01-15 10:30:45 - INFO - Loading input files from /viz/data/in
2024-01-15 10:30:45 - INFO - Validating graph data
2024-01-15 10:30:45 - INFO - Converting to NetworkX DiGraph
2024-01-15 10:30:45 - INFO - Graph statistics:
2024-01-15 10:30:45 - INFO -   - Nodes: 300
2024-01-15 10:30:45 - INFO -   - Edges: 500
2024-01-15 10:30:45 - INFO -   - Weakly connected components: 1
2024-01-15 10:30:45 - INFO -   - Is directed: True
2024-01-15 10:30:45 - INFO - Computing centrality metrics
2024-01-15 10:30:45 - INFO - Computing degree metrics
2024-01-15 10:30:45 - INFO - Computing PageRank (damping=0.85, max_iter=100)
2024-01-15 10:30:45 - INFO - Computing betweenness centrality (normalized=True)
2024-01-15 10:30:45 - INFO - Computing closeness centrality (harmonic=True)
2024-01-15 10:30:45 - INFO - Adding metrics to 300 nodes
2024-01-15 10:30:45 - INFO - Metrics added to 300 nodes
2024-01-15 10:30:45 - INFO - PageRank range: [0.001234, 0.045678]
2024-01-15 10:30:45 - INFO - Computing clustering (stub)
2024-01-15 10:30:45 - INFO - Generating demo path (stub)
2024-01-15 10:30:45 - INFO - Saving output files to /viz/data/out
2024-01-15 10:30:45 - INFO - === SUCCESS graph2metrics ===
```

## Usage Examples

### Basic Usage
```bash
# Activate virtual environment
source .venv/bin/activate

# Run on production data
python -m viz.graph2metrics

# Check output
ls -la viz/data/out/
```

### Test Mode
```bash
# Run on test data
python -m viz.graph2metrics --test-mode

# Verify output
python -c "import json; g=json.load(open('viz/data/out/LearningChunkGraph_wow.json')); print(f'Nodes: {len(g[\"nodes\"])}')"
```

### Validation Check
```bash
# Validate output files
python -c "from src.utils.validation import validate_json; import json; \
g=json.load(open('viz/data/out/LearningChunkGraph_wow.json')); \
validate_json(g, 'LearningChunkGraph'); print('Graph valid ✓')"

python -c "from src.utils.validation import validate_json; import json; \
c=json.load(open('viz/data/out/ConceptDictionary_wow.json')); \
validate_json(c, 'ConceptDictionary'); print('Concepts valid ✓')"
```

### Check Logs
```bash
# View recent log entries
tail -20 viz/logs/graph2metrics.log

# Search for errors
grep ERROR viz/logs/graph2metrics.log
```

## Future Enhancements

### ✅ VIZ-METRICS-02: Centrality Metrics (COMPLETED)
- ✓ PageRank for node importance
- ✓ Degree centrality (in/out/normalized)
- ✓ Betweenness centrality for bridges
- ✓ Closeness centrality for connectivity
- ✓ Safe handling of NaN/inf values for isolated nodes
- ✓ Configuration-driven parameters

### VIZ-METRICS-03: Clustering
- Louvain community detection
- Connected components analysis
- Clustering coefficients
- Bridge score calculation

### VIZ-METRICS-04: Demo Path
- Generate educational paths based on strategy
- Add path metadata to graph
- Support different path generation algorithms

## Performance Notes

- **Betweenness centrality**: O(n*m) for unweighted, O(n*m + n^2*log(n)) for weighted graphs
- **PageRank**: O(iterations * edges)
- **Closeness centrality**: O(n^2) using Floyd-Warshall for all pairs shortest paths
- **Expected times**:
  - 16 nodes (test): < 0.1 seconds
  - 100 nodes: ~1 second
  - 250 nodes: ~2-5 seconds
  - 1000 nodes: ~30-60 seconds

## Notes

- Graph must be converted to `nx.DiGraph` (directed), not undirected
- All output files use same names regardless of mode for consistency
- Test mode adds `[TEST MODE]` prefix to console output only
- Configuration parameters are fully used for centrality metrics computation
- Isolated nodes and disconnected components are handled safely (NaN/inf → 0.0)
- UTF-8 encoding is enforced for all file I/O operations
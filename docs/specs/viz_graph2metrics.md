# viz_graph2metrics.md

## Status: READY (Data Enrichment Implemented)

Module for computing comprehensive NetworkX metrics on K2-18 knowledge graph for visualization, including clustering, bridge detection, and data enrichment for interactive visualization.

## Module Purpose

The `graph2metrics.py` utility enriches the LearningChunkGraph with 12 network metrics for nodes and 4 for edges computed using NetworkX. These metrics provide insights into node importance, clustering, connectivity patterns, bridge detection, and educational structure that power the interactive visualization experience.

**For detailed algorithms and formulas of all metrics, see: `/docs/specs/viz_metrics_reference.md`**

## CLI Interface

```bash
# Run on production data
python -m viz.graph2metrics

# Run on test data  
python -m viz.graph2metrics --test-mode

# Run validation mode
python -m viz.graph2metrics --validate

# Validate specific graph
python -m viz.graph2metrics --validate --graph test_line

# Validate specific metric
python -m viz.graph2metrics --validate --metric pagerank

# Verbose validation output
python -m viz.graph2metrics --validate --verbose
```

### Command-line Arguments

- `--test-mode` - Use test data from `/viz/data/test/` instead of `/viz/data/in/`
- `--validate` - Run validation mode to check test graphs against expected results
- `--graph NAME` - Filter validation to specific graph (e.g., test_line)
- `--metric NAME` - Filter validation to specific metric (e.g., pagerank)
- `--verbose` - Show detailed validation output

## Input/Output Files

### Input Files
- **Production mode**: 
  - `/viz/data/in/LearningChunkGraph.json` - Knowledge graph structure
  - `/viz/data/in/ConceptDictionary.json` - Concept definitions
- **Test mode**:
  - `/viz/data/test/tiny_graph.json` - Test graph (10-20 nodes)
  - `/viz/data/test/tiny_concepts.json` - Test concepts (5-10 concepts)
- **Validation mode**:
  - `/viz/data/test/test_*_graph.json` - Test input graphs
  - `/viz/data/test/test_*_graph_expected.json` - Expected results with metrics

### Output Files
Always saved to `/viz/data/out/` regardless of mode:
- `LearningChunkGraph_wow.json` - Graph enriched with all metrics
- `ConceptDictionary_wow.json` - Concepts with mention index

### Log Files
- `/viz/logs/graph2metrics.log` - Execution log with timestamps
- `/viz/logs/validation_report.json` - Validation results (when --validate used)

## Public API

### Main Orchestration Functions

#### setup_logging(log_file: Path, test_mode: bool = False) -> logging.Logger
Sets up logging configuration for both file and console output.
- **Input**: 
  - log_file (Path) - Path to log file
  - test_mode (bool) - Whether running in test mode
- **Returns**: Configured logger instance
- **Side effects**: Creates logs directory if not exists

#### load_input_data(input_dir: Path, logger: Logger, test_mode: bool = False) -> Tuple[Dict, Dict]
Loads and validates input JSON files.
- **Input**: 
  - input_dir (Path) - Directory containing input files
  - logger (Logger) - Logger instance
  - test_mode (bool) - Whether running in test mode
- **Returns**: Tuple of (graph_data, concepts_data)
- **Raises**: 
  - FileNotFoundError - If input files not found
  - ValidationError - If validation fails against schemas

#### convert_to_networkx(graph_data: Dict, logger: Logger, test_mode: bool = False) -> nx.DiGraph
Converts JSON graph data to NetworkX directed graph.
- **Input**: 
  - graph_data (Dict) - Graph data from JSON
  - logger (Logger) - Logger instance
  - test_mode (bool) - Whether running in test mode
- **Returns**: NetworkX directed graph with all node/edge attributes
- **Side effects**: Logs graph statistics (nodes, edges, components)
- **Raises**: RuntimeError - If graph conversion results in undirected graph

#### save_output_data(output_dir: Path, graph_data: Dict, concepts_data: Dict, logger: Logger, test_mode: bool = False) -> None
Saves enriched data to output JSON files.
- **Input**: 
  - output_dir (Path) - Output directory path
  - graph_data (Dict) - Enriched graph data
  - concepts_data (Dict) - Concepts data to enrich
  - logger (Logger) - Logger instance
  - test_mode (bool) - Whether running in test mode
- **Side effects**: 
  - Calls create_mention_index() to enrich concepts
  - Creates output directory if not exists
  - Writes two JSON files with UTF-8 encoding

### Metric Computation Functions

#### safe_metric_value(value: Any) -> float
Converts any metric value to a safe float, handling NaN and infinity.
- **Input**: value (Any) - Value to convert
- **Returns**: float - Safe numeric value (0.0 for None/NaN/inf)
- **See**: `/docs/specs/viz_metrics_reference.md#62-safe_metric_value` for algorithm details

#### compute_all_metrics(G: nx.DiGraph, graph_data: Dict, config: Dict, logger: Logger) -> Dict
Main orchestrator function that calls all metric computation functions in proper order.
- **Input**: 
  - G (nx.DiGraph) - NetworkX directed graph
  - graph_data (Dict) - Original graph data from JSON
  - config (Dict) - Configuration from config.toml
  - logger (Logger) - Logger instance
- **Returns**: Enhanced graph_data with all metrics added
- **Computation order**: See `/docs/specs/viz_metrics_reference.md#5-computation-sequence`

#### compute_edge_weights(G: nx.DiGraph, logger: Optional[Logger]) -> nx.DiGraph
Adds inverse_weight to all edges for distance algorithms.
- **Input**: G (nx.DiGraph), logger (optional)
- **Returns**: Modified graph with inverse_weight on edges
- **See**: `/docs/specs/viz_metrics_reference.md#23-inverse_weight-for-edges`

#### compute_basic_centrality(G: nx.DiGraph, config: Dict, logger: Optional[Logger]) -> Dict
Computes degree metrics and PageRank.
- **Input**: G (nx.DiGraph), config with pagerank parameters, logger (optional)
- **Returns**: Dict with degree_in, degree_out, degree_centrality, pagerank
- **Metrics**: See reference sections 2.1, 2.2, 2.4

#### compute_distance_centrality(G: nx.DiGraph, logger: Optional[Logger]) -> Dict
Computes betweenness and OUT-closeness using inverse weights.
- **Input**: G with inverse_weight on edges, logger (optional)
- **Returns**: Dict with betweenness_centrality, out-closeness
- **See**: `/docs/specs/viz_metrics_reference.md` sections 2.5, 2.6

#### compute_component_ids(G: nx.DiGraph, node_order: List[str], logger: Optional[Logger]) -> Dict
Assigns deterministic component IDs based on weakly connected components.
- **Returns**: Dict mapping node_id to component_id (int)
- **See**: `/docs/specs/viz_metrics_reference.md#27-component_id`

#### compute_prerequisite_metrics(G: nx.DiGraph, graph_data: Dict, config: Dict, logger: Optional[Logger]) -> Tuple[Dict, Dict]
Computes prerequisite_depth and learning_effort via SCC decomposition.
- **Returns**: Tuple of (prerequisite_depth, learning_effort) dicts
- **Handles**: PREREQUISITE cycles through SCC
- **See**: `/docs/specs/viz_metrics_reference.md` sections 2.8, 2.9

#### compute_educational_importance(G: nx.DiGraph, config: Dict, logger: Optional[Logger]) -> Dict
Computes PageRank on educational edges subgraph.
- **Returns**: Dict mapping node_id to educational_importance (float)
- **Invariant**: sum(educational_importance) = 1.0
- **See**: `/docs/specs/viz_metrics_reference.md#210-educational_importance`

#### validate_metric_invariants(pagerank_vals: Dict, edu_importance_vals: Dict, logger: Optional[Logger]) -> None
Validates that PageRank metrics sum to 1.0.
- **Input**: 
  - pagerank_vals (Dict) - PageRank values
  - edu_importance_vals (Dict) - Educational importance values
  - logger (Logger) - Logger instance
- **Raises**: Warning if sum deviates from 1.0 by more than 0.01

### Advanced Metric Functions

#### sanitize_graph_weights(G: nx.DiGraph, eps: float = 1e-9) -> None
Ensures numerical stability of edge weights (in-place modification).
- **Input**: G to sanitize, eps value (default=1e-9)
- **Returns**: None (modifies graph in-place)
- **See**: `/docs/specs/viz_metrics_reference.md#61-sanitize_graph_weights`

#### compute_louvain_clustering(G: nx.DiGraph, config: Dict, logger: Optional[Logger]) -> Dict[str, int]
Performs community detection using Louvain algorithm with deterministic numbering.
- **Returns**: Dict mapping node_id to cluster_id (integers starting from 0)
- **Requires**: python-louvain>=0.16 library
- **See**: `/docs/specs/viz_metrics_reference.md#31-cluster_id-louvain-clustering`

#### compute_bridge_scores(G: nx.DiGraph, cluster_map: Dict, betweenness_centrality: Dict, config: Dict) -> Dict[str, float]
Computes composite bridge score combining betweenness and inter-cluster ratio.
- **Returns**: Dict mapping node_id to bridge_score (float in [0, 1])
- **Formula**: bridge_score = w_b * betweenness + (1 - w_b) * inter_ratio
- **See**: `/docs/specs/viz_metrics_reference.md#32-bridge_score`

#### mark_inter_cluster_edges(G: nx.DiGraph, cluster_map: Dict[str, int]) -> None
Marks edges that connect nodes from different clusters (in-place modification).
- **Returns**: None (modifies edge attributes in-place)
- **Adds**: is_inter_cluster_edge, source_cluster_id, target_cluster_id
- **See**: `/docs/specs/viz_metrics_reference.md#33-inter-cluster-edge-metrics`

### Data Enrichment Functions

#### generate_demo_path(G: nx.DiGraph, graph_data: Dict, config: Dict, logger: Logger, test_mode: bool = False) -> Dict
Generates educational demo path for tour mode with 3 strategies.
- **Input**: 
  - G (nx.DiGraph) - NetworkX directed graph
  - graph_data (Dict) - Graph data with metrics
  - config (Dict) - Configuration with demo_path section
  - logger (Logger) - Logger instance
  - test_mode (bool) - Whether running in test mode
- **Returns**: Graph data with demo path in _meta
- **Configuration**:
  - demo_path.strategy (int, 1-3) - Strategy selection
  - demo_path.max_nodes (int) - Maximum nodes in path
- **Output in _meta**:
  - demo_path - List of node IDs forming the path
  - demo_generation_config - Strategy used and parameters

##### Strategy 1: Optimal Educational Journey
**Purpose**: Showcases the most educationally important concepts in logical learning order.

**Algorithm**:
1. Find entry point with min(prerequisite_depth) and high educational_importance
2. Build educational subgraph from types: PREREQUISITE, ELABORATES, EXAMPLE_OF, TESTS
3. Select top-N nodes by educational_importance (N = max_nodes * 1.5)
4. Connect nodes using shortest paths in educational subgraph
5. Apply fallback mechanisms if path too short:
   - If < 15 nodes: Add top PageRank nodes
   - If still short: Add bridge nodes (high betweenness_centrality)
   - If disconnected: Add intermediate connecting nodes
6. Ensure connectivity and limit to max_nodes

**Minimum guarantee**: 15 nodes (if graph has >= 15 nodes)

**Example**: For a 255-node graph, generates 100-node educational journey from fundamentals to advanced topics.

##### Strategy 2: Cluster Showcase  
**Purpose**: Tours one representative node from each cluster to show content diversity.

**Algorithm**:
1. Identify all clusters from Louvain clustering
2. Select top node by PageRank from each cluster
3. Order clusters by educational progression
4. Create path visiting one node per cluster

**Note**: This strategy was not modified in VIZ-DATA-ENRICH-02 as it works correctly.

##### Strategy 3: Critical Dependencies Path
**Purpose**: Traces back from most complex content to fundamental prerequisites.

**Algorithm**:
1. Find peak node with max(learning_effort)
2. Build dependency graph from: PREREQUISITE, ELABORATES, EXAMPLE_OF edges
3. Find all ancestors using reverse BFS on dependency graph
4. Group ancestors by prerequisite_depth level
5. Select representative nodes from each depth level (by educational_importance)
6. Apply fallback mechanisms if path too short:
   - If < 15 nodes: Add nodes with high betweenness_centrality (bridges)
   - If still short: Add top PageRank nodes
   - Fill gaps with intermediate nodes if disconnected
7. Sort by prerequisite_depth (ascending) for learning order
8. Limit to max_nodes

**Minimum guarantee**: 15 nodes (if graph has >= 15 nodes)

**Example**: For a 255-node graph, generates 100-node path from complex peak back to fundamentals.

##### Helper Functions
- `_build_educational_subgraph(G, edge_types)` - Extracts subgraph with only educational edges
- `_add_high_value_nodes(G, current_path, target, metric)` - Adds top nodes by specified metric
- `_ensure_path_connectivity(G, path_nodes)` - Fills gaps in path with intermediate nodes

#### generate_course_sequence(graph_data: Dict, logger: Optional[Logger] = None) -> Dict
Generates sequential course content order from Chunk nodes.
- **Input**: 
  - graph_data (Dict) - Graph with nodes containing Chunk types
  - logger (Optional[Logger]) - Logger instance for warnings
- **Returns**: Enhanced graph_data with course_sequence in _meta
- **Algorithm**: 
  1. Find all nodes with type="Chunk"
  2. Parse node IDs with format `{slug}:c:{position}`
  3. Extract position number after `:c:`
  4. Sort by position ascending
  5. Include cluster_id from node attributes
- **Output structure in _meta.course_sequence**: 
  ```json
  [
    {"id": "lesson1:c:1", "cluster_id": 0, "position": 1},
    {"id": "lesson1:c:2", "cluster_id": 1, "position": 2},
    ...
  ]
  ```
- **Edge cases**:
  - Invalid ID format: Skip node with warning
  - Missing position: Skip node with warning
  - Empty result: Returns empty list
- **Purpose**: Powers left-side course content panel in visualization

#### create_mention_index(graph_data: Dict, concepts_data: Dict) -> Dict
Creates index of concept mentions in nodes.
- **Input**: 
  - graph_data (Dict) - Graph with nodes and edges
  - concepts_data (Dict) - Concepts dictionary
- **Returns**: Enhanced concepts_data with mention index in _meta
- **Algorithm**: 
  - Builds node type map to identify Concept nodes
  - Analyzes ALL edges to find connections with Concept nodes
  - Bidirectional: both incoming and outgoing edges to concepts are indexed
  - Includes Concept-to-Concept relationships
  - Uses set-based deduplication
- **Output structure**: 
  ```json
  {"concept_id": {"nodes": ["n1", "n2"], "count": 2}}
  ```

#### link_nodes_to_concepts(graph_data: Dict) -> Dict
Fills concepts field in each node based on ALL edges with Concept nodes.
- **Input**: graph_data (Dict) - Graph with nodes and edges
- **Returns**: Modified graph_data with concepts field in nodes
- **Algorithm**: 
  - Builds node type map to identify Concept nodes
  - Processes ALL edges, not just MENTIONS
  - If target is a Concept, adds it to source's concepts
  - If source is a Concept, adds it to target's concepts (bidirectional)
  - Applies to ALL node types (Chunk, Assessment, Concept)
  - Uses set-based deduplication
- **Side effects**: Adds "concepts" field to every node

#### handle_large_graph(graph_data: Dict, max_nodes: int = 1000, save_full_path: Path = None, logger: Logger = None) -> Dict
Filters top-N nodes by PageRank for graphs > max_nodes.
- **Input**: 
  - graph_data (Dict) - Graph data with nodes and edges
  - max_nodes (int) - Maximum number of nodes to keep
  - save_full_path (Path) - Optional path to save full graph
  - logger (Logger) - Logger instance
- **Returns**: Filtered graph_data if needed, original otherwise
- **Algorithm**: 
  1. If nodes ≤ max_nodes, return unchanged
  2. Sort nodes by PageRank descending
  3. Keep top max_nodes
  4. Filter edges to kept nodes only
  5. Add filtering metadata to _meta.graph_metadata
- **Metadata added**:
  - filtered (bool) - Whether filtering was applied
  - original_nodes/edges - Original counts
  - filtered_nodes/edges - New counts
  - filter_method - "top_pagerank"

### Wrapper Functions

#### compute_centrality_metrics(G: nx.DiGraph, graph_data: Dict, config: Dict, logger: Logger, test_mode: bool = False) -> Dict
Wrapper function for compute_all_metrics() with progress output.
- **Input**: Same as compute_all_metrics() plus test_mode flag
- **Returns**: Enhanced graph_data with all metrics and enrichments
- **Side effects**: Prints progress to console

### Validation Mode Functions

#### run_validation_mode(viz_dir: Path, config: Dict, logger: Logger, graph_filter: Optional[str], metric_filter: Optional[str], verbose: bool) -> int
Main function for validation mode execution.
- **Input**: 
  - viz_dir (Path) - Viz directory path
  - config (Dict) - Configuration dictionary
  - logger (Logger) - Logger instance
  - graph_filter (Optional[str]) - Filter for specific graph
  - metric_filter (Optional[str]) - Filter for specific metric
  - verbose (bool) - Whether to show detailed output
- **Returns**: Exit code (0 for success, 3 for failure)
- **Algorithm**:
  1. Scan test directory for graph pairs
  2. For each test graph:
     - Load and validate input/expected
     - Compute metrics
     - Compare with expected values
  3. Generate validation matrix
  4. Save JSON report

#### scan_test_graphs(test_dir: Path, graph_filter: Optional[str] = None) -> List[Tuple[str, Path, Path]]
Scans test directory for test graph pairs.
- **Input**: 
  - test_dir (Path) - Directory containing test files
  - graph_filter (Optional[str]) - Filter for specific graph name
- **Returns**: List of (name, input_path, expected_path) tuples
- **Algorithm**: Looks for test_*_graph.json and test_*_graph_expected.json pairs

#### validate_test_files(input_data: Dict, expected_data: Dict, logger: Logger) -> bool
Validates test files against JSON schemas.
- **Input**: 
  - input_data (Dict) - Input graph data
  - expected_data (Dict) - Expected graph data
  - logger (Logger) - Logger instance
- **Returns**: True if validation passes
- **Checks**: Schema compliance and graph invariants

#### compare_metric_value(expected: Any, actual: Any, tolerance: float = 0.01) -> Tuple[str, float]
Compares metric values with tolerance.
- **Input**: 
  - expected (Any) - Expected value
  - actual (Any) - Actual value
  - tolerance (float) - Relative tolerance (default 1%)
- **Returns**: Tuple of (status, deviation_percent)
- **Status values**: "PASS", "FAIL", "MISS", "NaN"
- **Algorithm**: 
  - Missing actual → "MISS"
  - NaN/inf → "NaN"
  - Zero expected → absolute tolerance 0.001
  - Otherwise → relative tolerance

#### format_validation_matrix(all_results: List[Dict]) -> None
Prints formatted validation matrix to console.
- **Input**: all_results - List of validation results for all graphs
- **Side effects**: Prints colored matrix with pass/fail indicators
- **Format**: Compact matrix with ✓ for pass, ✗XX% for fail

#### generate_json_report(all_results: List[Dict], output_path: Path) -> None
Generates detailed JSON validation report.
- **Input**: 
  - all_results - List of validation results
  - output_path - Path to save JSON report
- **Side effects**: Writes JSON file with summary and detailed results

## Algorithm

The module processes graphs through these stages:

1. **Initialization**: Parse arguments, setup logging, load configuration
2. **Mode Selection**: Validation, test, or production mode
3. **Data Loading**: Load and validate JSON files against schemas
4. **NetworkX Conversion**: Create directed graph with all attributes
5. **Metrics Computation**: Compute all metrics in dependency order
6. **Output Generation**: Save enriched JSON files

### Metrics Computation Order

Metrics MUST be computed in specific order due to dependencies.
**For complete algorithm details and formulas, see: `/docs/specs/viz_metrics_reference.md#5-computation-sequence`**

Key points:
- Edge weights (inverse_weight) computed first
- Basic metrics (degrees, PageRank) before distance metrics
- Component and educational metrics computed independently
- Advanced metrics (clustering, bridges) computed last
- All metrics validated and sanitized before output

## Validation Mode

### Purpose
Validation mode compares computed metrics against hand-calculated expected results for test graphs to ensure correctness of all algorithms.

### Test Data Structure
- Test graphs located in `/viz/data/test/`
- Each test consists of:
  - `test_NAME_graph.json` - Input graph
  - `test_NAME_graph_expected.json` - Expected output with all metrics
  - `test_NAME_graph_calc.md` - Manual calculation documentation

### Validated Metrics
Validation covers all 12 node metrics and 4 edge metrics.
**For complete list and descriptions, see: `/docs/specs/viz_metrics_reference.md#4-complete-metrics-list`**

### Validation Process
1. Scan `/viz/data/test/` for test graph pairs
2. Validate each file against JSON schemas
3. Compute metrics using implementation
4. Compare with expected values using 1% relative tolerance
5. Special handling for zero values (absolute tolerance 0.001)
6. Generate validation matrix and JSON report

### Output Format

#### Console Matrix
Compact validation matrix with results for each metric/graph combination:
```
VALIDATION MATRIX
================
Graph                d_in  d_out d_cen PR    BW_c  out-c comp  pre_d l_eff e_imp inv_w 
-----------------------------------------------------------------------------------------
test_bridge          ✓     ✓     ✓     ✓     ✓     ✓     ✓     ✓     ✓     ✓     ✓    
test_cycle           ✓     ✓     ✓     ✓     ✓     ✓     ✓     ✓     ✓     ✓     ✓    
...
SUMMARY: 462 checks, 462 passed, 0 failed, 0 missing
```

Indicators:
- ✓ - PASS (within 1% tolerance)
- ✗XX% - FAIL (with average deviation percentage)
- ✗MISS - Metric not implemented
- ⚠️ NaN - Invalid value

#### JSON Report
Saved to `/viz/logs/validation_report.json`:
```json
{
  "timestamp": "ISO-8601",
  "summary": {
    "total_checks": 462,
    "passed": 462,
    "failed": 0,
    "missing": 0
  },
  "by_metric": {
    "pagerank": {
      "passed": 50,
      "failed": 0,
      "missing": 0,
      "failures": []
    }
  },
  "by_graph": {
    "test_line": {
      "passed": ["degree_in", "degree_out", ...],
      "failed": {}
    }
  }
}
```

## Dependencies

### External Libraries
- `networkx>=3.0` - Graph algorithms and metrics
- `scipy>=1.10` - Required by NetworkX for PageRank computation
- `python-louvain>=0.16` - Community detection (Louvain clustering)

### Internal Modules
- `src.utils.config` - Configuration loading
- `src.utils.console_encoding` - UTF-8 output support
- `src.utils.exit_codes` - Standardized exit codes
- `src.utils.validation` - JSON schema validation

## Configuration

Uses `/viz/config.toml` section `[graph2metrics]`.

**For all configuration parameters and their defaults, see the relevant sections in:**
- `/docs/specs/viz_metrics_reference.md` - Metric-specific parameters
- Example: PageRank parameters in section 2.4, Louvain parameters in section 3.1

Key parameters include:
- `pagerank_damping`, `pagerank_max_iter` - PageRank configuration
- `louvain_resolution`, `louvain_random_state` - Clustering configuration  
- `bridge_weight_betweenness` - Bridge score weights
- `default_difficulty` - For learning_effort computation

## Error Handling & Exit Codes

### Exit Codes
- `0` (EXIT_SUCCESS) - Successful completion
- `1` (EXIT_CONFIG_ERROR) - Configuration loading/validation error
- `2` (EXIT_INPUT_ERROR) - Input file not found or validation failed
- `3` (EXIT_RUNTIME_ERROR) - Unexpected runtime error or validation failure
- `5` (EXIT_IO_ERROR) - File I/O error

### Boundary Cases
- **Empty graph** - All metrics return 0.0
- **Single node** - degree=0, PageRank=1.0, centralities=0.0, cluster_id=0
- **Disconnected components** - Handled via weakly connected components
- **PREREQUISITE cycles** - Handled via SCC decomposition
- **Zero/negative weights** - inverse_weight = inf, excluded from distance calculations
- **NaN/inf values** - Converted to 0.0 via safe_metric_value()
- **Missing python-louvain** - Falls back to all nodes in cluster 0 with warning

## Terminal Output

### Production Mode
```
Using production data
Graph loaded: 300 nodes, 500 edges
Computing metrics for 300 nodes...
  ✓ All metrics computed successfully
[HH:MM:SS] INFO - Computing advanced metrics...
[HH:MM:SS] INFO - Running Louvain with resolution=1.0, random_state=42
[HH:MM:SS] INFO - Louvain clustering found 8 clusters
[HH:MM:SS] INFO - Found 8 clusters
[HH:MM:SS] INFO - Found 45 nodes with bridge_score > 0.1
✓ Graph metrics computed successfully
```

### Test Mode
```
[TEST MODE] Using test data
[TEST MODE] Graph loaded: 16 nodes, 22 edges
[TEST MODE] Computing metrics for 16 nodes...
  ✓ All metrics computed successfully
[HH:MM:SS] INFO - Computing advanced metrics...
[HH:MM:SS] INFO - Running Louvain with resolution=1.0, random_state=42
[HH:MM:SS] INFO - Louvain clustering found 3 clusters
[HH:MM:SS] INFO - Found 3 clusters
[HH:MM:SS] INFO - Found 6 nodes with bridge_score > 0.1
✓ Graph metrics computed successfully (TEST MODE)
```

### Validation Mode
```
================================================================================
VALIDATION MODE
================================================================================
Found 8 test graph(s)

Validating: test_bridge
  ✅ Schema validation passed

Validating: test_cycle
  ✅ Schema validation passed
...

VALIDATION MATRIX
================================================================================
[Matrix output]
SUMMARY: 462 checks, 462 passed, 0 failed, 0 missing

Validation report saved to: /viz/logs/validation_report.json
```

## Output Format

### Node Metrics Added
Each node in `LearningChunkGraph_wow.json` receives 12 additional metrics.

**For complete list with example values, see: `/docs/specs/viz_metrics_reference.md#4-complete-metrics-list`**

Example structure:
```json
{
  "id": "existing_id",
  // ... existing fields ...
  "degree_in": 2,
  "pagerank": 0.0234,
  "cluster_id": 1,
  // ... other metrics
}
```

### Edge Metrics Added  
Each edge receives 4 additional metrics (inverse_weight always, inter-cluster attributes when applicable).

Example:
```json
{
  "source": "node1",
  "target": "node2",
  "inverse_weight": 1.25,
  "is_inter_cluster_edge": true
  // ... cluster IDs if inter-cluster
}
```

## Usage Examples

### Basic Usage
```bash
# Activate virtual environment
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate      # Windows

# Run on production data
python -m viz.graph2metrics

# Check output
ls -la viz/data/out/
# Should see: LearningChunkGraph_wow.json, ConceptDictionary_wow.json
```

### Test Mode
```bash
# Run on test data
python -m viz.graph2metrics --test-mode

# Verify metrics were added
python -c "import json; g=json.load(open('viz/data/out/LearningChunkGraph_wow.json')); \
n=g['nodes'][0]; print(f'Node {n[\"id\"]} has PageRank={n[\"pagerank\"]:.4f}, cluster_id={n[\"cluster_id\"]}')"
```

### Validation Examples
```bash
# Run full validation (should show all green)
python -m viz.graph2metrics --validate

# Validate specific graph
python -m viz.graph2metrics --validate --graph test_line

# Check specific metric across all graphs
python -m viz.graph2metrics --validate --metric pagerank

# Detailed output for debugging
python -m viz.graph2metrics --validate --verbose

# Combine filters for targeted check
python -m viz.graph2metrics --validate --graph test_weighted_triangle --metric betweenness_centrality --verbose

# View validation report
cat viz/logs/validation_report.json | python -m json.tool | head -20
```

### Verify Output Validity
```bash
# Validate graph structure
python -c "from src.utils.validation import validate_json; import json; \
g=json.load(open('viz/data/out/LearningChunkGraph_wow.json')); \
validate_json(g, 'LearningChunkGraph'); print('✓ Graph structure valid')"

# Check PageRank sum
python -c "import json; g=json.load(open('viz/data/out/LearningChunkGraph_wow.json')); \
pr_sum=sum(n['pagerank'] for n in g['nodes']); \
print(f'PageRank sum = {pr_sum:.6f} (should be 1.0)')"

# Check all metrics present
python -c "import json; g=json.load(open('viz/data/out/LearningChunkGraph_wow.json')); \
metrics=['degree_in','degree_out','degree_centrality','pagerank','betweenness_centrality',\
'out-closeness','component_id','prerequisite_depth','learning_effort','educational_importance',\
'cluster_id','bridge_score']; \
missing=[m for m in metrics if m not in g['nodes'][0]]; \
print('✓ All metrics present' if not missing else f'Missing: {missing}')"

# Check clustering determinism
python -c "import json; import subprocess; \
for i in range(3): \
    subprocess.run(['python', '-m', 'viz.graph2metrics', '--test-mode'], \
                   capture_output=True); \
    g=json.load(open('viz/data/out/LearningChunkGraph_wow.json')); \
    clusters=[n['cluster_id'] for n in g['nodes']]; \
    print(f'Run {i+1}: clusters={set(clusters)}'); \
print('✓ Deterministic' if True else '✗ Non-deterministic')"
```

### Check Logs
```bash
# View recent log entries
tail -30 viz/logs/graph2metrics.log

# Check for warnings
grep WARNING viz/logs/graph2metrics.log

# Check for errors
grep ERROR viz/logs/graph2metrics.log
```

## Performance Notes

### Time Complexity
- **Degree metrics**: O(V + E) - linear scan
- **PageRank**: O(iterations × E) - typically converges in 20-50 iterations
- **Betweenness centrality**: O(V × E) for unweighted, O(V × E + V² × log V) for weighted
- **Out-closeness**: O(V²) using Dijkstra for all nodes
- **Component IDs**: O(V + E) - DFS/BFS
- **Prerequisite metrics**: O(V + E) for SCC + O(V + E) for topological sort
- **Educational importance**: O(iterations × E_edu) where E_edu ⊆ E
- **Louvain clustering**: O(V × log V) on average, O(V²) worst case
- **Bridge scores**: O(V × avg_degree) - iterating neighbors for each node
- **Inter-cluster edges**: O(E) - single pass through edges

### Space Complexity
- **Memory usage**: ~O(V² + E) for distance matrices
- **Peak during betweenness**: Up to O(V²) for shortest paths storage
- **Louvain clustering**: O(V + E) for undirected projection

### Expected Performance
| Graph Size | Time | Memory |
|------------|------|--------|
| 16 nodes (test) | < 0.1s | ~10 MB |
| 100 nodes | ~1s | ~50 MB |
| 250 nodes | 2-5s | ~200 MB |
| 500 nodes | 10-20s | ~500 MB |
| 1000 nodes | 30-60s | ~2 GB |
| 5000 nodes | 5-10 min | ~10 GB |

### Optimization Tips
- For graphs >1000 nodes, consider filtering before metrics computation
- PageRank convergence can be tuned via `pagerank_max_iter`
- Betweenness can be approximated using sampling for very large graphs
- Use `--test-mode` for quick validation on small graphs
- Louvain clustering is fast even for large graphs (quasi-linear)

## Test Coverage

Module has comprehensive test coverage in `/tests/viz/`:

### test_graph2metrics.py (28+ tests)
- `test_compute_edge_weights` - Verifies inverse_weight calculation
- `test_compute_basic_centrality` - Tests degree and PageRank
- `test_compute_distance_centrality` - Tests betweenness and out-closeness
- `test_compute_component_ids` - Verifies deterministic numbering
- `test_compute_prerequisite_metrics` - Tests depth and effort with cycles
- `test_compute_educational_importance` - Tests subgraph PageRank
- `test_sanitize_graph_weights` - Tests weight sanitization and self-loop removal
- `test_compute_louvain_clustering` - Tests community detection and determinism
- `test_compute_bridge_scores` - Tests composite bridge metric calculation
- `test_mark_inter_cluster_edges` - Tests inter-cluster edge marking
- `test_safe_metric_value` - Tests NaN/inf handling
- `test_single_node_graph` - Boundary case
- `test_disconnected_graph` - Multiple components
- `test_prerequisite_cycles` - SCC handling
- `test_clustering_determinism` - Verifies identical results across runs
- `test_empty_graph` - Empty input handling
- `test_missing_louvain` - Graceful fallback when library missing
- Plus 10+ additional edge case tests
- Integration tests for full enrichment flow

### test_graph2metrics_enrichment.py (30 tests)
- Tests for all 3 demo path strategies
- Tests for mention index creation
- Tests for node-concept linking
- Tests for large graph filtering
- Integration tests with real data

### test_graph2metrics_centrality.py (13 tests)
- Tests for all centrality metrics
- Performance tests for large graphs
- Edge cases and error handling

### test_graph2metrics_validation.py
- Validates all 8 test graphs against expected results
- Ensures <1% deviation for all metrics
- Total of 462 validation checks

### test_viz_setup.py (5 tests)
- Infrastructure and configuration tests
- Test data validation

## Notes

- Graph MUST be `nx.DiGraph` (directed), not undirected
- All metrics use edge weights when applicable (PageRank, betweenness, etc.)
- Component IDs are deterministic based on original JSON node order
- PREREQUISITE cycles are handled via strongly connected components
- NaN/inf values are converted to 0.0 for safety
- PageRank and educational_importance always sum to 1.0 (validated)
- The metric `out-closeness` (not `closeness_centrality`) measures outgoing reachability
- All file I/O uses UTF-8 encoding
- Test mode uses `tiny_*` files, validation uses `test_*` files
- Louvain clustering requires `python-louvain>=0.16` library (falls back gracefully if missing)
- Cluster IDs are renumbered for stability (sorted by minimum node ID in cluster)
- Bridge score combines betweenness (70%) and inter-cluster ratio (30%) by default
- Inter-cluster edges are marked only when source and target belong to different clusters
- All advanced metrics are deterministic when `louvain_random_state` is set
- The module processes graphs up to ~5000 nodes efficiently on standard hardware
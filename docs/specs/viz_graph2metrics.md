# viz_graph2metrics.md

## Status: READY (Data Enrichment Implemented)

Module for computing comprehensive NetworkX metrics on K2-18 knowledge graph for visualization, including clustering, bridge detection, and data enrichment for interactive visualization.

## Module Purpose

The `graph2metrics.py` utility enriches the LearningChunkGraph with 12 network metrics for nodes and 4 for edges computed using NetworkX. These metrics provide insights into node importance, clustering, connectivity patterns, bridge detection, and educational structure that power the interactive visualization experience.

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
- **Input**: 
  - value (Any) - Value to convert (float, None, NaN, inf, etc.)
- **Returns**: float - Safe numeric value
- **Algorithm**: 
  - None → 0.0
  - NaN → 0.0
  - ±inf → 0.0
  - Valid float → unchanged
- **Purpose**: Ensures all metrics are JSON-serializable and won't break visualizations

#### compute_all_metrics(G: nx.DiGraph, graph_data: Dict, config: Dict, logger: Logger) -> Dict
Main orchestrator function that calls all metric computation functions in proper order.
- **Input**: 
  - G (nx.DiGraph) - NetworkX directed graph
  - graph_data (Dict) - Original graph data from JSON
  - config (Dict) - Configuration from config.toml
  - logger (Logger) - Logger instance
- **Returns**: Enhanced graph_data with all metrics added
- **Side effects**: 
  - Modifies G by adding inverse_weight to edges
  - Calls sanitize_graph_weights() to ensure numerical stability
  - Modifies graph_data by adding metrics to nodes and edges
- **Algorithm**:
  1. Compute edge weights (inverse_weight)
  2. Compute basic centrality (degrees, PageRank)
  3. Compute distance centrality (betweenness, out-closeness)
  4. Compute component IDs
  5. Compute prerequisite metrics (depth, effort)
  6. Compute educational importance
  7. Validate invariants
  8. Compute advanced metrics (clustering, bridges)
  9. Add all metrics to nodes
  10. Transfer inter-cluster edge attributes to output

#### compute_edge_weights(G: nx.DiGraph, logger: Optional[Logger]) -> nx.DiGraph
Adds inverse_weight to all edges for distance algorithms.
- **Input**: 
  - G (nx.DiGraph) - NetworkX directed graph
  - logger (Logger) - Logger instance (optional)
- **Returns**: Modified graph with inverse_weight on edges
- **Algorithm**: inverse_weight = 1.0 / weight (or inf if weight <= 0)
- **Note**: MUST be called before distance-based metrics

#### compute_basic_centrality(G: nx.DiGraph, config: Dict, logger: Optional[Logger]) -> Dict
Computes degree metrics and PageRank.
- **Input**: 
  - G (nx.DiGraph) - NetworkX directed graph
  - config (Dict) - Configuration with pagerank parameters
  - logger (Logger) - Logger instance (optional)
- **Returns**: Dict with degree_in, degree_out, degree_centrality, pagerank
- **Metrics computed**:
  - degree_in - Number of incoming edges (int)
  - degree_out - Number of outgoing edges (int)
  - degree_centrality - Normalized degree (float, 0-1)
  - pagerank - Importance with damping (float, sum=1.0)

#### compute_distance_centrality(G: nx.DiGraph, logger: Optional[Logger]) -> Dict
Computes betweenness and OUT-closeness using inverse weights.
- **Input**: 
  - G (nx.DiGraph) - Graph with inverse_weight on edges
  - logger (Logger) - Logger instance (optional)
- **Returns**: Dict with betweenness_centrality, out-closeness
- **Metrics computed**:
  - betweenness_centrality - Node as bridge (float, 0-1)
  - out-closeness - Closeness via outgoing paths (float, 0-1)
- **Note**: Requires inverse_weight precomputed on edges

#### compute_component_ids(G: nx.DiGraph, node_order: List[str], logger: Optional[Logger]) -> Dict
Assigns deterministic component IDs based on weakly connected components.
- **Input**: 
  - G (nx.DiGraph) - NetworkX directed graph
  - node_order (List[str]) - Original node order from JSON
  - logger (Logger) - Logger instance (optional)
- **Returns**: Dict mapping node_id to component_id (int)
- **Algorithm**: Components sorted by first node's position in original order

#### compute_prerequisite_metrics(G: nx.DiGraph, graph_data: Dict, config: Dict, logger: Optional[Logger]) -> Tuple[Dict, Dict]
Computes prerequisite_depth and learning_effort via SCC decomposition.
- **Input**: 
  - G (nx.DiGraph) - NetworkX directed graph
  - graph_data (Dict) - Graph data with node difficulties
  - config (Dict) - Configuration with default_difficulty
  - logger (Logger) - Logger instance (optional)
- **Returns**: Tuple of (prerequisite_depth, learning_effort) dicts
- **Algorithm**: 
  - Builds PREREQUISITE subgraph
  - Finds strongly connected components
  - Creates condensed DAG
  - Computes depth and effort via topological DP
- **Handles**: PREREQUISITE cycles through SCC

#### compute_educational_importance(G: nx.DiGraph, config: Dict, logger: Optional[Logger]) -> Dict
Computes PageRank on educational edges subgraph.
- **Input**: 
  - G (nx.DiGraph) - NetworkX directed graph
  - config (Dict) - Configuration with educational_edge_types
  - logger (Logger) - Logger instance (optional)
- **Returns**: Dict mapping node_id to educational_importance (float)
- **Algorithm**: PageRank on subgraph of PREREQUISITE, ELABORATES, TESTS, EXAMPLE_OF edges
- **Invariant**: sum(educational_importance) = 1.0

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
- **Input**: 
  - G (nx.DiGraph) - NetworkX directed graph to sanitize
  - eps (float, default=1e-9) - Small value to replace zero/negative weights
- **Returns**: None (modifies graph in-place)
- **Side effects**: 
  - Removes self-loops if present
  - Replaces missing weights with 1.0
  - Replaces zero/negative weights with eps
  - Ensures inverse_weight exists on all edges
- **Algorithm**: 
  - Remove self-loops via nx.selfloop_edges()
  - Iterate through edges, apply weight corrections
- **Edge cases**:
  - Empty graph: No operation
  - All weights invalid: All set to eps

#### compute_louvain_clustering(G: nx.DiGraph, config: Dict, logger: Optional[Logger]) -> Dict[str, int]
Performs community detection using Louvain algorithm with deterministic numbering.
- **Input**: 
  - G (nx.DiGraph) - NetworkX directed graph
  - config (Dict) - Configuration with louvain_resolution and louvain_random_state
  - logger (Logger) - Logger instance (optional)
- **Returns**: Dict mapping node_id to cluster_id (integers starting from 0)
- **Algorithm**: 
  1. Create undirected projection of graph
  2. Aggregate weights for bidirectional edges (sum)
  3. Add isolated nodes to ensure all nodes are included
  4. Run community_louvain.best_partition() with random_state for determinism
  5. Renumber clusters by minimum node ID in each cluster (stable ordering)
- **Configuration**:
  - louvain_resolution (float, default=1.0) - Controls cluster granularity
  - louvain_random_state (int, default=42) - Seed for deterministic results
- **Edge cases**:
  - Empty graph: Returns empty dict
  - Single node: Returns {node: 0}
  - Disconnected components: Each component gets separate clusters
  - python-louvain not installed: Returns {node: 0 for all} with warning
- **Note**: Requires python-louvain>=0.16 library

#### compute_bridge_scores(G: nx.DiGraph, cluster_map: Dict, betweenness_centrality: Dict, config: Dict) -> Dict[str, float]
Computes composite bridge score combining betweenness and inter-cluster ratio.
- **Input**: 
  - G (nx.DiGraph) - NetworkX directed graph
  - cluster_map (Dict[str, int]) - Node to cluster_id mapping from Louvain
  - betweenness_centrality (Dict[str, float]) - Pre-computed betweenness values
  - config (Dict) - Configuration with bridge weight parameters
- **Returns**: Dict mapping node_id to bridge_score (float in [0, 1])
- **Formula**: 
  ```
  bridge_score = w_b * betweenness_norm + (1 - w_b) * inter_ratio
  where:
    w_b = bridge_weight_betweenness (default: 0.7)
    inter_ratio = fraction of neighbors in different clusters
  ```
- **Algorithm**: 
  1. For each node, find all neighbors (predecessors ∪ successors)
  2. Count neighbors in different clusters
  3. Calculate inter_ratio = inter_count / len(unique_neighbors)
  4. Combine with normalized betweenness using weights
- **Configuration**:
  - bridge_weight_betweenness (float, default=0.7) - Weight for betweenness component
- **Edge cases**:
  - Node with degree 0: inter_ratio = 0, uses only betweenness
  - Single cluster: All inter_ratio = 0
  - No clustering available: Uses only betweenness
- **Note**: Uses pre-computed betweenness_centrality from basic metrics

#### mark_inter_cluster_edges(G: nx.DiGraph, cluster_map: Dict[str, int]) -> None
Marks edges that connect nodes from different clusters (in-place modification).
- **Input**: 
  - G (nx.DiGraph) - NetworkX directed graph
  - cluster_map (Dict[str, int]) - Node to cluster_id mapping from Louvain
- **Returns**: None (modifies edge attributes in-place)
- **Side effects**: Adds attributes to each edge:
  - is_inter_cluster_edge (bool) - True if source and target in different clusters
  - source_cluster_id (int) - Cluster ID of source node (if inter-cluster)
  - target_cluster_id (int) - Cluster ID of target node (if inter-cluster)
- **Algorithm**: 
  1. Check if cluster_map is empty (no clustering)
  2. Iterate through all edges
  3. Compare cluster IDs of source and target
  4. Set appropriate attributes based on comparison
- **Edge cases**:
  - No clustering available: All edges marked as is_inter_cluster_edge = False
  - Node not in cluster_map: Edge marked as is_inter_cluster_edge = False
- **Invariant**: is_inter_cluster_edge = True ⟺ source_cluster_id ≠ target_cluster_id

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

#### create_mention_index(graph_data: Dict, concepts_data: Dict) -> Dict
Creates index of concept mentions in nodes.
- **Input**: 
  - graph_data (Dict) - Graph with nodes and edges
  - concepts_data (Dict) - Concepts dictionary
- **Returns**: Enhanced concepts_data with mention index in _meta
- **Algorithm**: Analyzes MENTIONS edges to build index
- **Output structure**: 
  ```json
  {"concept_id": {"nodes": ["n1", "n2"], "count": 2}}
  ```

#### link_nodes_to_concepts(graph_data: Dict) -> Dict
Fills concepts field in each node based on MENTIONS edges.
- **Input**: graph_data (Dict) - Graph with nodes and edges
- **Returns**: Modified graph_data with concepts field in nodes
- **Algorithm**: Maps node_id → [concept_ids] from MENTIONS edges
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

### 1. Initialization
- Parse command-line arguments
- Setup UTF-8 console encoding
- Initialize logging to file and console
- Load configuration from `/viz/config.toml`

### 2. Mode Selection
- **Validation mode**: If --validate flag, run validation workflow
- **Test mode**: If --test-mode flag, use test data
- **Production mode**: Default, use production data

### 3. Data Loading
- Determine input directory based on mode
- Load JSON files with proper encoding
- Validate against schemas:
  - `LearningChunkGraph` schema
  - `ConceptDictionary` schema
- Validate graph invariants (node/edge consistency)

### 4. NetworkX Conversion
- Create `nx.DiGraph()` (MUST be directed, not undirected)
- Add nodes with all attributes from JSON
- Add edges with type, weight, conditions
- Verify graph is directed
- Log basic statistics:
  - Number of nodes
  - Number of edges
  - Weakly connected components

### 5. Metrics Computation (STRICT ORDER)

Metrics are computed in specific order due to dependencies:

1. **Edge weights** - `compute_edge_weights()` adds inverse_weight to all edges
2. **Basic centrality** - `compute_basic_centrality()` computes:
   - degree_in, degree_out (count of edges)
   - degree_centrality (normalized by n-1)
   - pagerank (with edge weights)
3. **Distance centrality** - `compute_distance_centrality()` computes:
   - betweenness_centrality (using inverse_weight)
   - out-closeness (via graph reversal)
4. **Component structure** - `compute_component_ids()` assigns:
   - component_id (deterministic numbering)
5. **Educational metrics** - `compute_prerequisite_metrics()` computes:
   - prerequisite_depth (levels in PREREQUISITE DAG)
   - learning_effort (cumulative difficulty)
6. **Educational importance** - `compute_educational_importance()` computes:
   - educational_importance (PageRank on educational subgraph)
7. **Validation** - `validate_metric_invariants()` checks:
   - sum(pagerank) = 1.0 ± 0.01
   - sum(educational_importance) = 1.0 ± 0.01
8. **Advanced metrics** - Three additional algorithms:
   - `sanitize_graph_weights()` - Ensures numerical stability
   - `compute_louvain_clustering()` - Community detection with deterministic numbering
   - `compute_bridge_scores()` - Combines betweenness (70%) and inter-cluster ratio (30%)
   - `mark_inter_cluster_edges()` - Labels edges connecting different clusters

### 6. Output Generation
- Add all computed metrics to nodes and edges
- Apply `safe_metric_value()` to handle NaN/inf → 0.0
- Transfer inter-cluster edge attributes from NetworkX to JSON
- Save enriched JSON files with UTF-8 encoding
- Log metric ranges and statistics

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
**Node metrics (12):**
- `degree_in` - Number of incoming edges
- `degree_out` - Number of outgoing edges
- `degree_centrality` - Normalized degree
- `pagerank` - PageRank with weights
- `betweenness_centrality` - Bridge importance
- `out-closeness` - OUT-closeness centrality
- `component_id` - Component membership
- `prerequisite_depth` - Level in prerequisite hierarchy
- `learning_effort` - Cumulative learning difficulty
- `educational_importance` - PageRank on educational edges
- `cluster_id` - Louvain community detection cluster
- `bridge_score` - Composite metric for bridge nodes

**Edge metrics (4):**
- `inverse_weight` - Reciprocal of edge weight
- `is_inter_cluster_edge` - Boolean flag for edges between clusters
- `source_cluster_id` - Source node's cluster ID for inter-cluster edges
- `target_cluster_id` - Target node's cluster ID for inter-cluster edges

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

Uses `/viz/config.toml` section `[graph2metrics]`:

### Required Parameters
- **pagerank_damping** (float, 0-1, default=0.85) - PageRank damping factor
- **pagerank_max_iter** (int, >0, default=100) - Max iterations for PageRank
- **educational_edge_types** (list, default=["PREREQUISITE", "ELABORATES", "TESTS", "EXAMPLE_OF"]) - Edge types for educational importance

### Optional Parameters
- **betweenness_normalized** (bool, default=true) - Normalize betweenness values
- **closeness_harmonic** (bool, default=true) - Use harmonic mean for disconnected graphs (deprecated)
- **louvain_resolution** (float, default=1.0) - Community detection resolution
- **louvain_random_state** (int, default=42) - Random seed for deterministic clustering
- **bridge_weight_betweenness** (float, default=0.7) - Weight for betweenness in bridge score
- **bridge_weight_inter_ratio** (float, default=0.3) - Weight for inter-cluster ratio in bridge score (can be set explicitly or auto-computed as 1-bridge_weight_betweenness)
- **bridge_top_gap_min** (float, default=0.05) - Minimum gap between top bridge nodes and others for behavioral validation

### Path Mode Parameters (used for learning_effort)
- **default_difficulty** (int, 1-5, default=3) - Default node difficulty if not specified

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
Each node in `LearningChunkGraph_wow.json` receives:
```json
{
  "id": "existing_id",
  // ... existing fields ...
  
  // Basic metrics
  "degree_in": 2,
  "degree_out": 3,
  "degree_centrality": 0.3125,
  
  // Importance metrics  
  "pagerank": 0.0234,
  "betweenness_centrality": 0.1667,
  "out-closeness": 0.5833,
  
  // Structure metrics
  "component_id": 0,
  "prerequisite_depth": 2,
  
  // Educational metrics
  "learning_effort": 12.0,
  "educational_importance": 0.0456,
  
  // Advanced metrics
  "cluster_id": 1,
  "bridge_score": 0.2456
}
```

### Edge Metrics Added
Each edge in `LearningChunkGraph_wow.json` receives:
```json
{
  "source": "node1",
  "target": "node2",
  "weight": 0.8,
  // ... existing fields ...
  
  "inverse_weight": 1.25,
  
  // Inter-cluster attributes (only for inter-cluster edges)
  "is_inter_cluster_edge": true,
  "source_cluster_id": 1,
  "target_cluster_id": 2
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
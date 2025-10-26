# viz_graph_split.md

## Status: READY

Utility for splitting enriched knowledge graph into separate files by clusters. Extracts each cluster as independent subgraph with proper node/edge filtering, statistics, and metadata.

## CLI Interface

### Usage
```bash
python -m viz.graph_split
```

### Input Files
- **Source**: `/viz/data/out/LearningChunkGraph_wow.json`
- **Format**: Enriched graph with cluster_id on all nodes (from graph2metrics)

### Output Files
- **Target**: `/viz/data/out/LearningChunkGraph_cluster_{ID}.json` (one per cluster)
- **Naming**: Cluster ID in filename (e.g., `LearningChunkGraph_cluster_0.json`)

### Log Files
- `/viz/logs/graph_split.log` - Execution log with timestamps and statistics

### Exit Codes
- **0 (SUCCESS)** - Successful completion
- **1 (CONFIG_ERROR)** - Configuration errors
- **2 (INPUT_ERROR)** - Input file not found or invalid
- **3 (RUNTIME_ERROR)** - Processing errors
- **5 (IO_ERROR)** - File system errors

## Terminal Output

### Format
```
[HH:MM:SS] LEVEL    | Message
```

### Levels
- **START**: Beginning of processing (cyan+bright)
- **INFO**: Informational messages (blue)
- **WARNING**: Warning messages (yellow)
- **SUCCESS**: Success message (green+bright)
- **ERROR**: Error messages (red+bright)

### Example Output
```
[12:34:56] START    | Graph Split by Clusters
[12:34:56] INFO     | Loading: LearningChunkGraph_wow.json
[12:34:56] INFO     | Graph: 255 nodes, 847 edges
[12:34:56] INFO     | Found 16 clusters
[12:34:56] INFO     | Finding inter-cluster links (PREREQUISITE, ELABORATES)...
[12:34:56] INFO     | Found inter-cluster links for 14 clusters

Processing clusters:
[12:34:56] INFO     | Cluster 0: 45 nodes, 123 edges kept, 34 inter-cluster edges removed
[12:34:56] INFO     |   └─ Inter-cluster links: 0 incoming, 3 outgoing
[12:34:56] INFO     | Cluster 1: 32 nodes, 87 edges kept, 21 inter-cluster edges removed
[12:34:56] INFO     |   └─ Inter-cluster links: 2 incoming, 1 outgoing
[12:34:56] WARNING  | Skipping cluster 3: only 1 node
[12:34:56] INFO     | Cluster 2: 28 nodes, 76 edges kept, 18 inter-cluster edges removed
...

[12:34:57] SUCCESS  | ✅ Split completed successfully
[12:34:57] INFO     | Output files saved to: /viz/data/out/
[12:34:57] INFO     | Created 15 cluster files (1 skipped)
```

## Core Algorithm

### Overview
Splits enriched knowledge graph into separate subgraphs by cluster_id, with proper filtering, sorting, and statistics.

### Processing Steps

#### 1. Filter Nodes
- Select nodes where `node['cluster_id'] == current_cluster_id`
- Create set of selected node IDs for fast lookup
- **Important**: ALL nodes MUST have `cluster_id` (validated by graph2metrics), no special handling needed

#### 2. Filter Edges
- Select edges where BOTH source AND target are in the selected node set
- Count inter-cluster edges for statistics:
  ```python
  # Inter-cluster edges = edges where exactly one endpoint is in cluster
  inter_cluster_count = 0
  for edge in all_edges:
      source_in_cluster = edge['source'] in cluster_nodes
      target_in_cluster = edge['target'] in cluster_nodes
      if source_in_cluster != target_in_cluster:  # XOR - exactly one is in cluster
          inter_cluster_count += 1
  ```
- This shows "how connected the cluster was to the rest of the graph"

#### 3. Sort Nodes
- Split into two groups:
  - Concept nodes: `type == "Concept"`
  - Other nodes: `type == "Chunk"` or `type == "Assessment"`
- Sort Concept nodes by `id` (alphabetically)
- Keep other nodes in original order (they're already sorted by position in their IDs)
- Concatenate: all Concepts first, then all others

#### 4. Edges
- Keep in original order (no sorting)

#### 5. Metadata
- **DELETE** entire `_meta` section from source graph (but remember `title` field)
- **CREATE NEW** `_meta` with only:
  ```json
  {
    "_meta": {
      "title": "<original title>",
      "subtitle": "Cluster {cluster_id} | Nodes {node_count} | Edges {edge_count}"
    }
  }
  ```

#### 6. Statistics for Console Output
For each cluster, calculate and display:
- Node count: nodes in cluster
- Edge count: edges kept (both endpoints in cluster)
- Inter-cluster edges: edges removed (exactly one endpoint in cluster)

#### 7. Find Inter-Cluster Links
- Extract all PREREQUISITE and ELABORATES edges between Concept nodes from different clusters
- For each cluster, identify top-3 incoming and top-3 outgoing links
- Sort by source node's educational_importance (descending)
- Include both link types without artificial priority

See detailed algorithm in "Inter-Cluster Links Metadata" section below.

#### 8. Boundary Cases
- **Cluster with 1 node, 0 edges:**
  - Skip file creation
  - Print WARNING to console: "Skipping cluster {id}: only 1 node"
- **Processing order:** Sort clusters by ID ascending (0, 1, 2, ...)

## Inter-Cluster Links Metadata

### Purpose
Preserve semantic dependencies between clusters to enable navigation and content generation.

### Link Types Included
- **PREREQUISITE** (≥0.80 weight) - Learning order dependencies (A must be understood before B)
- **ELABORATES** (0.50-0.75 weight) - Conceptual depth relationships (B deepens understanding of A)
- **EXAMPLE_OF** (0.50-0.75 weight) - Illustrative examples and demonstrations
- **TESTS** (≥0.80 weight) - Assessment relationships (A tests understanding of B)

Other edge types (PARALLEL, HINT_FORWARD, REFER_BACK, etc.) are excluded as they don't affect learning trajectory.

### Algorithm
For each cluster, find top-3 inter-cluster links in each direction:
- **incoming**: Concepts from other clusters that are prerequisites for or elaborate on concepts in this cluster
- **outgoing**: Concepts in this cluster that are prerequisites for or elaborate on concepts in other clusters

### Selection Criteria
1. Edge types: PREREQUISITE, ELABORATES, EXAMPLE_OF, TESTS
2. Node requirements:
   - For TESTS: source must be Assessment (target can be any type)
   - For others: at least one node must be Concept (source OR target)
3. Source and target must be in different clusters
4. Sort by `educational_importance` of source node (descending)
5. Keep top 3 for each direction (no artificial priority by edge type)

### Metadata Format
```json
{
  "_meta": {
    "title": "Course Title",
    "subtitle": "Cluster 1 | Nodes 28 | Edges 76",
    "inter_cluster_links": {
      "incoming": [
        {
          "from_cluster": 0,
          "source": "concept_id",
          "source_text": "Concept Name",
          "source_type": "Concept",
          "source_importance": 0.045,
          "target": "chunk_id",
          "target_text": "Chunk content...",
          "target_type": "Chunk",
          "target_importance": 0.032,
          "type": "PREREQUISITE",
          "weight": 0.9,
          "conditions": "Optional semantic explanation"
        }
      ],
      "outgoing": [...]
    }
  }
}
```

### Fields
- `from_cluster`/`to_cluster` (int) - ID of the other cluster
- `source` (str) - Source node ID
- `source_text` (str) - Source node text (for readability)
- `source_type` (str) - Source node type ("Concept", "Chunk", or "Assessment")
- `source_importance` (float) - Source educational_importance (used for sorting)
- `target` (str) - Target node ID
- `target_text` (str) - Target node text (for readability)
- `target_type` (str) - Target node type ("Concept", "Chunk", or "Assessment")
- `target_importance` (float) - Target educational_importance (full context)
- `type` (str) - Edge type ("PREREQUISITE", "ELABORATES", "EXAMPLE_OF", or "TESTS")
- `weight` (float) - Edge weight [0.0-1.0]
- `conditions` (str, optional) - Semantic explanation of the relationship

## Public API

### setup_logging(log_file: Path) -> logging.Logger
Set up logging configuration with file and console handlers.
- **Input**:
  - log_file (Path) - Path to log file
- **Returns**: Configured logger instance
- **Side effects**: Creates logs directory if not exists

### load_graph(input_file: Path, logger: Logger) -> Dict
Load and validate graph against LearningChunkGraph schema.
- **Input**:
  - input_file (Path) - Path to graph JSON file
  - logger (Logger) - Logger instance
- **Returns**: Graph data dictionary
- **Raises**:
  - SystemExit(EXIT_INPUT_ERROR) - If file not found or validation fails
  - SystemExit(EXIT_IO_ERROR) - If file read fails
- **Validation**: Uses `validate_json(data, "LearningChunkGraph")`

### identify_clusters(graph_data: Dict, logger: Logger) -> List[int]
Find all unique cluster_id values, sorted ascending.
- **Input**:
  - graph_data (Dict) - Graph data with nodes
  - logger (Logger) - Logger instance
- **Returns**: Sorted list of cluster IDs (integers)
- **Algorithm**: Extract unique cluster_id from all nodes, sort ascending

### extract_cluster(graph_data: Dict, cluster_id: int, logger: Logger) -> Tuple[Dict, int, int, int]
Extract single cluster subgraph with statistics.
- **Input**:
  - graph_data (Dict) - Full graph data
  - cluster_id (int) - Cluster ID to extract
  - logger (Logger) - Logger instance
- **Returns**: Tuple of (cluster_graph, node_count, edge_count, inter_cluster_edges)
  - cluster_graph (Dict) - Subgraph containing only cluster nodes/edges
  - node_count (int) - Number of nodes in cluster
  - edge_count (int) - Number of edges kept (both endpoints in cluster)
  - inter_cluster_edges (int) - Number of inter-cluster edges removed
- **Algorithm**:
  1. Filter nodes by cluster_id
  2. Create node_ids set for fast lookup
  3. Filter edges: keep if BOTH endpoints in cluster
  4. Count inter-cluster edges using XOR logic
  5. Return tuple with statistics

### sort_nodes(nodes: List[Dict]) -> List[Dict]
Sort nodes: Concepts first (by id alphabetically), then others (preserve order).
- **Input**:
  - nodes (List[Dict]) - List of graph nodes
- **Returns**: Sorted list of nodes
- **Algorithm**:
  - Split into Concepts (type=="Concept") and Others
  - Sort Concepts by 'id' field alphabetically
  - Concatenate: Concepts + Others (original order preserved)

### find_inter_cluster_links(graph_data: Dict, cluster_map: Dict[str, int], logger: Logger) -> Dict[int, Dict[str, List[Dict]]]
Find PREREQUISITE, ELABORATES, EXAMPLE_OF, and TESTS links between nodes from different clusters.
- **Input**:
  - graph_data (Dict) - Full graph with nodes and edges
  - cluster_map (Dict[str, int]) - Mapping {node_id: cluster_id}
  - logger (Logger) - Logger instance
- **Returns**: Dictionary mapping cluster_id to {"incoming": [...], "outgoing": [...]}
  - Each link contains: source, source_text, source_type, source_importance, target, target_text, target_type, target_importance, type, weight, conditions (optional), and from_cluster/to_cluster depending on direction
- **Algorithm**:
  1. Filter edges by type (PREREQUISITE, ELABORATES, EXAMPLE_OF, TESTS)
  2. Filter nodes by type:
     - For TESTS: source must be Assessment
     - For others: at least one node must be Concept
  3. Find edges crossing cluster boundaries
  4. Sort by source node educational_importance
  5. Keep top-3 for each direction per cluster

### create_cluster_metadata(cluster_id: int, node_count: int, edge_count: int, original_title: str, inter_cluster_links: Optional[Dict[str, List[Dict]]] = None) -> Dict
Create new _meta section with subtitle and optional inter-cluster links.
- **Input**:
  - cluster_id (int) - Cluster ID
  - node_count (int) - Number of nodes in cluster
  - edge_count (int) - Number of edges in cluster
  - original_title (str) - Title from source graph
  - inter_cluster_links (Dict, optional) - Dict with "incoming" and "outgoing" lists
- **Returns**: New _meta dictionary with optional inter_cluster_links section
- **Format**:
  ```json
  {
    "title": "<original_title>",
    "subtitle": "Cluster {cluster_id} | Nodes {node_count} | Edges {edge_count}",
    "inter_cluster_links": {  // optional, only if links exist
      "incoming": [...],
      "outgoing": [...]
    }
  }
  ```

### save_cluster_graph(cluster_graph: Dict, cluster_id: int, output_dir: Path, logger: Logger) -> None
Validate and save cluster graph to file.
- **Input**:
  - cluster_graph (Dict) - Cluster subgraph to save
  - cluster_id (int) - Cluster ID for filename
  - output_dir (Path) - Output directory path
  - logger (Logger) - Logger instance
- **Side effects**:
  - Validates graph against LearningChunkGraph schema
  - Writes file to `{output_dir}/LearningChunkGraph_cluster_{cluster_id}.json`
- **Raises**:
  - SystemExit(EXIT_RUNTIME_ERROR) - If validation fails
  - SystemExit(EXIT_IO_ERROR) - If save fails

### main() -> int
Main entry point for the utility.
- **Returns**: Exit code (0 for success, non-zero for errors)
- **Algorithm**:
  1. Setup console encoding
  2. Setup logging
  3. Load and validate input graph
  4. Identify all clusters (sorted)
  5. For each cluster:
     - Extract cluster subgraph
     - Check if single node → skip with WARNING
     - Sort nodes (Concepts first)
     - Create metadata
     - Save cluster graph
     - Log statistics
  6. Print success summary
  7. Return EXIT_SUCCESS

## Error Handling & Exit Codes

### Exit Codes
- **0 (EXIT_SUCCESS)** - Successful completion
- **1 (EXIT_CONFIG_ERROR)** - Configuration file not found or invalid
- **2 (EXIT_INPUT_ERROR)** - Input file not found or validation failed
- **3 (EXIT_RUNTIME_ERROR)** - Processing error or validation failure
- **5 (EXIT_IO_ERROR)** - File read/write errors

### Error Conditions
- **Missing input file**: Exit with EXIT_INPUT_ERROR
- **Invalid JSON**: Exit with EXIT_INPUT_ERROR
- **Schema validation failure**: Exit with EXIT_INPUT_ERROR
- **Empty cluster list**: Continue with warning (all nodes may lack cluster_id)
- **File write error**: Exit with EXIT_IO_ERROR

### Boundary Cases
- **Single-node cluster**: Skip with WARNING, don't create file
- **Cluster with 0 inter-cluster edges**: Normal case, proceed
- **All nodes in one cluster**: Valid case, creates single output file
- **Isolated cluster**: Cluster with no connections to other clusters (inter_cluster_edges=0)

## Test Coverage

Module has comprehensive test coverage in `/tests/viz/test_graph_split.py`:

### Unit Tests (8 tests)
- `test_identify_clusters` - Finding unique cluster IDs, sorted
- `test_identify_clusters_empty_graph` - Empty graph handling
- `test_sort_nodes` - Concepts first, others preserve order
- `test_sort_nodes_all_concepts` - Only Concept nodes
- `test_extract_cluster` - Node and edge filtering
- `test_extract_cluster_statistics` - Correct counts
- `test_inter_cluster_edges_calculation` - XOR logic for inter-cluster edges
- `test_create_cluster_metadata` - Metadata format and subtitle

### Integration Tests (3 tests)
- `test_full_split_flow` - Load → split → save → validate complete flow
- `test_full_split_validation` - Schema validation of output files
- `test_metadata_in_output` - Verify metadata format in saved files

### Boundary Cases (3 tests)
- `test_single_node_cluster_skipped` - Single node cluster skip with warning
- `test_isolated_cluster` - Cluster with no inter-cluster edges (inter_cluster_count=0)
- `test_all_nodes_one_cluster` - Edge case with single cluster containing all nodes

### Coverage
- Line coverage: >90% (all critical paths covered)
- All functions tested
- Edge cases and error handling covered

## Dependencies

### Standard Library
- json - JSON file handling
- logging - Logging configuration
- sys - Exit codes and path manipulation
- pathlib - Path operations
- typing - Type hints

### External Libraries
- colorama (optional) - Colored console output, falls back gracefully if not available

### Internal Modules
- `src.utils.exit_codes` - Standardized exit codes (EXIT_SUCCESS, EXIT_INPUT_ERROR, etc.)
- `src.utils.validation` - JSON schema validation (validate_json)
- `src.utils.console_encoding` - UTF-8 output support (setup_console_encoding)

## Usage Examples

### Basic Usage
```bash
# Activate virtual environment
source .venv/bin/activate

# Run split on production data
python -m viz.graph_split

# Check output files
ls -la viz/data/out/LearningChunkGraph_cluster_*.json
```

### Verify Output
```bash
# Count output files
ls viz/data/out/LearningChunkGraph_cluster_*.json | wc -l

# Validate one cluster file
python -c "
from src.utils.validation import validate_json
import json

g = json.load(open('viz/data/out/LearningChunkGraph_cluster_0.json'))
validate_json(g, 'LearningChunkGraph')
print(f\"Cluster 0: {len(g['nodes'])} nodes, {len(g['edges'])} edges\")
print(f\"Subtitle: {g['_meta']['subtitle']}\")
"
```

### Check Logs
```bash
# View recent log entries
tail -30 viz/logs/graph_split.log

# Check for warnings (skipped clusters)
grep WARNING viz/logs/graph_split.log
```

### Analyze Split Results
```bash
# Count nodes per cluster
for file in viz/data/out/LearningChunkGraph_cluster_*.json; do
    python -c "import json; g=json.load(open('$file')); \
    print(f'$(basename $file): {len(g[\"nodes\"])} nodes, {len(g[\"edges\"])} edges')"
done

# Verify total nodes (should match original)
python -c "
import json
from pathlib import Path

# Original graph
orig = json.load(open('viz/data/out/LearningChunkGraph_wow.json'))
orig_nodes = len(orig['nodes'])

# Sum cluster nodes
cluster_files = Path('viz/data/out').glob('LearningChunkGraph_cluster_*.json')
total_cluster_nodes = sum(
    len(json.load(open(f))['nodes'])
    for f in cluster_files
)

print(f'Original: {orig_nodes} nodes')
print(f'Clusters total: {total_cluster_nodes} nodes')
print(f'Match: {orig_nodes == total_cluster_nodes}')
"
```

## Notes

- Requires input graph to have `cluster_id` on all nodes (populated by graph2metrics)
- Single-node clusters are skipped with WARNING (not an error)
- Metadata is completely replaced, not merged with original
- All file I/O uses UTF-8 encoding
- Cluster IDs processed in ascending order (0, 1, 2, ...)
- Inter-cluster edges use XOR logic: exactly one endpoint in cluster
- Validation occurs both before processing (input) and after extraction (output)
- Colorama is optional - gracefully falls back to plain text if not available
- Each output file is a valid standalone LearningChunkGraph

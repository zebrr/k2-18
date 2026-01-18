# viz_graph_split.md

## Status: READY

Utility for splitting enriched knowledge graph into separate files by clusters. Extracts each cluster as independent subgraph with proper node/edge filtering, statistics, and metadata.

## CLI Interface

### Usage
```bash
python -m viz.graph_split
```

### Input Files
- **Graph**: `/viz/data/out/LearningChunkGraph_wow.json` - Enriched graph with cluster_id on all nodes (from graph2metrics)
- **Dictionary**: `/viz/data/out/ConceptDictionary_wow.json` - Concept dictionary for extracting cluster-specific concepts

### Output Files
- **Cluster graph**: `/viz/data/out/LearningChunkGraph_cluster_{ID}.json` (one per cluster)
- **Cluster dictionary**: `/viz/data/out/LearningChunkGraph_cluster_{ID}_dict.json` (one per cluster)
- **Naming**: Cluster ID with zero-padding based on max cluster count:
  - 16 clusters (0-15) → 2 digits: `LearningChunkGraph_cluster_00.json`, `LearningChunkGraph_cluster_00_dict.json`
  - 101 clusters (0-100) → 3 digits: `LearningChunkGraph_cluster_000.json`, `LearningChunkGraph_cluster_000_dict.json`

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
[12:34:56] INFO     | Loading: ConceptDictionary_wow.json
[12:34:56] INFO     | Graph: 255 nodes, 847 edges
[12:34:56] INFO     | Dictionary: 115 concepts
[12:34:56] INFO     | Found 16 clusters (zero-padding: 2 digits)
[12:34:56] INFO     | Finding inter-cluster links (PREREQUISITE, ELABORATES)...
[12:34:56] INFO     | Found inter-cluster links for 14 clusters

Processing clusters:
[12:34:56] INFO     | Cluster 00: 45 nodes, 123 edges, 23 concepts
[12:34:56] INFO     |   └─ Inter-cluster links: 0 incoming, 3 outgoing
[12:34:56] INFO     | Cluster 01: 32 nodes, 87 edges, 18 concepts
[12:34:56] INFO     |   └─ Inter-cluster links: 2 incoming, 1 outgoing
[12:34:56] WARNING  | Skipping cluster 03: only 1 node
[12:34:56] INFO     | Cluster 02: 28 nodes, 76 edges, 15 concepts
...

[12:34:57] SUCCESS  | ✅ Split completed successfully
[12:34:57] INFO     | Output files saved to: /viz/data/out/
[12:34:57] INFO     | Created 15 cluster graphs + 15 cluster dictionaries (1 skipped)
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

#### 8. Extract Cluster Dictionary
For each cluster, create a dictionary file with concepts used by nodes in that cluster:
1. Collect all unique concept IDs from `concepts: []` field of all nodes in cluster
2. Look up each concept ID in source `ConceptDictionary_wow.json`
3. Create cluster dictionary with format:
   ```json
   {
     "_meta": {
       "title": "<original title from graph>",
       "cluster_id": 12,
       "concepts_used": 23
     },
     "concepts": [
       { ... full concept object from dictionary ... },
       { ... }
     ]
   }
   ```
4. Save to `LearningChunkGraph_cluster_{ID}_dict.json`

**Edge cases:**
- Concept ID not found in dictionary → log WARNING, skip concept
- No concepts in cluster → create file with empty `concepts: []`

#### 9. Zero-Padding for Filenames
- Calculate padding width: `len(str(max_cluster_id))`
- Apply to all output filenames: `f"LearningChunkGraph_cluster_{cluster_id:0{width}d}.json"`

#### 10. Boundary Cases
- **Cluster with 1 node, 0 edges:**
  - Skip file creation (both graph and dictionary)
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
- `source_text` (str) - Source node text, truncated to 500 chars (for readability)
- `source_type` (str) - Source node type ("Concept", "Chunk", or "Assessment")
- `source_importance` (float) - Source educational_importance (used for sorting)
- `target` (str) - Target node ID
- `target_text` (str) - Target node text, truncated to 500 chars (for readability)
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

### load_dictionary(input_file: Path, logger: Logger) -> Dict
Load and validate concept dictionary against ConceptDictionary schema.
- **Input**:
  - input_file (Path) - Path to dictionary JSON file
  - logger (Logger) - Logger instance
- **Returns**: Dictionary data with concepts list
- **Raises**:
  - SystemExit(EXIT_INPUT_ERROR) - If file not found or validation fails
  - SystemExit(EXIT_IO_ERROR) - If file read fails
- **Validation**: Uses `validate_json(data, "ConceptDictionary")`

### get_filename_padding(cluster_ids: List[int]) -> int
Calculate zero-padding width for cluster filenames.
- **Input**:
  - cluster_ids (List[int]) - List of all cluster IDs
- **Returns**: Padding width (number of digits)
- **Algorithm**: `len(str(max(cluster_ids)))` or 1 if empty

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

### extract_cluster_concepts(cluster_nodes: List[Dict], concepts_data: Dict, logger: Logger) -> Tuple[List[Dict], int]
Extract concepts used by nodes in a cluster.
- **Input**:
  - cluster_nodes (List[Dict]) - Nodes belonging to cluster
  - concepts_data (Dict) - Full concept dictionary
  - logger (Logger) - Logger instance
- **Returns**: Tuple of (concepts_list, concepts_count)
  - concepts_list (List[Dict]) - Full concept objects from dictionary
  - concepts_count (int) - Number of unique concepts found
- **Algorithm**:
  1. Collect all unique concept IDs from `concepts: []` field of all nodes
  2. Build lookup map from concepts_data by concept_id
  3. For each ID, find concept in dictionary
  4. Log WARNING for missing concepts
  5. Return list of found concept objects

### create_cluster_dictionary(cluster_id: int, concepts_list: List[Dict], original_title: str) -> Dict
Create cluster dictionary structure with metadata.
- **Input**:
  - cluster_id (int) - Cluster ID
  - concepts_list (List[Dict]) - List of concept objects
  - original_title (str) - Title from source graph
- **Returns**: Cluster dictionary with _meta and concepts
- **Format**:
  ```json
  {
    "_meta": {
      "title": "<original_title>",
      "cluster_id": 12,
      "concepts_used": 23
    },
    "concepts": [...]
  }
  ```

### save_cluster_graph(cluster_graph: Dict, cluster_id: int, output_dir: Path, padding: int, logger: Logger) -> None
Validate and save cluster graph to file with zero-padded filename.
- **Input**:
  - cluster_graph (Dict) - Cluster subgraph to save
  - cluster_id (int) - Cluster ID for filename
  - output_dir (Path) - Output directory path
  - padding (int) - Zero-padding width for filename
  - logger (Logger) - Logger instance
- **Side effects**:
  - Validates graph against LearningChunkGraph schema
  - Writes file to `{output_dir}/LearningChunkGraph_cluster_{cluster_id:0{padding}d}.json`
- **Raises**:
  - SystemExit(EXIT_RUNTIME_ERROR) - If validation fails
  - SystemExit(EXIT_IO_ERROR) - If save fails

### save_cluster_dictionary(cluster_dict: Dict, cluster_id: int, output_dir: Path, padding: int, logger: Logger) -> None
Validate and save cluster dictionary to file with zero-padded filename.
- **Input**:
  - cluster_dict (Dict) - Cluster dictionary to save
  - cluster_id (int) - Cluster ID for filename
  - output_dir (Path) - Output directory path
  - padding (int) - Zero-padding width for filename
  - logger (Logger) - Logger instance
- **Side effects**:
  - Validates dictionary against ConceptDictionary schema
  - Writes file to `{output_dir}/LearningChunkGraph_cluster_{cluster_id:0{padding}d}_dict.json`
- **Raises**:
  - SystemExit(EXIT_INPUT_ERROR) - If validation fails
  - SystemExit(EXIT_IO_ERROR) - If save fails

### main() -> int
Main entry point for the utility.
- **Returns**: Exit code (0 for success, non-zero for errors)
- **Algorithm**:
  1. Setup console encoding
  2. Setup logging
  3. Load and validate input graph
  4. Load and validate concept dictionary
  5. Identify all clusters (sorted)
  6. Calculate filename padding width
  7. Find inter-cluster links for all clusters
  8. For each cluster:
     - Extract cluster subgraph
     - Check if single node → skip with WARNING
     - Sort nodes (Concepts first)
     - Create graph metadata
     - Extract cluster concepts from dictionary
     - Create cluster dictionary
     - Save cluster graph (with padding)
     - Save cluster dictionary (with padding)
     - Log statistics (nodes, edges, concepts)
  9. Print success summary
  10. Return EXIT_SUCCESS

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

Module has comprehensive test coverage in `/tests/viz/test_graph_split.py` (30 tests total):

### Unit Tests (15 tests)
- `test_identify_clusters` - Finding unique cluster IDs, sorted
- `test_identify_clusters_empty_graph` - Empty graph handling
- `test_identify_clusters_no_cluster_id` - Graph without cluster_id fields
- `test_sort_nodes` - Concepts first, others preserve order
- `test_sort_nodes_all_concepts` - Only Concept nodes
- `test_sort_nodes_no_concepts` - No Concept nodes
- `test_extract_cluster` - Node and edge filtering
- `test_extract_cluster_statistics` - Correct counts
- `test_inter_cluster_edges_calculation` - XOR logic for inter-cluster edges
- `test_create_cluster_metadata` - Metadata format and subtitle
- `test_get_filename_padding` - Zero-padding calculation (1, 2, 3 digits)
- `test_get_filename_padding_empty` - Empty list returns 1
- `test_extract_cluster_concepts` - Concept extraction from nodes
- `test_extract_cluster_concepts_missing` - Missing concept handling with warning
- `test_create_cluster_dictionary` - Dictionary format and metadata

### Integration Tests (7 tests)
- `test_full_split_flow` - Load → split → save → validate complete flow
- `test_full_split_validation` - Schema validation of output files
- `test_metadata_in_output` - Verify metadata format in saved files
- `test_inter_cluster_links_in_metadata` - Inter-cluster links in saved metadata
- `test_dictionary_files_created` - Verify dictionary files created alongside graphs
- `test_zero_padding_in_filenames` - Verify consistent padding in output filenames
- `test_cluster_with_no_concepts` - Empty concepts array handling

### Boundary Cases (4 tests)
- `test_single_node_cluster_skipped` - Single node cluster skip with warning (both files)
- `test_isolated_cluster` - Cluster with no inter-cluster edges (inter_cluster_count=0)
- `test_all_nodes_one_cluster` - Edge case with single cluster containing all nodes
- `test_inter_cluster_links_four_types` - All 4 edge types with proper node filtering

### Inter-Cluster Links Tests (4 tests)
- `test_inter_cluster_links_top3_selection` - Top-3 by source importance
- `test_inter_cluster_links_importance_fields` - Both importance fields included
- `test_inter_cluster_links_node_type_filtering` - Node type requirements enforced
- `test_inter_cluster_links_type_fields` - source_type and target_type populated

### Coverage
- Line coverage: >90% (all critical paths covered)
- All public functions tested
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

# Check output files (graphs and dictionaries)
ls -la viz/data/out/LearningChunkGraph_cluster_*.json
```

### Verify Output
```bash
# Count output files (should be 2x clusters: graph + dict)
ls viz/data/out/LearningChunkGraph_cluster_*.json | wc -l

# Validate one cluster graph
python -c "
from src.utils.validation import validate_json
import json

g = json.load(open('viz/data/out/LearningChunkGraph_cluster_00.json'))
validate_json(g, 'LearningChunkGraph')
print(f\"Cluster 00: {len(g['nodes'])} nodes, {len(g['edges'])} edges\")
print(f\"Subtitle: {g['_meta']['subtitle']}\")
"

# Check cluster dictionary
python -c "
import json
d = json.load(open('viz/data/out/LearningChunkGraph_cluster_00_dict.json'))
print(f\"Cluster {d['_meta']['cluster_id']}: {d['_meta']['concepts_used']} concepts\")
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
- Requires input graph to have `concepts: []` on nodes (populated by graph2metrics via link_nodes_to_concepts)
- Requires ConceptDictionary_wow.json in same directory as input graph
- Single-node clusters are skipped with WARNING (not an error), no files created
- Metadata is completely replaced, not merged with original
- All file I/O uses UTF-8 encoding
- Cluster IDs processed in ascending order (0, 1, 2, ...)
- Filenames use zero-padding based on max cluster ID (e.g., 00, 01, ... or 000, 001, ...)
- Inter-cluster edges use XOR logic: exactly one endpoint in cluster
- Validation occurs both before processing (input) and after extraction (output)
- Both cluster graphs and dictionaries are validated against their respective schemas
- Colorama is optional - gracefully falls back to plain text if not available
- Each cluster graph file is a valid standalone LearningChunkGraph
- Each cluster dictionary file contains subset of concepts from source dictionary

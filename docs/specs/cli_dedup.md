# cli_dedup.md

## Status: READY

Utility for removing duplicate nodes from the knowledge graph. Uses vector embeddings and FAISS to find similar nodes of type Chunk and Assessment. Adds metadata tracking for full traceability of deduplication process.

## CLI Interface

**Launch:**
```bash
python -m src.dedup
```

**Input directories:**
- `/data/out/LearningChunkGraph_raw.json` - graph after itext2kg

**Output directories:**
- `/data/out/LearningChunkGraph_dedup.json` - graph without duplicates
- `/logs/dedup_map.csv` - duplicate mapping

## Core Algorithm

1. **Load and validate** - check input graph against schema
2. **Filter nodes** - select Chunk/Assessment with non-empty text
3. **Get embeddings** - via OpenAI API (text-embedding-3-small)
4. **Build FAISS index** - HNSW for fast search
5. **Find duplicates** - by cosine similarity and length ratio
6. **Cluster duplicates** - Union-Find for transitive duplicates
7. **Rewrite graph** - remove duplicates and empty nodes

Note: The module now handles graphs with `_meta` section from itext2kg_graph v2, updates existing metadata and adds deduplication statistics

## Terminal Output

The utility uses standard logging with format:
```
[HH:MM:SS] LEVEL    | Message
```

Example output:
```
[10:30:00] INFO     | Loading knowledge graph...
[10:30:01] INFO     | Filtered 156 nodes out of 189 for deduplication
[10:30:02] INFO     | Getting embeddings for 156 nodes...
[10:30:15] INFO     | Building FAISS index...
[10:30:15] INFO     | Searching for duplicates...
[10:30:16] INFO     | Found 23 potential duplicates
[10:30:16] INFO     | Clustering duplicates...
[10:30:16] INFO     | Formed 8 clusters, 15 nodes marked as duplicates
[10:30:16] INFO     | Rewriting graph...
[10:30:16] INFO     | Removed 15 duplicate nodes, 3 empty nodes
[10:30:16] INFO     | Updated 42 edges, final count: 287
[10:30:17] INFO     | Saving results...
[10:30:17] INFO     | Saved duplicate mapping to logs/dedup_map.csv
[10:30:17] INFO     | Deduplication completed in 17.24 seconds
[10:30:17] INFO     | Nodes were: 189, became: 171
[10:30:17] INFO     | Edges were: 312, became: 287
```

Error messages:
```
[10:30:00] ERROR    | Input file not found: data/out/LearningChunkGraph_raw.json
[10:30:00] INFO     | Not enough nodes for deduplication, copying graph without changes
[10:30:00] ERROR    | API limit exceeded: Rate limit exceeded
```

## Public Functions

### extract_global_position(node_id: str) -> int
Extract global token position from Chunk or Assessment node ID.
- **Input**: node_id - Node identifier in format {slug}:c:{position} or {slug}:q:{position}:{index}
- **Returns**: Global position in tokens
- **Raises**: ValueError if ID format is unexpected (indicates a bug)
- **Purpose**: Used to determine master node when deduplicating (earlier position = master)
- **Note**: Only called for nodes that passed filter_nodes_for_dedup (Chunk/Assessment types)

### filter_nodes_for_dedup(nodes: List[Dict]) -> List[Dict]
Filter nodes for deduplication.
- **Input**: nodes - list of all graph nodes
- **Returns**: filtered nodes (Chunk/Assessment with text)

### build_faiss_index(embeddings: np.ndarray, config: Dict) -> faiss.IndexHNSWFlat
Create FAISS index.
- **Input**: embeddings - embedding matrix, config - FAISS parameters
- **Returns**: built index
- **Note**: Always uses METRIC_INNER_PRODUCT for normalized vectors

### find_duplicates(nodes, embeddings, index, config) -> List[Tuple[str, str, float]]
Find duplicate candidates.
- **Input**: nodes - nodes, embeddings - vectors, index - FAISS, config - parameters
- **Returns**: list of (master_id, duplicate_id, similarity)

### cluster_duplicates(duplicates) -> Tuple[Dict[str, str], int]
Cluster through Union-Find.
- **Input**: duplicates - duplicate pairs
- **Returns**: tuple of (dedup_map, num_clusters) where:
  - dedup_map: Dictionary {duplicate_id: master_id}
  - num_clusters: Number of clusters formed

### rewrite_graph(graph, dedup_map) -> Tuple[Dict, Dict]
Rewrite graph removing duplicates and empty nodes.
- **Input**: graph - original graph, dedup_map - duplicate mapping
- **Returns**: tuple of (new_graph, statistics) where:
  - new_graph: Graph without duplicates
  - statistics: Dict with removal and update counts

### save_dedup_map(dedup_map, duplicates)
Save duplicate mapping to CSV file.
- **Input**: dedup_map - {duplicate: master}, duplicates - original pairs with similarities
- **Creates**: /logs/dedup_map.csv

### update_metadata(existing_meta, config, statistics, processing_time) -> Dict
Update or create metadata with deduplication information.
- **Input**: 
  - existing_meta - Existing metadata from input graph (if any)
  - config - Deduplication configuration
  - statistics - Deduplication statistics collected during processing
  - processing_time - Time spent on deduplication in seconds
- **Returns**: Updated metadata dictionary with quality_issues and deduplication sections
- **Purpose**: Maintain full traceability of graph transformations

## Internal Classes

### UnionFind
Union-Find data structure for clustering duplicates.
- **find(x)**: Find root with path compression
- **union(x, y)**: Union two elements by rank
- **get_clusters()**: Return all clusters as Dict[root, List[elements]]

## Output Format

### LearningChunkGraph_dedup.json
```json
{
  "_meta": {
    // Original metadata from itext2kg_graph (if present)
    "quality_issues": {
      "duplicate_nodes_removed": 15
      // other quality issues
    },
    "deduplication": {
      "performed_at": "2024-01-15T10:45:00",
      "config": {
        "similarity_threshold": 0.97,
        "length_ratio_threshold": 0.8,
        "top_k": 5,
        "min_similarity": 0.97,
        "model": "text-embedding-3-small"
      },
      "statistics": {
        "nodes_analyzed": 156,
        "embeddings_created": 156,
        "potential_duplicates": 23,
        "clusters_formed": 8,
        "nodes_removed": {
          "duplicates": 15,
          "empty": 3,
          "total": 18
        },
        "edges_updated": 42,
        "processing_time_seconds": 17.24
      },
      "before_after": {
        "nodes_before": 189,
        "nodes_after": 171,
        "edges_before": 312,
        "edges_after": 287
      }
    }
  },
  "nodes": [
    {
      "id": "string",
      "type": "Chunk|Concept|Assessment",
      "text": "string",
      "node_offset": 0,
      // other fields according to schema
    }
  ],
  "edges": [
    {
      "source": "string",
      "target": "string", 
      "type": "PREREQUISITE|ELABORATES|...",
      "weight": 0.5
    }
  ]
}
```

### dedup_map.csv
```csv
duplicate_id,master_id,similarity
handbook:c:1000,handbook:c:500,0.9823
handbook:c:1500,handbook:c:500,0.9756
```

## Configuration

Section `[dedup]` in config.toml:

- **embedding_model**: "text-embedding-3-small" - model for embeddings
- **embedding_api_key**: "sk-..." - API key (optional, falls back to api_key)
- **embedding_tpm_limit**: 350000 - tokens per minute limit
- **sim_threshold**: 0.97 - cosine similarity threshold
- **len_ratio_min**: 0.8 - minimum text length ratio
- **faiss_M**: 32 - HNSW parameter (graph connectivity)
- **faiss_efC**: 200 - HNSW parameter (construction quality)
- **faiss_metric**: "INNER_PRODUCT" - not used, hardcoded in code
- **k_neighbors**: 5 - number of nearest neighbors

## Error Handling & Exit Codes

- **0 (SUCCESS)**: Successful execution
- **1 (CONFIG_ERROR)**: Configuration errors (invalid parameters)
- **2 (INPUT_ERROR)**: Missing input file or validation failure
- **3 (RUNTIME_ERROR)**: Runtime errors (FAISS, data processing)
- **4 (API_LIMIT_ERROR)**: OpenAI API limits exceeded
- **5 (IO_ERROR)**: File write errors

## Boundary Cases

**Not enough nodes for deduplication (<2):**
- Graph is copied without changes
- Empty dedup_map.csv is created
- Exit code: 0 (SUCCESS)

**Empty nodes (Chunk/Assessment):**
- Removed from final graph
- Edges referencing them are also removed

**Texts too long (>8192 tokens):**
- Handled by llm_embeddings module
- Processing continues

**No duplicates found:**
- Graph is copied with empty nodes removed
- Empty dedup_map.csv is created

**Master selection with equal global position:**
- Node with lexicographically smaller ID is chosen
- Global position extracted from node ID format {slug}:c:{position}

**Dangling edges:**
- Edges referencing non-existent nodes (including removed empty ones) are dropped
- Logged at DEBUG level

## Test Coverage

- **test_dedup**: 30 tests
  - test_union_find_basic
  - test_filter_nodes_for_dedup
  - test_find_duplicates_with_mock
  - test_cluster_duplicates (updated for tuple return)
  - test_rewrite_graph (updated for tuple return and statistics)
  - test_update_metadata (new tests for metadata generation)
  - test_main_success
  - test_edge_cases

- **test_dedup_integration**: 5 tests
  - test_full_dedup_process
  - test_no_duplicates_case
  - test_transitive_duplicates
  - test_edge_cases_with_real_api
  - test_performance_large_graph

## Changes History

### v2.1 - Metadata tracking update (DEDUP-META-ADD)
- Added comprehensive metadata tracking for deduplication process
- Updated `cluster_duplicates` to return tuple with cluster count
- Updated `rewrite_graph` to return tuple with statistics
- Added `update_metadata` function for metadata generation
- Updates `quality_issues` section with duplicate removal count
- Creates new `deduplication` section with full statistics and configuration
- Tracks nodes analyzed, embeddings created, duplicates found, clusters formed
- Records before/after counts for nodes and edges
- Added processing time tracking

### v2.0 - Post-refactoring update (DEDUP-REFACTOR-01)
- Changed from `local_start` field to extracting position from node IDs
- Added `extract_global_position` function to parse {slug}:c:{position} format
- Added support for graphs with `_meta` section from itext2kg_graph v2
- Updated master selection logic to work with new ID format
- Preserved metadata section in output graph

## Dependencies

- **Standard Library**: sys, json, logging, time, pathlib, csv
- **External**: numpy, faiss-cpu, python-dotenv
- **Internal**: utils.config, utils.validation, utils.llm_embeddings, utils.exit_codes, utils.console_encoding

## Performance Notes

- **Embeddings**: Batch processing up to 2048 texts per request
- **FAISS index**: HNSW provides O(log N) neighbor search
- **Memory**: ~2GB for 10K nodes (embeddings + index)
- **Speed**: ~1000 nodes/minute (including API requests)
- **TPM control**: Automatic via EmbeddingsClient

## Usage Examples

### Running deduplication
```bash
# Ensure input file exists
dir data/out/LearningChunkGraph_raw.json

# Run deduplication
python -m src.dedup

# Check results
dir data/out/LearningChunkGraph_dedup.json
type logs/dedup_map.csv
```

### Checking results
```python
import json

# Load graphs
with open('data/out/LearningChunkGraph_raw.json') as f:
    raw_graph = json.load(f)
    
with open('data/out/LearningChunkGraph_dedup.json') as f:
    dedup_graph = json.load(f)

# Statistics
print(f"Nodes were: {len(raw_graph['nodes'])}")
print(f"Nodes became: {len(dedup_graph['nodes'])}")
print(f"Removed: {len(raw_graph['nodes']) - len(dedup_graph['nodes'])}")
```
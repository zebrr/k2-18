# cli_dedup.md

## Status: READY

Utility for removing duplicate nodes from the knowledge graph. Uses vector embeddings and FAISS to find similar nodes of type Chunk and Assessment.

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

### cluster_duplicates(duplicates) -> Dict[str, str]
Cluster through Union-Find.
- **Input**: duplicates - duplicate pairs
- **Returns**: dictionary {duplicate_id: master_id}

### rewrite_graph(graph, dedup_map) -> Dict
Rewrite graph removing duplicates and empty nodes.
- **Input**: graph - original graph, dedup_map - duplicate mapping
- **Returns**: new graph without duplicates

### save_dedup_map(dedup_map, duplicates)
Save duplicate mapping to CSV file.
- **Input**: dedup_map - {duplicate: master}, duplicates - original pairs with similarities
- **Creates**: /logs/dedup_map.csv

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
  "nodes": [
    {
      "id": "string",
      "type": "Chunk|Concept|Assessment",
      "text": "string",
      "local_start": 0,
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
chunk_123,chunk_042,0.9823
chunk_789,chunk_042,0.9756
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

**Master selection with equal local_start:**
- Node with lexicographically smaller ID is chosen

**Dangling edges:**
- Edges referencing non-existent nodes (including removed empty ones) are dropped
- Logged at DEBUG level

## Test Coverage

- **test_dedup**: 24 tests
  - test_union_find_basic
  - test_filter_nodes_for_dedup
  - test_find_duplicates_with_mock
  - test_cluster_duplicates
  - test_rewrite_graph
  - test_main_success
  - test_edge_cases

- **test_dedup_integration**: 5 tests
  - test_full_dedup_process
  - test_no_duplicates_case
  - test_transitive_duplicates
  - test_edge_cases_with_real_api
  - test_performance_large_graph

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
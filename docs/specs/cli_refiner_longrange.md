# cli_refiner_longrange.md

## Status: READY

Rename from refiner to refiner_longrange completed. Module now:
- Uses extract_global_position() instead of local_start field
- Adds _meta section with refiner_longrange metadata
- Outputs to LearningChunkGraph_longrange.json
- Supports context management parameters for OpenAIClient

CLI utility for adding long-range connections to knowledge graphs. Searches for missed connections between nodes that didn't appear in the same context during initial processing. Uses semantic similarity through embeddings to find candidates and LLM for analyzing connection types.

## CLI Interface

### Usage
```bash
python -m src.refiner_longrange
```

### Input Files
- **Primary**: `/data/out/LearningChunkGraph_dedup.json` - graph after deduplication
- **Prompt**: `/prompts/refiner_longrange.md` - prompt for LLM with weight substitutions

### Output Files
- **Result**: `/data/out/LearningChunkGraph_longrange.json` - graph with long-range connections
- **Logs**: `/logs/refiner_longrange_YYYY-MM-DD_HH-MM-SS.log` - JSON Lines logs
- **Errors**: `/logs/{node_id}_bad.json` - problematic LLM responses (on errors)

### Exit Codes
- 0 (SUCCESS) - successful execution
- 1 (CONFIG_ERROR) - configuration errors
- 2 (INPUT_ERROR) - missing input file
- 3 (RUNTIME_ERROR) - critical execution errors
- 4 (API_LIMIT_ERROR) - API limits exceeded
- 5 (IO_ERROR) - file write errors

## Core Algorithm

### Pipeline Overview
1. **Configuration check**: if `run = false` - just copy file (without JSON logging initialization, uses simple output)
2. **Graph loading**: validation by schema and target nodes extraction
3. **Candidate generation**:
   - Get embeddings for Chunk/Assessment nodes
   - Build FAISS index for fast search
   - Find top-K similar pairs with similarity threshold
4. **LLM connection analysis**:
   - Sequential processing with context preservation
   - Form requests with texts and existing edges
   - Response validation with repair-retry on errors
5. **Graph update**: add/update edges with marking
6. **Final validation**: check invariants and save

### Input Format for LLM
Each node is analyzed with its candidate pairs in the following format:
```json
{
  "source_node": {
    "id": "handbook:c:220",
    "text": "Text of source node (Chunk or Assessment)"
  },
  "candidates": [
    {
      "node_id": "handbook:c:480",
      "text": "Text of candidate node", 
      "similarity": 0.87,
      "existing_edges": [
        {"source": "handbook:c:220", "target": "handbook:c:480", "type": "PREREQUISITE", "weight": 0.8},
        {"source": "handbook:c:480", "target": "handbook:c:220", "type": "REFER_BACK", "weight": 0.6}
      ]
    },
    {
      "node_id": "handbook:q:650:2",
      "text": "Another candidate (Assessment)",
      "similarity": 0.82,
      "existing_edges": []  // No existing connections
    },
    {
      "node_id": "handbook:c:890",
      "text": "Third candidate",
      "similarity": 0.81,
      "existing_edges": [
        {"source": "handbook:c:220", "target": "handbook:c:890", "type": "HINT_FORWARD", "weight": 0.4}
      ]
    }
    // ... up to max_pairs_per_node candidates (default 20, configurable)
  ]
}
```
- **candidates**: 1 to `max_pairs_per_node` elements, sorted by descending similarity
- **existing_edges**: Shows current connections between the pair (can be empty, one-way, or bidirectional)
- **similarity**: Cosine similarity from `sim_threshold` to 1.0

### Embeddings Processing
- **API**: OpenAI Embeddings API (/v1/embeddings)
- **Model**: configured via `embedding_model` (usually text-embedding-3-small)
- **TPM Limit**: controlled via `embedding_tpm_limit` (1,000,000 default)
- **Batching**: up to 2048 texts per request, up to 8192 tokens per text
- **Token counting**: via cl100k_base for embeddings API
- **Normalization**: vectors automatically normalized (L2 norm = 1)

### FAISS Index Configuration
- **Index type**: HNSW (Hierarchical Navigable Small World)
- **Parameters**:
  - M: configured via `faiss_M` (graph connectivity)
  - efConstruction: configured via `faiss_efC` (construction precision) - not in config, defaults to 200
  - Metric: INNER_PRODUCT or L2 via `faiss_metric`
- **Ordering**: deterministic order via sorting by global position extracted from node IDs using extract_global_position()

### LLM Analysis
- **Sequential processing**: with previous_response_id preservation between nodes
- **Context preservation**: LLM sees analysis history of previous nodes
- **Auto-detection**: reasoning models detected by "o*" prefix
- **Retry logic**: one repair-retry on invalid JSON
- **Rate limiting**: TPM control via response headers
- **Prompt template**: loaded from file with weight substitutions

### Context Management
- **Response chain**: Controlled via `response_chain_depth` parameter
  - `0` (recommended for refiner) - independent requests for each node
  - `1-N` - sliding window with automatic deletion of old responses
  - Undefined/commented - unlimited chain (not recommended due to context accumulation)
- **Truncation**: Optional `truncation` parameter for API-level context management
  - `"auto"` - automatic truncation by API when context exceeds limit
  - `"disabled"` - no truncation
  - Commented out - parameter not sent to API
- **Model selection**: Based on context size via `max_context_tokens` and `max_context_tokens_test`
  - If context < max_context_tokens → use main model
  - Otherwise → use test model (if configured)
- **Parameters passed to OpenAIClient**:
  - `is_reasoning` (required) - determines if model is reasoning type
  - `response_chain_depth` (optional) - context window management
  - `truncation` (optional) - API truncation strategy
  - `max_context_tokens` - threshold for main model
  - `max_context_tokens_test` - threshold for test model

### Graph Update Logic
- **New edges**: added with `conditions: "added_by=refiner_longrange_v1"`
- **Duplicate edges**: weight updated to maximum of old and new
- **Type replacement**: only if new weight ≥ old, with `conditions: "fixed_by=refiner_longrange_v1"`
- **Direction conflicts**: A→B and B→A can coexist
- **Self-loop cleanup**: remove PREREQUISITE self-loops after update

## Terminal Output

The utility uses structured progress output with unified format:
```
[HH:MM:SS] TAG      | Data
```

### Progress Output Format

**START - processing start:**
```
[10:30:00] START    | 89 nodes | model=gpt-5-2025-08-07 | tpm=450k
```

**NODE - node processing:**
```
[10:30:05] NODE     | ✅ 001/089 | pairs=15 | tokens=1240 | 320ms | edges_added=3
[10:30:12] NODE     | ✅ 002/089 | pairs=8 | tokens=890 | 285ms | edges_added=1
```

**ERROR - processing errors:**
```
[10:30:45] ERROR    | ⚠️ RateLimitError | will retry...
[10:31:02] ERROR    | ⚠️ JSONDecodeError: Expecting value: line 1 column 1...
```

**END - completion:**
```
[10:45:30] END      | Done | nodes=89 | edges_added=47 | time=8m 12s
```

### Simple Output when run=false
When `run = false` in config, uses simple print without logging:
```
Refiner longrange is disabled (run=false), copying file without changes
Copied data/out/LearningChunkGraph_dedup.json to data/out/LearningChunkGraph_longrange.json
```

### JSON Lines Logging
Log files use structured format for detailed analysis:
```json
{"timestamp": "2024-01-15T10:30:00Z", "level": "INFO", "event": "node_processed", "node_id": "n1", "pairs_count": 15, "tokens_used": 1240, "duration_ms": 320, "edges_added": 3}
{"timestamp": "2024-01-15T10:30:01Z", "level": "DEBUG", "event": "edge_added", "source": "n1", "target": "n2", "type": "PREREQUISITE", "weight": 0.8, "conditions": "added_by=refiner_longrange_v1"}
```

**Note**: Terminal timestamps use UTC+3 timezone for display.

## Public Functions

### validate_refiner_longrange_config(config: Dict) -> None
Validate configuration parameters.
- **Checks**: required parameters, value ranges, weight consistency
- **Raises**: ValueError with problem description

### load_and_validate_graph(graph_path: Path) -> Dict
Load and validate graph from file.
- **Input**: path to JSON graph file
- **Returns**: valid graph with preserved _meta section
- **Validates**: JSON schema compliance

### extract_target_nodes(graph: Dict) -> List[Dict]
Extract nodes for analysis.
- **Filters**: only Chunk and Assessment types
- **Returns**: list of nodes with non-empty texts

### get_node_embeddings(nodes: List[Dict], config: Dict, logger) -> Dict[str, np.ndarray]
Get embeddings for nodes via OpenAI API.
- **Batching**: automatic request grouping
- **Filtering**: skip nodes with empty texts
- **Returns**: dictionary {node_id: embedding_vector}

### build_similarity_index(embeddings_dict, nodes, config, logger) -> Tuple[faiss.Index, List[str]]
Build FAISS index for search.
- **Ordering**: sort by global position extracted from node IDs using extract_global_position()
- **Configuration**: uses faiss_M, faiss_metric from config
- **Returns**: (index, ordered_node_ids)

### generate_candidate_pairs(nodes, embeddings_dict, index, node_ids, edges_index, config, logger) -> List[Dict]
Generate candidate pairs for analysis.
- **Filtering**: similarity ≥ sim_threshold, source.position < target.position (positions from IDs)
- **Limit**: max_pairs_per_node nearest neighbors
- **Includes**: existing edges between each pair for context

### analyze_candidate_pairs(candidate_pairs, graph, config, logger) -> List[Dict]
Analyze pairs via LLM to determine connection types.
- **Sequential**: processing with previous_response_id
- **Retry**: repair-reprompt on JSON errors
- **Validation**: check edge types and weights
- **Returns**: list of new edges

### update_graph_with_new_edges(graph, new_edges, logger) -> Dict
Update graph with new edges.
- **Logic**: add/update/replace according to TZ rules
- **Cleanup**: remove PREREQUISITE self-loops
- **Statistics**: count changes for logging

### extract_global_position(node_id: str) -> int
Extract global position from Chunk or Assessment node ID.
- **Input**: node_id in format {slug}:c:{position} or {slug}:q:{position}:{index}
- **Returns**: Global position in tokens from the beginning of corpus
- **Raises**: ValueError if ID format is unexpected
- **Purpose**: Replaces the deprecated local_start field with position extracted from ID
- **Note**: Only works with Chunk and Assessment nodes, never called for Concept nodes

### add_refiner_meta(graph: Dict, config: Dict, stats: Dict) -> None
Add refiner_longrange metadata to graph.
- **Updates**: graph["_meta"]["refiner_longrange"] with processing info
- **Preserves**: Existing _meta sections from other pipeline stages (slicer, itext2kg, dedup)
- **Creates**: _meta if not present in input graph
- **Includes**: timestamp, config parameters, and update statistics

## Internal Functions

### setup_json_logging(config: Dict) -> logging.Logger
Set up JSON Lines logging.
- **Creates**: timestamped log file in /logs
- **Format**: JSON Lines with structured data
- **Levels**: INFO or DEBUG based on config

### build_edges_index(graph: Dict) -> Dict[str, Dict[str, List[Dict]]]
Build index of existing edges for fast lookup.
- **Structure**: {source_id: {target_id: [edges]}}
- **Purpose**: quick edge existence checks

### load_refiner_longrange_prompt(config: Dict) -> str
Load and prepare prompt for connection analysis.
- **Substitutions**: {weight_low}, {weight_mid}, {weight_high} from config
- **Returns**: prepared prompt text
- **Raises**: FileNotFoundError if prompt file missing

### log_edge_operation(logger: Logger, operation: str, edge: Dict, **kwargs)
Log edge operations in structured format.
- **Operations**: added, updated, replaced, removed
- **Format**: structured logging with edge details
- **Levels**: INFO for updates/replacements, DEBUG for others

### validate_llm_edges(edges_response, source_id, candidates, graph, logger) -> List[Dict]
Validate edges received from LLM.
- **Checks**: completeness, valid types, weight ranges
- **Filters**: null types, self-loops, invalid targets
- **Returns**: list of valid edges only

## Configuration

### Required Parameters
```toml
[refiner]
# Core functionality control
run = true                           # false = just copy file without processing
is_reasoning = true                  # REQUIRED! Must match model type (o*/gpt-5* models = true)

# Model configuration
model = "gpt-5-2025-08-07"          # Main model for analysis
api_key = "sk-..."                   # OpenAI API key
tpm_limit = 450000                   # Tokens per minute limit
max_completion = 100000              # Max response tokens

# Context management thresholds
max_context_tokens = 400000         # Context limit for main model
max_context_tokens_test = 400000    # Context limit for test model

# Test model (optional, for large contexts)
model_test = "gpt-5-mini-2025-08-07"  # Fallback model for large contexts
tpm_limit_test = 450000              # TPM limit for test model
max_completion_test = 100000        # Max completion for test model

# Embeddings configuration
embedding_model = "text-embedding-3-small"
embedding_api_key = ""               # Optional, uses api_key if empty
embedding_tpm_limit = 1000000       # Embeddings API token limit
max_batch_tokens = 100000           # Soft limit per batch
max_texts_per_batch = 2048          # Max texts per request
truncate_tokens = 8000              # Truncate texts > 8192 tokens

# Similarity search parameters
sim_threshold = 0.80                # Cosine similarity threshold [0,1]
max_pairs_per_node = 20             # Max candidates per node (1-100)

# FAISS index parameters
faiss_M = 32                        # HNSW connectivity parameter
faiss_metric = "INNER_PRODUCT"      # Similarity metric (or "L2")

# Edge weight configuration
weight_low = 0.3                    # Weight for weak connections
weight_mid = 0.6                    # Weight for medium connections
weight_high = 0.9                   # Weight for strong connections

# Processing parameters
tpm_safety_margin = 0.15            # Safety margin for TPM control
timeout = 360                       # Request timeout in seconds
max_retries = 3                     # Number of retry attempts
poll_interval = 7                   # Seconds between retries
log_level = "info"                  # Logging level (info/debug)

# Model-specific parameters (for regular models)
# temperature = 0.6                  # Generation temperature

# Model-specific parameters (for reasoning models)
reasoning_effort = "high"           # low/medium/high for o*/gpt-5* models
reasoning_summary = "auto"          # auto/concise/detailed for o*/gpt-5* models
```

### Optional Parameters (Context Management)
```toml
# response_chain_depth = 0           # 0 = independent (recommended for refiner)
                                     # 1-N = sliding window with auto-deletion
                                     # commented = unlimited chain
# truncation = "auto"                # auto/disabled, or comment out
```

### Validation Rules
- api_key not empty
- sim_threshold ∈ [0,1]
- max_pairs_per_node > 0
- 0 ≤ weight_low < weight_mid < weight_high ≤ 1
- faiss_metric ∈ ["INNER_PRODUCT", "L2"]
- is_reasoning must match model type (true for o*/gpt-5* models)

## Error Handling

### API Errors
- **RateLimitError**: exponential backoff with retry
- **Embeddings limit**: truncate texts > 8192 tokens with logging
- **Network errors**: retry with same context

### LLM Response Errors
- **Invalid JSON**: one repair-retry with error prompt
  - Uses two-phase confirmation: response confirmed only after successful JSON parsing
  - On parsing failure, repair uses last confirmed response_id (prevents "Previous response not found" error)
- **Invalid edges**: filter with logging
- **Failed nodes**: save to `/logs/{node_id}_bad.json`

### Critical Errors
- **Missing config params**: EXIT_CONFIG_ERROR
- **Input file not found**: EXIT_INPUT_ERROR
- **All retries exhausted**: EXIT_API_LIMIT_ERROR
- **File write errors**: EXIT_IO_ERROR

## Performance Notes

### Embeddings Optimization
- Batch processing up to 2048 texts to minimize API calls
- In-memory caching during utility execution
- No parallel processing (sequential for context)

### FAISS Performance
- HNSW index provides O(log N) search
- Memory: ~4KB per vector (1536 dims × 4 bytes + overhead)
- Index construction: O(N log N) for N nodes

### LLM Optimization
- Prompt minimization through compact format
- Context preservation reduces repetition
- Early termination when no candidates

## Test Coverage

- **test_refiner_longrange_config**: 8 tests
  - Configuration parameter validation
  - Weight and range checks
  - run=false handling

- **test_embeddings_processing**: 10 tests  
  - Getting embeddings
  - Building FAISS index
  - Finding candidates

- **test_llm_analysis**: 12 tests
  - Request formation
  - Response validation
  - Repair mechanism
  - Invalid JSON handling

- **test_graph_update**: 15 tests
  - Adding new edges
  - Weight updates
  - Type replacement
  - Self-loop cleanup

- **test_extract_global_position**: 3 tests
  - Chunk ID parsing
  - Assessment ID parsing
  - Invalid ID handling

- **test_meta_handling**: 2 tests
  - Meta preservation
  - Meta creation and update

## Dependencies

- **Standard Library**: json, logging, shutil, sys, pathlib, typing, datetime, time
- **External**: numpy, faiss-cpu, python-dotenv
- **Internal**: utils.config, utils.validation, utils.exit_codes, utils.llm_embeddings, utils.llm_client, utils.console_encoding

## Usage Examples

### Basic Run
```bash
# Normal run with default config
python -m src.refiner_longrange
```

### Skip Processing
```bash
# Set run = false in config.toml
# Just copies dedup → longrange without processing
python -m src.refiner_longrange
```

### Debug Mode
```bash
# Set log_level = "debug" in config.toml
# Detailed logs with prompts and responses
python -m src.refiner_longrange
```

### Checking Results
```python
import json

# Compare graphs before and after
with open('data/out/LearningChunkGraph_dedup.json') as f:
    before = json.load(f)
    
with open('data/out/LearningChunkGraph_longrange.json') as f:
    after = json.load(f)

# Statistics
edges_before = len(before['edges'])
edges_after = len(after['edges'])
edges_added = edges_after - edges_before

print(f"Edges before: {edges_before}")
print(f"Edges after: {edges_after}")
print(f"Added: {edges_added}")

# Check new edges
new_edges = [e for e in after['edges'] if e.get('conditions') == 'added_by=refiner_longrange_v1']
print(f"New edges with marking: {len(new_edges)}")

# Check _meta section
if '_meta' in after and 'refiner_longrange' in after['_meta']:
    meta = after['_meta']['refiner_longrange']
    print(f"Processed at: {meta['processed_at']}")
    print(f"Stats: {meta['stats']}")
```

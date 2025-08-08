# cli_itext2kg_graph.md

## Status: READY

CLI utility for incremental knowledge graph construction from educational texts. Processes slices sequentially, sends them to LLM while preserving context through previous_response_id. Builds LearningChunkGraph using the pre-extracted ConceptDictionary.

## CLI Interface

### Usage
```bash
python -m src.itext2kg_graph
```

### Input Directory/Files
- **Source**: `/data/staging/*.slice.json` - slices from slicer.py
- **Required**: `/data/out/ConceptDictionary.json` - pre-extracted concepts (MUST EXIST)
- **Formats**: JSON files with slice and concept dictionary structures
- **Note**: Only `text` field from slice is used for processing

### Output Directory/Files
- **Target**: `/data/out/LearningChunkGraph_raw.json` - knowledge graph with nodes and edges
- **Logs**: `/logs/itext2kg_graph_YYYY-MM-DD_HH-MM-SS.log` - detailed logs
- **Debug**: `/logs/{slice_id}_bad.json` - problematic LLM responses (on errors)
- **Recovery**: `/logs/LearningChunkGraph_temp_*.json` - temporary graph dumps (on critical errors)
- **Stats**: `/logs/processing_stats_*.json` - processing statistics (on critical errors)

## Key Features

### ID Post-processing
- **Automatic ID correction**: LLM generates temporary IDs (chunk_1, assessment_1), which are automatically replaced with correct position-based IDs
- **Position calculation**: `final_id = {slug}:c:{slice_token_start + node_offset}` for Chunks
- **Assessment IDs**: `final_id = {slug}:q:{slice_token_start + node_offset}:{index}` for Assessments
- **Post-processing eliminates repair needs**: No more ID calculation errors from LLM
- **Critical for data integrity**: Prevents ~75% content loss from ID collisions
- **Implementation**: See `_assign_final_ids()` method

### JSON Repair Mechanism
Single repair attempt for JSON parsing errors:
- **JSON parsing errors**:
  - Add emphasis on valid JSON format
  - Remove markdown formatting instructions
  - Rollback to previous_response_id

### Node Processing
- **Chunk nodes**: Keep longer text version on duplicates
- **Assessment nodes**: Ignore duplicates with WARNING
- **Concept nodes**: Add as-is, always use definition from ConceptDictionary
- **Missing difficulty**: Set to default value (3) with WARNING

### Edge Validation
- Drop PREREQUISITE self-loops
- Fix invalid weights (clamp to [0,1])
- Filter duplicate edges by (source, target, type) key
- Check node existence in graph or ConceptDictionary

### Automatic MENTIONS Edges
Safety net for MENTIONS edges the LLM might miss:
- Case-insensitive whole word matching
- Search primary term and all aliases
- Skip existing MENTIONS edges
- Weight always 1.0

### Context Management
- Full ConceptDictionary passed with each slice
- Previous_response_id preservation for incremental processing
- Rollback mechanism for repair attempts
- Context window up to 128K tokens

## Core Algorithm

1. **Load ConceptDictionary** - must exist from itext2kg_concepts.py
2. **Load slices** from staging in lexicographic order
3. **Sequential processing** with previous_response_id preservation:
   - Format input data (FULL ConceptDictionary + Slice)
   - Call LLM via Responses API
   - Parse and validate response
   - **Apply ID post-processing**: Replace temporary IDs with final position-based IDs using `_assign_final_ids()`
   - **JSON repair mechanism** (if needed):
     - JSON errors → repair with format emphasis
   - Process nodes (handle duplicates)
   - Validate edges
   - Add automatic MENTIONS edges
   - Intermediate graph validation
4. **Critical error handling**:
   - If any slice fails → exit with RUNTIME_ERROR
   - Save temporary dumps for recovery
5. **Save results** to LearningChunkGraph_raw.json

## Terminal Output

The utility uses structured progress output with unified format:
```
[HH:MM:SS] TAG      | Data
```

### Output Format

**START - processing start:**
```
[10:30:00] START    | 157 slices | model=o4-mini-2025-04-16 | tpm=200k
```

**SLICE - successful slice processing:**
```
[10:30:05] SLICE    | ✅ 001/157 | tokens_used=12.35k | 5s | nodes=23 | edges=45
[10:30:12] SLICE    | ✅ 002/157 | tokens_used=112.34k | 8s | nodes=48 | edges=92
```

**REPAIR - error fix attempts:**
```
[10:30:45] REPAIR   | 🔧 Attempting to fix JSON validation error...
[10:30:50] REPAIR   | ✅ JSON validation fixed successfully!
```

**INFO - informational messages:**
```
[10:30:06] INFO     | Added 12 automatic MENTIONS edges
```

**ERROR - processing errors:**
```
[10:30:45] ERROR    | ❌ 042/157 | slice_042 | Processing failed
```

**FAILED - critical errors (stops processing):**
```
[10:45:30] FAILED   | ❌ Cannot continue without slice slice_042
[10:45:30] FAILED   | ❌ All slices failed processing
```

**SUCCESS - successful completion:**
```
[10:45:00] SUCCESS  | ✅ Results saved to /data/out/
                    | - LearningChunkGraph_raw.json
                    | - Nodes: 523
                    | - Edges: 1247
```

**END - processing complete:**
```
[10:45:00] END      | Done | slices=157 | time=15m 0s
```

## Configuration

Section `[itext2kg]` in config.toml:

### Required Parameters
- **model** (str) - LLM model (o4-mini-2025-04-16)
- **tpm_limit** (int, >0) - tokens per minute limit
- **log_level** (str) - logging level (debug/info/warning/error)

### Optional Parameters
- **tpm_safety_margin** (float, 0-1, default=0.15) - TPM safety margin
- **max_completion** (int, >0) - maximum tokens per generation
- **temperature** (float, 0-2) - for regular models
- **reasoning_effort** (str) - for reasoning models (low/medium/high)
- **reasoning_summary** (str) - summary format for reasoning models
- **timeout** (int, >0, default=360) - request timeout in seconds
- **max_retries** (int, >0, default=3) - number of retries on API errors
- **poll_interval** (int, >0, default=5) - polling interval for async requests

## Error Handling & Exit Codes

### Exit Codes
- **0 (SUCCESS)** - successful execution
- **1 (CONFIG_ERROR)** - configuration errors
- **2 (INPUT_ERROR)** - no slices OR no ConceptDictionary.json
- **3 (RUNTIME_ERROR)** - any slice failed (including repair failure) OR critical error
- **4 (API_LIMIT_ERROR)** - API limits exhausted
- **5 (IO_ERROR)** - file write errors

### Recovery Strategy
1. Try to parse response
2. If JSON error → repair with format emphasis
3. Post-process IDs with `_assign_final_ids()` (no repair needed)
4. If repair fails → save dumps and exit with RUNTIME_ERROR

### Graceful Degradation
- Save bad responses for debugging
- Create temporary dumps on critical errors
- Exit immediately if slice fails (graph completeness required)

### Critical Failures
- **Any slice fails** → save dumps → EXIT_RUNTIME_ERROR (3)
- **Missing ConceptDictionary** → EXIT_INPUT_ERROR (2)
- **I/O errors** → save dumps → EXIT_IO_ERROR (5)

## Public Classes

### ProcessingStats
Graph processing statistics.
- **Attributes**: 
  - total_slices (int) - total number of slices to process
  - processed_slices (int) - successfully processed count
  - failed_slices (int) - failed slice count
  - total_nodes (int) - total nodes in graph
  - total_edges (int) - total edges in graph
  - total_tokens_used (int) - cumulative token usage
  - start_time (datetime) - processing start timestamp

### SliceData
Single slice data container.
- **Attributes**: 
  - id (str) - unique slice identifier
  - order (int) - sequential order number
  - source_file (str) - original source filename
  - slug (str) - textbook identifier
  - text (str) - slice text content
  - slice_token_start (int) - starting token position
  - slice_token_end (int) - ending token position

### SliceProcessor
Main processing class for graph construction.
- **__init__(config: Dict)** - initialization with configuration and ConceptDictionary loading
- **run() -> int** - main processing launch method, returns exit code

## Internal Methods

### SliceProcessor._format_tokens(tokens: int) -> str
Format token count to readable form.
- **Input**: tokens - number of tokens
- **Returns**: formatted string like "123", "45.61k", "1.22M"
- **Algorithm**:
  - Numbers < 1000: unchanged ("123")
  - Thousands (1K-999K): formatted as "45.61k" with two decimal places
  - Millions (1M+): formatted as "1.22M" with two decimal places
- **Side effects**: None
- **Note**: Used for terminal output formatting

### SliceProcessor._assign_final_ids(patch: Dict, slice_data: SliceData) -> None
Replace temporary IDs with final position-based IDs.
- **Input**: 
  - patch - graph patch from LLM with temporary IDs (chunk_1, assessment_1)
  - slice_data - current slice data with slice_token_start
- **Algorithm**:
  1. Build ID mapping dictionary
  2. For each Chunk node with temporary ID:
     - Calculate final_position = slice_token_start + node_offset
     - Generate new_id = "{slug}:c:{final_position}"
     - Update node ID and store mapping
  3. For each Assessment node with temporary ID:
     - Calculate final_position = slice_token_start + node_offset
     - Extract index from temporary ID (assessment_1 → 1)
     - Generate new_id = "{slug}:q:{final_position}:{index}"
     - Update node ID and store mapping
  4. Update all edges to use new IDs from mapping
  5. Log ID replacements for debugging
- **Side effects**: Modifies patch nodes and edges in-place
- **Note**: Must be called BEFORE any validation or processing

### SliceProcessor.validate_node_positions(nodes: List[Dict], slice_token_start: int) -> Optional[List[Dict]]
**DEPRECATED**: Position validation no longer needed after ID post-processing.
- **Input**: 
  - nodes - list of nodes to validate
  - slice_token_start - start token position of current slice
- **Returns**: Always returns nodes (validation is skipped)
- **Note**: Kept for backward compatibility but performs no validation

### SliceProcessor._validate_node_positions_legacy(nodes: List[Dict], slice_token_start: int) -> Optional[List[Dict]]
Legacy validation method - kept for reference but not used.
- **Purpose**: Original validation logic that checked node_position calculations
- **Algorithm**:
  1. For each Chunk/Assessment node
  2. Check required fields (node_offset, node_position) exist
  3. Verify math: node_position = slice_token_start + node_offset
  4. Check position >= slice_token_start
  5. Verify ID consistency with stated position
  6. Return None if any issues found
- **Note**: Replaced by _assign_final_ids() which fixes IDs automatically

### SliceProcessor._process_single_slice(slice_file: Path) -> bool
Process single slice with repair mechanism.
- **Input**: slice_file - Path to slice JSON file
- **Returns**: True on success, False on failure
- **Algorithm**:
  1. Load slice data
  2. Format input with full ConceptDictionary
  3. Call LLM with previous_response_id
  4. Parse response and check for JSON errors
  5. If JSON error: repair with format emphasis and rollback
  6. Apply ID post-processing with `_assign_final_ids()`
  7. Process nodes and edges
  8. Add automatic MENTIONS edges
  9. Validate graph intermediate state
  10. Update previous_response_id on success
- **Side effects**: 
  - Updates graph_nodes and graph_edges
  - Updates previous_response_id
  - May create bad response files
- **Error handling**: Returns False on any failure

### SliceProcessor._process_chunk_nodes(new_nodes: List[Dict]) -> List[Dict]
Handle duplicate Chunks and Assessments.
- **Input**: new_nodes - list of nodes from LLM response
- **Returns**: List of nodes to add to graph
- **Algorithm**:
  1. For each node check if ID already exists
  2. For duplicate Chunks: keep version with longer text
  3. For duplicate Assessments: skip with WARNING
  4. For Concepts: add as-is (duplicates allowed)
  5. Add default difficulty=3 if missing from Chunks
- **Side effects**: May update existing nodes in graph_nodes
- **Note**: Concept duplicates handled later by dedup.py

### SliceProcessor._validate_edges(edges: List[Dict]) -> List[Dict]
Validate and filter edges.
- **Input**: edges - list of edge dictionaries
- **Returns**: List of valid edges
- **Algorithm**:
  1. Build set of existing edge keys
  2. For each new edge:
     - Drop PREREQUISITE self-loops
     - Clamp weights to [0,1] range
     - Check node existence in graph or ConceptDictionary
     - Filter duplicates by (source, target, type) key
- **Side effects**: None
- **Error handling**: Logs warnings for invalid edges

### SliceProcessor._add_mentions_edges(chunk_nodes: List[Dict]) -> int
Automatically create MENTIONS edges.
- **Input**: chunk_nodes - list of Chunk type nodes
- **Returns**: Number of edges added
- **Algorithm**:
  1. Build set of existing MENTIONS edges
  2. For each chunk:
     - Get chunk text (lowercase)
     - For each concept in dictionary:
       - Search for primary term (whole word, case-insensitive)
       - Search for aliases (whole word, case-insensitive)
       - If found and not exists: add MENTIONS edge
- **Side effects**: Adds edges to graph_edges
- **Note**: All automatic edges have weight=1.0

### SliceProcessor._validate_graph_intermediate() -> bool
Validate graph invariants after each slice.
- **Returns**: True if valid, False on violations
- **Checks**:
  - ID uniqueness for Chunks and Assessments
  - Allows Concept duplicates (for dedup.py)
- **Note**: Critical violation stops processing

### SliceProcessor._save_bad_response(slice_id: str, original_response: str, error: str, repair_response: Optional[str] = None) -> None
Save problematic LLM response.
- **Input**: 
  - slice_id - identifier of failed slice
  - original_response - first LLM response text
  - error - error description
  - repair_response - response after repair (optional)
- **Output**: JSON file `/logs/{slice_id}_bad.json`
- **Side effects**: Creates file in logs directory
- **Error handling**: Logs error if file write fails

### SliceProcessor._save_temp_dumps(reason: str) -> None
Save temporary dumps on critical errors.
- **Input**: reason - save reason (critical_error/interrupted/validation_failed/io_error)
- **Output**:
  - LearningChunkGraph_temp_{reason}_{timestamp}.json
  - processing_stats_{reason}_{timestamp}.json
- **Side effects**: Creates two files in logs directory
- **Note**: Used for recovery from failures

### SliceProcessor._process_llm_response(response_text: str, slice_id: str) -> Tuple[bool, Optional[Dict]]
Parse and validate LLM response with HTML attribute cleanup.
- **Input**: 
  - response_text - raw LLM response string
  - slice_id - current slice ID for logging
- **Returns**: (success: bool, parsed_data: Optional[Dict])
- **Algorithm**:
  1. Strip markdown code fences if present
  2. Fix common HTML attribute issues (escaped quotes)
  3. Parse JSON
  4. Validate required structure (chunk_graph_patch field)
  5. Check nodes and edges are lists
- **Error handling**: Returns (False, None) on any error
- **Note**: HTML cleanup handles LLM quirks

### SliceProcessor._add_to_graph(patch: Dict, slice_data: SliceData) -> None
Add patch to graph with full processing.
- **Input**: 
  - patch - graph patch from LLM response
  - slice_data - current slice data
- **Algorithm**:
  1. Process nodes with duplicate handling
  2. Add nodes to graph
  3. Validate and filter edges
  4. Add edges to graph
  5. Add automatic MENTIONS edges
- **Side effects**: 
  - Updates graph_nodes and graph_edges
  - Updates statistics
- **Note**: Combines all graph update operations

## Testing

### test_itext2kg_graph: 32 tests

**Initialization & Configuration:**
- test_initialization - processor setup with ConceptDictionary loading
- test_missing_concept_dictionary - error handling

**ID Post-processing:**
- test_assign_final_ids - ID replacement logic
- test_temporary_id_patterns - chunk_1, assessment_1 detection
- test_position_calculation - correct final ID generation

**Deprecated Validation:**
- test_validate_node_positions_deprecated - always returns nodes
- test_legacy_validation_preserved - legacy method exists

**Repair Mechanism:**
- test_dual_repair_json - JSON error repair
- test_repair_rollback - previous_response_id handling
- test_repair_failure - exit on repair failure

**Node Processing:**
- test_chunk_duplicate_handling - longer text wins
- test_assessment_duplicate_handling - skip duplicates
- test_concept_node_processing - use dictionary definition
- test_missing_difficulty - default value assignment

**Edge Processing:**
- test_edge_validation - weight clamping, self-loops
- test_edge_filtering - duplicate detection
- test_node_existence_check - reference validation
- test_automatic_mentions - safety net creation

**Graph Validation:**
- test_intermediate_validation - invariant checks
- test_duplicate_id_detection - critical error handling
- test_concept_duplicates_allowed - for dedup.py

**Integration:**
- test_full_pipeline - end-to-end with mocked LLM
- test_context_preservation - previous_response_id chain
- test_critical_error_handling - dumps and exit
- test_partial_success - proper failure mode

**LLM Response:**
- test_markdown_removal - code fence stripping
- test_html_cleanup - attribute fixing
- test_json_parsing - various formats
- test_structure_validation - required fields

**Error Recovery:**
- test_bad_response_saving - debug file creation
- test_temp_dumps - recovery files
- test_interrupt_handling - Ctrl+C processing

**Coverage: 82%**

## Dependencies

### Standard Library
- json, logging, sys, time, pathlib
- datetime, typing, dataclasses
- re (for HTML cleanup and MENTIONS search)

### External
- python-dotenv - environment variable loading
- openai - API client (via llm_client)

### Internal
- utils.config - configuration loading and validation
- utils.exit_codes - standardized exit codes
- utils.llm_client - OpenAI API client with repair_response() method for rollback
- utils.validation - JSON schema validation (currently unused but available)
- utils.console_encoding - UTF-8 console setup for Windows

## Boundary Cases

- **Empty staging** → EXIT_INPUT_ERROR (2)
- **Missing ConceptDictionary.json** → EXIT_INPUT_ERROR (2)
- **Corrupted slice.json** → log error, attempt to continue → EXIT_RUNTIME_ERROR (3)
- **Invalid temporary IDs** → handled by _assign_final_ids() post-processing
- **Duplicate Chunk/Assessment in intermediate validation** → EXIT_RUNTIME_ERROR (3)
- **Ctrl+C interruption** → save temporary dumps → EXIT_RUNTIME_ERROR
- **All slices failed** → save dumps → EXIT_RUNTIME_ERROR (3)
- **I/O error saving output** → save dumps → EXIT_IO_ERROR (5)
- **Empty graph (no nodes)** → log warning but save → SUCCESS
- **Graph with nodes but no edges** → log warning but save → SUCCESS

## Output Validation

Final validation uses:
- `validate_json()` - check against LearningChunkGraph.schema.json (if implemented)
- `validate_graph_invariants_intermediate()` - check graph structure:
  - ID uniqueness for Chunks and Assessments
  - Valid edge references
  - Concept duplicates allowed (handled by dedup.py)

## Output Format

### LearningChunkGraph_raw.json
```json
{
  "nodes": [
    {
      "id": "algo101:c:1000",
      "type": "Chunk",
      "text": "Example chunk text...",
      "node_offset": 0,
      "node_position": 1000,
      "difficulty": 3,
      "language": "ru",
      "metadata": {}
    },
    {
      "id": "algo101:q:5000:0",
      "type": "Assessment",
      "text": "Question text...",
      "node_offset": 200,
      "node_position": 5200,
      "difficulty": 4
    },
    {
      "id": "algo101:p:stack",
      "type": "Concept",
      "definition": "LIFO data structure..."
    }
  ],
  "edges": [
    {
      "source": "algo101:c:1000",
      "target": "algo101:p:stack",
      "type": "MENTIONS",
      "weight": 1.0
    },
    {
      "source": "algo101:c:1000",
      "target": "algo101:c:2000",
      "type": "PREREQUISITE",
      "weight": 0.8
    }
  ]
}
```

### Bad Response Format ({slice_id}_bad.json)
```json
{
  "slice_id": "slice_042",
  "timestamp": "2024-01-15T10:30:00Z",
  "original_response": "invalid LLM response text",
  "error": "JSON parse failed",
  "repair_response": "response after repair (if any)"
}
```

### Temporary Dumps Format

**LearningChunkGraph_temp_{reason}_{timestamp}.json:**
```json
{
  "nodes": [...],  // Current graph nodes
  "edges": [...]   // Current graph edges
}
```

**processing_stats_{reason}_{timestamp}.json:**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "reason": "critical_error",
  "stats": {
    "total_slices": 157,
    "processed_slices": 42,
    "failed_slices": 1,
    "total_nodes": 523,
    "total_edges": 1247,
    "total_tokens_used": 125000,
    "processing_time": "5m 23s"
  }
}
```

## Performance Notes

- **Sequential processing** required for context preservation
- **Memory usage**: proportional to graph size (~500MB for 10K nodes)
- **Speed**: ~10-15 slices/minute with o4-mini model
- **TPM control** via llm_client with safety margin
- **Bottleneck**: LLM API calls (limited by TPM)
- **ID post-processing**: eliminates repair overhead, faster than validation
- **Optimization**: automatic MENTIONS reduces need for LLM to find all mentions
- **Logging**: JSON Lines format for efficient parsing
- **Critical path**: Post-processing IDs is deterministic and fast

## Usage Examples

### Basic Usage
```bash
# Ensure ConceptDictionary exists
python -m src.itext2kg_concepts

# Run graph construction
python -m src.itext2kg_graph

# Check results
ls data/out/
# ConceptDictionary.json
# LearningChunkGraph_raw.json

# Verify graph structure
python -c "import json; g=json.load(open('data/out/LearningChunkGraph_raw.json')); print(f'Nodes: {len(g[\"nodes\"])}, Edges: {len(g[\"edges\"])}')"
```

### Error Recovery
```bash
# View error logs
cat logs/itext2kg_graph_*.log | grep ERROR

# Analyze bad responses
ls logs/*_bad.json
cat logs/slice_042_bad.json | jq .error

# Check for temporary ID replacements
cat logs/itext2kg_graph_*.log | grep "Replaced"

# Recover from temporary dumps
ls logs/LearningChunkGraph_temp_*.json
# LearningChunkGraph_temp_critical_error_20240115_103045.json
# processing_stats_critical_error_20240115_103045.json
```

### Debugging
```bash
# Enable debug logging in config.toml
# log_level = "debug"

# Check MENTIONS edge creation
cat logs/itext2kg_graph_*.log | grep "Added automatic MENTIONS"

# Analyze duplicate handling
cat logs/itext2kg_graph_*.log | grep "duplicate"

# Monitor repair attempts
cat logs/itext2kg_graph_*.log | grep REPAIR

# Check ID post-processing
cat logs/itext2kg_graph_*.log | grep "Post-processing"

# Validate final graph structure
python -c "
import json
g = json.load(open('data/out/LearningChunkGraph_raw.json'))
chunks = [n for n in g['nodes'] if n['type'] == 'Chunk']
mentions = [e for e in g['edges'] if e['type'] == 'MENTIONS']
print(f'Chunks: {len(chunks)}, MENTIONS edges: {len(mentions)}')
"
```

## See Also

- `/docs/specs/cli_itext2kg_concepts.md` - concept extraction utility
- `/docs/specs/util_llm_client.md` - LLM client implementation
- `/src/prompts/itext2kg_graph_extraction.md` - LLM prompt
- `/src/schemas/LearningChunkGraph.schema.json` - output schema
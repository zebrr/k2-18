# cli_itext2kg.md

## Status: READY

CLI utility for incremental knowledge graph construction from educational texts. Processes slices sequentially, sends them to LLM while preserving context through previous_response_id. Includes error recovery mechanisms and intermediate result saving.

## CLI Interface

**Launch:**
```bash
python -m src.itext2kg
```

**Input data:**
- `/data/staging/*.slice.json` - slices from slicer.py

**Output data:**
- `/data/out/ConceptDictionary.json` - concept dictionary
- `/data/out/LearningChunkGraph_raw.json` - knowledge graph
- `/logs/itext2kg_YYYY-MM-DD_HH-MM-SS.log` - detailed logs
- `/logs/{slice_id}_bad.json` - problematic LLM responses (on errors)
- `/logs/*_temp_*.json` - temporary dumps (on critical errors)

**Exit codes:**
- 0 (SUCCESS) - successful processing
- 1 (CONFIG_ERROR) - configuration errors
- 2 (INPUT_ERROR) - no slices in staging
- 3 (RUNTIME_ERROR) - all slices failed or critical error
- 4 (API_LIMIT_ERROR) - API limits exhausted
- 5 (IO_ERROR) - file write errors

## Core Algorithm

1. **Load slices** from staging in lexicographic order
2. **Sequential processing** with previous_response_id preservation:
   - Format input data (ConceptDictionary + Slice)
   - Call LLM via Responses API
   - Validate and parse response
   - Repair-reprompt on errors (1 attempt)
   - Incremental data structure update
   - **Intermediate validation** after each slice
   - Automatic addition of MENTIONS edges for all processed Chunks
3. **Error handling** with graceful degradation:
   - Continue on partial failures
   - Save temporary dumps on critical errors
4. **Final validation** using intermediate validation (allows concept duplicates)
5. **Save results** to output

## Terminal Output

The utility uses structured progress output with unified format:
```
[HH:MM:SS] TAG      | Data
```

### Progress Output Format

**START - processing start:**
```
[10:30:00] START    | 157 slices | model=o4-mini-2025-04-16 | tpm=100k
```

**SLICE - successful slice processing:**
```
[10:30:05] SLICE    | ✅ 001/157 | tokens_used=12.35k | tokens_current=1.23k | 5s | concepts=23 | nodes=156 | edges=287
[10:30:12] SLICE    | ✅ 002/157 | tokens_used=112.34k | tokens_current=11.23k incl. reasoning=567 | 8s | concepts=25 | nodes=163 | edges=301
```

**REPAIR - validation error fix attempts:**
```
[10:30:45] REPAIR   | 🔧 Attempting to fix JSON validation error...
[10:30:45] REPAIR   | 📝 Adding clarification to prompt and retrying...
[10:30:50] REPAIR   | ✅ JSON validation fixed successfully!
```

**ERROR - processing errors:**
```
[10:30:45] ERROR    | ❌ 042/157 | slice_042 | JSON validation failed after repair
[10:30:45] ERROR    | ❌ Incremental validation failed for slice_042
[10:30:45] ERROR    | 📋 Error: Duplicate node ID (Assessment): algo101:q:1234:0...
[10:31:02] ERROR    | ⚠️ RateLimitError | waiting for retry...
[10:31:15] ERROR    | ⚠️ APIError | slice slice_055
```

**FAILED - critical errors:**
```
[10:45:30] FAILED   | ❌ All slices failed processing
[10:45:30] FAILED   | ❌ Critical error: Connection timeout...
[10:45:30] FAILED   | ❌ Validation failed: Invalid graph structure...
```

**SAVING - saving temporary files:**
```
[10:45:30] SAVING   | 💾 Attempting to save empty structures...
[10:45:30] SAVING   | 💾 Emergency dump of current state...
[10:45:30] SAVING   | 💾 Attempting to save partial results...
```

**INFO - informational messages:**
```
[10:45:31] INFO     | Check /logs/ for temporary files and diagnostics
```

**SUCCESS - successful completion:**
```
[10:45:30] SUCCESS  | ✅ Results saved to /data/out/
                    | - ConceptDictionary.json
                    | - LearningChunkGraph_raw.json
```

**END - work completion:**
```
[10:45:30] END      | Done | slices=157 | time=15m 30s
```

### File Logging

Log files use JSON Lines format for structured analysis:
- **INFO level**: main processing events
- **DEBUG level**: full LLM prompts and responses (when log_level=debug)
- **ERROR level**: validation and API errors

Errors are also output to console via standard logger:
```
[10:30:00] ERROR    | No slice files found in staging directory
```

## Public Classes

### ProcessingStats
Slice processing statistics.
- **Attributes**: total_slices, processed_slices, failed_slices, total_concepts, total_nodes, total_edges, total_tokens_used, start_time

### SliceData
Single slice data.
- **Attributes**: id, order, source_file, slug, text, slice_token_start, slice_token_end

### SliceProcessor
Main processing class.
- **__init__(config)** - initialization with configuration
- **run()** - main processing launch method

## Internal Methods

### SliceProcessor._format_tokens(tokens)
Format token count to readable form.
- **Input**: tokens - number of tokens
- **Returns**: string like "123", "45.61k", "1.22M"
- **Features**:
  - Numbers < 1000: unchanged ("123")
  - Thousands (1K-999K): formatted as "45.61k" with two decimal places
  - Millions (1M+): formatted as "1.22M" with two decimal places

### SliceProcessor._process_chunk_nodes(new_nodes)
Process Chunk and Assessment nodes with duplicate checking.
- **Input**: new_nodes - list of new nodes from patch
- **Returns**: list of nodes to add to graph
- **Features**:
  - For Chunk: text length comparison, update if new is longer
  - For Assessment: ignore duplicates with warning logging
  - For other types: add without changes

### SliceProcessor._validate_edges(edges)
Edge validation with node existence checking and duplicate filtering.
- **Input**: edges - list of edges to check
- **Returns**: list of valid edges
- **Features**:
  - Check source/target node existence (including concepts from ConceptDictionary)
  - Drop PREREQUISITE self-loops
  - Check weights in range [0,1]
  - **Duplicate edge filtering**:
    - Check against existing edges in graph
    - Check duplicates within current patch
    - Duplicates determined by combination (source, target, type)
    - Weight ignored when determining duplicate
  - Invalid edges dropped with WARNING logging
  - Duplicate edges dropped with INFO logging

### SliceProcessor._save_bad_response(slice_id, original_response, error, repair_response=None)
Save incorrect LLM response for analysis.
- **Input**: slice_id, original response, error description, repair response (if any)
- **Output**: file `/logs/{slice_id}_bad.json` with full information

### SliceProcessor._save_temp_dumps(reason)
Save temporary dumps on critical errors.
- **Input**: reason - save reason (interrupted, validation_failed, io_error, all_failed, critical_error, validation_error_slice_{id})
- **Output**: 
  - ConceptDictionary_temp_{reason}_{timestamp}.json
  - LearningChunkGraph_temp_{reason}_{timestamp}.json
  - processing_stats_{reason}_{timestamp}.json

### SliceProcessor._process_single_slice(slice_file)
Process single slice with full error handling.
- **Returns**: True on success, False on failure
- **Features**: 
  - repair-reprompt on invalid JSON
  - save bad responses
  - intermediate validation after patch application
  - graceful error handling

### SliceProcessor._add_mentions_edges(chunk_nodes)
Automatically add MENTIONS edges from Chunks to Concepts based on text search.
- **Input**: chunk_nodes - list of Chunk type nodes for processing
- **Returns**: number of added MENTIONS edges
- **Algorithm**:
  - For each Chunk searches for mentions of all concepts from ConceptDictionary
  - Search performed on term.primary and all term.aliases
  - **Search rules**:
    - Full word matches only (uses regex `\b` word boundaries)
    - Case-insensitive (lowercase comparison)
    - Exact forms only (no morphology, "stacks" ≠ "stack")
  - Avoids duplicating existing MENTIONS edges
  - All MENTIONS edges have weight=1.0

### SliceProcessor._process_llm_response(response_text, slice_id)
Process and validate LLM response with pre-cleaning of known issues.
- **Input**: response_text - raw LLM response, slice_id - current slice ID
- **Returns**: (success, parsed_data) - success and parsed data or None
- **Features**:
  - **HTML attribute pre-cleaning**:
    - Fixes patterns like `href='\"url\"'` → `href="url"`
    - Fixes patterns like `src="'url'"` → `src="url"` 
    - Applied to all responses before JSON parsing
    - Processes attributes: href, src, target, action, name, frameborder, width, height, align
    - Uses regular expressions for replacement
  - Parse cleaned JSON
  - Validate response structure (presence of concepts_added and chunk_graph_patch)
  - Basic schema validation for ConceptDictionary and LearningChunkGraph
  - Logs details for debugging on error

## Key Features

### Context Management
- Automatic previous_response_id management
- Context preservation between slices up to 128K tokens
- Same previous_response_id used for retry and repair

### Incremental ConceptDictionary Update
- New concepts added entirely with automatic case-insensitive duplicate cleanup in aliases
- For existing concepts:
  - Only aliases updated with case-insensitive uniqueness check
  - Primary term and definition preserved from first appearance
  - New aliases added only if their lowercase versions not already in list
- Creation of Concept type nodes for new concepts
- **Case-insensitive logic**:
  - When adding new concept: removes duplicate aliases (e.g., ["Stack", "stack"] → ["Stack"])
  - When updating existing: new aliases checked case-insensitive against existing
  - First occurrence of each unique alias preserved with original case

**Note:** The system automatically ensures case-insensitive uniqueness of aliases within each concept, preventing validation errors during incremental processing. LLM may return duplicates like ["Brute Force", "brute force"], but system will keep only first variant.

### Node Duplicate Processing
- **Chunk nodes**: for identical IDs keeps longer text version
- **Assessment nodes**: duplicates ignored with warning
- **Concept nodes**: duplicates NOT prevented (semantic deduplication in dedup.py)
- All changes and warnings logged

### Intermediate Validation
- Performed after processing each slice
- Uses `validate_graph_invariants_intermediate()`:
  - Checks ID uniqueness for Chunk and Assessment
  - Does NOT check ID uniqueness for Concept (duplicates allowed)
  - Checks all other graph invariants
- On validation error:
  - Slice marked as failed
  - Temporary state saved for debugging
  - Processing continues with next slice

### Edge Validation
- Check source/target node existence
- Drop PREREQUISITE self-loops
- Check weights in range [0,1]
- **Duplicate edge filtering** (added to solve previous_response_id issue):
  - LLM may recreate MENTIONS for nodes from previous slices
  - All duplicates automatically filtered
  - INFO level logging for tracking
- Support for references to concepts from ConceptDictionary

### Automatic MENTIONS Edges Addition
- After applying each patch automatically searches for concept mentions
- Processes both new Chunk nodes and updated existing ones
- Search across all terms from ConceptDictionary (primary + aliases)
- Ensures graph completeness even if LLM missed obvious connections
- Example:
  ```
  Concept: {"primary": "Stack", "aliases": ["стек", "LIFO"]}
  Chunk text: "We use стек for storage. Stack is a LIFO structure."
  Result: 1 MENTIONS edge (all three mentions lead to same concept)
  ```

### Error Recovery
- **Repair-reprompt**: on invalid JSON makes repeat request with clarification
  - Method `repair_response()` automatically uses saved previous_response_id
  - Repair prompt adds explicit error indication and valid JSON requirement
- **HTML attributes cleanup**: before JSON parsing automatically fixes known issues with quotes in HTML attributes that LLM sometimes generates when copying links from slices
- **Graceful degradation**: process continues on partial failures
- **Temporary dumps**: state saving on critical errors
- **Interrupt handling**: correct Ctrl+C handling with result saving

## Configuration

Section `[itext2kg]` in config.toml:
- **model** - LLM model (o4-mini-2025-04-16)
- **tpm_limit** - tokens per minute limit
- **tpm_safety_margin** - TPM safety margin (0.15)
- **max_completion** - maximum tokens per generation
- **log_level** - logging level (debug/info)
- **temperature** - for regular models
- **reasoning_effort** - for reasoning models
- **reasoning_summary** - summary format for reasoning models
- **timeout** - request timeout in seconds
- **max_retries** - number of retries on API errors

## Error Handling & Exit Codes

### Recoverable Errors
- **JSON validation errors** → repair-reprompt (1 attempt) → bad response saved
- **Incremental validation errors** → slice marked failed → temporary dump → continue
- **API errors** → exponential backoff via llm_client (20s → 40s → 80s...)
- **Rate limits** → automatic wait via TPMBucket with recovery

### Non-recoverable Errors
- **All slices failed** → temporary dumps → EXIT_RUNTIME_ERROR (3)
- **Configuration errors** → EXIT_CONFIG_ERROR (1)
- **I/O errors** → temporary dumps → EXIT_IO_ERROR (5)

### Partial Failures
- Process continues if at least some slices successful
- Warning on failure rate > 50%
- Statistics saved in logs and temporary dumps

## Boundary Cases

- **Empty staging** → EXIT_INPUT_ERROR (2)
- **Corrupted slice.json** → logging, skip slice, continue
- **Invalid LLM response after repair** → save to logs/{slice_id}_bad.json
- **Ctrl+C interruption** → save temporary dumps → EXIT_RUNTIME_ERROR
- **Validation failed (final)** → temporary dumps with validation_failed prefix
- **Validation failed (intermediate)** → temporary dump for slice → slice failed
- **High failure rate** → warning, but continue work
- **Concept ID duplicate** → allowed, semantic deduplication in dedup.py
- **Duplicate edges from LLM** → automatically filtered in _validate_edges → INFO logging
- **Incorrect HTML attributes in LLM response** → automatic cleanup of patterns `='\"...\"'` → successful parsing

## Output Validation

Final validation uses:
- `validate_json()` - check against ConceptDictionary and LearningChunkGraph schemas
- `validate_concept_dictionary_invariants()` - check dictionary invariants
- `validate_graph_invariants_intermediate()` - intermediate graph validation (allows concept duplicates)

**Important:** Final validation does NOT use full `validate_graph_invariants()`, as concept duplicates may exist at this stage that will be processed in dedup.py.

## Output Format

**ConceptDictionary.json:**
```json
{
  "concepts": [
    {
      "concept_id": "slug:p:term",
      "term": {"primary": "Term", "aliases": ["term", "synonym"]},
      "definition": "Concept definition"
    }
  ]
}
```

**LearningChunkGraph_raw.json:**
```json
{
  "nodes": [
    {
      "id": "slug:c:token_start",
      "type": "Chunk|Concept|Assessment",
      "text": "Node text",
      "local_start": 0,
      "difficulty": 1,
      "definition": "For Concept type nodes"
    }
  ],
  "edges": [
    {
      "source": "node_id",
      "target": "node_id", 
      "type": "PREREQUISITE|MENTIONS|...",
      "weight": 0.8
    }
  ]
}
```

**Bad Response Format ({slice_id}_bad.json):**
```json
{
  "slice_id": "slice_001",
  "timestamp": "2024-01-15T10:30:00Z",
  "original_response": "invalid LLM response",
  "validation_error": "error description",
  "repair_response": "response after repair (if any)"
}
```

## Test Coverage

- **test_slice_processor**: 21 tests
  - Initialization and prompt loading
  - Input data formatting
  - Concept and node updates
  - Edge validation
  - LLM response processing
  - **Automatic MENTIONS edges addition (5 tests)**
  
- **test_processing_flow**: 8 tests
  - Slice loading
  - Patch application
  - Bad responses saving
  - Successful processing and repair
  - Full pipeline run

- **test_itext2kg_error_handling**: 9 tests
  - Bad responses saving
  - Temporary dumps creation
  - Repair-reprompt mechanism
  - Graceful degradation
  - Interrupt handling
  - All failure types

- **test_itext2kg_deduplication**: 11 tests
  - Node deduplication (Chunk/Assessment)
  - Overlapping text processing
  - Incremental validation
  - **Edge deduplication (4 tests)**
  - Duplicate MENTIONS filtering
  - Scenario with previous_response_id

## Dependencies
- **Standard Library**: json, logging, re, sys, time, pathlib, datetime, typing, dataclasses
- **External**: python-dotenv
- **Internal**: utils.config, utils.exit_codes, utils.llm_client, utils.validation (including validate_graph_invariants_intermediate), utils.console_encoding

## Performance Notes
- Sequential processing for context preservation
- TPM control via llm_client with safety margin
- Detailed logging in JSON Lines format
- Real-time progress output to terminal
- Checkpoint logging every 10 slices
- Minimal delay on repair due to previous_response_id preservation
- Intermediate validation adds minimal overhead (< 50ms per slice)
- Duplicate edge checking adds minimal overhead (< 10ms per patch)
- HTML attribute pre-cleaning adds minimal overhead (< 1ms per response)

## Usage Examples
```bash
# Prepare slices
python -m src.slicer

# Run extraction
python -m src.itext2kg

# Check results
ls data/out/
# ConceptDictionary.json
# LearningChunkGraph_raw.json

# View error logs
cat logs/itext2kg_*.log | grep ERROR

# Analyze bad responses
ls logs/*_bad.json

# Recover from temporary dumps
ls logs/*_temp_*.json
# ConceptDictionary_temp_interrupted_20240115_103045.json
# LearningChunkGraph_temp_interrupted_20240115_103045.json
# processing_stats_interrupted_20240115_103045.json
# ConceptDictionary_temp_validation_error_slice_042_20240115_103045.json
```
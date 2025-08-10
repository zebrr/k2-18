# cli_itext2kg_concepts.md

## Status: READY

CLI utility for incremental concept extraction from educational texts. Processes slices sequentially, sends them to LLM while preserving context through previous_response_id. Extracts only concepts to build ConceptDictionary without any graph construction. Uses two-phase confirmation pattern for robust error recovery.

## CLI Interface

### Usage
```bash
python -m src.itext2kg_concepts
```

### Input Directory/Files
- **Source**: `/data/staging/*.slice.json` - slices from slicer.py
- **Formats**: JSON files with slice structure

### Output Directory/Files
- **Target**: `/data/out/ConceptDictionary.json` - concept dictionary with metadata
- **Logs**: `/logs/itext2kg_concepts_YYYY-MM-DD_HH-MM-SS.log` - detailed logs
- **Debug**: `/logs/{slice_id}_bad.json` - problematic LLM responses (on errors)
- **Recovery**: `/logs/*_temp_*.json` - temporary dumps (on critical errors)

## Key Features

### Two-Phase Confirmation Pattern
Module uses two-phase confirmation from llm_client for robust error recovery:

1. **Response Phase**: 
   - `create_response()` returns response but doesn't finalize chain
   - Response stored as unconfirmed in llm_client
   - Previous unconfirmed responses auto-cleared
   
2. **Validation Phase**:
   - Response parsed and validated against schema
   - On success: `confirm_response()` called to finalize
   - On failure: proceed to repair without confirmation
   
3. **Repair Phase**:
   - `repair_response()` automatically uses last confirmed response as context
   - Ensures clean rollback without broken JSON in chain
   - Repair responses don't need explicit confirmation (handled internally by llm_client)

This pattern prevents invalid responses from corrupting the context chain and ensures reliable incremental processing.

### Context Management
- Automatic previous_response_id management for incremental processing
- Context preservation between slices up to 128K tokens
- Two-phase confirmation ensures only valid responses enter the chain
- **CRITICAL**: Context helps LLM remember which concepts were already extracted
- **Implementation**: 
  - `self.previous_response_id` initialized as None in `__init__()`
  - Each `create_response()` call passes `previous_response_id=self.previous_response_id`
  - After successful processing AND confirmation, updated with new response_id
  - On successful repair, updated with repair_id (repair responses are auto-confirmed)
  - On repair, uses rollback to last confirmed response_id

### Incremental ConceptDictionary Update
- New concepts added with automatic case-insensitive duplicate cleanup in aliases
- Primary term excluded from aliases during deduplication
- For existing concepts:
  - Only aliases updated with case-insensitive uniqueness check
  - Aliases matching primary term (case-insensitive) are filtered out
  - Primary term and definition preserved from first appearance
  - New aliases added only if lowercase versions not already in list
- **Case-insensitive logic**:
  - When adding new concept: removes duplicate aliases AND those matching primary (e.g., ["Stack", "stack", "Стек"] with primary="Стек" → ["Stack", "stack"])
  - When updating existing: new aliases checked case-insensitive against existing AND primary
  - First occurrence of each unique alias preserved with original case

### Error Recovery
- **Repair-reprompt with two-phase confirmation**: 
  - On invalid JSON, makes repair request with clarification
  - Repair uses `repair_response()` which automatically rolls back to last confirmed response
  - Prevents "anchoring" on broken JSON structure in context
  - Repair prompt adds explicit error indication and valid JSON requirement
  - Successful repairs are auto-confirmed by llm_client
- **Graceful degradation**: process continues on partial failures
- **Temporary dumps**: state saving on critical errors
- **Interrupt handling**: correct Ctrl+C handling with result saving

### Debug Mode
- When `log_level = "debug"` in config:
  - Logs complete LLM prompts with formatted input data
  - Logs complete LLM responses with usage statistics
  - Enables detailed tracing of concept dictionary updates
  - Useful for debugging extraction issues and prompt optimization

### API Usage Tracking
- Tracks total API requests, input tokens, and output tokens
- Statistics included in final metadata for cost analysis
- Accumulated throughout processing for monitoring

## Core Algorithm

1. **Load slices** from staging in lexicographic order
2. **Sequential processing** with previous_response_id preservation:
   - Format input data (ConceptDictionary + Slice)
   - Call LLM via Responses API
   - Validate and parse response
   - Confirm successful responses via two-phase pattern
   - Repair-reprompt on errors (1 attempt with automatic rollback)
   - Incremental concept dictionary update
3. **Error handling** with graceful degradation:
   - Continue on partial failures
   - Save temporary dumps on critical errors
4. **Final validation** of ConceptDictionary invariants
5. **Save results** with metadata to output

## Terminal Output

The utility uses structured progress output with unified format:
```
[HH:MM:SS] TAG      | Data
```

### Output Format

**START - processing start:**
```
[10:30:00] START    | 157 slices | model=o4-mini-2025-04-16 | tpm=100k
```

**SLICE - successful slice processing:**
```
[10:30:05] SLICE    | ✅ 001/157 | tokens_used=12.35k | tokens_current=1.23k | 5s | concepts=23
[10:30:12] SLICE    | ✅ 002/157 | tokens_used=112.34k | tokens_current=11.23k incl. reasoning=567 | 8s | concepts=25
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
[10:31:02] ERROR    | ⚠️ RateLimitError | waiting for retry...
[10:31:15] ERROR    | ⚠️ APIError | slice slice_055
```

**FAILED - critical errors:**
```
[10:45:30] FAILED   | ❌ All slices failed processing
[10:45:30] FAILED   | ❌ Critical error: Connection timeout...
[10:45:30] FAILED   | ❌ Validation failed: Invalid concept structure...
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
```

**END - work completion:**
```
[10:45:30] END      | Done | slices=157 | time=15m 30s
```

## Configuration

Section `[itext2kg]` in config.toml:

### Required Parameters
- **model** (str) - LLM model (o4-mini-2025-04-16)
- **tpm_limit** (int, >0) - tokens per minute limit
- **log_level** (str) - logging level (debug/info/warning/error)
- **is_reasoning** (bool) - whether model is a reasoning model (REQUIRED for llm_client)

### Optional Parameters
- **tpm_safety_margin** (float, 0-1, default=0.15) - TPM safety margin
- **max_completion** (int, >0) - maximum tokens per generation
- **temperature** (float, 0-2) - for regular models
- **reasoning_effort** (str) - for reasoning models (low/medium/high)
- **reasoning_summary** (str) - summary format for reasoning models
- **timeout** (int, >0, default=360) - request timeout in seconds
- **max_retries** (int, >0, default=3) - number of retries on API errors
- **max_context_tokens** (int, >=1000, default=128000) - maximum context size
- **max_context_tokens_test** (int, >=1000, default=128000) - max context for testing
- **poll_interval** (int, >0, default=5) - polling interval for async requests
- **response_chain_depth** (int, optional) - depth of response chain (None=unlimited, 0=independent, >0=sliding window)
- **truncation** (str, optional) - truncation strategy ("auto", "disabled", or comment out to omit)

### Validation
- Configuration parameters are validated at startup
- `max_context_tokens` and `max_context_tokens_test` must be integers >= 1000
- `is_reasoning` must be explicitly provided for llm_client
- Invalid configuration causes EXIT_CONFIG_ERROR

## Error Handling & Exit Codes

### Exit Codes
- **0 (SUCCESS)** - successful execution
- **1 (CONFIG_ERROR)** - configuration errors
- **2 (INPUT_ERROR)** - no slices in staging
- **3 (RUNTIME_ERROR)** - all slices failed or critical error
- **4 (API_LIMIT_ERROR)** - API limits exhausted
- **5 (IO_ERROR)** - file write errors

### Recoverable Errors
- **JSON validation errors** → repair-reprompt with automatic rollback
  - Uses two-phase confirmation: response confirmed only after successful validation
  - On validation failure, `repair_response()` uses last confirmed response_id
  - Prevents "Previous response not found" error and context corruption
  - Repair responses are auto-confirmed internally by llm_client
- **API errors** → exponential backoff via llm_client (20s → 40s → 80s...)
- **Rate limits** → automatic wait via TPMBucket with recovery

### Non-recoverable Errors
- **All slices failed** → temporary dumps → EXIT_RUNTIME_ERROR (3)
- **Configuration errors** → EXIT_CONFIG_ERROR (1)
- **I/O errors** → temporary dumps → EXIT_IO_ERROR (5)

### Partial Failures
- Process continues if at least some slices successful
- Statistics saved in logs and temporary dumps

## Public Classes

### ProcessingStats
Slice processing statistics.
- **Attributes**: 
  - total_slices (int) - total number of slices to process
  - processed_slices (int) - successfully processed count
  - total_concepts (int) - total concepts extracted
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
Main processing class for concept extraction.
- **__init__(config: Dict)** - initialization with configuration
- **run() -> int** - main processing launch method, returns exit code
- **Internal state tracking**:
  - api_usage - tracks total requests and token usage for cost monitoring
  - source_slug - extracted from first slice for metadata
  - total_source_tokens - calculated from last slice for statistics
  - previous_response_id - maintains context chain between slices

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

### SliceProcessor._save_bad_response(slice_id: str, original_response: str, error: str, repair_response: Optional[str] = None) -> None
Save incorrect LLM response for analysis.
- **Input**: 
  - slice_id - identifier of failed slice
  - original_response - first LLM response text
  - error - error description
  - repair_response - response after repair attempt (optional)
- **Output**: JSON file `/logs/{slice_id}_bad.json`
- **Side effects**: Creates file in logs directory
- **Error handling**: Logs error if file write fails

### SliceProcessor._save_temp_dumps(reason: str) -> None
Save temporary dumps on critical errors.
- **Input**: reason - save reason (interrupted/validation_failed/io_error/all_failed/critical_error)
- **Output**: 
  - ConceptDictionary_temp_{reason}_{timestamp}.json
  - processing_stats_{reason}_{timestamp}.json
- **Side effects**: Creates two files in logs directory
- **Note**: Used for recovery from failures

### SliceProcessor._process_single_slice(slice_file: Path) -> bool
Process single slice with two-phase confirmation pattern.
- **Input**: slice_file - Path to slice JSON file
- **Returns**: True on success, False on failure
- **Algorithm**:
  1. Load and validate slice data
  2. Format input with current ConceptDictionary state
  3. Call LLM with previous_response_id
  4. Track API usage (requests, tokens)
  5. Parse and validate response
  6. **On success**: Call `confirm_response()` to finalize
  7. **On JSON error**: 
     - Attempt repair with `repair_response()`
     - Repair automatically uses last confirmed response as context
     - No explicit confirmation needed (handled internally)
     - Save bad responses for debugging
  8. Update concept dictionary on success
  9. Update previous_response_id:
     - If repair was successful: use repair_id
     - Otherwise: use original response_id
- **Side effects**: 
  - Updates self.concept_dictionary
  - Updates self.previous_response_id
  - Updates self.api_usage counters
  - May create bad response files
  - Confirms valid responses via llm_client
- **Error handling**: Catches all exceptions, logs errors, returns False
- **Critical**: Uses two-phase confirmation to prevent context corruption

### SliceProcessor._process_llm_response(response_text: str, slice_id: str) -> Tuple[bool, Optional[Dict]]
Process and validate LLM response.
- **Input**: 
  - response_text - raw LLM response string
  - slice_id - current slice ID for logging
- **Returns**: (success: bool, parsed_data: Optional[Dict])
- **Algorithm**:
  1. Strip markdown code fences if present
  2. Parse JSON
  3. Validate required structure (concepts_added field)
  4. Basic schema validation for ConceptDictionary format
- **Error handling**: Returns (False, None) on any parsing/validation error
- **Note**: Logs detailed debug info on errors

### SliceProcessor._update_concept_dictionary(new_concepts: List[Dict]) -> None
Update concept dictionary with new concepts and aliases.
- **Input**: new_concepts - list of concept objects from LLM
- **Algorithm**:
  1. For each new concept:
     - If concept_id exists: update only aliases (case-insensitive check)
     - Filter out aliases matching primary term
     - If new: add entire concept with alias deduplication
  2. Case-insensitive duplicate removal within aliases
  3. Primary term excluded from aliases list
  4. Preserve first occurrence of each unique alias
- **Side effects**: Modifies self.concept_dictionary
- **Note**: Critical for incremental processing consistency

### SliceProcessor._apply_concepts(response_data: Dict) -> None
Apply concepts from validated LLM response to concept dictionary.
- **Input**: response_data - parsed LLM response with concepts_added field
- **Algorithm**:
  1. Extract concepts list from response_data["concepts_added"]["concepts"]
  2. Call _update_concept_dictionary() with new concepts
- **Side effects**: Modifies self.concept_dictionary via _update_concept_dictionary()
- **Note**: Separated from response processing for clarity and testability

### SliceProcessor._print_start_status() -> None
Output initial processing status to terminal.
- **Output**: Formatted status line with slice count, model, and TPM limit
- **Format**: `[HH:MM:SS] START    | {slices} slices | model={model} | tpm={limit}k`
- **Side effects**: Prints to stdout
- **Note**: Called once at the beginning of run()

### SliceProcessor._print_end_status() -> None
Output final processing status to terminal.
- **Output**: Formatted status line with processed count and elapsed time
- **Format**: `[HH:MM:SS] END      | Done | slices={processed} | time={m}m {s}s`
- **Side effects**: Prints to stdout
- **Note**: Called after successful completion before returning EXIT_SUCCESS

### SliceProcessor._finalize_and_save() -> int
Final validation and save results with comprehensive metadata.
- **Returns**: Exit code (SUCCESS or error code)
- **Algorithm**:
  1. Validate against schema and invariants
  2. Calculate concept statistics (aliases, counts)
  3. Collect comprehensive metadata including API usage
  4. Merge metadata with concept dictionary
  5. Save to output file
- **Side effects**: Creates ConceptDictionary.json with metadata
- **Error handling**: Returns appropriate exit codes, saves temp dumps on failure

## Testing

### test_itext2kg_concepts: 19 tests

**Initialization & Configuration:**
- test_initialization - processor setup with config
- test_configuration_loading - config.toml parsing
- test_configuration_validation - validation of max_context_tokens parameters

**Concept Processing:**
- test_concept_dictionary_updates - adding new concepts
- test_case_insensitive_deduplication - alias uniqueness
- test_primary_term_exclusion - primary not in aliases
- test_updating_existing_concepts - alias merging
- test_empty_response_handling - graceful empty handling

**LLM Integration:**
- test_llm_response_processing - response parsing
- test_bad_response_saving - error file creation
- test_repair_reprompt_mechanism - JSON error recovery with two-phase confirmation
- test_context_preservation - previous_response_id usage
- test_response_id_chaining - multi-slice context
- test_repair_id_update - correct ID after successful repair

**Pipeline:**
- test_full_pipeline_run - end-to-end processing
- test_temporary_dumps_creation - error recovery files
- test_partial_failure_handling - graceful degradation
- test_interrupt_handling - Ctrl+C processing

**Validation:**
- test_json_validation - schema compliance
- test_invariant_validation - dictionary consistency
- test_duplicate_concept_detection - ID uniqueness
- test_alias_case_sensitivity - case handling

**Coverage: 95%**

## Dependencies

### Standard Library
- json, logging, sys, time, pathlib
- datetime, typing, dataclasses
- re (for JSON cleanup)

### External
- python-dotenv - environment variable loading
- openai>=1.0.0 - via llm_client

### Internal
- utils.config - configuration loading and validation
- utils.exit_codes - standardized exit codes
- utils.llm_client - OpenAI API wrapper with retry logic and two-phase confirmation
- utils.validation - JSON schema validation
- utils.console_encoding - UTF-8 console setup for Windows

## Boundary Cases

- **Empty staging** → EXIT_INPUT_ERROR (2)
- **Corrupted slice.json** → log error, skip slice, continue
- **Invalid LLM response after repair** → save to logs/{slice_id}_bad.json, skip slice
- **Ctrl+C interruption** → save temporary dumps → EXIT_RUNTIME_ERROR
- **Validation failed (final)** → temporary dumps with validation_failed prefix
- **All slices have no concepts** → save empty ConceptDictionary.json → SUCCESS
- **Duplicate concept_id from LLM** → merge aliases, log warning
- **API timeout** → retry via llm_client, eventual fail after max_retries
- **Invalid configuration parameters** → EXIT_CONFIG_ERROR at startup
- **Broken JSON in response** → repair with automatic rollback to last confirmed

## Output Validation

Final validation uses:
- `validate_json()` - check against ConceptDictionary.schema.json
- `validate_concept_dictionary_invariants()` - imported function from `utils.validation` that checks custom invariants:
  - Unique concept_ids
  - Required fields present (concept_id, term, definition)
  - Primary term is non-empty string
  - Case-insensitive alias uniqueness within each concept
  - Primary term not in aliases list

## Output Format

### ConceptDictionary.json (with metadata)
```json
{
  "_meta": {
    "generated_at": "2024-01-15 10:30:00",
    "generator": "itext2kg_concepts",
    "config": {
      "model": "o4-mini-2025-04-16",
      "temperature": 0.1,
      "max_output_tokens": 25000,
      "reasoning_effort": "medium",
      "overlap": 500,
      "slice_size": 5000
    },
    "source": {
      "total_slices": 157,
      "processed_slices": 155,
      "total_tokens": 785000,
      "slug": "algo101"
    },
    "api_usage": {
      "total_requests": 312,
      "total_input_tokens": 1250000,
      "total_output_tokens": 450000,
      "total_tokens": 1700000
    },
    "concepts_stats": {
      "total_concepts": 234,
      "concepts_with_aliases": 189,
      "total_aliases": 567,
      "avg_aliases_per_concept": 2.42
    },
    "processing_time": {
      "start": "2024-01-15 10:00:00",
      "end": "2024-01-15 10:30:00",
      "duration_minutes": 30.5
    }
  },
  "concepts": [
    {
      "concept_id": "algo101:p:stack",
      "term": {
        "primary": "Стек",
        "aliases": ["stack", "LIFO", "стековая структура"]
      },
      "definition": "LIFO-структура данных, где элементы добавляются и удаляются с одного конца"
    }
  ]
}
```

### Bad Response Format ({slice_id}_bad.json)
```json
{
  "slice_id": "slice_001",
  "timestamp": "2024-01-15T10:30:00Z",
  "original_response": "invalid LLM response text",
  "validation_error": "JSON decode error: Expecting value: line 1 column 1",
  "repair_response": "response after repair attempt (if any)"
}
```

### Temporary Dumps Format

**ConceptDictionary_temp_{reason}_{timestamp}.json:**
```json
{
  "concepts": [...]  // Current state of concept dictionary
}
```

**processing_stats_{reason}_{timestamp}.json:**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "reason": "interrupted",
  "stats": {
    "total_slices": 157,
    "processed_slices": 42,
    "total_concepts": 234,
    "total_tokens_used": 125000,
    "processing_time": "5m 23s"
  }
}
```

## Performance Notes

- **Sequential processing** required for context preservation
- **Memory usage**: ~100MB for 1000 concepts
- **Speed**: ~20-30 slices/minute with o4-mini model
- **TPM control** via llm_client with safety margin (15% default)
- **Bottleneck**: LLM API calls (limited by TPM)
- **Optimization**: Previous_response_id reduces repeated concept extraction
- **Two-phase confirmation**: Small overhead for increased reliability
- **API usage tracking**: Collects statistics for cost analysis (total requests, input/output tokens)
- **Logging**: JSON Lines format for efficient parsing
- **Checkpoint**: Progress logged every 10 slices for recovery

## Usage Examples

### Basic Usage
```bash
# Prepare slices
python -m src.slicer

# Run concept extraction
python -m src.itext2kg_concepts

# Check results
ls data/out/
# ConceptDictionary.json

# Verify concept count
python -c "import json; d=json.load(open('data/out/ConceptDictionary.json')); print(f'Concepts: {len(d[\"concepts\"])}')"

# Check API usage from metadata
python -c "import json; d=json.load(open('data/out/ConceptDictionary.json')); m=d['_meta']['api_usage']; print(f'Total API cost estimate: {m[\"total_tokens\"]} tokens')"
```

### Error Recovery
```bash
# View error logs
cat logs/itext2kg_concepts_*.log | grep ERROR

# Analyze bad responses
ls logs/*_bad.json
cat logs/slice_042_bad.json | jq .error

# Recover from temporary dumps
ls logs/*_temp_*.json
# ConceptDictionary_temp_interrupted_20240115_103045.json
# processing_stats_interrupted_20240115_103045.json

# Resume with partial results
cp logs/ConceptDictionary_temp_interrupted_*.json data/out/ConceptDictionary.json
# Then modify config to process only remaining slices
```

### Debugging
```bash
# Enable debug logging in config.toml
# log_level = "debug"

# Run with specific slice range (manual intervention)
# Move processed slices to backup, run again

# Check for duplicate concepts
python -c "
import json
d = json.load(open('data/out/ConceptDictionary.json'))
ids = [c['concept_id'] for c in d['concepts']]
print(f'Total: {len(ids)}, Unique: {len(set(ids))}')
"

# View debug logs for prompt/response analysis
cat logs/itext2kg_concepts_*.log | jq 'select(.level == "DEBUG" and .event == "llm_request")'
cat logs/itext2kg_concepts_*.log | jq 'select(.level == "DEBUG" and .event == "llm_response")'

# Analyze API usage patterns
python -c "
import json
d = json.load(open('data/out/ConceptDictionary.json'))
m = d['_meta']['api_usage']
avg_in = m['total_input_tokens'] / m['total_requests']
avg_out = m['total_output_tokens'] / m['total_requests']
print(f'Avg tokens per request: {avg_in:.0f} in, {avg_out:.0f} out')
"
```

## See Also

- `/docs/specs/cli_itext2kg_graph.md` - graph construction utility
- `/docs/specs/util_llm_client.md` - LLM client with two-phase confirmation
- `/src/prompts/itext2kg_concepts_extraction.md` - LLM prompt
- `/src/schemas/ConceptDictionary.schema.json` - output schema
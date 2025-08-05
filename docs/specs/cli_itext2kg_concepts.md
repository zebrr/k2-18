# cli_itext2kg_concepts.md

## Status: READY

CLI utility for incremental concept extraction from educational texts. Processes slices sequentially, sends them to LLM while preserving context through previous_response_id. Extracts only concepts to build ConceptDictionary without any graph construction.

## CLI Interface

**Launch:**
```bash
python -m src.itext2kg_concepts
```

**Input data:**
- `/data/staging/*.slice.json` - slices from slicer.py

**Output data:**
- `/data/out/ConceptDictionary.json` - concept dictionary
- `/logs/itext2kg_concepts_YYYY-MM-DD_HH-MM-SS.log` - detailed logs
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
   - Incremental concept dictionary update
3. **Error handling** with graceful degradation:
   - Continue on partial failures
   - Save temporary dumps on critical errors
4. **Final validation** of ConceptDictionary invariants
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
- **Attributes**: total_slices, processed_slices, total_concepts, total_tokens_used, start_time

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

### SliceProcessor._save_bad_response(slice_id, original_response, error, repair_response=None)
Save incorrect LLM response for analysis.
- **Input**: slice_id, original response, error description, repair response (if any)
- **Output**: file `/logs/{slice_id}_bad.json` with full information

### SliceProcessor._save_temp_dumps(reason)
Save temporary dumps on critical errors.
- **Input**: reason - save reason (interrupted, validation_failed, io_error, all_failed, critical_error)
- **Output**: 
  - ConceptDictionary_temp_{reason}_{timestamp}.json
  - processing_stats_{reason}_{timestamp}.json

### SliceProcessor._process_single_slice(slice_file)
Process single slice with full error handling.
- **Returns**: True on success, False on failure
- **Features**: 
  - repair-reprompt on invalid JSON
  - save bad responses
  - graceful error handling

### SliceProcessor._process_llm_response(response_text, slice_id)
Process and validate LLM response.
- **Input**: response_text - raw LLM response, slice_id - current slice ID
- **Returns**: (success, parsed_data) - success and parsed data or None
- **Features**:
  - Parse JSON response
  - Validate response structure (presence of concepts_added)
  - Basic schema validation for ConceptDictionary
  - Logs details for debugging on error

## Key Features

### Context Management
- Automatic previous_response_id management
- Context preservation between slices up to 128K tokens
- Same previous_response_id used for retry and repair
- **CRITICAL**: Context helps LLM remember which concepts were already extracted
- **Implementation**: 
  - `self.previous_response_id` is initialized as None in `__init__()`
  - Each `create_response()` call passes `previous_response_id=self.previous_response_id`
  - After successful slice processing, `self.previous_response_id` is updated with the new response_id
  - This ensures incremental processing with full context awareness

### Incremental ConceptDictionary Update
- New concepts added entirely with automatic case-insensitive duplicate cleanup in aliases
- For existing concepts:
  - Only aliases updated with case-insensitive uniqueness check
  - Primary term and definition preserved from first appearance
  - New aliases added only if their lowercase versions not already in list
- **Case-insensitive logic**:
  - When adding new concept: removes duplicate aliases (e.g., ["Stack", "stack"] → ["Stack"])
  - When updating existing: new aliases checked case-insensitive against existing
  - First occurrence of each unique alias preserved with original case

**Note:** The system automatically ensures case-insensitive uniqueness of aliases within each concept, preventing validation errors during incremental processing. LLM may return duplicates like ["Brute Force", "brute force"], but system will keep only first variant.

### Error Recovery
- **Repair-reprompt**: on invalid JSON makes repeat request with clarification
  - Repair uses rollback to last successful response_id (not the failed one)
  - Prevents "anchoring" on broken JSON structure
  - Repair prompt adds explicit error indication and valid JSON requirement
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
- **max_context_tokens** - maximum context size
- **max_context_tokens_test** - context size for tests

## Error Handling & Exit Codes

### Recoverable Errors
- **JSON validation errors** → repair-reprompt (1 attempt) → bad response saved
- **API errors** → exponential backoff via llm_client (20s → 40s → 80s...)
- **Rate limits** → automatic wait via TPMBucket with recovery

### Non-recoverable Errors
- **All slices failed** → temporary dumps → EXIT_RUNTIME_ERROR (3)
- **Configuration errors** → EXIT_CONFIG_ERROR (1)
- **I/O errors** → temporary dumps → EXIT_IO_ERROR (5)

### Partial Failures
- Process continues if at least some slices successful
- Statistics saved in logs and temporary dumps

## Boundary Cases

- **Empty staging** → EXIT_INPUT_ERROR (2)
- **Corrupted slice.json** → logging, skip slice, continue
- **Invalid LLM response after repair** → save to logs/{slice_id}_bad.json
- **Ctrl+C interruption** → save temporary dumps → EXIT_RUNTIME_ERROR
- **Validation failed (final)** → temporary dumps with validation_failed prefix

## Output Validation

Final validation uses:
- `validate_json()` - check against ConceptDictionary schema
- `validate_concept_dictionary_invariants()` - check dictionary invariants

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

- **test_itext2kg_concepts**: 19 tests
  - Initialization and configuration
  - Concept dictionary updates with case-insensitive deduplication
  - Updating existing concepts with new aliases
  - LLM response processing and validation
  - Empty response handling
  - Bad response saving
  - Repair-reprompt mechanism
  - Context preservation through previous_response_id
  - Previous response ID chaining across multiple slices
  - Full pipeline run
  - Temporary dumps creation

## Dependencies
- **Standard Library**: json, logging, sys, time, pathlib, datetime, typing, dataclasses
- **External**: python-dotenv
- **Internal**: utils.config, utils.exit_codes, utils.llm_client, utils.validation, utils.console_encoding

## Performance Notes
- Sequential processing for context preservation
- TPM control via llm_client with safety margin
- Detailed logging in JSON Lines format
- Real-time progress output to terminal
- Checkpoint logging every 10 slices
- Minimal delay on repair due to previous_response_id preservation

## Usage Examples
```bash
# Prepare slices
python -m src.slicer

# Run concept extraction
python -m src.itext2kg_concepts

# Check results
ls data/out/
# ConceptDictionary.json

# View error logs
cat logs/itext2kg_concepts_*.log | grep ERROR

# Analyze bad responses
ls logs/*_bad.json

# Recover from temporary dumps
ls logs/*_temp_*.json
# ConceptDictionary_temp_interrupted_20240115_103045.json
# processing_stats_interrupted_20240115_103045.json
```
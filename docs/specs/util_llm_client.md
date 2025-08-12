# util_llm_client.md

## Status: READY

Client for working with OpenAI Responses API with support for asynchronous mode, token limit control, response chains, reasoning models, and two-phase confirmation pattern.

## Public API

### ResponseUsage
Dataclass for tracking used tokens.
- **Fields**: 
  - input_tokens (int) - tokens in input data
  - output_tokens (int) - tokens in response
  - total_tokens (int) - total count (input + output)
  - reasoning_tokens (int) - reasoning tokens (for o* models)

### TPMBucket
Class for tokens per minute limit control via response headers.

#### TPMBucket.__init__(initial_limit: int)
Initialize bucket with initial limit.
- **Input**: initial_limit - initial tokens per minute limit
- **Attributes**: 
  - initial_limit - saved initial limit
  - remaining_tokens - current token balance
  - reset_time - Unix timestamp for limit reset

#### TPMBucket.update_from_headers(headers: Dict[str, str]) -> None
Update state from response headers.
- **Input**: headers - dictionary with API headers (x-ratelimit-remaining-tokens, x-ratelimit-reset-tokens)
- **Logic**: Parses remaining tokens and reset time (format "XXXms" or "Xs")

#### TPMBucket.wait_if_needed(required_tokens: int, safety_margin: float = 0.15) -> None
Check token sufficiency and wait if necessary.
- **Input**: required_tokens - required amount, safety_margin - safety margin
- **Logic**: Waits until reset_time if insufficient tokens
- **Terminal Output**: Shows informational message about limit reset when waiting

### IncompleteResponseError
Exception for handling incomplete response status (token limit exceeded).

### OpenAIClient
Main client for working with OpenAI Responses API.

#### OpenAIClient.__init__(config: Dict[str, Any])
Initialize client with configuration.
- **Input**: config - dictionary with parameters:
  - api_key (str) - OpenAI API key
  - model (str) - model (gpt-4o, o4-mini, gpt-5, etc.)
  - is_reasoning (bool, REQUIRED) - whether model is a reasoning model
  - tpm_limit (int) - tokens per minute limit
  - tpm_safety_margin (float) - safety margin (default 0.15)
  - max_completion (int) - maximum tokens for generation
  - max_context_tokens (int) - maximum context window size (default 128000)
  - timeout (int) - request timeout in seconds
  - max_retries (int) - number of retry attempts
  - temperature (float, optional) - temperature parameter (sent only if not None)
  - reasoning_effort (str, optional) - reasoning effort level (sent only if not None)
  - reasoning_summary (str, optional) - summary type for reasoning (sent only if not None)
  - verbosity (str, optional) - verbosity level for GPT-5 models (sent only if not None)
  - response_chain_depth (int, optional) - response chain depth (None=unlimited, 0=independent, >0=sliding window)
  - truncation (str, optional) - truncation strategy ("auto", "disabled", or None to omit)
  - poll_interval (int) - polling interval in seconds (default 5)
  - probe_model (str, optional) - model for TPM probe requests (default: "gpt-4.1-nano-2025-04-14")
- **Attributes**:
  - last_response_id (str) - ID of last successful response (for backward compatibility)
  - last_confirmed_response_id (str) - ID of last confirmed response for chains
  - unconfirmed_response_id (str) - ID of response awaiting confirmation
  - last_usage (ResponseUsage) - usage info from last response for context accumulation
  - encoder - tokenizer (tiktoken with o200k_base) for precise token counting
  - is_reasoning_model (bool) - explicitly set from config
  - response_chain_depth (int|None) - chain management mode
  - response_chain (deque|None) - sliding window of response IDs (only if depth > 0)
  - truncation (str|None) - truncation strategy for API
  - probe_model (str) - current model used for TPM probes
  - probe_fallback_models (list) - fallback chain for probe failures
  - _cached_tpm_limit (int) - cached TPM limit for ultimate fallback
  - _cached_tpm_remaining (int) - cached remaining tokens for ultimate fallback
- **Raises**: ValueError - if required parameter 'is_reasoning' is missing

#### OpenAIClient.create_response(instructions: str, input_data: str, previous_response_id: Optional[str] = None, is_repair: bool = False) -> Tuple[str, str, ResponseUsage]
Create response via OpenAI Responses API (public interface).
- **Input**: 
  - instructions - system prompt
  - input_data - user data
  - previous_response_id - ID of previous response (optional)
  - is_repair - if True, response won't be added to chain (for repair requests)
- **Returns**: (response_text, response_id, usage_info)
- **Side effects**:
  - Automatically clears forgotten unconfirmed responses
  - Saves new response as unconfirmed (unless is_repair=True)
  - Updates last_usage for context tracking
- **Raises**: 
  - TimeoutError - when timeout exceeded
  - IncompleteResponseError - for incomplete status (NO RETRY)
  - ValueError - for failed status or model refusal
  - openai.RateLimitError - when rate limit exceeded

#### OpenAIClient.repair_response(instructions: str, input_data: str, previous_response_id: Optional[str] = None) -> Tuple[str, str, ResponseUsage]
Repair request with specified previous_response_id (transport layer).
- **Input**: 
  - instructions - system prompt (caller should include repair instructions)
  - input_data - user data
  - previous_response_id - ID of previous response (optional, uses last_confirmed_response_id if None)
- **Returns**: (response_text, response_id, usage_info)
- **Logic**: 
  - Uses last_confirmed_response_id by default (not last_response_id)
  - Delegates to create_response with is_repair=True (response NOT added to chain)
  - Repair responses don't require confirmation

#### OpenAIClient.confirm_response() -> None
Confirm that the last unconfirmed response is valid.
- **Logic**: 
  - Updates last_confirmed_response_id from unconfirmed_response_id
  - Updates last_response_id for backward compatibility
  - Adds response to chain if configured
  - Deletes old responses if chain exceeds depth
  - Clears unconfirmed_response_id
- **Note**: Should be called after successful validation of response content
- **Safe**: Can be called multiple times (no-op if no unconfirmed response)

## Internal Methods

### OpenAIClient._delete_response(response_id: str) -> None
Delete response via OpenAI API.
- **Input**: response_id - ID of response to delete
- **Logic**: Calls client.responses.delete(response_id)
- **Raises**: ValueError if deletion fails (except for 404 errors)
- **Note**: Used for managing response chain when sliding window exceeds depth

### OpenAIClient._update_tpm_via_probe() -> None
Get current rate limit data via probe request with fallback chain.
- **Logic**: 
  - Tries probe_model first, then fallback models in order
  - Fallback chain: cheapest nano models → mini models → main model
  - Caches TPM limits for future use
  - Uses cached limits if all models fail
- **Note**: Necessary because OpenAI doesn't return rate limit headers in background mode

### OpenAIClient._create_response_async(instructions, input_data, previous_response_id, is_repair) -> Tuple[str, str, ResponseUsage]
Main logic for creating response in asynchronous mode.
- **Token Calculation**:
  1. If previous_response_id exists and last_usage available:
     - Use last_usage.input_tokens as base (includes all accumulated context)
     - Add only new content tokens
     - Check against max_context_tokens limit
  2. Otherwise: count tokens in full prompt (instructions + input_data)
- **Steps**:
  1. Auto-clear forgotten unconfirmed responses (if not repair)
  2. TPM probe to update limits
  3. Create background request (background=true)
  4. Polling loop with status check
  5. Handle statuses: completed, incomplete, failed, cancelled, queued
  6. Save as unconfirmed_response_id (unless is_repair)
  7. Cancel on timeout with cancellation attempt
- **Progress Display**: 
  - Queued status shown at first detection
  - Progress displayed every 3 polling iterations with elapsed time

### OpenAIClient._prepare_request_params(instructions, input_data, previous_response_id) -> Dict[str, Any]
Prepare parameters for Responses API.
- **Logic**: 
  - Base parameters + store=true
  - Adds only non-null parameters to request:
    - temperature: included if not None (for any model type)
    - reasoning: included if reasoning_effort or reasoning_summary not None
    - verbosity: included if not None (top-level parameter)
    - truncation: included if not None (for context management)
  - Parameters with None values are NOT sent to API

### OpenAIClient._extract_response_content(response) -> str
Extract text from response object.
- **Logic**:
  - Uses is_reasoning_model from config to determine parsing:
    - If is_reasoning=true: output[0]=reasoning, output[1]=message
    - If is_reasoning=false: output[0]=message
  - Handle refusal and incomplete statuses
  - Support both completed and incomplete statuses

### OpenAIClient._extract_usage_info(response) -> ResponseUsage
Extract token usage information from response.

### OpenAIClient._clean_json_response(response_text: str) -> str
Clean response from markdown wrappers.
- **Logic**: Removes ```json...``` and ```...``` wrappers

## Terminal Output

Module uses structured terminal output with format `[HH:MM:SS] TAG | message`:

### Asynchronous Response Generation
- **QUEUE** - waiting in OpenAI queue (shown at first detection)
  ```
  [10:30:01] QUEUE    | ⏳ Response 6871a162606c... in progress
  ```

- **PROGRESS** - waiting progress (every 3 polling iterations)
  ```
  [10:30:08] PROGRESS | ⏳ Elapsed: 7s
  [10:30:15] PROGRESS | ⏳ Elapsed: 14s
  [10:30:22] PROGRESS | ⏳ Elapsed: 21s
  [10:31:08] PROGRESS | ⏳ Elapsed: 1m 7s
  ```

- **ERROR** - critical errors
  ```
  [10:30:15] ERROR    | ❌ Response incomplete: max_output_tokens
  [10:30:15] ERROR    |    Generated only 4000 tokens
  [10:30:20] ERROR    | ❌ Response generation failed: Server error
  ```

- **HINT** - useful hints for reasoning models
  ```
  [10:30:16] HINT     | 💡 Reasoning model needs more tokens. Current limit: 10000
  ```

- **RETRY** - retry information
  ```
  [10:30:25] RETRY    | 🔄 Increasing token limit: 10000 → 15000
  [10:30:30] RETRY    | ⏳ Waiting 40s before retry 2/3...
  ```

### TPM Control
- **INFO** - limit reset after waiting
  ```
  [10:31:15] INFO     | ✅ TPM limit reset, continuing...
  ```

## Usage Examples

### Two-Phase Confirmation Pattern (RECOMMENDED)
```python
# Phase 1: Get response
response_text, response_id, usage = client.create_response(instructions, data)

# Phase 2: Validate and confirm
try:
    validated_data = json.loads(response_text)
    # Success - confirm the response
    client.confirm_response()
except json.JSONDecodeError:
    # Error - repair automatically uses last confirmed response
    repair_text, repair_id, _ = client.repair_response(
        instructions + "\nReturn valid JSON",
        data
    )
    # Repair responses don't need confirmation
    validated_data = json.loads(repair_text)
```

### Handling Multiple Responses with Confirmation
```python
last_successful_id = None

for data_chunk in data_chunks:
    try:
        # Get response
        text, resp_id, usage = client.create_response(
            instructions,
            data_chunk,
            previous_response_id=last_successful_id
        )
        
        # Validate
        result = validate_response(text)
        
        # Confirm on success
        client.confirm_response()
        last_successful_id = resp_id
        
    except ValidationError:
        # Repair with rollback to last confirmed
        repair_text, _, _ = client.repair_response(
            instructions + "\nFix the error",
            data_chunk
        )
        result = validate_response(repair_text)
        # Note: repair doesn't update last_successful_id
```

## Test Coverage

- **test_llm_client**: 28+ tests
  - Initialization tests - verify config validation and required parameters
  - TPM bucket tests - token limit control and waiting logic
  - Parameter preparation tests - request params formatting for API
  - Response extraction tests - parsing different response structures
  - Async response creation tests - background mode and polling
  - Retry logic tests - exponential backoff and error handling
  - Incomplete handling - NO RETRY, immediate error
  - Timeout handling - request cancellation on timeout
  - Response chain management - three modes (unlimited, independent, sliding window)
  - Response deletion - via OpenAI API
  - Truncation parameter - context management
  - Repair flag - responses not added to chain
  - Two-phase confirmation - unconfirmed/confirmed response management
  - Probe failure with fallback - fallback chain when probe model unavailable
  - All probes fail uses cache - cached TPM limits as ultimate fallback
  - Probe model configuration - custom probe model from config
  
- **test_llm_client_integration**: 10 active tests (3 skipped)
  - Simple response - basic API call verification
  - JSON response - structured output parsing
  - Response chain - context preservation with previous_response_id
  - TPM limiting - rate limit control with artificial throttling
  - Error handling - authentication and API errors
  - Reasoning model - o* models with reasoning tokens
  - Background mode verification - console output capture

- **test_llm_client_integration_chain**: 6 tests (require API key)
  - Response chain window - sliding window management
  - Independent requests - no context preservation
  - Repair not in chain - repair responses excluded
  - Truncation parameter - automatic context truncation
  - Chain management with confirmations - proper deletion of old responses
  - Two-phase confirmation - proper chain management

## Configuration Notes

### Response Chain Management
The `response_chain_depth` parameter controls how previous responses are managed:
- **None** (default) - Unlimited chain, all responses kept (current behavior)
- **0** - Independent requests, previous_response_id always None
- **>0** - Sliding window of N responses, older ones deleted via API

When using sliding window mode:
1. Response IDs added to deque after successful confirmation
2. When chain exceeds depth, oldest response deleted via API
3. Deletion failures logged as WARNING but don't stop execution
4. Repair responses (is_repair=True) NOT added to chain

### Two-Phase Confirmation Pattern
The client implements a two-phase confirmation pattern for robust error recovery:

1. **Response Phase**: `create_response()` returns a response but doesn't finalize chain management
   - Response saved as `unconfirmed_response_id`
   - Chain not updated yet
   - Previous unconfirmed responses auto-cleared

2. **Confirmation Phase**: `confirm_response()` finalizes the successful response
   - Updates `last_confirmed_response_id`
   - Adds to response chain (if configured)
   - Manages sliding window deletions

3. **Repair Phase**: `repair_response()` uses last confirmed context
   - Automatically uses `last_confirmed_response_id` as context
   - Repair responses not added to chain
   - No confirmation needed for repairs

This pattern ensures that invalid responses don't break the context chain and allows safe rollback to the last known good state.

### Test Model Parameters
The client does NOT handle `tpm_limit_test` or `max_completion_test` parameters directly. These are utility-level configuration options used by itext2kg and refiner utilities when deciding between main and test models. The OpenAIClient receives already selected model parameters.

### Required vs Optional Parameters
- **is_reasoning** (bool) - REQUIRED parameter that must be explicitly provided
- **temperature**, **reasoning_effort**, **reasoning_summary**, **verbosity**, **truncation** - optional parameters sent to API only when not None
- **response_chain_depth** - optional, controls chain management mode

## Dependencies
- **Standard Library**: time, logging, json, typing, dataclasses, datetime, collections
- **External**: openai>=1.0.0, tiktoken
- **Internal**: None

## Performance Notes

### Context Accumulation
- When using previous_response_id, the client tracks accumulated context size
- **Smart token calculation**: 
  - Uses last_usage.input_tokens as base (includes all previous context)
  - Adds only new content tokens to estimate total
  - Warns when approaching max_context_tokens limit
- OpenAI automatically truncates oldest content when exceeding the limit
- TPM calculations now account for full context size, not just new content
- Context limits by model:
  - **o1/o4 models**: 200K tokens context window
  - **gpt-4 models**: 128K tokens context window

### Asynchronous Mode
- All requests executed in background mode for timeout control
- Polling with adaptive interval:
  - First 3 checks: every 2 seconds (fast response)
  - Then: use poll_interval from config
- On timeout, attempt to cancel request

### Progress Display
- **Optimized for readability**: progress shown every 3 polling iterations
- **Time formatting**: below 60 seconds shown as "42s", after as "2m 15s"
- **Single intermediate status**: only queued, no intermediate token information

### TPM Probe Mechanism
- **IMPORTANT**: OpenAI doesn't return rate limit headers in background mode
- Probe request to cheap gpt-4.1-nano model before main request
- Overhead: ~20 tokens (0.02% of typical limit)

### Console Progress Output
- [QUEUE] ⏳ - request in queue (first time only)
- [PROGRESS] ⏳ - waiting progress with time (every 3 checks)
- [ERROR] ❌ - generation errors
- [RETRY] ⏳ - waiting before retry
- [RETRY] 🔄 - token limit increase on incomplete
- [HINT] 💡 - hints for reasoning models
- [INFO] ✅ - TPM limit reset

## Usage Examples

### Basic Usage
```python
from src.utils.llm_client import OpenAIClient

config = {
    'api_key': 'sk-...',
    'model': 'gpt-4.1-mini-2025-04-14',
    'is_reasoning': False,  # REQUIRED: explicitly specify model type
    'tpm_limit': 120000,
    'tpm_safety_margin': 0.15,
    'max_completion': 4096,
    'max_context_tokens': 128000,
    'timeout': 45,
    'max_retries': 6,
    'temperature': 0.7,
    'poll_interval': 7
}

client = OpenAIClient(config)
response_text, response_id, usage = client.create_response(
    "You are a helpful assistant",
    "What is the capital of France?"
)
client.confirm_response()  # Confirm successful response
```

### Request Chain with Context
```python
# First request
text1, id1, usage1 = client.create_response(
    "You are a math tutor",
    "My name is Alice. What is 5 + 3?"
)
client.confirm_response()

# Second request remembers context
text2, id2, usage2 = client.create_response(
    "Continue being a math tutor",
    "What was my name?",
    previous_response_id=id1
)
client.confirm_response()
# Response will contain "Alice"

# Check accumulated context
print(f"Context tokens accumulated: {usage2.input_tokens}")
```

### Working with Reasoning Models
```python
config = {
    'api_key': 'sk-...',
    'model': 'o4-mini-2025-04-16',
    'is_reasoning': True,  # REQUIRED: explicitly specify
    'tpm_limit': 100000,
    'max_completion': 25000,
    'max_context_tokens': 200000,
    'reasoning_effort': 'medium',
    'reasoning_summary': 'auto',
}

client = OpenAIClient(config)
response_text, _, usage = client.create_response(
    "Solve step by step",
    "What is 123 * 456?"
)
client.confirm_response()
print(f"Reasoning tokens: {usage.reasoning_tokens}")
```

### JSON Response Handling with Repair
```python
import json

# Process multiple JSON responses with validation
for data in json_data_list:
    try:
        # Get response
        response_text, response_id, _ = client.create_response(
            "Return a JSON object",
            data
        )
        
        # Validate JSON
        result = json.loads(response_text)
        
        # Confirm on success
        client.confirm_response()
        
    except json.JSONDecodeError:
        # Repair with automatic rollback to last confirmed
        repair_instructions = (
            "Return a JSON object\n\n"
            "CRITICAL: Return ONLY valid JSON."
        )
        response_text, _, _ = client.repair_response(
            instructions=repair_instructions,
            input_data=data
        )
        result = json.loads(response_text)
        # No confirmation needed for repair
```

### Error Handling
```python
try:
    response_text, response_id, usage = client.create_response(
        instructions="Complex task",
        input_data="Generate very long text..."
    )
    # Validate response here
    client.confirm_response()
    
except IncompleteResponseError as e:
    print(f"Response was incomplete: {e}")
    # Consider increasing max_completion or using truncation='auto'
    
except TimeoutError as e:
    print(f"Request timed out: {e}")
    
except openai.RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
```

### TPM and Context Monitoring
```python
# Check current TPM state
print(f"TPM remaining: {client.tpm_bucket.remaining_tokens}")
print(f"TPM reset at: {client.tpm_bucket.reset_time}")

# Monitor context accumulation
response_text, _, usage = client.create_response(...)
client.confirm_response()
print(f"Input tokens (with context): {usage.input_tokens}")

# Check if approaching context limit
if client.last_usage and client.last_usage.input_tokens > 100000:
    print("Warning: Context size is large, may be truncated soon")
```
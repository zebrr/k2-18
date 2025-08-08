# util_llm_client.md

## Status: READY

Client for working with OpenAI Responses API with support for asynchronous mode, token limit control, response chains, and reasoning models.

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
  - model (str) - model (gpt-4o, o4-mini, etc.)
  - tpm_limit (int) - tokens per minute limit
  - tpm_safety_margin (float) - safety margin (default 0.15)
  - max_completion (int) - maximum tokens for generation
  - max_context_tokens (int) - maximum context window size (default 128000)
  - timeout (int) - request timeout in seconds
  - max_retries (int) - number of retry attempts
  - temperature (float, optional) - for regular models
  - reasoning_effort (str, optional) - for reasoning models
  - reasoning_summary (str, optional) - summary type for reasoning
  - poll_interval (int) - polling interval in seconds (default 5)
- **Attributes**:
  - last_response_id (str) - ID of last successful response for chains
  - last_usage (ResponseUsage) - usage info from last response for context accumulation
  - encoder - tokenizer (tiktoken with o200k_base) for precise token counting

#### OpenAIClient.create_response(instructions: str, input_data: str, previous_response_id: Optional[str] = None) -> Tuple[str, str, ResponseUsage]
Create response via OpenAI Responses API (public interface).
- **Input**: 
  - instructions - system prompt
  - input_data - user data
  - previous_response_id - ID of previous response (optional)
- **Returns**: (response_text, response_id, usage_info)
- **Raises**: 
  - TimeoutError - when timeout exceeded
  - IncompleteResponseError - for incomplete status
  - ValueError - for failed status or model refusal
  - openai.RateLimitError - when rate limit exceeded

#### OpenAIClient.repair_response(instructions: str, input_data: str, previous_response_id: Optional[str] = None) -> Tuple[str, str, ResponseUsage]
Repair request with specified previous_response_id (transport layer).
- **Input**: 
  - instructions - system prompt (caller should include repair instructions)
  - input_data - user data
  - previous_response_id - ID of previous response (optional, uses last_response_id if None)
- **Returns**: (response_text, response_id, usage_info)
- **Logic**: Pure transport layer - simply delegates to create_response with provided previous_response_id

## Internal Methods

### OpenAIClient._update_tpm_via_probe() -> None
Get current rate limit data via probe request.
- **Logic**: Synchronous request to gpt-4.1-nano with "2+2=?" to get headers
- **Note**: Necessary because OpenAI doesn't return rate limit headers in background mode

### OpenAIClient._create_response_async(instructions, input_data, previous_response_id) -> Tuple[str, str, ResponseUsage]
Main logic for creating response in asynchronous mode.
- **Token Calculation**:
  1. If previous_response_id exists and last_usage available:
     - Use last_usage.input_tokens as base (includes all accumulated context)
     - Add only new content tokens
     - Check against max_context_tokens limit
  2. Otherwise: count tokens in full prompt (instructions + input_data)
- **Steps**:
  1. TPM probe to update limits
  2. Create background request (background=true)
  3. Polling loop with status check
  4. Handle statuses: completed, incomplete, failed, cancelled, queued
  5. Automatic token increase on incomplete (×1.5, ×2.0)
  6. Cancel on timeout with cancellation attempt
- **Progress Display**: 
  - Queued status shown at first detection
  - Progress displayed every 3 polling iterations with elapsed time

### OpenAIClient._prepare_request_params(instructions, input_data, previous_response_id) -> Dict[str, Any]
Prepare parameters for Responses API.
- **Logic**: 
  - Base parameters + store=true
  - For reasoning models: reasoning parameters, no temperature
  - For regular models: temperature, no reasoning

### OpenAIClient._extract_response_content(response) -> str
Extract text from response object.
- **Logic**:
  - Reasoning models: output[0]=reasoning, output[1]=message
  - Regular models: output[0]=message
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

## Test Coverage

- **test_llm_client**: 23 tests
  - test_initialization
  - test_tpm_bucket_*
  - test_prepare_request_params_*
  - test_extract_response_content_*
  - test_create_response_async_*
  - test_retry_logic_*
  - test_incomplete_handling
  - test_timeout_handling
  
- **test_llm_client_integration**: 13 tests
  - test_simple_response
  - test_json_response
  - test_response_chain
  - test_tpm_limiting
  - test_error_handling
  - test_reasoning_model
  - test_headers_update
  - test_background_mode_verification
  - test_incomplete_response_handling
  - test_timeout_cancellation
  - test_console_progress_output
  - test_incomplete_with_reasoning_model
  - test_tpm_probe_mechanism

## Dependencies
- **Standard Library**: time, logging, json, typing, dataclasses, datetime
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
- Code has commented attempts to update TPM from async headers

### Incomplete Handling
- Automatic max_output_tokens increase on retry:
  - 1st retry: ×1.5 from original
  - 2nd retry: ×2.0 from original
  - 3rd retry: critical error
- **Bug fixed**: old_limit variable now correctly defined before use

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
    'tpm_limit': 120000,
    'tpm_safety_margin': 0.15,
    'max_completion': 4096,
    'max_context_tokens': 128000,  # Optional, default 128000
    'timeout': 45,
    'max_retries': 6,
    'temperature': 0.7,
    'poll_interval': 7
}

client = OpenAIClient(config)

# Simple request
response_text, response_id, usage = client.create_response(
    "You are a helpful assistant",
    "What is the capital of France?"
)
print(f"Response: {response_text}")
print(f"Tokens used: {usage.total_tokens}")
```

### Request Chain with Context
```python
# First request
text1, id1, usage1 = client.create_response(
    "You are a math tutor",
    "My name is Alice. What is 5 + 3?"
)

# Second request remembers context
text2, id2, usage2 = client.create_response(
    "Continue being a math tutor",
    "What was my name?",
    previous_response_id=id1  # Explicit context passing
)
# Response will contain "Alice"

# Check accumulated context
print(f"Context tokens accumulated: {usage2.input_tokens}")
```

### Working with Reasoning Models
```python
config = {
    'api_key': 'sk-...',
    'model': 'o4-mini-2025-04-16',  # Reasoning model
    'tpm_limit': 100000,
    'max_completion': 25000,  # More tokens for reasoning
    'max_context_tokens': 200000,  # o4 models have larger context
    'reasoning_effort': 'medium',
    'reasoning_summary': 'auto',
    # don't specify temperature for reasoning models!
}

client = OpenAIClient(config)
response_text, _, usage = client.create_response(
    "Solve step by step",
    "What is 123 * 456?"
)
print(f"Reasoning tokens: {usage.reasoning_tokens}")
```

### JSON Response Handling with Repair
```python
import json

# Track successful response for potential rollback
last_successful_id = None

try:
    response_text, response_id, _ = client.create_response(
        "Return a JSON object",
        "Create user object with name and age",
        previous_response_id=last_successful_id
    )
    data = json.loads(response_text)
    last_successful_id = response_id  # Update on success
except json.JSONDecodeError:
    # Repair with rollback to last successful context
    repair_instructions = (
        "Return a JSON object\n\n"
        "CRITICAL: Return ONLY valid JSON. "
        "No markdown formatting, no explanations."
    )
    response_text, response_id, _ = client.repair_response(
        instructions=repair_instructions,
        input_data="Create user object with name and age",
        previous_response_id=last_successful_id  # Rollback!
    )
    data = json.loads(response_text)
    last_successful_id = response_id  # Update on success
```

### Error Handling
```python
try:
    response_text, response_id, usage = client.create_response(
        instructions="Complex task",
        input_data="Generate very long text..."
    )
except IncompleteResponseError as e:
    print(f"Response was incomplete: {e}")
    # Can retry with larger max_completion
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
print(f"Input tokens (with context): {usage.input_tokens}")
print(f"Output tokens: {usage.output_tokens}")
print(f"Total tokens: {usage.total_tokens}")

# Check if approaching context limit
if client.last_usage and client.last_usage.input_tokens > 100000:
    print("Warning: Context size is large, may be truncated soon")
```
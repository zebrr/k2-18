# util_llm_embeddings.md

## Status: READY

Module for working with OpenAI Embeddings API. Provides functions for obtaining vector representations of text and calculating cosine similarity between vectors.

## Public API

### EmbeddingsClient
Client for working with OpenAI Embeddings API with support for batch processing and TPM limit control.

#### EmbeddingsClient.__init__(config: Dict[str, Any])
Client initialization.
- **Input**: config - configuration dictionary with keys:
  - embedding_api_key (str, optional) - API key for embeddings (fallback to api_key)
  - api_key (str) - main API key
  - embedding_model (str) - model for embeddings (default "text-embedding-3-small")
  - embedding_tpm_limit (int) - TPM limit (default 350000)
  - max_retries (int) - number of retry attempts (default 3)
- **Raises**: ValueError - if API key not found

#### EmbeddingsClient.get_embeddings(texts: List[str]) -> np.ndarray
Get embeddings for a list of texts.
- **Input**: texts - list of texts to process
- **Returns**: numpy array shape (n_texts, 1536) with normalized vectors
- **Terminal Output**: Shows progress and status when processing multiple batches
- **Features**: 
  - Automatic batching (up to 2048 texts per request)
  - Truncation of texts longer than 8192 tokens
  - TPM limit control with waiting
  - Retry logic with exponential backoff
  - **Empty string handling**: empty strings (after strip()) are not sent to API, zero vectors are returned for them
  - **Order preservation**: original text order is preserved in results
- **Raises**: Exception - on API errors after all retry attempts

### get_embeddings(texts: List[str], config: Dict[str, Any]) -> np.ndarray
Simple wrapper for getting embeddings.
- **Input**: 
  - texts - list of texts
  - config - configuration (must contain embedding_api_key or api_key)
- **Returns**: numpy array with embeddings

### cosine_similarity_batch(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray
Calculate cosine similarity between two sets of vectors.
- **Input**: 
  - embeddings1 - vector array shape (n1, dim)
  - embeddings2 - vector array shape (n2, dim)
- **Returns**: cosine similarity array shape (n1, n2)
- **Note**: Vectors must be normalized (OpenAI API returns normalized)

## Internal Methods

#### _count_tokens(text: str) -> int
Count tokens in text using cl100k_base tokenizer.

#### _truncate_text(text: str, max_tokens: int = 8000) -> str
Truncate text to specified number of tokens.

#### _update_tpm_state(tokens_used: int, headers: Optional[Dict[str, str]] = None)
Update TPM bucket state from response headers or by simple subtraction.

#### _wait_for_tokens(required_tokens: int, safety_margin: float = 0.15)
Wait for token availability with safety margin.
- **Terminal Output**: Informs user about waiting and limit reset

#### _batch_texts(texts: List[str]) -> List[List[str]]
Split texts into batches considering API limits (2048 texts, ~100K tokens).
- **Terminal Output**: Warns about truncation of texts longer than 8192 tokens

## Terminal Output

Module outputs processing progress information to terminal with format `[HH:MM:SS] EMBEDDINGS | message`:

### Batch Processing
- Processing start information (when >1 batch):
  ```
  [10:30:00] EMBEDDINGS | Processing 457 texts in 5 batches...
  ```

- Batch progress (when >1 batch):
  ```
  [10:30:05] EMBEDDINGS | ✅ Batch 1/5 completed
  [10:30:10] EMBEDDINGS | ✅ Batch 2/5 completed
  ```

- Final message (when >1 batch):
  ```
  [10:31:08] EMBEDDINGS | ✅ Completed: 457 texts processed
  ```

### Text Processing
- Warning about long text truncation:
  ```
  [10:30:01] EMBEDDINGS | ⚠️ Text truncated: 9234 → 8000 tokens
  ```

### TPM Control
- Waiting for limit reset:
  ```
  [10:30:12] EMBEDDINGS | ⏳ Waiting 42.3s for TPM limit reset...
  [10:30:55] EMBEDDINGS | ✅ TPM limit reset, continuing...
  ```

### Error Handling
- Rate limit with retry:
  ```
  [10:31:20] EMBEDDINGS | ⏳ Rate limit hit, retry 1/3 in 10s...
  ```

**Note**: Terminal output only occurs for operations with multiple batches. When processing small amounts of text (1 batch), the module works "silently".

## Test Coverage

### Unit Tests (test_llm_embeddings.py)
- **TestEmbeddingsClient**: 14 tests
  - test_init_with_embedding_api_key
  - test_init_fallback_to_api_key
  - test_init_no_api_key
  - test_count_tokens
  - test_truncate_text
  - test_batch_texts
  - test_batch_texts_with_long_text
  - test_update_tpm_state_with_headers
  - test_update_tpm_state_without_headers
  - test_wait_for_tokens
  - test_get_embeddings_success
  - test_get_embeddings_empty_input
  - test_get_embeddings_rate_limit_retry
  - test_get_embeddings_max_retries_exceeded

- **TestHelperFunctions**: 3 tests
  - test_get_embeddings_wrapper
  - test_cosine_similarity_batch
  - test_cosine_similarity_batch_normalized

### Integration Tests (test_llm_embeddings_integration.py)
- **TestSingleEmbedding**: 2 tests (single texts)
- **TestBatchProcessing**: 3 tests (batches of different sizes)
- **TestLongTexts**: 3 tests (long text processing)
- **TestCosineSimilarity**: 4 tests (similarity verification)
- **TestEdgeCases**: 5 tests (empty strings, special characters)
- **TestTPMLimits**: 2 tests (limit control)
- **TestErrorHandling**: 2 tests (error handling)
- **TestVectorProperties**: 2 tests (vector properties)
- **TestPerformance**: 1 test (performance)

## Dependencies

- **Standard Library**: time, logging
- **External**: openai, tiktoken, numpy
- **Internal**: None

## Configuration

Module uses the following configuration parameters:
- `embedding_api_key` - API key for embeddings (optional)
- `api_key` - main API key (fallback)
- `embedding_model` - OpenAI model for embeddings
- `embedding_tpm_limit` - tokens per minute limit (350000 for embedding models)
- `max_retries` - number of retry attempts on errors

## Performance Notes

- **Tokenizer**: Uses cl100k_base (NOT o200k_base!) as required by OpenAI Embeddings API
- **Batching**: Automatic batching for optimal performance
- **TPM control**: Built-in limit control with wait for reset
- **Dimensions**: text-embedding-3-small returns vectors of dimension 1536
- **Normalization**: Vectors are automatically normalized (norm = 1)
- **Empty strings**: OpenAI API doesn't accept empty strings, module automatically filters them and returns zero vectors

## Usage Examples

```python
from utils import load_config, get_embeddings, cosine_similarity_batch

# Load configuration
config = load_config()

# Get embeddings for texts
texts = ["Python programming", "Machine learning", "Data science"]
embeddings = get_embeddings(texts, config['dedup'])

# Calculate pairwise cosine similarity
similarity_matrix = cosine_similarity_batch(embeddings, embeddings)
print(f"Similarity between text 0 and 1: {similarity_matrix[0, 1]:.3f}")

# Using the class directly for finer control
from utils.llm_embeddings import EmbeddingsClient

client = EmbeddingsClient(config['dedup'])
# Processing large number of texts
large_text_list = ["text"] * 5000  # Will be automatically split into batches
embeddings = client.get_embeddings(large_text_list)

# Processing texts with empty strings
mixed_texts = ["Hello", "", "World", "   ", "!"]
embeddings = client.get_embeddings(mixed_texts)
# Result: shape (5, 1536), where embeddings[1] and embeddings[3] are zero vectors
```

## Error Handling

```python
try:
    embeddings = get_embeddings(texts, config)
except ValueError as e:
    # Configuration error (no API key)
    print(f"Config error: {e}")
except Exception as e:
    # API errors after all retry attempts
    if "rate_limit" in str(e).lower():
        print("Rate limit exceeded, try again later")
    else:
        print(f"API error: {e}")
```

## Edge Cases

- **Empty strings**: Automatically replaced with zero vectors without API call
- **Too long texts**: Truncated to 8000 tokens with warning logging
- **Empty input array**: Returns empty numpy array shape (0,)
- **Mixed languages**: Supported, model is multilingual
- **Special characters and emoji**: Processed correctly
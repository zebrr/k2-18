# util_config.md

## Status: READY

Module for loading and validating iText2KG configuration from TOML file. Supports Python 3.11+ (tomllib) and earlier versions (tomli).

## Public API

### load_config(config_path: Union[str, Path] = None) -> Dict[str, Any]
Loads and validates configuration from TOML file.
- **Input**: config_path (optional) - path to configuration file, defaults to src/config.toml
- **Returns**: Dict[str, Any] - dictionary with validated configuration
- **Raises**: 
  - ConfigValidationError - on validation errors
  - FileNotFoundError - if configuration file not found

Automatically injects API keys from environment variables if config values are empty or placeholders (starting with "sk-...").

### ConfigValidationError(Exception)
Exception for configuration validation errors.

## Internal Functions

### _inject_env_api_keys(config: Dict[str, Any]) -> None
Injects API keys from environment variables. Priority order:
1. Environment variable (if set)
2. Value from config.toml (if not placeholder)
3. Validation error

Environment variables:
- `OPENAI_API_KEY` - for itext2kg.api_key and refiner.api_key
- `OPENAI_EMBEDDING_API_KEY` - for embedding API keys (fallback to OPENAI_API_KEY)

## Configuration Sections

### [slicer]
- max_tokens (int, >0) - window size in tokens
- overlap (int, ≥0) - overlap between slices
- soft_boundary (bool) - use soft boundaries
- soft_boundary_max_shift (int, ≥0) - maximum shift for soft boundaries
- tokenizer (str, ="o200k_base") - tokenizer model
- allowed_extensions (list, non-empty) - allowed file extensions

### [itext2kg]
- model (str) - LLM model
- tpm_limit (int, >0) - tokens per minute limit
- max_completion (int, 1-100000) - maximum generated tokens
- log_level (str, debug/info/warning/error) - logging level
- api_key (str, non-empty) - OpenAI API key
- timeout (int, >0) - request timeout in seconds
- max_retries (int, ≥0) - number of retries
- poll_interval (int, >0) - status check interval in seconds (for async mode)

Additional parameters not validated in code:
- tpm_safety_margin - safety margin for TPM calculations
- temperature - generation temperature (for non-reasoning models)
- reasoning_effort - reasoning effort level (for o-models)
- reasoning_summary - reasoning summary format (for o-models)

### [dedup]
- embedding_model (str) - embedding model
- embedding_api_key (str, optional) - API key for embeddings (uses env vars if not set)
- sim_threshold (float, 0.0-1.0) - similarity threshold
- len_ratio_min (float, 0.0-1.0) - minimum length ratio
- faiss_M (int, >0) - HNSW graph parameter
- faiss_efC (int, >0) - HNSW construction parameter
- faiss_metric (str, INNER_PRODUCT/L2) - FAISS metric
- k_neighbors (int, >0) - number of neighbors

Additional parameters not validated:
- embedding_tpm_limit - TPM limit for embedding models

### [refiner]
- run (bool) - whether to run refiner
- embedding_model (str) - embedding model
- embedding_api_key (str, optional) - API key for embeddings (uses env vars if not set)
- sim_threshold (float, 0.0-1.0) - similarity threshold
- max_pairs_per_node (int, >0) - maximum pairs per node
- model (str) - LLM model
- api_key (str, non-empty) - OpenAI API key
- tpm_limit (int, >0) - tokens per minute limit
- max_completion (int, 1-100000) - maximum generated tokens
- timeout (int, >0) - request timeout
- max_retries (int, ≥0) - number of retries
- poll_interval (int, >0) - status check interval in seconds (for async mode)
- weight_low (float, 0.0-1.0) - low connection weight
- weight_mid (float, 0.0-1.0) - medium connection weight
- weight_high (float, 0.0-1.0) - high connection weight

Additional parameters not validated in code:
- tpm_safety_margin - safety margin for TPM calculations
- temperature - generation temperature (for non-reasoning models)
- reasoning_effort - reasoning effort level (for o-models)
- reasoning_summary - reasoning summary format (for o-models)
- embedding_tpm_limit - TPM limit for embedding models

## Validation Rules

- All sections are required: [slicer], [itext2kg], [dedup], [refiner]
- When overlap > 0: soft_boundary_max_shift ≤ overlap * 0.8
- Weights must satisfy: weight_low < weight_mid < weight_high
- API keys cannot be empty (including whitespace) unless provided via environment variables
- Strict type checking for all fields
- Placeholder keys (starting with "sk-...") trigger environment variable lookup

## Test Coverage

- **test_config_loading**: 3 tests
  - test_load_valid_config
  - test_missing_config_file
  - test_invalid_toml_syntax

- **test_slicer_validation**: 3 tests
  - test_missing_slicer_section
  - test_invalid_max_tokens
  - test_overlap_soft_boundary_validation

- **test_itext2kg_validation**: 2 tests
  - test_invalid_log_level
  - test_empty_api_key

- **test_refiner_validation**: 2 tests
  - test_invalid_weight_order
  - test_weight_out_of_range

- **test_type_validation**: 1 test
  - test_wrong_type_validation

## Dependencies
- **Standard Library**: os, sys, pathlib, typing
- **External**: tomllib (Python 3.11+), tomli (Python <3.11)
- **Internal**: None

## Usage Examples
```python
from src.utils.config import load_config, ConfigValidationError

# Load with default path
try:
    config = load_config()
    slicer_config = config["slicer"]
    max_tokens = slicer_config["max_tokens"]
except ConfigValidationError as e:
    print(f"Configuration error: {e}")

# Load with custom path
config = load_config("custom_config.toml")

# Environment variables override
# Set OPENAI_API_KEY environment variable to avoid storing keys in config
os.environ["OPENAI_API_KEY"] = "your-api-key"
config = load_config()  # Will use env var for api_key fields
```

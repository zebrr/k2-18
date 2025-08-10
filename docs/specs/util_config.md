# util_config.md

## Status: READY

Module for loading and validating iText2KG configuration from TOML file. Supports Python 3.11+ (tomllib) and earlier versions (tomli).

## Public API

### load_config(config_path: Union[str, Path] = None) -> Dict[str, Any]
Loads and validates configuration from TOML file.
- **Input**: 
  - config_path (Optional[Union[str, Path]]) - path to configuration file, defaults to src/config.toml
- **Returns**: Dict[str, Any] - dictionary with validated configuration
- **Raises**: 
  - ConfigValidationError - on validation errors
  - FileNotFoundError - if configuration file not found
- **Side effects**: 
  - Automatically injects API keys from environment variables if config values are empty or placeholders (starting with "sk-...")
  - Logs warnings for consistency issues (reasoning model with temperature, non-reasoning with reasoning_effort)

### ConfigValidationError(Exception)
Exception for configuration validation errors.
- **Usage**: Raised when configuration fails validation with descriptive error message

## Internal Methods

### _inject_env_api_keys(config: Dict[str, Any]) -> None
Injects API keys from environment variables into configuration. Priority order:
1. Environment variable (if set)
2. Value from config.toml (if not placeholder)
3. Validation error if neither available

Environment variables:
- `OPENAI_API_KEY` - for itext2kg.api_key and refiner.api_key
- `OPENAI_EMBEDDING_API_KEY` - for embedding API keys (fallback to OPENAI_API_KEY if not set)

### _validate_config(config: Dict[str, Any]) -> None
Validates complete configuration structure.
- **Logic**: 
  - Checks presence of all required sections: [slicer], [itext2kg], [dedup], [refiner]
  - Delegates to section-specific validators
  - Validates is_reasoning parameter presence in itext2kg and refiner
  - Logs consistency warnings for parameter combinations

### _validate_slicer_section(section: Dict[str, Any]) -> None
Validates [slicer] section parameters.
- **Checks**:
  - Required fields presence and types
  - max_tokens > 0
  - overlap >= 0
  - soft_boundary_max_shift >= 0
  - When overlap > 0: soft_boundary_max_shift <= overlap * 0.8
  - tokenizer must be "o200k_base"
  - allowed_extensions not empty

### _validate_itext2kg_section(section: Dict[str, Any]) -> None
Validates [itext2kg] section parameters.
- **Checks**:
  - Required fields presence and types
  - tpm_limit > 0
  - max_completion between 1 and 100000
  - log_level in ["debug", "info", "warning", "error"]
  - api_key not empty or available via environment
  - timeout > 0
  - max_retries >= 0
  - response_chain_depth >= 0 (if present)
  - truncation in ["auto", "disabled"] (if present)

### _validate_dedup_section(section: Dict[str, Any]) -> None
Validates [dedup] section parameters.
- **Checks**:
  - Required fields presence and types
  - sim_threshold between 0.0 and 1.0
  - len_ratio_min between 0.0 and 1.0
  - faiss_M > 0
  - faiss_efC > 0
  - faiss_metric in ["INNER_PRODUCT", "L2"]
  - k_neighbors > 0
  - embedding_api_key available (optional, checks env vars)

### _validate_refiner_section(section: Dict[str, Any]) -> None
Validates [refiner] section parameters.
- **Checks**:
  - Required fields presence and types
  - run is boolean
  - sim_threshold between 0.0 and 1.0
  - max_pairs_per_node > 0
  - api_key not empty or available via environment
  - tpm_limit > 0
  - max_completion between 1 and 100000
  - timeout > 0
  - max_retries >= 0
  - response_chain_depth >= 0 (if present)
  - truncation in ["auto", "disabled"] (if present)
  - Weights validation: 0.0 <= weight_low < weight_mid < weight_high <= 1.0

### _validate_required_fields(section: Dict[str, Any], required_fields: Dict[str, type], section_name: str) -> None
Generic validator for required fields presence and types.
- **Input**:
  - section - configuration section to validate
  - required_fields - dictionary of field_name -> expected_type
  - section_name - section name for error messages
- **Raises**: ConfigValidationError with specific field and type information

## Configuration

### [slicer] - Required section
**Validated parameters:**
- **max_tokens** (int, >0) - window size in tokens
- **overlap** (int, ≥0) - overlap between slices
- **soft_boundary** (bool) - use soft boundaries
- **soft_boundary_max_shift** (int, ≥0) - maximum shift for soft boundaries
- **tokenizer** (str, ="o200k_base") - tokenizer model
- **allowed_extensions** (list, non-empty) - allowed file extensions

### [itext2kg] - Required section
**Validated parameters:**
- **is_reasoning** (bool, REQUIRED) - whether model is a reasoning model
- **model** (str) - LLM model name
- **tpm_limit** (int, >0) - tokens per minute limit
- **max_completion** (int, 1-100000) - maximum generated tokens
- **log_level** (str, debug/info/warning/error) - logging level
- **api_key** (str, non-empty) - OpenAI API key (or via OPENAI_API_KEY env)
- **timeout** (int, >0) - request timeout in seconds
- **max_retries** (int, ≥0) - number of retries
- **response_chain_depth** (int, ≥0, optional) - depth of response chain (0=independent requests, 1-N=chain depth, not set=unlimited)
- **truncation** (str, "auto"/"disabled", optional) - truncation strategy for context overflow

**Non-validated parameters (passed through to LLM client):**
- **poll_interval** (int, >0) - status check interval in seconds for async mode
- **tpm_safety_margin** (float) - safety margin for TPM calculations
- **max_context_tokens** (int) - maximum context window size
- **temperature** (float, optional) - generation temperature (sent to API only if not None)
- **reasoning_effort** (str, optional) - reasoning effort level (sent to API only if not None)
- **reasoning_summary** (str, optional) - reasoning summary format (sent to API only if not None)
- **verbosity** (str, optional) - verbosity level for GPT-5 models (sent to API only if not None)
- **model_test** (str) - test model name
- **max_context_tokens_test** (int) - context limit for test model
- **tpm_limit_test** (int) - TPM limit for test model
- **max_completion_test** (int) - max completion for test model

### [dedup] - Required section
**Validated parameters:**
- **embedding_model** (str) - embedding model name
- **sim_threshold** (float, 0.0-1.0) - similarity threshold
- **len_ratio_min** (float, 0.0-1.0) - minimum length ratio
- **faiss_M** (int, >0) - HNSW graph parameter
- **faiss_efC** (int, >0) - HNSW construction parameter
- **faiss_metric** (str, INNER_PRODUCT/L2) - FAISS metric
- **k_neighbors** (int, >0) - number of neighbors

**Non-validated parameters:**
- **embedding_api_key** (str, optional) - API key for embeddings (uses OPENAI_EMBEDDING_API_KEY or OPENAI_API_KEY env if not set)
- **embedding_tpm_limit** (int) - TPM limit for embedding models

### [refiner] - Required section
**Validated parameters:**
- **is_reasoning** (bool, REQUIRED) - whether model is a reasoning model
- **run** (bool) - whether to run refiner
- **embedding_model** (str) - embedding model name
- **sim_threshold** (float, 0.0-1.0) - similarity threshold
- **max_pairs_per_node** (int, >0) - maximum pairs per node
- **model** (str) - LLM model name
- **api_key** (str, non-empty) - OpenAI API key (or via OPENAI_API_KEY env)
- **tpm_limit** (int, >0) - tokens per minute limit
- **max_completion** (int, 1-100000) - maximum generated tokens
- **timeout** (int, >0) - request timeout
- **max_retries** (int, ≥0) - number of retries
- **response_chain_depth** (int, ≥0, optional) - depth of response chain (0=independent requests, 1-N=chain depth, not set=unlimited)
- **truncation** (str, "auto"/"disabled", optional) - truncation strategy for context overflow
- **weight_low** (float, 0.0-1.0) - low connection weight
- **weight_mid** (float, 0.0-1.0) - medium connection weight
- **weight_high** (float, 0.0-1.0) - high connection weight

**Non-validated parameters (passed through to LLM client):**
- **embedding_api_key** (str, optional) - API key for embeddings (uses OPENAI_EMBEDDING_API_KEY or OPENAI_API_KEY env if not set)
- **poll_interval** (int, >0) - status check interval in seconds for async mode
- **tpm_safety_margin** (float) - safety margin for TPM calculations
- **max_context_tokens** (int) - maximum context window size
- **temperature** (float, optional) - generation temperature (sent to API only if not None)
- **reasoning_effort** (str, optional) - reasoning effort level (sent to API only if not None)
- **reasoning_summary** (str, optional) - reasoning summary format (sent to API only if not None)
- **verbosity** (str, optional) - verbosity level for GPT-5 models (sent to API only if not None)
- **embedding_tpm_limit** (int) - TPM limit for embedding models
- **faiss_M** (int) - HNSW graph parameter for similarity search
- **faiss_metric** (str) - FAISS metric for similarity search
- **model_test** (str) - test model name
- **max_context_tokens_test** (int) - context limit for test model
- **tpm_limit_test** (int) - TPM limit for test model
- **max_completion_test** (int) - max completion for test model
- **log_level** (str) - logging level for refiner

## Validation Rules

- **Required sections**: All four sections must be present: [slicer], [itext2kg], [dedup], [refiner]
- **is_reasoning parameter**: REQUIRED in [itext2kg] and [refiner] sections, raises ConfigValidationError if missing
- **Dependency validations**:
  - When overlap > 0: soft_boundary_max_shift must be ≤ overlap * 0.8
  - Weights must satisfy: weight_low < weight_mid < weight_high
- **API keys handling**:
  - Empty or placeholder keys (starting with "sk-...") trigger environment variable lookup
  - Falls back to environment variables: OPENAI_API_KEY, OPENAI_EMBEDDING_API_KEY
  - Raises ConfigValidationError if neither config nor env provides valid key
- **Type checking**: Strict type validation for all required fields
- **Consistency warnings** (logged but not errors):
  - Reasoning model (is_reasoning=true) with temperature parameter - might be ignored by API
  - Non-reasoning model (is_reasoning=false) with reasoning_effort - will be ignored by API

## Dependencies
- **Standard Library**: os, sys, pathlib, typing, logging
- **External**: tomllib (Python 3.11+), tomli (Python <3.11, optional)
- **Internal**: None

## Test Coverage

- **test_config_loading**: 3 tests
  - test_load_valid_config - loads and validates correct configuration
  - test_missing_config_file - handles missing file correctly
  - test_invalid_toml_syntax - handles malformed TOML

- **test_slicer_validation**: 3 tests
  - test_missing_slicer_section - validates section presence
  - test_invalid_max_tokens - validates positive values
  - test_overlap_soft_boundary_validation - validates dependency rules

- **test_itext2kg_validation**: 5 tests
  - test_invalid_log_level - validates enum values
  - test_empty_api_key - validates API key availability
  - test_missing_is_reasoning_parameter - validates required is_reasoning
  - test_consistency_warning_reasoning_with_temperature - warns for reasoning models with temperature
  - test_consistency_warning_non_reasoning_with_effort - warns for non-reasoning models with reasoning_effort

- **test_refiner_validation**: 3 tests
  - test_invalid_weight_order - validates weight ordering
  - test_weight_out_of_range - validates weight ranges
  - test_missing_is_reasoning_parameter - validates required is_reasoning

- **test_type_validation**: 1 test
  - test_wrong_type_validation - validates type checking

- **test_environment_variables**: 3 tests
  - test_api_key_from_environment - validates API key injection from env
  - test_embedding_api_key_from_environment - validates embedding key injection
  - test_placeholder_detection - validates placeholder key detection

- **test_dedup_validation**: 3 tests
  - test_embedding_model_required - validates required embedding_model
  - test_invalid_similarity_threshold - validates threshold range
  - test_invalid_faiss_parameters - validates positive FAISS parameters

- **test_config_integration**: 3 tests
  - test_load_real_config - integration test with real config.toml
  - test_config_file_exists - verifies config.toml location
  - test_config_accessibility_from_utils - tests import from other modules

## Usage Examples

### Basic Usage
```python
from src.utils.config import load_config, ConfigValidationError

# Load with default path
try:
    config = load_config()
    slicer_config = config["slicer"]
    max_tokens = slicer_config["max_tokens"]
except ConfigValidationError as e:
    print(f"Configuration error: {e}")
```

### Custom Configuration Path
```python
# Load with custom path
config = load_config("custom_config.toml")

# Access nested configuration
itext2kg_config = config["itext2kg"]
is_reasoning = itext2kg_config["is_reasoning"]
```

### Environment Variables Override
```python
import os

# Set API key via environment variable
os.environ["OPENAI_API_KEY"] = "sk-your-actual-key"
os.environ["OPENAI_EMBEDDING_API_KEY"] = "sk-embedding-key"

# Config will use env vars for empty/placeholder keys
config = load_config()  # Automatically uses env vars
```

### Working with is_reasoning Parameter
```python
config = load_config()

# Check model type for itext2kg
if config["itext2kg"]["is_reasoning"]:
    print("Using reasoning model, temperature will be ignored")
    # Reasoning models ignore temperature
else:
    print(f"Using regular model with temperature={config['itext2kg'].get('temperature', 1.0)}")

# Same for refiner
if config["refiner"]["is_reasoning"]:
    # Reasoning model configuration
    reasoning_effort = config["refiner"].get("reasoning_effort", "medium")
```

### Working with Context Management Parameters
```python
config = load_config()

# Check response chain depth for context management
itext2kg_config = config["itext2kg"]
chain_depth = itext2kg_config.get("response_chain_depth")

if chain_depth is not None:
    if chain_depth == 0:
        print("Independent requests mode - no context sharing")
    else:
        print(f"Chain mode with depth {chain_depth}")
else:
    print("Unlimited chain depth")

# Check truncation strategy
truncation = itext2kg_config.get("truncation")
if truncation == "auto":
    print("Will truncate context automatically if needed")
elif truncation == "disabled":
    print("No truncation - may fail if context too large")
```

### Error Handling
```python
try:
    config = load_config()
except FileNotFoundError as e:
    print(f"Config file not found: {e}")
    sys.exit(1)
except ConfigValidationError as e:
    # Detailed validation error
    if "is_reasoning" in str(e):
        print("Missing required is_reasoning parameter")
    elif "api_key" in str(e):
        print("API key not configured - set OPENAI_API_KEY environment variable")
    elif "response_chain_depth" in str(e):
        print("Invalid response_chain_depth - must be non-negative integer")
    elif "truncation" in str(e):
        print("Invalid truncation - must be 'auto' or 'disabled'")
    else:
        print(f"Config validation failed: {e}")
    sys.exit(1)
```

## Performance Notes

- Configuration loading is fast (~1ms for typical config)
- Validation overhead is minimal
- No caching - config is re-read and validated on each load_config() call
- Thread-safe - can be called from multiple threads

## Model Support

The configuration supports both reasoning and non-reasoning models from OpenAI:

**Reasoning models** (is_reasoning=true):
- gpt-5-2025-08-07 (400K context, 128K output)
- gpt-5-mini-2025-08-07 (400K context, 128K output)
- o4-mini-2025-04-16
- o3-2025-04-16

**Non-reasoning models** (is_reasoning=false):
- gpt-4.1-2025-04-14 (1M context, 32K output)
- gpt-4.1-mini-2025-04-14 (1M context, 32K output)
- gpt-4.1-nano-2025-04-14
- gpt-4o and variants

Choose models based on task complexity and budget constraints. Test models (_test suffix parameters) should match the type (reasoning/non-reasoning) of the main model.

# Task K2-18-CONFIG-SPLIT-001 Completion Report

## Summary
Successfully split the `[itext2kg]` configuration section into two independent sections:
- `[itext2kg_concepts]` — for `itext2kg_concepts.py` (concept extraction)
- `[itext2kg_graph]` — for `itext2kg_graph.py` (graph construction)

Both sections now support independent configuration allowing different LLM models, parameters, and settings for each processing stage.

## Changes Made

### Core Configuration
- **src/config.toml**: Split `[itext2kg]` into `[itext2kg_concepts]` and `[itext2kg_graph]` with identical parameters, except `auto_mentions_weight` which is only in `[itext2kg_graph]`

### Validation Module
- **src/utils/config.py**:
  - Updated `_inject_env_api_keys()` to inject API keys for both new sections
  - Updated `_validate_config()` with new `required_sections = ["slicer", "itext2kg_concepts", "itext2kg_graph", "dedup", "refiner"]`
  - Created `_validate_itext2kg_concepts_section()` — validates concepts-specific parameters
  - Created `_validate_itext2kg_graph_section()` — validates graph-specific parameters including `auto_mentions_weight` (0.0-1.0)
  - Removed old `_validate_itext2kg_section()`
  - Updated `is_reasoning` validation for both new sections
  - Updated consistency warnings loop

### Processing Modules
- **src/itext2kg_concepts.py**: Line 95 changed from `config["itext2kg"]` to `config["itext2kg_concepts"]`
- **src/itext2kg_graph.py**: Line 94 changed from `config["itext2kg"]` to `config["itext2kg_graph"]`

### Backup Files Created
- `src/config_backup_CONFIG-SPLIT-001.toml`
- `src/utils/config_backup_CONFIG-SPLIT-001.py`

### Test Files Updated
- **tests/test_config.py**: 85 tests — comprehensive updates to all config fixtures and assertions
- **tests/test_config_integration.py**: Updated required sections and config access
- **tests/test_llm_client.py**: Updated `test_config_has_test_parameters` for both new sections
- **tests/test_llm_client_integration.py**: Updated `integration_config` fixture to use `itext2kg_concepts`
- **tests/test_itext2kg_concepts.py**: Updated `mock_config` fixture
- **tests/test_itext2kg_graph.py**: Updated `sample_config` fixture
- **tests/test_itext2kg_graph_deduplication.py**: Updated config in processor fixture
- **tests/test_itext2kg_graph_postprocessing.py**: Updated config in processor fixture
- **tests/test_itext2kg_graph_timeout.py**: Updated config in processor fixture

### Specifications Updated
- **docs/specs/util_config.md**: Updated required sections, validation functions, examples
- **docs/specs/cli_itext2kg_concepts.md**: Changed `[itext2kg]` to `[itext2kg_concepts]`
- **docs/specs/cli_itext2kg_graph.md**: Changed `[itext2kg]` to `[itext2kg_graph]`

## Tests

### Result: PASS

**Full test suite (non-integration)**: 780 passed, 5 skipped, 50 deselected

**Integration tests**: 48 passed, 2 failed (pre-existing issue), 1 skipped
- test_llm_client_integration.py: 14 passed, 1 skipped
- test_llm_client_integration_chain.py: 7 passed
- test_llm_embeddings_integration.py: 27 passed, 2 failed (see note below)

**Note**: 2 failures in `test_llm_embeddings_integration.py` are **not related to config split**:
- `test_tpm_tracking` and `test_real_headers_tracking` fail because OpenAI API returns actual TPM limit (5M) via headers, which is higher than configured `embedding_tpm_limit: 1000000`. This is a pre-existing issue with TPM tracking logic.

**test_config.py**: 85 tests passed
- All section validation tests updated for both `itext2kg_concepts` and `itext2kg_graph`
- All parameterized tests updated with correct section names
- All environment variable injection tests pass

**test_config_integration.py**: All tests pass
- Updated `required_sections` to include both new sections
- Updated config access to use `itext2kg_concepts`

**test_llm_client.py**: All tests pass
- `test_config_has_test_parameters` updated for both new sections

**test_llm_client_integration.py**: All tests pass
- Updated `integration_config` fixture to use `itext2kg_concepts` section

**itext2kg tests**: 91 tests passed
- test_itext2kg_concepts.py: All tests pass
- test_itext2kg_graph.py: All tests pass
- test_itext2kg_graph_deduplication.py: All tests pass
- test_itext2kg_graph_postprocessing.py: All tests pass
- test_itext2kg_graph_timeout.py: All tests pass

## Quality Checks

- **ruff check**: PASS (pre-existing warnings in other files, not related to changes)
- **ruff format**: PASS (files reformatted)
- **mypy**: PASS (no new errors in config.py; pre-existing errors in imported modules)

## Issues Encountered

1. **Multiple test configs missing `[itext2kg_graph]` section**: Fixed by running Python script to add missing sections (18 sections added)

2. **Incorrect error message patterns**: Several tests expected old `itext2kg.` prefix in error messages, updated to `itext2kg_concepts.` or `itext2kg_graph.`

3. **Assertions using old section name**: Fixed `result["itext2kg"]` → `result["itext2kg_concepts"]` in 4 test assertions

4. **Test for missing is_reasoning in refiner**: Config was incorrectly modified by bulk replace, fixed by removing accidentally added `is_reasoning` parameter

5. **Integration tests missed in initial update**: `test_llm_client_integration.py` fixture used `config["itext2kg"]` — fixed to `config["itext2kg_concepts"]`

## Next Steps
None — task complete.

## Commit Proposal
```
feat: split [itext2kg] config into [itext2kg_concepts] and [itext2kg_graph]

- Separate configuration sections for concept extraction and graph construction
- Each section can now use different LLM models and parameters
- auto_mentions_weight parameter exclusive to itext2kg_graph section
- Updated all validation, tests, and specifications
```

## Specs Updated

1. **docs/specs/util_config.md**:
   - Updated required sections from 4 to 5
   - Added documentation for both new validation functions
   - Added `[itext2kg_concepts]` and `[itext2kg_graph]` section documentation
   - Updated all code examples

2. **docs/specs/cli_itext2kg_concepts.md**:
   - Configuration section reference updated to `[itext2kg_concepts]`

3. **docs/specs/cli_itext2kg_graph.md**:
   - Configuration section reference updated to `[itext2kg_graph]`

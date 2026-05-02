# util_exit_codes.md

## Status: READY

Standard exit codes module for all CLI utilities in the project. Provides consistent error handling through a unified code system with support for readable names and logging.

## Public API

### Constants

#### Exit Codes
- **EXIT_SUCCESS** = 0 - Successful execution
- **EXIT_CONFIG_ERROR** = 1 - Configuration errors (missing API key, broken config.toml)
- **EXIT_INPUT_ERROR** = 2 - Input data errors (empty files, broken JSON schemas)
- **EXIT_RUNTIME_ERROR** = 3 - Runtime errors (LLM failures, broken slice after repair)
- **EXIT_API_LIMIT_ERROR** = 4 - TPM limits, rate limits (require retry/waiting)
- **EXIT_IO_ERROR** = 5 - File write errors, directory access issues

#### Dictionaries
- **EXIT_CODE_NAMES** - Dictionary {code: name} for logging
- **EXIT_CODE_DESCRIPTIONS** - Dictionary {code: description} for documentation

### Functions

#### get_exit_code_name(code: int) -> str
Returns readable name for exit code.
- **Input**: code - exit code
- **Returns**: code name (e.g. "CONFIG_ERROR") or "UNKNOWN(code)" for unknown codes
- **Usage**: for creating readable logs and error messages

#### get_exit_code_description(code: int) -> str
Returns description for exit code.
- **Input**: code - exit code
- **Returns**: code description or "Unknown exit code: {code}"
- **Usage**: for detailed error messages and documentation

#### log_exit(logger, code: int, message: str = None) -> None
Logs exit code with optional message.
- **Input**: 
  - logger - logger object (logging.Logger)
  - code - exit code
  - message - additional message (optional)
- **Behavior**: 
  - SUCCESS is logged via logger.info()
  - All other codes via logger.error()
  - Includes code name and description in log

## Test Coverage

- **TestExitCodeConstants**: 3 tests
  - test_exit_code_values - verify constant values
  - test_exit_code_names_completeness - all codes have names
  - test_exit_code_descriptions_completeness - all codes have descriptions

- **TestGetExitCodeName**: 2 tests
  - test_valid_codes - get names for all codes
  - test_unknown_code - handle unknown codes

- **TestGetExitCodeDescription**: 2 tests
  - test_valid_codes - get descriptions for all codes
  - test_unknown_code - handle unknown codes

- **TestLogExit**: 5 tests
  - test_log_success_without_message
  - test_log_success_with_message
  - test_log_error_without_message
  - test_log_error_with_message
  - test_log_all_error_types

## Dependencies

- **Standard Library**: logging
- **External**: None
- **Internal**: None

## Usage Examples

### Basic usage in CLI utilities
```python
from utils.exit_codes import EXIT_SUCCESS, EXIT_CONFIG_ERROR, EXIT_INPUT_ERROR
import sys

# Successful completion
if all_processed:
    sys.exit(EXIT_SUCCESS)

# Configuration error
if not api_key:
    print("Error: API key not found in configuration", file=sys.stderr)
    sys.exit(EXIT_CONFIG_ERROR)

# Input data error
if not input_files:
    print("Error: No files to process", file=sys.stderr)
    sys.exit(EXIT_INPUT_ERROR)
```

### Usage with logging
```python
from utils.exit_codes import EXIT_SUCCESS, EXIT_RUNTIME_ERROR, log_exit
import logging

logger = logging.getLogger(__name__)

try:
    # Process data
    process_data()
    log_exit(logger, EXIT_SUCCESS, f"Processed {count} files")
    sys.exit(EXIT_SUCCESS)
except Exception as e:
    log_exit(logger, EXIT_RUNTIME_ERROR, f"Critical error: {e}")
    sys.exit(EXIT_RUNTIME_ERROR)
```

### Main function handling
```python
from utils.exit_codes import (
    EXIT_SUCCESS, EXIT_CONFIG_ERROR, EXIT_INPUT_ERROR,
    EXIT_RUNTIME_ERROR, EXIT_API_LIMIT_ERROR, EXIT_IO_ERROR,
    get_exit_code_name
)

def main():
    try:
        # Main logic
        return EXIT_SUCCESS
    except ConfigError:
        return EXIT_CONFIG_ERROR
    except InputError:
        return EXIT_INPUT_ERROR
    except RateLimitError:
        return EXIT_API_LIMIT_ERROR
    except IOError:
        return EXIT_IO_ERROR
    except Exception:
        return EXIT_RUNTIME_ERROR

if __name__ == "__main__":
    exit_code = main()
    if exit_code != EXIT_SUCCESS:
        print(f"Exited with error: {get_exit_code_name(exit_code)}")
    sys.exit(exit_code)
```

## Error Code Guidelines

**Codes 1-2**: Configuration/input problems, fixable by user
- CONFIG_ERROR: check config.toml, API keys, parameters
- INPUT_ERROR: check input files, formats, schemas

**Code 3**: Runtime errors, require log analysis
- RUNTIME_ERROR: unexpected exceptions, broken data after recovery attempts

**Code 4**: Temporary API limitations, can retry later
- API_LIMIT_ERROR: wait and run again

**Code 5**: Filesystem problems
- IO_ERROR: check access permissions, disk space availability
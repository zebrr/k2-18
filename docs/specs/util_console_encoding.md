# util_console_encoding.md

## Status: READY

Utility for safe UTF-8 encoding setup in Windows console. Solves issues with displaying Cyrillic and special characters without breaking pytest and other tools.

## Public API

### setup_console_encoding() -> None
Sets up UTF-8 encoding for Windows console in a safe way.
- **Input**: None
- **Returns**: None
- **Raises**: Never raises exceptions (fail-safe design)

The function automatically detects:
- Operating system (only works on Windows)
- If running under pytest (doesn't modify streams during tests)
- If streams are already overridden (prevents double wrapping)
- If output is a terminal (doesn't modify when redirected)

Implementation details:
- For Python 3.7+: Uses `reconfigure()` method for safer encoding change
- For older Python: Creates `TextIOWrapper` with preserved `line_buffering`
- Sets `PYTHONIOENCODING` environment variable for child processes
- Marks modified streams with `_original_stream` attribute to prevent re-wrapping

## Test Coverage

No test file exists for this module.

## Dependencies
- **Standard Library**: sys, os, io
- **External**: None
- **Internal**: None

## Performance Notes
- Executes instantly (< 1ms)
- No impact on output performance
- Safe for multiple calls (idempotent)
- Fail-safe: silently continues on any errors

## Usage Examples
```python
from src.utils.console_encoding import setup_console_encoding

# At the start of CLI utility
setup_console_encoding()

# Now you can safely output Cyrillic text
print("Привет, мир!")
logging.info("Processing file: Алгоритмы.txt")
```

### Replacing Unsafe Code
Instead of the unsafe approach:
```python
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
```

Use:
```python
from src.utils.console_encoding import setup_console_encoding
setup_console_encoding()
```

### Platform-Specific Behavior
- **Windows**: Configures UTF-8 encoding using the safest method available
- **Non-Windows**: No-op (returns immediately)
- **Under pytest**: No-op (preserves pytest's stream capture)
- **Non-TTY**: No-op (preserves redirection behavior)

#!/usr/bin/env python3
"""
Standard exit codes for all iText2KG project utilities.

Exit code system provides consistent error handling:
- Codes 1-2: configuration/input problems, fixable by user
- Code 3: runtime errors, require log analysis
- Code 4: temporary API limitations, can retry later
- Code 5: filesystem problems, check access permissions
"""

# Exit codes according to specification
EXIT_SUCCESS = 0  # Successful execution
EXIT_CONFIG_ERROR = 1  # Configuration errors (missing API key, broken config.toml)
EXIT_INPUT_ERROR = 2  # Input data errors (empty files, broken JSON schemas)
EXIT_RUNTIME_ERROR = 3  # Runtime errors (LLM failures, broken slice after repair)
EXIT_API_LIMIT_ERROR = 4  # TPM limits, rate limits (require retry/waiting)
EXIT_IO_ERROR = 5  # File write errors, directory access issues

# Dictionary for readable names (for logging)
EXIT_CODE_NAMES = {
    EXIT_SUCCESS: "SUCCESS",
    EXIT_CONFIG_ERROR: "CONFIG_ERROR",
    EXIT_INPUT_ERROR: "INPUT_ERROR",
    EXIT_RUNTIME_ERROR: "RUNTIME_ERROR",
    EXIT_API_LIMIT_ERROR: "API_LIMIT_ERROR",
    EXIT_IO_ERROR: "IO_ERROR",
}

# Dictionary with descriptions for documentation/logs
EXIT_CODE_DESCRIPTIONS = {
    EXIT_SUCCESS: "Successful execution",
    EXIT_CONFIG_ERROR: "Configuration errors",
    EXIT_INPUT_ERROR: "Input data errors",
    EXIT_RUNTIME_ERROR: "Runtime errors",
    EXIT_API_LIMIT_ERROR: "API limits",
    EXIT_IO_ERROR: "Filesystem errors",
}


def get_exit_code_name(code: int) -> str:
    """
    Returns readable name for exit code.

    Args:
        code: Exit code

    Returns:
        Code name or 'UNKNOWN' for unknown codes
    """
    return EXIT_CODE_NAMES.get(code, f"UNKNOWN({code})")


def get_exit_code_description(code: int) -> str:
    """
    Returns description for exit code.

    Args:
        code: Exit code

    Returns:
        Code description or 'Unknown exit code' for unknown codes
    """
    return EXIT_CODE_DESCRIPTIONS.get(code, f"Unknown exit code: {code}")


def log_exit(logger, code: int, message: str = None) -> None:
    """
    Logs exit code with optional message.

    Args:
        logger: Logger object
        code: Exit code
        message: Additional message (optional)
    """
    code_name = get_exit_code_name(code)
    code_desc = get_exit_code_description(code)

    if code == EXIT_SUCCESS:
        if message:
            logger.info(f"Exit: {code_name} - {message}")
        else:
            logger.info(f"Exit: {code_name}")
    else:
        if message:
            logger.error(f"Exit with error: {code_name} ({code_desc}) - {message}")
        else:
            logger.error(f"Exit with error: {code_name} ({code_desc})")

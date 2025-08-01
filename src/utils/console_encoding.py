"""
Utility for safe UTF-8 encoding setup in Windows console.
"""

import io
import os
import sys


def setup_console_encoding():
    """
    Sets up UTF-8 encoding for Windows console in a safe way.
    Doesn't break pytest and other tools that intercept stdout.
    """
    # Check if we're running on Windows
    if sys.platform != "win32":
        return

    # Check if we're running under pytest
    if "pytest" in sys.modules:
        # Under pytest don't touch stdout/stderr
        return

    # Check if streams are already overridden
    if hasattr(sys.stdout, "_original_stream") or hasattr(
        sys.stderr, "_original_stream"
    ):
        # Already configured, don't touch
        return

    # Check if streams are TTY (terminal)
    if not sys.stdout.isatty() or not sys.stderr.isatty():
        # Not a terminal (possibly redirected to file) - don't touch
        return

    try:
        # Try to set encoding via environment variable
        # This is a safer approach for Python 3.7+
        if sys.version_info >= (3, 7):
            # Set environment variable for future processes
            os.environ["PYTHONIOENCODING"] = "utf-8"

            # For current process use reconfigure (Python 3.7+)
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
                sys.stderr.reconfigure(encoding="utf-8")
        else:
            # For older Python versions use wrapper, but save original
            # and add flag to avoid re-defining
            original_stdout = sys.stdout
            original_stderr = sys.stderr

            sys.stdout = io.TextIOWrapper(
                original_stdout.buffer,
                encoding="utf-8",
                line_buffering=original_stdout.line_buffering,
            )
            sys.stderr = io.TextIOWrapper(
                original_stderr.buffer,
                encoding="utf-8",
                line_buffering=original_stderr.line_buffering,
            )

            # Mark that streams were modified
            sys.stdout._original_stream = original_stdout
            sys.stderr._original_stream = original_stderr

    except Exception:
        # If something went wrong, don't crash, just work as is
        pass

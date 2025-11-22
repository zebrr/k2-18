#!/usr/bin/env python3
"""
CLI utility for splitting educational texts into slices.

Reads files from /data/raw/, applies preprocessing and cuts them into fragments
of fixed size with consideration for semantic boundaries.

Usage:
    python slicer.py

Output files are saved in /data/staging/ in *.slice.json format
"""

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import argparse
import json
import logging
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.tokenizer import find_safe_token_boundary_with_fallback

# Add project root to PYTHONPATH for correct imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import tiktoken
from bs4 import BeautifulSoup

# External dependencies
from unidecode import unidecode

# Import project utilities
from src.utils.config import load_config

# Setup UTF-8 encoding for Windows console
from src.utils.console_encoding import setup_console_encoding
from src.utils.exit_codes import (
    EXIT_CONFIG_ERROR,
    EXIT_INPUT_ERROR,
    EXIT_IO_ERROR,
    EXIT_RUNTIME_ERROR,
    EXIT_SUCCESS,
    log_exit,
)

setup_console_encoding()


def setup_logging(log_level: str = "info") -> None:
    """
    Setup logging for slicer.

    Args:
        log_level: Logging level (debug, info, warning, error)
    """
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }

    level = level_map.get(log_level.lower(), logging.INFO)

    # Setup log format
    formatter = logging.Formatter("[%(asctime)s] %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(console_handler)


def validate_config_parameters(config: Dict[str, Any]) -> None:
    """
    Validate slicer configuration parameters.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: For invalid parameters
    """
    slicer_config = config.get("slicer", {})

    # Check required parameters
    required_params = [
        "max_tokens",
        "soft_boundary_max_shift",
        "allowed_extensions",
    ]
    for param in required_params:
        if param not in slicer_config:
            raise ValueError(f"Missing required parameter slicer.{param}")

    # Check types and ranges
    max_tokens = slicer_config["max_tokens"]
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError(f"slicer.max_tokens must be a positive integer, got: {max_tokens}")

    soft_boundary_max_shift = slicer_config["soft_boundary_max_shift"]
    if not isinstance(soft_boundary_max_shift, int) or soft_boundary_max_shift < 0:
        raise ValueError("slicer.soft_boundary_max_shift must be a non-negative integer")

    allowed_extensions = slicer_config["allowed_extensions"]
    if not isinstance(allowed_extensions, list) or not allowed_extensions:
        raise ValueError("slicer.allowed_extensions must be a non-empty list")


def create_slug(filename: str) -> str:
    """
    Creates slug from filename according to specification.

    Rules:
    - Remove extension
    - Transliterate Cyrillic to Latin
    - Convert to lowercase
    - Replace spaces with _
    - Leave other characters unchanged

    Args:
        filename: Filename with extension

    Returns:
        Processed slug

    Examples:
        >>> create_slug("Алгоритмы и Структуры.txt")
        'algoritmy_i_struktury'
        >>> create_slug("My Course Chapter 1.md")
        'my_course_chapter_1'
        >>> create_slug("python-basics.html")
        'python-basics'
    """
    # Remove extension
    name_without_ext = Path(filename).stem

    # Transliterate Cyrillic to Latin
    transliterated = unidecode(name_without_ext)

    # Convert to lowercase
    lowercased = transliterated.lower()

    # Replace spaces with underscores
    slug = lowercased.replace(" ", "_")

    return slug


def preprocess_text(text: str) -> str:
    """
    Applies preprocessing to text according to specification.

    Includes:
    - Removing content of <script> and <style> tags
    - Unicode NFC normalization
    - Other content remains unchanged

    Args:
        text: Source text

    Returns:
        Processed text
    """
    if not isinstance(text, str):
        raise ValueError("Input parameter must be a string")

    # First apply Unicode normalization
    normalized_text = unicodedata.normalize("NFC", text)

    # Check for presence of HTML script or style tags
    if "<script" in normalized_text.lower() or "<style" in normalized_text.lower():
        # Use BeautifulSoup for safe tag removal
        soup = BeautifulSoup(normalized_text, "html.parser")

        # Remove all script tags and their content
        for script in soup.find_all("script"):
            script.decompose()

        # Remove all style tags and their content
        for style in soup.find_all("style"):
            style.decompose()

        # Get processed text
        processed_text = str(soup)
    else:
        # If no script/style tags, leave as is
        processed_text = normalized_text

    return processed_text


class InputError(Exception):
    """Exception for input data errors."""

    pass


def load_and_validate_file(file_path: Path, allowed_extensions: List[str]) -> str:
    """
    Loads and validates file.

    Args:
        file_path: Path to file
        allowed_extensions: List of allowed extensions

    Returns:
        File content

    Raises:
        InputError: For empty file or unsupported extension
    """
    # Check extension
    if file_path.suffix.lstrip(".").lower() not in [ext.lower() for ext in allowed_extensions]:
        raise InputError(f"Unsupported file extension: {file_path.suffix}")

    try:
        # Load with encoding auto-detection
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        # Fallback to other encodings
        try:
            with open(file_path, "r", encoding="cp1251") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin1") as f:
                content = f.read()

    # Apply preprocessing
    content = preprocess_text(content)

    # Check for emptiness after preprocessing
    if not content.strip():
        raise InputError(
            f"Empty file detected: {file_path.name}. Please remove empty files from /data/raw/"
        )

    return content


def slice_text_with_window(
    text: str,
    max_tokens: int,
    soft_boundary: bool,
    soft_boundary_max_shift: int,
    file_name: Optional[str] = None,
) -> List[Tuple[str, int, int]]:
    """
    Slices text into chunks using incremental tokenization with improved progress reporting.

    Key optimizations:
    - Tokenizes only current window + buffer, not entire file
    - Tracks global token count for slice_token_start/end
    - Uses character positions for navigation between windows
    - Uses smart candidate selection for boundary detection

    Args:
        text: Source text for slicing
        max_tokens: Maximum slice size in tokens
        soft_boundary: Whether to use soft boundaries
        soft_boundary_max_shift: Maximum shift for soft boundaries (in tokens)
        file_name: Optional file name for improved logging

    Returns:
        List of tuples (slice_text, slice_token_start, slice_token_end)
    """
    if not text or not text.strip():
        return []

    encoding = tiktoken.get_encoding("o200k_base")

    # Quick check if entire text fits in one slice
    # We'll do a full tokenization only for small texts
    if len(text) <= max_tokens * 10:  # Conservative estimate
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            logging.info(f"File fits in single slice: {len(tokens)} tokens")
            return [(text, 0, len(tokens))]

    # For large files, estimate total tokens (without full tokenization!)
    estimated_total_tokens = len(text) // 4  # Rough estimate
    estimated_slices = (estimated_total_tokens // max_tokens) + 1

    if file_name:
        file_size_mb = len(text) / (1024 * 1024)
        logging.info(
            f"Processing file: {file_name} "
            f"({file_size_mb:.1f}MB, ~{estimated_total_tokens:,} tokens)"
        )
        logging.info(f"Estimated slices: ~{estimated_slices}")

    slices = []
    char_pos = 0
    global_token_offset = 0  # For tracking slice_token_start/end
    slice_num = 1

    while char_pos < len(text):
        # Calculate window size with buffer
        # Factor 10 guarantees coverage for any language
        window_chars = max_tokens * 10
        buffer_chars = soft_boundary_max_shift * 10 if soft_boundary else 0
        total_window_chars = window_chars + buffer_chars

        # Extract window text
        window_text = text[char_pos : char_pos + total_window_chars]
        if not window_text:
            break

        window_tokens = encoding.encode(window_text)

        # Handle last window
        if len(window_tokens) <= max_tokens:
            # Take everything remaining
            slice_text = window_text
            slice_token_count = len(window_tokens)

            logging.info(
                f"Creating slice_{slice_num:03d} "
                f"(tokens {global_token_offset}-{global_token_offset + slice_token_count}) "
                f"[100%]"
            )

            slices.append(
                (
                    slice_text,
                    global_token_offset,  # slice_token_start
                    global_token_offset + slice_token_count,  # slice_token_end
                )
            )
            break
        else:
            # Find boundary in window
            target_pos = min(max_tokens, len(window_tokens))

            if soft_boundary and soft_boundary_max_shift > 0:
                # Use new fallback function (logging is inside)
                boundary_token_pos = find_safe_token_boundary_with_fallback(
                    text=window_text,
                    tokens=window_tokens,
                    encoding=encoding,
                    target_token_pos=target_pos,
                    max_shift_tokens=soft_boundary_max_shift,
                    max_tokens=max_tokens,
                )
            else:
                boundary_token_pos = target_pos

            # Get slice text
            slice_text = encoding.decode(window_tokens[:boundary_token_pos])
            slice_token_count = boundary_token_pos

            # Log progress with exact token range
            progress_pct = (char_pos / len(text)) * 100
            logging.info(
                f"Creating slice_{slice_num:03d} "
                f"(tokens {global_token_offset}-{global_token_offset + slice_token_count}) "
                f"[{progress_pct:.0f}%]"
            )

        # Verify no gaps
        if slices and global_token_offset != slices[-1][2]:
            logging.warning(
                f"Gap detected: previous ended at {slices[-1][2]}, "
                f"current starts at {global_token_offset}"
            )

        # Add slice with global token positions
        slices.append(
            (
                slice_text,
                global_token_offset,  # slice_token_start
                global_token_offset + slice_token_count,  # slice_token_end
            )
        )

        # Update positions for next iteration
        char_pos += len(slice_text)
        global_token_offset += slice_token_count
        slice_num += 1

    if file_name:
        logging.info(f"Completed: {len(slices)} slices from {file_name}")

    return slices


def save_slice(slice_data: Dict[str, Any], output_dir: Path) -> None:
    """
    Saves slice to JSON file.

    Args:
        slice_data: Slice data
        output_dir: Directory for saving

    Raises:
        IOError: For file writing errors
    """
    slice_id = slice_data["id"]
    output_file = output_dir / f"{slice_id}.slice.json"

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(slice_data, f, ensure_ascii=False, indent=2)

        logging.info(f"Slice saved: {output_file}")
    except Exception as e:
        logging.error(f"Error saving slice {slice_id}: {e}")
        raise IOError(f"Failed to save slice {slice_id}: {e}")


def process_file(
    file_path: Path, config: Dict[str, Any], global_slice_counter: int
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Processes one file and returns list of slices.

    Args:
        file_path: Path to file
        config: Slicer configuration
        global_slice_counter: Global slice counter

    Returns:
        Tuple (slice list, updated counter)

    Raises:
        InputError: For input data errors
        RuntimeError: For processing errors
    """
    slicer_config = config["slicer"]

    # Calculate file size for logging
    file_size = file_path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)

    logging.info(f"Processing file: {file_path.name} ({file_size_mb:.2f}MB)")

    try:
        # Load and validate file
        content = load_and_validate_file(file_path, slicer_config["allowed_extensions"])

        # Create slug
        slug = create_slug(file_path.name)

        # Slice into chunks (pass filename for improved logging)
        slices_data = slice_text_with_window(
            content,
            slicer_config["max_tokens"],
            slicer_config["soft_boundary"],
            slicer_config["soft_boundary_max_shift"],
            file_name=file_path.name,
        )

        # Create slice objects
        slices = []
        for i, (slice_text, slice_token_start, slice_token_end) in enumerate(slices_data):
            slice_obj = {
                "id": f"slice_{global_slice_counter:03d}",
                "order": global_slice_counter,
                "source_file": file_path.name,
                "slug": slug,
                "text": slice_text,
                "slice_token_start": slice_token_start,
                "slice_token_end": slice_token_end,
            }
            slices.append(slice_obj)
            global_slice_counter += 1

        logging.info(f"File {file_path.name}: created {len(slices)} slices")
        return slices, global_slice_counter

    except InputError:
        # Re-raise InputError as is
        raise
    except Exception as e:
        logging.error(f"Error processing file {file_path.name}: {e}")
        raise RuntimeError(f"Failed to process file {file_path.name}: {e}")


def main(argv=None):
    """Main slicer function."""

    # CLI setup
    parser = argparse.ArgumentParser(
        description="Utility for splitting educational texts into slices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # No parameters yet - config is hardcoded
    _ = parser.parse_args(argv)

    try:
        # Load configuration
        config = load_config()

        # Setup logging
        log_level = config.get("slicer", {}).get("log_level", "info")
        setup_logging(log_level)

        logging.info("Starting slicer.py")

        # Validate configuration parameters
        try:
            validate_config_parameters(config)
        except ValueError as e:
            logging.error(f"Configuration error: {e}")
            return EXIT_CONFIG_ERROR

        # Define paths
        raw_dir = Path("data/raw")
        staging_dir = Path("data/staging")

        # Check directory existence
        if not raw_dir.exists():
            logging.error(f"Directory {raw_dir} does not exist")
            return EXIT_INPUT_ERROR

        try:
            staging_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.error(f"Failed to create directory {staging_dir}: {e}")
            return EXIT_IO_ERROR

        # Get file list in lexicographic order
        allowed_extensions = config["slicer"]["allowed_extensions"]
        input_files = []

        for ext in allowed_extensions:
            pattern = f"*.{ext.lower()}"
            input_files.extend(raw_dir.glob(pattern))
            # Also search with uppercase letters
            pattern_upper = f"*.{ext.upper()}"
            input_files.extend(raw_dir.glob(pattern_upper))

        # Remove duplicates and sort
        input_files = sorted(set(input_files))

        if not input_files:
            logging.warning(f"No files found for processing in {raw_dir}")
            logging.warning(f"Supported extensions: {allowed_extensions}")
            return EXIT_SUCCESS

        # Output files that will be skipped
        all_files = list(raw_dir.iterdir())
        for file_path in all_files:
            if file_path.is_file() and file_path not in input_files:
                logging.warning(f"Unsupported file skipped: {file_path.name}")

        logging.info(f"Found {len(input_files)} files for processing")

        # Process files
        global_slice_counter = 1
        total_slices = 0

        for file_path in input_files:
            try:
                slices, global_slice_counter = process_file(file_path, config, global_slice_counter)

                # Save slices
                for slice_data in slices:
                    save_slice(slice_data, staging_dir)

                total_slices += len(slices)

            except InputError as e:
                logging.error(f"Input data error in file {file_path.name}: {e}")
                return EXIT_INPUT_ERROR
            except IOError as e:
                logging.error(f"I/O error when processing {file_path.name}: {e}")
                return EXIT_IO_ERROR
            except RuntimeError as e:
                logging.error(f"Runtime error when processing {file_path.name}: {e}")
                return EXIT_RUNTIME_ERROR
            except Exception as e:
                logging.error(f"Unexpected error when processing {file_path.name}: {e}")
                return EXIT_RUNTIME_ERROR

        logging.info(f"Processing completed: {total_slices} slices saved in {staging_dir}")
        log_exit(logging.getLogger(), EXIT_SUCCESS)
        return EXIT_SUCCESS

    except Exception as e:
        logging.error(f"Critical error: {e}")
        log_exit(logging.getLogger(), EXIT_RUNTIME_ERROR)
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

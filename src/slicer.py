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
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.utils.tokenizer import count_tokens

# Add project root to PYTHONPATH for correct imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bs4 import BeautifulSoup
# External dependencies
from unidecode import unidecode

# Import project utilities
from src.utils.config import load_config
# Setup UTF-8 encoding for Windows console
from src.utils.console_encoding import setup_console_encoding
from src.utils.exit_codes import (EXIT_CONFIG_ERROR, EXIT_INPUT_ERROR,
                                  EXIT_IO_ERROR, EXIT_RUNTIME_ERROR,
                                  EXIT_SUCCESS, log_exit)

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
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
    )

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
        "overlap",
        "soft_boundary_max_shift",
        "allowed_extensions",
    ]
    for param in required_params:
        if param not in slicer_config:
            raise ValueError(f"Missing required parameter slicer.{param}")

    # Check types and ranges
    max_tokens = slicer_config["max_tokens"]
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError(
            f"slicer.max_tokens must be a positive integer, got: {max_tokens}"
        )

    overlap = slicer_config["overlap"]
    if not isinstance(overlap, int) or overlap < 0:
        raise ValueError(
            f"slicer.overlap must be a non-negative integer, got: {overlap}"
        )

    if overlap >= max_tokens:
        raise ValueError(
            f"slicer.overlap ({overlap}) must be less than max_tokens ({max_tokens})"
        )

    soft_boundary_max_shift = slicer_config["soft_boundary_max_shift"]
    if not isinstance(soft_boundary_max_shift, int) or soft_boundary_max_shift < 0:
        raise ValueError(
            "slicer.soft_boundary_max_shift must be a non-negative integer"
        )

    # Special validation for overlap > 0
    if overlap > 0:
        max_allowed_shift = int(overlap * 0.8)
        if soft_boundary_max_shift > max_allowed_shift:
            raise ValueError(
                f"When overlap > 0, soft_boundary_max_shift ({soft_boundary_max_shift}) "
                f"must not exceed overlap*0.8 ({max_allowed_shift})"
            )

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
    if file_path.suffix.lstrip(".").lower() not in [
        ext.lower() for ext in allowed_extensions
    ]:
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
    overlap: int,
    soft_boundary: bool,
    soft_boundary_max_shift: int,
) -> List[Tuple[str, int, int]]:
    """
    Slices text into chunks using sliding window.

    Args:
        text: Source text for slicing
        max_tokens: Maximum slice size in tokens
        overlap: Number of overlapping tokens
        soft_boundary: Whether to use soft boundaries
        soft_boundary_max_shift: Maximum shift for soft boundaries

    Returns:
        List of tuples (slice_text, slice_token_start, slice_token_end)
    """
    if not text or not text.strip():
        return []

    # Count total tokens in text
    total_tokens = count_tokens(text)

    # If entire text fits in one slice
    if total_tokens <= max_tokens:
        return [(text, 0, total_tokens)]

    # Tokenize entire text for position handling
    import tiktoken

    encoding = tiktoken.get_encoding("o200k_base")
    tokens = encoding.encode(text)

    slices = []
    current_token_start = 0

    while current_token_start < len(tokens):
        # Determine end of current window
        window_end = min(current_token_start + max_tokens, len(tokens))

        # If this is the last fragment, take to the end
        if window_end == len(tokens):
            slice_tokens = tokens[current_token_start:]
            slice_text = encoding.decode(slice_tokens)
            slice_token_start = current_token_start
            slice_token_end = len(tokens)

            slices.append((slice_text, slice_token_start, slice_token_end))
            break

        # Look for soft boundary if enabled
        actual_end = window_end
        if soft_boundary and soft_boundary_max_shift > 0:
            # Convert soft_boundary_max_shift from characters to approximate token count
            # Approximate ratio: 1 token ≈ 4 characters (for o200k_base)
            max_shift_tokens = max(1, soft_boundary_max_shift // 4)

            # Use new function to find safe boundary
            from src.utils.tokenizer import find_safe_token_boundary

            safe_end = find_safe_token_boundary(
                text=text,
                tokens=tokens,
                encoding=encoding,
                target_token_pos=window_end,
                max_shift_tokens=max_shift_tokens,
            )

            # Logging for debugging
            if safe_end != window_end:
                shift = safe_end - window_end
                logging.info(f"Soft boundary found: shift {shift:+d} tokens")

                # Show boundary type for debugging
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    context = encoding.decode(tokens[max(0, safe_end - 20) : safe_end])
                    if context.endswith("\n\n"):
                        logging.debug("Boundary type: double line break")
                    elif re.search(r"[.!?]\s*$", context):
                        logging.debug("Boundary type: end of sentence")
                    elif re.search(r"</h[1-6]>\s*$", context):
                        logging.debug("Boundary type: HTML heading")

            actual_end = safe_end

        # Create final slice
        final_slice_tokens = tokens[current_token_start:actual_end]
        slice_text = encoding.decode(final_slice_tokens)
        slice_token_start = current_token_start
        slice_token_end = actual_end

        slices.append((slice_text, slice_token_start, slice_token_end))

        # Calculate start of next window
        if overlap == 0:
            current_token_start = actual_end  # Without overlaps
        else:
            current_token_start = actual_end - overlap  # With overlap
            # Protection from infinite loop
            if current_token_start <= slice_token_start:
                current_token_start = slice_token_start + 1

    # Handle edge cases for overlap > 0 according to specification
    if overlap > 0 and len(slices) >= 2:
        last_slice = slices[-1]
        prev_slice = slices[-2]

        # If last fragment is smaller than overlap, merge with previous
        last_slice_size = last_slice[2] - last_slice[1]  # token_end - token_start
        if last_slice_size < overlap:
            # Update previous slice
            prev_start = prev_slice[1]
            combined_end = last_slice[2]
            combined_tokens = tokens[prev_start:combined_end]
            combined_text = encoding.decode(combined_tokens)

            # Replace previous slice with updated one
            slices[-2] = (combined_text, prev_start, combined_end)
            # Remove last slice
            slices.pop()

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

    logging.info(f"Processing file: {file_path.name}")

    try:
        # Load and validate file
        content = load_and_validate_file(file_path, slicer_config["allowed_extensions"])

        # Create slug
        slug = create_slug(file_path.name)

        # Slice into chunks
        slices_data = slice_text_with_window(
            content,
            slicer_config["max_tokens"],
            slicer_config["overlap"],
            slicer_config["soft_boundary"],
            slicer_config["soft_boundary_max_shift"],
        )

        # Create slice objects
        slices = []
        for i, (slice_text, slice_token_start, slice_token_end) in enumerate(
            slices_data
        ):
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
    args = parser.parse_args(argv)

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
                slices, global_slice_counter = process_file(
                    file_path, config, global_slice_counter
                )

                # Save slices
                for slice_data in slices:
                    save_slice(slice_data, staging_dir)

                total_slices += len(slices)

            except InputError as e:
                logging.error(f"Input data error in file {file_path.name}: {e}")
                return EXIT_INPUT_ERROR
            except IOError as e:
                logging.error(
                    f"I/O error when processing {file_path.name}: {e}"
                )
                return EXIT_IO_ERROR
            except RuntimeError as e:
                logging.error(f"Runtime error when processing {file_path.name}: {e}")
                return EXIT_RUNTIME_ERROR
            except Exception as e:
                logging.error(f"Unexpected error when processing {file_path.name}: {e}")
                return EXIT_RUNTIME_ERROR

        logging.info(
            f"Processing completed: {total_slices} slices saved in {staging_dir}"
        )
        log_exit(logging.getLogger(), EXIT_SUCCESS)
        return EXIT_SUCCESS

    except Exception as e:
        logging.error(f"Critical error: {e}")
        log_exit(logging.getLogger(), EXIT_RUNTIME_ERROR)
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

"""
Module for text tokenization and semantic boundary detection.

Uses the o200k_base codec, which is supported by all modern OpenAI models
"""

import logging
import re
from typing import List, Optional, Tuple

import tiktoken


def count_tokens(text: str) -> int:
    """
    Counts the number of tokens in text using the o200k_base codec.

    Args:
        text: Input text for tokenization

    Returns:
        Number of tokens in the text

    Raises:
        ValueError: If text is not a string
    """
    if not isinstance(text, str):
        raise ValueError("Input parameter must be a string")

    # Initialize o200k_base tokenizer
    encoding = tiktoken.get_encoding("o200k_base")

    # Count tokens
    tokens = encoding.encode(text)
    return len(tokens)


def find_soft_boundary(text: str, target_pos: int, max_shift: int) -> Optional[int]:
    """
    Finds the nearest semantic boundary in text for soft breaks.
    Uses a priority hierarchy for educational content.

    Boundary priorities (from highest to lowest):
    1. Section boundaries (headers)
    2. Paragraph boundaries (double line breaks, code blocks)
    3. Sentence boundaries (periods, exclamation and question marks)
    4. Phrase boundaries (commas, colons, semicolons)
    5. Word boundaries (spaces) - fallback

    Args:
        text: Source text
        target_pos: Target position in characters
        max_shift: Maximum offset from target_pos

    Returns:
        Boundary position in characters or None if boundary not found
    """
    if not isinstance(text, str) or len(text) == 0:
        return None

    if target_pos < 0 or target_pos > len(text):
        return None

    if max_shift < 0:
        return None

    # Define search range
    start_pos = max(0, target_pos - max_shift)
    end_pos = min(len(text), target_pos + max_shift)

    # Hierarchy of boundaries with weights (lower weight = better boundary)
    boundary_types = {
        "section": {"weight": 1, "candidates": []},
        "paragraph": {"weight": 2, "candidates": []},
        "sentence": {"weight": 3, "candidates": []},
        "phrase": {"weight": 4, "candidates": []},
        "word": {"weight": 5, "candidates": []},
    }

    # 1. Section boundaries (priority 1)
    # HTML headers
    for match in re.finditer(r"</h[1-6]>\s*(?=\n|$)", text, re.IGNORECASE):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            boundary_types["section"]["candidates"].append(pos)

    # Markdown headers
    for match in re.finditer(r"(?:^|\n)(#{1,6})\s+.*?(?=\n|$)", text):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            boundary_types["section"]["candidates"].append(pos)

    # Text headers
    section_pattern = (
        r"(?:^|\n)(?:Глава|Параграф|Часть|Chapter|Section|Раздел|Урок|Тема)" r"\s+.*?(?=\n|$)"
    )
    for match in re.finditer(section_pattern, text, re.IGNORECASE | re.MULTILINE):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            boundary_types["section"]["candidates"].append(pos)

    # 2. Paragraph boundaries (priority 2)
    # Double line break
    for match in re.finditer(r"\n\n+", text):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            boundary_types["paragraph"]["candidates"].append(pos)

    # End of code block
    for match in re.finditer(r"(?:^|\n)```\s*(?=\n|$)", text):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            boundary_types["paragraph"]["candidates"].append(pos)

    # End of formula block
    for match in re.finditer(r"(?:^|\n)\$\$\s*(?=\n|$)", text):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            boundary_types["paragraph"]["candidates"].append(pos)

    # HTML/Markdown links
    for match in re.finditer(r"</a>|\]\([^)]+\)", text):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            boundary_types["paragraph"]["candidates"].append(pos)

    # 3. Sentence boundaries (priority 3)
    # End of sentence
    for match in re.finditer(r"[.!?]\s+", text):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            # Check that this is not an abbreviation
            before_pos = match.start()
            if before_pos > 0:
                # Simple heuristic for abbreviations
                word_before = text[max(0, before_pos - 10) : before_pos].strip()
                if not word_before.endswith(
                    (
                        "Dr",
                        "Mr",
                        "Mrs",
                        "Ms",
                        "Prof",
                        "St",
                        "vs",
                        "etc",
                        "т.д",
                        "т.п",
                        "и.д",
                        "и.п",
                    )
                ):
                    boundary_types["sentence"]["candidates"].append(pos)

    # Semicolon
    for match in re.finditer(r";\s+", text):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            boundary_types["sentence"]["candidates"].append(pos)

    # 4. Phrase boundaries (priority 4)
    # Comma
    for match in re.finditer(r",\s+", text):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            boundary_types["phrase"]["candidates"].append(pos)

    # Colon
    for match in re.finditer(r":\s+", text):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            boundary_types["phrase"]["candidates"].append(pos)

    # Dash
    for match in re.finditer(r"\s+[—–-]\s+", text):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            boundary_types["phrase"]["candidates"].append(pos)

    # 5. Word boundaries (priority 5 - fallback)
    for match in re.finditer(r"\s+", text):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            boundary_types["word"]["candidates"].append(pos)

    # Select the best boundary considering priorities
    best_boundary = None
    best_score = float("inf")

    for boundary_type, data in boundary_types.items():
        weight = data["weight"]
        candidates = data["candidates"]

        # Remove duplicates
        candidates = list(set(candidates))

        for pos in candidates:
            # Calculate score = weight * distance
            # Lower score is better
            distance = abs(pos - target_pos)
            score = weight * distance

            # Small bonus for boundaries after target_pos (prefer not to cut)
            if pos > target_pos:
                score *= 0.9

            if score < best_score:
                best_score = score
                best_boundary = pos

    return best_boundary


def _build_token_char_mapping(tokens: List[int], encoding) -> dict:
    """
    Build mapping from token index to character position.

    Args:
        tokens: List of token IDs
        encoding: tiktoken encoding object

    Returns:
        Dictionary mapping token index to character position
    """
    token_to_char = {0: 0}

    for i in range(1, len(tokens) + 1):
        # Decode prefix to get char position
        # This is more accurate than decoding individual tokens
        prefix = encoding.decode(tokens[:i])
        token_to_char[i] = len(prefix)

    return token_to_char


def _get_text_at_boundary(decoded_text: str, char_pos: int) -> tuple:
    """
    Split decoded text at character position.

    Args:
        decoded_text: The decoded text
        char_pos: Character position to split at

    Returns:
        Tuple (text_before, text_after)
    """
    return decoded_text[:char_pos], decoded_text[char_pos:]


def find_boundary_candidates(
    decoded_text: str, target_char_pos: int, max_char_shift: int
) -> List[Tuple[int, str]]:
    """
    Find boundary candidates with priorities.

    Key change: boundaries are now BEFORE headers, not after.
    Uses lookahead patterns (?=...) to find positions before patterns.

    Priority levels:
        1: BEFORE headers (best boundaries)
        2: BEFORE subheaders, AFTER paragraphs
        3: AFTER sentences
        4: AFTER lines
        5: AFTER phrases
        6: Between words (fallback)

    Args:
        decoded_text: The decoded text to search in
        target_char_pos: Target position in characters
        max_char_shift: Maximum offset from target_pos

    Returns:
        List of (position, boundary_type) tuples sorted by score (priority * 1000 + distance)
    """
    candidates = []

    start_pos = max(0, target_char_pos - max_char_shift)
    end_pos = min(len(decoded_text), target_char_pos + max_char_shift)

    # Priority 1: BEFORE headers (best boundaries)
    # Using lookahead (?=...) to find position BEFORE header

    # Before HTML headers
    for match in re.finditer(r"(?:^|\n)(?=<h[1-6][^>]*>)", decoded_text):
        pos = match.end()  # Position after \n and before <h1>
        if start_pos <= pos <= end_pos:
            candidates.append((pos, 1, "html_header"))

    # Before Markdown headers
    for match in re.finditer(r"(?:^|\n)(?=#{1,6}\s+)", decoded_text):
        pos = match.end()  # Position after \n and before #
        if start_pos <= pos <= end_pos:
            candidates.append((pos, 1, "markdown_header"))

    # Before text headers (Russian and English)
    pattern = r"(?:^|\n)(?=(?:Глава|Параграф|Часть|Chapter|Section|Раздел|Урок|Тема)\s+)"
    for match in re.finditer(pattern, decoded_text, re.IGNORECASE):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            candidates.append((pos, 1, "text_header"))

    # Priority 2: BEFORE subheaders and AFTER paragraphs
    # Before H2-H4 (subheaders)
    for match in re.finditer(r"(?:^|\n)(?=#{2,4}\s+)", decoded_text):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            candidates.append((pos, 2, "subheader"))

    # After double line break (paragraph end)
    for match in re.finditer(r"\n\n+", decoded_text):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            candidates.append((pos, 2, "paragraph"))

    # After code block end
    for match in re.finditer(r"```\s*\n", decoded_text):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            candidates.append((pos, 2, "code_block_end"))

    # Priority 3: After sentences
    for match in re.finditer(r"[.!?]\s+", decoded_text):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            # Check for abbreviations
            before = decoded_text[max(0, match.start() - 10) : match.start()]
            if not before.endswith(
                ("Dr", "Mr", "Mrs", "Ms", "Prof", "St", "vs", "etc", "т.д", "т.п", "и.д", "и.п")
            ):
                candidates.append((pos, 3, "sentence"))

    # Priority 4: After lines (single newline)
    for match in re.finditer(r"\n", decoded_text):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            candidates.append((pos, 4, "line"))

    # Priority 5: After phrases (comma, semicolon, colon)
    for match in re.finditer(r"[,;:]\s+", decoded_text):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            candidates.append((pos, 5, "phrase"))

    # Priority 6: Between words (fallback)
    for match in re.finditer(r"\s+", decoded_text):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            candidates.append((pos, 6, "word"))

    # Sort by score (priority * 1000 + distance)
    scored_candidates = []
    for pos, priority, boundary_type in candidates:
        distance = abs(pos - target_char_pos)
        score = priority * 1000 + distance  # Priority is more important
        scored_candidates.append((score, pos, boundary_type))

    scored_candidates.sort()

    # Return top candidates (limit to 50 to avoid too many checks)
    return [(pos, boundary_type) for _, pos, boundary_type in scored_candidates[:50]]


def find_safe_token_boundary(
    text: str, tokens: List[int], encoding, target_token_pos: int, max_shift_tokens: int
) -> Tuple[int, str]:
    """
    Finds a safe boundary for cutting at the token level using smart candidate selection.

    Key optimizations:
    - Decode working range ONCE, build mapping
    - Use find_boundary_candidates() to select smart candidates instead of checking all positions
    - Prioritize boundaries BEFORE headers

    Args:
        text: Source text (unused but kept for compatibility)
        tokens: List of tokens
        encoding: tiktoken encoding object
        target_token_pos: Target position in tokens
        max_shift_tokens: Maximum offset in tokens

    Returns:
        Tuple of (safe position in tokens for cutting, boundary_type)
    """
    # Handle edge cases
    if len(tokens) == 0:
        return 0, "empty"

    # For negative positions, just return 0 (beginning of text)
    if target_token_pos < 0:
        # If target is negative but within shift range of 0, search near beginning
        if target_token_pos + max_shift_tokens >= 0:
            # Search in valid range near beginning
            original_target = target_token_pos
            target_token_pos = 0
        else:
            # Target is too far negative, just return 0
            return 0, "edge"
    else:
        original_target = target_token_pos

    if target_token_pos > len(tokens):
        target_token_pos = len(tokens)
        original_target = target_token_pos

    # Define search range
    start_pos = max(0, target_token_pos - max_shift_tokens)
    end_pos = min(len(tokens), target_token_pos + max_shift_tokens)

    # OPTIMIZATION: Decode working range ONCE
    working_tokens = tokens[start_pos : end_pos + 1]
    decoded_text = encoding.decode(working_tokens)

    # Build token_index -> char_index mapping for decoded_text
    token_to_char = {}
    for i in range(len(working_tokens) + 1):
        # Map each token boundary to character position
        if i == 0:
            token_to_char[0] = 0
        elif i == len(working_tokens):
            token_to_char[i] = len(decoded_text)
        else:
            # Decode prefix to get char position
            prefix = encoding.decode(working_tokens[:i])
            token_to_char[i] = len(prefix)

    # Build reverse mapping: char_pos -> local_token_pos (nearest)
    char_to_token = {}
    for local_token, char_pos in token_to_char.items():
        char_to_token[char_pos] = local_token

    # Calculate target_char_pos
    target_local_token = target_token_pos - start_pos
    if target_local_token in token_to_char:
        target_char_pos = token_to_char[target_local_token]
    else:
        target_char_pos = len(decoded_text) // 2

    # Get smart boundary candidates
    max_char_shift = max_shift_tokens * 4  # Approximate chars per token
    candidates = find_boundary_candidates(decoded_text, target_char_pos, max_char_shift)

    best_pos = target_token_pos
    best_score = float("inf")
    best_type = "none"

    # If we have candidates, check them first
    if candidates:
        # Create sorted list of token boundaries for finding nearest
        sorted_char_positions = sorted(char_to_token.keys())

        for char_pos, boundary_type in candidates:
            # Find nearest token position for this char position
            local_token = None

            # Check if exact match
            if char_pos in char_to_token:
                local_token = char_to_token[char_pos]
            else:
                # Find nearest token boundary
                for i, cp in enumerate(sorted_char_positions):
                    if cp >= char_pos:
                        # Use this position or the previous one (whichever is closer)
                        if i > 0 and (cp - char_pos) > (char_pos - sorted_char_positions[i - 1]):
                            local_token = char_to_token[sorted_char_positions[i - 1]]
                        else:
                            local_token = char_to_token[cp]
                        break
                if local_token is None and sorted_char_positions:
                    local_token = char_to_token[sorted_char_positions[-1]]

            if local_token is None:
                continue

            global_pos = start_pos + local_token

            # Ensure we stay within the allowed search range
            if abs(global_pos - original_target) > max_shift_tokens:
                continue

            # Get text before/after from cache
            actual_char_pos = token_to_char[local_token]
            text_before = decoded_text[:actual_char_pos]
            text_after = decoded_text[actual_char_pos:]

            # Check if safe using cached text
            if is_safe_cut_position(
                text=None,
                tokens=None,
                encoding=None,
                pos=global_pos,
                text_before=text_before,
                text_after=text_after,
            ):
                # Calculate score based on boundary type priority
                priority = {
                    "html_header": 1,
                    "markdown_header": 1,
                    "text_header": 1,
                    "subheader": 2,
                    "paragraph": 2,
                    "code_block_end": 2,
                    "sentence": 3,
                    "line": 4,
                    "phrase": 5,
                    "word": 6,
                }.get(boundary_type, 7)

                distance = abs(global_pos - target_token_pos)
                score = priority * 1000 + distance

                if score < best_score:
                    best_score = score
                    best_pos = global_pos
                    best_type = boundary_type

    # Fallback: if no good candidate found, check all positions (original behavior)
    if best_type == "none":
        for local_pos in range(len(working_tokens) + 1):
            global_pos = start_pos + local_pos

            # Ensure we stay within the allowed search range from original target
            if abs(global_pos - original_target) > max_shift_tokens:
                continue

            # Get text before/after from cache
            char_pos = token_to_char[local_pos]
            text_before = decoded_text[:char_pos]
            text_after = decoded_text[char_pos:]

            # Check if safe using cached text
            if is_safe_cut_position(
                text=None,
                tokens=None,
                encoding=None,
                pos=global_pos,
                text_before=text_before,
                text_after=text_after,
            ):
                # Evaluate quality using cached text
                score = evaluate_boundary_quality(
                    text=None,
                    tokens=None,
                    encoding=None,
                    pos=global_pos,
                    text_before=text_before,
                    text_after=text_after,
                )

                # Add distance penalty
                distance_penalty = abs(global_pos - target_token_pos)
                total_score = score + distance_penalty * 0.1

                if total_score < best_score:
                    best_score = total_score
                    best_pos = global_pos
                    best_type = "fallback"

    return best_pos, best_type


def find_safe_token_boundary_with_fallback(
    text: str,
    tokens: List[int],
    encoding,
    target_token_pos: int,
    max_shift_tokens: int,
    max_tokens: int,
) -> int:
    """
    Find safe boundary with smart fallback for large blocks.

    If no safe boundary found in normal range, expands search up to 30% of window size.
    Returns only position (for compatibility with slice_text_with_window).

    Args:
        text: Source text (unused but kept for compatibility)
        tokens: List of tokens
        encoding: tiktoken encoding object
        target_token_pos: Target position in tokens
        max_shift_tokens: Maximum offset in tokens
        max_tokens: Window size for calculating extended range

    Returns:
        Safe position in tokens for cutting
    """
    # First try normal search
    best_pos, best_type = find_safe_token_boundary(
        text, tokens, encoding, target_token_pos, max_shift_tokens
    )

    # Log the boundary type if we found something useful
    if best_type not in ("none", "empty", "edge"):
        shift = best_pos - target_token_pos
        if shift != 0:
            logging.info(f"Soft boundary found: shift {shift:+d} tokens")
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(f"Boundary type: {best_type}")
        return best_pos

    # Check if we found a truly safe position
    if best_pos != target_token_pos and best_pos > 0 and best_pos < len(tokens):
        text_before = encoding.decode(tokens[:best_pos])
        text_after = encoding.decode(tokens[best_pos:])

        if is_safe_cut_position(
            text=None,
            tokens=None,
            encoding=None,
            pos=best_pos,
            text_before=text_before,
            text_after=text_after,
        ):
            return best_pos

    # Fallback: expand search range
    logging.warning(f"No safe boundary found at position {target_token_pos}, expanding search")

    # Try up to 30% of window size
    extended_shift = int(max_tokens * 0.3)

    # Try forward first (prefer not cutting in the middle of content)
    for offset in range(max_shift_tokens + 1, extended_shift, 10):
        test_pos = min(len(tokens), target_token_pos + offset)
        if test_pos < len(tokens):
            text_before = encoding.decode(tokens[:test_pos])
            text_after = encoding.decode(tokens[test_pos:])

            if is_safe_cut_position(
                text=None,
                tokens=None,
                encoding=None,
                pos=test_pos,
                text_before=text_before,
                text_after=text_after,
            ):
                logging.info(
                    f"Extended boundary found: shift {test_pos - target_token_pos:+d} tokens"
                )
                return test_pos

    # Try backward
    for offset in range(max_shift_tokens + 1, extended_shift, 10):
        test_pos = max(0, target_token_pos - offset)
        if test_pos > 0:
            text_before = encoding.decode(tokens[:test_pos])
            text_after = encoding.decode(tokens[test_pos:])

            if is_safe_cut_position(
                text=None,
                tokens=None,
                encoding=None,
                pos=test_pos,
                text_before=text_before,
                text_after=text_after,
            ):
                logging.info(
                    f"Extended boundary found: shift {test_pos - target_token_pos:+d} tokens"
                )
                return test_pos

    # Ultimate fallback: force cut at end of normal range
    fallback_pos = min(len(tokens), target_token_pos + max_shift_tokens)
    logging.warning(f"Forcing boundary at position {fallback_pos} (no safe position found)")
    return fallback_pos


def is_safe_cut_position(
    text: Optional[str] = None,
    tokens: Optional[List[int]] = None,
    encoding=None,
    pos: Optional[int] = None,
    text_before: Optional[str] = None,
    text_after: Optional[str] = None,
) -> bool:
    """
    Checks if it's safe to cut at the given token position.

    If text_before/text_after are provided, uses them (cached mode).
    Otherwise, decodes from tokens (legacy mode).

    Args:
        text: Source text (legacy mode, unused)
        tokens: List of tokens (legacy mode)
        encoding: tiktoken encoding object (legacy mode)
        pos: Token position (legacy mode)
        text_before: Pre-decoded text before boundary (cached mode)
        text_after: Pre-decoded text after boundary (cached mode)

    Returns:
        True if it's safe to cut at this position
    """
    # Handle cached mode
    if text_before is not None and text_after is not None:
        # Use provided cached text
        pass
    # Handle legacy mode
    elif tokens is not None and encoding is not None and pos is not None:
        if pos <= 0 or pos >= len(tokens):
            return pos == 0 or pos == len(tokens)

        # Decode text before and after position
        text_before = encoding.decode(tokens[:pos])
        text_after = encoding.decode(tokens[pos:])
    else:
        raise ValueError("Either provide text_before/text_after or tokens/encoding/pos")

    # Integrity checks for structures
    checks = [
        # Don't cut inside a word
        not (text_before and text_after and text_before[-1].isalnum() and text_after[0].isalnum()),
        # Don't cut inside URL
        not is_inside_url(text_before, text_after),
        # Don't cut inside markdown link
        not is_inside_markdown_link(text_before, text_after),
        # Don't cut inside HTML tag
        not is_inside_html_tag(text_before, text_after),
        # Don't cut inside formula
        not is_inside_formula(text_before, text_after),
        # Don't cut inside code block
        not is_inside_code_block(text_before, text_after),
        # Don't cut inside lists (up to 2 levels of nesting)
        not is_inside_list(text_before, text_after),
        # Don't cut inside tables
        not is_inside_table(text_before, text_after),
    ]

    return all(checks)


def is_inside_url(text_before: str, text_after: str) -> bool:
    """Checks if we are inside a URL."""
    # Look for URL start in text_before
    url_pattern = r"https?://[^\s\)>\]]*$"
    if re.search(url_pattern, text_before):
        # Check if URL continues in text_after
        if text_after and re.match(r"^[^\s\)>\]]+", text_after):
            return True
    return False


def is_inside_markdown_link(text_before: str, text_after: str) -> bool:
    """Checks if we are inside a markdown link."""
    # Check [text](url) structure
    # Count unclosed square and round brackets
    open_square = text_before.count("[") - text_before.count("]")
    open_round = text_before.count("(") - text_before.count(")")

    # If there's an unclosed [ but no ], we're inside [text] part
    if open_square > 0:
        return True

    # If we just closed ] and ( follows, we're at the boundary
    if text_before.endswith("]") and text_after.startswith("("):
        return True

    # If inside (url) part after ](
    if "](h" in text_before[-10:] or (text_before.endswith("](") and open_round > 0):
        return True

    # Check if we're in the middle of ]( transition
    if text_before.endswith("]") and text_after and text_after[0] == "(":
        return True

    return False


def is_inside_html_tag(text_before: str, text_after: str) -> bool:
    """Checks if we are inside an HTML tag."""
    # Check if there's an unclosed <
    last_open = text_before.rfind("<")
    last_close = text_before.rfind(">")

    # If the last < comes after the last >, we're inside a tag
    return last_open > last_close


def is_inside_formula(text_before: str, text_after: str) -> bool:
    """Checks if we are inside a mathematical formula."""
    # Check $...$ and $$...$$
    # Count number of $ before position
    dollar_count = text_before.count("$")

    # If odd number of $, we're inside a formula
    return dollar_count % 2 == 1


def is_inside_code_block(text_before: str, text_after: str) -> bool:
    """Checks if we are inside a code block."""
    # Count triple quotes
    triple_quotes = text_before.count("```")

    # If odd number, we're inside a code block
    return triple_quotes % 2 == 1


def is_inside_list(text_before: str, text_after: str) -> bool:
    """
    Check if we're inside a list structure (up to 2 levels of nesting).

    Detects numbered lists (1., 2., a., b.) and bullet lists (-, *, +, •)
    with optional indentation for nested items.

    Args:
        text_before: Text before the boundary
        text_after: Text after the boundary

    Returns:
        True if boundary is inside a list structure
    """
    if not text_before or not text_after:
        return False

    # Get last few lines before cut
    lines_before = text_before.split("\n")[-3:]

    # Check for numbered lists
    # Level 1: "1. ", "2. ", etc.
    # Level 2: "  1. ", "  a. ", etc.
    numbered_patterns = [
        r"^\d+\.\s+",  # 1. First level
        r"^  \d+\.\s+",  # 1. Second level (indented)
        r"^  [a-z]\.\s+",  # a. Second level (indented)
        r"^\t\d+\.\s+",  # Tab indent
        r"^\t[a-z]\.\s+",  # Tab indent
    ]

    # Check for bullet lists
    # Level 1: "- ", "* ", "+ "
    # Level 2: "  - ", "  * ", etc.
    bullet_patterns = [
        r"^[-*+]\s+",  # - First level
        r"^  [-*+]\s+",  # - Second level
        r"^\t[-*+]\s+",  # Tab indent
        r"^•\s+",  # • bullet
        r"^  •\s+",  # Second level bullet
    ]

    all_patterns = numbered_patterns + bullet_patterns

    # Check if any recent line is a list item
    for line in lines_before:
        for pattern in all_patterns:
            if re.match(pattern, line):
                # We're after a list item, check if next line continues the list
                lines_after = text_after.split("\n")
                if lines_after:
                    first_line_after = lines_after[0]
                    for next_pattern in all_patterns:
                        if re.match(next_pattern, first_line_after):
                            return True  # We're between list items

    return False


def is_inside_table(text_before: str, text_after: str) -> bool:
    """
    Check if we're inside a table structure.

    Detects Markdown tables (|---|---| format) and HTML tables (<table>...</table>).

    Args:
        text_before: Text before the boundary
        text_after: Text after the boundary

    Returns:
        True if boundary is inside a table structure
    """
    if not text_before or not text_after:
        return False

    # Check for Markdown table
    # Look for table separator line like |---|---|
    lines_before = text_before.split("\n")[-5:]
    lines_after = text_after.split("\n")[:5]

    # Markdown table patterns
    table_separator = r"^\s*\|[\s\-:]+\|"  # |---|---| or | :--- | :---: |
    table_row = r"^\s*\|.*\|"  # | cell | cell |

    # Check if we have table structure around
    has_table_before = any(
        re.match(table_separator, line) or re.match(table_row, line) for line in lines_before
    )
    has_table_after = any(
        re.match(table_separator, line) or re.match(table_row, line) for line in lines_after
    )

    if has_table_before and has_table_after:
        return True

    # Check for HTML table
    # Don't cut if we're between <table> and </table>
    html_before = text_before[-200:] if len(text_before) > 200 else text_before
    # html_after not needed for this check

    # Simple check: more <table> than </table> before means we're inside
    tables_open_before = html_before.count("<table") - html_before.count("</table>")
    if tables_open_before > 0:
        return True

    return False


def evaluate_boundary_quality(
    text: Optional[str] = None,
    tokens: Optional[List[int]] = None,
    encoding=None,
    pos: Optional[int] = None,
    text_before: Optional[str] = None,
    text_after: Optional[str] = None,
) -> float:
    """
    Evaluates boundary quality (lower is better).

    If text_before/text_after are provided, uses them (cached mode).
    Otherwise, decodes from tokens (legacy mode).

    Args:
        text: Source text (legacy mode, unused)
        tokens: List of tokens (legacy mode)
        encoding: tiktoken encoding object (legacy mode)
        pos: Token position (legacy mode)
        text_before: Pre-decoded text before boundary (cached mode)
        text_after: Pre-decoded text after boundary (cached mode)

    Returns:
        Score where lower is better boundary
    """
    # Handle cached mode
    if text_before is not None and text_after is not None:
        # Get context from cached text
        context_before = text_before[-50:] if len(text_before) > 50 else text_before
        # context_after not used but available if needed in future
        # context_after = text_after[:50] if len(text_after) > 50 else text_after
    # Handle legacy mode
    elif tokens is not None and encoding is not None and pos is not None:
        if pos <= 0 or pos >= len(tokens):
            return 0.0  # Text boundaries are ideal

        # Decode context around boundary
        context_before = encoding.decode(tokens[max(0, pos - 10) : pos])
        # context_after not currently used in scoring
        # context_after = encoding.decode(tokens[pos : min(len(tokens), pos + 10)])
    else:
        raise ValueError("Either provide text_before/text_after or tokens/encoding/pos")

    score = 100.0  # Base score

    # Check various boundary types and assign scores
    # (lower score = better boundary)

    # Headers - best boundaries
    if re.search(r"</h[1-6]>\s*$", context_before, re.IGNORECASE):
        score = 1.0
    elif re.search(r"\n#{1,6}\s+.*$", context_before):
        score = 1.0
    elif re.search(r"\n(?:Глава|Chapter|Раздел)\s+.*$", context_before, re.IGNORECASE):
        score = 1.0

    # Double line break - good boundary
    elif context_before.endswith("\n\n"):
        score = 5.0

    # End of sentence
    elif re.search(r"[.!?]\s*$", context_before):
        score = 10.0

    # End of paragraph (single line break)
    elif context_before.endswith("\n"):
        score = 15.0

    # After comma or semicolon
    elif re.search(r"[,;]\s*$", context_before):
        score = 20.0

    # Between words (space)
    elif context_before.endswith(" "):
        score = 50.0

    return score

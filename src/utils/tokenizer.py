"""
Module for text tokenization and semantic boundary detection.

Uses the o200k_base codec, which is supported by all modern OpenAI models
"""

import re
from typing import List, Optional

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
    section_pattern = r"(?:^|\n)(?:Глава|Параграф|Часть|Chapter|Section|Раздел|Урок|Тема)\s+.*?(?=\n|$)"
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


def find_safe_token_boundary(
    text: str, tokens: List[int], encoding, target_token_pos: int, max_shift_tokens: int
) -> int:
    """
    Finds a safe boundary for cutting at the token level.

    Args:
        text: Source text
        tokens: List of tokens
        encoding: tiktoken encoding object
        target_token_pos: Target position in tokens
        max_shift_tokens: Maximum offset in tokens

    Returns:
        Safe position in tokens for cutting
    """
    # Define search range
    start_pos = max(0, target_token_pos - max_shift_tokens)
    end_pos = min(len(tokens), target_token_pos + max_shift_tokens)

    best_pos = target_token_pos
    best_score = float("inf")

    # Check each possible position in range
    for pos in range(start_pos, end_pos + 1):
        # Check that we're not cutting inside a structure
        if is_safe_cut_position(text, tokens, encoding, pos):
            # Evaluate the quality of this position
            score = evaluate_boundary_quality(text, tokens, encoding, pos)

            # Add penalty for distance from target
            distance_penalty = abs(pos - target_token_pos)
            total_score = score + distance_penalty * 0.1

            if total_score < best_score:
                best_score = total_score
                best_pos = pos

    return best_pos


def is_safe_cut_position(text: str, tokens: List[int], encoding, pos: int) -> bool:
    """
    Checks if it's safe to cut at the given token position.
    """
    if pos <= 0 or pos >= len(tokens):
        return pos == 0 or pos == len(tokens)

    # Decode text before and after position
    text_before = encoding.decode(tokens[:pos])
    text_after = encoding.decode(tokens[pos:])

    # Integrity checks for structures
    checks = [
        # Don't cut inside a word
        not (
            text_before
            and text_after
            and text_before[-1].isalnum()
            and text_after[0].isalnum()
        ),
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

    # If there's an unclosed [ and follows ](
    if open_square > 0 and "](h" in text_before[-20:] + text_after[:5]:
        return True

    # If inside (url) part
    if open_round > 0 and text_before.endswith("]("):
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


def evaluate_boundary_quality(
    text: str, tokens: List[int], encoding, pos: int
) -> float:
    """
    Evaluates boundary quality (lower is better).
    """
    if pos <= 0 or pos >= len(tokens):
        return 0.0  # Text boundaries are ideal

    # Decode context around boundary
    context_before = encoding.decode(tokens[max(0, pos - 10) : pos])
    context_after = encoding.decode(tokens[pos : min(len(tokens), pos + 10)])

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

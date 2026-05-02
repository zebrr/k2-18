# util_tokenizer.md

## Status: READY

Module for text tokenization and semantic boundary detection. Uses o200k_base codec (compatible with all modern OpenAI models). Supports hierarchical boundary search and safe cutting at token level.

## Public API

### count_tokens(text: str) -> int
Counts the number of tokens in text using o200k_base codec.
- **Input**: text (str) - input text for tokenization
- **Returns**: int - number of tokens in text
- **Raises**: ValueError - if text is not a string

### find_soft_boundary(text: str, target_pos: int, max_shift: int) -> Optional[int]
Finds the nearest semantic boundary in text for soft breaks using priority hierarchy.
- **Input**: 
  - text (str) - source text
  - target_pos (int) - target position in characters
  - max_shift (int) - maximum offset from target_pos
- **Returns**: Optional[int] - boundary position in characters or None if not found
- **Algorithm**: Uses weight system for selecting best boundary: score = weight × distance

#### Boundary hierarchy (from highest to lowest priority):
1. **section (weight 1)** - section boundaries:
   - HTML headers (`</h1-6>`)
   - Markdown headers (`#-####`)
   - Text headers (Глава, Параграф, Часть, Chapter, Section, Раздел, Урок, Тема)
2. **paragraph (weight 2)** - paragraph boundaries:
   - Double line breaks (`\n\n`)
   - End of code block (` ``` `)
   - End of formula block (`$$`)
   - HTML/Markdown links (`</a>`, `](url)`)
3. **sentence (weight 3)** - sentence boundaries:
   - End of sentence (`. ! ?` + space) with abbreviation checking
   - Semicolon (`;` + space)
4. **phrase (weight 4)** - phrase boundaries:
   - Comma (`,` + space)
   - Colon (`:` + space)
   - Dash (`—–-` + spaces)
5. **word (weight 5)** - word boundaries (fallback):
   - Any whitespace character

### find_safe_token_boundary(text: str, tokens: List[int], encoding, target_token_pos: int, max_shift_tokens: int) -> Tuple[int, str]
Finds a safe boundary for cutting at token level using smart candidate selection.
- **Input**:
  - text (str) - source text (unused, kept for compatibility)
  - tokens (List[int]) - list of tokens
  - encoding - tiktoken encoding object
  - target_token_pos (int) - target position in tokens
  - max_shift_tokens (int) - maximum offset in tokens
- **Returns**: Tuple[int, str] - (safe position in tokens, boundary_type)
- **Boundary types**: "markdown_header", "html_header", "text_header", "subheader", "paragraph", "code_block_end", "sentence", "line", "phrase", "word", "fallback", "empty", "edge", "none"
- **Algorithm**:
  - Decodes working range ONCE
  - Uses `find_boundary_candidates()` for smart candidate selection
  - Prioritizes boundaries BEFORE headers (not after)
  - Falls back to checking all positions if no candidates found
- **Optimization**: Smart candidate selection instead of checking all positions

### find_boundary_candidates(decoded_text: str, target_char_pos: int, max_char_shift: int) -> List[Tuple[int, str]]
Find boundary candidates with priorities. Key change: boundaries are now BEFORE headers.
- **Input**:
  - decoded_text (str) - the decoded text to search in
  - target_char_pos (int) - target position in characters
  - max_char_shift (int) - maximum offset from target_pos
- **Returns**: List of (position, boundary_type) tuples sorted by score
- **Priority levels**:
  1. BEFORE headers (markdown, HTML, text headers)
  2. BEFORE subheaders, AFTER paragraphs, code block ends
  3. AFTER sentences
  4. AFTER lines
  5. AFTER phrases
  6. Between words (fallback)
- **Scoring**: score = priority × 1000 + distance

### find_safe_token_boundary_with_fallback(text: str, tokens: List[int], encoding, target_token_pos: int, max_shift_tokens: int, max_tokens: int) -> int
Find safe boundary with smart fallback for large blocks.
- **Input**:
  - text (str) - source text (unused, kept for compatibility)
  - tokens (List[int]) - list of tokens
  - encoding - tiktoken encoding object
  - target_token_pos (int) - target position in tokens
  - max_shift_tokens (int) - maximum offset in tokens
  - max_tokens (int) - window size for calculating extended range
- **Returns**: int - safe position in tokens for cutting
- **Algorithm**:
  - First tries normal search via `find_safe_token_boundary()`
  - If no safe boundary found, expands search up to 30% of window size
  - Logs boundary type and shift for debugging
- **Use case**: Main entry point for slicer.py

## Helper Functions

### _build_token_char_mapping(tokens: List[int], encoding) -> dict
Build mapping from token index to character position.
- **Input**:
  - tokens (List[int]) - list of token IDs
  - encoding - tiktoken encoding object
- **Returns**: dict - mapping token index to character position
- **Algorithm**: Decodes prefixes to build accurate mapping

### _get_text_at_boundary(decoded_text: str, char_pos: int) -> tuple
Split decoded text at character position.
- **Input**:
  - decoded_text (str) - the decoded text
  - char_pos (int) - character position to split at
- **Returns**: tuple (text_before, text_after)

## Internal Methods

### is_safe_cut_position(text=None, tokens=None, encoding=None, pos=None, text_before=None, text_after=None) -> bool
Checks if it's safe to cut at given token position. Supports two modes:
- **Legacy mode**: Provide text, tokens, encoding, pos
- **Cached mode**: Provide text_before, text_after (for performance)
- Doesn't cut inside a word
- Doesn't cut inside URL
- Doesn't cut inside markdown link
- Doesn't cut inside HTML tag
- Doesn't cut inside formula
- Doesn't cut inside code block
- Doesn't cut inside lists (up to 2 levels of nesting)
- Doesn't cut inside tables (Markdown and HTML)

### evaluate_boundary_quality(text=None, tokens=None, encoding=None, pos=None, text_before=None, text_after=None) -> float
Evaluates boundary quality (lower value is better). Supports two modes:
- **Legacy mode**: Provide text, tokens, encoding, pos
- **Cached mode**: Provide text_before, text_after (for performance)
- Headers: score = 1.0
- Double line break: score = 5.0
- End of sentence: score = 10.0
- End of paragraph: score = 15.0
- After comma/semicolon: score = 20.0
- Between words: score = 50.0
- Other: score = 100.0

### is_inside_url(text_before: str, text_after: str) -> bool
Checks if position is inside URL.

### is_inside_markdown_link(text_before: str, text_after: str) -> bool
Checks if position is inside markdown link `[text](url)`.

### is_inside_html_tag(text_before: str, text_after: str) -> bool
Checks if position is inside HTML tag.

### is_inside_formula(text_before: str, text_after: str) -> bool
Checks if position is inside mathematical formula `$...$` or `$$...$$`.

### is_inside_code_block(text_before: str, text_after: str) -> bool
Checks if position is inside code block ` ```...``` `.

### is_inside_list(text_before: str, text_after: str) -> bool
Checks if position is inside a list structure (up to 2 levels of nesting).
- Detects numbered lists (1., 2., a., b.) and bullet lists (-, *, +, •)
- Supports indented nested items

### is_inside_table(text_before: str, text_after: str) -> bool
Checks if position is inside a table structure.
- Detects Markdown tables (|---|---| format)
- Detects HTML tables (<table>...</table>)

## Test Coverage

- **test_count_tokens**: 7 tests
  - test_basic_text
  - test_empty_string
  - test_russian_text
  - test_mixed_languages
  - test_special_characters
  - test_code_content
  - test_invalid_input_type

- **test_find_soft_boundary**: 14 tests
  - test_invalid_inputs
  - test_no_boundaries_found
  - test_markdown_headers
  - test_html_headers
  - test_text_headers_russian
  - test_text_headers_english
  - test_double_newlines
  - test_sentence_endings
  - test_code_blocks
  - test_formula_blocks
  - test_html_links
  - test_markdown_links
  - test_closest_boundary_selection
  - test_boundary_within_range

- **test_fixtures**: 4 tests
  - test_markdown_fixture
  - test_html_fixture
  - test_mixed_fixture
  - test_token_consistency

- **TestNewBoundaryFunctions**: 9 tests (added in SLICER_REFACTOR_03)
  - test_is_inside_list_numbered
  - test_is_inside_list_bullet
  - test_is_inside_list_nested
  - test_is_inside_table_markdown
  - test_is_inside_table_html
  - test_find_boundary_candidates_headers
  - test_find_boundary_candidates_priorities
  - test_find_safe_token_boundary_with_fallback_normal
  - test_find_safe_token_boundary_with_fallback_returns_int

## Dependencies
- **Standard Library**: re, logging, typing
- **External**: tiktoken
- **Internal**: None

## Performance Notes
- Uses regular expressions for boundary search - can be slow on very large texts
- find_soft_boundary performs multiple regex searches by priority hierarchy
- Selection algorithm optimized for educational content
- tiktoken is fast enough for processing large texts
- **Decode caching optimization**: find_safe_token_boundary now uses single decode operation (4000x speedup)
- Helper functions enable efficient text manipulation without repeated decoding

## Usage Examples
```python
from src.utils.tokenizer import count_tokens, find_soft_boundary, find_safe_token_boundary
import tiktoken

# Count tokens
tokens = count_tokens("Hello world!")  # returns token count

# Find soft boundary with priority hierarchy
boundary = find_soft_boundary(text, target_pos=1000, max_shift=100)
if boundary:
    # Found optimal boundary at position boundary
    left_part = text[:boundary]
    right_part = text[boundary:]

# Safe cutting at token level
encoding = tiktoken.get_encoding("o200k_base")
tokens = encoding.encode(text)
safe_pos = find_safe_token_boundary(
    text, tokens, encoding, 
    target_token_pos=40000, 
    max_shift_tokens=500
)
# Cut at safe position
left_tokens = tokens[:safe_pos]
right_tokens = tokens[safe_pos:]
```
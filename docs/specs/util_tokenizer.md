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

### find_safe_token_boundary(text: str, tokens: List[int], encoding, target_token_pos: int, max_shift_tokens: int) -> int
Finds a safe boundary for cutting at token level.
- **Input**:
  - text (str) - source text
  - tokens (List[int]) - list of tokens
  - encoding - tiktoken encoding object
  - target_token_pos (int) - target position in tokens
  - max_shift_tokens (int) - maximum offset in tokens
- **Returns**: int - safe position in tokens for cutting
- **Algorithm**: Checks each position for safety and selects best by quality

## Internal Methods

### is_safe_cut_position(text: str, tokens: List[int], encoding, pos: int) -> bool
Checks if it's safe to cut at given token position.
- Doesn't cut inside a word
- Doesn't cut inside URL
- Doesn't cut inside markdown link
- Doesn't cut inside HTML tag
- Doesn't cut inside formula
- Doesn't cut inside code block

### evaluate_boundary_quality(text: str, tokens: List[int], encoding, pos: int) -> float
Evaluates boundary quality (lower value is better).
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

## Dependencies
- **Standard Library**: re, logging, typing
- **External**: tiktoken
- **Internal**: None

## Performance Notes
- Uses regular expressions for boundary search - can be slow on very large texts
- find_soft_boundary performs multiple regex searches by priority hierarchy
- Selection algorithm optimized for educational content
- tiktoken is fast enough for processing large texts

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
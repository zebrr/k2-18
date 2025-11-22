# cli_slicer.md

## Status: READY

CLI utility for splitting educational texts into slices. Reads files from /data/raw/, applies preprocessing and cuts them into fragments of fixed size with consideration for semantic boundaries.

## CLI Interface

### Usage
```bash
python -m src.slicer
```

### Input Directory
- **Source**: `/data/raw/` - source corpus files
- **Supported formats**: .json, .txt, .md, .html (configurable)
- **Processing order**: lexicographic (for determinism)
- **Case handling**: file search considers extension case (finds both .txt and .TXT)

### Output Directory  
- **Target**: `/data/staging/` - slices in *.slice.json format
- **Naming**: slice_001.slice.json, slice_002.slice.json, etc.

## Core Algorithm

### File Processing Pipeline
1. **Environment setup**: Load environment variables via dotenv
2. **File Discovery**: search files by allowed_extensions in lexicographic order (case-aware)
3. **Preprocessing**: Unicode normalization NFC + remove <script>/<style> tags via BeautifulSoup
4. **Validation**: check for emptiness after preprocessing
5. **Slicing**: cut with sliding window considering soft boundaries
6. **Output**: save in JSON format with validation

### Sliding Window Algorithm
- **Window size**: max_tokens (configurable)
- **Incremental processing**: Tokenizes only current window + buffer (not entire file)
- **Window buffer**: (max_tokens + soft_boundary_max_shift * 2) * 10 characters
- **Soft boundaries**: search for semantic boundaries within soft_boundary_max_shift tokens
- **Boundary detection**: uses optimized find_safe_token_boundary with decode caching
- **Token calculation**: precise counting via tiktoken o200k_base
- **No overlap**: slices are sequential without overlapping tokens
- **Performance**: Handles files of any size without memory issues

### Boundary Rules
- Next slice starts where previous ends: slice_token_start(next) = slice_token_end(current)
- File boundaries: hard boundaries between files, each file processed independently
- Last fragment: processed as-is regardless of size

## Terminal Output

### Console Encoding
UTF-8 console encoding is automatically configured on Windows via `setup_console_encoding()`.

### Log Format
The utility uses standard logging with format:
```
[HH:MM:SS] LEVEL    | Message
```

### Standard Output Examples
```
[10:30:00] INFO     | Starting slicer.py
[10:30:00] INFO     | Found 3 files for processing
[10:30:00] INFO     | Processing file: chapter1.md (1.25MB)
[10:30:00] INFO     | Processing file: chapter1.md (1.3MB, ~325,000 tokens)
[10:30:00] INFO     | Estimated slices: ~9
[10:30:00] INFO     | Creating slice_001 (tokens 0-5000) [0%]
[10:30:01] INFO     | Soft boundary found: shift +15 tokens
[10:30:01] INFO     | Creating slice_002 (tokens 5000-10000) [12%]
[10:30:01] INFO     | Slice saved: data\staging\slice_001.slice.json
[10:30:01] INFO     | Completed: 9 slices from chapter1.md
[10:30:02] INFO     | Processing completed: 8 slices saved in data\staging
[10:30:02] INFO     | Terminating with code: SUCCESS (0)
```

**Note**: Token ranges use Python convention (exclusive end), so slice ending at 5000 contains tokens 0-4999, and next slice starts at 5000.

### Debug Output Examples
In DEBUG mode, boundary type is shown:
```
[10:30:01] INFO     | Soft boundary found: shift -23 tokens
[10:30:01] DEBUG    | Boundary type: paragraph
[10:30:01] INFO     | Soft boundary found: shift +8 tokens
[10:30:01] DEBUG    | Boundary type: sentence
[10:30:01] INFO     | Soft boundary found: shift +12 tokens
[10:30:01] DEBUG    | Boundary type: markdown_header
```

### Warning and Error Examples
```
[10:30:00] WARNING  | No files found for processing in data\raw
[10:30:00] WARNING  | Supported extensions: ['json', 'txt', 'md', 'html']
[10:30:00] WARNING  | Unsupported file skipped: image.png
[10:30:00] ERROR    | Configuration error: slicer.max_tokens must be a positive integer
[10:30:00] ERROR    | Input data error in file empty.md: Empty file detected: empty.md. Please remove empty files from /data/raw/
[10:30:00] ERROR    | I/O error when processing file.txt: Failed to save slice slice_001
```

## Public Functions

### create_slug(filename: str) -> str
Creates slug from filename.
- **Rules**: remove extension, transliterate Cyrillic (unidecode), lowercase, spaces → "_"
- **Examples**: "Алгоритмы.txt" → "algoritmy", "My Course 1.md" → "my_course_1"

### preprocess_text(text: str) -> str
Applies preprocessing to text.
- **Steps**: Unicode normalization NFC, remove <script>/<style> via BeautifulSoup
- **Preservation**: other content remains unchanged
- **Raises**: ValueError if input parameter is not a string

### validate_config_parameters(config: Dict[str, Any]) -> None
Validates slicer configuration parameters.
- **Checks**: required parameters, types, ranges
- **Raises**: ValueError with detailed error description

### slice_text_with_window(text: str, max_tokens: int, soft_boundary: bool, soft_boundary_max_shift: int, file_name: Optional[str] = None) -> List[Tuple[str, int, int]]
Main text slicing algorithm with incremental tokenization and improved progress reporting.
- **Returns**: list of tuples (slice_text, slice_token_start, slice_token_end)
- **Features**:
  - Incremental tokenization (processes only current window + buffer)
  - Soft boundary detection via `find_safe_token_boundary_with_fallback`
  - Smart candidate selection (prioritizes boundaries BEFORE headers)
  - Fallback for large blocks (expands search up to 30% of window)
  - Protection for lists and tables
  - Sequential slicing without overlaps
  - Memory-efficient for large files
  - Improved progress logging with file_name parameter
- **soft_boundary_max_shift**: interpreted directly as tokens (no conversion needed)
- **Imports**: uses find_safe_token_boundary_with_fallback from utils.tokenizer

### load_and_validate_file(file_path: Path, allowed_extensions: List[str]) -> str
Loads file with automatic encoding detection.
- **Encodings**: utf-8 → cp1251 → latin1 (fallback chain)
- **Validation**: extension check, emptiness check after preprocessing
- **Raises**: InputError for empty files or unsupported extensions

### process_file(file_path: Path, config: Dict[str, Any], global_slice_counter: int) -> Tuple[List[Dict[str, Any]], int]
Processes one file and returns list of slices.
- **Input**: file path, configuration, global slice counter
- **Returns**: tuple (slice list, updated counter)
- **Workflow**: load → create slug → slice → form slice objects
- **Raises**: InputError, RuntimeError, IOError

### save_slice(slice_data: Dict[str, Any], output_dir: Path) -> None
Saves slice to JSON file.
- **Filename**: {slice_id}.slice.json
- **Encoding**: UTF-8 with ensure_ascii=False
- **Raises**: IOError on write errors

### setup_logging(log_level: str = "info") -> None
Sets up logging for slicer.
- **Levels**: debug, info, warning, error
- **Format**: [HH:MM:SS] LEVEL | Message
- **Handler**: console output to stdout

## Output Format

### Slice JSON Structure
```json
{
  "id": "slice_042",
  "order": 42,
  "source_file": "chapter03.md", 
  "slug": "chapter03",
  "text": "processed content...",
  "slice_token_start": 52000,
  "slice_token_end": 92000
}
```

### ID Generation
- **Pattern**: slice_{order:03d} (slice_001, slice_002, ...)
- **Uniqueness**: global counter across all files
- **Deterministic**: repeated runs produce identical IDs

## Configuration

### Required Parameters (slicer section)
- **max_tokens** (int, >0) - window size in tokens
- **soft_boundary** (bool) - use soft boundaries
- **soft_boundary_max_shift** (int, ≥0) - maximum shift for boundary search (in tokens)
- **tokenizer** (str, ="o200k_base") - tokenizer
- **allowed_extensions** (list, non-empty) - allowed file extensions
- **log_level** (str) - logging level (debug/info/warning/error)

### Validation Rules
- max_tokens must be positive integer
- soft_boundary_max_shift must be non-negative integer
- allowed_extensions non-empty list

## Error Handling & Exit Codes

### Custom Exceptions
- **InputError** - special exception for input data errors (empty files, unsupported extensions)

### Exit Codes
- **0 (EXIT_SUCCESS)** - successful execution
- **1 (EXIT_CONFIG_ERROR)** - configuration errors  
- **2 (EXIT_INPUT_ERROR)** - empty files, unsupported extensions
- **3 (EXIT_RUNTIME_ERROR)** - processing errors
- **5 (EXIT_IO_ERROR)** - file write errors, directory access issues

### Error Types
- **InputError** - empty files, unsupported extensions
- **ValueError** - invalid configuration parameters
- **IOError** - problems writing to /data/staging/
- **RuntimeError** - unexpected processing errors

### Exit Code Logging
Uses `log_exit()` function to log exit code in readable format.

## Boundary Cases

### Empty Files
EXIT_INPUT_ERROR with message: "Empty file detected: {filename}. Please remove empty files from /data/raw/"

### Unsupported Files  
Warning: "Unsupported file skipped: {filename}"

### Last Fragment Handling
- Separate slice created regardless of size
- No merging with previous slices

### No Files Found
When no files to process:
- Warns about supported extensions
- Returns EXIT_SUCCESS (not considered an error)


## Test Coverage

- **test_create_slug**: 6 tests
  - test_cyrillic_transliteration
  - test_english_with_spaces  
  - test_hyphens_preserved
  - test_extension_removal
  - test_complex_filename
  - test_special_characters

- **test_preprocess_text**: multiple tests
  - test_unicode_normalization
  - test_script_tag_removal (via BeautifulSoup)
  - test_style_tag_removal (via BeautifulSoup)
  - test_plain_text_unchanged
  - test_invalid_input_type

- **test_validate_config_parameters**: validation tests
  - test_valid_config
  - test_missing_parameters
  - test_invalid_types

- **test_slice_text_with_window**: slicing algorithm tests
  - test_single_slice
  - test_multiple_slices_no_overlap
  - test_soft_boundary_detection
  - test_no_token_loss

- **test_process_file**: file processing tests
  - test_successful_processing
  - test_empty_file_error
  - test_unsupported_extension

- **integration tests**: full pipeline tests
- **large file tests**: performance on large files

## Dependencies
- **Standard Library**: argparse, json, logging, sys, unicodedata, pathlib, typing
- **External**: unidecode, beautifulsoup4 (bs4), tiktoken, python-dotenv
- **Internal**: utils.config, utils.tokenizer (find_safe_token_boundary_with_fallback), utils.validation, utils.exit_codes, utils.console_encoding

## Performance Optimizations
- **Incremental tokenization**: Processes files in windows to avoid tokenizing entire file
- **Decode caching**: Single decode operation per boundary search (vs 4000 in v1)
- **Memory usage**: O(window_size) instead of O(file_size)
- **Scalability**: Can process multi-GB files without hanging

## Performance Notes
- **Deterministic processing**: lexicographic file order
- **Accurate token counting**: via tiktoken o200k_base
- **Efficient large file handling**: through incremental tokenization
- **Soft boundary search**: using token-based max_shift directly
- **Cross-platform file search**: case-aware extension matching
- **Environment variables**: loaded via dotenv for flexibility

## Usage Examples
```bash
# Simple run (uses config.toml)
python -m src.slicer

# Check results
dir data\staging\
# slice_001.slice.json
# slice_002.slice.json
# ...

# File structure before:
/data/raw/
  chapter1.md
  lesson2.txt
  exercises.json
  README.TXT    # will be processed (case-insensitive extension)

# File structure after:  
/data/staging/
  slice_001.slice.json  # from chapter1.md
  slice_002.slice.json  # from chapter1.md (continuation)
  slice_003.slice.json  # from lesson2.txt
  slice_004.slice.json  # from exercises.json
  slice_005.slice.json  # from README.TXT
  
# View slice
type data\staging\slice_001.slice.json

# Check file encodings
python -m src.slicer
# Automatically handles different encodings (utf-8, cp1251, latin1)

# Debug mode for soft boundary analysis
# Set log_level = "debug" in config.toml
python -m src.slicer
# [10:30:01] INFO     | Soft boundary found: shift +15 tokens
# [10:30:01] DEBUG    | Boundary type: double line break
```
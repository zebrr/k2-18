# viz_graph2viewer.md

## Status: ACTIVE

Module for generating HTML viewer for methodologists to explore knowledge graph in detail.
Supports both production and test modes with tabular data exploration interface.

## Quick Start

```bash
# Activate virtual environment first
source .venv/bin/activate

# Generate production viewer from enriched graph data
python -m viz.graph2viewer

# Generate test viewer for development
python -m viz.graph2viewer --test

# Open generated HTML in browser
open viz/data/out/knowledge_graph_viewer.html  # production
open viz/data/out/test_viewer.html            # test mode
```

## Module Purpose

The `graph2viewer.py` utility generates a single HTML file containing an interactive viewer for detailed exploration of the knowledge graph. It processes enriched graph data with metrics, combines it with concept dictionary, and produces a self-contained HTML viewer with a three-column layout for methodical examination of graph structure and content. Unlike the visualization tool (graph2html.py), this focuses on detailed tabular exploration rather than graph visualization.

## CLI Interface

### Usage
```bash
# Production mode (default) - uses production data
python -m viz.graph2viewer

# Test mode - uses test data for development
python -m viz.graph2viewer --test
```

### Command Line Arguments
- `--test` - Use test data from `/viz/data/test/` for development and testing

### Input Directory/Files

#### Production Mode (default)
- **Source**: `/viz/data/out/`
  - `LearningChunkGraph_wow.json` - Enriched graph with metrics
  - `ConceptDictionary_wow.json` - Concepts with mention index
- **Templates**: `/viz/templates/viewer/`
  - `index.html` - Main HTML template (Jinja2)
  - `viewer_styles.css` - CSS styles for viewer interface
- **Vendor Libraries**: `/viz/vendor/` (if embed_libraries=true)
  - Formatting libraries only (marked.js, highlight.js, MathJax)
- **Static Assets**: `/viz/static/viewer/`
  - `viewer_core.js` - Main controller module
  - `node_explorer.js` - Node exploration functionality
  - `edge_inspector.js` - Edge inspection functionality
  - `navigation_history.js` - Navigation history management
  - `formatters.js` - Text formatting utilities
  - `search_filter.js` - Search and filter functionality

#### Test Mode (`--test`)
- **Source**: `/viz/data/test/`
  - `tiny_html_data.json` - Test graph data (26 nodes)
  - `tiny_html_concepts.json` - Test concept dictionary

### Output Directory/Files
- **Production**: `/viz/data/out/knowledge_graph_viewer.html` - Standalone HTML viewer
- **Test Mode**: `/viz/data/out/test_viewer.html` - Test viewer with same structure

## Terminal Output

### Output Format
Utility uses timestamped structured output:
```
[HH:MM:SS] LEVEL - Message
```

### Progress Messages
```
[10:30:00] INFO - ================================================================================
[10:30:00] INFO - Starting HTML viewer generation (mode: production)
[10:30:00] INFO - Configuration loaded
[10:30:01] INFO - Loaded concepts: 150 concepts
[10:30:01] INFO - Loaded production graph: 543 nodes, 892 edges
[10:30:01] INFO - Loaded vendor file: marked.min.js (89234 bytes)
[10:30:01] INFO - Loaded vendor file: highlight.min.js (45234 bytes)
[10:30:01] INFO - Loaded vendor file: mathjax-tex-mml-chtml.js (234567 bytes)
[10:30:01] INFO - Loaded viewer_core.js (2345 bytes)
[10:30:01] INFO - Loaded node_explorer.js (1234 bytes)
[10:30:02] INFO - HTML generated successfully
[10:30:02] INFO - HTML saved to: /viz/data/out/knowledge_graph_viewer.html
[10:30:02] INFO - File size: 0.45 MB
[10:30:02] INFO - HTML viewer generation completed successfully (mode: production)

[SUCCESS] HTML viewer created: /viz/data/out/knowledge_graph_viewer.html
```

### Warning Messages
```
[10:30:01] WARNING - Vendor file marked.min.js not found, downloading from CDN
[10:30:02] INFO - Downloaded marked.min.js from CDN (89234 bytes)
[10:30:03] WARNING - viewer_styles.css not found, using empty styles
[10:30:03] WARNING - node_explorer.js not found - feature disabled
```

### Error Messages
```
[10:30:00] ERROR - Failed to load config: File not found
[10:30:00] ERROR - Graph file not found: /viz/data/out/LearningChunkGraph_wow.json
[10:30:00] ERROR - Invalid JSON: Expecting value: line 1 column 1 (char 0)
[10:30:00] ERROR - Template rendering failed: 'graph_data' is undefined
[10:30:00] ERROR - Failed to save HTML: Permission denied
```

## Core Algorithm

1. **Initialization**
   - Parse command line arguments (test mode flag)
   - Setup logging to `/viz/logs/graph2viewer.log`
   - Load configuration from `/viz/config.toml`

2. **Data Loading**
   - Determine data paths based on mode (production/test)
   - Load enriched graph JSON with metrics
   - Load concept dictionary JSON
   - Handle missing concepts in test mode with empty stub
   - Extract real node and edge counts for header display

3. **Template Preparation**
   - Initialize Jinja2 environment with `/viz/templates/viewer/`
   - Load `index.html` template
   - Load `viewer_styles.css` for embedding

4. **Library Management**
   - Check `embed_libraries` configuration
   - Load formatting libraries only (marked, highlight, MathJax)
   - No Cytoscape.js or graph-related libraries
   - If embedding:
     - Load each vendor file or download from CDN
     - Concatenate all JavaScript/CSS content
   - If CDN mode:
     - Generate script/link tags with CDN URLs

5. **Module Loading**
   - Load all viewer JavaScript modules from `/viz/static/viewer/`
   - Include modules in template context for embedding
   - Handle missing modules with warnings (non-critical)

6. **HTML Generation**
   - Prepare template context with all data and configuration
   - Include real node/edge counts
   - Render Jinja2 template with context
   - Optionally minify JSON data
   - Optionally minify final HTML

7. **Output**
   - Create output directory if needed
   - Save HTML to file
   - Log file size and completion

## Public API

### setup_logging(log_file: Path) -> logging.Logger
Setup logging configuration with file and console handlers.
- **Input**: 
  - log_file (Path) - Path to log file
- **Returns**: Logger - Configured logger instance
- **Side effects**: Creates log directory if not exists

### get_library_content(lib_name: str, lib_path: Path, cdn_url: str, logger: logging.Logger) -> str
Get library content from local file or CDN with automatic fallback.
- **Input**:
  - lib_name (str) - Library filename for logging
  - lib_path (Path) - Local path to library file
  - cdn_url (str) - CDN URL for fallback
  - logger (Logger) - Logger instance
- **Returns**: str - Library content as string
- **Raises**:
  - RuntimeError - If library cannot be loaded from file or CDN
- **Note**: Requires `requests` library for CDN fallback

### load_graph_data(data_dir: Path, logger: logging.Logger, test_mode: bool = False) -> tuple[Dict, Dict]
Load enriched graph and concept data from JSON files with real counts.
- **Input**:
  - data_dir (Path) - Base data directory
  - logger (Logger) - Logger instance
  - test_mode (bool) - Use test data if True
- **Returns**: tuple[Dict, Dict] - (graph_data, concepts_data)
- **Raises**: 
  - SystemExit(EXIT_INPUT_ERROR) - Missing required files
  - SystemExit(EXIT_IO_ERROR) - File read or JSON parse errors
- **Note**: Returns actual node/edge counts from loaded data

### minify_json_data(data: Dict, minify: bool) -> str
Convert data to JSON string, optionally minified.
- **Input**:
  - data (Dict) - Data to serialize
  - minify (bool) - If True, produce compact JSON
- **Returns**: str - JSON string
- **Note**: Preserves Unicode characters without escaping

### load_vendor_content(viz_dir: Path, file_list: list, logger: logging.Logger) -> str
Load and concatenate vendor files with CDN fallback.
- **Input**:
  - viz_dir (Path) - Viz module directory
  - file_list (list) - List of relative file paths
  - logger (Logger) - Logger instance
- **Returns**: str - Concatenated content with file markers
- **Note**: Non-critical for viewer functionality

### generate_script_tags(file_list: list, embed: bool) -> str
Generate script tags for JavaScript libraries.
- **Input**:
  - file_list (list) - List of JavaScript file paths
  - embed (bool) - If True, return empty (content embedded)
- **Returns**: str - HTML script tags or empty string

### generate_style_tags(file_list: list, embed: bool) -> str
Generate link tags for CSS libraries.
- **Input**:
  - file_list (list) - List of CSS file paths
  - embed (bool) - If True, return empty (content embedded)
- **Returns**: str - HTML link tags or empty string

### generate_html(graph_data: Dict, concepts_data: Dict, config: Dict, viz_dir: Path, logger: logging.Logger, test_mode: bool = False) -> str
Generate the complete HTML file for viewer.
- **Input**:
  - graph_data (Dict) - Enriched graph with metrics
  - concepts_data (Dict) - Concept dictionary
  - config (Dict) - Full configuration dictionary
  - viz_dir (Path) - Viz module directory
  - logger (Logger) - Logger instance
  - test_mode (bool) - Test mode flag
- **Returns**: str - Complete HTML document
- **Algorithm**:
  1. Setup Jinja2 environment and load templates
  2. Load CSS from templates/viewer/viewer_styles.css
  3. Extract text formatting configuration
  4. Calculate real node/edge counts from data
  5. Process vendor files (formatting libraries only)
  6. Load viewer JavaScript modules
  7. Prepare template context with all data
  8. Render template and return HTML
- **Raises**:
  - SystemExit(EXIT_IO_ERROR) - Template loading errors
  - SystemExit(EXIT_RUNTIME_ERROR) - Template rendering errors

### minify_html_content(html: str, config: Dict, logger: logging.Logger) -> str
Minify HTML if configured.
- **Input**:
  - html (str) - HTML content to minify
  - config (Dict) - Configuration with graph2html.minify_html
  - logger (Logger) - Logger instance
- **Returns**: str - Minified or original HTML
- **Note**: Falls back to original on minification errors

### save_html(html: str, output_path: Path, logger: logging.Logger) -> None
Save HTML to file with size logging.
- **Input**:
  - html (str) - HTML content to save
  - output_path (Path) - Output file path
  - logger (Logger) - Logger instance
- **Raises**:
  - SystemExit(EXIT_IO_ERROR) - File write errors
- **Side effects**: Creates parent directories if needed

### main() -> int
Main entry point with command line argument parsing.
- **Returns**: int - Exit code (EXIT_SUCCESS on success)
- **Side effects**: 
  - Creates log file
  - Generates and saves HTML file
  - Prints completion message to stdout

## Configuration

Uses `/viz/config.toml`:

### [graph2html] section
- **minify_json** (bool, default=true) - Minify embedded JSON data
- **minify_html** (bool, default=true) - Minify final HTML output
- **embed_libraries** (bool, default=true) - Embed vs CDN mode

### [text_formatting] section
- **enable_markdown** (bool, default=true) - Enable markdown parsing
- **enable_code_highlighting** (bool, default=true) - Enable code highlighting
- **enable_math** (bool, default=true) - Enable math formula rendering
- **math_renderer** (str, default="mathjax") - Math rendering engine

## Template System

### Template Files

#### `/viz/templates/viewer/index.html`
Main Jinja2 template that structures the HTML document with three-column layout:
- Header with K2-18 branding and real node/edge counts
- Navigation history section
- Three columns (20%-40%-40%):
  - Column A: Search and node selection
  - Column B: Active node details
  - Column C: Related node details

#### `/viz/templates/viewer/viewer_styles.css`
CSS styles for three-column layout and viewer interface.

### Template Context Variables

```python
{
    # Data (JSON strings)
    "graph_data_json": str,        # Minified/formatted graph JSON
    "concepts_data_json": str,     # Minified/formatted concepts JSON
    
    # Title and counts
    "title": str,                  # Graph title from metadata
    "node_count": int,             # Real node count
    "edge_count": int,             # Real edge count
    
    # Configuration
    "text_formatting": {           # From [text_formatting] section
        "enable_markdown": bool,
        "enable_code_highlighting": bool,
        "enable_math": bool,
        "math_renderer": str
    },
    
    # Library content (when embed_libraries=true)
    "embed_libraries": bool,       # Embedding mode flag
    "vendor_js_content": str,      # Concatenated JS libraries
    "vendor_css_content": str,     # Concatenated CSS libraries
    "styles_content": str,         # viewer_styles.css content
    
    # Viewer modules
    "viewer_core_content": str,           # viewer_core.js content
    "node_explorer_content": str,         # node_explorer.js content
    "edge_inspector_content": str,        # edge_inspector.js content
    "navigation_history_content": str,    # navigation_history.js content
    "formatters_content": str,            # formatters.js content
    "search_filter_content": str,         # search_filter.js content
    
    # CDN tags (when embed_libraries=false)
    "script_tags": str,            # HTML script tags for CDN
    "link_tags": str               # HTML link tags for CDN
}
```

## Frontend Architecture

For detailed information about JavaScript modules, DOM structure, and frontend implementation, see `viz_graph2viewer_frontend.md`.

## Error Handling & Exit Codes

### Exit Codes
- **0 (SUCCESS)** - HTML generated successfully
- **1 (CONFIG_ERROR)** - Configuration file errors
- **2 (INPUT_ERROR)** - Missing or invalid input files
- **3 (RUNTIME_ERROR)** - Template rendering errors
- **5 (IO_ERROR)** - File system or network errors

### Error Types
- **Configuration Errors** - Missing or invalid config.toml
- **Input Errors** - Missing graph/concept files, invalid JSON
- **Template Errors** - Missing templates, render failures
- **Library Errors** - Non-critical, warnings only
- **IO Errors** - File write failures, network issues

### Boundary Cases
- **Missing concept dictionary in test mode** → Use empty stub
- **Missing vendor files** → Try CDN fallback, warn if fails
- **Missing viewer modules** → Log warning, continue (non-critical)
- **HTML minification failure** → Use non-minified output
- **Large graphs (>10000 nodes)** → May impact initial load time

## Dependencies

### Python Dependencies
- **Standard Library**: argparse, json, logging, sys, pathlib
- **External**: 
  - Jinja2>=3.1.0 - Template engine (required)
  - minify-html>=0.16.0 - HTML minification (required)
  - requests - CDN fallback (optional)
- **Internal**: 
  - src.utils.config - Configuration loading
  - src.utils.console_encoding - Console encoding setup
  - src.utils.exit_codes - Standardized exit codes

### JavaScript Libraries (Runtime)
- **marked.min.js** - Markdown parser (optional)
- **highlight.min.js** - Code syntax highlighting (optional)
- **mathjax-tex-mml-chtml.js** - Math formula rendering (optional)

Note: No Cytoscape.js or graph visualization libraries needed.

## Performance Notes

- **Memory**: ~50MB for 500 nodes, ~200MB for 5000 nodes
- **Generation Speed**: 1-2 seconds for production graph
- **File Sizes**:
  - Test mode: 200-300 KB
  - Production embedded: 2-5 MB
  - Production CDN: 0.5-1 MB
- **Browser Performance**: Instant load for tabular data

## Test Coverage

Module tested by `/tests/viz/test_graph2viewer.py`:
- **Coverage**: 11 test cases
- **setup_logging**: Logger creation and configuration
- **load_graph_data**: Production/test modes, missing files, invalid JSON
- **minify_json_data**: Minification, Unicode handling
- **generate_html**: Template rendering, missing modules
- **save_html**: File creation, directory handling
- **main**: Production/test modes, error handling


## Usage Examples

### Basic Production Build
```bash
# Generate production viewer
python -m viz.graph2viewer

# Output
[SUCCESS] HTML viewer created: /viz/data/out/knowledge_graph_viewer.html
```

### Test Mode
```bash
# Generate test viewer
python -m viz.graph2viewer --test

# Output
[SUCCESS] HTML viewer created: /viz/data/out/test_viewer.html
```


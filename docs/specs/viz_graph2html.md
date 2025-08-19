# viz_graph2html.md

## Status: READY

Module for generating standalone HTML visualization of enriched knowledge graph.
Supports both production and test modes with embedded or CDN-linked libraries.

## Module Purpose

The `graph2html.py` utility generates a single HTML file containing an interactive visualization of the knowledge graph. It processes enriched graph data with metrics, combines it with concept dictionary, and produces a self-contained HTML visualization using Cytoscape.js. The module supports flexible library loading strategies (embedded vs CDN) and includes test mode for development.

## CLI Interface

### Usage
```bash
# Production mode (default) - uses production data
python -m viz.graph2html

# Test mode - uses test data for development
python -m viz.graph2html --test
```

### Command Line Arguments
- `--test` - Use test data from `/viz/data/test/` for development and testing

### Input Directory/Files

#### Production Mode (default)
- **Source**: `/viz/data/out/`
  - `LearningChunkGraph_wow.json` - Enriched graph with metrics
  - `ConceptDictionary_wow.json` - Concepts with mention index
- **Templates**: `/viz/templates/`
  - `index.html` - Main HTML template (Jinja2)
  - `styles.css` - CSS styles for visualization
- **Vendor Libraries**: `/viz/vendor/` (if embed_libraries=true)
  - JavaScript libraries (Cytoscape.js and extensions)
  - CSS libraries (navigator styles)
- **Static Assets**: `/viz/static/`
  - `edge_styles.js` - Edge styling definitions for all 9 types
  - `animation_controller.js` - Animation sequences controller
  - `graph_core.js` - Core graph initialization module
  - `ui_controls.js` - UI controls and user interactions
  - `debug_helpers.js` - Debug utilities (test mode only)

#### Test Mode (`--test`)
- **Source**: `/viz/data/test/`
  - `tiny_html_data.json` - Test graph data (26 nodes)
  - `tiny_html_concepts.json` - Test concept dictionary
- **Additional**: `/viz/static/debug_helpers.js` - Debug utilities (auto-included in test mode)

### Output Directory/Files
- **Production**: `/viz/data/out/knowledge_graph.html` - Standalone HTML visualization
- **Test Mode**: `/viz/data/out/test_graph.html` - Test HTML visualization

## Terminal Output

### Output Format
Utility uses timestamped structured output:
```
[HH:MM:SS] LEVEL - Message
```

### Progress Messages
```
[10:30:00] INFO - ================================================================================
[10:30:00] INFO - Starting HTML generation (mode: production)
[10:30:00] INFO - Configuration loaded
[10:30:01] INFO - Loaded concepts: 150 concepts
[10:30:01] INFO - Loaded production graph: 543 nodes, 892 edges
[10:30:01] INFO - Loaded vendor file: cytoscape.min.js (892341 bytes)
[10:30:01] INFO - Loaded vendor file: layout-base.js (45234 bytes)
[10:30:01] INFO - Loaded vendor file: cose-base.js (23456 bytes)
[10:30:01] INFO - Loaded vendor file: cytoscape-cose-bilkent.js (67890 bytes)
[10:30:02] INFO - Loaded graph_core.js (12345 bytes)
[10:30:02] INFO - HTML generated successfully
[10:30:02] INFO - HTML minification: 1,234,567 → 987,654 bytes
[10:30:02] INFO - Saved: 246,913 bytes (20.0%)
[10:30:02] INFO - HTML saved to: /viz/data/out/knowledge_graph.html
[10:30:02] INFO - File size: 0.94 MB
[10:30:02] INFO - HTML generation completed successfully (mode: production)

✓ HTML visualization created: /viz/data/out/knowledge_graph.html
```

### Warning Messages
```
[10:30:01] WARNING - Vendor file cytoscape.min.js not found, downloading from CDN: https://unpkg.com/cytoscape@3.30.2/dist/cytoscape.min.js
[10:30:02] INFO - Downloaded cytoscape.min.js from CDN (892341 bytes)
[10:30:03] WARNING - styles.css not found, using empty styles
[10:30:03] WARNING - HTML minification failed: Syntax error
[10:30:03] WARNING - Using non-minified HTML
```

### Error Messages
```
[10:30:00] ERROR - Failed to load config: File not found
[10:30:00] ERROR - Graph file not found: /viz/data/out/LearningChunkGraph_wow.json
[10:30:00] ERROR - Invalid JSON: Expecting value: line 1 column 1 (char 0)
[10:30:00] ERROR - requests library not available for CDN fallback
[10:30:00] ERROR - Cannot load critical library cytoscape.min.js
[10:30:00] ERROR - graph_core.js not found - graph won't be initialized!
[10:30:00] ERROR - Template rendering failed: 'graph_data' is undefined
[10:30:00] ERROR - Failed to save HTML: Permission denied
```

## Core Algorithm

1. **Initialization**
   - Parse command line arguments (test mode flag)
   - Setup logging to `/viz/logs/graph2html.log`
   - Load configuration from `/viz/config.toml`

2. **Data Loading**
   - Determine data paths based on mode (production/test)
   - Load enriched graph JSON with metrics
   - Load concept dictionary JSON
   - Handle missing concepts in test mode with empty stub

3. **Template Preparation**
   - Initialize Jinja2 environment with `/viz/templates/`
   - Load `index.html` template
   - Load `styles.css` for embedding

4. **Library Management**
   - Check `embed_libraries` configuration
   - If embedding:
     - Reorder vendor files according to `LIBRARY_ORDER`
     - Load each vendor file or download from CDN
     - Concatenate all JavaScript/CSS content
     - Add cose-bilkent registration script
     - Include graph_core.js module
     - Add debug_helpers.js in test mode
   - If CDN mode:
     - Generate script/link tags with CDN URLs

5. **HTML Generation**
   - Prepare template context with all data and configuration
   - Render Jinja2 template with context
   - Optionally minify JSON data
   - Optionally minify final HTML

6. **Output**
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
Load enriched graph and concept data from JSON files.
- **Input**:
  - data_dir (Path) - Base data directory
  - logger (Logger) - Logger instance
  - test_mode (bool) - Use test data if True
- **Returns**: tuple[Dict, Dict] - (graph_data, concepts_data)
- **Raises**: 
  - SystemExit(EXIT_INPUT_ERROR) - Missing required files
  - SystemExit(EXIT_IO_ERROR) - File read or JSON parse errors
- **Note**: In test mode, uses empty concepts stub if file missing

### minify_json_data(data: Dict, minify: bool) -> str
Convert data to JSON string, optionally minified.
- **Input**:
  - data (Dict) - Data to serialize
  - minify (bool) - If True, produce compact JSON
- **Returns**: str - JSON string
- **Note**: Preserves Unicode characters without escaping

### load_vendor_content(viz_dir: Path, file_list: list, logger: logging.Logger, test_mode: bool = False) -> str
Load and concatenate vendor files with CDN fallback and proper ordering.
- **Input**:
  - viz_dir (Path) - Viz module directory
  - file_list (list) - List of relative file paths
  - logger (Logger) - Logger instance
  - test_mode (bool) - Include debug_helpers.js if True
- **Returns**: str - Concatenated content with file markers
- **Algorithm**:
  1. Reorder files according to LIBRARY_ORDER for dependencies
  2. Load each file locally or via CDN
  3. Add file path comments between sections
  4. Append debug_helpers.js in test mode
- **Note**: Critical libraries raise error if unavailable

### generate_script_tags(file_list: list, embed: bool) -> str
Generate script tags for JavaScript libraries.
- **Input**:
  - file_list (list) - List of JavaScript file paths
  - embed (bool) - If True, return empty (content embedded)
- **Returns**: str - HTML script tags or empty string
- **Note**: Uses CDN URLs from CDN_URLS mapping when available

### generate_style_tags(file_list: list, embed: bool) -> str
Generate link tags for CSS libraries.
- **Input**:
  - file_list (list) - List of CSS file paths
  - embed (bool) - If True, return empty (content embedded)
- **Returns**: str - HTML link tags or empty string
- **Note**: Uses CDN URLs from CDN_URLS mapping when available

### generate_html(graph_data: Dict, concepts_data: Dict, config: Dict, viz_dir: Path, logger: logging.Logger, test_mode: bool = False) -> str
Generate the complete HTML file for visualization.
- **Input**:
  - graph_data (Dict) - Enriched graph with metrics
  - concepts_data (Dict) - Concept dictionary
  - config (Dict) - Full configuration dictionary
  - viz_dir (Path) - Viz module directory
  - logger (Logger) - Logger instance
  - test_mode (bool) - Test mode flag
- **Returns**: str - Complete HTML document
- **Raises**:
  - SystemExit(EXIT_IO_ERROR) - Template loading errors
  - SystemExit(EXIT_RUNTIME_ERROR) - Template rendering errors
- **Note**: Adds cose-bilkent registration script automatically

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

## Internal Methods

### _reorder_vendor_files(file_list: list) -> list
Reorder vendor files to maintain critical library dependencies.
- **Purpose**: Ensure Cytoscape.js plugins load after core library
- **Algorithm**: 
  1. Extract files matching LIBRARY_ORDER
  2. Place them first in order
  3. Append remaining files
- **Note**: Critical for cose-bilkent layout functionality

## Configuration

Section `[graph2html]` in `/viz/config.toml`:

### Required Parameters
- **output_filename** (str) - Output HTML filename
- **vendor_js** (list[str]) - JavaScript libraries to include
- **vendor_css** (list[str]) - CSS libraries to include

### Optional Parameters
- **minify_json** (bool, default=true) - Minify embedded JSON data
- **minify_html** (bool, default=true) - Minify final HTML output
- **embed_libraries** (bool, default=true) - Embed vs CDN mode
- **default_language** (str, default="ru") - Default UI language
- **enable_language_switch** (bool, default=true) - Show language toggle

### Additional Configuration Sections Used
- `[visualization]` - Graph visualization parameters
- `[colors]` - Color schemes and palettes
- `[ui]` - Interface settings
- `[node_shapes]` - Node shape mappings

## Template System

### Template Files

#### `/viz/templates/index.html`
Main Jinja2 template that structures the HTML document. Expected blocks:
- `{% block title %}` - Page title
- `{% block styles %}` - CSS inclusion point
- `{% block scripts %}` - JavaScript inclusion point
- `{% block content %}` - Main content area

#### `/viz/templates/styles.css`
CSS styles for the visualization interface. Includes:
- Graph container styles
- Sidebar and control panel styles
- Node and edge appearance rules
- Responsive layout definitions

### Template Context Variables

The template receives the following context:

```python
{
    # Data (JSON strings)
    "graph_data_json": str,        # Minified/formatted graph JSON
    "concepts_data_json": str,     # Minified/formatted concepts JSON
    
    # Configuration objects
    "viz_config": {                # From [visualization] section
        "initial_layout": str,
        "animation_duration": int,
        "tour_speed": int,
        "tour_auto_loop": bool,
        "node_size_min": int,
        "node_size_max": int,
        "physics_enabled": bool,
        "physics_duration": int
    },
    "colors_config": {             # From [colors] section
        "theme": str,
        "cluster_palette": list,
        "chunk_color": str,
        "concept_color": str,
        "assessment_color": str,
        "path_fast": str,
        "path_easy": str,
        "gradient_easy": str,
        "gradient_medium": str,
        "gradient_hard": str
    },
    "ui_config": {                 # From [ui] section
        "sidebar_width": int,
        "sidebar_position": str,
        "search_debounce": int,
        "max_popup_width": int,
        "show_fps_counter": bool,
        "show_legend": bool,
        "show_minimap": bool,
        "show_stats": bool
    },
    "node_shapes": {               # From [node_shapes] section
        "chunk_shape": str,
        "concept_shape": str,
        "assessment_shape": str
    },
    
    # Library content (when embed_libraries=true)
    "embed_libraries": bool,       # Embedding mode flag
    "vendor_js_content": str,      # Concatenated JS libraries
    "vendor_css_content": str,     # Concatenated CSS libraries
    "styles_content": str,         # styles.css content
    "graph_core_content": str,     # graph_core.js content (empty if CDN mode)
    
    # CDN tags (when embed_libraries=false)
    "script_tags": str,            # HTML script tags for CDN
    "link_tags": str,              # HTML link tags for CDN
    
    # Metadata
    "graph_stats": {
        "nodes": int,              # Node count
        "edges": int,              # Edge count
        "concepts": int            # Concept count
    }
}
```

### Working with Styles

#### Style Loading Priority
1. **Vendor CSS** (if configured) - External library styles
2. **styles.css** - Main application styles
3. **Inline styles** - Template-specific overrides

#### Style Embedding vs Linking
- **embed_libraries=true**: All CSS embedded in `<style>` tags
- **embed_libraries=false**: CSS linked via `<link>` tags to CDN

#### Customizing Styles
1. Edit `/viz/templates/styles.css` for persistent changes
2. Use `colors_config` in template for dynamic theming
3. Override via inline styles in template for specific elements

## Library Loading Strategy

### Critical Library Order
The following libraries MUST load in this exact order (defined in `LIBRARY_ORDER` constant):

1. **cytoscape.min.js** - Core graph library (MUST be first!)
2. **layout-base.js** - Base dependency for layout algorithms
3. **cose-base.js** - Cose layout dependency (requires layout-base)
4. **cytoscape-cose-bilkent.js** - Advanced layout plugin (requires cose-base)

After these, the registration script runs:
```javascript
cytoscape.use(cytoscapeCoseBilkent);
```

5. **graph_core.js** - Application initialization module
6. **debug_helpers.js** - Debug utilities (test mode only)

### Library Loading Modes

#### Embed Mode (embed_libraries=true)
1. Read vendor files from local `/viz/vendor/` directory
2. If file missing, attempt CDN download via `get_library_content()`
3. Concatenate all content with file markers
4. Embed directly in HTML `<script>` and `<style>` tags
5. Result: Self-contained HTML, works offline

#### CDN Mode (embed_libraries=false)
1. Generate `<script src="CDN_URL">` tags for JavaScript
2. Generate `<link href="CDN_URL">` tags for CSS
3. Use CDN_URLS mapping for known libraries
4. Fallback to relative paths for unknown libraries
5. Result: Smaller HTML, requires internet

### CDN Fallback Mechanism

When vendor file is missing in embed mode:

1. **Detection**: File.exists() returns False
2. **Logging**: WARNING logged with file path and CDN URL
3. **Download Attempt**:
   - Import `requests` library (optional dependency)
   - GET request to CDN URL with 30s timeout
   - Validate response status
4. **Success Path**:
   - Log successful download with size
   - Return content for embedding
5. **Failure Paths**:
   - No requests library: RuntimeError (critical libraries only)
   - Download failed: RuntimeError (critical libraries only)
   - Optional libraries: Continue with warning

### CDN URLs Mapping

Predefined CDN URLs for common libraries:
```python
CDN_URLS = {
    "cytoscape.min.js": "https://unpkg.com/cytoscape@3.30.2/dist/cytoscape.min.js",
    "layout-base.js": "https://unpkg.com/layout-base@2.0.1/layout-base.js",
    "cose-base.js": "https://unpkg.com/cose-base@2.2.0/cose-base.js",
    "cytoscape-cose-bilkent.js": "https://unpkg.com/cytoscape-cose-bilkent@4.1.0/cytoscape-cose-bilkent.js",
    "cytoscape-navigator.js": "https://unpkg.com/cytoscape.js-navigator@2.0.2/cytoscape.js-navigator.js",
    "cytoscape.js-navigator.css": "https://unpkg.com/cytoscape.js-navigator@2.0.2/cytoscape.js-navigator.css"
}
```

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
- **Library Errors** - Critical libraries unavailable
- **IO Errors** - File write failures, network issues

### Boundary Cases
- **Missing concept dictionary in test mode** → Use empty stub
- **Missing optional vendor files** → Log warning and continue
- **Missing critical vendor files** → Try CDN fallback
- **CDN download failure for critical library** → RuntimeError
- **HTML minification failure** → Use non-minified output
- **Large graphs (>10MB)** → May cause browser performance issues

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
- **cytoscape.min.js** - Core graph library (critical)
- **layout-base.js** - Layout dependency (critical)
- **cose-base.js** - Cose dependency (critical)
- **cytoscape-cose-bilkent.js** - Layout plugin (critical)
- **cytoscape-navigator.js** - Minimap (optional)
- **edge_styles.js** - Edge styling module (custom)
- **animation_controller.js** - Animation control module (custom)
- **graph_core.js** - Main visualization controller (custom)

## Performance Notes

- **Memory**: ~100MB for typical graph (500 nodes)
- **Generation Speed**: ~2-3 seconds for production graph
- **File Sizes**:
  - Test mode: 700-800 KB
  - Production embedded: 5-15 MB
  - Production CDN: 1-3 MB
- **Browser Performance**: 
  - Smooth up to 1000 nodes
  - May lag with >2000 nodes
- **Optimization**:
  - JSON minification saves 30-40%
  - HTML minification saves 10-20%
  - CDN mode reduces file size by 70-80%

## Test Coverage

The module is tested by `/tests/viz/test_graph2html.py` with 32 tests covering:

### Core Functions
- **setup_logging()** - Directory creation, logger configuration
- **get_library_content()** - Local file loading, CDN fallback, error handling
- **load_graph_data()** - Production/test modes, missing files, invalid JSON
- **minify_json_data()** - Minification, pretty print, Unicode support
- **load_vendor_content()** - Library ordering, test mode debug helpers, CDN fallback
- **generate_script_tags()** / **generate_style_tags()** - CDN vs embed modes
- **generate_html()** - Template rendering, context preparation, cose-bilkent registration
- **minify_html_content()** - HTML minification, error recovery
- **save_html()** - File writing, size logging, I/O errors

### Test Classes (32 tests total)
- **TestSetupLogging** - 1 test
- **TestGetLibraryContent** - 5 tests (critical CDN fallback logic)
- **TestLoadGraphData** - 5 tests
- **TestMinifyJsonData** - 3 tests
- **TestLoadVendorContent** - 4 tests (including library ordering)
- **TestGenerateTags** - 4 tests
- **TestGenerateHtml** - 5 tests (including graph_core.js loading)
- **TestMinifyHtmlContent** - 3 tests
- **TestSaveHtml** - 2 tests

All tests use mocks for filesystem operations and external dependencies.

## Visual Encoding System

### Node Visual Encoding
The visualization uses distinct shapes and colors for the three node types:

#### Node Types and Shapes
- **Chunk**: hexagon shape, blue (#3498db)
  - Represents learning content chunks
  - Size scaled by PageRank metric
  - Opacity based on difficulty (0.5 for difficulty=1, 1.0 for difficulty=5)

- **Concept**: star shape, green (#2ecc71)
  - Represents semantic concepts
  - Size scaled by PageRank metric
  - Opacity based on difficulty

- **Assessment**: roundrectangle shape, orange (#f39c12)
  - Represents assessment/test nodes
  - Size scaled by PageRank metric
  - Opacity based on difficulty

### Edge Visual Encoding
All 9 edge types from the schema have distinct visual styles:

#### Strong Dependencies (4px width)
- **PREREQUISITE**: solid red (#e74c3c), 4px
  - Strong educational dependency
  - Most prominent visual weight
  
- **TESTS**: solid orange (#f39c12), 4px
  - Assessment relationship
  - High visual prominence

#### Clear Relationships (2.5px width)
- **ELABORATES**: dashed blue (#3498db), 2.5px
  - Elaboration/detail relationship
  - Dash pattern: [8, 4]
  
- **EXAMPLE_OF**: dotted purple (#9b59b6), 2.5px
  - Example relationship
  - Dot pattern: [2, 4]
  
- **PARALLEL**: solid gray (#95a5a6), 2.5px
  - Parallel topic relationship
  
- **REVISION_OF**: dashed green (#27ae60), 2.5px
  - Revision/update relationship
  - Dash pattern: [6, 3]

#### Weak References (1px width)
- **HINT_FORWARD**: dotted light blue (#5dade2), 1px
  - Hint to future content
  - Dot pattern: [2, 6]
  - Lower opacity (0.5)
  
- **REFER_BACK**: dotted pink (#ec7063), 1px
  - Reference to past content
  - Dot pattern: [2, 6]
  - Lower opacity (0.5)
  
- **MENTIONS**: dashed light gray (#bdc3c7), 1px
  - Simple mention
  - Dash pattern: [4, 4]
  - Lowest opacity (0.4)

#### Inter-cluster Edges
- Edges connecting nodes from different clusters are rendered 50% thicker
- Higher z-index for visual prominence
- Slightly increased opacity

### Animation System

#### Node Appearance Animation
1. Nodes grouped by `prerequisite_depth` metric
2. Each depth level appears sequentially:
   - 500ms fade-in animation per node
   - 200ms delay between depth levels
   - Opacity animated to difficulty-based value
3. Lower depths (prerequisites) appear first

#### Edge Appearance Animation
- All edges appear after nodes complete
- 500ms fade-in to type-specific opacity
- Inter-cluster edges highlighted with higher opacity

#### Physics Simulation
- Runs after initial placement
- 3000ms duration (configurable)
- Uses cose-bilkent layout algorithm
- Nodes remain draggable after simulation

### Label Display Behavior

#### Hover Labels
- Labels hidden by default (no visual clutter)
- Show on mouse hover with 500ms delay
- White text with dark outline for readability
- Font weight increased to 600
- Hide immediately on mouse leave
- Prevents label flashing on quick mouse movements

### Interactive Features

#### Node Interactions
- Hover: 20% size increase, label display
- Click: Opens node information popup with full details
- Drag: Repositioning (maintained after physics)

#### Edge Interactions
- Hover: 50% width increase, full opacity
- Click: Selection with orange highlight

#### Dictionary Interactions
- Click on concept: Opens concept definition popup
- Hover on concept: Highlights nodes containing that concept

#### TOP Nodes Interactions
- Click on TOP node: Centers view on that node (no popup)
- Hover on TOP node: Pulse effect on the node

#### Popup Interactions
- Click on edge target in node popup: Updates popup with target node
- Click outside popup: Closes popup
- Escape key: Closes active popup (priority: node > concept > info > side panel)

### JavaScript Module Architecture

#### Static Modules (loaded in order)
1. **edge_styles.js**
   - Defines all 9 edge type styles
   - Exports EdgeStyles global object
   - Functions: generateEdgeStyles(), getEdgeColor(), isEducationalEdge()

2. **animation_controller.js**
   - Controls animation sequences
   - Exports AnimationController class
   - Methods: animateGraph(), highlightPath(), stopAnimation()

3. **graph_core.js**
   - Main visualization controller
   - Integrates edge styles and animations
   - Exports GraphCore class
   - Provides base styles for UI interactions

4. **ui_controls.js**
   - User interface controls and interactions
   - Exports UIControls object
   - Features:
     * Top header filters for node types (Chunks, Concepts, Assessments)
     * Dynamic counters showing visible/total nodes and edges
     * Right side panel with Dictionary and TOP nodes tabs
     * Node hover effects with red highlighting
     * Edge highlighting on node hover
     * Tooltips with 500ms delay
     * Info popup with graph statistics (i key)
     * **Node Information Popup**: Detailed node information with metrics, edges, and navigation
     * **Concept Definition Popup**: Concept definition with aliases and mention count
     * **Popup Management**: Exclusive popup display (only one at a time)
     * **Educational Tooltips**: Explanations for PageRank, Betweenness, and Learning Effort metrics
     * Keyboard shortcuts (Esc, i, d)
   - Auto-initializes via "k2-graph-ready" event

## Usage Examples

### Basic Production Build
```bash
# Generate production visualization
python -m viz.graph2html

# Output
✓ HTML visualization created: /viz/data/out/knowledge_graph.html
```

### Test Mode with Debug Output
```bash
# Generate test visualization
python -m viz.graph2html --test

# Output
✓ HTML visualization created: /viz/data/out/test_graph.html

Test mode notes:
  - Using data from /viz/data/test/tiny_html_data.json
  - debug_helpers.js included (if available)
  - Open in browser to test: file:///path/to/test_graph.html
```

### Custom Configuration
```python
# Modify /viz/config.toml before running
[graph2html]
embed_libraries = false  # Use CDN for smaller file
minify_html = false      # Keep readable for debugging
output_filename = "custom_graph.html"

# Then run
python -m viz.graph2html
```

### Browser Console Debug (Test Mode)
```javascript
// When test_graph.html is opened
debugHelpers.help()  // Show available debug commands
window.graphData     // Access graph data
window.conceptData   // Access concept dictionary
cy.nodes().size()    // Count nodes via Cytoscape
```

## Notes

- Template system uses Jinja2 with autoescape disabled for raw HTML
- Graph visualization uses Cytoscape.js without build tools
- Test mode automatically includes debug helpers for development
- CDN fallback requires `requests` library (graceful degradation)
- File paths in config are relative to `/viz/` directory
- Both production and test modes use same template system
- Minification settings are independent (JSON vs HTML)
- Library order is critical for cose-bilkent layout to work

## Known Issues and Fixes

- **Fixed**: Chrome on Windows showing unwanted scrollbars in info popup (overflow management with nested containers)
- Node and Concept popups use exclusive display (only one popup at a time)
- Educational tooltips provide context for learning metrics
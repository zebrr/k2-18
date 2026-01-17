# viz_graph2viewer_frontend.md

## Status: READY

Frontend architecture and JavaScript modules for the knowledge graph viewer.

## Module Architecture

The viewer uses a modular JavaScript architecture with global objects for inter-module communication. All modules are loaded sequentially and initialized through the main `initViewer()` function.

## JavaScript Modules

### viewer_core.js
Main controller that orchestrates all viewer functionality.
- **Entry Point**: `initViewer(graphData, conceptData)`
- **Responsibilities**:
  - Initialize all viewer modules in correct order
  - Manage global state
  - Coordinate module interactions
- **Module Initialization Order**:
  1. Formatters.init()
  2. SearchFilter.init(nodes, conceptDict)
  3. NodeExplorer.init(graphData, conceptDict)
  4. EdgeInspector.init(graphData)
  5. NavigationHistory.init()

### node_explorer.js
Handles node selection and display functionality.
- **Global Object**: `NodeExplorer`
- **Key Methods**:
  - `init(graphData, conceptDict)` - Initialize with data
  - `selectNode(nodeId)` - Select and display a node
  - `renderActiveNode(node)` - Render node in Column B
  - `renderNodeList(nodes)` - Render filtered nodes in Column A
  - `toggleViewMode()` - Switch between formatted/JSON views
- **Features**:
  - Formatted view with Markdown/code/math support
  - JSON view with syntax highlighting
  - Metrics display with tooltips
  - Edges table with clickable rows

### edge_inspector.js
Manages edge selection and related node display.
- **Global Object**: `EdgeInspector`
- **Key Methods**:
  - `init(graphData)` - Initialize with graph data
  - `selectEdge(edgeData)` - Display edge and related node
  - `clearEdge()` - Clear edge selection
  - `renderEdgePanel(edge, sourceNode, targetNode, isIncoming)` - Visualize edge
  - `renderRelatedNode(node, edge)` - Display node in Column C
  - `makeNodeActive(node)` - Switch to selected node
- **Edge Types Supported**:
  - PREREQUISITE (red)
  - ELABORATES (blue)
  - RELATED_TO (green)
  - CONTINUES (purple)
  - CONTRASTS (orange)
  - DEMONSTRATES (cyan)
  - PART_OF (pink)
  - ALTERNATIVE (brown)
  - APPLIED_IN (teal)

### search_filter.js
Provides search and filtering capabilities.
- **Global Object**: `SearchFilter`
- **Key Methods**:
  - `init(nodes, conceptDict)` - Initialize with data
  - `applyFilter(searchText, typeFilters)` - Filter nodes
  - `getFilteredNodes()` - Get current filtered list
- **Features**:
  - Real-time search with 300ms debouncing
  - Type filters (CHUNK, CONCEPT, ASSESSMENT)
  - Result count display
  - Custom events for filter changes

### formatters.js
Text formatting and rendering utilities.
- **Global Object**: `Formatters`
- **Key Methods**:
  - `init()` - Configure formatting libraries
  - `formatMarkdown(text)` - Convert Markdown to HTML
  - `highlightCode(code, language)` - Syntax highlighting
  - `renderMath(element)` - Trigger MathJax rendering
  - `formatNodeText(text)` - Combined formatting
  - `formatDifficulty(difficulty)` - Traffic-light indicators
  - `formatMetricValue(value)` - Format numeric values
  - `formatEdgeType(type)` - Translate edge types
- **Dependencies**:
  - marked.js for Markdown
  - highlight.js for code
  - MathJax for math formulas

### navigation_history.js
Tracks user navigation through the graph.
- **Global Object**: `NavigationHistory`
- **Key Methods**:
  - `init()` - Initialize history tracking
  - `addToHistory(nodeId)` - Add node to history
  - `goBack()` - Navigate to previous node
  - `goForward()` - Navigate to next node
  - `clearHistory()` - Reset navigation history
- **Features**:
  - Back/forward navigation
  - History limit (50 items)
  - Visual history display

## Three-Column Layout

### Column A (20% width): Search and Selection
- **Header**: Search input with real-time filtering
- **Filters**: Type checkboxes (Chunk, Concept, Assessment)
- **Node List**: Filtered scrollable list of node cards
  - Type badge (colored by node type)
  - Node ID and title
  - Cluster indicator (background color)
  - Click to select functionality

### Column B (40% width): Active Node Details
- **Header**:
  - Node type badge with color coding
  - Node ID
  - Difficulty indicators (traffic-light colors)
  - View toggle button (Formatted/JSON)
- **Content Sections**:
  - Title and main content (gray background)
  - Definition if concept (light blue background)
  - Related concepts (light yellow background)
- **Metrics Grid**: 12 educational metrics in 3x4 layout
  - Click for detailed tooltip (no arrow)
- **Edges Table**:
  - 5-column layout: Direction | Type | Node Type | Node ID | Weight
  - Incoming edges (← arrow)
  - Outgoing edges (→ arrow)
  - Click row to inspect edge

### Column C (40% width): Related Node
- **Edge Panel** (between B and C):
  - Single-line edge visualization
  - Source and target node badges
  - Edge type and weight
  - Inter-cluster indicator
  - Optional conditions
- **Related Node Content**:
  - "Make active node" button
  - Full node details (same format as Column B)
  - No clickable elements in edges table

## DOM Structure

```html
<div id="viewer-container">
  <header id="viewer-header">
    <h1>K2-18 Knowledge Graph Viewer</h1>
    <div class="stats">Nodes: X | Edges: Y</div>
    <nav id="navigation-history"></nav>
  </header>

  <main id="viewer-main">
    <aside id="column-a" class="viewer-column">
      <div class="search-section">
        <input id="search-input" type="text">
        <div id="filter-checkboxes"></div>
        <div id="result-count"></div>
      </div>
      <div id="node-list" class="scrollable"></div>
    </aside>

    <section id="column-b" class="viewer-column">
      <header class="node-header">
        <div class="node-badges"></div>
        <button id="view-toggle">Toggle View</button>
      </header>
      <div id="node-content" class="scrollable"></div>
      <div id="metrics-grid"></div>
      <div id="edges-table"></div>
    </section>

    <section id="column-c" class="viewer-column">
      <div id="edge-panel"></div>
      <div id="related-node-content" class="scrollable"></div>
    </section>
  </main>
</div>
```

## Event System

### Custom Events
- `filter-changed` - Fired when search/filter updates
- `node-selected` - Fired when node is selected
- `edge-selected` - Fired when edge is selected
- `view-mode-changed` - Fired when toggling formatted/JSON

### Event Flow
1. User interaction (click, search, etc.)
2. Module processes event
3. Module updates DOM
4. Module fires custom event
5. Other modules listen and react

## CSS Classes

### Node Type Classes
- `.node-type-chunk` - Blue background
- `.node-type-concept` - Green background
- `.node-type-assessment` - Orange background

### Difficulty Classes
- `.difficulty-low` - Green (levels 1-2)
- `.difficulty-medium` - Yellow (level 3)
- `.difficulty-high` - Red (levels 4-5)

### State Classes
- `.selected` - Currently selected item
- `.active` - Active/clickable element
- `.disabled` - Non-interactive element
- `.filtered-out` - Hidden by filter

### Layout Classes
- `.viewer-column` - Main column container
- `.scrollable` - Scrollable content area
- `.node-card` - Node list item
- `.metric-item` - Metric grid cell
- `.edge-row` - Edge table row

## Browser Compatibility

Requires modern browser with:
- ES6 JavaScript support
- CSS Grid and Flexbox
- Custom Events API
- JSON parsing
- Local storage (for preferences)

## Performance Considerations

- Node list uses virtual scrolling for >1000 nodes
- Debounced search (300ms) prevents excessive filtering
- Lazy loading of node details
- Metrics calculated once and cached
- JSON view syntax highlighting on-demand
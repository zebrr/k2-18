# viz_ui_controls.md

## Status: READY

Comprehensive UI controls module that manages all user interactions, filters, panels, popups, and visualization modes for the K2-18 knowledge graph.

## Module Purpose

The `ui_controls.js` module handles all user interface elements and interactions for the visualization. It manages the filter panel, mode switching, side panels (dictionary and TOP nodes), hover effects, tooltips, information popups, node/concept detail popups, and keyboard shortcuts. This module is the primary interface between users and the graph visualization.

## Public API

### UIControls Object

The main object exposed globally as `window.UIControls`.

#### Properties

##### Components
- **topPanel** (HTMLElement|null) - Top panel reference (deprecated)
- **sidePanel** (HTMLElement) - Right side panel for dictionary/TOP nodes
- **sidePanelTab** (HTMLElement) - Tab button for side panel
- **hoverController** (object|null) - Hover effect controller
- **infoPopup** (HTMLElement) - Graph statistics popup
- **tooltip** (HTMLElement) - Node hover tooltip
- **nodePopup** (HTMLElement) - Node details popup
- **conceptPopup** (HTMLElement) - Concept definition popup

##### State Management
```javascript
state: {
    visibleTypes: {
        'Chunk': true,
        'Concept': true,
        'Assessment': true
    },
    visibleEdgeCategories: {
        'strong': true,    // PREREQUISITE, TESTS
        'medium': true,    // ELABORATES, EXAMPLE_OF, PARALLEL, REVISION_OF
        'weak': true       // HINT_FORWARD, REFER_BACK, MENTIONS
    },
    sidePanelOpen: false,
    activeTab: 'dictionary',    // 'dictionary' | 'top-nodes'
    hoveredNode: null,
    hoveredConcept: null,
    infoPanelOpen: false,
    nodePopupOpen: false,
    conceptPopupOpen: false,
    currentPopupNode: null,
    currentPopupConcept: null,
    activeMode: null            // 'path' | 'clusters' | 'tour' | null
}
```

### Initialization

#### init(cy, graphCore, appContainer, statsContainer, graphData, conceptData, config)
Initializes all UI components and event handlers.
- **Input**:
  - cy (Cytoscape) - Cytoscape instance
  - graphCore (GraphCore) - Core visualization module
  - appContainer (HTMLElement) - Main app container
  - statsContainer (HTMLElement) - Stats display container
  - graphData (object) - Graph data
  - conceptData (object) - Concept dictionary
  - config (object) - UI configuration
- **Side effects**:
  - Creates all UI components
  - Sets up event handlers
  - Updates initial counters
  - Auto-triggered by 'k2-graph-ready' event

## UI Components

### 1. Header Components

#### Stats Counters
Display in header right:
- Visible/total nodes count
- Visible/total edges count

Note: Mode buttons have been moved to the Bottom Badge component.

### 2. Filter Panel

Separate panel below header with node and edge filters.

#### Node Filters
- Chunks checkbox
- Concepts checkbox
- Assessments checkbox

#### Edge Filters
- Strong connections (PREREQUISITE, TESTS)
- Medium connections (ELABORATES, EXAMPLE_OF, PARALLEL, REVISION_OF)
- Weak connections (HINT_FORWARD, REFER_BACK, MENTIONS)

#### Appearance
- Initially hidden
- Animates in after 1 second or 'graph-animation-complete' event
- Slide down animation with opacity fade

### 3. Side Panel (Right)

Sliding panel with two tabs.

#### Dictionary Tab
- Alphabetically sorted concept list
- Shows concept name and mention count
- Hover highlights nodes containing concept
- Click shows concept definition popup

#### TOP Nodes Tab
Three sections showing top 5 nodes by:
- **PageRank** (Importance)
- **Betweenness** (Bridges)
- **Degree** (Hubs)

Features:
- Hover causes pulse effect on node
- Click centers view on node

### 4. Popups

#### Node Information Popup
Detailed node information with:
- **Header**: Type, ID, difficulty circles
- **Content**: Full text with formatting support
- **Source**: Original definition if available
- **Metrics**:
  - PageRank with info tooltip
  - Betweenness with info tooltip
  - Learning Effort with info tooltip
- **Connectivity**: In/out degree counts
- **Edges**: List of connections with:
  - Direction arrow
  - Edge type
  - Target node (clickable for navigation)
  - Difficulty circles
  - Weight value

#### Concept Definition Popup
Concept details with:
- Primary term
- Full definition with formatting
- Aliases/synonyms
- Mention count across nodes

#### Graph Statistics Popup
Overall graph information:
- Node counts by type
- Total edge count
- Connected components count
- Cluster count (if available)
- Visual legend for nodes and edges

### 5. Tooltips

#### Node Hover Tooltip
- Shows truncated node text (300 chars)
- 500ms delay before showing
- Positioned near cursor
- Max width 400px

#### Metric Info Tooltips
Click-based explanations for metrics:
- PageRank explanation
- Betweenness explanation
- Learning Effort explanation

### 6. Bottom Badge

Compact information panel in the lower left corner of the screen.

#### Structure
- **Info button (ℹ️)** - Opens graph statistics popup
- **Beta features label** - Indicator for experimental features
- **Mode buttons**:
  - Path Mode (🛤️) - Learning path discovery
  - Clusters Mode (🎨) - Cluster visualization
- **GitHub link** - Link to project repository

#### Appearance
- Position: fixed, bottom: 20px, left: 20px
- Size: height 36px, width by content
- Style: semi-transparent background with blur effect
- Appears together with filter panel after animation completes

#### Behavior
- Appears 1 second after 'graph-animation-complete' event
- Mode buttons show active state
- GitHub link opens in new tab

## Core Methods

### Filter Methods

#### Filter Panel Appearance
Filter panel appears 1000ms after 'graph-animation-complete' event or 1000ms after initialization if animation is already complete.

#### toggleNodeType(type, visible)
Shows/hides nodes of specific type.
- **Algorithm**:
  1. Update state.visibleTypes
  2. Add/remove 'hidden' class from nodes
  3. Update connected edges visibility
  4. Update counters
- **Edge logic**: Edge visible only if both endpoints visible AND category enabled

#### toggleEdgeCategory(category, visible)
Shows/hides edges by category.
- **Categories**:
  - strong: PREREQUISITE, TESTS
  - medium: ELABORATES, EXAMPLE_OF, PARALLEL, REVISION_OF
  - weak: HINT_FORWARD, REFER_BACK, MENTIONS
- **Algorithm**:
  1. Update state.visibleEdgeCategories
  2. Show/hide edges of category types
  3. Check endpoint visibility
  4. Update counters

### Mode Management

#### toggleMode(mode)
Switches between visualization modes.
- **Modes**: 'path', 'clusters', 'tour', null
- **Behavior**:
  - Same mode: Deactivate
  - Different mode: Switch
  - null: Deactivate all
- **Events**: Dispatches 'mode-changed' custom event

### Panel Methods

#### toggleSidePanel()
Opens/closes right side panel.
- Slides panel in/out
- Moves tab button with panel
- Updates state.sidePanelOpen

#### switchTab(tabName)
Switches between dictionary and TOP nodes tabs.

### Hover Effects

#### setupHoverEffects()
Configures node hover interactions:
- **On hover**:
  - Add 'hover-highlight' class (red)
  - Highlight connected edges
  - Show tooltip after 500ms
- **On leave**:
  - Remove highlight classes
  - Hide tooltip
- **Skip conditions**:
  - Path mode dimmed nodes
  - Nodes with 'ghost' property

### Popup Methods

#### showNodePopup(node)
Displays detailed node information.
- Closes other popups
- Formats all data and metrics
- Renders difficulty as colored circles
- Makes edges clickable for navigation
- Supports text formatting (Markdown, Math, Code)

#### showConceptPopup(concept)
Displays concept definition.
- Closes other popups
- Shows definition with formatting
- Lists aliases
- Shows mention count

#### showGraphStats()
Displays graph statistics and legend.
- Node/edge counts
- Component analysis
- Cluster information
- Complete visual legend

### Helper Methods

#### updateCounters()
Updates visible/total counts in header.
- Uses `.not('.hidden')` for accurate counts
- Updates all counter displays

#### highlightConceptNodes(conceptId)
Highlights nodes containing a concept.
- Adds 'pulse' class to matching nodes

#### centerOnNode(nodeId)
Animates view to center on node.
- Zoom level 2
- 500ms animation

#### formatTextContent(content)
Formats text with Markdown, Math, and Code.
- Markdown parsing with marked.js
- Math delimiters: `$...$` inline, `$$...$$` display
- Returns formatted HTML

#### renderFormattedContent(element)
Triggers rendering of formatted content.
- Code highlighting with highlight.js
- Math rendering with MathJax

## Event Handling

### Mouse Events
- **Node hover**: Highlight and tooltip
- **Node click**: Show node popup (unless in path mode)
- **Concept item hover**: Highlight related nodes
- **Concept item click**: Show concept popup
- **TOP node hover**: Pulse effect
- **TOP node click**: Center view

### Keyboard Shortcuts
- **Escape**: Close popups/panels/modes (priority order)
- **i**: Toggle info popup
- **d**: Toggle dictionary panel (right)
- **t/T**: Toggle Table of Contents (left course panel)
- **p/P**: Toggle Path mode
- **c/C**: Toggle Clusters mode

### Custom Events

#### Listens for:
- **k2-graph-ready**: Auto-initialize
- **graph-animation-complete**: Show filter panel

#### Dispatches:
- **mode-changed**: When switching modes
  - Detail: `{mode: 'path'|'clusters'|'tour'|null}`

## Visual Effects

### Node Effects
- **hover-highlight**: Red color with full opacity
- **pulse**: Pulsing animation
- **hidden**: Display none

### Edge Effects
- **hover-connected**: Red color, width 6px
- **hidden-edge**: Display none

### Difficulty Visualization
Traffic light pattern with 5 circles:
- 1-2: Green (#2ecc71)
- 3: Yellow (#f39c12)
- 4-5: Red (#e74c3c)

## State Management

### Visibility State
Tracks which elements are visible:
- Node types (Chunk, Concept, Assessment)
- Edge categories (strong, medium, weak)
- Persisted during session

### Popup State
Manages exclusive popup display:
- Only one popup active at a time
- Priority: Node > Concept > Info > Side Panel
- Escape key follows priority

### Mode State
Tracks active visualization mode:
- Only one mode active at a time
- Modes control their own behavior
- UIControls coordinates switching

## Text Formatting System

### Supported Formats
1. **Markdown**: Headers, bold, italic, lists, links
2. **Mathematics**: LaTeX with explicit delimiters
   - Inline: `$formula$`
   - Display: `$$formula$$`
3. **Code**: Syntax highlighting for code blocks

### Processing Pipeline
1. Parse Markdown if enabled
2. Convert math delimiters to MathJax format
3. Return formatted HTML
4. Trigger rendering (highlighting, MathJax)

## Integration with Other Modules

### With GraphCore
- Receives cy instance and references
- Uses node/edge data for display
- Applies visual classes

### With Path Mode
- Checks activeMode before showing popup
- Skips hover effects for dimmed nodes
- Respects 'ghost' property

### With Clusters Mode
- Dispatches mode-changed events
- Coordinates mode switching

### With Configuration
- Uses window.tooltipConfig for tooltip settings
- Uses window.textFormattingConfig for text processing
- Uses window.uiConfig for general settings

## Performance Optimizations

### Batch Operations
- Filter changes use cy.batch()
- Multiple class changes grouped
- Single counter update per operation

### Event Delegation
- Single handler for multiple similar elements
- Efficient selector-based filtering

### Lazy Loading
- Popups created on demand
- Content rendered only when visible
- Formatted content processed once

## CSS Classes and Selectors

### Filter Classes
- `.hidden` - Hidden nodes
- `.hidden-edge` - Hidden edges

### Effect Classes
- `.hover-highlight` - Red node highlight
- `.hover-connected` - Red edge highlight
- `.pulse` - Pulsing animation
- `.path-dimmed` - Dimmed in path mode

### UI Classes
- `.filter-panel` - Filter panel container
- `.mode-btn` - Mode toggle buttons
- `.side-panel` - Right panel
- `.node-popup` - Node detail popup
- `.concept-popup` - Concept popup

## Configuration

### Tooltip Configuration
```javascript
{
    max_width: 400,      // Maximum width in pixels
    preview_length: 300, // Characters to show
    show_delay: 500,     // Milliseconds before show
    hide_delay: 200      // Milliseconds before hide
}
```

### Text Formatting Configuration
```javascript
{
    enable_markdown: true,
    enable_code_highlighting: true,
    enable_math: true,
    math_renderer: "mathjax"
}
```

## Error Handling

### Missing Data
- Empty concept dictionary: Shows message
- Missing node text: Shows "Text unavailable"
- Invalid metrics: Shows "N/A"

### Missing Libraries
- marked.js: Falls back to plain text
- MathJax: Skips math rendering
- highlight.js: No code highlighting

## Testing Considerations

### Console Testing
```javascript
UIControls.state                    // View current state
UIControls.toggleNodeType('Chunk', false)  // Hide chunks
UIControls.updateCounters()          // Refresh counts
UIControls.toggleMode('path')        // Activate path mode
```

### Visual Testing
1. Filter toggles update graph immediately
2. Mode buttons show active state
3. Popups are mutually exclusive
4. Keyboard shortcuts work correctly
5. Hover effects don't interfere with modes

## Internal Methods

### _createBottomBadge()
Creates compact panel with control elements.
- **Purpose**: Group secondary UI elements
- **Returns**: HTMLElement - bottom badge element
- **Contains**: Info button, mode buttons, GitHub link
- **Side effects**: Adds event handlers

## Visual Effects

### Bottom Badge Effects
- `.bottom-badge` - Main container
- `.bottom-badge.visible` - Visible state
- `.mode-btn.active` - Active mode button

## Known Limitations

1. **Mobile Support**: Limited touch interaction support
2. **Large Graphs**: Performance degrades with >1000 nodes
3. **Text Overflow**: Very long text may exceed popup bounds
4. **Math Rendering**: Requires explicit $ delimiters in source
5. **Mode Buttons**: Located in bottom badge, Tour mode hidden from UI but available via API

## Dependencies

### External Libraries
- **Cytoscape.js**: Graph interaction (required)
- **marked.js**: Markdown parsing (optional)
- **MathJax**: Math rendering (optional)
- **highlight.js**: Code highlighting (optional)

### Internal Modules
- **GraphCore**: Provides cy instance
- **PathFinder**: Path mode implementation
- **ClustersBridges**: Clusters mode implementation
- **TourMode**: Tour mode implementation

## Notes

- Central hub for all user interactions
- Modular design allows feature toggling
- Extensive state management for complex UI
- Performance optimized for responsive interaction
- Comprehensive keyboard and mouse support
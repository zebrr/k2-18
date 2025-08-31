# viz_graph_core.md

## Status: READY

Core visualization module that handles Cytoscape.js initialization, graph data preparation, and base styling for the K2-18 knowledge graph.

## Module Purpose

The `graph_core.js` module is the foundation of the visualization system. It initializes Cytoscape.js with the graph data, applies visual encoding based on node/edge types and metrics, manages the initial layout, and coordinates with other modules through events. This module transforms raw graph data into an interactive visual representation.

## Public API

### GraphCore Class

The main class exposed globally as `window.GraphCore`.

#### Constructor

```javascript
new GraphCore(container, config = {})
```

- **container** (HTMLElement) - DOM element for Cytoscape rendering
- **config** (object) - Configuration overrides

#### Properties

- **container** (HTMLElement) - Container element reference
- **config** (object) - Merged configuration
- **cy** (Cytoscape) - Cytoscape instance (after initialization)
- **graphData** (object) - Original graph data
- **conceptData** (object) - Original concept dictionary
- **animationController** (AnimationController|null) - Animation controller instance

#### Configuration Object

Default configuration merged with user config:

```javascript
{
    // Node visual encoding
    nodeShapes: {
        'Chunk': 'hexagon',
        'Concept': 'star',
        'Assessment': 'roundrectangle'
    },
    nodeColors: {
        'Chunk': '#3498db',      // Blue
        'Concept': '#2ecc71',    // Green
        'Assessment': '#f39c12'  // Orange
    },
    // Size mapping
    minNodeSize: 20,
    maxNodeSize: 60,
    // Opacity mapping for difficulty
    minOpacity: 0.5,
    maxOpacity: 1.0,
    // Animation
    animationDuration: 500,
    layoutAnimationDuration: 3000,
    physicsDuration: 3000,
    // Layout
    initialLayout: 'cose-bilkent',
    // Labels
    showLabelsOnHover: true,    // DEPRECATED - handled by UIControls
    hoverDelay: 500,
    // Animation on load
    animateOnLoad: true
}
```

### Core Methods

#### initialize(graphData, conceptData)
Main initialization method that sets up the entire visualization.
- **Input**:
  - graphData (object) - Graph with nodes and edges arrays
  - conceptData (object) - Concept dictionary
- **Returns**: Promise<Cytoscape> - Initialized Cytoscape instance
- **Algorithm**:
  1. Store graph and concept data
  2. Prepare elements for Cytoscape (prepareElements)
  3. Generate styles including edge styles (generateStyles)
  4. Initialize Cytoscape with elements, styles, and layout
  5. Pre-position nodes in wide horizontal band for better layout
  6. Setup hover labels if enabled (deprecated, now in UIControls)
  7. Initialize AnimationController if available
  8. Run initial animation if enabled
  9. Return Cytoscape instance
- **Side effects**:
  - Creates Cytoscape instance
  - Triggers initial layout and animation
  - Logs initialization progress

#### prepareElements()
Transforms raw graph data into Cytoscape elements format.
- **Returns**: object - {nodes: [], edges: []}
- **Node data fields**:
  - id, label (truncated), fullText, type, difficulty
  - pagerank, cluster_id, bridge_score, prerequisite_depth
  - All other fields from original node data
- **Edge data fields**:
  - id (generated as "source-target")
  - source, target, type, weight
  - is_inter_cluster_edge
  - All other fields from original edge data
- **Classes**: Adds lowercase type as class (chunk, concept, assessment)

#### generateStyles()
Generates Cytoscape stylesheet array with all visual styles.
- **Returns**: Array<StyleObject> - Cytoscape style definitions
- **Style categories**:
  1. Base node styles (opacity initially 0 for animation)
  2. Type-specific node styles (shapes and colors)
  3. Selected node styles (orange border)
  4. Hover highlight styles (red highlight)
  5. Pulse effect styles
  6. Hidden element styles
  7. Edge styles (from EdgeStyles module if available)
  8. Hover-connected edge styles
- **Integration**: Loads edge styles from window.EdgeStyles if available
- **Fallback**: Basic gray edges if EdgeStyles module missing

### Visual Encoding Methods

#### calculateNodeSize(ele)
Calculates node size based on PageRank metric.
- **Input**: ele (CyNode) - Cytoscape node element
- **Returns**: number - Size in pixels
- **Algorithm**:
  - Logarithmic scale for better distribution
  - Maps pagerank to minNodeSize-maxNodeSize range
  - Formula: `log(pagerank * 1000 + 1) / log(1000)`

#### calculateOpacity(ele)
Calculates node opacity based on difficulty.
- **Input**: ele (CyNode) - Cytoscape node element
- **Returns**: number - Opacity value (0.5-1.0)
- **Algorithm**:
  - Linear mapping of difficulty (1-5) to opacity range
  - Lower difficulty = lower opacity (more transparent)

#### truncateLabel(text, maxLength = 30)
Truncates text for display with ellipsis.
- **Input**:
  - text (string) - Original text
  - maxLength (number) - Maximum characters
- **Returns**: string - Truncated text with "..." if needed

### Layout Methods

#### getLayoutConfig()
Returns layout configuration based on initialLayout setting.
- **Returns**: object - Cytoscape layout configuration
- **Supported layouts**:
  - **cose-bilkent** (default): Advanced force-directed layout
    - Wide horizontal initial placement
    - High node repulsion (15000)
    - Large ideal edge length (300)
    - Low gravity (0.01) for spread
    - Tiling with horizontal padding (300px)
    - 1500 iterations for quality
  - **grid**: Simple grid layout (fallback)

### Animation Methods

#### animateAppearance()
Simple fallback animation when AnimationController not available.
- **Algorithm**:
  1. Hide all nodes (opacity 0)
  2. Group nodes by prerequisite_depth
  3. Animate each depth level sequentially
  4. 500ms fade-in per level
  5. 200ms delay between levels
  6. Show all edges after nodes

### Statistics Methods

#### getStats()
Returns graph statistics.
- **Returns**: object
  - nodes: Total node count
  - edges: Total edge count
  - nodeTypes: Count by type
  - edgeTypes: Count by type

#### getNodeTypeCounts()
Counts nodes by type.
- **Returns**: object - {type: count}

#### getEdgeTypeCounts()
Counts edges by type.
- **Returns**: object - {type: count}

## Integration Points

### With EdgeStyles Module
- Checks for `window.EdgeStyles`
- Calls `EdgeStyles.generateEdgeStyles()` if available
- Falls back to basic edge styling if missing

### With AnimationController Module
- Checks for `window.AnimationController`
- Creates instance with configuration
- Delegates animation to controller if available
- Falls back to simple animation otherwise

### With UIControls Module
- Dispatches `k2-graph-ready` event after initialization
- Passes cy and graphCore references
- UIControls handles all user interactions

## Visual Encoding System

### Node Visual Properties

#### Size (PageRank-based)
- Minimum: 20px
- Maximum: 60px
- Logarithmic scaling for better distribution
- Larger nodes = higher importance

#### Opacity (Difficulty-based)
- Minimum: 0.5 (difficulty 1)
- Maximum: 1.0 (difficulty 5)
- Linear scaling
- More opaque = more difficult

#### Shape (Type-based)
- Chunk: hexagon
- Concept: star
- Assessment: roundrectangle

#### Color (Type-based)
- Chunk: #3498db (blue)
- Concept: #2ecc71 (green)
- Assessment: #f39c12 (orange)

### Edge Visual Properties
Delegated to EdgeStyles module, but includes:
- Color by relationship type
- Width by importance (1-4px)
- Style (solid, dashed, dotted) by type
- Opacity by relevance

## CSS Classes

### Node Classes
- `chunk` - Learning chunk nodes
- `concept` - Concept nodes
- `assessment` - Assessment nodes
- `hover-highlight` - Red highlight on hover
- `pulse` - Pulsing animation effect
- `hidden` - Hidden nodes

### Edge Classes
- `hover-connected` - Red highlight for connected edges
- `hidden-edge` - Hidden edges

## Events

### Dispatched Events

#### k2-graph-ready
Fired after graph initialization completes.
- **Detail**: `{cy: Cytoscape, graphCore: GraphCore}`
- **Timing**: After layout and initial animation
- **Listeners**: UIControls, PathFinder, ClustersBridges

#### graph-animation-complete
Fired when initial animation finishes.
- **Detail**: None
- **Timing**: After all nodes and edges visible
- **Listeners**: Filter panel appearance

## Pre-positioning Strategy

Before running layout, nodes are pre-positioned in a wide horizontal band:
- X: Random in range `[-width*1.5, width*1.5]`
- Y: Random in range `[-height*0.15, height*0.15]`

This helps cose-bilkent create a wider, more horizontal layout suitable for landscape viewing.

## Performance Considerations

### Batch Operations
- Element preparation done in single pass
- Styles generated once and cached
- Pre-positioning before layout reduces iterations

### Lazy Loading
- AnimationController loaded only if needed
- EdgeStyles module optional
- Fallback animations simpler

### Memory Management
- Single Cytoscape instance
- Data references (not copies)
- Styles array reused

## Error Handling

### Missing Modules
- EdgeStyles: Falls back to basic edge styling
- AnimationController: Falls back to simple animation
- Both logged as warnings, not errors

### Invalid Data
- Missing node fields: Uses defaults
- Invalid difficulty: Defaults to 3
- Missing pagerank: Defaults to 0.01

## Configuration Sources

Configuration is merged from multiple sources:
1. Default configuration (hardcoded)
2. User config passed to constructor
3. Window configuration objects:
   - `window.vizConfig`
   - `window.colorsConfig`
   - `window.nodeShapes`

## Known Issues

### Label Display
The `showLabelsOnHover` feature is deprecated. Label display is now handled by UIControls through tooltips for better performance and flexibility.

### Layout Convergence
Large graphs (>1000 nodes) may take significant time to converge with cose-bilkent layout. Consider using simpler layouts for very large graphs.

## Testing Considerations

### Console Testing
```javascript
// After initialization
window.graphCore.getStats()          // View statistics
window.cy.nodes().length             // Node count
window.cy.edges('[type="PREREQUISITE"]').length  // Specific edge count
window.cy.zoom()                      // Current zoom level
```

### Visual Testing
1. Check node sizes vary with PageRank
2. Verify opacity changes with difficulty
3. Confirm shapes match node types
4. Test edge colors match types
5. Verify animation sequence by depth

## Dependencies

### External Libraries
- **Cytoscape.js**: Core graph library (required)
- **cytoscape-cose-bilkent**: Layout algorithm (required for default layout)

### Internal Modules
- **EdgeStyles**: Edge styling definitions (optional but recommended)
- **AnimationController**: Animation sequencing (optional)
- **UIControls**: User interface (required for interactions)

## Data Structure Requirements

### Node Data Fields
Required fields for proper visualization:
- **id** (string): Unique node identifier
- **type** (string): One of: "Chunk", "Concept", "Assessment"
- **text** (string): Display label and content
- **difficulty** (number): 1-5 scale for complexity visualization
- **pagerank** (number): For node size calculation
- **betweenness_centrality** (number): Network centrality metric
- **learning_effort** (number): Educational effort metric
- **cluster_id** (number, optional): For cluster-based coloring
- **prerequisite_depth** (number): For animation sequence
- **degree_in** (number): Incoming edge count
- **degree_out** (number): Outgoing edge count
- **definition** (string, optional): Source text for node content

### Edge Data Fields
- **id** (string): Unique edge identifier
- **source** (string): Source node ID
- **target** (string): Target node ID
- **type** (string): Relationship type (PREREQUISITE, ELABORATES, etc.)
- **weight** (number, optional): Edge importance weight

### Concept Dictionary Structure
```javascript
{
  "concepts": [
    {
      "id": "concept_001",
      "name": "Machine Learning",
      "definition": "A subset of artificial intelligence...",
      "aliases": ["ML", "Машинное обучение"],
      "mention_count": {
        "chunk_001": 3,
        "chunk_002": 1
      }
    }
  ]
}
```

## Window Global Objects

The following objects are exposed globally for debugging and integration:
- **window.graphData**: Complete graph data structure
- **window.conceptData**: Concept dictionary
- **window.cy**: Cytoscape instance (after initialization)
- **window.graphCore**: GraphCore instance
- **window.vizConfig**: Visualization configuration
- **window.colorsConfig**: Color scheme configuration
- **window.uiConfig**: UI settings configuration
- **window.nodeShapes**: Node shape mappings

## Notes

- Core module focuses on visualization setup, not interaction
- All user interactions delegated to UIControls
- Modular design allows optional enhancement modules
- Performance optimized for 100-1000 node graphs
- Wide horizontal layout optimized for desktop viewing
# viz_clusters_bridges.md

## Status: READY

Module for visualizing knowledge clusters and bridge nodes in the K2-18 graph visualization.

## Module Purpose

The `clusters_bridges.js` module provides interactive visualization of knowledge clusters identified through Louvain community detection. It highlights thematic groupings of nodes, bridge nodes that connect different clusters, and inter-cluster relationships. The module emphasizes the graph's modular structure and helps users understand how different knowledge areas are connected.

## Public API

### ClustersBridges Object

The main object exposed globally as `window.ClustersBridges` with the following structure:

#### Properties

- **active** (boolean) - Whether cluster mode is currently active
- **originalStyles** (object|null) - Storage for original element styles before cluster visualization
- **clusterColors** (object) - Mapping of cluster_id to assigned color
- **hoveredCluster** (number|null) - ID of currently hovered cluster
- **overlayEnabled** (boolean) - Whether overlay effects are applied
- **boundHandleNodeHover** (function|null) - Bound hover handler for cleanup
- **boundHandleNodeUnhover** (function|null) - Bound unhover handler for cleanup
- **config** (object) - Configuration parameters for visualization

#### Configuration Object

```javascript
config: {
    bridge_threshold: 0.7,           // Minimum bridge_score to highlight
    overlay_padding: 30,             // Padding for cluster overlay effect
    overlay_opacity_normal: 0.1,     // Default overlay opacity
    overlay_opacity_hover: 0.3,      // Hover overlay opacity
    bridge_border_width: 4,          // Border width for bridge nodes
    bridge_border_color: '#ff6b6b',  // Red border for bridges
    inter_cluster_dash: [8, 4],      // Dash pattern for inter-cluster edges
    animation_duration: 300          // Animation duration in ms
}
```

### Core Methods

#### init(cy, graphCore, config)
Initializes the module with Cytoscape instance and configuration.
- **Input**:
  - cy (Cytoscape) - The Cytoscape instance
  - graphCore (GraphCore) - Reference to GraphCore module
  - config (object) - Configuration object containing colors palette
- **Side effects**: Stores cluster palette from config.colors.cluster_palette

#### activate()
Activates cluster visualization mode.
- **Algorithm**:
  1. Check if already active (early return)
  2. Save original styles of all elements
  3. Apply cluster visualization styles
  4. Setup hover event handlers
- **Side effects**: 
  - Modifies node and edge styles
  - Adds event listeners
  - Sets active flag to true

#### deactivate()
Deactivates cluster visualization mode and restores original state.
- **Algorithm**:
  1. Check if inactive (early return)
  2. Remove hover event handlers
  3. Restore all original styles
  4. Clear internal state
- **Side effects**:
  - Restores original element styles
  - Removes event listeners
  - Resets all state properties

### Style Management Methods

#### saveOriginalStyles()
Saves current styles of all nodes and edges before applying cluster visualization.
- **Saved node properties**:
  - background-color, background-opacity
  - border-width, border-color, border-opacity
  - opacity, overlay-color, overlay-padding, overlay-opacity
- **Saved edge properties**:
  - line-color, line-style, line-dash-pattern
  - width, opacity, line-opacity
  - target-arrow-color, source-arrow-color
- **Storage**: this.originalStyles = { nodes: {}, edges: {} }

#### restoreOriginalStyles()
Restores all saved styles to their original values.
- **Algorithm**:
  1. Check if original styles exist
  2. Batch restore operations for performance
  3. Remove overlay styles completely
  4. Remove inline styles for nodes to let original class styles take over
  5. Restore only specific edge styles: line-style, line-dash-pattern, width, opacity
- **Note**: Only 4 edge style properties are restored, not all saved properties. Node styles are removed rather than restored to allow CSS class styles to take precedence

### Cluster Visualization Methods

#### applyClusterStyles()
Main method that orchestrates the cluster visualization.
- **Steps**:
  1. assignClusterColors() - Map clusters to colors
  2. styleNodesByCluster() - Color nodes by cluster
  3. highlightBridgeNodes() - Add borders to bridges
  4. styleInterClusterEdges() - Style cross-cluster edges
  5. applyClusterOverlays() - Add overlay effects

#### assignClusterColors()
Maps each cluster_id to a color from the palette.
- **Algorithm**:
  1. Collect unique cluster_ids from all nodes
  2. Sort cluster IDs numerically
  3. Assign colors from palette cyclically
- **Palette**: 12 colors from config.colors.cluster_palette
- **Result**: this.clusterColors = { cluster_id: color_hex }

#### styleNodesByCluster()
Attempts to apply cluster colors to nodes.
- **Current behavior**:
  - Adds inline styles for background-color with cluster color
  - Sets background-opacity to 1 (full opacity for bright colors)
  - Adds 'cluster-mode-active' marker class
- **Known limitation**: Due to Cytoscape.js stylesheet priority, type-based class selectors (.chunk, .concept, .assessment) override inline styles, preventing color changes
- **Performance**: Uses cy.batch() for bulk updates

#### highlightBridgeNodes()
Highlights nodes with high bridge_score.
- **Criteria**: bridge_score > config.bridge_threshold (0.7)
- **Styling**:
  - border-width: 4px
  - border-color: #ff6b6b (red)
  - border-opacity: 1

#### styleInterClusterEdges()
Styles edges that connect different clusters.
- **Detection**: source.cluster_id !== target.cluster_id
- **Styling**:
  - line-style: dashed
  - line-dash-pattern: [8, 4]
  - width: 1.5x original
  - line-opacity: 0.8

#### applyClusterOverlays()
Applies overlay effects to visually group cluster nodes.
- **Condition**: Applied only to clusters with >2 nodes
- **Styling**:
  - overlay-color: cluster color
  - overlay-padding: 30px
  - overlay-opacity: 0.1 (normal), 0.3 (hover)
- **Note**: Cytoscape doesn't support true convex hulls, uses node overlays

### Interaction Methods

#### setupHoverEffects()
Sets up mouse event handlers for cluster highlighting.
- **Events**:
  - mouseover on nodes → handleNodeHover()
  - mouseout on nodes → handleNodeUnhover()
- **Implementation**: Creates bound handlers (boundHandleNodeHover, boundHandleNodeUnhover) for proper cleanup

#### removeHoverEffects()
Removes all hover event handlers.
- **Implementation**: Only removes the specific bound handlers to avoid conflicts with other modules

#### handleNodeHover(evt)
Handles mouse entering a node.
- **Algorithm**:
  1. Check if mode is active
  2. Get cluster_id from hovered node
  3. Call highlightCluster() if different cluster

#### handleNodeUnhover(evt)
Handles mouse leaving a node.
- **Algorithm**:
  1. Check if mode is active
  2. Get cluster_id from unhovered node
  3. Call unhighlightCluster() if same as hoveredCluster

#### highlightCluster(clusterId)
Emphasizes all nodes in a cluster.
- **Effects**:
  - Animate overlay-opacity to 0.3 (if overlay enabled)
- **Animation**: 300ms duration using Cytoscape animate API
- **Note**: No background-opacity change in current implementation

#### unhighlightCluster(clusterId)
Returns cluster to normal state.
- **Effects**:
  - Animate overlay-opacity to 0.1 (if overlay enabled)
- **Animation**: 300ms duration using Cytoscape animate API
- **Note**: No background-opacity change in current implementation

### Utility Methods

#### getClusterStats()
Returns statistics about current clustering.
- **Returns**:
```javascript
{
    active: boolean,              // Mode active status
    totalClusters: number,        // Number of clusters
    clusterSizes: {              // Nodes per cluster
        cluster_id: node_count
    },
    bridgeNodes: number,          // Nodes with bridge_score > 0.7
    interClusterEdges: number     // Edges between clusters
}
```

## Visual Effects

### Node Visualization
1. **Cluster coloring**: Currently limited - nodes retain type-based colors due to stylesheet priority
2. **Bridge highlighting**: Red 4px border for nodes with bridge_score > 0.7 (working)
3. **Overlay grouping**: Semi-transparent square overlay around cluster nodes (working)
4. **Hover emphasis**: Increased overlay opacity on cluster hover (working)

### Edge Visualization
1. **Inter-cluster edges**: Dashed lines with [8,4] pattern, 1.2x width, opacity 0.5 (working)
2. **Intra-cluster edges**: Dimmed to opacity 0.3 (working)

### Color Palette
Default 12-color palette from config.toml:
- #3498db (blue), #e74c3c (red), #2ecc71 (green)
- #f39c12 (orange), #9b59b6 (purple), #1abc9c (turquoise)
- #34495e (gray), #e67e22 (dark orange), #16a085 (dark turquoise)
- #8e44ad (dark purple), #2c3e50 (dark gray), #27ae60 (dark green)

## Event System

### Mode Activation
The module listens for the custom `mode-changed` event:
```javascript
document.addEventListener('mode-changed', (e) => {
    if (e.detail.mode === 'clusters') {
        ClustersBridges.activate();
    } else if (ClustersBridges.active) {
        ClustersBridges.deactivate();
    }
});
```

### Integration Points
- **UIControls**: Triggers mode-changed events via mode buttons
- **GraphCore**: Provides cy instance and base styles
- **Config**: Supplies color palette and visualization parameters

## Boundary Cases

### Missing Cluster Data
- **No cluster_id**: Nodes without cluster_id maintain original color
- **Empty graph**: No visualization changes applied
- **Single cluster**: All nodes get same color, no inter-cluster edges

### Bridge Detection
- **No bridge_score**: Nodes without score are not highlighted
- **All bridges**: If all nodes are bridges, all get red borders
- **No bridges**: No border styling applied

### Edge Cases
- **Self-loops**: Ignored for inter-cluster detection
- **Disconnected nodes**: Maintain cluster color if assigned
- **Missing data**: Gracefully handles undefined/null values

## Performance Considerations

### Batch Operations
All style changes wrapped in `cy.batch()` for:
- Reduced redraws
- Better performance with large graphs
- Atomic visual updates

### Memory Management
- Original styles stored once on activation
- Cleared completely on deactivation
- Cluster colors computed once per activation

### Animation Performance
- 300ms transitions for smooth effects
- Overlay animations use Cytoscape native animation API
- No animation queuing to prevent lag

## CSS Classes and Styles

### Added CSS Classes
None directly - all styling via Cytoscape style API

### CSS Support (styles.css)
```css
/* Cluster mode overlay animation */
.cy-cluster-mode .node[overlay-opacity] {
    transition: overlay-opacity 0.3s ease;
}

/* Bridge node emphasis */
.cy-cluster-mode .node.bridge-node {
    border-width: 4px !important;
    border-color: #ff6b6b !important;
}

/* Inter-cluster edge styling */
.cy-cluster-mode .edge.inter-cluster {
    line-style: dashed !important;
    line-dash-pattern: 8 4 !important;
}

/* Cluster hover effect */
.cluster-highlight {
    transition: all 0.3s ease;
    background-opacity: 1 !important;
}
```

## Testing

### Console Testing
```javascript
// Activate cluster mode
ClustersBridges.activate()

// Check state
ClustersBridges.active                    // Should be true
ClustersBridges.getClusterStats()         // View statistics
Object.keys(ClustersBridges.clusterColors) // See cluster count

// Test hover
// Hover over nodes to see cluster highlighting

// Deactivate
ClustersBridges.deactivate()
ClustersBridges.active                    // Should be false
```

### Visual Testing
1. **Activation**: Click Clusters button or press 'C'
   - Nodes should change colors by cluster
   - Bridge nodes get red borders
   - Inter-cluster edges become dashed

2. **Hover**: Mouse over any node
   - Entire cluster should highlight
   - Overlay opacity increases
   - Smooth animation transition

3. **Deactivation**: Click button again or press Esc
   - All original styles restored
   - No cluster coloring remains
   - Hover effects removed

### Edge Case Testing
1. **Mode switching**: Activate Path mode after Clusters
   - Clusters should fully deactivate
   - No style artifacts remain

2. **Rapid toggling**: Quickly toggle mode on/off
   - Styles should restore correctly
   - No accumulation of event handlers

3. **Large graphs**: Test with >500 nodes
   - Performance should remain smooth
   - Batch operations prevent lag

## Dependencies

### External Libraries
- **Cytoscape.js**: Core graph library (required)
- No additional Cytoscape extensions needed

### Internal Modules
- **GraphCore**: Provides cy instance and base configuration
- **UIControls**: Manages mode switching and button states
- **Config**: Supplies colors and visualization parameters

## Notes

- Overlay effects provide visual grouping without true convex hulls
- Bridge detection uses pre-computed bridge_score from graph2metrics
- Cluster IDs come from Louvain community detection in backend
- Color assignment is deterministic (sorted cluster IDs)
- Full style restoration ensures clean mode switching
- Module designed for graphs with 2-20 clusters optimally

## Known Limitations

1. **Node color override issue**: Cytoscape.js stylesheet selectors have priority over inline styles set via `node.style()`. Type-based classes (.chunk, .concept, .assessment) prevent cluster colors from being applied to nodes. This requires modifying the Cytoscape stylesheet dynamically or using a different approach.

2. **Square overlays**: Cytoscape.js overlay shapes are limited to rectangles for regular nodes. Circular overlays would require compound nodes or custom rendering.

## Future Improvements

1. Implement dynamic stylesheet modification to properly apply cluster colors
2. Consider using compound nodes for better cluster visualization
3. Add cluster statistics display on hover
4. Implement cluster-based layout options
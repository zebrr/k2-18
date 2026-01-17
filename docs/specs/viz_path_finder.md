# viz_path_finder.md - Path Finder Module Specification

## Status: IN_PROGRESS

## Overview
Path Finder module (`/viz/static/path_finder.js`) implements the Learning Path Finder (GPS Mode) for the K2-18 knowledge graph visualization. It allows users to select two nodes and visualize the shortest path between them.

## Purpose
Provide an interactive way to find and visualize learning paths between concepts in the knowledge graph, helping users understand the connections and progression between different topics.

## Dependencies
- **Cytoscape.js**: Core graph library for path calculations and visualization
- **UIControls**: Integration with mode system and UI state management
- **Graph data**: Access to nodes and edges with their properties

## Core Functionality

### 1. Mode Activation/Deactivation
- **activate()**: Enables path finding mode
  - Disables node dragging (ungrabify)
  - Changes cursor to crosshair
  - Shows instruction toast
  - Does NOT modify any node/edge styles on activation

- **deactivate()**: Exits path finding mode
  - Restores all elements to original state
  - Re-enables node dragging (grabify)
  - Resets cursor
  - Clears any active paths and selections

### 2. Node Selection
- First click: Selects start node
  - Adds pulsing animation (node size changes from original to 1.5x)
  - Shows orange border (#ff6b00)
  - Displays toast for second node selection
  - Uses Map to store animation intervals per node

- Second click: Selects end node
  - Adds pulsing animation to second node
  - Triggers path finding algorithm
  - Shows "Searching..." toast

### 3. Path Finding Algorithm
- **findShortestPath(startId, endId)**: Uses Dijkstra's algorithm
  - Finds path with minimum number of steps (hops)
  - Each edge has weight = 1 (unweighted shortest path)
  - Calculates total difficulty (sum of node difficulties along path)
  - Returns: `{found: boolean, path: Collection, distance: number, totalDifficulty: number}`

- **findSimplePath(startId, endId)** [DEPRECATED - kept in code but not used]:
  - Uses weighted Dijkstra based on node difficulty
  - Weight function: w(u→v) = α + β·difficulty(v) + σ·(1−edge.weight)
  - Returns path with minimum total difficulty

- **findFastPath_OLD(startId, endId)** [DEPRECATED - legacy code]:
  - Progressive edge type ladder expansion
  - Not used in current implementation

### 4. Path Visualization
- **displayPath(result)**: Visualizes the found path
  - Path edges: Bright green (#00ff00), thick lines (width: 6)
  - Straight lines for clarity (no curves)
  - High z-index (1000) to appear on top
  - Weak segments (REFER_BACK, HINT_FORWARD, MENTIONS) shown as dashed lines

- **dimOtherElements()**: Dims non-path elements
  - Other edges: opacity 0.1, events disabled
  - Other nodes: opacity 0.4, but remain clickable
  - Path nodes and edges remain fully visible

### 5. Visual Feedback

#### Pulsing Animation
- Applied to selected nodes (start and end)
- Animates node size from original to 1.5x
- Orange border (#ff6b00)
- Smooth pulsing at 50ms intervals
- Uses Map (pulseAnimations) to track individual node animations
- Proper cleanup when stopping animations

#### Metrics Panel
- Shows path statistics:
  - Number of steps (edges in path)
  - Total difficulty (sum of node difficulties)
- Dark semi-transparent background
- Positioned below filter panel (top: 120px)
- Auto-shows when path is found

#### Toast Notifications
- Instructions and feedback messages
- Auto-dismiss after specified duration
- Different messages for different states
- Messages in Russian:
  - "📍 Select two nodes for path finding" (Выберите два узла для поиска пути)
  - "📍 Select second node" (Выберите второй узел)
  - "🔍 Searching..." (Поиск пути...)
  - "⚠️ Path not found" (Путь не найден)
  - "❌ Selection cancelled" (Выбор отменён)
  - "🔄 Mode reset" (Режим сброшен)

### 6. State Management
- **reset()**: Clears current selection
  - Stops pulsing animations
  - Clears selected nodes
  - Restores elements ONLY if path was displayed (checks this.shortestPath)
  - Resets internal state variables
  - Note: Uses this.shortestPath instead of this.path

- **restoreAllElements()**: Restores graph to original state
  - **Critical**: Sets opacity to 1 for all nodes and edges (not removes)
  - Removes all added classes (path-dimmed, path-selected, etc.)
  - Removes path-specific inline styles
  - Preserves original graph styles

- **dimOtherElements()**: Dims non-path elements
  - Edges: opacity 0.1, events disabled
  - Nodes: opacity 0.4, remain clickable
  - Uses 'ghost' custom property for dimmed state indication

## Configuration
Loaded from `window.pathModeConfig`:
```javascript
{
    edge_type_ladder: [...],  // Not used in current simplified version
    default_difficulty: 3,    // Default node difficulty if not specified
    animation_duration: 500,  // Toast and panel animations
    // ... other config options
}
```

## Event Handling

### Node Events
- **tap**: Handles node selection
  - First tap: Select start node
  - Second tap: Select end node (triggers path finding)
  - Same node tap: Cancel selection

### Background Events
- **tap on background**: Cancels current selection

### Keyboard Events
- **Escape**: Resets current selection

## Integration with UIControls

### Hover Behavior
- UIControls checks for `path-dimmed` class
- Skips hover effects for dimmed nodes/edges
- Prevents hover from restoring dimmed elements

### Mode Coordination
- UIControls checks `activeMode === 'path'`
- Disables node popup in path mode
- Allows PathFinder to handle all interactions

## CSS Classes

### Node Classes
- `path-selected`: Selected start/end nodes
- `path-dimmed`: Dimmed nodes not in path
- `path-no-connection`: Nodes when no path found (red pulse)

### Custom Properties
- `ghost`: 'yes' - Custom property to indicate dimmed state for hover skip

### Edge Classes
- `path-shortest`: Edges in the shortest path
- `path-dimmed`: Dimmed edges not in path
- `path-weak-segment`: Weak connection types (dashed)

## Error Handling
- No path found: Shows warning toast and red pulse on nodes
- Auto-reset after 3 seconds if no path exists
- Console logging for debugging

## Performance Considerations
- Batch updates using `cy.batch()` for style changes
- Efficient Dijkstra implementation from Cytoscape
- Minimal DOM manipulation (reuses panels)
- Cleanup of intervals and timeouts

## Known Limitations
1. Currently finds only shortest path (minimum hops)
2. Does not consider edge types or semantic relationships
3. Path difficulty is simple sum, not weighted by edge types
4. No alternative paths shown
5. Directed graph - some nodes may not be reachable

## Legacy Code
The module contains deprecated methods that are kept for reference:
- **findSimplePath()**: Weighted path finding by difficulty
- **findFastPath_OLD()**: Edge type ladder expansion
- **displayFastPath_OLD()**: Old visualization for fast path
- **displaySimplePath()**: Old visualization for simple path
- **dimOtherEdges_OLD()**: Old dimming method
- Properties: fastPath, simplePath (unused)

These methods are not used but preserved in code.

## Future Enhancements (from original spec, not implemented)
- Multiple path types (fast vs. easy)
- Edge type ladder for progressive filtering
- Semantic path finding based on relationships
- Path comparison and recommendations
- Export/save path functionality

## Testing Checklist
- [ ] Mode activation/deactivation without side effects
- [ ] Node selection and pulsing animation
- [ ] Path finding between connected nodes
- [ ] No path scenario handling
- [ ] Hover behavior with dimmed elements
- [ ] Restoration of all elements on exit
- [ ] Metrics panel display and content
- [ ] Toast notifications timing
- [ ] Keyboard shortcuts (Escape)
- [ ] Integration with other modes

## Version History
- v1.0: Initial implementation with two path types (fast/simple)
- v2.0: Simplified to single shortest path algorithm
- v2.1: Fixed node visibility and hover interaction issues
- v2.2: Current version with legacy code preserved
# K2-18 - Educational Knowledge Graph Visualization Module

## Why You Need This

After running the main K2-18 pipeline, you get a knowledge graph with thousands of nodes and edges. But how do you know if it's good? How do you explore it? How do you show it to others?

The visualization tools solves four key problems:

1. **Quality Check** - Verify the graph structure makes educational sense
2. **Educational Metrics** - Measure learning complexity and prerequisite chains  
3. **Research Tool** - Explore knowledge clusters and connections
4. **Presentation** - Show results to stakeholders with interactive visualization

## What You Get

Two complementary HTML tools that work in any browser:

**Interactive Graph** (`knowledge_graph.html`):
- Visual knowledge graph you can explore
- Computed metrics showing which concepts are fundamental
- Automatic clustering of related topics
- Tools to analyze learning paths

**Detailed Viewer** (`knowledge_graph_viewer.html`):
- Three-column tabular interface for methodical exploration
- Full text content with markdown/code/math rendering
- Educational metrics with explanations
- Edge inspection with related nodes

## Pipeline

```
1. Copy results from main pipeline to viz folder:
   /data/out/ConceptDictionary.json         → /viz/data/in/ConceptDictionary.json
   /data/out/LearningChunkGraph_*.json      → /viz/data/in/LearningChunkGraph.json

2. Run visualization pipeline:
   python -m viz.graph2metrics              # Compute educational metrics
   python -m viz.graph_fix                  # (Optional) Mark LLM content for QA
   python -m viz.graph2html                 # Generate interactive graph
   python -m viz.graph2viewer               # Generate detailed viewer

3. Open results:
   viz/data/out/knowledge_graph.html        # Interactive graph visualization
   viz/data/out/knowledge_graph_viewer.html # Detailed tabular viewer
```

Works in browser. No servers. No installation.

## Using the Tools

### Graph Visualization (graph2html)

#### Main Interface

**Top Panel**
- Live counters of visible nodes and edges
- Checkboxes to show/hide node types (Chunks, Concepts, Assessments) and edges

**Right Panel** (click 📚 tab to open)
- **Dictionary tab**: All concepts with usage counts, click for definitions
- **TOP Nodes tab**: Most important nodes by different metrics

**Left Panel** (click 📖 tab to open)
- Course content in sequence
- Color-coded by topic clusters
- Click to navigate to specific chunk

**Graph Interaction**
- Click node - see detailed information
- Drag to pan, scroll to zoom
- Hover - shows preview text and highlights all connections

#### Planned Features

Three advanced modes accessible via buttons under header:
- 🛤️ **Path Mode** - Find learning paths between concepts (_in Beta_)
- 🎨 **Clusters** - Visualize topic groupings (_in Beta_)
- ▶️ **Tour** - Automated presentation (_NOT READY_)

Currently these buttons may be visible but don't work yet.

### Detailed Viewer (graph2viewer)

For methodologists who need to examine every node and edge in detail:

**Column A - Search & Filter**
- Real-time search across all nodes
- Filter by type (Chunk, Concept, Assessment)
- Nodes sorted: Concepts first (alphabetically), then Chunks/Assessments by position
- Color-coded by cluster with cluster ID badge

**Column B - Active Node**
- Toggle between formatted view and raw JSON
- Full text content with markdown/code/math rendering
- All 12 educational metrics with click-to-show explanations
- Complete edges table - click any edge to inspect

**Column C - Edge Analysis**
- Shows relationship visualization when edge selected
- Displays related node with full content
- Button to make related node active for navigation

## Metrics Computed

The module calculates 12 metrics that reveal graph structure:

### Educational Metrics (3)
- **prerequisite_depth** - How many steps from basics (0 = entry level)
- **learning_effort** - Cumulative difficulty through prerequisites
- **educational_importance** - Importance in learning context

### Centrality Metrics (6)
- **degree_in** - Number of incoming connections
- **degree_out** - Number of outgoing connections
- **degree_centrality** - Normalized connectivity
- **pagerank** - Overall importance in the graph
- **betweenness_centrality** - Concepts that bridge different areas
- **out-closeness** - How easily node reaches others

### Structure Analysis (3)
- **cluster_id** - Automatic grouping of related content
- **component_id** - Identifies disconnected parts
- **bridge_score** - Nodes connecting different clusters

These metrics help identify:
- Entry points for learning (depth = 0)
- Fundamental concepts (high PageRank)
- Knowledge bridges (high betweenness)
- Topic clusters (same cluster_id)

## Configuration

Edit `/viz/config.toml` to adjust:
- Clustering sensitivity
- Visual appearance
- Performance settings
- Library embedding (CDN vs local)

## Quality Control

Check if metrics make sense:
```bash
python -m viz.anomaly_detector
```

This will report:
- Invalid metric values
- Statistical outliers
- Potential data issues

## Troubleshooting

**Graph too large (>1000 nodes)**
- May experience SEVERE performance issues
- Consider testing on smaller subsets first

**Nothing appears**
- Check browser console for errors
- Verify input files exist
- Try Chrome/Firefox/Edge (no IE11)

## Technical Details

For developers:
- Metrics implementation: `/docs/specs/viz_metrics_reference.md`
- Module specifications: `/docs/specs/viz_*.md`
- Reference graphs with validated metrics: `/viz/data/test/`

Dependencies:
- Python: NetworkX, python-louvain, Jinja2, minify-html
- JavaScript (embedded in HTML):
  - Cytoscape.js - graph visualization
  - Marked.js - markdown parsing
  - Highlight.js - code syntax highlighting
  - MathJax - mathematical formula rendering
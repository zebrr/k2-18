# K2-18 - Educational Knowledge Graph Visualization Module

## Why You Need This

After running the main K2-18 pipeline, you get a knowledge graph with thousands of nodes and edges. But how do you know if it's good? How do you explore it? How do you show it to others?

The visualization module solves four key problems:

1. **Quality Check** - Verify the graph structure makes educational sense
2. **Educational Metrics** - Measure learning complexity and prerequisite chains  
3. **Research Tool** - Explore knowledge clusters and connections
4. **Presentation** - Show results to stakeholders with interactive visualization

## What You Get

A standalone HTML file that works in any browser with:
- Interactive knowledge graph you can explore
- Computed metrics showing which concepts are fundamental
- Automatic clustering of related topics
- Tools to analyze learning paths

## Pipeline

```
1. Copy results from main pipeline to viz folder:
   /data/out/ConceptDictionary.json     → /viz/data/in/ConceptDictionary.json
   /data/out/LearningChunkGraph_*.json  → /viz/data/in/LearningChunkGraph.json

2. Run visualization pipeline:
   python -m viz.graph2metrics          # Compute educational metrics
   python -m viz.graph_fix              # (Optional) Mark LLM content for QA
   python -m viz.graph2html             # Generate interactive HTML

3. Open result:
   viz/data/out/knowledge_graph.html
```

That's it. No servers. No installation. Works in browser.

## Using the Visualization

### Main Interface

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

### Planned Features

Three advanced modes accessible via buttons under header:
- 🛤️ **Path Mode** - Find learning paths between concepts (in Beta)
- 🎨 **Clusters** - Visualize topic groupings (in Beta)
- ▶️ **Tour** - Automated presentation (NOT READY)

Currently these buttons may be visible but don't work yet.

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
- Python: NetworkX, python-louvain, Jinja2
- JavaScript: Cytoscape.js (embedded)
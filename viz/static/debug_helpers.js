/**
 * Debug helpers for K2-18 Knowledge Graph visualization
 * Usage: Open browser console and use debugHelpers.* functions
 */

const debugHelpers = {
    // Show graph statistics
    stats: () => {
        if (!window.cy) {
            console.error('Cytoscape not initialized yet');
            return;
        }
        
        const cy = window.cy;
        const nodes = cy.nodes();
        const edges = cy.edges();
        
        console.log('=== Graph Statistics ===');
        console.log(`Total nodes: ${nodes.length}`);
        console.log(`Total edges: ${edges.length}`);
        
        // Node type breakdown
        const nodeTypes = {};
        nodes.forEach(n => {
            const type = n.data('type');
            nodeTypes[type] = (nodeTypes[type] || 0) + 1;
        });
        console.log('Node types:', nodeTypes);
        
        // Edge type breakdown
        const edgeTypes = {};
        edges.forEach(e => {
            const type = e.data('type');
            edgeTypes[type] = (edgeTypes[type] || 0) + 1;
        });
        console.log('Edge types:', edgeTypes);
        
        // Components
        const components = cy.elements().components();
        console.log(`Connected components: ${components.length}`);
        
        // Clusters
        const clusters = new Set(nodes.map(n => n.data('cluster_id')));
        console.log(`Clusters: ${clusters.size}`, [...clusters]);
        
        return {
            nodes: nodes.length,
            edges: edges.length,
            nodeTypes,
            edgeTypes,
            components: components.length,
            clusters: clusters.size
        };
    },
    
    // Find nodes by text (partial match)
    findByText: (text) => {
        if (!window.cy) {
            console.error('Cytoscape not initialized yet');
            return [];
        }
        
        const searchText = text.toLowerCase();
        const matches = window.cy.nodes().filter(n => {
            const nodeText = (n.data('text') || n.data('label') || '').toLowerCase();
            return nodeText.includes(searchText);
        });
        
        console.log(`Found ${matches.length} nodes matching "${text}":`);
        matches.forEach(n => {
            console.log(`  - ${n.id()}: "${n.data('text')}" (${n.data('type')})`);
        });
        
        return matches;
    },
    
    // Show neighbors of a node
    neighbors: (nodeId) => {
        if (!window.cy) {
            console.error('Cytoscape not initialized yet');
            return;
        }
        
        const node = window.cy.$(`#${nodeId}`);
        if (node.length === 0) {
            console.error(`Node "${nodeId}" not found`);
            return;
        }
        
        const incoming = node.incomers().nodes();
        const outgoing = node.outgoers().nodes();
        
        console.log(`=== Neighbors of ${nodeId} ===`);
        console.log('Incoming nodes:', incoming.map(n => ({
            id: n.id(),
            text: n.data('text'),
            type: n.data('type')
        })));
        console.log('Outgoing nodes:', outgoing.map(n => ({
            id: n.id(),
            text: n.data('text'),
            type: n.data('type')
        })));
        
        return {
            incoming: incoming.map(n => n.id()),
            outgoing: outgoing.map(n => n.id())
        };
    },
    
    // Highlight shortest path between two nodes
    highlightPath: (fromId, toId) => {
        if (!window.cy) {
            console.error('Cytoscape not initialized yet');
            return;
        }
        
        const cy = window.cy;
        
        // Clear previous highlights
        cy.elements().removeClass('highlighted dimmed');
        
        const source = cy.$(`#${fromId}`);
        const target = cy.$(`#${toId}`);
        
        if (source.length === 0) {
            console.error(`Source node "${fromId}" not found`);
            return;
        }
        if (target.length === 0) {
            console.error(`Target node "${toId}" not found`);
            return;
        }
        
        // Find shortest path
        const dijkstra = cy.elements().dijkstra(source, function(edge) {
            return 1 / (edge.data('weight') || 0.1); // Inverse weight for distance
        });
        
        const path = dijkstra.pathTo(target);
        
        if (path.length === 0) {
            console.log(`No path found from ${fromId} to ${toId}`);
            return;
        }
        
        // Highlight path
        cy.elements().addClass('dimmed');
        path.removeClass('dimmed').addClass('highlighted');
        
        // Add styles for highlighting
        cy.style()
            .selector('.highlighted')
            .style({
                'background-color': '#e74c3c',
                'line-color': '#e74c3c',
                'target-arrow-color': '#e74c3c',
                'z-index': 999
            })
            .selector('.dimmed')
            .style({
                'opacity': 0.2
            })
            .update();
        
        console.log(`Path from ${fromId} to ${toId}: ${path.nodes().map(n => n.id()).join(' -> ')}`);
        console.log(`Path length: ${path.nodes().length - 1} steps`);
        
        return path;
    },
    
    // Clear all highlights
    clearHighlights: () => {
        if (!window.cy) {
            console.error('Cytoscape not initialized yet');
            return;
        }
        
        window.cy.elements().removeClass('highlighted dimmed');
        window.cy.style().resetToDefault().update();
        console.log('Highlights cleared');
    },
    
    // Show nodes by cluster
    showCluster: (clusterId) => {
        if (!window.cy) {
            console.error('Cytoscape not initialized yet');
            return;
        }
        
        const nodes = window.cy.nodes().filter(n => n.data('cluster_id') === clusterId);
        
        console.log(`=== Cluster ${clusterId} ===`);
        console.log(`Nodes (${nodes.length}):`);
        nodes.forEach(n => {
            console.log(`  - ${n.id()}: "${n.data('text')}" (${n.data('type')})`);
        });
        
        // Highlight cluster
        window.cy.elements().addClass('dimmed');
        nodes.removeClass('dimmed').addClass('highlighted');
        nodes.connectedEdges().removeClass('dimmed').addClass('highlighted');
        
        return nodes;
    },
    
    // Show top nodes by PageRank
    topByPageRank: (n = 5) => {
        if (!window.cy) {
            console.error('Cytoscape not initialized yet');
            return;
        }
        
        const nodes = window.cy.nodes().sort((a, b) => {
            return (b.data('pagerank') || 0) - (a.data('pagerank') || 0);
        });
        
        const top = nodes.slice(0, n);
        
        console.log(`=== Top ${n} nodes by PageRank ===`);
        top.forEach((node, i) => {
            console.log(`${i + 1}. ${node.id()}: ${node.data('pagerank').toFixed(4)} - "${node.data('text')}"`);
        });
        
        return top;
    },
    
    // Show demo path
    showDemoPath: () => {
        if (!window.cy) {
            console.error('Cytoscape not initialized yet');
            return;
        }
        
        // Try to get demo path from graph data
        const graphData = window.graphData || {};
        const demoPath = graphData._meta?.demo_path || [];
        
        if (demoPath.length === 0) {
            console.log('No demo path available in graph data');
            return;
        }
        
        console.log(`=== Demo Path (${demoPath.length} nodes) ===`);
        
        const cy = window.cy;
        const pathNodes = [];
        
        demoPath.forEach((nodeId, i) => {
            const node = cy.$(`#${nodeId}`);
            if (node.length > 0) {
                console.log(`${i + 1}. ${nodeId}: "${node.data('text')}"`);
                pathNodes.push(node);
            } else {
                console.warn(`Node ${nodeId} not found in graph`);
            }
        });
        
        // Highlight demo path
        cy.elements().addClass('dimmed');
        pathNodes.forEach(n => {
            n.removeClass('dimmed').addClass('highlighted');
            n.connectedEdges().removeClass('dimmed');
        });
        
        return pathNodes;
    },
    
    // Export current view as JSON
    exportView: () => {
        if (!window.cy) {
            console.error('Cytoscape not initialized yet');
            return;
        }
        
        const cy = window.cy;
        const positions = {};
        
        cy.nodes().forEach(n => {
            positions[n.id()] = n.position();
        });
        
        const viewData = {
            positions: positions,
            zoom: cy.zoom(),
            pan: cy.pan(),
            timestamp: new Date().toISOString()
        };
        
        console.log('View exported:', viewData);
        console.log('Copy this JSON to save the current layout');
        
        return viewData;
    },
    
    // Apply saved view
    applyView: (viewData) => {
        if (!window.cy) {
            console.error('Cytoscape not initialized yet');
            return;
        }
        
        const cy = window.cy;
        
        if (viewData.positions) {
            Object.entries(viewData.positions).forEach(([nodeId, pos]) => {
                const node = cy.$(`#${nodeId}`);
                if (node.length > 0) {
                    node.position(pos);
                }
            });
        }
        
        if (viewData.zoom) {
            cy.zoom(viewData.zoom);
        }
        
        if (viewData.pan) {
            cy.pan(viewData.pan);
        }
        
        console.log('View applied from:', viewData.timestamp || 'unknown time');
    },
    
    // Check edge styles
    checkEdgeStyles: () => {
        if (!window.cy) {
            console.error('Cytoscape not initialized');
            return;
        }
        
        // Get all edge styles from Cytoscape
        const styles = window.cy.style().json();
        const edgeStyles = styles.filter(s => s.selector && s.selector.includes('edge'));
        
        console.log('=== EDGE STYLES IN CYTOSCAPE ===');
        console.log(`Total edge styles: ${edgeStyles.length}`);
        
        // Group by selector type
        const byType = {};
        edgeStyles.forEach(style => {
            const match = style.selector.match(/edge\[type="(\w+)"\]/);
            if (match) {
                const type = match[1];
                if (!byType[type]) byType[type] = [];
                byType[type].push(style);
            }
        });
        
        console.log('\nStyles by edge type:');
        Object.keys(byType).forEach(type => {
            const styles = byType[type];
            console.log(`  ${type}: ${styles.length} styles`);
            if (styles[0] && styles[0].style) {
                console.log(`    - line-color: ${styles[0].style['line-color']}`);
                console.log(`    - width: ${styles[0].style['width']}`);
            }
        });
        
        // Check actual edges
        console.log('\n=== ACTUAL EDGES IN GRAPH ===');
        const edges = window.cy.edges();
        const edgeTypes = {};
        edges.forEach(edge => {
            const type = edge.data('type');
            edgeTypes[type] = (edgeTypes[type] || 0) + 1;
        });
        
        console.log('Edge type distribution:');
        Object.keys(edgeTypes).forEach(type => {
            console.log(`  ${type}: ${edgeTypes[type]} edges`);
        });
        
        // Sample edge check
        console.log('\n=== SAMPLE EDGE CHECK ===');
        const sampleEdge = edges[0];
        if (sampleEdge) {
            console.log('First edge data:', sampleEdge.data());
            console.log('First edge computed style:');
            console.log('  line-color:', sampleEdge.style('line-color'));
            console.log('  width:', sampleEdge.style('width'));
            console.log('  opacity:', sampleEdge.style('opacity'));
        }
        
        // Check specific edge types
        console.log('\n=== CHECKING SPECIFIC EDGE TYPES ===');
        ['PREREQUISITE', 'ELABORATES', 'MENTIONS'].forEach(type => {
            const edge = window.cy.edges(`[type="${type}"]`).first();
            if (edge && edge.length > 0) {
                console.log(`${type} edge:`);
                console.log('  Computed line-color:', edge.style('line-color'));
                console.log('  Expected color:', window.EdgeStyles?.EDGE_STYLES[type]?.lineColor);
            }
        });
    },

    // Help message
    help: () => {
        console.log(`
=== K2-18 Debug Helpers ===

Available functions:
  debugHelpers.stats()                     - Show graph statistics
  debugHelpers.findByText(text)           - Find nodes containing text
  debugHelpers.neighbors(nodeId)          - Show node neighbors
  debugHelpers.highlightPath(from, to)    - Highlight shortest path
  debugHelpers.clearHighlights()          - Clear all highlights
  debugHelpers.showCluster(clusterId)     - Show nodes in cluster
  debugHelpers.topByPageRank(n)          - Show top N nodes by PageRank
  debugHelpers.showDemoPath()            - Show and highlight demo path
  debugHelpers.exportView()              - Export current layout
  debugHelpers.applyView(viewData)       - Apply saved layout
  debugHelpers.checkEdgeStyles()          - Check edge styles and colors
  debugHelpers.help()                    - Show this help

Global objects:
  window.cy                              - Cytoscape instance
  window.graphData                       - Raw graph data (if available)
        `);
    }
};

// Initialize on load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        console.log('Debug helpers loaded. Type debugHelpers.help() for available commands.');
    });
} else {
    console.log('Debug helpers loaded. Type debugHelpers.help() for available commands.');
}
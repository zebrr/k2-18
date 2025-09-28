/**
 * Clusters & Bridges Module for K2-18 Visualization
 * Visualizes knowledge clusters and bridge nodes
 */

const ClustersBridges = {
    // State management
    active: false,
    originalStyles: null,      // Storage for original styles
    clusterColors: {},         // Mapping cluster_id -> color
    hoveredCluster: null,      // ID of currently hovered cluster
    overlayEnabled: false,     // Whether overlay is supported
    boundHandleNodeHover: null,    // Bound hover handler for cleanup
    boundHandleNodeUnhover: null,  // Bound unhover handler for cleanup
    
    // Configuration
    config: {
        bridge_threshold: 0.7,  // Minimum bridge_score to highlight
        overlay_padding: 30,    // Padding for cluster overlay
        overlay_opacity_normal: 0.1,   // Normal overlay opacity
        overlay_opacity_hover: 0.3,    // Hover overlay opacity
        bridge_border_width: 4,         // Border width for bridge nodes
        bridge_border_color: '#ff6b6b', // Red border for bridges
        inter_cluster_dash: [8, 4],     // Dash pattern for inter-cluster edges
        animation_duration: 300         // Animation duration in ms
    },
    
    // Initialize module
    init(cy, graphCore, config) {
        this.cy = cy;
        this.graphCore = graphCore;
        
        // Merge configuration
        if (config && config.colors) {
            this.clusterPalette = config.colors.cluster_palette || [];
        }
        
        console.log('[ClustersBridges] Module initialized');
    },
    
    // Activation
    activate() {
        if (this.active) return;
        
        console.log('[ClustersBridges] Activating...');
        this.active = true;
        
        // Save original styles before applying cluster styles
        this.saveOriginalStyles();
        
        // Apply cluster visualization
        this.applyClusterStyles();
        
        // Setup hover effects
        this.setupHoverEffects();
        
        console.log('[ClustersBridges] Activated successfully');
    },
    
    // Deactivation
    deactivate() {
        if (!this.active) return;
        
        console.log('[ClustersBridges] Deactivating...');
        this.active = false;
        
        // Remove hover handlers
        this.removeHoverEffects();
        
        // Restore original styles
        this.restoreOriginalStyles();
        
        // Clear state
        this.clusterColors = {};
        this.hoveredCluster = null;
        this.originalStyles = null;
        
        console.log('[ClustersBridges] Deactivated successfully');
    },
    
    // Save original styles for all elements
    saveOriginalStyles() {
        console.log('[ClustersBridges] Saving original styles...');
        
        this.originalStyles = {
            nodes: {},
            edges: {}
        };
        
        // Save node styles
        this.cy.nodes().forEach(node => {
            const id = node.id();
            this.originalStyles.nodes[id] = {
                'background-color': node.style('background-color'),
                'background-opacity': node.style('background-opacity'),
                'border-width': node.style('border-width'),
                'border-color': node.style('border-color'),
                'border-opacity': node.style('border-opacity'),
                'opacity': node.style('opacity'),
                'overlay-color': node.style('overlay-color'),
                'overlay-padding': node.style('overlay-padding'),
                'overlay-opacity': node.style('overlay-opacity')
            };
        });
        
        // Save edge styles
        this.cy.edges().forEach(edge => {
            const id = edge.id();
            this.originalStyles.edges[id] = {
                'line-color': edge.style('line-color'),
                'line-style': edge.style('line-style'),
                'line-dash-pattern': edge.style('line-dash-pattern'),
                'width': edge.style('width'),
                'opacity': edge.style('opacity'),
                'line-opacity': edge.style('line-opacity'),
                'target-arrow-color': edge.style('target-arrow-color'),
                'source-arrow-color': edge.style('source-arrow-color')
            };
        });
        
        console.log('[ClustersBridges] Saved styles for', 
                    Object.keys(this.originalStyles.nodes).length, 'nodes and',
                    Object.keys(this.originalStyles.edges).length, 'edges');
    },
    
    // Restore original styles for all elements
    restoreOriginalStyles() {
        if (!this.originalStyles) {
            console.warn('[ClustersBridges] No original styles to restore');
            return;
        }
        
        console.log('[ClustersBridges] Restoring original styles...');
        
        this.cy.batch(() => {
            // Restore node styles
            this.cy.nodes().forEach(node => {
                // Remove cluster mode marker
                node.removeClass('cluster-mode-active');
                
                // Remove overlay styles completely
                node.removeStyle('overlay-color overlay-padding overlay-opacity');
                
                // Remove inline styles to let original class styles take over
                node.removeStyle('background-color background-opacity border-width border-color border-opacity');
            });
            
            // Restore edge styles
            Object.entries(this.originalStyles.edges).forEach(([id, styles]) => {
                const edge = this.cy.getElementById(id);
                if (edge.length > 0) {
                    // Restore all styles that we modified (including opacity!)
                    const stylesToRestore = ['line-style', 'line-dash-pattern', 'width', 'opacity'];
                    
                    stylesToRestore.forEach(key => {
                        if (styles[key] !== undefined) {
                            edge.style(key, styles[key]);
                        }
                    });
                }
            });
        });
        
        console.log('[ClustersBridges] Original styles restored');
    },
    
    // Apply cluster visualization styles
    applyClusterStyles() {
        console.log('[ClustersBridges] Applying cluster styles...');
        
        // Step 1: Assign colors to clusters
        this.assignClusterColors();
        
        // Step 2: Style nodes by cluster
        this.styleNodesByCluster();
        
        // Step 3: Highlight bridge nodes
        this.highlightBridgeNodes();
        
        // Step 4: Style inter-cluster edges
        this.styleInterClusterEdges();
        
        // Step 5: Apply cluster overlays (if supported)
        this.applyClusterOverlays();
        
        console.log('[ClustersBridges] Cluster styles applied');
    },
    
    // Assign colors to clusters
    assignClusterColors() {
        // Get unique cluster IDs from nodes
        const clusterIds = new Set();
        this.cy.nodes().forEach(node => {
            const clusterId = node.data('cluster_id');
            if (clusterId !== undefined && clusterId !== null) {
                clusterIds.add(clusterId);
            }
        });
        
        const sortedClusterIds = Array.from(clusterIds).sort((a, b) => a - b);
        console.log('[ClustersBridges] Found', sortedClusterIds.length, 'clusters:', sortedClusterIds);
        
        // Assign colors from palette (36 colors for better distinction)
        const palette = this.clusterPalette || [
            // Основные яркие цвета (12)
            "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
            "#1abc9c", "#e67e22", "#16a085", "#8e44ad", "#27ae60",
            "#2980b9", "#c0392b",

            // Пастельные оттенки (12)
            "#f1948a", "#85c1e9", "#82e0aa", "#f8c471", "#bb8fce",
            "#76d7c4", "#f0b27a", "#73c6b6", "#af7ac5", "#7dcea0",
            "#7fb3d5", "#ec7063",

            // Темные насыщенные (12)
            "#922b21", "#1a5490", "#196f3d", "#9a7d0a", "#6c3483",
            "#0e6251", "#935116", "#0b5345", "#5b2c6f", "#186a3b",
            "#1f618d", "#7b241c"
        ];
        
        this.clusterColors = {};
        sortedClusterIds.forEach((clusterId, index) => {
            this.clusterColors[clusterId] = palette[index % palette.length];
        });
        
        console.log('[ClustersBridges] Assigned colors to clusters:', this.clusterColors);
    },
    
    // Style nodes by their cluster
    styleNodesByCluster() {
        this.cy.batch(() => {
            this.cy.nodes().forEach(node => {
                const clusterId = node.data('cluster_id');
                const color = this.clusterColors[clusterId];
                
                if (color) {
                    // Don't remove type classes, just override with inline styles
                    // Inline styles should have higher priority than class styles
                    node.style({
                        'background-color': color,
                        'background-opacity': 1  // Full opacity for bright colors
                    });
                    
                    // Add a marker class to indicate cluster mode is active
                    node.addClass('cluster-mode-active');
                }
            });
        });
        
        console.log('[ClustersBridges] Nodes styled by cluster');
    },
    
    // Highlight nodes with high bridge scores
    highlightBridgeNodes() {
        const threshold = this.config.bridge_threshold;
        let bridgeCount = 0;
        
        this.cy.batch(() => {
            this.cy.nodes().forEach(node => {
                const bridgeScore = node.data('bridge_score');
                
                if (bridgeScore !== undefined && bridgeScore > threshold) {
                    node.style({
                        'border-width': this.config.bridge_border_width,
                        'border-color': this.config.bridge_border_color,
                        'border-opacity': 1
                    });
                    bridgeCount++;
                }
            });
        });
        
        console.log('[ClustersBridges] Highlighted', bridgeCount, 'bridge nodes');
    },
    
    // Style inter-cluster edges
    styleInterClusterEdges() {
        let interClusterCount = 0;
        let intraClusterCount = 0;
        
        this.cy.batch(() => {
            // First, dim ALL edges
            this.cy.edges().forEach(edge => {
                const sourceNode = edge.source();
                const targetNode = edge.target();
                const sourceCluster = sourceNode.data('cluster_id');
                const targetCluster = targetNode.data('cluster_id');
                
                // Check if edge connects different clusters
                const isInterCluster = sourceCluster !== undefined && 
                                      targetCluster !== undefined && 
                                      sourceCluster !== targetCluster;
                
                if (isInterCluster) {
                    // Inter-cluster edges: slightly more visible and dashed
                    const currentWidth = parseFloat(edge.style('width')) || 1;
                    edge.style({
                        'line-style': 'dashed',
                        'line-dash-pattern': this.config.inter_cluster_dash,
                        'width': currentWidth * 1.2,  // Slightly thicker
                        'opacity': 0.5  // More visible than intra-cluster
                    });
                    interClusterCount++;
                } else {
                    // Intra-cluster edges: very dim
                    edge.style({
                        'opacity': 0.3  // Very dim
                    });
                    intraClusterCount++;
                }
            });
        });
        
        console.log('[ClustersBridges] Styled', interClusterCount, 'inter-cluster edges and', 
                    intraClusterCount, 'intra-cluster edges');
    },
    
    // Apply overlay effects for clusters (visual grouping)
    applyClusterOverlays() {
        // Check if we have clusters to visualize
        if (Object.keys(this.clusterColors).length === 0) {
            console.log('[ClustersBridges] No clusters to create overlays for');
            return;
        }
        
        // Note: Cytoscape.js doesn't have true convex hull overlays,
        // but we can use node overlay styles for visual grouping
        console.log('[ClustersBridges] Applying cluster overlays...');
        
        this.cy.batch(() => {
            Object.keys(this.clusterColors).forEach(clusterId => {
                const clusterNodes = this.cy.nodes(`[cluster_id = ${clusterId}]`);
                
                if (clusterNodes.length > 2) {
                    // Apply subtle overlay effect to visually group nodes
                    clusterNodes.style({
                        'overlay-color': this.clusterColors[clusterId],
                        'overlay-padding': this.config.overlay_padding,
                        'overlay-opacity': this.config.overlay_opacity_normal
                    });
                }
            });
        });
        
        this.overlayEnabled = true;
        console.log('[ClustersBridges] Cluster overlays applied');
    },
    
    // Setup hover effects
    setupHoverEffects() {
        // Remove any existing handlers first
        this.removeHoverEffects();
        
        // Create bound handlers and store them for later removal
        this.boundHandleNodeHover = this.handleNodeHover.bind(this);
        this.boundHandleNodeUnhover = this.handleNodeUnhover.bind(this);
        
        // Node hover handlers with namespace to avoid conflicts
        this.cy.on('mouseover', 'node', this.boundHandleNodeHover);
        this.cy.on('mouseout', 'node', this.boundHandleNodeUnhover);
        
        console.log('[ClustersBridges] Hover effects setup');
    },
    
    // Remove hover effects
    removeHoverEffects() {
        // Only remove our specific handlers, not all mouseover/mouseout handlers
        if (this.boundHandleNodeHover) {
            this.cy.off('mouseover', 'node', this.boundHandleNodeHover);
        }
        if (this.boundHandleNodeUnhover) {
            this.cy.off('mouseout', 'node', this.boundHandleNodeUnhover);
        }
        
        // Clean up references
        this.boundHandleNodeHover = null;
        this.boundHandleNodeUnhover = null;
    },
    
    // Handle node hover
    handleNodeHover(evt) {
        if (!this.active) return;
        
        const node = evt.target;
        const clusterId = node.data('cluster_id');
        
        if (clusterId !== undefined && clusterId !== this.hoveredCluster) {
            this.highlightCluster(clusterId);
        }
    },
    
    // Handle node unhover
    handleNodeUnhover(evt) {
        if (!this.active) return;
        
        const node = evt.target;
        const clusterId = node.data('cluster_id');
        
        if (clusterId !== undefined && clusterId === this.hoveredCluster) {
            this.unhighlightCluster(clusterId);
        }
    },
    
    // Highlight a cluster
    highlightCluster(clusterId) {
        if (this.hoveredCluster === clusterId) return;
        
        this.hoveredCluster = clusterId;
        const clusterNodes = this.cy.nodes(`[cluster_id = ${clusterId}]`);
        
        if (clusterNodes.length > 0 && this.overlayEnabled) {
            // Animate overlay opacity increase
            clusterNodes.animate({
                style: {
                    'overlay-opacity': this.config.overlay_opacity_hover
                }
            }, {
                duration: this.config.animation_duration
            });
        }
        
        // Node opacity already at 1, no need to change
    },
    
    // Unhighlight a cluster
    unhighlightCluster(clusterId) {
        if (this.hoveredCluster !== clusterId) return;
        
        this.hoveredCluster = null;
        const clusterNodes = this.cy.nodes(`[cluster_id = ${clusterId}]`);
        
        if (clusterNodes.length > 0 && this.overlayEnabled) {
            // Animate overlay opacity decrease
            clusterNodes.animate({
                style: {
                    'overlay-opacity': this.config.overlay_opacity_normal
                }
            }, {
                duration: this.config.animation_duration
            });
        }
        
        // Node opacity stays at 1, no need to reset
    },
    
    // Get cluster statistics for debugging
    getClusterStats() {
        const stats = {
            active: this.active,
            totalClusters: Object.keys(this.clusterColors).length,
            clusterSizes: {},
            bridgeNodes: 0,
            interClusterEdges: 0
        };
        
        // Count nodes per cluster
        Object.keys(this.clusterColors).forEach(clusterId => {
            const nodes = this.cy.nodes(`[cluster_id = ${clusterId}]`);
            stats.clusterSizes[clusterId] = nodes.length;
        });
        
        // Count bridge nodes
        const threshold = this.config.bridge_threshold;
        this.cy.nodes().forEach(node => {
            if (node.data('bridge_score') > threshold) {
                stats.bridgeNodes++;
            }
        });
        
        // Count inter-cluster edges
        this.cy.edges().forEach(edge => {
            const sourceCluster = edge.source().data('cluster_id');
            const targetCluster = edge.target().data('cluster_id');
            if (sourceCluster !== undefined && targetCluster !== undefined && 
                sourceCluster !== targetCluster) {
                stats.interClusterEdges++;
            }
        });
        
        return stats;
    }
};

// Auto-initialize when graph is ready
document.addEventListener('k2-graph-ready', (e) => {
    const { cy, graphCore } = e.detail;
    
    // Get colors config from window
    const fullConfig = {
        colors: window.colorsConfig || {}
    };
    
    ClustersBridges.init(cy, graphCore, fullConfig);
    console.log('[ClustersBridges] Auto-initialized via k2-graph-ready event');
});

// Listen for mode changes
document.addEventListener('mode-changed', (e) => {
    if (e.detail.mode === 'clusters') {
        ClustersBridges.activate();
    } else if (ClustersBridges.active) {
        ClustersBridges.deactivate();
    }
});

// Export to window for debugging
window.ClustersBridges = ClustersBridges;
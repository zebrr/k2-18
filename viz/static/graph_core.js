/**
 * K2-18 Knowledge Graph - Core Visualization Module
 * Handles Cytoscape initialization and base styling
 */

class GraphCore {
    constructor(container, config = {}) {
        this.container = container;
        this.config = this.mergeConfig(config);
        this.cy = null;
        this.graphData = null;
        this.conceptData = null;
    }

    mergeConfig(userConfig) {
        const defaults = {
            // Node visual encoding - will be overridden by config
            nodeShapes: {
                'Chunk': 'hexagon',
                'Concept': 'star', 
                'Assessment': 'roundrectangle'
            },
            nodeColors: {
                'Chunk': '#3498db',
                'Concept': '#2ecc71',
                'Assessment': '#f39c12'
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
            showLabelsOnHover: true,
            hoverDelay: 500,
            // Animation on load
            animateOnLoad: true
        };
        return { ...defaults, ...userConfig };
    }

    async initialize(graphData, conceptData) {
        this.graphData = graphData;
        this.conceptData = conceptData;
        
        // Prepare elements for Cytoscape
        const elements = this.prepareElements();
        
        // Generate styles including edge styles if available
        const styles = this.generateStyles();
        
        // Debug: check styles array before passing to Cytoscape
        console.log('Total styles to apply:', styles.length);
        console.log('Styles array includes edge styles?', 
            styles.some(s => s.selector && s.selector.includes('edge')));
        
        // Initialize Cytoscape
        this.cy = cytoscape({
            container: this.container,
            elements: elements,
            style: styles,
            layout: this.getLayoutConfig(),
            wheelSensitivity: 0.2,
            minZoom: 0.1,
            maxZoom: 5
        });
        
        // Предварительно размещаем узлы в горизонтальной полосе для широкого layout
        const width = this.container.clientWidth || 1920;
        const height = this.container.clientHeight || 800;
        
        this.cy.nodes().forEach((node, i) => {
            node.position({
                x: (Math.random() - 0.5) * width * 3.0,  // Широко по X
                y: (Math.random() - 0.5) * height * 0.3  // Узко по Y
            });
        });
        
        // Debug: check what styles Cytoscape actually has
        console.log('Cytoscape initialized, checking applied styles...');
        const cyStyles = this.cy.style().json();
        console.log('Total styles in Cytoscape:', cyStyles.length);
        const edgeStylesInCy = cyStyles.filter(s => s.selector && s.selector.includes('edge'));
        console.log('Edge styles in Cytoscape:', edgeStylesInCy.length);
        console.log('Sample Cytoscape edge styles:', edgeStylesInCy.slice(0, 3));
        
        // Setup hover labels if enabled
        if (this.config.showLabelsOnHover) {
            this.setupHoverLabels();
        }
        
        // Initialize animation controller
        if (window.AnimationController) {
            this.animationController = new window.AnimationController(this.cy, {
                levelDelay: 200,
                nodeAnimDuration: 500,
                edgeAnimDuration: 500,
                physicsDuration: this.config.physicsDuration,
                animateOnLoad: this.config.animateOnLoad
            });
            
            // Run animation if enabled
            if (this.config.animateOnLoad) {
                await this.animationController.animateGraph();
            }
        } else if (this.config.animateOnLoad) {
            // Fallback to simple animation if controller not available
            await this.animateAppearance();
        }

        return this.cy;
    }

    prepareElements() {
        const elements = {
            nodes: [],
            edges: []
        };

        // Process nodes
        this.graphData.nodes.forEach(node => {
            elements.nodes.push({
                data: {
                    id: node.id,
                    label: this.truncateLabel(node.text || node.id),
                    fullText: node.text,
                    type: node.type,
                    difficulty: node.difficulty || 3,
                    pagerank: node.pagerank || 0.01,
                    cluster_id: node.cluster_id,
                    bridge_score: node.bridge_score || 0,
                    prerequisite_depth: node.prerequisite_depth || 0,
                    ...node // Include all other metrics
                },
                classes: node.type.toLowerCase()
            });
        });

        // Process edges
        this.graphData.edges.forEach(edge => {
            elements.edges.push({
                data: {
                    id: `${edge.source}-${edge.target}`,
                    source: edge.source,
                    target: edge.target,
                    type: edge.type,
                    weight: edge.weight || 0.5,
                    is_inter_cluster_edge: edge.is_inter_cluster_edge || false,
                    ...edge
                }
            });
        });

        return elements;
    }

    generateStyles() {
        console.log('generateStyles called');
        console.log('EdgeStyles available?', window.EdgeStyles);
        
        const styles = [
            // Base node styles
            {
                selector: 'node',
                style: {
                    'opacity': 0,  // first visible frame should be empty
                    // No labels - handled by ui_controls.js tooltip
                    'label': '',
                    'background-opacity': (ele) => this.calculateOpacity(ele),
                    'width': (ele) => this.calculateNodeSize(ele),
                    'height': (ele) => this.calculateNodeSize(ele),
                    'border-width': 2,
                    'border-color': '#ffffff',
                    'border-opacity': 0.8,
                    'transition-property': 'width, height, background-opacity, background-color',
                    'transition-duration': '250ms',
                    'transition-timing-function': 'ease-out'
                }
            },
            // Type-specific styles
            {
                selector: 'node.chunk',
                style: {
                    'shape': this.config.nodeShapes['Chunk'] || 'hexagon',
                    'background-color': this.config.nodeColors['Chunk']
                }
            },
            {
                selector: 'node.concept',
                style: {
                    'shape': this.config.nodeShapes['Concept'] || 'star',
                    'background-color': this.config.nodeColors['Concept']
                }
            },
            {
                selector: 'node.assessment',
                style: {
                    'shape': this.config.nodeShapes['Assessment'] || 'roundrectangle',
                    'background-color': this.config.nodeColors['Assessment']
                }
            },
            {
                selector: 'node:selected',
                style: {
                    'border-width': 4,
                    'border-color': '#f39c12'
                }
            },
            // UI Controls hover effects
            {
                selector: 'node.hover-highlight',
                style: {
                    'background-color': '#ff0000',
                    'background-opacity': 1,
                    'border-width': 3,
                    'border-color': '#ffffff',
                    'z-index': 9999
                }
            },
            {
                selector: 'node.pulse',
                style: {
                    'background-color': '#ff0000',
                    'background-opacity': 1,
                    'border-width': 3,
                    'border-color': '#ffffff',
                    'z-index': 9999
                }
            },
            {
                selector: '.hidden',
                style: {
                    'display': 'none'
                }
            },
            {
                selector: '.hidden-edge',
                style: {
                    'display': 'none'
                }
            }
        ];
        
        // Add edge styles if module is available
        if (window.EdgeStyles && window.EdgeStyles.generateEdgeStyles) {
            const edgeStyles = window.EdgeStyles.generateEdgeStyles({
                interClusterMultiplier: 1.5
            });
            styles.push(...edgeStyles);
            console.log('Edge styles loaded:', edgeStyles.length, 'styles');
            console.log('EdgeStyles module available:', window.EdgeStyles);
            console.log('Generated edge styles:', edgeStyles);
            
            // Debug: log unique edge types in data
            const edgeTypes = new Set(this.graphData.edges.map(e => e.type));
            console.log('Edge types in data:', Array.from(edgeTypes));
            
            // Debug: log first few edge styles for verification
            console.log('Sample edge styles:', edgeStyles.slice(0, 5).map(s => ({
                selector: s.selector,
                color: s.style['line-color']
            })));
        } else {
            console.warn('EdgeStyles module not available - edges will use default styling');
            // Fallback base edge style if EdgeStyles not available
            styles.push({
                selector: 'edge',
                style: {
                    'width': 2,
                    'line-color': '#95a5a6',
                    'target-arrow-color': '#95a5a6',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier',
                    'opacity': 0
                }
            });
        }
        
        // Add hover styles AFTER edge styles to ensure higher priority
        styles.push({
            selector: 'edge.hover-connected',
            style: {
                'line-color': '#ff0000',
                'target-arrow-color': '#ff0000',
                'source-arrow-color': '#ff0000',
                'opacity': 1,
                'width': 6,
                'z-index': 999
            }
        });
        
        return styles;
    }

    calculateNodeSize(ele) {
        const pagerank = ele.data('pagerank') || 0.01;
        const minSize = this.config.minNodeSize;
        const maxSize = this.config.maxNodeSize;
        
        // Logarithmic scale for better distribution
        const scaledValue = Math.log(pagerank * 1000 + 1) / Math.log(1000);
        return minSize + (maxSize - minSize) * Math.min(1, scaledValue);
    }

    calculateOpacity(ele) {
        const difficulty = ele.data('difficulty') || 3;
        const minOpacity = this.config.minOpacity;
        const maxOpacity = this.config.maxOpacity;
        
        // Map difficulty 1-5 to opacity range
        const normalized = (difficulty - 1) / 4;
        return minOpacity + (maxOpacity - minOpacity) * normalized;
    }

    truncateLabel(text, maxLength = 30) {
        if (!text) return '';
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength - 3) + '...';
    }
    
    /**
     * Setup hover label display with delay
     * DISABLED - now handled by ui_controls.js with better tooltip
     */
    setupHoverLabels() {
        // Functionality moved to ui_controls.js
        // Keeping empty method for compatibility
    }

    getLayoutConfig() {
        const layoutName = this.config.initialLayout;
        
        if (layoutName === 'cose-bilkent') {
            return {
                name: 'cose-bilkent',
                animate: 'end',
                animationDuration: this.config.layoutAnimationDuration,
                randomize: false,               // Используем наши начальные позиции из широкой полосы
                nodeRepulsion: 15000,
                idealEdgeLength: 300,
                edgeElasticity: 0.1,
                gravity: 0.01,
                gravityRange: 2.0,              // Действие гравитации на отдаленные узлы
                gravityCompound: 4.0,           // Принягие вание отдаленных компонент к центру
                gravityRangeCompound: 1.5,
                tile: true,
                tilingPaddingHorizontal: 300,   // Большой отступ по горизонтали
                tilingPaddingVertical: 10,      // Маленький по вертикали
                numIter: 4000,
                nestingFactor: 0.05
            };
        }
        
        // Fallback to grid layout
        return {
            name: 'grid',
            animate: true,
            animationDuration: this.config.animationDuration
        };
    }

    async animateAppearance() {
        // Hide all nodes initially
        this.cy.nodes().style('opacity', 0);
        
        // Group nodes by prerequisite_depth
        const nodesByDepth = {};
        this.cy.nodes().forEach(node => {
            const depth = node.data('prerequisite_depth') || 0;
            if (!nodesByDepth[depth]) {
                nodesByDepth[depth] = [];
            }
            nodesByDepth[depth].push(node);
        });
        
        // Animate appearance by depth level
        const depths = Object.keys(nodesByDepth).sort((a, b) => a - b);
        
        for (let depth of depths) {
            const nodes = nodesByDepth[depth];
            nodes.forEach(node => {
                node.animate({
                    style: { opacity: 1 },
                    duration: 500,
                    easing: 'ease-out'
                });
            });
            
            // Wait before showing next level
            await new Promise(resolve => setTimeout(resolve, 200));
        }
        
        // Show all edges
        this.cy.edges().animate({
            style: { opacity: 0.6 },
            duration: 500
        });
    }

    // Public API methods
    getStats() {
        return {
            nodes: this.cy.nodes().length,
            edges: this.cy.edges().length,
            nodeTypes: this.getNodeTypeCounts(),
            edgeTypes: this.getEdgeTypeCounts()
        };
    }

    getNodeTypeCounts() {
        const counts = {};
        this.cy.nodes().forEach(node => {
            const type = node.data('type');
            counts[type] = (counts[type] || 0) + 1;
        });
        return counts;
    }

    getEdgeTypeCounts() {
        const counts = {};
        this.cy.edges().forEach(edge => {
            const type = edge.data('type');
            counts[type] = (counts[type] || 0) + 1;
        });
        return counts;
    }
}

// Export for use in HTML
window.GraphCore = GraphCore;
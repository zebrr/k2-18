/**
 * Path Finder Module for K2-18 Visualization
 * Implements learning path discovery between nodes
 */

const PathFinder = {
    // State management
    active: false,
    isSelecting: false,
    firstNode: null,
    secondNode: null,
    fastPath: null,
    simplePath: null,
    metricsPanel: null,
    pulseAnimation: null,
    
    // Configuration from config.toml
    config: {
        alpha_step: 0,
        beta_difficulty: 1,
        sigma_confidence: 0.5,
        default_difficulty: 3,
        edge_type_ladder: [
            ["PREREQUISITE", "ELABORATES", "EXAMPLE_OF", "PARALLEL", "REVISION_OF", "TESTS"],
            ["REFER_BACK", "HINT_FORWARD"],
            ["MENTIONS"]
        ]
    },
    
    // Initialize module
    init(cy, graphCore, config) {
        this.cy = cy;
        this.graphCore = graphCore;
        
        // Merge configuration
        if (config && config.path_mode) {
            Object.assign(this.config, config.path_mode);
        }
        
        // Create metrics panel
        this.createMetricsPanel();
        
        // Setup event handlers
        this.setupEventHandlers();
        
        console.log('[PathFinder] Module initialized');
    },
    
    // Activation/deactivation
    activate() {
        if (this.active) return;
        
        this.active = true;
        this.isSelecting = true;
        this.reset();
        
        // Disable node dragging in path mode
        this.cy.nodes().ungrabify();
        
        // Change cursor to indicate selection mode
        document.getElementById('cy').style.cursor = 'crosshair';
        
        console.log('[PathFinder] Activated - click two nodes to find paths');
        this.showToast('📍 Выберите два узла для поиска пути', 2000);
    },
    
    deactivate() {
        if (!this.active) return;
        
        this.active = false;
        this.isSelecting = false;
        
        // Always restore all elements when deactivating
        this.restoreAllElements();
        
        // Then reset state (without calling restoreAllElements again)
        this.reset();
        this.hideMetricsPanel();
        
        // Re-enable node dragging
        this.cy.nodes().grabify();
        
        // Reset cursor
        document.getElementById('cy').style.cursor = '';
        
        console.log('[PathFinder] Deactivated');
    },
    
    // Reset state
    reset() {
        // Stop pulse animation
        this.stopPulseAnimation();
        
        // Clear selections
        if (this.firstNode) {
            this.firstNode.removeClass('path-selected');
        }
        if (this.secondNode) {
            this.secondNode.removeClass('path-selected');
        }
        
        // Only restore elements if we had a path (not on initial activation)
        if (this.shortestPath) {
            this.restoreAllElements();
            
            // Clear path
            this.shortestPath.edges().removeClass('path-shortest path-weak-segment');
            // Reset edge styles
            this.shortestPath.edges().forEach(edge => {
                edge.removeStyle();
            });
        }
        
        // Reset state
        this.firstNode = null;
        this.secondNode = null;
        this.shortestPath = null;
        this.isSelecting = true;
        
        // Hide metrics panel
        this.hideMetricsPanel();
    },
    
    // Event handlers
    setupEventHandlers() {
        // Node tap handler (tap = click without drag)
        this.cy.on('tap', 'node', (evt) => {
            if (!this.active || !this.isSelecting) return;
            
            evt.stopPropagation();
            evt.preventDefault();
            this.handleNodeClick(evt.target);
        });
        
        // Background tap - cancel selection
        this.cy.on('tap', (evt) => {
            if (!this.active || evt.target !== this.cy) return;
            
            if (this.firstNode && !this.secondNode) {
                this.reset();
                this.showToast('❌ Выбор отменён', 1500);
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (!this.active) return;
            
            if (e.key === 'Escape') {
                this.reset();
                this.showToast('🔄 Режим сброшен', 1500);
            }
        });
    },
    
    // Handle node click
    handleNodeClick(node) {
        if (!this.firstNode) {
            // First node selection
            this.firstNode = node;
            this.firstNode.addClass('path-selected');
            
            // Add animated pulse using interval
            this.startPulseAnimation(this.firstNode);
            
            console.log('[PathFinder] First node selected:', node.id());
            this.showToast('📍 Выберите второй узел', 1500);
            
        } else if (node.id() === this.firstNode.id()) {
            // Clicking same node - cancel
            this.reset();
            this.showToast('❌ Выбор отменён', 1500);
            
        } else {
            // Second node selection - find paths
            this.secondNode = node;
            this.secondNode.addClass('path-selected');
            this.isSelecting = false;
            
            // Start pulsing animation for second node too
            this.startPulseAnimation(this.secondNode);
            
            console.log('[PathFinder] Second node selected:', node.id());
            console.log('[PathFinder] Finding paths from', this.firstNode.id(), 'to', node.id());
            
            // Show loading toast
            this.showToast('🔍 Поиск пути...', 1000);
            
            // Find shortest path with a small delay for visual feedback
            setTimeout(() => this.findPath(), 100);
        }
    },
    
    // Main path finding logic - simplified to single shortest path
    findPath() {
        const startId = this.firstNode.id();
        const endId = this.secondNode.id();
        
        console.log('[PathFinder] ========================================');
        console.log('[PathFinder] Starting shortest path search');
        console.log('[PathFinder] From:', startId, 'To:', endId);
        console.log('[PathFinder] Total edges in graph:', this.cy.edges().length);
        
        // Find shortest path (minimum steps) using Dijkstra
        const result = this.findShortestPath(startId, endId);
        
        console.log('[PathFinder] Search complete.');
        console.log('[PathFinder] Path found:', result.found);
        if (result.found) {
            console.log('[PathFinder] Path length:', result.distance, 'steps');
            console.log('[PathFinder] Path difficulty:', result.totalDifficulty);
        }
        console.log('[PathFinder] ========================================');
        
        // Handle results
        if (!result.found) {
            this.handleNoPath();
        } else {
            this.displayPath(result);
            this.showMetrics(result);
        }
    },
    
    // Find shortest path using standard Dijkstra (minimum steps)
    findShortestPath(startId, endId) {
        const startNode = this.cy.getElementById(startId);
        const endNode = this.cy.getElementById(endId);
        
        // Use Dijkstra with uniform weights (each edge = 1 step)
        const dijkstra = this.cy.elements().dijkstra({
            root: startNode,
            weight: () => 1,  // Each edge counts as 1 step
            directed: true
        });
        
        const distance = dijkstra.distanceTo(endNode);
        
        if (distance === Infinity) {
            return { found: false };
        }
        
        // Get the path
        const path = dijkstra.pathTo(endNode);
        
        // Calculate total difficulty
        let totalDifficulty = 0;
        path.nodes().forEach((node, i) => {
            if (i > 0) {  // Skip start node
                totalDifficulty += node.data('difficulty') || this.config.default_difficulty || 3;
            }
        });
        
        return {
            found: true,
            path: path,
            distance: distance,
            totalDifficulty: totalDifficulty
        };
    },
    
    // DEPRECATED - keeping for reference but not using
    findFastPath_OLD(startId, endId) {
        const ladder = this.config.edge_type_ladder;
        
        // Try each level of the ladder progressively
        for (let levelIndex = 0; levelIndex < ladder.length; levelIndex++) {
            // Accumulate allowed types up to current level
            const allowedTypes = ladder.slice(0, levelIndex + 1).flat();
            
            console.log('[PathFinder] Trying fast path with edge types:', allowedTypes);
            
            // Filter edges by allowed types
            const filteredEdges = this.cy.edges().filter(edge => {
                const edgeType = edge.data('type');
                return allowedTypes.includes(edgeType);
            });
            
            // Create subgraph with only allowed edges
            const subgraph = this.cy.collection().union(this.cy.nodes()).union(filteredEdges);
            
            // Use Dijkstra with uniform weights (all edges = 1)
            // Note: we pass the node directly, not a selector
            const startNode = this.cy.getElementById(startId);
            const dijkstra = subgraph.dijkstra({
                root: startNode,
                weight: () => 1,  // Uniform weight for step counting
                directed: true
            });
            
            const endNode = this.cy.getElementById(endId);
            const distance = dijkstra.distanceTo(endNode);
            
            if (distance !== Infinity) {
                // Path found at this level
                const path = dijkstra.pathTo(endNode);
                console.log('[PathFinder] Fast path found at level', levelIndex, 'distance:', distance);
                
                return {
                    found: true,
                    path: path,
                    distance: distance,
                    level: levelIndex,
                    allowedTypes: allowedTypes
                };
            }
        }
        
        console.log('[PathFinder] No fast path found through any level');
        return { found: false };
    },
    
    // Easiest path - minimum total difficulty (weighted by node difficulty)
    findSimplePath(startId, endId) {
        const alpha = this.config.alpha_step;
        const beta = this.config.beta_difficulty;
        const sigma = this.config.sigma_confidence;
        const defaultDifficulty = this.config.default_difficulty;
        
        // Weight function: w(u→v) = α + β·difficulty(v) + σ·(1−edge.weight)
        const weightFunction = (edge) => {
            const targetNode = edge.target();
            const difficulty = targetNode.data('difficulty') || defaultDifficulty;
            const edgeWeight = edge.data('weight') || 1;
            
            const weight = alpha + beta * difficulty + sigma * (1 - edgeWeight);
            return weight;
        };
        
        // Use all edges for simple path
        const startNode = this.cy.getElementById(startId);
        const endNode = this.cy.getElementById(endId);
        
        const dijkstra = this.cy.elements().dijkstra({
            root: startNode,
            weight: weightFunction,
            directed: true
        });
        
        const distance = dijkstra.distanceTo(endNode);
        
        if (distance !== Infinity) {
            const path = dijkstra.pathTo(endNode);
            console.log('[PathFinder] Simple path found, total weight:', distance);
            
            // Calculate total difficulty
            let totalDifficulty = 0;
            path.nodes().forEach(node => {
                if (node.id() !== startId) {  // Don't count start node
                    totalDifficulty += node.data('difficulty') || defaultDifficulty;
                }
            });
            
            return {
                found: true,
                path: path,
                distance: distance,
                totalDifficulty: totalDifficulty,
                steps: path.edges().length
            };
        }
        
        console.log('[PathFinder] No simple path found');
        return { found: false };
    },
    
    // Display the shortest path
    displayPath(result) {
        this.shortestPath = result.path;
        
        console.log('[PathFinder] Shortest path found with', this.shortestPath.edges().length, 'edges');
        console.log('[PathFinder] Path edges:', this.shortestPath.edges().map(e => e.id()));
        
        // Apply styles in batch for better performance
        this.cy.batch(() => {
            // First, dim all other edges and nodes
            this.dimOtherElements();
            
            this.shortestPath.edges().forEach(edge => {
                // Add class
                edge.addClass('path-shortest');
                
                // Apply bright green color for the path
                edge.style({
                    'line-color': '#00ff00',  // Bright green
                    'width': 6,  // Thick line
                    'curve-style': 'straight',  // Straight lines for clarity
                    'z-index': 1000,  // On top
                    'z-compound-depth': 'top',
                    'opacity': 1,
                    'target-arrow-color': '#00ff00',
                    'source-arrow-color': '#00ff00',
                    'target-arrow-shape': 'triangle',
                    'source-arrow-shape': 'none',
                    'arrow-scale': 2,
                    'overlay-color': '#00ff00',
                    'overlay-padding': 4,
                    'overlay-opacity': 0.3
                });
                
                // Mark weak segments with dashed line
                const weakTypes = ['REFER_BACK', 'HINT_FORWARD', 'MENTIONS'];
                if (weakTypes.includes(edge.data('type'))) {
                    edge.addClass('path-weak-segment');
                    edge.style({
                        'line-style': 'dashed',
                        'line-dash-pattern': [6, 3]
                    });
                }
            });
        });
        
        console.log('[PathFinder] Path displayed with styles');
    },
    
    // DEPRECATED - old display methods
    displayFastPath_OLD(result) {
        this.fastPath = result.path;
        
        console.log('[PathFinder] Fast path found with', this.fastPath.edges().length, 'edges');
        console.log('[PathFinder] Path edges:', this.fastPath.edges().map(e => e.id()));
        
        // Apply styles in batch for better performance
        this.cy.batch(() => {
            // First, dim all other edges
            this.dimOtherEdges();
            
            this.fastPath.edges().forEach(edge => {
                // Add class first
                edge.addClass('path-fast');
                
                // Apply inline styles - bright blue for fast path
                edge.style({
                    'line-color': '#ff6b00',  // Bright orange for better visibility
                    'width': 6,  // Thicker line
                    'curve-style': 'bezier',
                    'control-point-distances': 30,  // More curve
                    'z-index': 1000,  // Much higher z-index
                    'z-compound-depth': 'top',  // On top
                    'opacity': 1,  // Full opacity
                    'target-arrow-color': '#ff6b00',
                    'source-arrow-color': '#ff6b00',
                    'target-arrow-shape': 'triangle',
                    'source-arrow-shape': 'none',
                    'arrow-scale': 2,  // Bigger arrows
                    'overlay-color': '#ff6b00',
                    'overlay-padding': 4,
                    'overlay-opacity': 0.3  // Glow effect
                });
                
                // Mark weak segments
                const weakTypes = ['REFER_BACK', 'HINT_FORWARD', 'MENTIONS'];
                if (weakTypes.includes(edge.data('type'))) {
                    edge.addClass('path-weak-segment');
                    edge.style({
                        'line-style': 'dashed',
                        'line-dash-pattern': [4, 4]
                    });
                }
            });
        });
        
        console.log('[PathFinder] Fast path displayed with styles');
    },
    
    // Display simple path
    displaySimplePath(result) {
        this.simplePath = result.path;
        
        console.log('[PathFinder] Simple path found with', this.simplePath.edges().length, 'edges');
        console.log('[PathFinder] Path edges:', this.simplePath.edges().map(e => e.id()));
        
        // Apply styles in batch for better performance
        this.cy.batch(() => {
            // Dim other edges if not already done
            if (!this.fastPath) {
                this.dimOtherEdges();
            }
            
            this.simplePath.edges().forEach(edge => {
                // Add class first
                edge.addClass('path-simple');
                
                // Apply inline styles - bright green for simple path
                edge.style({
                    'line-color': '#00ff88',  // Bright green for better visibility
                    'width': 5,  // Thicker line
                    'curve-style': 'bezier',
                    'control-point-distances': -30,  // More curve in opposite direction
                    'z-index': 999,  // High z-index
                    'z-compound-depth': 'top',  // On top
                    'opacity': 1,  // Full opacity
                    'target-arrow-color': '#00ff88',
                    'source-arrow-color': '#00ff88',
                    'target-arrow-shape': 'triangle',
                    'source-arrow-shape': 'none',
                    'arrow-scale': 2,  // Bigger arrows
                    'overlay-color': '#00ff88',
                    'overlay-padding': 4,
                    'overlay-opacity': 0.3  // Glow effect
                });
                
                // Mark weak segments
                const weakTypes = ['REFER_BACK', 'HINT_FORWARD', 'MENTIONS'];
                if (weakTypes.includes(edge.data('type'))) {
                    edge.addClass('path-weak-segment');
                    edge.style({
                        'line-style': 'dashed',
                        'line-dash-pattern': [4, 4]
                    });
                }
            });
        });
        
        console.log('[PathFinder] Simple path displayed with styles');
    },
    
    // Handle no path found
    handleNoPath() {
        console.log('[PathFinder] No paths found between nodes');
        
        // Red pulse animation on both nodes
        this.firstNode.addClass('path-no-connection');
        this.secondNode.addClass('path-no-connection');
        
        // Show toast notification
        this.showToast('⚠️ Путь не найден', 3000);
        
        // Auto-reset after 3 seconds
        setTimeout(() => {
            this.firstNode.removeClass('path-no-connection');
            this.secondNode.removeClass('path-no-connection');
            this.reset();
        }, 3000);
    },
    
    // Create metrics panel
    createMetricsPanel() {
        const panel = document.createElement('div');
        panel.className = 'path-metrics-panel';
        panel.style.display = 'none';
        
        panel.innerHTML = `
            <div class="metrics-content">
                <div class="path-metrics">
                    <span class="path-label">🌿 Найден путь:</span>
                    <span class="path-steps"></span>
                    <span class="path-difficulty"></span>
                </div>
            </div>
        `;
        
        document.body.appendChild(panel);
        this.metricsPanel = panel;
    },
    
    // Show metrics panel with single path info
    showMetrics(result) {
        if (!this.metricsPanel) return;
        
        const panel = this.metricsPanel;
        const pathMetrics = panel.querySelector('.path-metrics');
        
        // Update metrics for the single path
        const steps = result.path.edges().length;
        pathMetrics.querySelector('.path-steps').textContent = `Шагов: ${steps}`;
        pathMetrics.querySelector('.path-difficulty').textContent = `Сложность: ${result.totalDifficulty.toFixed(0)}`;
        pathMetrics.style.display = 'flex';
        
        // Show panel with animation
        panel.style.display = 'block';
        setTimeout(() => {
            panel.classList.add('show');
        }, 10);
    },
    
    // Hide metrics panel
    hideMetricsPanel() {
        if (!this.metricsPanel) return;
        
        this.metricsPanel.classList.remove('show');
        setTimeout(() => {
            this.metricsPanel.style.display = 'none';
        }, 300);
    },
    
    // Toast notification system
    showToast(message, duration = 3000) {
        // Remove existing toast if any
        const existingToast = document.querySelector('.toast-notification');
        if (existingToast) {
            existingToast.remove();
        }
        
        // Create new toast
        const toast = document.createElement('div');
        toast.className = 'toast-notification';
        toast.innerHTML = message;
        document.body.appendChild(toast);
        
        // Animate in
        setTimeout(() => toast.classList.add('show'), 10);
        
        // Auto remove
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, duration);
    },
    
    // Pulse animation methods
    startPulseAnimation(node) {
        // Store animation for this specific node
        if (!this.pulseAnimations) {
            this.pulseAnimations = new Map();
        }
        
        // Stop existing animation for this node if any
        if (this.pulseAnimations.has(node.id())) {
            clearInterval(this.pulseAnimations.get(node.id()));
        }
        
        // Animate the NODE ITSELF, not overlay
        let growing = true;
        // Get numeric size (remove 'px' if present)
        const originalSize = parseInt(node.style('width')) || parseInt(node.style('height')) || 30;
        const maxSize = originalSize * 1.5;
        let currentSize = originalSize;
        
        const animationId = setInterval(() => {
            if (growing) {
                currentSize += 2;
                if (currentSize >= maxSize) {
                    growing = false;
                }
            } else {
                currentSize -= 2;
                if (currentSize <= originalSize) {
                    growing = true;
                }
            }
            
            // Pulse the actual node size and color
            node.style({
                'width': currentSize,
                'height': currentSize,
                'border-width': 3,
                'border-color': '#ff6b00',
                'border-opacity': 1
            });
        }, 50);
        
        this.pulseAnimations.set(node.id(), animationId);
    },
    
    stopPulseAnimation(node = null) {
        if (!this.pulseAnimations) {
            // Fallback for old animation system
            if (this.pulseAnimation) {
                clearInterval(this.pulseAnimation);
                this.pulseAnimation = null;
            }
            return;
        }
        
        if (node) {
            // Stop animation for specific node
            if (this.pulseAnimations.has(node.id())) {
                clearInterval(this.pulseAnimations.get(node.id()));
                this.pulseAnimations.delete(node.id());
                
                // Reset node size and border
                node.removeStyle('width height border-width border-color border-opacity');
            }
        } else {
            // Stop all animations
            for (const [nodeId, animationId] of this.pulseAnimations) {
                clearInterval(animationId);
            }
            this.pulseAnimations.clear();
            
            // Reset node styles for both nodes
            if (this.firstNode) {
                this.firstNode.removeStyle('width height border-width border-color border-opacity');
            }
            if (this.secondNode) {
                this.secondNode.removeStyle('width height border-width border-color border-opacity');
            }
        }
    },
    
    // Dim all elements except the path
    dimOtherElements() {
        // Get path edges and nodes
        const pathEdges = new Set();
        const pathNodes = new Set();
        
        if (this.shortestPath) {
            this.shortestPath.edges().forEach(e => pathEdges.add(e.id()));
            this.shortestPath.nodes().forEach(n => pathNodes.add(n.id()));
        }
        
        // Dim all other edges - disable pointer events to prevent hover
        this.cy.edges().forEach(edge => {
            if (!pathEdges.has(edge.id())) {
                edge.addClass('path-dimmed');
                edge.style({
                    'opacity': 0.1,
                    'z-index': 1,
                    'events': 'no',  // Disable all events
                    'text-events': 'no'  // Also disable text events
                });
            }
        });
        
        // Dim all other nodes but keep them clickable for path selection
        this.cy.nodes().forEach(node => {
            if (!pathNodes.has(node.id())) {
                node.addClass('path-dimmed');
                node.style({
                    'opacity': 0.4,  // Keep visible
                    'z-index': 1,
                    // Don't disable events - we need them clickable!
                    // But disable hover effects by using a special class
                    'ghost': 'yes'  // Custom property to indicate dimmed state
                });
            }
        });
    },
    
    // DEPRECATED - old dimming method
    dimOtherEdges_OLD() {
        // Get all path edges
        const pathEdges = new Set();
        if (this.fastPath) {
            this.fastPath.edges().forEach(e => pathEdges.add(e.id()));
        }
        if (this.simplePath) {
            this.simplePath.edges().forEach(e => pathEdges.add(e.id()));
        }
        
        // Dim all other edges with pointer-events disabled to prevent hover
        this.cy.edges().forEach(edge => {
            if (!pathEdges.has(edge.id())) {
                edge.addClass('path-dimmed');
                edge.style({
                    'opacity': 0.15,
                    'z-index': 1,
                    'events': 'no'  // Disable hover events
                });
            }
        });
        
        // Also dim nodes not in the path
        const pathNodes = new Set();
        if (this.fastPath) {
            this.fastPath.nodes().forEach(n => pathNodes.add(n.id()));
        }
        if (this.simplePath) {
            this.simplePath.nodes().forEach(n => pathNodes.add(n.id()));
        }
        
        this.cy.nodes().forEach(node => {
            if (!pathNodes.has(node.id())) {
                node.addClass('path-dimmed');
                node.style({
                    'opacity': 0.3,
                    'events': 'no'  // Disable hover events
                });
            }
        });
    },
    
    // Restore all elements to normal
    restoreAllElements() {
        console.log('[PathFinder] Restoring all elements');
        
        // Batch restore for better performance
        this.cy.batch(() => {
            // Restore ALL edges
            this.cy.edges().forEach(edge => {
                // Remove all our classes
                edge.removeClass('path-dimmed path-shortest path-weak-segment');
                
                // RESTORE OPACITY TO 1, NOT REMOVE IT!
                edge.style('opacity', 1);
                
                // Remove other dimming/path styles
                edge.removeStyle('z-index events text-events');
                
                // If it was a path edge, remove path styles
                if (edge.hasClass('path-shortest') || edge.style('line-color') === '#00ff00' || edge.style('line-color') === 'rgb(0, 255, 0)') {
                    edge.removeStyle('line-color width curve-style target-arrow-color source-arrow-color target-arrow-shape source-arrow-shape arrow-scale overlay-color overlay-padding overlay-opacity line-style line-dash-pattern');
                }
            });
            
            // RESTORE ALL NODES - CRITICAL FIX
            this.cy.nodes().forEach(node => {
                // Remove all our classes
                node.removeClass('path-dimmed path-selected path-no-connection');
                
                // RESTORE OPACITY TO 1, NOT REMOVE IT!!!
                // Removing opacity style makes nodes invisible!
                node.style('opacity', 1);
                
                // Remove other styles we might have added
                node.removeStyle('ghost z-index events');
                
                // Remove pulsing styles if present
                node.removeStyle('width height border-width border-color border-opacity');
            });
        });
        
        console.log('[PathFinder] Elements restored - all nodes should be visible');
    }
};

// Listen for mode changes
document.addEventListener('mode-changed', (e) => {
    if (e.detail.mode === 'path') {
        PathFinder.activate();
    } else if (PathFinder.active) {
        PathFinder.deactivate();
    }
});

// Auto-initialize when graph is ready
document.addEventListener('k2-graph-ready', (e) => {
    const { cy, graphCore, config } = e.detail;
    
    // Get path_mode config from window
    const fullConfig = window.vizConfig || {};
    if (window.pathModeConfig) {
        fullConfig.path_mode = window.pathModeConfig;
    }
    
    PathFinder.init(cy, graphCore, fullConfig);
    console.log('[PathFinder] Auto-initialized via k2-graph-ready event');
});

// Export to window
window.PathFinder = PathFinder;
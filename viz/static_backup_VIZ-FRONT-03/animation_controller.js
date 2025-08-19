/**
 * K2-18 Knowledge Graph - Animation Controller Module
 * Manages node appearance animations and physics simulation
 */

class AnimationController {
    constructor(cy, config = {}) {
        this.cy = cy;
        this.config = {
            levelDelay: 200,           // Delay between depth levels (ms)
            nodeAnimDuration: 500,     // Node fade-in duration (ms)
            edgeAnimDuration: 500,     // Edge fade-in duration (ms)
            edgeDelay: 100,           // Delay before showing edges (ms)
            physicsDuration: 3000,     // Physics simulation duration (ms)
            animateOnLoad: true,       // Auto-animate on initialization
            ...config
        };
        
        this.isAnimating = false;
        this.animationPromise = null;
    }
    
    /**
     * Main animation sequence: nodes by depth, then edges
     * @returns {Promise} Resolves when animation completes
     */
    async animateGraph() {
        if (this.isAnimating) {
            console.log('Animation already in progress');
            return this.animationPromise;
        }
        
        this.isAnimating = true;
        console.log('Starting graph animation sequence...');
        
        try {

            // Hide canvas until elements are fully hidden to prevent first-paint flash
            const container = this.cy.container && this.cy.container();
            if (container) container.style.visibility = 'hidden';

            // Store initial positions if layout is running
            this.storeInitialPositions();
            
            // Hide all elements initially
            this.hideAllElements();

            // Reveal canvas after elements are hidden (first visible frame is empty graph)
            if (container) container.style.visibility = 'visible';
            
            // Animate nodes by prerequisite depth
            await this.animateNodesByDepth();
            
            // Small delay before edges
            await this.delay(this.config.edgeDelay);
            
            // Animate edges
            await this.animateEdges();
            
            // Run physics simulation if configured
            if (this.config.physicsDuration > 0) {
                await this.runPhysicsSimulation();
            }
            
            console.log('Animation sequence completed');
            
        } catch (error) {
            console.error('Animation error:', error);
        } finally {
            this.isAnimating = false;
            this.animationPromise = null;
        }
    }
    
    /**
     * Hide all graph elements
     */
    hideAllElements() {
        // Hide nodes & edges in a single batch to avoid intermediate frames
        this.cy.batch(() => {
            this.cy.edges().forEach(edge => {
                edge.style('opacity', 0);
            });
            this.cy.nodes().style({ 'opacity': 0, 'events': 'no' });
            this.cy.edges().style('opacity', 0);
        });
    }
    
    /**
     * Store initial node positions for smooth animation
     */
    storeInitialPositions() {
        this.cy.nodes().forEach(node => {
            node.data('initialPosition', {
                x: node.position('x'),
                y: node.position('y')
            });
        });
    }
    
    /**
     * Animate nodes appearance by prerequisite_depth levels
     * @returns {Promise} Resolves when all nodes are visible
     */
    async animateNodesByDepth() {
        // Group nodes by prerequisite_depth
        const nodesByDepth = this.groupNodesByDepth();
        const depths = Object.keys(nodesByDepth).sort((a, b) => a - b);
        
        console.log(`Animating ${depths.length} depth levels`);
        
        // Animate each depth level
        for (let i = 0; i < depths.length; i++) {
            const depth = depths[i];
            const nodes = nodesByDepth[depth];
            
            console.log(`Level ${depth}: animating ${nodes.length} nodes`);
            
            // Animate all nodes at this depth simultaneously
            const animations = nodes.map(node => this.animateNode(node));
            await Promise.all(animations);
            
            // Delay before next level (except for last level)
            if (i < depths.length - 1) {
                await this.delay(this.config.levelDelay);
            }
        }
        
        // Re-enable events on all nodes
        this.cy.nodes().style('events', 'yes');
    }
    
    /**
     * Group nodes by their prerequisite_depth
     * @returns {Object} Map of depth to node arrays
     */
    groupNodesByDepth() {
        const groups = {};
        
        this.cy.nodes().forEach(node => {
            const depth = node.data('prerequisite_depth') || 0;
            if (!groups[depth]) {
                groups[depth] = [];
            }
            groups[depth].push(node);
        });
        
        return groups;
    }
    
    /**
     * Animate single node appearance
     * @param {Object} node - Cytoscape node element
     * @returns {Promise} Resolves when animation completes
     */
    animateNode(node) {
        return new Promise(resolve => {
            // Calculate target opacity based on difficulty
            const difficulty = node.data('difficulty') || 3;
            const targetOpacity = this.calculateOpacity(difficulty);
            
            node.animate({
                style: { 
                    'opacity': targetOpacity
                },
                duration: this.config.nodeAnimDuration,
                easing: 'ease-out-cubic',
                complete: resolve
            });
        });
    }
    
    /**
     * Calculate node opacity based on difficulty
     * @param {number} difficulty - Node difficulty (1-5)
     * @returns {number} Opacity value (0.5-1.0)
     */
    calculateOpacity(difficulty) {
        // Map difficulty 1-5 to opacity 0.5-1.0
        const minOpacity = 0.5;
        const maxOpacity = 1.0;
        const normalized = (difficulty - 1) / 4;
        return minOpacity + (maxOpacity - minOpacity) * normalized;
    }
    
    /**
     * Animate edges appearance
     * @returns {Promise} Resolves when all edges are visible
     */
    async animateEdges() {
        console.log(`Animating ${this.cy.edges().length} edges`);
        
        // Animate only opacity to preserve colors from EdgeStyles
        return new Promise(resolve => {
            this.cy.batch(() => {
                this.cy.edges().forEach(edge => {
                    const edgeType = edge.data('type');
                    let targetOpacity = 0.6; // Default
                    
                    // Get edge-specific opacity if available
                    if (window.EdgeStyles && window.EdgeStyles.EDGE_STYLES[edgeType]) {
                        targetOpacity = window.EdgeStyles.EDGE_STYLES[edgeType].opacity;
                    }
                    
                    edge.animate({
                        style: { 'opacity': targetOpacity },
                        duration: this.config.edgeAnimDuration,
                        easing: 'ease-in-out'
                    });
                });
            });
            
            // Resolve after animation duration
            setTimeout(resolve, this.config.edgeAnimDuration);
        });
    }
    
    /**
     * Run physics simulation with cose-bilkent layout
     * @returns {Promise} Resolves when simulation completes
     */
    async runPhysicsSimulation() {
        console.log(`Running physics simulation for ${this.config.physicsDuration}ms`);
        
        // Check if cose-bilkent is available
        if (!this.cy.layout || typeof this.cy.layout !== 'function') {
            console.warn('Layout function not available');
            return;
        }
        
        return new Promise(resolve => {
            const layout = this.cy.layout({
                name: 'cose-bilkent',
                animate: 'end',
                animationDuration: this.config.physicsDuration,
                animationEasing: 'ease-out-cubic',
                nodeRepulsion: 4500,
                idealEdgeLength: 100,
                edgeElasticity: 0.45,
                nestingFactor: 0.1,
                gravity: 0.25,
                numIter: 2500,
                tile: true,
                tilingPaddingVertical: 10,
                tilingPaddingHorizontal: 10,
                gravityRangeCompound: 1.5,
                gravityCompound: 1.0,
                gravityRange: 3.8,
                stop: resolve
            });
            
            layout.run();
        });
    }
    
    /**
     * Utility delay function
     * @param {number} ms - Milliseconds to delay
     * @returns {Promise} Resolves after delay
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    /**
     * Stop any running animation
     */
    stopAnimation() {
        if (this.isAnimating) {
            this.cy.stop();
            this.isAnimating = false;
            
            // Show all elements
            this.cy.nodes().style('opacity', 1);
            this.cy.edges().style('opacity', 0.6);
            this.cy.nodes().style('events', 'yes');
            
            console.log('Animation stopped');
        }
    }
    
    /**
     * Reset graph to pre-animation state
     */
    reset() {
        this.stopAnimation();
        this.hideAllElements();
    }
    
    /**
     * Highlight specific path through the graph
     * @param {Array} nodeIds - Array of node IDs in path order
     * @param {Object} options - Highlight options
     */
    async highlightPath(nodeIds, options = {}) {
        const config = {
            stepDelay: 300,
            highlightDuration: 500,
            dimOthers: true,
            ...options
        };
        
        // Dim all elements if requested
        if (config.dimOthers) {
            this.cy.elements().addClass('dimmed');
        }
        
        // Highlight nodes in sequence
        for (let i = 0; i < nodeIds.length; i++) {
            const node = this.cy.getElementById(nodeIds[i]);
            if (node.length === 0) continue;
            
            // Highlight node
            node.removeClass('dimmed').addClass('highlighted');
            
            // Highlight edges to next node
            if (i < nodeIds.length - 1) {
                const nextNode = this.cy.getElementById(nodeIds[i + 1]);
                const edge = node.edgesWith(nextNode);
                edge.removeClass('dimmed').addClass('highlighted');
            }
            
            await this.delay(config.stepDelay);
        }
    }
    
    /**
     * Clear all highlights
     */
    clearHighlights() {
        this.cy.elements().removeClass('highlighted dimmed');
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AnimationController;
}

// Export for browser use
if (typeof window !== 'undefined') {
    window.AnimationController = AnimationController;
}
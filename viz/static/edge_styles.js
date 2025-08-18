/**
 * K2-18 Knowledge Graph - Edge Styles Module
 * Defines visual styles for all 9 edge types from the schema
 */

// Edge style configurations for all relationship types
const EDGE_STYLES = {
    // Strong educational dependencies (4px width)
    PREREQUISITE: {
        lineColor: '#e74c3c',      // Red - strong dependency
        lineStyle: 'solid',
        baseWidth: 4,
        arrow: 'triangle',
        opacity: 0.8
    },
    TESTS: {
        lineColor: '#f39c12',      // Orange - assessment relationship
        lineStyle: 'solid',
        baseWidth: 4,
        arrow: 'triangle',
        opacity: 0.8
    },
    
    // Clear relationships (2.5px width)
    ELABORATES: {
        lineColor: '#3498db',      // Blue - elaboration/detail
        lineStyle: 'dashed',
        dashPattern: [8, 4],
        baseWidth: 2.5,
        arrow: 'triangle',
        opacity: 0.7
    },
    EXAMPLE_OF: {
        lineColor: '#9b59b6',      // Purple - example
        lineStyle: 'dotted',
        dashPattern: [2, 4],
        baseWidth: 2.5,
        arrow: 'triangle',
        opacity: 0.7
    },
    PARALLEL: {
        lineColor: '#95a5a6',      // Gray - parallel topic
        lineStyle: 'solid',
        baseWidth: 2.5,
        arrow: 'triangle',
        opacity: 0.6
    },
    REVISION_OF: {
        lineColor: '#27ae60',      // Green - revision/update
        lineStyle: 'dashed',
        dashPattern: [6, 3],
        baseWidth: 2.5,
        arrow: 'triangle',
        opacity: 0.7
    },
    
    // Weak hints and references (1px width)
    HINT_FORWARD: {
        lineColor: '#5dade2',      // Light blue - hint to future content
        lineStyle: 'dotted',
        dashPattern: [2, 6],
        baseWidth: 1,
        arrow: 'tee',
        opacity: 0.5
    },
    REFER_BACK: {
        lineColor: '#ec7063',      // Pink - reference to past content
        lineStyle: 'dotted',
        dashPattern: [2, 6],
        baseWidth: 1,
        arrow: 'tee',
        opacity: 0.5
    },
    MENTIONS: {
        lineColor: '#bdc3c7',      // Light gray - simple mention
        lineStyle: 'dashed',
        dashPattern: [4, 4],
        baseWidth: 1,
        arrow: 'triangle-tee',
        opacity: 0.4
    }
};

/**
 * Generate Cytoscape style definitions for all edge types
 * @param {Object} options - Configuration options
 * @param {number} options.interClusterMultiplier - Multiplier for inter-cluster edge width (default: 1.5)
 * @returns {Array} Array of Cytoscape style objects
 */
function generateEdgeStyles(options = {}) {
    const { interClusterMultiplier = 1.5 } = options;
    const styles = [];
    
    // Add base edge style FIRST (lowest priority in Cytoscape)
    styles.push({
        selector: 'edge',
        style: {
            'width': 2,
            'line-color': '#95a5a6',  // Default gray for edges without type
            'target-arrow-color': '#95a5a6',
            'target-arrow-shape': 'triangle',
            'opacity': 0.6,
            'curve-style': 'bezier',
            'control-point-step-size': 40,
            'transition-property': 'opacity, width',
            'transition-duration': '250ms',
            'transition-timing-function': 'ease-in-out',
            'text-rotation': 'autorotate',
            'text-margin-y': -10,
            'font-size': '10px',
            'font-family': 'Inter, sans-serif',
            'color': '#4a5568',
            'text-outline-width': 2,
            'text-outline-color': '#ffffff'
        }
    });
    
    // Generate styles for each edge type AFTER base (higher priority)
    Object.entries(EDGE_STYLES).forEach(([type, config]) => {
        const selector = `edge[type="${type}"]`;
        const style = {
            'line-color': config.lineColor,
            'target-arrow-color': config.lineColor,
            'source-arrow-color': config.lineColor,
            'target-arrow-shape': config.arrow,
            'width': config.baseWidth,
            'opacity': config.opacity
        };
        
        // Add line style specific properties
        if (config.lineStyle === 'dashed') {
            style['line-style'] = 'dashed';
            style['line-dash-pattern'] = config.dashPattern || [6, 3];
        } else if (config.lineStyle === 'dotted') {
            style['line-style'] = 'dotted';
            style['line-dash-pattern'] = config.dashPattern || [2, 4];
        } else {
            style['line-style'] = 'solid';
        }
        
        styles.push({ selector, style });
        
        // Add inter-cluster variant
        const interClusterSelector = `edge[type="${type}"][is_inter_cluster_edge]`;
        styles.push({
            selector: interClusterSelector,
            style: {
                'width': config.baseWidth * interClusterMultiplier,
                'opacity': Math.min(1, config.opacity + 0.1),
                'z-index': 10
            }
        });
    });
    
    // Selected state only (hover is not supported properly in Cytoscape)
    styles.push(
        {
            selector: 'edge:selected',
            style: {
                'opacity': 1,
                'z-index': 1000,
                'line-color': '#f39c12',
                'target-arrow-color': '#f39c12',
                'source-arrow-color': '#f39c12'
            }
        },
        {
            selector: 'edge.highlighted',
            style: {
                'opacity': 1,
                'z-index': 998,
                'width': (ele) => {
                    const baseWidth = EDGE_STYLES[ele.data('type')]?.baseWidth || 2;
                    const isInterCluster = ele.data('is_inter_cluster_edge');
                    return baseWidth * (isInterCluster ? interClusterMultiplier : 1) * 1.3;
                }
            }
        },
        {
            selector: 'edge.dimmed',
            style: {
                'opacity': 0.2
            }
        }
    );
    
    return styles;
}

/**
 * Get edge style configuration by type
 * @param {string} type - Edge type from schema
 * @returns {Object} Style configuration object
 */
function getEdgeStyle(type) {
    return EDGE_STYLES[type] || EDGE_STYLES.MENTIONS;
}

/**
 * Get color for edge type
 * @param {string} type - Edge type from schema
 * @returns {string} Hex color code
 */
function getEdgeColor(type) {
    return EDGE_STYLES[type]?.lineColor || '#bdc3c7';
}

/**
 * Check if edge type represents a strong relationship
 * @param {string} type - Edge type from schema
 * @returns {boolean} True if strong relationship (width >= 4px)
 */
function isStrongRelationship(type) {
    return EDGE_STYLES[type]?.baseWidth >= 4;
}

/**
 * Check if edge type is educational (used in educational_importance metric)
 * @param {string} type - Edge type from schema
 * @returns {boolean} True if educational edge type
 */
function isEducationalEdge(type) {
    const educationalTypes = ['PREREQUISITE', 'ELABORATES', 'TESTS', 'EXAMPLE_OF'];
    return educationalTypes.includes(type);
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        EDGE_STYLES,
        generateEdgeStyles,
        getEdgeStyle,
        getEdgeColor,
        isStrongRelationship,
        isEducationalEdge
    };
}

// Export for browser use
if (typeof window !== 'undefined') {
    window.EdgeStyles = {
        EDGE_STYLES,
        generateEdgeStyles,
        getEdgeStyle,
        getEdgeColor,
        isStrongRelationship,
        isEducationalEdge
    };
}
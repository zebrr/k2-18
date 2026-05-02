// Main controller for viewer application
// All modules are now global variables

// Global state
const ViewerCore = {
    state: {
        graphData: null,
        conceptData: null,
        activeNodeId: null,
        selectedEdge: null
    },

    modules: {
        searchFilter: null,
        nodeExplorer: null,
        edgeInspector: null,
        formatters: null
    }
};

function initViewer(graphData, conceptData) {
    console.log('Viewer core initializing...');
    console.log(`Loaded ${graphData.nodes.length} nodes and ${graphData.edges.length} edges`);

    // Store data
    ViewerCore.state.graphData = graphData;
    ViewerCore.state.conceptData = conceptData;

    // Check if modules are available
    if (typeof SearchFilter === 'undefined' || typeof NodeExplorer === 'undefined') {
        console.error('Required modules not loaded');
        return;
    }

    // Initialize modules
    ViewerCore.modules.searchFilter = SearchFilter;
    ViewerCore.modules.nodeExplorer = NodeExplorer;
    ViewerCore.modules.edgeInspector = EdgeInspector;
    ViewerCore.modules.formatters = Formatters;

    // Make NodeExplorer globally available for EdgeInspector
    window.NodeExplorer = NodeExplorer;

    // Initialize formatters (make globally available)
    if (typeof Formatters !== 'undefined') {
        window.Formatters = Formatters;
        Formatters.init();
    }

    // Initialize search filter
    SearchFilter.init(graphData);

    // Initialize node explorer
    NodeExplorer.init(graphData, conceptData);

    // Initialize edge inspector
    if (typeof EdgeInspector !== 'undefined') {
        EdgeInspector.init(graphData);
    }
    
    // Setup event listeners
    setupEventListeners();
    
    // Initial render
    SearchFilter.applyFilters();
    
    console.log('Viewer core initialized successfully');
}

function setupEventListeners() {
    // Listen for node selection events
    document.addEventListener('node-selected', (e) => {
        handleNodeSelection(e.detail);
    });
    
    // Listen for filter changes
    document.addEventListener('filter-changed', (e) => {
        console.log(`Filter changed: ${e.detail.filteredNodes.length} nodes visible`);
    });
}

function handleNodeSelection(detail) {
    const { nodeId, node, previousNodeId } = detail;
    
    console.log(`Node selection changed: ${previousNodeId} -> ${nodeId}`);
    
    // Update global state
    ViewerCore.state.activeNodeId = nodeId;
    
    // TODO: In future stages, update Column B with node details
    // For now, just log the selection
    if (node) {
        console.log('Selected node details:', {
            id: node.id,
            type: node.type,
            text: node.text?.substring(0, 100) + '...',
            cluster: node.cluster_id
        });
    }
}

// Make ViewerCore global for debugging
window.ViewerCore = ViewerCore;

// Make initViewer globally available
window.initViewer = initViewer;
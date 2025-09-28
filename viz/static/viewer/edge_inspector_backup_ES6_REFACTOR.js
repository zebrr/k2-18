// Edge inspection functionality
export const EdgeInspector = {
    selectedEdge: null,
    graphData: null,

    init(graphData) {
        console.log('Edge inspector initializing...');
        this.graphData = graphData;
        this.bindEvents();
        console.log('Edge inspector initialized');
    },

    bindEvents() {
        // Listen for edge selection from NodeExplorer
        document.addEventListener('edge-selected', (e) => {
            this.selectEdge(e.detail);
        });

        // Listen for edge deselection
        document.addEventListener('edge-deselected', () => {
            this.clearEdge();
        });

        // Listen for node changes to clear edge
        document.addEventListener('node-selected', () => {
            this.clearEdge();
        });
    },

    selectEdge(edgeData) {
        const { edge, otherNode } = edgeData;

        // Get current active node
        const activeNodeId = NodeExplorer.state.activeNodeId;
        if (!activeNodeId) return;

        const sourceNode = this.graphData.nodes.find(n => n.id === activeNodeId);
        if (!sourceNode) return;

        // Determine direction
        const isIncoming = edge.target === activeNodeId;
        const targetNode = otherNode;

        this.selectedEdge = edge;

        // Render edge panel
        this.renderEdgePanel(edge, sourceNode, targetNode, isIncoming);

        // Render target node in column C
        this.renderRelatedNode(targetNode, edge);
    },

    clearEdge() {
        this.selectedEdge = null;
        this.renderEdgePanel(null);
        this.clearRelatedNode();
    },

    renderEdgePanel(edge, sourceNode, targetNode, isIncoming) {
        const panel = document.getElementById('edge-panel');
        if (!panel) return;

        if (!edge) {
            panel.innerHTML = `
                <div class="edge-panel-empty">
                    Выберите связь в таблице для анализа
                </div>
            `;
            return;
        }

        // Get display values
        const fromType = isIncoming ? this.getNodeTypeShort(targetNode.type) : this.getNodeTypeShort(sourceNode.type);
        const toType = isIncoming ? this.getNodeTypeShort(sourceNode.type) : this.getNodeTypeShort(targetNode.type);
        const edgeColor = this.getEdgeColor(edge.type);
        const thickness = this.getLineThickness(edge.weight);

        const isInterCluster = this.isInterClusterEdge(sourceNode, targetNode);

        panel.innerHTML = `
            <!-- Line 1: types and edge with metrics -->
            <div class="edge-line-types">
                <span class="node-type-badge badge-${(isIncoming ? targetNode.type : sourceNode.type || 'unknown').toLowerCase()}">
                    ${fromType}
                </span>
                <div class="edge-arrow-inline">
                    <span class="edge-arrow-label" style="color: ${edgeColor}">
                        ${edge.type}
                        <span style="font-weight: normal; font-size: 0.9em; color: #6b7280">
                            (вес: ${(edge.weight || 1).toFixed(2)}${isInterCluster ? ' | межкластерное: true' : ''})
                        </span>
                    </span>
                    <div class="edge-arrow-line" style="background: ${edgeColor}; height: ${thickness}px; margin-left: 10px"></div>
                    <div class="edge-arrow-head" style="border-left-color: ${edgeColor}"></div>
                </div>
                <span class="node-type-badge badge-${(isIncoming ? sourceNode.type : targetNode.type || 'unknown').toLowerCase()}">
                    ${toType}
                </span>
            </div>

            <!-- Line 2: condition (if exists) -->
            ${edge.conditions ? `
                <div class="edge-line-condition" title="${this.escapeHtml(edge.conditions)}">
                    Условие: ${this.escapeHtml(edge.conditions)}
                </div>
            ` : ''}
        `;
    },

    renderRelatedNode(node, edge) {
        const container = document.getElementById('related-node-content');
        if (!container) return;

        // Clear and add header with action button
        container.innerHTML = `
            <div class="related-node-header">
                <button class="make-active-btn" data-node-id="${node.id}">
                    ← Сделать активным узлом
                </button>
            </div>
            <div class="related-node-content"></div>
        `;

        // Render node using NodeExplorer's method
        const contentContainer = container.querySelector('.related-node-content');
        this.renderNodeDetails(node, contentContainer, true); // true = isSecondary

        // Bind action button
        container.querySelector('.make-active-btn').addEventListener('click', () => {
            this.makeNodeActive(node);
        });
    },

    renderNodeDetails(node, container, isSecondary = false) {
        // Reuse NodeExplorer's rendering logic but without edge interaction
        const tempContainer = document.createElement('div');

        // Get NodeExplorer to render the node
        if (typeof NodeExplorer !== 'undefined' && NodeExplorer.renderFormattedView) {
            // Temporarily save the container
            const originalContainer = container;

            // Create a wrapper div
            const wrapper = document.createElement('div');
            wrapper.id = 'active-node-content';
            tempContainer.appendChild(wrapper);

            // Render using NodeExplorer
            NodeExplorer.renderFormattedView(node, wrapper);

            // Remove the header elements we don't need
            const header = wrapper.querySelector('.active-node-header');
            if (header) {
                header.remove();
            }

            // Transfer content to original container
            originalContainer.innerHTML = wrapper.innerHTML;

            // If secondary, disable edge table interactions
            if (isSecondary) {
                const edgeRows = originalContainer.querySelectorAll('.edge-row');
                edgeRows.forEach(row => {
                    row.style.cursor = 'default';
                    row.onclick = null;
                    row.classList.remove('active');
                });
            }
        } else {
            // Fallback: basic rendering
            container.innerHTML = `
                <div class="node-content">
                    <div class="node-header">
                        <span class="node-type-badge badge-${(node.type || 'unknown').toLowerCase()}">
                            ${node.type || 'Unknown'}
                        </span>
                        <span class="node-id">${node.id}</span>
                    </div>
                    <div class="content-section">
                        <h4>Содержание</h4>
                        <div>${node.text || '-'}</div>
                    </div>
                </div>
            `;
        }
    },

    clearRelatedNode() {
        const container = document.getElementById('related-node-content');
        if (container) {
            container.innerHTML = `
                <div class="placeholder-message">
                    Выберите ребро для исследования
                </div>
            `;
        }
    },

    makeNodeActive(node) {
        // Clear current edge
        this.clearEdge();

        // Trigger node selection via NodeExplorer
        if (typeof NodeExplorer !== 'undefined' && NodeExplorer.selectNode) {
            NodeExplorer.selectNode(node.id);
        } else {
            // Fallback: dispatch event
            document.dispatchEvent(new CustomEvent('make-node-active', {
                detail: { nodeId: node.id }
            }));
        }
    },

    // Helper methods
    getNodeTypeShort(type) {
        const typeMap = {
            'Chunk': 'CHUNK',
            'Concept': 'CONCEPT',
            'Assessment': 'ASMNT'
        };
        return typeMap[type] || type;
    },

    getEdgeColor(type) {
        const colors = {
            'PREREQUISITE': '#3b82f6',
            'ELABORATES': '#10b981',
            'EXAMPLE_OF': '#f59e0b',
            'PARALLEL': '#8b5cf6',
            'TESTS': '#ef4444',
            'REVISION_OF': '#06b6d4',
            'HINT_FORWARD': '#a3a3a3',
            'REFER_BACK': '#a3a3a3',
            'MENTIONS': '#d4d4d4'
        };
        return colors[type] || '#6b7280';
    },

    getLineThickness(weight) {
        // weight 0.3-1.0 → thickness 1-4px
        const minWeight = 0.3;
        const maxWeight = 1.0;
        const minThickness = 1;
        const maxThickness = 4;

        const normalized = Math.max(0, Math.min(1,
            (weight - minWeight) / (maxWeight - minWeight)
        ));
        return minThickness + normalized * (maxThickness - minThickness);
    },

    isInterClusterEdge(sourceNode, targetNode) {
        return sourceNode.cluster_id !== targetNode.cluster_id;
    },

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
};

// Export for module system
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EdgeInspector;
}
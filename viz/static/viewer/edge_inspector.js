// Edge inspection functionality
const EdgeInspector = {
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

        // Get display values - ALWAYS show B on left, C on right
        const leftType = this.getNodeTypeShort(sourceNode.type);  // Column B node
        const rightType = this.getNodeTypeShort(targetNode.type); // Column C node
        const edgeColor = this.getEdgeColor(edge.type);
        const thickness = this.getLineThickness(edge.weight);

        const isInterCluster = this.isInterClusterEdge(sourceNode, targetNode);

        // Choose arrow based on actual direction
        if (isIncoming) {
            // Arrow from right to left: C -> B
            panel.innerHTML = `
                <div class="edge-panel-oneline">
                    <span class="node-type-badge badge-${(sourceNode.type || 'unknown').toLowerCase()}">
                        ${leftType}
                    </span>
                    <div class="edge-arrow-head edge-arrow-head-left" style="border-right-color: ${edgeColor}"></div>
                    <div class="edge-arrow-line" style="background: ${edgeColor}; height: ${thickness}px; width: 40px"></div>
                    <span class="edge-label-center" style="color: ${edgeColor}">
                        ${edge.type}
                        <span class="edge-metrics">(вес: ${(edge.weight || 1).toFixed(2)}${isInterCluster ? ', межкластерное: true' : ''})</span>
                    </span>
                    <div class="edge-arrow-line" style="background: ${edgeColor}; height: ${thickness}px; width: 40px"></div>
                    <span class="node-type-badge badge-${(targetNode.type || 'unknown').toLowerCase()}">
                        ${rightType}
                    </span>
                </div>
            `;
        } else {
            // Arrow from left to right: B -> C
            panel.innerHTML = `
                <div class="edge-panel-oneline">
                    <span class="node-type-badge badge-${(sourceNode.type || 'unknown').toLowerCase()}">
                        ${leftType}
                    </span>
                    <div class="edge-arrow-line" style="background: ${edgeColor}; height: ${thickness}px; width: 40px"></div>
                    <span class="edge-label-center" style="color: ${edgeColor}">
                        ${edge.type}
                        <span class="edge-metrics">(вес: ${(edge.weight || 1).toFixed(2)}${isInterCluster ? ', межкластерное: true' : ''})</span>
                    </span>
                    <div class="edge-arrow-line" style="background: ${edgeColor}; height: ${thickness}px; width: 40px"></div>
                    <div class="edge-arrow-head" style="border-left-color: ${edgeColor}"></div>
                    <span class="node-type-badge badge-${(targetNode.type || 'unknown').toLowerCase()}">
                        ${rightType}
                    </span>
                </div>
            `;
        }

        panel.innerHTML += `
            ${edge.conditions ? `
                <div class="edge-condition-line">
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
        // Use NodeExplorer to render formatted content
        if (typeof NodeExplorer !== 'undefined' && NodeExplorer.renderFormattedView) {
            // Create temporary wrapper to get formatted content
            const wrapper = document.createElement('div');
            NodeExplorer.renderFormattedView(node, wrapper);

            // Remove the header that we don't need in column C
            const header = wrapper.querySelector('.active-node-header');
            if (header) {
                header.remove();
            }

            // Transfer formatted content to container
            container.innerHTML = wrapper.innerHTML;

            // Apply syntax highlighting to code blocks
            if (typeof hljs !== 'undefined') {
                container.querySelectorAll('pre code').forEach((block) => {
                    hljs.highlightElement(block);
                });
            }

            // Trigger math rendering for all formatted content sections
            setTimeout(() => {
                if (window.Formatters && window.Formatters.renderMath) {
                    container.querySelectorAll('.formatted-content').forEach((element) => {
                        window.Formatters.renderMath(element);
                    });
                }
            }, 0);

            // Re-attach metric tooltip click handlers for column C
            container.querySelectorAll('.metric-info').forEach(infoIcon => {
                const tooltip = infoIcon.querySelector('.metric-tooltip');
                if (tooltip) {
                    infoIcon.onclick = (e) => {
                        e.stopPropagation();
                        const isVisible = tooltip.style.display === 'block';
                        // Hide all other tooltips
                        document.querySelectorAll('.metric-tooltip').forEach(t => {
                            t.style.display = 'none';
                        });
                        // Toggle this tooltip
                        tooltip.style.display = isVisible ? 'none' : 'block';
                    };
                }
            });

            // If secondary, disable edge table interactions
            if (isSecondary) {
                const edgeRows = container.querySelectorAll('.edge-row');
                edgeRows.forEach(row => {
                    row.style.cursor = 'default';
                    row.onclick = null;
                    row.classList.remove('active');
                });
            }
        } else {
            // Fallback: basic rendering without formatting
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
                        <div class="formatted-content">${node.text || '-'}</div>
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
        // Match colors from edge_styles.js
        const colors = {
            'PREREQUISITE': '#e74c3c',  // Red - strong dependency
            'ELABORATES': '#3498db',     // Blue - elaboration/detail
            'EXAMPLE_OF': '#9b59b6',     // Purple - example
            'PARALLEL': '#95a5a6',       // Gray - parallel topic
            'TESTS': '#f39c12',          // Orange - assessment
            'REVISION_OF': '#27ae60',    // Green - revision/update
            'HINT_FORWARD': '#95a5a6',   // Light gray - weak forward reference
            'REFER_BACK': '#95a5a6',     // Light gray - weak backward reference
            'MENTIONS': '#bdc3c7'        // Lighter gray - mention
        };
        return colors[type] || '#bdc3c7';
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

// EdgeInspector is now global
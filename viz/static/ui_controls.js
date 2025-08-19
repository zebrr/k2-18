/**
 * UI Controls Module for K2-18 Visualization
 * Handles top panel, side panel, hover effects, and info popup
 */

const UIControls = {
    // Components
    topPanel: null,
    sidePanel: null,
    hoverController: null,
    infoPopup: null,
    tooltip: null,
    
    // State
    state: {
        visibleTypes: {
            'Chunk': true,
            'Concept': true,
            'Assessment': true
        },
        sidePanelOpen: false,
        activeTab: 'dictionary',
        hoveredNode: null,
        hoveredConcept: null,
        infoPanelOpen: false
    },
    
    // Initialization
    init(cy, graphData, conceptData, config) {
        this.cy = cy;
        this.graphData = graphData;
        this.conceptData = conceptData || { concepts: [] };
        this.config = config;
        
        // Create UI components
        this.createTopPanel();
        this.createSidePanel();
        this.setupHoverEffects();
        this.createInfoPopup();
        this.createTooltip();
        this.setupKeyboardHandlers();
        
        // Initial update
        this.updateCounters();
        
        console.log('[UIControls] Initialized');
    },
    
    // Top Panel Implementation - REMOVED, filters moved to main header
    createTopPanel() {
        // Add filters to existing header instead of creating new panel
        const header = document.getElementById('header');
        if (!header) {
            console.warn('[UIControls] Header not found, cannot add filters');
            return;
        }
        
        // Find stats container or create if doesn't exist
        let statsContainer = document.getElementById('stats-container');
        if (!statsContainer) {
            statsContainer = document.createElement('div');
            statsContainer.id = 'stats-container';
            header.appendChild(statsContainer);
        }
        
        // Add filters and counters to stats container
        statsContainer.innerHTML = `
            <div class="filters">
                <label class="filter-checkbox">
                    <input type="checkbox" id="filter-chunk" checked>
                    <span>Chunks</span>
                </label>
                <label class="filter-checkbox">
                    <input type="checkbox" id="filter-concept" checked>
                    <span>Concepts</span>
                </label>
                <label class="filter-checkbox">
                    <input type="checkbox" id="filter-assessment" checked>
                    <span>Assessments</span>
                </label>
            </div>
            <div class="counters">
                <span>Видимые: <strong id="visible-nodes">0</strong>/<strong id="total-nodes">0</strong></span>
                <span class="separator">|</span>
                <span>Рёбра: <strong id="visible-edges">0</strong>/<strong id="total-edges">0</strong></span>
            </div>
        `;
        
        // Make stats container visible
        statsContainer.style.visibility = 'visible';
        
        // Add event listeners for checkboxes
        document.getElementById('filter-chunk').addEventListener('change', (e) => {
            this.toggleNodeType('Chunk', e.target.checked);
        });
        
        document.getElementById('filter-concept').addEventListener('change', (e) => {
            this.toggleNodeType('Concept', e.target.checked);
        });
        
        document.getElementById('filter-assessment').addEventListener('change', (e) => {
            this.toggleNodeType('Assessment', e.target.checked);
        });
    },
    
    toggleNodeType(type, visible) {
        // Update state
        this.state.visibleTypes[type] = visible;
        
        // Use batch for performance
        this.cy.batch(() => {
            // Process nodes
            this.cy.nodes().forEach(node => {
                if (node.data('type') === type) {
                    if (visible) {
                        node.removeClass('hidden');
                    } else {
                        node.addClass('hidden');
                    }
                }
            });
            
            // Process edges - show only if both ends are visible
            this.cy.edges().forEach(edge => {
                const source = edge.source();
                const target = edge.target();
                
                if (source.hasClass('hidden') || target.hasClass('hidden')) {
                    edge.addClass('hidden-edge');
                } else {
                    edge.removeClass('hidden-edge');
                }
            });
        });
        
        // Update counters
        this.updateCounters();
    },
    
    // Side Panel Implementation
    createSidePanel() {
        // Create panel container
        const panel = document.createElement('div');
        panel.className = 'side-panel';
        panel.innerHTML = `
            <div class="panel-content">
                <div class="tabs">
                    <button class="tab active" data-tab="dictionary">Словарь</button>
                    <button class="tab" data-tab="top-nodes">TOP-узлы</button>
                </div>
                <div class="tab-content" id="dictionary-content">
                    <div class="concept-list" id="concept-list">
                        <!-- Will be populated dynamically -->
                    </div>
                </div>
                <div class="tab-content" id="top-nodes-content" style="display: none;">
                    <!-- Will be populated dynamically -->
                </div>
            </div>
        `;
        
        // Create tab button as a separate element (not child of panel)
        const tabButton = document.createElement('div');
        tabButton.className = 'side-panel-tab';
        tabButton.title = 'Словарь и TOP-узлы';
        tabButton.innerHTML = '📚';
        
        document.body.appendChild(panel);
        document.body.appendChild(tabButton);
        this.sidePanel = panel;
        
        // Add event listeners
        tabButton.addEventListener('click', () => this.toggleSidePanel());
        
        // Tab switching
        panel.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => this.switchTab(tab.dataset.tab));
        });
        
        // Populate content
        this.populateDictionary();
        this.populateTopNodes();
    },
    
    toggleSidePanel() {
        this.state.sidePanelOpen = !this.state.sidePanelOpen;
        this.sidePanel.classList.toggle('open', this.state.sidePanelOpen);
    },
    
    switchTab(tabName) {
        // Update active tab
        this.state.activeTab = tabName;
        
        // Update tab buttons
        this.sidePanel.querySelectorAll('.tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.tab === tabName);
        });
        
        // Show/hide content
        document.getElementById('dictionary-content').style.display = 
            tabName === 'dictionary' ? 'block' : 'none';
        document.getElementById('top-nodes-content').style.display = 
            tabName === 'top-nodes' ? 'block' : 'none';
    },
    
    populateDictionary() {
        const listElement = document.getElementById('concept-list');
        
        // Check if we have concepts
        if (!this.conceptData || !this.conceptData.concepts || this.conceptData.concepts.length === 0) {
            listElement.innerHTML = '<div class="empty-message">Нет данных о концептах</div>';
            return;
        }
        
        // Sort concepts alphabetically by primary term
        const sortedConcepts = [...this.conceptData.concepts].sort((a, b) => {
            // Handle different data structures
            const termA = a.term?.primary || a.primary?.term || '';
            const termB = b.term?.primary || b.primary?.term || '';
            return termA.localeCompare(termB, 'ru');
        });
        
        // Create list items
        const html = sortedConcepts.map(concept => {
            // Handle different id fields
            const conceptId = concept.concept_id || concept.id;
            // Handle different term structures
            const term = concept.term?.primary || concept.primary?.term || 'Unknown';
            
            // Count mentions using mention_index if available
            let count = 0;
            if (this.conceptData._meta?.mention_index?.[conceptId]) {
                count = this.conceptData._meta.mention_index[conceptId].count || 0;
            }
            
            return `
                <div class="concept-item" data-concept-id="${conceptId}">
                    <span class="concept-name">${term}</span>
                    <span class="concept-count">${count}</span>
                </div>
            `;
        }).join('');
        
        listElement.innerHTML = html;
        
        // Add hover listeners
        listElement.querySelectorAll('.concept-item').forEach(item => {
            item.addEventListener('mouseenter', () => {
                const conceptId = item.dataset.conceptId;
                this.highlightConceptNodes(conceptId);
            });
            
            item.addEventListener('mouseleave', () => {
                this.clearConceptHighlight();
            });
        });
    },
    
    populateTopNodes() {
        const container = document.getElementById('top-nodes-content');
        
        // Calculate top nodes by different metrics
        const nodeData = this.cy.nodes().map(node => ({
            id: node.id(),
            label: this.truncateText(node.data('text') || node.id(), 30),
            pagerank: node.data('pagerank') || 0,
            betweenness: node.data('betweenness_centrality') || 0,
            degree: (node.data('degree_in') || 0) + (node.data('degree_out') || 0)
        }));
        
        // Sort and get top 5 for each metric
        const topByPagerank = [...nodeData].sort((a, b) => b.pagerank - a.pagerank).slice(0, 5);
        const topByBetweenness = [...nodeData].sort((a, b) => b.betweenness - a.betweenness).slice(0, 5);
        const topByDegree = [...nodeData].sort((a, b) => b.degree - a.degree).slice(0, 5);
        
        // Generate HTML
        container.innerHTML = `
            <div class="top-section">
                <h4>Важность (PageRank)</h4>
                <ul class="top-list">
                    ${topByPagerank.map(node => `
                        <li class="top-node-item" data-node-id="${node.id}">
                            ${node.label}
                            <span class="metric-value">${node.pagerank.toFixed(3)}</span>
                        </li>
                    `).join('')}
                </ul>
            </div>
            <div class="top-section">
                <h4>Мосты (Betweenness)</h4>
                <ul class="top-list">
                    ${topByBetweenness.map(node => `
                        <li class="top-node-item" data-node-id="${node.id}">
                            ${node.label}
                            <span class="metric-value">${node.betweenness.toFixed(3)}</span>
                        </li>
                    `).join('')}
                </ul>
            </div>
            <div class="top-section">
                <h4>Хабы (Degree)</h4>
                <ul class="top-list">
                    ${topByDegree.map(node => `
                        <li class="top-node-item" data-node-id="${node.id}">
                            ${node.label}
                            <span class="metric-value">${node.degree}</span>
                        </li>
                    `).join('')}
                </ul>
            </div>
        `;
        
        // Add event listeners
        container.querySelectorAll('.top-node-item').forEach(item => {
            item.addEventListener('mouseenter', () => {
                const nodeId = item.dataset.nodeId;
                this.pulseNode(nodeId);
            });
            
            item.addEventListener('mouseleave', () => {
                this.clearPulse();
            });
            
            item.addEventListener('click', () => {
                const nodeId = item.dataset.nodeId;
                this.centerOnNode(nodeId);
            });
        });
    },
    
    // Hover Effects
    setupHoverEffects() {
        let hoverTimeout = null;
        
        // A. Hover on graph nodes
        this.cy.on('mouseover', 'node', (evt) => {
            const node = evt.target;
            
            // Clear any existing timeout
            if (hoverTimeout) {
                clearTimeout(hoverTimeout);
                hoverTimeout = null;
            }
            
            // Increase size
            node.addClass('hover-node');
            
            // Highlight connected edges
            node.connectedEdges().addClass('hover-connected');
            
            // Show tooltip after delay
            hoverTimeout = setTimeout(() => {
                this.showTooltip(node, evt.renderedPosition);
            }, 500);
        });
        
        this.cy.on('mouseout', 'node', (evt) => {
            const node = evt.target;
            
            // Clear timeout
            if (hoverTimeout) {
                clearTimeout(hoverTimeout);
                hoverTimeout = null;
            }
            
            // Remove hover classes
            node.removeClass('hover-node');
            node.connectedEdges().removeClass('hover-connected');
            
            // Hide tooltip
            this.hideTooltip();
        });
    },
    
    highlightConceptNodes(conceptId) {
        // Find nodes that contain this concept
        const nodes = this.cy.nodes().filter(node => {
            const concepts = node.data('concepts') || [];
            return concepts.includes(conceptId);
        });
        
        // Add pulse effect
        nodes.addClass('pulse');
    },
    
    clearConceptHighlight() {
        this.cy.nodes().removeClass('pulse');
    },
    
    pulseNode(nodeId) {
        const node = this.cy.getElementById(nodeId);
        if (node) {
            node.addClass('pulse');
        }
    },
    
    clearPulse() {
        this.cy.nodes().removeClass('pulse');
    },
    
    centerOnNode(nodeId) {
        const node = this.cy.getElementById(nodeId);
        if (node) {
            this.cy.animate({
                center: { eles: node },
                zoom: 2,
                duration: 500
            });
        }
    },
    
    // Tooltip
    createTooltip() {
        const tooltip = document.createElement('div');
        tooltip.className = 'node-tooltip';
        tooltip.style.display = 'none';
        document.body.appendChild(tooltip);
        this.tooltip = tooltip;
    },
    
    showTooltip(node, position) {
        const text = node.data('text') || node.data('label') || node.id();
        const truncated = this.truncateText(text, 100);
        
        this.tooltip.textContent = truncated;
        this.tooltip.style.display = 'block';
        
        // Position near cursor
        const offset = 10;
        this.tooltip.style.left = position.x + offset + 'px';
        this.tooltip.style.top = position.y + offset + 'px';
    },
    
    hideTooltip() {
        if (this.tooltip) {
            this.tooltip.style.display = 'none';
        }
    },
    
    // Info Popup
    createInfoPopup() {
        // Create info button
        const button = document.createElement('button');
        button.className = 'info-button';
        button.innerHTML = 'ℹ️';
        button.title = 'Информация о графе';
        document.body.appendChild(button);
        
        // Create popup
        const popup = document.createElement('div');
        popup.className = 'info-popup';
        popup.style.display = 'none';
        popup.innerHTML = `
            <button class="close-button">×</button>
            <h3>Статистика графа</h3>
            <div class="stats-content">
                <!-- Will be populated dynamically -->
            </div>
        `;
        document.body.appendChild(popup);
        this.infoPopup = popup;
        
        // Event listeners
        button.addEventListener('click', () => this.toggleInfoPopup());
        popup.querySelector('.close-button').addEventListener('click', () => this.hideInfoPopup());
    },
    
    toggleInfoPopup() {
        this.state.infoPanelOpen = !this.state.infoPanelOpen;
        if (this.state.infoPanelOpen) {
            this.showGraphStats();
        } else {
            this.hideInfoPopup();
        }
    },
    
    showGraphStats() {
        // Collect statistics
        const nodeTypes = this.getNodeTypeCounts();
        const totalNodes = this.cy.nodes().length;
        const totalEdges = this.cy.edges().length;
        
        // Count components
        const components = this.countComponents();
        
        // Count clusters if available
        const clusters = new Set();
        this.cy.nodes().forEach(node => {
            const clusterId = node.data('cluster_id');
            if (clusterId !== undefined) {
                clusters.add(clusterId);
            }
        });
        
        // Generate HTML
        const html = `
            <div class="stat-section">
                <strong>Узлов:</strong> ${totalNodes}<br>
                <ul>
                    <li>Chunks: ${nodeTypes['Chunk'] || 0}</li>
                    <li>Concepts: ${nodeTypes['Concept'] || 0}</li>
                    <li>Assessments: ${nodeTypes['Assessment'] || 0}</li>
                </ul>
            </div>
            <div class="stat-section">
                <strong>Рёбер:</strong> ${totalEdges}
            </div>
            <div class="stat-section">
                <strong>Компонент связности:</strong> ${components}
            </div>
            ${clusters.size > 0 ? `
            <div class="stat-section">
                <strong>Кластеров:</strong> ${clusters.size}
            </div>
            ` : ''}
            <div class="stat-section legend">
                <h4>Легенда</h4>
                <div class="legend-item">
                    <span class="legend-shape chunk">⬢</span> Chunk
                </div>
                <div class="legend-item">
                    <span class="legend-shape concept">★</span> Concept
                </div>
                <div class="legend-item">
                    <span class="legend-shape assessment">▬</span> Assessment
                </div>
            </div>
        `;
        
        this.infoPopup.querySelector('.stats-content').innerHTML = html;
        this.infoPopup.style.display = 'block';
    },
    
    hideInfoPopup() {
        this.state.infoPanelOpen = false;
        this.infoPopup.style.display = 'none';
    },
    
    // Keyboard Handlers
    setupKeyboardHandlers() {
        document.addEventListener('keydown', (e) => {
            // Skip if user is typing in an input
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                return;
            }
            
            if (e.key === 'Escape') {
                // Close any open panels
                if (this.state.infoPanelOpen) {
                    this.hideInfoPopup();
                } else if (this.state.sidePanelOpen) {
                    this.toggleSidePanel();
                }
            } else if (e.key === 'i' && !e.ctrlKey && !e.metaKey) {
                // Toggle info popup
                this.toggleInfoPopup();
            } else if (e.key === 'd' && !e.ctrlKey && !e.metaKey) {
                // Toggle dictionary panel
                this.toggleSidePanel();
            }
        });
    },
    
    // Helper Methods
    updateCounters() {
        const visibleNodes = this.cy.nodes(':visible').length;
        const totalNodes = this.cy.nodes().length;
        const visibleEdges = this.cy.edges(':visible').length;
        const totalEdges = this.cy.edges().length;
        
        document.getElementById('visible-nodes').textContent = visibleNodes;
        document.getElementById('total-nodes').textContent = totalNodes;
        document.getElementById('visible-edges').textContent = visibleEdges;
        document.getElementById('total-edges').textContent = totalEdges;
    },
    
    getNodeTypeCounts() {
        const counts = {};
        this.cy.nodes().forEach(node => {
            const type = node.data('type');
            counts[type] = (counts[type] || 0) + 1;
        });
        return counts;
    },
    
    countComponents() {
        // Simple BFS to count connected components
        const visited = new Set();
        let components = 0;
        
        this.cy.nodes().forEach(node => {
            if (!visited.has(node.id())) {
                components++;
                // BFS from this node
                const queue = [node];
                while (queue.length > 0) {
                    const current = queue.shift();
                    if (visited.has(current.id())) continue;
                    visited.add(current.id());
                    
                    // Add connected nodes
                    current.neighborhood('node').forEach(neighbor => {
                        if (!visited.has(neighbor.id())) {
                            queue.push(neighbor);
                        }
                    });
                }
            }
        });
        
        return components;
    },
    
    truncateText(text, maxLength) {
        if (!text) return '';
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    },
    
    getCySelector(conceptId) {
        // Return Cytoscape selector for nodes with this concept
        return `node[concepts *= "${conceptId}"]`;
    }
};

// Auto-initialize when DOM ready
document.addEventListener('DOMContentLoaded', () => {
    // Wait for graph initialization - check periodically
    let initAttempts = 0;
    const maxAttempts = 20;
    
    const tryInit = setInterval(() => {
        initAttempts++;
        
        // Check if cy is ready and is actual Cytoscape instance
        if (window.cy && window.cy.nodes && typeof window.cy.nodes === 'function') {
            // Verify it's working
            try {
                const nodeCount = window.cy.nodes().length;
                console.log(`[UIControls] Graph ready with ${nodeCount} nodes, initializing...`);
                
                UIControls.init(
                    window.cy,
                    window.graphData,
                    window.conceptData,
                    window.uiConfig || {}
                );
                
                clearInterval(tryInit);
            } catch (e) {
                console.warn('[UIControls] Graph not fully ready:', e);
            }
        } else if (initAttempts >= maxAttempts) {
            console.error('[UIControls] Failed to initialize after', maxAttempts, 'attempts');
            clearInterval(tryInit);
        } else {
            console.log(`[UIControls] Waiting for graph initialization... (attempt ${initAttempts})`);
        }
    }, 500); // Check every 500ms
});

// Export for debugging
window.UIControls = UIControls;
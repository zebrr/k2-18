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
    nodePopup: null,
    conceptPopup: null,
    
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
        infoPanelOpen: false,
        nodePopupOpen: false,
        conceptPopupOpen: false,
        currentPopupNode: null,
        currentPopupConcept: null
    },
    
    // Initialization
    init(cy, graphCore, appContainer, statsContainer, graphData, conceptData, config) {
        this.cy = cy;
        this.graphCore = graphCore;
        this.appContainer = appContainer;
        this.statsContainer = statsContainer;
        this.graphData = graphData;
        this.conceptData = conceptData || { concepts: [] };
        this.config = config;
        
        // Create UI components
        this.createTopPanel();
        this.createSidePanel();
        this.setupHoverEffects();
        this.createInfoPopup();
        this.createNodePopup();
        this.createConceptPopup();
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
        
        // Add hover and click listeners
        listElement.querySelectorAll('.concept-item').forEach(item => {
            item.addEventListener('mouseenter', () => {
                const conceptId = item.dataset.conceptId;
                this.highlightConceptNodes(conceptId);
            });
            
            item.addEventListener('mouseleave', () => {
                this.clearConceptHighlight();
            });
            
            item.addEventListener('click', () => {
                const conceptId = item.dataset.conceptId;
                const concept = this.findConceptById(conceptId);
                if (concept) {
                    this.showConceptPopup(concept);
                }
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
            
            // Make node red (same as pulse effect)
            node.addClass('hover-highlight');
            
            // Highlight connected edges
            const edges = node.connectedEdges();
            edges.addClass('hover-connected');
            console.log(`Hover: highlighting ${edges.length} edges for node ${node.id()}`);
            
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
            node.removeClass('hover-highlight');
            node.connectedEdges().removeClass('hover-connected');
            
            // Hide tooltip
            this.hideTooltip();
        });
        
        // Click on graph nodes
        this.cy.on('click', 'node', (evt) => {
            evt.stopPropagation();
            const node = evt.target;
            this.showNodePopup(node);
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
            <div class="stats-grid">
                <div class="stat-column">
                    <div class="stat-row">
                        <span class="stat-label">Узлов:</span>
                        <span class="stat-value">${totalNodes}</span>
                    </div>
                    <div class="stat-row-detail">
                        <span class="stat-label-sub">Chunks:</span>
                        <span class="stat-value-sub">${nodeTypes['Chunk'] || 0}</span>
                    </div>
                    <div class="stat-row-detail">
                        <span class="stat-label-sub">Concepts:</span>
                        <span class="stat-value-sub">${nodeTypes['Concept'] || 0}</span>
                    </div>
                    <div class="stat-row-detail">
                        <span class="stat-label-sub">Assessments:</span>
                        <span class="stat-value-sub">${nodeTypes['Assessment'] || 0}</span>
                    </div>
                </div>
                <div class="stat-column">
                    <div class="stat-row">
                        <span class="stat-label">Рёбер:</span>
                        <span class="stat-value">${totalEdges}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Компонент:</span>
                        <span class="stat-value">${components}</span>
                    </div>
                    ${clusters.size > 0 ? `
                    <div class="stat-row">
                        <span class="stat-label">Кластеров:</span>
                        <span class="stat-value">${clusters.size}</span>
                    </div>
                    ` : ''}
                </div>
            </div>
            
            <div class="legend-section">
                <h4>Легенда визуализации</h4>
                
                <div class="legend-subsection">
                    <h5>Узлы</h5>
                    <div class="legend-nodes">
                        <div class="legend-item">
                            <span class="legend-shape chunk">⬢</span>
                            <span>Chunk (учебный блок)</span>
                        </div>
                        <div class="legend-item">
                            <span class="legend-shape concept">★</span>
                            <span>Concept (концепт)</span>
                        </div>
                        <div class="legend-item">
                            <span class="legend-shape assessment">▬</span>
                            <span>Assessment (тест)</span>
                        </div>
                    </div>
                    <div class="encoding-note">
                        📏 Размер узла = важность (PageRank)<br>
                        👁 Прозрачность = сложность (1-5)
                    </div>
                </div>
                
                <div class="legend-subsection">
                    <h5>Связи между узлами</h5>
                    <div class="legend-edges">
                        <div class="edge-group">
                            <div class="edge-group-title">Сильные (4px)</div>
                            <div class="legend-item">
                                <svg class="edge-svg" width="30" height="10">
                                    <line x1="0" y1="5" x2="30" y2="5" stroke="#e74c3c" stroke-width="3"/>
                                </svg>
                                <span>PREREQUISITE</span>
                            </div>
                            <div class="legend-item">
                                <svg class="edge-svg" width="30" height="10">
                                    <line x1="0" y1="5" x2="30" y2="5" stroke="#f39c12" stroke-width="3"/>
                                </svg>
                                <span>TESTS</span>
                            </div>
                        </div>
                        
                        <div class="edge-group">
                            <div class="edge-group-title">Средние (2.5px)</div>
                            <div class="legend-item">
                                <svg class="edge-svg" width="30" height="10">
                                    <line x1="0" y1="5" x2="30" y2="5" stroke="#3498db" stroke-width="2" stroke-dasharray="5,2"/>
                                </svg>
                                <span>ELABORATES</span>
                            </div>
                            <div class="legend-item">
                                <svg class="edge-svg" width="30" height="10">
                                    <line x1="0" y1="5" x2="30" y2="5" stroke="#9b59b6" stroke-width="2" stroke-dasharray="2,3"/>
                                </svg>
                                <span>EXAMPLE_OF</span>
                            </div>
                            <div class="legend-item">
                                <svg class="edge-svg" width="30" height="10">
                                    <line x1="0" y1="5" x2="30" y2="5" stroke="#95a5a6" stroke-width="2"/>
                                </svg>
                                <span>PARALLEL</span>
                            </div>
                            <div class="legend-item">
                                <svg class="edge-svg" width="30" height="10">
                                    <line x1="0" y1="5" x2="30" y2="5" stroke="#27ae60" stroke-width="2" stroke-dasharray="4,2"/>
                                </svg>
                                <span>REVISION_OF</span>
                            </div>
                        </div>
                        
                        <div class="edge-group">
                            <div class="edge-group-title">Слабые (1px)</div>
                            <div class="legend-item">
                                <svg class="edge-svg" width="30" height="10">
                                    <line x1="0" y1="5" x2="30" y2="5" stroke="#5dade2" stroke-width="1" stroke-dasharray="2,4" opacity="0.6"/>
                                </svg>
                                <span>HINT_FORWARD</span>
                            </div>
                            <div class="legend-item">
                                <svg class="edge-svg" width="30" height="10">
                                    <line x1="0" y1="5" x2="30" y2="5" stroke="#ec7063" stroke-width="1" stroke-dasharray="2,4" opacity="0.6"/>
                                </svg>
                                <span>REFER_BACK</span>
                            </div>
                            <div class="legend-item">
                                <svg class="edge-svg" width="30" height="10">
                                    <line x1="0" y1="5" x2="30" y2="5" stroke="#bdc3c7" stroke-width="1" stroke-dasharray="3,3" opacity="0.5"/>
                                </svg>
                                <span>MENTIONS</span>
                            </div>
                        </div>
                    </div>
                    <div class="encoding-note">
                        ⚡ Межкластерные связи отображаются толще
                    </div>
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
                // Priority order for closing
                if (this.state.nodePopupOpen) {
                    this.hideNodePopup();
                } else if (this.state.conceptPopupOpen) {
                    this.hideConceptPopup();
                } else if (this.state.infoPanelOpen) {
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
    },
    
    // Node Popup Methods
    createNodePopup() {
        const backdrop = document.createElement('div');
        backdrop.className = 'node-popup-backdrop';
        backdrop.style.display = 'none';
        
        const popup = document.createElement('div');
        popup.className = 'node-popup';
        
        backdrop.appendChild(popup);
        document.body.appendChild(backdrop);
        this.nodePopup = backdrop;
        
        // Click on backdrop to close
        backdrop.addEventListener('click', (e) => {
            if (e.target === backdrop) {
                this.hideNodePopup();
            }
        });
        
        console.log('[UIControls] Node popup created');
    },
    
    showNodePopup(node) {
        // Close other popups
        this.hideConceptPopup();
        this.hideInfoPopup();
        
        // Get node data
        const nodeData = node.data();
        const nodeId = node.id();
        
        // Format metrics
        const pagerank = this.formatMetricValue(nodeData.pagerank, 'pagerank');
        const betweenness = this.formatMetricValue(nodeData.betweenness_centrality, 'betweenness');
        const learningEffort = this.formatMetricValue(nodeData.learning_effort, 'effort');
        
        // Get edges
        const edges = this.getNodeEdges(node);
        const showAllButton = edges.length > 3 ? `
            <button class="show-all-edges" onclick="UIControls.toggleAllEdges(this)">
                показать все ${edges.length - 3}...
            </button>` : '';
        
        // Generate HTML
        const html = `
            <button class="popup-close" onclick="UIControls.hideNodePopup()">×</button>
            <div class="popup-header">
                <span class="node-type node-type-${nodeData.type}">${nodeData.type}</span>
                <span class="node-id" title="${nodeId}">${nodeId}</span>
                <span class="node-difficulty">${this.renderDifficulty(nodeData.difficulty)}</span>
            </div>
            <div class="popup-content">
                <div class="node-text-section">
                    <h4>Содержание:</h4>
                    <div class="node-text-scroll">
                        ${nodeData.text || 'Текст недоступен'}
                    </div>
                </div>
                ${nodeData.definition ? `
                <div class="node-definition-section">
                    <h4>Источник:</h4>
                    <div class="node-definition-scroll">
                        ${nodeData.definition}
                    </div>
                </div>
                ` : ''}
                <div class="metrics-section">
                    <h4>Образовательные метрики:</h4>
                    <div class="metric-row">
                        <span class="metric-label">Важность (PageRank):</span>
                        <span class="metric-value">${pagerank}</span>
                        <span class="metric-info" onclick="UIControls.toggleMetricTooltip(this, 'pagerank')">ℹ️
                            <span class="metric-tooltip">Определяет значимость узла в структуре знаний. Чем выше значение, тем больше важных концептов ссылается на этот материал. Изучение узлов с высокой важностью критично для полного понимания темы.</span>
                        </span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Мост (Betweenness):</span>
                        <span class="metric-value">${betweenness}</span>
                        <span class="metric-info" onclick="UIControls.toggleMetricTooltip(this, 'betweenness')">ℹ️
                            <span class="metric-tooltip">Показывает, насколько узел является мостом между разными частями учебного материала. Высокое значение означает, что узел соединяет различные темы. Пропуск такого материала может нарушить целостность понимания.</span>
                        </span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Учебная нагрузка (Learning Effort):</span>
                        <span class="metric-value">${learningEffort}</span>
                        <span class="metric-info" onclick="UIControls.toggleMetricTooltip(this, 'effort')">ℹ️
                            <span class="metric-tooltip">Суммарная сложность изучения с учётом всех предварительных тем. Отражает общий объём усилий, необходимых для освоения этого узла и всех его зависимостей.</span>
                        </span>
                    </div>
                </div>
                <div class="connectivity-section">
                    <span>Связность: ${nodeData.degree_in || 0} входящих, ${nodeData.degree_out || 0} исходящих</span>
                </div>
                <div class="edges-section">
                    <h4>Связи (${edges.length}):</h4>
                    <div class="edges-header">
                        <span class="edges-header-direction"></span>
                        <span class="edges-header-type">Тип</span>
                        <span class="edges-header-target">Узел</span>
                        <span class="edges-header-difficulty">Сложность</span>
                        <span class="edges-header-weight">Вес</span>
                    </div>
                    <div class="edges-list" data-expanded="false">
                        ${edges.slice(0, 3).map(edge => this.renderEdgeItem(edge)).join('')}
                        <div class="edges-expanded" style="display: none;">
                            ${edges.slice(3).map(edge => this.renderEdgeItem(edge)).join('')}
                        </div>
                    </div>
                    ${showAllButton}
                </div>
            </div>
        `;
        
        this.nodePopup.querySelector('.node-popup').innerHTML = html;
        this.nodePopup.style.display = 'flex';
        this.state.nodePopupOpen = true;
        this.state.currentPopupNode = nodeId;
        
        console.log(`[UIControls] Node popup shown for: ${nodeId}`);
    },
    
    hideNodePopup() {
        if (this.nodePopup) {
            this.nodePopup.style.display = 'none';
            this.state.nodePopupOpen = false;
            this.state.currentPopupNode = null;
        }
    },
    
    updateNodePopup(nodeId) {
        const node = this.cy.getElementById(nodeId);
        if (node && node.length > 0) {
            this.showNodePopup(node);
            console.log(`[UIControls] Popup updated to node: ${nodeId}`);
        }
    },
    
    toggleAllEdges(button) {
        const edgesList = button.parentElement.querySelector('.edges-list');
        const expandedDiv = edgesList.querySelector('.edges-expanded');
        const isExpanded = edgesList.dataset.expanded === 'true';
        
        if (isExpanded) {
            expandedDiv.style.display = 'none';
            button.textContent = `показать все ${expandedDiv.children.length}...`;
            edgesList.dataset.expanded = 'false';
        } else {
            expandedDiv.style.display = 'block';
            button.textContent = 'скрыть';
            edgesList.dataset.expanded = 'true';
        }
    },
    
    renderEdgeItem(edge) {
        const direction = edge.direction === 'incoming' ? '←' : '→';
        const targetId = edge.targetId;
        const targetLabel = this.truncateText(edge.targetLabel, 30);
        const weight = edge.weight ? `(${edge.weight.toFixed(1)})` : '';
        
        // Render small difficulty circles
        const difficulty = Math.min(5, Math.max(1, parseInt(edge.targetDifficulty) || 1));
        let difficultyCircles = '';
        for (let i = 1; i <= 5; i++) {
            const filled = i <= difficulty;
            const color = i <= 2 ? '#2ecc71' : i <= 3 ? '#f39c12' : '#e74c3c';
            difficultyCircles += `<span class="edge-difficulty-circle ${filled ? 'filled' : ''}" style="${filled ? `background: ${color};` : ''}"></span>`;
        }
        
        return `
            <div class="edge-item">
                <span class="edge-direction">${direction}</span>
                <span class="edge-type edge-type-${edge.type}">${edge.type}</span>
                <span class="edge-target clickable" onclick="UIControls.updateNodePopup('${targetId}')">
                    ${targetLabel}
                </span>
                <span class="edge-difficulty">${difficultyCircles}</span>
                <span class="edge-weight">${weight}</span>
            </div>
        `;
    },
    
    getNodeEdges(node) {
        const edges = [];
        
        // Incoming edges
        node.incomers('edge').forEach(edge => {
            const source = edge.source();
            edges.push({
                direction: 'incoming',
                type: edge.data('type') || 'UNKNOWN',
                targetId: source.id(),
                targetLabel: source.data('text') || source.id(),
                targetDifficulty: source.data('difficulty') || 1,
                weight: edge.data('weight')
            });
        });
        
        // Outgoing edges
        node.outgoers('edge').forEach(edge => {
            const target = edge.target();
            edges.push({
                direction: 'outgoing',
                type: edge.data('type') || 'UNKNOWN',
                targetId: target.id(),
                targetLabel: target.data('text') || target.id(),
                targetDifficulty: target.data('difficulty') || 1,
                weight: edge.data('weight')
            });
        });
        
        // Sort by type importance
        const typeOrder = ['PREREQUISITE', 'TESTS', 'ELABORATES', 'EXAMPLE_OF', 'PARALLEL', 'REVISION_OF', 'HINT_FORWARD', 'REFER_BACK', 'MENTIONS'];
        edges.sort((a, b) => {
            const aIndex = typeOrder.indexOf(a.type);
            const bIndex = typeOrder.indexOf(b.type);
            return (aIndex === -1 ? 999 : aIndex) - (bIndex === -1 ? 999 : bIndex);
        });
        
        return edges;
    },
    
    // Concept Popup Methods
    createConceptPopup() {
        const backdrop = document.createElement('div');
        backdrop.className = 'concept-popup-backdrop';
        backdrop.style.display = 'none';
        
        const popup = document.createElement('div');
        popup.className = 'concept-popup';
        
        backdrop.appendChild(popup);
        document.body.appendChild(backdrop);
        this.conceptPopup = backdrop;
        
        // Click on backdrop to close
        backdrop.addEventListener('click', (e) => {
            if (e.target === backdrop) {
                this.hideConceptPopup();
            }
        });
        
        console.log('[UIControls] Concept popup created');
    },
    
    showConceptPopup(concept) {
        // Close other popups
        this.hideNodePopup();
        this.hideInfoPopup();
        
        // Get concept data
        const conceptId = concept.concept_id || concept.id;
        const primaryTerm = concept.term?.primary || concept.primary?.term || 'Unknown';
        const definition = concept.definition || 'Определение недоступно';
        const aliases = concept.term?.aliases || concept.aliases || [];
        
        // Count mentions
        let mentionCount = 0;
        if (this.conceptData._meta?.mention_index?.[conceptId]) {
            mentionCount = this.conceptData._meta.mention_index[conceptId].count || 0;
        }
        
        // Generate HTML
        const html = `
            <button class="popup-close" onclick="UIControls.hideConceptPopup()">×</button>
            <div class="popup-header">
                <h3>Концепт: ${primaryTerm}</h3>
            </div>
            <div class="popup-content">
                <div class="definition-section">
                    <h4>Определение:</h4>
                    <p>${definition}</p>
                </div>
                ${aliases.length > 0 ? `
                <div class="aliases-section">
                    <h4>Синонимы:</h4>
                    <p>${aliases.join(', ')}</p>
                </div>
                ` : ''}
                <div class="mentions-section">
                    <p>Упоминается в <strong>${mentionCount}</strong> узлах</p>
                </div>
            </div>
        `;
        
        this.conceptPopup.querySelector('.concept-popup').innerHTML = html;
        this.conceptPopup.style.display = 'flex';
        this.state.conceptPopupOpen = true;
        this.state.currentPopupConcept = conceptId;
        
        console.log(`[UIControls] Concept popup shown for: ${conceptId}`);
    },
    
    hideConceptPopup() {
        if (this.conceptPopup) {
            this.conceptPopup.style.display = 'none';
            this.state.conceptPopupOpen = false;
            this.state.currentPopupConcept = null;
        }
    },
    
    findConceptById(conceptId) {
        if (!this.conceptData || !this.conceptData.concepts) {
            return null;
        }
        
        return this.conceptData.concepts.find(c => 
            (c.concept_id || c.id) === conceptId
        );
    },
    
    formatMetricValue(value, type) {
        if (value === undefined || value === null) {
            return 'N/A';
        }
        
        switch (type) {
            case 'pagerank':
            case 'betweenness':
                return value.toFixed(3);
            case 'effort':
                return Math.round(value).toString();
            default:
                return value.toString();
        }
    },
    
    renderDifficulty(difficulty) {
        // Parse difficulty, ensuring it's a number between 1-5
        const level = Math.min(5, Math.max(1, parseInt(difficulty) || 1));
        let circles = '';
        for (let i = 1; i <= 5; i++) {
            const filled = i <= level;
            // Traffic light colors: green (1-2), yellow (3), red (4-5)
            const color = i <= 2 ? '#2ecc71' : i <= 3 ? '#f39c12' : '#e74c3c';
            circles += `<span class="difficulty-circle ${filled ? 'filled' : ''}" style="${filled ? `background: ${color};` : ''}"></span>`;
        }
        return `<span class="difficulty-label">Сложность:</span> <span class="difficulty-circles">${circles}</span>`;
    },
    
    toggleMetricTooltip(element, metricType) {
        // Close all other tooltips first
        document.querySelectorAll('.metric-tooltip.active').forEach(tooltip => {
            if (tooltip !== element.querySelector('.metric-tooltip')) {
                tooltip.classList.remove('active');
            }
        });
        
        // Toggle current tooltip
        const tooltip = element.querySelector('.metric-tooltip');
        if (tooltip) {
            tooltip.classList.toggle('active');
        }
        
        // Close tooltip when clicking elsewhere
        if (!this.tooltipClickHandler) {
            this.tooltipClickHandler = (e) => {
                if (!e.target.closest('.metric-info')) {
                    document.querySelectorAll('.metric-tooltip.active').forEach(t => {
                        t.classList.remove('active');
                    });
                }
            };
            document.addEventListener('click', this.tooltipClickHandler);
        }
    },
    
    closeAllPopups() {
        this.hideNodePopup();
        this.hideConceptPopup();
        this.hideInfoPopup();
    }
};

document.addEventListener("k2-graph-ready", (e) => {
  const { cy, graphCore } = e.detail || {};
  UIControls.init(
    cy,
    graphCore,
    document.getElementById("app-container"),
    document.getElementById("stats-container"),
    window.graphData,
    window.conceptData,
    window.uiConfig || {}
  );
  const statsEl = document.getElementById("stats-container");
  if (statsEl) statsEl.style.visibility = "visible";
  console.log("✓ UIControls initialized via k2-graph-ready");
});


// Export for debugging
window.UIControls = UIControls;
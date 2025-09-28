// Node exploration functionality
const NodeExplorer = {
    state: {
        activeNodeId: null,
        nodeCards: [],
        allNodes: [],
        conceptData: {},
        graphData: {},
        viewMode: 'formatted' // or 'json'
    },
    
    // Cluster colors from main visualization (36 colors for better distinction)
    clusterColors: [
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
    ],
    
    init(graphData, conceptData) {
        console.log('Node explorer initializing...');
        this.allNodes = graphData.nodes || [];
        this.graphData = graphData || { nodes: [], edges: [] };
        this.conceptData = conceptData || { concepts: [] };

        // Listen for filter changes
        document.addEventListener('filter-changed', (e) => {
            this.renderNodeList(e.detail.filteredNodes);
        });

        // Listen for node selection
        document.addEventListener('node-selected', (e) => {
            if (e.detail.nodeId) {
                this.renderActiveNode(e.detail.node);
            } else {
                this.clearActiveNode();
            }
        });

        // Close tooltips when clicking elsewhere
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.metric-info')) {
                document.querySelectorAll('.metric-tooltip').forEach(t => {
                    t.style.display = 'none';
                });
            }
        });

        console.log('Node explorer initialized');
    },
    
    renderNodeList(filteredNodes) {
        const nodeListContainer = document.getElementById('node-list');
        if (!nodeListContainer) {
            console.error('Node list container not found');
            return;
        }
        
        // Sort nodes according to requirements
        const sortedNodes = this.sortNodes(filteredNodes);
        
        // Clear existing content
        nodeListContainer.innerHTML = '';
        
        // Create node cards
        sortedNodes.forEach(node => {
            const card = this.createNodeCard(node);
            nodeListContainer.appendChild(card);
        });
        
        // Update counter is handled by SearchFilter
        console.log(`Rendered ${sortedNodes.length} node cards`);
    },
    
    sortNodes(nodes) {
        return nodes.sort((a, b) => {
            // First sort by cluster_id
            const clusterA = a.cluster_id || 0;
            const clusterB = b.cluster_id || 0;
            
            if (clusterA !== clusterB) {
                return clusterA - clusterB;
            }
            
            // Within same cluster
            const typeA = a.type || 'Unknown';
            const typeB = b.type || 'Unknown';

            // Concepts go to the beginning
            if (typeA === 'Concept' && typeB !== 'Concept') return -1;
            if (typeA !== 'Concept' && typeB === 'Concept') return 1;
            
            // Both are Concepts - sort alphabetically by text
            if (typeA === 'Concept' && typeB === 'Concept') {
                const textA = (a.text || '').toLowerCase();
                const textB = (b.text || '').toLowerCase();
                return textA.localeCompare(textB);
            }
            
            // Both are Chunks or Assessments - sort by position token
            const posA = this.extractPosition(a);
            const posB = this.extractPosition(b);
            
            return posA - posB;
        });
    },
    
    extractPosition(node) {
        const id = node.id || '';
        
        if (node.type === 'Chunk') {
            // Extract number after :c:
            const match = id.match(/:c:(\d+)/);
            return match ? parseInt(match[1], 10) : 0;
        }
        
        if (node.type === 'Assessment') {
            // Extract number after :q:
            const match = id.match(/:q:(\d+)/);
            return match ? parseInt(match[1], 10) : 0;
        }
        
        return 0;
    },
    
    getNodeTypeLabel(type) {
        switch(type) {
            case 'Chunk': return 'CHUNK';
            case 'Concept': return 'CONCEPT';
            case 'Assessment': return 'ASMNT';
            default: return 'UNKNOWN';
        }
    },

    createNodeCard(node) {
        const card = document.createElement('div');
        card.className = 'node-card';

        // Add active class if this is the active node
        if (node.id === this.state.activeNodeId) {
            card.classList.add('active');
        }

        // Apply cluster color styling
        const clusterId = node.cluster_id || 0;
        const clusterColor = this.clusterColors[clusterId % this.clusterColors.length];

        // Create gradient background
        card.style.background = `linear-gradient(135deg,
            ${this.hexToRgba(clusterColor, 0.1)} 0%,
            ${this.hexToRgba(clusterColor, 0.05)} 100%)`;
        card.style.borderLeftColor = clusterColor;

        // Create card header
        const header = document.createElement('div');
        header.className = 'node-card-header';

        // Type badge with label
        const typeBadge = document.createElement('span');
        const nodeType = node.type || 'Unknown';
        const label = this.getNodeTypeLabel(nodeType);

        // Add appropriate class based on type
        let badgeClass = 'node-type-badge ';
        switch(nodeType) {
            case 'Chunk': badgeClass += 'badge-chunk'; break;
            case 'Concept': badgeClass += 'badge-concept'; break;
            case 'Assessment': badgeClass += 'badge-assessment'; break;
            default: badgeClass += 'badge-unknown';
        }
        typeBadge.className = badgeClass;
        typeBadge.textContent = label;
        
        // Shortened ID
        const idShort = document.createElement('span');
        idShort.className = 'node-id-short';
        idShort.textContent = this.shortenId(node);
        
        header.appendChild(typeBadge);
        header.appendChild(idShort);
        
        // Text preview
        const textPreview = document.createElement('div');
        textPreview.className = 'node-text-preview';
        const text = node.text || node.definition || '';
        textPreview.textContent = text.length > 300 ? text.substring(0, 300) + '...' : text;

        card.appendChild(header);
        card.appendChild(textPreview);

        // Add cluster number badge at bottom right
        const clusterBadge = document.createElement('div');
        clusterBadge.className = 'cluster-badge';
        clusterBadge.style.cssText = `
            position: absolute;
            bottom: 8px;
            right: 8px;
            background: ${clusterColor};
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            opacity: 0.9;
        `;
        clusterBadge.textContent = `C${clusterId}`;
        card.appendChild(clusterBadge);

        // Make card position relative for absolute positioning to work
        card.style.position = 'relative';
        
        // Add click handler
        card.addEventListener('click', () => {
            this.selectNode(node.id);
        });
        
        // Add hover effect enhancement
        card.addEventListener('mouseenter', () => {
            card.style.background = `linear-gradient(135deg,
                ${this.hexToRgba(clusterColor, 0.2)} 0%,
                ${this.hexToRgba(clusterColor, 0.15)} 100%)`;
        });

        card.addEventListener('mouseleave', () => {
            if (!card.classList.contains('active')) {
                card.style.background = `linear-gradient(135deg,
                    ${this.hexToRgba(clusterColor, 0.1)} 0%,
                    ${this.hexToRgba(clusterColor, 0.05)} 100%)`;
            }
        });
        
        return card;
    },
    
    shortenId(node) {
        const id = node.id || '';
        const type = node.type || 'Unknown';
        
        if (type === 'Chunk') {
            // Extract position number
            const match = id.match(/:c:(\d+)/);
            return match ? match[1] : id.substring(0, 15);
        }
        
        if (type === 'Assessment') {
            // Extract position number
            const match = id.match(/:q:(\d+)/);
            return match ? match[1] : id.substring(0, 15);
        }
        
        if (type === 'Concept') {
            // Get first 15 chars after :p:
            const match = id.match(/:p:(.+)/);
            if (match) {
                const conceptPart = match[1];
                return conceptPart.length > 15 ? conceptPart.substring(0, 15) + '...' : conceptPart;
            }
            return id.substring(0, 15);
        }
        
        return id.substring(0, 15);
    },
    
    selectNode(nodeId) {
        console.log(`Node selected: ${nodeId}`);

        // Update active state
        const previousActive = this.state.activeNodeId;

        // If clicking the same node, deselect it
        if (this.state.activeNodeId === nodeId) {
            this.state.activeNodeId = null;
            nodeId = null;
        } else {
            this.state.activeNodeId = nodeId;
        }

        // Update visual state of cards
        const cards = document.querySelectorAll('.node-card');
        cards.forEach(card => {
            card.classList.remove('active');
        });

        // Find the selected node
        let selectedNode = null;

        // Find and activate the selected card if nodeId is not null
        if (nodeId) {
            selectedNode = this.allNodes.find(n => n.id === nodeId);
            if (selectedNode) {
                const cards = document.querySelectorAll('.node-card');
                cards.forEach(card => {
                    const cardHeader = card.querySelector('.node-card-header');
                    if (cardHeader) {
                        const idText = cardHeader.querySelector('.node-id-short')?.textContent;
                        const typeText = cardHeader.querySelector('.node-type-badge')?.textContent;

                        // Check if this card represents the selected node
                        const expectedTypeText = this.getNodeTypeLabel(selectedNode.type);

                        if (typeText === expectedTypeText) {
                            const shortId = this.shortenId(selectedNode);
                            if (idText === shortId) {
                                card.classList.add('active');

                                // Update background for active state
                                const clusterId = selectedNode.cluster_id || 0;
                                const clusterColor = this.clusterColors[clusterId % this.clusterColors.length];
                                card.style.background = `linear-gradient(135deg,
                                    ${this.hexToRgba(clusterColor, 0.2)} 0%,
                                    ${this.hexToRgba(clusterColor, 0.15)} 100%)`;
                            }
                        }
                    }
                });
            }
        }

        // Dispatch custom event
        const event = new CustomEvent('node-selected', {
            detail: {
                nodeId: nodeId,
                node: selectedNode,
                previousNodeId: previousActive
            }
        });
        document.dispatchEvent(event);
    },
    
    updateCounter(shown, total) {
        // This is now handled by SearchFilter
        console.log(`Counter update: ${shown} of ${total}`);
    },
    
    // Helper function to convert hex to rgba
    hexToRgba(hex, alpha) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    },

    // Metric information from spec
    metricsInfo: {
        'degree_in': {
            label: 'Входящая степень',
            tooltip: 'degree_in: Количество узлов, от которых зависит данный материал. Высокое значение указывает на тему, требующую много предварительных знаний.'
        },
        'degree_out': {
            label: 'Исходящая степень',
            tooltip: 'degree_out: Количество узлов, которые зависят от данного материала. Высокое значение означает, что это базовая тема, открывающая доступ к новым концепциям.'
        },
        'degree_centrality': {
            label: 'Нормализованная связность',
            tooltip: 'degree_centrality: Доля всех возможных связей узла относительно размера графа. Показывает, насколько узел интегрирован в общую структуру знаний.'
        },
        'pagerank': {
            label: 'Важность',
            tooltip: 'pagerank: Определяет значимость узла в структуре знаний. Чем выше значение, тем больше важных концептов ссылается на этот материал. Изучение узлов с высокой важностью критично для полного понимания темы.'
        },
        'betweenness_centrality': {
            label: 'Мост между частями',
            tooltip: 'betweenness_centrality: Показывает, насколько узел является мостом между разными частями учебного материала. Высокое значение означает, что узел соединяет различные темы. Пропуск такого материала может нарушить целостность понимания.'
        },
        'out-closeness': {
            label: 'Исходящая близость',
            tooltip: 'out-closeness: Насколько быстро из данного узла можно достичь других тем. Высокое значение означает хорошую стартовую точку для изучения.'
        },
        'component_id': {
            label: 'Компонента связности',
            tooltip: 'component_id: Идентификатор группы взаимосвязанных тем. Узлы с одинаковым ID образуют единый связный блок знаний.'
        },
        'prerequisite_depth': {
            label: 'Уровень зависимостей',
            tooltip: 'prerequisite_depth: Глубина в дереве предварительных требований. 0 = базовые концепты, далее по возрастанию сложности.'
        },
        'learning_effort': {
            label: 'Учебная сложность',
            tooltip: 'learning_effort: Суммарная сложность изучения с учётом всех предварительных тем. Отражает общий объём усилий, необходимых для освоения этого узла и всех его зависимостей.'
        },
        'educational_importance': {
            label: 'Образовательная важность',
            tooltip: 'educational_importance: PageRank только по образовательным связям (PREREQUISITE, ELABORATES, TESTS, EXAMPLE_OF). Показывает важность в контексте обучения.'
        },
        'cluster_id': {
            label: 'Кластер',
            tooltip: 'cluster_id: Идентификатор тематического блока. Узлы одного кластера объединены общей темой или подходом.'
        },
        'bridge_score': {
            label: 'Метрика моста',
            tooltip: 'bridge_score: Композитная метрика узла-моста между тематическими блоками. Высокое значение = узел связывает разные области знаний.'
        }
    },

    renderActiveNode(node) {
        // Get container #active-node-content
        const container = document.getElementById('active-node-content');
        if (!container || !node) return;

        // Clear existing content
        container.innerHTML = '';

        // Add view mode toggle button
        const header = document.createElement('div');
        header.className = 'active-node-header';

        const toggleBtn = document.createElement('button');
        toggleBtn.className = 'view-mode-toggle' + (this.state.viewMode === 'json' ? ' active' : '');
        toggleBtn.innerHTML = '{JSON}';
        toggleBtn.title = 'Переключить в режим JSON';
        toggleBtn.onclick = () => this.toggleViewMode();
        header.appendChild(toggleBtn);

        const indicator = document.createElement('div');
        indicator.className = 'active-indicator';
        indicator.textContent = 'Активный узел';
        header.appendChild(indicator);

        container.appendChild(header);

        // Call renderFormattedView or renderJsonView
        if (this.state.viewMode === 'formatted') {
            this.renderFormattedView(node, container);
        } else {
            this.renderJsonView(node, container);
        }
    },

    renderFormattedView(node, container) {
        // Create header with type, ID, difficulty dots
        const nodeHeader = document.createElement('div');
        nodeHeader.className = 'node-header';

        // Type badge
        const typeBadge = document.createElement('span');
        typeBadge.className = `node-type-badge badge-${(node.type || 'unknown').toLowerCase()}`;
        typeBadge.textContent = node.type || 'Unknown';
        nodeHeader.appendChild(typeBadge);

        // ID
        const nodeId = document.createElement('span');
        nodeId.className = 'node-id';
        nodeId.textContent = this.shortenId(node);
        nodeHeader.appendChild(nodeId);

        // Difficulty dots
        if (node.difficulty) {
            const difficulty = document.createElement('div');
            difficulty.className = 'difficulty-dots';

            const label = document.createElement('span');
            label.className = 'difficulty-label';
            label.textContent = 'Сложность:';
            difficulty.appendChild(label);

            const circles = document.createElement('div');
            circles.className = 'difficulty-circles';
            circles.innerHTML = this.formatDifficulty(node.difficulty);
            difficulty.appendChild(circles);

            nodeHeader.appendChild(difficulty);
        }

        container.appendChild(nodeHeader);

        // Content sections
        const content = document.createElement('div');
        content.className = 'node-content';

        // 1. Содержание (node.text)
        this.addContentSection(content, 'Содержание', node.text || '-');

        // 2. Определение (node.definition)
        this.addContentSection(content, 'Определение', node.definition || '-');

        // 3. Связанные концепты (node.concepts)
        this.addConceptsSection(content, node.concepts || []);

        // 4. Образовательные метрики (all 12 with [i] tooltips)
        this.renderMetricsBlock(content, node);

        // 5. Связи table (5 columns)
        this.renderEdgesTable(content, node);

        container.appendChild(content);
    },

    renderJsonView(node, container) {
        // Show raw JSON with syntax highlighting
        const jsonContainer = document.createElement('div');
        jsonContainer.className = 'json-view';

        const pre = document.createElement('pre');
        const code = document.createElement('code');
        code.className = 'language-json';

        // Format JSON with indentation
        const jsonString = JSON.stringify(node, null, 2);

        // Use highlight.js with 'json' language
        if (typeof hljs !== 'undefined') {
            code.innerHTML = hljs.highlight(jsonString, { language: 'json' }).value;
        } else {
            code.textContent = jsonString;
        }

        pre.appendChild(code);
        jsonContainer.appendChild(pre);
        container.appendChild(jsonContainer);
    },

    renderMetricsBlock(container, node) {
        // Create metrics list with all 12 metrics
        const section = document.createElement('div');
        section.className = 'content-section metrics-section';

        const header = document.createElement('h4');
        header.textContent = 'Образовательные метрики';
        section.appendChild(header);

        const metricsList = document.createElement('div');
        metricsList.className = 'metrics-list';

        // Add [i] icons with tooltips
        // Use exact labels and descriptions from Technical_Specification_VIEW_K2-18.md section "Пояснения для метрик"
        const metricKeys = [
            'degree_in', 'degree_out', 'degree_centrality', 'pagerank',
            'betweenness_centrality', 'out-closeness', 'component_id',
            'prerequisite_depth', 'learning_effort', 'educational_importance',
            'cluster_id', 'bridge_score'
        ];

        metricKeys.forEach(key => {
            const metric = this.metricsInfo[key];
            if (!metric) return;

            const metricItem = document.createElement('div');
            metricItem.className = 'metric-item';

            const label = document.createElement('span');
            label.className = 'metric-label';
            label.textContent = metric.label;

            const infoIcon = document.createElement('span');
            infoIcon.className = 'metric-info';
            infoIcon.innerHTML = 'ℹ️';

            const tooltip = document.createElement('span');
            tooltip.className = 'metric-tooltip';
            tooltip.textContent = metric.tooltip;
            tooltip.style.display = 'none';
            infoIcon.appendChild(tooltip);

            // Toggle tooltip on click
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

            const value = document.createElement('span');
            value.className = 'metric-value';
            value.textContent = this.formatMetricValue(node[key]);

            metricItem.appendChild(label);
            metricItem.appendChild(value);
            metricItem.appendChild(infoIcon);

            metricsList.appendChild(metricItem);
        });

        section.appendChild(metricsList);
        container.appendChild(section);
    },

    renderEdgesTable(container, node) {
        // Find all incoming and outgoing edges
        const edges = this.getNodeEdges(node.id);
        const totalEdges = edges.incoming.length + edges.outgoing.length;

        const section = document.createElement('div');
        section.className = 'content-section edges-section';

        const header = document.createElement('h4');
        header.innerHTML = `Связи <span class="edge-counts">(всего: ${totalEdges} | вх: ${edges.incoming.length} | исх: ${edges.outgoing.length})</span>`;
        section.appendChild(header);

        if (totalEdges === 0) {
            const noEdges = document.createElement('div');
            noEdges.className = 'no-edges';
            noEdges.textContent = 'Нет связей';
            section.appendChild(noEdges);
        } else {
            // Create 5-column table
            const table = document.createElement('table');
            table.className = 'edges-table';

            // Table header (merged Type into Link column)
            const thead = document.createElement('thead');
            const headerRow = document.createElement('tr');
            ['', 'Связь / Тип', 'Узел', 'Вес'].forEach((text, index) => {
                const th = document.createElement('th');
                th.textContent = text;
                // Set column widths
                if (index === 0) th.style.width = '30px';  // Direction
                if (index === 1) th.style.width = '150px'; // Link/Type
                if (index === 3) th.style.width = '50px';  // Weight
                headerRow.appendChild(th);
            });
            thead.appendChild(headerRow);
            table.appendChild(thead);

            // Table body
            const tbody = document.createElement('tbody');

            // Add incoming edges
            edges.incoming.forEach(edge => {
                const row = this.createEdgeRow(edge, 'incoming', node.id);
                tbody.appendChild(row);
            });

            // Add outgoing edges
            edges.outgoing.forEach(edge => {
                const row = this.createEdgeRow(edge, 'outgoing', node.id);
                tbody.appendChild(row);
            });

            table.appendChild(tbody);
            section.appendChild(table);
        }

        container.appendChild(section);
    },

    createEdgeRow(edge, direction, currentNodeId) {
        const row = document.createElement('tr');
        row.className = 'edge-row';
        row.dataset.edgeId = `${edge.source}_${edge.target}_${edge.type}`;

        // Direction arrow (← or →)
        const dirCell = document.createElement('td');
        dirCell.className = 'edge-direction';
        dirCell.textContent = direction === 'incoming' ? '←' : '→';
        row.appendChild(dirCell);

        // Find the other node
        const otherNodeId = direction === 'incoming' ? edge.source : edge.target;
        const otherNode = this.allNodes.find(n => n.id === otherNodeId);

        // Merged cell: Edge type + Node type
        const mergedCell = document.createElement('td');
        mergedCell.className = 'edge-type-merged';

        // Edge type on first line
        const edgeTypeDiv = document.createElement('div');
        edgeTypeDiv.className = 'edge-type-name';
        edgeTypeDiv.textContent = edge.type || 'UNKNOWN';
        mergedCell.appendChild(edgeTypeDiv);

        // Node type badge on second line
        if (otherNode) {
            const badge = document.createElement('span');
            badge.className = `node-type-badge badge-${(otherNode.type || 'unknown').toLowerCase()}`;
            const nodeType = otherNode.type || 'Unknown';
            // Uniform uppercase abbreviations
            const displayType = nodeType === 'Assessment' ? 'ASMNT' :
                               nodeType === 'Chunk' ? 'CHUNK' :
                               nodeType === 'Concept' ? 'CONCEPT' :
                               nodeType.toUpperCase();
            badge.textContent = displayType;
            mergedCell.appendChild(badge);
        } else {
            const unknownDiv = document.createElement('div');
            unknownDiv.className = 'node-type-unknown';
            unknownDiv.textContent = 'Unknown';
            mergedCell.appendChild(unknownDiv);
        }
        row.appendChild(mergedCell);

        // Node text + conditions + межкластерное
        const nodeCell = document.createElement('td');
        nodeCell.className = 'node-info';

        const nodeText = document.createElement('div');
        nodeText.className = 'node-text';
        nodeText.textContent = otherNode ?
            (otherNode.text || otherNode.definition || '').substring(0, 100) +
            ((otherNode.text || otherNode.definition || '').length > 100 ? '...' : '')
            : 'Unknown node';
        nodeCell.appendChild(nodeText);

        if (edge.conditions) {
            const conditions = document.createElement('div');
            conditions.className = 'edge-conditions';
            conditions.textContent = `Условие: ${edge.conditions}`;
            nodeCell.appendChild(conditions);
        }

        const interCluster = document.createElement('div');
        interCluster.className = 'inter-cluster';
        const isInterCluster = otherNode && otherNode.cluster_id !== (this.allNodes.find(n => n.id === currentNodeId)?.cluster_id);
        interCluster.textContent = `Межкластерное: ${isInterCluster ? 'true' : 'false'}`;
        if (isInterCluster) {
            interCluster.classList.add('highlighted');
        }
        nodeCell.appendChild(interCluster);

        row.appendChild(nodeCell);

        // Weights (weight + inverse_weight)
        const weightCell = document.createElement('td');
        weightCell.className = 'edge-weights';

        const weight = document.createElement('div');
        weight.className = 'weight';
        weight.textContent = this.formatMetricValue(edge.weight || 1);
        weightCell.appendChild(weight);

        const invWeight = document.createElement('div');
        invWeight.className = 'inverse-weight';
        invWeight.textContent = this.formatMetricValue(edge.inverse_weight || 0);
        weightCell.appendChild(invWeight);

        row.appendChild(weightCell);

        // Make rows clickable (add data-edge-id attribute)
        row.addEventListener('click', (e) => {
            this.selectEdge(e, row, edge, otherNode);
        });

        return row;
    },

    getNodeEdges(nodeId) {
        // Filter graphData.edges for source=nodeId or target=nodeId
        const edges = this.graphData.edges || [];

        const incoming = edges.filter(e => e.target === nodeId);
        const outgoing = edges.filter(e => e.source === nodeId);

        // Return { incoming: [], outgoing: [] }
        return { incoming, outgoing };
    },

    toggleViewMode() {
        // Switch between 'formatted' and 'json'
        this.state.viewMode = this.state.viewMode === 'formatted' ? 'json' : 'formatted';

        // Re-render active node
        const activeNode = this.allNodes.find(n => n.id === this.state.activeNodeId);
        if (activeNode) {
            this.renderActiveNode(activeNode);
        }
    },

    clearActiveNode() {
        const container = document.getElementById('active-node-content');
        if (!container) return;

        container.innerHTML = '<div class="placeholder-message">Выберите узел для исследования</div>';
    },

    addContentSection(container, title, text) {
        const section = document.createElement('div');
        // Add specific class based on section type
        const sectionType = title === 'Содержание' ? 'node-text-section' : 'node-definition-section';
        section.className = `content-section ${sectionType}`;

        const header = document.createElement('h4');
        header.textContent = title;
        section.appendChild(header);

        const content = document.createElement('div');
        const contentClass = title === 'Содержание' ? 'node-text-scroll' : 'node-definition-scroll';
        content.className = `${contentClass} formatted-content`;

        if (text === '-' || !text) {
            content.innerHTML = '<span class="empty-content">—</span>';
        } else {
            // Use formatters if available
            if (typeof window.Formatters !== 'undefined' && window.Formatters.formatNodeText) {
                content.innerHTML = window.Formatters.formatNodeText(text);

                // Apply syntax highlighting to code blocks
                if (typeof hljs !== 'undefined') {
                    content.querySelectorAll('pre code').forEach((block) => {
                        hljs.highlightElement(block);
                    });
                }

                // Trigger math rendering
                setTimeout(() => {
                    if (window.Formatters.renderMath) {
                        window.Formatters.renderMath(content);
                    }
                }, 0);
            } else {
                content.textContent = text;
            }
        }

        section.appendChild(content);
        container.appendChild(section);
    },

    addConceptsSection(container, concepts) {
        const section = document.createElement('div');
        section.className = 'content-section related-concepts-section';

        const header = document.createElement('h4');
        header.textContent = 'Связанные концепты';
        section.appendChild(header);

        const contentWrapper = document.createElement('div');
        contentWrapper.className = 'related-concepts-scroll';

        if (!concepts || concepts.length === 0) {
            const empty = document.createElement('span');
            empty.className = 'no-concepts';
            empty.textContent = 'Нет связанных концептов';
            contentWrapper.appendChild(empty);
        } else {
            const list = document.createElement('ol');
            list.className = 'concept-list';

            concepts.forEach(conceptId => {
                const li = document.createElement('li');

                // Find concept in conceptData or nodes
                let conceptText = conceptId;

                // First try conceptData
                if (this.conceptData.concepts) {
                    const concept = this.conceptData.concepts.find(c => c.id === conceptId);
                    if (concept) {
                        conceptText = concept.text || concept.definition || conceptId;
                    }
                }

                // Then try nodes
                if (conceptText === conceptId) {
                    const conceptNode = this.allNodes.find(n => n.id === conceptId && n.type === 'Concept');
                    if (conceptNode) {
                        conceptText = conceptNode.text || conceptNode.definition || conceptId;
                    }
                }

                li.textContent = conceptText;
                list.appendChild(li);
            });

            contentWrapper.appendChild(list);
        }
        section.appendChild(contentWrapper);

        container.appendChild(section);
    },

    formatDifficulty(difficulty) {
        const maxDifficulty = 5;
        const circles = [];

        for (let i = 1; i <= maxDifficulty; i++) {
            const filled = i <= difficulty;
            // Traffic light colors: green (1-2), yellow (3), red (4-5)
            const color = i <= 2 ? '#2ecc71' : i <= 3 ? '#f39c12' : '#e74c3c';
            circles.push(`<span class="difficulty-circle ${filled ? 'filled' : ''}" style="${filled ? `background: ${color};` : ''}"></span>`);
        }

        return circles.join('');
    },

    formatMetricValue(value) {
        if (value === null || value === undefined) {
            return '-';
        }

        if (typeof value === 'number') {
            if (value === 0) {
                return '0';
            } else if (value < 0.001) {
                return value.toExponential(2);
            } else if (value < 1) {
                return value.toFixed(5);
            } else if (value < 100) {
                return value.toFixed(2);
            } else {
                return Math.round(value).toString();
            }
        }

        return value.toString();
    },

    selectEdge(event, clickedRow, edge, otherNode) {
        // Prevent event bubbling
        event.stopPropagation();

        // Check if this row is already active
        const wasActive = clickedRow.classList.contains('active');

        // Remove active class from all edge rows
        document.querySelectorAll('.edge-row.active').forEach(row => {
            row.classList.remove('active');
        });

        // If the row wasn't active, make it active
        if (!wasActive) {
            clickedRow.classList.add('active');

            // Dispatch event for edge selection
            const customEvent = new CustomEvent('edge-selected', {
                detail: { edge, otherNode }
            });
            document.dispatchEvent(customEvent);
        } else {
            // Dispatch event for edge deselection
            const customEvent = new CustomEvent('edge-deselected', {
                detail: {}
            });
            document.dispatchEvent(customEvent);
        }
    }
};

// NodeExplorer is now global
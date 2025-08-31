/**
 * Course Panel Module for K2-18 Visualization
 * Displays sequential course content (Chunk nodes) in order
 */

window.CoursePanel = (function() {
    let isOpen = false;
    let panelElement = null;
    let buttonElement = null;
    let cy = null;
    let graphData = null;
    let courseSequence = [];
    
    const PANEL_WIDTH = 320; // Panel width in pixels
    
    /**
     * Initialize the course panel
     */
    function init(cyInstance, graphDataInstance) {
        cy = cyInstance;
        graphData = graphDataInstance;
        
        // Extract course sequence from _meta
        courseSequence = graphData._meta?.course_sequence || [];
        
        console.log('[CoursePanel] Initializing with', courseSequence.length, 'items');
        
        createButton();
        createPanel();
        populateContent();
        setupInteractions();
        
        console.log('[CoursePanel] Initialized');
    }
    
    /**
     * Create the tab button on the left side
     */
    function createButton() {
        buttonElement = document.createElement('div');
        buttonElement.className = 'course-panel-button';
        buttonElement.title = 'Содержание курса';
        buttonElement.innerHTML = '📖';
        
        // EXACT copy of right panel button but mirrored
        buttonElement.style.cssText = `
            position: fixed;
            left: 0;
            top: 50%;
            transform: translateY(-50%);
            width: 45px;
            height: 70px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: 2px solid white;
            border-left: none;
            border-radius: 0 12px 12px 0;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 28px;
            z-index: 1001; /* Above bottom badge (1000) */
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
            color: white;
            text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
        `;
        
        // Add hover effect - EXACT copy from right panel
        buttonElement.addEventListener('mouseenter', () => {
            buttonElement.style.background = 'linear-gradient(135deg, #5a72e5 0%, #6b4299 100%)';
            buttonElement.style.transform = 'translateY(-50%) translateX(4px)';
            buttonElement.style.boxShadow = '4px 4px 12px rgba(0, 0, 0, 0.3)';
        });
        
        buttonElement.addEventListener('mouseleave', () => {
            buttonElement.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
            buttonElement.style.transform = 'translateY(-50%)';
            buttonElement.style.boxShadow = '2px 2px 10px rgba(0, 0, 0, 0.2)';
        });
        
        // Click handler
        buttonElement.addEventListener('click', togglePanel);
        
        document.body.appendChild(buttonElement);
    }
    
    /**
     * Create the main panel
     */
    function createPanel() {
        panelElement = document.createElement('div');
        panelElement.className = 'course-panel';
        
        panelElement.style.cssText = `
            position: fixed;
            left: -${PANEL_WIDTH}px;
            top: 106px;
            width: ${PANEL_WIDTH}px;
            height: calc(100vh - 106px);
            background: white;
            box-shadow: 2px 0 5px rgba(0,0,0,0.1);
            transition: left 0.3s ease;
            z-index: 1001; /* Above bottom badge (1000) */
            display: flex;
            flex-direction: column;
            overflow: hidden;
        `;
        
        // Create panel header
        const header = document.createElement('div');
        header.style.cssText = `
            padding: 16px;
            border-bottom: 1px solid #ddd;
            background: #f8f8f8;
        `;
        header.innerHTML = `
            <h3 style="margin: 0; font-size: 18px; font-weight: 600;">
                📖 Содержание курса
            </h3>
            <div style="margin-top: 8px; font-size: 12px; color: #666;">
                ${courseSequence.length} учебных блока (Chunks)
            </div>
        `;
        
        // Create scrollable content container
        const contentContainer = document.createElement('div');
        contentContainer.id = 'course-content';
        contentContainer.style.cssText = `
            flex: 1;
            overflow-y: auto;
            padding: 8px;
        `;
        
        panelElement.appendChild(header);
        panelElement.appendChild(contentContainer);
        
        document.body.appendChild(panelElement);
    }
    
    /**
     * Populate panel with course content
     */
    function populateContent() {
        const container = document.getElementById('course-content');
        
        if (courseSequence.length === 0) {
            container.innerHTML = `
                <div style="padding: 20px; text-align: center; color: #999;">
                    Нет элементов курса в формате {slug}:c:{position}
                </div>
            `;
            return;
        }
        
        // Get cluster colors from config or use defaults
        const clusterColors = [
            "#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6",
            "#1abc9c", "#34495e", "#e67e22", "#16a085", "#8e44ad",
            "#2c3e50", "#27ae60"
        ];
        
        // Create items for each course element
        const html = courseSequence.map((item, index) => {
            // Find the corresponding node data
            const node = cy.getElementById(item.id);
            const nodeData = node.length > 0 ? node.data() : null;
            
            // Get text preview (first 60 characters)
            let textPreview = '';
            if (nodeData && nodeData.text) {
                textPreview = nodeData.text.substring(0, 60);
                if (nodeData.text.length > 60) {
                    textPreview += '...';
                }
            }
            
            // Get cluster color
            const clusterColor = clusterColors[item.cluster_id % clusterColors.length];
            
            return `
                <div class="course-item" 
                     data-node-id="${item.id}"
                     data-position="${item.position}"
                     style="
                        padding: 10px 12px;
                        margin-bottom: 8px;
                        border: 1px solid #e0e0e0;
                        border-radius: 6px;
                        cursor: pointer;
                        transition: all 0.2s ease;
                        background: linear-gradient(to right, 
                            ${hexToRgba(clusterColor, 0.1)} 0%, 
                            ${hexToRgba(clusterColor, 0.05)} 100%);
                        border-left: 3px solid ${clusterColor};
                     ">
                    <div style="
                        font-size: 12px;
                        color: #333;
                        line-height: 1.5;
                    ">
                        <span style="font-family: monospace; font-weight: bold;">${item.position}</span> | ${textPreview || '<em>Нет текста</em>'}
                    </div>
                </div>
            `;
        }).join('');
        
        container.innerHTML = html;
    }
    
    /**
     * Setup hover and click interactions
     */
    function setupInteractions() {
        const container = document.getElementById('course-content');
        
        // Delegate event handling to container
        container.addEventListener('mouseover', (e) => {
            const item = e.target.closest('.course-item');
            if (item) {
                handleItemHover(item.dataset.nodeId);
                item.style.background = item.style.background.replace('0.1', '0.2').replace('0.05', '0.15');
                item.style.transform = 'translateX(4px)';
            }
        });
        
        container.addEventListener('mouseout', (e) => {
            const item = e.target.closest('.course-item');
            if (item) {
                clearHighlight();
                item.style.background = item.style.background.replace('0.2', '0.1').replace('0.15', '0.05');
                item.style.transform = 'translateX(0)';
            }
        });
        
        container.addEventListener('click', (e) => {
            const item = e.target.closest('.course-item');
            if (item) {
                handleItemClick(item.dataset.nodeId);
            }
        });
    }
    
    /**
     * Handle hover on course item - USE PULSE LIKE RIGHT PANEL
     */
    function handleItemHover(nodeId) {
        if (!cy) return;
        
        const node = cy.getElementById(nodeId);
        if (node.length > 0) {
            // Use pulse class EXACTLY like right panel
            node.addClass('pulse');
        }
    }
    
    /**
     * Clear node highlighting - USE PULSE LIKE RIGHT PANEL
     */
    function clearHighlight() {
        if (!cy) return;
        
        // Remove pulse class from all nodes - EXACTLY like right panel
        cy.nodes().removeClass('pulse');
    }
    
    /**
     * Handle click on course item - zoom and center WITHOUT popup
     */
    function handleItemClick(nodeId) {
        if (!cy) return;
        
        const node = cy.getElementById(nodeId);
        if (node.length > 0) {
            // Zoom and center on the node
            cy.animate({
                zoom: 1.5,
                center: {
                    eles: node
                }
            }, {
                duration: 500
            });
            
            // Flash the node for visual feedback
            node.addClass('flashing');
            setTimeout(() => {
                node.removeClass('flashing');
            }, 1000);
        }
    }
    
    /**
     * Toggle panel open/closed
     */
    function togglePanel() {
        isOpen = !isOpen;
        
        if (isOpen) {
            panelElement.style.left = '0';
            buttonElement.style.left = `${PANEL_WIDTH}px`;
        } else {
            panelElement.style.left = `-${PANEL_WIDTH}px`;
            buttonElement.style.left = '0';
        }
    }
    
    /**
     * Helper function to convert hex to rgba
     */
    function hexToRgba(hex, alpha) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }
    
    // Public API
    return {
        init: init,
        togglePanel: togglePanel  // Export toggle function for keyboard shortcut
    };
})();
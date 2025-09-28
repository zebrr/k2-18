// Search and filter functionality
const SearchFilter = {
    state: {
        searchQuery: '',
        activeTypes: {
            Chunk: true,
            Concept: true,
            Assessment: true
        },
        filteredNodes: [],
        allNodes: []
    },
    
    init(graphData) {
        console.log('Search filter initializing...');
        this.state.allNodes = graphData.nodes || [];
        this.state.filteredNodes = [...this.state.allNodes];
        this.renderFilters();
        this.setupEventListeners();
        this.applyFilters();
        console.log(`Search filter initialized with ${this.state.allNodes.length} nodes`);
    },
    
    renderFilters() {
        const searchContainer = document.getElementById('search-container');
        if (!searchContainer) {
            console.error('Search container not found');
            return;
        }
        
        // Clear existing content
        searchContainer.innerHTML = '';
        
        // Create counter
        const counter = document.createElement('div');
        counter.id = 'node-counter';
        counter.innerHTML = `
            Показано: <span id="shown-count">0</span> из <span id="total-count">${this.state.allNodes.length}</span>
        `;
        searchContainer.appendChild(counter);
        
        // Create type filters
        const typeFilters = document.createElement('div');
        typeFilters.id = 'type-filters';
        typeFilters.innerHTML = `
            <label><input type="checkbox" data-type="Chunk" ${this.state.activeTypes.Chunk ? 'checked' : ''}> Chunks</label>
            <label><input type="checkbox" data-type="Concept" ${this.state.activeTypes.Concept ? 'checked' : ''}> Concepts</label>
            <label><input type="checkbox" data-type="Assessment" ${this.state.activeTypes.Assessment ? 'checked' : ''}> Asmnt</label>
        `;
        searchContainer.appendChild(typeFilters);
        
        // Create search input container
        const searchInputContainer = document.createElement('div');
        searchInputContainer.id = 'search-input-container';
        searchInputContainer.innerHTML = `
            <input type="text" id="search-input" placeholder="Поиск..." value="${this.state.searchQuery}">
            <button id="clear-search">×</button>
        `;
        searchContainer.appendChild(searchInputContainer);
    },
    
    setupEventListeners() {
        // Type filter checkboxes
        const checkboxes = document.querySelectorAll('#type-filters input[type="checkbox"]');
        checkboxes.forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                const type = e.target.dataset.type;
                this.toggleType(type, e.target.checked);
            });
        });
        
        // Search input
        const searchInput = document.getElementById('search-input');
        if (searchInput) {
            searchInput.addEventListener('keyup', (e) => {
                if (e.key === 'Enter') {
                    this.handleSearch(e.target.value);
                }
            });
            
            // Also handle input event for real-time search
            searchInput.addEventListener('input', (e) => {
                this.state.searchQuery = e.target.value;
                // Debounce search
                clearTimeout(this.searchTimeout);
                this.searchTimeout = setTimeout(() => {
                    this.handleSearch(e.target.value);
                }, 300);
            });
        }
        
        // Clear search button
        const clearButton = document.getElementById('clear-search');
        if (clearButton) {
            clearButton.addEventListener('click', () => {
                this.clearSearch();
            });
        }
    },
    
    handleSearch(query) {
        console.log(`Searching for: ${query}`);
        this.state.searchQuery = query;
        this.applyFilters();
    },
    
    clearSearch() {
        this.state.searchQuery = '';
        const searchInput = document.getElementById('search-input');
        if (searchInput) {
            searchInput.value = '';
        }
        this.applyFilters();
    },
    
    toggleType(type, checked) {
        console.log(`Toggle ${type}: ${checked}`);
        this.state.activeTypes[type] = checked;
        this.applyFilters();
    },
    
    applyFilters() {
        // Filter by type
        let filtered = this.state.allNodes.filter(node => {
            const nodeType = node.type || 'Unknown';
            return this.state.activeTypes[nodeType];
        });

        // Filter by search query (case-insensitive)
        if (this.state.searchQuery) {
            const query = this.state.searchQuery.toLowerCase();
            filtered = filtered.filter(node => {
                const inId = (node.id || '').toLowerCase().includes(query);
                const inText = (node.text || '').toLowerCase().includes(query);
                const inDefinition = (node.definition || '').toLowerCase().includes(query);
                return inId || inText || inDefinition;
            });
        }

        this.state.filteredNodes = filtered;
        this.updateCounter();

        // Dispatch custom event
        const event = new CustomEvent('filter-changed', {
            detail: {
                filteredNodes: this.state.filteredNodes,
                totalNodes: this.state.allNodes.length
            }
        });
        document.dispatchEvent(event);

        console.log(`Filters applied: ${this.state.filteredNodes.length} nodes shown`);
    },
    
    updateCounter() {
        const shownCount = document.getElementById('shown-count');
        const totalCount = document.getElementById('total-count');

        if (shownCount) {
            shownCount.textContent = this.state.filteredNodes.length;
        }
        if (totalCount) {
            totalCount.textContent = this.state.allNodes.length;
        }
    },

    getFilteredNodes() {
        return this.state.filteredNodes;
    }
};

// Export for ES6 modules
export { SearchFilter };
/**
 * Tour Mode Module for K2-18 Visualization
 * Implements automatic presentation tour through the graph
 */

const TourMode = {
    active: false,
    
    init(cy, graphCore, config) {
        this.cy = cy;
        this.graphCore = graphCore;
        this.config = config;
        console.log('[TourMode] Module loaded (stub)');
    },
    
    activate() {
        this.active = true;
        console.log('[TourMode] Activated - implementation pending');
        // TODO: Implement in VIZ-FRONT-10
    },
    
    deactivate() {
        this.active = false;
        console.log('[TourMode] Deactivated');
    }
};

// Listen for mode changes
document.addEventListener('mode-changed', (e) => {
    if (e.detail.mode === 'tour') {
        TourMode.activate();
    } else if (TourMode.active) {
        TourMode.deactivate();
    }
});

window.TourMode = TourMode;
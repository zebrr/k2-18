// Text formatting utilities
const Formatters = {
    initialized: false,

    init() {
        console.log('Formatters initializing...');

        // Initialize marked options for Markdown
        if (typeof marked !== 'undefined') {
            marked.setOptions({
                highlight: function(code, lang) {
                    if (lang && typeof hljs !== 'undefined') {
                        try {
                            return hljs.highlight(code, { language: lang }).value;
                        } catch (e) {
                            console.warn('Error highlighting code:', e);
                        }
                    }
                    return code;
                },
                breaks: true,
                gfm: true,
                headerIds: false,
                mangle: false
            });
        }

        // Initialize highlight.js
        if (typeof hljs !== 'undefined') {
            hljs.configure({
                languages: ['javascript', 'python', 'json', 'html', 'css', 'bash', 'sql']
            });
        }

        // Setup MathJax configuration
        if (typeof MathJax !== 'undefined') {
            window.MathJax = {
                tex: {
                    inlineMath: [['$', '$'], ['\\(', '\\)']],
                    displayMath: [['$$', '$$'], ['\\[', '\\]']],
                    processEscapes: true
                },
                svg: {
                    fontCache: 'global'
                },
                startup: {
                    typeset: false // We'll trigger manually
                }
            };
        }

        this.initialized = true;
        console.log('Formatters initialized');
    },

    formatMarkdown(text) {
        // Handle empty text case
        if (!text || text.trim() === '') {
            return '';
        }

        // Use marked.parse() to convert markdown to HTML
        if (typeof marked !== 'undefined') {
            try {
                return marked.parse(text);
            } catch (e) {
                console.error('Error parsing markdown:', e);
                return text;
            }
        }

        // Fallback if marked is not available
        return text.replace(/\n/g, '<br>');
    },

    highlightCode(code, language) {
        // Handle empty code case
        if (!code) {
            return '';
        }

        // Use hljs.highlight() or hljs.highlightAuto()
        if (typeof hljs !== 'undefined') {
            try {
                if (language) {
                    return hljs.highlight(code, { language: language }).value;
                } else {
                    return hljs.highlightAuto(code).value;
                }
            } catch (e) {
                console.warn('Error highlighting code:', e);
                return code;
            }
        }

        // Return code wrapped in pre/code tags as fallback
        return `<pre><code>${this.escapeHtml(code)}</code></pre>`;
    },

    renderMath(element) {
        // Trigger MathJax.typesetPromise() on element
        if (typeof MathJax !== 'undefined' && MathJax.typesetPromise) {
            // Handle async rendering
            MathJax.typesetPromise([element])
                .then(() => {
                    console.log('Math rendering completed');
                })
                .catch((e) => {
                    console.error('Error rendering math:', e);
                });
        }
    },

    formatNodeText(text) {
        // Combined formatter:
        // Return empty string for empty text
        if (!text || text.trim() === '') {
            return '';
        }

        // 1. Parse markdown
        let formatted = this.formatMarkdown(text);

        // 2. Find and highlight code blocks (already done by marked if configured)
        // 3. Math rendering will be triggered after DOM insertion

        // Return formatted HTML
        return formatted;
    },

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },

    formatDifficulty(difficulty) {
        // Create difficulty dots visualization
        if (typeof difficulty !== 'number') {
            return '';
        }

        const maxDifficulty = 5;
        const dots = [];

        for (let i = 1; i <= maxDifficulty; i++) {
            if (i <= difficulty) {
                dots.push('●');
            } else {
                dots.push('○');
            }
        }

        return dots.join('');
    },

    formatMetricValue(value) {
        // Format metric values for display
        if (value === null || value === undefined) {
            return '-';
        }

        if (typeof value === 'number') {
            // Format based on value range
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

    formatEdgeType(type) {
        // Format edge type for display
        const typeMap = {
            'PREREQUISITE': 'Требуется для',
            'ELABORATES': 'Развивает',
            'TESTS': 'Тестирует',
            'EXAMPLE_OF': 'Пример',
            'SIMILAR_TO': 'Похоже на',
            'PART_OF': 'Часть',
            'CONTRAST_WITH': 'Контраст с',
            'LEADS_TO': 'Ведёт к',
            'SUPPORTS': 'Поддерживает'
        };

        return typeMap[type] || type;
    },

    // Create formatted content block
    createFormattedBlock(content, className = 'formatted-content') {
        const div = document.createElement('div');
        div.className = className;
        div.innerHTML = this.formatNodeText(content);

        // Trigger math rendering after adding to DOM
        setTimeout(() => {
            this.renderMath(div);
        }, 0);

        return div;
    }
};

// Export for ES6 modules
export { Formatters };
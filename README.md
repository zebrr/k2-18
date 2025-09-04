# K2-18 - Educational Knowledge Graph Converter

## Why K2-18?

Traditional educational content is linear text, but learning is a network of interconnected concepts. K2-18 bridges this gap by automatically extracting the hidden knowledge structure from textbooks and educational materials.

**The Problem**: Educational platforms need structured content for adaptive learning, prerequisite tracking, and personalized paths, but manually creating this structure is prohibitively expensive.

**The Solution**: K2-18 automatically converts any educational text into a semantic knowledge graph with:
- Extracted concepts with definitions and relationships
- Learning dependencies (what to learn first)
- Difficulty levels and assessment points
- Semantic connections between distant but related topics

## What You Get

The pipeline produces two main outputs:

- **ConceptDictionary** - comprehensive vocabulary of all concepts with definitions, aliases, and cross-references
- **LearningChunkGraph** - semantic graph connecting content chunks, concepts, and assessments with typed relationships

## Architecture

K2-18 implements the **iText2KG (Incremental Text to Knowledge Graph)** approach - incremental knowledge graph construction from text, designed to work within LLM context window limitations.

### Processing Pipeline

```
Raw Content (.md, .txt, .html)
    ↓
1. Slicer             → Semantic Chunks (respecting paragraph boundaries)
    ↓
2. iText2KG Concepts  → Concept Dictionary (with all concepts extracted)
    ↓
3. iText2KG Graph     → Knowledge Graph (using Concept Dictionary)
    ↓
4. Dedup              → Knowledge Graph (with semantic duplicates removed)
    ↓
5. Refiner Longrange  → Knowledge Graph (with long-range connections added)
```

## Use Cases

After running the pipeline, you can:

- **Import into Neo4j**: Use the JSON graph directly with Neo4j's import tools (see Neo4j documentation for import details)
- **Build Learning Paths**: Find optimal prerequisite chains between topics
- **Curriculum Analysis**: Identify knowledge gaps and redundancies
- **Adaptive Learning**: Power recommendation systems with concept dependencies
- **Content Quality**: Detect missing prerequisites or circular dependencies

### Key Features

- **Incremental Processing**: Handles books of 100-1000 pages by processing in chunks
- **Context Preservation**: Maintains semantic continuity across chunk boundaries
- **Smart Deduplication**: Uses embeddings to identify and merge semantically identical content
- **Long-range Connections**: Discovers relationships between concepts separated by many pages (forward/backward pass)
- **Language Support**: Any UTF-8 text content

## Visualization & Analytics Module

K2-18 includes a powerful visualization module that enriches your knowledge graphs with educational metrics and creates interactive visualizations.

**What it does:**
- Computes 12 network metrics revealing graph structure and learning paths
- Identifies fundamental concepts, knowledge bridges, and topic clusters
- Generates a standalone interactive HTML that works in any browser

Perfect for quality control, curriculum analysis, and presenting results to stakeholders.

For detailed documentation, see [Visualization Module Guide](./viz/README.md).

## Requirements

- Python 3.11+
- OpenAI API access (Responses API)
- Memory: Not tested (entire corpus processed in memory)
- OS: Windows, macOS

## Installation

### For Users

```bash
# Clone the repository
git clone https://github.com/yourusername/k2-18.git
cd k2-18

# Create virtual environment
python -m venv .venv

# Activate it (choose your platform):
source .venv/bin/activate         # Linux/macOS
.venv\Scripts\activate            # Windows

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key"  # Linux/macOS
set OPENAI_API_KEY=your-api-key       # Windows

# Set your SOY_TOKEN key
export SOY_TOKEN="your-api-key"  # Linux/macOS
set SOY_TOKEN=your-api-key       # Windows

# Optional: Set INTERNAL_EMBEDDING_API_KEY for internal embeddings
# Use when enabling internal embeddings via embedding_use_internal_auth=true
export INTERNAL_EMBEDDING_API_KEY="your-internal-oauth-token"  # Linux/macOS
set INTERNAL_EMBEDDING_API_KEY=your-internal-oauth-token       # Windows
```

### For Developers

```bash
# Same initial setup as above, then:

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

## Quick Start

1. **Prepare content**:
   ```bash
   # Place educational materials in:
   data/raw/
   ```
   Supported formats: `.md`, `.txt`, `.html`

2. **Configure processing** (optional):
   Edit `src/config.toml` to adjust parameters like chunk size, overlap, and model selection.

3. **Run the pipeline**:
   ```bash
   # Step-by-step processing
   python -m src.slicer               # Split into chunks
   python -m src.itext2kg_concepts    # Extract concepts
   python -m src.itext2kg_graph       # Build knowledge graph
   python -m src.dedup                # Remove duplicates if any
   python -m src.refiner_longrange    # Add distant connections
   ```

4. **Find your results**:
   ```bash
   data/out/
   ├── ConceptDictionary.json             # All extracted concepts
   ├── LearningChunkGraph_raw.json        # Initial graph
   ├── LearningChunkGraph_dedup.json      # After deduplication
   └── LearningChunkGraph_longrange.json  # Final graph
   ```

## Configuration

Main settings in `src/config.toml`:

```toml
[slicer]
max_tokens = 5000          # Chunk size in tokens
overlap = 0                # Context window size managed by response_chain_depth
soft_boundary = true       # Respect semantic boundaries

[itext2kg]
model = "..."              # OpenAI model selection
tpm_limit = 150000         # API rate limit (tokens/minute) based on your Tier
max_output_tokens = 25000  # Max response size

[dedup]
sim_threshold = 0.85       # Similarity threshold for duplicates

[refiner]
run = true                 # Enable/disable refiner stage
sim_threshold = 0.7        # Threshold for new connections
```

## Data Formats

All data formats are defined by JSON schemas in `/src/schemas/`:
- `ConceptDictionary.schema.json` - concept vocabulary structure
- `LearningChunkGraph.schema.json` - knowledge graph structure

## Documentation

- **Component specifications**: [`/docs/specs/`](docs/specs/)
  - Pipeline utilities: `cli_slicer.md`, `cli_itext2kg_concepts.md`, `cli_itext2kg_graph.md`, `cli_dedup.md`, `cli_refiner_longrange.md`  
  - Core modules: `util_llm_client.md`, `util_config.md`, `util_validation.md`, `util_tokenizer.md`, etc.
  
- **Data schemas**: [`/src/schemas/`](src/schemas/)
  - `ConceptDictionary.schema.json` - concept vocabulary structure
  - `LearningChunkGraph.schema.json` - knowledge graph structure
  
- **LLM prompts**: [`/src/prompts/`](src/prompts/)
  - `itext2kg_concepts_extraction.md` - concept extraction from text
  - `itext2kg_graph_extraction.md` - knowledge graph construction
  - `refiner_longrange_fw.md` - forward pass for long-range connections
  - `refiner_longrange_bw.md` - backward pass for long-range connections
  
**⚠️ Important**: Current prompts are optimized for Computer Science or Economics content in Russian. For other domains (management, history, biology, etc.) or languages, prompts **REQUIRE** adaptation to domain-specific terminology and concept patterns.

## Limitations

- **Memory-bound**: Entire corpus processed in memory
- **Sequential**: No parallel processing (to maintain context/TPM limits)
- **API-dependent**: Requires stable OpenAI API access
- **Token limits**: Constrained by LLM context windows

## Troubleshooting

### API Rate Limits

- Check your OpenAI API Tier limits
- Adjust `tpm_limit` in config
- Pipeline will auto-retry with backoff

### Incomplete Processing

- Check exit codes and logs in `/logs/`
- Use `previous_response_id` for context continuity
- Use `timeout` and `max_retries ` to manage retries
- Utilities do not support resuming from last successful slice

## Development

### Contributing

1. Follow TDD approach - write tests first
2. All functions must have type hints
3. Run quality checks before commits
4. Update relevant specifications in `/docs/specs/`

### Code Quality

```bash
# Format code
black src/
isort src/

# Check quality
ruff check src/
flake8 src/
mypy src/
```

### Running Tests

```bash
# Activate virtual environment first
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Run all tests but integration (fast!)
pytest tests/ -v -m "not integration"

# Full test suite incl. integration (~5-7m and require OpenAI API key)
pytest tests/ -v

```
Test markers:
- `integration` - Tests requiring real API calls
- `slow` - Tests taking >30 seconds
- `timeout` - Tests with explicit timeout settings

## License — Non-Commercial Educational & Research Use

Copyright (c) 2025 @zebrr. All rights reserved.

Permission is granted to use, copy, modify, and distribute this project **solely for non-commercial educational or academic research purposes**, subject to the following conditions:

1) You must retain this license notice and a link to the original repository.  
2) You may not charge users, sell access, or integrate this project into paid products or services.  
3) **Corporate/for-profit use is prohibited**, including CSR/PR/brand initiatives, recruiting/lead-gen, or internal employee training—even if the training itself is free.

Any **commercial or corporate use (including CSR)** requires a separate commercial license.  
To discuss terms, please open an Issue in this repository addressed to **@zebrr**.

This software is provided **“AS IS”**, without warranty of any kind. Use may require third-party API keys; all related costs are the user’s responsibility.

## Support

- Check `/docs/specs/` for detailed component documentation
- Review logs in `/logs/` for debugging
- Open a Github Issue
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
1. Slicer → Semantic chunks (respecting paragraph boundaries)
    ↓
2. iText2KG Concepts → ConceptDictionary (all concepts extracted)
    ↓
3. iText2KG Graph → Raw knowledge graph (using ConceptDictionary)
    ↓
4. Dedup → Clean graph (semantic duplicates removed)
    ↓
5. Refiner → Final graph (long-range connections added)
```

### Key Features

- **Incremental Processing**: Handles books of 100-1000 pages by processing in chunks
- **Context Preservation**: Maintains semantic continuity across chunk boundaries
- **Smart Deduplication**: Uses embeddings to identify and merge semantically identical content
- **Long-range Connections**: Discovers relationships between concepts separated by many pages
- **Language Support**: Russian and English content

## Requirements

- Python 3.11+
- OpenAI API access (Responses API)
- Memory: ~2GB per 100 pages of text
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
```

### For Developers

```bash
# Same initial setup as above, then:

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Check code quality
ruff check src/
black src/ --check
```

## Quick Start

1. **Prepare your content**:
   ```bash
   # Place your educational materials in:
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
   python -m src.dedup                # Remove duplicates
   python -m src.refiner              # Add distant connections
   ```

4. **Find your results**:
   ```bash
   data/out/
   ├── ConceptDictionary.json       # All extracted concepts
   ├── LearningChunkGraph_raw.json  # Initial graph
   ├── LearningChunkGraph_dedup.json # After deduplication
   └── LearningChunkGraph.json      # Final graph
   ```

## Configuration

Main settings in `src/config.toml`:

```toml
[slicer]
max_tokens = 5000        # Chunk size in tokens
overlap = 0              # Token overlap between chunks
soft_boundary = true     # Respect semantic boundaries

[itext2kg]
model = "..."           # OpenAI model selection
tpm_limit = 150000      # API rate limit (tokens/minute)
max_output_tokens = 25000  # Max response size

[dedup]
sim_threshold = 0.97    # Similarity threshold for duplicates

[refiner]
run = true              # Enable/disable refiner stage
sim_threshold = 0.7     # Threshold for new connections
```

## Data Formats

All data formats are defined by JSON schemas in `/src/schemas/`:
- `ConceptDictionary.schema.json` - concept vocabulary structure
- `LearningChunkGraph.schema.json` - knowledge graph structure

## Error Handling

The pipeline uses consistent exit codes:
- `0` - Success
- `1` - Configuration error
- `2` - Input data error
- `3` - Runtime error
- `4` - API rate limit exceeded
- `5` - I/O error

See `/docs/specs/util_exit_codes.md` for details.

## Project Structure

```
k2-18/
├── src/
│   ├── slicer.py              # Text chunking
│   ├── itext2kg_concepts.py   # Concept extraction
│   ├── itext2kg_graph.py      # Graph construction
│   ├── dedup.py               # Deduplication
│   ├── refiner.py             # Connection refinement
│   ├── config.toml            # Configuration
│   ├── prompts/               # LLM prompts
│   ├── schemas/               # JSON schemas
│   └── utils/                 # Shared utilities
├── data/
│   ├── raw/                   # Input files
│   ├── staging/               # Intermediate files
│   └── out/                   # Results
├── docs/
│   └── specs/                 # Module specifications
├── tests/                     # Test suite
└── logs/                      # Processing logs
```

## Documentation

Detailed specifications for each component are in `/docs/specs/`:
- CLI utilities: `cli_*.md`
- Utility modules: `util_*.md`
- Architecture overview: see main Technical Specification

## Troubleshooting

### Common Issues

**Out of Memory**
- The pipeline processes everything in memory
- Estimate: ~2GB RAM per 100 pages
- Solution: Process smaller batches or increase available memory

**API Rate Limits**
- Check your OpenAI tier limits
- Adjust `tpm_limit` in config
- Pipeline will auto-retry with backoff

**Incomplete Processing**
- Check exit codes and logs in `/logs/`
- Most utilities support resuming from last successful slice
- Use `previous_response_id` for context continuity

## Development

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_slicer.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

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

### Contributing

1. Follow TDD approach - write tests first
2. All functions must have type hints
3. Update relevant specifications in `/docs/specs/`
4. Run quality checks before commits
5. Keep specifications in sync with code

## Limitations

- **Memory-bound**: Entire corpus processed in memory
- **Sequential**: No parallel processing (to maintain context)
- **API-dependent**: Requires stable OpenAI API access
- **Token limits**: Constrained by LLM context windows

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
- See Technical Specification for architecture details
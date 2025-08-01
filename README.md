# K2-18 - Semantic Knowledge Graph Converter

K2-18 is a semantic knowledge graph converter that processes educational content into structured knowledge representations. It extracts concepts, builds relationships, and creates a semantic graph suitable for educational applications.

## Overview

The project transforms raw educational text into:
- **ConceptDictionary** - structured vocabulary of concepts with definitions
- **LearningChunkGraph** - semantic graph with relationships between content chunks, concepts, and assessments

## Architecture

The processing pipeline consists of four main stages:

1. **Slicer** - Splits raw content into manageable chunks with soft boundaries
2. **iText2KG** - Extracts concepts and builds initial knowledge graph using LLM
3. **Dedup** - Removes duplicate nodes using semantic similarity (embeddings + FAISS)
4. **Refiner** - Discovers and adds long-range semantic connections

## Installation

### Requirements

- Python 3.8+
- OpenAI API key
- Dependencies: `numpy`, `faiss-cpu`, `openai`, `jsonschema`, `python-dotenv`

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/k2-18.git
cd k2-18
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY="your-api-key-here"
# Or create a .env file with:
# OPENAI_API_KEY=your-api-key-here
```

## Usage

### Basic Workflow

All tools are configured via `src/config.toml` and work with predefined directory structure:

1. **Prepare your content**:
   - Place raw text files in `data/raw/` directory
   - Supported formats: `.txt`, `.md`, `.html`

2. **Run the pipeline**:
```bash
# Step 1: Slice raw content into chunks
python -m src.slicer

# Step 2: Extract knowledge graph from slices
python -m src.itext2kg

# Step 3: Remove semantic duplicates
python -m src.dedup

# Step 4: Add long-range connections (optional)
python -m src.refiner
```

### Directory Structure

The tools automatically work with these directories:
- **Input**: `data/raw/` - place your source files here
- **Staging**: `data/staging/` - intermediate slice files (*.slice.json)
- **Output**: `data/out/` - final knowledge graphs
- **Logs**: `logs/` - processing logs and temporary files

## Configuration

Edit `src/config.toml` to adjust processing parameters:

```toml
[slicer]
max_tokens = 5000           # Maximum tokens per chunk
overlap = 0                 # Token overlap between chunks
soft_boundary = true        # Enable semantic boundary detection

[itext2kg]
model = "o4-mini-2025-04-16"  # LLM model for extraction
tpm_limit = 150000            # Tokens per minute limit

[dedup]
sim_threshold = 0.97        # Similarity threshold for duplicates
embedding_model = "text-embedding-3-small"

[refiner]
run = true                  # Enable/disable refiner
sim_threshold = 0.7         # Threshold for new connections
```

## Output Formats

### ConceptDictionary Schema

```json
{
  "concepts": [
    {
      "concept_id": "unique-id",
      "term": {
        "primary": "Main Term",
        "aliases": ["synonym1", "synonym2"]
      },
      "definition": "Clear definition of the concept"
    }
  ]
}
```

### LearningChunkGraph Schema

```json
{
  "nodes": [
    {
      "id": "node-id",
      "type": "Chunk|Concept|Assessment",
      "text": "Node content",
      "local_start": 0,
      "difficulty": 3
    }
  ],
  "edges": [
    {
      "source": "node-id-1",
      "target": "node-id-2", 
      "type": "MENTIONS|PREREQUISITE|TESTS|RELATES_TO",
      "weight": 0.85
    }
  ]
}
```

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific module tests
python -m pytest tests/test_slicer.py -v

# Run with coverage
python -m pytest tests/ --cov=src
```

## Development

### Project Structure

```
k2-18/
├── src/
│   ├── slicer.py          # Text chunking
│   ├── itext2kg.py        # Knowledge extraction
│   ├── dedup.py           # Deduplication
│   ├── refiner.py         # Relationship refinement
│   ├── config.toml        # Configuration
│   └── utils/             # Shared utilities
├── data/
│   ├── raw/               # Input files
│   ├── staging/           # Intermediate files
│   └── out/               # Final outputs
├── tests/                 # Test suite
└── docs/                  # Documentation
```

### Contributing

1. Follow PEP 8 style guidelines
2. Add type hints to all functions
3. Write tests for new features
4. Update documentation as needed

## License

[License information here]

## Support

For issues and questions:
- Create an issue on GitHub
- Check existing documentation in `/docs`
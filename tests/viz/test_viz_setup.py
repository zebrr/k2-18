"""
Tests for visualization infrastructure setup.
"""

from pathlib import Path

import pytest


@pytest.mark.viz
def test_viz_infrastructure():
    """Test that viz infrastructure is properly set up."""
    viz_root = Path("viz")
    assert viz_root.exists(), "viz directory does not exist"
    assert viz_root.is_dir(), "viz is not a directory"

    # Check config file
    config_file = viz_root / "config.toml"
    assert config_file.exists(), "viz/config.toml does not exist"
    assert config_file.is_file(), "viz/config.toml is not a file"

    # Check directory structure
    assert (viz_root / "data").exists(), "viz/data directory does not exist"
    assert (viz_root / "data" / "in").exists(), "viz/data/in directory does not exist"
    assert (viz_root / "data" / "out").exists(), "viz/data/out directory does not exist"
    assert (viz_root / "data" / "test").exists(), "viz/data/test directory does not exist"
    assert (viz_root / "templates").exists(), "viz/templates directory does not exist"
    assert (viz_root / "static").exists(), "viz/static directory does not exist"
    assert (viz_root / "logs").exists(), "viz/logs directory does not exist"

    # Check test data files
    test_graph = viz_root / "data" / "test" / "tiny_graph.json"
    test_concepts = viz_root / "data" / "test" / "tiny_concepts.json"
    assert test_graph.exists(), "tiny_graph.json does not exist"
    assert test_concepts.exists(), "tiny_concepts.json does not exist"

    # Check placeholder modules
    assert (viz_root / "graph2metrics.py").exists(), "graph2metrics.py does not exist"
    assert (viz_root / "graph2html.py").exists(), "graph2html.py does not exist"


@pytest.mark.viz
def test_config_loading_with_path():
    """Test that config.py can load viz config."""
    from src.utils.config import load_config

    # Load viz config with explicit path
    config = load_config("viz/config.toml")
    assert config is not None, "Failed to load viz config"

    # Check that viz-specific sections are present
    assert "graph2metrics" in config, "graph2metrics section missing"
    assert "graph2html" in config, "graph2html section missing"
    assert "visualization" in config, "visualization section missing"
    assert "colors" in config, "colors section missing"
    assert "ui" in config, "ui section missing"
    assert "filters" in config, "filters section missing"
    assert "performance" in config, "performance section missing"

    # Check specific values from config
    assert config["graph2metrics"]["pagerank_damping"] == 0.85
    assert config["graph2metrics"]["demo_strategy"] == 1
    assert config["graph2html"]["output_filename"] == "knowledge_graph.html"
    assert config["visualization"]["initial_layout"] == "cose-bilkent"
    assert config["colors"]["theme"] == "modern"
    assert len(config["colors"]["cluster_palette"]) == 12
    assert config["ui"]["sidebar_width"] == 320
    assert config["filters"]["show_chunks"] is True
    assert config["performance"]["viewport_culling"] is True


@pytest.mark.viz
def test_test_data_validity():
    """Test that test data files are valid JSON."""
    import json
    from pathlib import Path

    viz_root = Path("viz")

    # Test tiny_concepts.json
    concepts_file = viz_root / "data" / "test" / "tiny_concepts.json"
    with open(concepts_file, encoding="utf-8") as f:
        concepts_data = json.load(f)

    assert "concepts" in concepts_data, "concepts key missing in tiny_concepts.json"
    assert len(concepts_data["concepts"]) == 8, "Expected 8 concepts in test data"

    # Check structure of first concept
    first_concept = concepts_data["concepts"][0]
    assert "concept_id" in first_concept
    assert "term" in first_concept
    assert "definition" in first_concept
    assert "primary" in first_concept["term"]

    # Test tiny_graph.json
    graph_file = viz_root / "data" / "test" / "tiny_graph.json"
    with open(graph_file, encoding="utf-8") as f:
        graph_data = json.load(f)

    assert "nodes" in graph_data, "nodes key missing in tiny_graph.json"
    assert "edges" in graph_data, "edges key missing in tiny_graph.json"
    assert len(graph_data["nodes"]) == 16, "Expected 16 nodes in test data"
    assert len(graph_data["edges"]) == 22, "Expected 22 edges in test data"

    # Check node types
    node_types = {node["type"] for node in graph_data["nodes"]}
    assert "Chunk" in node_types, "No Chunk nodes found"
    assert "Assessment" in node_types, "No Assessment nodes found"
    assert "Concept" in node_types, "No Concept nodes found"

    # Count nodes by type
    chunks = sum(1 for n in graph_data["nodes"] if n["type"] == "Chunk")
    assessments = sum(1 for n in graph_data["nodes"] if n["type"] == "Assessment")
    concepts = sum(1 for n in graph_data["nodes"] if n["type"] == "Concept")

    assert chunks == 5, f"Expected 5 Chunk nodes, got {chunks}"
    assert assessments == 3, f"Expected 3 Assessment nodes, got {assessments}"
    assert concepts == 8, f"Expected 8 Concept nodes, got {concepts}"


@pytest.mark.viz
def test_test_data_schema_validation():
    """Test that test data passes schema validation."""
    import json
    from pathlib import Path

    from src.utils.validation import ValidationError, validate_json

    # Validate tiny_concepts.json
    concepts_file = Path("viz/data/test/tiny_concepts.json")
    with open(concepts_file, encoding="utf-8") as f:
        concepts_data = json.load(f)

    try:
        validate_json(concepts_data, "ConceptDictionary")
    except ValidationError as e:
        pytest.fail(f"tiny_concepts.json validation failed: {e}")

    # Validate tiny_graph.json
    graph_file = Path("viz/data/test/tiny_graph.json")
    with open(graph_file, encoding="utf-8") as f:
        graph_data = json.load(f)

    try:
        validate_json(graph_data, "LearningChunkGraph")
    except ValidationError as e:
        pytest.fail(f"tiny_graph.json validation failed: {e}")


@pytest.mark.viz
def test_test_data_id_conventions():
    """Test that test data follows ID conventions from prompts."""
    import json
    from pathlib import Path
    
    viz_root = Path("viz")
    
    # Check concept IDs follow the pattern
    concepts_file = viz_root / "data" / "test" / "tiny_concepts.json"
    with open(concepts_file, encoding="utf-8") as f:
        concepts_data = json.load(f)
    
    for concept in concepts_data["concepts"]:
        concept_id = concept["concept_id"]
        # Should follow pattern: slug:p:slugified-term
        assert ":" in concept_id, f"Concept ID {concept_id} missing colon separator"
        parts = concept_id.split(":")
        assert len(parts) == 3, f"Concept ID {concept_id} should have 3 parts"
        assert parts[0] == "algo101", f"Concept ID should start with slug 'algo101'"
        assert parts[1] == "p", f"Concept ID should have 'p' as second part"
        assert parts[2].replace("-", "").isalnum(), f"Third part should be alphanumeric with hyphens"
    
    # Check graph node IDs
    graph_file = viz_root / "data" / "test" / "tiny_graph.json"
    with open(graph_file, encoding="utf-8") as f:
        graph_data = json.load(f)
    
    for node in graph_data["nodes"]:
        node_id = node["id"]
        node_type = node["type"]
        
        if node_type == "Chunk":
            assert node_id.startswith("chunk_"), f"Chunk ID should start with 'chunk_'"
        elif node_type == "Assessment":
            assert node_id.startswith("assessment_"), f"Assessment ID should start with 'assessment_'"
        elif node_type == "Concept":
            assert node_id.startswith("algo101:p:"), f"Concept ID should start with 'algo101:p:'"
    
    # Check edge types and weights
    valid_edge_types = {
        "PREREQUISITE", "ELABORATES", "EXAMPLE_OF", "HINT_FORWARD", 
        "REFER_BACK", "PARALLEL", "TESTS", "REVISION_OF", "MENTIONS"
    }
    
    for edge in graph_data["edges"]:
        assert edge["type"] in valid_edge_types, f"Invalid edge type: {edge['type']}"
        assert 0 <= edge["weight"] <= 1, f"Edge weight out of range: {edge['weight']}"
        
        # Check weight ranges according to prompts
        if edge["type"] in ["PREREQUISITE", "TESTS", "REVISION_OF"]:
            assert edge["weight"] >= 0.8, f"{edge['type']} should have weight >= 0.8"
        elif edge["type"] in ["ELABORATES", "EXAMPLE_OF", "PARALLEL"]:
            assert 0.5 <= edge["weight"] <= 0.9, f"{edge['type']} should have weight 0.5-0.9"
        elif edge["type"] in ["HINT_FORWARD", "REFER_BACK"]:
            assert 0.3 <= edge["weight"] <= 0.5, f"{edge['type']} should have weight 0.3-0.5"

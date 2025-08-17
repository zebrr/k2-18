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

    # Check that test validation graphs exist
    test_files = list((viz_root / "data" / "test").glob("test_*_graph.json"))
    assert len(test_files) > 0, "No test_*_graph.json files found for validation"
    
    # Check that at least one expected test graph exists
    expected_test_graphs = ["test_line_graph.json", "test_cycle_graph.json", "test_bridge_graph.json"]
    found_graphs = [(viz_root / "data" / "test" / name).exists() for name in expected_test_graphs]
    assert any(found_graphs), "No expected test graphs found"

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
    assert config["demo_path"]["strategy"] in [1, 2, 3], "Strategy should be 1, 2, or 3"
    assert config["graph2html"]["output_filename"] == "knowledge_graph.html"
    assert config["visualization"]["initial_layout"] == "cose-bilkent"
    assert config["colors"]["theme"] == "modern"
    assert len(config["colors"]["cluster_palette"]) == 12
    assert config["ui"]["sidebar_width"] == 320
    assert config["filters"]["show_chunks"] is True
    assert config["performance"]["viewport_culling"] is True


@pytest.mark.viz
def test_validation_test_data():
    """Test that validation test graphs are valid JSON."""
    import json
    from pathlib import Path

    viz_root = Path("viz")
    test_dir = viz_root / "data" / "test"
    
    # Find all test graphs for validation
    test_graphs = list(test_dir.glob("test_*_graph.json"))
    assert len(test_graphs) > 0, "No test graphs found for validation"
    
    # Check each test graph
    for graph_file in test_graphs:
        with open(graph_file, encoding="utf-8") as f:
            graph_data = json.load(f)
        
        # Basic structure checks
        assert "nodes" in graph_data, f"nodes key missing in {graph_file.name}"
        assert "edges" in graph_data, f"edges key missing in {graph_file.name}"
        assert len(graph_data["nodes"]) > 0, f"No nodes in {graph_file.name}"
        
        # Check node structure
        for node in graph_data["nodes"]:
            assert "id" in node, f"Node missing id in {graph_file.name}"
            assert "type" in node, f"Node missing type in {graph_file.name}"
        
        # Check edge structure  
        for edge in graph_data["edges"]:
            assert "source" in edge, f"Edge missing source in {graph_file.name}"
            assert "target" in edge, f"Edge missing target in {graph_file.name}"
            assert "type" in edge, f"Edge missing type in {graph_file.name}"
            assert "weight" in edge, f"Edge missing weight in {graph_file.name}"


@pytest.mark.viz
def test_validation_graphs_schema_compliance():
    """Test that validation test graphs pass schema validation."""
    import json
    from pathlib import Path

    from src.utils.validation import ValidationError, validate_json

    test_dir = Path("viz/data/test")
    
    # Validate all test graphs
    for graph_file in test_dir.glob("test_*_graph.json"):
        with open(graph_file, encoding="utf-8") as f:
            graph_data = json.load(f)
        
        try:
            validate_json(graph_data, "LearningChunkGraph")
        except ValidationError as e:
            pytest.fail(f"{graph_file.name} validation failed: {e}")


@pytest.mark.viz
def test_edge_type_conventions():
    """Test that test graphs follow edge type conventions."""
    import json
    from pathlib import Path
    
    test_dir = Path("viz/data/test")
    
    # Valid edge types from specifications
    valid_edge_types = {
        "PREREQUISITE", "ELABORATES", "EXAMPLE_OF", "HINT_FORWARD", 
        "REFER_BACK", "PARALLEL", "TESTS", "REVISION_OF", "MENTIONS"
    }
    
    # Check all test graphs
    for graph_file in test_dir.glob("test_*_graph.json"):
        with open(graph_file, encoding="utf-8") as f:
            graph_data = json.load(f)
        
        for edge in graph_data.get("edges", []):
            # Check edge type validity
            edge_type = edge.get("type")
            assert edge_type in valid_edge_types, f"Invalid edge type '{edge_type}' in {graph_file.name}"
            
            # Check weight range
            weight = edge.get("weight", 1.0)
            assert 0 <= weight <= 1, f"Edge weight {weight} out of range [0,1] in {graph_file.name}"

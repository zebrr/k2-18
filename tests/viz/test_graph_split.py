"""Tests for viz.graph_split module."""

import json

import pytest

from viz.graph_split import (
    create_cluster_metadata,
    extract_cluster,
    identify_clusters,
    sort_nodes,
)

# Test Fixtures


@pytest.fixture
def sample_graph():
    """Sample graph with 3 clusters for testing."""
    return {
        "nodes": [
            {
                "id": "node_1",
                "type": "Chunk",
                "text": "Content 1",
                "cluster_id": 0,
                "node_offset": 0,
            },
            {
                "id": "node_2",
                "type": "Concept",
                "text": "Concept A",
                "cluster_id": 0,
                "node_offset": 10,
            },
            {
                "id": "node_3",
                "type": "Chunk",
                "text": "Content 2",
                "cluster_id": 1,
                "node_offset": 20,
            },
            {
                "id": "node_4",
                "type": "Concept",
                "text": "Concept B",
                "cluster_id": 1,
                "node_offset": 30,
            },
            {
                "id": "node_5",
                "type": "Assessment",
                "text": "Test",
                "cluster_id": 2,
                "node_offset": 40,
            },
            {
                "id": "node_6",
                "type": "Concept",
                "text": "Concept C",
                "cluster_id": 0,
                "node_offset": 50,
            },
        ],
        "edges": [
            {"source": "node_1", "target": "node_2", "type": "PREREQUISITE"},  # Within cluster 0
            {"source": "node_2", "target": "node_6", "type": "ELABORATES"},  # Within cluster 0
            {"source": "node_1", "target": "node_3", "type": "ELABORATES"},  # Inter-cluster 0→1
            {"source": "node_3", "target": "node_4", "type": "PREREQUISITE"},  # Within cluster 1
            {"source": "node_4", "target": "node_5", "type": "TESTS"},  # Inter-cluster 1→2
            {"source": "node_5", "target": "node_1", "type": "REFER_BACK"},  # Inter-cluster 2→0
        ],
        "_meta": {"title": "Test Graph", "version": "1.0"},
    }


@pytest.fixture
def single_node_cluster_graph():
    """Graph with a single-node cluster."""
    return {
        "nodes": [
            {"id": "node_1", "type": "Chunk", "text": "Content", "cluster_id": 0, "node_offset": 0},
            {
                "id": "node_2",
                "type": "Chunk",
                "text": "Content",
                "cluster_id": 1,
                "node_offset": 10,
            },
            {
                "id": "node_3",
                "type": "Chunk",
                "text": "Content",
                "cluster_id": 1,
                "node_offset": 20,
            },
        ],
        "edges": [{"source": "node_2", "target": "node_3", "type": "ELABORATES"}],
        "_meta": {"title": "Single Node Test"},
    }


@pytest.fixture
def isolated_cluster_graph():
    """Graph with an isolated cluster (no inter-cluster edges)."""
    return {
        "nodes": [
            {"id": "node_1", "type": "Chunk", "text": "Content", "cluster_id": 0, "node_offset": 0},
            {
                "id": "node_2",
                "type": "Chunk",
                "text": "Content",
                "cluster_id": 0,
                "node_offset": 10,
            },
            {
                "id": "node_3",
                "type": "Chunk",
                "text": "Content",
                "cluster_id": 1,
                "node_offset": 20,
            },
        ],
        "edges": [
            {"source": "node_1", "target": "node_2", "type": "ELABORATES"},  # Within cluster 0
            # No edges involving node_3 (cluster 1 is isolated)
        ],
        "_meta": {"title": "Isolated Cluster Test"},
    }


# Unit Tests


def test_identify_clusters(sample_graph):
    """Test finding unique cluster IDs, sorted."""
    import logging

    logger = logging.getLogger("test")

    clusters = identify_clusters(sample_graph, logger)

    assert clusters == [0, 1, 2]
    assert isinstance(clusters, list)
    assert all(isinstance(c, int) for c in clusters)


def test_identify_clusters_empty_graph():
    """Test empty graph handling."""
    import logging

    logger = logging.getLogger("test")

    graph = {"nodes": [], "edges": []}
    clusters = identify_clusters(graph, logger)

    assert clusters == []


def test_identify_clusters_no_cluster_id():
    """Test graph without cluster_id fields."""
    import logging

    logger = logging.getLogger("test")

    graph = {
        "nodes": [
            {"id": "node_1", "type": "Chunk", "text": "Content", "node_offset": 0},
            {"id": "node_2", "type": "Chunk", "text": "Content", "node_offset": 10},
        ],
        "edges": [],
    }
    clusters = identify_clusters(graph, logger)

    assert clusters == []


def test_sort_nodes():
    """Test node sorting: Concepts first (by id), then others (preserve order)."""
    nodes = [
        {"id": "chunk_1", "type": "Chunk", "text": "A"},
        {"id": "concept_z", "type": "Concept", "text": "Z"},
        {"id": "assessment_1", "type": "Assessment", "text": "Test"},
        {"id": "concept_a", "type": "Concept", "text": "A"},
        {"id": "chunk_2", "type": "Chunk", "text": "B"},
    ]

    sorted_nodes = sort_nodes(nodes)

    # Check Concepts are first
    assert sorted_nodes[0]["type"] == "Concept"
    assert sorted_nodes[1]["type"] == "Concept"

    # Check Concepts sorted by id
    assert sorted_nodes[0]["id"] == "concept_a"
    assert sorted_nodes[1]["id"] == "concept_z"

    # Check Others preserve order
    assert sorted_nodes[2]["id"] == "chunk_1"
    assert sorted_nodes[3]["id"] == "assessment_1"
    assert sorted_nodes[4]["id"] == "chunk_2"


def test_sort_nodes_all_concepts():
    """Test sorting when all nodes are Concepts."""
    nodes = [
        {"id": "concept_c", "type": "Concept", "text": "C"},
        {"id": "concept_a", "type": "Concept", "text": "A"},
        {"id": "concept_b", "type": "Concept", "text": "B"},
    ]

    sorted_nodes = sort_nodes(nodes)

    assert [n["id"] for n in sorted_nodes] == ["concept_a", "concept_b", "concept_c"]


def test_sort_nodes_no_concepts():
    """Test sorting when no Concepts present."""
    nodes = [
        {"id": "chunk_1", "type": "Chunk", "text": "A"},
        {"id": "chunk_2", "type": "Chunk", "text": "B"},
        {"id": "assessment_1", "type": "Assessment", "text": "Test"},
    ]

    sorted_nodes = sort_nodes(nodes)

    # Order preserved (no Concepts to sort first)
    assert [n["id"] for n in sorted_nodes] == ["chunk_1", "chunk_2", "assessment_1"]


def test_extract_cluster(sample_graph):
    """Test cluster extraction logic."""
    import logging

    logger = logging.getLogger("test")

    # Extract cluster 0 (contains node_1, node_2, node_6)
    cluster_graph, node_count, edge_count, inter_cluster_count = extract_cluster(
        sample_graph, 0, logger
    )

    # Check node count
    assert node_count == 3
    node_ids = {n["id"] for n in cluster_graph["nodes"]}
    assert node_ids == {"node_1", "node_2", "node_6"}

    # Check edge count (only edges within cluster 0)
    # Edges within cluster 0: node_1→node_2, node_2→node_6
    assert edge_count == 2

    # Check inter-cluster edges
    # Inter-cluster edges FROM cluster 0: node_1→node_3 (1 edge)
    # Inter-cluster edges TO cluster 0: node_5→node_1 (1 edge)
    # Total: 2
    assert inter_cluster_count == 2


def test_extract_cluster_statistics(sample_graph):
    """Test statistics for different clusters."""
    import logging

    logger = logging.getLogger("test")

    # Cluster 1 (node_3, node_4)
    _, node_count, edge_count, inter_cluster_count = extract_cluster(sample_graph, 1, logger)

    assert node_count == 2
    # Edges within cluster 1: node_3→node_4 (1 edge)
    assert edge_count == 1
    # Inter-cluster: node_1→node_3 (to cluster 1), node_4→node_5 (from cluster 1)
    assert inter_cluster_count == 2

    # Cluster 2 (node_5)
    _, node_count, edge_count, inter_cluster_count = extract_cluster(sample_graph, 2, logger)

    assert node_count == 1
    # No edges within cluster 2
    assert edge_count == 0
    # Inter-cluster: node_4→node_5 (to cluster 2), node_5→node_1 (from cluster 2)
    assert inter_cluster_count == 2


def test_inter_cluster_edges_calculation():
    """Test XOR logic for inter-cluster edge counting."""
    import logging

    logger = logging.getLogger("test")

    graph = {
        "nodes": [
            {"id": "n1", "type": "Chunk", "text": "A", "cluster_id": 0, "node_offset": 0},
            {"id": "n2", "type": "Chunk", "text": "B", "cluster_id": 0, "node_offset": 10},
            {"id": "n3", "type": "Chunk", "text": "C", "cluster_id": 1, "node_offset": 20},
        ],
        "edges": [
            {"source": "n1", "target": "n2", "type": "ELABORATES"},  # Within cluster 0
            {"source": "n1", "target": "n3", "type": "PREREQUISITE"},  # Inter-cluster 0→1
            {"source": "n2", "target": "n3", "type": "ELABORATES"},  # Inter-cluster 0→1
            {"source": "n3", "target": "n1", "type": "REFER_BACK"},  # Inter-cluster 1→0
        ],
    }

    # Extract cluster 0
    _, _, _, inter_cluster_count = extract_cluster(graph, 0, logger)

    # Inter-cluster edges for cluster 0:
    # - n1→n3 (source in cluster 0, target not)
    # - n2→n3 (source in cluster 0, target not)
    # - n3→n1 (source not in cluster 0, target in)
    # Total: 3
    assert inter_cluster_count == 3


def test_create_cluster_metadata():
    """Test metadata creation with subtitle format."""
    metadata = create_cluster_metadata(
        cluster_id=5, node_count=42, edge_count=87, original_title="Original Graph"
    )

    assert metadata["title"] == "Original Graph"
    assert metadata["subtitle"] == "Cluster 5 | Nodes 42 | Edges 87"
    assert len(metadata) == 2  # Only title and subtitle


# Integration Tests


def test_full_split_flow(tmp_path, sample_graph):
    """Test complete split flow: load → split → save → validate."""
    import logging

    from src.utils.validation import validate_json
    from viz.graph_split import save_cluster_graph

    logger = logging.getLogger("test")

    # Create input file
    input_file = tmp_path / "test_graph.json"
    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(sample_graph, f)

    # Extract and save each cluster
    clusters = identify_clusters(sample_graph, logger)

    for cluster_id in clusters:
        cluster_graph, node_count, edge_count, _ = extract_cluster(sample_graph, cluster_id, logger)

        # Skip single-node clusters
        if node_count == 1:
            continue

        # Sort nodes
        cluster_graph["nodes"] = sort_nodes(cluster_graph["nodes"])

        # Create metadata
        cluster_graph["_meta"] = create_cluster_metadata(
            cluster_id, node_count, edge_count, sample_graph["_meta"]["title"]
        )

        # Save
        save_cluster_graph(cluster_graph, cluster_id, tmp_path, logger)

    # Verify files exist
    cluster_files = list(tmp_path.glob("LearningChunkGraph_cluster_*.json"))
    assert len(cluster_files) == 2  # Cluster 0 and 1 (cluster 2 has 1 node, skipped)

    # Validate each file
    for cluster_file in cluster_files:
        with open(cluster_file, encoding="utf-8") as f:
            data = json.load(f)

        # Validate against schema
        validate_json(data, "LearningChunkGraph")

        # Check metadata
        assert "_meta" in data
        assert "title" in data["_meta"]
        assert "subtitle" in data["_meta"]
        assert "Cluster" in data["_meta"]["subtitle"]
        assert "Nodes" in data["_meta"]["subtitle"]
        assert "Edges" in data["_meta"]["subtitle"]


def test_full_split_validation(tmp_path, sample_graph):
    """Test schema validation of output files."""
    import logging

    from src.utils.validation import validate_json
    from viz.graph_split import save_cluster_graph

    logger = logging.getLogger("test")

    # Extract cluster 0
    cluster_graph, node_count, edge_count, _ = extract_cluster(sample_graph, 0, logger)

    # Add metadata
    cluster_graph["_meta"] = create_cluster_metadata(0, node_count, edge_count, "Test Graph")

    # Save
    output_file = tmp_path / "LearningChunkGraph_cluster_0.json"
    save_cluster_graph(cluster_graph, 0, tmp_path, logger)

    # Load and validate
    with open(output_file, encoding="utf-8") as f:
        loaded_data = json.load(f)

    # Should not raise
    validate_json(loaded_data, "LearningChunkGraph")

    # Check structure
    assert "nodes" in loaded_data
    assert "edges" in loaded_data
    assert "_meta" in loaded_data


def test_metadata_in_output(tmp_path, sample_graph):
    """Test metadata format in saved files."""
    import logging

    from viz.graph_split import save_cluster_graph

    logger = logging.getLogger("test")

    cluster_graph, node_count, edge_count, _ = extract_cluster(sample_graph, 0, logger)

    cluster_graph["_meta"] = create_cluster_metadata(
        cluster_id=0,
        node_count=node_count,
        edge_count=edge_count,
        original_title="My Graph",
    )

    save_cluster_graph(cluster_graph, 0, tmp_path, logger)

    # Load and check metadata
    output_file = tmp_path / "LearningChunkGraph_cluster_0.json"
    with open(output_file, encoding="utf-8") as f:
        data = json.load(f)

    assert data["_meta"]["title"] == "My Graph"
    assert data["_meta"]["subtitle"] == f"Cluster 0 | Nodes {node_count} | Edges {edge_count}"

    # OLD metadata should NOT be present (complete replacement)
    assert "version" not in data["_meta"]


# Boundary Cases


def test_single_node_cluster_skipped(single_node_cluster_graph):
    """Test single-node cluster detection."""
    import logging

    logger = logging.getLogger("test")

    # Extract cluster 0 (single node)
    cluster_graph, node_count, edge_count, inter_cluster_count = extract_cluster(
        single_node_cluster_graph, 0, logger
    )

    assert node_count == 1
    assert edge_count == 0
    # Should be skipped in main() - verified by integration test


def test_isolated_cluster(isolated_cluster_graph):
    """Test cluster with no inter-cluster edges."""
    import logging

    logger = logging.getLogger("test")

    # Extract cluster 1 (isolated, no edges)
    _, node_count, edge_count, inter_cluster_count = extract_cluster(
        isolated_cluster_graph, 1, logger
    )

    assert node_count == 1
    assert edge_count == 0
    assert inter_cluster_count == 0  # No inter-cluster edges


def test_all_nodes_one_cluster():
    """Test edge case with all nodes in single cluster."""
    import logging

    logger = logging.getLogger("test")

    graph = {
        "nodes": [
            {"id": "n1", "type": "Chunk", "text": "A", "cluster_id": 0, "node_offset": 0},
            {"id": "n2", "type": "Chunk", "text": "B", "cluster_id": 0, "node_offset": 10},
            {"id": "n3", "type": "Chunk", "text": "C", "cluster_id": 0, "node_offset": 20},
        ],
        "edges": [
            {"source": "n1", "target": "n2", "type": "ELABORATES"},
            {"source": "n2", "target": "n3", "type": "PREREQUISITE"},
        ],
    }

    clusters = identify_clusters(graph, logger)
    assert clusters == [0]

    _, node_count, edge_count, inter_cluster_count = extract_cluster(graph, 0, logger)

    assert node_count == 3
    assert edge_count == 2
    assert inter_cluster_count == 0  # No other clusters, so no inter-cluster edges

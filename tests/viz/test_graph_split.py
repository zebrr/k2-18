"""Tests for viz.graph_split module."""

import json

import pytest

from viz.graph_split import (
    build_concept_cluster_map,
    create_cluster_dictionary,
    create_cluster_metadata,
    extract_cluster,
    extract_cluster_concepts,
    get_filename_padding,
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

        # Save (padding=1 for single-digit cluster IDs)
        save_cluster_graph(cluster_graph, cluster_id, tmp_path, 1, logger)

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

    # Save (padding=1 for single-digit cluster ID)
    output_file = tmp_path / "LearningChunkGraph_cluster_0.json"
    save_cluster_graph(cluster_graph, 0, tmp_path, 1, logger)

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

    save_cluster_graph(cluster_graph, 0, tmp_path, 1, logger)

    # Load and check metadata
    output_file = tmp_path / "LearningChunkGraph_cluster_0.json"
    with open(output_file, encoding="utf-8") as f:
        data = json.load(f)

    assert data["_meta"]["title"] == "My Graph"
    assert data["_meta"]["subtitle"] == f"Cluster 0 | Nodes {node_count} | Edges {edge_count}"

    # OLD metadata should NOT be present (complete replacement)
    assert "version" not in data["_meta"]

    # Check inter_cluster_links format if present
    if "inter_cluster_links" in data["_meta"]:
        links = data["_meta"]["inter_cluster_links"]
        assert "incoming" in links
        assert "outgoing" in links

        # Verify structure of links
        for direction in ["incoming", "outgoing"]:
            for link in links[direction]:
                assert "source" in link
                assert "source_text" in link
                assert "source_type" in link
                assert link["source_type"] in ["Concept", "Chunk", "Assessment"]
                assert "source_importance" in link
                assert "target" in link
                assert "target_text" in link
                assert "target_type" in link
                assert link["target_type"] in ["Concept", "Chunk", "Assessment"]
                assert "target_importance" in link
                assert "type" in link
                assert link["type"] in ["PREREQUISITE", "ELABORATES", "EXAMPLE_OF", "TESTS"]
                assert "weight" in link


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


# New Tests for Inter-Cluster Links


def test_find_inter_cluster_links_four_types():
    """Test finding all 4 allowed link types with proper node filtering."""
    import logging

    from viz.graph_split import find_inter_cluster_links

    logger = logging.getLogger("test")

    # Create test graph with all 4 edge types and different node type combinations
    # Cluster 0: concept_a (Concept), chunk_e (Chunk)
    # Cluster 1: concept_b (Concept), chunk_f (Chunk), assessment_g (Assessment)
    # Cluster 2: concept_d (Concept)
    graph = {
        "nodes": [
            {
                "id": "concept_a",
                "type": "Concept",
                "text": "Concept A",
                "educational_importance": 0.05,
                "node_offset": 0,
            },
            {
                "id": "concept_b",
                "type": "Concept",
                "text": "Concept B",
                "educational_importance": 0.03,
                "node_offset": 10,
            },
            {
                "id": "concept_d",
                "type": "Concept",
                "text": "Concept D",
                "educational_importance": 0.02,
                "node_offset": 20,
            },
            {
                "id": "chunk_e",
                "type": "Chunk",
                "text": "Chunk E",
                "educational_importance": 0.01,
                "node_offset": 30,
            },
            {
                "id": "chunk_f",
                "type": "Chunk",
                "text": "Chunk F",
                "educational_importance": 0.04,
                "node_offset": 40,
            },
            {
                "id": "assessment_g",
                "type": "Assessment",
                "text": "Assessment G",
                "educational_importance": 0.06,
                "node_offset": 50,
            },
            {
                "id": "chunk_i",
                "type": "Chunk",
                "text": "Chunk I",
                "educational_importance": 0.001,
                "node_offset": 60,
            },
            {
                "id": "chunk_j",
                "type": "Chunk",
                "text": "Chunk J",
                "educational_importance": 0.002,
                "node_offset": 70,
            },
        ],
        "edges": [
            # 1. PREREQUISITE: Concept A (c0) -> Concept B (c1) - should be included
            {"source": "concept_a", "target": "concept_b", "type": "PREREQUISITE", "weight": 0.9},
            # 2. ELABORATES: Chunk F (c1) -> Concept D (c2) - should be included (has Concept)
            {"source": "chunk_f", "target": "concept_d", "type": "ELABORATES", "weight": 0.8},
            # 3. EXAMPLE_OF: Chunk E (c0) -> Concept B (c1) - should be included (has Concept)
            {"source": "chunk_e", "target": "concept_b", "type": "EXAMPLE_OF", "weight": 0.7},
            # 4. TESTS: Assessment G (c1) -> Concept D (c2) - should be included (source=Assessment)
            {"source": "assessment_g", "target": "concept_d", "type": "TESTS", "weight": 0.85},
            # 5. PARALLEL: should be IGNORED (wrong edge type)
            {"source": "concept_a", "target": "concept_d", "type": "PARALLEL", "weight": 0.6},
            # 6. EXAMPLE_OF: Chunk I (c0) -> Chunk J (c1) - should be IGNORED (no Concept)
            {"source": "chunk_i", "target": "chunk_j", "type": "EXAMPLE_OF", "weight": 0.5},
        ],
    }

    cluster_map = {
        "concept_a": 0,
        "chunk_e": 0,
        "chunk_i": 0,
        "concept_b": 1,
        "chunk_f": 1,
        "assessment_g": 1,
        "chunk_j": 1,
        "concept_d": 2,
    }

    result = find_inter_cluster_links(graph, cluster_map, logger)

    # Collect all edge types found
    all_types = set()
    for cluster_id in result:
        for link in result[cluster_id]["incoming"] + result[cluster_id]["outgoing"]:
            all_types.add(link["type"])

    # Verify all 4 types are captured
    assert "PREREQUISITE" in all_types
    assert "ELABORATES" in all_types
    assert "EXAMPLE_OF" in all_types
    assert "TESTS" in all_types

    # Verify PARALLEL is not included
    assert "PARALLEL" not in all_types

    # Verify Chunk->Chunk EXAMPLE_OF is not included
    all_sources = []
    for cluster_id in result:
        all_sources.extend([link["source"] for link in result[cluster_id]["incoming"]])
        all_sources.extend([link["source"] for link in result[cluster_id]["outgoing"]])

    assert "chunk_i" not in all_sources  # Chunk->Chunk EXAMPLE_OF ignored

    # Verify source_type and target_type fields are present in all links
    for cluster_id in result:
        for link in result[cluster_id]["incoming"] + result[cluster_id]["outgoing"]:
            assert "source_type" in link
            assert link["source_type"] in ["Concept", "Chunk", "Assessment"]
            assert "target_type" in link
            assert link["target_type"] in ["Concept", "Chunk", "Assessment"]


def test_inter_cluster_links_top3_selection():
    """Test that only top-3 links by source importance are kept."""
    import logging

    from viz.graph_split import find_inter_cluster_links

    logger = logging.getLogger("test")

    # Create graph with 5 concepts in cluster 0, all connecting to 1 concept in cluster 1
    # Different importance values: 0.05, 0.04, 0.03, 0.02, 0.01
    graph = {
        "nodes": [
            {
                "id": "c1",
                "type": "Concept",
                "text": "C1",
                "educational_importance": 0.05,
                "node_offset": 0,
            },
            {
                "id": "c2",
                "type": "Concept",
                "text": "C2",
                "educational_importance": 0.04,
                "node_offset": 10,
            },
            {
                "id": "c3",
                "type": "Concept",
                "text": "C3",
                "educational_importance": 0.03,
                "node_offset": 20,
            },
            {
                "id": "c4",
                "type": "Concept",
                "text": "C4",
                "educational_importance": 0.02,
                "node_offset": 30,
            },
            {
                "id": "c5",
                "type": "Concept",
                "text": "C5",
                "educational_importance": 0.01,
                "node_offset": 40,
            },
            {
                "id": "target",
                "type": "Concept",
                "text": "Target",
                "educational_importance": 0.06,
                "node_offset": 50,
            },
        ],
        "edges": [
            {"source": "c1", "target": "target", "type": "PREREQUISITE", "weight": 0.9},
            {"source": "c2", "target": "target", "type": "ELABORATES", "weight": 0.8},
            {"source": "c3", "target": "target", "type": "PREREQUISITE", "weight": 0.7},
            {"source": "c4", "target": "target", "type": "ELABORATES", "weight": 0.6},
            {"source": "c5", "target": "target", "type": "PREREQUISITE", "weight": 0.5},
        ],
    }

    cluster_map = {"c1": 0, "c2": 0, "c3": 0, "c4": 0, "c5": 0, "target": 1}

    result = find_inter_cluster_links(graph, cluster_map, logger)

    # Cluster 1 should have exactly 3 incoming links
    assert len(result[1]["incoming"]) == 3

    # Top-3 by source_importance: c1 (0.05), c2 (0.04), c3 (0.03)
    incoming = result[1]["incoming"]
    assert incoming[0]["source"] == "c1"
    assert incoming[0]["source_importance"] == 0.05
    assert incoming[1]["source"] == "c2"
    assert incoming[1]["source_importance"] == 0.04
    assert incoming[2]["source"] == "c3"
    assert incoming[2]["source_importance"] == 0.03

    # c4 and c5 should NOT be included (lower importance)
    source_ids = [link["source"] for link in incoming]
    assert "c4" not in source_ids
    assert "c5" not in source_ids

    # Verify no artificial priority by edge type (mix of PREREQUISITE and ELABORATES)
    types = [link["type"] for link in incoming]
    assert "PREREQUISITE" in types  # c1
    assert "ELABORATES" in types  # c2


def test_inter_cluster_links_importance_fields():
    """Test that both source_importance and target_importance are included."""
    import logging

    from viz.graph_split import find_inter_cluster_links

    logger = logging.getLogger("test")

    graph = {
        "nodes": [
            {
                "id": "src",
                "type": "Concept",
                "text": "Source",
                "educational_importance": 0.05,
                "node_offset": 0,
            },
            {
                "id": "tgt",
                "type": "Concept",
                "text": "Target",
                "educational_importance": 0.03,
                "node_offset": 10,
            },
        ],
        "edges": [
            {
                "source": "src",
                "target": "tgt",
                "type": "PREREQUISITE",
                "weight": 0.9,
                "conditions": "Test condition",
            },
        ],
    }

    cluster_map = {"src": 0, "tgt": 1}

    result = find_inter_cluster_links(graph, cluster_map, logger)

    # Check incoming link for cluster 1
    link = result[1]["incoming"][0]

    # Verify all required fields
    assert "source" in link
    assert link["source"] == "src"
    assert "source_text" in link
    assert link["source_text"] == "Source"
    assert "source_importance" in link
    assert link["source_importance"] == 0.05

    assert "target" in link
    assert link["target"] == "tgt"
    assert "target_text" in link
    assert link["target_text"] == "Target"
    assert "target_importance" in link
    assert link["target_importance"] == 0.03

    assert "type" in link
    assert link["type"] == "PREREQUISITE"
    assert "weight" in link
    assert link["weight"] == 0.9
    assert "conditions" in link
    assert link["conditions"] == "Test condition"

    assert "from_cluster" in link
    assert link["from_cluster"] == 0


def test_inter_cluster_links_node_type_filtering():
    """Test that node type requirements are enforced correctly."""
    import logging

    from viz.graph_split import find_inter_cluster_links

    logger = logging.getLogger("test")

    # Create graph with various node type combinations to test filtering rules
    graph = {
        "nodes": [
            # Cluster 0
            {
                "id": "concept_1",
                "type": "Concept",
                "text": "C1",
                "educational_importance": 0.05,
                "node_offset": 0,
            },
            {
                "id": "chunk_1",
                "type": "Chunk",
                "text": "Ch1",
                "educational_importance": 0.04,
                "node_offset": 10,
            },
            {
                "id": "assessment_1",
                "type": "Assessment",
                "text": "A1",
                "educational_importance": 0.03,
                "node_offset": 20,
            },
            # Cluster 1
            {
                "id": "concept_2",
                "type": "Concept",
                "text": "C2",
                "educational_importance": 0.02,
                "node_offset": 30,
            },
            {
                "id": "chunk_2",
                "type": "Chunk",
                "text": "Ch2",
                "educational_importance": 0.01,
                "node_offset": 40,
            },
            {
                "id": "assessment_2",
                "type": "Assessment",
                "text": "A2",
                "educational_importance": 0.06,
                "node_offset": 50,
            },
        ],
        "edges": [
            # Test case 1: TESTS with Assessment source -> INCLUDED
            {"source": "assessment_1", "target": "concept_2", "type": "TESTS", "weight": 0.9},
            # Test case 2: TESTS with Chunk source -> EXCLUDED
            {"source": "chunk_1", "target": "concept_2", "type": "TESTS", "weight": 0.9},
            # Test case 3: PREREQUISITE with Concept->Chunk -> INCLUDED (has Concept)
            {"source": "concept_1", "target": "chunk_2", "type": "PREREQUISITE", "weight": 0.8},
            # Test case 4: PREREQUISITE with Chunk->Chunk -> EXCLUDED (no Concept)
            {"source": "chunk_1", "target": "chunk_2", "type": "PREREQUISITE", "weight": 0.8},
            # Test case 5: EXAMPLE_OF with Assessment->Concept -> INCLUDED (has Concept)
            {"source": "assessment_2", "target": "concept_1", "type": "EXAMPLE_OF", "weight": 0.7},
            # Test case 6: EXAMPLE_OF with Chunk->Assessment -> EXCLUDED (no Concept)
            {"source": "chunk_2", "target": "assessment_1", "type": "EXAMPLE_OF", "weight": 0.7},
        ],
    }

    cluster_map = {
        "concept_1": 0,
        "chunk_1": 0,
        "assessment_1": 0,
        "concept_2": 1,
        "chunk_2": 1,
        "assessment_2": 1,
    }

    result = find_inter_cluster_links(graph, cluster_map, logger)

    # Collect all links
    all_links = []
    for cluster_id in result:
        all_links.extend(result[cluster_id]["incoming"])
        all_links.extend(result[cluster_id]["outgoing"])

    # Test case 1: TESTS with Assessment source -> INCLUDED
    assert any(
        link["source"] == "assessment_1" and link["type"] == "TESTS" for link in all_links
    ), "TESTS with Assessment source should be included"

    # Test case 2: TESTS with Chunk source -> EXCLUDED
    assert not any(link["source"] == "chunk_1" and link["type"] == "TESTS" for link in all_links), (
        "TESTS with Chunk source should be excluded"
    )

    # Test case 3: PREREQUISITE with Concept->Chunk -> INCLUDED
    assert any(
        link["source"] == "concept_1"
        and link["target"] == "chunk_2"
        and link["type"] == "PREREQUISITE"
        for link in all_links
    ), "PREREQUISITE with Concept->Chunk should be included"

    # Test case 4: PREREQUISITE with Chunk->Chunk -> EXCLUDED
    assert not any(
        link["source"] == "chunk_1"
        and link["target"] == "chunk_2"
        and link["type"] == "PREREQUISITE"
        for link in all_links
    ), "PREREQUISITE with Chunk->Chunk should be excluded"

    # Test case 5: EXAMPLE_OF with Assessment->Concept -> INCLUDED
    assert any(
        link["source"] == "assessment_2"
        and link["target"] == "concept_1"
        and link["type"] == "EXAMPLE_OF"
        for link in all_links
    ), "EXAMPLE_OF with Assessment->Concept should be included"

    # Test case 6: EXAMPLE_OF with Chunk->Assessment -> EXCLUDED
    assert not any(
        link["source"] == "chunk_2"
        and link["target"] == "assessment_1"
        and link["type"] == "EXAMPLE_OF"
        for link in all_links
    ), "EXAMPLE_OF with Chunk->Assessment should be excluded"


def test_inter_cluster_links_type_fields():
    """Test that source_type and target_type are correctly populated."""
    import logging

    from viz.graph_split import find_inter_cluster_links

    logger = logging.getLogger("test")

    # Create links with different node type combinations
    graph = {
        "nodes": [
            {
                "id": "concept_a",
                "type": "Concept",
                "text": "CA",
                "educational_importance": 0.05,
                "node_offset": 0,
            },
            {
                "id": "chunk_b",
                "type": "Chunk",
                "text": "ChB",
                "educational_importance": 0.04,
                "node_offset": 10,
            },
            {
                "id": "assessment_c",
                "type": "Assessment",
                "text": "AC",
                "educational_importance": 0.03,
                "node_offset": 20,
            },
            {
                "id": "concept_d",
                "type": "Concept",
                "text": "CD",
                "educational_importance": 0.02,
                "node_offset": 30,
            },
        ],
        "edges": [
            # Concept -> Chunk
            {"source": "concept_a", "target": "chunk_b", "type": "PREREQUISITE", "weight": 0.9},
            # Chunk -> Concept
            {"source": "chunk_b", "target": "concept_d", "type": "ELABORATES", "weight": 0.8},
            # Assessment -> Concept
            {"source": "assessment_c", "target": "concept_d", "type": "TESTS", "weight": 0.85},
        ],
    }

    cluster_map = {
        "concept_a": 0,
        "chunk_b": 1,
        "assessment_c": 1,
        "concept_d": 2,
    }

    result = find_inter_cluster_links(graph, cluster_map, logger)

    # Collect all links
    all_links = []
    for cluster_id in result:
        all_links.extend(result[cluster_id]["incoming"])
        all_links.extend(result[cluster_id]["outgoing"])

    # Find Concept -> Chunk link
    concept_chunk_link = next(
        link for link in all_links if link["source"] == "concept_a" and link["target"] == "chunk_b"
    )
    assert concept_chunk_link["source_type"] == "Concept"
    assert concept_chunk_link["target_type"] == "Chunk"

    # Find Chunk -> Concept link
    chunk_concept_link = next(
        link for link in all_links if link["source"] == "chunk_b" and link["target"] == "concept_d"
    )
    assert chunk_concept_link["source_type"] == "Chunk"
    assert chunk_concept_link["target_type"] == "Concept"

    # Find Assessment -> Concept link
    assessment_concept_link = next(
        link
        for link in all_links
        if link["source"] == "assessment_c" and link["target"] == "concept_d"
    )
    assert assessment_concept_link["source_type"] == "Assessment"
    assert assessment_concept_link["target_type"] == "Concept"


def test_inter_cluster_links_in_metadata(tmp_path):
    """Test that inter-cluster links appear in cluster metadata."""
    import json
    import logging

    from viz.graph_split import (
        create_cluster_metadata,
        extract_cluster,
        find_inter_cluster_links,
        save_cluster_graph,
        sort_nodes,
    )

    logger = logging.getLogger("test")

    # Create test graph with inter-cluster links
    graph = {
        "nodes": [
            {
                "id": "concept_a",
                "type": "Concept",
                "text": "Concept A",
                "educational_importance": 0.05,
                "cluster_id": 0,
                "node_offset": 0,
            },
            {
                "id": "concept_b",
                "type": "Concept",
                "text": "Concept B",
                "educational_importance": 0.03,
                "cluster_id": 1,
                "node_offset": 10,
            },
            {
                "id": "chunk_1",
                "type": "Chunk",
                "text": "Content",
                "cluster_id": 0,
                "node_offset": 20,
            },
        ],
        "edges": [
            {"source": "concept_a", "target": "concept_b", "type": "PREREQUISITE", "weight": 0.9},
        ],
        "_meta": {"title": "Test Graph"},
    }

    cluster_map = {"concept_a": 0, "concept_b": 1, "chunk_1": 0}

    # Find inter-cluster links
    all_inter_links = find_inter_cluster_links(graph, cluster_map, logger)

    # Extract cluster 0
    cluster_graph, node_count, edge_count, _ = extract_cluster(graph, 0, logger)
    cluster_graph["nodes"] = sort_nodes(cluster_graph["nodes"])

    # Get inter-cluster links for cluster 0
    inter_links = all_inter_links.get(0, {"incoming": [], "outgoing": []})

    # Create metadata with inter-cluster links
    cluster_graph["_meta"] = create_cluster_metadata(
        cluster_id=0,
        node_count=node_count,
        edge_count=edge_count,
        original_title="Test Graph",
        inter_cluster_links=inter_links,
    )

    # Save (padding=1 for single-digit cluster ID)
    save_cluster_graph(cluster_graph, 0, tmp_path, 1, logger)

    # Load and verify
    output_file = tmp_path / "LearningChunkGraph_cluster_0.json"
    with open(output_file, encoding="utf-8") as f:
        data = json.load(f)

    # Verify structure
    assert "_meta" in data
    assert "inter_cluster_links" in data["_meta"]

    links = data["_meta"]["inter_cluster_links"]
    assert "incoming" in links
    assert "outgoing" in links

    # Verify cluster 0 has 1 outgoing link (concept_a -> concept_b)
    assert len(links["outgoing"]) == 1
    assert links["outgoing"][0]["source"] == "concept_a"
    assert links["outgoing"][0]["target"] == "concept_b"
    assert links["outgoing"][0]["type"] == "PREREQUISITE"

    # Verify both importance fields present
    assert "source_importance" in links["outgoing"][0]
    assert links["outgoing"][0]["source_importance"] == 0.05
    assert "target_importance" in links["outgoing"][0]
    assert links["outgoing"][0]["target_importance"] == 0.03


# New Tests for Cluster Dictionary and Zero-Padding


def test_get_filename_padding():
    """Test padding calculation for various cluster ID ranges."""
    # Single digit (0-9) -> padding 1
    assert get_filename_padding([0, 1, 2, 3]) == 1
    assert get_filename_padding([9]) == 1

    # Two digits (10-99) -> padding 2
    assert get_filename_padding([0, 1, 10]) == 2
    assert get_filename_padding(list(range(16))) == 2  # 0-15
    assert get_filename_padding([0, 99]) == 2

    # Three digits (100-999) -> padding 3
    assert get_filename_padding([0, 100]) == 3
    assert get_filename_padding(list(range(101))) == 3  # 0-100


def test_get_filename_padding_empty():
    """Test padding returns 1 for empty list."""
    assert get_filename_padding([]) == 1


def test_extract_cluster_concepts():
    """Test extraction of concepts referenced by cluster nodes."""
    import logging

    logger = logging.getLogger("test")

    # Sample cluster nodes with concepts field
    cluster_nodes = [
        {"id": "chunk_1", "type": "Chunk", "concepts": ["concept_a", "concept_b"]},
        {"id": "chunk_2", "type": "Chunk", "concepts": ["concept_b", "concept_c"]},
        {"id": "chunk_3", "type": "Chunk", "concepts": []},  # No concepts
    ]

    # Sample dictionary data
    concepts_data = {
        "concepts": [
            {"concept_id": "concept_a", "term": {"primary": "A"}, "definition": "Def A"},
            {"concept_id": "concept_b", "term": {"primary": "B"}, "definition": "Def B"},
            {"concept_id": "concept_c", "term": {"primary": "C"}, "definition": "Def C"},
            {"concept_id": "concept_d", "term": {"primary": "D"}, "definition": "Def D"},
        ]
    }

    concepts_list, count = extract_cluster_concepts(cluster_nodes, concepts_data, {}, logger)

    # Should extract 3 unique concepts: a, b, c (sorted)
    assert count == 3
    assert len(concepts_list) == 3

    # Verify sorted order by concept_id
    assert concepts_list[0]["concept_id"] == "concept_a"
    assert concepts_list[1]["concept_id"] == "concept_b"
    assert concepts_list[2]["concept_id"] == "concept_c"

    # concept_d should NOT be included (not referenced)
    concept_ids = [c["concept_id"] for c in concepts_list]
    assert "concept_d" not in concept_ids


def test_extract_cluster_concepts_missing(caplog):
    """Test warning when concept referenced but not found in dictionary."""
    import logging

    logger = logging.getLogger("test")
    logger.setLevel(logging.WARNING)

    cluster_nodes = [
        {"id": "chunk_1", "type": "Chunk", "concepts": ["concept_missing", "concept_a"]},
    ]

    concepts_data = {
        "concepts": [
            {"concept_id": "concept_a", "term": {"primary": "A"}, "definition": "Def A"},
        ]
    }

    with caplog.at_level(logging.WARNING):
        concepts_list, count = extract_cluster_concepts(cluster_nodes, concepts_data, {}, logger)

    # Should still return the found concept
    assert count == 1
    assert concepts_list[0]["concept_id"] == "concept_a"

    # Should have logged a warning for missing concept
    assert "concept_missing not found in dictionary" in caplog.text


def test_create_cluster_dictionary():
    """Test cluster dictionary structure and metadata."""
    concepts_list = [
        {"concept_id": "concept_a", "term": {"primary": "A"}, "definition": "Def A"},
        {"concept_id": "concept_b", "term": {"primary": "B"}, "definition": "Def B"},
    ]

    result = create_cluster_dictionary(
        cluster_id=5,
        concepts_list=concepts_list,
        original_title="Test Knowledge Graph",
    )

    # Verify structure
    assert "_meta" in result
    assert "concepts" in result

    # Verify metadata
    assert result["_meta"]["title"] == "Test Knowledge Graph"
    assert result["_meta"]["cluster_id"] == 5
    assert result["_meta"]["concepts_used"] == 2

    # Verify concepts
    assert len(result["concepts"]) == 2
    assert result["concepts"][0]["concept_id"] == "concept_a"


def test_build_concept_cluster_map():
    """Test building concept_id -> cluster_id mapping."""
    graph_data = {
        "nodes": [
            {"id": "concept_1", "type": "Concept", "cluster_id": 0},
            {"id": "concept_2", "type": "Concept", "cluster_id": 1},
            {"id": "chunk_1", "type": "Chunk", "cluster_id": 0},  # Should be excluded
            {"id": "concept_3", "type": "Concept"},  # No cluster_id - should be excluded
        ]
    }

    result = build_concept_cluster_map(graph_data)

    assert result == {"concept_1": 0, "concept_2": 1}
    assert "chunk_1" not in result
    assert "concept_3" not in result


def test_extract_cluster_concepts_with_cluster_id():
    """Test that cluster_id is added to concepts from graph."""
    import logging

    logger = logging.getLogger("test")

    cluster_nodes = [{"id": "chunk_1", "concepts": ["concept_a", "concept_b"]}]
    concepts_data = {
        "concepts": [
            {"concept_id": "concept_a", "term": {"primary": "A"}, "definition": "Def A"},
            {"concept_id": "concept_b", "term": {"primary": "B"}, "definition": "Def B"},
        ]
    }
    concept_cluster_map = {"concept_a": 5, "concept_b": 3}

    result, count = extract_cluster_concepts(
        cluster_nodes, concepts_data, concept_cluster_map, logger
    )

    assert count == 2
    assert result[0]["cluster_id"] == 5  # concept_a -> cluster 5
    assert result[1]["cluster_id"] == 3  # concept_b -> cluster 3


def test_extract_cluster_concepts_missing_in_graph():
    """Test that cluster_id is null when concept not found in graph."""
    import logging

    logger = logging.getLogger("test")

    cluster_nodes = [{"id": "chunk_1", "concepts": ["concept_a", "concept_b"]}]
    concepts_data = {
        "concepts": [
            {"concept_id": "concept_a", "term": {"primary": "A"}, "definition": "Def A"},
            {"concept_id": "concept_b", "term": {"primary": "B"}, "definition": "Def B"},
        ]
    }
    # concept_b not in map
    concept_cluster_map = {"concept_a": 5}

    result, count = extract_cluster_concepts(
        cluster_nodes, concepts_data, concept_cluster_map, logger
    )

    assert count == 2
    assert result[0]["cluster_id"] == 5
    assert result[1]["cluster_id"] is None  # concept_b not in graph


def test_extract_cluster_concepts_preserves_existing_cluster_id():
    """Test that existing cluster_id is not overwritten (forward-compatible)."""
    import logging

    logger = logging.getLogger("test")

    cluster_nodes = [{"id": "chunk_1", "concepts": ["concept_a"]}]
    concepts_data = {
        "concepts": [
            {
                "concept_id": "concept_a",
                "cluster_id": 99,  # Already has cluster_id
                "term": {"primary": "A"},
                "definition": "Def A",
            },
        ]
    }
    concept_cluster_map = {"concept_a": 5}  # Different value

    result, count = extract_cluster_concepts(
        cluster_nodes, concepts_data, concept_cluster_map, logger
    )

    assert count == 1
    assert result[0]["cluster_id"] == 99  # Preserved original, not 5


def test_dictionary_files_created(tmp_path):
    """Test that _dict.json files are created alongside cluster graphs."""
    import logging

    from viz.graph_split import save_cluster_dictionary

    logger = logging.getLogger("test")

    cluster_dict = {
        "_meta": {
            "title": "Test Graph",
            "cluster_id": 0,
            "concepts_used": 2,
        },
        "concepts": [
            {"concept_id": "concept_a", "term": {"primary": "A"}, "definition": "Def A"},
            {"concept_id": "concept_b", "term": {"primary": "B"}, "definition": "Def B"},
        ],
    }

    # Save with padding=2
    save_cluster_dictionary(cluster_dict, 0, tmp_path, 2, logger)

    # Verify file exists with correct zero-padded name
    output_file = tmp_path / "LearningChunkGraph_cluster_00_dict.json"
    assert output_file.exists()

    # Verify content
    with open(output_file, encoding="utf-8") as f:
        data = json.load(f)

    assert data["_meta"]["cluster_id"] == 0
    assert data["_meta"]["concepts_used"] == 2
    assert len(data["concepts"]) == 2


def test_zero_padding_in_filenames(tmp_path):
    """Test that filenames use consistent zero-padding."""
    import logging

    from viz.graph_split import save_cluster_dictionary, save_cluster_graph

    logger = logging.getLogger("test")

    # Create minimal cluster graph
    cluster_graph = {
        "_meta": {"title": "Test", "subtitle": "Cluster 3"},
        "nodes": [
            {"id": "n1", "type": "Chunk", "text": "A", "node_offset": 0},
            {"id": "n2", "type": "Chunk", "text": "B", "node_offset": 10},
        ],
        "edges": [{"source": "n1", "target": "n2", "type": "ELABORATES"}],
    }

    # Create minimal cluster dictionary
    cluster_dict = {
        "_meta": {"title": "Test", "cluster_id": 3, "concepts_used": 0},
        "concepts": [],
    }

    # Save with padding=2 (for clusters 0-99)
    save_cluster_graph(cluster_graph, 3, tmp_path, 2, logger)
    save_cluster_dictionary(cluster_dict, 3, tmp_path, 2, logger)

    # Verify zero-padded filenames
    graph_file = tmp_path / "LearningChunkGraph_cluster_03.json"
    dict_file = tmp_path / "LearningChunkGraph_cluster_03_dict.json"

    assert graph_file.exists(), "Graph file should have zero-padded name"
    assert dict_file.exists(), "Dictionary file should have zero-padded name"

    # Verify non-padded files don't exist
    wrong_graph = tmp_path / "LearningChunkGraph_cluster_3.json"
    wrong_dict = tmp_path / "LearningChunkGraph_cluster_3_dict.json"

    assert not wrong_graph.exists(), "Non-padded graph filename should not exist"
    assert not wrong_dict.exists(), "Non-padded dictionary filename should not exist"


def test_cluster_with_no_concepts(tmp_path):
    """Test cluster dictionary creation when nodes have no concepts."""
    import logging

    from viz.graph_split import save_cluster_dictionary

    logger = logging.getLogger("test")

    # Cluster nodes with empty concepts field
    cluster_nodes = [
        {"id": "chunk_1", "type": "Chunk", "concepts": []},
        {"id": "chunk_2", "type": "Chunk", "concepts": []},
    ]

    concepts_data = {
        "concepts": [
            {"concept_id": "concept_a", "term": {"primary": "A"}, "definition": "Def A"},
        ]
    }

    # Extract (should return empty list)
    concepts_list, count = extract_cluster_concepts(cluster_nodes, concepts_data, {}, logger)

    assert count == 0
    assert concepts_list == []

    # Create and save dictionary with empty concepts
    cluster_dict = create_cluster_dictionary(
        cluster_id=7,
        concepts_list=concepts_list,
        original_title="Empty Concepts Test",
    )

    save_cluster_dictionary(cluster_dict, 7, tmp_path, 1, logger)

    # Verify file exists and has empty concepts array
    output_file = tmp_path / "LearningChunkGraph_cluster_7_dict.json"
    assert output_file.exists()

    with open(output_file, encoding="utf-8") as f:
        data = json.load(f)

    assert data["_meta"]["concepts_used"] == 0
    assert data["concepts"] == []

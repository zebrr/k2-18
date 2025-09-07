#!/usr/bin/env python3
"""
Tests for data enrichment functions in graph2metrics module.

Tests demo path generation, mention indexing, node-concept linking,
and large graph filtering.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import networkx as nx
import pytest

# Add src to path for imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from viz.graph2metrics import (
    _generate_critical_path,
    _generate_optimal_path,
    _generate_showcase_path,
    create_mention_index,
    generate_demo_path,
    handle_large_graph,
    link_nodes_to_concepts,
)


@pytest.fixture
def sample_graph_data():
    """Create sample graph data for testing."""
    return {
        "nodes": [
            {
                "id": "n1",
                "type": "Chunk",
                "pagerank": 0.15,
                "prerequisite_depth": 0,
                "learning_effort": 1.0,
                "cluster_id": 0,
                "bridge_score": 0.1,
            },
            {
                "id": "n2",
                "type": "Chunk",
                "pagerank": 0.25,
                "prerequisite_depth": 1,
                "learning_effort": 3.0,
                "cluster_id": 0,
                "bridge_score": 0.3,
            },
            {
                "id": "n3",
                "type": "Chunk",
                "pagerank": 0.35,
                "prerequisite_depth": 2,
                "learning_effort": 5.0,
                "cluster_id": 1,
                "bridge_score": 0.5,
            },
            {
                "id": "n4",
                "type": "Chunk",
                "pagerank": 0.25,
                "prerequisite_depth": 3,
                "learning_effort": 8.0,
                "cluster_id": 1,
                "bridge_score": 0.2,
            },
        ],
        "edges": [
            {"source": "n1", "target": "n2", "type": "PREREQUISITE", "weight": 0.8},
            {"source": "n2", "target": "n3", "type": "PREREQUISITE", "weight": 0.7},
            {"source": "n3", "target": "n4", "type": "PREREQUISITE", "weight": 0.9},
            {"source": "n1", "target": "concept_1", "type": "MENTIONS", "weight": 0.6},
            {"source": "n2", "target": "concept_1", "type": "MENTIONS", "weight": 0.7},
            {"source": "n3", "target": "concept_2", "type": "MENTIONS", "weight": 0.8},
        ],
    }


@pytest.fixture
def sample_concepts_data():
    """Create sample concepts data for testing."""
    return {
        "concepts": [
            {"id": "concept_1", "definition": "Basic concept"},
            {"id": "concept_2", "definition": "Advanced concept"},
        ]
    }


@pytest.fixture
def sample_networkx_graph(sample_graph_data):
    """Create NetworkX graph from sample data."""
    G = nx.DiGraph()

    # Add nodes
    for node in sample_graph_data["nodes"]:
        G.add_node(node["id"], **node)

    # Add edges
    for edge in sample_graph_data["edges"]:
        if edge["target"] not in ["concept_1", "concept_2"]:  # Skip concept edges
            G.add_edge(edge["source"], edge["target"], **edge)

    return G


class TestDemoPathStrategies:
    """Test demo path generation strategies."""

    def test_optimal_path_basic(self, sample_networkx_graph, sample_graph_data):
        """Test optimal strategy for demo path."""
        nodes_dict = {n["id"]: n for n in sample_graph_data["nodes"]}

        path = _generate_optimal_path(sample_networkx_graph, nodes_dict, max_nodes=3)

        # Should produce a valid educational path
        assert len(path) <= 3
        assert len(path) >= 1  # Should have at least one node
        # Check all nodes in path exist
        for node in path:
            assert node in nodes_dict

    def test_optimal_path_empty_graph(self):
        """Test optimal path with empty graph."""
        G = nx.DiGraph()
        path = _generate_optimal_path(G, {}, max_nodes=5)
        assert path == []

    def test_optimal_path_single_node(self):
        """Test optimal path with single node."""
        G = nx.DiGraph()
        G.add_node("n1")
        nodes_dict = {"n1": {"prerequisite_depth": 0, "pagerank": 1.0, "learning_effort": 1.0}}

        path = _generate_optimal_path(G, nodes_dict, max_nodes=5)
        assert path == ["n1"]

    def test_showcase_path_basic(self, sample_networkx_graph, sample_graph_data):
        """Test showcase strategy (one per cluster)."""
        nodes_dict = {n["id"]: n for n in sample_graph_data["nodes"]}

        path = _generate_showcase_path(sample_networkx_graph, nodes_dict, max_nodes=5)

        # Should have nodes from different clusters
        clusters_in_path = set()
        for node_id in path:
            clusters_in_path.add(nodes_dict[node_id]["cluster_id"])

        assert len(clusters_in_path) == 2  # We have 2 clusters
        assert len(path) <= 5

    def test_showcase_path_single_cluster(self, sample_networkx_graph):
        """Test showcase path when all nodes in same cluster."""
        nodes_dict = {
            "n1": {"cluster_id": 0, "pagerank": 0.4, "prerequisite_depth": 0},
            "n2": {"cluster_id": 0, "pagerank": 0.6, "prerequisite_depth": 1},
        }

        # Should fallback to optimal strategy
        path = _generate_showcase_path(sample_networkx_graph, nodes_dict, max_nodes=2)
        assert len(path) <= 2

    def test_showcase_path_more_clusters_than_max(self):
        """Test showcase path with more clusters than max_nodes."""
        G = nx.DiGraph()
        nodes_dict = {}

        # Create 10 clusters, but max_nodes = 3
        for i in range(10):
            node_id = f"n{i}"
            G.add_node(node_id)
            nodes_dict[node_id] = {
                "cluster_id": i,
                "pagerank": 0.1 * (i + 1),
                "prerequisite_depth": i,
            }

        path = _generate_showcase_path(G, nodes_dict, max_nodes=3)

        assert len(path) == 3
        # Should select from largest clusters (by node count, all have 1 node here)

    def test_critical_path_basic(self, sample_networkx_graph, sample_graph_data):
        """Test critical strategy (reverse from complex)."""
        nodes_dict = {n["id"]: n for n in sample_graph_data["nodes"]}

        path = _generate_critical_path(sample_networkx_graph, nodes_dict, max_nodes=4)

        # Should end at max learning_effort node (n4)
        assert path[-1] == "n4"
        # Path should be in simple -> complex order
        assert len(path) <= 4

    def test_critical_path_zero_effort(self):
        """Test critical path when all efforts are zero."""
        G = nx.DiGraph()
        G.add_node("n1")
        G.add_node("n2")
        nodes_dict = {
            "n1": {"learning_effort": 0.0, "pagerank": 0.5, "prerequisite_depth": 0},
            "n2": {"learning_effort": 0.0, "pagerank": 0.5, "prerequisite_depth": 1},
        }

        # Should fallback to optimal strategy
        path = _generate_critical_path(G, nodes_dict, max_nodes=2)
        assert len(path) <= 2

    def test_critical_path_multiple_max_effort(self, sample_networkx_graph):
        """Test critical path with multiple nodes having max effort."""
        nodes_dict = {
            "n1": {"learning_effort": 10.0, "pagerank": 0.3, "prerequisite_depth": 0},
            "n2": {"learning_effort": 10.0, "pagerank": 0.7, "prerequisite_depth": 1},
        }

        path = _generate_critical_path(sample_networkx_graph, nodes_dict, max_nodes=2)

        # Should select n2 (higher PageRank among max effort nodes)
        assert "n2" in path


class TestGenerateDemoPath:
    """Test main demo path generation function."""

    def test_generate_demo_path_strategy_1(self, sample_networkx_graph, sample_graph_data):
        """Test demo path generation with strategy 1."""
        config = {"demo_path": {"strategy": 1, "max_nodes": 3}}
        logger = MagicMock()

        result = generate_demo_path(
            sample_networkx_graph, sample_graph_data, config, logger, test_mode=True
        )

        assert "_meta" in result
        assert "demo_path" in result["_meta"]
        assert "demo_generation_config" in result["_meta"]

        config_meta = result["_meta"]["demo_generation_config"]
        assert config_meta["strategy"] == 1
        assert config_meta["strategy_name"] == "optimal"
        assert config_meta["max_nodes"] == 3
        assert config_meta["actual_nodes"] <= 3

    def test_generate_demo_path_strategy_2(self, sample_networkx_graph, sample_graph_data):
        """Test demo path generation with strategy 2."""
        config = {"demo_path": {"strategy": 2, "max_nodes": 5}}
        logger = MagicMock()

        result = generate_demo_path(
            sample_networkx_graph, sample_graph_data, config, logger
        )

        config_meta = result["_meta"]["demo_generation_config"]
        assert config_meta["strategy"] == 2
        assert config_meta["strategy_name"] == "showcase"

    def test_generate_demo_path_strategy_3(self, sample_networkx_graph, sample_graph_data):
        """Test demo path generation with strategy 3."""
        config = {"demo_path": {"strategy": 3, "max_nodes": 4}}
        logger = MagicMock()

        result = generate_demo_path(
            sample_networkx_graph, sample_graph_data, config, logger
        )

        config_meta = result["_meta"]["demo_generation_config"]
        assert config_meta["strategy"] == 3
        assert config_meta["strategy_name"] == "critical"

    def test_generate_demo_path_invalid_strategy(self, sample_networkx_graph, sample_graph_data):
        """Test demo path with invalid strategy number."""
        config = {"demo_path": {"strategy": 99, "max_nodes": 3}}
        logger = MagicMock()

        result = generate_demo_path(
            sample_networkx_graph, sample_graph_data, config, logger
        )

        # Should default to strategy 1
        config_meta = result["_meta"]["demo_generation_config"]
        assert config_meta["strategy"] == 99
        assert config_meta["strategy_name"] == "optimal"  # Defaults to optimal

    def test_generate_demo_path_no_config(self, sample_networkx_graph, sample_graph_data):
        """Test demo path with missing config."""
        config = {}  # No demo_path section
        logger = MagicMock()

        result = generate_demo_path(
            sample_networkx_graph, sample_graph_data, config, logger
        )

        # Should use defaults
        config_meta = result["_meta"]["demo_generation_config"]
        assert config_meta["strategy"] == 1  # Default
        assert config_meta["max_nodes"] == 15  # Default


class TestCreateMentionIndex:
    """Test mention index creation."""

    def test_create_mention_index_basic(self, sample_graph_data, sample_concepts_data):
        """Test basic mention index creation."""
        # Add Concept nodes to graph for testing
        sample_graph_data["nodes"].extend([
            {"id": "concept_1", "type": "Concept"},
            {"id": "concept_2", "type": "Concept"},
        ])
        
        result = create_mention_index(sample_graph_data, sample_concepts_data)

        assert "_meta" in result
        assert "mention_index" in result["_meta"]

        index = result["_meta"]["mention_index"]

        # Check concept_1 mentions (now includes ALL edges)
        assert "concept_1" in index
        assert index["concept_1"]["count"] == 2
        assert set(index["concept_1"]["nodes"]) == {"n1", "n2"}

        # Check concept_2 mentions
        assert "concept_2" in index
        assert index["concept_2"]["count"] == 1
        assert set(index["concept_2"]["nodes"]) == {"n3"}

    def test_create_mention_index_no_mentions(self):
        """Test mention index with no Concept nodes."""
        graph_data = {
            "nodes": [{"id": "n1", "type": "Chunk"}, {"id": "n2", "type": "Chunk"}],
            "edges": [{"source": "n1", "target": "n2", "type": "PREREQUISITE"}],
        }
        concepts_data = {"concepts": [{"id": "c1"}]}

        result = create_mention_index(graph_data, concepts_data)

        assert result["_meta"]["mention_index"] == {}

    def test_create_mention_index_empty_graph(self):
        """Test mention index with empty graph."""
        graph_data = {"nodes": [], "edges": []}
        concepts_data = {"concepts": []}

        result = create_mention_index(graph_data, concepts_data)

        assert result["_meta"]["mention_index"] == {}


class TestLinkNodesToConcepts:
    """Test node-concept linking."""

    def test_link_nodes_to_concepts_basic(self, sample_graph_data):
        """Test basic node-concept linking."""
        # Add Concept nodes to graph
        sample_graph_data["nodes"].extend([
            {"id": "concept_1", "type": "Concept"},
            {"id": "concept_2", "type": "Concept"},
        ])
        
        result = link_nodes_to_concepts(sample_graph_data.copy())

        # Check that nodes have concepts field
        nodes_by_id = {n["id"]: n for n in result["nodes"]}

        assert "concepts" in nodes_by_id["n1"]
        assert set(nodes_by_id["n1"]["concepts"]) == {"concept_1"}

        assert "concepts" in nodes_by_id["n2"]
        assert set(nodes_by_id["n2"]["concepts"]) == {"concept_1"}

        assert "concepts" in nodes_by_id["n3"]
        assert set(nodes_by_id["n3"]["concepts"]) == {"concept_2"}

        assert "concepts" in nodes_by_id["n4"]
        assert nodes_by_id["n4"]["concepts"] == []  # No connections to concepts

    def test_link_nodes_to_concepts_no_concepts(self):
        """Test linking with no Concept nodes."""
        graph_data = {
            "nodes": [{"id": "n1", "type": "Chunk"}, {"id": "n2", "type": "Chunk"}],
            "edges": [{"source": "n1", "target": "n2", "type": "PREREQUISITE"}],
        }

        result = link_nodes_to_concepts(graph_data)

        for node in result["nodes"]:
            assert node["concepts"] == []

    def test_link_nodes_to_concepts_multiple_concepts(self):
        """Test linking when node connects to multiple concepts."""
        graph_data = {
            "nodes": [
                {"id": "n1", "type": "Chunk"},
                {"id": "c1", "type": "Concept"},
                {"id": "c2", "type": "Concept"},
                {"id": "c3", "type": "Concept"},
            ],
            "edges": [
                {"source": "n1", "target": "c1", "type": "MENTIONS"},
                {"source": "n1", "target": "c2", "type": "MENTIONS"},
                {"source": "n1", "target": "c3", "type": "MENTIONS"},
            ],
        }

        result = link_nodes_to_concepts(graph_data)
        
        nodes_by_id = {n["id"]: n for n in result["nodes"]}
        assert set(nodes_by_id["n1"]["concepts"]) == {"c1", "c2", "c3"}


class TestHandleLargeGraph:
    """Test large graph filtering."""

    def test_handle_large_graph_no_filtering(self, sample_graph_data):
        """Test that small graphs are not filtered."""
        logger = MagicMock()

        result = handle_large_graph(sample_graph_data, max_nodes=100, logger=logger)

        # Should return unchanged
        assert result == sample_graph_data
        assert "_meta" not in result or "graph_metadata" not in result.get("_meta", {})

    def test_handle_large_graph_filtering(self):
        """Test filtering of large graph."""
        # Create graph with 10 nodes
        graph_data = {
            "nodes": [{"id": f"n{i}", "pagerank": 0.1 * i} for i in range(10)],
            "edges": [
                {"source": f"n{i}", "target": f"n{i+1}", "type": "PREREQUISITE"}
                for i in range(9)
            ],
        }
        logger = MagicMock()

        result = handle_large_graph(graph_data, max_nodes=5, logger=logger)

        # Should keep top 5 by PageRank
        assert len(result["nodes"]) == 5
        assert len(result["edges"]) == 4  # Edges between top 5 nodes

        # Check metadata
        assert result["_meta"]["graph_metadata"]["filtered"] is True
        assert result["_meta"]["graph_metadata"]["original_nodes"] == 10
        assert result["_meta"]["graph_metadata"]["filtered_nodes"] == 5

        # Check that highest PageRank nodes were kept
        kept_ids = {n["id"] for n in result["nodes"]}
        assert kept_ids == {"n5", "n6", "n7", "n8", "n9"}

    def test_handle_large_graph_save_full(self, tmp_path):
        """Test saving full graph before filtering."""
        graph_data = {
            "nodes": [{"id": f"n{i}", "pagerank": 0.1 * i} for i in range(10)],
            "edges": [],
        }
        logger = MagicMock()
        full_path = tmp_path / "full_graph.json"

        result = handle_large_graph(graph_data, max_nodes=5, save_full_path=full_path, logger=logger)

        # Check that full graph was saved
        assert full_path.exists()
        with open(full_path, "r") as f:
            saved_data = json.load(f)
        assert len(saved_data["nodes"]) == 10  # Full graph

        # Check that result is filtered
        assert len(result["nodes"]) == 5

    def test_handle_large_graph_edge_filtering(self):
        """Test that edges are correctly filtered."""
        graph_data = {
            "nodes": [{"id": f"n{i}", "pagerank": 0.1 * i} for i in range(5)],
            "edges": [
                {"source": "n0", "target": "n1"},  # Will be filtered out
                {"source": "n0", "target": "n4"},  # Will be filtered out (n0 removed)
                {"source": "n3", "target": "n4"},  # Will be kept
                {"source": "n2", "target": "n3"},  # Will be kept
            ],
        }
        logger = MagicMock()

        result = handle_large_graph(graph_data, max_nodes=3, logger=logger)

        # Top 3 nodes by PageRank: n2, n3, n4
        kept_ids = {n["id"] for n in result["nodes"]}
        assert kept_ids == {"n2", "n3", "n4"}

        # Only edges between kept nodes
        assert len(result["edges"]) == 2
        edge_pairs = {(e["source"], e["target"]) for e in result["edges"]}
        assert edge_pairs == {("n3", "n4"), ("n2", "n3")}

    @pytest.mark.parametrize(
        "nodes,threshold,should_filter",
        [
            (100, 1000, False),
            (1000, 1000, False),
            (1001, 1000, True),
            (2000, 500, True),
        ],
    )
    def test_filtering_threshold(self, nodes, threshold, should_filter):
        """Test that filtering only applies when needed."""
        graph_data = {
            "nodes": [{"id": f"n{i}", "pagerank": 1.0 / (i + 1)} for i in range(nodes)],
            "edges": [],
        }

        result = handle_large_graph(graph_data, max_nodes=threshold)

        if should_filter:
            assert len(result["nodes"]) == threshold
            assert result["_meta"]["graph_metadata"]["filtered"] is True
        else:
            assert len(result["nodes"]) == nodes
            assert "_meta" not in result or "filtered" not in result.get("_meta", {}).get(
                "graph_metadata", {}
            )


class TestIntegration:
    """Integration tests for enrichment functions."""

    def test_full_enrichment_flow(self, sample_networkx_graph, sample_graph_data, sample_concepts_data):
        """Test complete enrichment pipeline."""
        config = {"demo_path": {"strategy": 1, "max_nodes": 3}}
        logger = MagicMock()

        # Generate demo path
        graph_data = generate_demo_path(
            sample_networkx_graph, sample_graph_data, config, logger
        )

        # Link nodes to concepts
        graph_data = link_nodes_to_concepts(graph_data)

        # Create mention index
        concepts_data = create_mention_index(graph_data, sample_concepts_data)

        # Verify all enrichments
        assert "demo_path" in graph_data["_meta"]
        assert "demo_generation_config" in graph_data["_meta"]
        assert all("concepts" in node for node in graph_data["nodes"])
        assert "mention_index" in concepts_data["_meta"]

    def test_enrichment_with_tiny_graph(self):
        """Test enrichment with actual tiny_graph.json structure."""
        # Load tiny_graph.json
        viz_dir = Path(__file__).parent.parent.parent / "viz"
        tiny_graph_path = viz_dir / "data" / "test" / "tiny_graph.json"

        if tiny_graph_path.exists():
            with open(tiny_graph_path, "r") as f:
                graph_data = json.load(f)

            # Create NetworkX graph
            G = nx.DiGraph()
            for node in graph_data["nodes"]:
                G.add_node(node["id"], **node)
            for edge in graph_data["edges"]:
                G.add_edge(edge["source"], edge["target"], **edge)

            # Test demo path generation
            config = {"demo_path": {"strategy": 1, "max_nodes": 3}}
            logger = MagicMock()

            result = generate_demo_path(G, graph_data, config, logger)

            assert "demo_path" in result["_meta"]
            assert len(result["_meta"]["demo_path"]) <= 3

            # Test node-concept linking
            result = link_nodes_to_concepts(result)

            # Check that chunk nodes have concepts
            chunk_nodes = [n for n in result["nodes"] if n.get("type") == "Chunk"]
            for node in chunk_nodes:
                assert "concepts" in node
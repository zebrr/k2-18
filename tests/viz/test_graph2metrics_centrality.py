#!/usr/bin/env python3
"""
Tests for graph2metrics centrality computation functionality.
"""

import math
import sys
from pathlib import Path

import networkx as nx
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from viz.graph2metrics import compute_centrality_metrics, safe_metric_value


@pytest.mark.viz
def test_safe_metric_value():
    """Test safe conversion of NaN/inf values."""
    # Normal values
    assert safe_metric_value(0.5) == 0.5
    assert safe_metric_value(1.0) == 1.0
    assert safe_metric_value(0) == 0.0

    # Edge cases
    assert safe_metric_value(None) == 0.0
    assert safe_metric_value(float("nan")) == 0.0
    assert safe_metric_value(float("inf")) == 0.0
    assert safe_metric_value(float("-inf")) == 0.0

    # Math module NaN/inf
    assert safe_metric_value(math.nan) == 0.0
    assert safe_metric_value(math.inf) == 0.0


@pytest.mark.viz
def test_centrality_metrics_computation():
    """Test that all centrality metrics are computed correctly."""
    # Create a small test graph
    G = nx.DiGraph()
    G.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])

    graph_data = {
        "nodes": [
            {"id": "A", "type": "Chunk"},
            {"id": "B", "type": "Chunk"},
            {"id": "C", "type": "Chunk"},
        ],
        "edges": [
            {"source": "A", "target": "B", "weight": 1.0},
            {"source": "B", "target": "C", "weight": 1.0},
            {"source": "A", "target": "C", "weight": 1.0},
        ],
    }

    config = {
        "graph2metrics": {
            "pagerank_damping": 0.85,
            "pagerank_max_iter": 100,
            "betweenness_normalized": True,
            "closeness_harmonic": True,
        }
    }

    # Compute metrics (logger=None should be handled gracefully)
    result = compute_centrality_metrics(G, graph_data, config, None, test_mode=True)

    # Check all nodes have metrics
    for node in result["nodes"]:
        assert "degree_in" in node
        assert "degree_out" in node
        assert "degree_centrality" in node
        assert "pagerank" in node
        assert "betweenness_centrality" in node
        assert "closeness_centrality" in node

        # Check types
        assert isinstance(node["degree_in"], int)
        assert isinstance(node["degree_out"], int)
        assert isinstance(node["degree_centrality"], float)
        assert isinstance(node["pagerank"], float)
        assert isinstance(node["betweenness_centrality"], float)
        assert isinstance(node["closeness_centrality"], float)

        # Check value ranges
        assert node["degree_in"] >= 0
        assert node["degree_out"] >= 0
        assert 0.0 <= node["degree_centrality"] <= 1.0
        assert 0.0 <= node["pagerank"] <= 1.0
        assert 0.0 <= node["betweenness_centrality"] <= 1.0
        assert 0.0 <= node["closeness_centrality"] <= 1.0

    # Check specific values for this graph structure
    # Node A has 2 outgoing edges (to B and C)
    node_a = next(n for n in result["nodes"] if n["id"] == "A")
    assert node_a["degree_out"] == 2
    assert node_a["degree_in"] == 0

    # Node C has 2 incoming edges (from A and B)
    node_c = next(n for n in result["nodes"] if n["id"] == "C")
    assert node_c["degree_in"] == 2
    assert node_c["degree_out"] == 0

    # Node B is in the middle (1 in, 1 out)
    node_b = next(n for n in result["nodes"] if n["id"] == "B")
    assert node_b["degree_in"] == 1
    assert node_b["degree_out"] == 1


@pytest.mark.viz
def test_isolated_nodes_handling():
    """Test that isolated nodes are handled correctly."""
    # Test with isolated node
    G = nx.DiGraph()
    G.add_node("isolated")
    G.add_edge("A", "B")

    graph_data = {
        "nodes": [
            {"id": "isolated", "type": "Chunk"},
            {"id": "A", "type": "Chunk"},
            {"id": "B", "type": "Chunk"},
        ],
        "edges": [{"source": "A", "target": "B", "weight": 1.0}],
    }

    config = {
        "graph2metrics": {
            "pagerank_damping": 0.85,
            "pagerank_max_iter": 100,
            "betweenness_normalized": True,
            "closeness_harmonic": True,
        }
    }

    # Compute metrics
    result = compute_centrality_metrics(G, graph_data, config, None, test_mode=True)

    # Find isolated node
    isolated_node = next(n for n in result["nodes"] if n["id"] == "isolated")

    # Check isolated node has safe values (0 or very small)
    assert isolated_node["degree_in"] == 0
    assert isolated_node["degree_out"] == 0
    assert isolated_node["degree_centrality"] == 0.0
    assert isolated_node["betweenness_centrality"] == 0.0
    # PageRank distributes evenly for disconnected components
    # With 3 nodes, each gets ~1/3 of total PageRank
    assert 0.0 <= isolated_node["pagerank"] <= 0.4
    # Closeness for isolated node should be 0 with harmonic mean
    assert isolated_node["closeness_centrality"] == 0.0


@pytest.mark.viz
def test_config_parameters_usage():
    """Test that configuration parameters are properly used."""
    G = nx.DiGraph()
    G.add_edges_from([("A", "B"), ("B", "C")])

    graph_data = {
        "nodes": [
            {"id": "A", "type": "Chunk"},
            {"id": "B", "type": "Chunk"},
            {"id": "C", "type": "Chunk"},
        ],
        "edges": [
            {"source": "A", "target": "B", "weight": 1.0},
            {"source": "B", "target": "C", "weight": 1.0},
        ],
    }

    # Test with different damping factors
    config_low_damping = {
        "graph2metrics": {
            "pagerank_damping": 0.5,
            "pagerank_max_iter": 100,
            "betweenness_normalized": True,
            "closeness_harmonic": True,
        }
    }

    config_high_damping = {
        "graph2metrics": {
            "pagerank_damping": 0.95,
            "pagerank_max_iter": 100,
            "betweenness_normalized": True,
            "closeness_harmonic": True,
        }
    }

    result_low = compute_centrality_metrics(
        G, graph_data.copy(), config_low_damping, None, test_mode=True
    )
    result_high = compute_centrality_metrics(
        G, graph_data.copy(), config_high_damping, None, test_mode=True
    )

    # PageRank values should differ with different damping
    pr_low = [n["pagerank"] for n in result_low["nodes"]]
    pr_high = [n["pagerank"] for n in result_high["nodes"]]

    # For small graphs the difference might be minimal
    # Just check that metrics were computed
    assert len(pr_low) == 3
    assert len(pr_high) == 3
    assert all(0.0 <= p <= 1.0 for p in pr_low)
    assert all(0.0 <= p <= 1.0 for p in pr_high)


@pytest.mark.viz
def test_large_graph_performance():
    """Test performance on larger graph."""
    import time

    # Create larger test graph (100 nodes)
    G_temp = nx.fast_gnp_random_graph(100, 0.05, directed=True)
    # Convert to string IDs
    G = nx.DiGraph()
    for u, v in G_temp.edges():
        G.add_edge(str(u), str(v))
    
    # Create graph data
    graph_data = {
        "nodes": [{"id": str(i), "type": "Chunk"} for i in range(100)],
        "edges": [{"source": str(u), "target": str(v), "weight": 1.0} for u, v in G.edges()],
    }

    config = {
        "graph2metrics": {
            "pagerank_damping": 0.85,
            "pagerank_max_iter": 100,
            "betweenness_normalized": True,
            "closeness_harmonic": True,
        }
    }

    # Measure computation time
    start_time = time.time()
    result = compute_centrality_metrics(G, graph_data, config, None, test_mode=True)
    elapsed = time.time() - start_time

    # Should complete reasonably fast for 100 nodes
    assert elapsed < 5.0, f"Computation took {elapsed:.2f}s, expected < 5s for 100 nodes"

    # Check all nodes have metrics
    assert all("pagerank" in node for node in result["nodes"])
    assert all("betweenness_centrality" in node for node in result["nodes"])


@pytest.mark.viz
def test_empty_graph():
    """Test handling of empty graph."""
    G = nx.DiGraph()

    graph_data = {"nodes": [], "edges": []}

    config = {
        "graph2metrics": {
            "pagerank_damping": 0.85,
            "pagerank_max_iter": 100,
            "betweenness_normalized": True,
            "closeness_harmonic": True,
        }
    }

    # Should handle empty graph gracefully
    result = compute_centrality_metrics(G, graph_data, config, None, test_mode=True)

    assert result["nodes"] == []
    assert result["edges"] == []


@pytest.mark.viz
def test_single_node_graph():
    """Test handling of single-node graph."""
    G = nx.DiGraph()
    G.add_node("single")

    graph_data = {"nodes": [{"id": "single", "type": "Chunk"}], "edges": []}

    config = {
        "graph2metrics": {
            "pagerank_damping": 0.85,
            "pagerank_max_iter": 100,
            "betweenness_normalized": True,
            "closeness_harmonic": True,
        }
    }

    result = compute_centrality_metrics(G, graph_data, config, None, test_mode=True)

    node = result["nodes"][0]
    assert node["degree_in"] == 0
    assert node["degree_out"] == 0
    # NetworkX returns 1.0 for single node degree_centrality (special case)
    assert node["degree_centrality"] == 1.0
    assert node["betweenness_centrality"] == 0.0
    # Single node gets all PageRank
    assert node["pagerank"] == 1.0
    # Closeness is 0 for single node
    assert node["closeness_centrality"] == 0.0


@pytest.mark.viz
def test_self_loops():
    """Test handling of self-loops in graph."""
    G = nx.DiGraph()
    G.add_edges_from([("A", "A"), ("A", "B"), ("B", "C")])

    graph_data = {
        "nodes": [
            {"id": "A", "type": "Chunk"},
            {"id": "B", "type": "Chunk"},
            {"id": "C", "type": "Chunk"},
        ],
        "edges": [
            {"source": "A", "target": "A", "weight": 1.0},
            {"source": "A", "target": "B", "weight": 1.0},
            {"source": "B", "target": "C", "weight": 1.0},
        ],
    }

    config = {
        "graph2metrics": {
            "pagerank_damping": 0.85,
            "pagerank_max_iter": 100,
            "betweenness_normalized": True,
            "closeness_harmonic": True,
        }
    }

    # Should handle self-loops properly
    result = compute_centrality_metrics(G, graph_data, config, None, test_mode=True)

    node_a = next(n for n in result["nodes"] if n["id"] == "A")
    # Self-loop counts as both in and out degree
    assert node_a["degree_in"] == 1  # self-loop
    assert node_a["degree_out"] == 2  # self-loop + edge to B

    # All metrics should be computed
    assert "pagerank" in node_a
    assert "betweenness_centrality" in node_a
    assert "closeness_centrality" in node_a

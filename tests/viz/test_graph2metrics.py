#!/usr/bin/env python3
"""
Unit tests for graph2metrics.py module.
Tests all 11 metrics implementation.
"""

# Add project root to path
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from viz.graph2metrics import (
    compute_all_metrics,
    compute_component_ids,
    compute_distance_centrality,
    compute_edge_weights,
    compute_educational_importance,
    compute_prerequisite_metrics,
    create_mention_index,
    generate_demo_path,
    link_nodes_to_concepts,
    safe_metric_value,
    validate_metric_invariants,
)


class TestEdgeWeights(unittest.TestCase):
    """Test inverse_weight computation."""

    def test_inverse_weight_normal(self):
        """Test inverse_weight for normal weights."""
        G = nx.DiGraph()
        G.add_edge("a", "b", weight=2.0)
        G.add_edge("b", "c", weight=0.5)
        G.add_edge("c", "d", weight=1.0)

        compute_edge_weights(G, None)

        self.assertAlmostEqual(G["a"]["b"]["inverse_weight"], 0.5)
        self.assertAlmostEqual(G["b"]["c"]["inverse_weight"], 2.0)
        self.assertAlmostEqual(G["c"]["d"]["inverse_weight"], 1.0)

    def test_inverse_weight_zero(self):
        """Test inverse_weight for zero weight."""
        G = nx.DiGraph()
        G.add_edge("a", "b", weight=0.0)

        compute_edge_weights(G, None)

        self.assertEqual(G["a"]["b"]["inverse_weight"], float("inf"))

    def test_inverse_weight_missing(self):
        """Test inverse_weight for missing weight (defaults to 1.0)."""
        G = nx.DiGraph()
        G.add_edge("a", "b")  # No weight attribute

        compute_edge_weights(G, None)

        self.assertAlmostEqual(G["a"]["b"]["inverse_weight"], 1.0)


class TestOutCloseness(unittest.TestCase):
    """Test out-closeness centrality computation."""

    def test_out_closeness_directed(self):
        """Test OUT-closeness on directed graph."""
        G = nx.DiGraph()
        G.add_edge("a", "b", weight=1.0)
        G.add_edge("b", "c", weight=1.0)
        G.add_edge("c", "d", weight=1.0)

        # Add inverse weights first
        compute_edge_weights(G, None)

        # Compute distance centrality
        result = compute_distance_centrality(G, None)

        # Check out-closeness values
        # Node 'a' can reach all others, should have highest out-closeness
        # Node 'd' can't reach any others, should have 0
        self.assertGreater(result["out_closeness"]["a"], result["out_closeness"]["d"])
        self.assertEqual(result["out_closeness"]["d"], 0.0)

    def test_out_closeness_single_node(self):
        """Test out-closeness for single node graph."""
        G = nx.DiGraph()
        G.add_node("a")

        result = compute_distance_centrality(G, None)

        self.assertEqual(result["out_closeness"]["a"], 0.0)


class TestComponentIds(unittest.TestCase):
    """Test component ID assignment."""

    def test_component_id_single(self):
        """Test single connected component."""
        G = nx.DiGraph()
        G.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])
        node_order = ["a", "b", "c"]

        result = compute_component_ids(G, node_order, None)

        # All nodes should be in component 0
        self.assertEqual(result["a"], 0)
        self.assertEqual(result["b"], 0)
        self.assertEqual(result["c"], 0)

    def test_component_id_multiple(self):
        """Test multiple disconnected components."""
        G = nx.DiGraph()
        G.add_edges_from([("a", "b"), ("c", "d")])
        node_order = ["a", "b", "c", "d"]

        result = compute_component_ids(G, node_order, None)

        # First component (a,b) should be 0
        self.assertEqual(result["a"], 0)
        self.assertEqual(result["b"], 0)
        # Second component (c,d) should be 1
        self.assertEqual(result["c"], 1)
        self.assertEqual(result["d"], 1)

    def test_component_id_deterministic(self):
        """Test deterministic numbering based on node order."""
        G = nx.DiGraph()
        G.add_nodes_from(["x", "y", "z"])  # Three isolated nodes

        # Different orders should give different component IDs
        result1 = compute_component_ids(G, ["x", "y", "z"], None)
        result2 = compute_component_ids(G, ["z", "y", "x"], None)

        # With order ["x", "y", "z"], x=0, y=1, z=2
        self.assertEqual(result1["x"], 0)
        self.assertEqual(result1["y"], 1)
        self.assertEqual(result1["z"], 2)

        # With order ["z", "y", "x"], z=0, y=1, x=2
        self.assertEqual(result2["z"], 0)
        self.assertEqual(result2["y"], 1)
        self.assertEqual(result2["x"], 2)


class TestPrerequisiteMetrics(unittest.TestCase):
    """Test prerequisite_depth and learning_effort computation."""

    def test_prerequisite_linear_chain(self):
        """Test metrics for linear prerequisite chain."""
        G = nx.DiGraph()
        G.add_edge("a", "b", type="PREREQUISITE", weight=1.0)
        G.add_edge("b", "c", type="PREREQUISITE", weight=1.0)
        G.add_edge("c", "d", type="PREREQUISITE", weight=1.0)

        graph_data = {
            "nodes": [
                {"id": "a", "difficulty": 1},
                {"id": "b", "difficulty": 2},
                {"id": "c", "difficulty": 3},
                {"id": "d", "difficulty": 4},
            ]
        }

        config = {"path_mode": {"default_difficulty": 3}}

        depth, effort = compute_prerequisite_metrics(G, graph_data, config, None)

        # Check depths
        self.assertEqual(depth["a"], 0)
        self.assertEqual(depth["b"], 1)
        self.assertEqual(depth["c"], 2)
        self.assertEqual(depth["d"], 3)

        # Check efforts (cumulative difficulty)
        self.assertAlmostEqual(effort["a"], 1.0)
        self.assertAlmostEqual(effort["b"], 3.0)  # 1 + 2
        self.assertAlmostEqual(effort["c"], 6.0)  # 1 + 2 + 3
        self.assertAlmostEqual(effort["d"], 10.0)  # 1 + 2 + 3 + 4

    def test_prerequisite_with_cycle(self):
        """Test handling of cycles in prerequisites."""
        G = nx.DiGraph()
        G.add_edge("a", "b", type="PREREQUISITE", weight=1.0)
        G.add_edge("b", "c", type="PREREQUISITE", weight=1.0)
        G.add_edge("c", "a", type="PREREQUISITE", weight=1.0)  # Cycle!

        graph_data = {
            "nodes": [
                {"id": "a", "difficulty": 1},
                {"id": "b", "difficulty": 2},
                {"id": "c", "difficulty": 3},
            ]
        }

        config = {"path_mode": {"default_difficulty": 3}}

        depth, effort = compute_prerequisite_metrics(G, graph_data, config, None)

        # All nodes in cycle should have same depth (0)
        self.assertEqual(depth["a"], 0)
        self.assertEqual(depth["b"], 0)
        self.assertEqual(depth["c"], 0)

        # All nodes in cycle should have same effort (sum of difficulties)
        expected_effort = 6.0  # 1 + 2 + 3
        self.assertAlmostEqual(effort["a"], expected_effort)
        self.assertAlmostEqual(effort["b"], expected_effort)
        self.assertAlmostEqual(effort["c"], expected_effort)

    def test_prerequisite_no_edges(self):
        """Test when there are no PREREQUISITE edges."""
        G = nx.DiGraph()
        G.add_edge("a", "b", type="MENTIONS", weight=1.0)
        G.add_edge("b", "c", type="ELABORATES", weight=1.0)

        graph_data = {
            "nodes": [
                {"id": "a", "difficulty": 1},
                {"id": "b", "difficulty": 2},
                {"id": "c", "difficulty": 3},
            ]
        }

        config = {"path_mode": {"default_difficulty": 3}}

        depth, effort = compute_prerequisite_metrics(G, graph_data, config, None)

        # All nodes should have depth 0
        self.assertEqual(depth["a"], 0)
        self.assertEqual(depth["b"], 0)
        self.assertEqual(depth["c"], 0)

        # All nodes should have effort equal to their difficulty
        self.assertAlmostEqual(effort["a"], 1.0)
        self.assertAlmostEqual(effort["b"], 2.0)
        self.assertAlmostEqual(effort["c"], 3.0)


class TestEducationalImportance(unittest.TestCase):
    """Test educational_importance computation."""

    def test_educational_importance_subgraph(self):
        """Test PageRank on educational edges only."""
        G = nx.DiGraph()
        # Educational edges
        G.add_edge("a", "b", type="PREREQUISITE", weight=1.0)
        G.add_edge("b", "c", type="ELABORATES", weight=1.0)
        # Non-educational edge
        G.add_edge("c", "d", type="MENTIONS", weight=1.0)

        config = {
            "graph2metrics": {
                "educational_edge_types": ["PREREQUISITE", "ELABORATES", "TESTS", "EXAMPLE_OF"],
                "pagerank_damping": 0.85,
            }
        }

        result = compute_educational_importance(G, config, None)

        # Should compute PageRank only on educational subgraph
        # All nodes should be present
        self.assertIn("a", result)
        self.assertIn("b", result)
        self.assertIn("c", result)
        self.assertIn("d", result)

        # Sum should be approximately 1.0
        total = sum(result.values())
        self.assertAlmostEqual(total, 1.0, places=2)

    def test_educational_importance_no_edges(self):
        """Test when there are no educational edges."""
        G = nx.DiGraph()
        G.add_edge("a", "b", type="MENTIONS", weight=1.0)
        G.add_edge("b", "c", type="MENTIONS", weight=1.0)

        config = {
            "graph2metrics": {
                "educational_edge_types": ["PREREQUISITE", "ELABORATES", "TESTS", "EXAMPLE_OF"],
                "pagerank_damping": 0.85,
            }
        }

        result = compute_educational_importance(G, config, None)

        # Should return uniform distribution
        expected = 1.0 / G.number_of_nodes()
        for node in G.nodes():
            self.assertAlmostEqual(result[node], expected, places=5)


class TestMetricInvariants(unittest.TestCase):
    """Test metric invariant validation."""

    def test_pagerank_sum_valid(self):
        """Test valid PageRank sum."""
        pagerank_vals = {"a": 0.25, "b": 0.35, "c": 0.40}
        edu_importance_vals = {"a": 0.30, "b": 0.30, "c": 0.40}

        logger = MagicMock()
        validate_metric_invariants(pagerank_vals, edu_importance_vals, logger)

        # Should not log any warnings
        logger.warning.assert_not_called()

    def test_pagerank_sum_invalid(self):
        """Test invalid PageRank sum."""
        pagerank_vals = {"a": 0.25, "b": 0.35, "c": 0.50}  # Sum = 1.1
        edu_importance_vals = {"a": 0.10, "b": 0.20, "c": 0.30}  # Sum = 0.6

        logger = MagicMock()
        validate_metric_invariants(pagerank_vals, edu_importance_vals, logger)

        # Should log warnings
        self.assertEqual(logger.warning.call_count, 2)


class TestSafeMetricValue(unittest.TestCase):
    """Test safe metric value conversion."""

    def test_safe_metric_normal(self):
        """Test normal float values."""
        self.assertEqual(safe_metric_value(0.5), 0.5)
        self.assertEqual(safe_metric_value(1.0), 1.0)
        self.assertEqual(safe_metric_value(0), 0.0)

    def test_safe_metric_nan(self):
        """Test NaN conversion."""
        self.assertEqual(safe_metric_value(float("nan")), 0.0)

    def test_safe_metric_inf(self):
        """Test infinity conversion."""
        self.assertEqual(safe_metric_value(float("inf")), 0.0)
        self.assertEqual(safe_metric_value(float("-inf")), 0.0)

    def test_safe_metric_none(self):
        """Test None conversion."""
        self.assertEqual(safe_metric_value(None), 0.0)


class TestFullEnrichmentFlow(unittest.TestCase):
    """Integration test for complete enrichment pipeline."""

    def test_full_enrichment_flow(self):
        """Test complete enrichment pipeline with tiny_graph.json."""
        # Create sample graph data
        graph_data = {
            "nodes": [
                {
                    "id": "chunk_1",
                    "type": "Chunk",
                    "text": "Introduction",
                    "difficulty": 1,
                },
                {
                    "id": "chunk_2",
                    "type": "Chunk",
                    "text": "Advanced topic",
                    "difficulty": 3,
                },
                {
                    "id": "concept_1",
                    "type": "Concept",
                    "text": "Basic concept",
                },
            ],
            "edges": [
                {
                    "source": "chunk_1",
                    "target": "chunk_2",
                    "type": "PREREQUISITE",
                    "weight": 0.8,
                },
                {
                    "source": "chunk_1",
                    "target": "concept_1",
                    "type": "MENTIONS",
                    "weight": 0.7,
                },
            ],
        }

        concepts_data = {
            "concepts": [{"id": "concept_1", "definition": "A basic concept"}]
        }

        # Create NetworkX graph
        G = nx.DiGraph()
        for node in graph_data["nodes"]:
            G.add_node(node["id"], **node)
        for edge in graph_data["edges"]:
            G.add_edge(edge["source"], edge["target"], **edge)

        # Configuration
        config = {
            "graph2metrics": {
                "pagerank_damping": 0.85,
                "pagerank_max_iter": 100,
                "educational_edge_types": ["PREREQUISITE"],
                "louvain_resolution": 1.0,
                "louvain_random_state": 42,
                "bridge_weight_betweenness": 0.7,
            },
            "demo_path": {"strategy": 1, "max_nodes": 3},
            "path_mode": {"default_difficulty": 3},
        }

        logger = MagicMock()

        # Run full pipeline
        enriched_graph = compute_all_metrics(G, graph_data, config, logger)

        # Check that all enrichments are present:
        # 1. Demo path in _meta
        self.assertIn("_meta", enriched_graph)
        self.assertIn("demo_path", enriched_graph["_meta"])
        self.assertIn("demo_generation_config", enriched_graph["_meta"])

        # 2. Concepts field in nodes
        for node in enriched_graph["nodes"]:
            if node["type"] == "Chunk":
                self.assertIn("concepts", node)

        # 3. All metrics present in nodes
        chunk_nodes = [n for n in enriched_graph["nodes"] if n["type"] == "Chunk"]
        for node in chunk_nodes:
            # Basic metrics
            self.assertIn("degree_in", node)
            self.assertIn("degree_out", node)
            self.assertIn("degree_centrality", node)
            self.assertIn("pagerank", node)
            self.assertIn("betweenness_centrality", node)
            self.assertIn("out-closeness", node)
            # Structure metrics
            self.assertIn("component_id", node)
            self.assertIn("prerequisite_depth", node)
            self.assertIn("learning_effort", node)
            self.assertIn("educational_importance", node)
            # Advanced metrics
            self.assertIn("cluster_id", node)
            self.assertIn("bridge_score", node)

        # 4. Create mention index
        enriched_concepts = create_mention_index(enriched_graph, concepts_data)
        self.assertIn("_meta", enriched_concepts)
        self.assertIn("mention_index", enriched_concepts["_meta"])

        # Check mention index content
        mention_index = enriched_concepts["_meta"]["mention_index"]
        if "concept_1" in mention_index:
            self.assertIn("nodes", mention_index["concept_1"])
            self.assertIn("count", mention_index["concept_1"])

    def test_demo_path_generation(self):
        """Test that demo path is correctly generated."""
        # Create a simple linear graph
        graph_data = {
            "nodes": [
                {
                    "id": "n1",
                    "pagerank": 0.4,
                    "prerequisite_depth": 0,
                    "learning_effort": 1.0,
                    "cluster_id": 0,
                    "bridge_score": 0.1,
                },
                {
                    "id": "n2",
                    "pagerank": 0.3,
                    "prerequisite_depth": 1,
                    "learning_effort": 2.0,
                    "cluster_id": 0,
                    "bridge_score": 0.2,
                },
                {
                    "id": "n3",
                    "pagerank": 0.3,
                    "prerequisite_depth": 2,
                    "learning_effort": 3.0,
                    "cluster_id": 1,
                    "bridge_score": 0.3,
                },
            ],
            "edges": [
                {"source": "n1", "target": "n2", "type": "PREREQUISITE"},
                {"source": "n2", "target": "n3", "type": "PREREQUISITE"},
            ],
        }

        G = nx.DiGraph()
        for node in graph_data["nodes"]:
            G.add_node(node["id"], **node)
        for edge in graph_data["edges"]:
            G.add_edge(edge["source"], edge["target"], **edge)

        config = {"demo_path": {"strategy": 1, "max_nodes": 3}}
        logger = MagicMock()

        result = generate_demo_path(G, graph_data, config, logger)

        # Check demo path
        self.assertIn("demo_path", result["_meta"])
        path = result["_meta"]["demo_path"]
        self.assertIsInstance(path, list)
        self.assertLessEqual(len(path), 3)

        # For strategy 1 (optimal), should go from entry to deepest
        if path:
            self.assertEqual(path[0], "n1")  # Entry point
            self.assertEqual(path[-1], "n3")  # Deepest node

    def test_node_concept_linking(self):
        """Test that nodes are correctly linked to concepts."""
        graph_data = {
            "nodes": [{"id": "n1"}, {"id": "n2"}],
            "edges": [
                {"source": "n1", "target": "c1", "type": "MENTIONS"},
                {"source": "n1", "target": "c2", "type": "MENTIONS"},
                {"source": "n2", "target": "c1", "type": "MENTIONS"},
            ],
        }

        result = link_nodes_to_concepts(graph_data)

        # Check concepts field
        nodes_by_id = {n["id"]: n for n in result["nodes"]}
        self.assertEqual(set(nodes_by_id["n1"]["concepts"]), {"c1", "c2"})
        self.assertEqual(nodes_by_id["n2"]["concepts"], ["c1"])


if __name__ == "__main__":
    unittest.main()

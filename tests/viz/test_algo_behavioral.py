"""
Behavioral tests for advanced graph metrics (Louvain, bridge_score, inter-cluster edges).

These tests verify qualitative properties rather than exact numeric values.
They use small synthetic graphs created in-memory.
"""

import sys
from pathlib import Path

import networkx as nx
import pytest

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from viz.graph2metrics import (
    compute_bridge_scores,
    compute_edge_weights,
    compute_louvain_clustering,
    mark_inter_cluster_edges,
)


@pytest.mark.viz
class TestAlgorithmBehavior:
    """Test behavioral properties of advanced algorithms."""

    def test_louvain_three_cliques(self):
        """Test that Louvain correctly identifies 3 isolated cliques."""
        # Create graph: three K4 cliques (12 nodes total), no connections between them
        G = nx.DiGraph()  # noqa: N806

        # Clique 1: nodes 0-3
        for i in range(4):
            for j in range(4):
                if i != j:
                    G.add_edge(f"c1_{i}", f"c1_{j}", weight=1.0)

        # Clique 2: nodes 4-7
        for i in range(4):
            for j in range(4):
                if i != j:
                    G.add_edge(f"c2_{i}", f"c2_{j}", weight=1.0)

        # Clique 3: nodes 8-11
        for i in range(4):
            for j in range(4):
                if i != j:
                    G.add_edge(f"c3_{i}", f"c3_{j}", weight=1.0)

        # Run Louvain
        config = {"louvain_resolution": 1.0, "louvain_random_state": 42}
        cluster_map = compute_louvain_clustering(G, config, logger=None)

        # Behavioral assertions:
        # 1. Should have exactly 3 clusters
        unique_clusters = set(cluster_map.values())
        assert len(unique_clusters) == 3, f"Expected 3 clusters, got {len(unique_clusters)}"

        # 2. Each clique should be in its own cluster (check as sets)
        clique1_nodes = {f"c1_{i}" for i in range(4)}
        clique2_nodes = {f"c2_{i}" for i in range(4)}
        clique3_nodes = {f"c3_{i}" for i in range(4)}

        # Get cluster assignments for each clique
        clique1_clusters = {cluster_map[n] for n in clique1_nodes}
        clique2_clusters = {cluster_map[n] for n in clique2_nodes}
        clique3_clusters = {cluster_map[n] for n in clique3_nodes}

        # Each clique should have exactly one cluster ID
        assert len(clique1_clusters) == 1, "Clique 1 split across clusters"
        assert len(clique2_clusters) == 1, "Clique 2 split across clusters"
        assert len(clique3_clusters) == 1, "Clique 3 split across clusters"

        # All three should be different
        assert clique1_clusters.isdisjoint(clique2_clusters)
        assert clique2_clusters.isdisjoint(clique3_clusters)
        assert clique1_clusters.isdisjoint(clique3_clusters)

    def test_bridge_score_barbell(self):
        """Test bridge_score identifies bridge nodes in barbell graph."""
        # Create barbell: two K4 cliques connected by single edge
        G = nx.DiGraph()  # noqa: N806

        # Left clique (nodes 0-3)
        for i in range(4):
            for j in range(4):
                if i != j:
                    G.add_edge(f"left_{i}", f"left_{j}", weight=1.0)

        # Right clique (nodes 4-7)
        for i in range(4):
            for j in range(4):
                if i != j:
                    G.add_edge(f"right_{i}", f"right_{j}", weight=1.0)

        # Bridge edge: connect left_3 to right_0
        G.add_edge("left_3", "right_0", weight=1.0)
        G.add_edge("right_0", "left_3", weight=1.0)  # Make bidirectional

        # Compute prerequisites
        # First need betweenness centrality (required for bridge_score)
        # Add inverse weights
        compute_edge_weights(G, logger=None)

        # Compute betweenness
        betweenness = nx.betweenness_centrality(G, weight="inverse_weight", normalized=True)

        # Run Louvain to get clusters
        config = {
            "louvain_resolution": 1.0,
            "louvain_random_state": 42,
            "bridge_weight_betweenness": 0.7,
            "bridge_top_gap_min": 0.05,
        }
        cluster_map = compute_louvain_clustering(G, config, logger=None)

        # Compute bridge scores
        bridge_scores = compute_bridge_scores(G, cluster_map, betweenness, config)

        # Behavioral assertions:
        # 1. Bridge nodes (left_3, right_0) should be top-2
        sorted_nodes = sorted(bridge_scores.items(), key=lambda x: x[1], reverse=True)
        top_2_nodes = {sorted_nodes[0][0], sorted_nodes[1][0]}
        bridge_nodes = {"left_3", "right_0"}

        assert bridge_nodes == top_2_nodes, f"Expected {bridge_nodes} as top-2, got {top_2_nodes}"

        # 2. Gap between top-2 and rest should be >= bridge_top_gap_min
        if len(sorted_nodes) > 2:
            top_2_min = sorted_nodes[1][1]
            third_score = sorted_nodes[2][1]
            gap = top_2_min - third_score
            min_gap = config.get("bridge_top_gap_min", 0.05)

            assert gap >= min_gap, f"Gap {gap:.3f} < minimum {min_gap}"

        # 3. Mark inter-cluster edges
        mark_inter_cluster_edges(G, cluster_map)

        # The bridge edge should be marked as inter-cluster
        assert G["left_3"]["right_0"]["is_inter_cluster_edge"] is True
        assert G["right_0"]["left_3"]["is_inter_cluster_edge"] is True

    def test_inter_cluster_edges_ring(self):
        """Test inter-cluster edge marking in ring of 3 cliques."""
        # Create 3 K4 cliques in a ring
        G = nx.DiGraph()  # noqa: N806

        # Clique A
        for i in range(4):
            for j in range(4):
                if i != j:
                    G.add_edge(f"A_{i}", f"A_{j}", weight=1.0)

        # Clique B
        for i in range(4):
            for j in range(4):
                if i != j:
                    G.add_edge(f"B_{i}", f"B_{j}", weight=1.0)

        # Clique C
        for i in range(4):
            for j in range(4):
                if i != j:
                    G.add_edge(f"C_{i}", f"C_{j}", weight=1.0)

        # Connect in ring: A-B, B-C, C-A (bidirectional)
        bridges = [("A_3", "B_0"), ("B_3", "C_0"), ("C_3", "A_0")]

        for u, v in bridges:
            G.add_edge(u, v, weight=1.0)
            G.add_edge(v, u, weight=1.0)

        # Run Louvain
        config = {"louvain_resolution": 1.0, "louvain_random_state": 42}
        cluster_map = compute_louvain_clustering(G, config, logger=None)

        # Mark inter-cluster edges
        mark_inter_cluster_edges(G, cluster_map)

        # Behavioral assertions:
        # 1. Should have 3 clusters
        assert len(set(cluster_map.values())) == 3

        # 2. All bridge edges should be marked as inter-cluster
        for u, v in bridges:
            assert G[u][v]["is_inter_cluster_edge"] is True, f"Edge {u}->{v} not marked"
            assert G[v][u]["is_inter_cluster_edge"] is True, f"Edge {v}->{u} not marked"

            # Check cluster IDs are different
            assert G[u][v]["source_cluster_id"] != G[u][v]["target_cluster_id"]
            assert cluster_map[u] != cluster_map[v], f"Nodes {u} and {v} in same cluster"

        # 3. Intra-cluster edges should NOT be marked
        # Check some edges within clique A
        assert G["A_0"]["A_1"]["is_inter_cluster_edge"] is False
        assert G["A_1"]["A_2"]["is_inter_cluster_edge"] is False

    def test_clustering_determinism(self):
        """Test that Louvain produces identical results across multiple runs."""
        # Create a moderately complex graph
        G = nx.karate_club_graph().to_directed()  # noqa: N806

        # Add weights
        for u, v in G.edges():
            G[u][v]["weight"] = 1.0

        config = {"louvain_resolution": 1.0, "louvain_random_state": 42}

        # Run 10 times
        results = []
        for _ in range(10):
            cluster_map = compute_louvain_clustering(G, config, logger=None)
            # Convert to sorted tuple for comparison
            result = tuple(sorted(cluster_map.items()))
            results.append(result)

        # All results should be identical
        first_result = results[0]
        for i, result in enumerate(results[1:], 1):
            assert result == first_result, f"Run {i+1} differs from run 1"

        print("✓ All 10 runs produced identical clustering")

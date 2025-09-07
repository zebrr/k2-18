"""Extended tests for enhanced create_mention_index and link_nodes_to_concepts functions."""

import pytest

from viz.graph2metrics import create_mention_index, link_nodes_to_concepts


class TestCreateMentionIndexExtended:
    """Extended tests for create_mention_index with all edge types."""

    def test_mention_index_with_prerequisite_edges(self):
        """Test that PREREQUISITE edges are included in mention index."""
        graph_data = {
            "nodes": [
                {"id": "n1", "type": "Chunk"},
                {"id": "c1", "type": "Concept"},
                {"id": "c2", "type": "Concept"},
            ],
            "edges": [
                {"source": "n1", "target": "c1", "type": "PREREQUISITE"},
                {"source": "c1", "target": "c2", "type": "PREREQUISITE"},  # Concept-to-Concept
            ],
        }
        concepts_data = {"concepts": []}

        result = create_mention_index(graph_data, concepts_data)
        index = result["_meta"]["mention_index"]

        # c1 should have n1 and c2 in its index
        assert "c1" in index
        assert set(index["c1"]["nodes"]) == {"n1", "c2"}
        assert index["c1"]["count"] == 2

        # c2 should have c1 in its index
        assert "c2" in index
        assert set(index["c2"]["nodes"]) == {"c1"}
        assert index["c2"]["count"] == 1

    def test_mention_index_with_elaborates_edges(self):
        """Test that ELABORATES edges are included in mention index."""
        graph_data = {
            "nodes": [
                {"id": "n1", "type": "Chunk"},
                {"id": "n2", "type": "Chunk"},
                {"id": "c1", "type": "Concept"},
            ],
            "edges": [
                {"source": "n1", "target": "c1", "type": "ELABORATES"},
                {"source": "n2", "target": "c1", "type": "ELABORATES"},
            ],
        }
        concepts_data = {"concepts": []}

        result = create_mention_index(graph_data, concepts_data)
        index = result["_meta"]["mention_index"]

        assert "c1" in index
        assert set(index["c1"]["nodes"]) == {"n1", "n2"}
        assert index["c1"]["count"] == 2

    def test_mention_index_with_tests_edges(self):
        """Test that TESTS edges are included in mention index."""
        graph_data = {
            "nodes": [
                {"id": "a1", "type": "Assessment"},
                {"id": "c1", "type": "Concept"},
            ],
            "edges": [
                {"source": "a1", "target": "c1", "type": "TESTS"},
            ],
        }
        concepts_data = {"concepts": []}

        result = create_mention_index(graph_data, concepts_data)
        index = result["_meta"]["mention_index"]

        assert "c1" in index
        assert set(index["c1"]["nodes"]) == {"a1"}
        assert index["c1"]["count"] == 1

    def test_mention_index_concept_to_concept(self):
        """Test Concept-to-Concept relationships are indexed."""
        graph_data = {
            "nodes": [
                {"id": "c1", "type": "Concept"},
                {"id": "c2", "type": "Concept"},
                {"id": "c3", "type": "Concept"},
            ],
            "edges": [
                {"source": "c1", "target": "c2", "type": "PREREQUISITE"},
                {"source": "c2", "target": "c3", "type": "ELABORATES"},
            ],
        }
        concepts_data = {"concepts": []}

        result = create_mention_index(graph_data, concepts_data)
        index = result["_meta"]["mention_index"]

        # Each concept should reference the others it's connected to
        assert set(index["c1"]["nodes"]) == {"c2"}
        assert set(index["c2"]["nodes"]) == {"c1", "c3"}
        assert set(index["c3"]["nodes"]) == {"c2"}

    def test_mention_index_deduplication(self):
        """Test that multiple edges between same nodes are deduplicated."""
        graph_data = {
            "nodes": [
                {"id": "n1", "type": "Chunk"},
                {"id": "c1", "type": "Concept"},
            ],
            "edges": [
                {"source": "n1", "target": "c1", "type": "MENTIONS"},
                {"source": "n1", "target": "c1", "type": "PREREQUISITE"},
                {"source": "n1", "target": "c1", "type": "ELABORATES"},
            ],
        }
        concepts_data = {"concepts": []}

        result = create_mention_index(graph_data, concepts_data)
        index = result["_meta"]["mention_index"]

        # n1 should appear only once despite multiple edges
        assert index["c1"]["nodes"] == ["n1"]
        assert index["c1"]["count"] == 1

    def test_mention_index_mixed_edge_types(self):
        """Test mention index with mixed edge types."""
        graph_data = {
            "nodes": [
                {"id": "n1", "type": "Chunk"},
                {"id": "n2", "type": "Chunk"},
                {"id": "a1", "type": "Assessment"},
                {"id": "c1", "type": "Concept"},
                {"id": "c2", "type": "Concept"},
            ],
            "edges": [
                {"source": "n1", "target": "c1", "type": "MENTIONS"},
                {"source": "n2", "target": "c1", "type": "PREREQUISITE"},
                {"source": "a1", "target": "c1", "type": "TESTS"},
                {"source": "c1", "target": "c2", "type": "ELABORATES"},
                {"source": "n1", "target": "n2", "type": "PREREQUISITE"},  # Non-concept edge
            ],
        }
        concepts_data = {"concepts": []}

        result = create_mention_index(graph_data, concepts_data)
        index = result["_meta"]["mention_index"]

        # c1 should have n1, n2, a1, and c2
        assert set(index["c1"]["nodes"]) == {"n1", "n2", "a1", "c2"}
        assert index["c1"]["count"] == 4

        # c2 should have c1
        assert set(index["c2"]["nodes"]) == {"c1"}
        assert index["c2"]["count"] == 1


class TestLinkNodesToConceptsExtended:
    """Extended tests for link_nodes_to_concepts with all edge types."""

    def test_link_with_prerequisite_edges(self):
        """Test that PREREQUISITE edges link nodes to concepts."""
        graph_data = {
            "nodes": [
                {"id": "n1", "type": "Chunk"},
                {"id": "c1", "type": "Concept"},
            ],
            "edges": [
                {"source": "n1", "target": "c1", "type": "PREREQUISITE"},
            ],
        }

        result = link_nodes_to_concepts(graph_data)
        nodes_by_id = {n["id"]: n for n in result["nodes"]}

        assert set(nodes_by_id["n1"]["concepts"]) == {"c1"}
        # Concept nodes don't reference non-Concept nodes
        assert nodes_by_id["c1"]["concepts"] == []

    def test_link_with_elaborates_edges(self):
        """Test that ELABORATES edges link nodes to concepts."""
        graph_data = {
            "nodes": [
                {"id": "n1", "type": "Chunk"},
                {"id": "c1", "type": "Concept"},
            ],
            "edges": [
                {"source": "n1", "target": "c1", "type": "ELABORATES"},
            ],
        }

        result = link_nodes_to_concepts(graph_data)
        nodes_by_id = {n["id"]: n for n in result["nodes"]}

        assert set(nodes_by_id["n1"]["concepts"]) == {"c1"}
        # Concept nodes don't reference non-Concept nodes
        assert nodes_by_id["c1"]["concepts"] == []

    def test_link_with_example_of_edges(self):
        """Test that EXAMPLE_OF edges link nodes to concepts."""
        graph_data = {
            "nodes": [
                {"id": "n1", "type": "Chunk"},
                {"id": "c1", "type": "Concept"},
            ],
            "edges": [
                {"source": "n1", "target": "c1", "type": "EXAMPLE_OF"},
            ],
        }

        result = link_nodes_to_concepts(graph_data)
        nodes_by_id = {n["id"]: n for n in result["nodes"]}

        assert set(nodes_by_id["n1"]["concepts"]) == {"c1"}
        # Concept nodes don't reference non-Concept nodes
        assert nodes_by_id["c1"]["concepts"] == []

    def test_link_assessment_with_tests_edges(self):
        """Test that Assessment nodes get linked via TESTS edges."""
        graph_data = {
            "nodes": [
                {"id": "a1", "type": "Assessment"},
                {"id": "c1", "type": "Concept"},
            ],
            "edges": [
                {"source": "a1", "target": "c1", "type": "TESTS"},
            ],
        }

        result = link_nodes_to_concepts(graph_data)
        nodes_by_id = {n["id"]: n for n in result["nodes"]}

        assert set(nodes_by_id["a1"]["concepts"]) == {"c1"}
        # Concept nodes don't reference non-Concept nodes
        assert nodes_by_id["c1"]["concepts"] == []

    def test_link_concept_to_concept(self):
        """Test that Concept-to-Concept relationships are established."""
        graph_data = {
            "nodes": [
                {"id": "c1", "type": "Concept"},
                {"id": "c2", "type": "Concept"},
                {"id": "c3", "type": "Concept"},
            ],
            "edges": [
                {"source": "c1", "target": "c2", "type": "PREREQUISITE"},
                {"source": "c2", "target": "c3", "type": "ELABORATES"},
            ],
        }

        result = link_nodes_to_concepts(graph_data)
        nodes_by_id = {n["id"]: n for n in result["nodes"]}

        # c1 connects to c2
        assert set(nodes_by_id["c1"]["concepts"]) == {"c2"}
        # c2 connects to both c1 and c3
        assert set(nodes_by_id["c2"]["concepts"]) == {"c1", "c3"}
        # c3 connects to c2
        assert set(nodes_by_id["c3"]["concepts"]) == {"c2"}

    def test_link_deduplication(self):
        """Test that multiple edges are deduplicated."""
        graph_data = {
            "nodes": [
                {"id": "n1", "type": "Chunk"},
                {"id": "c1", "type": "Concept"},
            ],
            "edges": [
                {"source": "n1", "target": "c1", "type": "MENTIONS"},
                {"source": "n1", "target": "c1", "type": "PREREQUISITE"},
                {"source": "n1", "target": "c1", "type": "ELABORATES"},
            ],
        }

        result = link_nodes_to_concepts(graph_data)
        nodes_by_id = {n["id"]: n for n in result["nodes"]}

        # c1 should appear only once
        assert nodes_by_id["n1"]["concepts"] == ["c1"]
        assert len(nodes_by_id["n1"]["concepts"]) == 1

    def test_link_bidirectional_concept_edges(self):
        """Test bidirectional linking when Concept is source."""
        graph_data = {
            "nodes": [
                {"id": "n1", "type": "Chunk"},
                {"id": "c1", "type": "Concept"},
            ],
            "edges": [
                {"source": "c1", "target": "n1", "type": "ELABORATES"},  # Concept as source
            ],
        }

        result = link_nodes_to_concepts(graph_data)
        nodes_by_id = {n["id"]: n for n in result["nodes"]}

        # n1 should reference c1, but c1 shouldn't reference n1 (only other Concepts)
        assert set(nodes_by_id["n1"]["concepts"]) == {"c1"}
        assert nodes_by_id["c1"]["concepts"] == []

    def test_link_mixed_types_comprehensive(self):
        """Test comprehensive scenario with all node and edge types."""
        graph_data = {
            "nodes": [
                {"id": "ch1", "type": "Chunk"},
                {"id": "ch2", "type": "Chunk"},
                {"id": "a1", "type": "Assessment"},
                {"id": "c1", "type": "Concept"},
                {"id": "c2", "type": "Concept"},
                {"id": "c3", "type": "Concept"},
            ],
            "edges": [
                # Chunk to Concept edges
                {"source": "ch1", "target": "c1", "type": "MENTIONS"},
                {"source": "ch2", "target": "c1", "type": "PREREQUISITE"},
                {"source": "ch2", "target": "c2", "type": "ELABORATES"},
                # Assessment to Concept
                {"source": "a1", "target": "c2", "type": "TESTS"},
                {"source": "a1", "target": "c3", "type": "EXAMPLE_OF"},
                # Concept to Concept
                {"source": "c1", "target": "c2", "type": "PREREQUISITE"},
                {"source": "c2", "target": "c3", "type": "ELABORATES"},
                # Non-concept edges (should be ignored)
                {"source": "ch1", "target": "ch2", "type": "PREREQUISITE"},
                {"source": "ch2", "target": "a1", "type": "FOLLOWED_BY"},
            ],
        }

        result = link_nodes_to_concepts(graph_data)
        nodes_by_id = {n["id"]: n for n in result["nodes"]}

        # Check Chunk nodes
        assert set(nodes_by_id["ch1"]["concepts"]) == {"c1"}
        assert set(nodes_by_id["ch2"]["concepts"]) == {"c1", "c2"}

        # Check Assessment node
        assert set(nodes_by_id["a1"]["concepts"]) == {"c2", "c3"}

        # Check Concept nodes (only reference other Concepts)
        assert set(nodes_by_id["c1"]["concepts"]) == {"c2"}
        assert set(nodes_by_id["c2"]["concepts"]) == {"c1", "c3"}
        assert set(nodes_by_id["c3"]["concepts"]) == {"c2"}

    def test_link_empty_graph(self):
        """Test with empty graph."""
        graph_data = {"nodes": [], "edges": []}

        result = link_nodes_to_concepts(graph_data)

        assert result["nodes"] == []
        assert result["edges"] == []

    def test_link_no_concept_nodes(self):
        """Test graph with no Concept nodes."""
        graph_data = {
            "nodes": [
                {"id": "n1", "type": "Chunk"},
                {"id": "n2", "type": "Chunk"},
                {"id": "a1", "type": "Assessment"},
            ],
            "edges": [
                {"source": "n1", "target": "n2", "type": "PREREQUISITE"},
                {"source": "n2", "target": "a1", "type": "FOLLOWED_BY"},
            ],
        }

        result = link_nodes_to_concepts(graph_data)
        
        # All nodes should have empty concepts list
        for node in result["nodes"]:
            assert node["concepts"] == []
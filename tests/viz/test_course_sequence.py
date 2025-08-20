#!/usr/bin/env python3
"""Tests for course sequence generation in graph2metrics."""

from pathlib import Path

import pytest

# Add src and viz to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from viz.graph2metrics import generate_course_sequence


class TestCourseSequence:
    """Test course sequence generation functionality."""

    def test_course_sequence_generation(self):
        """Test basic course sequence generation from Chunk nodes."""
        # Create sample graph with Chunk nodes
        graph_data = {
            "nodes": [
                {"id": "test:c:1", "type": "Chunk", "cluster_id": 0},
                {"id": "test:c:3", "type": "Chunk", "cluster_id": 1},
                {"id": "test:c:2", "type": "Chunk", "cluster_id": 0},
                {"id": "concept:1", "type": "Concept", "cluster_id": 2},
            ],
            "edges": [],
        }

        # Generate course sequence
        result = generate_course_sequence(graph_data)

        # Check that _meta was created
        assert "_meta" in result
        assert "course_sequence" in result["_meta"]

        # Check sequence content
        sequence = result["_meta"]["course_sequence"]
        assert len(sequence) == 3  # Only Chunk nodes

        # Check each item has required fields
        for item in sequence:
            assert "id" in item
            assert "cluster_id" in item
            assert "position" in item

    def test_course_sequence_sorting(self):
        """Test that course sequence is sorted by position."""
        graph_data = {
            "nodes": [
                {"id": "test:c:5", "type": "Chunk", "cluster_id": 0},
                {"id": "test:c:1", "type": "Chunk", "cluster_id": 1},
                {"id": "test:c:3", "type": "Chunk", "cluster_id": 2},
                {"id": "test:c:2", "type": "Chunk", "cluster_id": 1},
                {"id": "test:c:4", "type": "Chunk", "cluster_id": 0},
            ],
            "edges": [],
        }

        result = generate_course_sequence(graph_data)
        sequence = result["_meta"]["course_sequence"]

        # Check sorting
        assert len(sequence) == 5
        positions = [item["position"] for item in sequence]
        assert positions == [1, 2, 3, 4, 5]

        # Check IDs are in correct order
        assert sequence[0]["id"] == "test:c:1"
        assert sequence[1]["id"] == "test:c:2"
        assert sequence[2]["id"] == "test:c:3"
        assert sequence[3]["id"] == "test:c:4"
        assert sequence[4]["id"] == "test:c:5"

    def test_course_sequence_with_clusters(self):
        """Test that cluster_id is preserved in course sequence."""
        graph_data = {
            "nodes": [
                {"id": "test:c:1", "type": "Chunk", "cluster_id": 3},
                {"id": "test:c:2", "type": "Chunk", "cluster_id": 1},
                {"id": "test:c:3", "type": "Chunk", "cluster_id": 2},
            ],
            "edges": [],
        }

        result = generate_course_sequence(graph_data)
        sequence = result["_meta"]["course_sequence"]

        # Check cluster IDs are preserved
        assert sequence[0]["cluster_id"] == 3
        assert sequence[1]["cluster_id"] == 1
        assert sequence[2]["cluster_id"] == 2

    def test_empty_chunks(self):
        """Test with graph without Chunk nodes."""
        graph_data = {
            "nodes": [
                {"id": "concept:1", "type": "Concept"},
                {"id": "assessment:1", "type": "Assessment"},
            ],
            "edges": [],
        }

        result = generate_course_sequence(graph_data)
        sequence = result["_meta"]["course_sequence"]

        # Should be empty list
        assert sequence == []

    def test_invalid_chunk_id_format(self):
        """Test handling of Chunk nodes with non-standard ID format."""
        graph_data = {
            "nodes": [
                {"id": "test:c:1", "type": "Chunk", "cluster_id": 0},
                {"id": "invalid_chunk", "type": "Chunk", "cluster_id": 1},  # No :c:
                {"id": "test:c:2", "type": "Chunk", "cluster_id": 0},
                {"id": "test:c:invalid", "type": "Chunk", "cluster_id": 2},  # Non-numeric position
            ],
            "edges": [],
        }

        result = generate_course_sequence(graph_data)
        sequence = result["_meta"]["course_sequence"]

        # Should only include valid chunks
        assert len(sequence) == 2
        assert sequence[0]["id"] == "test:c:1"
        assert sequence[1]["id"] == "test:c:2"

    def test_missing_cluster_id(self):
        """Test handling of nodes without cluster_id."""
        graph_data = {
            "nodes": [
                {"id": "test:c:1", "type": "Chunk"},  # No cluster_id
                {"id": "test:c:2", "type": "Chunk", "cluster_id": 5},
            ],
            "edges": [],
        }

        result = generate_course_sequence(graph_data)
        sequence = result["_meta"]["course_sequence"]

        # Default cluster_id should be 0
        assert sequence[0]["cluster_id"] == 0
        assert sequence[1]["cluster_id"] == 5

    def test_preserves_existing_meta(self):
        """Test that existing _meta content is preserved."""
        graph_data = {
            "nodes": [{"id": "test:c:1", "type": "Chunk"}],
            "edges": [],
            "_meta": {"existing_field": "value"},
        }

        result = generate_course_sequence(graph_data)

        # Check existing field is preserved
        assert result["_meta"]["existing_field"] == "value"
        assert "course_sequence" in result["_meta"]

    def test_large_position_numbers(self):
        """Test handling of large position numbers."""
        graph_data = {
            "nodes": [
                {"id": "test:c:999", "type": "Chunk", "cluster_id": 0},
                {"id": "test:c:1", "type": "Chunk", "cluster_id": 1},
                {"id": "test:c:100", "type": "Chunk", "cluster_id": 2},
            ],
            "edges": [],
        }

        result = generate_course_sequence(graph_data)
        sequence = result["_meta"]["course_sequence"]

        # Check sorting with large numbers
        positions = [item["position"] for item in sequence]
        assert positions == [1, 100, 999]

    def test_duplicate_positions(self):
        """Test handling of duplicate position numbers."""
        # This shouldn't happen in real data, but test stability
        graph_data = {
            "nodes": [
                {"id": "test:c:1", "type": "Chunk", "cluster_id": 0},
                {"id": "another:c:1", "type": "Chunk", "cluster_id": 1},  # Same position
                {"id": "test:c:2", "type": "Chunk", "cluster_id": 2},
            ],
            "edges": [],
        }

        result = generate_course_sequence(graph_data)
        sequence = result["_meta"]["course_sequence"]

        # Should include all nodes, stable sort
        assert len(sequence) == 3
        positions = [item["position"] for item in sequence]
        assert positions == [1, 1, 2]

    @pytest.mark.parametrize(
        "node_id,expected_position",
        [
            ("test:c:0", 0),
            ("test:c:1", 1),
            ("test:c:10", 10),
            ("test:c:100", 100),
            ("slug:c:5", 5),
            ("long_slug_name:c:7", 7),
        ],
    )
    def test_position_extraction(self, node_id, expected_position):
        """Test position extraction from various ID formats."""
        graph_data = {
            "nodes": [{"id": node_id, "type": "Chunk"}],
            "edges": [],
        }

        result = generate_course_sequence(graph_data)
        sequence = result["_meta"]["course_sequence"]

        assert len(sequence) == 1
        assert sequence[0]["position"] == expected_position

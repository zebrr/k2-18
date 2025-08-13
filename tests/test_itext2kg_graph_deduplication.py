#!/usr/bin/env python3
"""
Tests for node deduplication functionality in itext2kg_graph.
"""

from unittest.mock import Mock, patch

import pytest

from src.itext2kg_graph import SliceData, SliceProcessor


class TestNodeDeduplication:
    """Test node deduplication functionality."""

    @pytest.fixture
    def processor(self):
        """Create a SliceProcessor instance for testing."""
        config = {
            "itext2kg": {
                "model": "test-model",
                "tpm_limit": 100000,
                "log_level": "debug",
                "max_completion": 25000,
                "temperature": 0.6,
                "api_key": "test-key",  # Add api_key for OpenAIClient
            },
            "slicer": {"overlap": 500, "max_tokens": 5000},
        }

        # Mock OpenAIClient and ConceptDictionary loading
        with patch("src.itext2kg_graph.OpenAIClient") as mock_client_class, patch.object(
            SliceProcessor, "_load_concept_dictionary"
        ) as mock_load:

            # Setup mock client
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            # Setup concept dictionary
            mock_load.return_value = {
                "concepts": [
                    {
                        "concept_id": "test:p:concept1",
                        "term": {"primary": "Concept 1"},
                        "definition": "Definition of concept 1",
                    },
                    {
                        "concept_id": "test:p:concept2",
                        "term": {"primary": "Concept 2"},
                        "definition": "Definition of concept 2",
                    },
                ]
            }

            processor = SliceProcessor(config)
            processor.llm_client = mock_client
            return processor

    def test_removes_duplicate_concepts_silently(self, processor):
        """Duplicate Concept nodes should be removed without warnings."""
        # Setup existing graph with a concept
        processor.graph_nodes = [{"id": "test:p:concept1", "type": "Concept", "text": "Concept 1"}]

        # Create patch with duplicate concept
        patch = {
            "nodes": [
                {"id": "test:p:concept1", "type": "Concept", "text": "Concept 1 duplicate"},
                {"id": "test:c:100", "type": "Chunk", "text": "New chunk"},
            ],
            "edges": [],
        }

        # Apply deduplication
        result = processor._deduplicate_patch_nodes(
            {"nodes": processor.graph_nodes}, patch, "slice_01"
        )

        # Assert only new chunk remains
        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["id"] == "test:c:100"

        # Check statistics
        assert processor.quality_issues["duplicate_concepts_removed"] == 1
        assert len(processor.quality_issues["anomalous_duplicates"]) == 0

    def test_warns_on_duplicate_chunks(self, processor, caplog):
        """Duplicate Chunk nodes should trigger warnings."""
        # Setup existing graph with a chunk
        processor.graph_nodes = [{"id": "test:c:100", "type": "Chunk", "text": "Existing chunk"}]

        # Create patch with duplicate chunk
        patch = {
            "nodes": [
                {"id": "test:c:100", "type": "Chunk", "text": "Duplicate chunk"},
                {"id": "test:c:200", "type": "Chunk", "text": "New chunk"},
            ],
            "edges": [],
        }

        # Apply deduplication
        result = processor._deduplicate_patch_nodes(
            {"nodes": processor.graph_nodes}, patch, "slice_02"
        )

        # Assert only new chunk remains
        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["id"] == "test:c:200"

        # Check warning was logged
        assert "Unexpected duplicate Chunk node: test:c:100" in caplog.text

        # Check anomalous duplicates tracked
        assert len(processor.quality_issues["anomalous_duplicates"]) == 1
        assert processor.quality_issues["anomalous_duplicates"][0]["node_id"] == "test:c:100"
        assert processor.quality_issues["anomalous_duplicates"][0]["slice_id"] == "slice_02"

    def test_warns_on_duplicate_assessments(self, processor, caplog):
        """Duplicate Assessment nodes should trigger warnings."""
        # Setup existing graph with an assessment
        processor.graph_nodes = [{"id": "test:q:100:1", "type": "Assessment", "text": "Question 1"}]

        # Create patch with duplicate assessment
        patch = {
            "nodes": [
                {"id": "test:q:100:1", "type": "Assessment", "text": "Question 1 duplicate"},
                {"id": "test:q:200:1", "type": "Assessment", "text": "Question 2"},
            ],
            "edges": [],
        }

        # Apply deduplication
        result = processor._deduplicate_patch_nodes(
            {"nodes": processor.graph_nodes}, patch, "slice_03"
        )

        # Assert only new assessment remains
        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["id"] == "test:q:200:1"

        # Check warning was logged
        assert "Unexpected duplicate Assessment node: test:q:100:1" in caplog.text

        # Check anomalous duplicates tracked
        assert len(processor.quality_issues["anomalous_duplicates"]) == 1

    def test_preserves_edges_after_deduplication(self, processor):
        """All edges should remain valid after removing duplicates."""
        # Setup existing graph
        processor.graph_nodes = [
            {"id": "test:c:100", "type": "Chunk", "text": "Chunk 1"},
            {"id": "test:p:concept1", "type": "Concept", "text": "Concept 1"},
        ]

        # Create patch with duplicate concept and edges
        patch = {
            "nodes": [
                {"id": "test:p:concept1", "type": "Concept", "text": "Duplicate"},
                {"id": "test:c:200", "type": "Chunk", "text": "Chunk 2"},
            ],
            "edges": [
                {"source": "test:c:200", "target": "test:p:concept1", "type": "MENTIONS"},
                {"source": "test:c:100", "target": "test:c:200", "type": "PREREQUISITE"},
            ],
        }

        # Apply deduplication
        result = processor._deduplicate_patch_nodes(
            {"nodes": processor.graph_nodes}, patch, "slice_04"
        )

        # Edges should remain unchanged
        assert len(result["edges"]) == 2
        assert result["edges"] == patch["edges"]

    def test_tracks_deduplication_statistics(self, processor):
        """Metadata should contain deduplication stats."""
        # Reset statistics
        processor.quality_issues = {"duplicate_concepts_removed": 0, "anomalous_duplicates": []}

        # Setup existing graph
        processor.graph_nodes = [
            {"id": "test:p:concept1", "type": "Concept", "text": "Concept 1"},
            {"id": "test:p:concept2", "type": "Concept", "text": "Concept 2"},
            {"id": "test:c:100", "type": "Chunk", "text": "Chunk 1"},
        ]

        # Apply multiple patches with duplicates
        patch1 = {
            "nodes": [
                {"id": "test:p:concept1", "type": "Concept", "text": "Dup 1"},
                {"id": "test:p:concept2", "type": "Concept", "text": "Dup 2"},
            ]
        }
        processor._deduplicate_patch_nodes({"nodes": processor.graph_nodes}, patch1, "slice_05")

        patch2 = {
            "nodes": [
                {"id": "test:c:100", "type": "Chunk", "text": "Duplicate chunk"},
                {"id": "test:p:concept1", "type": "Concept", "text": "Another dup"},
            ]
        }
        processor._deduplicate_patch_nodes({"nodes": processor.graph_nodes}, patch2, "slice_06")

        # Check statistics
        assert processor.quality_issues["duplicate_concepts_removed"] == 3  # 2 + 1
        assert len(processor.quality_issues["anomalous_duplicates"]) == 1  # 1 chunk

    def test_validate_node_uniqueness_detects_duplicates(self, processor):
        """Validation should detect any remaining duplicates."""
        # Create graph with duplicates
        graph = {
            "nodes": [
                {"id": "test:c:100", "type": "Chunk", "text": "Chunk 1"},
                {"id": "test:p:concept1", "type": "Concept", "text": "Concept 1"},
                {"id": "test:c:100", "type": "Chunk", "text": "Duplicate chunk"},
                {"id": "test:p:concept1", "type": "Concept", "text": "Duplicate concept"},
            ],
            "edges": [],
        }

        # Run validation
        is_valid, duplicates = processor._validate_node_uniqueness(graph)

        # Should detect duplicates
        assert is_valid is False
        assert len(duplicates) == 2

        # Check duplicate details
        duplicate_ids = [d["id"] for d in duplicates]
        assert "test:c:100" in duplicate_ids
        assert "test:p:concept1" in duplicate_ids

    def test_validate_node_uniqueness_passes_clean_graph(self, processor):
        """Validation should pass for graph without duplicates."""
        # Create clean graph
        graph = {
            "nodes": [
                {"id": "test:c:100", "type": "Chunk", "text": "Chunk 1"},
                {"id": "test:c:200", "type": "Chunk", "text": "Chunk 2"},
                {"id": "test:p:concept1", "type": "Concept", "text": "Concept 1"},
                {"id": "test:q:300:1", "type": "Assessment", "text": "Question 1"},
            ],
            "edges": [],
        }

        # Run validation
        is_valid, duplicates = processor._validate_node_uniqueness(graph)

        # Should pass
        assert is_valid is True
        assert len(duplicates) == 0

    def test_deduplication_within_patch(self, processor):
        """Should detect duplicates within the same patch."""
        # Empty graph
        processor.graph_nodes = []

        # Create patch with internal duplicates
        patch = {
            "nodes": [
                {"id": "test:c:100", "type": "Chunk", "text": "Chunk 1"},
                {"id": "test:c:100", "type": "Chunk", "text": "Duplicate in same patch"},
                {"id": "test:p:concept1", "type": "Concept", "text": "Concept 1"},
                {"id": "test:p:concept1", "type": "Concept", "text": "Duplicate concept"},
            ],
            "edges": [],
        }

        # Apply deduplication
        result = processor._deduplicate_patch_nodes(
            {"nodes": processor.graph_nodes}, patch, "slice_07"
        )

        # Should keep only first occurrence of each
        assert len(result["nodes"]) == 2
        node_ids = [n["id"] for n in result["nodes"]]
        assert "test:c:100" in node_ids
        assert "test:p:concept1" in node_ids

        # Check that it kept the first occurrence
        assert result["nodes"][0]["text"] == "Chunk 1"
        assert result["nodes"][1]["text"] == "Concept 1"

    def test_empty_patch_handling(self, processor):
        """Should handle empty patches gracefully."""
        # Setup existing graph
        processor.graph_nodes = [{"id": "test:c:100", "type": "Chunk", "text": "Existing chunk"}]

        # Empty patch
        patch = {"nodes": [], "edges": []}

        # Apply deduplication
        result = processor._deduplicate_patch_nodes(
            {"nodes": processor.graph_nodes}, patch, "slice_08"
        )

        # Should return empty nodes list
        assert result["nodes"] == []
        assert result["edges"] == []

    @pytest.mark.timeout(120)
    def test_integration_with_add_to_graph(self, processor):
        """Test deduplication integration in _add_to_graph method."""
        # Setup processor state
        processor.graph_nodes = [
            {"id": "test:p:concept1", "type": "Concept", "text": "Existing concept"}
        ]
        processor.graph_edges = []
        processor.node_ids = {"test:p:concept1": 0}

        # Create slice data
        slice_data = SliceData(
            id="slice_09",
            order=1,
            source_file="test.txt",
            slug="test",
            text="Test text",
            slice_token_start=0,
            slice_token_end=100,
        )

        # Create patch with duplicate concept
        patch = {
            "nodes": [
                {"id": "test:p:concept1", "type": "Concept", "text": "Duplicate", "node_offset": 0},
                {
                    "id": "test:c:100",
                    "type": "Chunk",
                    "text": "New chunk",
                    "node_offset": 50,
                    "difficulty": 3,
                },
            ],
            "edges": [],
        }

        # Call _add_to_graph
        processor._add_to_graph(patch, slice_data)

        # Should have only added the new chunk
        assert len(processor.graph_nodes) == 2  # Original concept + new chunk
        assert processor.graph_nodes[1]["id"] == "test:c:100"

        # Check deduplication statistics
        assert processor.quality_issues["duplicate_concepts_removed"] == 1

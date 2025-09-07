#!/usr/bin/env python3
"""
Tests for graph_fix.py utility.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from viz.graph_fix import (
    load_input_files,
    process_chunk_assessment_definitions,
    process_concept_text,
    process_edge_conditions,
    update_metadata,
)


@pytest.mark.viz
class TestGraphFix:
    """Tests for graph_fix utility."""

    @pytest.fixture
    def sample_graph(self):
        """Sample graph data for testing."""
        return {
            "nodes": [
                {
                    "id": "chunk_1",
                    "type": "Chunk",
                    "text": "Sample chunk",
                    "node_offset": 0,
                    "definition": "This is a chunk definition",
                },
                {
                    "id": "assessment_1",
                    "type": "Assessment",
                    "text": "Test question",
                    "node_offset": 100,
                    "definition": "Assessment of understanding",
                },
                {"id": "concept_1", "type": "Concept", "text": "Algorithm", "node_offset": 200},
                {
                    "id": "chunk_2",
                    "type": "Chunk",
                    "text": "Another chunk",
                    "node_offset": 300,
                    "definition": "[added_by=LLM] Already marked",
                },
                {
                    "id": "chunk_3",
                    "type": "Chunk",
                    "text": "Empty def chunk",
                    "node_offset": 400,
                    "definition": "",
                },
            ],
            "edges": [
                {
                    "source": "chunk_1",
                    "target": "concept_1",
                    "type": "MENTIONS",
                    "weight": 0.8,
                    "conditions": "Chunk mentions the concept",
                },
                {
                    "source": "chunk_1",
                    "target": "assessment_1",
                    "type": "TESTS",
                    "weight": 0.9,
                    "conditions": "Tests understanding",
                },
                {
                    "source": "chunk_2",
                    "target": "concept_1",
                    "type": "ELABORATES",
                    "weight": 0.7,
                    "conditions": "Has added_by=LLM marker already",
                },
                {
                    "source": "chunk_1",
                    "target": "chunk_2",
                    "type": "PREREQUISITE",
                    "weight": 0.6,
                    "conditions": "auto_generated connection",
                },
                {"source": "assessment_1", "target": "chunk_3", "type": "TESTS", "weight": 0.5},
            ],
        }

    @pytest.fixture
    def sample_concepts(self):
        """Sample concept dictionary for testing."""
        return {
            "concepts": [
                {
                    "concept_id": "concept_1",
                    "term": {"primary": "Algorithm", "aliases": ["алгоритм", "procedure"]},
                    "definition": "A step-by-step procedure",
                },
                {
                    "concept_id": "concept_2",
                    "term": {"primary": "Data Structure"},
                    "definition": "Organization of data",
                },
            ]
        }

    def test_process_chunk_assessment_definitions(self, sample_graph):
        """Test marking definitions in Chunk and Assessment nodes."""
        logger = MagicMock()

        chunks_marked, assessments_marked, examples = process_chunk_assessment_definitions(
            sample_graph["nodes"], dry_run=False, logger=logger
        )

        # Check counts
        assert chunks_marked == 1  # Only chunk_1 should be marked
        assert assessments_marked == 1  # assessment_1 should be marked

        # Check modifications
        assert sample_graph["nodes"][0]["definition"] == "[added_by=LLM] This is a chunk definition"
        assert (
            sample_graph["nodes"][1]["definition"] == "[added_by=LLM] Assessment of understanding"
        )
        assert (
            sample_graph["nodes"][3]["definition"] == "[added_by=LLM] Already marked"
        )  # No double marking
        assert sample_graph["nodes"][4]["definition"] == ""  # Empty not touched

    def test_process_chunk_assessment_definitions_dry_run(self, sample_graph):
        """Test dry-run mode doesn't modify data."""
        logger = MagicMock()
        original_def = sample_graph["nodes"][0]["definition"]

        chunks_marked, assessments_marked, examples = process_chunk_assessment_definitions(
            sample_graph["nodes"], dry_run=True, logger=logger
        )

        # Check counts
        assert chunks_marked == 1
        assert assessments_marked == 1
        assert len(examples) == 2

        # Check no modifications
        assert sample_graph["nodes"][0]["definition"] == original_def

    def test_process_chunk_assessment_definitions_idempotency(self, sample_graph):
        """Test that running twice doesn't double-mark."""
        logger = MagicMock()

        # First run
        process_chunk_assessment_definitions(sample_graph["nodes"], dry_run=False, logger=logger)

        # Second run
        chunks_marked, assessments_marked, _ = process_chunk_assessment_definitions(
            sample_graph["nodes"], dry_run=False, logger=logger
        )

        # Should not mark again
        assert chunks_marked == 0
        assert assessments_marked == 0

        # Check no double marking
        assert not sample_graph["nodes"][0]["definition"].startswith(
            "[added_by=LLM] [added_by=LLM]"
        )

    def test_process_concept_text(self, sample_graph, sample_concepts):
        """Test updating Concept text from dictionary."""
        logger = MagicMock()

        concepts_updated, examples = process_concept_text(
            sample_graph["nodes"], sample_concepts, dry_run=False, logger=logger
        )

        # Check count
        assert concepts_updated == 1

        # Check modification
        assert sample_graph["nodes"][2]["text"] == "Algorithm (алгоритм, procedure)"

    def test_process_concept_text_no_aliases(self, sample_concepts):
        """Test concept without aliases."""
        nodes = [{"id": "concept_2", "type": "Concept", "text": "Old text", "node_offset": 0}]
        logger = MagicMock()

        concepts_updated, _ = process_concept_text(
            nodes, sample_concepts, dry_run=False, logger=logger
        )

        assert concepts_updated == 1
        assert nodes[0]["text"] == "Data Structure"

    def test_process_concept_text_missing_concept(self, sample_graph, sample_concepts):
        """Test handling missing concept in dictionary."""
        # Add a concept node that's not in dictionary
        sample_graph["nodes"].append(
            {"id": "concept_missing", "type": "Concept", "text": "Missing", "node_offset": 500}
        )

        logger = MagicMock()

        with patch("builtins.print") as mock_print:
            concepts_updated, _ = process_concept_text(
                sample_graph["nodes"], sample_concepts, dry_run=False, logger=logger
            )

        # Should log warning
        logger.warning.assert_called_with("Concept not found in dictionary: concept_missing")
        mock_print.assert_called_with("WARNING: Concept not found in dictionary: concept_missing")

        # Should update only the found concept
        assert concepts_updated == 1

    def test_process_edge_conditions(self, sample_graph):
        """Test marking conditions in edges."""
        logger = MagicMock()

        edges_marked, examples = process_edge_conditions(
            sample_graph["edges"], dry_run=False, logger=logger
        )

        # Check count - only 2 should be marked
        assert edges_marked == 2

        # Check modifications
        assert sample_graph["edges"][0]["conditions"] == "[added_by=LLM] Chunk mentions the concept"
        assert sample_graph["edges"][1]["conditions"] == "[added_by=LLM] Tests understanding"

        # These should not be marked (already contain skip markers)
        assert sample_graph["edges"][2]["conditions"] == "Has added_by=LLM marker already"  # Not changed
        assert not sample_graph["edges"][3]["conditions"].startswith(
            "[added_by=LLM]"
        )  # Has auto_generated

    def test_process_edge_conditions_skip_markers(self):
        """Test that edges with skip markers are not processed."""
        edges = [
            {"source": "a", "target": "b", "type": "TEST", "conditions": "Normal condition"},
            {"source": "c", "target": "d", "type": "TEST", "conditions": "Has added_by= in middle"},
            {"source": "e", "target": "f", "type": "TEST", "conditions": "fixed_by=user somewhere"},
            {"source": "g", "target": "h", "type": "TEST", "conditions": "auto_generated edge"},
        ]

        logger = MagicMock()
        edges_marked, _ = process_edge_conditions(edges, dry_run=False, logger=logger)

        # Only first edge should be marked
        assert edges_marked == 1
        assert edges[0]["conditions"].startswith("[added_by=LLM]")
        assert not edges[1]["conditions"].startswith("[added_by=LLM]")
        assert not edges[2]["conditions"].startswith("[added_by=LLM]")
        assert not edges[3]["conditions"].startswith("[added_by=LLM]")

    def test_process_edge_conditions_empty(self, sample_graph):
        """Test that edges without conditions are skipped."""
        logger = MagicMock()

        # Edge at index 4 has no conditions
        original_edges = len(sample_graph["edges"])
        assert "conditions" not in sample_graph["edges"][4]

        edges_marked, _ = process_edge_conditions(
            sample_graph["edges"], dry_run=False, logger=logger
        )

        # Should not crash, edge should be skipped
        assert len(sample_graph["edges"]) == original_edges
        assert "conditions" not in sample_graph["edges"][4]

    def test_update_metadata(self, sample_graph):
        """Test metadata update."""
        logger = MagicMock()
        stats = {
            "chunks_marked": 5,
            "assessments_marked": 3,
            "concepts_updated": 10,
            "edges_marked": 7,
        }

        update_metadata(sample_graph, stats, logger)

        assert "graph_fix_applied" in sample_graph["_meta"]
        fix_meta = sample_graph["_meta"]["graph_fix_applied"]
        assert fix_meta["chunks_definitions_marked"] == 5
        assert fix_meta["assessments_definitions_marked"] == 3
        assert fix_meta["concepts_text_updated"] == 10
        assert fix_meta["edges_conditions_marked"] == 7
        assert "timestamp" in fix_meta

    def test_load_input_files_missing(self):
        """Test handling of missing input files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            logger = MagicMock()

            with pytest.raises(SystemExit) as exc_info:
                load_input_files(data_dir, logger)

            assert exc_info.value.code == 2  # EXIT_INPUT_ERROR

    def test_load_input_files_invalid_json(self):
        """Test handling of invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            # Create invalid JSON files
            (data_dir / "ConceptDictionary_wow.json").write_text("invalid json{")
            (data_dir / "LearningChunkGraph_wow.json").write_text('{"valid": "json"}')

            logger = MagicMock()

            with pytest.raises(SystemExit) as exc_info:
                load_input_files(data_dir, logger)

            assert exc_info.value.code == 2  # EXIT_INPUT_ERROR

    def test_full_integration(self, sample_graph, sample_concepts):
        """Test full processing flow."""
        logger = MagicMock()

        # Process all components
        chunks_marked, assessments_marked, _ = process_chunk_assessment_definitions(
            sample_graph["nodes"], dry_run=False, logger=logger
        )

        concepts_updated, _ = process_concept_text(
            sample_graph["nodes"], sample_concepts, dry_run=False, logger=logger
        )

        edges_marked, _ = process_edge_conditions(
            sample_graph["edges"], dry_run=False, logger=logger
        )

        stats = {
            "chunks_marked": chunks_marked,
            "assessments_marked": assessments_marked,
            "concepts_updated": concepts_updated,
            "edges_marked": edges_marked,
        }

        update_metadata(sample_graph, stats, logger)

        # Verify results
        assert chunks_marked == 1
        assert assessments_marked == 1
        assert concepts_updated == 1
        assert edges_marked == 2

        # Verify metadata
        assert "graph_fix_applied" in sample_graph["_meta"]
        assert sample_graph["_meta"]["graph_fix_applied"]["chunks_definitions_marked"] == 1

    def test_malformed_data_handling(self):
        """Test handling of malformed data structures."""
        logger = MagicMock()

        # Nodes without required fields
        malformed_nodes = [
            {"type": "Chunk"},  # No id
            {"id": "test", "type": "Unknown"},  # Unknown type
            {"id": "chunk", "type": "Chunk", "definition": None},  # None definition
        ]

        # Should not crash
        chunks_marked, assessments_marked, _ = process_chunk_assessment_definitions(
            malformed_nodes, dry_run=False, logger=logger
        )

        assert chunks_marked == 0
        assert assessments_marked == 0

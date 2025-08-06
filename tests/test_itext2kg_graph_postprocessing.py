#!/usr/bin/env python3
"""
Tests for ID post-processing in itext2kg_graph module.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.itext2kg_graph import SliceData, SliceProcessor


class TestIDPostProcessing:
    """Test suite for _assign_final_ids method."""

    @pytest.fixture
    def mock_config(self):
        """Create minimal config for processor."""
        return {
            "itext2kg": {
                "model": "test-model",
                "tpm_limit": 100000,
                "log_level": "debug",
                "temperature": 0.6,
                "reasoning_effort": "medium",
                "reasoning_summary": "auto",
                "timeout": 360,
                "max_retries": 3,
                "poll_interval": 7,
            }
        }

    @pytest.fixture
    def processor(self, mock_config, tmp_path):
        """Create processor with mocked dependencies."""
        # Create necessary directories
        (tmp_path / "out").mkdir()
        (tmp_path / "logs").mkdir()

        # Create minimal ConceptDictionary
        concept_dict = {"concepts": []}
        concept_path = tmp_path / "out" / "ConceptDictionary.json"
        with open(concept_path, "w", encoding="utf-8") as f:
            json.dump(concept_dict, f)

        # Patch paths
        import src.itext2kg_graph as module

        module.OUTPUT_DIR = tmp_path / "out"
        module.LOGS_DIR = tmp_path / "logs"
        module.PROMPTS_DIR = Path(__file__).parent.parent / "src" / "prompts"
        module.SCHEMAS_DIR = Path(__file__).parent.parent / "src" / "schemas"

        # Mock LLM client
        with patch("src.itext2kg_graph.OpenAIClient"):
            processor = SliceProcessor(mock_config)
            return processor

    def test_assign_final_ids_chunks(self, processor):
        """Test that chunk IDs are correctly calculated."""
        slice_data = SliceData(
            id="slice_001",
            order=1,
            source_file="test.md",
            slug="test_book",
            text="Test content",
            slice_token_start=5000,
            slice_token_end=6000,
        )

        patch = {
            "nodes": [
                {"id": "chunk_1", "type": "Chunk", "node_offset": 100, "text": "First chunk"},
                {"id": "chunk_2", "type": "Chunk", "node_offset": 500, "text": "Second chunk"},
            ],
            "edges": [{"source": "chunk_1", "target": "chunk_2", "type": "PREREQUISITE"}],
        }

        processor._assign_final_ids(patch, slice_data)

        # Check IDs were replaced
        assert patch["nodes"][0]["id"] == "test_book:c:5100"
        assert patch["nodes"][1]["id"] == "test_book:c:5500"

        # Check edges were updated
        assert patch["edges"][0]["source"] == "test_book:c:5100"
        assert patch["edges"][0]["target"] == "test_book:c:5500"

    def test_assign_final_ids_assessments(self, processor):
        """Test that assessment IDs are correctly calculated."""
        slice_data = SliceData(
            id="slice_002",
            order=2,
            source_file="test.md",
            slug="algo101",
            text="Test content",
            slice_token_start=10000,
            slice_token_end=11000,
        )

        patch = {
            "nodes": [
                {
                    "id": "assessment_1",
                    "type": "Assessment",
                    "node_offset": 200,
                    "text": "Question 1",
                },
                {
                    "id": "assessment_2",
                    "type": "Assessment",
                    "node_offset": 600,
                    "text": "Question 2",
                },
                {"id": "chunk_1", "type": "Chunk", "node_offset": 50, "text": "Content"},
            ],
            "edges": [
                {"source": "chunk_1", "target": "assessment_1", "type": "TESTS"},
                {"source": "assessment_1", "target": "assessment_2", "type": "PREREQUISITE"},
            ],
        }

        processor._assign_final_ids(patch, slice_data)

        # Check assessment IDs
        assert patch["nodes"][0]["id"] == "algo101:q:10200:1"
        assert patch["nodes"][1]["id"] == "algo101:q:10600:2"

        # Check chunk ID
        assert patch["nodes"][2]["id"] == "algo101:c:10050"

        # Check edges were updated
        assert patch["edges"][0]["source"] == "algo101:c:10050"
        assert patch["edges"][0]["target"] == "algo101:q:10200:1"
        assert patch["edges"][1]["source"] == "algo101:q:10200:1"
        assert patch["edges"][1]["target"] == "algo101:q:10600:2"

    def test_concept_ids_unchanged(self, processor):
        """Test that concept IDs from dictionary are not modified."""
        slice_data = SliceData(
            id="slice_003",
            order=3,
            source_file="test.md",
            slug="test",
            text="Test content",
            slice_token_start=15000,
            slice_token_end=16000,
        )

        patch = {
            "nodes": [
                {
                    "id": "test:p:stack",
                    "type": "Concept",
                    "node_offset": 100,
                    "definition": "Stack def",
                },
                {"id": "chunk_1", "type": "Chunk", "node_offset": 200, "text": "Content"},
            ],
            "edges": [{"source": "chunk_1", "target": "test:p:stack", "type": "MENTIONS"}],
        }

        processor._assign_final_ids(patch, slice_data)

        # Concept ID should remain unchanged
        assert patch["nodes"][0]["id"] == "test:p:stack"

        # Chunk ID should be updated
        assert patch["nodes"][1]["id"] == "test:c:15200"

        # Edge should use updated chunk ID but original concept ID
        assert patch["edges"][0]["source"] == "test:c:15200"
        assert patch["edges"][0]["target"] == "test:p:stack"

    def test_missing_node_offset_warning(self, processor, caplog):
        """Test warning is logged for nodes without node_offset."""
        slice_data = SliceData(
            id="slice_004",
            order=4,
            source_file="test.md",
            slug="test",
            text="Test content",
            slice_token_start=20000,
            slice_token_end=21000,
        )

        patch = {
            "nodes": [
                {"id": "chunk_1", "type": "Chunk", "text": "No offset"},  # Missing node_offset
                {"id": "chunk_2", "type": "Chunk", "node_offset": 500, "text": "Has offset"},
            ],
            "edges": [],
        }

        processor._assign_final_ids(patch, slice_data)

        # Only chunk_2 should have updated ID
        assert patch["nodes"][0]["id"] == "chunk_1"  # Unchanged due to missing offset
        assert patch["nodes"][1]["id"] == "test:c:20500"

        # Check warning was logged
        assert "missing required node_offset field" in caplog.text

    def test_mixed_node_types(self, processor):
        """Test processing a mix of chunks, assessments, and concepts."""
        slice_data = SliceData(
            id="slice_005",
            order=5,
            source_file="test.md",
            slug="course",
            text="Test content",
            slice_token_start=25000,
            slice_token_end=26000,
        )

        patch = {
            "nodes": [
                {"id": "chunk_1", "type": "Chunk", "node_offset": 0, "text": "Intro"},
                {
                    "id": "course:p:recursion",
                    "type": "Concept",
                    "node_offset": 100,
                    "definition": "Recursion",
                },
                {"id": "chunk_2", "type": "Chunk", "node_offset": 200, "text": "Details"},
                {"id": "assessment_1", "type": "Assessment", "node_offset": 400, "text": "Quiz"},
                {"id": "chunk_3", "type": "Chunk", "node_offset": 600, "text": "Summary"},
            ],
            "edges": [
                {"source": "chunk_1", "target": "course:p:recursion", "type": "MENTIONS"},
                {"source": "chunk_2", "target": "chunk_3", "type": "PREREQUISITE"},
                {"source": "assessment_1", "target": "chunk_3", "type": "TESTS"},
            ],
        }

        processor._assign_final_ids(patch, slice_data)

        # Check all IDs
        assert patch["nodes"][0]["id"] == "course:c:25000"
        assert patch["nodes"][1]["id"] == "course:p:recursion"  # Unchanged
        assert patch["nodes"][2]["id"] == "course:c:25200"
        assert patch["nodes"][3]["id"] == "course:q:25400:1"
        assert patch["nodes"][4]["id"] == "course:c:25600"

        # Check edges
        assert patch["edges"][0]["source"] == "course:c:25000"
        assert patch["edges"][0]["target"] == "course:p:recursion"
        assert patch["edges"][1]["source"] == "course:c:25200"
        assert patch["edges"][1]["target"] == "course:c:25600"
        assert patch["edges"][2]["source"] == "course:q:25400:1"
        assert patch["edges"][2]["target"] == "course:c:25600"

    def test_assessment_index_extraction(self, processor):
        """Test correct extraction of assessment indices."""
        slice_data = SliceData(
            id="slice_006",
            order=6,
            source_file="test.md",
            slug="test",
            text="Test content",
            slice_token_start=30000,
            slice_token_end=31000,
        )

        patch = {
            "nodes": [
                {"id": "assessment_1", "type": "Assessment", "node_offset": 100, "text": "Q1"},
                {"id": "assessment_10", "type": "Assessment", "node_offset": 200, "text": "Q10"},
                {"id": "assessment_99", "type": "Assessment", "node_offset": 300, "text": "Q99"},
            ],
            "edges": [],
        }

        processor._assign_final_ids(patch, slice_data)

        # Check indices are correctly extracted
        assert patch["nodes"][0]["id"] == "test:q:30100:1"
        assert patch["nodes"][1]["id"] == "test:q:30200:10"
        assert patch["nodes"][2]["id"] == "test:q:30300:99"

    def test_empty_patch(self, processor):
        """Test handling of empty patch."""
        slice_data = SliceData(
            id="slice_007",
            order=7,
            source_file="test.md",
            slug="test",
            text="Test content",
            slice_token_start=35000,
            slice_token_end=36000,
        )

        patch = {"nodes": [], "edges": []}

        # Should not raise any errors
        processor._assign_final_ids(patch, slice_data)

        assert patch["nodes"] == []
        assert patch["edges"] == []

    def test_edge_with_nonexistent_nodes(self, processor):
        """Test edges referencing non-existent nodes are handled gracefully."""
        slice_data = SliceData(
            id="slice_008",
            order=8,
            source_file="test.md",
            slug="test",
            text="Test content",
            slice_token_start=40000,
            slice_token_end=41000,
        )

        patch = {
            "nodes": [
                {"id": "chunk_1", "type": "Chunk", "node_offset": 100, "text": "Content"},
            ],
            "edges": [
                {"source": "chunk_1", "target": "nonexistent_node", "type": "PREREQUISITE"},
                {"source": "another_missing", "target": "chunk_1", "type": "ELABORATES"},
            ],
        }

        processor._assign_final_ids(patch, slice_data)

        # Check chunk ID was updated
        assert patch["nodes"][0]["id"] == "test:c:40100"

        # Check edges - only existing nodes in mapping should be updated
        assert patch["edges"][0]["source"] == "test:c:40100"
        assert patch["edges"][0]["target"] == "nonexistent_node"  # Unchanged
        assert patch["edges"][1]["source"] == "another_missing"  # Unchanged
        assert patch["edges"][1]["target"] == "test:c:40100"

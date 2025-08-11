#!/usr/bin/env python3
"""
Tests for itext2kg_graph module.
"""

import json
import os

# Add project root to PYTHONPATH for imports
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.itext2kg_graph import ProcessingStats, SliceData, SliceProcessor
from src.utils.exit_codes import (
    EXIT_INPUT_ERROR,
    EXIT_RUNTIME_ERROR,
    EXIT_SUCCESS,
)


@pytest.fixture
def sample_config(tmp_path):
    """Create sample configuration for testing."""
    # Create necessary directories
    (tmp_path / "staging").mkdir()
    (tmp_path / "out").mkdir()
    (tmp_path / "logs").mkdir()

    config = {
        "itext2kg": {
            "model": "test-model",
            "tpm_limit": 100000,
            "log_level": "debug",
            "temperature": 0.6,
            "reasoning_effort": "medium",
            "reasoning_summary": "auto",
            "timeout": 360,
            "max_retries": 3,
            "poll_interval": 2,
        }
    }

    # Monkey-patch paths
    import src.itext2kg_graph as module

    module.STAGING_DIR = tmp_path / "staging"
    module.OUTPUT_DIR = tmp_path / "out"
    module.LOGS_DIR = tmp_path / "logs"
    module.PROMPTS_DIR = Path(__file__).parent.parent / "src" / "prompts"
    module.SCHEMAS_DIR = Path(__file__).parent.parent / "src" / "schemas"

    return config


@pytest.fixture
def sample_concept_dict():
    """Create sample ConceptDictionary."""
    return {
        "concepts": [
            {
                "concept_id": "test:p:stack",
                "term": {"primary": "Stack", "aliases": ["стек", "LIFO"]},
                "definition": "LIFO data structure",
            },
            {
                "concept_id": "test:p:queue",
                "term": {"primary": "Queue", "aliases": ["очередь", "FIFO"]},
                "definition": "FIFO data structure",
            },
        ]
    }


@pytest.fixture
def sample_slice():
    """Create sample slice data."""
    return {
        "id": "slice_001",
        "order": 1,
        "source_file": "test.md",
        "slug": "test",
        "text": "We use стек for storage. Stack is a LIFO structure.",
        "slice_token_start": 1000,
        "slice_token_end": 1500,
    }


@pytest.fixture
def processor_with_mocks(sample_config, sample_concept_dict, tmp_path):
    """Create processor with mocked dependencies."""
    # Create ConceptDictionary file
    concept_path = tmp_path / "out" / "ConceptDictionary.json"
    with open(concept_path, "w", encoding="utf-8") as f:
        json.dump(sample_concept_dict, f)

    # Create prompt file
    prompt_path = tmp_path / "prompts" / "itext2kg_graph_extraction.md"
    prompt_path.parent.mkdir(exist_ok=True)
    prompt_path.write_text("Test prompt {learning_chunk_graph_schema}")

    # Create schema file
    schema_path = tmp_path / "schemas" / "LearningChunkGraph.schema.json"
    schema_path.parent.mkdir(exist_ok=True)
    schema_path.write_text('{"test": "schema"}')

    # Update module paths
    import src.itext2kg_graph as module

    module.PROMPTS_DIR = tmp_path / "prompts"
    module.SCHEMAS_DIR = tmp_path / "schemas"

    with patch("src.itext2kg_graph.OpenAIClient") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        processor = SliceProcessor(sample_config)
        processor.llm_client = mock_client

        return processor, mock_client


class TestSliceProcessor:
    """Test the main processor class."""

    def test_initialization(self, sample_config, sample_concept_dict, tmp_path):
        """Test processor initialization."""
        # Create ConceptDictionary
        concept_path = tmp_path / "out" / "ConceptDictionary.json"
        with open(concept_path, "w", encoding="utf-8") as f:
            json.dump(sample_concept_dict, f)

        # Create prompt and schema files
        prompt_path = tmp_path / "prompts" / "itext2kg_graph_extraction.md"
        prompt_path.parent.mkdir(exist_ok=True)
        prompt_path.write_text("Test prompt {learning_chunk_graph_schema}")

        schema_path = tmp_path / "schemas" / "LearningChunkGraph.schema.json"
        schema_path.parent.mkdir(exist_ok=True)
        schema_path.write_text('{"test": "schema"}')

        # Update module paths
        import src.itext2kg_graph as module

        module.PROMPTS_DIR = tmp_path / "prompts"
        module.SCHEMAS_DIR = tmp_path / "schemas"

        with patch("src.itext2kg_graph.OpenAIClient"):
            processor = SliceProcessor(sample_config)

            assert processor.concept_dict == sample_concept_dict
            assert processor.graph_nodes == []
            assert processor.graph_edges == []
            assert processor.node_ids == {}
            assert processor.previous_response_id is None

    def test_load_concept_dictionary_not_found(self, sample_config, tmp_path):
        """Test error when ConceptDictionary not found."""
        with patch("src.itext2kg_graph.OpenAIClient"):
            with pytest.raises(SystemExit) as exc_info:
                SliceProcessor(sample_config)

            assert exc_info.value.code == EXIT_INPUT_ERROR

    def test_format_tokens(self, processor_with_mocks):
        """Test token formatting."""
        processor, _ = processor_with_mocks

        assert processor._format_tokens(123) == "123"
        assert processor._format_tokens(1234) == "1.23k"
        assert processor._format_tokens(45678) == "45.68k"
        assert processor._format_tokens(1234567) == "1.23M"

    def test_validate_node_positions(self, processor_with_mocks):
        """Test node position validation logic."""
        processor, _ = processor_with_mocks

        # Test correct calculation
        valid_nodes = [
            {
                "id": "test:c:5238",
                "type": "Chunk",
                "text": "test",
                "node_offset": 245,
                "node_position": 5238,
                "_calculation": "slice_token_start(4993) + node_offset(245) = node_position(5238)",
            },
            {
                "id": "test:q:5420:0",
                "type": "Assessment",
                "text": "question",
                "node_offset": 427,
                "node_position": 5420,
                "_calculation": "slice_token_start(4993) + node_offset(427) = node_position(5420)",
            },
            {"id": "test:p:concept", "type": "Concept", "definition": "test"},  # Ignored
        ]
        result = processor.validate_node_positions(valid_nodes, 4993)
        assert result == valid_nodes  # Now always returns nodes

        # Test math error - after refactor, validation is skipped
        invalid_nodes = [
            {
                "id": "test:c:5238",
                "type": "Chunk",
                "text": "test",
                "node_offset": 245,
                "node_position": 5000,  # Wrong, but validation is now skipped
                "_calculation": "incorrect",
            }
        ]
        result = processor.validate_node_positions(invalid_nodes, 4993)
        assert result == invalid_nodes  # Now always returns nodes (validation skipped)

        # Test position before slice start
        invalid_nodes2 = [
            {
                "id": "test:c:100",
                "type": "Chunk",
                "text": "test",
                "node_offset": 100,
                "node_position": 100,  # Less than slice_token_start!
                "_calculation": "wrong",
            }
        ]
        result = processor.validate_node_positions(invalid_nodes2, 4993)
        assert result == invalid_nodes2  # Now always returns nodes (validation skipped)

        # Test ID mismatch - after refactor, validation is skipped
        invalid_nodes3 = [
            {
                "id": "test:c:5000",  # ID says 5000
                "type": "Chunk",
                "text": "test",
                "node_offset": 245,
                "node_position": 5238,  # But position is 5238, validation is now skipped
                "_calculation": "slice_token_start(4993) + node_offset(245) = node_position(5238)",
            }
        ]
        result = processor.validate_node_positions(invalid_nodes3, 4993)
        assert result == invalid_nodes3  # Now always returns nodes (validation skipped)

        # Test missing fields - after refactor, validation is skipped
        invalid_nodes4 = [
            {
                "id": "test:c:5238",
                "type": "Chunk",
                "text": "test",
                # Missing node_offset and node_position, but validation is now skipped
            }
        ]
        result = processor.validate_node_positions(invalid_nodes4, 4993)
        assert result == invalid_nodes4  # Now always returns nodes (validation skipped)

    def test_process_llm_response_valid_json(self, processor_with_mocks):
        """Test processing valid JSON response."""
        processor, _ = processor_with_mocks

        response = json.dumps(
            {
                "chunk_graph_patch": {
                    "nodes": [{"id": "test:c:1000", "type": "Chunk", "text": "Test"}],
                    "edges": [
                        {"source": "test:c:1000", "target": "test:p:stack", "type": "MENTIONS"}
                    ],
                }
            }
        )

        success, parsed = processor._process_llm_response(response, "slice_001")
        assert success is True
        assert parsed is not None
        assert "chunk_graph_patch" in parsed

    def test_process_llm_response_with_markdown(self, processor_with_mocks):
        """Test processing response with markdown fences."""
        processor, _ = processor_with_mocks

        response = """```json
{
    "chunk_graph_patch": {
        "nodes": [],
        "edges": []
    }
}
```"""

        success, parsed = processor._process_llm_response(response, "slice_001")
        assert success is True
        assert parsed is not None

    def test_process_llm_response_invalid_json(self, processor_with_mocks):
        """Test processing invalid JSON."""
        processor, _ = processor_with_mocks

        response = "{invalid json"

        success, parsed = processor._process_llm_response(response, "slice_001")
        assert success is False
        assert parsed is None

    def test_process_llm_response_missing_patch(self, processor_with_mocks):
        """Test processing response without chunk_graph_patch."""
        processor, _ = processor_with_mocks

        response = json.dumps({"other_field": "value"})

        success, parsed = processor._process_llm_response(response, "slice_001")
        assert success is False
        assert parsed is None

    def test_process_chunk_nodes_new_chunk(self, processor_with_mocks):
        """Test processing new Chunk node."""
        processor, _ = processor_with_mocks

        new_nodes = [{"id": "test:c:1000", "type": "Chunk", "text": "Test chunk", "difficulty": 2}]

        nodes_to_add = processor._process_chunk_nodes(new_nodes)
        assert len(nodes_to_add) == 1
        assert "test:c:1000" in processor.node_ids

    def test_process_chunk_nodes_duplicate_chunk(self, processor_with_mocks):
        """Test handling duplicate Chunk with longer text."""
        processor, _ = processor_with_mocks

        # Add initial chunk
        processor.graph_nodes = [{"id": "test:c:1000", "type": "Chunk", "text": "Short"}]
        processor.node_ids = {"test:c:1000": 0}

        # Try to add duplicate with longer text
        new_nodes = [{"id": "test:c:1000", "type": "Chunk", "text": "Much longer text here"}]

        nodes_to_add = processor._process_chunk_nodes(new_nodes)
        assert len(nodes_to_add) == 0  # Not added as new
        assert processor.graph_nodes[0]["text"] == "Much longer text here"  # Updated

    def test_process_chunk_nodes_missing_difficulty(self, processor_with_mocks):
        """Test adding default difficulty when missing."""
        processor, _ = processor_with_mocks

        new_nodes = [{"id": "test:c:1000", "type": "Chunk", "text": "Test"}]  # No difficulty

        nodes_to_add = processor._process_chunk_nodes(new_nodes)
        assert len(nodes_to_add) == 1
        assert nodes_to_add[0]["difficulty"] == 3  # Default value

    def test_process_chunk_nodes_duplicate_assessment(self, processor_with_mocks):
        """Test ignoring duplicate Assessment."""
        processor, _ = processor_with_mocks

        # Add initial assessment
        processor.graph_nodes = [{"id": "test:q:1000:0", "type": "Assessment", "text": "Question"}]
        processor.node_ids = {"test:q:1000:0": 0}

        # Try to add duplicate
        new_nodes = [{"id": "test:q:1000:0", "type": "Assessment", "text": "Question"}]

        nodes_to_add = processor._process_chunk_nodes(new_nodes)
        assert len(nodes_to_add) == 0  # Ignored

    def test_process_chunk_nodes_concept(self, processor_with_mocks):
        """Test processing Concept node."""
        processor, _ = processor_with_mocks

        new_nodes = [{"id": "test:p:stack", "type": "Concept", "definition": "Wrong def"}]

        nodes_to_add = processor._process_chunk_nodes(new_nodes)
        assert len(nodes_to_add) == 1
        # Should use definition from ConceptDictionary
        assert nodes_to_add[0]["definition"] == "LIFO data structure"

    def test_validate_edges_valid(self, processor_with_mocks):
        """Test validating valid edges."""
        processor, _ = processor_with_mocks

        # Add nodes to graph
        processor.node_ids = {"test:c:1000": 0}

        edges = [
            {"source": "test:c:1000", "target": "test:p:stack", "type": "MENTIONS", "weight": 1.0}
        ]

        valid = processor._validate_edges(edges)
        assert len(valid) == 1

    def test_validate_edges_self_loop_prerequisite(self, processor_with_mocks):
        """Test dropping PREREQUISITE self-loops."""
        processor, _ = processor_with_mocks

        processor.node_ids = {"test:c:1000": 0}

        edges = [{"source": "test:c:1000", "target": "test:c:1000", "type": "PREREQUISITE"}]

        valid = processor._validate_edges(edges)
        assert len(valid) == 0  # Dropped

    def test_validate_edges_invalid_weight(self, processor_with_mocks):
        """Test fixing invalid edge weight."""
        processor, _ = processor_with_mocks

        processor.node_ids = {"test:c:1000": 0, "test:c:2000": 1}

        edges = [
            {"source": "test:c:1000", "target": "test:c:2000", "type": "ELABORATES", "weight": 1.5}
        ]

        valid = processor._validate_edges(edges)
        assert len(valid) == 1
        assert valid[0]["weight"] == 0.5  # Fixed

    def test_validate_edges_duplicate(self, processor_with_mocks):
        """Test filtering duplicate edges."""
        processor, _ = processor_with_mocks

        processor.node_ids = {"test:c:1000": 0}
        processor.graph_edges = [
            {"source": "test:c:1000", "target": "test:p:stack", "type": "MENTIONS"}
        ]

        edges = [
            {"source": "test:c:1000", "target": "test:p:stack", "type": "MENTIONS"}  # Duplicate
        ]

        valid = processor._validate_edges(edges)
        assert len(valid) == 0  # Filtered

    def test_add_mentions_edges(self, processor_with_mocks):
        """Test automatic MENTIONS edge creation."""
        processor, _ = processor_with_mocks

        chunks = [
            {"id": "test:c:1000", "type": "Chunk", "text": "We use стек for storage"},
            {"id": "test:c:2000", "type": "Chunk", "text": "Stack is a LIFO structure"},
        ]

        added = processor._add_mentions_edges(chunks)
        assert added == 2
        assert len(processor.graph_edges) == 2

        # Check edges
        edge_pairs = [(e["source"], e["target"]) for e in processor.graph_edges]
        assert ("test:c:1000", "test:p:stack") in edge_pairs
        assert ("test:c:2000", "test:p:stack") in edge_pairs

    def test_add_mentions_edges_no_duplicates(self, processor_with_mocks):
        """Test that existing MENTIONS edges are not duplicated."""
        processor, _ = processor_with_mocks

        # Add existing MENTIONS edge
        processor.graph_edges = [
            {"source": "test:c:1000", "target": "test:p:stack", "type": "MENTIONS", "weight": 1.0}
        ]

        chunks = [{"id": "test:c:1000", "type": "Chunk", "text": "Stack is used here"}]

        added = processor._add_mentions_edges(chunks)
        assert added == 0  # No new edges
        assert len(processor.graph_edges) == 1  # Still just one edge

    def test_validate_graph_intermediate_valid(self, processor_with_mocks):
        """Test intermediate validation with valid graph."""
        processor, _ = processor_with_mocks

        processor.graph_nodes = [
            {"id": "test:c:1000", "type": "Chunk"},
            {"id": "test:c:2000", "type": "Chunk"},
            {"id": "test:q:3000:0", "type": "Assessment"},
            {"id": "test:p:stack", "type": "Concept"},  # Duplicate Concepts allowed
            {"id": "test:p:stack", "type": "Concept"},
        ]

        assert processor._validate_graph_intermediate() is True

    def test_validate_graph_intermediate_duplicate_chunk(self, processor_with_mocks):
        """Test intermediate validation with duplicate Chunk ID."""
        processor, _ = processor_with_mocks

        processor.graph_nodes = [
            {"id": "test:c:1000", "type": "Chunk"},
            {"id": "test:c:1000", "type": "Chunk"},  # Duplicate!
        ]

        assert processor._validate_graph_intermediate() is False

    def test_validate_graph_intermediate_duplicate_assessment(self, processor_with_mocks):
        """Test intermediate validation with duplicate Assessment ID."""
        processor, _ = processor_with_mocks

        processor.graph_nodes = [
            {"id": "test:q:1000:0", "type": "Assessment"},
            {"id": "test:q:1000:0", "type": "Assessment"},  # Duplicate!
        ]

        assert processor._validate_graph_intermediate() is False

    def test_process_single_slice_success(self, processor_with_mocks, sample_slice, tmp_path):
        """Test successful slice processing."""
        processor, mock_client = processor_with_mocks

        # Create slice file
        slice_file = tmp_path / "staging" / "slice_001.slice.json"
        with open(slice_file, "w", encoding="utf-8") as f:
            json.dump(sample_slice, f)

        # Mock LLM response
        mock_client.create_response.return_value = (
            json.dumps(
                {
                    "chunk_graph_patch": {
                        "nodes": [
                            {
                                "id": "test:c:1100",
                                "type": "Chunk",
                                "text": "Test",
                                "node_offset": 100,
                                "node_position": 1100,
                                "_calculation": "slice_token_start(1000) + node_offset(100) = node_position(1100)",
                                "difficulty": 2,
                            }
                        ],
                        "edges": [],
                    }
                }
            ),
            "resp_1",
            Mock(total_tokens=100, input_tokens=50, output_tokens=40, reasoning_tokens=10),
        )

        success = processor._process_single_slice(slice_file)
        assert success is True
        assert processor.previous_response_id == "resp_1"
        assert len(processor.graph_nodes) == 1

    def test_process_single_slice_json_repair(self, processor_with_mocks, sample_slice, tmp_path):
        """Test repair for JSON parsing errors."""
        processor, mock_client = processor_with_mocks

        # Create slice file
        slice_file = tmp_path / "staging" / "slice_001.slice.json"
        with open(slice_file, "w", encoding="utf-8") as f:
            json.dump(sample_slice, f)

        # First response - invalid JSON
        mock_client.create_response.return_value = (
            "```json\n{invalid json",
            "resp_1",
            Mock(total_tokens=100, input_tokens=50, output_tokens=40, reasoning_tokens=10),
        )

        # Repair response - valid JSON
        mock_client.repair_response.return_value = (
            json.dumps(
                {
                    "chunk_graph_patch": {
                        "nodes": [
                            {
                                "id": "test:c:1100",
                                "type": "Chunk",
                                "text": "Test",
                                "node_offset": 100,
                                "node_position": 1100,
                                "_calculation": "slice_token_start(1000) + node_offset(100) = node_position(1100)",
                            }
                        ],
                        "edges": [],
                    }
                }
            ),
            "resp_2",
            Mock(total_tokens=50, input_tokens=25, output_tokens=20, reasoning_tokens=5),
        )

        success = processor._process_single_slice(slice_file)
        assert success is True
        assert mock_client.repair_response.called
        assert processor.previous_response_id == "resp_2"  # Uses repair_id

    def test_process_single_slice_id_repair(self, processor_with_mocks, sample_slice, tmp_path):
        """Test that IDs are automatically fixed via post-processing (no repair needed)."""
        processor, mock_client = processor_with_mocks

        # Create slice file
        slice_file = tmp_path / "staging" / "slice_001.slice.json"
        with open(slice_file, "w", encoding="utf-8") as f:
            json.dump(sample_slice, f)

        # Response with temporary IDs (as per new convention)
        mock_client.create_response.return_value = (
            json.dumps(
                {
                    "chunk_graph_patch": {
                        "nodes": [
                            {
                                "id": "chunk_1",  # Temporary ID
                                "type": "Chunk",
                                "text": "Test",
                                "node_offset": 100,
                                "difficulty": 3,
                            }
                        ],
                        "edges": [],
                    }
                }
            ),
            "resp_1",
            Mock(total_tokens=100, input_tokens=50, output_tokens=40, reasoning_tokens=10),
        )

        success = processor._process_single_slice(slice_file)
        assert success is True
        # Repair should NOT be called - IDs are fixed automatically
        assert not mock_client.repair_response.called

        # Check that the ID was properly assigned
        assert len(processor.graph_nodes) == 1
        # The ID should be fixed to test:c:1100 (slice_token_start=1000 + node_offset=100)
        assert processor.graph_nodes[0]["id"] == "test:c:1100"
        # No repair was called, so should use the original response_id
        assert processor.previous_response_id == "resp_1"

    def test_process_single_slice_repair_failure(
        self, processor_with_mocks, sample_slice, tmp_path
    ):
        """Test handling when repair fails."""
        processor, mock_client = processor_with_mocks

        # Create slice file
        slice_file = tmp_path / "staging" / "slice_001.slice.json"
        with open(slice_file, "w", encoding="utf-8") as f:
            json.dump(sample_slice, f)

        # First response - invalid JSON
        mock_client.create_response.return_value = (
            "{invalid",
            "resp_1",
            Mock(total_tokens=100, input_tokens=50, output_tokens=40, reasoning_tokens=10),
        )

        # Repair also fails
        mock_client.repair_response.return_value = (
            "{still invalid",
            "resp_2",
            Mock(total_tokens=50, input_tokens=25, output_tokens=20, reasoning_tokens=5),
        )

        success = processor._process_single_slice(slice_file)
        assert success is False
        assert processor.previous_response_id is None  # Not updated on failure

    def test_run_no_slices(self, processor_with_mocks, tmp_path):
        """Test run with no slice files."""
        processor, _ = processor_with_mocks

        exit_code = processor.run()
        assert exit_code == EXIT_INPUT_ERROR

    def test_run_success(self, processor_with_mocks, sample_slice, tmp_path):
        """Test successful run."""
        processor, mock_client = processor_with_mocks

        # Create slice file
        slice_file = tmp_path / "staging" / "slice_001.slice.json"
        with open(slice_file, "w", encoding="utf-8") as f:
            json.dump(sample_slice, f)

        # Mock LLM response
        mock_client.create_response.return_value = (
            json.dumps(
                {
                    "chunk_graph_patch": {
                        "nodes": [
                            {
                                "id": "test:c:1100",
                                "type": "Chunk",
                                "text": "Test",
                                "node_offset": 100,
                                "node_position": 1100,
                                "_calculation": "slice_token_start(1000) + node_offset(100) = node_position(1100)",
                            }
                        ],
                        "edges": [
                            {"source": "test:c:1100", "target": "test:p:stack", "type": "MENTIONS"}
                        ],
                    }
                }
            ),
            "resp_1",
            Mock(total_tokens=100, input_tokens=50, output_tokens=40, reasoning_tokens=10),
        )

        exit_code = processor.run()
        assert exit_code == EXIT_SUCCESS

        # Check output file
        output_file = tmp_path / "out" / "LearningChunkGraph_raw.json"
        assert output_file.exists()

        with open(output_file, encoding="utf-8") as f:
            graph = json.load(f)

        # Check metadata exists
        assert "_meta" in graph
        assert graph["_meta"]["generator"] == "itext2kg_graph"
        assert "api_usage" in graph["_meta"]
        assert "graph_stats" in graph["_meta"]
        assert "processing_time" in graph["_meta"]
        
        # Check data structure
        assert "nodes" in graph
        assert "edges" in graph
        assert len(graph["nodes"]) > 0
        assert len(graph["edges"]) > 0

    def test_run_slice_failure(self, processor_with_mocks, sample_slice, tmp_path):
        """Test run with slice processing failure."""
        processor, mock_client = processor_with_mocks

        # Create slice file
        slice_file = tmp_path / "staging" / "slice_001.slice.json"
        with open(slice_file, "w", encoding="utf-8") as f:
            json.dump(sample_slice, f)

        # Mock LLM to fail
        mock_client.create_response.side_effect = Exception("API error")

        exit_code = processor.run()
        assert exit_code == EXIT_RUNTIME_ERROR

        # Check temp dumps were created
        log_files = list((tmp_path / "logs").glob("LearningChunkGraph_temp_*.json"))
        assert len(log_files) > 0


class TestProcessingStats:
    """Test ProcessingStats dataclass."""

    def test_initialization(self):
        """Test stats initialization."""
        stats = ProcessingStats()
        assert stats.total_slices == 0
        assert stats.processed_slices == 0
        assert stats.failed_slices == 0
        assert stats.total_nodes == 0
        assert stats.total_edges == 0
        assert stats.total_tokens_used == 0
        assert stats.start_time is not None


class TestSliceData:
    """Test SliceData dataclass."""

    def test_initialization(self):
        """Test slice data initialization."""
        slice_data = SliceData(
            id="slice_001",
            order=1,
            source_file="test.md",
            slug="test",
            text="Test text",
            slice_token_start=1000,
            slice_token_end=1500,
        )

        assert slice_data.id == "slice_001"
        assert slice_data.order == 1
        assert slice_data.source_file == "test.md"
        assert slice_data.slug == "test"
        assert slice_data.text == "Test text"
        assert slice_data.slice_token_start == 1000
        assert slice_data.slice_token_end == 1500


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No API key")
class TestIntegration:
    """Integration tests with real API (requires OPENAI_API_KEY)."""

    def test_full_pipeline(self, tmp_path):
        """Test full processing with real LLM."""
        # This would be a real integration test
        # For now, just check that we can import the module
        from src.itext2kg_graph import main

        assert main is not None

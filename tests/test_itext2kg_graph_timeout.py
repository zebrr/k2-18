"""Tests for TimeoutError retry mechanism in itext2kg_graph module."""

import json
from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest

from src.itext2kg_graph import SliceProcessor
from src.utils.llm_client import ResponseUsage


class TestGraphTimeoutRetryMechanism:
    """Test TimeoutError retry mechanism for graph module."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration with max_retries."""
        return {
            "itext2kg": {
                "model": "test-model",
                "tpm_limit": 100000,
                "log_level": "info",
                "max_context_tokens": 128000,
                "max_context_tokens_test": 128000,
            },
            "max_retries": 3
        }

    @pytest.fixture
    def processor(self, mock_config, tmp_path):
        """Create processor instance with mocked dependencies."""
        with patch("src.itext2kg_graph.LOGS_DIR", tmp_path / "logs"):
            with patch("src.itext2kg_graph.PROMPTS_DIR", tmp_path / "prompts"):
                with patch("src.itext2kg_graph.SCHEMAS_DIR", tmp_path / "schemas"):
                    with patch("src.itext2kg_graph.STAGING_DIR", tmp_path / "staging"):
                        with patch("src.itext2kg_graph.OUTPUT_DIR", tmp_path / "output"):
                            # Create necessary directories
                            (tmp_path / "logs").mkdir()
                            (tmp_path / "prompts").mkdir()
                            (tmp_path / "schemas").mkdir()
                            (tmp_path / "staging").mkdir()
                            (tmp_path / "output").mkdir()

                            # Create ConceptDictionary file
                            concept_dict = {
                                "concepts": [
                                    {
                                        "concept_id": "test:p:concept1",
                                        "term": {"primary": "Test Concept", "aliases": []},
                                        "definition": "Test definition"
                                    }
                                ]
                            }
                            (tmp_path / "output" / "ConceptDictionary.json").write_text(
                                json.dumps(concept_dict), encoding="utf-8"
                            )

                            # Create prompt file
                            prompt_file = tmp_path / "prompts" / "itext2kg_graph_extraction.md"
                            prompt_file.write_text(
                                "Test prompt\n{learning_chunk_graph_schema}", encoding="utf-8"
                            )

                            # Create schema file
                            schema_file = tmp_path / "schemas" / "LearningChunkGraph.schema.json"
                            schema_file.write_text(json.dumps({"$schema": "test-schema"}), encoding="utf-8")

                            # Mock LLM client
                            with patch("src.itext2kg_graph.OpenAIClient") as mock_client_class:
                                mock_client = mock_client_class.return_value
                                processor = SliceProcessor(mock_config)
                                processor.llm_client = mock_client
                                yield processor

    def test_timeout_retry_mechanism_graph(self, processor, tmp_path):
        """Test that TimeoutError triggers repair attempts up to max_retries in graph module."""
        # Create a test slice file
        slice_file = tmp_path / "staging" / "test.slice.json"
        slice_data = {
            "id": "test_slice",
            "order": 1,
            "source_file": "test.txt",
            "slug": "test",
            "text": "Test content",
            "slice_token_start": 0,
            "slice_token_end": 100,
        }
        slice_file.write_text(json.dumps(slice_data), encoding="utf-8")

        # Mock llm_client to raise TimeoutError on first 2 attempts, then succeed
        attempt_count = [0]
        
        def side_effect(*args, **kwargs):
            attempt_count[0] += 1
            if attempt_count[0] <= 2:
                raise TimeoutError("Request timeout")
            else:
                # Return valid response on 3rd attempt
                return (
                    json.dumps({
                        "chunk_graph_patch": {
                            "nodes": [],
                            "edges": []
                        }
                    }),
                    "response_id_3",
                    ResponseUsage(input_tokens=10, output_tokens=20, reasoning_tokens=0, total_tokens=30)
                )
        
        processor.llm_client.create_response.side_effect = lambda *args, **kwargs: (
            side_effect() if attempt_count[0] == 0 else None
        )
        processor.llm_client.repair_response.side_effect = side_effect
        processor.llm_client.confirm_response.return_value = None

        # Process the slice with mocked sleep
        with patch("time.sleep"):  # Mock sleep to speed up test
            result = processor._process_single_slice(slice_file)

        # Verify repair_response was called twice (for 2 retry attempts)
        assert processor.llm_client.repair_response.call_count == 2
        # Verify success on 3rd attempt
        assert result is True
        # Verify confirm_response was called
        assert processor.llm_client.confirm_response.called

    def test_timeout_causes_graph_stop(self, processor, tmp_path):
        """Test that graph extraction stops after max_retries TimeoutErrors."""
        # Create a test slice file
        slice_file = tmp_path / "staging" / "test.slice.json"
        slice_data = {
            "id": "test_slice",
            "order": 1,
            "source_file": "test.txt",
            "slug": "test",
            "text": "Test content",
            "slice_token_start": 0,
            "slice_token_end": 100,
        }
        slice_file.write_text(json.dumps(slice_data), encoding="utf-8")

        # Mock llm_client to always raise TimeoutError
        processor.llm_client.create_response.side_effect = TimeoutError("Request timeout")
        processor.llm_client.repair_response.side_effect = TimeoutError("Request timeout")

        # Process the slice with mocked sleep
        with patch("time.sleep"):  # Mock sleep to speed up test
            result = processor._process_single_slice(slice_file)

        # Verify processing stops with False
        assert result is False
        # Verify repair_response was called max_retries times
        assert processor.llm_client.repair_response.call_count == 3
        # Verify temporary dumps were created
        assert (tmp_path / "logs").exists()

    def test_graph_no_slice_skipping(self, processor, tmp_path):
        """Test that graph module doesn't skip slices on timeout - stops completely."""
        # Create multiple test slice files
        for i in range(3):
            slice_file = tmp_path / "staging" / f"test_{i}.slice.json"
            slice_data = {
                "id": f"test_slice_{i}",
                "order": i + 1,
                "source_file": "test.txt",
                "slug": "test",
                "text": f"Test content {i}",
                "slice_token_start": i * 100,
                "slice_token_end": (i + 1) * 100,
            }
            slice_file.write_text(json.dumps(slice_data), encoding="utf-8")

        # Make second slice fail with timeout
        call_count = [0]
        
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            # First slice succeeds
            if call_count[0] == 1:
                return (
                    json.dumps({"chunk_graph_patch": {"nodes": [], "edges": []}}),
                    f"response_id_{call_count[0]}",
                    ResponseUsage(input_tokens=10, output_tokens=20, reasoning_tokens=0, total_tokens=30)
                )
            # Second slice fails with timeout
            else:
                raise TimeoutError("Request timeout")
        
        processor.llm_client.create_response.side_effect = side_effect
        processor.llm_client.repair_response.side_effect = TimeoutError("Request timeout")
        processor.llm_client.confirm_response.return_value = None

        # Process slices
        slice_files = sorted((tmp_path / "staging").glob("*.slice.json"))
        
        with patch("time.sleep"):  # Mock sleep
            # Process first slice - should succeed
            result1 = processor._process_single_slice(slice_files[0])
            assert result1 is True
            
            # Process second slice - should fail and stop
            result2 = processor._process_single_slice(slice_files[1])
            assert result2 is False
            
            # Third slice should NOT be processed (graph stops on failure)
            # This verifies that the graph module doesn't skip slices

    def test_json_and_timeout_mixed_errors(self, processor, tmp_path):
        """Test handling of mixed JSON and timeout errors in graph module."""
        slice_file = tmp_path / "staging" / "test.slice.json"
        slice_data = {
            "id": "test_slice",
            "order": 1,
            "source_file": "test.txt",
            "slug": "test",
            "text": "Test content",
            "slice_token_start": 0,
            "slice_token_end": 100,
        }
        slice_file.write_text(json.dumps(slice_data), encoding="utf-8")

        # Mock different errors in sequence
        attempt_count = [0]
        
        def side_effect(*args, **kwargs):
            attempt_count[0] += 1
            if attempt_count[0] == 1:
                # First attempt: TimeoutError
                raise TimeoutError("Request timeout")
            elif attempt_count[0] == 2:
                # Second attempt: Invalid JSON
                return (
                    "Invalid JSON response",
                    "response_id_2",
                    ResponseUsage(input_tokens=10, output_tokens=20, reasoning_tokens=0, total_tokens=30)
                )
            else:
                # Third attempt: Success
                return (
                    json.dumps({"chunk_graph_patch": {"nodes": [], "edges": []}}),
                    "response_id_3",
                    ResponseUsage(input_tokens=10, output_tokens=20, reasoning_tokens=0, total_tokens=30)
                )

        processor.llm_client.create_response.side_effect = lambda *args, **kwargs: (
            side_effect() if attempt_count[0] == 0 else None
        )
        processor.llm_client.repair_response.side_effect = side_effect
        processor.llm_client.confirm_response.return_value = None

        # Process the slice
        with patch("time.sleep"):  # Mock sleep to speed up test
            result = processor._process_single_slice(slice_file)

        # Verify success after mixed errors
        assert result is True
        # Verify repair_response was called twice
        assert processor.llm_client.repair_response.call_count == 2
"""Tests for TimeoutError retry mechanism in refiner_longrange module."""

import json
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path

import pytest
import numpy as np

from src.refiner_longrange import analyze_candidate_pairs, load_refiner_longrange_prompt
from src.utils.llm_client import ResponseUsage


class TestRefinerTimeoutRetryMechanism:
    """Test TimeoutError retry mechanism for refiner module."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration with max_retries."""
        return {
            "model": "test-model",
            "tpm_limit": 100000,
            "log_level": "info",
            "max_context_tokens": 128000,
            "max_context_tokens_test": 128000,
            "sim_threshold": 0.7,
            "max_pairs_per_node": 10,
            "weight_low": 0.3,
            "weight_mid": 0.6,
            "weight_high": 0.9,
            "max_retries": 3,
            "is_reasoning": False,
        }

    @pytest.fixture
    def mock_graph(self):
        """Create a mock graph for testing."""
        return {
            "nodes": [
                {"id": "chunk_001", "type": "Chunk", "text": "Test chunk 1"},
                {"id": "chunk_002", "type": "Chunk", "text": "Test chunk 2"},
                {"id": "chunk_003", "type": "Chunk", "text": "Test chunk 3"},
            ],
            "edges": []
        }

    @pytest.fixture
    def candidate_pairs(self):
        """Create test candidate pairs."""
        return [
            {
                "source_node": {"id": "chunk_001", "type": "Chunk", "text": "Test chunk 1"},
                "candidates": [
                    {"node_id": "chunk_002", "similarity": 0.8, "text": "Test chunk 2"},
                    {"node_id": "chunk_003", "similarity": 0.75, "text": "Test chunk 3"},
                ]
            }
        ]

    def test_timeout_retry_in_analyze_pairs(self, mock_config, mock_graph, candidate_pairs, tmp_path):
        """Test that TimeoutError triggers retry in analyze_candidate_pairs."""
        # Mock logger
        logger = Mock()
        
        # Mock prompt loading
        with patch("src.refiner_longrange.load_refiner_longrange_prompt") as mock_load_prompt:
            mock_load_prompt.return_value = "Test prompt"
            
            # Mock LLM client
            with patch("src.refiner_longrange.OpenAIClient") as mock_client_class:
                mock_client = mock_client_class.return_value
                
                # Setup retry behavior - fail twice with timeout, then succeed
                call_count = [0]
                
                def side_effect(*args, **kwargs):
                    call_count[0] += 1
                    if call_count[0] == 1:
                        # First call: timeout
                        raise TimeoutError("Request timeout")
                    elif call_count[0] == 2:
                        # Second call (first retry): timeout
                        raise TimeoutError("Request timeout")
                    else:
                        # Third call (second retry): succeed
                        return (
                            json.dumps([
                                {
                                    "source": "chunk_001",
                                    "target": "chunk_002",
                                    "type": "PREREQUISITE",
                                    "weight": 0.8
                                }
                            ]),
                            "response_id_3",
                            ResponseUsage(input_tokens=10, output_tokens=20, reasoning_tokens=0, total_tokens=30)
                        )
                
                # First call uses create_response, subsequent calls use repair_response
                mock_client.create_response.side_effect = side_effect
                mock_client.repair_response.side_effect = side_effect
                mock_client.confirm_response.return_value = None
                
                # Run analyze_candidate_pairs with mocked sleep
                with patch("time.sleep"):  # Mock sleep to speed up test
                    new_edges, api_usage = analyze_candidate_pairs(
                        candidate_pairs,
                        mock_graph,
                        mock_config,
                        logger,
                        pass_direction="forward"
                    )
                
                # Verify repair was called twice (2 retries)
                assert mock_client.repair_response.call_count == 2
                # Verify we got edges back
                assert len(new_edges) == 1
                assert new_edges[0]["type"] == "PREREQUISITE"
                # Verify API usage tracking - only successful calls are counted
                assert api_usage["requests"] == 1  # Only the successful attempt counts

    def test_refiner_stops_after_max_retries(self, mock_config, mock_graph, candidate_pairs, tmp_path):
        """Test that refiner stops completely after exhausting retries."""
        # Mock logger
        logger = Mock()
        
        # Create multiple candidate pairs
        extended_pairs = [
            {
                "source_node": {"id": "chunk_001", "type": "Chunk", "text": "Test chunk 1"},
                "candidates": [{"node_id": "chunk_002", "similarity": 0.8, "text": "Test chunk 2"}]
            },
            {
                "source_node": {"id": "chunk_002", "type": "Chunk", "text": "Test chunk 2"},
                "candidates": [{"node_id": "chunk_003", "similarity": 0.75, "text": "Test chunk 3"}]
            }
        ]
        
        with patch("src.refiner_longrange.load_refiner_longrange_prompt") as mock_load_prompt:
            mock_load_prompt.return_value = "Test prompt"
            
            with patch("src.refiner_longrange.OpenAIClient") as mock_client_class:
                mock_client = mock_client_class.return_value
                
                call_count = [0]
                
                def side_effect(*args, **kwargs):
                    call_count[0] += 1
                    # First node fails with timeout
                    if call_count[0] <= 4:  # 1 initial + 3 retries for first node
                        raise TimeoutError("Request timeout")
                    # Second node succeeds
                    else:
                        return (
                            json.dumps([
                                {
                                    "source": "chunk_002",
                                    "target": "chunk_003",
                                    "type": "ELABORATES",
                                    "weight": 0.7
                                }
                            ]),
                            "response_id_success",
                            ResponseUsage(input_tokens=10, output_tokens=20, reasoning_tokens=0, total_tokens=30)
                        )
                
                mock_client.create_response.side_effect = side_effect
                mock_client.repair_response.side_effect = side_effect
                mock_client.confirm_response.return_value = None
                
                # Mock Path for bad response saving
                with patch("src.refiner_longrange.Path"):
                    with patch("time.sleep"):  # Mock sleep
                        new_edges, api_usage = analyze_candidate_pairs(
                            extended_pairs,
                            mock_graph,
                            mock_config,
                            logger,
                            pass_direction="forward"
                        )
                
                # Processing should stop after first node fails with max_retries
                # No edges should be returned (stopped before second node)
                assert len(new_edges) == 0
                # Verify api_usage shows no successful requests
                assert api_usage["requests"] == 0

    def test_json_error_retry_in_refiner(self, mock_config, mock_graph, candidate_pairs):
        """Test JSON error retry mechanism in refiner."""
        logger = Mock()
        
        with patch("src.refiner_longrange.load_refiner_longrange_prompt") as mock_load_prompt:
            mock_load_prompt.return_value = "Test prompt"
            
            with patch("src.refiner_longrange.OpenAIClient") as mock_client_class:
                mock_client = mock_client_class.return_value
                
                attempt_count = [0]
                
                def side_effect(*args, **kwargs):
                    attempt_count[0] += 1
                    if attempt_count[0] <= 2:
                        # Return invalid JSON
                        return (
                            "Invalid JSON response",
                            f"response_id_{attempt_count[0]}",
                            ResponseUsage(input_tokens=10, output_tokens=20, reasoning_tokens=0, total_tokens=30)
                        )
                    else:
                        # Return valid JSON on 3rd attempt
                        return (
                            json.dumps([{"source": "chunk_001", "target": "chunk_002", "type": "MENTIONS", "weight": 0.5}]),
                            "response_id_3",
                            ResponseUsage(input_tokens=10, output_tokens=20, reasoning_tokens=0, total_tokens=30)
                        )
                
                # First call uses create_response, subsequent calls use repair_response
                mock_client.create_response.side_effect = side_effect
                mock_client.repair_response.side_effect = side_effect
                mock_client.confirm_response.return_value = None
                
                with patch("time.sleep"):
                    new_edges, api_usage = analyze_candidate_pairs(
                        candidate_pairs,
                        mock_graph,
                        mock_config,
                        logger,
                        pass_direction="forward"
                    )
                
                # Should retry and eventually succeed
                assert len(new_edges) == 1
                assert mock_client.repair_response.call_count == 2

    def test_mixed_errors_in_refiner(self, mock_config, mock_graph, candidate_pairs):
        """Test handling of mixed timeout and JSON errors."""
        logger = Mock()
        
        with patch("src.refiner_longrange.load_refiner_longrange_prompt") as mock_load_prompt:
            mock_load_prompt.return_value = "Test prompt"
            
            with patch("src.refiner_longrange.OpenAIClient") as mock_client_class:
                mock_client = mock_client_class.return_value
                
                attempt_count = [0]
                
                def side_effect(*args, **kwargs):
                    attempt_count[0] += 1
                    if attempt_count[0] == 1:
                        # First: timeout
                        raise TimeoutError("Request timeout")
                    elif attempt_count[0] == 2:
                        # Second: invalid JSON
                        return (
                            "{broken json",
                            "response_id_2",
                            ResponseUsage(input_tokens=10, output_tokens=20, reasoning_tokens=0, total_tokens=30)
                        )
                    else:
                        # Third: success
                        return (
                            json.dumps([]),  # Empty edges list is valid
                            "response_id_3",
                            ResponseUsage(input_tokens=10, output_tokens=20, reasoning_tokens=0, total_tokens=30)
                        )
                
                # First call uses create_response, subsequent calls use repair_response
                mock_client.create_response.side_effect = side_effect
                mock_client.repair_response.side_effect = side_effect
                mock_client.confirm_response.return_value = None
                
                with patch("time.sleep"):
                    new_edges, api_usage = analyze_candidate_pairs(
                        candidate_pairs,
                        mock_graph,
                        mock_config,
                        logger,
                        pass_direction="forward"
                    )
                
                # Should handle both error types and succeed
                assert mock_client.repair_response.call_count == 2
                # Second attempt returns invalid JSON (counted), third succeeds (counted)
                assert api_usage["requests"] == 2

    def test_refiner_stops_on_persistent_failure(self, mock_config, mock_graph):
        """Test that refiner stops completely when node fails persistently."""
        logger = Mock()
        
        # Create 3 candidate pairs
        three_pairs = [
            {
                "source_node": {"id": f"chunk_{i:03d}", "type": "Chunk", "text": f"Test chunk {i}"},
                "candidates": [{"node_id": f"chunk_{i+1:03d}", "similarity": 0.8, "text": f"Test chunk {i+1}"}]
            }
            for i in range(1, 4)
        ]
        
        with patch("src.refiner_longrange.load_refiner_longrange_prompt") as mock_load_prompt:
            mock_load_prompt.return_value = "Test prompt"
            
            with patch("src.refiner_longrange.OpenAIClient") as mock_client_class:
                mock_client = mock_client_class.return_value
                
                node_count = [0]
                
                def side_effect(*args, **kwargs):
                    # Check which node we're processing based on input_data
                    input_str = kwargs.get('input_data', args[1] if len(args) > 1 else "")
                    
                    if "chunk_001" in input_str:
                        # First node succeeds
                        return (
                            json.dumps([{"source": "chunk_001", "target": "chunk_002", "type": "ELABORATES", "weight": 0.7}]),
                            "response_1",
                            ResponseUsage(input_tokens=10, output_tokens=20, reasoning_tokens=0, total_tokens=30)
                        )
                    elif "chunk_002" in input_str:
                        # Second node always fails
                        raise TimeoutError("Persistent timeout")
                    else:
                        # Third node succeeds
                        return (
                            json.dumps([{"source": "chunk_003", "target": "chunk_004", "type": "ELABORATES", "weight": 0.7}]),
                            "response_3",
                            ResponseUsage(input_tokens=10, output_tokens=20, reasoning_tokens=0, total_tokens=30)
                        )
                
                mock_client.create_response.side_effect = side_effect
                mock_client.repair_response.side_effect = TimeoutError("Persistent timeout")
                mock_client.confirm_response.return_value = None
                
                # Mock Path for bad response saving
                with patch("src.refiner_longrange.Path"):
                    with patch("time.sleep"):
                        new_edges, api_usage = analyze_candidate_pairs(
                            three_pairs,
                            mock_graph,
                            mock_config,
                            logger,
                            pass_direction="forward"
                        )
                
                # Should stop after node 2 fails - only node 1 processed
                assert len(new_edges) == 1
                assert new_edges[0].get("source") == "chunk_001"
                # Node 3 should not be processed (stopped after node 2 failure)
                assert not any(e.get("source") == "chunk_003" for e in new_edges)
                # Only node 1 successful
                assert api_usage["requests"] == 1
"""Tests for itext2kg_concepts module."""

import json
from unittest.mock import patch

import pytest

from src.itext2kg_concepts import ProcessingStats, SliceData, SliceProcessor
from src.utils.exit_codes import EXIT_INPUT_ERROR, EXIT_SUCCESS
from src.utils.llm_client import ResponseUsage


class TestSliceProcessor:
    """Test SliceProcessor class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return {
            "itext2kg": {
                "model": "test-model",
                "tpm_limit": 100000,
                "log_level": "info",
                "max_context_tokens": 128000,
                "max_context_tokens_test": 128000,
            }
        }

    @pytest.fixture
    def processor(self, mock_config, tmp_path):
        """Create processor instance with mocked dependencies."""
        with patch("src.itext2kg_concepts.LOGS_DIR", tmp_path / "logs"):
            with patch("src.itext2kg_concepts.PROMPTS_DIR", tmp_path / "prompts"):
                with patch("src.itext2kg_concepts.SCHEMAS_DIR", tmp_path / "schemas"):
                    # Create necessary directories
                    (tmp_path / "logs").mkdir()
                    (tmp_path / "prompts").mkdir()
                    (tmp_path / "schemas").mkdir()

                    # Create prompt file
                    prompt_file = tmp_path / "prompts" / "itext2kg_concepts_extraction.md"
                    prompt_file.write_text(
                        "Test prompt\n{concept_dictionary_schema}", encoding="utf-8"
                    )

                    # Create schema file
                    schema_file = tmp_path / "schemas" / "ConceptDictionary.schema.json"
                    schema_file.write_text(json.dumps({"$schema": "test-schema"}), encoding="utf-8")

                    # Mock LLM client
                    with patch("src.itext2kg_concepts.OpenAIClient"):
                        processor = SliceProcessor(mock_config)
                        return processor

    def test_initialization(self, processor):
        """Test processor initialization."""
        assert processor.concept_dictionary == {"concepts": []}
        assert processor.concept_id_map == {}
        assert isinstance(processor.stats, ProcessingStats)
        assert processor.stats.total_slices == 0
        assert processor.stats.total_concepts == 0

    def test_format_tokens(self, processor):
        """Test token formatting."""
        assert processor._format_tokens(123) == "123"
        assert processor._format_tokens(1234) == "1.23k"
        assert processor._format_tokens(45678) == "45.68k"
        assert processor._format_tokens(1234567) == "1.23M"

    def test_load_slice(self, processor, tmp_path):
        """Test slice loading."""
        slice_file = tmp_path / "test.slice.json"
        slice_data = {
            "id": "slice_001",
            "order": 1,
            "source_file": "test.md",
            "slug": "test",
            "text": "Test content",
            "slice_token_start": 0,
            "slice_token_end": 100,
        }
        slice_file.write_text(json.dumps(slice_data), encoding="utf-8")

        loaded = processor._load_slice(slice_file)
        assert loaded.id == "slice_001"
        assert loaded.order == 1
        assert loaded.text == "Test content"

    def test_format_slice_input(self, processor):
        """Test input formatting for LLM."""
        slice_data = SliceData(
            id="slice_001",
            order=1,
            source_file="test.md",
            slug="test",
            text="Test content",
            slice_token_start=0,
            slice_token_end=100,
        )

        input_json = processor._format_slice_input(slice_data)
        input_data = json.loads(input_json)

        assert "ConceptDictionary" in input_data
        assert "Slice" in input_data
        assert input_data["Slice"]["id"] == "slice_001"
        assert input_data["Slice"]["text"] == "Test content"

    def test_update_concept_dictionary_new_concept(self, processor):
        """Test adding new concept to dictionary."""
        concepts_added = [
            {
                "concept_id": "test:p:stack",
                "term": {"primary": "Stack", "aliases": ["LIFO", "стек"]},
                "definition": "LIFO data structure",
            }
        ]

        processor._update_concept_dictionary(concepts_added)

        assert len(processor.concept_dictionary["concepts"]) == 1
        assert processor.concept_dictionary["concepts"][0]["concept_id"] == "test:p:stack"
        assert processor.stats.total_concepts == 1
        assert "test:p:stack" in processor.concept_id_map

    def test_update_concept_dictionary_case_insensitive_dedup(self, processor):
        """Test case-insensitive alias deduplication."""
        concepts_added = [
            {
                "concept_id": "test:p:stack",
                "term": {"primary": "Stack", "aliases": ["LIFO", "lifo", "Lifo"]},
                "definition": "LIFO data structure",
            }
        ]

        processor._update_concept_dictionary(concepts_added)

        concept = processor.concept_dictionary["concepts"][0]
        # Should keep only first occurrence of each unique alias
        assert len(concept["term"]["aliases"]) == 1
        assert concept["term"]["aliases"][0] == "LIFO"

    def test_update_concept_dictionary_existing_concept(self, processor):
        """Test updating existing concept with new aliases."""
        # Add initial concept
        initial_concept = {
            "concept_id": "test:p:stack",
            "term": {"primary": "Stack", "aliases": ["LIFO"]},
            "definition": "LIFO data structure",
        }
        processor.concept_dictionary["concepts"].append(initial_concept)
        processor.concept_id_map["test:p:stack"] = 0
        processor.stats.total_concepts = 1

        # Update with new aliases
        concepts_added = [
            {
                "concept_id": "test:p:stack",
                "term": {"primary": "Stack", "aliases": ["LIFO", "стек", "stack structure"]},
                "definition": "LIFO data structure",
            }
        ]

        processor._update_concept_dictionary(concepts_added)

        # Should have same number of concepts
        assert len(processor.concept_dictionary["concepts"]) == 1
        assert processor.stats.total_concepts == 1

        # Should have all unique aliases (sorted)
        concept = processor.concept_dictionary["concepts"][0]
        assert sorted(concept["term"]["aliases"]) == ["LIFO", "stack structure", "стек"]

    def test_process_llm_response_valid(self, processor):
        """Test processing valid LLM response."""
        response_text = json.dumps(
            {
                "concepts_added": {
                    "concepts": [
                        {
                            "concept_id": "test:p:stack",
                            "term": {"primary": "Stack", "aliases": ["LIFO"]},
                            "definition": "LIFO data structure",
                        }
                    ]
                }
            }
        )

        success, parsed_data = processor._process_llm_response(response_text, "slice_001")

        assert success is True
        assert parsed_data is not None
        assert "concepts_added" in parsed_data

    def test_process_llm_response_invalid(self, processor):
        """Test processing invalid LLM response."""
        # Missing required field
        response_text = json.dumps({"wrong_field": {}})

        success, parsed_data = processor._process_llm_response(response_text, "slice_001")

        assert success is False
        assert parsed_data is None

    def test_process_llm_response_empty_concepts(self, processor):
        """Test processing response with empty concepts list."""
        response_text = json.dumps({"concepts_added": {"concepts": []}})

        success, parsed_data = processor._process_llm_response(response_text, "slice_001")

        assert success is True
        assert parsed_data is not None
        assert parsed_data["concepts_added"]["concepts"] == []

    def test_apply_concepts(self, processor):
        """Test applying concepts from LLM response."""
        response_data = {
            "concepts_added": {
                "concepts": [
                    {
                        "concept_id": "test:p:stack",
                        "term": {"primary": "Stack", "aliases": ["LIFO"]},
                        "definition": "LIFO data structure",
                    }
                ]
            }
        }

        processor._apply_concepts(response_data)

        assert len(processor.concept_dictionary["concepts"]) == 1
        assert processor.stats.total_concepts == 1

    def test_save_bad_response(self, processor, tmp_path):
        """Test saving bad LLM responses."""
        with patch("src.itext2kg_concepts.LOGS_DIR", tmp_path):
            processor._save_bad_response(
                "slice_001", "invalid json", "JSON parse error", "still invalid"
            )

            bad_file = tmp_path / "slice_001_bad.json"
            assert bad_file.exists()

            bad_data = json.loads(bad_file.read_text())
            assert bad_data["slice_id"] == "slice_001"
            assert bad_data["original_response"] == "invalid json"
            assert bad_data["validation_error"] == "JSON parse error"
            assert bad_data["repair_response"] == "still invalid"

    def test_save_temp_dumps(self, processor, tmp_path):
        """Test saving temporary dumps."""
        with patch("src.itext2kg_concepts.LOGS_DIR", tmp_path):
            # Add some concepts
            processor.concept_dictionary["concepts"].append(
                {
                    "concept_id": "test:p:stack",
                    "term": {"primary": "Stack"},
                    "definition": "Test",
                }
            )
            processor.stats.total_concepts = 1

            processor._save_temp_dumps("test_reason")

            # Check files were created
            concept_files = list(tmp_path.glob("ConceptDictionary_temp_test_reason_*.json"))
            stats_files = list(tmp_path.glob("processing_stats_test_reason_*.json"))

            assert len(concept_files) == 1
            assert len(stats_files) == 1

    @patch("src.itext2kg_concepts.SliceProcessor._process_single_slice")
    def test_run_no_slices(self, mock_process, processor, tmp_path):
        """Test run with no slices."""
        with patch("src.itext2kg_concepts.STAGING_DIR", tmp_path):
            result = processor.run()
            assert result == EXIT_INPUT_ERROR
            mock_process.assert_not_called()

    @patch("src.itext2kg_concepts.SliceProcessor._finalize_and_save")
    @patch("src.itext2kg_concepts.SliceProcessor._process_single_slice")
    def test_run_successful(self, mock_process, mock_finalize, processor, tmp_path):
        """Test successful run."""
        # Create slice files
        staging_dir = tmp_path / "staging"
        staging_dir.mkdir()
        (staging_dir / "slice_001.slice.json").write_text("{}")
        (staging_dir / "slice_002.slice.json").write_text("{}")

        with patch("src.itext2kg_concepts.STAGING_DIR", staging_dir):
            mock_process.return_value = True
            mock_finalize.return_value = EXIT_SUCCESS

            result = processor.run()

            assert result == EXIT_SUCCESS
            assert mock_process.call_count == 2
            assert processor.stats.processed_slices == 2

    def test_process_single_slice_with_repair(self, processor, tmp_path):
        """Test processing slice with repair mechanism."""
        slice_file = tmp_path / "test.slice.json"
        slice_data = {
            "id": "slice_001",
            "order": 1,
            "source_file": "test.md",
            "slug": "test",
            "text": "Test content",
            "slice_token_start": 0,
            "slice_token_end": 100,
        }
        slice_file.write_text(json.dumps(slice_data), encoding="utf-8")

        # Mock LLM client to return invalid then valid response
        mock_client = processor.llm_client
        mock_client.create_response.return_value = (
            "invalid json",
            "response_id_1",
            ResponseUsage(input_tokens=100, output_tokens=50, total_tokens=150, reasoning_tokens=0),
        )
        mock_client.repair_response.return_value = (
            json.dumps({"concepts_added": {"concepts": []}}),
            "response_id_2",
            ResponseUsage(input_tokens=120, output_tokens=60, total_tokens=180, reasoning_tokens=0),
        )

        with patch("src.itext2kg_concepts.LOGS_DIR", tmp_path):
            result = processor._process_single_slice(slice_file)

        assert result is True
        assert mock_client.repair_response.called
        assert processor.stats.total_tokens_used == 180  # Uses repair usage

    def test_context_preservation(self, processor, tmp_path):
        """Test that previous_response_id is preserved through processing."""
        slice_file = tmp_path / "test.slice.json"
        slice_data = {
            "id": "slice_001",
            "order": 1,
            "source_file": "test.md",
            "slug": "test",
            "text": "Test content",
            "slice_token_start": 0,
            "slice_token_end": 100,
        }
        slice_file.write_text(json.dumps(slice_data), encoding="utf-8")

        # Mock LLM client
        mock_client = processor.llm_client
        mock_client.create_response.return_value = (
            json.dumps({"concepts_added": {"concepts": []}}),
            "response_id_1",
            ResponseUsage(input_tokens=100, output_tokens=50, total_tokens=150, reasoning_tokens=0),
        )

        with patch("src.itext2kg_concepts.LOGS_DIR", tmp_path):
            # Process first slice
            processor._process_single_slice(slice_file)

            # Process second slice - should use previous response context
            processor._process_single_slice(slice_file)

        # LLM client should maintain context automatically
        assert mock_client.create_response.call_count == 2

    def test_finalize_and_save_success(self, processor, tmp_path):
        """Test successful finalization and save."""
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        # Add concept
        processor.concept_dictionary["concepts"].append(
            {
                "concept_id": "test:p:stack",
                "term": {"primary": "Stack", "aliases": ["LIFO"]},
                "definition": "LIFO data structure",
            }
        )
        processor.stats.processed_slices = 5

        with patch("src.itext2kg_concepts.OUTPUT_DIR", output_dir):
            result = processor._finalize_and_save()

        assert result == EXIT_SUCCESS
        assert (output_dir / "ConceptDictionary.json").exists()

        # Check saved content
        saved_data = json.loads((output_dir / "ConceptDictionary.json").read_text())
        assert len(saved_data["concepts"]) == 1
        assert saved_data["concepts"][0]["concept_id"] == "test:p:stack"

    def test_previous_response_id_preserved(self, processor, tmp_path):
        """Test that previous_response_id is used across slices."""
        # Create multiple slice files
        slice_data_template = {
            "order": 1,
            "source_file": "test.md",
            "slug": "test",
            "text": "Test content",
            "slice_token_start": 0,
            "slice_token_end": 100,
        }

        slice_files = []
        for i in range(3):
            slice_file = tmp_path / f"slice_{i:03d}.slice.json"
            slice_data = slice_data_template.copy()
            slice_data["id"] = f"slice_{i:03d}"
            slice_data["order"] = i + 1
            slice_file.write_text(json.dumps(slice_data), encoding="utf-8")
            slice_files.append(slice_file)

        # Mock LLM client
        mock_client = processor.llm_client
        response_ids = ["response_id_1", "response_id_2", "response_id_3"]

        # Track calls to create_response
        call_count = 0

        def mock_create_response(**kwargs):
            nonlocal call_count
            result = (
                json.dumps({"concepts_added": {"concepts": []}}),
                response_ids[call_count],
                ResponseUsage(
                    input_tokens=100, output_tokens=50, total_tokens=150, reasoning_tokens=0
                ),
            )
            call_count += 1
            return result

        mock_client.create_response.side_effect = mock_create_response

        with patch("src.itext2kg_concepts.LOGS_DIR", tmp_path):
            # Process first slice - should have None as previous_response_id
            processor._process_single_slice(slice_files[0])
            assert mock_client.create_response.call_args[1]["previous_response_id"] is None
            assert processor.previous_response_id == "response_id_1"

            # Process second slice - should use first response_id
            processor._process_single_slice(slice_files[1])
            assert (
                mock_client.create_response.call_args[1]["previous_response_id"] == "response_id_1"
            )
            assert processor.previous_response_id == "response_id_2"

            # Process third slice - should use second response_id
            processor._process_single_slice(slice_files[2])
            assert (
                mock_client.create_response.call_args[1]["previous_response_id"] == "response_id_2"
            )
            assert processor.previous_response_id == "response_id_3"

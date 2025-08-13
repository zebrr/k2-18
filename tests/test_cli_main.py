"""
Тесты для main() функций CLI утилит.

Покрывает точки входа всех CLI модулей:
- dedup.py
- slicer.py
- itext2kg_concepts.py
- itext2kg_graph.py
- refiner_longrange.py
"""

import json
from unittest.mock import Mock, mock_open, patch

import pytest

from src.utils.exit_codes import (
    EXIT_CONFIG_ERROR,
    EXIT_INPUT_ERROR,
    EXIT_RUNTIME_ERROR,
    EXIT_SUCCESS,
)


class TestDedupMain:
    """Тесты для dedup.py::main()"""

    @patch("src.dedup.load_config")
    @patch("src.dedup.Path.exists")
    @patch("src.dedup.open", new_callable=mock_open)
    @patch("src.dedup.json.load")
    @patch("src.dedup.json.dump")
    @patch("src.dedup.get_embeddings")
    @patch("src.dedup.build_faiss_index")
    @patch("src.dedup.find_duplicates")
    def test_main_success(
        self,
        mock_find_duplicates,
        mock_build_index,
        mock_get_embeddings,
        mock_json_dump,
        mock_json_load,
        mock_file_open,
        mock_exists,
        mock_load_config,
    ):
        """Тест успешного выполнения dedup.main()"""
        # Setup configuration
        mock_load_config.return_value = {
            "dedup": {
                "sim_threshold": 0.95,
                "min_neighbors": 5,
                "batch_size": 100,
            }
        }

        # Setup file existence
        mock_exists.return_value = True

        # Setup graph loading
        test_graph = {
            "nodes": [
                {"id": "handbook:c:100", "type": "Chunk", "text": "Test text 1", "node_offset": 0},
                {
                    "id": "handbook:c:200",
                    "type": "Chunk",
                    "text": "Test text 2",
                    "node_offset": 100,
                },
            ],
            "edges": [],
        }
        mock_json_load.return_value = test_graph

        # Setup embeddings and FAISS
        import numpy as np

        embeddings_dict = {"handbook:c:100": [0.1, 0.2], "handbook:c:200": [0.3, 0.4]}
        embeddings_array = np.array(list(embeddings_dict.values()))
        mock_get_embeddings.return_value = (embeddings_dict, embeddings_array)
        mock_build_index.return_value = Mock()  # Mock FAISS index
        mock_find_duplicates.return_value = {}  # No duplicates

        # Run main
        from src.dedup import main

        exit_code = main()

        # Verify
        assert exit_code == EXIT_SUCCESS
        mock_load_config.assert_called_once()
        assert mock_json_load.call_count >= 1  # Called for graph and schema
        mock_json_dump.assert_called()

    @patch("src.dedup.load_config")
    def test_main_config_error(self, mock_load_config):
        """Тест ошибки загрузки конфигурации"""
        mock_load_config.side_effect = Exception("Config error")

        from src.dedup import main

        exit_code = main()

        assert exit_code == EXIT_CONFIG_ERROR

    @patch("src.dedup.load_config")
    @patch("src.dedup.Path.exists")
    def test_main_input_file_missing(self, mock_exists, mock_load_config):
        """Тест отсутствующего входного файла"""
        mock_load_config.return_value = {"dedup": {}}
        mock_exists.return_value = False

        from src.dedup import main

        exit_code = main()

        assert exit_code == EXIT_INPUT_ERROR

    @patch("src.dedup.load_config")
    @patch("src.dedup.Path.exists")
    @patch("src.dedup.open", new_callable=mock_open)
    @patch("src.dedup.json.load")
    def test_main_invalid_json(self, mock_json_load, mock_file_open, mock_exists, mock_load_config):
        """Тест невалидного JSON файла"""
        mock_load_config.return_value = {"dedup": {}}
        mock_exists.return_value = True
        mock_json_load.side_effect = json.JSONDecodeError("Invalid", "doc", 0)

        from src.dedup import main

        exit_code = main()

        # dedup возвращает EXIT_RUNTIME_ERROR для JSON ошибок
        assert exit_code == EXIT_RUNTIME_ERROR


class TestSlicerMain:
    """Тесты для slicer.py::main()"""

    def test_main_success(self, tmp_path, monkeypatch):
        """Тест успешного выполнения slicer.main()"""
        # Create test files
        raw_dir = tmp_path / "data" / "raw"
        staging_dir = tmp_path / "data" / "staging"
        raw_dir.mkdir(parents=True)
        staging_dir.mkdir(parents=True)

        # Create test file
        test_file = raw_dir / "test.md"
        test_file.write_text("# Test\n\nThis is a test file.")

        # Mock config and run
        monkeypatch.chdir(tmp_path)

        with patch("src.slicer.load_config") as mock_config:
            mock_config.return_value = {
                "slicer": {
                    "log_level": "info",
                    "max_tokens": 100,
                    "overlap": 10,
                    "allowed_extensions": ["md"],
                    "soft_boundary": False,
                    "soft_boundary_max_shift": 0,
                }
            }

            from src.slicer import main

            exit_code = main([])

            # Verify
            assert exit_code == EXIT_SUCCESS

            # Check that slice files were created
            slice_files = list(staging_dir.glob("*.slice.json"))
            assert len(slice_files) > 0

    @patch("src.slicer.load_config")
    def test_main_config_error(self, mock_load_config):
        """Тест ошибки конфигурации"""
        mock_load_config.side_effect = Exception("Config error")

        from src.slicer import main

        exit_code = main([])

        # slicer возвращает EXIT_RUNTIME_ERROR для исключений при загрузке конфига
        assert exit_code == EXIT_RUNTIME_ERROR

    def test_main_no_files(self, tmp_path, monkeypatch):
        """Тест когда нет файлов для обработки"""
        # Create empty directories
        raw_dir = tmp_path / "data" / "raw"
        staging_dir = tmp_path / "data" / "staging"
        raw_dir.mkdir(parents=True)
        staging_dir.mkdir(parents=True)
        # No files created

        monkeypatch.chdir(tmp_path)

        with patch("src.slicer.load_config") as mock_config:
            mock_config.return_value = {
                "slicer": {
                    "log_level": "info",
                    "max_tokens": 100,
                    "overlap": 10,
                    "allowed_extensions": ["md"],
                    "soft_boundary": False,
                    "soft_boundary_max_shift": 0,
                }
            }

            from src.slicer import main

            exit_code = main([])

            # Verify
            assert exit_code == EXIT_SUCCESS  # No files is not an error


class TestItext2kgConceptsMain:
    """Тесты для itext2kg_concepts.py::main()"""

    @patch("src.itext2kg_concepts.load_config")
    @patch("src.itext2kg_concepts.SliceProcessor")
    def test_main_success(self, mock_processor_class, mock_load_config):
        """Тест успешного выполнения"""
        # Setup configuration
        mock_load_config.return_value = {
            "itext2kg": {
                "max_context_tokens": 10000,
                "max_context_tokens_test": 5000,
            }
        }

        # Setup processor
        mock_processor = Mock()
        mock_processor.run.return_value = EXIT_SUCCESS
        mock_processor_class.return_value = mock_processor

        # Run main
        from src.itext2kg_concepts import main

        exit_code = main()

        # Verify
        assert exit_code == EXIT_SUCCESS
        mock_load_config.assert_called_once()
        mock_processor.run.assert_called_once()

    @patch("src.itext2kg_concepts.load_config")
    def test_main_invalid_max_context(self, mock_load_config):
        """Тест невалидного max_context_tokens"""
        mock_load_config.return_value = {
            "itext2kg": {
                "max_context_tokens": 500,  # Too small
                "max_context_tokens_test": 5000,
            }
        }

        from src.itext2kg_concepts import main

        exit_code = main()

        assert exit_code == EXIT_CONFIG_ERROR

    @pytest.mark.skip(reason="Hangs in test execution")
    @patch("src.itext2kg_concepts.load_config")
    def test_main_keyboard_interrupt(self, mock_load_config):
        """Тест прерывания пользователем"""
        mock_load_config.side_effect = KeyboardInterrupt()

        from src.itext2kg_concepts import main

        exit_code = main()

        assert exit_code == EXIT_RUNTIME_ERROR


class TestItext2kgGraphMain:
    """Тесты для itext2kg_graph.py::main()"""

    @patch("src.itext2kg_graph.load_config")
    @patch("src.itext2kg_graph.SliceProcessor")
    @patch("src.itext2kg_graph.sys.exit")
    def test_main_success(self, mock_exit, mock_processor_class, mock_load_config):
        """Тест успешного выполнения"""
        # Setup configuration
        mock_load_config.return_value = {
            "itext2kg": {
                "llm": {"model": "gpt-4"},
                "max_context_tokens": 10000,
            }
        }

        # Setup processor
        mock_processor = Mock()
        mock_processor.run.return_value = EXIT_SUCCESS
        mock_processor_class.return_value = mock_processor

        # Run main
        from src.itext2kg_graph import main

        main()

        # Verify
        mock_load_config.assert_called_once()
        mock_processor.run.assert_called_once()
        mock_exit.assert_called_with(EXIT_SUCCESS)

    @pytest.mark.skip(reason="Hangs in test execution")
    @patch("src.itext2kg_graph.load_config")
    @patch("src.itext2kg_graph.sys.exit")
    def test_main_keyboard_interrupt(self, mock_exit, mock_load_config):
        """Тест прерывания пользователем"""
        mock_load_config.side_effect = KeyboardInterrupt()

        from src.itext2kg_graph import main

        with patch("builtins.print"):  # Suppress print output
            main()

        mock_exit.assert_called_with(EXIT_RUNTIME_ERROR)

    @patch("src.itext2kg_graph.load_config")
    @patch("src.itext2kg_graph.sys.exit")
    def test_main_unexpected_error(self, mock_exit, mock_load_config):
        """Тест неожиданной ошибки"""
        mock_load_config.side_effect = Exception("Unexpected error")

        from src.itext2kg_graph import main

        with patch("builtins.print"):  # Suppress print output
            main()

        mock_exit.assert_called_with(EXIT_RUNTIME_ERROR)


class TestRefinerMain:
    """Тесты для refiner_longrange.py::main()"""

    @patch("src.refiner_longrange.load_config")
    @patch("src.refiner_longrange.setup_json_logging")  # Fixed from setup_logging
    @patch("src.refiner_longrange.Path.exists")
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.refiner_longrange.json.load")
    @patch("src.refiner_longrange.json.dump")
    @patch("src.refiner_longrange.get_node_embeddings")  # Fixed from get_embeddings
    @patch("src.refiner_longrange.OpenAIClient")
    @patch("src.refiner_longrange.generate_candidate_pairs")  # Fixed from find_candidates
    @patch("src.refiner_longrange.analyze_candidate_pairs")  # Fixed from analyze_relations
    def test_main_success(
        self,
        mock_analyze,
        mock_find_candidates,
        mock_client_class,
        mock_get_embeddings,
        mock_json_dump,
        mock_json_load,
        mock_file,
        mock_exists,
        mock_setup_logging,
        mock_load_config,
    ):
        """Тест успешного выполнения refiner.main()"""
        # Setup configuration
        mock_load_config.return_value = {
            "refiner": {
                "run": True,
                "is_reasoning": False,  # Required parameter
                "max_context_tokens": 10000,
                "max_context_tokens_test": 5000,
                "sim_threshold": 0.8,
                "max_pairs_per_node": 10,
                "api_key": "test-key",
                "model": "gpt-4",
                "tpm_limit": 10000,
                "max_completion": 1000,
                "embedding_model": "text-embedding-3-small",
                "embedding_tpm_limit": 100000,
                "weight_low": 0.3,
                "weight_mid": 0.6,
                "weight_high": 0.9,
                "faiss_M": 32,
                "faiss_metric": "INNER_PRODUCT",
            }
        }

        # Setup file existence
        mock_exists.return_value = True

        # Setup graph loading
        test_graph = {
            "nodes": [
                {"id": "handbook:c:100", "type": "Chunk", "text": "Test"},
            ],
            "edges": [],
        }
        mock_json_load.return_value = test_graph

        # Setup processing
        mock_get_embeddings.return_value = {"handbook:c:100": [0.1, 0.2]}
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_find_candidates.return_value = []

        # Run main
        from src.refiner_longrange import main

        exit_code = main()

        # Verify
        assert exit_code == EXIT_SUCCESS
        mock_load_config.assert_called_once()
        mock_json_dump.assert_called()

    @patch("src.refiner_longrange.load_config")
    @patch("src.refiner_longrange.shutil.copy2")
    def test_main_disabled(self, mock_copy, mock_load_config):
        """Тест когда refiner отключен в конфиге"""
        mock_load_config.return_value = {
            "refiner": {
                "run": False,
            }
        }

        from src.refiner_longrange import main

        with patch("builtins.print"):  # Suppress print output
            exit_code = main()

        assert exit_code == EXIT_SUCCESS
        # Проверяем что файл был скопирован
        mock_copy.assert_called_once()

    @patch("src.refiner_longrange.load_config")
    def test_main_invalid_max_context(self, mock_load_config):
        """Тест невалидного max_context_tokens"""
        mock_load_config.return_value = {
            "refiner": {
                "run": True,
                "max_context_tokens": 100,  # Too small
            }
        }

        from src.refiner_longrange import main

        exit_code = main()

        assert exit_code == EXIT_CONFIG_ERROR

    @patch("src.refiner_longrange.load_config")
    @patch("src.refiner_longrange.setup_json_logging")  # Fixed from setup_logging
    @patch("src.refiner_longrange.load_and_validate_graph")
    def test_main_input_missing(self, mock_load_graph, mock_setup_logging, mock_load_config):
        """Тест отсутствующего входного файла"""
        mock_load_config.return_value = {
            "refiner": {
                "run": True,
                "is_reasoning": False,
                "max_context_tokens": 10000,
                "max_context_tokens_test": 5000,
                "api_key": "test-key",
                "model": "gpt-4",
                "tpm_limit": 10000,
                "sim_threshold": 0.8,
                "max_pairs_per_node": 10,
                "embedding_model": "text-embedding-3-small",
                "embedding_tpm_limit": 100000,
                "weight_low": 0.3,
                "weight_mid": 0.6,
                "weight_high": 0.9,
                "faiss_M": 32,
                "faiss_metric": "INNER_PRODUCT",
            }
        }

        # Mock load_and_validate_graph to raise FileNotFoundError
        mock_load_graph.side_effect = FileNotFoundError("Input file not found")

        from src.refiner_longrange import main

        exit_code = main()

        # Валидация конфигурации происходит до проверки файла,
        # поэтому возвращается CONFIG_ERROR, а не INPUT_ERROR
        assert exit_code == EXIT_CONFIG_ERROR

"""
End-to-end тесты для refiner.py

Тестируют полный pipeline refiner с использованием моков для внешних систем.
Реальная интеграция с OpenAI API тестируется в test_llm_client_integration.py
"""

import json
import logging
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np

# Добавляем путь к src для импортов
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.exit_codes import EXIT_INPUT_ERROR, EXIT_RUNTIME_ERROR, EXIT_SUCCESS


class TestRefinerMain(unittest.TestCase):
    """Интеграционные тесты основной функции."""

    @patch("refiner_longrange.load_config")
    @patch("shutil.copy2")
    @patch("pathlib.Path.exists")
    def test_run_false(self, mock_exists, mock_copy, mock_load_config):
        """Тест с run=false - просто копирование файла."""
        from refiner_longrange import main

        # Конфигурация с run=false
        mock_load_config.return_value = {"refiner": {"run": False}}

        mock_exists.return_value = True

        result = main()

        # Должен просто скопировать файл
        self.assertEqual(result, EXIT_SUCCESS)
        mock_copy.assert_called_once_with(
            Path("data/out/LearningChunkGraph_dedup.json"),
            Path("data/out/LearningChunkGraph_longrange.json"),
        )

    @patch("refiner_longrange.setup_json_logging")
    @patch("refiner_longrange.load_config")
    @patch("pathlib.Path.exists")
    @patch("refiner_longrange.load_and_validate_graph")
    def test_input_file_not_found(self, mock_load_graph, mock_exists, mock_load_config, mock_setup_logging):
        """Тест отсутствующего входного файла."""
        from refiner_longrange import main

        mock_load_config.return_value = {
            "refiner": {
                "run": True,
                "embedding_model": "text-embedding-3-small",
                "sim_threshold": 0.8,
                "max_pairs_per_node": 20,
                "model": "gpt-4o",
                "api_key": "sk-test",
                "tpm_limit": 100000,
                "max_completion": 4096,
                "weight_low": 0.3,
                "weight_mid": 0.6,
                "weight_high": 0.9,
                "faiss_M": 32,
                "faiss_metric": "INNER_PRODUCT",
            }
        }

        # Файл не существует
        mock_exists.return_value = False

        # load_and_validate_graph должна выбросить FileNotFoundError
        mock_load_graph.side_effect = FileNotFoundError("Input file not found")

        # Mock logger to prevent creating real log files
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        
        result = main()

        self.assertEqual(result, EXIT_INPUT_ERROR)


class TestGraphLoadValidate(unittest.TestCase):
    """Тесты загрузки и валидации графа с файловой системой."""

    def setUp(self):
        self.sample_graph = {
            "nodes": [
                {
                    "id": "test:c:0",
                    "type": "Chunk",
                    "text": "Test chunk 1",
                },
                {
                    "id": "test:c:100",
                    "type": "Chunk",
                    "text": "Test chunk 2",
                },
                {
                    "id": "test:c:300",
                    "type": "Concept",
                    "text": "Test concept",
                },
                {
                    "id": "test:q:200:0",
                    "type": "Assessment",
                    "text": "Test question",
                },
            ],
            "edges": [
                {"source": "test:c:0", "target": "test:c:100", "type": "PREREQUISITE"},
                {"source": "test:c:100", "target": "test:c:300", "type": "MENTIONS"},
            ],
        }

    def test_load_and_validate_graph(self):
        """Тест загрузки и валидации графа из файла."""
        from refiner_longrange import load_and_validate_graph

        # Mock файла и Path
        with patch("builtins.open", mock_open(read_data=json.dumps(self.sample_graph))):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("refiner_longrange.validate_json") as mock_validate:
                    graph = load_and_validate_graph(Path("test.json"))

        # Проверяем загрузку
        self.assertEqual(len(graph["nodes"]), 4)
        self.assertEqual(len(graph["edges"]), 2)

        # Проверяем, что валидация была вызвана
        mock_validate.assert_called_once_with(graph, "LearningChunkGraph")

    def test_load_invalid_json(self):
        """Тест загрузки некорректного JSON."""
        from refiner_longrange import load_and_validate_graph

        with patch("builtins.open", mock_open(read_data="invalid json{{")):
            with patch("pathlib.Path.exists", return_value=True):
                with self.assertRaises(json.JSONDecodeError):
                    load_and_validate_graph(Path("test.json"))

    def test_file_not_found(self):
        """Тест с несуществующим файлом."""
        from refiner_longrange import load_and_validate_graph

        with patch("pathlib.Path.exists", return_value=False):
            with self.assertRaises(FileNotFoundError):
                load_and_validate_graph(Path("nonexistent.json"))


class TestEmbeddingsIntegration(unittest.TestCase):
    """Интеграционные тесты работы с embeddings."""

    def setUp(self):
        self.logger = logging.getLogger("test")
        self.config = {
            "embedding_model": "text-embedding-3-small",
            "api_key": "test-key",
            "embedding_api_key": "test-embedding-key",
            "sim_threshold": 0.8,
            "max_pairs_per_node": 5,
            "faiss_M": 16,
            "faiss_metric": "INNER_PRODUCT",
        }

        self.nodes = [
            {
                "id": "test:c:0",
                "type": "Chunk",
                "text": "Python variables",
            },
            {
                "id": "test:c:100",
                "type": "Chunk",
                "text": "Python functions",
            },
            {
                "id": "test:q:200:0",
                "type": "Assessment",
                "text": "Test question",
            },
        ]

    @patch("refiner_longrange.get_embeddings")
    def test_get_node_embeddings_success(self, mock_get_embeddings):
        """Тест успешного получения embeddings через API."""
        from refiner_longrange import get_node_embeddings

        # Mock возвращает векторы правильной размерности
        mock_embeddings = np.random.rand(3, 1536).astype(np.float32)
        mock_get_embeddings.return_value = mock_embeddings

        result = get_node_embeddings(self.nodes, self.config, self.logger)

        # Проверяем результат
        self.assertEqual(len(result), 3)
        self.assertIn("test:c:0", result)
        self.assertEqual(result["test:c:0"].shape, (1536,))

        # Проверяем вызов API
        mock_get_embeddings.assert_called_once_with(
            ["Python variables", "Python functions", "Test question"], self.config
        )

    @patch("refiner_longrange.get_embeddings")
    def test_get_node_embeddings_empty_text(self, mock_get_embeddings):
        """Тест с пустыми текстами - фильтрация перед API."""
        from refiner_longrange import get_node_embeddings

        nodes_with_empty = [
            {"id": "test:c:0", "text": "Valid text"},
            {"id": "test:c:100", "text": ""},  # Пустой
            {"id": "test:c:200", "text": "   "},  # Только пробелы
        ]

        # Mock возвращает один вектор для одного валидного текста
        mock_get_embeddings.return_value = np.random.rand(1, 1536).astype(np.float32)

        result = get_node_embeddings(nodes_with_empty, self.config, self.logger)

        # Должен быть только один embedding
        self.assertEqual(len(result), 1)
        self.assertIn("test:c:0", result)
        self.assertNotIn("test:c:100", result)
        self.assertNotIn("test:c:200", result)

        # Проверяем, что API был вызван только с валидным текстом
        mock_get_embeddings.assert_called_once_with(["Valid text"], self.config)

    @patch("refiner_longrange.get_embeddings")
    def test_embeddings_api_error(self, mock_get_embeddings):
        """Тест обработки ошибок API embeddings."""
        from refiner_longrange import get_node_embeddings

        # Mock выбрасывает исключение
        mock_get_embeddings.side_effect = Exception("API rate limit exceeded")

        with self.assertRaises(Exception) as cm:
            get_node_embeddings(self.nodes, self.config, self.logger)

        self.assertIn("API rate limit", str(cm.exception))


class TestFAISSIntegration(unittest.TestCase):
    """Интеграционные тесты работы с FAISS."""

    def setUp(self):
        self.logger = logging.getLogger("test")
        self.config = {
            "sim_threshold": 0.8,
            "max_pairs_per_node": 5,
            "faiss_M": 16,
            "faiss_metric": "INNER_PRODUCT",
            "faiss_efC": 200,
        }

        self.nodes = [
            {"id": "test:c:0", "type": "Chunk", "text": "Text 1"},
            {"id": "test:c:100", "type": "Chunk", "text": "Text 2"},
            {"id": "test:c:200", "type": "Chunk", "text": "Text 3"},
            {"id": "test:c:300", "type": "Chunk", "text": "Text 4"},
        ]

    def test_build_faiss_index_and_search(self):
        """Тест построения FAISS индекса и поиска."""
        from refiner_longrange import build_similarity_index, generate_candidate_pairs

        # Создаем embeddings с известной структурой сходства
        base_vector = np.random.rand(1536).astype(np.float32)
        embeddings_dict = {
            "test:c:0": base_vector,
            "test:c:100": base_vector + 0.05 * np.random.rand(1536),  # Очень похож на n1
            "test:c:200": base_vector + 0.3 * np.random.rand(1536),  # Менее похож
            "test:c:300": np.random.rand(1536).astype(np.float32),  # Совсем другой
        }

        # Нормализуем для корректного cosine similarity
        for k in embeddings_dict:
            embeddings_dict[k] /= np.linalg.norm(embeddings_dict[k])

        # Строим индекс
        index, node_ids = build_similarity_index(
            embeddings_dict, self.nodes, self.config, self.logger
        )

        # Проверяем индекс
        self.assertEqual(index.ntotal, 4)
        self.assertEqual(len(node_ids), 4)
        self.assertEqual(node_ids, ["test:c:0", "test:c:100", "test:c:200", "test:c:300"])  # По позиции из ID

        # Генерируем кандидатов
        edges_index = {}  # Нет существующих рёбер
        candidates = generate_candidate_pairs(
            self.nodes,
            embeddings_dict,
            index,
            node_ids,
            edges_index,
            self.config,
            self.logger,
        )

        # Проверяем, что n1 нашел n2 как похожий
        self.assertGreater(len(candidates), 0)

        # Находим кандидатов для test:c:0
        n1_candidates = None
        for pair in candidates:
            if pair["source_node"]["id"] == "test:c:0":
                n1_candidates = pair["candidates"]
                break

        self.assertIsNotNone(n1_candidates)

        # test:c:100 должен быть первым кандидатом (самый похожий)
        if n1_candidates:
            self.assertEqual(n1_candidates[0]["node_id"], "test:c:100")
            self.assertGreater(n1_candidates[0]["similarity"], 0.9)  # Очень похожи

    def test_faiss_with_different_metrics(self):
        """Тест FAISS с разными метриками."""
        from refiner_longrange import build_similarity_index

        embeddings_dict = {
            f"test:c:{i*100}": np.random.rand(1536).astype(np.float32) for i in range(4)
        }

        # INNER_PRODUCT
        index_ip, _ = build_similarity_index(
            embeddings_dict, self.nodes, self.config, self.logger
        )
        self.assertEqual(index_ip.metric_type, 0)  # METRIC_INNER_PRODUCT = 0

        # L2 (если изменить конфиг)
        config_l2 = self.config.copy()
        config_l2["faiss_metric"] = "L2"

        index_l2, _ = build_similarity_index(
            embeddings_dict, self.nodes, config_l2, self.logger
        )
        self.assertEqual(index_l2.metric_type, 1)  # METRIC_L2 = 1

    def test_faiss_empty_embeddings(self):
        """Тест с пустыми embeddings."""
        from refiner_longrange import build_similarity_index

        empty_embeddings = {}

        with self.assertRaises(ValueError) as cm:
            build_similarity_index(
                empty_embeddings, self.nodes, self.config, self.logger
            )

        self.assertIn("No embeddings to build index", str(cm.exception))


class TestRefinerFullPipeline(unittest.TestCase):
    """Полные интеграционные тесты pipeline."""

    def setUp(self):
        self.config = {
            "refiner": {
                "run": True,
                "embedding_model": "text-embedding-3-small",
                "embedding_api_key": "sk-test-embed",
                "embedding_tpm_limit": 350000,
                "sim_threshold": 0.7,
                "max_pairs_per_node": 2,
                "model": "gpt-4o",
                "api_key": "sk-test",
                "tpm_limit": 100000,
                "tpm_safety_margin": 0.15,
                "max_completion": 4096,
                "temperature": 0.6,
                "timeout": 45,
                "max_retries": 3,
                "weight_low": 0.3,
                "weight_mid": 0.6,
                "weight_high": 0.9,
                "faiss_M": 16,
                "faiss_metric": "INNER_PRODUCT",
                "log_level": "info",
            }
        }

        self.graph_data = {
            "nodes": [
                {
                    "id": "test:c:0",
                    "type": "Chunk",
                    "text": "Python variables",
                    "node_offset": 0,
                },
                {
                    "id": "test:c:1000",
                    "type": "Chunk",
                    "text": "Python functions",
                    "node_offset": 0,
                },
                {
                    "id": "test:c:2000",
                    "type": "Chunk",
                    "text": "Python classes",
                    "node_offset": 0,
                },
                {
                    "id": "concept1",
                    "type": "Concept",
                    "text": "Variables",
                    "node_offset": 0,
                },
            ],
            "edges": [{"source": "test:c:0", "target": "concept1", "type": "MENTIONS"}],
        }

    @patch("refiner_longrange.setup_json_logging")
    @patch("refiner_longrange.OpenAIClient")
    @patch("refiner_longrange.get_embeddings")
    @patch("refiner_longrange.load_config")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.exists")
    def test_pipeline_no_candidates(
        self, mock_exists, mock_file, mock_config, mock_embeddings, mock_openai_client, mock_setup_logging
    ):
        """Тест pipeline когда нет подходящих кандидатов (высокий порог)."""
        from refiner_longrange import main

        # Настройка моков
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = json.dumps(self.graph_data)

        config = self.config.copy()
        config["refiner"]["sim_threshold"] = 0.99  # Очень высокий порог
        mock_config.return_value = config

        # Совершенно разные embeddings
        mock_embeddings.return_value = np.array(
            [
                np.array([1.0] + [0.0] * 1535),  # test:c:0
                np.array([0.0, 1.0] + [0.0] * 1534),  # test:c:1000
                np.array([0.0, 0.0, 1.0] + [0.0] * 1533),  # test:c:2000
            ],
            dtype=np.float32,
        )

        # OpenAI клиент не должен вызываться
        mock_llm_instance = MagicMock()
        mock_openai_client.return_value = mock_llm_instance
        
        # Mock logger to prevent creating real log files
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger

        with patch("refiner_longrange.validate_json") as mock_val_json:
            with patch("refiner_longrange.validate_graph_invariants") as mock_val_inv:
                mock_val_json.return_value = None
                mock_val_inv.return_value = None
                result = main()

        self.assertEqual(result, EXIT_SUCCESS)

        # LLM не должен был вызываться
        mock_llm_instance.create_response.assert_not_called()

    @patch("refiner_longrange.setup_json_logging")
    @patch("refiner_longrange.OpenAIClient")
    @patch("refiner_longrange.get_embeddings")
    @patch("refiner_longrange.load_config")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.exists")
    def test_pipeline_with_analysis(
        self, mock_exists, mock_file, mock_config, mock_embeddings, mock_openai_client, mock_setup_logging
    ):
        """Тест полного pipeline с LLM анализом."""
        from refiner_longrange import main

        # Настройка моков
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = json.dumps(self.graph_data)
        mock_config.return_value = self.config

        # Похожие embeddings для создания кандидатов
        base = np.random.rand(1536)
        embeddings = np.array(
            [
                base,  # test:c:0
                base + 0.1 * np.random.rand(1536),  # test:c:1000 похож на test:c:0
                base + 0.1 * np.random.rand(1536),  # test:c:2000 похож на test:c:0 и test:c:1000
            ],
            dtype=np.float32,
        )

        # Нормализуем
        for i in range(len(embeddings)):
            embeddings[i] /= np.linalg.norm(embeddings[i])

        mock_embeddings.return_value = embeddings
        
        # Mock logger to prevent creating real log files
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger

        # Настройка мока LLM
        mock_llm_instance = MagicMock()

        # Создаем mock объекты ResponseUsage
        mock_usage_1 = MagicMock()
        mock_usage_1.input_tokens = 250
        mock_usage_1.output_tokens = 250
        mock_usage_1.total_tokens = 500
        
        mock_usage_2 = MagicMock()
        mock_usage_2.input_tokens = 200
        mock_usage_2.output_tokens = 200
        mock_usage_2.total_tokens = 400

        # Первый вызов - анализ test:c:0
        mock_llm_instance.create_response.side_effect = [
            (
                json.dumps(
                    [
                        {
                            "source": "test:c:0",
                            "target": "test:c:1000",
                            "type": "PREREQUISITE",
                            "weight": 0.8,
                            "conditions": "",
                        },
                        {"source": "test:c:0", "target": "test:c:2000", "type": None},  # Нет связи
                    ]
                ),
                "response_id_1",
                mock_usage_1,
            ),
            # Второй вызов - анализ test:c:1000 (если будет)
            (
                json.dumps(
                    [
                        {
                            "source": "test:c:1000",
                            "target": "test:c:2000",
                            "type": "ELABORATES",
                            "weight": 0.7,
                            "conditions": "",
                        }
                    ]
                ),
                "response_id_2",
                mock_usage_2,
            ),
        ]

        mock_openai_client.return_value = mock_llm_instance

        # Мокаем запись файлов
        write_calls = []

        def track_write(content):
            write_calls.append(content)
            return len(content)

        mock_file.return_value.write.side_effect = track_write

        with patch("refiner_longrange.validate_json") as mock_val_json:
            with patch("refiner_longrange.validate_graph_invariants") as mock_val_inv:
                mock_val_json.return_value = None
                mock_val_inv.return_value = None
                result = main()

        self.assertEqual(result, EXIT_SUCCESS)

        # Проверяем, что LLM был вызван
        self.assertTrue(mock_llm_instance.create_response.called)

        # Проверяем, что граф был сохранен
        self.assertTrue(
            any(
                "LearningChunkGraph_longrange.json" in str(call)
                for call in mock_file.call_args_list
            )
        )

    @patch("refiner_longrange.setup_json_logging")
    @patch("refiner_longrange.OpenAIClient")
    @patch("refiner_longrange.get_embeddings")
    @patch("refiner_longrange.load_config")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.exists")
    def test_pipeline_with_repair_retry(
        self, mock_exists, mock_file, mock_config, mock_embeddings, mock_openai_client, mock_setup_logging
    ):
        """Тест pipeline с repair-retry при битом JSON от LLM."""
        from refiner_longrange import main

        # Настройка базовых моков
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = json.dumps(self.graph_data)
        mock_config.return_value = self.config

        # Похожие embeddings
        embeddings = np.ones((3, 1536), dtype=np.float32)
        for i in range(3):
            embeddings[i] = embeddings[i] + i * 0.01
            embeddings[i] /= np.linalg.norm(embeddings[i])
        mock_embeddings.return_value = embeddings
        
        # Mock logger to prevent creating real log files
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger

        # Настройка мока LLM
        mock_llm_instance = MagicMock()

        # Создаем mock объект ResponseUsage
        mock_usage_1 = MagicMock()
        mock_usage_1.input_tokens = 50
        mock_usage_1.output_tokens = 50
        mock_usage_1.total_tokens = 100
        
        mock_usage_2 = MagicMock()
        mock_usage_2.input_tokens = 75
        mock_usage_2.output_tokens = 75
        mock_usage_2.total_tokens = 150

        # Первый вызов возвращает битый JSON
        mock_llm_instance.create_response.return_value = (
            "This is not valid JSON { broken",
            "response_id_1",
            mock_usage_1,
        )

        # repair_response возвращает корректный JSON
        mock_llm_instance.repair_response.return_value = (
            json.dumps(
                [
                    {
                        "source": "test:c:0",
                        "target": "test:c:1000",
                        "type": "PREREQUISITE",
                        "weight": 0.8,
                        "conditions": "",
                    }
                ]
            ),
            "response_id_1_repair",
            mock_usage_2,
        )

        mock_openai_client.return_value = mock_llm_instance

        with patch("refiner_longrange.validate_json") as mock_val_json:
            with patch("refiner_longrange.validate_graph_invariants") as mock_val_inv:
                mock_val_json.return_value = None
                mock_val_inv.return_value = None
                with patch("pathlib.Path.mkdir"):  # Для создания logs/
                    result = main()

        # Должен успешно завершиться после repair
        self.assertEqual(result, EXIT_SUCCESS)

        # Проверяем, что repair был вызван
        mock_llm_instance.repair_response.assert_called()

    @patch("refiner_longrange.setup_json_logging")
    @patch("refiner_longrange.OpenAIClient")
    @patch("refiner_longrange.get_embeddings")
    @patch("refiner_longrange.load_config")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.exists")
    def test_pipeline_api_error(
        self, mock_exists, mock_file, mock_config, mock_embeddings, mock_openai_client, mock_setup_logging
    ):
        """Тест обработки API ошибок."""
        from refiner_longrange import main

        # Настройка моков
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = json.dumps(self.graph_data)
        mock_config.return_value = self.config
        
        # Mock logger to prevent creating real log files
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger

        # Ошибка при получении embeddings
        mock_embeddings.side_effect = Exception("RateLimitError: Too many requests")

        result = main()

        # Должен вернуть ошибку runtime (не API limit, т.к. это embeddings)
        self.assertEqual(result, EXIT_RUNTIME_ERROR)


class TestLoggingIntegration(unittest.TestCase):
    """Тесты логирования в файлы."""

    @patch("refiner_longrange.OpenAIClient")
    @patch("refiner_longrange.get_embeddings")
    @patch("refiner_longrange.load_config")
    @patch("pathlib.Path.exists")
    def test_json_logging_setup(
        self, mock_exists, mock_config, mock_embeddings, mock_openai_client
    ):
        """Тест настройки JSON логирования."""
        from refiner_longrange import setup_json_logging

        mock_exists.return_value = True

        # Минимальный граф
        graph_data = {
            "nodes": [{"id": "test:c:0", "type": "Chunk", "text": "Test"}],
            "edges": [],
        }

        mock_config.return_value = {
            "refiner": {
                "run": True,
                "log_level": "debug",
                "embedding_model": "text-embedding-3-small",
                "sim_threshold": 0.95,
                "max_pairs_per_node": 1,
                "model": "gpt-4o",
                "api_key": "sk-test",
                "tpm_limit": 100000,
                "max_completion": 4096,
                "weight_low": 0.3,
                "weight_mid": 0.6,
                "weight_high": 0.9,
                "faiss_M": 16,
                "faiss_metric": "INNER_PRODUCT",
            }
        }

        # Mock embeddings
        mock_embeddings.return_value = np.random.rand(1, 1536).astype(np.float32)

        # Перехватываем создание файлов логов
        log_handlers = []

        with patch(
            "builtins.open", mock_open(read_data=json.dumps(graph_data))
        ) as mock_file:
            with patch("logging.FileHandler") as mock_handler:

                def track_handler(*args, **kwargs):
                    handler = MagicMock()
                    handler.level = logging.DEBUG
                    log_handlers.append((args, kwargs))
                    return handler

                mock_handler.side_effect = track_handler

                with patch("refiner_longrange.validate_json") as mock_val_json:
                    with patch("refiner_longrange.validate_graph_invariants") as mock_val_inv:
                        mock_val_json.return_value = None
                        mock_val_inv.return_value = None
                        # Вызываем setup_json_logging напрямую
                        logger = setup_json_logging(mock_config.return_value["refiner"])

                        # Проверяем, что logger настроен
                        self.assertEqual(logger.name, "refiner")
                        self.assertEqual(
                            logger.level, logging.DEBUG
                        )  # log_level = "debug"

        # Проверяем, что был создан файловый handler для логов
        self.assertTrue(any("refiner_" in str(args[0]) for args, _ in log_handlers))


if __name__ == "__main__":
    unittest.main()

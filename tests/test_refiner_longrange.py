"""
Тесты для refiner.py
"""

import logging
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import mock_open, patch

import numpy as np

# Добавляем путь к src для импортов
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))



class TestRefinerConfig(unittest.TestCase):
    """Тесты валидации конфигурации."""

    def test_valid_config(self):
        """Тест валидной конфигурации."""
        from refiner_longrange import validate_refiner_longrange_config

        config = {
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

        # Не должно быть исключений
        validate_refiner_longrange_config(config)

    def test_missing_required_param(self):
        """Тест отсутствующего обязательного параметра."""
        from refiner_longrange import validate_refiner_longrange_config

        config = {
            "sim_threshold": 0.8,
            # Отсутствует embedding_model
        }

        with self.assertRaises(ValueError) as cm:
            validate_refiner_longrange_config(config)

        self.assertIn("Missing required parameter", str(cm.exception))

    def test_empty_api_key(self):
        """Тест пустого API ключа."""
        from refiner_longrange import validate_refiner_longrange_config

        config = {
            "embedding_model": "text-embedding-3-small",
            "sim_threshold": 0.8,
            "max_pairs_per_node": 20,
            "model": "gpt-4o",
            "api_key": "",  # Пустой
            "tpm_limit": 100000,
            "max_completion": 4096,
            "weight_low": 0.3,
            "weight_mid": 0.6,
            "weight_high": 0.9,
            "faiss_M": 32,
            "faiss_metric": "INNER_PRODUCT",
        }

        with self.assertRaises(ValueError) as cm:
            validate_refiner_longrange_config(config)

        self.assertIn("api_key cannot be empty", str(cm.exception))

    def test_invalid_threshold(self):
        """Тест некорректного порога similarity."""
        from refiner_longrange import validate_refiner_longrange_config

        config = {
            "embedding_model": "text-embedding-3-small",
            "sim_threshold": 1.5,  # > 1
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

        with self.assertRaises(ValueError) as cm:
            validate_refiner_longrange_config(config)

        self.assertIn("sim_threshold must be in [0,1]", str(cm.exception))

    def test_invalid_weights_order(self):
        """Тест некорректного порядка весов."""
        from refiner_longrange import validate_refiner_longrange_config

        config = {
            "embedding_model": "text-embedding-3-small",
            "sim_threshold": 0.8,
            "max_pairs_per_node": 20,
            "model": "gpt-4o",
            "api_key": "sk-test",
            "tpm_limit": 100000,
            "max_completion": 4096,
            "weight_low": 0.8,  # Больше чем mid
            "weight_mid": 0.6,
            "weight_high": 0.9,
            "faiss_M": 32,
            "faiss_metric": "INNER_PRODUCT",
        }

        with self.assertRaises(ValueError) as cm:
            validate_refiner_longrange_config(config)

        self.assertIn("Weights must satisfy", str(cm.exception))

    def test_reasoning_model_params(self):
        """Тест параметров для reasoning моделей."""
        from refiner_longrange import validate_refiner_longrange_config

        # Корректные параметры для o-модели
        config = {
            "embedding_model": "text-embedding-3-small",
            "sim_threshold": 0.8,
            "max_pairs_per_node": 20,
            "model": "o4-mini",
            "api_key": "sk-test",
            "tpm_limit": 100000,
            "max_completion": 4096,
            "weight_low": 0.3,
            "weight_mid": 0.6,
            "weight_high": 0.9,
            "faiss_M": 32,
            "faiss_metric": "INNER_PRODUCT",
            "reasoning_effort": "medium",
            "reasoning_summary": "auto",
        }

        validate_refiner_longrange_config(config)

        # Некорректный reasoning_effort
        config["reasoning_effort"] = "super-high"

        with self.assertRaises(ValueError) as cm:
            validate_refiner_longrange_config(config)

        self.assertIn("reasoning_effort must be", str(cm.exception))


class TestExtractGlobalPosition(unittest.TestCase):
    """Тесты функции extract_global_position."""

    def test_chunk_id(self):
        """Тест извлечения позиции из Chunk ID."""
        from refiner_longrange import extract_global_position
        
        # Test Chunk ID format
        self.assertEqual(extract_global_position("handbook:c:220"), 220)
        self.assertEqual(extract_global_position("chapter01:c:0"), 0)
        self.assertEqual(extract_global_position("text:c:99999"), 99999)
    
    def test_assessment_id(self):
        """Тест извлечения позиции из Assessment ID."""
        from refiner_longrange import extract_global_position
        
        # Test Assessment ID format  
        self.assertEqual(extract_global_position("handbook:q:500:1"), 500)
        self.assertEqual(extract_global_position("chapter02:q:1000:0"), 1000)
        self.assertEqual(extract_global_position("test:q:42:5"), 42)
    
    def test_invalid_id(self):
        """Тест обработки невалидных ID."""
        from refiner_longrange import extract_global_position
        
        # Test invalid IDs
        with self.assertRaises(ValueError) as cm:
            extract_global_position("invalid_id")
        self.assertIn("Unexpected node ID format", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            extract_global_position("handbook:x:100")
        self.assertIn("Unexpected node ID format", str(cm.exception))
        
        # Test invalid position
        with self.assertRaises(ValueError) as cm:
            extract_global_position("handbook:c:abc")
        self.assertIn("Cannot parse position", str(cm.exception))


class TestGraphProcessing(unittest.TestCase):
    """Тесты обработки графа."""

    def setUp(self):
        self.sample_graph = {
            "nodes": [
                {"id": "node1", "type": "Chunk", "text": "Test chunk 1"},
                {"id": "node2", "type": "Chunk", "text": "Test chunk 2"},
                {"id": "node3", "type": "Concept", "text": "Test concept"},
                {"id": "node4", "type": "Assessment", "text": "Test question"},
            ],
            "edges": [
                {"source": "node1", "target": "node2", "type": "PREREQUISITE"},
                {"source": "node2", "target": "node3", "type": "MENTIONS"},
            ],
        }

    def test_extract_target_nodes(self):
        """Тест извлечения целевых узлов."""
        from refiner_longrange import extract_target_nodes

        target_nodes = extract_target_nodes(self.sample_graph)

        # Должны быть только Chunk и Assessment
        self.assertEqual(len(target_nodes), 3)

        node_types = {node["type"] for node in target_nodes}
        self.assertEqual(node_types, {"Chunk", "Assessment"})

        # Concept не должен попасть
        node_ids = {node["id"] for node in target_nodes}
        self.assertNotIn("node3", node_ids)

    def test_build_edges_index(self):
        """Тест построения индекса рёбер."""
        from refiner_longrange import build_edges_index

        edges_index = build_edges_index(self.sample_graph)

        # Проверяем структуру
        self.assertIn("node1", edges_index)
        self.assertIn("node2", edges_index["node1"])
        self.assertEqual(len(edges_index["node1"]["node2"]), 1)
        self.assertEqual(edges_index["node1"]["node2"][0]["type"], "PREREQUISITE")

        self.assertIn("node2", edges_index)
        self.assertIn("node3", edges_index["node2"])

    def test_empty_target_nodes(self):
        """Тест с пустым списком целевых узлов."""
        from refiner_longrange import extract_target_nodes

        empty_graph = {
            "nodes": [{"id": "node1", "type": "Concept", "text": "Only concepts"}],
            "edges": [],
        }

        target_nodes = extract_target_nodes(empty_graph)
        self.assertEqual(len(target_nodes), 0)


class TestCandidateGeneration(unittest.TestCase):
    """Тесты генерации кандидатов."""

    def setUp(self):
        self.logger = logging.getLogger("test")
        self.config = {
            "embedding_model": "text-embedding-3-small",
            "api_key": "test-key",
            "sim_threshold": 0.8,
            "max_pairs_per_node": 5,
            "faiss_M": 16,
            "faiss_metric": "INNER_PRODUCT",
        }

        # Тестовые узлы
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

    def test_build_similarity_index(self):
        """Тест построения FAISS индекса."""
        from refiner_longrange import build_similarity_index

        # Создаем тестовые embeddings
        embeddings_dict = {
            "test:c:0": np.random.rand(1536).astype(np.float32),
            "test:c:100": np.random.rand(1536).astype(np.float32),
            "test:q:200:0": np.random.rand(1536).astype(np.float32),
        }

        index, node_ids = build_similarity_index(
            embeddings_dict, self.nodes, self.config, self.logger
        )

        # Проверяем индекс
        self.assertEqual(index.ntotal, 3)
        self.assertEqual(len(node_ids), 3)

        # Проверяем порядок (по позиции извлеченной из ID)
        self.assertEqual(node_ids, ["test:c:0", "test:c:100", "test:q:200:0"])

    def test_generate_candidate_pairs(self):
        """Тест генерации пар кандидатов."""
        from refiner_longrange import build_similarity_index, generate_candidate_pairs

        # Создаем похожие embeddings (высокая корреляция между первым и вторым узлом)
        base_vector = np.random.rand(1536).astype(np.float32)
        embeddings_dict = {
            "test:c:0": base_vector,
            "test:c:100": base_vector + np.random.rand(1536) * 0.1,  # Похожий
            "test:q:200:0": np.random.rand(1536).astype(np.float32),  # Случайный
        }

        # Нормализуем для корректного cosine similarity
        for k in embeddings_dict:
            embeddings_dict[k] /= np.linalg.norm(embeddings_dict[k])

        index, node_ids = build_similarity_index(
            embeddings_dict, self.nodes, self.config, self.logger
        )

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

        # Должны найти хотя бы одну пару
        self.assertGreater(len(candidates), 0)

        # Проверяем структуру
        first_pair = candidates[0]
        self.assertIn("source_node", first_pair)
        self.assertIn("candidates", first_pair)

        # Проверяем, что local_start соблюдается
        source_node = next(
            n for n in self.nodes if n["id"] == first_pair["source_node"]["id"]
        )
        for candidate in first_pair["candidates"]:
            target_node = next(n for n in self.nodes if n["id"] == candidate["node_id"])
            from refiner_longrange import extract_global_position
            self.assertLess(
                extract_global_position(source_node["id"]),
                extract_global_position(target_node["id"])
            )

    def test_generate_candidate_pairs_with_existing_edges(self):
        """Тест с существующими рёбрами."""
        from refiner_longrange import build_similarity_index, generate_candidate_pairs

        # Похожие embeddings
        embeddings_dict = {
            "test:c:0": np.ones(1536, dtype=np.float32),
            "test:c:100": np.ones(1536, dtype=np.float32) * 0.99,
        }

        for k in embeddings_dict:
            embeddings_dict[k] /= np.linalg.norm(embeddings_dict[k])

        nodes = self.nodes[:2]  # Только два узла
        index, node_ids = build_similarity_index(
            embeddings_dict, nodes, self.config, self.logger
        )

        # Добавляем существующее ребро
        edges_index = {
            "test:c:0": {
                "test:c:100": [
                    {"source": "test:c:0", "target": "test:c:100", "type": "PREREQUISITE"}
                ]
            }
        }

        candidates = generate_candidate_pairs(
            nodes,
            embeddings_dict,
            index,
            node_ids,
            edges_index,
            self.config,
            self.logger,
        )

        # Проверяем, что существующее ребро включено
        if candidates:
            first_candidate = candidates[0]["candidates"][0]
            self.assertEqual(len(first_candidate["existing_edges"]), 1)
            self.assertEqual(
                first_candidate["existing_edges"][0]["type"], "PREREQUISITE"
            )


class TestPromptLoading(unittest.TestCase):
    """Тесты загрузки и подготовки промпта."""

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="Test prompt with {weight_low} and {weight_mid} and {weight_high}",
    )
    @patch("pathlib.Path.exists", return_value=True)
    def test_load_refiner_prompt(self, mock_exists, mock_file):
        """Тест загрузки и подстановки весов в промпт."""
        from refiner_longrange import load_refiner_longrange_prompt

        config = {"weight_low": 0.3, "weight_mid": 0.6, "weight_high": 0.9}

        result = load_refiner_longrange_prompt(config)

        # Проверяем подстановку
        self.assertIn("0.3", result)
        self.assertIn("0.6", result)
        self.assertIn("0.9", result)
        self.assertNotIn("{weight_low}", result)
        self.assertNotIn("{weight_mid}", result)
        self.assertNotIn("{weight_high}", result)

    @patch("pathlib.Path.exists", return_value=False)
    def test_load_refiner_prompt_not_found(self, mock_exists):
        """Тест с отсутствующим файлом промпта."""
        from refiner_longrange import load_refiner_longrange_prompt

        config = {"weight_low": 0.3, "weight_mid": 0.6, "weight_high": 0.9}

        with self.assertRaises(FileNotFoundError):
            load_refiner_longrange_prompt(config)


class TestLLMEdgeValidation(unittest.TestCase):
    """Тесты валидации рёбер от LLM."""

    def setUp(self):
        self.logger = logging.getLogger("test")
        self.graph = {
            "nodes": [
                {"id": "n1", "type": "Chunk"},
                {"id": "n2", "type": "Chunk"},
                {"id": "n3", "type": "Chunk"},
            ]
        }
        self.candidates = [{"node_id": "n2"}, {"node_id": "n3"}]

    def test_validate_valid_edges(self):
        """Тест валидации корректных рёбер."""
        from refiner_longrange import validate_llm_edges

        edges_response = [
            {
                "source": "n1",
                "target": "n2",
                "type": "PREREQUISITE",
                "weight": 0.8,
                "conditions": "added_by=refiner_v1",
            },
            {"source": "n1", "target": "n3", "type": "ELABORATES", "weight": 0.6},
        ]

        result = validate_llm_edges(
            edges_response, "n1", self.candidates, self.graph, self.logger
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["type"], "PREREQUISITE")
        self.assertEqual(result[1]["type"], "ELABORATES")

    def test_skip_null_type(self):
        """Тест пропуска записей с type: null."""
        from refiner_longrange import validate_llm_edges

        edges_response = [
            {"source": "n1", "target": "n2", "type": None},
            {"source": "n1", "target": "n3", "type": "PARALLEL", "weight": 0.5},
        ]

        result = validate_llm_edges(
            edges_response, "n1", self.candidates, self.graph, self.logger
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "PARALLEL")

    def test_invalid_source(self):
        """Тест с некорректным source."""
        from refiner_longrange import validate_llm_edges

        edges_response = [
            {"source": "wrong_id", "target": "n2", "type": "PREREQUISITE"}
        ]

        result = validate_llm_edges(
            edges_response, "n1", self.candidates, self.graph, self.logger
        )

        self.assertEqual(len(result), 0)

    def test_target_not_in_candidates(self):
        """Тест с target не из кандидатов."""
        from refiner_longrange import validate_llm_edges

        edges_response = [
            {
                "source": "n1",
                "target": "n4",
                "type": "PREREQUISITE",
            }  # n4 не в кандидатах
        ]

        result = validate_llm_edges(
            edges_response, "n1", self.candidates, self.graph, self.logger
        )

        self.assertEqual(len(result), 0)

    def test_prerequisite_self_loop(self):
        """Тест PREREQUISITE self-loop."""
        from refiner_longrange import validate_llm_edges

        edges_response = [{"source": "n1", "target": "n1", "type": "PREREQUISITE"}]

        # Добавляем n1 в кандидаты для теста
        candidates_with_self = self.candidates + [{"node_id": "n1"}]

        result = validate_llm_edges(
            edges_response, "n1", candidates_with_self, self.graph, self.logger
        )

        self.assertEqual(len(result), 0)  # Должен быть отфильтрован

    def test_weight_validation(self):
        """Тест валидации весов."""
        from refiner_longrange import validate_llm_edges

        edges_response = [
            {
                "source": "n1",
                "target": "n2",
                "type": "PREREQUISITE",
                "weight": 1.5,
            },  # > 1
            {
                "source": "n1",
                "target": "n3",
                "type": "ELABORATES",
                "weight": "invalid",
            },  # не число
        ]

        result = validate_llm_edges(
            edges_response, "n1", self.candidates, self.graph, self.logger
        )

        # Оба должны быть добавлены с исправленными весами
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["weight"], 0.5)  # Исправлено на default
        self.assertEqual(result[1]["weight"], 0.5)  # Исправлено на default


class TestGraphUpdate(unittest.TestCase):
    """Тесты для функции update_graph_with_new_edges."""

    def setUp(self):
        from refiner_longrange import update_graph_with_new_edges

        self.update_graph_with_new_edges = update_graph_with_new_edges
        self.logger = logging.getLogger("test")
        self.base_graph = {
            "nodes": [
                {"id": "n1", "type": "Chunk", "text": "Text 1"},
                {"id": "n2", "type": "Chunk", "text": "Text 2"},
                {"id": "n3", "type": "Chunk", "text": "Text 3"},
            ],
            "edges": [
                {"source": "n1", "target": "n2", "type": "PREREQUISITE", "weight": 0.5},
                {"source": "n2", "target": "n3", "type": "ELABORATES", "weight": 0.7},
            ],
        }

    def test_add_new_edge(self):
        """Тест сценария 1: Добавление нового ребра."""
        graph = deepcopy(self.base_graph)

        new_edges = [
            {"source": "n1", "target": "n3", "type": "EXAMPLE_OF", "weight": 0.8}
        ]

        stats = self.update_graph_with_new_edges(graph, new_edges, self.logger)

        # Проверяем статистику
        self.assertEqual(stats["added"], 1)
        self.assertEqual(stats["updated"], 0)
        self.assertEqual(stats["replaced"], 0)

        # Проверяем, что ребро добавлено
        self.assertEqual(len(graph["edges"]), 3)
        new_edge = graph["edges"][-1]
        self.assertEqual(new_edge["source"], "n1")
        self.assertEqual(new_edge["target"], "n3")
        self.assertEqual(new_edge["type"], "EXAMPLE_OF")
        self.assertEqual(new_edge["conditions"], "added_by=refiner_longrange_v1")

    def test_update_existing_edge_weight(self):
        """Тест сценария 2: Обновление веса существующего ребра."""
        graph = deepcopy(self.base_graph)

        new_edges = [
            {"source": "n1", "target": "n2", "type": "PREREQUISITE", "weight": 0.9}
        ]

        stats = self.update_graph_with_new_edges(graph, new_edges, self.logger)

        # Проверяем статистику
        self.assertEqual(stats["added"], 0)
        self.assertEqual(stats["updated"], 1)
        self.assertEqual(stats["replaced"], 0)

        # Проверяем, что вес обновлен
        self.assertEqual(len(graph["edges"]), 2)
        updated_edge = graph["edges"][0]
        self.assertEqual(updated_edge["weight"], 0.9)

    def test_replace_edge_type(self):
        """Тест сценария 3: Замена типа ребра."""
        graph = deepcopy(self.base_graph)

        new_edges = [
            {"source": "n1", "target": "n2", "type": "PARALLEL", "weight": 0.8}
        ]

        stats = self.update_graph_with_new_edges(graph, new_edges, self.logger)

        # Проверяем статистику
        self.assertEqual(stats["added"], 0)
        self.assertEqual(stats["updated"], 0)
        self.assertEqual(stats["replaced"], 1)

        # Проверяем, что тип заменен
        self.assertEqual(len(graph["edges"]), 2)
        # Находим ребро n1->n2
        n1_n2_edge = None
        for edge in graph["edges"]:
            if edge["source"] == "n1" and edge["target"] == "n2":
                n1_n2_edge = edge
                break

        self.assertIsNotNone(n1_n2_edge)
        self.assertEqual(n1_n2_edge["type"], "PARALLEL")
        self.assertEqual(n1_n2_edge["conditions"], "fixed_by=refiner_longrange_v1")

    def test_no_replace_lower_weight(self):
        """Тест: не заменяем ребро если новый вес меньше."""
        graph = deepcopy(self.base_graph)

        new_edges = [
            {"source": "n1", "target": "n2", "type": "PARALLEL", "weight": 0.3}
        ]

        stats = self.update_graph_with_new_edges(graph, new_edges, self.logger)

        # Проверяем, что ничего не изменилось
        self.assertEqual(stats["added"], 0)
        self.assertEqual(stats["updated"], 0)
        self.assertEqual(stats["replaced"], 0)

        # Проверяем, что тип остался прежним
        self.assertEqual(graph["edges"][0]["type"], "PREREQUISITE")

    def test_remove_prerequisite_self_loops(self):
        """Тест удаления PREREQUISITE self-loops."""
        graph = deepcopy(self.base_graph)

        # Добавляем self-loop
        graph["edges"].append(
            {"source": "n1", "target": "n1", "type": "PREREQUISITE", "weight": 0.5}
        )

        new_edges = [
            {"source": "n2", "target": "n1", "type": "REFER_BACK", "weight": 0.6}
        ]

        stats = self.update_graph_with_new_edges(graph, new_edges, self.logger)

        # Проверяем статистику
        self.assertEqual(stats["self_loops_removed"], 1)

        # Проверяем, что self-loop удален
        for edge in graph["edges"]:
            if edge["source"] == edge["target"] and edge["type"] == "PREREQUISITE":
                self.fail("PREREQUISITE self-loop was not removed")

    def test_multiple_edges_update(self):
        """Тест обработки нескольких новых рёбер."""
        graph = deepcopy(self.base_graph)

        new_edges = [
            {"source": "n1", "target": "n3", "type": "HINT_FORWARD", "weight": 0.7},
            {"source": "n1", "target": "n2", "type": "PREREQUISITE", "weight": 0.95},
            {"source": "n3", "target": "n1", "type": "REFER_BACK", "weight": 0.6},
        ]

        stats = self.update_graph_with_new_edges(graph, new_edges, self.logger)

        # Проверяем статистику
        self.assertEqual(stats["added"], 2)  # n1->n3 и n3->n1
        self.assertEqual(stats["updated"], 1)  # n1->n2 вес обновлен
        self.assertEqual(stats["replaced"], 0)
        self.assertEqual(stats["total_processed"], 3)

        # Проверяем финальное количество рёбер
        self.assertEqual(len(graph["edges"]), 4)  # 2 было + 2 добавлено


class TestMetaHandling(unittest.TestCase):
    """Тесты для обработки _meta секции."""

    def test_add_refiner_meta(self):
        """Тест добавления метаданных refiner_longrange."""
        from refiner_longrange import add_refiner_meta
        
        graph = {
            "nodes": [],
            "edges": []
        }
        
        config = {
            "model": "gpt-4o",
            "sim_threshold": 0.8,
            "max_pairs_per_node": 20,
            "weight_low": 0.3,
            "weight_mid": 0.6,
            "weight_high": 0.9
        }
        
        stats = {
            "added": 5,
            "updated": 2,
            "replaced": 1,
            "self_loops_removed": 0
        }
        
        add_refiner_meta(graph, config, stats)
        
        # Проверяем, что _meta добавлена
        self.assertIn("_meta", graph)
        self.assertIn("refiner_longrange", graph["_meta"])
        
        # Проверяем содержимое метаданных
        meta = graph["_meta"]["refiner_longrange"]
        self.assertIn("processed_at", meta)
        self.assertIn("config", meta)
        self.assertIn("stats", meta)
        
        # Проверяем конфигурацию
        self.assertEqual(meta["config"]["model"], "gpt-4o")
        self.assertEqual(meta["config"]["sim_threshold"], 0.8)
        self.assertEqual(meta["config"]["weights"]["low"], 0.3)
        
        # Проверяем статистику
        self.assertEqual(meta["stats"]["added"], 5)
        self.assertEqual(meta["stats"]["updated"], 2)
    
    def test_preserve_existing_meta(self):
        """Тест сохранения существующих метаданных."""
        from refiner_longrange import add_refiner_meta
        
        graph = {
            "nodes": [],
            "edges": [],
            "_meta": {
                "existing_tool": {
                    "data": "should be preserved"
                }
            }
        }
        
        config = {
            "model": "gpt-4o",
            "sim_threshold": 0.8,
            "max_pairs_per_node": 20,
            "weight_low": 0.3,
            "weight_mid": 0.6,
            "weight_high": 0.9
        }
        
        stats = {"added": 1, "updated": 0, "replaced": 0, "self_loops_removed": 0}
        
        add_refiner_meta(graph, config, stats)
        
        # Проверяем, что старые метаданные сохранены
        self.assertIn("existing_tool", graph["_meta"])
        self.assertEqual(graph["_meta"]["existing_tool"]["data"], "should be preserved")
        
        # И новые добавлены
        self.assertIn("refiner_longrange", graph["_meta"])


if __name__ == "__main__":
    unittest.main()

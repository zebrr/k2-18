"""
Тесты для dedup.py - утилиты дедупликации узлов графа
"""

from unittest.mock import MagicMock, patch

import faiss
import numpy as np

from src.dedup import (
    UnionFind,
    build_faiss_index,
    cluster_duplicates,
    extract_global_position,
    filter_nodes_for_dedup,
    find_duplicates,
    rewrite_graph,
    save_dedup_map,
    update_metadata,
)


class TestUnionFind:
    """Тесты для Union-Find структуры данных"""

    def test_init(self):
        """Проверка инициализации"""
        uf = UnionFind()
        assert uf.parent == {}
        assert uf.rank == {}

    def test_find_single_element(self):
        """Проверка поиска одиночного элемента"""
        uf = UnionFind()
        assert uf.find("a") == "a"
        assert "a" in uf.parent
        assert uf.parent["a"] == "a"
        assert uf.rank["a"] == 0

    def test_union_two_elements(self):
        """Проверка объединения двух элементов"""
        uf = UnionFind()
        uf.union("a", "b")
        # После объединения должны иметь общий корень
        assert uf.find("a") == uf.find("b")

    def test_union_chain(self):
        """Проверка объединения цепочки элементов"""
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("b", "c")
        uf.union("c", "d")

        # Все должны иметь один корень
        root = uf.find("a")
        assert uf.find("b") == root
        assert uf.find("c") == root
        assert uf.find("d") == root

    def test_get_clusters(self):
        """Проверка получения кластеров"""
        uf = UnionFind()
        # Создаем два кластера
        uf.union("a", "b")
        uf.union("b", "c")
        uf.union("d", "e")

        clusters = uf.get_clusters()

        # Должно быть 2 кластера
        assert len(clusters) == 2

        # Проверяем, что элементы правильно сгруппированы
        cluster_sets = [set(cluster) for cluster in clusters.values()]
        assert {"a", "b", "c"} in cluster_sets
        assert {"d", "e"} in cluster_sets


class TestExtractGlobalPosition:
    """Тесты для извлечения глобальной позиции из ID узла"""

    def test_chunk_id(self):
        """Проверка извлечения позиции из Chunk ID"""
        assert extract_global_position("handbook:c:100") == 100
        assert extract_global_position("handbook:c:5505") == 5505
        assert extract_global_position("textbook:c:0") == 0

    def test_assessment_id(self):
        """Проверка извлечения позиции из Assessment ID"""
        assert extract_global_position("handbook:q:200:1") == 200
        assert extract_global_position("handbook:q:1000:3") == 1000
        assert extract_global_position("textbook:q:0:0") == 0

    def test_invalid_formats(self):
        """Проверка обработки некорректных форматов ID"""
        import pytest

        # Concept ID - не должен обрабатываться этой функцией
        with pytest.raises(ValueError, match="Unexpected node ID format"):
            extract_global_position("handbook:p:sorting")

        # Некорректный формат
        with pytest.raises(ValueError, match="Unexpected node ID format"):
            extract_global_position("invalid")

        # Пустой ID
        with pytest.raises(ValueError, match="Unexpected node ID format"):
            extract_global_position("")

    def test_malformed_position(self):
        """Проверка обработки некорректной позиции"""
        import pytest

        with pytest.raises(ValueError, match="Cannot parse position"):
            extract_global_position("handbook:c:not_a_number")

        with pytest.raises(ValueError, match="Cannot parse position"):
            extract_global_position("handbook:q:abc:1")


class TestFilterNodes:
    """Тесты для фильтрации узлов"""

    def test_filter_empty_list(self):
        """Проверка фильтрации пустого списка"""
        result = filter_nodes_for_dedup([])
        assert result == []

    def test_filter_chunk_and_assessment(self):
        """Проверка фильтрации Chunk и Assessment узлов"""
        nodes = [
            {"id": "1", "type": "Chunk", "text": "Test chunk"},
            {"id": "2", "type": "Assessment", "text": "Test assessment"},
            {"id": "3", "type": "Concept", "text": "Test concept"},
            {"id": "4", "type": "Other", "text": "Other type"},
        ]

        result = filter_nodes_for_dedup(nodes)
        assert len(result) == 2
        assert result[0]["id"] == "1"
        assert result[1]["id"] == "2"

    def test_filter_empty_text(self):
        """Проверка фильтрации узлов с пустым текстом"""
        nodes = [
            {"id": "1", "type": "Chunk", "text": "Valid text"},
            {"id": "2", "type": "Chunk", "text": ""},
            {"id": "3", "type": "Chunk", "text": "  \n  "},
            {"id": "4", "type": "Assessment", "text": None},
        ]

        result = filter_nodes_for_dedup(nodes)
        assert len(result) == 1
        assert result[0]["id"] == "1"


class TestFAISSIndex:
    """Тесты для построения FAISS индекса"""

    def test_build_index(self):
        """Проверка создания индекса"""
        embeddings = np.random.rand(10, 1536).astype(np.float32)
        # Нормализуем векторы
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        config = {"faiss_M": 32, "faiss_efC": 200, "faiss_metric": "INNER_PRODUCT"}

        index = build_faiss_index(embeddings, config)

        assert isinstance(index, faiss.IndexHNSWFlat)
        assert index.ntotal == 10
        assert index.d == 1536

    def test_search_in_index(self):
        """Проверка поиска в индексе"""
        # Создаем тестовые эмбеддинги
        embeddings = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0.9, 0.1, 0],  # Похож на первый
            ],
            dtype=np.float32,
        )

        # Нормализуем
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        config = {"faiss_M": 2, "faiss_efC": 10, "faiss_metric": "INNER_PRODUCT"}

        index = build_faiss_index(embeddings, config)

        # Поиск ближайших для первого вектора
        distances, indices = index.search(embeddings[0:1], 3)

        # Первый результат должен быть сам вектор
        assert indices[0, 0] == 0
        assert distances[0, 0] > 0.99  # Почти 1 (нормализованный)

        # Второй результат должен быть четвертый вектор (похожий)
        assert indices[0, 1] == 3


class TestFindDuplicates:
    """Тесты для поиска дубликатов"""

    def test_no_duplicates(self):
        """Проверка когда нет дубликатов"""
        nodes = [
            {"id": "handbook:c:0", "text": "First unique text", "node_offset": 0},
            {
                "id": "handbook:c:100",
                "text": "Second completely different text",
                "node_offset": 100,
            },
        ]

        # Создаем различные эмбеддинги
        embeddings = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
            ],
            dtype=np.float32,
        )
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        config = {
            "faiss_M": 2,
            "faiss_efC": 10,
            "k_neighbors": 5,
            "sim_threshold": 0.97,
            "len_ratio_min": 0.8,
        }

        index = build_faiss_index(embeddings, config)
        duplicates = find_duplicates(nodes, embeddings, index, config)

        assert len(duplicates) == 0

    def test_find_duplicate_pair(self):
        """Проверка нахождения пары дубликатов"""
        nodes = [
            {"id": "handbook:c:0", "text": "This is a test text about Python", "node_offset": 0},
            {
                "id": "handbook:c:100",
                "text": "This is a test text about Python!",
                "node_offset": 100,
            },
        ]

        # Создаем почти одинаковые эмбеддинги
        embeddings = np.array(
            [
                [1, 0.1, 0],
                [0.99, 0.11, 0],
            ],
            dtype=np.float32,
        )
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        config = {
            "faiss_M": 2,
            "faiss_efC": 10,
            "k_neighbors": 5,
            "sim_threshold": 0.9,
            "len_ratio_min": 0.8,
        }

        index = build_faiss_index(embeddings, config)
        duplicates = find_duplicates(nodes, embeddings, index, config)

        assert len(duplicates) == 1
        master_id, dup_id, sim = duplicates[0]
        assert master_id == "handbook:c:0"  # Меньший global position
        assert dup_id == "handbook:c:100"
        assert sim > 0.9

    def test_length_ratio_filter(self):
        """Проверка фильтрации по length ratio"""
        nodes = [
            {"id": "handbook:c:0", "text": "Short", "node_offset": 0},
            {
                "id": "handbook:c:100",
                "text": "This is a much longer text that should not match",
                "node_offset": 100,
            },
        ]

        # Создаем похожие эмбеддинги
        embeddings = np.array(
            [
                [1, 0, 0],
                [0.98, 0.02, 0],
            ],
            dtype=np.float32,
        )
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        config = {
            "faiss_M": 2,
            "faiss_efC": 10,
            "k_neighbors": 5,
            "sim_threshold": 0.9,
            "len_ratio_min": 0.8,  # 5/49 = 0.1 < 0.8
        }

        index = build_faiss_index(embeddings, config)
        duplicates = find_duplicates(nodes, embeddings, index, config)

        assert len(duplicates) == 0  # Не должны найтись из-за length ratio


class TestClusterDuplicates:
    """Тесты для кластеризации дубликатов"""

    def test_empty_duplicates(self):
        """Проверка с пустым списком дубликатов"""
        dedup_map, num_clusters = cluster_duplicates([])
        assert dedup_map == {}
        assert num_clusters == 0

    def test_single_pair(self):
        """Проверка с одной парой дубликатов"""
        duplicates = [("master1", "dup1", 0.98)]
        dedup_map, num_clusters = cluster_duplicates(duplicates)

        assert len(dedup_map) == 1
        assert dedup_map["dup1"] == "master1"
        assert num_clusters == 1

    def test_transitive_duplicates(self):
        """Проверка транзитивных дубликатов A→B→C"""
        duplicates = [
            ("A", "B", 0.98),
            ("B", "C", 0.97),
        ]
        dedup_map, num_clusters = cluster_duplicates(duplicates)

        assert len(dedup_map) == 2
        assert dedup_map["B"] == "A"
        assert dedup_map["C"] == "A"
        assert num_clusters == 1

    def test_multiple_clusters(self):
        """Проверка нескольких независимых кластеров"""
        duplicates = [
            ("A", "B", 0.98),
            ("C", "D", 0.97),
            ("D", "E", 0.96),
        ]
        dedup_map, num_clusters = cluster_duplicates(duplicates)

        assert len(dedup_map) == 3
        assert dedup_map["B"] == "A"
        assert dedup_map["D"] == "C"
        assert dedup_map["E"] == "C"
        assert num_clusters == 2


class TestRewriteGraph:
    """Тесты для перезаписи графа"""

    def test_empty_dedup_map(self):
        """Проверка когда нет дубликатов"""
        graph = {
            "nodes": [
                {"id": "n1", "text": "Node 1"},
                {"id": "n2", "text": "Node 2"},
            ],
            "edges": [{"source": "n1", "target": "n2", "type": "PREREQUISITE"}],
        }

        new_graph, stats = rewrite_graph(graph, {})

        assert len(new_graph["nodes"]) == 2
        assert len(new_graph["edges"]) == 1
        assert new_graph == graph
        assert stats["nodes_removed_duplicates"] == 0
        assert stats["nodes_removed_empty"] == 0
        assert stats["nodes_removed_total"] == 0
        assert stats["edges_updated"] == 0

    def test_remove_duplicate_nodes(self):
        """Проверка удаления узлов-дубликатов"""
        graph = {
            "nodes": [
                {"id": "n1", "text": "Master node"},
                {"id": "n2", "text": "Duplicate node"},
                {"id": "n3", "text": "Another node"},
            ],
            "edges": [],
        }

        dedup_map = {"n2": "n1"}
        new_graph, stats = rewrite_graph(graph, dedup_map)

        assert len(new_graph["nodes"]) == 2
        node_ids = [n["id"] for n in new_graph["nodes"]]
        assert "n1" in node_ids
        assert "n2" not in node_ids
        assert "n3" in node_ids
        assert stats["nodes_removed_duplicates"] == 1
        assert stats["nodes_removed_empty"] == 0
        assert stats["nodes_removed_total"] == 1

    def test_update_edges(self):
        """Проверка обновления рёбер"""
        graph = {
            "nodes": [
                {"id": "n1", "text": "Node 1"},
                {"id": "n2", "text": "Node 2"},
                {"id": "n3", "text": "Node 3"},
            ],
            "edges": [
                {"source": "n2", "target": "n3", "type": "PREREQUISITE", "weight": 0.8},
                {"source": "n3", "target": "n2", "type": "ELABORATES", "weight": 0.7},
            ],
        }

        dedup_map = {"n2": "n1"}
        new_graph, stats = rewrite_graph(graph, dedup_map)

        assert len(new_graph["nodes"]) == 2
        assert len(new_graph["edges"]) == 2

        # Проверяем, что рёбра обновлены
        edge1 = new_graph["edges"][0]
        assert edge1["source"] == "n1"  # n2 → n1
        assert edge1["target"] == "n3"

        edge2 = new_graph["edges"][1]
        assert edge2["source"] == "n3"
        assert edge2["target"] == "n1"  # n2 → n1

        assert stats["edges_updated"] == 2

    def test_remove_duplicate_edges(self):
        """Проверка удаления дублированных рёбер"""
        graph = {
            "nodes": [
                {"id": "n1", "text": "Node 1"},
                {"id": "n2", "text": "Node 2"},
                {"id": "n3", "text": "Node 3"},
            ],
            "edges": [
                {"source": "n1", "target": "n3", "type": "PREREQUISITE"},
                {"source": "n2", "target": "n3", "type": "PREREQUISITE"},
            ],
        }

        dedup_map = {"n2": "n1"}
        new_graph, stats = rewrite_graph(graph, dedup_map)

        # Оба ребра станут n1→n3, останется только одно
        assert len(new_graph["edges"]) == 1
        assert new_graph["edges"][0]["source"] == "n1"
        assert new_graph["edges"][0]["target"] == "n3"
        # Второе ребро обновлено (n2->n3 стало n1->n3), но удалено как дубликат
        # Первое ребро не требовало обновления
        assert stats["edges_updated"] == 0  # Обновлений не было, т.к. измененное ребро было удалено как дубликат

    def test_remove_dangling_edges(self):
        """Проверка удаления висячих рёбер"""
        graph = {
            "nodes": [
                {"id": "n1", "text": "Node 1"},
                {"id": "n2", "text": "Node 2"},
            ],
            "edges": [
                {"source": "n1", "target": "n2", "type": "PREREQUISITE"},
                {
                    "source": "n2",
                    "target": "n3",
                    "type": "ELABORATES",
                },  # n3 не существует
            ],
        }

        new_graph, stats = rewrite_graph(graph, {})

        # Второе ребро должно быть удалено
        assert len(new_graph["edges"]) == 1
        assert stats["edges_updated"] == 0
        assert new_graph["edges"][0]["target"] == "n2"


class TestSaveDedupMap:
    """Тесты для сохранения маппинга дубликатов"""

    @patch("builtins.open", new_callable=MagicMock)
    @patch("pathlib.Path.mkdir")
    def test_save_empty_map(self, mock_mkdir, mock_open):
        """Проверка сохранения пустого маппинга"""
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        save_dedup_map({}, [])

        # Проверяем, что файл открыт для записи
        mock_open.assert_called_once()

        # Проверяем, что записан заголовок
        write_calls = mock_file.write.call_args_list
        # CSV writer вызывает write для каждой строки
        assert any("duplicate_id" in str(call) for call in write_calls)

    @patch("builtins.open", new_callable=MagicMock)
    @patch("pathlib.Path.mkdir")
    def test_save_with_duplicates(self, mock_mkdir, mock_open):
        """Проверка сохранения с дубликатами"""
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        dedup_map = {
            "dup1": "master1",
            "dup2": "master1",
            "dup3": "master2",
        }

        duplicates = [
            ("master1", "dup1", 0.98),
            ("master1", "dup2", 0.95),
            ("master2", "dup3", 0.97),
        ]

        save_dedup_map(dedup_map, duplicates)

        # Проверяем, что файл создан
        mock_open.assert_called_once()

        # Проверяем вызовы записи
        write_calls = mock_file.write.call_args_list
        write_content = "".join(str(call) for call in write_calls)

        # Проверяем наличие данных
        assert "dup1" in write_content
        assert "master1" in write_content
        assert "0.98" in write_content


class TestUpdateMetadata:
    """Тесты для обновления метаданных"""

    def test_create_new_metadata(self):
        """Проверка создания новых метаданных"""
        config = {
            "sim_threshold": 0.95,
            "len_ratio_min": 0.8,
            "k_neighbors": 5,
            "embedding_model": "text-embedding-3-small",
        }

        statistics = {
            "nodes_analyzed": 150,
            "embeddings_created": 150,
            "potential_duplicates": 23,
            "clusters_formed": 8,
            "nodes_removed_duplicates": 15,
            "nodes_removed_empty": 3,
            "nodes_removed_total": 18,
            "edges_updated": 42,
            "nodes_before": 189,
            "nodes_after": 171,
            "edges_before": 312,
            "edges_after": 287,
        }

        metadata = update_metadata(None, config, statistics, 17.24)

        # Проверяем качественные показатели (теперь внутри deduplication)
        assert "deduplication" in metadata
        assert "quality_issues" in metadata["deduplication"]
        assert metadata["deduplication"]["quality_issues"]["duplicate_nodes_removed"] == 15
        assert metadata["deduplication"]["quality_issues"]["empty_nodes_removed"] == 3
        assert metadata["deduplication"]["quality_issues"]["total_nodes_removed"] == 18

        # Получаем секцию дедупликации
        dedup = metadata["deduplication"]

        # Проверяем конфигурацию
        assert dedup["config"]["similarity_threshold"] == 0.95
        assert dedup["config"]["length_ratio_threshold"] == 0.8
        assert dedup["config"]["top_k"] == 5
        assert dedup["config"]["model"] == "text-embedding-3-small"

        # Проверяем статистику
        assert dedup["statistics"]["nodes_analyzed"] == 150
        assert dedup["statistics"]["embeddings_created"] == 150
        assert dedup["statistics"]["potential_duplicates"] == 23
        assert dedup["statistics"]["clusters_formed"] == 8
        assert dedup["statistics"]["nodes_removed"]["duplicates"] == 15
        assert dedup["statistics"]["nodes_removed"]["empty"] == 3
        assert dedup["statistics"]["nodes_removed"]["total"] == 18
        assert dedup["statistics"]["edges_updated"] == 42
        assert dedup["statistics"]["processing_time_seconds"] == 17.24

        # Проверяем before/after
        assert dedup["before_after"]["nodes_before"] == 189
        assert dedup["before_after"]["nodes_after"] == 171
        assert dedup["before_after"]["edges_before"] == 312
        assert dedup["before_after"]["edges_after"] == 287

        # Проверяем временную метку
        assert "performed_at" in dedup

    def test_update_existing_metadata(self):
        """Проверка обновления существующих метаданных"""
        existing_meta = {
            "generated_at": "2024-01-15 10:45:00",
            "generator": "itext2kg_graph",
            "config": {
                "model": "o4-mini-2025-04-16",
                "temperature": 0.6,
            },
            "quality_issues": {
                "some_other_issue": 5,
            },
        }

        config = {
            "sim_threshold": 0.97,
            "len_ratio_min": 0.8,
            "k_neighbors": 5,
        }

        statistics = {
            "nodes_removed_duplicates": 10,
            "nodes_removed_empty": 2,
            "nodes_removed_total": 12,
            "nodes_analyzed": 100,
            "embeddings_created": 100,
            "potential_duplicates": 15,
            "clusters_formed": 5,
            "edges_updated": 20,
            "nodes_before": 150,
            "nodes_after": 138,
            "edges_before": 200,
            "edges_after": 180,
        }

        metadata = update_metadata(existing_meta, config, statistics, 10.5)

        # Проверяем, что существующие поля сохранились
        assert metadata["generated_at"] == "2024-01-15 10:45:00"
        assert metadata["generator"] == "itext2kg_graph"
        assert metadata["config"]["model"] == "o4-mini-2025-04-16"

        # Проверяем quality_issues в секции deduplication
        # (старые quality_issues не сохраняются, т.к. теперь всё внутри deduplication)
        assert "quality_issues" in metadata["deduplication"]
        assert metadata["deduplication"]["quality_issues"]["duplicate_nodes_removed"] == 10
        assert metadata["deduplication"]["quality_issues"]["empty_nodes_removed"] == 2
        assert metadata["deduplication"]["quality_issues"]["total_nodes_removed"] == 12

        # Проверяем секцию дедупликации
        assert "deduplication" in metadata
        assert metadata["deduplication"]["statistics"]["nodes_removed"]["duplicates"] == 10

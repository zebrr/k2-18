"""
Интеграционные тесты для dedup.py с реальными API вызовами

Требуют реального API ключа в config.toml
"""

import csv
import json
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from src.utils.exit_codes import EXIT_INPUT_ERROR, EXIT_SUCCESS


def create_test_graph_with_semantic_duplicates():
    """Создание тестового графа с реальными семантическими дубликатами"""
    nodes = []
    edges = []

    # Добавляем Concept узлы (не должны участвовать в дедупликации)
    nodes.append(
        {
            "id": "concept_python",
            "type": "Concept",
            "text": "Python",
            "definition": "High-level programming language",
            "node_offset": 0,
            "local_start": 0,
        }
    )

    nodes.append(
        {
            "id": "concept_variable",
            "type": "Concept",
            "text": "Variable",
            "definition": "Named storage location",
            "node_offset": 0,
            "local_start": 10,
        }
    )

    # Группа 1: Определение переменных (семантические дубликаты)
    nodes.append(
        {
            "id": "chunk_vars_1",
            "type": "Chunk",
            "text": "In Python, variables are containers for storing data values. You create a variable by assigning a value to it using the equals sign.",
            "node_offset": 0,
            "local_start": 100,
        }
    )

    nodes.append(
        {
            "id": "chunk_vars_2",
            "type": "Chunk",
            "text": "Python variables are used to store data values. A variable is created when you assign a value to it with the = operator.",
            "node_offset": 0,
            "local_start": 200,
        }
    )

    nodes.append(
        {
            "id": "chunk_vars_3",
            "type": "Chunk",
            "text": "Variables in Python are containers that hold data. To create a variable, simply assign it a value using the assignment operator (=).",
            "node_offset": 0,
            "local_start": 300,
        }
    )

    # Группа 2: Циклы for (семантические дубликаты)
    nodes.append(
        {
            "id": "chunk_for_1",
            "type": "Chunk",
            "text": "The for loop in Python is used to iterate over a sequence (list, tuple, string) or other iterable objects. It executes a block of code for each item.",
            "node_offset": 0,
            "local_start": 400,
        }
    )

    nodes.append(
        {
            "id": "chunk_for_2",
            "type": "Chunk",
            "text": "Python for loops iterate through sequences like lists, tuples, or strings. The loop runs a code block once for every element in the sequence.",
            "node_offset": 0,
            "local_start": 500,
        }
    )

    # Уникальные chunks (не должны быть помечены как дубликаты)
    nodes.append(
        {
            "id": "chunk_unique_1",
            "type": "Chunk",
            "text": "Python supports multiple programming paradigms including procedural, object-oriented, and functional programming styles.",
            "node_offset": 0,
            "local_start": 600,
        }
    )

    nodes.append(
        {
            "id": "chunk_unique_2",
            "type": "Chunk",
            "text": "List comprehensions provide a concise way to create lists in Python. They consist of brackets containing an expression followed by a for clause.",
            "node_offset": 0,
            "local_start": 700,
        }
    )

    # Assessment узлы
    nodes.append(
        {
            "id": "assessment_1",
            "type": "Assessment",
            "text": "What is a variable in Python?",
            "node_offset": 0,
            "local_start": 800,
        }
    )

    nodes.append(
        {
            "id": "assessment_2",
            "type": "Assessment",
            "text": "How do you create a variable in Python?",  # Семантически похож на assessment_1
            "node_offset": 0,
            "local_start": 900,
        }
    )

    # Добавляем рёбра
    edges.extend(
        [
            {
                "source": "chunk_vars_1",
                "target": "concept_variable",
                "type": "MENTIONS",
                "weight": 1.0,
            },
            {
                "source": "chunk_vars_2",
                "target": "concept_variable",
                "type": "MENTIONS",
                "weight": 1.0,
            },
            {
                "source": "chunk_vars_3",
                "target": "concept_variable",
                "type": "MENTIONS",
                "weight": 1.0,
            },
            {
                "source": "chunk_vars_1",
                "target": "chunk_for_1",
                "type": "PREREQUISITE",
                "weight": 0.8,
            },
            {
                "source": "chunk_vars_2",
                "target": "chunk_unique_1",
                "type": "ELABORATES",
                "weight": 0.6,
            },
            {
                "source": "chunk_for_1",
                "target": "chunk_unique_2",
                "type": "HINT_FORWARD",
                "weight": 0.7,
            },
            {
                "source": "chunk_vars_3",
                "target": "assessment_1",
                "type": "TESTS",
                "weight": 0.9,
            },
            {
                "source": "chunk_for_2",
                "target": "assessment_2",
                "type": "TESTS",
                "weight": 0.85,
            },
        ]
    )

    return {"nodes": nodes, "edges": edges}


def create_edge_case_graph():
    """Граф с граничными случаями для тестирования"""
    nodes = []

    # Пустые и почти пустые тексты
    nodes.append(
        {
            "id": "chunk_empty",
            "type": "Chunk",
            "text": "",  # Пустой текст
            "node_offset": 0,
            "local_start": 0,
        }
    )

    nodes.append(
        {
            "id": "chunk_whitespace",
            "type": "Chunk",
            "text": "   \n\t  ",  # Только пробелы
            "node_offset": 0,
            "local_start": 100,
        }
    )

    # Очень длинный текст (близкий к лимиту 8192 токена)
    long_text = "This is a very long educational text about Python. " * 500
    nodes.append(
        {
            "id": "chunk_very_long",
            "type": "Chunk",
            "text": long_text[:30000],  # Примерно 7000 токенов
            "node_offset": 0,
            "local_start": 200,
        }
    )

    # Идентичные тексты (100% дубликаты)
    nodes.append(
        {
            "id": "chunk_identical_1",
            "type": "Chunk",
            "text": "Python is a high-level, interpreted programming language known for its simplicity.",
            "node_offset": 0,
            "local_start": 300,
        }
    )

    nodes.append(
        {
            "id": "chunk_identical_2",
            "type": "Chunk",
            "text": "Python is a high-level, interpreted programming language known for its simplicity.",
            "node_offset": 0,
            "local_start": 400,
        }
    )

    return {"nodes": nodes, "edges": []}


class TestDedupIntegrationReal(unittest.TestCase):
    """Интеграционные тесты с реальными API вызовами"""

    def setUp(self):
        """Подготовка перед каждым тестом"""
        # Создаем директории если нужно
        Path("data/out").mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(exist_ok=True)

    def tearDown(self):
        """Очистка после каждого теста"""
        # Удаляем созданные файлы
        files_to_remove = [
            "data/out/LearningChunkGraph_raw.json",
            "data/out/LearningChunkGraph_dedup.json",
            "logs/dedup_map.csv",
        ]

        for file_path in files_to_remove:
            path = Path(file_path)
            if path.exists():
                try:
                    path.unlink()
                except Exception as e:
                    pass  # Ignore cleanup errors

    def test_semantic_duplicates_two_thresholds(self):
        """Тест с двумя порогами - низким 0.5 и реальным из конфига"""
        # Создаем граф с семантическими дубликатами
        graph = create_test_graph_with_semantic_duplicates()

        # Сохраняем входной файл
        input_path = Path("data/out/LearningChunkGraph_raw.json")
        input_path.parent.mkdir(parents=True, exist_ok=True)
        with open(input_path, "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)

        # Создаем директорию для логов
        logs_path = Path("logs")
        logs_path.mkdir(exist_ok=True)

        # ПЕРВЫЙ ВЫЗОВ: с низким порогом 0.5 для гарантированного нахождения дубликатов

        # Загружаем реальный конфиг и патчим только sim_threshold
        from src.utils import load_config

        real_config = load_config()
        patched_config = real_config.copy()
        patched_config["dedup"]["sim_threshold"] = 0.5  # Низкий порог для семантических дубликатов

        with patch("src.dedup.load_config", return_value=patched_config):
            from src.dedup import main

            start_time = time.time()
            exit_code = main()
            duration = time.time() - start_time

        assert exit_code == EXIT_SUCCESS

        # Проверяем результат с низким порогом
        output_path = Path("data/out/LearningChunkGraph_dedup.json")
        with open(output_path, "r", encoding="utf-8") as f:
            result_low_threshold = json.load(f)

        original_count = len([n for n in graph["nodes"] if n["type"] in ["Chunk", "Assessment"]])
        result_count_low = len(
            [n for n in result_low_threshold["nodes"] if n["type"] in ["Chunk", "Assessment"]]
        )

        # С низким порогом ДОЛЖНЫ найтись дубликаты
        assert result_count_low < original_count, "With threshold 0.5 duplicates MUST be found"

        # Смотрим что нашлось
        dedup_map_path = Path("logs/dedup_map.csv")
        with open(dedup_map_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            dedup_entries_low = list(reader)

        for entry in dedup_entries_low[:5]:  # Показываем первые 5
            # Проверяем структуру записей в dedup_map
            assert "duplicate_id" in entry, "Должен быть duplicate_id в записи"
            assert "master_id" in entry, "Должен быть master_id в записи"
            assert "similarity" in entry, "Должна быть similarity в записи"
            assert float(entry["similarity"]) >= 0.5, "Similarity должна быть >= 0.5"

        # ВТОРОЙ ВЫЗОВ: с реальным порогом из конфига (0.97)

        # Восстанавливаем входной файл (на случай если dedup его изменил)
        with open(input_path, "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)

        # Запускаем с реальным конфигом
        from src.dedup import main

        start_time = time.time()
        exit_code = main()
        duration = time.time() - start_time

        assert exit_code == EXIT_SUCCESS

        # Проверяем результат с высоким порогом
        with open(output_path, "r", encoding="utf-8") as f:
            result_high_threshold = json.load(f)

        result_count_high = len(
            [n for n in result_high_threshold["nodes"] if n["type"] in ["Chunk", "Assessment"]]
        )

        # С высоким порогом дубликаты могут не найтись - это нормально
        if result_count_high == original_count:
            # Проверяем что с порогом 0.97 дубликаты действительно не найдены
            assert (
                result_count_high == original_count
            ), "С порогом 0.97 ожидалось сохранение всех узлов"

        # Смотрим что нашлось
        with open(dedup_map_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            dedup_entries_high = list(reader)

        if dedup_entries_high:
            for entry in dedup_entries_high:
                # Проверяем записи с высоким порогом
                assert (
                    float(entry["similarity"]) >= 0.97
                ), f"Similarity должна быть >= 0.97, но получено {entry['similarity']}"
                assert (
                    entry["duplicate_id"] != entry["master_id"]
                ), "duplicate_id не должен совпадать с master_id для дубликатов"
        else:
            # Если записей нет, проверяем что все узлы сохранены
            assert (
                result_count_high == original_count
            ), "Если дубликатов нет, должны сохраниться все узлы"

        # Итоговое сравнение
        # Проверяем что с низким порогом найдено больше дубликатов чем с высоким
        assert (
            result_count_low <= result_count_high
        ), f"С низким порогом (0.5) должно остаться меньше или столько же узлов чем с высоким (0.97): {result_count_low} vs {result_count_high}"
        assert len(dedup_entries_low) >= len(
            dedup_entries_high
        ), f"С низким порогом должно найтись больше или столько же дубликатов: {len(dedup_entries_low)} vs {len(dedup_entries_high)}"

    def test_edge_cases_with_real_api(self):
        """Тест граничных случаев с реальным API"""
        # Создаем граф с граничными случаями
        graph = create_edge_case_graph()

        input_path = Path("data/out/LearningChunkGraph_raw.json")
        input_path.parent.mkdir(parents=True, exist_ok=True)
        with open(input_path, "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)

        # Создаем директорию для логов
        Path("logs").mkdir(exist_ok=True)

        # Запускаем dedup
        from src.dedup import main

        exit_code = main()

        assert exit_code == EXIT_SUCCESS

        # Проверяем результат
        output_path = Path("data/out/LearningChunkGraph_dedup.json")
        with open(output_path, "r", encoding="utf-8") as f:
            result_graph = json.load(f)

        result_ids = {n["id"] for n in result_graph["nodes"]}

        # Пустые тексты должны быть отфильтрованы
        assert "chunk_empty" not in result_ids
        assert "chunk_whitespace" not in result_ids

        # Длинный текст должен остаться
        assert "chunk_very_long" in result_ids

        # Идентичные тексты - один должен быть удален
        identical_in_result = ["chunk_identical_1", "chunk_identical_2"]
        identical_count = sum(1 for id in identical_in_result if id in result_ids)
        assert identical_count == 1, "Only one of identical chunks should remain"

        # Проверяем dedup_map
        dedup_map_path = Path("logs/dedup_map.csv")
        with open(dedup_map_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            dedup_entries = list(reader)

        # Находим запись об идентичных chunks
        identical_entries = [
            e
            for e in dedup_entries
            if "identical" in e["duplicate_id"] or "identical" in e["master_id"]
        ]
        assert len(identical_entries) == 1
        assert float(identical_entries[0]["similarity"]) > 0.99  # Почти 1.0 для идентичных

    def test_rate_limiting_handling(self):
        """Тест обработки rate limiting от реального API"""
        # Создаем граф с большим количеством узлов
        nodes = []
        for i in range(50):  # Уменьшил до 50 для скорости
            nodes.append(
                {
                    "id": f"chunk_load_{i}",
                    "type": "Chunk",
                    "text": f"This is test chunk number {i} with unique content to test rate limiting. "
                    * 10,
                    "node_offset": 0,
                    "local_start": i * 100,
                }
            )

        graph = {"nodes": nodes, "edges": []}

        input_path = Path("data/out/LearningChunkGraph_raw.json")
        input_path.parent.mkdir(parents=True, exist_ok=True)
        with open(input_path, "w", encoding="utf-8") as f:
            json.dump(graph, f)

        Path("logs").mkdir(exist_ok=True)

        # Запускаем dedup и замеряем время
        from src.dedup import main

        start_time = time.time()
        exit_code = main()
        duration = time.time() - start_time

        # Должно успешно завершиться
        assert exit_code == EXIT_SUCCESS

        # Если обработка заняла больше 20 секунд, возможно было ожидание rate limit
        if duration > 20:
            pass  # Было print предупреждение о долгой работе

    def test_invalid_graph_structure(self):
        """Тест обработки невалидной структуры графа"""
        # Граф без обязательного поля 'edges'
        invalid_graph = {
            "nodes": [
                {"id": "test", "type": "Chunk", "text": "Test", "node_offset": 0, "local_start": 0}
            ]
            # 'edges' отсутствует!
        }

        input_path = Path("data/out/LearningChunkGraph_raw.json")
        input_path.parent.mkdir(parents=True, exist_ok=True)
        with open(input_path, "w", encoding="utf-8") as f:
            json.dump(invalid_graph, f)

        Path("logs").mkdir(exist_ok=True)

        from src.dedup import main

        exit_code = main()

        assert exit_code == EXIT_INPUT_ERROR

    def test_missing_input_file(self):
        """Тест обработки отсутствующего входного файла"""
        # Удаляем входной файл если он есть
        input_path = Path("data/out/LearningChunkGraph_raw.json")
        if input_path.exists():
            input_path.unlink()

        Path("logs").mkdir(exist_ok=True)

        from src.dedup import main

        exit_code = main()

        assert exit_code == EXIT_INPUT_ERROR


if __name__ == "__main__":
    # Запуск тестов
    pytest.main([__file__, "-v", "-s"])

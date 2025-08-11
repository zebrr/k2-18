"""
Базовые тесты для быстрой проверки модуля валидации.
"""

import sys
from pathlib import Path

import pytest

# Добавляем корневую папку проекта в sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.validation import (GraphInvariantError, _load_schema,
                                  validate_concept_dictionary_invariants,
                                  validate_graph_invariants, validate_json)


def test_basic_functionality():
    """Базовый тест загрузки модуля и схем."""


    # Тест 1: Проверяем, что можем загрузить схемы
    try:
        concept_schema = _load_schema("ConceptDictionary")
    except Exception as e:
        pytest.fail(f"Ошибка загрузки ConceptDictionary.schema.json: {e}")

    try:
        graph_schema = _load_schema("LearningChunkGraph")
    except Exception as e:
        pytest.fail(f"Ошибка загрузки LearningChunkGraph.schema.json: {e}")

    # Тест 2: Проверяем валидацию простых корректных данных
    try:
        # Минимальный корректный ConceptDictionary
        valid_concept_dict = {
            "concepts": [
                {
                    "concept_id": "test_concept_1",
                    "term": {"primary": "Тестовый концепт"},
                    "definition": "Это тестовое определение",
                }
            ]
        }

        validate_json(valid_concept_dict, "ConceptDictionary")
        validate_concept_dictionary_invariants(valid_concept_dict)

    except Exception as e:
        pytest.fail(f"Ошибка валидации ConceptDictionary: {e}")

    try:
        # Минимальный корректный LearningChunkGraph
        valid_graph = {
            "nodes": [
                {
                    "id": "test_chunk_1",
                    "type": "Chunk",
                    "text": "Это тестовый текст чанка",
                    "node_offset": 0,
                    "local_start": 0,
                    "difficulty": 3,
                }
            ],
            "edges": [],
        }

        validate_json(valid_graph, "LearningChunkGraph")
        validate_graph_invariants(valid_graph)

    except Exception as e:
        pytest.fail(f"Ошибка валидации LearningChunkGraph: {e}")

    # Тест 3: Проверяем обнаружение ошибок
    try:
        invalid_graph = {
            "nodes": [
                {
                    "id": "test_chunk_1",
                    "type": "Chunk",
                    "text": "Тест",
                    "node_offset": 0,
                    "local_start": 0,
                    "difficulty": 1,
                }
            ],
            "edges": [
                {
                    "source": "test_chunk_1",
                    "target": "test_chunk_1",
                    "type": "PREREQUISITE",  # Self-loop должен быть запрещён
                }
            ],
        }

        validate_graph_invariants(invalid_graph)
        pytest.fail("Не обнаружил PREREQUISITE self-loop")

    except GraphInvariantError as e:
        if "self-loop" in str(e):
            pass  # Ожидаемая ошибка обнаружена
        else:
            pytest.fail(f"Неожиданная ошибка: {e}")
    except Exception as e:
        pytest.fail(f"Неожиданное исключение: {e}")

    # Тест завершается успешно - не возвращаем значение


if __name__ == "__main__":
    # Для запуска напрямую используем pytest
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))

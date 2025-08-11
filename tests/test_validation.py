"""
Тесты для модуля валидации.
"""

from unittest.mock import mock_open, patch

import pytest

from src.utils.validation import (GraphInvariantError, ValidationError,
                                  _load_schema,
                                  validate_concept_dictionary_invariants,
                                  validate_graph_invariants,
                                  validate_graph_invariants_intermediate,
                                  validate_json)


class TestLoadSchema:
    """Тесты загрузки схем."""

    def test_load_valid_schema(self):
        """Тест загрузки существующей валидной схемы."""
        # Предполагаем, что схемы уже существуют
        schema = _load_schema("ConceptDictionary")
        assert isinstance(schema, dict)
        assert "$schema" in schema
        assert "properties" in schema

    def test_load_nonexistent_schema(self):
        """Тест загрузки несуществующей схемы."""
        with pytest.raises(FileNotFoundError, match="JSON Schema not found"):
            _load_schema("NonExistentSchema")

    @patch("builtins.open", mock_open(read_data='{"invalid": json}'))
    def test_load_invalid_json(self):
        """Тест загрузки схемы с некорректным JSON."""
        with patch("pathlib.Path.exists", return_value=True):
            with pytest.raises(ValidationError, match="Invalid JSON in schema"):
                _load_schema("InvalidSchema")


class TestValidateJson:
    """Тесты валидации JSON по схемам."""

    def setup_method(self):
        """Очистить кэш схем перед каждым тестом для изоляции."""
        from src.utils import validation
        validation._SCHEMA_CACHE.clear()

    def test_valid_concept_dictionary(self):
        """Тест валидации корректного ConceptDictionary."""
        valid_data = {
            "concepts": [
                {
                    "concept_id": "test:concept:1",
                    "term": {
                        "primary": "Тестовый концепт",
                        "aliases": ["test_concept", "концепт"],
                    },
                    "definition": "Определение тестового концепта",
                }
            ]
        }

        # Не должно выбросить исключение
        validate_json(valid_data, "ConceptDictionary")

    def test_valid_learning_chunk_graph(self):
        """Тест валидации корректного LearningChunkGraph."""
        valid_data = {
            "nodes": [
                {
                    "id": "test:chunk:1",
                    "type": "Chunk",
                    "text": "Тестовый текст чанка",
                    "node_offset": 0,
                    "local_start": 0,
                    "difficulty": 3,
                },
                {
                    "id": "test:concept:1",
                    "type": "Concept",
                    "text": "Концепт",
                    "node_offset": 0,
                    "local_start": 10,
                    "definition": "Определение концепта",
                },
            ],
            "edges": [
                {
                    "source": "test:chunk:1",
                    "target": "test:concept:1",
                    "type": "MENTIONS",
                    "weight": 0.8,
                }
            ],
        }

        # Не должно выбросить исключение
        validate_json(valid_data, "LearningChunkGraph")

    def test_invalid_concept_dictionary_missing_required(self):
        """Тест валидации ConceptDictionary с отсутствующими обязательными полями."""
        invalid_data = {
            "concepts": [
                {
                    "concept_id": "test:concept:1",
                    # Отсутствует обязательное поле "term"
                    "definition": "Определение",
                }
            ]
        }

        with pytest.raises(ValidationError, match="Schema validation error.*term"):
            validate_json(invalid_data, "ConceptDictionary")

    def test_invalid_graph_wrong_edge_type(self):
        """Тест валидации графа с некорректным типом ребра."""
        invalid_data = {
            "nodes": [
                {
                    "id": "test:chunk:1",
                    "type": "Chunk",
                    "text": "Текст",
                    "node_offset": 0,
                    "local_start": 0,
                }
            ],
            "edges": [
                {
                    "source": "test:chunk:1",
                    "target": "test:chunk:1",
                    "type": "INVALID_TYPE",  # Неверный тип ребра
                }
            ],
        }

        with pytest.raises(ValidationError, match="INVALID_TYPE"):
            validate_json(invalid_data, "LearningChunkGraph")


class TestValidateGraphInvariants:
    """Тесты проверки инвариантов графа."""

    def setup_method(self):
        """Очистить кэш схем перед каждым тестом для изоляции."""
        from src.utils import validation
        validation._SCHEMA_CACHE.clear()

    def test_valid_graph_invariants(self):
        """Тест корректного графа."""
        valid_graph = {
            "nodes": [
                {
                    "id": "test:chunk:1",
                    "type": "Chunk",
                    "text": "Первый чанк",
                    "node_offset": 0,
                    "local_start": 0,
                    "difficulty": 2,
                },
                {
                    "id": "test:chunk:2",
                    "type": "Chunk",
                    "text": "Второй чанк",
                    "node_offset": 0,
                    "local_start": 50,
                    "difficulty": 3,
                },
            ],
            "edges": [
                {
                    "source": "test:chunk:1",
                    "target": "test:chunk:2",
                    "type": "PREREQUISITE",
                    "weight": 0.9,
                }
            ],
        }

        # Не должно выбросить исключение
        validate_graph_invariants(valid_graph)

    def test_duplicate_node_ids(self):
        """Тест обнаружения дублированных ID узлов."""
        invalid_graph = {
            "nodes": [
                {
                    "id": "duplicate:id",
                    "type": "Chunk",
                    "text": "Первый",
                    "node_offset": 0,
                    "local_start": 0,
                },
                {
                    "id": "duplicate:id",  # Дубликат
                    "type": "Chunk",
                    "text": "Второй",
                    "node_offset": 0,
                    "local_start": 10,
                },
            ],
            "edges": [],
        }

        with pytest.raises(GraphInvariantError, match="Duplicate node ID"):
            validate_graph_invariants(invalid_graph)

    def test_prerequisite_self_loop(self):
        """Тест обнаружения PREREQUISITE self-loop."""
        invalid_graph = {
            "nodes": [
                {
                    "id": "test:chunk:1",
                    "type": "Chunk",
                    "text": "Чанк",
                    "node_offset": 0,
                    "local_start": 0,
                    "difficulty": 1,
                }
            ],
            "edges": [
                {
                    "source": "test:chunk:1",
                    "target": "test:chunk:1",  # Self-loop
                    "type": "PREREQUISITE",
                }
            ],
        }

        with pytest.raises(
            GraphInvariantError, match="PREREQUISITE self-loop forbidden"
        ):
            validate_graph_invariants(invalid_graph)

    def test_nonexistent_edge_target(self):
        """Тест обнаружения ребра с несуществующим target."""
        invalid_graph = {
            "nodes": [
                {
                    "id": "test:chunk:1",
                    "type": "Chunk",
                    "text": "Чанк",
                    "node_offset": 0,
                    "local_start": 0,
                    "difficulty": 1,
                }
            ],
            "edges": [
                {
                    "source": "test:chunk:1",
                    "target": "nonexistent:id",  # Несуществующий узел
                    "type": "MENTIONS",
                }
            ],
        }

        with pytest.raises(GraphInvariantError, match="target.*does not exist"):
            validate_graph_invariants(invalid_graph)

    def test_invalid_weight_range(self):
        """Тест обнаружения веса вне диапазона [0,1] - схема должна отловить это."""
        invalid_graph = {
            "nodes": [
                {
                    "id": "test:chunk:1",
                    "type": "Chunk",
                    "text": "Первый",
                    "node_offset": 0,
                    "local_start": 0,
                    "difficulty": 1,
                },
                {
                    "id": "test:chunk:2",
                    "type": "Chunk",
                    "text": "Второй",
                    "node_offset": 0,
                    "local_start": 10,
                    "difficulty": 1,
                },
            ],
            "edges": [
                {
                    "source": "test:chunk:1",
                    "target": "test:chunk:2",
                    "type": "MENTIONS",
                    "weight": 1.5,  # Вне диапазона
                }
            ],
        }

        # Схема должна отловить это раньше validate_graph_invariants
        with pytest.raises(ValidationError, match="greater than the maximum"):
            validate_graph_invariants(invalid_graph)

    def test_duplicate_edges(self):
        """Тест обнаружения дублированных рёбер."""
        invalid_graph = {
            "nodes": [
                {
                    "id": "test:chunk:1",
                    "type": "Chunk",
                    "text": "Первый",
                    "node_offset": 0,
                    "local_start": 0,
                    "difficulty": 1,
                },
                {
                    "id": "test:chunk:2",
                    "type": "Chunk",
                    "text": "Второй",
                    "node_offset": 0,
                    "local_start": 10,
                    "difficulty": 1,
                },
            ],
            "edges": [
                {
                    "source": "test:chunk:1",
                    "target": "test:chunk:2",
                    "type": "MENTIONS",
                    "weight": 0.8,
                },
                {
                    "source": "test:chunk:1",
                    "target": "test:chunk:2",
                    "type": "MENTIONS",  # Дубликат
                    "weight": 0.9,
                },
            ],
        }

        with pytest.raises(GraphInvariantError, match="duplicate edge"):
            validate_graph_invariants(invalid_graph)


class TestValidateConceptDictionary:
    """Тесты проверки инвариантов словаря концептов."""

    def test_valid_concept_dictionary(self):
        """Тест корректного словаря концептов."""
        valid_dict = {
            "concepts": [
                {
                    "concept_id": "test:concept:1",
                    "term": {"primary": "Переменная", "aliases": ["variable", "var"]},
                    "definition": "Именованная область памяти",
                },
                {
                    "concept_id": "test:concept:2",
                    "term": {"primary": "Функция", "aliases": ["function"]},
                    "definition": "Блок организованного кода",
                },
            ]
        }

        # Не должно выбросить исключение
        validate_concept_dictionary_invariants(valid_dict)

    def test_duplicate_concept_ids(self):
        """Тест обнаружения дублированных concept_id."""
        invalid_dict = {
            "concepts": [
                {
                    "concept_id": "duplicate:id",
                    "term": {"primary": "Первый"},
                    "definition": "Определение первого",
                },
                {
                    "concept_id": "duplicate:id",  # Дубликат
                    "term": {"primary": "Второй"},
                    "definition": "Определение второго",
                },
            ]
        }

        with pytest.raises(ValidationError, match="duplicate concept_id"):
            validate_concept_dictionary_invariants(invalid_dict)

    def test_duplicate_primary_terms_allowed(self):
        """Тест что дублированные primary термины теперь разрешены."""
        valid_dict = {
            "concepts": [
                {
                    "concept_id": "test:concept:1",
                    "term": {"primary": "Переменная"},
                    "definition": "Первое определение",
                },
                {
                    "concept_id": "test:concept:2",
                    "term": {"primary": "переменная"},  # Дубликат разрешен!
                    "definition": "Второе определение",
                },
            ]
        }

        # НЕ должно выбросить исключение
        validate_concept_dictionary_invariants(valid_dict)

    def test_primary_in_aliases(self):
        """Тест обнаружения primary термина в aliases."""
        invalid_dict = {
            "concepts": [
                {
                    "concept_id": "test:concept:1",
                    "term": {
                        "primary": "Переменная",
                        "aliases": ["variable", "переменная"],  # Дубликат primary
                    },
                    "definition": "Определение",
                }
            ]
        }

        with pytest.raises(
            ValidationError, match="primary term.*duplicated in aliases"
        ):
            validate_concept_dictionary_invariants(invalid_dict)

    def test_duplicate_aliases_within_concept(self):
        """Тест обнаружения дублированных aliases в одном концепте."""
        invalid_dict = {
            "concepts": [
                {
                    "concept_id": "test:concept:1",
                    "term": {
                        "primary": "Переменная",
                        "aliases": [
                            "variable",
                            "var",
                            "Variable",
                        ],  # Variable = variable
                    },
                    "definition": "Определение",
                }
            ]
        }

        with pytest.raises(ValidationError, match="duplicate alias"):
            validate_concept_dictionary_invariants(invalid_dict)

    def test_duplicate_aliases_across_concepts_allowed(self):
        """Тест что дублированные aliases между концептами теперь разрешены."""
        valid_dict = {
            "concepts": [
                {
                    "concept_id": "test:concept:1",
                    "term": {"primary": "Переменная", "aliases": ["var"]},
                    "definition": "Первое определение",
                },
                {
                    "concept_id": "test:concept:2",
                    "term": {
                        "primary": "Функция",
                        "aliases": ["VAR"],  # Дубликат разрешен!
                    },
                    "definition": "Второе определение",
                },
            ]
        }

        # НЕ должно выбросить исключение
        validate_concept_dictionary_invariants(valid_dict)


class TestIntermediateValidation:
    """Тесты для промежуточной валидации графа."""

    def test_intermediate_allows_duplicate_concepts(self):
        """Промежуточная валидация разрешает дубликаты концептов."""
        graph_data = {
            "nodes": [
                {
                    "id": "test:p:concept1",
                    "type": "Concept",
                    "text": "Концепт 1",
                    "node_offset": 0,
                    "local_start": 0,
                },
                {
                    "id": "test:p:concept1",
                    "type": "Concept",
                    "text": "Концепт 1",
                    "node_offset": 0,
                    "local_start": 100,
                },  # Дубликат
            ],
            "edges": [],
        }

        # Обычная валидация должна упасть
        with pytest.raises(GraphInvariantError, match="Duplicate node ID"):
            validate_graph_invariants(graph_data)

        # Промежуточная валидация должна пройти
        validate_graph_invariants_intermediate(graph_data)  # Не должно быть исключения

    def test_intermediate_checks_chunk_duplicates(self):
        """Промежуточная валидация проверяет дубликаты Chunk узлов."""
        graph_data = {
            "nodes": [
                {
                    "id": "test:c:100",
                    "type": "Chunk",
                    "text": "Текст 1",
                    "node_offset": 0,
                    "local_start": 100,
                },
                {
                    "id": "test:c:100",
                    "type": "Chunk",
                    "text": "Текст 2",
                    "node_offset": 0,
                    "local_start": 200,
                },  # Дубликат
            ],
            "edges": [],
        }

        with pytest.raises(
            GraphInvariantError, match="Duplicate node ID \\(Chunk\\)"
        ):
            validate_graph_invariants_intermediate(graph_data)

    def test_intermediate_checks_assessment_duplicates(self):
        """Промежуточная валидация проверяет дубликаты Assessment узлов."""
        graph_data = {
            "nodes": [
                {
                    "id": "test:q:100:0",
                    "type": "Assessment",
                    "text": "Вопрос 1",
                    "node_offset": 0,
                    "local_start": 100,
                },
                {
                    "id": "test:q:100:0",
                    "type": "Assessment",
                    "text": "Вопрос 1",
                    "node_offset": 0,
                    "local_start": 200,
                },  # Дубликат
            ],
            "edges": [],
        }

        with pytest.raises(
            GraphInvariantError, match="Duplicate node ID \\(Assessment\\)"
        ):
            validate_graph_invariants_intermediate(graph_data)

    def test_intermediate_checks_other_invariants(self):
        """Промежуточная валидация проверяет остальные инварианты."""
        # Проверка PREREQUISITE self-loops
        graph_data = {
            "nodes": [
                {
                    "id": "test:c:100",
                    "type": "Chunk",
                    "text": "Текст",
                    "node_offset": 0,
                    "local_start": 100,
                }
            ],
            "edges": [
                {"source": "test:c:100", "target": "test:c:100", "type": "PREREQUISITE"}
            ],
        }

        with pytest.raises(GraphInvariantError, match="PREREQUISITE self-loop"):
            validate_graph_invariants_intermediate(graph_data)

        # Проверка битых ссылок
        graph_data = {
            "nodes": [
                {
                    "id": "test:c:100",
                    "type": "Chunk",
                    "text": "Текст",
                    "node_offset": 0,
                    "local_start": 100,
                }
            ],
            "edges": [
                {
                    "source": "test:c:100",
                    "target": "test:c:200",
                    "type": "MENTIONS",
                }  # Несуществующий target
            ],
        }

        with pytest.raises(GraphInvariantError, match="does not exist"):
            validate_graph_invariants_intermediate(graph_data)

    def test_intermediate_allows_mixed_nodes(self):
        """Промежуточная валидация работает со смешанными типами узлов."""
        graph_data = {
            "nodes": [
                {
                    "id": "test:p:concept1",
                    "type": "Concept",
                    "text": "Концепт 1",
                    "node_offset": 0,
                    "local_start": 0,
                },
                {
                    "id": "test:p:concept1",
                    "type": "Concept",
                    "text": "Концепт 1 дубликат",
                    "node_offset": 0,
                    "local_start": 50,
                },
                {
                    "id": "test:c:100",
                    "type": "Chunk",
                    "text": "Текст чанка",
                    "node_offset": 0,
                    "local_start": 100,
                },
                {
                    "id": "test:q:200:0",
                    "type": "Assessment",
                    "text": "Вопрос",
                    "node_offset": 0,
                    "local_start": 200,
                },
            ],
            "edges": [
                {
                    "source": "test:c:100",
                    "target": "test:p:concept1",
                    "type": "MENTIONS",
                    "weight": 1.0,
                },
                {
                    "source": "test:q:200:0",
                    "target": "test:c:100",
                    "type": "TESTS",
                    "weight": 0.8,
                },
            ],
        }

        # Должна пройти успешно несмотря на дубликат концепта
        validate_graph_invariants_intermediate(graph_data)  # Не должно быть исключения


if __name__ == "__main__":
    pytest.main([__file__])

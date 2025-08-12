"""
Тесты для модуля валидации.
"""

from unittest.mock import patch

import pytest

from src.utils.validation import (
    GraphInvariantError,
    ValidationError,
    _load_schema,
    validate_concept_dictionary_invariants,
    validate_graph_invariants,
    validate_graph_invariants_intermediate,
    validate_json,
)


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

    def test_load_invalid_json(self):
        """Тест загрузки схемы с некорректным JSON и некорректным форматом JSON Schema."""
        import json
        from pathlib import Path
        from unittest.mock import mock_open

        # Тест 1: Некорректный JSON
        # Патчим open для возврата некорректного JSON
        with patch("builtins.open", mock_open(read_data='{"invalid": json}')):
            with patch.object(Path, "exists", return_value=True):
                with pytest.raises(ValidationError, match="Invalid JSON in schema"):
                    _load_schema("InvalidSchema")

        # Тест 2: Валидный JSON, но некорректная JSON Schema
        invalid_schema = {
            "type": "invalid_type",  # Некорректный тип для JSON Schema
            "properties": "should_be_object",  # properties должно быть объектом
        }
        # Патчим open для возврата некорректной схемы
        with patch("builtins.open", mock_open(read_data=json.dumps(invalid_schema))):
            with patch.object(Path, "exists", return_value=True):
                with pytest.raises(ValidationError, match="Invalid JSON Schema"):
                    _load_schema("BadSchema")


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

        with pytest.raises(GraphInvariantError, match="PREREQUISITE self-loop forbidden"):
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

    def test_weight_validation_by_schema(self):
        """Тест валидации веса на уровне JSON Schema - вес вне диапазона [0,1]."""
        # Этот тест проверяет, что JSON Schema корректно валидирует веса
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


class TestEdgeCases:
    """Тесты граничных случаев для модуля валидации."""

    def setup_method(self):
        """Очистить кэш схем перед каждым тестом для изоляции."""
        from src.utils import validation

        validation._SCHEMA_CACHE.clear()

    def test_empty_graph(self):
        """Тест валидации пустого графа - должен проходить без ошибок."""
        empty_graph = {"nodes": [], "edges": []}

        # Не должно выбросить исключение
        validate_graph_invariants(empty_graph)

    def test_graph_with_nodes_only(self):
        """Тест графа с узлами, но без рёбер."""
        graph_nodes_only = {
            "nodes": [
                {
                    "id": "test:chunk:1",
                    "type": "Chunk",
                    "text": "Текст первого чанка",
                    "node_offset": 0,
                    "local_start": 0,
                    "difficulty": 2,
                },
                {
                    "id": "test:concept:1",
                    "type": "Concept",
                    "text": "Концепт",
                    "node_offset": 0,
                    "local_start": 50,
                    "definition": "Определение концепта",
                },
                {
                    "id": "test:assessment:1",
                    "type": "Assessment",
                    "text": "Вопрос для проверки",
                    "node_offset": 0,
                    "local_start": 100,
                },
            ],
            "edges": [],
        }

        # Не должно выбросить исключение
        validate_graph_invariants(graph_nodes_only)

    def test_node_without_id(self):
        """Тест обнаружения узла без поля ID."""
        graph_no_id = {
            "nodes": [
                {
                    # Отсутствует поле id
                    "type": "Chunk",
                    "text": "Текст чанка",
                    "node_offset": 0,
                    "local_start": 0,
                }
            ],
            "edges": [],
        }

        # JSON Schema ловит отсутствие обязательного поля 'id' раньше инвариантов
        with pytest.raises(ValidationError, match="'id' is a required property"):
            validate_graph_invariants(graph_no_id)

    def test_nonexistent_source_in_edge(self):
        """Тест ребра с несуществующим source узлом."""
        graph_bad_source = {
            "nodes": [
                {
                    "id": "test:chunk:1",
                    "type": "Chunk",
                    "text": "Текст чанка",
                    "node_offset": 0,
                    "local_start": 0,
                    "difficulty": 1,
                }
            ],
            "edges": [
                {
                    "source": "nonexistent:node",  # Несуществующий источник
                    "target": "test:chunk:1",
                    "type": "MENTIONS",
                    "weight": 0.5,
                }
            ],
        }

        with pytest.raises(GraphInvariantError, match="source.*does not exist"):
            validate_graph_invariants(graph_bad_source)

    def test_schema_caching(self):
        """Тест кэширования схем после первой загрузки."""
        from src.utils import validation

        # Очищаем кэш
        validation._SCHEMA_CACHE.clear()
        assert len(validation._SCHEMA_CACHE) == 0

        # Первая загрузка - схема попадает в кэш
        schema1 = _load_schema("ConceptDictionary")
        assert "ConceptDictionary" in validation._SCHEMA_CACHE
        assert len(validation._SCHEMA_CACHE) == 1

        # Вторая загрузка - используется кэш (проверяем, что возвращается тот же объект)
        schema2 = _load_schema("ConceptDictionary")
        assert schema1 is schema2  # Тот же объект из кэша
        assert len(validation._SCHEMA_CACHE) == 1  # Кэш не изменился

        # Загрузка другой схемы
        _load_schema("LearningChunkGraph")
        assert "LearningChunkGraph" in validation._SCHEMA_CACHE
        assert len(validation._SCHEMA_CACHE) == 2  # Теперь две схемы в кэше

    def test_nested_validation_error_path(self):
        """Тест форматирования пути ошибки для вложенных ошибок валидации."""
        invalid_nested = {
            "concepts": [
                {
                    "concept_id": "test:concept:1",
                    "term": {
                        # primary отсутствует - обязательное поле во вложенном объекте
                        "aliases": ["alias1"]
                    },
                    "definition": "Определение",
                }
            ]
        }

        try:
            validate_json(invalid_nested, "ConceptDictionary")
            pytest.fail("Должна была быть ошибка валидации")
        except ValidationError as e:
            error_msg = str(e)
            # Проверяем, что путь ошибки правильно форматируется
            assert "concepts -> 0 -> term" in error_msg or "term" in error_msg
            assert "primary" in error_msg.lower()


class TestAllEdgeTypes:
    """Тесты для всех типов рёбер в графе знаний."""

    def setup_method(self):
        """Очистить кэш схем перед каждым тестом для изоляции."""
        from src.utils import validation

        validation._SCHEMA_CACHE.clear()

    def _create_test_graph_with_edge(self, edge_type: str, weight: float = 0.8):
        """Вспомогательный метод для создания тестового графа с определённым типом ребра."""
        return {
            "nodes": [
                {
                    "id": "test:chunk:1",
                    "type": "Chunk",
                    "text": "Первый чанк с базовой информацией",
                    "node_offset": 0,
                    "local_start": 0,
                    "difficulty": 2,
                },
                {
                    "id": "test:chunk:2",
                    "type": "Chunk",
                    "text": "Второй чанк с дополнительной информацией",
                    "node_offset": 0,
                    "local_start": 100,
                    "difficulty": 3,
                },
                {
                    "id": "test:concept:1",
                    "type": "Concept",
                    "text": "Важный концепт",
                    "node_offset": 0,
                    "local_start": 50,
                    "definition": "Определение важного концепта",
                },
                {
                    "id": "test:assessment:1",
                    "type": "Assessment",
                    "text": "Проверочный вопрос",
                    "node_offset": 0,
                    "local_start": 200,
                },
            ],
            "edges": [
                {
                    "source": "test:chunk:1",
                    "target": (
                        "test:chunk:2"
                        if edge_type != "MENTIONS" and edge_type != "TESTS"
                        else "test:concept:1"
                    ),
                    "type": edge_type,
                    "weight": weight,
                }
            ],
        }

    def test_prerequisite_edge(self):
        """Тест PREREQUISITE - необходимо понять A перед B (уже протестировано в базовых тестах)."""
        graph = self._create_test_graph_with_edge("PREREQUISITE", 0.9)
        # Не должно выбросить исключение
        validate_graph_invariants(graph)

    def test_elaborates_edge(self):
        """Тест ELABORATES - B детализирует A."""
        graph = self._create_test_graph_with_edge("ELABORATES", 0.85)
        # Не должно выбросить исключение
        validate_graph_invariants(graph)

    def test_example_of_edge(self):
        """Тест EXAMPLE_OF - A является примером B."""
        graph = self._create_test_graph_with_edge("EXAMPLE_OF", 0.75)
        # Не должно выбросить исключение
        validate_graph_invariants(graph)

    def test_hint_forward_edge(self):
        """Тест HINT_FORWARD - подсказка для перехода вперёд."""
        graph = self._create_test_graph_with_edge("HINT_FORWARD", 0.7)
        # Не должно выбросить исключение
        validate_graph_invariants(graph)

    def test_refer_back_edge(self):
        """Тест REFER_BACK - ссылка назад на предыдущий материал."""
        graph = self._create_test_graph_with_edge("REFER_BACK", 0.65)
        # Не должно выбросить исключение
        validate_graph_invariants(graph)

    def test_parallel_edge(self):
        """Тест PARALLEL - параллельные или альтернативные концепты."""
        graph = self._create_test_graph_with_edge("PARALLEL", 0.8)
        # Не должно выбросить исключение
        validate_graph_invariants(graph)

    def test_tests_edge(self):
        """Тест TESTS - Assessment проверяет знание материала."""
        graph = {
            "nodes": [
                {
                    "id": "test:chunk:1",
                    "type": "Chunk",
                    "text": "Учебный материал",
                    "node_offset": 0,
                    "local_start": 0,
                    "difficulty": 2,
                },
                {
                    "id": "test:assessment:1",
                    "type": "Assessment",
                    "text": "Вопрос по материалу",
                    "node_offset": 0,
                    "local_start": 100,
                },
            ],
            "edges": [
                {
                    "source": "test:assessment:1",
                    "target": "test:chunk:1",
                    "type": "TESTS",
                    "weight": 0.9,
                }
            ],
        }
        # Не должно выбросить исключение
        validate_graph_invariants(graph)

    def test_revision_of_edge(self):
        """Тест REVISION_OF - обновлённая версия материала."""
        graph = self._create_test_graph_with_edge("REVISION_OF", 0.95)
        # Не должно выбросить исключение
        validate_graph_invariants(graph)

    def test_mentions_edge(self):
        """Тест MENTIONS - фрагмент упоминает концепт (уже протестировано в базовых тестах)."""
        graph = {
            "nodes": [
                {
                    "id": "test:chunk:1",
                    "type": "Chunk",
                    "text": "Текст, упоминающий концепт",
                    "node_offset": 0,
                    "local_start": 0,
                    "difficulty": 2,
                },
                {
                    "id": "test:concept:1",
                    "type": "Concept",
                    "text": "Упоминаемый концепт",
                    "node_offset": 0,
                    "local_start": 50,
                    "definition": "Определение концепта",
                },
            ],
            "edges": [
                {
                    "source": "test:chunk:1",
                    "target": "test:concept:1",
                    "type": "MENTIONS",
                    "weight": 1.0,
                }
            ],
        }
        # Не должно выбросить исключение
        validate_graph_invariants(graph)

    def test_multiple_edge_types_in_graph(self):
        """Тест графа с несколькими типами рёбер одновременно."""
        complex_graph = {
            "nodes": [
                {
                    "id": "test:chunk:1",
                    "type": "Chunk",
                    "text": "Базовый материал",
                    "node_offset": 0,
                    "local_start": 0,
                    "difficulty": 1,
                },
                {
                    "id": "test:chunk:2",
                    "type": "Chunk",
                    "text": "Продвинутый материал",
                    "node_offset": 0,
                    "local_start": 100,
                    "difficulty": 3,
                },
                {
                    "id": "test:chunk:3",
                    "type": "Chunk",
                    "text": "Пример",
                    "node_offset": 0,
                    "local_start": 200,
                    "difficulty": 2,
                },
                {
                    "id": "test:concept:1",
                    "type": "Concept",
                    "text": "Ключевой концепт",
                    "node_offset": 0,
                    "local_start": 50,
                    "definition": "Определение",
                },
                {
                    "id": "test:assessment:1",
                    "type": "Assessment",
                    "text": "Тест",
                    "node_offset": 0,
                    "local_start": 300,
                },
            ],
            "edges": [
                {
                    "source": "test:chunk:1",
                    "target": "test:chunk:2",
                    "type": "PREREQUISITE",
                    "weight": 0.9,
                },
                {
                    "source": "test:chunk:2",
                    "target": "test:chunk:1",
                    "type": "ELABORATES",
                    "weight": 0.8,
                },
                {
                    "source": "test:chunk:3",
                    "target": "test:chunk:2",
                    "type": "EXAMPLE_OF",
                    "weight": 0.7,
                },
                {
                    "source": "test:chunk:1",
                    "target": "test:concept:1",
                    "type": "MENTIONS",
                    "weight": 1.0,
                },
                {
                    "source": "test:assessment:1",
                    "target": "test:chunk:2",
                    "type": "TESTS",
                    "weight": 0.85,
                },
                {
                    "source": "test:chunk:2",
                    "target": "test:chunk:3",
                    "type": "HINT_FORWARD",
                    "weight": 0.6,
                },
                {
                    "source": "test:chunk:3",
                    "target": "test:chunk:1",
                    "type": "REFER_BACK",
                    "weight": 0.65,
                },
            ],
        }
        # Не должно выбросить исключение
        validate_graph_invariants(complex_graph)


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

        with pytest.raises(ValidationError, match="primary term.*duplicated in aliases"):
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

    def test_empty_concept_dictionary(self):
        """Тест валидации пустого, но корректного словаря концептов."""
        empty_dict = {"concepts": []}

        # Не должно выбросить исключение - пустой словарь валиден
        validate_concept_dictionary_invariants(empty_dict)

    def test_case_insensitive_alias_check(self):
        """Тест что проверка уникальности алиасов внутри концепта нечувствительна к регистру."""
        invalid_dict = {
            "concepts": [
                {
                    "concept_id": "test:concept:1",
                    "term": {
                        "primary": "Функция",
                        "aliases": [
                            "function",
                            "Function",
                            "FUNCTION",
                        ],  # Дубликаты с разным регистром
                    },
                    "definition": "Определение функции",
                }
            ]
        }

        # Должна быть ошибка - алиасы дублируются несмотря на разный регистр
        with pytest.raises(ValidationError, match="duplicate alias"):
            validate_concept_dictionary_invariants(invalid_dict)

    def test_primary_term_case_variations(self):
        """Тест обнаружения primary термина в алиасах с различным регистром."""
        invalid_dict = {
            "concepts": [
                {
                    "concept_id": "test:concept:1",
                    "term": {
                        "primary": "Variable",
                        "aliases": [
                            "var",
                            "VARIABLE",
                            "переменная",
                        ],  # VARIABLE - тот же термин с другим регистром
                    },
                    "definition": "Определение переменной",
                }
            ]
        }

        # Должна быть ошибка - primary term дублируется в aliases с другим регистром
        with pytest.raises(ValidationError, match="primary term.*duplicated in aliases"):
            validate_concept_dictionary_invariants(invalid_dict)


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

        with pytest.raises(GraphInvariantError, match="Duplicate node ID \\(Chunk\\)"):
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

        with pytest.raises(GraphInvariantError, match="Duplicate node ID \\(Assessment\\)"):
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
            "edges": [{"source": "test:c:100", "target": "test:c:100", "type": "PREREQUISITE"}],
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

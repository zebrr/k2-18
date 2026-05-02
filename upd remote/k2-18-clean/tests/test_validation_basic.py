"""
Smoke test для модуля валидации - минимальная проверка работоспособности.
"""

import pytest

from src.utils.validation import (
    _load_schema,
    validate_concept_dictionary_invariants,
    validate_graph_invariants,
    validate_json,
)


def test_smoke():
    """Базовый smoke test - проверка, что модуль работает."""
    # 1. Проверка загрузки схем
    concept_schema = _load_schema("ConceptDictionary")
    assert concept_schema is not None
    assert "$schema" in concept_schema

    graph_schema = _load_schema("LearningChunkGraph")
    assert graph_schema is not None
    assert "$schema" in graph_schema

    # 2. Проверка валидации минимальных корректных данных
    minimal_concept_dict = {
        "concepts": [
            {
                "concept_id": "smoke:test:1",
                "term": {"primary": "Test"},
                "definition": "Test definition",
            }
        ]
    }

    validate_json(minimal_concept_dict, "ConceptDictionary")
    validate_concept_dictionary_invariants(minimal_concept_dict)

    minimal_graph = {
        "nodes": [
            {
                "id": "smoke:node:1",
                "type": "Chunk",
                "text": "Test text",
                "node_offset": 0,
                "local_start": 0,
            }
        ],
        "edges": [],
    }

    validate_json(minimal_graph, "LearningChunkGraph")
    validate_graph_invariants(minimal_graph)

    # Если мы дошли сюда - базовая функциональность работает


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

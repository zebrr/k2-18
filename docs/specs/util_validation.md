# util_validation.md

## Status: READY

Модуль валидации JSON Schema и инвариантов графа знаний. Обеспечивает проверку данных по схемам и бизнес-логики графовых структур.

## Public API

### validate_json(data: Dict[str, Any], schema_name: str) -> None
Валидирует данные по JSON Schema.
- **Input**: 
  - data - данные для валидации
  - schema_name - имя схемы без расширения (ConceptDictionary, LearningChunkGraph)
- **Returns**: None (успех) или выбрасывает исключение
- **Raises**: ValidationError - если данные не соответствуют схеме

### validate_graph_invariants(graph_data: Dict[str, Any]) -> None
Проверяет инварианты графа знаний после валидации по схеме.
- **Input**: graph_data - данные графа в формате LearningChunkGraph
- **Returns**: None (успех) или выбрасывает исключение
- **Raises**: GraphInvariantError - если нарушены инварианты графа
- **Проверки**: уникальность ID узлов, существование source/target, запрет PREREQUISITE self-loops, отсутствие дублированных рёбер

### validate_graph_invariants_intermediate(graph_data: Dict[str, Any]) -> None
Промежуточная валидация графа для использования в itext2kg.
Проверяет всё, КРОМЕ уникальности ID концептов.
- **Input**: graph_data - данные графа в формате LearningChunkGraph
- **Returns**: None (успех) или выбрасывает исключение
- **Raises**: GraphInvariantError - если нарушены инварианты графа
- **Проверки**: 
  - уникальность ID для узлов типа Chunk и Assessment
  - существование source/target
  - запрет PREREQUISITE self-loops
  - отсутствие дублированных рёбер
- **Примечание**: Используется при инкрементальной обработке, когда дубликаты концептов являются допустимыми и будут обработаны позже в dedup

### validate_concept_dictionary_invariants(concept_data: Dict[str, Any]) -> None
Проверяет инварианты словаря концептов после валидации по схеме.
- **Input**: concept_data - данные словаря в формате ConceptDictionary
- **Returns**: None (успех) или выбрасывает исключение
- **Raises**: ValidationError - если нарушены инварианты
- **Проверки**: 
  - Уникальность concept_id
  - Primary термин не дублируется в aliases того же концепта
  - Уникальность aliases внутри концепта (case-insensitive)
  - **Примечание**: Алиасы могут повторяться между разными концептами

### ValidationError(Exception)
Исключение для ошибок валидации данных.

### GraphInvariantError(ValidationError)
Исключение для ошибок инвариантов графа.

## Schema Loading

### _load_schema(schema_name: str) -> Dict[str, Any]
Внутренняя функция загрузки JSON Schema с кэшированием.
- **Input**: schema_name - имя схемы без расширения
- **Returns**: Dict - загруженная схема
- **Features**: кэширование схем, валидация самих схем через Draft202012Validator

## Validation Rules

### Graph Invariants
- Уникальность ID узлов в пределах графа
- Все source/target в рёбрах должны ссылаться на существующие узлы
- Запрещены PREREQUISITE self-loops (A→A)
- Запрещены дублированные рёбра (одинаковые source+target+type)
- Веса проверяются на уровне JSON Schema

### Concept Dictionary Invariants
- Уникальность concept_id в пределах словаря
- Primary термин не должен дублироваться в aliases того же концепта
- Уникальность aliases внутри концепта (case-insensitive)
- Primary термины могут повторяться между разными концептами
- Aliases могут повторяться между разными концептами

## Test Coverage

- **test_load_schema**: 3 теста
  - test_load_valid_schema
  - test_load_nonexistent_schema
  - test_load_invalid_json

- **test_validate_json**: 4 теста
  - test_valid_concept_dictionary
  - test_valid_learning_chunk_graph
  - test_invalid_concept_dictionary_missing_required
  - test_invalid_graph_wrong_edge_type

- **test_validate_graph_invariants**: 6 тестов
  - test_valid_graph_invariants
  - test_duplicate_node_ids
  - test_prerequisite_self_loop
  - test_nonexistent_edge_target
  - test_invalid_weight_range
  - test_duplicate_edges

- **test_validate_concept_dictionary**: 6 тестов
  - test_valid_concept_dictionary
  - test_duplicate_concept_ids
  - test_duplicate_primary_terms_allowed (проверяет что дубликаты разрешены)
  - test_primary_in_aliases
  - test_duplicate_aliases_within_concept
  - test_duplicate_aliases_across_concepts_allowed (проверяет что дубликаты разрешены)

## Dependencies
- **Standard Library**: json, pathlib, typing
- **External**: jsonschema
- **Internal**: использует схемы из /src/schemas/

## Performance Notes
- Схемы кэшируются после первой загрузки
- Валидация инвариантов выполняется после JSON Schema валидации
- Сложность проверки графа: O(nodes + edges)

## Usage Examples
```python
from src.utils.validation import (
    validate_json, 
    validate_graph_invariants,
    validate_concept_dictionary_invariants,
    ValidationError,
    GraphInvariantError
)

# Валидация по схеме
try:
    validate_json(concept_data, "ConceptDictionary")
    validate_json(graph_data, "LearningChunkGraph")
except ValidationError as e:
    print(f"Schema validation failed: {e}")

# Проверка инвариантов
try:
    validate_graph_invariants(graph_data)
    validate_concept_dictionary_invariants(concept_data)
except GraphInvariantError as e:
    print(f"Graph invariant violated: {e}")
```
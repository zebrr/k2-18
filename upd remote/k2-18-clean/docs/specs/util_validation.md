# util_validation.md

## Status: READY

JSON Schema validation and knowledge graph invariants module. Provides data validation against schemas and business logic verification for graph structures.

## Public API

### validate_json(data: Dict[str, Any], schema_name: str) -> None
Validates data against JSON Schema.
- **Input**: 
  - data - data to validate
  - schema_name - schema name without extension (ConceptDictionary, LearningChunkGraph)
- **Returns**: None (success) or raises exception
- **Raises**: ValidationError - if data does not match schema

### validate_graph_invariants(graph_data: Dict[str, Any]) -> None
Checks knowledge graph invariants after schema validation.
- **Input**: graph_data - graph data in LearningChunkGraph format
- **Returns**: None (success) or raises exception
- **Raises**: GraphInvariantError - if graph invariants are violated
- **Checks**: node ID uniqueness, source/target existence, PREREQUISITE self-loop prohibition, no duplicate edges

### validate_graph_invariants_intermediate(graph_data: Dict[str, Any]) -> None
Intermediate graph validation for use in itext2kg.
Checks everything EXCEPT concept ID uniqueness.
- **Input**: graph_data - graph data in LearningChunkGraph format
- **Returns**: None (success) or raises exception
- **Raises**: GraphInvariantError - if graph invariants are violated
- **Checks**: 
  - ID uniqueness for Chunk and Assessment node types
  - source/target existence
  - PREREQUISITE self-loop prohibition
  - no duplicate edges
- **Note**: Used during incremental processing when concept duplicates are acceptable and will be processed later in dedup

### validate_concept_dictionary_invariants(concept_data: Dict[str, Any]) -> None
Checks concept dictionary invariants after schema validation.
- **Input**: concept_data - dictionary data in ConceptDictionary format
- **Returns**: None (success) or raises exception
- **Raises**: ValidationError - if invariants are violated
- **Checks**: 
  - concept_id uniqueness
  - Primary term not duplicated in aliases of the same concept
  - Alias uniqueness within concept (case-insensitive)
  - **Note**: Aliases may repeat between different concepts

### ValidationError(Exception)
Data validation error exception.

### GraphInvariantError(ValidationError)
Graph invariant error exception.

## Schema Loading

### _load_schema(schema_name: str) -> Dict[str, Any]
Internal function for loading JSON Schema with caching.
- **Input**: schema_name - schema name without extension
- **Returns**: Dict - loaded schema
- **Features**: schema caching, schema validation via Draft202012Validator

## Validation Rules

### Graph Invariants
- Node ID uniqueness within graph
- All source/target in edges must reference existing nodes
- PREREQUISITE self-loops (A→A) are prohibited
- Duplicate edges are prohibited (same source+target+type)
- Weights are validated at JSON Schema level

### Concept Dictionary Invariants
- concept_id uniqueness within dictionary
- Primary term must not be duplicated in aliases of the same concept
- Alias uniqueness within concept (case-insensitive)
- Primary terms may repeat between different concepts
- Aliases may repeat between different concepts

## Test Coverage

- **test_load_schema**: 3 tests
  - test_load_valid_schema
  - test_load_nonexistent_schema
  - test_load_invalid_json

- **test_validate_json**: 4 tests
  - test_valid_concept_dictionary
  - test_valid_learning_chunk_graph
  - test_invalid_concept_dictionary_missing_required
  - test_invalid_graph_wrong_edge_type

- **test_validate_graph_invariants**: 6 tests
  - test_valid_graph_invariants
  - test_duplicate_node_ids
  - test_prerequisite_self_loop
  - test_nonexistent_edge_target
  - test_invalid_weight_range
  - test_duplicate_edges

- **test_validate_concept_dictionary**: 6 tests
  - test_valid_concept_dictionary
  - test_duplicate_concept_ids
  - test_duplicate_primary_terms_allowed (verifies duplicates are allowed)
  - test_primary_in_aliases
  - test_duplicate_aliases_within_concept
  - test_duplicate_aliases_across_concepts_allowed (verifies duplicates are allowed)

## Dependencies
- **Standard Library**: json, pathlib, typing
- **External**: jsonschema
- **Internal**: uses schemas from /src/schemas/

## Performance Notes
- Schemas are cached after first load
- Invariant validation is performed after JSON Schema validation
- Graph checking complexity: O(nodes + edges)

## Usage Examples
```python
from src.utils.validation import (
    validate_json, 
    validate_graph_invariants,
    validate_concept_dictionary_invariants,
    ValidationError,
    GraphInvariantError
)

# Schema validation
try:
    validate_json(concept_data, "ConceptDictionary")
    validate_json(graph_data, "LearningChunkGraph")
except ValidationError as e:
    print(f"Schema validation failed: {e}")

# Invariant checking
try:
    validate_graph_invariants(graph_data)
    validate_concept_dictionary_invariants(concept_data)
except GraphInvariantError as e:
    print(f"Graph invariant violated: {e}")
```
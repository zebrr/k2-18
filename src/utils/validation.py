"""
Module for JSON Schema validation and knowledge graph invariants.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Set

import jsonschema
from jsonschema import ValidationError

__all__ = [
    "ValidationError",
    "GraphInvariantError",
    "validate_json",
    "validate_graph_invariants",
    "validate_graph_invariants_intermediate",  # NEW
    "validate_concept_dictionary_invariants",
]


class ValidationError(Exception):
    """Data validation error."""

    pass


class GraphInvariantError(ValidationError):
    """Graph invariant error."""

    pass


# Cache for loaded schemas
_SCHEMA_CACHE: Dict[str, Dict] = {}


def _load_schema(schema_name: str) -> Dict[str, Any]:
    """
    Loads JSON Schema from file.

    Args:
        schema_name: Schema name without extension (e.g., 'ConceptDictionary')

    Returns:
        Dictionary with JSON Schema

    Raises:
        FileNotFoundError: If schema file is not found
        ValidationError: If schema is invalid
    """
    if schema_name in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[schema_name]

    # Path to schemas relative to current file
    schema_path = (
        Path(__file__).parent.parent / "schemas" / f"{schema_name}.schema.json"
    )

    if not schema_path.exists():
        raise FileNotFoundError(f"JSON Schema not found: {schema_path}")

    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        # Check that the schema itself is valid
        jsonschema.Draft202012Validator.check_schema(schema)

        _SCHEMA_CACHE[schema_name] = schema
        return schema

    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON in schema {schema_name}: {e}")
    except jsonschema.SchemaError as e:
        raise ValidationError(f"Invalid JSON Schema {schema_name}: {e}")


def validate_json(data: Dict[str, Any], schema_name: str) -> None:
    """
    Validates data against JSON Schema.

    Args:
        data: Data to validate
        schema_name: Schema name without extension

    Raises:
        ValidationError: If data does not match the schema
    """
    schema = _load_schema(schema_name)

    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as e:
        # Format a clear error message
        error_path = (
            " -> ".join(str(p) for p in e.absolute_path)
            if e.absolute_path
            else "root"
        )
        raise ValidationError(
            f"Schema validation error '{schema_name}' in field '{error_path}': {e.message}"
        )


def validate_graph_invariants(graph_data: Dict[str, Any]) -> None:
    """
    Checks knowledge graph invariants.

    Args:
        graph_data: Graph data in LearningChunkGraph format

    Raises:
        GraphInvariantError: If graph invariants are violated
    """
    # First validate against schema
    validate_json(graph_data, "LearningChunkGraph")

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    # Collect all node IDs
    node_ids: Set[str] = set()
    for node in nodes:
        node_id = node.get("id")
        if not node_id:
            raise GraphInvariantError("Found node without ID")

        if node_id in node_ids:
            raise GraphInvariantError(f"Duplicate node ID: {node_id}")

        node_ids.add(node_id)

    # Check edges
    edge_keys: Set[tuple] = set()
    prerequisite_edges: List[tuple] = []

    for i, edge in enumerate(edges):
        source = edge.get("source")
        target = edge.get("target")
        edge_type = edge.get("type")
        weight = edge.get("weight")

        # Check existence of source and target
        if source not in node_ids:
            raise GraphInvariantError(f"Edge {i}: source '{source}' does not exist")

        if target not in node_ids:
            raise GraphInvariantError(f"Edge {i}: target '{target}' does not exist")

        # Weight validation is performed at JSON Schema level
        # Check for PREREQUISITE self-loops
        if edge_type == "PREREQUISITE" and source == target:
            raise GraphInvariantError(
                f"Edge {i}: PREREQUISITE self-loop forbidden {source} -> {target}"
            )

        # Check for duplicate edges
        edge_key = (source, target, edge_type)
        if edge_key in edge_keys:
            raise GraphInvariantError(
                f"Edge {i}: duplicate edge {source} -> {target} ({edge_type})"
            )

        edge_keys.add(edge_key)

        # Collect PREREQUISITE edges for cycle checking (if needed in the future)
        if edge_type == "PREREQUISITE":
            prerequisite_edges.append((source, target))


def validate_graph_invariants_intermediate(graph_data: Dict[str, Any]) -> None:
    """
    Intermediate graph validation for use in itext2kg.
    Checks everything EXCEPT concept ID uniqueness.

    Used during incremental processing, when concept duplicates
    are acceptable and will be processed later in dedup.

    Args:
        graph_data: Graph data in LearningChunkGraph format

    Raises:
        GraphInvariantError: If graph invariants are violated
    """
    # First validate against schema
    validate_json(graph_data, "LearningChunkGraph")

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    # Collect all node IDs and check uniqueness (except concepts)
    node_ids: Set[str] = set()
    chunk_assessment_ids: Set[str] = set()

    for node in nodes:
        node_id = node.get("id")
        node_type = node.get("type")

        if not node_id:
            raise GraphInvariantError("Found node without ID")

        # Add all nodes to common set (for edge checking)
        node_ids.add(node_id)

        # Check uniqueness only for Chunk and Assessment
        if node_type in ("Chunk", "Assessment"):
            if node_id in chunk_assessment_ids:
                raise GraphInvariantError(
                    f"Duplicate node ID ({node_type}): {node_id}"
                )
            chunk_assessment_ids.add(node_id)

    # Check edges (same logic as in main function)
    edge_keys: Set[tuple] = set()

    for i, edge in enumerate(edges):
        source = edge.get("source")
        target = edge.get("target")
        edge_type = edge.get("type")

        # Check existence of source and target
        if source not in node_ids:
            raise GraphInvariantError(f"Edge {i}: source '{source}' does not exist")

        if target not in node_ids:
            raise GraphInvariantError(f"Edge {i}: target '{target}' does not exist")

        # Check for PREREQUISITE self-loops
        if edge_type == "PREREQUISITE" and source == target:
            raise GraphInvariantError(
                f"Edge {i}: PREREQUISITE self-loop forbidden {source} -> {target}"
            )

        # Check for duplicate edges
        edge_key = (source, target, edge_type)
        if edge_key in edge_keys:
            raise GraphInvariantError(
                f"Edge {i}: duplicate edge {source} -> {target} ({edge_type})"
            )

        edge_keys.add(edge_key)


def validate_concept_dictionary_invariants(concept_data: Dict[str, Any]) -> None:
    """
    Checks concept dictionary invariants.

    Args:
        concept_data: Dictionary data in ConceptDictionary format

    Raises:
        ValidationError: If invariants are violated
    """
    # First validate against schema
    validate_json(concept_data, "ConceptDictionary")

    concepts = concept_data.get("concepts", [])

    # Check concept_id uniqueness
    concept_ids: Set[str] = set()

    for i, concept in enumerate(concepts):
        concept_id = concept.get("concept_id")

        if concept_id in concept_ids:
            raise ValidationError(
                f"Concept {i}: duplicate concept_id '{concept_id}'"
            )

        concept_ids.add(concept_id)

        # Check terms
        term = concept.get("term", {})
        primary = term.get("primary")
        aliases = term.get("aliases", [])

        if primary:
            # if primary in primary_terms:
            #   raise ValidationError(f"Concept {i}: duplicate primary term '{primary}'")
            # primary_terms.add(primary.lower())

            # Check that primary does not repeat in aliases
            if primary.lower() in [alias.lower() for alias in aliases]:
                raise ValidationError(
                    f"Concept {i}: primary term '{primary}' duplicated in aliases"
                )

        # Check aliases for duplicates WITHIN the concept
        alias_set = set()
        for alias in aliases:
            alias_lower = alias.lower()
            if alias_lower in alias_set:
                raise ValidationError(f"Concept {i}: duplicate alias '{alias}'")

            alias_set.add(alias_lower)
            # Removed all_aliases check - aliases can repeat between concepts

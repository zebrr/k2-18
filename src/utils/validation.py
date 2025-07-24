"""
Модуль валидации JSON Schema и инвариантов графа знаний.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Set
import jsonschema
from jsonschema import ValidationError

__all__ = [
    'ValidationError',
    'GraphInvariantError', 
    'validate_json',
    'validate_graph_invariants',
    'validate_graph_invariants_intermediate',  # НОВОЕ
    'validate_concept_dictionary_invariants'
]

class ValidationError(Exception):
    """Ошибка валидации данных."""
    pass


class GraphInvariantError(ValidationError):
    """Ошибка инвариантов графа."""
    pass


# Кэш загруженных схем
_SCHEMA_CACHE: Dict[str, Dict] = {}


def _load_schema(schema_name: str) -> Dict[str, Any]:
    """
    Загружает JSON Schema из файла.
    
    Args:
        schema_name: Имя схемы без расширения (например, 'ConceptDictionary')
        
    Returns:
        Словарь с JSON Schema
        
    Raises:
        FileNotFoundError: Если файл схемы не найден
        ValidationError: Если схема некорректна
    """
    if schema_name in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[schema_name]
    
    # Путь к схемам относительно текущего файла
    schema_path = Path(__file__).parent.parent / "schemas" / f"{schema_name}.schema.json"
    
    if not schema_path.exists():
        raise FileNotFoundError(f"JSON Schema не найдена: {schema_path}")
    
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        
        # Проверяем, что сама схема валидна
        jsonschema.Draft202012Validator.check_schema(schema)
        
        _SCHEMA_CACHE[schema_name] = schema
        return schema
        
    except json.JSONDecodeError as e:
        raise ValidationError(f"Некорректный JSON в схеме {schema_name}: {e}")
    except jsonschema.SchemaError as e:
        raise ValidationError(f"Некорректная JSON Schema {schema_name}: {e}")


def validate_json(data: Dict[str, Any], schema_name: str) -> None:
    """
    Валидирует данные по JSON Schema.
    
    Args:
        data: Данные для валидации
        schema_name: Имя схемы без расширения
        
    Raises:
        ValidationError: Если данные не соответствуют схеме
    """
    schema = _load_schema(schema_name)
    
    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as e:
        # Формируем понятное сообщение об ошибке
        error_path = " -> ".join(str(p) for p in e.absolute_path) if e.absolute_path else "корень"
        raise ValidationError(
            f"Ошибка валидации схемы '{schema_name}' в поле '{error_path}': {e.message}"
        )


def validate_graph_invariants(graph_data: Dict[str, Any]) -> None:
    """
    Проверяет инварианты графа знаний.
    
    Args:
        graph_data: Данные графа в формате LearningChunkGraph
        
    Raises:
        GraphInvariantError: Если нарушены инварианты графа
    """
    # Сначала валидируем по схеме
    validate_json(graph_data, "LearningChunkGraph")
    
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    
    # Собираем ID всех узлов
    node_ids: Set[str] = set()
    for node in nodes:
        node_id = node.get("id")
        if not node_id:
            raise GraphInvariantError("Найден узел без ID")
        
        if node_id in node_ids:
            raise GraphInvariantError(f"Дублированный ID узла: {node_id}")
        
        node_ids.add(node_id)
    
    # Проверяем рёбра
    edge_keys: Set[tuple] = set()
    prerequisite_edges: List[tuple] = []
    
    for i, edge in enumerate(edges):
        source = edge.get("source")
        target = edge.get("target")
        edge_type = edge.get("type")
        weight = edge.get("weight")
        
        # Проверка существования source и target
        if source not in node_ids:
            raise GraphInvariantError(f"Ребро {i}: source '{source}' не существует")
        
        if target not in node_ids:
            raise GraphInvariantError(f"Ребро {i}: target '{target}' не существует")
        
        # Проверка весов выполняется на уровне JSON Schema
        # Проверка на PREREQUISITE self-loops
        if edge_type == "PREREQUISITE" and source == target:
            raise GraphInvariantError(f"Ребро {i}: запрещён PREREQUISITE self-loop {source} -> {target}")
        
        # Проверка дублированных рёбер
        edge_key = (source, target, edge_type)
        if edge_key in edge_keys:
            raise GraphInvariantError(f"Ребро {i}: дублированное ребро {source} -> {target} ({edge_type})")
        
        edge_keys.add(edge_key)
        
        # Собираем PREREQUISITE рёбра для проверки циклов (если понадобится в будущем)
        if edge_type == "PREREQUISITE":
            prerequisite_edges.append((source, target))


def validate_graph_invariants_intermediate(graph_data: Dict[str, Any]) -> None:
    """
    Промежуточная валидация графа для использования в itext2kg.
    Проверяет всё, КРОМЕ уникальности ID концептов.
    
    Используется при инкрементальной обработке, когда дубликаты концептов
    являются допустимыми и будут обработаны позже в dedup.
    
    Args:
        graph_data: Данные графа в формате LearningChunkGraph
        
    Raises:
        GraphInvariantError: Если нарушены инварианты графа
    """
    # Сначала валидируем по схеме
    validate_json(graph_data, "LearningChunkGraph")
    
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    
    # Собираем ID всех узлов и проверяем уникальность (кроме концептов)
    node_ids: Set[str] = set()
    chunk_assessment_ids: Set[str] = set()
    
    for node in nodes:
        node_id = node.get("id")
        node_type = node.get("type")
        
        if not node_id:
            raise GraphInvariantError("Найден узел без ID")
        
        # Для всех узлов добавляем в общий набор (для проверки рёбер)
        node_ids.add(node_id)
        
        # Проверяем уникальность только для Chunk и Assessment
        if node_type in ("Chunk", "Assessment"):
            if node_id in chunk_assessment_ids:
                raise GraphInvariantError(f"Дублированный ID узла ({node_type}): {node_id}")
            chunk_assessment_ids.add(node_id)
    
    # Проверяем рёбра (та же логика, что и в основной функции)
    edge_keys: Set[tuple] = set()
    
    for i, edge in enumerate(edges):
        source = edge.get("source")
        target = edge.get("target")
        edge_type = edge.get("type")
        
        # Проверка существования source и target
        if source not in node_ids:
            raise GraphInvariantError(f"Ребро {i}: source '{source}' не существует")
        
        if target not in node_ids:
            raise GraphInvariantError(f"Ребро {i}: target '{target}' не существует")
        
        # Проверка на PREREQUISITE self-loops
        if edge_type == "PREREQUISITE" and source == target:
            raise GraphInvariantError(f"Ребро {i}: запрещён PREREQUISITE self-loop {source} -> {target}")
        
        # Проверка дублированных рёбер
        edge_key = (source, target, edge_type)
        if edge_key in edge_keys:
            raise GraphInvariantError(f"Ребро {i}: дублированное ребро {source} -> {target} ({edge_type})")
        
        edge_keys.add(edge_key)


def validate_concept_dictionary_invariants(concept_data: Dict[str, Any]) -> None:
    """
    Проверяет инварианты словаря концептов.
    
    Args:
        concept_data: Данные словаря в формате ConceptDictionary
        
    Raises:
        ValidationError: Если нарушены инварианты
    """
    # Сначала валидируем по схеме
    validate_json(concept_data, "ConceptDictionary")
    
    concepts = concept_data.get("concepts", [])
    
    # Проверяем уникальность concept_id
    concept_ids: Set[str] = set()
    
    for i, concept in enumerate(concepts):
        concept_id = concept.get("concept_id")
        
        if concept_id in concept_ids:
            raise ValidationError(f"Концепт {i}: дублированный concept_id '{concept_id}'")
        
        concept_ids.add(concept_id)
        
        # Проверяем термы
        term = concept.get("term", {})
        primary = term.get("primary")
        aliases = term.get("aliases", [])
        
        if primary:
            # if primary in primary_terms:
            #   raise ValidationError(f"Концепт {i}: дублированный primary термин '{primary}'")
            # primary_terms.add(primary.lower())
            
            # Проверяем, что primary не повторяется в aliases
            if primary.lower() in [alias.lower() for alias in aliases]:
                raise ValidationError(f"Концепт {i}: primary термин '{primary}' дублируется в aliases")
        
        # Проверяем aliases на дубликаты ВНУТРИ концепта
        alias_set = set()
        for alias in aliases:
            alias_lower = alias.lower()
            if alias_lower in alias_set:
                raise ValidationError(f"Концепт {i}: дублированный alias '{alias}'")
            
            alias_set.add(alias_lower)
            # Убрали проверку all_aliases - алиасы могут повторяться между концептами
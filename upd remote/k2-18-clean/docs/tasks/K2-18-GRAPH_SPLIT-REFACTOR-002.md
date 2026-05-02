# K2-18-GRAPH_SPLIT-REFACTOR-002: Add cluster_id to cluster dictionary concepts

## References

**ОБЯЗАТЕЛЬНО изучить перед началом работы:**
- `docs/specs/viz_graph_split.md` — спецификация модуля (обновить после изменений)
- `viz/graph_split.py` — текущая реализация
- `tests/viz/test_graph_split.py` — существующие тесты
- `src/schemas/ConceptDictionary.schema.json` — схема словаря (минимальное изменение)

## Context

При разбиении графа на кластеры создаются словари концептов для каждого кластера. Сейчас концепты копируются из исходного словаря без информации о принадлежности к кластеру.

**Задача:** При формировании словаря кластера добавлять каждому концепту поле `cluster_id`, которое берётся из соответствующей вершины графа (type=Concept).

**Ключевые моменты:**
- `concept_id` в словаре === `id` вершины типа Concept в графе
- У вершины графа есть `cluster_id` после graph2metrics
- Нужно добавлять `cluster_id` ТОЛЬКО если его ещё нет в концепте (forward-compatible)
- Если вершина не найдена в графе — записываем `null` (не пропускаем поле)

**Формат записи концепта после изменений:**
```json
{
  "concept_id": "handbook_osnovy_algoritmov:p:algoritm",
  "cluster_id": 3,
  "term": {"primary": "Алгоритм", "aliases": [...]},
  "definition": "..."
}
```

Если концепт не найден в графе:
```json
{
  "concept_id": "handbook_osnovy_algoritmov:p:unknown",
  "cluster_id": null,
  "term": {...},
  "definition": "..."
}
```

## Steps

### 0. Изменение схемы `src/schemas/ConceptDictionary.schema.json`

Разрешить дополнительные поля в объекте concept (строка 36):

```json
// Было:
"additionalProperties": false

// Стало:
"additionalProperties": true
```

Это позволит добавлять `cluster_id` без нарушения валидации.

### 1. Модификация `viz/graph_split.py`

#### 1.1 Добавить функцию построения маппинга

```python
def build_concept_cluster_map(graph_data: Dict) -> Dict[str, int]:
    """Build mapping from concept_id to cluster_id.
    
    Args:
        graph_data: Full graph with nodes
        
    Returns:
        Dictionary mapping concept node id to its cluster_id.
        Only includes nodes with type="Concept".
    """
    concept_map = {}
    for node in graph_data.get("nodes", []):
        if node.get("type") == "Concept" and "cluster_id" in node:
            concept_map[node["id"]] = node["cluster_id"]
    return concept_map
```

#### 1.2 Модифицировать `extract_cluster_concepts`

Добавить параметр `concept_cluster_map: Dict[str, int]` и логику добавления `cluster_id`:

```python
def extract_cluster_concepts(
    cluster_nodes: List[Dict], 
    concepts_data: Dict, 
    concept_cluster_map: Dict[str, int],  # NEW PARAMETER
    logger: logging.Logger
) -> Tuple[List[Dict], int]:
    """Extract concepts referenced by cluster nodes.
    
    Args:
        cluster_nodes: List of nodes in the cluster
        concepts_data: Full dictionary data with concepts array
        concept_cluster_map: Mapping {concept_id: cluster_id} from graph
        logger: Logger instance
        
    Returns:
        Tuple of (concepts_list, count)
    """
    # 1. Collect unique concept IDs from all nodes' concepts: [] field
    concept_ids = set()
    for node in cluster_nodes:
        concept_ids.update(node.get("concepts", []))

    # 2. Build lookup map from concepts_data
    concept_lookup = {c["concept_id"]: c for c in concepts_data.get("concepts", [])}

    # 3. Extract matching concepts with cluster_id
    concepts_list = []
    for cid in sorted(concept_ids):
        if cid in concept_lookup:
            # Make a copy to avoid modifying original
            concept_copy = concept_lookup[cid].copy()
            
            # Add cluster_id only if not already present (forward-compatible)
            if "cluster_id" not in concept_copy:
                # Use null if not found in graph
                concept_copy["cluster_id"] = concept_cluster_map.get(cid)
            
            concepts_list.append(concept_copy)
        else:
            logger.warning(f"Concept {cid} not found in dictionary")

    return concepts_list, len(concepts_list)
```

#### 1.3 Обновить вызов в `main()`

В `main()` после загрузки графа построить маппинг и передать в `extract_cluster_concepts`:

```python
# После загрузки графа, перед циклом по кластерам:
concept_cluster_map = build_concept_cluster_map(graph_data)
logger.info(f"Built concept cluster map: {len(concept_cluster_map)} concepts")

# В цикле по кластерам обновить вызов:
concepts_list, concepts_count = extract_cluster_concepts(
    cluster_graph["nodes"], 
    dictionary_data, 
    concept_cluster_map,  # NEW ARGUMENT
    logger
)
```

### 2. Обновить спецификацию `docs/specs/viz_graph_split.md`

#### 2.1 Обновить секцию "Extract Cluster Dictionary"

Добавить описание поля `cluster_id`:

```markdown
#### 8. Extract Cluster Dictionary
For each cluster, create a dictionary file with concepts used by nodes in that cluster:
1. Collect all unique concept IDs from `concepts: []` field of all nodes in cluster
2. Build concept_cluster_map from graph nodes (type=Concept) for cluster_id lookup
3. Look up each concept ID in source `ConceptDictionary_wow.json`
4. For each concept, add `cluster_id` field:
   - If concept_id matches a Concept node in graph → use node's cluster_id
   - If not found → set to `null`
   - If already has cluster_id → preserve existing value (forward-compatible)
5. Create cluster dictionary with format:
   ```json
   {
     "_meta": {
       "title": "<original title from graph>",
       "cluster_id": 12,
       "concepts_used": 23
     },
     "concepts": [
       { 
         "concept_id": "...",
         "cluster_id": 3,  // or null if not found
         "term": {...},
         "definition": "..."
       }
     ]
   }
   ```
```

#### 2.2 Добавить в Public API новую функцию

```markdown
### build_concept_cluster_map(graph_data: Dict) -> Dict[str, int]
Build mapping from concept node IDs to their cluster_id values.
- **Input**:
  - graph_data (Dict) - Full graph with nodes
- **Returns**: Dictionary {concept_id: cluster_id} for all Concept nodes
- **Algorithm**: Filter nodes by type=="Concept", extract id → cluster_id pairs
```

#### 2.3 Обновить сигнатуру `extract_cluster_concepts`

```markdown
### extract_cluster_concepts(cluster_nodes: List[Dict], concepts_data: Dict, concept_cluster_map: Dict[str, int], logger: Logger) -> Tuple[List[Dict], int]
Extract concepts used by nodes in a cluster, enriched with cluster_id.
- **Input**:
  - cluster_nodes (List[Dict]) - Nodes belonging to cluster
  - concepts_data (Dict) - Full concept dictionary
  - concept_cluster_map (Dict[str, int]) - Mapping {concept_id: cluster_id} from graph
  - logger (Logger) - Logger instance
- **Returns**: Tuple of (concepts_list, concepts_count)
- **Algorithm**:
  1. Collect all unique concept IDs from `concepts: []` field of all nodes
  2. Build lookup map from concepts_data by concept_id
  3. For each ID, find concept in dictionary
  4. Add cluster_id from concept_cluster_map (null if not found, skip if already present)
  5. Log WARNING for missing concepts
  6. Return list of enriched concept objects
```

### 3. Обновить тесты `tests/viz/test_graph_split.py`

#### 3.1 Добавить тест для `build_concept_cluster_map`

```python
def test_build_concept_cluster_map():
    """Test building concept_id -> cluster_id mapping."""
    graph_data = {
        "nodes": [
            {"id": "concept_1", "type": "Concept", "cluster_id": 0},
            {"id": "concept_2", "type": "Concept", "cluster_id": 1},
            {"id": "chunk_1", "type": "Chunk", "cluster_id": 0},  # Should be excluded
            {"id": "concept_3", "type": "Concept"},  # No cluster_id - should be excluded
        ]
    }
    
    result = build_concept_cluster_map(graph_data)
    
    assert result == {"concept_1": 0, "concept_2": 1}
    assert "chunk_1" not in result
    assert "concept_3" not in result
```

#### 3.2 Добавить тест для cluster_id в extract_cluster_concepts

```python
def test_extract_cluster_concepts_with_cluster_id():
    """Test that cluster_id is added to concepts from graph."""
    cluster_nodes = [
        {"id": "chunk_1", "concepts": ["concept_a", "concept_b"]}
    ]
    concepts_data = {
        "concepts": [
            {"concept_id": "concept_a", "term": {"primary": "A"}, "definition": "Def A"},
            {"concept_id": "concept_b", "term": {"primary": "B"}, "definition": "Def B"},
        ]
    }
    concept_cluster_map = {"concept_a": 5, "concept_b": 3}
    
    result, count = extract_cluster_concepts(
        cluster_nodes, concepts_data, concept_cluster_map, mock_logger
    )
    
    assert count == 2
    assert result[0]["cluster_id"] == 5  # concept_a -> cluster 5
    assert result[1]["cluster_id"] == 3  # concept_b -> cluster 3
```

#### 3.3 Добавить тест для null когда концепт не в графе

```python
def test_extract_cluster_concepts_missing_in_graph():
    """Test that cluster_id is null when concept not found in graph."""
    cluster_nodes = [
        {"id": "chunk_1", "concepts": ["concept_a", "concept_b"]}
    ]
    concepts_data = {
        "concepts": [
            {"concept_id": "concept_a", "term": {"primary": "A"}, "definition": "Def A"},
            {"concept_id": "concept_b", "term": {"primary": "B"}, "definition": "Def B"},
        ]
    }
    # concept_b not in map
    concept_cluster_map = {"concept_a": 5}
    
    result, count = extract_cluster_concepts(
        cluster_nodes, concepts_data, concept_cluster_map, mock_logger
    )
    
    assert count == 2
    assert result[0]["cluster_id"] == 5
    assert result[1]["cluster_id"] is None  # concept_b not in graph
```

#### 3.4 Добавить тест для forward-compatibility

```python
def test_extract_cluster_concepts_preserves_existing_cluster_id():
    """Test that existing cluster_id is not overwritten (forward-compatible)."""
    cluster_nodes = [
        {"id": "chunk_1", "concepts": ["concept_a"]}
    ]
    concepts_data = {
        "concepts": [
            {"concept_id": "concept_a", "cluster_id": 99, "term": {"primary": "A"}, "definition": "Def A"},
        ]
    }
    concept_cluster_map = {"concept_a": 5}  # Different value
    
    result, count = extract_cluster_concepts(
        cluster_nodes, concepts_data, concept_cluster_map, mock_logger
    )
    
    assert count == 1
    assert result[0]["cluster_id"] == 99  # Preserved original, not 5
```

#### 3.5 Обновить существующие тесты

Все существующие тесты, вызывающие `extract_cluster_concepts`, должны передавать `concept_cluster_map` параметр. Добавь пустой словарь `{}` если cluster_id не важен для теста.

**Найти и обновить:**
- `test_extract_cluster_concepts`
- `test_extract_cluster_concepts_missing`
- `test_cluster_with_no_concepts`
- Любые интеграционные тесты использующие эту функцию

## Testing

**ВАЖНО: Выполнять проверки качества ПЕРЕД тестами!**

```bash
# 1. Активировать venv
source .venv/bin/activate

# 2. Проверка качества кода (ОБЯЗАТЕЛЬНО ПЕРЕД тестами)
ruff check viz/graph_split.py tests/viz/test_graph_split.py
ruff format viz/graph_split.py tests/viz/test_graph_split.py
mypy viz/graph_split.py

# 3. Запуск тестов модуля
pytest tests/viz/test_graph_split.py -v

# 4. Проверка на реальных данных (если есть)
python -m viz.graph_split

# 5. Проверить что cluster_id появился в выходных файлах
python -c "
import json
from pathlib import Path

# Check first cluster dictionary
dict_files = list(Path('viz/data/out').glob('*_cluster_*_dict.json'))
if dict_files:
    d = json.load(open(dict_files[0]))
    concepts = d.get('concepts', [])
    if concepts:
        print(f'First concept: {concepts[0].get(\"concept_id\")}')
        print(f'cluster_id: {concepts[0].get(\"cluster_id\")}')
        print(f'cluster_id field present: {\"cluster_id\" in concepts[0]}')
"
```

**Ожидаемые результаты:**
- Все тесты проходят
- ruff check без ошибок
- mypy без ошибок типизации
- В выходных словарях кластеров у каждого концепта есть поле `cluster_id`

## Deliverables

1. **Изменённая схема:** `src/schemas/ConceptDictionary.schema.json`
   - `additionalProperties: true` в объекте concept

2. **Модифицированный файл:** `viz/graph_split.py`
   - Новая функция `build_concept_cluster_map`
   - Обновлённая функция `extract_cluster_concepts` с параметром `concept_cluster_map`
   - Обновлённый вызов в `main()`

3. **Обновлённая спецификация:** `docs/specs/viz_graph_split.md`
   - Описание нового поведения
   - Новая функция в Public API
   - Обновлённая сигнатура `extract_cluster_concepts`

4. **Обновлённые тесты:** `tests/viz/test_graph_split.py`
   - 4 новых теста для cluster_id логики
   - Обновлённые существующие тесты с новым параметром

5. **Отчёт:** `docs/tasks/K2-18-GRAPH_SPLIT-REFACTOR-002_REPORT.md`

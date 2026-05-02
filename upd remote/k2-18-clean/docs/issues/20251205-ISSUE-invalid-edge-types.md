# ISSUE: Invalid Edge Types in Graph Extraction

## Problem Statement

При анализе графа знаний после полного прогона pipeline обнаружено 19+ типов рёбер вместо разрешённых 9.

### Масштаб проблемы

- **Рёбра имеют невалидные типы**
- Наиболее частые нарушители:
  - `RELATED` — 320 рёбер
  - `ASSOCIATED` — 338 рёбер  
  - `REQUIRES`, `DEPENDS_ON`, `USES` — десятки рёбер каждого типа
  - `EXEMPLIFIES`, `LEADS_TO`, `SEQUENTIAL` и другие

### Допустимые типы (из схемы)

```json
"enum": [
  "PREREQUISITE", "ELABORATES", "EXAMPLE_OF",
  "HINT_FORWARD", "REFER_BACK", "PARALLEL",
  "TESTS", "REVISION_OF", "MENTIONS"
]
```

---

## Root Cause Analysis

### 1. Отсутствие валидации enum в коде

**Файл:** `src/itext2kg_graph.py`, метод `_validate_edges()` (строки ~415-455)

Метод проверяет:
- ✅ PREREQUISITE self-loops
- ✅ Weight range [0,1]
- ✅ Node existence
- ✅ Duplicate edges
- ❌ **Edge type enum — НЕ ПРОВЕРЯЕТСЯ**

```python
def _validate_edges(self, edges: List[Dict]) -> List[Dict]:
    valid_edges = []
    # ...
    for edge in edges:
        source = edge.get("source", "")
        target = edge.get("target", "")
        edge_type = edge.get("type", "")  # <-- Используется, но не валидируется!

        # Check self-loop PREREQUISITE
        if edge_type == "PREREQUISITE" and source == target:
            # ...
        
        # Check weight range
        # ...
        
        # Check node existence
        # ...
        
        # NO CHECK FOR VALID EDGE TYPE!
        valid_edges.append(edge)
```

### 2. JSON Schema валидация НЕ реализована

**Факт:** Библиотека `jsonschema` не импортируется и не используется нигде в проекте.

**Что есть:** Метод `_process_llm_response()` делает только базовые проверки:
```python
# Check required structure
if "chunk_graph_patch" not in parsed:
    return False, None

patch = parsed["chunk_graph_patch"]
if "nodes" not in patch or "edges" not in patch:
    return False, None

# Validate against schema (basic check)  <-- ЛОЖЬ! Это не валидация против схемы
if not isinstance(patch["nodes"], list) or not isinstance(patch["edges"], list):
    return False, None
```

Схема подставляется в промпт как текст (`{learning_chunk_graph_schema}`), но **программная валидация против неё не выполняется**.

### 3. Промпт-инструкции недостаточно строгие

- Edge types упоминались разрозненно в тексте, не консолидированно
- Фраза "other semantic relationships" подразумевала открытый список
- MENTIONS отсутствовал в примерах (кроме CS)
- Нет явного запрета на изобретение новых типов

### 4. Гипотеза о context drift

**Маленькие датасеты:** LLM держит инструкции в фокусе → следует enum строго

**Большие датасеты:** 
- ConceptDictionary на сотни концептов выдавливает инструкции из активного контекста
- LLM видит паттерн "relationship type" → генерирует семантически похожие (RELATED, USES, DEPENDS_ON)
- Валидация не ловит → невалидные типы попадают в граф

---

## Applied Fixes

### Prompt-level (DONE ✅)

6 правок во всех 4 доменных промптах (`_cs`, `_econ`, `_mgmt`, `_comm`):

1. **Instructions section** — явное указание "exactly 9 types", "No other edge types exist"
2. **Phase 2 header** — список типов с пометкой "exactly 9, no others"
3. **Phase 2 step 3** — убрана фраза "other semantic relationships"
4. **Edge Types Heuristics** — добавлен пример MENTIONS для каждого домена
5. **Planning and Verification** — добавлено "Use ONLY the 9 allowed edge types"
6. **Stop Conditions** — добавлено "Reject any edge with a type not in the allowed list"

### Schema-level (DONE ✅)

Усилено описание поля `type` в `LearningChunkGraph.schema.json`:
```json
"description": "Type of semantic relationship. Use ONLY these 9 allowed edge types. No other edge types exist."
```

---

## Pending Fixes

### Code-level validation (TODO)

Добавить проверку enum в `_validate_edges()`:

```python
VALID_EDGE_TYPES = {
    "PREREQUISITE", "ELABORATES", "EXAMPLE_OF",
    "HINT_FORWARD", "REFER_BACK", "PARALLEL",
    "TESTS", "REVISION_OF", "MENTIONS"
}

def _validate_edges(self, edges: List[Dict]) -> List[Dict]:
    valid_edges = []
    
    for edge in edges:
        edge_type = edge.get("type", "")
        
        # NEW: Check edge type against allowed enum
        if edge_type not in VALID_EDGE_TYPES:
            self.logger.warning(f"Invalid edge type '{edge_type}', skipping edge")
            continue
        
        # ... existing checks ...
```

### Optional: Full JSON Schema validation

Можно добавить `jsonschema.validate()` для всего патча, но это дороже по производительности и избыточно если enum проверяется явно.

---

## Testing

После внесения правок:
1. Прогнать pipeline на тестовом датасете
2. Проверить в итоговом графе: `jq '.edges[].type' LearningChunkGraph_raw.json | sort | uniq -c`
3. Все типы должны быть из списка 9 разрешённых

---

## Related Files

- `src/itext2kg_graph.py` — основная логика
- `src/schemas/LearningChunkGraph.schema.json` — схема данных
- `src/prompts/itext2kg_graph_extraction_*.md` — промпты (4 домена)
- `docs/LLM_Reference_K2-18.md` — референс по типам связей

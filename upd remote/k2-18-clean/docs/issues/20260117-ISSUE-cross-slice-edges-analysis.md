# K2-18-ISSUE: Анализ межслайсовых связей и роли контекстной цепочки LLM

**Дата:** 2025-01-17
**Статус:** Исследовано, выводы сделаны
**Компоненты:** itext2kg_graph, slicer, config

## Контекст задачи

После удаления overlap в slicer (в пользу контекстной цепочки LLM через `previous_response_id`) возник вопрос: действительно ли LLM использует контекст предыдущих слайсов для построения связей между вершинами?

**Гипотеза:** При обработке слайса N, LLM должен "подцеплять" вершины из слайса N-1 благодаря контекстной цепочке.

## Методология

### Способы проверки

**Границы слайсов:** Каждый слайс в `data/staging/` содержит поля `slice_token_start` и `slice_token_end` — абсолютные токенные позиции в исходном документе. Например:
```
Slice  1: tokens [     0,   5429)
Slice  2: tokens [  5429,  10658)
...
Slice 16: tokens [ 75219,  79931)
```

**1. Определение слайса для Chunk/Assessment:**
Позиция извлекается из финального ID вершины:
- `handbook:c:75979` → position = 75979
- `handbook:q:5200:1` → position = 5200
- Слайс определяется по попаданию: `slice_token_start <= position < slice_token_end`

**2. Определение слайса для Concept:**
ID не содержит позиции (`handbook:p:algoritm`), но nodes в JSON идут в порядке добавления. Когда `node_offset` текущей вершины меньше предыдущего — произошёл переход на новый слайс. Concept наследует слайс от ближайших Chunk/Assessment.

**3. Классификация рёбер:**
- INTRA-SLICE: оба конца в одном слайсе
- INTER-SLICE: концы в разных слайсах, с подсчётом distance

**4. Сравнение по стадиям пайплайна:**
Анализ `_raw.json`, `_dedup.json`, `_longrange.json` для понимания где добавляются cross-slice связи.

## Результаты анализа

### 1. Chunk↔Chunk/Assessment связи (RAW граф)

```
Total analyzed edges:  78
INTRA-SLICE (same slice):    78 (100.0%)
INTER-SLICE (cross slice):    0 (  0.0%)
```

**Вывод:** LLM создаёт связи между Chunk/Assessment **только внутри одного слайса**.

### 2. Сравнение по стадиям пайплайна

| Стадия | INTRA | INTER | Комментарий |
|--------|-------|-------|-------------|
| RAW | 100% | 0% | После itext2kg_graph |
| DEDUP | 98.7% | 1.3% | 1 ребро — артефакт |
| LONGRANGE | 72.8% | **27.2%** | Refiner добавил 46 cross-slice |

**Вывод:** Cross-slice связи между Chunk/Assessment добавляются только на этапе `refiner_longrange`.

### 3. Причина: архитектура ID и ограничения промпта

В `itext2kg_graph_extraction.md` два связанных момента:

**ID Convention — LLM использует временные ID:**
```
For nodes in this slice use the following IDs:
- Chunks: chunk_1, chunk_2, chunk_3...
- Assessments: assessment_1, assessment_2...
- Concepts: use exact concept_id from ConceptDictionary
```

**Явное ограничение scope:**
> All `source`/`target` edges must be nodes created from **the same slice**.

Эти ограничения взаимосвязаны: LLM работает с временными ID (`chunk_1`, `chunk_2`), которые существуют только в контексте текущего слайса. Финальные position-based ID (`handbook:c:75979`) присваивает Python в `_assign_final_ids()`. У LLM физически нет возможности сослаться на вершину из другого слайса — он не знает её временный ID.

### 4. Проверка порядка построения

```
✅ MONOTONIC: Slice numbers never decrease
   Graph is built strictly slice-by-slice
```

Граф строится строго последовательно: slice 1 → 2 → 3 → ... → 16. Нет признаков использования контекста для ordering.

### 5. Полный анализ включая Concept

```
Total edges: 533
Analyzed: 522

INTRA-SLICE:  280 ( 53.6%)
INTER-SLICE:  242 ( 46.4%)

Inter-slice by edge type:
  MENTIONS: 187      ← автоматические (Python код)
  PREREQUISITE: 29   ← LLM через ConceptDictionary
  ELABORATES: 14
  EXAMPLE_OF: 10
  TESTS: 1
  PARALLEL: 1

Inter-slice by node pattern:
  Chunk→Concept: 208
  Concept→Chunk: 28
  Concept→Concept: 5
```

**Ключевое наблюдение:** 
- 187 MENTIONS — генерируются автоматически в `_add_mentions_edges()` (текстовый поиск)
- Остальные inter-slice — связи с Concept нодами через **полный ConceptDictionary в input**

Concept'ы не "создаются" в слайсе — они берутся из словаря по `concept_id`, поэтому связь Chunk@slice5 → Concept@slice1 не нарушает промпт и возможна без контекстной цепочки.

## Выводы

### Факты

1. **Chunk↔Chunk/Assessment:** 100% intra-slice в RAW графе — архитектурно невозможно иначе (временные ID)
2. **Chunk↔Concept:** Inter-slice возможны через полный ConceptDictionary в input (не через контекстную цепочку)
3. **MENTIONS:** Генерируются Python кодом, не LLM
4. **Cross-slice связи:** Добавляются только на этапе `refiner_longrange`

### Роль контекстной цепочки для itext2kg_graph

**Контекстная цепочка НЕ влияет на построение графа:**
- Связи между Chunk/Assessment — только внутри слайса (архитектура временных ID)
- Связи с Concept — через словарь в input, не через контекст предыдущих ответов
- Ordering — строго последовательный, контекст не используется

### Рекомендации

1. **Для itext2kg_graph:** Можно использовать `response_chain_depth = 0`
   - Экономия токенов (не передаём предыдущие ответы)
   - Возможность параллельной обработки слайсов

2. **Качество границ слайсов:** `soft_boundary` работает хорошо, но если нужна страховка на стыках — рассмотреть небольшой overlap (100-200 токенов) в дополнение
   - Дешевле контекстной цепочки
   - Явный, контролируемый
   - Текст физически присутствует

3. **Архитектура подтверждена:** Двухфазный подход работает корректно
   - Phase 1: itext2kg_graph — плотные локальные связи внутри слайсов
   - Phase 2: refiner_longrange — дальние связи между слайсами

## Связанные задачи

- [ ] Рассмотреть разделение конфига на `[itext2kg_concepts]` и `[itext2kg_graph]` с разными `response_chain_depth`
- [ ] Рассмотреть небольшой overlap в slicer как страховку в дополнение к `soft_boundary`

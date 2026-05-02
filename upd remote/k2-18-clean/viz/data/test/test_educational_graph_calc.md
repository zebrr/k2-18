# test_educational_graph_calculations.md

## Граф: Educational Graph with Types

### Структура
- 4 узла: 1 Concept, 2 Chunks, 1 Assessment
- 5 рёбер: PREREQUISITE (2), EXAMPLE_OF (1), ELABORATES (1), TESTS (1)
- Все веса = 1.0

### Визуализация
```
concept_1 (Concept, difficulty=1)
    ↓ PREREQUISITE
chunk_1 (Chunk, difficulty=2)
    ↓ PREREQUISITE → chunk_2
    ↓ EXAMPLE_OF → concept_1
    ↑ ELABORATES ← chunk_2
    
chunk_2 (Chunk, difficulty=3)
    ↑ TESTS ← assessment_1
    
assessment_1 (Assessment, difficulty=3)
```

## Пошаговые вычисления

### 1. Degree метрики

**Подсчёт рёбер:**
- concept_1: in=[chunk_1], out=[chunk_1]
- chunk_1: in=[concept_1, chunk_2], out=[chunk_2, concept_1]
- chunk_2: in=[chunk_1, assessment_1], out=[chunk_1]
- assessment_1: in=[], out=[chunk_2]

**Degree centrality** = (degree_in + degree_out) / (n - 1), где n = 4:
- concept_1: (1 + 1) / 3 = 2/3 = 0.6667
- chunk_1: (2 + 2) / 3 = 4/3 = 1.3333
- chunk_2: (2 + 1) / 3 = 3/3 = 1.0000
- assessment_1: (0 + 1) / 3 = 1/3 = 0.3333

### 2. PageRank (alpha = 0.85)

**Итерация 0**: PR = [0.25, 0.25, 0.25, 0.25]

**Итерация 1**:
- teleport = (1 - 0.85) / 4 = 0.0375
- Висячих узлов нет (все имеют исходящие рёбра)
- concept_1: 0.0375 + 0.85 * (0.25/2 от chunk_1) = 0.0375 + 0.10625 = 0.14375
- chunk_1: 0.0375 + 0.85 * (0.25/1 от concept_1 + 0.25/1 от chunk_2) = 0.0375 + 0.425 = 0.4625
- chunk_2: 0.0375 + 0.85 * (0.25/2 от chunk_1 + 0.25/1 от assessment_1) = 0.0375 + 0.31875 = 0.35625
- assessment_1: 0.0375 + 0 = 0.0375

**Итерация 2**:
- concept_1: 0.0375 + 0.85 * (0.4625/2) = 0.234063
- chunk_1: 0.0375 + 0.85 * (0.14375 + 0.35625) = 0.4625
- chunk_2: 0.0375 + 0.85 * (0.4625/2 + 0.0375) = 0.265937
- assessment_1: 0.0375

**Итерация 3**: Сходимость
- concept_1: 0.2341
- chunk_1: 0.4625
- chunk_2: 0.2659
- assessment_1: 0.0375
- Сумма = 1.0000 ✓

### 3. Betweenness Centrality

**Все кратчайшие пути** (с учётом направленности):
1. concept_1 → chunk_1: прямой (1)
2. concept_1 → chunk_2: через chunk_1 (2)
3. chunk_1 → concept_1: прямой (1)
4. chunk_1 → chunk_2: прямой (1)
5. chunk_2 → chunk_1: прямой (1)
6. chunk_2 → concept_1: через chunk_1 (2)
7. assessment_1 → chunk_2: прямой (1)
8. assessment_1 → chunk_1: через chunk_2 (2)
9. assessment_1 → concept_1: через chunk_2 и chunk_1 (3)

**Подсчёт промежуточных узлов:**
- chunk_1: в путях #2, #6, #9 = 3
- chunk_2: в путях #8, #9 = 2
- concept_1: 0
- assessment_1: 0

**Нормализация** для directed: делим на (n-1)(n-2) = 6:
- chunk_1: 3/6 = 0.5
- chunk_2: 2/6 = 0.3333

### 4. Out-Closeness Centrality

**Расстояния от каждого узла** (кратчайшие пути):
- concept_1 → {chunk_1: 1, chunk_2: 2} → достигает 2, сумма = 3
- chunk_1 → {concept_1: 1, chunk_2: 1} → достигает 2, сумма = 2
- chunk_2 → {chunk_1: 1, concept_1: 2} → достигает 2, сумма = 3
- assessment_1 → {chunk_2: 1, chunk_1: 2, concept_1: 3} → достигает 3, сумма = 6

**Формула Wasserman-Faust**: `(k/sum_dist) * (k/(n-1))`:
- concept_1: (2/3) * (2/3) = 4/9 = 0.4444
- chunk_1: (2/2) * (2/3) = 2/3 = 0.6667
- chunk_2: (2/3) * (2/3) = 4/9 = 0.4444
- assessment_1: (3/6) * (3/3) = 1/2 = 0.5

### 5. Component ID
Все узлы слабо связаны → component_id = 0 для всех

### 6. Prerequisite Depth
Анализ PREREQUISITE подграфа:
- concept_1: нет входящих → depth = 0
- chunk_1: от concept_1 (depth 0) → depth = 1
- chunk_2: от chunk_1 (depth 1) → depth = 2
- assessment_1: нет PREREQUISITE связей → depth = 0

### 7. Learning Effort
Формула: difficulty(узел) + max(effort предшественников по PREREQUISITE):
- concept_1: 1 + 0 = 1
- chunk_1: 2 + 1 = 3
- chunk_2: 3 + 3 = 6
- assessment_1: 3 + 0 = 3

### 8. Inverse Weights
Все weight = 1.0 → inverse_weight = 1.0 для всех рёбер

### 9. Educational Importance для test_educational_graph:

- concept_1: 0.23406
- chunk_1: 0.46250
- chunk_2: 0.26594
- assessment_1: 0.03750

Сумма: 1.00000 ✓

Значения полностью совпадают с обычным PageRank, так как все 5 рёбер графа относятся к образовательным типам (PREREQUISITE, ELABORATES, EXAMPLE_OF, TESTS) — подграф равен полному графу.

## Проверка инвариантов
✓ Сумма PageRank = 1.0000
✓ chunk_1 - центральный узел (max PageRank и betweenness)
✓ assessment_1 - висячий узел (min PageRank)
✓ Все метрики в допустимых диапазонах

# Детальные расчеты метрик для test_realistic_mini.json

## Структура графа
- **Узлов**: 10 (4 Chunk, 3 Concept, 2 Assessment)
- **Рёбер**: 16
- **Направленный граф**: да
- **Взвешенный граф**: да

## Узлы и их типы
1. chunk_intro (Chunk, difficulty=1)
2. concept_complexity (Concept, difficulty=3 default)
3. chunk_sorting (Chunk, difficulty=2)
4. chunk_bubble (Chunk, difficulty=2)
5. concept_sorting (Concept, difficulty=3 default)
6. chunk_quicksort (Chunk, difficulty=3)
7. concept_recursion (Concept, difficulty=3 default)
8. assessment_sorting (Assessment, difficulty=3)
9. chunk_optimization (Chunk, difficulty=4)
10. assessment_final (Assessment, difficulty=4)

## 1. Degree метрики

### Подсчет входящих и исходящих рёбер

| Узел | Входящие рёбра | Исходящие рёбра | degree_in | degree_out |
|------|---------------|-----------------|-----------|------------|
| chunk_intro | - | →complexity(0.35), →sorting(0.9) | 0 | 2 |
| concept_complexity | ←intro(0.35) | →sorting(0.85) | 1 | 1 |
| chunk_sorting | ←intro(0.9), ←complexity(0.85) | →concept_sorting(0.4), →bubble(0.85), →final(0.4) | 2 | 3 |
| chunk_bubble | ←sorting(0.85), ←concept_sorting(0.65), ←assessment(0.9) | →quicksort(0.6) | 3 | 1 |
| concept_sorting | ←sorting(0.4) | →bubble(0.65), →quicksort(0.7) | 1 | 2 |
| chunk_quicksort | ←bubble(0.6), ←concept_sorting(0.7), ←assessment(0.9), ←optimization(0.85) | →recursion(0.35), →optimization(0.95) | 4 | 2 |
| concept_recursion | ←quicksort(0.35) | →optimization(0.8) | 1 | 1 |
| assessment_sorting | - | →bubble(0.9), →quicksort(0.9) | 0 | 2 |
| chunk_optimization | ←quicksort(0.95), ←recursion(0.8), ←assessment_final(0.85) | →quicksort(0.85) | 3 | 1 |
| assessment_final | ←sorting(0.4) | →optimization(0.85) | 1 | 1 |

### Degree centrality
**Формула**: `(degree_in + degree_out) / (n - 1)` где n=10

- chunk_intro: (0+2)/9 = **0.222222**
- concept_complexity: (1+1)/9 = **0.222222**
- chunk_sorting: (2+3)/9 = **0.555556**
- chunk_bubble: (3+1)/9 = **0.444444**
- concept_sorting: (1+2)/9 = **0.333333**
- chunk_quicksort: (4+2)/9 = **0.666667**
- concept_recursion: (1+1)/9 = **0.222222**
- assessment_sorting: (0+2)/9 = **0.222222**
- chunk_optimization: (3+1)/9 = **0.444444**
- assessment_final: (1+1)/9 = **0.222222**

## 2. Inverse weights для рёбер

| Ребро | Weight | Inverse weight |
|-------|--------|----------------|
| chunk_intro → concept_complexity | 0.35 | 2.8571 |
| chunk_intro → chunk_sorting | 0.9 | 1.1111 |
| concept_complexity → chunk_sorting | 0.85 | 1.1765 |
| chunk_sorting → concept_sorting | 0.4 | 2.5000 |
| chunk_sorting → chunk_bubble | 0.85 | 1.1765 |
| chunk_bubble → chunk_quicksort | 0.6 | 1.6667 |
| concept_sorting → chunk_bubble | 0.65 | 1.5385 |
| concept_sorting → chunk_quicksort | 0.7 | 1.4286 |
| chunk_quicksort → concept_recursion | 0.35 | 2.8571 |
| assessment_sorting → chunk_bubble | 0.9 | 1.1111 |
| assessment_sorting → chunk_quicksort | 0.9 | 1.1111 |
| chunk_quicksort → chunk_optimization | 0.95 | 1.0526 |
| chunk_optimization → chunk_quicksort | 0.85 | 1.1765 |
| concept_recursion → chunk_optimization | 0.8 | 1.2500 |
| assessment_final → chunk_optimization | 0.85 | 1.1765 |
| chunk_sorting → assessment_final | 0.4 | 2.5000 |

## 3. PageRank (alpha=0.85)

Используя алгоритм с правильной обработкой dangling nodes:
- Итеративное вычисление до сходимости
- Учет массы висячих узлов (нет в этом графе)
- Нормализация: сумма = 1.0

**Результаты:**
- chunk_intro: **0.01500000**
- concept_complexity: **0.01857000**
- chunk_sorting: **0.03996450**
- chunk_bubble: **0.04838379**
- concept_sorting: **0.02323511**
- chunk_quicksort: **0.36884547**
- concept_recursion: **0.09940887**
- assessment_sorting: **0.01500000**
- chunk_optimization: **0.34835716**
- assessment_final: **0.02323511**

**Проверка**: Σ = 1.00000000 ✓

## 4. Component ID

Все узлы находятся в одной слабосвязной компоненте: **component_id = 0** для всех.

## 5. Prerequisite Depth

Анализ PREREQUISITE рёбер:
- chunk_intro → chunk_sorting
- concept_complexity → chunk_sorting
- chunk_sorting → chunk_bubble
- chunk_quicksort → chunk_optimization
- concept_recursion → chunk_optimization

**Уровни:**
- **Уровень 0** (без входящих PREREQUISITE): chunk_intro, concept_complexity, concept_sorting, chunk_quicksort, concept_recursion, assessment_sorting, assessment_final
- **Уровень 1** (зависят от уровня 0): chunk_sorting (от intro и complexity), chunk_optimization (от quicksort и recursion)
- **Уровень 2** (зависят от уровня 1): chunk_bubble (от sorting)

## 6. Learning Effort

**Формула**: `difficulty(node) + max(effort предшественников по PREREQUISITE)`

Используя default_difficulty = 3 для Concept узлов:

- chunk_intro: 1 (нет предшественников)
- concept_complexity: 3 (нет предшественников)
- chunk_sorting: 2 + max(1, 3) = **5**
- chunk_bubble: 2 + 5 = **7**
- concept_sorting: 3 (нет PREREQUISITE предшественников)
- chunk_quicksort: 3 (нет предшественников)
- concept_recursion: 3 (нет предшественников)
- assessment_sorting: 3 (нет предшественников)
- chunk_optimization: 4 + max(3, 3) = **7**
- assessment_final: 4 (нет предшественников)

## 7. Educational Importance

PageRank по подграфу с типами: PREREQUISITE, ELABORATES, EXAMPLE_OF, TESTS

Рёбра образовательного подграфа:
- PREREQUISITE: intro→sorting(0.9), complexity→sorting(0.85), sorting→bubble(0.85), quicksort→optimization(0.95), recursion→optimization(0.8)
- ELABORATES: concept_sorting→bubble(0.65), concept_sorting→quicksort(0.7)
- TESTS: assessment_sorting→bubble(0.9), assessment_sorting→quicksort(0.9), assessment_final→optimization(0.85)

**Результаты:**
- chunk_intro: **0.05268460**
- concept_complexity: **0.05268460**
- chunk_sorting: **0.14224841**
- chunk_bubble: **0.21754836**
- concept_sorting: **0.05268460**
- chunk_quicksort: **0.09829580**
- concept_recursion: **0.05268460**
- assessment_sorting: **0.05268460**
- chunk_optimization: **0.22579984**
- assessment_final: **0.05268460**

**Проверка**: Σ = 1.00000000 ✓

## 8. Betweenness Centrality

Используя алгоритм Брандеса с inverse_weight:
- Нормализация для directed: делим на (n-1)*(n-2) = 72

**Результаты:**
- chunk_intro: **0.000000**
- concept_complexity: **0.000000**
- chunk_sorting: **0.166667** (важный разветвитель)
- chunk_bubble: **0.083333**
- concept_sorting: **0.000000**
- chunk_quicksort: **0.152778** (центральный мост)
- concept_recursion: **0.000000**
- assessment_sorting: **0.000000**
- chunk_optimization: **0.041667**
- assessment_final: **0.041667**

## 9. Out-Closeness Centrality

Используя формулу Wasserman-Faust с inverse_weight:
`closeness = (k/sum_distances) * (k/(n-1))` где k - количество достижимых узлов

**Результаты:**
- chunk_intro: **0.244946** (достигает 8/9)
- concept_complexity: **0.204432** (достигает 7/9)
- chunk_sorting: **0.217433** (достигает 6/9)
- chunk_bubble: **0.112236** (достигает 3/9)
- concept_sorting: **0.182636** (достигает 4/9)
- chunk_quicksort: **0.113677** (достигает 2/9)
- concept_recursion: **0.120888** (достигает 2/9)
- assessment_sorting: **0.212803** (достигает 4/9)
- chunk_optimization: **0.085304** (достигает 2/9)
- assessment_final: **0.114422** (достигает 3/9)

---

## Сводная таблица всех метрик

| Node ID            | degree_in | degree_out | degree_centrality | pagerank   | betweenness | out-closeness | component_id | prerequisite_depth | learning_effort | educational_importance |
|--------------------|-----------|------------|-------------------|------------|-----------------------------|--------------|--------------------|-----------------|------------------------|
| chunk_intro        | 0         | 2          | 0.222222          | 0.01500000 | 0.000000    | 0.244946      | 0            | 0                  | 1               | 0.05268460             |
| concept_complexity | 1         | 1          | 0.222222          | 0.01857000 | 0.000000    | 0.204432      | 0            | 0                  | 3               | 0.05268460             |
| chunk_sorting      | 2         | 3          | 0.555556          | 0.03996450 | 0.166667    | 0.217433      | 0            | 1                  | 5               | 0.14224841             |
| chunk_bubble       | 3         | 1          | 0.444444          | 0.04838379 | 0.083333    | 0.112236      | 0            | 2                  | 7               | 0.21754836             |
| concept_sorting    | 1         | 2          | 0.333333          | 0.02323511 | 0.000000    | 0.182636      | 0            | 0                  | 3               | 0.05268460             |
| chunk_quicksort    | 4         | 2          | 0.666667          | 0.36884547 | 0.152778    | 0.113677      | 0            | 0                  | 3               | 0.09829580             |
| concept_recursion  | 1         | 1          | 0.222222          | 0.09940887 | 0.000000    | 0.120888      | 0            | 0                  | 3               | 0.05268460             |
| assessment_sorting | 0         | 2          | 0.222222          | 0.01500000 | 0.000000    | 0.212803      | 0            | 0                  | 3               | 0.05268460             |
| chunk_optimization | 3         | 1          | 0.444444          | 0.34835716 | 0.041667    | 0.085304      | 0            | 1                  | 7               | 0.22579984             |
| assessment_final   | 1         | 1          | 0.222222          | 0.02323511 | 0.041667    | 0.114422      | 0            | 0                  | 4               | 0.05268460             |

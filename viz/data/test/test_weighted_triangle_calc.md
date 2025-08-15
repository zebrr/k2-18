# test_weighted_triangle.json - Пошаговые вычисления метрик

## Структура графа
- **Узлы**: A, B, C (все типа Chunk)
- **Рёбра**: 
  - A → B (weight=0.9, type=PREREQUISITE)
  - B → C (weight=0.3, type=EXAMPLE_OF)
  - C → A (weight=0.6, type=ELABORATES)

## 1. Degree метрики

### degree_in и degree_out (количество рёбер)
- **A**: degree_in = 1 (от C), degree_out = 1 (к B)
- **B**: degree_in = 1 (от A), degree_out = 1 (к C)
- **C**: degree_in = 1 (от B), degree_out = 1 (к A)

### degree_centrality
Для directed графов: `(degree_in + degree_out) / (n - 1)` где n=3
- **A**: (1 + 1) / (3 - 1) = 2/2 = 1.0
- **B**: (1 + 1) / (3 - 1) = 2/2 = 1.0
- **C**: (1 + 1) / (3 - 1) = 2/2 = 1.0

## 2. inverse_weight (добавляется к рёбрам)
- A → B: inverse_weight = 1.0 / 0.9 = 1.111...
- B → C: inverse_weight = 1.0 / 0.3 = 3.333...
- C → A: inverse_weight = 1.0 / 0.6 = 1.666...

## 3. PageRank (с учётом весов)

PageRank использует ПРЯМЫЕ веса. В цикле с разными весами:
- Телепортация: (1-α)/n = 0.15/3 = 0.05
- α = 0.85 (damping factor)

### Итеративный расчёт
Начальные значения: PR(A) = PR(B) = PR(C) = 1/3

#### Итерация 1:
PR(A) = 0.05 + 0.85 * (PR(C) * w(C→A)/Σw_out(C))
      = 0.05 + 0.85 * (0.333 * 0.6/0.6)  
      = 0.05 + 0.85 * 0.333 = 0.333

PR(B) = 0.05 + 0.85 * (PR(A) * w(A→B)/Σw_out(A))
      = 0.05 + 0.85 * (0.333 * 0.9/0.9)
      = 0.05 + 0.85 * 0.333 = 0.333

PR(C) = 0.05 + 0.85 * (PR(B) * w(B→C)/Σw_out(B))
      = 0.05 + 0.85 * (0.333 * 0.3/0.3)
      = 0.05 + 0.85 * 0.333 = 0.333

**Результат**: В симметричном цикле (где каждый узел имеет ровно одно входящее и одно исходящее ребро), PageRank распределяется равномерно независимо от весов!

### Финальные значения PageRank
- **A**: 0.333333
- **B**: 0.333333
- **C**: 0.333333
- **Сумма**: 1.0 ✓

## 4. Betweenness Centrality

В цикле из 3 узлов НЕТ кратчайших путей, проходящих ЧЕРЕЗ промежуточные узлы:
- Путь A→B: прямое ребро (не через промежуточные)
- Путь B→C: прямое ребро (не через промежуточные)
- Путь C→A: прямое ребро (не через промежуточные)
- Путь A→C: A→B→C (через B), но есть альтернативный путь через A→B→C→A→B→C
- Путь B→A: B→C→A (через C)
- Путь C→B: C→A→B (через A)

С учётом inverse_weight для кратчайших путей:
- A→C: прямого нет, через B = 1.111 + 3.333 = 4.444
- B→A: прямого нет, через C = 3.333 + 1.666 = 4.999
- C→B: прямого нет, через A = 1.666 + 1.111 = 2.777

### Результат Betweenness
В цикле каждый узел лежит на некоторых кратчайших путях:
- **A**: лежит на пути C→B (1 путь)
- **B**: лежит на пути A→C (1 путь)
- **C**: лежит на пути B→A (1 путь)

Normalized betweenness = count / ((n-1)*(n-2)) = 1 / (2*1) = 0.5

### Финальные значения Betweenness
- **A**: 0.5
- **B**: 0.5
- **C**: 0.5

## 5. Out-closeness (с учётом inverse_weight)

Out-closeness = достижимость исходящими путями.
Для каждого узла считаем сумму расстояний до достижимых узлов.

### Расстояния (по inverse_weight):
От A:
- A→B: 1.111
- A→C: 1.111 + 3.333 = 4.444
- Сумма: 5.555, достигает 2 узла

От B:
- B→C: 3.333
- B→A: 3.333 + 1.666 = 4.999
- Сумма: 8.332, достигает 2 узла

От C:
- C→A: 1.666
- C→B: 1.666 + 1.111 = 2.777
- Сумма: 4.443, достигает 2 узла

### Closeness (Wasserman-Faust):
closeness = (k/sum_distances) * (k/(n-1)) где k=2 (достижимых), n=3

- **A**: (2/5.555) * (2/2) = 0.360
- **B**: (2/8.332) * (2/2) = 0.240
- **C**: (2/4.443) * (2/2) = 0.450

## 6. Component ID
Все узлы в одной weakly connected component:
- **A**: component_id = 0
- **B**: component_id = 0
- **C**: component_id = 0

## 7. Prerequisite Depth
Только PREREQUISITE ребро: A → B
- **A**: depth = 0 (нет входящих PREREQUISITE)
- **B**: depth = 1 (от A)
- **C**: depth = 0 (нет входящих PREREQUISITE)

## 8. Learning Effort
С учётом difficulty и PREREQUISITE путей:
- **A**: effort = difficulty(A) = 1
- **B**: effort = difficulty(B) + effort(A) = 3 + 1 = 4
- **C**: effort = difficulty(C) = 2 (нет PREREQUISITE предков)

## 9. Educational Importance
PageRank по подграфу из [PREREQUISITE, ELABORATES, EXAMPLE_OF]:
Все три ребра входят в educational_edge_types, поэтому:
- **A**: 0.333333 (как обычный PageRank)
- **B**: 0.333333
- **C**: 0.333333

## Итоговая таблица метрик

| Метрика                | A        | B        | C        |
|------------------------|----------|----------|----------|
| degree_in              | 1        | 1        | 1        |
| degree_out             | 1        | 1        | 1        |
| degree_centrality      | 1.0      | 1.0      | 1.0      |
| pagerank               | 0.333333 | 0.333333 | 0.333333 |
| betweenness_centrality | 0.5      | 0.5      | 0.5      |
| out-closeness          | 0.360    | 0.240    | 0.450    |
| component_id           | 0        | 0        | 0        |
| prerequisite_depth     | 0        | 1        | 0        |
| learning_effort        | 1        | 4        | 2        |
| educational_importance | 0.333333 | 0.333333 | 0.333333 |

## Метрики рёбер

| Ребро | weight | inverse_weight |
|-------|--------|----------------|
| A→B | 0.9 | 1.111111 |
| B→C | 0.3 | 3.333333 |
| C→A | 0.6 | 1.666667 |

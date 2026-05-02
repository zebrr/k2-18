# test_star_graph: Пошаговые вычисления метрик

## Структура графа
```
           branch_1
              ↑
              |
branch_2 ← central → branch_3
              |
              ↓
           branch_4
```

- **Узлы**: 5 (central, branch_1, branch_2, branch_3, branch_4)
- **Рёбра**: 4 (все из central в branch_*)
- **Веса**: все = 1.0
- **Направленный граф** (directed)

## 1. Degree метрики

### Подсчет степеней
- **central**: degree_in = 0, degree_out = 4
- **branch_1**: degree_in = 1, degree_out = 0
- **branch_2**: degree_in = 1, degree_out = 0
- **branch_3**: degree_in = 1, degree_out = 0
- **branch_4**: degree_in = 1, degree_out = 0

### Degree Centrality
Формула для directed: `(degree_in + degree_out) / (n - 1)` где n = 5

- **central**: (0 + 4) / 4 = 1.0
- **branch_1**: (1 + 0) / 4 = 0.25
- **branch_2**: (1 + 0) / 4 = 0.25
- **branch_3**: (1 + 0) / 4 = 0.25
- **branch_4**: (1 + 0) / 4 = 0.25

## 2. PageRank (alpha = 0.85)

### Итерация 0 (начальные значения)
Все узлы: PR = 1/5 = 0.2

### Анализ структуры
- central - висячий узел (dangling node, нет исходящих рёбер? НЕТ! Есть 4 исходящих!)
- branch_* - висячие узлы (нет исходящих рёбер)

### Итеративный расчет
Формула: `PR(v) = (1-α)/n + α * [Σ(PR(u)/outdeg(u)) + s/n]`
где s - сумма PR висячих узлов

**Итерация 1:**
- Висячие узлы: branch_1, branch_2, branch_3, branch_4
- s = 0.2 * 4 = 0.8
- teleport = (1-0.85)/5 = 0.03

PR(central) = 0.03 + 0.85 * (0 + 0.8/5) = 0.03 + 0.85 * 0.16 = 0.03 + 0.136 = 0.166
PR(branch_1) = 0.03 + 0.85 * (0.2/4 + 0.8/5) = 0.03 + 0.85 * (0.05 + 0.16) = 0.03 + 0.1785 = 0.2085
PR(branch_2) = 0.2085 (симметрично)
PR(branch_3) = 0.2085 (симметрично)
PR(branch_4) = 0.2085 (симметрично)

Проверка суммы: 0.166 + 4*0.2085 = 0.166 + 0.834 = 1.0 ✓

**Итерация 2:**
- s = 0.2085 * 4 = 0.834

PR(central) = 0.03 + 0.85 * (0 + 0.834/5) = 0.03 + 0.85 * 0.1668 = 0.03 + 0.14178 = 0.17178
PR(branch_1) = 0.03 + 0.85 * (0.166/4 + 0.834/5) = 0.03 + 0.85 * (0.0415 + 0.1668) = 0.03 + 0.177055 = 0.207055
PR(branch_2-4) = 0.207055

Проверка: 0.17178 + 4*0.207055 = 0.17178 + 0.82822 = 1.0 ✓

**Продолжаем итерации до сходимости...**

### Финальные значения PageRank (сходимость на 16-й итерации):
- **central**: 0.1709402
- **branch_1**: 0.2072650
- **branch_2**: 0.2072650
- **branch_3**: 0.2072650
- **branch_4**: 0.2072650

Сумма: 0.1709402 + 4*0.2072650 = 1.0000002 ✓

## 3. Betweenness Centrality

### Кратчайшие пути между всеми парами
Для directed графа учитываем направление.

**Пути через central:**
- branch_1 → branch_2: НЕТ ПУТИ (нужно идти против стрелок)
- branch_1 → branch_3: НЕТ ПУТИ
- branch_1 → branch_4: НЕТ ПУТИ
- branch_1 → central: НЕТ ПУТИ
- branch_2 → branch_1: НЕТ ПУТИ
- branch_2 → branch_3: НЕТ ПУТИ
- branch_2 → branch_4: НЕТ ПУТИ
- branch_2 → central: НЕТ ПУТИ
- branch_3 → branch_1: НЕТ ПУТИ
- branch_3 → branch_2: НЕТ ПУТИ
- branch_3 → branch_4: НЕТ ПУТИ
- branch_3 → central: НЕТ ПУТИ
- branch_4 → branch_1: НЕТ ПУТИ
- branch_4 → branch_2: НЕТ ПУТИ
- branch_4 → branch_3: НЕТ ПУТИ
- branch_4 → central: НЕТ ПУТИ
- central → branch_1: ПРЯМОЙ ПУТЬ (central не промежуточный)
- central → branch_2: ПРЯМОЙ ПУТЬ
- central → branch_3: ПРЯМОЙ ПУТЬ
- central → branch_4: ПРЯМОЙ ПУТЬ

**Важно**: В directed графе-звезде с рёбрами ИЗ центра, между периферийными узлами НЕТ путей!

### Подсчет betweenness
- **central**: 0 (не является промежуточным ни для одного пути)
- **branch_1**: 0
- **branch_2**: 0
- **branch_3**: 0
- **branch_4**: 0

### Нормализация для directed
Делим на (n-1)*(n-2) = 4*3 = 12

Все значения остаются 0.

## 4. Out-Closeness (исходящая близость для directed графов)

### Расстояния от каждого узла (с учетом inverse_weight = 1/weight = 1)

**От central:**
- до branch_1: 1
- до branch_2: 1
- до branch_3: 1
- до branch_4: 1
- Достигает 4 узла из 4, сумма расстояний = 4

**От branch_1:**
- Не может достичь никого (нет исходящих рёбер)
- Достигает 0 узлов из 4

**От branch_2, branch_3, branch_4:**
- Аналогично, достигают 0 узлов

### Расчет out_closeness (формула Wasserman-Faust)
Если узел достигает k узлов из n-1:
- closeness = (k / sum_distances) * (k / (n-1))

**central**: (4/4) * (4/4) = 1.0 * 1.0 = 1.0
**branch_1**: 0 (не достигает никого)
**branch_2**: 0
**branch_3**: 0
**branch_4**: 0

## 5. Component ID

Все узлы в одной weakly connected component (если игнорировать направление):
- **Все узлы**: component_id = 0

## 6. Prerequisite Depth

Нет PREREQUISITE рёбер в графе, поэтому:
- **Все узлы**: prerequisite_depth = 0

## 7. Learning Effort

Формула: сумма difficulty по PREREQUISITE пути + своя difficulty
Так как нет PREREQUISITE рёбер:
- **central**: 2 (только своя difficulty)
- **branch_1**: 1
- **branch_2**: 3
- **branch_3**: 3
- **branch_4**: 4

## 8. Inverse Weight для рёбер

Для всех рёбер: inverse_weight = 1.0 / 1.0 = 1.0

## 9. educational_importance

Значения educational_importance:

- central: 0.1709402
- branch_1: 0.2072650
- branch_2: 0.2072650
- branch_3: 0.2072650
- branch_4: 0.2072650
- Сумма: 1.0 ✓

Важное наблюдение: Educational importance совпадает с обычным PageRank!

В нашем графе ВСЕ рёбра относятся к educational типам:
- 3 ребра EXAMPLE_OF
- 1 ребро TESTS

Все эти типы входят в educational_edge_types = ["PREREQUISITE", "ELABORATES", "TESTS", "EXAMPLE_OF"]
Следовательно, подграф для educational importance = полный граф
PageRank на полном графе = PageRank на подграфе (когда подграф совпадает с полным)

Это отличная проверка корректности алгоритма! В графах где не все рёбра "образовательные", значения будут отличаться.

## Итоговая таблица метрик

| Узел     | degree_in | degree_out | degree_centrality | pagerank  | betweenness | closeness | component_id | prereq_depth | learning_effort | educational_importance |
|----------|-----------|------------|-------------------|-----------|-------------|-----------|--------------|--------------|-----------------|------------------------|
| central  | 0         | 4          | 1.0               | 0.1709402 | 0.0         | 1.0       | 0            | 0            | 2               | 0.1709402              |
| branch_1 | 1         | 0          | 0.25              | 0.2072650 | 0.0         | 0.0       | 0            | 0            | 1               | 0.2072650              |
| branch_2 | 1         | 0          | 0.25              | 0.2072650 | 0.0         | 0.0       | 0            | 0            | 3               | 0.2072650              |
| branch_3 | 1         | 0          | 0.25              | 0.2072650 | 0.0         | 0.0       | 0            | 0            | 3               | 0.2072650              |
| branch_4 | 1         | 0          | 0.25              | 0.2072650 | 0.0         | 0.0       | 0            | 0            | 4               | 0.2072650              |

## Ключевые инсайты для валидации

1. **Betweenness = 0 для всех** - в directed звезде с рёбрами ИЗ центра нет путей между периферийными узлами
2. **PageRank максимален у периферийных узлов** - они висячие и перераспределяют свой вес равномерно
3. **Out-closeness = 1.0 только у центра** - только он может достичь других узлов
4. **Degree centrality максимальна у центра** = 1.0

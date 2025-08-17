# K2-18 VIZ: Референс алгоритмов вычисления метрик графа

## 1. Введение

Этот документ — единый источник правды для вычисления метрик образовательного графа знаний в проекте K2-18.

### Покрываемые метрики
**10 метрик для узлов:**
1. degree_in — входящая степень
2. degree_out — исходящая степень  
3. degree_centrality — нормализованная степень
4. pagerank — важность узла
5. betweenness_centrality — узел как мост
6. out-closeness — исходящая близость
7. component_id — компонента связности
8. prerequisite_depth — уровень в дереве зависимостей
9. learning_effort — накопленная сложность
10. educational_importance — важность в образовательном контексте

**1 метрика для рёбер:**
1. inverse_weight — обратный вес для distance-based алгоритмов

### Порядок вычисления
Метрики ДОЛЖНЫ вычисляться в указанном порядке из-за зависимостей.

---

## 2. Метрики узлов

### 2.1. degree_in, degree_out

**Образовательный смысл:** Показывает, от скольких узлов зависит данный (in) и на сколько узлов он влияет (out). Высокий degree_in = много предпосылок, высокий degree_out = открывает много нового.

**Алгоритм:** Простой подсчёт входящих и исходящих рёбер.

**Python реализация:**
```python
deg_in = dict(G.in_degree())
deg_out = dict(G.out_degree())
```

**Граничные случаи:**
- Изолированный узел: degree_in = 0, degree_out = 0
- Пустой граф: все степени = 0

---

### 2.2. degree_centrality

**Образовательный смысл:** Насколько узел "общительный" относительно размера графа. Показывает долю всех возможных связей, которые есть у узла. Если узел связан со всеми — близко к 1, если изолирован — 0.

**Алгоритм:** `(degree_in + degree_out) / (n - 1)` для directed графов.

**Python реализация:**
```python
deg_cent = nx.degree_centrality(G.to_undirected())
```

**Граничные случаи:**
- Граф с 1 узлом: degree_centrality = 0
- Полный граф: degree_centrality → 1

**Зависимости:** Использует degree_in и degree_out.

---

### 2.3. inverse_weight (для рёбер)

**Образовательный смысл:** Преобразование веса в "расстояние" — чем слабее связь, тем "дальше" узлы.

**Алгоритм:** `inverse_weight = 1.0 / weight`

**Python реализация:**
```python
for u, v, d in G.edges(data=True):
    w = float(d.get('weight', 1.0))
    inv = (1.0 / w) if (w and w > 0) else float("inf")
    G[u][v]['inverse_weight'] = inv
```

**Граничные случаи:**
- weight = 0 или отсутствует: inverse_weight = inf
- weight = 1: inverse_weight = 1

**Примечание:** ДОЛЖЕН быть вычислен ДО betweenness_centrality и out-closeness.

---

### 2.4. pagerank

**Образовательный смысл:** Важность узла с учётом важности указывающих на него узлов. Ключевая особенность — накопительный эффект: важность передаётся и накапливается по цепочкам. Узел важен не сам по себе, а если на него ссылаются другие важные узлы. Базовые концепты передают важность производным.

**Алгоритм:** Итеративный алгоритм со случайными переходами. Висячие узлы (без исходящих рёбер) распределяют свою массу равномерно.

**Python реализация:**
```python
if G.number_of_edges():
    pr = nx.pagerank(G, alpha=0.85, weight="weight")
else:
    pr = {u: 1.0/n for u in G.nodes()}
```

**Параметры конфига:**
- `pagerank_damping` (default: 0.85) — вероятность следования по рёбрам

**Граничные случаи:**
- Пустой граф: все узлы получают 1/n
- Граф без рёбер: все узлы получают 1/n
- Висячие узлы: их PageRank перераспределяется равномерно

**Инвариант:** `sum(PageRank) = 1.0 ± 0.01`

---

### 2.5. betweenness_centrality

**Образовательный смысл:** Узел как "мост" между частями графа. Высокое значение = узел критичен для связности знаний.

**Алгоритм:** Доля кратчайших путей, проходящих через узел. Использует ОБРАТНЫЕ веса (меньший вес = "дороже" путь).

**Python реализация:**
```python
if n >= 3:
    btw = nx.betweenness_centrality(G, weight="inverse_weight", normalized=True)
else:
    btw = {u: 0.0 for u in G.nodes()}
```

**Параметры конфига:**
- `betweenness_normalized` (default: true) — нормализация на `(n-1)*(n-2)`

**Граничные случаи:**
- Граф < 3 узлов: все betweenness = 0
- Линейный граф: средний узел имеет максимальное значение

**Зависимости:** Требует предвычисленный inverse_weight на рёбрах.

---

### 2.6. out-closeness

**Образовательный смысл:** Насколько узел "близок" к другим через исходящие пути. Высокое значение = узел может легко достичь многих других.

**Алгоритм:** OUT-closeness для directed графов через реверс. Формула Wasserman-Faust для частичной достижимости.

**Python реализация:**
```python
if n > 1:
    Gr = G.reverse(copy=True)
    out_close = nx.closeness_centrality(Gr, distance="inverse_weight", wf_improved=True)
else:
    out_close = {u: 0.0 for u in G.nodes()}
```

**Параметры конфига:**
- `closeness_harmonic` (default: true) — использовать гармоническую централизацию для несвязных графов

**Граничные случаи:**
- Граф с 1 узлом: out-closeness = 0
- Изолированный узел: out-closeness = 0
- Узел без исходящих рёбер: out-closeness = 0

**Зависимости:** Требует предвычисленный inverse_weight на рёбрах.

---

### 2.7. component_id

**Образовательный смысл:** Группировка узлов в связные подграфы. Узлы одной компоненты достижимы друг из друга.

**Алгоритм:** Weakly connected components с детерминированной нумерацией по порядку узлов в файле.

**Python реализация:**
```python
def component_ids(G, node_order):
    UG = G.to_undirected()
    comps = list(nx.connected_components(UG))
    order_map = {n: i for i, n in enumerate(node_order)}
    comps_sorted = sorted(comps, key=lambda c: min(order_map.get(n, 10**9) for n in c))
    mapping = {}
    for cid, comp in enumerate(comps_sorted):
        for n in comp:
            mapping[n] = cid
    return mapping
```

**Граничные случаи:**
- Полностью связный граф: все узлы имеют component_id = 0
- n изолированных узлов: component_id от 0 до n-1

**Инвариант:** component_id начинается с 0 и идёт последовательно.

---

### 2.8. prerequisite_depth

**Образовательный смысл:** Уровень узла в иерархии предварительных требований. 0 = базовые концепты, далее по возрастанию.

**Алгоритм:** Максимальная длина пути по PREREQUISITE рёбрам от узлов без входящих PREREQUISITE.

**Python реализация:**
```python
def prereq_subgraph(G):
    E = [(u, v, d) for u, v, d in G.edges(data=True) 
         if str(d.get("type", "")).upper() == "PREREQUISITE"]
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(E)
    return H

# В функции scc_dag_depth_and_effort:
# 1. Строим SCC (strongly connected components)
# 2. Создаём condensed DAG
# 3. Топологическая сортировка
# 4. DP: depth[c] = max(depth[pred]) + 1
```

**Граничные случаи:**
- Узлы без входящих PREREQUISITE: depth = 0
- Циклы PREREQUISITE: все узлы цикла получают одинаковую глубину
- Граф без PREREQUISITE рёбер: все узлы имеют depth = 0

**Зависимости:** Анализирует только подграф PREREQUISITE рёбер.

---

### 2.9. learning_effort

**Образовательный смысл:** Накопленная сложность изучения с учётом всех предварительных требований.

**Алгоритм:** 
1. Берём подграф PREREQUISITE
2. Находим SCC и сворачиваем в DAG
3. DP: `effort[c] = sum(difficulty в компоненте) + max(effort[предков])`
4. Узлы одной SCC получают одинаковое значение

**Python реализация:**
```python
def scc_dag_depth_and_effort(H, default_difficulty=3.0):
    # SCC компоненты
    comp_list = list(nx.strongly_connected_components(H))
    comp_index = {n: i for i, comp in enumerate(comp_list) for n in comp}
    
    # Condensed DAG
    C = nx.DiGraph()
    C.add_nodes_from(range(len(comp_list)))
    for u, v in H.edges():
        cu, cv = comp_index[u], comp_index[v]
        if cu != cv:
            C.add_edge(cu, cv)
    
    # Суммы difficulty по компонентам
    comp_difficulty = {}
    for i, comp in enumerate(comp_list):
        s = sum(H.nodes[n].get("difficulty", default_difficulty) for n in comp)
        comp_difficulty[i] = s
    
    # Топологическая DP
    topo = list(nx.topological_sort(C))
    comp_effort = {i: 0.0 for i in C.nodes()}
    for c in topo:
        preds = list(C.predecessors(c))
        if preds:
            comp_effort[c] = max(comp_effort[p] for p in preds) + comp_difficulty[c]
        else:
            comp_effort[c] = comp_difficulty[c]
    
    # Разворачиваем на узлы
    effort = {n: float(comp_effort[comp_index[n]]) for n in H.nodes()}
    return effort
```

**Параметры конфига:**
- `default_difficulty` (default: 3) — значение если difficulty отсутствует в узле

**Граничные случаи:**
- Узел без difficulty: используется default_difficulty
- Изолированный узел: effort = его difficulty
- Цикл PREREQUISITE: все узлы цикла получают сумму их difficulties + max(предков)

**Зависимости:** Использует prerequisite_depth логику (тот же подграф).

---

### 2.10. educational_importance

**Образовательный смысл:** PageRank только по "образовательным" типам рёбер. Показывает важность в контексте обучения.

**Алгоритм:** PageRank на подграфе из рёбер типов PREREQUISITE, ELABORATES, TESTS, EXAMPLE_OF.

**Python реализация:**
```python
def educational_subgraph(G):
    allowed = {"PREREQUISITE", "ELABORATES", "TESTS", "EXAMPLE_OF"}
    E = [(u, v, d) for u, v, d in G.edges(data=True) 
         if str(d.get("type", "")).upper() in allowed]
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(E)
    return H

E = educational_subgraph(G)
if E.number_of_edges():
    edu_pr = nx.pagerank(E, alpha=0.85, weight="weight")
else:
    edu_pr = {u: 1.0/n for u in G.nodes()}
```

**Параметры конфига:**
- `educational_edge_types` (default: ["PREREQUISITE", "ELABORATES", "TESTS", "EXAMPLE_OF"])
- `pagerank_damping` (default: 0.85)

**Граничные случаи:**
- Граф без образовательных рёбер: равномерное распределение 1/n
- Только образовательные рёбра: совпадает с обычным PageRank

**Инвариант:** `sum(educational_importance) = 1.0 ± 0.01`

**Зависимости:** Использует ту же логику что и PageRank, но на подграфе.

---

## 3. Продвинутые метрики (Advanced Metrics)

### 3.1. cluster_id (Louvain clustering)

**Образовательный смысл:** Группировка тематически связанных узлов. Узлы одного кластера образуют смысловой блок знаний.

**Алгоритм:** Louvain community detection на undirected проекции графа с детерминированной перенумерацией.

**Python реализация:**
```python
import community as community_louvain

UG = G.to_undirected()
# Агрегация весов для двунаправленных рёбер
for u, v in list(UG.edges()):
    if UG.has_edge(v, u):
        UG[u][v]['weight'] = G.get_edge_data(u, v, {}).get('weight', 1.0) + \
                              G.get_edge_data(v, u, {}).get('weight', 1.0)

partition = community_louvain.best_partition(
    UG, 
    resolution=config['louvain_resolution'],
    random_state=config['louvain_random_state']
)

# Детерминированная перенумерация по минимальному ID узла
clusters = {}
for node, cluster in partition.items():
    clusters.setdefault(cluster, []).append(node)
    
sorted_clusters = sorted(clusters.items(), 
                        key=lambda x: min(node_order.index(n) for n in x[1]))

cluster_map = {}
for new_id, (old_id, nodes) in enumerate(sorted_clusters):
    for node in nodes:
        cluster_map[node] = new_id
```

**Параметры конфига:**
- `louvain_resolution` (default: 1.0) — контроль размера кластеров
- `louvain_random_state` (default: 42) — seed для детерминизма

**Граничные случаи:**
- Пустой граф: пустой dict
- Один узел: cluster_id = 0
- Несвязные компоненты: каждая компонента кластеризуется отдельно

**Зависимости:** Требует python-louvain>=0.16

---

### 3.2. bridge_score

**Образовательный смысл:** Узлы-мосты между разными тематическими блоками. Высокий bridge_score = узел связывает разные области знаний.

**Алгоритм:** Взвешенная комбинация betweenness centrality и доли межкластерных связей.

**Формула:** 
```
bridge_score = w_b * betweenness_norm + (1 - w_b) * inter_ratio
```
где:
- `w_b` = bridge_weight_betweenness (default: 0.7)
- `inter_ratio` = доля соседей в других кластерах

**Python реализация:**
```python
def compute_bridge_scores(G, cluster_map, betweenness_centrality, config):
    w_b = config.get('bridge_weight_betweenness', 0.7)
    bridge_scores = {}
    
    for node in G.nodes():
        neighbors = set(G.predecessors(node)) | set(G.successors(node))
        if neighbors:
            inter_count = sum(1 for n in neighbors 
                            if cluster_map.get(n, -1) != cluster_map.get(node, -1))
            inter_ratio = inter_count / len(neighbors)
        else:
            inter_ratio = 0.0
            
        btw_norm = betweenness_centrality.get(node, 0.0)
        bridge_scores[node] = w_b * btw_norm + (1 - w_b) * inter_ratio
        
    return bridge_scores
```

**Параметры конфига:**
- `bridge_weight_betweenness` (default: 0.7) — вес для betweenness компоненты

**Граничные случаи:**
- Узел без соседей: inter_ratio = 0, использует только betweenness
- Один кластер: все inter_ratio = 0
- Нет кластеризации: использует только betweenness

**Зависимости:** Требует предвычисленные cluster_id и betweenness_centrality.

---

### 3.3. Межкластерные метрики рёбер

**Образовательный смысл:** Выявление связей между тематическими блоками для понимания междисциплинарных зависимостей.

**Алгоритм:** Проверка принадлежности source и target разным кластерам.

**Python реализация:**
```python
for u, v, d in G.edges(data=True):
    src_cluster = cluster_map.get(u, -1)
    tgt_cluster = cluster_map.get(v, -1)
    
    if src_cluster != tgt_cluster and src_cluster >= 0 and tgt_cluster >= 0:
        G[u][v]['is_inter_cluster_edge'] = True
        G[u][v]['source_cluster_id'] = src_cluster
        G[u][v]['target_cluster_id'] = tgt_cluster
    else:
        G[u][v]['is_inter_cluster_edge'] = False
```

**Инвариант:** `is_inter_cluster_edge = True ⟺ source_cluster_id ≠ target_cluster_id`

---

## 4. Полный список метрик

### Метрики узлов (12):
1. degree_in, degree_out — входящая/исходящая степень
2. degree_centrality — нормализованная степень
3. pagerank — важность с учетом источников
4. betweenness_centrality — узел как мост
5. out-closeness — исходящая близость
6. component_id — компонента связности
7. prerequisite_depth — уровень в иерархии
8. learning_effort — накопленная сложность
9. educational_importance — образовательная важность
10. cluster_id — ID кластера (Louvain)
11. bridge_score — композитная метрика моста

### Метрики рёбер (4):
1. inverse_weight — обратный вес
2. is_inter_cluster_edge — флаг межкластерного ребра
3. source_cluster_id — кластер источника (для межкластерных)
4. target_cluster_id — кластер цели (для межкластерных)

---

## 5. Итоговые замечания

### Последовательность вычисления
1. **degree_in, degree_out** — базовые метрики
2. **degree_centrality** — использует степени
3. **inverse_weight** — подготовка для distance-based метрик
4. **pagerank** — независимая метрика
5. **betweenness_centrality** — требует inverse_weight
6. **out-closeness** — требует inverse_weight
7. **component_id** — независимая метрика
8. **prerequisite_depth** — анализ PREREQUISITE подграфа
9. **learning_effort** — расширение prerequisite_depth
10. **educational_importance** — PageRank на подграфе
11. **cluster_id** — Louvain кластеризация
12. **bridge_score** — требует cluster_id и betweenness_centrality

### Обработка граничных случаев
- Все метрики должны возвращать числовые значения (не NaN/Inf)
- При делении на 0 или других исключениях — возвращать 0.0
- Логировать аномальные ситуации

### Проверка корректности
- sum(PageRank) ≈ 1.0
- sum(educational_importance) ≈ 1.0
- component_id от 0 до k-1 (k компонент)
- prerequisite_depth ≥ 0 для всех узлов
- cluster_id от 0 до c-1 (c кластеров)
- bridge_score ∈ [0, 1] для всех узлов

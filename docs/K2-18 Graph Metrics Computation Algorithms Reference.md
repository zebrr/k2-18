# K2-18 VIZ: Graph Metrics Computation Algorithms Reference

## 1. Introduction

This document is the single source of truth for computing metrics of the educational knowledge graph in the K2-18 project.

### Covered Metrics
**12 node metrics:**
1. degree_in — incoming degree
2. degree_out — outgoing degree  
3. degree_centrality — normalized degree
4. pagerank — node importance
5. betweenness_centrality — node as bridge
6. out-closeness — outgoing closeness
7. component_id — connected component
8. prerequisite_depth — level in dependency tree
9. learning_effort — cumulative difficulty
10. educational_importance — importance in educational context
11. cluster_id — Louvain cluster ID
12. bridge_score — composite bridge metric

**4 edge metrics:**
1. inverse_weight — inverse weight for distance-based algorithms
2. is_inter_cluster_edge — flag for inter-cluster edges
3. source_cluster_id — source node's cluster (for inter-cluster edges)
4. target_cluster_id — target node's cluster (for inter-cluster edges)

### Computation Order
Metrics MUST be computed in the specified order due to dependencies.

---

## 2. Node Metrics

### 2.1. degree_in, degree_out

**Educational meaning:** Shows how many nodes this one depends on (in) and how many nodes it affects (out). High degree_in = many prerequisites, high degree_out = opens many new concepts.

**Algorithm:** Simple count of incoming and outgoing edges.

**Python implementation:**
```python
deg_in = dict(G.in_degree())
deg_out = dict(G.out_degree())
```

**Edge cases:**
- Isolated node: degree_in = 0, degree_out = 0
- Empty graph: all degrees = 0

---

### 2.2. degree_centrality

**Educational meaning:** How "connected" the node is relative to graph size. Shows the fraction of all possible connections the node has. If connected to all — close to 1, if isolated — 0.

**Algorithm:** `(degree_in + degree_out) / (n - 1)` for directed graphs.

**Python implementation:**
```python
deg_cent = nx.degree_centrality(G.to_undirected())
```

**Edge cases:**
- Graph with 1 node: degree_centrality = 0
- Complete graph: degree_centrality → 1

**Dependencies:** Uses degree_in and degree_out.

---

### 2.3. inverse_weight (for edges)

**Educational meaning:** Transforms weight into "distance" — the weaker the connection, the "farther" the nodes.

**Algorithm:** `inverse_weight = 1.0 / weight`

**Python implementation:**
```python
for u, v, d in G.edges(data=True):
    w = float(d.get('weight', 1.0))
    inv = (1.0 / w) if (w and w > 0) else float("inf")
    G[u][v]['inverse_weight'] = inv
```

**Edge cases:**
- weight = 0 or missing: inverse_weight = inf
- weight = 1: inverse_weight = 1

**Note:** MUST be computed BEFORE betweenness_centrality and out-closeness.

---

### 2.4. pagerank

**Educational meaning:** Node importance considering the importance of nodes pointing to it. Key feature — cumulative effect: importance is transferred and accumulated through chains. A node is important not by itself, but if other important nodes reference it. Basic concepts transfer importance to derived ones.

**Algorithm:** Iterative algorithm with random jumps. Dangling nodes (without outgoing edges) distribute their mass uniformly.

**Python implementation:**
```python
if G.number_of_edges():
    pr = nx.pagerank(G, alpha=0.85, weight="weight")
else:
    pr = {u: 1.0/n for u in G.nodes()}
```

**Config parameters:**
- `pagerank_damping` (default: 0.85) — probability of following edges

**Edge cases:**
- Empty graph: all nodes get 1/n
- Graph without edges: all nodes get 1/n
- Dangling nodes: their PageRank is redistributed uniformly

**Invariant:** `sum(PageRank) = 1.0 ± 0.01`

---

### 2.5. betweenness_centrality

**Educational meaning:** Node as a "bridge" between parts of the graph. High value = node is critical for knowledge connectivity.

**Algorithm:** Fraction of shortest paths passing through the node. Uses INVERSE weights (smaller weight = "more expensive" path).

**Python implementation:**
```python
if n >= 3:
    btw = nx.betweenness_centrality(G, weight="inverse_weight", normalized=True)
else:
    btw = {u: 0.0 for u in G.nodes()}
```

**Config parameters:**
- `betweenness_normalized` (default: true) — normalization by `(n-1)*(n-2)`

**Edge cases:**
- Graph < 3 nodes: all betweenness = 0
- Linear graph: middle node has maximum value

**Dependencies:** Requires pre-computed inverse_weight on edges.

---

### 2.6. out-closeness

**Educational meaning:** How "close" the node is to others through outgoing paths. High value = node can easily reach many others.

**Algorithm:** OUT-closeness for directed graphs via reversal. Wasserman-Faust formula for partial reachability.

**Python implementation:**
```python
if n > 1:
    Gr = G.reverse(copy=True)
    out_close = nx.closeness_centrality(Gr, distance="inverse_weight", wf_improved=True)
else:
    out_close = {u: 0.0 for u in G.nodes()}
```

**Config parameters:**
- `closeness_harmonic` (default: true) — use harmonic centralization for disconnected graphs

**Edge cases:**
- Graph with 1 node: out-closeness = 0
- Isolated node: out-closeness = 0
- Node without outgoing edges: out-closeness = 0

**Dependencies:** Requires pre-computed inverse_weight on edges.

---

### 2.7. component_id

**Educational meaning:** Grouping nodes into connected subgraphs. Nodes of the same component are reachable from each other.

**Algorithm:** Weakly connected components with deterministic numbering by node order in file.

**Python implementation:**
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

**Edge cases:**
- Fully connected graph: all nodes have component_id = 0
- n isolated nodes: component_id from 0 to n-1

**Invariant:** component_id starts from 0 and goes sequentially.

---

### 2.8. prerequisite_depth

**Educational meaning:** Node level in the prerequisite hierarchy. 0 = basic concepts, then increasing.

**Algorithm:** Maximum path length via PREREQUISITE edges from nodes without incoming PREREQUISITE.

**Python implementation:**
```python
def prereq_subgraph(G):
    E = [(u, v, d) for u, v, d in G.edges(data=True) 
         if str(d.get("type", "")).upper() == "PREREQUISITE"]
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(E)
    return H

# In scc_dag_depth_and_effort function:
# 1. Build SCC (strongly connected components)
# 2. Create condensed DAG
# 3. Topological sort
# 4. DP: depth[c] = max(depth[pred]) + 1
```

**Edge cases:**
- Nodes without incoming PREREQUISITE: depth = 0
- PREREQUISITE cycles: all cycle nodes get the same depth
- Graph without PREREQUISITE edges: all nodes have depth = 0

**Dependencies:** Analyzes only PREREQUISITE edges subgraph.

---

### 2.9. learning_effort

**Educational meaning:** Cumulative learning difficulty considering all prerequisites.

**Algorithm:** 
1. Take PREREQUISITE subgraph
2. Find SCC and collapse into DAG
3. DP: `effort[c] = sum(difficulty in component) + max(effort[ancestors])`
4. Nodes of same SCC get same value

**Python implementation:**
```python
def scc_dag_depth_and_effort(H, default_difficulty=3.0):
    # SCC components
    comp_list = list(nx.strongly_connected_components(H))
    comp_index = {n: i for i, comp in enumerate(comp_list) for n in comp}
    
    # Condensed DAG
    C = nx.DiGraph()
    C.add_nodes_from(range(len(comp_list)))
    for u, v in H.edges():
        cu, cv = comp_index[u], comp_index[v]
        if cu != cv:
            C.add_edge(cu, cv)
    
    # Sum difficulties by component
    comp_difficulty = {}
    for i, comp in enumerate(comp_list):
        s = sum(H.nodes[n].get("difficulty", default_difficulty) for n in comp)
        comp_difficulty[i] = s
    
    # Topological DP
    topo = list(nx.topological_sort(C))
    comp_effort = {i: 0.0 for i in C.nodes()}
    for c in topo:
        preds = list(C.predecessors(c))
        if preds:
            comp_effort[c] = max(comp_effort[p] for p in preds) + comp_difficulty[c]
        else:
            comp_effort[c] = comp_difficulty[c]
    
    # Expand to nodes
    effort = {n: float(comp_effort[comp_index[n]]) for n in H.nodes()}
    return effort
```

**Config parameters:**
- `default_difficulty` (default: 3) — value if difficulty is missing in node

**Edge cases:**
- Node without difficulty: uses default_difficulty
- Isolated node: effort = its difficulty
- PREREQUISITE cycle: all cycle nodes get sum of their difficulties + max(ancestors)

**Dependencies:** Uses prerequisite_depth logic (same subgraph).

---

### 2.10. educational_importance

**Educational meaning:** PageRank only on "educational" edge types. Shows importance in learning context.

**Algorithm:** PageRank on subgraph from edges of types PREREQUISITE, ELABORATES, TESTS, EXAMPLE_OF.

**Python implementation:**
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

**Config parameters:**
- `educational_edge_types` (default: ["PREREQUISITE", "ELABORATES", "TESTS", "EXAMPLE_OF"])
- `pagerank_damping` (default: 0.85)

**Edge cases:**
- Graph without educational edges: uniform distribution 1/n
- Only educational edges: coincides with regular PageRank

**Invariant:** `sum(educational_importance) = 1.0 ± 0.01`

**Dependencies:** Uses same logic as PageRank, but on subgraph.

---

## 3. Advanced Metrics

### 3.1. cluster_id (Louvain clustering)

**Educational meaning:** Grouping of thematically related nodes. Nodes of the same cluster form a semantic knowledge block.

**Algorithm:** Louvain community detection on undirected graph projection with deterministic renumbering.

**Python implementation:**
```python
import community as community_louvain

UG = G.to_undirected()
# Aggregate weights for bidirectional edges
for u, v in list(UG.edges()):
    if UG.has_edge(v, u):
        UG[u][v]['weight'] = G.get_edge_data(u, v, {}).get('weight', 1.0) + \
                              G.get_edge_data(v, u, {}).get('weight', 1.0)

partition = community_louvain.best_partition(
    UG, 
    resolution=config['louvain_resolution'],
    random_state=config['louvain_random_state']
)

# Deterministic renumbering by minimum node ID
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

**Config parameters:**
- `louvain_resolution` (default: 1.0) — cluster size control
- `louvain_random_state` (default: 42) — seed for determinism

**Edge cases:**
- Empty graph: empty dict
- Single node: cluster_id = 0
- Disconnected components: each component is clustered separately

**Dependencies:** Requires python-louvain>=0.16

---

### 3.2. bridge_score

**Educational meaning:** Bridge nodes between different thematic blocks. High bridge_score = node connects different knowledge areas.

**Algorithm:** Weighted combination of betweenness centrality and inter-cluster connection ratio.

**Formula:** 
```
bridge_score = w_b * betweenness_norm + (1 - w_b) * inter_ratio
```
where:
- `w_b` = bridge_weight_betweenness (default: 0.7)
- `inter_ratio` = fraction of neighbors in other clusters

**Python implementation:**
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

**Config parameters:**
- `bridge_weight_betweenness` (default: 0.7) — weight for betweenness component

**Edge cases:**
- Node without neighbors: inter_ratio = 0, uses only betweenness
- Single cluster: all inter_ratio = 0
- No clustering: uses only betweenness

**Dependencies:** Requires pre-computed cluster_id and betweenness_centrality.

---

### 3.3. Inter-cluster Edge Metrics

**Educational meaning:** Identifying connections between thematic blocks for understanding interdisciplinary dependencies.

**Algorithm:** Check if source and target belong to different clusters.

**Python implementation:**
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

**Invariant:** `is_inter_cluster_edge = True ⟺ source_cluster_id ≠ target_cluster_id`

---

## 4. Complete Metrics List

### Node metrics (12):
1. degree_in, degree_out — incoming/outgoing degree
2. degree_centrality — normalized degree
3. pagerank — importance considering sources
4. betweenness_centrality — node as bridge
5. out-closeness — outgoing closeness
6. component_id — connected component
7. prerequisite_depth — level in hierarchy
8. learning_effort — cumulative difficulty
9. educational_importance — educational importance
10. cluster_id — Louvain cluster ID
11. bridge_score — composite bridge metric

### Edge metrics (4):
1. inverse_weight — inverse weight
2. is_inter_cluster_edge — inter-cluster edge flag
3. source_cluster_id — source cluster (for inter-cluster)
4. target_cluster_id — target cluster (for inter-cluster)

---

## 5. Computation Sequence

### Sequence Order
1. **degree_in, degree_out** — basic metrics
2. **degree_centrality** — uses degrees
3. **inverse_weight** — preparation for distance-based metrics
4. **pagerank** — independent metric
5. **betweenness_centrality** — requires inverse_weight
6. **out-closeness** — requires inverse_weight
7. **component_id** — independent metric
8. **prerequisite_depth** — PREREQUISITE subgraph analysis
9. **learning_effort** — extension of prerequisite_depth
10. **educational_importance** — PageRank on subgraph
11. **cluster_id** — Louvain clustering
12. **bridge_score** — requires cluster_id and betweenness_centrality

### Edge Case Handling
- All metrics must return numeric values (not NaN/Inf)
- On division by 0 or other exceptions — return 0.0
- Log anomalous situations

### Correctness Verification
- sum(PageRank) ≈ 1.0
- sum(educational_importance) ≈ 1.0
- component_id from 0 to k-1 (k components)
- prerequisite_depth ≥ 0 for all nodes
- cluster_id from 0 to c-1 (c clusters)
- bridge_score ∈ [0, 1] for all nodes

---

## 6. Helper Functions

### 6.1. sanitize_graph_weights

**Purpose:** Ensure numerical stability of edge weights before computing metrics.

**Algorithm:**
1. Remove self-loops via `nx.selfloop_edges()`
2. Replace missing weights with 1.0
3. Replace zero/negative weights with eps (1e-9)
4. Ensure inverse_weight exists on all edges

**Python implementation:**
```python
def sanitize_graph_weights(G: nx.DiGraph, eps: float = 1e-9) -> None:
    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))
    
    # Fix weights
    for u, v, d in G.edges(data=True):
        w = d.get('weight', 1.0)
        if not w or w <= 0:
            d['weight'] = eps
        if 'inverse_weight' not in d:
            d['inverse_weight'] = 1.0 / d['weight']
```

**When to apply:** Before `compute_louvain_clustering` and `compute_bridge_scores`

**Edge cases:**
- Empty graph: No operation
- All weights invalid: All set to eps

---

### 6.2. safe_metric_value

**Purpose:** Convert unsafe values to JSON-serializable format.

**Algorithm:**
- None → 0.0
- NaN → 0.0  
- ±inf → 0.0
- Valid number → unchanged

**Python implementation:**
```python
def safe_metric_value(value: Any) -> float:
    if value is None:
        return 0.0
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return float(value)
```

**When to apply:** When saving metrics to JSON

**Usage example:**
```python
for node in graph_data['nodes']:
    for metric in ['pagerank', 'betweenness_centrality', ...]:
        node[metric] = safe_metric_value(computed_metrics[node['id']][metric])
```
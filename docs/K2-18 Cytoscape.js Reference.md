# Cytoscape.js — LLM-Oriented Reference (Plain HTML, No Build System)

This reference is designed for LLM context: concise, machine-readable, with runnable code snippets.
Focus: **simple HTML environment** (no bundler), **graph algorithms**, **centralities**, **clustering**.

## 0) Quick HTML Setup (CDN, no build)
**Rules**
- Place CSS **before** Cytoscape init so the container has correct dimensions.
- Use any CDN; examples use `unpkg`.

```html
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Cytoscape.js Plain HTML</title>
    <style>
      /* Cytoscape reads container size at init */
      #cy {
        width: 100vw;
        height: 100vh;
        display: block;
      }
    </style>
    <!-- UMD build from CDN -->
    <script src="https://unpkg.com/cytoscape/dist/cytoscape.min.js"></script>
  </head>
  <body>
    <div id="cy"></div>
    <script>
      // Minimal graph for examples
      const cy = cytoscape({
        container: document.getElementById('cy'),
        elements: [
          { data: { id: 'a' } },
          { data: { id: 'b' } },
          { data: { id: 'c' } },
          { data: { id: 'd' } },
          { data: { id: 'ab', source: 'a', target: 'b', weight: 2 } },
          { data: { id: 'bc', source: 'b', target: 'c', weight: 1 } },
          { data: { id: 'cd', source: 'c', target: 'd', weight: 3 } },
          { data: { id: 'ad', source: 'a', target: 'd', weight: 10 } }
        ],
        style: [
          { selector: 'node', style: { 'background-color': '#555', 'label': 'data(id)' } },
          { selector: 'edge', style: { 'width': 2, 'curve-style': 'bezier', 'target-arrow-shape': 'triangle' } }
        ],
        layout: { name: 'grid', rows: 2 }
      });

      // Expose for console testing
      window.cy = cy;
    </script>
  </body>
</html>
````

---

## 1) Core Selection & Collections (for all algorithms)

* **Whole graph**: `cy.elements()`
* **Nodes/Edges**: `cy.nodes()`, `cy.edges()`
* **Selector**: `cy.$('node[id = "a"]')`, `cy.$('#a')`
* **Data access**: `ele.data('weight')`, `ele.data({ score: 0.42 })`
* **Path highlighting**: `path.select()` or add a class/style.

**Weight function pattern**

```js
const weight = e => Number(e.data('weight')) || 1; // positive weight default=1
```

**Directed flag pattern**

```js
const directed = false; // set true when edges should be treated as directed
```

---

## 2) Traversals

### 2.1 Breadth-First Search (BFS)

**Signature**: `eles.bfs({ roots, visit(v,e,u,i,depth), directed })`
**Returns**: `{ path: collection, found: node }`

```js
const bfs = cy.elements().bfs({
  roots: '#a',
  visit: (v, e, u, i, depth) => {
    // v: current node, e: edge from u->v, u: predecessor
    if (v.id() === 'd') return true; // stop when found
  },
  directed
});
bfs.path.addClass('bfs-path'); // style if needed
```

### 2.2 Depth-First Search (DFS)

**Signature**: `eles.dfs({ roots, visit, directed })`
**Returns**: `{ path: collection, found: node }`

```js
const dfs = cy.elements().dfs({
  roots: '#a',
  visit: v => v.data('marker') === 'target',
  directed
});
dfs.path.addClass('dfs-path');
```

---

## 3) Shortest Paths

### 3.1 Dijkstra (single-source)

**Signature**: `eles.dijkstra(root, weightFn?, directed?)` or `eles.dijkstra({ root, weight, directed })`
**Returns**: `{ distanceTo(node), pathTo(node) }`

```js
const dijkstra = cy.elements().dijkstra({
  root: '#a',
  weight,
  directed
});

const distAtoD = dijkstra.distanceTo('#d');     // number
const pathAtoD = dijkstra.pathTo('#d');         // collection
pathAtoD.addClass('shortest');
```

### 3.2 A\* (single-pair)

**Signature**: `eles.aStar({ root, goal, weight?, heuristic?, directed? })`
**Returns**: `{ found, distance, path }`
**Heuristic**: `node => 0` (Dijkstra), or estimate (if you store coords in data/position).

```js
const aStar = cy.elements().aStar({
  root: '#a',
  goal: '#d',
  weight,
  heuristic: node => 0, // or Euclidean on node.position()
  directed
});
if (aStar.found) aStar.path.addClass('astar');
```

### 3.3 Floyd–Warshall (all-pairs)

**Signature**: `eles.floydWarshall({ weight?, directed? })`
**Returns**: `{ distance(u,v), path(u,v) }`

```js
const fw = cy.elements().floydWarshall({ weight, directed });
const dist = fw.distance('#a', '#c');
const path = fw.path('#a', '#c');
path.addClass('fw-path');
```

### 3.4 Bellman–Ford (single-source, negative edges OK)

**Signature**: `eles.bellmanFord({ root, weight?, directed? })`
**Returns**: `{ distanceTo(node), pathTo(node), hasNegativeWeightCycle, negativeWeightCycles }`

```js
const bf = cy.elements().bellmanFord({ root: '#a', weight, directed });
if (!bf.hasNegativeWeightCycle) {
  const dist = bf.distanceTo('#d');
  bf.pathTo('#d').addClass('bf-path');
}
```

### 3.5 Euler Trail (Hierholzer)

**Signature**: `eles.hierholzer({ root?, directed? })`
**Returns**: `{ found, trail }`

```js
const euler = cy.elements().hierholzer({ directed });
if (euler.found) euler.trail.addClass('euler-trail');
```

### 3.6 Minimum Spanning Tree (Kruskal)

**Signature**: `eles.kruskal(weightFn?)`
**Returns**: MST as a **collection of edges/nodes** (undirected graphs).

```js
const mst = cy.elements().kruskal(weight);
mst.addClass('mst');
```

### 3.7 Global Min Cut (Karger–Stein)

**Signature**: `eles.kargerStein()`
**Returns**: `{ cut: collection, components: collection[] }`

```js
const ks = cy.elements().kargerStein();
ks.cut.addClass('mincut');
```

---

## 4) Connectivity

### 4.1 Biconnected Components (Hopcroft–Tarjan)

**Signature**: `eles.hopcroftTarjanBiconnectedComponents()` (aliases: `htb`, `htbc`)
**Returns**: `{ cut, components: collection[] }`  // cut typically articulation points

```js
const bc = cy.elements().hopcroftTarjanBiconnectedComponents();
bc.components.forEach(c => c.addClass('biconnected'));
```

### 4.2 Strongly Connected Components (Tarjan)

**Signature**: `eles.tarjanStronglyConnectedComponents()` (aliases: `tsc`, `tscc`)
**Returns**: `{ cut, components: collection[] }`

```js
const scc = cy.elements().tarjanStronglyConnectedComponents();
scc.components.forEach(c => c.addClass('scc'));
```

---

## 5) Centralities

### 5.1 Degree Centrality

**Signature**: `eles.degreeCentrality({ root?, weight?, alpha?, directed? })` (alias `dc`)
**Returns**:

* Undirected: `{ degree }`
* Directed: `{ indegree, outdegree }`

```js
const dc = cy.elements().degreeCentrality({ root: '#a', directed });
console.log(dc.degree); // or indegree/outdegree if directed:true
```

### 5.2 Degree Centrality (Normalized)

**Signature**: `eles.degreeCentralityNormalized({ weight?, alpha?, directed? })` (alias `dcn`)
**Returns**: object with methods like `degree(node)` or `{ indegree(node), outdegree(node) }`

```js
const dcn = cy.elements().degreeCentralityNormalized({ directed });
cy.nodes().forEach(n => n.data('deg_norm', dcn.degree(n)));
```

### 5.3 Closeness Centrality

**Signature**: `eles.closenessCentrality({ root, weight?, directed?, harmonic? })` (alias `cc`)
**Returns**: number (for root). `harmonic:true` works better on not-strongly-connected graphs.

```js
const cc = cy.elements().closenessCentrality({ root: '#a', weight, directed, harmonic: true });
cy.$('#a').data('closeness', cc);
```

### 5.4 Closeness Centrality (Normalized)

**Signature**: `eles.closenessCentralityNormalized({ weight?, directed?, harmonic? })` (alias `ccn`)
**Returns**: `{ closeness(node) }`

```js
const ccn = cy.elements().closenessCentralityNormalized({ weight, directed, harmonic: true });
cy.nodes().forEach(n => n.data('closeness_norm', ccn.closeness(n)));
```

### 5.5 Betweenness Centrality

**Signature**: `eles.betweennessCentrality({ weight?, directed? })` (alias `bc`)
**Returns**: `{ betweenness(node), betweennessNormalized(node) }`

```js
const bc = cy.elements().betweennessCentrality({ weight, directed });
cy.nodes().forEach(n => n.data('bc', bc.betweenness(n)));
```

### 5.6 PageRank

**Signature**: `eles.pageRank({ dampingFactor?, precision?, iterations? })`
**Returns**: `{ rank(node) }`

```js
const pr = cy.elements().pageRank({ dampingFactor: 0.8, iterations: 200 });
cy.nodes().forEach(n => n.data('pagerank', pr.rank(n)));
```

---

## 6) Clustering (attribute-based unless noted)

### 6.1 Markov Clustering (MCL)

**Signature**: `eles.markovClustering({ attributes, expandFactor?, inflateFactor?, multFactor?, maxIterations? })`
**Returns**: `collection[]` (clusters)

```js
const mcl = cy.elements().markovClustering({ attributes: [e => e.data('weight')] });
mcl.forEach(cluster => cluster.addClass('cluster-mcl'));
```

### 6.2 k-Means

**Signature**: `nodes.kMeans({ attributes, k, distance?, maxIterations?, sensitivityThreshold? })`
**Returns**: `collection[]`

```js
const kmeans = cy.nodes().kMeans({ k: 3, attributes: [n => n.degree(false)] });
kmeans.forEach((cluster, i) => cluster.addClass(`kmeans-${i}`));
```

### 6.3 k-Medoids

**Signature**: `nodes.kMedoids({ attributes, k, distance?, maxIterations? })`
**Returns**: `collection[]`

```js
const kmedoids = cy.nodes().kMedoids({ k: 2, attributes: [n => n.data('pagerank') || 0] });
kmedoids.forEach((cluster, i) => cluster.addClass(`kmedoids-${i}`));
```

### 6.4 Fuzzy c-Means

**Signature**: `nodes.fuzzyCMeans({ attributes, k, distance?, maxIterations?, sensitivityThreshold? })`
**Returns**: `{ clusters: collection[], degreeOfMembership: number[][] }`

```js
const fcm = cy.nodes().fuzzyCMeans({ k: 2, attributes: [n => n.data('pagerank') || 0] });
fcm.clusters.forEach((cluster, i) => cluster.addClass(`fcm-${i}`));
```

### 6.5 Hierarchical Clustering (Agglomerative)

**Signature**: `nodes.hierarchicalClustering({ attributes, distance?, linkage?, mode?, threshold?, dendrogramDepth?, addDendrogram? })`
**Returns**: `collection[]`

```js
const hca = cy.nodes().hierarchicalClustering({
  attributes: [n => n.data('pagerank') || 0],
  linkage: 'single', // 'single'|'complete'|'mean'
  mode: 'threshold',
  threshold: 0.5
});
hca.forEach((cluster, i) => cluster.addClass(`hca-${i}`));
```

### 6.6 Affinity Propagation

**Signature**: `nodes.affinityPropagation({ attributes, distance?, preference?, damping?, minIterations?, maxIterations? })`
**Returns**: `collection[]`

```js
const ap = cy.nodes().affinityPropagation({
  attributes: [n => n.data('pagerank') || 0],
  preference: 'median',
  damping: 0.8
});
ap.forEach((cluster, i) => cluster.addClass(`ap-${i}`));
```

---

## 7) Practical Patterns for Agents

### 7.1 Compute metrics → store → style / rank

```js
// Compute once:
const pr = cy.elements().pageRank();
const bc = cy.elements().betweennessCentrality({ weight });
const ccn = cy.elements().closenessCentralityNormalized({ weight, harmonic: true });

// Store to node data:
cy.nodes().forEach(n => {
  n.data({
    pagerank: pr.rank(n),
    bc: bc.betweenness(n),
    closeness_norm: ccn.closeness(n)
  });
});

// Example: top-N by PageRank
const topN = cy.nodes().sort((a, b) => b.data('pagerank') - a.data('pagerank')).slice(0, 10);
topN.forEach(n => n.addClass('topN'));

// Style by thresholds (append to stylesheet as needed)
cy.style().selector('node[topN]').style({ 'border-width': 4, 'border-color': '#f80' }).update();
```

### 7.2 Weighted shortest path with highlighted result

```js
const dijkstra = cy.elements().dijkstra({ root: '#a', weight });
const target = '#d';
const p = dijkstra.pathTo(target);
p.addClass('highlight');
```

### 7.3 Handle directed vs undirected

```js
const dir = true;
const prDir = cy.elements().pageRank(); // PageRank is inherently directed over out-edges
const dijkDir = cy.elements().dijkstra({ root: '#a', weight, directed: dir });
```

### 7.4 Negative weights

* Use **Bellman–Ford** for single-source shortest paths with negative edges.
* **Floyd–Warshall** supports negatives for all-pairs (no negative cycles).
* Detect cycles: `bf.hasNegativeWeightCycle`.

---

## 8) Styling Helpers for Paths/Clusters (optional)

```js
cy.style()
  .selector('.highlight').style({ 'line-color': '#f40', 'target-arrow-color': '#f40', 'width': 4 })
  .selector('.cluster-mcl').style({ 'background-color': '#3a7' })
  .selector('.mst').style({ 'line-color': '#0a7', 'width': 3 })
  .update();
```

---

## 9) Common Gotchas

* Container size must be set **before** `cytoscape({...})`.
* Weights must be **positive** for most algorithms (except Bellman–Ford/Floyd–Warshall).
* Algorithms operate on the **calling collection**; use `cy.elements()` for whole graph or filter first.
* For large graphs, prefer precomputing and caching metrics in `data()`.

#!/usr/bin/env python3
"""
graph2metrics.py - Compute metrics for K2-18 knowledge graph visualization.

This module enriches the LearningChunkGraph with NetworkX metrics for visualization.
Input: LearningChunkGraph.json and ConceptDictionary.json
Output: LearningChunkGraph_wow.json and ConceptDictionary_wow.json with metrics
"""

import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

# Try to import python-louvain for community detection
try:
    import community as community_louvain

    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    # Will log warning when trying to use

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import ConfigValidationError, load_config  # noqa: E402
from src.utils.console_encoding import setup_console_encoding  # noqa: E402
from src.utils.exit_codes import (  # noqa: E402
    EXIT_CONFIG_ERROR,
    EXIT_INPUT_ERROR,
    EXIT_IO_ERROR,
    EXIT_RUNTIME_ERROR,
    EXIT_SUCCESS,
    log_exit,
)
from src.utils.validation import (  # noqa: E402
    GraphInvariantError,
    ValidationError,
    validate_graph_invariants,
    validate_json,
)


def setup_logging(log_file: Path, test_mode: bool = False) -> logging.Logger:
    """Set up logging configuration.

    Args:
        log_file: Path to log file
        test_mode: Whether running in test mode

    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )

    logger = logging.getLogger(__name__)
    if test_mode:
        logger.info("[TEST MODE] Logging initialized")
    else:
        logger.info("Logging initialized")

    return logger


def load_input_data(
    input_dir: Path, logger: logging.Logger, test_mode: bool = False
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load and validate input JSON files.

    Args:
        input_dir: Directory containing input files
        logger: Logger instance
        test_mode: Whether running in test mode

    Returns:
        Tuple of (graph_data, concepts_data)

    Raises:
        FileNotFoundError: If input files not found
        ValidationError: If validation fails
    """
    mode_prefix = "[TEST MODE] " if test_mode else ""

    # Define file paths based on mode
    if test_mode:
        # Test mode uses tiny_* files
        graph_file = input_dir / "tiny_graph.json"
        concepts_file = input_dir / "tiny_concepts.json"
    else:
        # Production mode uses standard names
        graph_file = input_dir / "LearningChunkGraph.json"
        concepts_file = input_dir / "ConceptDictionary.json"

    # Check files exist
    if not graph_file.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_file}")
    if not concepts_file.exists():
        raise FileNotFoundError(f"Concepts file not found: {concepts_file}")

    logger.info(f"{mode_prefix}Loading input files from {input_dir}")

    # Load graph
    logger.info(f"{mode_prefix}Loading graph: {graph_file}")
    with open(graph_file, encoding="utf-8") as f:
        graph_data = json.load(f)

    # Load concepts
    logger.info(f"{mode_prefix}Loading concepts: {concepts_file}")
    with open(concepts_file, encoding="utf-8") as f:
        concepts_data = json.load(f)

    # Validate data
    logger.info(f"{mode_prefix}Validating graph data")
    validate_json(graph_data, "LearningChunkGraph")
    validate_graph_invariants(graph_data)

    logger.info(f"{mode_prefix}Validating concepts data")
    validate_json(concepts_data, "ConceptDictionary")

    return graph_data, concepts_data


def convert_to_networkx(
    graph_data: Dict[str, Any], logger: logging.Logger, test_mode: bool = False
) -> nx.DiGraph:
    """Convert JSON graph to NetworkX DiGraph.

    Args:
        graph_data: Graph data from JSON
        logger: Logger instance
        test_mode: Whether running in test mode

    Returns:
        NetworkX directed graph
    """
    mode_prefix = "[TEST MODE] " if test_mode else ""

    logger.info(f"{mode_prefix}Converting to NetworkX DiGraph")

    # Create directed graph (ВАЖНО: DiGraph, не Graph!)
    G = nx.DiGraph()  # noqa: N806

    # Add nodes with their attributes
    for node in graph_data["nodes"]:
        G.add_node(node["id"], **node)

    # Add edges with their attributes
    for edge in graph_data["edges"]:
        G.add_edge(
            edge["source"],
            edge["target"],
            type=edge.get("type"),
            weight=edge.get("weight", 1.0),
            conditions=edge.get("conditions"),
        )

    # Verify it's directed
    if not G.is_directed():
        raise RuntimeError("Graph conversion failed: result is not directed")

    # Log basic statistics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    num_components = nx.number_weakly_connected_components(G)

    logger.info(f"{mode_prefix}Graph statistics:")
    logger.info(f"{mode_prefix}  - Nodes: {num_nodes}")
    logger.info(f"{mode_prefix}  - Edges: {num_edges}")
    logger.info(f"{mode_prefix}  - Weakly connected components: {num_components}")
    logger.info(f"{mode_prefix}  - Is directed: {G.is_directed()}")

    return G


def safe_metric_value(value: Any) -> float:
    """Convert NaN/inf to 0.0 for isolated nodes.

    Args:
        value: Metric value that might be NaN/inf

    Returns:
        Safe float value (0.0 for NaN/inf)
    """
    if value is None or math.isnan(value) or math.isinf(value):
        return 0.0
    return float(value)


def compute_edge_weights(G: nx.DiGraph, logger: Optional[logging.Logger]) -> nx.DiGraph:
    """Add inverse_weight to all edges for distance algorithms.

    Args:
        G: NetworkX directed graph
        logger: Logger instance (optional)

    Returns:
        Graph with inverse_weight added to edges
    """
    if logger:
        logger.info("Computing inverse edge weights")

    for u, v, d in G.edges(data=True):
        weight = float(d.get("weight", 1.0))
        if weight > 0:
            G[u][v]["inverse_weight"] = 1.0 / weight
        else:
            G[u][v]["inverse_weight"] = float("inf")

    return G


def compute_distance_centrality(
    G: nx.DiGraph, logger: Optional[logging.Logger]
) -> Dict[str, Dict[str, float]]:
    """Compute betweenness and OUT-closeness using inverse weights.

    Args:
        G: NetworkX directed graph with inverse_weight on edges
        logger: Logger instance (optional)

    Returns:
        Dictionary with betweenness and out_closeness metrics
    """
    if logger:
        logger.info("Computing distance-based centrality metrics")

    # Betweenness with inverse_weight
    if G.number_of_nodes() >= 3:
        betweenness = nx.betweenness_centrality(G, weight="inverse_weight", normalized=True)
    else:
        betweenness = {n: 0.0 for n in G.nodes()}

    # OUT-closeness via REVERSE graph
    if G.number_of_nodes() > 1:
        Gr = G.reverse(copy=True)
        out_closeness = nx.closeness_centrality(Gr, distance="inverse_weight", wf_improved=True)
    else:
        out_closeness = {n: 0.0 for n in G.nodes()}

    return {"betweenness": betweenness, "out_closeness": out_closeness}


def compute_component_ids(
    G: nx.DiGraph, node_order: List[str], logger: Optional[logging.Logger]
) -> Dict[str, int]:
    """Compute weakly connected components with deterministic IDs.

    Args:
        G: NetworkX directed graph
        node_order: Original node order from JSON
        logger: Logger instance (optional)

    Returns:
        Dictionary mapping node ID to component ID
    """
    if logger:
        logger.info("Computing component IDs")

    UG = G.to_undirected()
    components = list(nx.connected_components(UG))

    # Sort by first node's position in original order
    order_map = {n: i for i, n in enumerate(node_order)}
    sorted_comps = sorted(components, key=lambda c: min(order_map.get(n, 10**9) for n in c))

    # Assign sequential IDs
    component_map = {}
    for cid, comp in enumerate(sorted_comps):
        for node in comp:
            component_map[node] = cid

    return component_map


def compute_prerequisite_metrics(
    G: nx.DiGraph,
    graph_data: Dict[str, Any],
    config: Dict[str, Any],
    logger: Optional[logging.Logger],
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """Compute prerequisite_depth and learning_effort via SCC and DAG.

    Args:
        G: NetworkX directed graph
        graph_data: Original graph data with node difficulties
        config: Configuration dictionary
        logger: Logger instance (optional)

    Returns:
        Tuple of (prerequisite_depth, learning_effort) dictionaries
    """
    if logger:
        logger.info("Computing prerequisite metrics")

    # Get default difficulty from config
    default_diff = config.get("path_mode", {}).get("default_difficulty", 3)

    # Build PREREQUISITE subgraph
    prereq_edges = [
        (u, v) for u, v, d in G.edges(data=True) if str(d.get("type", "")).upper() == "PREREQUISITE"
    ]

    H = nx.DiGraph()
    H.add_nodes_from(G.nodes())

    # Copy node attributes including difficulty
    node_map = {node["id"]: node for node in graph_data["nodes"]}
    for node_id in H.nodes():
        if node_id in node_map:
            H.nodes[node_id]["difficulty"] = node_map[node_id].get("difficulty", default_diff)
        else:
            H.nodes[node_id]["difficulty"] = default_diff

    H.add_edges_from(prereq_edges)

    # Handle empty prerequisite graph
    if H.number_of_edges() == 0:
        # No prerequisites - all nodes have depth 0 and effort = difficulty
        node_depth = {n: 0 for n in G.nodes()}
        node_effort = {n: float(H.nodes[n].get("difficulty", default_diff)) for n in G.nodes()}
        return node_depth, node_effort

    # Find SCCs
    scc_list = list(nx.strongly_connected_components(H))
    scc_index = {n: i for i, scc in enumerate(scc_list) for n in scc}

    # Build condensed DAG
    C = nx.DiGraph()
    C.add_nodes_from(range(len(scc_list)))

    for u, v in H.edges():
        cu, cv = scc_index[u], scc_index[v]
        if cu != cv and not C.has_edge(cu, cv):
            C.add_edge(cu, cv)

    # Compute difficulty sums per SCC
    scc_difficulty = {}
    for i, scc in enumerate(scc_list):
        total_diff = sum(H.nodes[n].get("difficulty", default_diff) for n in scc)
        scc_difficulty[i] = total_diff

    # Topological DP for depth and effort
    if C.number_of_nodes() > 0 and C.number_of_edges() > 0:
        try:
            topo_order = list(nx.topological_sort(C))
        except nx.NetworkXError:
            # Should not happen after SCC, but handle gracefully
            topo_order = list(C.nodes())

        scc_depth = {i: 0 for i in C.nodes()}
        scc_effort = {i: 0.0 for i in C.nodes()}

        for c in topo_order:
            preds = list(C.predecessors(c))
            if preds:
                scc_depth[c] = max(scc_depth[p] for p in preds) + 1
                scc_effort[c] = max(scc_effort[p] for p in preds) + scc_difficulty[c]
            else:
                scc_effort[c] = scc_difficulty[c]
    else:
        # All nodes in single SCC or no edges in condensed graph
        scc_depth = {i: 0 for i in range(len(scc_list))}
        scc_effort = {i: scc_difficulty[i] for i in range(len(scc_list))}

    # Map back to nodes
    node_depth = {n: scc_depth.get(scc_index.get(n, 0), 0) for n in G.nodes()}
    node_effort = {n: float(scc_effort.get(scc_index.get(n, 0), 0.0)) for n in G.nodes()}

    return node_depth, node_effort


def sanitize_graph_weights(G: nx.DiGraph, eps: float = 1e-9) -> None:
    """Sanitize edge weights for numerical stability (in-place).

    - Fill missing weights with 1.0
    - Clip invalid weights (<=0) to eps
    - Add inverse_weight (already done in basic metrics, just verify)
    - Remove self-loops if any

    Args:
        G: NetworkX directed graph
        eps: Minimum weight value for stability
    """
    # Remove self-loops
    self_loops = list(nx.selfloop_edges(G))
    if self_loops:
        G.remove_edges_from(self_loops)

    # Sanitize weights
    for u, v, d in G.edges(data=True):
        # Ensure weight exists and is valid
        weight = d.get("weight", 1.0)
        if not isinstance(weight, (int, float)) or weight <= 0:
            weight = eps
        d["weight"] = float(weight)

        # Ensure inverse_weight is computed
        if "inverse_weight" not in d:
            d["inverse_weight"] = min(1.0 / max(weight, eps), 1e9)


def compute_louvain_clustering(
    G: nx.DiGraph, config: Dict[str, Any], logger: Optional[logging.Logger]
) -> Dict[str, int]:
    """Compute Louvain clustering with deterministic numbering.

    Algorithm:
    1. Create undirected projection: UG = G.to_undirected()
    2. Aggregate weights for bidirectional edges (sum)
    3. Run community_louvain.best_partition() with:
       - weight="weight"
       - resolution=config["louvain_resolution"] (default 1.0)
       - random_state=config["louvain_random_state"] (default 42)
    4. Stable renumbering: sort clusters by min node ID in cluster
    5. Return {node_id: cluster_id} where cluster_id ∈ [0, num_clusters-1]

    Edge cases:
    - Empty graph: return {}
    - Single node: cluster_id = 0
    - Disconnected components: each gets separate clusters

    Args:
        G: NetworkX directed graph
        config: Configuration dictionary
        logger: Logger instance (optional)

    Returns:
        Dictionary mapping node_id to cluster_id
    """
    # Check if Louvain is available
    if not LOUVAIN_AVAILABLE:
        if logger:
            logger.warning("python-louvain not installed, skipping clustering")
        return {n: 0 for n in G.nodes()}  # All nodes in one cluster as fallback

    # Handle empty graph
    if G.number_of_nodes() == 0:
        return {}

    # Handle single node
    if G.number_of_nodes() == 1:
        return {list(G.nodes())[0]: 0}

    # Create undirected projection with aggregated weights
    UG = nx.Graph()
    for u, v, d in G.edges(data=True):
        weight = float(d.get("weight", 1.0))
        if UG.has_edge(u, v):
            # Sum weights for bidirectional edges
            UG[u][v]["weight"] += weight
        else:
            UG.add_edge(u, v, weight=weight)

    # Add isolated nodes
    for node in G.nodes():
        if node not in UG:
            UG.add_node(node)

    # Get configuration parameters
    resolution = config.get("louvain_resolution", 1.0)
    random_state = config.get("louvain_random_state", 42)

    if logger:
        logger.info(f"Running Louvain with resolution={resolution}, random_state={random_state}")

    # Run Louvain clustering
    try:
        partition = community_louvain.best_partition(
            UG, weight="weight", resolution=resolution, random_state=random_state
        )
    except Exception as e:
        if logger:
            logger.error(f"Louvain clustering failed: {e}")
        return {n: 0 for n in G.nodes()}

    # Stable renumbering: sort clusters by min node ID
    clusters = {}
    for node, cluster_id in partition.items():
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(node)

    # Sort clusters by their minimum node ID for deterministic numbering
    sorted_clusters = sorted(clusters.items(), key=lambda x: min(x[1]))

    # Create mapping with new sequential cluster IDs
    result = {}
    for new_id, (old_id, nodes) in enumerate(sorted_clusters):
        for node in nodes:
            result[node] = new_id

    if logger:
        num_clusters = len(sorted_clusters)
        logger.info(f"Louvain clustering found {num_clusters} clusters")

    return result


def compute_bridge_scores(
    G: nx.DiGraph,
    cluster_map: Dict[str, int],
    betweenness_centrality: Dict[str, float],
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Compute bridge scores for nodes.

    Formula: bridge_score = w_b * betweenness_norm + (1-w_b) * inter_ratio
    where:
    - w_b = config["bridge_weight_betweenness"] (default 0.7)
    - betweenness_norm = already computed in basic metrics (just use it!)
    - inter_ratio = fraction of neighbors in different clusters

    For directed graphs:
    - neighbors = predecessors ∪ successors (set union, no duplicates)
    - deg_total = in_degree + out_degree

    Edge cases:
    - deg_total = 0 → inter_ratio = 0
    - Single cluster → all inter_ratio = 0
    - No clustering → inter_ratio = 0

    Args:
        G: NetworkX directed graph
        cluster_map: Dictionary mapping node_id to cluster_id
        betweenness_centrality: Pre-computed betweenness centrality values
        config: Configuration dictionary

    Returns:
        Dictionary mapping node_id to bridge_score ∈ [0, 1]
    """
    # Get weight parameter
    w_b = config.get("bridge_weight_betweenness", 0.7)

    bridge_scores = {}

    for node in G.nodes():
        # Get betweenness (already normalized)
        betweenness = betweenness_centrality.get(node, 0.0)

        # Calculate inter-cluster ratio
        if G.is_directed():
            # For directed graph: union of predecessors and successors
            neighbors = set(G.predecessors(node)) | set(G.successors(node))
            deg_total = G.in_degree(node) + G.out_degree(node)
        else:
            neighbors = set(G.neighbors(node))
            deg_total = G.degree(node)

        # Count neighbors in different clusters
        if deg_total > 0 and cluster_map:
            node_cluster = cluster_map.get(node)
            inter_count = sum(
                1 for neighbor in neighbors if cluster_map.get(neighbor) != node_cluster
            )
            inter_ratio = inter_count / len(neighbors)  # Use unique neighbors count
        else:
            inter_ratio = 0.0

        # Compute bridge score
        bridge_score = w_b * float(betweenness) + (1.0 - w_b) * float(inter_ratio)
        bridge_scores[node] = bridge_score

    return bridge_scores


def mark_inter_cluster_edges(G: nx.DiGraph, cluster_map: Dict[str, int]) -> None:
    """Mark edges connecting different clusters (in-place).

    For each edge (u, v):
    - is_inter_cluster_edge = (cluster_map[u] != cluster_map[v])
    - if True, also add:
      - source_cluster_id = cluster_map[u]
      - target_cluster_id = cluster_map[v]

    Edge cases:
    - No clustering → all is_inter_cluster_edge = False
    - Missing nodes in cluster_map → is_inter_cluster_edge = False

    Args:
        G: NetworkX directed graph
        cluster_map: Dictionary mapping node_id to cluster_id
    """
    if not cluster_map:
        # No clustering - mark all edges as intra-cluster
        for u, v, d in G.edges(data=True):
            d["is_inter_cluster_edge"] = False
        return

    for u, v, d in G.edges(data=True):
        source_cluster = cluster_map.get(u)
        target_cluster = cluster_map.get(v)

        if source_cluster is not None and target_cluster is not None:
            is_inter = source_cluster != target_cluster
            d["is_inter_cluster_edge"] = bool(is_inter)

            if is_inter:
                d["source_cluster_id"] = source_cluster
                d["target_cluster_id"] = target_cluster
        else:
            # Missing cluster info - treat as intra-cluster
            d["is_inter_cluster_edge"] = False


def compute_educational_importance(
    G: nx.DiGraph, config: Dict[str, Any], logger: Optional[logging.Logger]
) -> Dict[str, float]:
    """PageRank on educational edges subgraph.

    Args:
        G: NetworkX directed graph
        config: Configuration dictionary
        logger: Logger instance (optional)

    Returns:
        Dictionary mapping node ID to educational importance
    """
    if logger:
        logger.info("Computing educational importance")

    edu_types = set(
        t.upper()
        for t in config.get("graph2metrics", {}).get(
            "educational_edge_types", ["PREREQUISITE", "ELABORATES", "TESTS", "EXAMPLE_OF"]
        )
    )

    # Build educational subgraph
    edu_edges = [
        (u, v, d) for u, v, d in G.edges(data=True) if str(d.get("type", "")).upper() in edu_types
    ]

    E = nx.DiGraph()
    E.add_nodes_from(G.nodes())
    E.add_weighted_edges_from([(u, v, d.get("weight", 1.0)) for u, v, d in edu_edges])

    # Compute PageRank on subgraph
    if E.number_of_edges() > 0:
        damping = config.get("graph2metrics", {}).get("pagerank_damping", 0.85)
        try:
            edu_pr = nx.pagerank(E, alpha=damping, weight="weight")
        except nx.PowerIterationFailedConvergence:
            # Fallback with more iterations
            edu_pr = nx.pagerank(E, alpha=damping, weight="weight", max_iter=200, tol=1e-3)
    else:
        # No educational edges - uniform distribution
        n = E.number_of_nodes()
        edu_pr = {node: 1.0 / n for node in E.nodes()} if n > 0 else {}

    return edu_pr


def validate_metric_invariants(
    pagerank_vals: Dict[str, float],
    edu_importance_vals: Dict[str, float],
    logger: Optional[logging.Logger],
) -> None:
    """Validate that PageRank sums to 1.0.

    Args:
        pagerank_vals: PageRank values
        edu_importance_vals: Educational importance values
        logger: Logger instance (optional)
    """
    if not pagerank_vals or not edu_importance_vals:
        return

    pr_sum = sum(pagerank_vals.values())
    edu_sum = sum(edu_importance_vals.values())

    if logger:
        if abs(pr_sum - 1.0) > 0.01:
            logger.warning(f"PageRank sum = {pr_sum:.6f}, expected 1.0")

        if abs(edu_sum - 1.0) > 0.01:
            logger.warning(f"Educational importance sum = {edu_sum:.6f}, expected 1.0")


def compute_basic_centrality(
    G: nx.DiGraph, config: Dict[str, Any], logger: Optional[logging.Logger]  # noqa: N803
) -> Dict[str, Dict[str, Any]]:
    """Compute basic centrality metrics (degrees and PageRank).

    Args:
        G: NetworkX directed graph
        config: Configuration dictionary
        logger: Logger instance (optional)

    Returns:
        Dictionary with degree and pagerank metrics
    """
    if logger:
        logger.info("Computing basic centrality metrics")

    # Degree metrics
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    degree_centrality = nx.degree_centrality(G)

    # PageRank with weights
    damping = config.get("graph2metrics", {}).get("pagerank_damping", 0.85)
    max_iter = config.get("graph2metrics", {}).get("pagerank_max_iter", 100)

    if G.number_of_edges() > 0:
        try:
            pagerank = nx.pagerank(G, alpha=damping, max_iter=max_iter, weight="weight")
        except nx.PowerIterationFailedConvergence:
            if logger:
                logger.warning("PageRank didn't converge, using partial results")
            pagerank = nx.pagerank(G, alpha=damping, max_iter=max_iter, weight="weight", tol=1e-3)
    else:
        # Empty graph - uniform distribution
        n = G.number_of_nodes()
        pagerank = {node: 1.0 / n for node in G.nodes()} if n > 0 else {}

    return {
        "degree_in": in_degrees,
        "degree_out": out_degrees,
        "degree_centrality": degree_centrality,
        "pagerank": pagerank,
    }


def compute_all_metrics(
    G: nx.DiGraph,  # noqa: N803
    graph_data: Dict[str, Any],
    config: Dict[str, Any],
    logger: Optional[logging.Logger],
) -> Dict[str, Any]:
    """Compute all 12 metrics in correct order.

    Args:
        G: NetworkX directed graph
        graph_data: Original graph data
        config: Configuration dictionary
        logger: Logger instance (optional)

    Returns:
        Enhanced graph data with all metrics
    """
    if logger:
        logger.info("Computing all graph metrics")

    # 1. Edge metrics FIRST (required for distance metrics)
    compute_edge_weights(G, logger)

    # Also add inverse_weight to edges in graph_data
    for edge in graph_data.get("edges", []):
        source = edge["source"]
        target = edge["target"]
        if G.has_edge(source, target):
            edge["inverse_weight"] = G[source][target]["inverse_weight"]

    # 2. Basic centrality (degrees and PageRank)
    basic_metrics = compute_basic_centrality(G, config, logger)

    # 3. Distance-based metrics (requires inverse_weight)
    distance_metrics = compute_distance_centrality(G, logger)

    # 4. Component IDs (deterministic based on node order)
    node_order = [n["id"] for n in graph_data["nodes"]]
    component_ids = compute_component_ids(G, node_order, logger)

    # 5. Prerequisite metrics
    prereq_depth, learning_effort = compute_prerequisite_metrics(G, graph_data, config, logger)

    # 6. Educational importance
    edu_importance = compute_educational_importance(G, config, logger)

    # 7. Validate invariants
    validate_metric_invariants(basic_metrics["pagerank"], edu_importance, logger)

    # 8. Advanced metrics (clustering and bridges)
    if logger:
        logger.info("Computing advanced metrics...")

    # Sanitize graph weights first
    sanitize_graph_weights(G)

    # 8.1 Louvain clustering
    graph2metrics_config = config.get("graph2metrics", {})
    cluster_map = compute_louvain_clustering(G, graph2metrics_config, logger)

    # 8.2 Bridge scores (uses betweenness from basic metrics)
    betweenness_centrality = distance_metrics["betweenness"]
    bridge_scores = compute_bridge_scores(
        G, cluster_map, betweenness_centrality, graph2metrics_config
    )

    # 8.3 Mark inter-cluster edges
    mark_inter_cluster_edges(G, cluster_map)

    # Log clustering statistics
    if logger and cluster_map:
        num_clusters = len(set(cluster_map.values()))
        logger.info(f"Found {num_clusters} clusters")
        high_bridges = sum(1 for s in bridge_scores.values() if s > 0.1)
        logger.info(f"Found {high_bridges} nodes with bridge_score > 0.1")

    # 9. Add all metrics to nodes
    for node in graph_data["nodes"]:
        node_id = node["id"]

        if node_id in G.nodes():
            # Degree metrics (integers)
            node["degree_in"] = basic_metrics["degree_in"].get(node_id, 0)
            node["degree_out"] = basic_metrics["degree_out"].get(node_id, 0)

            # Centrality metrics (floats, safe from NaN/inf)
            node["degree_centrality"] = safe_metric_value(
                basic_metrics["degree_centrality"].get(node_id, 0.0)
            )
            node["pagerank"] = safe_metric_value(basic_metrics["pagerank"].get(node_id, 0.0))
            node["betweenness_centrality"] = safe_metric_value(
                distance_metrics["betweenness"].get(node_id, 0.0)
            )
            node["out-closeness"] = safe_metric_value(
                distance_metrics["out_closeness"].get(node_id, 0.0)
            )

            # Structure metrics
            node["component_id"] = component_ids.get(node_id, 0)

            # Educational metrics
            node["prerequisite_depth"] = prereq_depth.get(node_id, 0)
            node["learning_effort"] = safe_metric_value(learning_effort.get(node_id, 0.0))
            node["educational_importance"] = safe_metric_value(edu_importance.get(node_id, 0.0))

            # Advanced metrics (clustering and bridges)
            node["cluster_id"] = cluster_map.get(node_id, 0)
            node["bridge_score"] = safe_metric_value(bridge_scores.get(node_id, 0.0))

    # 10. Transfer inter-cluster edge attributes to output
    for edge in graph_data.get("edges", []):
        u, v = edge["source"], edge["target"]
        if G.has_edge(u, v):
            edge_data = G[u][v]
            edge["is_inter_cluster_edge"] = edge_data.get("is_inter_cluster_edge", False)
            if edge["is_inter_cluster_edge"]:
                edge["source_cluster_id"] = edge_data.get("source_cluster_id")
                edge["target_cluster_id"] = edge_data.get("target_cluster_id")

    if logger:
        # Log statistics
        pr_values = [n.get("pagerank", 0) for n in graph_data["nodes"]]
        if pr_values:
            logger.info(f"PageRank range: [{min(pr_values):.6f}, {max(pr_values):.6f}]")
            logger.info(f"PageRank sum: {sum(pr_values):.6f}")

        edu_values = [n.get("educational_importance", 0) for n in graph_data["nodes"]]
        if edu_values:
            logger.info(f"Educational importance sum: {sum(edu_values):.6f}")

    # 11. Data enrichment for visualization
    if logger:
        logger.info("Adding visualization enrichments...")

    # Generate demo path
    graph_data = generate_demo_path(G, graph_data, config, logger)

    # Generate course sequence
    graph_data = generate_course_sequence(graph_data, logger)

    # Link nodes to concepts
    graph_data = link_nodes_to_concepts(graph_data)

    # Note: handle_large_graph will be called later in main(),
    # as it may change the number of nodes

    return graph_data


def compute_centrality_metrics(
    G: nx.DiGraph,  # noqa: N803
    graph_data: Dict[str, Any],
    config: Dict[str, Any],
    logger: logging.Logger,
    test_mode: bool = False,
) -> Dict[str, Any]:
    """Compute centrality metrics for nodes - wrapper for compute_all_metrics.

    Args:
        G: NetworkX directed graph
        graph_data: Original graph data
        config: Configuration dictionary
        logger: Logger instance
        test_mode: Whether running in test mode

    Returns:
        Enhanced graph data with metrics
    """
    mode_prefix = "[TEST MODE] " if test_mode else ""

    # Progress output to console
    num_nodes = G.number_of_nodes()
    if test_mode:
        print(f"[TEST MODE] Computing metrics for {num_nodes} nodes...")
    else:
        print(f"Computing metrics for {num_nodes} nodes...")

    # Log mode
    if logger:
        logger.info(f"{mode_prefix}Computing all metrics using new implementation")

    # Call the new comprehensive metrics computation
    result = compute_all_metrics(G, graph_data, config, logger)

    print("  ✓ All metrics computed successfully")

    return result


def create_mention_index(
    graph_data: Dict[str, Any],
    concepts_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Create index of concept mentions in nodes.

    Args:
        graph_data: Graph data with nodes and edges
        concepts_data: Concepts dictionary

    Returns:
        Enhanced concepts_data with mention index in _meta
    """
    mention_index = {}

    # Analyze MENTIONS edges (source=node, target=concept)
    for edge in graph_data.get("edges", []):
        if edge.get("type") == "MENTIONS":
            concept_id = edge["target"]
            node_id = edge["source"]

            if concept_id not in mention_index:
                mention_index[concept_id] = {"nodes": [], "count": 0}

            mention_index[concept_id]["nodes"].append(node_id)
            mention_index[concept_id]["count"] += 1

    # Add to concepts_data _meta
    if "_meta" not in concepts_data:
        concepts_data["_meta"] = {}

    concepts_data["_meta"]["mention_index"] = mention_index

    return concepts_data


def link_nodes_to_concepts(graph_data: Dict[str, Any]) -> Dict[str, Any]:
    """Fill concepts field in each node based on MENTIONS edges.

    Args:
        graph_data: Graph data with nodes and edges

    Returns:
        Modified graph_data with concepts field in nodes
    """
    # Build node_id -> [concept_ids] mapping from MENTIONS edges
    node_concepts = {}

    for edge in graph_data.get("edges", []):
        if edge.get("type") == "MENTIONS":
            node_id = edge["source"]
            concept_id = edge["target"]

            if node_id not in node_concepts:
                node_concepts[node_id] = []

            node_concepts[node_id].append(concept_id)

    # Add "concepts" field to each node
    for node in graph_data.get("nodes", []):
        node_id = node["id"]
        node["concepts"] = node_concepts.get(node_id, [])

    return graph_data


def handle_large_graph(
    graph_data: Dict[str, Any],
    max_nodes: int = 1000,
    save_full_path: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Filter top-N nodes by PageRank for graphs > max_nodes.

    Args:
        graph_data: Graph data with nodes and edges
        max_nodes: Maximum number of nodes to keep
        save_full_path: Optional path to save full graph before filtering
        logger: Logger instance

    Returns:
        Filtered graph_data if needed, original otherwise
    """
    current_nodes = len(graph_data.get("nodes", []))

    if current_nodes <= max_nodes:
        return graph_data  # No filtering needed

    if logger:
        logger.warning(f"Graph has {current_nodes} nodes, filtering to top-{max_nodes} by PageRank")

    # Save full graph if path provided
    if save_full_path:
        save_full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_full_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        if logger:
            logger.info(f"Full graph saved to {save_full_path}")

    # Sort nodes by PageRank descending
    nodes = graph_data.get("nodes", [])
    nodes.sort(key=lambda n: n.get("pagerank", 0.0), reverse=True)

    # Keep top max_nodes
    kept_nodes = nodes[:max_nodes]
    kept_node_ids = {n["id"] for n in kept_nodes}

    # Filter edges to keep only those between kept nodes
    edges = graph_data.get("edges", [])
    kept_edges = [e for e in edges if e["source"] in kept_node_ids and e["target"] in kept_node_ids]

    # Update graph_data
    graph_data["nodes"] = kept_nodes
    graph_data["edges"] = kept_edges

    # Add metadata about filtering
    if "_meta" not in graph_data:
        graph_data["_meta"] = {}

    if "graph_metadata" not in graph_data["_meta"]:
        graph_data["_meta"]["graph_metadata"] = {}

    graph_data["_meta"]["graph_metadata"]["filtered"] = True
    graph_data["_meta"]["graph_metadata"]["original_nodes"] = current_nodes
    graph_data["_meta"]["graph_metadata"]["original_edges"] = len(edges)
    graph_data["_meta"]["graph_metadata"]["filtered_nodes"] = len(kept_nodes)
    graph_data["_meta"]["graph_metadata"]["filtered_edges"] = len(kept_edges)
    graph_data["_meta"]["graph_metadata"]["filter_method"] = "top_pagerank"
    graph_data["_meta"]["graph_metadata"]["filter_threshold"] = max_nodes

    if logger:
        logger.info(
            f"Filtered graph: {len(kept_nodes)} nodes, {len(kept_edges)} edges "
            f"(was: {current_nodes} nodes, {len(edges)} edges)"
        )

    return graph_data


# Helper functions for path generation
def _build_educational_subgraph(
    G: nx.DiGraph,  # noqa: N803
    edge_types: Optional[List[str]] = None,
) -> nx.DiGraph:
    """Extract subgraph with only educational edge types."""
    if edge_types is None:
        edge_types = ["PREREQUISITE", "ELABORATES", "EXAMPLE_OF", "TESTS"]

    H = nx.DiGraph()  # noqa: N806
    H.add_nodes_from(G.nodes(data=True))

    for u, v, data in G.edges(data=True):
        if data.get("type") in edge_types:
            H.add_edge(u, v, **data)

    return H


def _add_high_value_nodes(
    current_path: List[str],
    nodes_dict: Dict[str, Dict[str, Any]],
    target_count: int,
    metric: str = "pagerank",
) -> List[str]:
    """Add top nodes by specified metric until target count."""
    if len(current_path) >= target_count:
        return current_path

    # Get nodes not in path, sorted by metric
    available_nodes = [
        (node_id, node.get(metric, 0.0))
        for node_id, node in nodes_dict.items()
        if node_id not in current_path
    ]
    available_nodes.sort(key=lambda x: x[1], reverse=True)

    # Add top nodes
    nodes_to_add = target_count - len(current_path)
    for node_id, _ in available_nodes[:nodes_to_add]:
        current_path.append(node_id)

    return current_path


def _ensure_path_connectivity(
    G: nx.DiGraph,  # noqa: N803
    path_nodes: List[str],
    nodes_dict: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Try to connect disconnected nodes in path."""
    if len(path_nodes) <= 1:
        return path_nodes

    connected_path = [path_nodes[0]]

    for i in range(1, len(path_nodes)):
        target = path_nodes[i]
        if target not in connected_path:
            # Try to find a short path from last connected node
            try:
                if G.has_node(connected_path[-1]) and G.has_node(target):
                    intermediate = nx.shortest_path(G, connected_path[-1], target)
                    # Add intermediate nodes (excluding start since it's already in path)
                    for node in intermediate[1:]:
                        if node not in connected_path:
                            connected_path.append(node)
                else:
                    # If can't connect, just add it anyway
                    connected_path.append(target)
            except nx.NetworkXNoPath:
                # Can't connect, add anyway
                connected_path.append(target)

    return connected_path


def _generate_optimal_path(
    G: nx.DiGraph,  # noqa: N803
    nodes_dict: Dict[str, Dict[str, Any]],
    max_nodes: int,
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """Generate optimal educational journey through most important concepts.

    Strategy 1: Build educational journey using educational_importance.

    Args:
        G: NetworkX directed graph
        nodes_dict: Dictionary mapping node_id to node data with metrics
        max_nodes: Maximum number of nodes in path
        config: Optional configuration dict
        logger: Optional logger for debugging

    Returns:
        List of node IDs forming the path
    """
    if not nodes_dict:
        return []

    if logger:
        logger.debug(f"Strategy 1: Starting with {len(nodes_dict)} nodes, max_nodes={max_nodes}")

    # Build educational subgraph
    edu_graph = _build_educational_subgraph(G)

    # 1. Find entry point: min prerequisite_depth with high educational_importance
    min_depth = min((n.get("prerequisite_depth", 0) for n in nodes_dict.values()), default=0)
    entry_candidates = [
        (node_id, node.get("educational_importance", 0.0))
        for node_id, node in nodes_dict.items()
        if node.get("prerequisite_depth", 0) == min_depth
    ]

    if not entry_candidates:
        # Fallback: just use node with highest educational_importance
        entry_candidates = [
            (node_id, node.get("educational_importance", 0.0))
            for node_id, node in nodes_dict.items()
        ]

    entry_candidates.sort(key=lambda x: x[1], reverse=True)
    start_node = entry_candidates[0][0] if entry_candidates else list(nodes_dict.keys())[0]

    # 2. Select top nodes by educational_importance
    edu_nodes = sorted(
        nodes_dict.items(), key=lambda x: x[1].get("educational_importance", 0.0), reverse=True
    )

    # Take more nodes than needed to have options
    top_edu_nodes = [node_id for node_id, _ in edu_nodes[: int(max_nodes * 1.5)]]

    # 3. Build initial path through educational nodes
    path = [start_node]
    visited = {start_node}

    # Try to connect educational nodes
    for target in top_edu_nodes:
        if target not in visited and len(path) < max_nodes:
            # Try to find path in educational subgraph
            try:
                if edu_graph.has_node(path[-1]) and edu_graph.has_node(target):
                    connecting_path = nx.shortest_path(edu_graph, path[-1], target)
                    for node in connecting_path[1:]:
                        if node not in visited and len(path) < max_nodes:
                            path.append(node)
                            visited.add(node)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # Can't connect via educational edges, try general graph
                try:
                    if G.has_node(path[-1]) and G.has_node(target):
                        connecting_path = nx.shortest_path(G, path[-1], target)
                        # Add only if path is reasonable
                        if len(connecting_path) <= 5:
                            for node in connecting_path[1:]:
                                if node not in visited and len(path) < max_nodes:
                                    path.append(node)
                                    visited.add(node)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass

    if logger:
        logger.debug(f"Strategy 1: After educational connections: {len(path)} nodes")

    # 4. If path too short, add high PageRank nodes
    min_nodes = min(15, max_nodes)  # Aim for at least 15 nodes
    target_nodes = min(max_nodes // 2, 50)  # Try to reach half of max or 50

    if len(path) < min_nodes:
        path = _add_high_value_nodes(path, nodes_dict, target_nodes, "pagerank")
        if logger:
            logger.debug(f"Strategy 1: After adding PageRank nodes: {len(path)} nodes")

    # 5. If still short, add bridge nodes
    if len(path) < target_nodes:
        path = _add_high_value_nodes(path, nodes_dict, target_nodes, "bridge_score")
        if logger:
            logger.debug(f"Strategy 1: After adding bridge nodes: {len(path)} nodes")

    # 6. If still short, add by betweenness centrality
    if len(path) < target_nodes:
        path = _add_high_value_nodes(path, nodes_dict, target_nodes, "betweenness_centrality")
        if logger:
            logger.debug(f"Strategy 1: After adding betweenness nodes: {len(path)} nodes")

    # 7. Fill remaining with educational_importance if still have room
    if len(path) < max_nodes:
        path = _add_high_value_nodes(path, nodes_dict, max_nodes, "educational_importance")
        if logger:
            logger.debug(
                f"Strategy 1: After adding educational_importance nodes: {len(path)} nodes"
            )

    # 8. Sort by prerequisite_depth for logical order
    path.sort(key=lambda x: nodes_dict[x].get("prerequisite_depth", 0))

    if logger:
        logger.debug(f"Strategy 1: Final path length: {len(path)}")

    return path[:max_nodes]


def _generate_showcase_path(
    G: nx.DiGraph,  # noqa: N803
    nodes_dict: Dict[str, Dict[str, Any]],
    max_nodes: int,
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """Generate showcase path with one node from each cluster.

    Strategy 2: Select top PageRank node from each cluster.

    Args:
        G: NetworkX directed graph
        nodes_dict: Dictionary mapping node_id to node data with metrics
        max_nodes: Maximum number of nodes in path

    Returns:
        List of node IDs forming the path
    """
    if not nodes_dict:
        return []

    # Group nodes by cluster_id
    clusters = {}
    for node_id, node in nodes_dict.items():
        cluster_id = node.get("cluster_id", 0)
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(node_id)

    if not clusters:
        return []

    # If no clustering or single cluster, fallback to Strategy 1
    if len(clusters) == 1:
        return _generate_optimal_path(G, nodes_dict, max_nodes, config, logger)

    # Select top PageRank node from each cluster
    showcase_nodes = []
    for cluster_id, node_ids in clusters.items():
        if node_ids:
            # Find best node in cluster by PageRank
            best_node = max(node_ids, key=lambda x: nodes_dict[x].get("pagerank", 0.0))
            showcase_nodes.append((best_node, nodes_dict[best_node].get("prerequisite_depth", 0)))

    # Sort by prerequisite_depth to create logical progression
    showcase_nodes.sort(key=lambda x: x[1])

    # If more clusters than max_nodes, prioritize by cluster size
    if len(showcase_nodes) > max_nodes:
        # Calculate cluster sizes
        cluster_sizes = {cid: len(nodes) for cid, nodes in clusters.items()}

        # Re-select based on largest clusters
        showcase_nodes = []
        for cluster_id, _ in sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)[
            :max_nodes
        ]:
            node_ids = clusters[cluster_id]
            best_node = max(node_ids, key=lambda x: nodes_dict[x].get("pagerank", 0.0))
            showcase_nodes.append((best_node, nodes_dict[best_node].get("prerequisite_depth", 0)))

        # Re-sort by depth
        showcase_nodes.sort(key=lambda x: x[1])

    return [node_id for node_id, _ in showcase_nodes[:max_nodes]]


def _generate_critical_path(
    G: nx.DiGraph,  # noqa: N803
    nodes_dict: Dict[str, Dict[str, Any]],
    max_nodes: int,
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """Generate critical path from most complex node to fundamentals.

    Strategy 3: Trace dependencies from complex to simple, ensuring coverage.

    Args:
        G: NetworkX directed graph
        nodes_dict: Dictionary mapping node_id to node data with metrics
        max_nodes: Maximum number of nodes in path
        config: Optional configuration dict
        logger: Optional logger for debugging

    Returns:
        List of node IDs forming the path (simple to complex)
    """
    if not nodes_dict:
        return []

    if logger:
        logger.debug(f"Strategy 3: Starting with {len(nodes_dict)} nodes, max_nodes={max_nodes}")

    # Find node with max learning_effort
    max_effort = max((n.get("learning_effort", 0.0) for n in nodes_dict.values()), default=0.0)

    if max_effort == 0:
        # All efforts are 0, fallback to using educational_importance
        complex_node = max(
            nodes_dict.keys(), key=lambda x: nodes_dict[x].get("educational_importance", 0.0)
        )
    else:
        # Get nodes with high effort
        complex_candidates = [
            (node_id, node.get("educational_importance", 0.0))
            for node_id, node in nodes_dict.items()
            if node.get("learning_effort", 0.0) >= max_effort * 0.9  # Top 10% effort
        ]
        # Among them, select one with max educational_importance
        complex_candidates.sort(key=lambda x: x[1], reverse=True)
        complex_node = (
            complex_candidates[0][0] if complex_candidates else list(nodes_dict.keys())[0]
        )

    # Build dependency graph with multiple edge types
    dep_graph = _build_educational_subgraph(G, ["PREREQUISITE", "ELABORATES", "EXAMPLE_OF"])

    # Find all ancestors using BFS on reversed dependency graph
    ancestors = set()
    to_visit = [complex_node]
    visited = {complex_node}
    depth_map = {complex_node: 0}

    while to_visit and len(ancestors) < max_nodes * 2:  # Get extra for selection
        current = to_visit.pop(0)
        current_depth = depth_map[current]

        # Look at predecessors in dependency graph
        if dep_graph.has_node(current):
            for pred in dep_graph.predecessors(current):
                if pred not in visited:
                    ancestors.add(pred)
                    visited.add(pred)
                    depth_map[pred] = current_depth + 1
                    to_visit.append(pred)

    if logger:
        logger.debug(f"Strategy 3: Found {len(ancestors)} ancestors")

    # Group ancestors by prerequisite_depth
    depth_groups = {}
    for node in ancestors:
        depth = nodes_dict[node].get("prerequisite_depth", 0)
        if depth not in depth_groups:
            depth_groups[depth] = []
        depth_groups[depth].append(node)

    # Select representative nodes from each depth
    path = [complex_node]
    for depth in sorted(depth_groups.keys()):
        # Sort by educational_importance and take top ones
        candidates = sorted(
            depth_groups[depth],
            key=lambda x: nodes_dict[x].get("educational_importance", 0.0),
            reverse=True,
        )
        # Add best from this depth level
        for node in candidates:
            if node not in path and len(path) < max_nodes:
                path.append(node)
                if len(path) >= max_nodes:
                    break

    if logger:
        logger.debug(f"Strategy 3: After depth selection: {len(path)} nodes")

    # If path too short, add nodes with high betweenness (bridges)
    min_nodes = min(15, max_nodes)  # Aim for at least 15 nodes
    target_nodes = min(max_nodes // 2, 50)  # Try to reach half of max or 50

    if len(path) < min_nodes:
        bridge_nodes = sorted(
            [
                (n, nodes_dict[n].get("betweenness_centrality", 0.0))
                for n in nodes_dict
                if n not in path
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        for node, _ in bridge_nodes:
            if len(path) >= target_nodes:
                break
            path.append(node)

        if logger:
            logger.debug(f"Strategy 3: After adding bridges: {len(path)} nodes")

    # If still short, add high PageRank nodes
    if len(path) < target_nodes:
        path = _add_high_value_nodes(path, nodes_dict, target_nodes, "pagerank")
        if logger:
            logger.debug(f"Strategy 3: After adding PageRank: {len(path)} nodes")

    # If still short, add by educational_importance
    if len(path) < target_nodes:
        path = _add_high_value_nodes(path, nodes_dict, target_nodes, "educational_importance")
        if logger:
            logger.debug(f"Strategy 3: After adding educational_importance: {len(path)} nodes")

    # Fill remaining with learning_effort if still have room
    if len(path) < max_nodes:
        path = _add_high_value_nodes(path, nodes_dict, max_nodes, "learning_effort")
        if logger:
            logger.debug(f"Strategy 3: After adding learning_effort nodes: {len(path)} nodes")

    # Sort by prerequisite_depth (ascending) for learning order
    path.sort(key=lambda x: nodes_dict[x].get("prerequisite_depth", 0))

    if logger:
        logger.debug(f"Strategy 3: Final path length: {len(path)}")

    return path[:max_nodes]


def generate_demo_path(
    G: nx.DiGraph,  # noqa: N803
    graph_data: Dict[str, Any],
    config: Dict[str, Any],
    logger: logging.Logger,
    test_mode: bool = False,
) -> Dict[str, Any]:
    """Generate demo path for tour mode.

    Args:
        G: NetworkX directed graph
        graph_data: Graph data with metrics
        config: Configuration dictionary
        logger: Logger instance
        test_mode: Whether running in test mode

    Returns:
        Graph data with demo path in _meta
    """
    mode_prefix = "[TEST MODE] " if test_mode else ""

    # Get configuration
    demo_config = config.get("demo_path", {})
    strategy = demo_config.get("strategy", 1)
    max_nodes = demo_config.get("max_nodes", 15)

    if logger:
        logger.info(f"{mode_prefix}Generating demo path (strategy={strategy}, max={max_nodes})")

    # Build nodes dictionary for easier access
    nodes_dict = {node["id"]: node for node in graph_data.get("nodes", [])}

    # Select strategy
    if strategy == 1:
        path = _generate_optimal_path(G, nodes_dict, max_nodes, config, logger)
        strategy_name = "optimal"
    elif strategy == 2:
        path = _generate_showcase_path(G, nodes_dict, max_nodes, config, logger)
        strategy_name = "showcase"
    elif strategy == 3:
        path = _generate_critical_path(G, nodes_dict, max_nodes, config, logger)
        strategy_name = "critical"
    else:
        if logger:
            logger.warning(f"Unknown demo path strategy: {strategy}, using default (1)")
        path = _generate_optimal_path(G, nodes_dict, max_nodes, config, logger)
        strategy_name = "optimal"

    # Add to _meta
    if "_meta" not in graph_data:
        graph_data["_meta"] = {}

    graph_data["_meta"]["demo_path"] = path
    graph_data["_meta"]["demo_generation_config"] = {
        "strategy": strategy,
        "strategy_name": strategy_name,
        "max_nodes": max_nodes,
        "actual_nodes": len(path),
    }

    if logger:
        logger.info(f"{mode_prefix}Generated {strategy_name} demo path with {len(path)} nodes")
        if path and len(path) <= 10:
            logger.debug(f"Demo path: {' → '.join(path)}")

    return graph_data


def generate_course_sequence(
    graph_data: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Generate course sequence from Chunk nodes ordered by position.

    Args:
        graph_data: Graph data with nodes
        logger: Optional logger instance

    Returns:
        Graph data with course_sequence in _meta
    """
    if logger:
        logger.info("Generating course sequence from Chunk nodes")

    # Find all Chunk nodes
    chunk_nodes = [node for node in graph_data.get("nodes", []) if node.get("type") == "Chunk"]

    if logger:
        logger.debug(f"Found {len(chunk_nodes)} Chunk nodes")

    # Extract position and build sequence
    course_sequence = []
    for node in chunk_nodes:
        node_id = node.get("id", "")
        # ID format: {slug}:c:{position}
        if ":c:" in node_id:
            try:
                position = int(node_id.split(":c:")[1])
                course_sequence.append(
                    {
                        "id": node_id,
                        "cluster_id": node.get("cluster_id", 0),
                        "position": position,
                    }
                )
            except (ValueError, IndexError) as e:
                if logger:
                    logger.warning(f"Could not extract position from Chunk ID '{node_id}': {e}")

    # Sort by position
    course_sequence.sort(key=lambda x: x["position"])

    # Add to _meta
    if "_meta" not in graph_data:
        graph_data["_meta"] = {}

    graph_data["_meta"]["course_sequence"] = course_sequence

    if logger:
        logger.info(f"Generated course sequence with {len(course_sequence)} items")

    return graph_data


def save_output_data(
    output_dir: Path,
    graph_data: Dict[str, Any],
    concepts_data: Dict[str, Any],
    logger: logging.Logger,
    test_mode: bool = False,
) -> None:
    """Save enriched data to output files.

    Args:
        output_dir: Output directory
        graph_data: Enriched graph data
        concepts_data: Enriched concepts data
        logger: Logger instance
        test_mode: Whether running in test mode
    """
    mode_prefix = "[TEST MODE] " if test_mode else ""

    # Enrich concepts with mention index
    concepts_data = create_mention_index(graph_data, concepts_data)

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output paths (always same names, regardless of mode)
    graph_output = output_dir / "LearningChunkGraph_wow.json"
    concepts_output = output_dir / "ConceptDictionary_wow.json"

    logger.info(f"{mode_prefix}Saving output files to {output_dir}")

    # Save graph
    logger.info(f"{mode_prefix}Saving graph: {graph_output}")
    with open(graph_output, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)

    # Save concepts
    logger.info(f"{mode_prefix}Saving concepts: {concepts_output}")
    with open(concepts_output, "w", encoding="utf-8") as f:
        json.dump(concepts_data, f, ensure_ascii=False, indent=2)

    logger.info(f"{mode_prefix}Output files saved successfully")


def scan_test_graphs(
    test_dir: Path, graph_filter: Optional[str] = None
) -> List[Tuple[str, Path, Path]]:
    """Scan test directory for test graph pairs.

    Args:
        test_dir: Directory containing test files
        graph_filter: Optional filter for specific graph name

    Returns:
        List of (name, input_path, expected_path) tuples
    """
    test_pairs = []

    # Find all test graph files (looking for test_*_graph.json)
    for input_file in sorted(test_dir.glob("test_*_graph.json")):
        # Extract base name (test_bridge_graph -> test_bridge)
        stem = input_file.stem  # test_bridge_graph
        name = stem[:-6]  # test_bridge

        # Skip if filter specified and doesn't match
        if graph_filter and graph_filter != name:
            continue

        # Look for corresponding expected file (test_bridge_graph_expected.json)
        expected_file = test_dir / f"{stem}_expected.json"

        if expected_file.exists():
            test_pairs.append((name, input_file, expected_file))

    return test_pairs


def validate_test_files(
    input_data: Dict[str, Any], expected_data: Dict[str, Any], logger: logging.Logger
) -> bool:
    """Validate test files against JSON schemas.

    Args:
        input_data: Input graph data
        expected_data: Expected graph data
        logger: Logger instance

    Returns:
        True if validation passes
    """
    try:
        # Validate input
        validate_json(input_data, "LearningChunkGraph")
        validate_graph_invariants(input_data)

        # Validate expected
        validate_json(expected_data, "LearningChunkGraph")
        validate_graph_invariants(expected_data)

        return True
    except (ValidationError, GraphInvariantError) as e:
        logger.error(f"Schema validation failed: {e}")
        return False


def compare_metric_value(expected: Any, actual: Any, tolerance: float = 0.01) -> Tuple[str, float]:
    """Compare metric values with tolerance.

    Args:
        expected: Expected value
        actual: Actual value
        tolerance: Relative tolerance (default 1%)

    Returns:
        Tuple of (status, deviation_percent)
        Status: "PASS", "FAIL", "MISS", "NaN"
    """
    # Check if metric is missing
    if actual is None:
        return "MISS", 0.0

    # Check for NaN or inf
    if isinstance(actual, float) and (math.isnan(actual) or math.isinf(actual)):
        return "NaN", 0.0

    # Handle expected = 0 case with absolute tolerance
    if expected == 0.0:
        abs_diff = abs(actual - expected)
        if abs_diff <= 0.001:  # Absolute tolerance for zero
            return "PASS", 0.0
        else:
            # Return large percentage for display
            return "FAIL", 999.9 if actual != 0 else 0.0

    # Calculate relative deviation
    deviation = abs((actual - expected) / expected)
    deviation_percent = deviation * 100

    if deviation <= tolerance:
        return "PASS", deviation_percent
    else:
        return "FAIL", deviation_percent


def format_validation_result(status: str, deviation: float) -> str:
    """Format validation result with emoji.

    Args:
        status: Validation status
        deviation: Deviation percentage

    Returns:
        Formatted string with emoji
    """
    if status == "PASS":
        return "✅"
    elif status == "FAIL":
        return f"❌ {deviation:.1f}%"
    elif status == "MISS":
        return "❌ MISS"
    elif status == "NaN":
        return "⚠️ NaN"
    else:
        return "?"


def validate_graph_metrics(
    graph_name: str,
    input_data: Dict[str, Any],
    expected_data: Dict[str, Any],
    config: Dict[str, Any],
    logger: logging.Logger,
    metric_filter: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Validate metrics for a single graph.

    Args:
        graph_name: Name of the test graph
        input_data: Input graph data
        expected_data: Expected graph data with metrics
        config: Configuration dictionary
        logger: Logger instance
        metric_filter: Optional filter for specific metric
        verbose: Whether to show detailed output

    Returns:
        Validation results dictionary
    """
    # Convert to NetworkX and compute metrics
    G = convert_to_networkx(input_data, logger, test_mode=False)  # noqa: N806
    computed_data = compute_centrality_metrics(
        G, input_data.copy(), config, logger, test_mode=False
    )

    # Define metrics to check
    node_metrics = [
        "degree_in",
        "degree_out",
        "degree_centrality",
        "pagerank",
        "betweenness_centrality",
        "out-closeness",
        "component_id",
        "prerequisite_depth",
        "learning_effort",
        "educational_importance",
    ]
    edge_metrics = ["inverse_weight"]

    results = {"graph": graph_name, "metrics": {}, "passed": 0, "failed": 0, "missing": 0}

    # Check node metrics
    for metric in node_metrics:
        if metric_filter and metric_filter != metric:
            continue

        metric_results = []

        # Compare each node
        for expected_node in expected_data["nodes"]:
            node_id = expected_node["id"]

            # Find corresponding computed node
            computed_node = next((n for n in computed_data["nodes"] if n["id"] == node_id), None)

            if computed_node:
                # Handle both naming conventions for out-closeness
                if metric == "out-closeness":
                    # Check for both "out-closeness" and "closeness_centrality" in expected
                    expected_val = expected_node.get("out-closeness") or expected_node.get(
                        "closeness_centrality"
                    )
                else:
                    expected_val = expected_node.get(metric)
                actual_val = computed_node.get(metric)

                if expected_val is not None:
                    status, deviation = compare_metric_value(expected_val, actual_val)
                    metric_results.append(
                        {
                            "node": node_id,
                            "expected": expected_val,
                            "actual": actual_val,
                            "status": status,
                            "deviation": deviation,
                        }
                    )

                    if verbose:
                        result_str = format_validation_result(status, deviation)
                        print(
                            f"  {graph_name}.{node_id}.{metric}: "
                            f"expected={expected_val}, actual={actual_val} {result_str}"
                        )

        # Aggregate results for this metric
        if metric_results:
            passed = sum(1 for r in metric_results if r["status"] == "PASS")
            failed = sum(1 for r in metric_results if r["status"] == "FAIL")
            missing = sum(1 for r in metric_results if r["status"] == "MISS")

            results["metrics"][metric] = {
                "passed": passed,
                "failed": failed,
                "missing": missing,
                "failures": [r for r in metric_results if r["status"] != "PASS"],
            }

            results["passed"] += passed
            results["failed"] += failed
            results["missing"] += missing

    # Check edge metrics
    for metric in edge_metrics:
        if metric_filter and metric_filter != metric:
            continue

        metric_results = []

        # Compare each edge
        for expected_edge in expected_data.get("edges", []):
            source = expected_edge["source"]
            target = expected_edge["target"]

            # Find corresponding computed edge
            computed_edge = next(
                (
                    e
                    for e in computed_data.get("edges", [])
                    if e["source"] == source and e["target"] == target
                ),
                None,
            )

            if computed_edge:
                expected_val = expected_edge.get(metric)
                actual_val = computed_edge.get(metric)

                if expected_val is not None:
                    status, deviation = compare_metric_value(expected_val, actual_val)
                    metric_results.append(
                        {
                            "edge": f"{source}->{target}",
                            "expected": expected_val,
                            "actual": actual_val,
                            "status": status,
                            "deviation": deviation,
                        }
                    )

                    if verbose:
                        result_str = format_validation_result(status, deviation)
                        print(
                            f"  {graph_name}.edge({source}->{target}).{metric}: "
                            f"expected={expected_val}, actual={actual_val} {result_str}"
                        )

        # Aggregate results for this metric
        if metric_results:
            passed = sum(1 for r in metric_results if r["status"] == "PASS")
            failed = sum(1 for r in metric_results if r["status"] == "FAIL")
            missing = sum(1 for r in metric_results if r["status"] == "MISS")

            results["metrics"][metric] = {
                "passed": passed,
                "failed": failed,
                "missing": missing,
                "failures": [r for r in metric_results if r["status"] != "PASS"],
            }

            results["passed"] += passed
            results["failed"] += failed
            results["missing"] += missing

    return results


def format_validation_matrix(all_results: List[Dict[str, Any]]) -> None:
    """Print formatted validation matrix.

    Args:
        all_results: List of validation results for all graphs
    """
    # Define all metrics with their display names
    metrics_map = [
        ("degree_in", "d_in"),
        ("degree_out", "d_out"),
        ("degree_centrality", "d_cen"),
        ("pagerank", "PR"),
        ("betweenness_centrality", "BW_c"),
        ("out-closeness", "out-c"),
        ("component_id", "comp"),
        ("prerequisite_depth", "pre_d"),
        ("learning_effort", "l_eff"),
        ("educational_importance", "e_imp"),
        ("inverse_weight", "inv_w"),
    ]

    # Print header
    print("\n" + "=" * 80)
    print("VALIDATION MATRIX")
    print("=" * 80)

    # Calculate column widths - make it more compact
    graph_width = max(len(r["graph"]) for r in all_results) + 1
    metric_width = 6  # 6 chars max for cell content

    # Print header row
    header = "Graph".ljust(graph_width)
    for metric_key, short_name in metrics_map:
        header += short_name.ljust(metric_width)
    print(header)
    print("-" * len(header))

    # Print each graph's results
    for result in all_results:
        row = result["graph"].ljust(graph_width)

        for metric_key, short_name in metrics_map:
            if metric_key in result["metrics"]:
                m = result["metrics"][metric_key]
                total = m["passed"] + m["failed"] + m["missing"]

                if m["missing"] == total:
                    cell = "✗MISS"
                elif m["failed"] == 0 and m["missing"] == 0:
                    cell = "✓"
                else:
                    # Calculate average deviation for failures
                    failures = m["failures"]
                    if failures:
                        avg_dev = sum(f["deviation"] for f in failures if f["status"] == "FAIL")
                        avg_dev = (
                            avg_dev / len([f for f in failures if f["status"] == "FAIL"])
                            if any(f["status"] == "FAIL" for f in failures)
                            else 0
                        )
                        # Compact format: ✗99%
                        cell = f"✗{avg_dev:.0f}%"
                    else:
                        cell = "✓"
            else:
                cell = "✗MISS"

            # Left-align in fixed width
            row += cell[:metric_width].ljust(metric_width)

        print(row)

    # Print summary
    print("-" * len(header))
    total_passed = sum(r["passed"] for r in all_results)
    total_failed = sum(r["failed"] for r in all_results)
    total_missing = sum(r["missing"] for r in all_results)
    total_checks = total_passed + total_failed + total_missing

    print(
        f"\nSUMMARY: {total_checks} checks, {total_passed} passed, "
        f"{total_failed} failed, {total_missing} missing"
    )

    if total_failed > 0 or total_missing > 0:
        pass_rate = (total_passed / total_checks * 100) if total_checks > 0 else 0
        print(f"Pass rate: {pass_rate:.1f}%")


def generate_json_report(all_results: List[Dict[str, Any]], output_path: Path) -> None:
    """Generate JSON validation report.

    Args:
        all_results: List of validation results
        output_path: Path to save JSON report
    """
    # Aggregate by metric
    by_metric = {}
    all_metrics = set()

    for result in all_results:
        for metric_name, metric_data in result["metrics"].items():
            all_metrics.add(metric_name)

            if metric_name not in by_metric:
                by_metric[metric_name] = {"passed": 0, "failed": 0, "missing": 0, "failures": []}

            by_metric[metric_name]["passed"] += metric_data["passed"]
            by_metric[metric_name]["failed"] += metric_data["failed"]
            by_metric[metric_name]["missing"] += metric_data["missing"]

            # Add failures with graph context
            for failure in metric_data["failures"]:
                failure_entry = failure.copy()
                failure_entry["graph"] = result["graph"]
                by_metric[metric_name]["failures"].append(failure_entry)

    # Build by_graph section
    by_graph = {}
    for result in all_results:
        passed_metrics = []
        failed_metrics = {}

        for metric_name, metric_data in result["metrics"].items():
            if metric_data["failed"] == 0 and metric_data["missing"] == 0:
                passed_metrics.append(metric_name)
            else:
                # Get first failure for display
                if metric_data["failures"]:
                    first_failure = metric_data["failures"][0]
                    if first_failure["status"] == "MISS":
                        failed_metrics[metric_name] = "MISSING"
                    else:
                        failed_metrics[metric_name] = {
                            "expected": first_failure["expected"],
                            "actual": first_failure["actual"],
                        }

        by_graph[result["graph"]] = {"passed": passed_metrics, "failed": failed_metrics}

    # Calculate totals
    total_passed = sum(r["passed"] for r in all_results)
    total_failed = sum(r["failed"] for r in all_results)
    total_missing = sum(r["missing"] for r in all_results)

    # Build report
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_checks": total_passed + total_failed + total_missing,
            "passed": total_passed,
            "failed": total_failed,
            "missing": total_missing,
        },
        "by_metric": by_metric,
        "by_graph": by_graph,
    }

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nValidation report saved to: {output_path}")


def run_validation_mode(
    viz_dir: Path,
    config: Dict[str, Any],
    logger: logging.Logger,
    graph_filter: Optional[str] = None,
    metric_filter: Optional[str] = None,
    verbose: bool = False,
) -> int:
    """Run validation mode to check test graphs.

    Args:
        viz_dir: Viz directory path
        config: Configuration dictionary
        logger: Logger instance
        graph_filter: Optional filter for specific graph
        metric_filter: Optional filter for specific metric
        verbose: Whether to show detailed output

    Returns:
        Exit code
    """
    print("\n" + "=" * 80)
    print("VALIDATION MODE")
    print("=" * 80)

    # Scan for test graphs
    test_dir = viz_dir / "data" / "test"
    test_pairs = scan_test_graphs(test_dir, graph_filter)

    if not test_pairs:
        print(f"No test graphs found in {test_dir}")
        return EXIT_INPUT_ERROR

    print(f"Found {len(test_pairs)} test graph(s)")

    # Process each test graph
    all_results = []

    for name, input_path, expected_path in test_pairs:
        print(f"\nValidating: {name}")

        # Load files
        with open(input_path, encoding="utf-8") as f:
            input_data = json.load(f)
        with open(expected_path, encoding="utf-8") as f:
            expected_data = json.load(f)

        # Validate schemas
        if not validate_test_files(input_data, expected_data, logger):
            print(f"  ❌ Schema validation failed for {name}")
            continue

        print("  ✅ Schema validation passed")

        # Validate metrics
        result = validate_graph_metrics(
            name, input_data, expected_data, config, logger, metric_filter, verbose
        )
        all_results.append(result)

    # Display results matrix
    if not metric_filter:  # Only show matrix if not filtering by metric
        format_validation_matrix(all_results)

    # Generate JSON report
    report_path = viz_dir / "logs" / "validation_report.json"
    generate_json_report(all_results, report_path)

    # Determine exit code
    total_failed = sum(r["failed"] for r in all_results)
    total_missing = sum(r["missing"] for r in all_results)

    if total_failed > 0 or total_missing > 0:
        return EXIT_RUNTIME_ERROR
    else:
        return EXIT_SUCCESS


def main() -> int:
    """Main entry point for graph2metrics utility.

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Setup console encoding for Windows
    setup_console_encoding()

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Compute metrics for K2-18 knowledge graph visualization"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Use test data from /viz/data/test/ instead of /viz/data/in/",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation mode to check test graphs against expected results",
    )
    parser.add_argument(
        "--graph",
        type=str,
        help="Filter validation to specific graph (e.g., test_line)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        help="Filter validation to specific metric (e.g., pagerank)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed validation output",
    )
    args = parser.parse_args()

    # Define paths
    viz_dir = Path(__file__).parent
    log_file = viz_dir / "logs" / "graph2metrics.log"

    # Setup logging
    logger = setup_logging(log_file, args.test_mode or args.validate)

    try:
        # Handle validation mode
        if args.validate:
            # Load configuration
            config_path = viz_dir / "config.toml"
            logger.info(f"Loading configuration from {config_path}")
            config = load_config(str(config_path))

            # Run validation
            return run_validation_mode(
                viz_dir, config, logger, args.graph, args.metric, args.verbose
            )
        # Log start
        mode_str = " (TEST MODE)" if args.test_mode else ""
        logger.info(f"=== START graph2metrics{mode_str} ===")

        # Load configuration
        config_path = viz_dir / "config.toml"
        logger.info(f"Loading configuration from {config_path}")
        config = load_config(str(config_path))

        # Log config check
        demo_strategy = config.get("graph2metrics", {}).get("demo_strategy", 1)
        logger.info(f"Config loaded, demo_strategy: {demo_strategy}")

        # Determine input directory
        if args.test_mode:
            input_dir = viz_dir / "data" / "test"
            print("[TEST MODE] Using test data")
        else:
            input_dir = viz_dir / "data" / "in"
            print("Using production data")

        # Load input data
        graph_data, concepts_data = load_input_data(input_dir, logger, args.test_mode)

        # Convert to NetworkX
        G = convert_to_networkx(graph_data, logger, args.test_mode)  # noqa: N806

        # Print statistics to console
        if args.test_mode:
            print(
                f"[TEST MODE] Graph loaded: {G.number_of_nodes()} nodes, "
                f"{G.number_of_edges()} edges"
            )
        else:
            print(f"Graph loaded: {G.number_of_nodes()} nodes, " f"{G.number_of_edges()} edges")

        # Compute metrics
        graph_data = compute_centrality_metrics(G, graph_data, config, logger, args.test_mode)

        # Handle large graphs if needed
        max_display = config.get("visualization", {}).get("max_display_nodes", 1000)
        if len(graph_data["nodes"]) > max_display:
            output_dir = viz_dir / "data" / "out"
            full_path = output_dir / "LearningChunkGraph_wow_full.json"
            graph_data = handle_large_graph(graph_data, max_display, full_path, logger)

        # Save output
        output_dir = viz_dir / "data" / "out"
        save_output_data(output_dir, graph_data, concepts_data, logger, args.test_mode)

        # Success
        mode_str = " (TEST MODE)" if args.test_mode else ""
        success_msg = f"Graph metrics computed successfully{mode_str}"
        print(f"✓ {success_msg}")
        logger.info(f"=== SUCCESS graph2metrics{mode_str} ===")
        log_exit(logger, EXIT_SUCCESS, success_msg)
        return EXIT_SUCCESS

    except FileNotFoundError as e:
        error_msg = f"Input file not found: {e}"
        print(f"✗ Error: {error_msg}")
        log_exit(logger, EXIT_INPUT_ERROR, error_msg)
        return EXIT_INPUT_ERROR

    except ConfigValidationError as e:
        error_msg = f"Configuration error: {e}"
        print(f"✗ Error: {error_msg}")
        log_exit(logger, EXIT_CONFIG_ERROR, error_msg)
        return EXIT_CONFIG_ERROR

    except (ValidationError, GraphInvariantError) as e:
        error_msg = f"Validation error: {e}"
        print(f"✗ Error: {error_msg}")
        log_exit(logger, EXIT_INPUT_ERROR, error_msg)
        return EXIT_INPUT_ERROR

    except OSError as e:
        error_msg = f"I/O error: {e}"
        print(f"✗ Error: {error_msg}")
        log_exit(logger, EXIT_IO_ERROR, error_msg)
        return EXIT_IO_ERROR

    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        print(f"✗ Error: {error_msg}")
        log_exit(logger, EXIT_RUNTIME_ERROR, error_msg)
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    sys.exit(main())

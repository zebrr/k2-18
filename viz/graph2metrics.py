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


def compute_centrality_metrics(
    G: nx.DiGraph,  # noqa: N803
    graph_data: Dict[str, Any],
    config: Dict[str, Any],
    logger: logging.Logger,
    test_mode: bool = False,
) -> Dict[str, Any]:
    """Compute centrality metrics for nodes.

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

    # Safe logging wrapper
    def log_info(msg: str) -> None:
        if logger:
            logger.info(msg)

    log_info(f"{mode_prefix}Computing centrality metrics")

    # Get config parameters
    metrics_config = config.get("graph2metrics", {})
    damping = metrics_config.get("pagerank_damping", 0.85)
    max_iter = metrics_config.get("pagerank_max_iter", 100)
    normalized = metrics_config.get("betweenness_normalized", True)
    harmonic = metrics_config.get("closeness_harmonic", True)

    # Progress output to console
    num_nodes = G.number_of_nodes()
    if test_mode:
        print(f"[TEST MODE] Computing metrics for {num_nodes} nodes...")
    else:
        print(f"Computing metrics for {num_nodes} nodes...")

    # 1. Compute degree metrics
    log_info(f"{mode_prefix}Computing degree metrics")
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    degree_centrality = nx.degree_centrality(G)

    # 2. Compute PageRank
    log_info(f"{mode_prefix}Computing PageRank (damping={damping}, max_iter={max_iter})")
    print("  • Computing PageRank...")
    try:
        # Compute PageRank
        pagerank = nx.pagerank(G, alpha=damping, max_iter=max_iter)
    except nx.PowerIterationFailedConvergence:
        log_info(f"{mode_prefix}PageRank didn't converge, using partial results")
        pagerank = nx.pagerank(G, alpha=damping, max_iter=max_iter, tol=1e-3)
    except Exception as e:
        # Fallback for empty graph or other edge cases
        log_info(f"{mode_prefix}PageRank computation failed: {e}, using defaults")
        pagerank = {node: 1.0 / max(1, G.number_of_nodes()) for node in G.nodes()}
    print("  ✓ PageRank computed")

    # 3. Compute betweenness centrality (slowest operation)
    log_info(f"{mode_prefix}Computing betweenness centrality (normalized={normalized})")
    print(f"  • Computing betweenness centrality for {num_nodes} nodes...")
    betweenness = nx.betweenness_centrality(G, normalized=normalized)
    print("  ✓ Betweenness centrality computed")

    # 4. Compute closeness centrality
    log_info(f"{mode_prefix}Computing closeness centrality (harmonic={harmonic})")
    print("  • Computing closeness centrality...")

    if harmonic:
        # Use harmonic centrality for potentially disconnected graphs
        # harmonic_centrality returns values that can be > 1, normalize them
        closeness = nx.harmonic_centrality(G)
        # Normalize to [0, 1] range
        if closeness and G.number_of_nodes() > 1:
            max_val = max(closeness.values()) if closeness.values() else 1.0
            if max_val > 0:
                closeness = {k: v / max_val for k, v in closeness.items()}
    else:
        # Standard closeness (may produce inf for disconnected components)
        closeness = nx.closeness_centrality(G)
    print("  ✓ Closeness centrality computed")

    # 5. Add metrics to all nodes
    log_info(f"{mode_prefix}Adding metrics to {len(graph_data['nodes'])} nodes")

    nodes_with_metrics = 0
    for node in graph_data["nodes"]:
        node_id = node["id"]

        if node_id in G.nodes():
            # Add degree metrics (integers)
            node["degree_in"] = in_degrees.get(node_id, 0)
            node["degree_out"] = out_degrees.get(node_id, 0)

            # Add centrality metrics (floats, safe from NaN/inf)
            node["degree_centrality"] = safe_metric_value(degree_centrality.get(node_id, 0.0))
            node["pagerank"] = safe_metric_value(pagerank.get(node_id, 0.0))
            node["betweenness_centrality"] = safe_metric_value(betweenness.get(node_id, 0.0))
            node["closeness_centrality"] = safe_metric_value(closeness.get(node_id, 0.0))

            nodes_with_metrics += 1

    log_info(f"{mode_prefix}Metrics added to {nodes_with_metrics} nodes")

    # Log statistics
    if nodes_with_metrics > 0:
        pr_values = [n["pagerank"] for n in graph_data["nodes"] if "pagerank" in n]
        if pr_values:
            log_info(f"{mode_prefix}PageRank range: [{min(pr_values):.6f}, {max(pr_values):.6f}]")

    print("  ✓ All metrics computed successfully")

    return graph_data


def compute_clustering(
    G: nx.DiGraph,  # noqa: N803
    graph_data: Dict[str, Any],
    logger: logging.Logger,
    test_mode: bool = False,
) -> Dict[str, Any]:
    """Compute clustering and community detection (stub).

    Args:
        G: NetworkX directed graph
        graph_data: Graph data with metrics
        logger: Logger instance
        test_mode: Whether running in test mode

    Returns:
        Enhanced graph data with clustering
    """
    mode_prefix = "[TEST MODE] " if test_mode else ""
    logger.info(f"{mode_prefix}Computing clustering (stub)")

    # TODO: Will be implemented in VIZ-METRICS-03
    # For now, just return data as is
    return graph_data


def generate_demo_path(
    G: nx.DiGraph,  # noqa: N803
    graph_data: Dict[str, Any],
    config: Dict[str, Any],
    logger: logging.Logger,
    test_mode: bool = False,
) -> Dict[str, Any]:
    """Generate demo path for tour mode (stub).

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
    logger.info(f"{mode_prefix}Generating demo path (stub)")

    # TODO: Will be implemented in VIZ-METRICS-04
    # For now, just return data as is
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
        "closeness_centrality",
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
        ("closeness_centrality", "out-c"),
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

        # Compute metrics (stubs for now)
        graph_data = compute_centrality_metrics(G, graph_data, config, logger, args.test_mode)
        graph_data = compute_clustering(G, graph_data, logger, args.test_mode)
        graph_data = generate_demo_path(G, graph_data, config, logger, args.test_mode)

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

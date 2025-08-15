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
from pathlib import Path
from typing import Any, Dict, Tuple

import networkx as nx

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import ConfigValidationError, load_config
from src.utils.console_encoding import setup_console_encoding
from src.utils.exit_codes import (
    EXIT_CONFIG_ERROR,
    EXIT_INPUT_ERROR,
    EXIT_IO_ERROR,
    EXIT_RUNTIME_ERROR,
    EXIT_SUCCESS,
    log_exit,
)
from src.utils.validation import (
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
    with open(graph_file, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    # Load concepts
    logger.info(f"{mode_prefix}Loading concepts: {concepts_file}")
    with open(concepts_file, "r", encoding="utf-8") as f:
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
    G = nx.DiGraph()

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
    G: nx.DiGraph,
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
        # Use power iteration method which doesn't require scipy
        pagerank = nx.pagerank(G, alpha=damping, max_iter=max_iter, method="power")
    except nx.PowerIterationFailedConvergence:
        log_info(f"{mode_prefix}PageRank didn't converge, using partial results")
        pagerank = nx.pagerank(G, alpha=damping, max_iter=max_iter, tol=1e-3, method="power")
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
    G: nx.DiGraph, graph_data: Dict[str, Any], logger: logging.Logger, test_mode: bool = False
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
    G: nx.DiGraph,
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
    args = parser.parse_args()

    # Define paths
    viz_dir = Path(__file__).parent
    log_file = viz_dir / "logs" / "graph2metrics.log"

    # Setup logging
    logger = setup_logging(log_file, args.test_mode)

    try:
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
        G = convert_to_networkx(graph_data, logger, args.test_mode)

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

    except (IOError, OSError) as e:
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

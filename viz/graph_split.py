#!/usr/bin/env python3
"""
Graph split by clusters utility.

Splits enriched knowledge graph into separate files by cluster_id.
Each cluster becomes an independent subgraph file.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from colorama import Fore, Style, init

    init(autoreset=True)
    HAS_COLORS = True
except ImportError:
    HAS_COLORS = False

    # Fallback if colorama not available
    class Fore:
        RED = GREEN = YELLOW = CYAN = BLUE = MAGENTA = WHITE = ""
        RESET = ""

    class Style:
        BRIGHT = DIM = NORMAL = RESET_ALL = ""


from src.utils.console_encoding import setup_console_encoding  # noqa: E402
from src.utils.exit_codes import (  # noqa: E402
    EXIT_INPUT_ERROR,
    EXIT_IO_ERROR,
    EXIT_SUCCESS,
    log_exit,
)
from src.utils.validation import ValidationError, validate_json  # noqa: E402


def setup_logging(log_file: Path) -> logging.Logger:
    """Set up logging configuration.

    Args:
        log_file: Path to log file

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
    logger.info("=" * 80)
    logger.info("Starting graph_split utility")

    return logger


def _log(level: str, message: str, error: bool = False) -> None:
    """Print formatted log message with timestamp and color.

    Args:
        level: Log level (START, INFO, WARNING, SUCCESS, ERROR)
        message: Message to log
        error: If True, print to stderr
    """
    timestamp = datetime.now().strftime("%H:%M:%S")

    # Color mapping
    colors = {
        "START": Fore.CYAN + Style.BRIGHT,
        "SUCCESS": Fore.GREEN + Style.BRIGHT,
        "ERROR": Fore.RED + Style.BRIGHT,
        "WARNING": Fore.YELLOW,
        "INFO": Fore.BLUE,
    }

    color = colors.get(level, "")
    reset = Style.RESET_ALL if HAS_COLORS else ""

    # Format message
    formatted = f"[{timestamp}] {color}{level:<8}{reset} | {message}"

    if error:
        print(formatted, file=sys.stderr)
    else:
        print(formatted)


def get_filename_padding(cluster_ids: List[int]) -> int:
    """Calculate zero-padding width based on max cluster ID.

    Args:
        cluster_ids: List of cluster IDs

    Returns:
        Number of digits needed for consistent filename padding.
        Returns 1 for empty list.

    Example:
        [0, 1, 2, ..., 15] -> 2 (for cluster_00, cluster_01, ..., cluster_15)
        [0, 1, ..., 99] -> 2
        [0, 1, ..., 100] -> 3
    """
    if not cluster_ids:
        return 1
    return len(str(max(cluster_ids)))


def load_graph(input_file: Path, logger: logging.Logger) -> Dict:
    """Load and validate graph against schema.

    Args:
        input_file: Path to graph JSON file
        logger: Logger instance

    Returns:
        Graph data dictionary

    Raises:
        SystemExit: If file not found or validation fails
    """
    _log("INFO", f"Loading: {input_file.name}")

    if not input_file.exists():
        logger.error(f"File not found: {input_file}")
        _log("ERROR", f"File not found: {input_file}", error=True)
        log_exit(logger, EXIT_INPUT_ERROR, f"File not found: {input_file}")
        sys.exit(EXIT_INPUT_ERROR)

    try:
        with open(input_file, encoding="utf-8") as f:
            graph_data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")
        _log("ERROR", f"Invalid JSON: {e}", error=True)
        log_exit(logger, EXIT_INPUT_ERROR, f"JSON parse error: {e}")
        sys.exit(EXIT_INPUT_ERROR)
    except Exception as e:
        logger.error(f"Failed to load file: {e}")
        _log("ERROR", f"Failed to load file: {e}", error=True)
        log_exit(logger, EXIT_IO_ERROR, f"File read error: {e}")
        sys.exit(EXIT_IO_ERROR)

    # Validate against schema
    try:
        validate_json(graph_data, "LearningChunkGraph")
    except ValidationError as e:
        logger.error(f"Validation failed: {e}")
        _log("ERROR", f"Validation failed: {e}", error=True)
        log_exit(logger, EXIT_INPUT_ERROR, f"Schema validation error: {e}")
        sys.exit(EXIT_INPUT_ERROR)

    nodes_count = len(graph_data.get("nodes", []))
    edges_count = len(graph_data.get("edges", []))
    _log("INFO", f"Graph: {nodes_count} nodes, {edges_count} edges")
    logger.info(f"Loaded graph: {nodes_count} nodes, {edges_count} edges")

    return graph_data


def load_dictionary(input_file: Path, logger: logging.Logger) -> Dict:
    """Load and validate concept dictionary against schema.

    Args:
        input_file: Path to dictionary JSON file
        logger: Logger instance

    Returns:
        Dictionary data dictionary

    Raises:
        SystemExit: If file not found or validation fails
    """
    _log("INFO", f"Loading: {input_file.name}")

    if not input_file.exists():
        logger.error(f"Dictionary file not found: {input_file}")
        _log("ERROR", f"Dictionary file not found: {input_file}", error=True)
        log_exit(logger, EXIT_IO_ERROR, f"Dictionary file not found: {input_file}")
        sys.exit(EXIT_IO_ERROR)

    try:
        with open(input_file, encoding="utf-8") as f:
            dictionary_data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in dictionary: {e}")
        _log("ERROR", f"Invalid JSON in dictionary: {e}", error=True)
        log_exit(logger, EXIT_INPUT_ERROR, f"Dictionary JSON parse error: {e}")
        sys.exit(EXIT_INPUT_ERROR)
    except Exception as e:
        logger.error(f"Failed to load dictionary file: {e}")
        _log("ERROR", f"Failed to load dictionary file: {e}", error=True)
        log_exit(logger, EXIT_IO_ERROR, f"Dictionary file read error: {e}")
        sys.exit(EXIT_IO_ERROR)

    # Validate against schema
    try:
        validate_json(dictionary_data, "ConceptDictionary")
    except ValidationError as e:
        logger.error(f"Dictionary validation failed: {e}")
        _log("ERROR", f"Dictionary validation failed: {e}", error=True)
        log_exit(logger, EXIT_INPUT_ERROR, f"Dictionary schema validation error: {e}")
        sys.exit(EXIT_INPUT_ERROR)

    concepts_count = len(dictionary_data.get("concepts", []))
    _log("INFO", f"Dictionary: {concepts_count} concepts")
    logger.info(f"Loaded dictionary: {concepts_count} concepts")

    return dictionary_data


def identify_clusters(graph_data: Dict, logger: logging.Logger) -> List[int]:
    """Find all unique cluster_id values, sorted ascending.

    Args:
        graph_data: Graph data with nodes
        logger: Logger instance

    Returns:
        Sorted list of cluster IDs (integers)
    """
    nodes = graph_data.get("nodes", [])

    # Extract unique cluster_ids
    cluster_ids = set()
    for node in nodes:
        if "cluster_id" in node:
            cluster_ids.add(node["cluster_id"])

    # Sort ascending
    sorted_clusters = sorted(cluster_ids)

    logger.info(f"Found {len(sorted_clusters)} clusters: {sorted_clusters}")
    return sorted_clusters


def extract_cluster(
    graph_data: Dict, cluster_id: int, logger: logging.Logger
) -> Tuple[Dict, int, int, int]:
    """Extract single cluster subgraph with statistics.

    Args:
        graph_data: Full graph data
        cluster_id: Cluster ID to extract
        logger: Logger instance

    Returns:
        Tuple of (cluster_graph, node_count, edge_count, inter_cluster_edges)
    """
    all_nodes = graph_data.get("nodes", [])
    all_edges = graph_data.get("edges", [])

    # 1. Filter nodes by cluster_id
    cluster_nodes = [n for n in all_nodes if n.get("cluster_id") == cluster_id]

    # 2. Create node_ids set for fast lookup
    cluster_node_ids = {n["id"] for n in cluster_nodes}

    # 3. Filter edges: keep if BOTH endpoints in cluster
    cluster_edges = []
    for edge in all_edges:
        source = edge.get("source")
        target = edge.get("target")
        if source in cluster_node_ids and target in cluster_node_ids:
            cluster_edges.append(edge)

    # 4. Count inter-cluster edges (XOR logic: exactly one endpoint in cluster)
    inter_cluster_count = 0
    for edge in all_edges:
        source = edge.get("source")
        target = edge.get("target")
        source_in_cluster = source in cluster_node_ids
        target_in_cluster = target in cluster_node_ids

        # XOR: exactly one is in cluster
        if source_in_cluster != target_in_cluster:
            inter_cluster_count += 1

    # Build cluster subgraph
    cluster_graph = {"nodes": cluster_nodes, "edges": cluster_edges}

    node_count = len(cluster_nodes)
    edge_count = len(cluster_edges)

    logger.info(
        f"Cluster {cluster_id}: {node_count} nodes, {edge_count} edges, "
        f"{inter_cluster_count} inter-cluster edges"
    )

    return cluster_graph, node_count, edge_count, inter_cluster_count


def extract_cluster_concepts(
    cluster_nodes: List[Dict], concepts_data: Dict, logger: logging.Logger
) -> Tuple[List[Dict], int]:
    """Extract concepts referenced by cluster nodes.

    Args:
        cluster_nodes: List of nodes in the cluster
        concepts_data: Full dictionary data with concepts array
        logger: Logger instance

    Returns:
        Tuple of (concepts_list, count) where concepts_list contains
        concept objects from the dictionary that are referenced by
        the cluster nodes.
    """
    # 1. Collect unique concept IDs from all nodes' concepts: [] field
    concept_ids = set()
    for node in cluster_nodes:
        concept_ids.update(node.get("concepts", []))

    # 2. Build lookup map from concepts_data
    concept_map = {c["concept_id"]: c for c in concepts_data.get("concepts", [])}

    # 3. Extract matching concepts (sorted for deterministic output)
    concepts_list = []
    for cid in sorted(concept_ids):
        if cid in concept_map:
            concepts_list.append(concept_map[cid])
        else:
            logger.warning(f"Concept {cid} not found in dictionary")

    return concepts_list, len(concepts_list)


def sort_nodes(nodes: List[Dict]) -> List[Dict]:
    """Sort nodes: Concepts first (by id), then others (preserve order).

    Args:
        nodes: List of graph nodes

    Returns:
        Sorted list of nodes
    """
    # Split into Concepts and Others
    concepts = [n for n in nodes if n.get("type") == "Concept"]
    others = [n for n in nodes if n.get("type") != "Concept"]

    # Sort Concepts by id alphabetically
    concepts_sorted = sorted(concepts, key=lambda n: n.get("id", ""))

    # Concatenate: Concepts first, then Others (original order preserved)
    return concepts_sorted + others


def create_cluster_metadata(
    cluster_id: int,
    node_count: int,
    edge_count: int,
    original_title: str,
    inter_cluster_links: Optional[Dict[str, List[Dict]]] = None,
) -> Dict:
    """Create new _meta section with subtitle and optional inter-cluster links.

    Args:
        cluster_id: Cluster ID
        node_count: Number of nodes in cluster
        edge_count: Number of edges in cluster
        original_title: Title from source graph
        inter_cluster_links: Optional dict with "incoming" and "outgoing" lists

    Returns:
        New _meta dictionary with optional inter_cluster_links section
    """
    meta = {
        "title": original_title,
        "subtitle": f"Cluster {cluster_id} | Nodes {node_count} | Edges {edge_count}",
    }

    # Add inter_cluster_links only if present and non-empty
    if inter_cluster_links and (
        inter_cluster_links.get("incoming") or inter_cluster_links.get("outgoing")
    ):
        meta["inter_cluster_links"] = inter_cluster_links

    return meta


def create_cluster_dictionary(
    cluster_id: int, concepts_list: List[Dict], original_title: str
) -> Dict:
    """Create cluster dictionary structure with metadata.

    Args:
        cluster_id: Cluster ID
        concepts_list: List of concept objects for this cluster
        original_title: Title from source graph

    Returns:
        Dictionary structure with _meta and concepts array
    """
    return {
        "_meta": {
            "title": original_title,
            "cluster_id": cluster_id,
            "concepts_used": len(concepts_list),
        },
        "concepts": concepts_list,
    }


def save_cluster_graph(
    cluster_graph: Dict,
    cluster_id: int,
    output_dir: Path,
    padding: int,
    logger: logging.Logger,
) -> None:
    """Validate and save cluster graph to file.

    Args:
        cluster_graph: Cluster subgraph to save
        cluster_id: Cluster ID for filename
        output_dir: Output directory path
        padding: Zero-padding width for filename (e.g., 2 for cluster_00.json)
        logger: Logger instance

    Raises:
        SystemExit: If validation or save fails
    """
    # Validate before saving
    try:
        validate_json(cluster_graph, "LearningChunkGraph")
    except ValidationError as e:
        logger.error(f"Cluster {cluster_id} validation failed: {e}")
        _log("ERROR", f"Cluster {cluster_id} validation failed: {e}", error=True)
        log_exit(logger, EXIT_INPUT_ERROR, f"Cluster validation error: {e}")
        sys.exit(EXIT_INPUT_ERROR)

    # Reorder: _meta first, then nodes, then edges
    ordered_graph = {
        "_meta": cluster_graph["_meta"],
        "nodes": cluster_graph["nodes"],
        "edges": cluster_graph["edges"],
    }

    # Save to file with zero-padded filename
    output_file = output_dir / f"LearningChunkGraph_cluster_{cluster_id:0{padding}d}.json"

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(ordered_graph, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved cluster {cluster_id} to: {output_file.name}")
    except Exception as e:
        logger.error(f"Failed to save cluster {cluster_id}: {e}")
        _log("ERROR", f"Failed to save cluster {cluster_id}: {e}", error=True)
        log_exit(logger, EXIT_IO_ERROR, f"Save error: {e}")
        sys.exit(EXIT_IO_ERROR)


def save_cluster_dictionary(
    cluster_dict: Dict,
    cluster_id: int,
    output_dir: Path,
    padding: int,
    logger: logging.Logger,
) -> None:
    """Validate and save cluster dictionary to file.

    Args:
        cluster_dict: Cluster dictionary to save
        cluster_id: Cluster ID for filename
        output_dir: Output directory path
        padding: Zero-padding width for filename (e.g., 2 for cluster_00_dict.json)
        logger: Logger instance

    Raises:
        SystemExit: If validation or save fails
    """
    # Validate before saving
    try:
        validate_json(cluster_dict, "ConceptDictionary")
    except ValidationError as e:
        logger.error(f"Cluster {cluster_id} dictionary validation failed: {e}")
        _log("ERROR", f"Cluster {cluster_id} dictionary validation failed: {e}", error=True)
        log_exit(logger, EXIT_INPUT_ERROR, f"Cluster dictionary validation error: {e}")
        sys.exit(EXIT_INPUT_ERROR)

    # Reorder: _meta first, then concepts
    ordered_dict = {
        "_meta": cluster_dict["_meta"],
        "concepts": cluster_dict["concepts"],
    }

    # Save to file with zero-padded filename
    output_file = output_dir / f"LearningChunkGraph_cluster_{cluster_id:0{padding}d}_dict.json"

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(ordered_dict, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved cluster {cluster_id} dictionary to: {output_file.name}")
    except Exception as e:
        logger.error(f"Failed to save cluster {cluster_id} dictionary: {e}")
        _log("ERROR", f"Failed to save cluster {cluster_id} dictionary: {e}", error=True)
        log_exit(logger, EXIT_IO_ERROR, f"Dictionary save error: {e}")
        sys.exit(EXIT_IO_ERROR)


def find_inter_cluster_links(
    graph_data: Dict, cluster_map: Dict[str, int], logger: logging.Logger
) -> Dict[int, Dict[str, List[Dict]]]:
    """Find PREREQUISITE and ELABORATES links between concepts from different clusters.

    Args:
        graph_data: Full graph with nodes and edges
        cluster_map: Mapping {node_id: cluster_id}
        logger: Logger instance

    Returns:
        Dictionary mapping cluster_id to {"incoming": [...], "outgoing": [...]}
        Each link contains: source, source_text, source_importance, target,
        target_text, target_importance, type, weight, conditions (optional),
        and from_cluster/to_cluster depending on direction
    """
    all_nodes = graph_data.get("nodes", [])
    all_edges = graph_data.get("edges", [])

    # Create node_map for fast lookup
    node_map = {node["id"]: node for node in all_nodes}

    # Initialize result structure
    result: Dict[int, Dict[str, List[Dict]]] = {}

    # Process each edge
    for edge in all_edges:
        # Filter edges by type (expand to 4 types)
        edge_type = edge.get("type")
        allowed_types = {"PREREQUISITE", "ELABORATES", "EXAMPLE_OF", "TESTS"}
        if edge_type not in allowed_types:
            continue

        # Get source and target nodes
        source_id = edge.get("source")
        target_id = edge.get("target")

        source_node = node_map.get(source_id)
        target_node = node_map.get(target_id)

        # Skip if nodes not found
        if not source_node:
            logger.warning(f"Source node {source_id} not found in graph")
            continue
        if not target_node:
            logger.warning(f"Target node {target_id} not found in graph")
            continue

        source_type = source_node.get("type")
        target_type = target_node.get("type")

        # Type-specific node requirements
        if edge_type == "TESTS":
            # TESTS requires source to be Assessment
            if source_type != "Assessment":
                continue
        else:
            # PREREQUISITE, ELABORATES, EXAMPLE_OF require at least one Concept
            if source_type != "Concept" and target_type != "Concept":
                continue

        # Get cluster IDs
        source_cluster = cluster_map.get(source_id)
        target_cluster = cluster_map.get(target_id)

        # Skip if cluster_id missing
        if source_cluster is None:
            logger.warning(f"Node {source_id} missing cluster_id")
            continue
        if target_cluster is None:
            logger.warning(f"Node {target_id} missing cluster_id")
            continue

        # Skip if same cluster (intra-cluster link)
        if source_cluster == target_cluster:
            continue

        # Get node attributes with fallbacks and warnings
        # Truncate text to 500 chars to avoid bloating metadata
        max_text_len = 500

        source_text = source_node.get("text")
        if not source_text:
            logger.warning(f"Node {source_id} missing text field, using node_id")
            source_text = source_id
        elif len(source_text) > max_text_len:
            source_text = source_text[:max_text_len] + "..."

        target_text = target_node.get("text")
        if not target_text:
            logger.warning(f"Node {target_id} missing text field, using node_id")
            target_text = target_id
        elif len(target_text) > max_text_len:
            target_text = target_text[:max_text_len] + "..."

        source_importance = source_node.get("educational_importance")
        if source_importance is None:
            logger.warning(f"Node {source_id} missing educational_importance, using 0.0")
            source_importance = 0.0

        target_importance = target_node.get("educational_importance")
        if target_importance is None:
            logger.warning(f"Node {target_id} missing educational_importance, using 0.0")
            target_importance = 0.0

        # Create base link record
        link = {
            "source": source_id,
            "source_text": source_text,
            "source_type": source_type,
            "source_importance": source_importance,
            "target": target_id,
            "target_text": target_text,
            "target_type": target_type,
            "target_importance": target_importance,
            "type": edge_type,
            "weight": edge.get("weight", 1.0),
        }

        # Add conditions only if present
        if "conditions" in edge:
            link["conditions"] = edge["conditions"]

        # Add to target_cluster incoming links
        if target_cluster not in result:
            result[target_cluster] = {"incoming": [], "outgoing": []}

        incoming_link = link.copy()
        incoming_link["from_cluster"] = source_cluster
        result[target_cluster]["incoming"].append(incoming_link)

        # Add to source_cluster outgoing links
        if source_cluster not in result:
            result[source_cluster] = {"incoming": [], "outgoing": []}

        outgoing_link = link.copy()
        outgoing_link["to_cluster"] = target_cluster
        result[source_cluster]["outgoing"].append(outgoing_link)

    # Select top-3 for each direction, sorted by source_importance
    for cluster_id in result:
        # Sort incoming by source_importance (descending) and take top 3
        result[cluster_id]["incoming"].sort(key=lambda x: x["source_importance"], reverse=True)
        result[cluster_id]["incoming"] = result[cluster_id]["incoming"][:3]

        # Sort outgoing by source_importance (descending) and take top 3
        result[cluster_id]["outgoing"].sort(key=lambda x: x["source_importance"], reverse=True)
        result[cluster_id]["outgoing"] = result[cluster_id]["outgoing"][:3]

    return result


def main() -> int:
    """Main entry point."""
    # Setup console encoding
    setup_console_encoding()

    # Setup paths
    viz_dir = Path(__file__).parent
    log_file = viz_dir / "logs" / "graph_split.log"
    input_file = viz_dir / "data" / "out" / "LearningChunkGraph_wow.json"
    dictionary_file = viz_dir / "data" / "out" / "ConceptDictionary_wow.json"
    output_dir = viz_dir / "data" / "out"

    # Setup logging
    logger = setup_logging(log_file)

    _log("START", "Graph Split by Clusters")

    # Load and validate input graph
    graph_data = load_graph(input_file, logger)

    # Load and validate concept dictionary
    dictionary_data = load_dictionary(dictionary_file, logger)

    # Get original title for metadata (from graph)
    original_title = graph_data.get("_meta", {}).get("title", "Knowledge Graph")

    # Identify all clusters (sorted)
    cluster_ids = identify_clusters(graph_data, logger)

    if not cluster_ids:
        logger.warning("No clusters found in graph")
        _log("WARNING", "No clusters found in graph")
        _log("INFO", "No output files created")
        log_exit(logger, EXIT_SUCCESS)
        return EXIT_SUCCESS

    # Calculate zero-padding for filenames
    padding = get_filename_padding(cluster_ids)
    _log("INFO", f"Found {len(cluster_ids)} clusters (zero-padding: {padding} digits)")

    # Find all inter-cluster links (PREREQUISITE and ELABORATES)
    logger.info("Finding inter-cluster links (PREREQUISITE, ELABORATES)...")
    _log("INFO", "Finding inter-cluster links (PREREQUISITE, ELABORATES)...")

    # Create cluster_map for fast lookup
    cluster_map = {node["id"]: node.get("cluster_id") for node in graph_data.get("nodes", [])}

    all_inter_links = find_inter_cluster_links(graph_data, cluster_map, logger)

    logger.info(f"Found inter-cluster links for {len(all_inter_links)} clusters")
    _log("INFO", f"Found inter-cluster links for {len(all_inter_links)} clusters")

    _log("", "")  # Blank line
    _log("", "Processing clusters:")

    # Track statistics
    files_created = 0
    files_skipped = 0

    # Process each cluster
    for cluster_id in cluster_ids:
        # Extract cluster
        cluster_graph, node_count, edge_count, inter_cluster_count = extract_cluster(
            graph_data, cluster_id, logger
        )

        # Check if single node → skip with WARNING (skip BOTH graph and dictionary)
        if node_count == 1:
            logger.warning(f"Skipping cluster {cluster_id}: only 1 node")
            _log("WARNING", f"Skipping cluster {cluster_id:0{padding}d}: only 1 node")
            files_skipped += 1
            continue

        # Extract concepts for this cluster
        concepts_list, concepts_count = extract_cluster_concepts(
            cluster_graph["nodes"], dictionary_data, logger
        )

        # Create cluster dictionary
        cluster_dict = create_cluster_dictionary(
            cluster_id=cluster_id,
            concepts_list=concepts_list,
            original_title=original_title,
        )

        # Sort nodes (Concepts first)
        cluster_graph["nodes"] = sort_nodes(cluster_graph["nodes"])

        # Get inter-cluster links for this cluster
        inter_links = all_inter_links.get(cluster_id, {"incoming": [], "outgoing": []})

        # Create metadata with inter-cluster links
        cluster_graph["_meta"] = create_cluster_metadata(
            cluster_id=cluster_id,
            node_count=node_count,
            edge_count=edge_count,
            original_title=original_title,
            inter_cluster_links=inter_links,
        )

        # Save cluster graph with zero-padded filename
        save_cluster_graph(cluster_graph, cluster_id, output_dir, padding, logger)

        # Save cluster dictionary with zero-padded filename
        save_cluster_dictionary(cluster_dict, cluster_id, output_dir, padding, logger)

        # Log statistics
        _log(
            "INFO",
            f"Cluster {cluster_id:0{padding}d}: {node_count} nodes, "
            f"{edge_count} edges, {concepts_count} concepts",
        )

        # Log inter-cluster links statistics
        if inter_links["incoming"] or inter_links["outgoing"]:
            _log(
                "INFO",
                f"  └─ Inter-cluster links: "
                f"{len(inter_links['incoming'])} incoming, "
                f"{len(inter_links['outgoing'])} outgoing",
            )

        files_created += 1

    # Summary
    _log("", "")  # Blank line
    _log("SUCCESS", "✅ Split completed successfully")
    _log("INFO", f"Output files saved to: {output_dir}/")
    _log(
        "INFO",
        f"Created {files_created} cluster graphs + "
        f"{files_created} cluster dictionaries ({files_skipped} skipped)",
    )

    logger.info(
        f"Split completed: {files_created} graphs, "
        f"{files_created} dictionaries, {files_skipped} skipped"
    )
    log_exit(logger, EXIT_SUCCESS)

    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())

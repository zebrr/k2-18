#!/usr/bin/env python
"""
refiner.py - Utility for adding long-range connections to knowledge graphs.

Searches for missed connections between nodes that did not appear in the same context
during initial processing. Uses semantic similarity to find candidates
and LLM for analyzing connection types.
"""

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to PYTHONPATH for correct imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import faiss
import numpy as np

from src.utils.config import load_config
# Set up UTF-8 encoding for Windows console
from src.utils.console_encoding import setup_console_encoding
from src.utils.exit_codes import (EXIT_API_LIMIT_ERROR, EXIT_CONFIG_ERROR,
                                  EXIT_INPUT_ERROR, EXIT_IO_ERROR,
                                  EXIT_RUNTIME_ERROR, EXIT_SUCCESS)
from src.utils.llm_client import OpenAIClient
from src.utils.llm_embeddings import get_embeddings
from src.utils.validation import validate_graph_invariants, validate_json

setup_console_encoding()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("refiner")


def setup_json_logging(config: Dict) -> logging.Logger:
    """
    Set up JSON Lines logging for refiner.

    Args:
        config: Refiner configuration

    Returns:
        Configured logger
    """
    import json
    import logging
    from datetime import datetime, timezone

    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Generate log file name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"refiner_{timestamp}.log"

    # Create custom formatter for JSON Lines
    class JSONLineFormatter(logging.Formatter):
        def format(self, record):
            log_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "event": getattr(record, "event", "log"),
            }

            # Add additional fields from record
            for key in [
                "slice_id",
                "node_id",
                "concept_id",
                "action",
                "source",
                "target",
                "type",
                "weight",
                "conditions",
                "pairs_count",
                "tokens_used",
                "duration_ms",
                "edges_added",
                "error",
            ]:
                if hasattr(record, key):
                    log_data[key] = getattr(record, key)

            # Add message if exists
            if record.getMessage():
                log_data["message"] = record.getMessage()

            # Add data for DEBUG level
            if record.levelname == "DEBUG":
                for key in [
                    "prompt",
                    "response",
                    "raw_response",
                    "new_aliases",
                    "old_len",
                    "new_len",
                    "similarity",
                ]:
                    if hasattr(record, key):
                        log_data[key] = getattr(record, key)

            return json.dumps(log_data, ensure_ascii=False)

    # Create logger
    logger = logging.getLogger("refiner")
    logger.setLevel(
        logging.DEBUG
        if config.get("log_level", "info").lower() == "debug"
        else logging.INFO
    )

    # Remove existing handlers
    logger.handlers = []

    # File handler для JSON Lines
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(JSONLineFormatter())
    logger.addHandler(file_handler)

    # Console handler for regular output (not JSON)
    # console_handler = logging.StreamHandler()
    # console_handler.setFormatter(logging.Formatter(
    #    '[%(asctime)s] %(levelname)-8s | %(message)s',
    # datefmt='%H:%M:%S'
    # ))
    # console_handler.setLevel(logging.INFO)  # Only important messages to console
    # logger.addHandler(console_handler)

    # Log start of work
    logger.info(
        "Refiner started",
        extra={
            "event": "refiner_start",
            "config": {
                "model": config["model"],
                "tpm_limit": config["tpm_limit"],
                "sim_threshold": config["sim_threshold"],
                "max_pairs_per_node": config["max_pairs_per_node"],
            },
        },
    )

    return logger


def log_edge_operation(logger: logging.Logger, operation: str, edge: Dict, **kwargs):
    """
    Log edge operations in structured format.

    Args:
        logger: Logger
        operation: Operation type (added, updated, replaced, removed)
        edge: Edge data
        **kwargs: Additional parameters for logging
    """
    extra = {
        "event": f"edge_{operation}",
        "source": edge.get("source"),
        "target": edge.get("target"),
        "type": edge.get("type"),
        "weight": edge.get("weight"),
    }
    extra.update(kwargs)

    message = f"Edge {operation}: {edge.get('source')} -> {edge.get('target')} ({edge.get('type')})"

    if operation in ["updated", "replaced"]:
        logger.info(message, extra=extra)
    else:
        logger.debug(message, extra=extra)


def validate_refiner_config(config: Dict) -> None:
    """
    Validate refiner configuration parameters.

    Args:
        config: [refiner] section from config

    Raises:
        ValueError: If parameters are invalid
    """
    # Check required parameters
    required = [
        "embedding_model",
        "sim_threshold",
        "max_pairs_per_node",
        "model",
        "api_key",
        "tpm_limit",
        "max_completion",
        "weight_low",
        "weight_mid",
        "weight_high",
        "faiss_M",
        "faiss_metric",
    ]

    for param in required:
        if param not in config:
            raise ValueError(f"Missing required parameter: {param}")

    # Check api_key
    if not config["api_key"].strip():
        raise ValueError("api_key cannot be empty")

    # Check ranges
    if not 0 <= config["sim_threshold"] <= 1:
        raise ValueError(
            f"sim_threshold must be in [0,1], got {config['sim_threshold']}"
        )

    if config["max_pairs_per_node"] <= 0:
        raise ValueError(
            f"max_pairs_per_node must be > 0, got {config['max_pairs_per_node']}"
        )

    # Check weights
    if not (
        0 <= config["weight_low"] < config["weight_mid"] < config["weight_high"] <= 1
    ):
        raise ValueError(
            f"Weights must satisfy: 0 <= weight_low < weight_mid < weight_high <= 1, "
            f"got {config['weight_low']}, {config['weight_mid']}, {config['weight_high']}"
        )

    # Check FAISS parameters
    if config["faiss_M"] <= 0:
        raise ValueError(f"faiss_M must be > 0, got {config['faiss_M']}")

    if config["faiss_metric"] not in ["INNER_PRODUCT", "L2"]:
        raise ValueError(
            f"faiss_metric must be INNER_PRODUCT or L2, got {config['faiss_metric']}"
        )

    # Check reasoning parameters for o-models
    if config.get("model", "").startswith("o"):
        if config.get("reasoning_effort") not in ["low", "medium", "high", None]:
            raise ValueError(
                f"reasoning_effort must be low/medium/high, got {config.get('reasoning_effort')}"
            )

        if config.get("reasoning_summary") not in ["auto", "concise", "detailed", None]:
            raise ValueError(
                f"reasoning_summary must be auto/concise/detailed, got {config.get('reasoning_summary')}"
            )


def load_and_validate_graph(input_path: Path) -> Dict:
    """
    Load and validate graph.

    Args:
        input_path: Path to graph file

    Returns:
        Loaded graph

    Raises:
        Exception: On loading or validation errors
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    # Schema validation
    validate_json(graph, "LearningChunkGraph")

    return graph


def extract_target_nodes(graph: Dict) -> List[Dict]:
    """
    Extract nodes of type Chunk and Assessment.

    Args:
        graph: Knowledge graph

    Returns:
        List of target nodes
    """
    target_types = {"Chunk", "Assessment"}
    return [node for node in graph.get("nodes", []) if node.get("type") in target_types]


def build_edges_index(graph: Dict) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Build index of existing edges for fast lookup.

    Args:
        graph: Knowledge graph

    Returns:
        Dictionary {source_id: {target_id: [edges]}}
    """
    edges_index = {}

    for edge in graph.get("edges", []):
        source = edge["source"]
        target = edge["target"]

        if source not in edges_index:
            edges_index[source] = {}

        if target not in edges_index[source]:
            edges_index[source][target] = []

        edges_index[source][target].append(edge)

    return edges_index


def get_node_embeddings(
    nodes: List[Dict], config: Dict, logger: logging.Logger
) -> Dict[str, np.ndarray]:
    """
    Get embeddings for all nodes.

    Args:
        nodes: List of nodes (Chunk/Assessment)
        config: Refiner configuration
        logger: Logger for output

    Returns:
        Dictionary {node_id: embedding_vector}
    """
    logger.info(f"Getting embeddings for {len(nodes)} nodes")

    # Extract texts in the same order as nodes
    texts = []
    node_ids = []

    for node in nodes:
        if node.get("text", "").strip():
            texts.append(node["text"])
            node_ids.append(node["id"])
        else:
            logger.warning(f"Node {node['id']} has empty text, skipping")

    if not texts:
        logger.error("No texts to get embeddings for")
        return {}

    try:
        # Get embeddings in batches
        embeddings = get_embeddings(texts, config)

        # Create dictionary for fast access
        embeddings_dict = {}
        for i, node_id in enumerate(node_ids):
            embeddings_dict[node_id] = embeddings[i]

        logger.info(
            f"Successfully obtained embeddings for {len(embeddings_dict)} nodes"
        )
        return embeddings_dict

    except Exception as e:
        logger.error(f"Failed to get embeddings: {e}")
        raise


def build_similarity_index(
    embeddings_dict: Dict[str, np.ndarray],
    nodes: List[Dict],
    config: Dict,
    logger: logging.Logger,
) -> Tuple[faiss.Index, List[str]]:
    """
    Build FAISS index for finding similar nodes.

    Args:
        embeddings_dict: Dictionary {node_id: embedding}
        nodes: List of nodes in ascending local_start order
        config: Refiner configuration
        logger: Logger

    Returns:
        (faiss_index, node_ids_list) - index and list of IDs in order of addition
    """
    # Sort nodes by local_start for determinism
    sorted_nodes = sorted(nodes, key=lambda n: (n.get("local_start", 0), n["id"]))

    # Collect embeddings in correct order
    embeddings_list = []
    node_ids_list = []

    for node in sorted_nodes:
        if node["id"] in embeddings_dict:
            embeddings_list.append(embeddings_dict[node["id"]])
            node_ids_list.append(node["id"])

    if not embeddings_list:
        raise ValueError("No embeddings to build index")

    # Convert to numpy array
    embeddings_matrix = np.array(embeddings_list, dtype=np.float32)
    dim = embeddings_matrix.shape[1]  # Should be 1536

    logger.info(
        f"Building FAISS index: dim={dim}, M={config['faiss_M']}, "
        f"metric={config['faiss_metric']}"
    )

    # Create HNSW index
    if config["faiss_metric"] == "INNER_PRODUCT":
        metric = faiss.METRIC_INNER_PRODUCT
    else:
        # In case other metrics are added
        metric = faiss.METRIC_L2

    index = faiss.IndexHNSWFlat(dim, config["faiss_M"], metric)
    index.hnsw.efConstruction = config.get("faiss_efC", 200)

    # Add vectors
    index.add(embeddings_matrix)

    logger.info(f"FAISS index built with {index.ntotal} vectors")
    return index, node_ids_list


def generate_candidate_pairs(
    nodes: List[Dict],
    embeddings_dict: Dict[str, np.ndarray],
    index: faiss.Index,
    node_ids_list: List[str],
    edges_index: Dict,
    config: Dict,
    logger: logging.Logger,
) -> List[Dict]:
    """
    Generate candidate node pairs for connection analysis.

    Args:
        nodes: List of all nodes
        embeddings_dict: Embeddings dictionary
        index: FAISS index
        node_ids_list: List of IDs in index order
        edges_index: Index of existing edges
        config: Configuration
        logger: Logger

    Returns:
        List of dictionaries with pair information for analysis
    """
    # Create fast access to nodes by ID
    nodes_by_id = {node["id"]: node for node in nodes}

    # Search parameters
    k_neighbors = min(
        config["max_pairs_per_node"] + 1, len(nodes)
    )  # +1 because it will find itself
    sim_threshold = config["sim_threshold"]

    candidate_pairs = []
    processed_pairs = set()  # To avoid duplicates (A,B) and (B,A)

    logger.info(
        f"Searching for candidates: k={k_neighbors-1}, threshold={sim_threshold}"
    )

    # For each node, search for candidates
    for i, node_id_a in enumerate(node_ids_list):
        node_a = nodes_by_id[node_id_a]
        embedding_a = embeddings_dict[node_id_a]

        # Search for nearest neighbors
        query = np.array([embedding_a], dtype=np.float32)
        similarities, indices = index.search(query, k_neighbors)

        candidates_for_a = []

        for j, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx == i:  # Skip the node itself
                continue

            if sim < sim_threshold:  # Filter by threshold
                continue

            node_id_b = node_ids_list[idx]
            node_b = nodes_by_id[node_id_b]

            # Check local_start order
            local_start_a = node_a.get("local_start", 0)
            local_start_b = node_b.get("local_start", 0)

            if local_start_a >= local_start_b:
                continue  # Process only pairs where A < B

            # Check that pair hasn't been processed yet
            pair_key = (node_id_a, node_id_b)
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)

            # Collect existing edges between nodes
            existing_edges = []

            # A -> B
            if node_id_a in edges_index and node_id_b in edges_index[node_id_a]:
                for edge in edges_index[node_id_a][node_id_b]:
                    existing_edges.append(edge)

            # B -> A
            if node_id_b in edges_index and node_id_a in edges_index[node_id_b]:
                for edge in edges_index[node_id_b][node_id_a]:
                    existing_edges.append(edge)

            candidates_for_a.append(
                {
                    "node_id": node_id_b,
                    "text": node_b["text"],
                    "similarity": float(sim),
                    "existing_edges": existing_edges,
                }
            )

        # Sort candidates by descending similarity and take top-K
        candidates_for_a.sort(key=lambda x: x["similarity"], reverse=True)
        candidates_for_a = candidates_for_a[: config["max_pairs_per_node"]]

        if candidates_for_a:
            candidate_pairs.append(
                {
                    "source_node": {"id": node_id_a, "text": node_a["text"]},
                    "candidates": candidates_for_a,
                }
            )

        # Progress logging
        if (i + 1) % 10 == 0:
            logger.debug(f"Processed {i + 1}/{len(node_ids_list)} nodes")

    logger.info(
        f"Generated {len(candidate_pairs)} nodes with candidates, "
        f"total {sum(len(p['candidates']) for p in candidate_pairs)} pairs"
    )

    return candidate_pairs


def load_refiner_prompt(config: Dict) -> str:
    """
    Load and prepare prompt for connection analysis.

    Args:
        config: Refiner configuration with weights

    Returns:
        Prepared prompt

    Raises:
        FileNotFoundError: If prompt file is not found
    """
    prompt_path = Path(__file__).parent / "prompts" / "refiner_relation.md"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    # Load prompt
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    # Substitute weights from config
    prompt = prompt_template.replace("{weight_low}", str(config["weight_low"]))
    prompt = prompt.replace(
        "{weight_mid}", str(config["weight_mid"])
    )  # Now working with prompt, not prompt_template
    prompt = prompt.replace("{weight_high}", str(config["weight_high"]))  # And here too

    return prompt


def analyze_candidate_pairs(
    candidate_pairs: List[Dict], graph: Dict, config: Dict, logger: logging.Logger
) -> List[Dict]:
    """
    Analyze candidate pairs through LLM to determine connection types.

    Args:
        candidate_pairs: List of node pairs for analysis
        graph: Source graph (for context)
        config: Refiner configuration
        logger: Logger

    Returns:
        List of new/updated edges
    """
    # Load prompt
    try:
        prompt = load_refiner_prompt(config)
        logger.info("Loaded refiner prompt with weight substitutions")
    except FileNotFoundError as e:
        logger.error(f"Failed to load prompt: {e}")
        raise

    # Initialize LLM client with correct config format
    llm_config = {
        "api_key": config["api_key"],
        "model": config["model"],
        "tpm_limit": config["tpm_limit"],
        "tpm_safety_margin": config.get("tpm_safety_margin", 0.15),
        "max_completion": config["max_completion"],
        "timeout": config.get("timeout", 45),
        "max_retries": config.get("max_retries", 3),
    }

    # Add parameters for regular models
    if not config.get("model", "").startswith("o"):
        llm_config["temperature"] = config.get("temperature", 0.6)

    # Add parameters for reasoning models
    if config.get("reasoning_effort"):
        llm_config["reasoning_effort"] = config["reasoning_effort"]
    if config.get("reasoning_summary"):
        llm_config["reasoning_summary"] = config["reasoning_summary"]

    llm_client = OpenAIClient(llm_config)

    all_new_edges = []
    previous_response_id = None

    # Terminal output START
    import time
    from datetime import datetime, timedelta, timezone

    start_time = time.time()
    utc3_tz = timezone(timedelta(hours=3))
    start_timestamp = datetime.now(utc3_tz).strftime("%H:%M:%S")

    print(
        f"[{start_timestamp}] START    | {len(candidate_pairs)} nodes | "
        f"model={config['model']} | tpm={config['tpm_limit']//1000}k"
    )

    logger.info(f"Starting LLM analysis of {len(candidate_pairs)} nodes")

    # Process each node sequentially
    for i, pair_data in enumerate(candidate_pairs):
        source_node = pair_data["source_node"]
        candidates = pair_data["candidates"]

        # Form input for LLM
        input_data = {"source_node": source_node, "candidates": candidates}

        logger.debug(
            f"Processing node {i+1}/{len(candidate_pairs)}: {source_node['id']} "
            f"with {len(candidates)} candidates"
        )

        request_start = time.time()

        # Log prompt in DEBUG mode
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "LLM request",
                extra={
                    "event": "llm_request",
                    "node_id": source_node["id"],
                    "prompt": prompt[:1000] + "..." if len(prompt) > 1000 else prompt,
                    "input_data": json.dumps(input_data, ensure_ascii=False)[:1000]
                    + "...",
                },
            )

        try:
            # Send request to LLM
            response_text, response_id, usage = llm_client.create_response(
                instructions=prompt,
                input_data=json.dumps(input_data, ensure_ascii=False, indent=2),
                previous_response_id=previous_response_id,
            )

            # Log response in DEBUG mode
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "LLM response",
                    extra={
                        "event": "llm_response",
                        "node_id": source_node["id"],
                        "response": (
                            response_text[:1000] + "..."
                            if len(response_text) > 1000
                            else response_text
                        ),
                        "usage": usage,
                    },
                )

            # Save response_id for next call
            previous_response_id = response_id

            # Parse response
            try:
                edges_response = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to parse LLM response for node {source_node['id']}: {e}"
                )
                logger.debug(f"Raw response: {response_text}")

                # One repair-retry
                logger.info("Attempting repair retry with clarification")
                repair_prompt = (
                    prompt + "\n\nPLEASE RETURN ONLY VALID JSON ARRAY, NO OTHER TEXT."
                )

                # Update instructions in client for repair
                llm_client.last_instructions = repair_prompt
                response_text, response_id, usage = llm_client.repair_response(
                    instructions=repair_prompt,
                    input_data=json.dumps(input_data, ensure_ascii=False, indent=2),
                )

                try:
                    edges_response = json.loads(response_text)
                except json.JSONDecodeError as e2:
                    logger.error(f"Failed to parse repaired response: {e2}")

                    # Save bad response
                    bad_response_path = Path(f"logs/{source_node['id']}_bad.json")
                    bad_response_path.parent.mkdir(exist_ok=True)

                    with open(bad_response_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "node_id": source_node["id"],
                                "original_response": response_text,
                                "error": str(e2),
                            },
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )

                    logger.error(f"Saved bad response to {bad_response_path}")
                    continue  # Skip this node

            # Validation and response processing
            valid_edges = validate_llm_edges(
                edges_response, source_node["id"], candidates, graph, logger
            )

            all_new_edges.extend(valid_edges)

            # Progress logging
            added_count = len([e for e in valid_edges if e.get("type")])
            request_time_ms = int((time.time() - request_start) * 1000)

            # Terminal output NODE
            node_timestamp = datetime.now(utc3_tz).strftime("%H:%M:%S")
            tokens_used = usage.get("total_tokens", 0) if usage else 0

            print(
                f"[{node_timestamp}] NODE     | ✅ {i+1:03d}/{len(candidate_pairs):03d} | "
                f"pairs={len(candidates)} | tokens={tokens_used} | {request_time_ms}ms | "
                f"edges_added={added_count}"
            )

            logger.info(
                f"[{i+1}/{len(candidate_pairs)}] Node {source_node['id']}: "
                f"{added_count} new edges from {len(candidates)} candidates"
            )

        except Exception as e:
            # Error output to terminal
            error_timestamp = datetime.now(utc3_tz).strftime("%H:%M:%S")

            if "RateLimitError" in str(e):
                print(
                    f"[{error_timestamp}] ERROR    | ⚠️ RateLimitError | will retry..."
                )
            else:
                print(
                    f"[{error_timestamp}] ERROR    | ⚠️ {type(e).__name__}: {str(e)[:50]}..."
                )

            logger.error(f"Error processing node {source_node['id']}: {e}")
            continue

    # Terminal output END
    end_time = time.time()
    elapsed = int(end_time - start_time)
    minutes = elapsed // 60
    seconds = elapsed % 60

    end_timestamp = datetime.now(utc3_tz).strftime("%H:%M:%S")
    total_added = len([e for e in all_new_edges if e.get("type")])

    print(
        f"[{end_timestamp}] END      | Done | nodes={len(candidate_pairs)} | "
        f"edges_added={total_added} | time={minutes}m {seconds}s"
    )

    logger.info(f"LLM analysis complete: {len(all_new_edges)} total edges to process")
    return all_new_edges


def validate_llm_edges(
    edges_response: List[Dict],
    source_id: str,
    candidates: List[Dict],
    graph: Dict,
    logger: logging.Logger,
) -> List[Dict]:
    """
    Validate edges received from LLM.

    Args:
        edges_response: LLM response (list of edges)
        source_id: Source node ID
        candidates: List of candidates
        graph: Graph for checking node existence
        logger: Logger

    Returns:
        List of valid edges
    """
    # Collect all node IDs for checking
    all_node_ids = {node["id"] for node in graph["nodes"]}
    candidate_ids = {c["node_id"] for c in candidates}

    # Valid edge types
    valid_edge_types = {
        "PREREQUISITE",
        "ELABORATES",
        "EXAMPLE_OF",
        "HINT_FORWARD",
        "REFER_BACK",
        "PARALLEL",
        "TESTS",
        "REVISION_OF",
        "MENTIONS",
    }

    valid_edges = []

    for edge_data in edges_response:
        # Skip records with type: null
        if edge_data.get("type") is None:
            continue

        source = edge_data.get("source")
        target = edge_data.get("target")
        edge_type = edge_data.get("type")
        weight = edge_data.get("weight", 0.5)
        conditions = edge_data.get("conditions", "")

        # Basic validation
        if not all([source, target, edge_type]):
            logger.warning(f"Incomplete edge data: {edge_data}")
            continue

        # Check source (should be current node)
        if source != source_id:
            logger.warning(f"Invalid source: expected {source_id}, got {source}")
            continue

        # Check target (should be in candidates)
        if target not in candidate_ids:
            logger.warning(f"Target {target} not in candidates")
            continue

        # Check type
        if edge_type not in valid_edge_types:
            logger.warning(f"Invalid edge type: {edge_type}")
            continue

        # Check weight
        try:
            weight = float(weight)
            if not 0 <= weight <= 1:
                logger.warning(f"Weight out of range: {weight}")
                weight = 0.5
        except (ValueError, TypeError):
            logger.warning(f"Invalid weight: {weight}, using 0.5")
            weight = 0.5

        # Check PREREQUISITE self-loops
        if edge_type == "PREREQUISITE" and source == target:
            logger.warning(f"PREREQUISITE self-loop detected: {source}")
            continue

        # Add valid edge
        valid_edges.append(
            {
                "source": source,
                "target": target,
                "type": edge_type,
                "weight": weight,
                "conditions": conditions,
            }
        )

    return valid_edges


def update_graph_with_new_edges(
    graph: Dict, new_edges: List[Dict], logger: logging.Logger
) -> Dict[str, int]:
    """
    Update graph with new edges from LLM according to specification logic.

    Args:
        graph: Source graph
        new_edges: List of new edges from LLM
        logger: Logger for tracking changes

    Returns:
        Change statistics: {added, updated, replaced, total_processed}
    """
    stats = {
        "added": 0,
        "updated": 0,
        "replaced": 0,
        "self_loops_removed": 0,
        "total_processed": 0,
    }

    # Create index of existing edges for fast lookup
    # Key: (source, target) -> list of edge indices
    edge_index = {}
    for i, edge in enumerate(graph["edges"]):
        key = (edge["source"], edge["target"])
        if key not in edge_index:
            edge_index[key] = []
        edge_index[key].append(i)

    # Process each new edge
    for new_edge in new_edges:
        stats["total_processed"] += 1

        source = new_edge["source"]
        target = new_edge["target"]
        edge_type = new_edge["type"]
        weight = new_edge["weight"]

        key = (source, target)

        # Scenario 1: New edge (no such source+target)
        if key not in edge_index:
            # Add with marking
            new_edge["conditions"] = "added_by=refiner_v1"
            graph["edges"].append(new_edge)
            stats["added"] += 1

            logger.debug(
                f"Added new edge: {source} -> {target} ({edge_type}, w={weight:.2f})"
            )
            log_edge_operation(logger, "added", new_edge)
            # Update index
            if key not in edge_index:
                edge_index[key] = []
            edge_index[key].append(len(graph["edges"]) - 1)

        else:
            # There is existing edge(s) with such source+target
            existing_indices = edge_index[key]

            # Look for edge with same type
            same_type_idx = None
            for idx in existing_indices:
                if graph["edges"][idx]["type"] == edge_type:
                    same_type_idx = idx
                    break

            if same_type_idx is not None:
                # Scenario 2: Duplicate (same source+target+type)
                existing_edge = graph["edges"][same_type_idx]
                old_weight = existing_edge.get("weight", 0.5)

                if weight > old_weight:
                    # Update to higher weight
                    existing_edge["weight"] = weight
                    stats["updated"] += 1

                    logger.debug(
                        f"Updated weight: {source} -> {target} ({edge_type}), "
                        f"old={old_weight:.2f}, new={weight:.2f}"
                    )
                    log_edge_operation(
                        logger, "updated", existing_edge, old_weight=old_weight
                    )
                else:
                    logger.debug(
                        f"Kept existing weight: {source} -> {target} ({edge_type}), "
                        f"existing={old_weight:.2f} >= new={weight:.2f}"
                    )

            else:
                # Scenario 3: Type replacement (same source+target, different type)
                # Find edge with maximum weight among existing
                max_weight_idx = None
                max_weight = -1

                for idx in existing_indices:
                    edge_weight = graph["edges"][idx].get("weight", 0.5)
                    if edge_weight > max_weight:
                        max_weight = edge_weight
                        max_weight_idx = idx

                # Replace only if new weight >= old
                if weight >= max_weight:
                    # Save old edges for logging
                    removed_edges = []

                    # Remove all old edges between these nodes
                    # (sort indices in reverse order for correct deletion)
                    for idx in sorted(existing_indices, reverse=True):
                        old_edge = graph["edges"].pop(idx)
                        removed_edges.append(old_edge)
                        logger.debug(
                            f"Removed old edge: {source} -> {target} "
                            f"({old_edge['type']}, w={old_edge.get('weight', 0.5):.2f})"
                        )
                    # Add new edge with marking
                    new_edge["conditions"] = "fixed_by=refiner_v1"
                    graph["edges"].append(new_edge)
                    stats["replaced"] += 1

                    logger.debug(
                        f"Replaced edge type: {source} -> {target}, "
                        f"new type={edge_type}, w={weight:.2f}"
                    )
                    log_edge_operation(
                        logger,
                        "replaced",
                        new_edge,
                        old_types=[e["type"] for e in removed_edges],
                    )

                    # Recreate index after changes
                    edge_index = {}
                    for i, edge in enumerate(graph["edges"]):
                        key = (edge["source"], edge["target"])
                        if key not in edge_index:
                            edge_index[key] = []
                        edge_index[key].append(i)
                else:
                    logger.debug(
                        f"Kept existing edge: {source} -> {target}, "
                        f"max existing weight={max_weight:.2f} > new={weight:.2f}"
                    )

    # Final cleanup: remove PREREQUISITE self-loops
    edges_before = len(graph["edges"])
    graph["edges"] = [
        edge
        for edge in graph["edges"]
        if not (edge["type"] == "PREREQUISITE" and edge["source"] == edge["target"])
    ]

    self_loops_removed = edges_before - len(graph["edges"])
    if self_loops_removed > 0:
        stats["self_loops_removed"] = self_loops_removed
        logger.info(f"Removed {self_loops_removed} PREREQUISITE self-loops")

    # Log final statistics
    logger.info(
        f"Graph update complete: added={stats['added']}, "
        f"updated={stats['updated']}, replaced={stats['replaced']}, "
        f"self-loops removed={stats['self_loops_removed']}"
    )

    return stats


def main():
    """Main function."""
    logger = None  # Initialize variable

    try:
        # Load configuration
        config = load_config()
        refiner_config = config["refiner"]

        # Check run flag BEFORE setting up logging
        if not refiner_config.get("run", True):
            # Simple logging for run=false case
            print("Refiner is disabled (run=false), copying file without changes")

            input_path = Path("data/out/LearningChunkGraph_dedup.json")
            output_path = Path("data/out/LearningChunkGraph.json")

            if not input_path.exists():
                print(f"ERROR: Input file not found: {input_path}")
                return EXIT_INPUT_ERROR

            shutil.copy2(input_path, output_path)
            print(f"Copied {input_path} to {output_path}")
            return EXIT_SUCCESS

        # Set up JSON logging only if run=true
        logger = setup_json_logging(refiner_config)

        # Configuration validation
        try:
            validate_refiner_config(refiner_config)
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            return EXIT_CONFIG_ERROR

        # File paths
        input_path = Path("data/out/LearningChunkGraph_dedup.json")
        output_path = Path("data/out/LearningChunkGraph.json")

        # Load and validate graph
        try:
            graph = load_and_validate_graph(input_path)
            logger.info(
                f"Loaded graph with {len(graph.get('nodes', []))} nodes "
                f"and {len(graph.get('edges', []))} edges",
                extra={
                    "event": "graph_loaded",
                    "nodes_count": len(graph.get("nodes", [])),
                    "edges_count": len(graph.get("edges", [])),
                },
            )
        except FileNotFoundError:
            logger.error(f"Input file not found: {input_path}")
            return EXIT_INPUT_ERROR
        except Exception as e:
            logger.error(f"Failed to load/validate graph: {e}")
            return EXIT_INPUT_ERROR

        # Extract target nodes
        target_nodes = extract_target_nodes(graph)
        logger.info(
            f"Found {len(target_nodes)} Chunk/Assessment nodes",
            extra={"event": "nodes_extracted", "target_nodes_count": len(target_nodes)},
        )

        if not target_nodes:
            logger.warning(
                "No Chunk/Assessment nodes found, saving graph without changes"
            )
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(graph, f, ensure_ascii=False, indent=2)
            return EXIT_SUCCESS

        # Build edge index
        edges_index = build_edges_index(graph)

        # Generate candidates for analysis
        try:
            # Get embeddings for all nodes
            embeddings_dict = get_node_embeddings(target_nodes, refiner_config, logger)

            if not embeddings_dict:
                logger.warning("No embeddings obtained, saving graph without changes")
                validate_json(graph, "LearningChunkGraph")

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(graph, f, ensure_ascii=False, indent=2)

                logger.info(f"Saved graph to {output_path} (no changes)")
                return EXIT_SUCCESS

            # Build FAISS index
            faiss_index, node_ids_list = build_similarity_index(
                embeddings_dict, target_nodes, refiner_config, logger
            )

            # Generate candidate pairs
            candidate_pairs = generate_candidate_pairs(
                target_nodes,
                embeddings_dict,
                faiss_index,
                node_ids_list,
                edges_index,
                refiner_config,
                logger,
            )

            if not candidate_pairs:
                logger.info("No candidate pairs found above similarity threshold")
            else:
                logger.info(
                    f"Found {len(candidate_pairs)} nodes with candidates for analysis"
                )

                # Analysis through LLM
                try:
                    new_edges = analyze_candidate_pairs(
                        candidate_pairs, graph, refiner_config, logger
                    )

                    # Update graph with new edges
                    if new_edges:
                        logger.info(
                            f"LLM analysis returned {len(new_edges)} edges to process"
                        )
                        update_stats = update_graph_with_new_edges(
                            graph, new_edges, logger
                        )
                        logger.info(
                            f"Graph updated: {update_stats['added']} added, "
                            f"{update_stats['updated']} updated, "
                            f"{update_stats['replaced']} replaced",
                            extra={"event": "graph_updated", "stats": update_stats},
                        )
                    else:
                        logger.info("No new edges found by LLM analysis")

                except Exception as e:
                    logger.error(f"LLM analysis failed: {e}")
                    if "RateLimitError" in str(e):
                        return EXIT_API_LIMIT_ERROR
                    return EXIT_RUNTIME_ERROR

        except Exception as e:
            logger.error(f"Error during candidate generation: {e}")
            return EXIT_RUNTIME_ERROR

        # Temporarily: save graph without changes
        # TODO: After LLM analysis, updated graph will be here

        # Final validation
        try:
            validate_json(graph, "LearningChunkGraph")
            validate_graph_invariants(graph)
        except Exception as e:
            logger.error(f"Graph validation failed: {e}")
            # Save problematic graph for analysis
            failed_path = Path("data/out/LearningChunkGraph_refiner_failed.json")
            with open(failed_path, "w", encoding="utf-8") as f:
                json.dump(graph, f, ensure_ascii=False, indent=2)
            logger.error(f"Saved failed graph to {failed_path}")
            return EXIT_RUNTIME_ERROR

        # Save result
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(graph, f, ensure_ascii=False, indent=2)
            logger.info(f"Successfully saved refined graph to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save output: {e}")
            return EXIT_IO_ERROR

        # Log completion
        logger.info(
            "Refiner completed successfully",
            extra={
                "event": "refiner_complete",
                "edges_added": (
                    update_stats.get("added", 0) if "update_stats" in locals() else 0
                ),
                "edges_updated": (
                    update_stats.get("updated", 0) if "update_stats" in locals() else 0
                ),
                "edges_replaced": (
                    update_stats.get("replaced", 0) if "update_stats" in locals() else 0
                ),
            },
        )

        return EXIT_SUCCESS

    except Exception as e:
        if logger:
            logger.error(
                f"Unexpected error: {e}",
                exc_info=True,
                extra={"event": "refiner_error", "error": str(e)},
            )
        else:
            # If logger not initialized, use print
            print(f"ERROR: Unexpected error: {e}")
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    sys.exit(main())

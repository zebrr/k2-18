#!/usr/bin/env python3
"""
dedup.py - Utility for removing duplicate nodes from knowledge graph

Removes duplicate nodes of type Chunk and Assessment that appeared due to overlap,
incorrect splitting or different Chunk boundaries. Uses vector embeddings
and FAISS for finding similar nodes.

Usage:
    python -m src.dedup
"""

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to PYTHONPATH for correct imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import faiss
import numpy as np

# Import project utilities
from src.utils.config import load_config

# Set UTF-8 encoding for Windows console
from src.utils.console_encoding import setup_console_encoding
from src.utils.exit_codes import (
    EXIT_API_LIMIT_ERROR,
    EXIT_CONFIG_ERROR,
    EXIT_INPUT_ERROR,
    EXIT_RUNTIME_ERROR,
    EXIT_SUCCESS,
)
from src.utils.llm_embeddings import get_embeddings
from src.utils.validation import validate_json

setup_console_encoding()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class UnionFind:
    """Union-Find data structure for clustering duplicates"""

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x: str) -> str:
        """Find root of element with path compression"""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str):
        """Union two elements"""
        px, py = self.find(x), self.find(y)
        if px == py:
            return

        # Union by rank
        if self.rank[px] < self.rank[py]:
            self.parent[px] = py
        elif self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[py] = px
            self.rank[px] += 1

    def get_clusters(self) -> Dict[str, List[str]]:
        """Get all clusters"""
        clusters = {}
        for x in self.parent:
            root = self.find(x)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(x)
        return clusters


def extract_global_position(node_id: str) -> int:
    """
    Extract global position from Chunk or Assessment node ID.

    Note: This function is only called for nodes that passed through
    filter_nodes_for_dedup, which guarantees only Chunk and Assessment types.
    Concept nodes never reach this function.

    ID formats:
    - Chunk: {slug}:c:{position}
    - Assessment: {slug}:q:{position}:{index}

    Args:
        node_id: Node identifier (must be Chunk or Assessment)

    Returns:
        Global position in tokens

    Raises:
        ValueError: If ID format is unexpected (indicates a bug)
    """
    parts = node_id.split(":")

    if len(parts) >= 3 and parts[1] in ["c", "q"]:
        try:
            return int(parts[2])
        except ValueError as e:
            raise ValueError(f"Cannot parse position from ID: {node_id}") from e

    # This should never happen for filtered nodes - indicates a bug
    raise ValueError(f"Unexpected node ID format in dedup: {node_id}")


def filter_nodes_for_dedup(nodes: List[Dict]) -> List[Dict]:
    """
    Filter nodes for deduplication

    Process only nodes of type Chunk and Assessment with non-empty text
    """
    filtered = []
    for node in nodes:
        if node.get("type") in ["Chunk", "Assessment"]:
            text = node.get("text")
            if text is not None and text.strip():
                filtered.append(node)

    logger.info(f"Filtered {len(filtered)} nodes out of {len(nodes)} for deduplication")
    return filtered


def build_faiss_index(embeddings: np.ndarray, config: Dict) -> faiss.IndexHNSWFlat:
    """
    Create FAISS index for fast similar vector search
    """
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(
        dim,
        config["faiss_M"],
        faiss.METRIC_INNER_PRODUCT,  # Use inner product for normalized vectors
    )
    index.hnsw.efConstruction = config["faiss_efC"]
    index.add(embeddings)

    logger.debug(f"Created FAISS index with {embeddings.shape[0]} vectors")
    return index


def find_duplicates(
    nodes: List[Dict], embeddings: np.ndarray, index: faiss.IndexHNSWFlat, config: Dict
) -> List[Tuple[str, str, float]]:
    """
    Search for duplicate candidates through FAISS

    Returns:
        List of tuples (master_id, duplicate_id, similarity)
    """
    duplicates = []
    k_neighbors = min(
        config["k_neighbors"] + 1, len(nodes)
    )  # +1 because the node itself is also in results

    # Search for nearest neighbors for all nodes at once
    similarities, indices = index.search(embeddings, k_neighbors)

    for i, node in enumerate(nodes):
        node_text_len = len(node["text"])

        for j in range(1, k_neighbors):  # Skip j=0 (the node itself)
            neighbor_idx = indices[i, j]
            if neighbor_idx == -1:  # No more neighbors
                break

            similarity = similarities[i, j]
            if similarity < config["sim_threshold"]:
                continue

            neighbor = nodes[neighbor_idx]
            neighbor_text_len = len(neighbor["text"])

            # Check length ratio
            len_ratio = min(node_text_len, neighbor_text_len) / max(
                node_text_len, neighbor_text_len
            )
            if len_ratio < config["len_ratio_min"]:
                continue

            # Extract global positions from IDs
            node_pos = extract_global_position(node["id"])
            neighbor_pos = extract_global_position(neighbor["id"])

            # Determine master by global position
            if node_pos < neighbor_pos:
                master, duplicate = node, neighbor
            elif node_pos > neighbor_pos:
                master, duplicate = neighbor, node
            else:  # positions are equal
                if node["id"] < neighbor["id"]:
                    master, duplicate = node, neighbor
                else:
                    master, duplicate = neighbor, node

            # Avoid pair duplication
            if i < neighbor_idx:
                duplicates.append((master["id"], duplicate["id"], similarity))

    logger.info(f"Found {len(duplicates)} potential duplicates")
    return duplicates


def cluster_duplicates(duplicates: List[Tuple[str, str, float]]) -> Tuple[Dict[str, str], int]:
    """
    Cluster duplicates using Union-Find

    Returns:
        Tuple of (dedup_map, num_clusters) where:
        - dedup_map: Dictionary {duplicate_id: master_id}
        - num_clusters: Number of clusters formed
    """
    if not duplicates:
        return {}, 0

    uf = UnionFind()

    # Save information about who was master in original pairs
    # In duplicates first element is always master (determined by local_start)
    initial_masters = {}
    for master_id, duplicate_id, _ in duplicates:
        uf.union(master_id, duplicate_id)
        # Remember that duplicate_id should point to master_id
        initial_masters[duplicate_id] = master_id

    # Get clusters
    clusters = uf.get_clusters()
    dedup_map = {}

    # For each cluster determine final master
    for cluster_nodes in clusters.values():
        if len(cluster_nodes) > 1:
            # Find node that was master in original pairs
            # If there are several, take the minimum
            masters_in_cluster = set()
            for node in cluster_nodes:
                if node not in initial_masters:
                    # This node was master in some pair
                    masters_in_cluster.add(node)

            # Choose final master
            if masters_in_cluster:
                master_id = min(masters_in_cluster)
            else:
                # All nodes were duplicates, take minimum
                master_id = min(cluster_nodes)

            # Create mapping for all non-master nodes
            for node_id in cluster_nodes:
                if node_id != master_id:
                    dedup_map[node_id] = master_id

    logger.info(f"Formed {len(clusters)} clusters, {len(dedup_map)} nodes marked as duplicates")
    return dedup_map, len(clusters)


def rewrite_graph(graph: Dict, dedup_map: Dict[str, str]) -> Tuple[Dict, Dict]:
    """
    Rewrite graph replacing duplicate IDs with master IDs
    and removing nodes with empty text.

    Returns:
        Tuple of (new_graph, statistics)
    """
    # Create new graph
    new_graph = {"nodes": [], "edges": []}

    # Filter nodes
    removed_duplicates = 0
    removed_empty = 0
    for node in graph["nodes"]:
        # Skip duplicates
        if node["id"] in dedup_map:
            removed_duplicates += 1
            continue

        # Remove nodes with empty text (only Chunk and Assessment)
        if node.get("type") in ["Chunk", "Assessment"]:
            text = node.get("text", "")
            if not text.strip():
                removed_empty += 1
                continue

        # Add node to new graph
        new_graph["nodes"].append(node)

    logger.info(f"Removed {removed_duplicates} duplicate nodes, {removed_empty} empty nodes")

    # Update edges
    seen_edges = set()  # For removing duplicate edges
    updated_edges_count = 0

    for edge in graph["edges"]:
        # Replace ID with master if it's a duplicate
        source = dedup_map.get(edge["source"], edge["source"])
        target = dedup_map.get(edge["target"], edge["target"])

        # Check that nodes exist (including removed empty ones)
        node_ids = {n["id"] for n in new_graph["nodes"]}
        if source not in node_ids or target not in node_ids:
            logger.debug(f"Dropped dangling edge: {source} -> {target}")
            continue

        # Create key for checking edge duplicates
        edge_key = (source, target, edge["type"])
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        # Update edge if IDs changed
        if source != edge["source"] or target != edge["target"]:
            updated_edges_count += 1

        new_edge = edge.copy()
        new_edge["source"] = source
        new_edge["target"] = target
        new_graph["edges"].append(new_edge)

    logger.info(f"Updated {updated_edges_count} edges, final count: {len(new_graph['edges'])}")

    # Collect statistics
    stats = {
        "nodes_removed_duplicates": removed_duplicates,
        "nodes_removed_empty": removed_empty,
        "nodes_removed_total": removed_duplicates + removed_empty,
        "edges_updated": updated_edges_count,
    }

    return new_graph, stats


def save_dedup_map(dedup_map: Dict[str, str], duplicates: List[Tuple[str, str, float]]):
    """
    Save duplicate mapping to CSV file
    """
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    csv_path = logs_dir / "dedup_map.csv"

    # Create similarity dictionary for fast access
    similarity_map = {(master, dup): sim for master, dup, sim in duplicates}

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["duplicate_id", "master_id", "similarity"])

        for duplicate_id, master_id in sorted(dedup_map.items()):
            # Search for similarity
            sim = similarity_map.get(
                (master_id, duplicate_id),
                similarity_map.get((duplicate_id, master_id), 0.0),
            )
            writer.writerow([duplicate_id, master_id, f"{sim:.4f}"])

    logger.info(f"Saved duplicate mapping to {csv_path}")


def update_metadata(
    existing_meta: Optional[Dict],
    config: Dict,
    statistics: Dict,
    processing_time: float,
) -> Dict:
    """
    Update or create metadata with deduplication information.

    Args:
        existing_meta: Existing metadata from input graph (if any)
        config: Deduplication configuration
        statistics: Deduplication statistics
        processing_time: Time spent on deduplication

    Returns:
        Updated metadata dictionary
    """
    # Start with existing metadata or create new
    metadata = existing_meta.copy() if existing_meta else {}

    # Add deduplication section
    metadata["deduplication"] = {
        "performed_at": datetime.now().isoformat(),
        "config": {
            "similarity_threshold": config.get("sim_threshold", 0.97),
            "length_ratio_threshold": config.get("len_ratio_min", 0.8),
            "top_k": config.get("k_neighbors", 5),
            "min_similarity": config.get("sim_threshold", 0.97),
            "model": config.get("embedding_model", "text-embedding-3-small"),
        },
        "statistics": {
            "nodes_analyzed": statistics.get("nodes_analyzed", 0),
            "embeddings_created": statistics.get("embeddings_created", 0),
            "potential_duplicates": statistics.get("potential_duplicates", 0),
            "clusters_formed": statistics.get("clusters_formed", 0),
            "nodes_removed": {
                "duplicates": statistics.get("nodes_removed_duplicates", 0),
                "empty": statistics.get("nodes_removed_empty", 0),
                "total": statistics.get("nodes_removed_total", 0),
            },
            "edges_updated": statistics.get("edges_updated", 0),
            "processing_time_seconds": processing_time,
        },
        "before_after": {
            "nodes_before": statistics.get("nodes_before", 0),
            "nodes_after": statistics.get("nodes_after", 0),
            "edges_before": statistics.get("edges_before", 0),
            "edges_after": statistics.get("edges_after", 0),
        },
        "quality_issues": {
            "duplicate_nodes_removed": statistics.get("nodes_removed_duplicates", 0),
            "empty_nodes_removed": statistics.get("nodes_removed_empty", 0),
            "total_nodes_removed": statistics.get("nodes_removed_total", 0),
        },
    }

    return metadata


def main():
    """Main function"""
    start_time = time.time()

    try:
        # Load configuration
        config = load_config()
        dedup_config = config["dedup"]

    except Exception as e:
        logger.error(f"Configuration loading error: {e}")
        return EXIT_CONFIG_ERROR

    # File paths
    input_path = Path("data/out/LearningChunkGraph_raw.json")
    output_path = Path("data/out/LearningChunkGraph_dedup.json")

    # Check input file
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return EXIT_INPUT_ERROR

    try:
        # Load graph
        logger.info("Loading knowledge graph...")
        with open(input_path, encoding="utf-8") as f:
            graph_data = json.load(f)

        # Extract graph structure (handle both old and new format with _meta)
        if "nodes" in graph_data and "edges" in graph_data:
            graph = {"nodes": graph_data["nodes"], "edges": graph_data["edges"]}
            # Preserve metadata if present
            metadata = graph_data.get("_meta")
        else:
            logger.error("Invalid graph structure: missing nodes or edges")
            return EXIT_INPUT_ERROR

        # Initialize statistics
        statistics = {
            "nodes_before": len(graph["nodes"]),
            "edges_before": len(graph["edges"]),
        }

        # Validate input graph
        try:
            validate_json(graph, "LearningChunkGraph")
        except Exception as e:
            logger.error(f"Input graph does not match schema: {e}")
            return EXIT_INPUT_ERROR

        # Filter nodes for deduplication
        nodes_to_dedup = filter_nodes_for_dedup(graph["nodes"])
        statistics["nodes_analyzed"] = len(nodes_to_dedup)

        if len(nodes_to_dedup) < 2:
            logger.info("Not enough nodes for deduplication, copying graph without changes")

            # Update statistics for no deduplication case
            statistics.update(
                {
                    "nodes_after": len(graph["nodes"]),
                    "edges_after": len(graph["edges"]),
                    "embeddings_created": 0,
                    "potential_duplicates": 0,
                    "clusters_formed": 0,
                    "nodes_removed_duplicates": 0,
                    "nodes_removed_empty": 0,
                    "nodes_removed_total": 0,
                    "edges_updated": 0,
                }
            )

            # Update metadata
            elapsed_time = time.time() - start_time
            metadata = update_metadata(metadata, dedup_config, statistics, elapsed_time)

            # Prepare output data with metadata
            output_data = {"_meta": metadata, **graph}

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            # Save empty dedup_map
            save_dedup_map({}, [])
            return EXIT_SUCCESS

        # Sort nodes by global position from ID for determinism
        nodes_to_dedup.sort(key=lambda n: (extract_global_position(n["id"]), n["id"]))

        # Get embeddings
        logger.info(f"Getting embeddings for {len(nodes_to_dedup)} nodes...")
        texts = [node["text"] for node in nodes_to_dedup]
        statistics["embeddings_created"] = len(texts)

        try:
            embeddings = get_embeddings(texts, dedup_config)
        except Exception as e:
            if "rate" in str(e).lower() or "limit" in str(e).lower():
                logger.error(f"API limit exceeded: {e}")
                return EXIT_API_LIMIT_ERROR
            else:
                logger.error(f"Error getting embeddings: {e}")
                return EXIT_RUNTIME_ERROR

        # Build FAISS index
        logger.info("Building FAISS index...")
        index = build_faiss_index(embeddings, dedup_config)

        # Search for duplicates
        logger.info("Searching for duplicates...")
        duplicates = find_duplicates(nodes_to_dedup, embeddings, index, dedup_config)
        statistics["potential_duplicates"] = len(duplicates)

        if not duplicates:
            logger.info("No duplicates found, removing only empty nodes")

            # Still need to check for empty nodes
            dedup_map = {}
            new_graph, rewrite_stats = rewrite_graph(graph, dedup_map)

            # Update statistics
            statistics.update(
                {
                    "nodes_after": len(new_graph["nodes"]),
                    "edges_after": len(new_graph["edges"]),
                    "clusters_formed": 0,
                    **rewrite_stats,
                }
            )

            # Update metadata
            elapsed_time = time.time() - start_time
            metadata = update_metadata(metadata, dedup_config, statistics, elapsed_time)

            # Prepare output data with metadata
            output_data = {"_meta": metadata, **new_graph}

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            # Save empty dedup_map
            save_dedup_map({}, [])

            logger.info(f"Nodes were: {len(graph['nodes'])}, became: {len(new_graph['nodes'])}")
            logger.info(f"Edges were: {len(graph['edges'])}, became: {len(new_graph['edges'])}")

            return EXIT_SUCCESS

        # Cluster duplicates
        logger.info("Clustering duplicates...")
        dedup_map, num_clusters = cluster_duplicates(duplicates)
        statistics["clusters_formed"] = num_clusters

        # Rewrite graph
        logger.info("Rewriting graph...")
        new_graph, rewrite_stats = rewrite_graph(graph, dedup_map)

        # Update statistics
        statistics.update(
            {
                "nodes_after": len(new_graph["nodes"]),
                "edges_after": len(new_graph["edges"]),
                **rewrite_stats,
            }
        )

        # Validate output graph
        try:
            validate_json(new_graph, "LearningChunkGraph")
        except Exception as e:
            logger.error(f"Output graph does not match schema: {e}")
            return EXIT_RUNTIME_ERROR

        # Save results
        logger.info("Saving results...")

        # Update metadata with deduplication information
        elapsed_time = time.time() - start_time
        metadata = update_metadata(metadata, dedup_config, statistics, elapsed_time)

        # Prepare output data with metadata
        output_data = {"_meta": metadata, **new_graph}

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        # Save duplicate mapping
        save_dedup_map(dedup_map, duplicates)

        logger.info(f"Deduplication completed in {elapsed_time:.2f} seconds")
        logger.info(f"Nodes were: {len(graph['nodes'])}, became: {len(new_graph['nodes'])}")
        logger.info(f"Edges were: {len(graph['edges'])}, became: {len(new_graph['edges'])}")

        return EXIT_SUCCESS

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    sys.exit(main())

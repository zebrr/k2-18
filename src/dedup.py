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
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to PYTHONPATH for correct imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import faiss
import numpy as np

# Import project utilities
from src.utils.config import load_config
# Set UTF-8 encoding for Windows console
from src.utils.console_encoding import setup_console_encoding
from src.utils.exit_codes import (EXIT_API_LIMIT_ERROR, EXIT_CONFIG_ERROR,
                                  EXIT_INPUT_ERROR, EXIT_RUNTIME_ERROR,
                                  EXIT_SUCCESS)
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

            # Determine master and duplicate by local_start, then by id
            if node["local_start"] < neighbor["local_start"]:
                master, duplicate = node, neighbor
            elif node["local_start"] > neighbor["local_start"]:
                master, duplicate = neighbor, node
            else:  # local_start are equal
                if node["id"] < neighbor["id"]:
                    master, duplicate = node, neighbor
                else:
                    master, duplicate = neighbor, node

            # Avoid pair duplication
            if i < neighbor_idx:
                duplicates.append((master["id"], duplicate["id"], similarity))

    logger.info(f"Found {len(duplicates)} potential duplicates")
    return duplicates


def cluster_duplicates(duplicates: List[Tuple[str, str, float]]) -> Dict[str, str]:
    """
    Cluster duplicates using Union-Find

    Returns:
        Dictionary {duplicate_id: master_id}
    """
    if not duplicates:
        return {}

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

    logger.info(
        f"Formed {len(clusters)} clusters, {len(dedup_map)} nodes marked as duplicates"
    )
    return dedup_map


def rewrite_graph(graph: Dict, dedup_map: Dict[str, str]) -> Dict:
    """
    Rewrite graph replacing duplicate IDs with master IDs
    and removing nodes with empty text
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

    logger.info(
        f"Removed {removed_duplicates} duplicate nodes, {removed_empty} empty nodes"
    )

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

    logger.info(
        f"Updated {updated_edges_count} edges, final count: {len(new_graph['edges'])}"
    )

    return new_graph


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
        with open(input_path, "r", encoding="utf-8") as f:
            graph = json.load(f)

        # Validate input graph
        try:
            validate_json(graph, "LearningChunkGraph")
        except Exception as e:
            logger.error(f"Input graph does not match schema: {e}")
            return EXIT_INPUT_ERROR

        # Filter nodes for deduplication
        nodes_to_dedup = filter_nodes_for_dedup(graph["nodes"])

        if len(nodes_to_dedup) < 2:
            logger.info(
                "Not enough nodes for deduplication, copying graph without changes"
            )
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(graph, f, ensure_ascii=False, indent=2)
            # Save empty dedup_map
            save_dedup_map({}, [])
            return EXIT_SUCCESS

        # Sort nodes by local_start for determinism
        nodes_to_dedup.sort(key=lambda n: (n.get("local_start", 0), n["id"]))

        # Get embeddings
        logger.info(f"Getting embeddings for {len(nodes_to_dedup)} nodes...")
        texts = [node["text"] for node in nodes_to_dedup]

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

        if not duplicates:
            logger.info("No duplicates found, copying graph without changes")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(graph, f, ensure_ascii=False, indent=2)
            # Save empty dedup_map
            save_dedup_map({}, [])
            return EXIT_SUCCESS

        # Cluster duplicates
        logger.info("Clustering duplicates...")
        dedup_map = cluster_duplicates(duplicates)

        # Rewrite graph
        logger.info("Rewriting graph...")
        new_graph = rewrite_graph(graph, dedup_map)

        # Validate output graph
        try:
            validate_json(new_graph, "LearningChunkGraph")
        except Exception as e:
            logger.error(f"Output graph does not match schema: {e}")
            return EXIT_RUNTIME_ERROR

        # Save results
        logger.info("Saving results...")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(new_graph, f, ensure_ascii=False, indent=2)

        # Save duplicate mapping
        save_dedup_map(dedup_map, duplicates)

        elapsed_time = time.time() - start_time
        logger.info(f"Deduplication completed in {elapsed_time:.2f} seconds")
        logger.info(
            f"Nodes were: {len(graph['nodes'])}, became: {len(new_graph['nodes'])}"
        )
        logger.info(
            f"Edges were: {len(graph['edges'])}, became: {len(new_graph['edges'])}"
        )

        return EXIT_SUCCESS

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    sys.exit(main())

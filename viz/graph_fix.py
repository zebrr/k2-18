#!/usr/bin/env python3
"""
graph_fix.py - Mark LLM-generated content in enriched knowledge graph.

This utility adds [added_by=LLM] markers to LLM-generated fields and updates
Concept nodes' text field with proper term formatting from ConceptDictionary.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    ValidationError,
    validate_json,
)


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
    logger.info("Starting graph_fix utility")

    return logger


def load_input_files(
    data_dir: Path, logger: logging.Logger
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load ConceptDictionary and LearningChunkGraph files.

    Args:
        data_dir: Directory containing input files
        logger: Logger instance

    Returns:
        Tuple of (concepts_data, graph_data)

    Raises:
        SystemExit: If files not found or invalid
    """
    concepts_file = data_dir / "ConceptDictionary_wow.json"
    graph_file = data_dir / "LearningChunkGraph_wow.json"

    # Check files exist
    if not concepts_file.exists():
        logger.error(f"Concepts file not found: {concepts_file}")
        log_exit(logger, EXIT_INPUT_ERROR, "Missing ConceptDictionary_wow.json")
        sys.exit(EXIT_INPUT_ERROR)

    if not graph_file.exists():
        logger.error(f"Graph file not found: {graph_file}")
        log_exit(logger, EXIT_INPUT_ERROR, "Missing LearningChunkGraph_wow.json")
        sys.exit(EXIT_INPUT_ERROR)

    logger.info(f"Loading concepts from: {concepts_file}")
    logger.info(f"Loading graph from: {graph_file}")

    try:
        # Load concepts
        with open(concepts_file, encoding="utf-8") as f:
            concepts_data = json.load(f)

        # Load graph
        with open(graph_file, encoding="utf-8") as f:
            graph_data = json.load(f)

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")
        log_exit(logger, EXIT_INPUT_ERROR, f"JSON parse error: {e}")
        sys.exit(EXIT_INPUT_ERROR)
    except Exception as e:
        logger.error(f"Failed to load files: {e}")
        log_exit(logger, EXIT_IO_ERROR, f"File read error: {e}")
        sys.exit(EXIT_IO_ERROR)

    # Validate structure
    logger.info("Validating data structure")
    try:
        validate_json(concepts_data, "ConceptDictionary")
        validate_json(graph_data, "LearningChunkGraph")
    except ValidationError as e:
        logger.error(f"Validation failed: {e}")
        log_exit(logger, EXIT_INPUT_ERROR, f"Schema validation error: {e}")
        sys.exit(EXIT_INPUT_ERROR)

    # Log statistics
    num_concepts = len(concepts_data.get("concepts", []))
    num_nodes = len(graph_data.get("nodes", []))
    num_edges = len(graph_data.get("edges", []))

    logger.info(f"Loaded {num_concepts} concepts")
    logger.info(f"Loaded graph: {num_nodes} nodes, {num_edges} edges")

    return concepts_data, graph_data


def process_chunk_assessment_definitions(
    nodes: List[Dict[str, Any]], dry_run: bool, logger: logging.Logger
) -> Tuple[int, int, List[str]]:
    """Process definitions in Chunk and Assessment nodes.

    Args:
        nodes: List of graph nodes
        dry_run: If True, don't modify nodes
        logger: Logger instance

    Returns:
        Tuple of (chunks_marked, assessments_marked, examples)
    """
    chunks_marked = 0
    assessments_marked = 0
    examples = []

    for node in nodes:
        node_type = node.get("type")
        if node_type not in ["Chunk", "Assessment"]:
            continue

        definition = node.get("definition")
        if not definition or definition.strip() == "":
            continue

        # Check if already marked
        if definition.startswith("[added_by=LLM]"):
            continue

        # Mark the definition
        new_definition = f"[added_by=LLM] {definition}"

        if node_type == "Chunk":
            chunks_marked += 1
        else:
            assessments_marked += 1

        # Collect examples for dry-run output
        if len(examples) < 5:
            examples.append(
                f"[{node_type}] {node['id']}: "
                f'"{definition[:50]}..." → '
                f'"[added_by=LLM] {definition[:50]}..."'
            )

        # Apply change if not dry-run
        if not dry_run:
            node["definition"] = new_definition

    return chunks_marked, assessments_marked, examples


def process_concept_text(
    nodes: List[Dict[str, Any]],
    concepts_data: Dict[str, Any],
    dry_run: bool,
    logger: logging.Logger,
) -> Tuple[int, List[str]]:
    """Update text field in Concept nodes from ConceptDictionary.

    Args:
        nodes: List of graph nodes
        concepts_data: ConceptDictionary data
        dry_run: If True, don't modify nodes
        logger: Logger instance

    Returns:
        Tuple of (concepts_updated, examples)
    """
    # Build concept lookup map
    concept_map = {}
    for concept in concepts_data.get("concepts", []):
        concept_id = concept.get("concept_id")
        if concept_id:
            concept_map[concept_id] = concept

    concepts_updated = 0
    examples = []

    for node in nodes:
        if node.get("type") != "Concept":
            continue

        node_id = node.get("id")
        concept = concept_map.get(node_id)

        if not concept:
            logger.warning(f"Concept not found in dictionary: {node_id}")
            print(f"WARNING: Concept not found in dictionary: {node_id}")
            continue

        # Build new text from term
        term = concept.get("term", {})
        primary = term.get("primary", "")
        aliases = term.get("aliases", [])

        if aliases:
            new_text = f"{primary} ({', '.join(aliases)})"
        else:
            new_text = primary

        old_text = node.get("text", "")

        # Only count if actually changed
        if new_text != old_text:
            concepts_updated += 1

            # Collect examples
            if len(examples) < 5:
                examples.append(
                    f"[Concept] {node_id}: " f'"{old_text[:50]}..." → ' f'"{new_text[:50]}..."'
                )

            # Apply change if not dry-run
            if not dry_run:
                node["text"] = new_text

    return concepts_updated, examples


def process_edge_conditions(
    edges: List[Dict[str, Any]], dry_run: bool, logger: logging.Logger
) -> Tuple[int, List[str]]:
    """Process conditions field in edges.

    Args:
        edges: List of graph edges
        dry_run: If True, don't modify edges
        logger: Logger instance

    Returns:
        Tuple of (edges_marked, examples)
    """
    edges_marked = 0
    examples = []

    skip_markers = ["added_by=", "fixed_by=", "auto_generated"]

    for edge in edges:
        conditions = edge.get("conditions")
        if not conditions or conditions.strip() == "":
            continue

        # Check for skip markers anywhere in the string
        should_skip = any(marker in conditions for marker in skip_markers)
        if should_skip:
            continue

        # Check if already marked
        if conditions.startswith("[added_by=LLM]"):
            continue

        # Mark the conditions
        new_conditions = f"[added_by=LLM] {conditions}"
        edges_marked += 1

        # Collect examples
        if len(examples) < 5:
            source = edge.get("source", "")
            target = edge.get("target", "")
            edge_type = edge.get("type", "")
            examples.append(
                f"[Edge] {source[:20]}→{target[:20]} ({edge_type}): "
                f'"{conditions[:50]}..." → '
                f'"[added_by=LLM] {conditions[:50]}..."'
            )

        # Apply change if not dry-run
        if not dry_run:
            edge["conditions"] = new_conditions

    return edges_marked, examples


def update_metadata(
    graph_data: Dict[str, Any], stats: Dict[str, int], logger: logging.Logger
) -> None:
    """Update graph metadata with fix statistics.

    Args:
        graph_data: Graph data to update
        stats: Statistics dictionary
        logger: Logger instance
    """
    if "_meta" not in graph_data:
        graph_data["_meta"] = {}

    graph_data["_meta"]["graph_fix_applied"] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "chunks_definitions_marked": stats["chunks_marked"],
        "assessments_definitions_marked": stats["assessments_marked"],
        "concepts_text_updated": stats["concepts_updated"],
        "edges_conditions_marked": stats["edges_marked"],
    }

    logger.info("Updated metadata with fix statistics")


def save_graph(graph_data: Dict[str, Any], output_file: Path, logger: logging.Logger) -> None:
    """Save modified graph to file.

    Args:
        graph_data: Graph data to save
        output_file: Output file path
        logger: Logger instance
    """
    try:
        # Validate before saving
        validate_json(graph_data, "LearningChunkGraph")

        # Save with proper formatting
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Graph saved to: {output_file}")

    except ValidationError as e:
        logger.error(f"Validation failed after modifications: {e}")
        log_exit(logger, EXIT_RUNTIME_ERROR, f"Modified graph invalid: {e}")
        sys.exit(EXIT_RUNTIME_ERROR)
    except Exception as e:
        logger.error(f"Failed to save graph: {e}")
        log_exit(logger, EXIT_IO_ERROR, f"Save error: {e}")
        sys.exit(EXIT_IO_ERROR)


def print_dry_run_summary(examples: Dict[str, List[str]], stats: Dict[str, int]) -> None:
    """Print dry-run summary to console.

    Args:
        examples: Dictionary of example changes
        stats: Statistics dictionary
    """
    print("\n" + "=" * 80)
    print("DRY-RUN MODE - No files were modified")
    print("=" * 80)

    # Show examples
    if examples["definitions"]:
        print("\nDefinition changes (showing first 5):")
        for example in examples["definitions"]:
            print(f"  {example}")

    if examples["concepts"]:
        print("\nConcept text updates (showing first 5):")
        for example in examples["concepts"]:
            print(f"  {example}")

    if examples["conditions"]:
        print("\nEdge condition changes (showing first 5):")
        for example in examples["conditions"]:
            print(f"  {example}")

    # Show summary
    print("\nSummary of changes that would be made:")
    print(f"  - Chunk definitions marked: {stats['chunks_marked']}")
    print(f"  - Assessment definitions marked: {stats['assessments_marked']}")
    print(f"  - Concept texts updated: {stats['concepts_updated']}")
    print(f"  - Edge conditions marked: {stats['edges_marked']}")
    print(f"  - Total changes: {sum(stats.values())}")
    print("=" * 80)


def main():
    """Main entry point."""
    # Setup console encoding
    setup_console_encoding()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Mark LLM-generated content in enriched knowledge graph"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show changes without modifying files"
    )
    args = parser.parse_args()

    # Setup paths
    viz_dir = Path(__file__).parent
    log_file = viz_dir / "logs" / "graph_fix.log"
    data_dir = viz_dir / "data" / "out"

    # Setup logging
    logger = setup_logging(log_file)

    if args.dry_run:
        logger.info("Running in DRY-RUN mode")
        print("\n[DRY-RUN MODE] Analyzing changes without modifying files...")

    try:
        # Load configuration (not used but following pattern)
        config_path = viz_dir / "config.toml"
        load_config(config_path)  # Load config to validate it exists
        logger.info("Configuration loaded")
    except (ConfigValidationError, FileNotFoundError) as e:
        logger.error(f"Failed to load config: {e}")
        log_exit(logger, EXIT_CONFIG_ERROR, str(e))
        sys.exit(EXIT_CONFIG_ERROR)

    # Load input files
    concepts_data, graph_data = load_input_files(data_dir, logger)

    # Process nodes and edges
    logger.info("Processing graph nodes and edges")

    # Process Chunk/Assessment definitions
    chunks_marked, assessments_marked, def_examples = process_chunk_assessment_definitions(
        graph_data["nodes"], args.dry_run, logger
    )

    # Process Concept text fields
    concepts_updated, concept_examples = process_concept_text(
        graph_data["nodes"], concepts_data, args.dry_run, logger
    )

    # Process edge conditions
    edges_marked, edge_examples = process_edge_conditions(graph_data["edges"], args.dry_run, logger)

    # Collect statistics
    stats = {
        "chunks_marked": chunks_marked,
        "assessments_marked": assessments_marked,
        "concepts_updated": concepts_updated,
        "edges_marked": edges_marked,
    }

    # Log statistics
    logger.info(f"Chunks definitions marked: {chunks_marked}")
    logger.info(f"Assessments definitions marked: {assessments_marked}")
    logger.info(f"Concepts text updated: {concepts_updated}")
    logger.info(f"Edges conditions marked: {edges_marked}")
    logger.info(f"Total changes: {sum(stats.values())}")

    if args.dry_run:
        # Show dry-run summary
        examples = {
            "definitions": def_examples,
            "concepts": concept_examples,
            "conditions": edge_examples,
        }
        print_dry_run_summary(examples, stats)
        logger.info("Dry-run completed successfully")
    else:
        # Update metadata
        update_metadata(graph_data, stats, logger)

        # Save modified graph
        output_file = data_dir / "LearningChunkGraph_wow.json"
        save_graph(graph_data, output_file, logger)

        # Print summary
        print("\n✓ Graph fix completed successfully")
        print(f"  - Chunks definitions marked: {chunks_marked}")
        print(f"  - Assessments definitions marked: {assessments_marked}")
        print(f"  - Concepts text updated: {concepts_updated}")
        print(f"  - Edge conditions marked: {edges_marked}")
        print(f"  - Total changes: {sum(stats.values())}")
        print(f"  - Output saved to: {output_file}")

    logger.info("Graph fix utility completed successfully")
    log_exit(logger, EXIT_SUCCESS)
    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())

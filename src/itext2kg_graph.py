#!/usr/bin/env python3
"""
iText2KG Graph - incremental knowledge graph construction from educational texts.

Utility sequentially processes slices from staging, sends them to LLM
while preserving context through previous_response_id, and builds
LearningChunkGraph using the pre-extracted ConceptDictionary.
"""

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to PYTHONPATH for correct imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config

# Set UTF-8 encoding for Windows console
from src.utils.console_encoding import setup_console_encoding
from src.utils.exit_codes import (
    EXIT_CONFIG_ERROR,
    EXIT_INPUT_ERROR,
    EXIT_IO_ERROR,
    EXIT_RUNTIME_ERROR,
    EXIT_SUCCESS,
)
from src.utils.llm_client import OpenAIClient

setup_console_encoding()

# Constants
CONFIG_PATH = Path(__file__).parent / "config.toml"
PROMPTS_DIR = Path(__file__).parent / "prompts"
SCHEMAS_DIR = Path(__file__).parent / "schemas"
STAGING_DIR = Path(__file__).parent.parent / "data" / "staging"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "out"
LOGS_DIR = Path(__file__).parent.parent / "logs"

GRAPH_EXTRACTION_PROMPT_FILE = "itext2kg_graph_extraction.md"
MAX_REPAIR_ATTEMPTS = 1
DEFAULT_DIFFICULTY = 3  # Default difficulty for Chunk nodes if not provided


@dataclass
class ProcessingStats:
    """Graph processing statistics."""

    total_slices: int = 0
    processed_slices: int = 0
    failed_slices: int = 0
    total_nodes: int = 0
    total_edges: int = 0
    total_tokens_used: int = 0
    start_time: datetime = field(default_factory=lambda: datetime.now())


@dataclass
class SliceData:
    """Single slice data."""

    id: str
    order: int
    source_file: str
    slug: str
    text: str
    slice_token_start: int
    slice_token_end: int


class SliceProcessor:
    """Main class for processing slices and building knowledge graph."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize processor.

        Args:
            config: Configuration from config.toml
        """
        self.config = config["itext2kg"]
        self.full_config = config  # Save full config for accessing slicer settings
        self.llm_client = OpenAIClient(self.config)
        self.logger = self._setup_logger()
        self.stats = ProcessingStats()

        # Load ConceptDictionary
        self.concept_dict = self._load_concept_dictionary()

        # Initialize graph structures
        self.graph_nodes: List[Dict] = []  # All nodes
        self.graph_edges: List[Dict] = []  # All edges
        self.node_ids: Dict[str, int] = {}  # {node_id: index} for duplicate detection

        # Context tracking for incremental processing
        self.previous_response_id: Optional[str] = None

        # API usage tracking
        self.api_usage = {
            "total_requests": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }

        # Load prompt and schema
        self.extraction_prompt = self._load_extraction_prompt()

    def _format_tokens(self, tokens: int) -> str:
        """
        Format token count into readable form.

        Args:
            tokens: Number of tokens

        Returns:
            String like "123", "45.61k", "1.22M"
        """
        if tokens < 1000:
            return str(tokens)
        elif tokens < 1_000_000:
            # Thousands with two decimal places
            return f"{tokens / 1000:.2f}k"
        else:
            # Millions with two decimal places
            return f"{tokens / 1_000_000:.2f}M"

    def _setup_logger(self) -> logging.Logger:
        """Setup logger with file and console output."""
        logger = logging.getLogger("itext2kg_graph")
        logger.setLevel(getattr(logging, self.config["log_level"].upper()))

        # File handler
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = LOGS_DIR / f"itext2kg_graph_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(message)s"))  # JSON Lines format
        logger.addHandler(file_handler)

        # Console handler for errors
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(console_handler)

        return logger

    def _load_concept_dictionary(self) -> Dict[str, Any]:
        """Load ConceptDictionary from output directory."""
        concept_dict_path = OUTPUT_DIR / "ConceptDictionary.json"

        if not concept_dict_path.exists():
            self.logger.error(f"ConceptDictionary not found: {concept_dict_path}")
            print(f"ERROR: ConceptDictionary.json not found at {concept_dict_path}")
            print("Please run itext2kg_concepts.py first to extract concepts.")
            sys.exit(EXIT_INPUT_ERROR)

        try:
            with open(concept_dict_path, encoding="utf-8") as f:
                concept_dict = json.load(f)

            # Validate structure
            if "concepts" not in concept_dict:
                self.logger.error("Invalid ConceptDictionary structure: missing 'concepts' field")
                sys.exit(EXIT_INPUT_ERROR)

            self.logger.info(
                f"Loaded {len(concept_dict['concepts'])} concepts from ConceptDictionary"
            )
            print(f"Loaded {len(concept_dict['concepts'])} concepts from ConceptDictionary.json")

            return concept_dict

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse ConceptDictionary.json: {e}")
            sys.exit(EXIT_INPUT_ERROR)
        except Exception as e:
            self.logger.error(f"Failed to load ConceptDictionary: {e}")
            sys.exit(EXIT_IO_ERROR)

    def _load_extraction_prompt(self) -> str:
        """Load prompt with schema substitution."""
        prompt_file = PROMPTS_DIR / GRAPH_EXTRACTION_PROMPT_FILE

        if not prompt_file.exists():
            self.logger.error(f"Graph extraction prompt not found: {prompt_file}")
            sys.exit(EXIT_CONFIG_ERROR)

        try:
            with open(prompt_file, encoding="utf-8") as f:
                prompt_template = f.read()

            # Load LearningChunkGraph schema
            schema_file = SCHEMAS_DIR / "LearningChunkGraph.schema.json"
            with open(schema_file, encoding="utf-8") as f:
                learning_chunk_schema = f.read()

            # Substitute schema into prompt
            prompt = prompt_template.replace("{learning_chunk_graph_schema}", learning_chunk_schema)

            return prompt

        except Exception as e:
            self.logger.error(f"Failed to load extraction prompt: {e}")
            sys.exit(EXIT_CONFIG_ERROR)

    def _load_slice(self, slice_file: Path) -> SliceData:
        """
        Load single slice from file.

        Args:
            slice_file: Path to slice JSON file

        Returns:
            SliceData object
        """
        try:
            with open(slice_file, encoding="utf-8") as f:
                data = json.load(f)

            return SliceData(
                id=data["id"],
                order=data["order"],
                source_file=data["source_file"],
                slug=data["slug"],
                text=data["text"],
                slice_token_start=data["slice_token_start"],
                slice_token_end=data["slice_token_end"],
            )
        except Exception as e:
            self.logger.error(f"Failed to load slice {slice_file}: {e}")
            raise

    def _format_slice_input(self, slice_data: SliceData) -> str:
        """
        Format slice data with ConceptDictionary for LLM input.

        Args:
            slice_data: Slice data

        Returns:
            JSON string with ConceptDictionary and Slice
        """
        input_obj = {
            "ConceptDictionary": self.concept_dict,  # Full dictionary
            "Slice": {
                "id": slice_data.id,
                "order": slice_data.order,
                "source_file": slice_data.source_file,
                "slug": slice_data.slug,
                "text": slice_data.text,
                "slice_token_start": slice_data.slice_token_start,
                "slice_token_end": slice_data.slice_token_end,
            },
        }

        return json.dumps(input_obj, ensure_ascii=False)

    def validate_node_positions(
        self, nodes: List[Dict], slice_token_start: int
    ) -> Optional[List[Dict]]:
        """
        DEPRECATED: Position validation is no longer needed after ID post-processing.
        Kept for backward compatibility but always returns nodes as valid.

        Args:
            nodes: List of nodes to validate
            slice_token_start: Start token position of current slice

        Returns:
            Always returns nodes (validation is skipped)
        """
        # Position validation is no longer needed since we fix IDs in post-processing
        return nodes

    def _validate_node_positions_legacy(
        self, nodes: List[Dict], slice_token_start: int
    ) -> Optional[List[Dict]]:
        """
        Legacy validation method - kept for reference but not used.
        The original validation logic that checked node_position calculations.
        """
        issues = []

        for node in nodes:
            if node.get("type") in ["Chunk", "Assessment"]:
                # Check required fields exist
                if "node_offset" not in node or "node_position" not in node:
                    issues.append(
                        {
                            "type": "missing_fields",
                            "node_id": node.get("id", "unknown"),
                            "error": "Missing node_offset or node_position",
                        }
                    )
                    continue

                node_offset = node.get("node_offset", 0)
                node_position = node.get("node_position", 0)
                node_id = node.get("id", "")

                # Check 1: Math consistency
                expected_position = slice_token_start + node_offset
                if node_position != expected_position:
                    issues.append(
                        {
                            "type": "math_error",
                            "node_id": node_id,
                            "stated_offset": node_offset,
                            "stated_position": node_position,
                            "expected_position": expected_position,
                            "error": f"Math error: {slice_token_start} + {node_offset} should be {expected_position}, not {node_position}",
                        }
                    )

                # Check 2: Position must be >= slice start
                if node_position < slice_token_start:
                    issues.append(
                        {
                            "type": "invalid_position",
                            "node_id": node_id,
                            "node_position": node_position,
                            "error": f"Position {node_position} is before slice start {slice_token_start}",
                        }
                    )

                # Check 3: ID consistency with stated position
                id_parts = node_id.split(":")
                if node["type"] == "Chunk" and len(id_parts) >= 3:
                    try:
                        id_position = int(id_parts[-1])
                        if id_position != node_position:
                            issues.append(
                                {
                                    "type": "id_mismatch",
                                    "node_id": node_id,
                                    "id_position": id_position,
                                    "stated_position": node_position,
                                    "error": f"ID contains {id_position} but node_position is {node_position}",
                                }
                            )
                    except ValueError:
                        pass  # ID format issue, handled elsewhere
                elif node["type"] == "Assessment" and len(id_parts) >= 4:
                    try:
                        id_position = int(id_parts[-2])
                        if id_position != node_position:
                            issues.append(
                                {
                                    "type": "id_mismatch",
                                    "node_id": node_id,
                                    "id_position": id_position,
                                    "stated_position": node_position,
                                    "error": f"ID contains {id_position} but node_position is {node_position}",
                                }
                            )
                    except ValueError:
                        pass

        if issues:
            self.logger.error(f"Found {len(issues)} node position validation issues:")
            for issue in issues:
                self.logger.error(f"  [{issue['type']}] {issue['error']}")
            return None  # Signal need for repair

        # Log successful validation at DEBUG level
        validated_count = len([n for n in nodes if n.get("type") in ["Chunk", "Assessment"]])
        if validated_count > 0:
            self.logger.debug(f"Validated {validated_count} nodes successfully")

        return nodes  # All good

    def _process_llm_response(
        self, response_text: str, slice_id: str
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Parse and validate LLM response with HTML attribute cleanup.

        Args:
            response_text: Raw LLM response
            slice_id: Current slice ID for logging

        Returns:
            Tuple of (success, parsed_data)
        """
        try:
            # Remove markdown fences if present
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                # Remove first line (```json or similar)
                if len(lines) > 1:
                    lines = lines[1:]
                # Remove last line if it's closing fence
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                cleaned = "\n".join(lines)

            # HTML attribute cleanup (fix common LLM issues)
            attributes = [
                "href",
                "src",
                "target",
                "action",
                "name",
                "frameborder",
                "width",
                "height",
                "align",
            ]
            for attr in attributes:
                # Fix patterns like href='\"url\"' → href="url"
                cleaned = re.sub(f'{attr}=[\'"]\\\\"([^"]*)\\\\"[\'"]', f'{attr}="\\1"', cleaned)
                # Fix patterns like src="'url'" → src="url"
                cleaned = re.sub(f"{attr}=\"'([^']*)'\"", f'{attr}="\\1"', cleaned)

            # Parse JSON
            parsed = json.loads(cleaned)

            # Check required structure
            if "chunk_graph_patch" not in parsed:
                self.logger.error(f"Missing 'chunk_graph_patch' in response for {slice_id}")
                return False, None

            patch = parsed["chunk_graph_patch"]
            if "nodes" not in patch or "edges" not in patch:
                self.logger.error(f"Invalid patch structure for {slice_id}: missing nodes or edges")
                return False, None

            # Validate against schema (basic check)
            if not isinstance(patch["nodes"], list) or not isinstance(patch["edges"], list):
                self.logger.error(
                    f"Invalid patch types for {slice_id}: nodes and edges must be lists"
                )
                return False, None

            return True, parsed

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error for {slice_id}: {e}")
            self.logger.debug(f"Failed to parse response: {response_text[:500]}...")
            return False, None
        except Exception as e:
            self.logger.error(f"Unexpected error processing response for {slice_id}: {e}")
            return False, None

    def _process_chunk_nodes(self, new_nodes: List[Dict]) -> List[Dict]:
        """
        Process Chunk and Assessment nodes with duplicate checking.
        For Concept nodes, add them as-is (dedup.py will handle later).

        Args:
            new_nodes: List of nodes from LLM response

        Returns:
            List of nodes to add to graph
        """
        nodes_to_add = []

        for node in new_nodes:
            node_id = node.get("id", "")
            node_type = node.get("type", "")

            if node_type in ["Chunk", "Assessment"]:
                if node_id in self.node_ids:
                    # Node already exists
                    existing_idx = self.node_ids[node_id]
                    existing_node = self.graph_nodes[existing_idx]

                    if node_type == "Chunk":
                        # For Chunks: keep longer text version
                        existing_text = existing_node.get("text", "")
                        new_text = node.get("text", "")
                        if len(new_text) > len(existing_text):
                            self.graph_nodes[existing_idx] = node
                            self.logger.debug(f"Updated Chunk {node_id} with longer text")
                        else:
                            self.logger.debug(f"Skipping shorter duplicate Chunk {node_id}")
                    elif node_type == "Assessment":
                        # For Assessments: ignore duplicates
                        self.logger.warning(f"Duplicate Assessment ID: {node_id}, skipping")
                else:
                    # New node, add it
                    # Check for missing difficulty in Chunk nodes
                    if node_type == "Chunk" and "difficulty" not in node:
                        self.logger.warning(
                            f"Chunk {node_id} missing difficulty, setting to {DEFAULT_DIFFICULTY}"
                        )
                        node["difficulty"] = DEFAULT_DIFFICULTY

                    nodes_to_add.append(node)
                    self.node_ids[node_id] = len(self.graph_nodes) + len(nodes_to_add) - 1

            elif node_type == "Concept":
                # For Concept nodes: always use definition from ConceptDictionary
                concept_id = node.get("id", "")
                concept_from_dict = None

                for concept in self.concept_dict["concepts"]:
                    if concept["concept_id"] == concept_id:
                        concept_from_dict = concept
                        break

                if concept_from_dict:
                    # Create proper Concept node with definition from dictionary
                    concept_node = {
                        "id": concept_id,
                        "type": "Concept",
                        "definition": concept_from_dict["definition"],
                    }
                    # Note: Concept nodes don't require node_offset
                    nodes_to_add.append(concept_node)
                    self.logger.debug(
                        f"Added Concept node {concept_id} with definition from dictionary"
                    )
                else:
                    self.logger.warning(
                        f"Concept {concept_id} not found in ConceptDictionary, skipping"
                    )
            else:
                # Unknown node type
                self.logger.warning(f"Unknown node type: {node_type}")

        return nodes_to_add

    def _validate_edges(self, edges: List[Dict]) -> List[Dict]:
        """
        Validate edges with node existence checking and duplicate filtering.

        Args:
            edges: List of edges to validate

        Returns:
            List of valid edges
        """
        valid_edges = []
        existing_edge_keys = set()

        # Build set of existing edge keys
        for edge in self.graph_edges:
            key = (edge["source"], edge["target"], edge["type"])
            existing_edge_keys.add(key)

        for edge in edges:
            source = edge.get("source", "")
            target = edge.get("target", "")
            edge_type = edge.get("type", "")

            # Check self-loop PREREQUISITE
            if edge_type == "PREREQUISITE" and source == target:
                self.logger.warning(f"Dropping PREREQUISITE self-loop: {source}")
                continue

            # Check weight range
            if "weight" in edge:
                weight = edge.get("weight", 0.5)
                if not (0 <= weight <= 1):
                    self.logger.warning(f"Invalid edge weight {weight}, setting to 0.5")
                    edge["weight"] = 0.5

            # Check node existence (in graph nodes or concept dictionary)
            source_exists = source in self.node_ids or any(
                c["concept_id"] == source for c in self.concept_dict["concepts"]
            )
            target_exists = target in self.node_ids or any(
                c["concept_id"] == target for c in self.concept_dict["concepts"]
            )

            if not source_exists:
                self.logger.warning(f"Edge source not found: {source}")
                continue
            if not target_exists:
                self.logger.warning(f"Edge target not found: {target}")
                continue

            # Check for duplicates
            edge_key = (source, target, edge_type)
            if edge_key in existing_edge_keys:
                self.logger.info(f"Duplicate edge filtered: {source} -> {target} ({edge_type})")
                continue

            valid_edges.append(edge)
            existing_edge_keys.add(edge_key)

        return valid_edges

    def _add_mentions_edges(self, chunk_nodes: List[Dict]) -> int:
        """
        Automatically add MENTIONS edges from Chunks to Concepts based on text search.

        Args:
            chunk_nodes: List of Chunk nodes

        Returns:
            Number of MENTIONS edges added
        """
        added_count = 0

        # Build set of ALL existing MENTIONS edges (including from LLM)
        existing_mentions = set()
        for edge in self.graph_edges:
            if edge["type"] == "MENTIONS":
                existing_mentions.add((edge["source"], edge["target"]))

        for chunk in chunk_nodes:
            if chunk.get("type") != "Chunk":
                continue

            chunk_id = chunk["id"]
            chunk_text = chunk.get("text", "").lower()  # Case-insensitive

            for concept in self.concept_dict["concepts"]:
                concept_id = concept["concept_id"]

                # Skip if MENTIONS edge already exists (from LLM or previous addition)
                if (chunk_id, concept_id) in existing_mentions:
                    continue

                found = False

                # Search for primary term
                primary = concept["term"]["primary"]
                pattern = r"\b" + re.escape(primary.lower()) + r"\b"
                if re.search(pattern, chunk_text):
                    found = True

                # Search for aliases if not found
                if not found:
                    for alias in concept["term"].get("aliases", []):
                        pattern = r"\b" + re.escape(alias.lower()) + r"\b"
                        if re.search(pattern, chunk_text):
                            found = True
                            break

                # Add MENTIONS edge if concept found
                if found:
                    edge = {
                        "source": chunk_id,
                        "target": concept_id,
                        "type": "MENTIONS",
                        "weight": 1.0,
                    }
                    self.graph_edges.append(edge)
                    existing_mentions.add((chunk_id, concept_id))
                    added_count += 1
                    self.logger.debug(f"Added automatic MENTIONS: {chunk_id} -> {concept_id}")

        return added_count

    def _assign_final_ids(self, patch: Dict, slice_data: SliceData) -> None:
        """
        Replace temporary IDs with final position-based IDs.
        Must be called BEFORE any validation or processing.

        Args:
            patch: Graph patch from LLM with temporary IDs
            slice_data: Current slice data with slice_token_start
        """
        # Build ID mapping for updates
        id_mapping = {}

        # Process nodes
        for node in patch.get("nodes", []):
            old_id = node.get("id", "")
            node_type = node.get("type", "")

            # Check for required node_offset field
            if "node_offset" not in node:
                self.logger.warning(f"Node {old_id} missing required node_offset field")
                continue

            node_offset = node["node_offset"]

            if node_type == "Chunk" and old_id.startswith("chunk_"):
                # Calculate final position
                final_position = slice_data.slice_token_start + node_offset
                new_id = f"{slice_data.slug}:c:{final_position}"

                # Update node ID
                node["id"] = new_id
                id_mapping[old_id] = new_id
                self.logger.debug(f"Replaced chunk ID: {old_id} -> {new_id}")

            elif node_type == "Assessment" and old_id.startswith("assessment_"):
                # Calculate final position
                final_position = slice_data.slice_token_start + node_offset

                # Extract index from temporary ID (assessment_1 → 1)
                try:
                    index = old_id.split("_")[-1] if "_" in old_id else "0"
                except Exception:
                    index = "0"
                    self.logger.warning(f"Could not extract index from {old_id}, using 0")

                new_id = f"{slice_data.slug}:q:{final_position}:{index}"

                # Update node ID
                node["id"] = new_id
                id_mapping[old_id] = new_id
                self.logger.debug(f"Replaced assessment ID: {old_id} -> {new_id}")

            # Note: Concept nodes keep their IDs from ConceptDictionary (no change needed)

        # Update edges to use new IDs
        for edge in patch.get("edges", []):
            if edge.get("source") in id_mapping:
                old_source = edge["source"]
                edge["source"] = id_mapping[old_source]
                self.logger.debug(f"Updated edge source: {old_source} -> {edge['source']}")

            if edge.get("target") in id_mapping:
                old_target = edge["target"]
                edge["target"] = id_mapping[old_target]
                self.logger.debug(f"Updated edge target: {old_target} -> {edge['target']}")

        self.logger.info(
            f"Post-processing: replaced {len(id_mapping)} temporary IDs with final position-based IDs"
        )

    def _add_to_graph(self, patch: Dict, slice_data: SliceData) -> None:
        """
        Add patch to graph with full processing.

        Args:
            patch: Graph patch from LLM
            slice_data: Current slice data
        """
        # Process nodes with duplicate handling
        new_nodes = patch.get("nodes", [])

        # Process nodes (handle duplicates for Chunks/Assessments)
        nodes_to_add = self._process_chunk_nodes(new_nodes)
        self.graph_nodes.extend(nodes_to_add)

        # Update stats
        self.stats.total_nodes = len(self.graph_nodes)

        # Process edges with validation
        new_edges = patch.get("edges", [])
        valid_edges = self._validate_edges(new_edges)
        self.graph_edges.extend(valid_edges)

        # Update stats
        self.stats.total_edges = len(self.graph_edges)

        # Add automatic MENTIONS edges for all Chunk nodes
        # This covers both new chunks and potentially missed mentions
        all_chunk_nodes = [n for n in self.graph_nodes if n.get("type") == "Chunk"]
        mentions_added = self._add_mentions_edges(all_chunk_nodes)

        if mentions_added > 0:
            self.logger.info(f"Added {mentions_added} automatic MENTIONS edges")
            self.stats.total_edges = len(self.graph_edges)

    def _validate_graph_intermediate(self) -> bool:
        """
        Validate graph invariants after each slice (allows Concept duplicates).

        Returns:
            True if validation passes, False otherwise
        """
        # Check ID uniqueness for Chunks and Assessments (but NOT Concepts)
        chunk_assessment_ids = set()

        for node in self.graph_nodes:
            if node.get("type") in ["Chunk", "Assessment"]:
                node_id = node.get("id", "")
                if node_id in chunk_assessment_ids:
                    self.logger.error(f"Duplicate {node['type']} ID found: {node_id}")
                    return False
                chunk_assessment_ids.add(node_id)

        # Additional invariant checks can be added here
        return True

    def _save_bad_response(
        self,
        slice_id: str,
        original_response: str,
        error: str,
        repair_response: Optional[str] = None,
    ) -> None:
        """
        Save incorrect response for analysis.

        Args:
            slice_id: Slice ID
            original_response: First LLM response
            error: Error description
            repair_response: Response after repair (if any)
        """
        bad_response_file = LOGS_DIR / f"{slice_id}_bad.json"
        bad_data = {
            "slice_id": slice_id,
            "timestamp": datetime.now().isoformat(),
            "original_response": original_response,
            "error": error,
            "repair_response": repair_response,
        }

        try:
            bad_response_file.write_text(
                json.dumps(bad_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self.logger.debug(f"Saved bad response to {bad_response_file}")
        except Exception as e:
            self.logger.error(f"Failed to save bad response: {e}")

    def _save_temp_dumps(self, reason: str) -> None:
        """
        Save temporary dumps on critical errors.

        Args:
            reason: Save reason (validation_failed, io_error, etc.)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save current graph state
        graph_path = LOGS_DIR / f"LearningChunkGraph_temp_{reason}_{timestamp}.json"
        graph_data = {"nodes": self.graph_nodes, "edges": self.graph_edges}

        try:
            graph_path.write_text(
                json.dumps(graph_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.logger.info(f"Saved temporary graph to: {graph_path}")
        except Exception as e:
            self.logger.error(f"Failed to save temporary graph: {e}")

        # Save processing stats
        stats_path = LOGS_DIR / f"processing_stats_{reason}_{timestamp}.json"
        stats_data = {
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "stats": {
                "total_slices": self.stats.total_slices,
                "processed_slices": self.stats.processed_slices,
                "failed_slices": self.stats.failed_slices,
                "total_nodes": self.stats.total_nodes,
                "total_edges": self.stats.total_edges,
                "total_tokens_used": self.stats.total_tokens_used,
                "processing_time": str(datetime.now() - self.stats.start_time),
            },
        }

        try:
            stats_path.write_text(
                json.dumps(stats_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self.logger.info(f"Saved processing stats to: {stats_path}")
        except Exception as e:
            self.logger.error(f"Failed to save processing stats: {e}")

    def _process_single_slice(self, slice_file: Path) -> bool:
        """
        Process single slice with dual repair for JSON and ID errors.

        Args:
            slice_file: Path to slice file

        Returns:
            True if successful, False on error
        """
        # Load slice data
        try:
            slice_data = self._load_slice(slice_file)
        except Exception as e:
            self.logger.error(f"Failed to load slice {slice_file}: {e}")
            return False

        slice_id = slice_data.id
        slice_token_start = slice_data.slice_token_start

        # Log processing start
        self.logger.info(
            json.dumps(
                {
                    "timestamp": datetime.now().isoformat(),
                    "level": "INFO",
                    "event": "slice_start",
                    "slice_id": slice_id,
                    "order": slice_data.order,
                    "slice_token_start": slice_token_start,
                    "total": self.stats.total_slices,
                }
            )
        )

        # Format input with FULL ConceptDictionary
        input_data = self._format_slice_input(slice_data)

        # Call LLM with previous_response_id
        try:
            response_text, response_id, usage = self.llm_client.create_response(
                instructions=self.extraction_prompt,
                input_data=input_data,
                previous_response_id=self.previous_response_id,
            )
            self.stats.total_tokens_used += usage.total_tokens

            # Track API usage
            self.api_usage["total_requests"] += 1
            self.api_usage["total_input_tokens"] += usage.input_tokens
            self.api_usage["total_output_tokens"] += usage.output_tokens

            # Log LLM response
            self.logger.debug(
                json.dumps(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "level": "DEBUG",
                        "event": "llm_response",
                        "slice_id": slice_id,
                        "response_id": response_id,
                        "usage": {
                            "input_tokens": usage.input_tokens,
                            "output_tokens": usage.output_tokens,
                            "reasoning_tokens": usage.reasoning_tokens,
                            "total_tokens": usage.total_tokens,
                        },
                    }
                )
            )
        except Exception as e:
            self.logger.error(f"LLM call failed for {slice_id}: {e}")
            return False

        # Try to parse JSON
        success, parsed = self._process_llm_response(response_text, slice_id)

        if success:
            # НОВОЕ: Подтверждаем response после успешной валидации
            self.llm_client.confirm_response()
        # Handle JSON parsing errors
        elif not success:
            self.logger.warning(f"JSON validation failed for {slice_id}, attempting repair...")

            # Add console output
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"[{current_time}] REPAIR   | 🔧 Attempting to fix JSON validation error...")

            # Repair with JSON format emphasis
            repair_instructions = (
                f"{self.extraction_prompt}\n\n"
                "CRITICAL: Return ONLY a valid JSON object. "
                "No markdown formatting, no explanations, no text outside the JSON structure."
            )

            try:
                repair_text, repair_id, repair_usage = self.llm_client.repair_response(
                    instructions=repair_instructions,
                    input_data=input_data,
                    previous_response_id=self.previous_response_id,  # Rollback!
                )
                self.stats.total_tokens_used += repair_usage.total_tokens

                # Track API usage for repair
                self.api_usage["total_requests"] += 1
                self.api_usage["total_input_tokens"] += repair_usage.input_tokens
                self.api_usage["total_output_tokens"] += repair_usage.output_tokens
                success, parsed = self._process_llm_response(repair_text, slice_id)

                if success:
                    # Repair successful
                    # НОВОЕ: Подтверждаем repair response
                    self.llm_client.confirm_response()
                    current_time = datetime.now().strftime("%H:%M:%S")
                    print(f"[{current_time}] REPAIR   | ✅ JSON validation fixed successfully!")
                    # Update response_id to repair_id for successful repair
                    response_id = repair_id
                else:
                    self.logger.error(f"JSON repair failed for {slice_id}")
                    self._save_bad_response(
                        slice_id, response_text, "JSON parse failed", repair_text
                    )
                    return False

            except Exception as e:
                self.logger.error(f"Repair failed for {slice_id}: {e}")
                self._save_bad_response(slice_id, response_text, f"Repair error: {e}")
                return False

        # JSON is valid, apply post-processing to fix IDs
        if parsed and "chunk_graph_patch" in parsed:
            patch = parsed["chunk_graph_patch"]

            # NEW: Replace temporary IDs with final position-based IDs
            self._assign_final_ids(patch, slice_data)

            # Position validation and repair are no longer needed - IDs are now correct
            self._add_to_graph(patch, slice_data)

            # Intermediate validation
            if not self._validate_graph_intermediate():
                self.logger.error(f"Intermediate validation failed after {slice_id}")
                self._save_temp_dumps(f"validation_error_{slice_id}")
                return False

            # Update previous_response_id ONLY on success
            self.previous_response_id = response_id

            # Log success
            self.logger.info(
                json.dumps(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "level": "INFO",
                        "event": "slice_complete",
                        "slice_id": slice_id,
                        "nodes_added": len(patch.get("nodes", [])),
                        "edges_added": len(patch.get("edges", [])),
                        "total_nodes": self.stats.total_nodes,
                        "total_edges": self.stats.total_edges,
                    }
                )
            )

            return True

        return False

    def run(self) -> int:
        """
        Main processing loop.

        Returns:
            Exit code
        """
        # Load slice files
        slice_files = sorted(STAGING_DIR.glob("*.slice.json"))
        if not slice_files:
            self.logger.error("No slice files found in staging directory")
            print(f"ERROR: No slice files found in {STAGING_DIR}")
            return EXIT_INPUT_ERROR

        self.stats.total_slices = len(slice_files)

        # Calculate total source tokens and get slug from first slice
        self.total_source_tokens = 0
        self.source_slug = "unknown"

        # Load first slice to get slug and calculate total tokens
        if slice_files:
            try:
                first_slice_data = json.loads(slice_files[0].read_text(encoding="utf-8"))
                self.source_slug = first_slice_data.get("slug", "unknown")

                # Calculate total tokens from last slice's end position
                last_slice_data = json.loads(slice_files[-1].read_text(encoding="utf-8"))
                self.total_source_tokens = last_slice_data.get("slice_token_end", 0)
            except Exception:
                pass  # Use defaults if can't read

        # Display start message
        model = self.config["model"]
        tpm_limit = self.config["tpm_limit"]
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(
            f"[{timestamp}] START    | {self.stats.total_slices} slices | model={model} | tpm={tpm_limit//1000}k"
        )

        # Process each slice
        for i, slice_file in enumerate(slice_files, 1):
            start_time = time.time()
            success = self._process_single_slice(slice_file)
            elapsed = int(time.time() - start_time)

            if success:
                self.stats.processed_slices += 1
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(
                    f"[{timestamp}] SLICE    | ✅ {i:03d}/{self.stats.total_slices} | "
                    f"tokens_used={self._format_tokens(self.stats.total_tokens_used)} | "
                    f"{elapsed}s | nodes={self.stats.total_nodes} | edges={self.stats.total_edges}"
                )
            else:
                self.stats.failed_slices += 1
                timestamp = datetime.now().strftime("%H:%M:%S")
                slice_id = slice_file.stem.replace(".slice", "")
                print(
                    f"[{timestamp}] ERROR    | ❌ {i:03d}/{self.stats.total_slices} | {slice_id} | Processing failed"
                )

                # If repair failed, we cannot continue (incomplete graph)
                print(f"[{timestamp}] FAILED   | ❌ Cannot continue without slice {slice_id}")
                self._save_temp_dumps("critical_error")
                return EXIT_RUNTIME_ERROR

        # Check if any slices were processed
        if self.stats.processed_slices == 0:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] FAILED   | ❌ All slices failed processing")
            return EXIT_RUNTIME_ERROR

        # Save results
        output_path = OUTPUT_DIR / "LearningChunkGraph_raw.json"
        try:
            # Calculate graph statistics
            graph_stats = {
                "total_nodes": len(self.graph_nodes),
                "chunks": len([n for n in self.graph_nodes if n.get("type") == "Chunk"]),
                "concepts": len([n for n in self.graph_nodes if n.get("type") == "Concept"]),
                "assessments": len([n for n in self.graph_nodes if n.get("type") == "Assessment"]),
                "total_edges": len(self.graph_edges),
                "edge_types": {},
            }

            # Count edge types
            for edge in self.graph_edges:
                edge_type = edge.get("type", "UNKNOWN")
                graph_stats["edge_types"][edge_type] = (
                    graph_stats["edge_types"].get(edge_type, 0) + 1
                )

            # Get config values safely
            config = self.config.copy()
            slicer_config = self.full_config.get("slicer", {})

            # Get concepts count from ConceptDictionary
            concepts_count = len(self.concept_dict.get("concepts", []))

            # Collect metadata
            end_time = datetime.now()
            duration_minutes = (end_time - self.stats.start_time).total_seconds() / 60

            metadata = {
                "_meta": {
                    "generated_at": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "generator": "itext2kg_graph",
                    "config": {
                        "model": config.get("model"),
                        "temperature": config.get("temperature"),
                        "max_output_tokens": config.get("max_completion"),
                        "reasoning_effort": config.get("reasoning_effort"),
                        "overlap": slicer_config.get("overlap", 0),
                        "slice_size": slicer_config.get("max_tokens", 5000),
                    },
                    "source": {
                        "total_slices": self.stats.total_slices,
                        "processed_slices": self.stats.processed_slices,
                        "total_tokens": self.total_source_tokens,
                        "slug": self.source_slug,
                        "concepts_used": concepts_count,
                    },
                    "api_usage": {
                        "total_requests": self.api_usage["total_requests"],
                        "total_input_tokens": self.api_usage["total_input_tokens"],
                        "total_output_tokens": self.api_usage["total_output_tokens"],
                        "total_tokens": self.api_usage["total_input_tokens"]
                        + self.api_usage["total_output_tokens"],
                    },
                    "graph_stats": graph_stats,
                    "processing_time": {
                        "start": self.stats.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "end": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "duration_minutes": round(duration_minutes, 2),
                    },
                }
            }

            # Merge metadata with graph data
            output_data = {**metadata, "nodes": self.graph_nodes, "edges": self.graph_edges}

            # Validate basic structure before saving
            if not self.graph_nodes:
                self.logger.warning("No nodes in final graph")
            if not self.graph_edges:
                self.logger.warning("No edges in final graph")

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] SUCCESS  | ✅ Results saved to /data/out/")
            print("                    | - LearningChunkGraph_raw.json")
            print(f"                    | - Nodes: {len(self.graph_nodes)}")
            print(f"                    | - Edges: {len(self.graph_edges)}")

        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            self._save_temp_dumps("io_error")
            return EXIT_IO_ERROR

        # Display completion
        elapsed = int(time.time() - self.stats.start_time.timestamp())
        elapsed_str = f"{elapsed // 60}m {elapsed % 60}s"
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(
            f"[{timestamp}] END      | Done | slices={self.stats.total_slices} | time={elapsed_str}"
        )

        # Final statistics log
        self.logger.info(
            json.dumps(
                {
                    "timestamp": datetime.now().isoformat(),
                    "level": "INFO",
                    "event": "processing_complete",
                    "stats": {
                        "total_slices": self.stats.total_slices,
                        "processed_slices": self.stats.processed_slices,
                        "failed_slices": self.stats.failed_slices,
                        "total_nodes": self.stats.total_nodes,
                        "total_edges": self.stats.total_edges,
                        "total_tokens_used": self.stats.total_tokens_used,
                        "processing_time": elapsed_str,
                    },
                }
            )
        )

        return EXIT_SUCCESS


def main():
    """Main entry point."""
    try:
        # Load configuration
        config = load_config(CONFIG_PATH)

        # Create processor and run
        processor = SliceProcessor(config)
        exit_code = processor.run()

        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Processing stopped by user")
        sys.exit(EXIT_RUNTIME_ERROR)
    except Exception as e:
        print("[FATAL] Unexpected error: " + str(e))
        sys.exit(EXIT_RUNTIME_ERROR)


if __name__ == "__main__":
    main()

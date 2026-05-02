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

import openai

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
from src.utils.validation import validate_graph_invariants_intermediate

setup_console_encoding()

# Constants
CONFIG_PATH = Path(__file__).parent / "config.toml"
PROMPTS_DIR = Path(__file__).parent / "prompts"
SCHEMAS_DIR = Path(__file__).parent / "schemas"
STAGING_DIR = Path(__file__).parent.parent / "data" / "staging"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "out"
LOGS_DIR = Path(__file__).parent.parent / "logs"
CHECKPOINTS_DIR_NAME = "checkpoints"
LATEST_CHECKPOINT_FILENAME = "LearningChunkGraph_raw_latest.json"
GRAPH_PATCHES_DIR_NAME = "graph_patches"

GRAPH_EXTRACTION_PROMPT_FILE = "itext2kg_graph_extraction.md"
MAX_REPAIR_ATTEMPTS = 1
DEFAULT_DIFFICULTY = 3  # Default difficulty for Chunk nodes if not provided
ALLOWED_EDGE_TYPES = {
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
        self.config = config["itext2kg_graph"]
        self.full_config = config  # Save full config for accessing slicer settings
        self.llm_client = OpenAIClient(self.config)
        self.logger = self._setup_logger()
        self.stats = ProcessingStats()

        # Initialize quality issues tracking for deduplication
        self.quality_issues = {
            "duplicate_concepts_removed": 0,
            "anomalous_duplicates": [],
            "invalid_edge_types_removed": 0,
            "unknown_edge_endpoints_removed": 0,
            "auto_concept_nodes_added": 0,
            "quality_repair_requests": 0,
        }

        # Load ConceptDictionary
        self.concept_dict = self._load_concept_dictionary()
        self.concept_by_id = {
            concept["concept_id"]: concept for concept in self.concept_dict.get("concepts", [])
        }
        self.mentions_blacklist = self._load_mentions_blacklist()

        # Initialize graph structures
        self.graph_nodes: List[Dict] = []  # All nodes
        self.graph_edges: List[Dict] = []  # All edges
        self.node_ids: Dict[str, int] = {}  # {node_id: index} for duplicate detection

        # Context tracking for incremental processing
        self.previous_response_id: Optional[str] = None
        self.last_completed_slice: Optional[SliceData] = None

        # API usage tracking
        self.api_usage = {
            "total_requests": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }

        # Source metadata is populated in run(), but defaults are needed for unit-level calls.
        self.total_source_tokens = 0
        self.source_slug = "unknown"

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

            self._apply_concept_alias_patch(concept_dict)

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

    def _resolve_config_path(self, path_value: str) -> Path:
        """Resolve a configured path relative to src/config/ by default."""
        path = Path(path_value)
        if path.is_absolute():
            return path
        config_dir_path = Path(__file__).parent / "config"
        return config_dir_path / path

    @staticmethod
    def _normalize_match_term(term: str) -> str:
        """Normalize terms for alias matching and blacklist checks."""
        return " ".join(term.casefold().replace("ё", "е").split())

    def _apply_concept_alias_patch(self, concept_dict: Dict[str, Any]) -> None:
        """Apply optional in-memory alias patch to ConceptDictionary."""
        patch_file = self.config.get("concept_alias_patch_file")
        self.alias_patch_report = {
            "patch_file": patch_file,
            "concepts_matched": 0,
            "aliases_added": 0,
            "missing_concept_ids": [],
        }
        if not patch_file:
            return

        patch_path = self._resolve_config_path(patch_file)
        if not patch_path.exists():
            self.logger.warning(f"Concept alias patch not found: {patch_path}")
            self.alias_patch_report["missing_patch_file"] = str(patch_path)
            return

        try:
            patch_data = json.loads(patch_path.read_text(encoding="utf-8"))
        except Exception as e:
            self.logger.warning(f"Failed to read concept alias patch {patch_path}: {e}")
            self.alias_patch_report["error"] = str(e)
            return

        concepts_by_id = {
            concept.get("concept_id"): concept for concept in concept_dict.get("concepts", [])
        }
        aliases_by_concept_id = patch_data.get("aliases_by_concept_id", {})
        if not isinstance(aliases_by_concept_id, dict):
            self.logger.warning(f"Invalid concept alias patch format: {patch_path}")
            self.alias_patch_report["error"] = "aliases_by_concept_id must be an object"
            return

        for concept_id, aliases in aliases_by_concept_id.items():
            concept = concepts_by_id.get(concept_id)
            if concept is None:
                self.alias_patch_report["missing_concept_ids"].append(concept_id)
                continue
            if not isinstance(aliases, list):
                continue

            term = concept.setdefault("term", {})
            primary = term.get("primary", "")
            current_aliases = term.setdefault("aliases", [])
            existing = {
                self._normalize_match_term(alias) for alias in current_aliases if isinstance(alias, str)
            }
            primary_normalized = self._normalize_match_term(primary)
            added_for_concept = 0

            for alias in aliases:
                if not isinstance(alias, str) or not alias.strip():
                    continue
                normalized = self._normalize_match_term(alias)
                if normalized == primary_normalized or normalized in existing:
                    continue
                current_aliases.append(alias)
                existing.add(normalized)
                added_for_concept += 1

            if added_for_concept:
                self.alias_patch_report["concepts_matched"] += 1
                self.alias_patch_report["aliases_added"] += added_for_concept

        self.logger.info(
            json.dumps(
                {
                    "timestamp": datetime.now().isoformat(),
                    "level": "INFO",
                    "event": "concept_alias_patch_applied",
                    **self.alias_patch_report,
                },
                ensure_ascii=False,
            )
        )

    def _load_mentions_blacklist(self) -> set:
        """Load optional term blacklist for automatic MENTIONS matching."""
        blacklist_file = self.config.get("mentions_blacklist_file")
        if not blacklist_file:
            return set()

        blacklist_path = self._resolve_config_path(blacklist_file)
        if not blacklist_path.exists():
            self.logger.warning(f"MENTIONS blacklist not found: {blacklist_path}")
            return set()

        terms = set()
        try:
            for line in blacklist_path.read_text(encoding="utf-8").splitlines():
                clean = line.split("#", 1)[0].strip()
                if clean:
                    terms.add(self._normalize_match_term(clean))
        except Exception as e:
            self.logger.warning(f"Failed to load MENTIONS blacklist {blacklist_path}: {e}")
            return set()

        return terms

    def _load_extraction_prompt(self) -> str:
        """Load prompt with schema substitution."""
        prompt_name = self.config.get("prompt_file", GRAPH_EXTRACTION_PROMPT_FILE)
        prompt_file = Path(prompt_name)
        if not prompt_file.is_absolute():
            prompt_file = PROMPTS_DIR / prompt_file

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
                concept_from_dict = self.concept_by_id.get(concept_id)

                if concept_from_dict:
                    # Create proper Concept node with definition from dictionary
                    concept_node = {
                        "id": concept_id,
                        "type": "Concept",
                        "text": node.get(
                            "text", concept_from_dict["term"]["primary"]
                        ),  # Take from LLM response or fallback to primary term
                        "node_offset": node.get(
                            "node_offset", 0
                        ),  # Take from LLM response or default to 0
                        "definition": concept_from_dict[
                            "definition"
                        ],  # Always use definition from ConceptDictionary
                    }
                    nodes_to_add.append(concept_node)
                    self.node_ids[concept_id] = len(self.graph_nodes) + len(nodes_to_add) - 1
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

            # Check edge type against schema enum
            if edge_type not in ALLOWED_EDGE_TYPES:
                self.logger.warning(
                    f"Dropping edge with invalid type {edge_type}: {source} -> {target}"
                )
                self.quality_issues["invalid_edge_types_removed"] += 1
                continue

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
            source_exists = source in self.node_ids
            target_exists = target in self.node_ids

            if not source_exists:
                self.logger.warning(f"Edge source not found: {source}")
                self.quality_issues["unknown_edge_endpoints_removed"] += 1
                continue
            if not target_exists:
                self.logger.warning(f"Edge target not found: {target}")
                self.quality_issues["unknown_edge_endpoints_removed"] += 1
                continue

            # Check for duplicates
            edge_key = (source, target, edge_type)
            if edge_key in existing_edge_keys:
                self.logger.info(f"Duplicate edge filtered: {source} -> {target} ({edge_type})")
                continue

            valid_edges.append(edge)
            existing_edge_keys.add(edge_key)

        return valid_edges

    def _ensure_concept_node_in_graph(self, concept_id: str, node_offset: int = 0) -> bool:
        """
        Ensure a Concept node exists in the accumulated graph.

        Args:
            concept_id: ConceptDictionary ID
            node_offset: Best known mention offset

        Returns:
            True if a node was added, False if it already existed or concept is unknown.
        """
        if concept_id in self.node_ids:
            return False

        concept = self.concept_by_id.get(concept_id)
        if concept is None:
            return False

        concept_node = {
            "id": concept_id,
            "type": "Concept",
            "text": concept["term"]["primary"],
            "node_offset": max(0, int(node_offset or 0)),
            "definition": concept["definition"],
        }
        self.node_ids[concept_id] = len(self.graph_nodes)
        self.graph_nodes.append(concept_node)
        self.stats.total_nodes = len(self.graph_nodes)
        self.quality_issues["auto_concept_nodes_added"] += 1
        return True

    def _ensure_referenced_concept_nodes(self, patch: Dict) -> int:
        """
        Add Concept nodes to the patch for any edge endpoint that references ConceptDictionary.

        This keeps LLM output recoverable when it creates a valid edge to a known concept but
        forgets to include the corresponding Concept node in `nodes`.
        """
        patch_nodes = patch.get("nodes", [])
        patch_node_ids = {node.get("id") for node in patch_nodes}
        graph_node_ids = {node.get("id") for node in self.graph_nodes}
        added = 0

        for edge in patch.get("edges", []):
            for endpoint in (edge.get("source"), edge.get("target")):
                if (
                    endpoint in self.concept_by_id
                    and endpoint not in patch_node_ids
                    and endpoint not in graph_node_ids
                ):
                    concept = self.concept_by_id[endpoint]
                    patch_nodes.append(
                        {
                            "id": endpoint,
                            "type": "Concept",
                            "text": concept["term"]["primary"],
                            "node_offset": 0,
                            "definition": concept["definition"],
                        }
                    )
                    patch_node_ids.add(endpoint)
                    added += 1

        if added:
            self.quality_issues["auto_concept_nodes_added"] += added
            self.logger.info(f"Added {added} missing Concept nodes referenced by edges")

        return added

    def _add_mentions_edges(self, chunk_nodes: List[Dict]) -> int:
        """
        Automatically add MENTIONS edges from Chunks to Concepts based on text search.
        Uses configurable weight from config and marks edges as auto-generated.

        Args:
            chunk_nodes: List of Chunk nodes

        Returns:
            Number of MENTIONS edges added
        """
        added_count = 0

        # Get weight from config (with default fallback)
        mentions_weight = self.config.get("auto_mentions_weight", 0.35)

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
            chunk_offset = int(chunk.get("node_offset", 0) or 0)

            for concept in self.concept_dict["concepts"]:
                concept_id = concept["concept_id"]

                # Skip if MENTIONS edge already exists (from LLM or previous addition)
                if (chunk_id, concept_id) in existing_mentions:
                    continue

                found = False

                # Search for primary term
                primary = concept["term"]["primary"]
                if self._normalize_match_term(primary) not in self.mentions_blacklist:
                    pattern = r"\b" + re.escape(primary.lower()) + r"\b"
                    if re.search(pattern, chunk_text):
                        found = True

                # Search for aliases if not found
                if not found:
                    for alias in concept["term"].get("aliases", []):
                        if self._normalize_match_term(alias) in self.mentions_blacklist:
                            continue
                        pattern = r"\b" + re.escape(alias.lower()) + r"\b"
                        if re.search(pattern, chunk_text):
                            found = True
                            break

                # Add MENTIONS edge if concept found
                if found:
                    self._ensure_concept_node_in_graph(concept_id, chunk_offset)
                    edge = {
                        "source": chunk_id,
                        "target": concept_id,
                        "type": "MENTIONS",
                        "weight": mentions_weight,
                        "conditions": "auto_generated",
                    }
                    self.graph_edges.append(edge)
                    existing_mentions.add((chunk_id, concept_id))
                    added_count += 1
                    self.logger.debug(
                        f"Added automatic MENTIONS: {chunk_id} -> {concept_id} "
                        f"(weight={mentions_weight})"
                    )

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

    def _deduplicate_patch_nodes(self, graph: Dict, patch: Dict, slice_id: str) -> Dict:
        """
        Remove duplicate nodes from patch before merging.

        Rules:
        - Silently remove duplicate Concept nodes (expected behavior)
        - Log WARNING for duplicate Chunk/Assessment nodes (anomaly)
        - Keep first occurrence, remove subsequent duplicates
        - Track statistics for metadata

        Args:
            graph: Current graph state (nodes list)
            patch: New patch to be deduplicated
            slice_id: Current slice identifier for logging

        Returns:
            Modified patch with duplicates removed
        """
        # Build set of existing node IDs from graph
        existing_ids = set()
        for node in self.graph_nodes:
            existing_ids.add(node.get("id", ""))

        # Process patch nodes
        deduplicated_nodes = []
        concepts_removed = 0
        anomalous_duplicates = []

        for node in patch.get("nodes", []):
            node_id = node.get("id", "")
            node_type = node.get("type", "")

            if node_id in existing_ids:
                # Duplicate found
                if node_type == "Concept":
                    # Expected behavior for concepts - remove silently
                    concepts_removed += 1
                    self.logger.debug(f"Removed duplicate Concept node: {node_id}")
                else:
                    # Anomaly for Chunk/Assessment - log warning
                    self.logger.warning(
                        f"Unexpected duplicate {node_type} node: {node_id} in slice {slice_id}"
                    )
                    anomalous_duplicates.append(
                        {"node_id": node_id, "node_type": node_type, "slice_id": slice_id}
                    )
            else:
                # Not a duplicate, keep it
                deduplicated_nodes.append(node)
                existing_ids.add(node_id)  # Add to set for checking within patch

        # Update statistics
        self.quality_issues["duplicate_concepts_removed"] += concepts_removed
        self.quality_issues["anomalous_duplicates"].extend(anomalous_duplicates)

        # Return modified patch
        return {**patch, "nodes": deduplicated_nodes}

    def _add_to_graph(self, patch: Dict, slice_data: SliceData) -> None:
        """
        Add patch to graph with full processing.

        Args:
            patch: Graph patch from LLM
            slice_data: Current slice data
        """
        # DEDUPLICATION - clean patch before merging
        patch = self._deduplicate_patch_nodes({"nodes": self.graph_nodes}, patch, slice_data.id)

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
        try:
            validate_graph_invariants_intermediate(
                {"nodes": self.graph_nodes, "edges": self.graph_edges}
            )
            return True
        except Exception as e:
            self.logger.error(f"Intermediate graph invariant validation failed: {e}")
            return False

    def _validate_patch_quality(self, patch: Dict, slice_data: SliceData) -> Tuple[bool, List[Dict]]:
        """
        Validate slice patch quality before it is merged into the graph.

        This catches semantically suspicious but JSON-valid responses early enough to trigger a
        repair-reprompt instead of silently carrying low-quality graph state forward.
        """
        issues = []
        nodes = patch.get("nodes", [])
        edges = patch.get("edges", [])
        patch_node_ids = {node.get("id") for node in nodes if node.get("id")}
        graph_node_ids = {node.get("id") for node in self.graph_nodes if node.get("id")}
        known_node_ids = patch_node_ids | graph_node_ids

        chunk_nodes = [node for node in nodes if node.get("type") == "Chunk"]
        concept_nodes = [node for node in nodes if node.get("type") == "Concept"]

        if len(chunk_nodes) < self.config.get("quality_min_chunk_nodes", 1):
            issues.append(
                {
                    "severity": "error",
                    "type": "missing_chunk",
                    "message": "Patch must contain at least one Chunk node",
                }
            )

        duplicate_patch_ids = set()
        seen_patch_ids = set()
        for node in nodes:
            node_id = node.get("id")
            if not node_id:
                issues.append(
                    {
                        "severity": "error",
                        "type": "missing_node_id",
                        "message": "Node without id",
                    }
                )
                continue
            if node_id in seen_patch_ids and node.get("type") != "Concept":
                duplicate_patch_ids.add(node_id)
            seen_patch_ids.add(node_id)

        for node_id in sorted(duplicate_patch_ids):
            issues.append(
                {
                    "severity": "error",
                    "type": "duplicate_patch_node",
                    "node_id": node_id,
                    "message": f"Duplicate non-Concept node in patch: {node_id}",
                }
            )

        unknown_concept_nodes = [
            node.get("id")
            for node in concept_nodes
            if node.get("id") and node.get("id") not in self.concept_by_id
        ]
        for concept_id in sorted(set(unknown_concept_nodes)):
            issues.append(
                {
                    "severity": "error",
                    "type": "unknown_concept_node",
                    "concept_id": concept_id,
                    "message": f"Concept node is not in ConceptDictionary: {concept_id}",
                }
            )

        max_edges_per_chunk = self.config.get("quality_max_edges_per_chunk", 8)
        if chunk_nodes and len(edges) > len(chunk_nodes) * max_edges_per_chunk:
            issues.append(
                {
                    "severity": "error",
                    "type": "edge_density_too_high",
                    "message": (
                        f"Patch has {len(edges)} edges for {len(chunk_nodes)} chunks; "
                        f"limit is {max_edges_per_chunk} per chunk"
                    ),
                }
            )

        unknown_endpoints = []
        invalid_edge_types = []
        missing_conditions = []
        for edge in edges:
            edge_type = edge.get("type")
            if edge_type not in ALLOWED_EDGE_TYPES:
                invalid_edge_types.append(edge_type)

            for field_name in ("source", "target"):
                endpoint = edge.get(field_name)
                if endpoint not in known_node_ids:
                    unknown_endpoints.append(
                        {
                            "field": field_name,
                            "endpoint": endpoint,
                            "edge_type": edge_type,
                        }
                    )

            if edge_type in {"PREREQUISITE", "PARALLEL", "REVISION_OF"} and not edge.get(
                "conditions"
            ):
                missing_conditions.append(
                    {
                        "source": edge.get("source"),
                        "target": edge.get("target"),
                        "edge_type": edge_type,
                    }
                )

        for edge_type in sorted({str(edge_type) for edge_type in invalid_edge_types}):
            issues.append(
                {
                    "severity": "error",
                    "type": "invalid_edge_type",
                    "edge_type": edge_type,
                    "message": f"Invalid edge type: {edge_type}",
                }
            )

        max_unknown_endpoints = self.config.get("quality_max_unknown_edge_endpoints", 0)
        if len(unknown_endpoints) > max_unknown_endpoints:
            issues.append(
                {
                    "severity": "error",
                    "type": "unknown_edge_endpoints",
                    "count": len(unknown_endpoints),
                    "items": unknown_endpoints[:10],
                    "message": "Edges reference nodes that are neither in the patch nor graph",
                }
            )

        max_missing_conditions = self.config.get("quality_max_missing_conditions", 2)
        if len(missing_conditions) > max_missing_conditions:
            issues.append(
                {
                    "severity": "error",
                    "type": "missing_conditions",
                    "count": len(missing_conditions),
                    "items": missing_conditions[:10],
                    "message": "Too many important semantic edges lack conditions",
                }
            )

        if not concept_nodes and self.config.get("quality_warn_no_concepts", True):
            issues.append(
                {
                    "severity": "warning",
                    "type": "no_concept_nodes",
                    "slice_id": slice_data.id,
                    "message": "Patch contains no Concept nodes",
                }
            )

        has_errors = any(issue.get("severity") == "error" for issue in issues)
        return not has_errors, issues

    def _format_quality_repair_hint(self, issues: List[Dict]) -> str:
        """Build concise repair instructions from patch quality issues."""
        issue_lines = []
        for issue in issues[:8]:
            message = issue.get("message") or issue.get("type", "quality issue")
            issue_lines.append(f"- {message}")

        return (
            "\nCRITICAL GRAPH QUALITY REPAIR REQUIRED:\n"
            "The previous JSON was parseable but failed graph quality checks:\n"
            + "\n".join(issue_lines)
            + "\nReturn a corrected JSON object only. Keep the same slice, include at least "
            "one Chunk, include Concept nodes for every Concept edge endpoint, use only the "
            "9 allowed edge types, and add concise Russian `conditions` for important "
            "PREREQUISITE/PARALLEL/REVISION_OF edges."
        )

    def _save_graph_patch_artifact(
        self,
        *,
        slice_data: SliceData,
        input_data: str,
        response_text: str,
        response_id: Optional[str],
        attempt: int,
        stage: str,
        parsed_response: Optional[Dict] = None,
        postprocessed_patch: Optional[Dict] = None,
        quality_issues: Optional[List[Dict]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Persist per-slice LLM input/response/patch artifacts for replay and audit."""
        if not self.config.get("archive_graph_patches", True):
            return

        archive_dir_name = self.config.get("graph_patch_archive_dir", GRAPH_PATCHES_DIR_NAME)
        archive_dir = OUTPUT_DIR / archive_dir_name
        safe_slice_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", slice_data.id)
        artifact_path = (
            archive_dir / f"{slice_data.order:03d}_{safe_slice_id}_attempt_{attempt}_{stage}.json"
        )

        try:
            input_obj: Any = json.loads(input_data)
        except json.JSONDecodeError:
            input_obj = input_data

        artifact = {
            "slice": {
                "id": slice_data.id,
                "order": slice_data.order,
                "source_file": slice_data.source_file,
                "slug": slice_data.slug,
                "slice_token_start": slice_data.slice_token_start,
                "slice_token_end": slice_data.slice_token_end,
            },
            "attempt": attempt,
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
            "response_id": response_id,
            "input": input_obj,
            "raw_response": response_text,
            "parsed_response": parsed_response,
            "postprocessed_patch": postprocessed_patch,
            "quality_issues": quality_issues or [],
            "error": error,
        }

        try:
            self._write_json_atomic(artifact_path, artifact)
        except Exception as e:
            self.logger.warning(f"Failed to save graph patch artifact {artifact_path}: {e}")

    def _validate_node_uniqueness(self, graph: Dict) -> Tuple[bool, List]:
        """
        Validate that all node IDs in the graph are unique.

        Args:
            graph: Graph dictionary with nodes list

        Returns:
            (is_valid, list_of_duplicates)
        """
        node_ids = {}
        duplicates = []

        # Check all nodes
        for node in graph.get("nodes", []):
            node_id = node.get("id", "")
            node_type = node.get("type", "")

            if node_id in node_ids:
                # Duplicate found
                duplicates.append(
                    {"id": node_id, "type": node_type, "first_occurrence_type": node_ids[node_id]}
                )
            else:
                node_ids[node_id] = node_type

        is_valid = len(duplicates) == 0
        return is_valid, duplicates

    def _calculate_graph_stats(self) -> Dict[str, Any]:
        """Calculate aggregate graph statistics for output metadata."""
        graph_stats = {
            "total_nodes": len(self.graph_nodes),
            "chunks": len([n for n in self.graph_nodes if n.get("type") == "Chunk"]),
            "concepts": len([n for n in self.graph_nodes if n.get("type") == "Concept"]),
            "assessments": len([n for n in self.graph_nodes if n.get("type") == "Assessment"]),
            "total_edges": len(self.graph_edges),
            "edge_types": {},
        }

        for edge in self.graph_edges:
            edge_type = edge.get("type", "UNKNOWN")
            graph_stats["edge_types"][edge_type] = (
                graph_stats["edge_types"].get(edge_type, 0) + 1
            )

        return graph_stats

    def _build_output_data(
        self,
        *,
        complete: bool,
        processed_slices: Optional[int] = None,
        checkpoint_slice: Optional[SliceData] = None,
        end_time: Optional[datetime] = None,
        is_unique: Optional[bool] = None,
        duplicates: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        Build graph output with metadata for final and checkpoint artifacts.

        Args:
            complete: True for final graph, False for recoverable checkpoints.
            processed_slices: Effective number of successfully processed slices.
            checkpoint_slice: Last fully integrated slice for checkpoint metadata.
            end_time: Timestamp to use for metadata.
            is_unique: Optional precomputed node uniqueness flag.
            duplicates: Optional precomputed duplicate list.

        Returns:
            Graph payload ready to serialize.
        """
        if end_time is None:
            end_time = datetime.now()

        graph_data = {"nodes": self.graph_nodes, "edges": self.graph_edges}
        if is_unique is None or duplicates is None:
            is_unique, duplicates = self._validate_node_uniqueness(graph_data)

        if processed_slices is None:
            processed_slices = self.stats.processed_slices

        config = self.config.copy()
        slicer_config = self.full_config.get("slicer", {})
        concepts_count = len(self.concept_dict.get("concepts", []))
        duration_minutes = (end_time - self.stats.start_time).total_seconds() / 60

        checkpoint_meta = {
            "complete": complete,
            "checkpoint_generated_at": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_completed_slice_id": None,
            "last_completed_slice_order": None,
            "last_completed_slice_token_start": None,
            "last_completed_slice_token_end": None,
            "processed_slices": processed_slices,
            "total_slices": self.stats.total_slices,
            "previous_response_id": self.previous_response_id,
            "stats": {
                "total_nodes": len(self.graph_nodes),
                "total_edges": len(self.graph_edges),
                "total_tokens_used": self.stats.total_tokens_used,
            },
        }

        if checkpoint_slice is not None:
            checkpoint_meta.update(
                {
                    "last_completed_slice_id": checkpoint_slice.id,
                    "last_completed_slice_order": checkpoint_slice.order,
                    "last_completed_slice_token_start": checkpoint_slice.slice_token_start,
                    "last_completed_slice_token_end": checkpoint_slice.slice_token_end,
                }
            )

        metadata = {
            "_meta": {
                "itext2kg_graph": {
                    "generated_at": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "config": {
                        "model": config.get("model"),
                        "prompt_file": config.get("prompt_file", GRAPH_EXTRACTION_PROMPT_FILE),
                        "temperature": config.get("temperature"),
                        "max_output_tokens": config.get("max_completion"),
                        "reasoning_effort": config.get("reasoning_effort"),
                        "overlap": slicer_config.get("overlap", 0),
                        "slice_size": slicer_config.get("max_tokens", 5000),
                        "auto_mentions_weight": config.get("auto_mentions_weight", 0.35),
                        "resume_from_latest": config.get("resume_from_latest", False),
                        "concept_alias_patch_file": config.get("concept_alias_patch_file"),
                        "mentions_blacklist_file": config.get("mentions_blacklist_file"),
                        "archive_graph_patches": config.get("archive_graph_patches", True),
                    },
                    "source": {
                        "total_slices": self.stats.total_slices,
                        "processed_slices": processed_slices,
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
                    "graph_stats": self._calculate_graph_stats(),
                    "concept_alias_patch": self.alias_patch_report,
                    "processing_time": {
                        "start": self.stats.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "end": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "duration_minutes": round(duration_minutes, 2),
                    },
                    "quality_issues": {
                        "duplicate_concepts_removed": self.quality_issues[
                            "duplicate_concepts_removed"
                        ],
                        "anomalous_duplicates": self.quality_issues["anomalous_duplicates"],
                        "graph_has_duplicates": not is_unique,
                        "remaining_duplicates": duplicates if not is_unique else [],
                    },
                    "checkpoint": checkpoint_meta,
                }
            }
        }

        return {**metadata, "nodes": self.graph_nodes, "edges": self.graph_edges}

    def _write_json_atomic(self, output_path: Path, data: Dict[str, Any]) -> None:
        """
        Write JSON atomically by replacing the target only after a full temp write.

        Args:
            output_path: Final path to write.
            data: JSON-serializable data.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_name(f".{output_path.name}.{time.time_ns()}.tmp")

        try:
            temp_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temp_path.replace(output_path)
        except Exception:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            finally:
                raise

    def _checkpoint_path_for_slice(self, slice_data: SliceData) -> Path:
        """Return deterministic checkpoint path for a completed slice."""
        safe_slice_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", slice_data.id)
        return (
            OUTPUT_DIR
            / CHECKPOINTS_DIR_NAME
            / f"LearningChunkGraph_raw_after_{safe_slice_id}.json"
        )

    def _save_success_checkpoint(self, slice_data: SliceData, processed_slices: int) -> None:
        """
        Save recoverable graph artifacts after a slice is fully integrated.

        Args:
            slice_data: Last completed slice.
            processed_slices: Effective successful slice count including slice_data.
        """
        output_data = self._build_output_data(
            complete=False,
            processed_slices=processed_slices,
            checkpoint_slice=slice_data,
        )

        checkpoint_path = self._checkpoint_path_for_slice(slice_data)
        latest_path = OUTPUT_DIR / LATEST_CHECKPOINT_FILENAME

        self._write_json_atomic(checkpoint_path, output_data)
        self._write_json_atomic(latest_path, output_data)

        self.logger.info(
            json.dumps(
                {
                    "timestamp": datetime.now().isoformat(),
                    "level": "INFO",
                    "event": "checkpoint_saved",
                    "slice_id": slice_data.id,
                    "checkpoint_path": str(checkpoint_path),
                    "latest_path": str(latest_path),
                }
            )
        )

    def _rebuild_node_ids(self) -> None:
        """Rebuild node index from restored graph nodes."""
        self.node_ids = {}
        for index, node in enumerate(self.graph_nodes):
            node_id = node.get("id", "")
            if node_id and node_id not in self.node_ids:
                self.node_ids[node_id] = index

    def _slice_by_order(self, slice_files: List[Path], order: int) -> Optional[SliceData]:
        """Find a staging slice by its order field."""
        for slice_file in slice_files:
            slice_data = self._load_slice(slice_file)
            if slice_data.order == order:
                return slice_data
        return None

    def _filter_slice_files_after_order(self, slice_files: List[Path], order: int) -> List[Path]:
        """Return staging slice files with order greater than the restored checkpoint."""
        remaining = []
        for slice_file in slice_files:
            slice_data = self._load_slice(slice_file)
            if slice_data.order > order:
                remaining.append(slice_file)
        return remaining

    def _int_from_meta(self, value: Any, field_name: str) -> int:
        """Read an integer metadata value with a clear error on malformed checkpoints."""
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Invalid checkpoint metadata: {field_name} must be an integer")
        return value

    def _restore_api_usage(self, api_usage: Dict[str, Any]) -> None:
        """Restore API usage counters from checkpoint metadata."""
        total_requests = api_usage.get("total_requests", 0)
        total_input_tokens = api_usage.get("total_input_tokens", 0)
        total_output_tokens = api_usage.get("total_output_tokens", 0)

        self.api_usage = {
            "total_requests": total_requests if isinstance(total_requests, int) else 0,
            "total_input_tokens": total_input_tokens if isinstance(total_input_tokens, int) else 0,
            "total_output_tokens": (
                total_output_tokens if isinstance(total_output_tokens, int) else 0
            ),
        }

    def _restore_quality_issues(self, quality_issues: Dict[str, Any]) -> None:
        """Restore quality issue counters from checkpoint metadata."""
        defaults = {
            "duplicate_concepts_removed": 0,
            "anomalous_duplicates": [],
            "invalid_edge_types_removed": 0,
            "unknown_edge_endpoints_removed": 0,
            "auto_concept_nodes_added": 0,
            "quality_repair_requests": 0,
        }

        restored = {}
        for key, default_value in defaults.items():
            value = quality_issues.get(key, default_value)
            if isinstance(default_value, list):
                restored[key] = value if isinstance(value, list) else []
            else:
                restored[key] = value if isinstance(value, int) else 0

        self.quality_issues = restored

    def _load_latest_checkpoint(self, slice_files: List[Path]) -> int:
        """
        Restore graph state from the latest successful checkpoint.

        Args:
            slice_files: Current staging slice files.

        Returns:
            Order of the last completed slice.

        Raises:
            ValueError: If checkpoint is missing or incompatible with staging.
        """
        checkpoint_path = OUTPUT_DIR / LATEST_CHECKPOINT_FILENAME
        if not checkpoint_path.exists():
            raise ValueError(f"Latest checkpoint not found: {checkpoint_path}")

        try:
            checkpoint_data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse latest checkpoint: {e}") from e
        except OSError as e:
            raise ValueError(f"Failed to read latest checkpoint: {e}") from e

        meta = checkpoint_data.get("_meta", {}).get("itext2kg_graph", {})
        checkpoint_meta = meta.get("checkpoint")
        if not isinstance(checkpoint_meta, dict):
            raise ValueError("Latest checkpoint is missing _meta.itext2kg_graph.checkpoint")

        last_order = self._int_from_meta(
            checkpoint_meta.get("last_completed_slice_order"),
            "last_completed_slice_order",
        )
        processed_slices = self._int_from_meta(
            checkpoint_meta.get("processed_slices"),
            "processed_slices",
        )
        checkpoint_total = self._int_from_meta(
            checkpoint_meta.get("total_slices"),
            "total_slices",
        )

        if checkpoint_total != len(slice_files):
            raise ValueError(
                "Latest checkpoint is incompatible with staging: "
                f"total_slices={checkpoint_total}, current={len(slice_files)}"
            )

        last_slice = self._slice_by_order(slice_files, last_order)
        if last_slice is None:
            raise ValueError(
                "Latest checkpoint is incompatible with staging: "
                f"slice order {last_order} not found"
            )

        expected_slice_id = checkpoint_meta.get("last_completed_slice_id")
        expected_start = checkpoint_meta.get("last_completed_slice_token_start")
        expected_end = checkpoint_meta.get("last_completed_slice_token_end")

        if expected_slice_id != last_slice.id:
            raise ValueError(
                "Latest checkpoint is incompatible with staging: "
                f"last_completed_slice_id={expected_slice_id}, current={last_slice.id}"
            )
        if expected_start != last_slice.slice_token_start:
            raise ValueError(
                "Latest checkpoint is incompatible with staging: "
                "last_completed_slice_token_start does not match current staging"
            )
        if expected_end != last_slice.slice_token_end:
            raise ValueError(
                "Latest checkpoint is incompatible with staging: "
                "last_completed_slice_token_end does not match current staging"
            )

        source_meta = meta.get("source", {})
        expected_slug = source_meta.get("slug")
        if expected_slug and expected_slug != "unknown" and expected_slug != last_slice.slug:
            raise ValueError(
                "Latest checkpoint is incompatible with staging: "
                f"slug={expected_slug}, current={last_slice.slug}"
            )

        nodes = checkpoint_data.get("nodes")
        edges = checkpoint_data.get("edges")
        if not isinstance(nodes, list) or not isinstance(edges, list):
            raise ValueError("Latest checkpoint must contain list fields: nodes and edges")

        self.graph_nodes = nodes
        self.graph_edges = edges
        self._rebuild_node_ids()
        self.previous_response_id = checkpoint_meta.get("previous_response_id")
        self.last_completed_slice = last_slice

        self.stats.processed_slices = processed_slices
        self.stats.total_nodes = len(self.graph_nodes)
        self.stats.total_edges = len(self.graph_edges)

        api_usage = meta.get("api_usage", {})
        if isinstance(api_usage, dict):
            self._restore_api_usage(api_usage)
            total_tokens = api_usage.get("total_tokens")
            if isinstance(total_tokens, int):
                self.stats.total_tokens_used = total_tokens
            else:
                self.stats.total_tokens_used = (
                    self.api_usage["total_input_tokens"] + self.api_usage["total_output_tokens"]
                )

        quality_issues = meta.get("quality_issues", {})
        if isinstance(quality_issues, dict):
            self._restore_quality_issues(quality_issues)

        self.logger.info(
            json.dumps(
                {
                    "timestamp": datetime.now().isoformat(),
                    "level": "INFO",
                    "event": "checkpoint_loaded",
                    "checkpoint_path": str(checkpoint_path),
                    "last_completed_slice_id": last_slice.id,
                    "last_completed_slice_order": last_slice.order,
                    "processed_slices": self.stats.processed_slices,
                    "total_nodes": self.stats.total_nodes,
                    "total_edges": self.stats.total_edges,
                }
            )
        )

        return last_order

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
        Process single slice with retry mechanism for TimeoutError and JSON errors.

        Args:
            slice_file: Path to slice file

        Returns:
            True if successful, False on critical error (will stop processing)
        """
        # Load slice data
        try:
            slice_data = self._load_slice(slice_file)
        except Exception as e:
            self.logger.error(f"Failed to load slice {slice_file}: {e}")
            return False

        slice_id = slice_data.id
        slice_token_start = slice_data.slice_token_start
        slice_order = slice_data.order

        # Log processing start
        self.logger.info(
            json.dumps(
                {
                    "timestamp": datetime.now().isoformat(),
                    "level": "INFO",
                    "event": "slice_start",
                    "slice_id": slice_id,
                    "order": slice_order,
                    "slice_token_start": slice_token_start,
                    "total": self.stats.total_slices,
                }
            )
        )

        # Format input with FULL ConceptDictionary
        input_data = self._format_slice_input(slice_data)

        # Get max_retries from config
        max_retries = self.config.get("max_retries", 3)
        last_error_type = None
        last_quality_issues: List[Dict] = []
        start_time = time.time()

        # Retry loop for recoverable errors
        for attempt in range(max_retries + 1):  # 0, 1, 2, 3 = first try + 3 retries
            try:
                if attempt == 0:
                    # First attempt - normal request
                    response_text, response_id, usage = self.llm_client.create_response(
                        instructions=self.extraction_prompt,
                        input_data=input_data,
                        previous_response_id=self.previous_response_id,
                    )
                else:
                    # Retry through repair
                    current_time = datetime.now().strftime("%H:%M:%S")
                    print(
                        f"[{current_time}] REPAIR   | 🔧 Attempt {attempt}/{max_retries} "
                        f"after {last_error_type}..."
                    )

                    # Add hint based on error type
                    repair_hint = ""
                    if last_error_type == "json":
                        repair_hint = (
                            "\nCRITICAL: Return ONLY a valid JSON object. "
                            "No markdown formatting, no explanations, "
                            "no text outside the JSON structure."
                        )
                    elif last_error_type == "timeout":
                        repair_hint = (
                            "\nIMPORTANT: Be concise to avoid timeout. "
                            "Focus on essential nodes and edges only."
                        )
                    elif last_error_type == "quality":
                        repair_hint = self._format_quality_repair_hint(last_quality_issues)

                    response_text, response_id, usage = self.llm_client.repair_response(
                        instructions=self.extraction_prompt + repair_hint,
                        input_data=input_data,
                        previous_response_id=self.previous_response_id,
                    )

                # Track API usage
                self.api_usage["total_requests"] += 1
                self.api_usage["total_input_tokens"] += usage.input_tokens
                self.api_usage["total_output_tokens"] += usage.output_tokens
                self.stats.total_tokens_used += usage.total_tokens

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

                # Try to parse JSON
                success, parsed = self._process_llm_response(response_text, slice_id)

                if not success:
                    self._save_graph_patch_artifact(
                        slice_data=slice_data,
                        input_data=input_data,
                        response_text=response_text,
                        response_id=response_id,
                        attempt=attempt,
                        stage="json_failed",
                        error="JSON validation failed",
                    )
                    last_error_type = "json"
                    if attempt == max_retries:
                        # CRITICAL: graph cannot continue without slice
                        self._save_bad_response(
                            slice_id,
                            response_text,
                            f"JSON validation failed after {max_retries} retries",
                        )
                        current_time = datetime.now().strftime("%H:%M:%S")
                        print(
                            f"[{current_time}] ERROR    | ❌ "
                            f"{slice_order:03d}/{self.stats.total_slices:03d} | "
                            f"{slice_id} | JSON validation failed after "
                            f"{max_retries} retries"
                        )
                        print(
                            f"[{current_time}] FAILED   | ❌ Cannot continue without slice "
                            f"{slice_id}"
                        )
                        self.logger.error(
                            f"JSON validation failed for slice {slice_id} after "
                            f"{max_retries} retries"
                        )
                        self._save_temp_dumps(f"critical_slice_failure_{slice_id}")
                        return False  # Will lead to EXIT_RUNTIME_ERROR
                    continue  # Next attempt

                # JSON is valid, apply post-processing to fix IDs
                if parsed and "chunk_graph_patch" in parsed:
                    patch = parsed["chunk_graph_patch"]

                    # NEW: Replace temporary IDs with final position-based IDs
                    self._assign_final_ids(patch, slice_data)
                    self._ensure_referenced_concept_nodes(patch)

                    quality_ok, quality_issues = self._validate_patch_quality(patch, slice_data)
                    if not quality_ok:
                        self.quality_issues["quality_repair_requests"] += 1
                        last_error_type = "quality"
                        last_quality_issues = quality_issues
                        self._save_graph_patch_artifact(
                            slice_data=slice_data,
                            input_data=input_data,
                            response_text=response_text,
                            response_id=response_id,
                            attempt=attempt,
                            stage="quality_failed",
                            parsed_response=parsed,
                            postprocessed_patch=patch,
                            quality_issues=quality_issues,
                        )

                        if attempt == max_retries:
                            current_time = datetime.now().strftime("%H:%M:%S")
                            print(
                                f"[{current_time}] ERROR    | ❌ "
                                f"{slice_order:03d}/{self.stats.total_slices:03d} | "
                                f"{slice_id} | Graph quality failed after {max_retries} retries"
                            )
                            self.logger.error(
                                f"Graph quality failed for slice {slice_id}: {quality_issues}"
                            )
                            self._save_temp_dumps(f"quality_failure_{slice_id}")
                            return False

                        continue

                    self._save_graph_patch_artifact(
                        slice_data=slice_data,
                        input_data=input_data,
                        response_text=response_text,
                        response_id=response_id,
                        attempt=attempt,
                        stage="accepted",
                        parsed_response=parsed,
                        postprocessed_patch=patch,
                        quality_issues=quality_issues,
                    )

                    # Success! Confirm response only after JSON and quality validation.
                    self.llm_client.confirm_response()

                    # Position validation and repair are no longer needed - IDs are now correct
                    self._add_to_graph(patch, slice_data)

                    # Intermediate validation
                    if not self._validate_graph_intermediate():
                        self.logger.error(f"Intermediate validation failed after {slice_id}")
                        self._save_temp_dumps(f"validation_error_{slice_id}")
                        return False

                    # Update previous_response_id ONLY on success
                    self.previous_response_id = response_id
                    self.last_completed_slice = slice_data

                    # Persist the recoverable graph state after the slice is fully integrated.
                    processed_slices = self.stats.processed_slices + 1
                    try:
                        self._save_success_checkpoint(slice_data, processed_slices)
                    except Exception as e:
                        self.logger.error(f"Failed to save checkpoint after {slice_id}: {e}")
                        current_time = datetime.now().strftime("%H:%M:%S")
                        print(
                            f"[{current_time}] ERROR    | ❌ "
                            f"{slice_order:03d}/{self.stats.total_slices:03d} | "
                            f"{slice_id} | Checkpoint save failed"
                        )
                        self._save_temp_dumps(f"checkpoint_failure_{slice_id}")
                        return False

                    # Log success
                    elapsed = int(time.time() - start_time)
                    current_time = datetime.now().strftime("%H:%M:%S")
                    print(
                        f"[{current_time}] SLICE    | ✅ "
                        f"{slice_order:03d}/{self.stats.total_slices} | "
                        f"tokens_used={self._format_tokens(self.stats.total_tokens_used)} | "
                        f"{elapsed}s | nodes={self.stats.total_nodes} | "
                        f"edges={self.stats.total_edges}"
                    )

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

            except TimeoutError as e:
                last_error_type = "timeout"
                current_time = datetime.now().strftime("%H:%M:%S")
                if attempt == max_retries:
                    # CRITICAL: graph cannot continue without slice
                    print(
                        f"[{current_time}] ERROR    | ❌ "
                        f"{slice_order:03d}/{self.stats.total_slices:03d} | "
                        f"{slice_id} | Timeout after {max_retries} retries"
                    )
                    print(
                        f"[{current_time}] FAILED   | ❌ Cannot continue without slice {slice_id}"
                    )
                    self.logger.error(
                        f"Timeout processing slice {slice_id} after {max_retries} retries: {e}"
                    )
                    self._save_temp_dumps(f"timeout_failure_{slice_id}")
                    return False  # Will lead to EXIT_RUNTIME_ERROR
                # Wait before next attempt (30s, 60s, 90s)
                wait_time = 30 * (attempt + 1)
                print(
                    f"[{current_time}] REPAIR   | ⏳ Timeout occurred, waiting {wait_time}s "
                    f"before retry..."
                )
                time.sleep(wait_time)

            except (openai.RateLimitError, openai.APIError) as e:
                # These are handled in llm_client with their own retry
                self.logger.error(f"API error processing slice {slice_id}: {e}")
                current_time = datetime.now().strftime("%H:%M:%S")
                print(
                    f"[{current_time}] ERROR    | ❌ "
                    f"{slice_order:03d}/{self.stats.total_slices:03d} | "
                    f"{slice_id} | API error: {type(e).__name__}"
                )
                if self.previous_response_id:
                    print(
                        f"[{current_time}] FAILED   | ❌ Cannot continue with response chain "
                        f"{self.previous_response_id}"
                    )
                self._save_temp_dumps(f"api_error_{slice_id}")
                return False  # Will lead to EXIT_RUNTIME_ERROR

            except Exception as e:
                # Unexpected errors - CRITICAL for graph
                current_time = datetime.now().strftime("%H:%M:%S")
                self.logger.error(f"Unexpected error processing slice {slice_id}: {e}")
                print(
                    f"[{current_time}] ERROR    | ❌ "
                    f"{slice_order:03d}/{self.stats.total_slices:03d} | "
                    f"{slice_id} | Unexpected error: {type(e).__name__}"
                )
                print(f"[{current_time}] FAILED   | ❌ Cannot continue without slice {slice_id}")
                self._save_temp_dumps(f"unexpected_error_{slice_id}")
                return False  # Will lead to EXIT_RUNTIME_ERROR

        # Should never reach here
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
            f"[{timestamp}] START    | {self.stats.total_slices} slices | "
            f"model={model} | tpm={tpm_limit // 1000}k"
        )

        slice_files_to_process = slice_files
        resume_from_latest = self.config.get("resume_from_latest", False)
        latest_checkpoint_path = OUTPUT_DIR / LATEST_CHECKPOINT_FILENAME
        should_resume = resume_from_latest is True or (
            resume_from_latest == "auto" and latest_checkpoint_path.exists()
        )
        if resume_from_latest == "auto" and not latest_checkpoint_path.exists():
            print(f"[{timestamp}] RESUME   | auto | no checkpoint found, starting fresh")

        if should_resume:
            try:
                last_completed_order = self._load_latest_checkpoint(slice_files)
                slice_files_to_process = self._filter_slice_files_after_order(
                    slice_files, last_completed_order
                )
            except ValueError as e:
                self.logger.error(f"Failed to resume from latest checkpoint: {e}")
                print(f"ERROR: {e}")
                return EXIT_INPUT_ERROR
            except Exception as e:
                self.logger.error(f"Unexpected resume error: {e}")
                print(f"ERROR: Failed to resume from latest checkpoint: {e}")
                return EXIT_INPUT_ERROR

            timestamp = datetime.now().strftime("%H:%M:%S")
            print(
                f"[{timestamp}] RESUME   | latest checkpoint | "
                f"last={self.last_completed_slice.id if self.last_completed_slice else 'unknown'} | "
                f"processed={self.stats.processed_slices}/{self.stats.total_slices} | "
                f"remaining={len(slice_files_to_process)}"
            )

        # Process each slice
        for slice_file in slice_files_to_process:
            success = self._process_single_slice(slice_file)

            if success:
                self.stats.processed_slices += 1
            else:
                self.stats.failed_slices += 1
                # Error messages already printed in _process_single_slice
                # Graph processing cannot continue with missing slices
                return EXIT_RUNTIME_ERROR

        # Check if any slices were processed
        if self.stats.processed_slices == 0:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] FAILED   | ❌ All slices failed processing")
            return EXIT_RUNTIME_ERROR

        # Validate node uniqueness before saving
        graph_data = {"nodes": self.graph_nodes, "edges": self.graph_edges}
        is_unique, duplicates = self._validate_node_uniqueness(graph_data)
        if not is_unique:
            self.logger.error(f"Graph contains duplicate node IDs: {duplicates}")
            # Continue but mark in metadata

        # Save results
        output_path = OUTPUT_DIR / "LearningChunkGraph_raw.json"
        try:
            output_data = self._build_output_data(
                complete=True,
                processed_slices=self.stats.processed_slices,
                checkpoint_slice=self.last_completed_slice,
                is_unique=is_unique,
                duplicates=duplicates,
            )

            # Validate basic structure before saving
            if not self.graph_nodes:
                self.logger.warning("No nodes in final graph")
            if not self.graph_edges:
                self.logger.warning("No edges in final graph")

            self._write_json_atomic(output_path, output_data)

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

#!/usr/bin/env python3
"""
iText2KG - incremental knowledge graph construction from educational texts.

Utility sequentially processes slices from staging, sends them to LLM
while preserving context through previous_response_id, and incrementally builds
ConceptDictionary and LearningChunkGraph.
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
from typing import Any, Dict, List, Optional, Set, Tuple

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
from src.utils.validation import (
    GraphInvariantError,
    ValidationError,
    validate_concept_dictionary_invariants,
    validate_graph_invariants_intermediate,
    validate_json,
)

setup_console_encoding()

# Constants
CONFIG_PATH = Path(__file__).parent / "config.toml"
PROMPTS_DIR = Path(__file__).parent / "prompts"
SCHEMAS_DIR = Path(__file__).parent / "schemas"
STAGING_DIR = Path(__file__).parent.parent / "data" / "staging"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "out"
LOGS_DIR = Path(__file__).parent.parent / "logs"

EXTRACTION_PROMPT_FILE = "itext2kg_extraction.md"
MAX_REPAIR_ATTEMPTS = 1


@dataclass
class ProcessingStats:
    """Slice processing statistics."""

    total_slices: int = 0
    processed_slices: int = 0
    failed_slices: int = 0
    total_concepts: int = 0
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
        self.llm_client = OpenAIClient(self.config)
        self.logger = self._setup_logger()
        self.stats = ProcessingStats()

        # Data accumulators
        self.concept_dictionary: Dict[str, List[Dict]] = {"concepts": []}
        self.learning_graph: Dict[str, List[Dict]] = {"nodes": [], "edges": []}
        self.known_node_ids: Set[str] = set()  # For fast existence check
        self.concept_id_map: Dict[str, int] = {}  # concept_id -> index in concepts

        # Load prompt and schemas
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
        logger = logging.getLogger("itext2kg")
        logger.setLevel(getattr(logging, self.config["log_level"].upper()))

        # File handler
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = LOGS_DIR / f"itext2kg_{timestamp}.log"
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

    def _load_extraction_prompt(self) -> str:
        """Load prompt with schema substitution."""
        prompt_path = PROMPTS_DIR / EXTRACTION_PROMPT_FILE
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        prompt_content = prompt_path.read_text(encoding="utf-8")

        # Load schemas for substitution
        concept_schema_path = SCHEMAS_DIR / "ConceptDictionary.schema.json"
        graph_schema_path = SCHEMAS_DIR / "LearningChunkGraph.schema.json"

        concept_schema = json.loads(concept_schema_path.read_text(encoding="utf-8"))
        graph_schema = json.loads(graph_schema_path.read_text(encoding="utf-8"))

        # Substitute schemas into prompt
        prompt_content = prompt_content.replace(
            "{concept_dictionary_schema}", json.dumps(concept_schema, indent=2)
        )
        prompt_content = prompt_content.replace(
            "{learning_chunk_graph_schema}", json.dumps(graph_schema, indent=2)
        )

        return prompt_content

    def _load_slice(self, slice_file: Path) -> SliceData:
        """
        Load slice data from file.

        Args:
            slice_file: Path to slice file

        Returns:
            SliceData object

        Raises:
            json.JSONDecodeError: If file contains invalid JSON
        """
        try:
            data = json.loads(slice_file.read_text(encoding="utf-8"))
            return SliceData(
                id=data["id"],
                order=data["order"],
                source_file=data["source_file"],
                slug=data["slug"],
                text=data["text"],
                slice_token_start=data["slice_token_start"],
                slice_token_end=data["slice_token_end"],
            )
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid slice file {slice_file}: {e}")

    def _format_slice_input(self, slice_data: SliceData) -> str:
        """
        Format input data for LLM.

        Args:
            slice_data: Slice data

        Returns:
            JSON string with ConceptDictionary and Slice
        """
        input_data = {
            "ConceptDictionary": self.concept_dictionary,
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

        return json.dumps(input_data, ensure_ascii=False, indent=2)

    def _update_concept_dictionary(self, concepts_added: List[Dict]) -> None:
        """
        Incremental ConceptDictionary update.

        Args:
            concepts_added: List of new/updated concepts from LLM response
        """
        for new_concept in concepts_added:
            concept_id = new_concept["concept_id"]

            if concept_id in self.concept_id_map:
                # Concept exists - update only aliases
                idx = self.concept_id_map[concept_id]
                existing_concept = self.concept_dictionary["concepts"][idx]

                # Create dictionary of existing aliases (lowercase -> original)
                existing_aliases = existing_concept["term"].get("aliases", [])
                existing_lower_map = {alias.lower(): alias for alias in existing_aliases}

                # Check new aliases
                new_aliases = new_concept["term"].get("aliases", [])
                added_aliases = []

                for new_alias in new_aliases:
                    # Check case-insensitive
                    if new_alias.lower() not in existing_lower_map:
                        existing_lower_map[new_alias.lower()] = new_alias
                        added_aliases.append(new_alias)

                if added_aliases:
                    # Update aliases list (take values - original strings)
                    existing_concept["term"]["aliases"] = sorted(existing_lower_map.values())

                    # Log update
                    self.logger.debug(
                        json.dumps(
                            {
                                "timestamp": datetime.now().isoformat(),
                                "level": "DEBUG",
                                "event": "concept_update",
                                "concept_id": concept_id,
                                "action": "added_aliases",
                                "new_aliases": sorted(added_aliases),
                            }
                        )
                    )
            else:
                # New concept - clean aliases from case-insensitive duplicates
                aliases = new_concept.get("term", {}).get("aliases", [])
                if aliases:
                    # Remove duplicates, keeping first occurrence
                    seen_lower = {}
                    unique_aliases = []
                    for alias in aliases:
                        alias_lower = alias.lower()
                        if alias_lower not in seen_lower:
                            seen_lower[alias_lower] = True
                            unique_aliases.append(alias)

                    new_concept["term"]["aliases"] = unique_aliases

                # Add concept
                self.concept_dictionary["concepts"].append(new_concept)
                self.concept_id_map[concept_id] = len(self.concept_dictionary["concepts"]) - 1
                self.stats.total_concepts += 1

                # Log addition
                self.logger.debug(
                    json.dumps(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "level": "DEBUG",
                            "event": "concept_added",
                            "concept_id": concept_id,
                        }
                    )
                )

    def _process_chunk_nodes(self, new_nodes: List[Dict]) -> List[Dict]:
        """
        Process Chunk and Assessment nodes with overlap checking.

        Args:
            new_nodes: List of new nodes from patch

        Returns:
            List of nodes to add to graph
        """
        nodes_to_add = []

        for node in new_nodes:
            node_type = node.get("type")
            node_id = node.get("id")

            # Process Chunk and Assessment nodes with existing IDs
            if node_type in ("Chunk", "Assessment") and node_id in self.known_node_ids:
                # Find existing node
                existing_node = None
                for idx, existing in enumerate(self.learning_graph["nodes"]):
                    if existing["id"] == node_id:
                        existing_node = existing
                        existing_idx = idx
                        break

                if existing_node:
                    # For Chunk compare text length
                    if node_type == "Chunk":
                        if len(node.get("text", "")) > len(existing_node.get("text", "")):
                            # Update existing node
                            self.learning_graph["nodes"][existing_idx] = node

                            self.logger.debug(
                                json.dumps(
                                    {
                                        "timestamp": datetime.now().isoformat(),
                                        "level": "DEBUG",
                                        "event": "chunk_updated",
                                        "node_id": node_id,
                                        "old_length": len(existing_node.get("text", "")),
                                        "new_length": len(node.get("text", "")),
                                    }
                                )
                            )
                        else:
                            # Ignore shorter variant
                            self.logger.debug(
                                json.dumps(
                                    {
                                        "timestamp": datetime.now().isoformat(),
                                        "level": "DEBUG",
                                        "event": "chunk_ignored",
                                        "node_id": node_id,
                                        "reason": "shorter_duplicate",
                                    }
                                )
                            )
                    else:
                        # For Assessment just log and skip duplicate
                        self.logger.warning(
                            json.dumps(
                                {
                                    "timestamp": datetime.now().isoformat(),
                                    "level": "WARN",
                                    "event": "assessment_duplicate_ignored",
                                    "node_id": node_id,
                                }
                            )
                        )
            else:
                # New node - add
                nodes_to_add.append(node)
                if node_id:  # Protection from empty IDs
                    self.known_node_ids.add(node_id)

        return nodes_to_add

    def _validate_edges(self, edges: List[Dict]) -> List[Dict]:
        """
        Edge validation with node existence and duplicate checking.

        Args:
            edges: List of edges to check

        Returns:
            List of valid edges
        """
        valid_edges = []

        # Collect all known IDs (nodes + concepts)
        all_known_ids = self.known_node_ids.copy()
        all_known_ids.update(self.concept_id_map.keys())

        # Collect existing edges for duplicate checking
        existing_edges = set()
        for edge in self.learning_graph.get("edges", []):
            existing_edges.add((edge["source"], edge["target"], edge["type"]))

        # Also track edges within current patch
        patch_edges = set()

        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            edge_type = edge.get("type")

            # Check node existence
            if source not in all_known_ids or target not in all_known_ids:
                self.logger.warning(
                    json.dumps(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "level": "WARN",
                            "event": "edge_dropped",
                            "reason": "invalid_reference",
                            "source": source,
                            "target": target,
                        }
                    )
                )
                continue

            # Check PREREQUISITE self-loops
            if edge_type == "PREREQUISITE" and source == target:
                self.logger.warning(
                    json.dumps(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "level": "WARN",
                            "event": "edge_dropped",
                            "reason": "prerequisite_self_loop",
                            "node_id": source,
                        }
                    )
                )
                continue

            # Check weight
            weight = edge.get("weight", 0.5)
            if not (0 <= weight <= 1):
                self.logger.warning(
                    json.dumps(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "level": "WARN",
                            "event": "edge_dropped",
                            "reason": "invalid_weight",
                            "weight": weight,
                        }
                    )
                )
                continue

            # Check for duplicate
            edge_key = (source, target, edge_type)

            # Check against existing edges in graph
            if edge_key in existing_edges:
                self.logger.info(
                    json.dumps(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "level": "INFO",
                            "event": "edge_dropped",
                            "reason": "duplicate_edge",
                            "source": source,
                            "target": target,
                            "type": edge_type,
                        }
                    )
                )
                continue

            # Check against edges in current patch
            if edge_key in patch_edges:
                self.logger.info(
                    json.dumps(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "level": "INFO",
                            "event": "edge_dropped",
                            "reason": "duplicate_in_patch",
                            "source": source,
                            "target": target,
                            "type": edge_type,
                        }
                    )
                )
                continue

            # Add valid edge
            valid_edges.append(edge)
            patch_edges.add(edge_key)

        return valid_edges

    def _add_mentions_edges(self, chunk_nodes: List[Dict]) -> int:
        """
        Automatically add MENTIONS edges from Chunks to Concepts.

        Search for concept mentions (primary term and aliases) in chunk text
        according to rules:
        - Full word matches only (not substrings)
        - Case-insensitive
        - Exact forms only (no morphology)

        Args:
            chunk_nodes: List of Chunk type nodes for processing

        Returns:
            Number of added MENTIONS edges
        """
        if not self.concept_dictionary.get("concepts"):
            return 0

        edges_added = 0

        # Collect existing MENTIONS edges to avoid duplication
        existing_mentions = set()
        for edge in self.learning_graph.get("edges", []):
            if edge.get("type") == "MENTIONS":
                existing_mentions.add((edge["source"], edge["target"]))

        # Process each Chunk node
        for chunk in chunk_nodes:
            if chunk.get("type") != "Chunk":
                continue

            chunk_text = chunk.get("text", "")
            if not chunk_text:
                continue

            chunk_id = chunk["id"]
            chunk_text_lower = chunk_text.lower()

            # Check each concept
            for concept in self.concept_dictionary["concepts"]:
                concept_id = concept["concept_id"]

                # Skip if MENTIONS edge already exists
                if (chunk_id, concept_id) in existing_mentions:
                    continue

                # Collect all terms for search (primary + aliases)
                terms_to_search = []

                primary_term = concept.get("term", {}).get("primary")
                if primary_term:
                    terms_to_search.append(primary_term)

                aliases = concept.get("term", {}).get("aliases", [])
                terms_to_search.extend(aliases)

                # Search each term
                found = False
                for term in terms_to_search:
                    if not term:
                        continue

                    # Create regular expression for full word match
                    # \b - word boundary, works with Unicode
                    pattern = r"\b" + re.escape(term.lower()) + r"\b"

                    if re.search(pattern, chunk_text_lower):
                        found = True
                        break

                # If found mention - add MENTIONS edge
                if found:
                    mentions_edge = {
                        "source": chunk_id,
                        "target": concept_id,
                        "type": "MENTIONS",
                        "weight": 1.0,
                    }

                    self.learning_graph["edges"].append(mentions_edge)
                    existing_mentions.add((chunk_id, concept_id))
                    edges_added += 1

                    # Logging in DEBUG mode
                    if self.config["log_level"].lower() == "debug":
                        self.logger.debug(
                            json.dumps(
                                {
                                    "timestamp": datetime.now().isoformat(),
                                    "level": "DEBUG",
                                    "event": "mentions_edge_added",
                                    "source": chunk_id,
                                    "target": concept_id,
                                    "found_term": term,
                                }
                            )
                        )

        if edges_added > 0:
            self.stats.total_edges += edges_added

            # Informational logging
            self.logger.info(
                json.dumps(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "level": "INFO",
                        "event": "mentions_edges_added",
                        "count": edges_added,
                        "chunks_processed": len(chunk_nodes),
                    }
                )
            )

        return edges_added

    def _process_llm_response(
        self, response_text: str, slice_id: str
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Process and validate LLM response.

        Args:
            response_text: Text of LLM response
            slice_id: ID of current slice

        Returns:
            (success, parsed_data) - success and parsed data or None
        """
        try:
            # Proactive cleanup of known issues before parsing
            cleaned_text = response_text

            # 1. Fix HTML attributes with incorrect quotes
            # Pattern: attr='\"value\"' -> attr="value"
            cleaned_text = re.sub(
                r"(\b(?:href|src|target|action|name|frameborder|width|height|align))=\'\"([^\"]*?)\"\'",
                r'\1="\2"',
                cleaned_text,
            )

            # 2. Fix reverse case: attr="'value'"  -> attr="value"
            cleaned_text = re.sub(
                r'(\b(?:href|src|target|action|name|frameborder|width|height|align))="\'([^\']*?)\'"',
                r'\1="\2"',
                cleaned_text,
            )

            # Log if there were changes
            if cleaned_text != response_text:
                self.logger.debug(
                    json.dumps(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "level": "DEBUG",
                            "event": "response_cleaned",
                            "slice_id": slice_id,
                            "message": "Applied HTML attribute cleanup",
                        }
                    )
                )

            # Parse JSON
            response_data = json.loads(cleaned_text)

            # Check structure
            if "concepts_added" not in response_data or "chunk_graph_patch" not in response_data:
                raise ValueError("Missing required fields in response")

            concepts_added = response_data["concepts_added"].get("concepts", [])
            patch = response_data["chunk_graph_patch"]

            # Basic schema validation (structure only)
            validate_json({"concepts": concepts_added}, "ConceptDictionary")
            validate_json(patch, "LearningChunkGraph")

            return True, response_data

        except (json.JSONDecodeError, ValueError, ValidationError) as e:
            self.logger.error(
                json.dumps(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "level": "ERROR",
                        "event": "response_validation_failed",
                        "slice_id": slice_id,
                        "error": str(e),
                    }
                )
            )
            return False, None

    def _apply_patch(self, patch_data: Dict) -> Tuple[int, int]:
        """
        Apply patch to knowledge graph.

        Args:
            patch_data: Patch data with concepts_added and chunk_graph_patch

        Returns:
            (nodes_added, edges_added) - number of added nodes and edges
        """
        nodes_added = 0
        edges_added = 0

        # Update ConceptDictionary
        concepts_to_add = patch_data["concepts_added"].get("concepts", [])
        self._update_concept_dictionary(concepts_to_add)

        # Create Concept type nodes for new concepts from concepts_added
        for concept in concepts_to_add:
            concept_id = concept["concept_id"]
            # Check if such node doesn't already exist
            if concept_id not in self.known_node_ids:
                concept_node = {
                    "id": concept_id,
                    "type": "Concept",
                    "text": concept["term"]["primary"],
                    "definition": concept["definition"],
                    "local_start": 0,  # Concepts don't have text position, set to 0
                }
                self.learning_graph["nodes"].append(concept_node)
                self.known_node_ids.add(concept_id)
                nodes_added += 1

        # Process nodes from patch
        new_nodes = patch_data["chunk_graph_patch"].get("nodes", [])
        nodes_to_add = self._process_chunk_nodes(new_nodes)
        self.learning_graph["nodes"].extend(nodes_to_add)
        nodes_added += len(nodes_to_add)
        self.stats.total_nodes += nodes_added

        # Process edges
        new_edges = patch_data["chunk_graph_patch"].get("edges", [])
        valid_edges = self._validate_edges(new_edges)
        self.learning_graph["edges"].extend(valid_edges)
        edges_added = len(valid_edges)
        self.stats.total_edges += edges_added

        # Add automatic MENTIONS edges
        # Process both new nodes and updated existing ones
        chunk_nodes_to_process = []

        # New Chunk type nodes
        for node in nodes_to_add:
            if node.get("type") == "Chunk":
                chunk_nodes_to_process.append(node)

        # Updated nodes (from _process_chunk_nodes)
        for node in new_nodes:
            if node.get("type") == "Chunk" and node["id"] in self.known_node_ids:
                # Find node in graph (it could have been updated)
                for graph_node in self.learning_graph["nodes"]:
                    if graph_node["id"] == node["id"]:
                        chunk_nodes_to_process.append(graph_node)
                        break

        # Add MENTIONS edges
        mentions_added = self._add_mentions_edges(chunk_nodes_to_process)
        edges_added += mentions_added

        return nodes_added, edges_added

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
            "validation_error": error,
            "repair_response": repair_response,
        }

        bad_response_file.write_text(
            json.dumps(bad_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _save_temp_dumps(self, reason: str) -> None:
        """
        Save temporary dumps on critical errors.

        Args:
            reason: Save reason (validation_failed, io_error, etc.)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Paths for temporary files
        temp_concept_path = LOGS_DIR / f"ConceptDictionary_temp_{reason}_{timestamp}.json"
        temp_graph_path = LOGS_DIR / f"LearningChunkGraph_temp_{reason}_{timestamp}.json"

        # Save ConceptDictionary
        if self.concept_dictionary and self.concept_dictionary.get("concepts"):
            temp_concept_path.write_text(
                json.dumps(self.concept_dictionary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"Temporary ConceptDictionary saved to: {temp_concept_path}",
                file=sys.stderr,
            )

        # Save LearningChunkGraph
        if self.learning_graph and (
            self.learning_graph.get("nodes") or self.learning_graph.get("edges")
        ):
            temp_graph_path.write_text(
                json.dumps(self.learning_graph, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"Temporary LearningChunkGraph saved to: {temp_graph_path}",
                file=sys.stderr,
            )

        # Save processing statistics
        stats_path = LOGS_DIR / f"processing_stats_{reason}_{timestamp}.json"
        stats_data = {
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "stats": {
                "total_slices": self.stats.total_slices,
                "processed_slices": self.stats.processed_slices,
                "failed_slices": self.stats.failed_slices,
                "total_concepts": self.stats.total_concepts,
                "total_nodes": self.stats.total_nodes,
                "total_edges": self.stats.total_edges,
                "total_tokens_used": self.stats.total_tokens_used,
                "processing_time": str(datetime.now() - self.stats.start_time),
            },
        }
        stats_path.write_text(
            json.dumps(stats_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Processing stats saved to: {stats_path}", file=sys.stderr)

    def _process_single_slice(self, slice_file: Path) -> bool:
        """
        Process single slice.

        Args:
            slice_file: Path to slice file

        Returns:
            True if successful, False on error
        """
        try:
            # Load slice
            slice_data = self._load_slice(slice_file)

            # Log processing start
            self.logger.info(
                json.dumps(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "level": "INFO",
                        "event": "slice_start",
                        "slice_id": slice_data.id,
                        "order": slice_data.order,
                        "total": self.stats.total_slices,
                    }
                )
            )

            # Format input data
            input_data = self._format_slice_input(slice_data)

            # Call LLM
            start_time = time.time()

            # DEBUG log prompt
            if self.config["log_level"].lower() == "debug":
                self.logger.debug(
                    json.dumps(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "level": "DEBUG",
                            "event": "llm_request",
                            "slice_id": slice_data.id,
                            "prompt": self.extraction_prompt,
                            "input_data": json.loads(input_data),
                        }
                    )
                )

            try:
                response_text, response_id, usage = self.llm_client.create_response(
                    instructions=self.extraction_prompt, input_data=input_data
                )

                # DEBUG log response
                if self.config["log_level"].lower() == "debug":
                    self.logger.debug(
                        json.dumps(
                            {
                                "timestamp": datetime.now().isoformat(),
                                "level": "DEBUG",
                                "event": "llm_response",
                                "slice_id": slice_data.id,
                                "response": response_text,
                                "response_id": response_id,
                                "usage": {
                                    "input_tokens": usage.input_tokens,
                                    "output_tokens": usage.output_tokens,
                                    "reasoning_tokens": usage.reasoning_tokens,
                                },
                            }
                        )
                    )

                # Process response
                success, parsed_data = self._process_llm_response(response_text, slice_data.id)

                if not success:
                    # Repair attempt with clarifying prompt
                    self.logger.info(
                        json.dumps(
                            {
                                "timestamp": datetime.now().isoformat(),
                                "level": "INFO",
                                "event": "repair_attempt",
                                "slice_id": slice_data.id,
                            }
                        )
                    )

                    # Add console output
                    current_time = datetime.now().strftime("%H:%M:%S")
                    print(
                        f"[{current_time}] REPAIR   | 🔧 Attempting to fix JSON validation error..."
                    )
                    print(
                        f"[{current_time}] REPAIR   | 📝 Adding clarification to prompt and retrying..."
                    )

                    # Form repair prompt with clarification
                    repair_instructions = (
                        f"{self.extraction_prompt}\n\n"
                        "IMPORTANT: Your previous response was not valid JSON or did not match the required schema. "
                        "Please ensure your response is EXACTLY one valid JSON object with the structure shown above. "
                        "Do not include any text before or after the JSON object."
                    )

                    # repair_response automatically uses saved previous_response_id
                    repair_text, repair_id, repair_usage = self.llm_client.repair_response(
                        instructions=repair_instructions, input_data=input_data
                    )

                    success, parsed_data = self._process_llm_response(repair_text, slice_data.id)

                    if success:
                        # Repair successful
                        current_time = datetime.now().strftime("%H:%M:%S")
                        print(f"[{current_time}] REPAIR   | ✅ JSON validation fixed successfully!")
                    else:
                        # Save bad responses
                        self._save_bad_response(
                            slice_data.id,
                            response_text,
                            "JSON validation failed after repair",
                            repair_text,
                        )

                        self.logger.error(
                            json.dumps(
                                {
                                    "timestamp": datetime.now().isoformat(),
                                    "level": "ERROR",
                                    "event": "slice_failed",
                                    "slice_id": slice_data.id,
                                    "error": "JSON validation failed after repair",
                                }
                            )
                        )

                        # Output error to terminal
                        current_time = datetime.now().strftime("%H:%M:%S")
                        print(
                            f"[{current_time}] ERROR    | ❌ {slice_data.order:03d}/{self.stats.total_slices:03d} | "
                            f"{slice_data.id} | JSON validation failed after repair"
                        )

                        return False

                    # Repair successful - use repair usage
                    usage = repair_usage

                # Apply patch
                nodes_added, edges_added = self._apply_patch(parsed_data)

                # Incremental validation after patch application
                try:
                    validate_graph_invariants_intermediate(self.learning_graph)
                    validate_concept_dictionary_invariants(self.concept_dictionary)
                except (ValidationError, GraphInvariantError) as e:
                    self.logger.error(
                        json.dumps(
                            {
                                "timestamp": datetime.now().isoformat(),
                                "level": "ERROR",
                                "event": "incremental_validation_failed",
                                "slice_id": slice_data.id,
                                "error": str(e),
                            }
                        )
                    )

                    # Save state for debugging
                    self._save_temp_dumps(f"validation_error_slice_{slice_data.id}")

                    # Output to console
                    current_time = datetime.now().strftime("%H:%M:%S")
                    print(
                        f"[{current_time}] ERROR    | ❌ Incremental validation failed for {slice_data.id}"
                    )
                    print(f"[{current_time}] ERROR    | 📋 Error: {str(e)[:100]}...")

                    # Don't crash immediately, mark slice as failed
                    return False

                # Update statistics
                self.stats.total_tokens_used += usage.total_tokens
                duration_sec = round(time.time() - start_time, 0)
                duration_ms = int((time.time() - start_time) * 1000)

                # Output progress to terminal
                current_time = datetime.now().strftime("%H:%M:%S")

                # Format token information
                tokens_info = f"tokens_used={self._format_tokens(self.stats.total_tokens_used)} | tokens_current={self._format_tokens(usage.total_tokens)}"
                if usage.reasoning_tokens > 0:
                    tokens_info += f" incl. reasoning={self._format_tokens(usage.reasoning_tokens)}"

                print(
                    f"[{current_time}] SLICE    | ✅ {slice_data.order:03d}/{self.stats.total_slices:03d} | "
                    f"{tokens_info} | {duration_sec}s | "
                    f"concepts={len(self.concept_dictionary['concepts'])} | "
                    f"nodes={self.stats.total_nodes} | edges={self.stats.total_edges}"
                )

                # Log success
                self.logger.info(
                    json.dumps(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "level": "INFO",
                            "event": "slice_success",
                            "slice_id": slice_data.id,
                            "tokens_used": usage.total_tokens,
                            "duration_ms": duration_ms,
                            "concepts_total": len(self.concept_dictionary["concepts"]),
                            "nodes_added": nodes_added,
                            "edges_added": edges_added,
                        }
                    )
                )

                return True

            except Exception as e:
                # Handle API errors
                error_type = type(e).__name__

                # IMPORTANT: Reset variables to avoid undefined
                response_text = None
                response_id = None
                usage = None

                # Output error to terminal
                current_time = datetime.now().strftime("%H:%M:%S")

                # Special handling for rate limit
                if "rate" in str(e).lower() or error_type == "RateLimitError":
                    # LLM client will already handle retry with backoff
                    print(f"[{current_time}] ERROR    | ⚠️ {error_type} | waiting for retry...")
                else:
                    print(f"[{current_time}] ERROR    | ⚠️ {error_type} | slice {slice_data.id}")

                self.logger.error(
                    json.dumps(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "level": "ERROR",
                            "event": "api_error",
                            "slice_id": slice_data.id,
                            "error_type": error_type,
                            "error": str(e),
                        }
                    )
                )

                # If all retries exhausted, consider slice failed
                return False

        except Exception as e:
            # General processing error
            self.logger.error(
                json.dumps(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "level": "ERROR",
                        "event": "slice_processing_error",
                        "slice_file": str(slice_file),
                        "error": str(e),
                    }
                )
            )
            return False

    def run(self) -> int:
        """
        Main processing method.

        Returns:
            Program exit code
        """
        try:
            # Check for slices
            slice_files = sorted(STAGING_DIR.glob("*.slice.json"))
            if not slice_files:
                self.logger.error("No slice files found in staging directory")
                return EXIT_INPUT_ERROR

            self.stats.total_slices = len(slice_files)

            # Output initial status
            self._print_start_status()

            # Process slices
            for slice_file in slice_files:
                try:
                    success = self._process_single_slice(slice_file)
                    if success:
                        self.stats.processed_slices += 1
                    else:
                        self.stats.failed_slices += 1

                    # Log intermediate progress
                    if self.stats.processed_slices % 10 == 0 and self.stats.processed_slices > 0:
                        self.logger.info(
                            json.dumps(
                                {
                                    "timestamp": datetime.now().isoformat(),
                                    "level": "INFO",
                                    "event": "progress_checkpoint",
                                    "processed": self.stats.processed_slices,
                                    "failed": self.stats.failed_slices,
                                    "total": self.stats.total_slices,
                                }
                            )
                        )

                except KeyboardInterrupt:
                    # Handle user interruption
                    self.logger.warning("Processing interrupted by user")

                    # Save intermediate results
                    if self.stats.processed_slices > 0:
                        self.logger.info(
                            f"Processed {self.stats.processed_slices}/{self.stats.total_slices} slices before interruption"
                        )
                        try:
                            self._save_temp_dumps("interrupted")
                            self.logger.info("Partial results saved to logs directory")
                        except Exception as e:
                            self.logger.error(f"Failed to save partial results: {e}")

                    return EXIT_RUNTIME_ERROR

                except Exception as e:
                    # Unexpected error processing slice
                    self.logger.error(f"Unexpected error processing {slice_file}: {e}")
                    self.stats.failed_slices += 1
                    # Continue processing remaining slices

            # Check results after processing all slices
            if self.stats.processed_slices == 0:
                self.logger.error("All slices failed processing")

                # Output error status
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"[{current_time}] FAILED   | ❌ All slices failed processing")
                print(f"[{current_time}] SAVING   | 💾 Attempting to save empty structures...")

                # Try to save at least empty structures
                try:
                    self._save_temp_dumps("all_failed")
                    print(
                        f"[{current_time}] INFO     | Check /logs/ for temporary files and diagnostics"
                    )
                except Exception as dump_error:
                    current_time = datetime.now().strftime("%H:%M:%S")
                    print(
                        f"[{current_time}] ERROR    | ⚠️ Failed to save temp dumps: {dump_error}",
                        file=sys.stderr,
                    )

                return EXIT_RUNTIME_ERROR

            # Warning if some slices failed
            if self.stats.failed_slices > 0:
                failure_rate = self.stats.failed_slices / self.stats.total_slices
                self.logger.warning(
                    f"Partial failure: {self.stats.failed_slices}/{self.stats.total_slices} slices failed ({failure_rate:.1%})"
                )

                # If more than 50% failed - warn
                if failure_rate > 0.5:
                    self.logger.warning(
                        f"High failure rate ({failure_rate:.1%}) - results may be incomplete"
                    )

            # Final validation and save
            return self._finalize_and_save()

        except Exception as e:
            # Critical error
            self.logger.error(f"Critical error in run(): {e}")

            # Output error status
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"[{current_time}] FAILED   | ❌ Critical error: {str(e)[:50]}...")
            print(f"[{current_time}] SAVING   | 💾 Emergency dump of current state...")

            # Last attempt to save data
            try:
                self._save_temp_dumps("critical_error")
                print(
                    f"[{current_time}] INFO     | Check /logs/ for temporary files and diagnostics"
                )
            except Exception as dump_error:
                current_time = datetime.now().strftime("%H:%M:%S")
                print(
                    f"[{current_time}] ERROR    | ⚠️ Failed to save emergency dumps: {dump_error}",
                    file=sys.stderr,
                )

            return EXIT_RUNTIME_ERROR

    def _print_start_status(self):
        """Output initial status to terminal."""
        current_time = datetime.now().strftime("%H:%M:%S")
        print(
            f"[{current_time}] START    | {self.stats.total_slices} slices | "
            f"model={self.config['model']} | tpm={self.config['tpm_limit']//1000}k"
        )

    def _finalize_and_save(self) -> int:
        """
        Final validation and save results.

        Returns:
            Exit code
        """
        try:
            # Schema validation
            validate_json(self.concept_dictionary, "ConceptDictionary")
            validate_json(self.learning_graph, "LearningChunkGraph")

            # Invariant validation
            # Use intermediate validation as there may be concept duplicates
            validate_concept_dictionary_invariants(self.concept_dictionary)
            validate_graph_invariants_intermediate(self.learning_graph)

            # Save files
            concept_path = OUTPUT_DIR / "ConceptDictionary.json"
            graph_path = OUTPUT_DIR / "LearningChunkGraph_raw.json"

            concept_path.write_text(
                json.dumps(self.concept_dictionary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            graph_path.write_text(
                json.dumps(self.learning_graph, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            # Output final status
            self._print_end_status()

            # Output saved files information
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"[{current_time}] SUCCESS  | ✅ Results saved to /data/out/")
            print("                           | - ConceptDictionary.json")
            print("                           | - LearningChunkGraph_raw.json")

            return EXIT_SUCCESS

        except (ValidationError, GraphInvariantError) as e:
            self.logger.error(f"Validation failed: {e}")

            # Output error status
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"[{current_time}] FAILED   | ❌ Validation failed: {str(e)[:50]}...")
            print(f"[{current_time}] SAVING   | 💾 Attempting to save partial results...")

            # Try to save temporary files
            try:
                self._save_temp_dumps("validation_failed")
                print(
                    f"[{current_time}] INFO     | Check /logs/ for temporary files and diagnostics"
                )
            except Exception as dump_error:
                current_time = datetime.now().strftime("%H:%M:%S")
                print(
                    f"[{current_time}] ERROR    | ⚠️ Failed to save temp dumps: {dump_error}",
                    file=sys.stderr,
                )

            return EXIT_RUNTIME_ERROR

        except Exception as e:
            self.logger.error(f"Failed to save output files: {e}")

            # Try to save temporary files
            try:
                self._save_temp_dumps("io_error")
            except Exception as dump_error:
                current_time = datetime.now().strftime("%H:%M:%S")
                print(
                    f"[{current_time}] ERROR    | ⚠️ Failed to save temp dumps: {dump_error}",
                    file=sys.stderr,
                )

            return EXIT_IO_ERROR

    def _print_end_status(self):
        """Output final status to terminal."""
        current_time = datetime.now().strftime("%H:%M:%S")
        duration = datetime.now() - self.stats.start_time
        minutes, seconds = divmod(int(duration.total_seconds()), 60)

        print(
            f"[{current_time}] END      | Done | slices={self.stats.processed_slices} | "
            f"time={minutes}m {seconds}s"
        )


def main():
    """Program entry point."""
    try:
        # Load configuration
        config = load_config(CONFIG_PATH)

        # Validate max_context_tokens parameters
        max_context = config["itext2kg"].get("max_context_tokens", 128000)
        if not isinstance(max_context, int) or max_context < 1000:
            raise ValueError(f"Invalid max_context_tokens: {max_context}. Must be integer >= 1000")

        max_context_test = config["itext2kg"].get("max_context_tokens_test", 128000)
        if not isinstance(max_context_test, int) or max_context_test < 1000:
            raise ValueError(
                f"Invalid max_context_tokens_test: {max_context_test}. Must be integer >= 1000"
            )

        # Create and run processor
        processor = SliceProcessor(config)
        return processor.run()

    except FileNotFoundError:
        return EXIT_CONFIG_ERROR
    except ValueError as e:
        print(f"Configuration error: {e}")
        return EXIT_CONFIG_ERROR
    except Exception:
        return EXIT_CONFIG_ERROR


if __name__ == "__main__":
    sys.exit(main())

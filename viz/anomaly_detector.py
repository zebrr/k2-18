#!/usr/bin/env python3
"""
Anomaly detection utility for enriched knowledge graphs.
Validates computed metrics, detects anomalies, and generates reports.
"""

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tomli

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


class AnomalyDetector:
    """Main class for detecting anomalies in knowledge graph metrics."""

    # Required node metrics (10 metrics)
    REQUIRED_NODE_METRICS = [
        "degree_in",
        "degree_out",
        "degree_centrality",
        "pagerank",
        "betweenness_centrality",
        "out-closeness",  # Correct name from spec
        "component_id",
        "prerequisite_depth",
        "learning_effort",
        "educational_importance",
    ]

    # Optional metrics
    OPTIONAL_NODE_METRICS = ["cluster_id", "bridge_score"]

    def __init__(self, config_path: Path = Path("viz/config.toml")):
        """Initialize detector with configuration."""
        self.config = self._load_config(config_path)
        self.graph_data: Optional[Dict] = None
        self.graph_file: Optional[str] = None

        # Counters for issues
        self.critical_issues: List[Dict] = []
        self.warnings: List[Dict] = []
        self.info_messages: List[Dict] = []

        # Statistics storage
        self.statistics: Dict[str, Dict] = {}

    def _load_config(self, config_path: Path) -> Dict:
        """Load configuration from TOML file."""
        if not config_path.exists():
            self._log("ERROR", f"Config file not found: {config_path}", error=True)
            sys.exit(1)

        with open(config_path, "rb") as f:
            config = tomli.load(f)

        # Extract anomaly_detection section with defaults
        ad_config = config.get("anomaly_detection", {})
        return {
            "pagerank_sum_tolerance": ad_config.get("pagerank_sum_tolerance", 0.01),
            "educational_sum_tolerance": ad_config.get("educational_sum_tolerance", 0.01),
            "min_modularity": ad_config.get("min_modularity", 0.1),
            "min_bridge_correlation": ad_config.get("min_bridge_correlation", 0.3),
            "outlier_method": ad_config.get("outlier_method", "iqr"),
            "outlier_threshold": ad_config.get("outlier_threshold", 1.5),
            "strict_mode": ad_config.get("strict_mode", False),
            "save_json_report": ad_config.get("save_json_report", True),
        }

    def _log(self, level: str, message: str, error: bool = False):
        """Print formatted log message."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Color mapping
        colors = {
            "START": Fore.CYAN + Style.BRIGHT,
            "SUCCESS": Fore.GREEN + Style.BRIGHT,
            "ERROR": Fore.RED + Style.BRIGHT,
            "WARNING": Fore.YELLOW,
            "INFO": Fore.BLUE,
            "CHECK": Fore.WHITE,
        }

        color = colors.get(level, "")
        reset = Style.RESET_ALL if HAS_COLORS else ""

        # Format message
        formatted = f"[{timestamp}] {color}{level:<8}{reset} | {message}"

        if error:
            print(formatted, file=sys.stderr)
        else:
            print(formatted)

    def load_graph(self, file_path: Path) -> bool:
        """Load graph from JSON file."""
        self._log("INFO", f"Loading: {file_path.name}")

        if not file_path.exists():
            self._log("ERROR", f"File not found: {file_path}", error=True)
            return False

        try:
            with open(file_path, encoding="utf-8") as f:
                self.graph_data = json.load(f)
                self.graph_file = file_path.name

            nodes = len(self.graph_data.get("nodes", []))
            edges = len(self.graph_data.get("edges", []))
            self._log("INFO", f"Graph: {nodes} nodes, {edges} edges")
            return True

        except Exception as e:
            self._log("ERROR", f"Failed to load graph: {e}", error=True)
            return False

    def run_critical_checks(self) -> bool:
        """Run critical checks that can stop execution."""
        self._log("", "")
        self._log("", "=== Critical Checks ===")

        nodes = self.graph_data.get("nodes", [])
        edges = self.graph_data.get("edges", [])

        all_passed = True

        # 1. Check PageRank sum
        pageranks = [n.get("pagerank", 0) for n in nodes]
        if pageranks:
            pr_sum = sum(pageranks)
            tolerance = self.config["pagerank_sum_tolerance"]
            if abs(pr_sum - 1.0) <= tolerance:
                self._log("CHECK", f"✅ PageRank sum: {pr_sum:.4f} (tolerance: ±{tolerance})")
            else:
                self._log("CHECK", f"❌ PageRank sum: {pr_sum:.4f} (expected: 1.0 ±{tolerance})")
                self.critical_issues.append(
                    {
                        "check": "pagerank_sum",
                        "message": f"PageRank sum {pr_sum:.4f} outside tolerance",
                        "expected": 1.0,
                        "actual": pr_sum,
                        "tolerance": tolerance,
                    }
                )
                all_passed = False

        # 2. Check educational_importance sum
        edu_importance = [n.get("educational_importance", 0) for n in nodes]
        if edu_importance:
            edu_sum = sum(edu_importance)
            tolerance = self.config["educational_sum_tolerance"]
            if abs(edu_sum - 1.0) <= tolerance:
                self._log("CHECK", f"✅ Educational importance sum: {edu_sum:.4f}")
            else:
                self._log(
                    "CHECK",
                    f"❌ Educational importance sum: {edu_sum:.4f} (expected: 1.0 ±{tolerance})",
                )
                self.critical_issues.append(
                    {
                        "check": "educational_importance_sum",
                        "message": f"Educational importance sum {edu_sum:.4f} outside tolerance",
                        "expected": 1.0,
                        "actual": edu_sum,
                        "tolerance": tolerance,
                    }
                )
                all_passed = False

        # 3. Check component_id sequential
        component_ids = [n.get("component_id") for n in nodes if "component_id" in n]
        if component_ids:
            unique_ids = sorted(set(component_ids))
            expected = list(range(len(unique_ids)))
            if unique_ids == expected:
                self._log("CHECK", f"✅ Component IDs: sequential 0-{len(unique_ids)-1}")
            else:
                self._log("CHECK", f"❌ Component IDs not sequential: {unique_ids[:10]}...")
                self.critical_issues.append(
                    {
                        "check": "component_id_sequential",
                        "message": "Component IDs are not sequential from 0",
                        "found": unique_ids[:10],
                    }
                )
                all_passed = False

        # 4. Check prerequisite_depth >= 0
        prereq_depths = [n.get("prerequisite_depth", 0) for n in nodes]
        negative_depths = [d for d in prereq_depths if d < 0]
        if negative_depths:
            self._log(
                "CHECK", f"❌ Found {len(negative_depths)} negative prerequisite_depth values"
            )
            self.critical_issues.append(
                {
                    "check": "prerequisite_depth_positive",
                    "message": f"Found {len(negative_depths)} negative prerequisite_depth values",
                    "count": len(negative_depths),
                }
            )
            all_passed = False
        else:
            self._log("CHECK", "✅ Prerequisite depth: all ≥ 0")

        # 5. Check for NaN/Inf values
        nan_inf_found = False
        for node in nodes:
            for metric in self.REQUIRED_NODE_METRICS:
                # Handle out-closeness vs closeness_centrality naming
                metric_key = metric
                if metric == "out-closeness" and metric not in node:
                    # Try alternative name
                    if "closeness_centrality" in node:
                        metric_key = "closeness_centrality"

                if metric_key in node:
                    value = node[metric_key]
                    if isinstance(value, float):
                        if math.isnan(value) or math.isinf(value):
                            if not nan_inf_found:
                                self._log("CHECK", "❌ Found NaN/Inf values in metrics")
                                self.critical_issues.append(
                                    {
                                        "check": "nan_inf_values",
                                        "message": "Found NaN or Inf values in metrics",
                                    }
                                )
                                all_passed = False
                                nan_inf_found = True
                            break
            if nan_inf_found:
                break

        if not nan_inf_found:
            self._log("CHECK", "✅ No NaN/Inf values found")

        # 6. Check all required metrics present
        missing_metrics = set()
        for node in nodes:
            for metric in self.REQUIRED_NODE_METRICS:
                # Handle naming variations
                metric_key = metric
                if metric == "out-closeness" and metric not in node:
                    if "closeness_centrality" in node:
                        metric_key = "closeness_centrality"

                if metric_key not in node:
                    missing_metrics.add(metric)

        if missing_metrics:
            self._log("CHECK", f"❌ Missing required metrics: {', '.join(missing_metrics)}")
            self.critical_issues.append(
                {
                    "check": "required_metrics",
                    "message": f"Missing required metrics: {', '.join(missing_metrics)}",
                    "missing": list(missing_metrics),
                }
            )
            all_passed = False
        else:
            self._log("CHECK", "✅ All required node metrics present")

        # 7. Check inverse_weight on edges
        edges_with_inverse = sum(1 for e in edges if "inverse_weight" in e)
        if edges and edges_with_inverse == 0:
            self._log("CHECK", "❌ No edges have inverse_weight metric")
            self.critical_issues.append(
                {"check": "inverse_weight", "message": "No edges have inverse_weight metric"}
            )
            all_passed = False
        elif edges:
            self._log("CHECK", f"✅ Inverse weight: {edges_with_inverse}/{len(edges)} edges")

        # 8. Check for bidirectional PREREQUISITE pairs
        prereq_edges = [
            (e["source"], e["target"]) for e in edges if e.get("type") == "PREREQUISITE"
        ]

        if prereq_edges:
            # Find bidirectional pairs (A → B and B → A)
            bidirectional_pairs = []
            prereq_set = set(prereq_edges)

            for source, target in prereq_edges:
                # Check if reverse edge exists
                if (target, source) in prereq_set:
                    # To avoid duplicates, only add if source < target (lexicographically)
                    if source < target:
                        bidirectional_pairs.append((source, target))

            if bidirectional_pairs:
                # Show first 10 examples
                examples = bidirectional_pairs[:10]
                examples_str = ", ".join([f"{s} ⇄ {t}" for s, t in examples])

                self._log(
                    "CHECK",
                    f"❌ Found {len(bidirectional_pairs)} bidirectional " f"PREREQUISITE pairs:",
                )
                self._log("", f"  {examples_str}")
                self.critical_issues.append(
                    {
                        "check": "prerequisite_cycles",
                        "message": (
                            "Found cycles in PREREQUISITE graph (bidirectional prerequisites)"
                        ),
                        "count": len(bidirectional_pairs),
                        "examples": examples_str,
                    }
                )
                all_passed = False
            else:
                self._log("CHECK", "✅ No bidirectional PREREQUISITE pairs found")
        else:
            self._log("CHECK", "✅ No PREREQUISITE edges to check")

        return all_passed

    def run_warning_checks(self):
        """Run warning checks for potential issues."""
        self._log("", "")
        self._log("", "=== Warning Checks ===")

        nodes = self.graph_data.get("nodes", [])

        # Check if clustering was applied
        has_clustering = any("cluster_id" in n for n in nodes)

        if has_clustering:
            # 1. Check all nodes have cluster_id
            nodes_with_cluster = sum(1 for n in nodes if "cluster_id" in n)
            if nodes_with_cluster < len(nodes):
                self._log(
                    "CHECK", f"⚠️  Not all nodes have cluster_id: {nodes_with_cluster}/{len(nodes)}"
                )
                self.warnings.append(
                    {
                        "check": "cluster_id_coverage",
                        "message": f"Only {nodes_with_cluster}/{len(nodes)} nodes have cluster_id",
                    }
                )

            # 2. Check cluster modularity (simplified check)
            cluster_ids = [n.get("cluster_id") for n in nodes if "cluster_id" in n]
            unique_clusters = len(set(cluster_ids))
            if unique_clusters > 1:
                # Simple modularity estimate based on cluster count vs nodes
                modularity_estimate = 1 - (unique_clusters / len(nodes))
                if modularity_estimate < self.config["min_modularity"]:
                    self._log(
                        "CHECK", f"⚠️  Weak cluster modularity estimate: {modularity_estimate:.2f}"
                    )
                    self.warnings.append(
                        {
                            "check": "cluster_modularity",
                            "message": f"Weak cluster modularity: {modularity_estimate:.2f}",
                            "threshold": self.config["min_modularity"],
                            "actual": modularity_estimate,
                        }
                    )
                else:
                    self._log("CHECK", f"✅ Cluster modularity estimate: {modularity_estimate:.2f}")

            # 3. Check bridge_score if present
            nodes_with_bridge = [n for n in nodes if "bridge_score" in n]
            if nodes_with_bridge:
                bridge_scores = [n["bridge_score"] for n in nodes_with_bridge]

                # Check range [0, 1]
                out_of_range = [s for s in bridge_scores if s < 0 or s > 1]
                if out_of_range:
                    self._log("CHECK", f"⚠️  {len(out_of_range)} bridge scores outside [0,1]")
                    self.warnings.append(
                        {
                            "check": "bridge_score_range",
                            "message": f"{len(out_of_range)} bridge scores outside [0,1]",
                        }
                    )
                else:
                    self._log("CHECK", "✅ Bridge scores: valid range [0, 1]")

                # Check at least one > 0
                positive_bridges = [s for s in bridge_scores if s > 0]
                if not positive_bridges:
                    self._log("CHECK", "⚠️  No nodes with bridge_score > 0")
                    self.warnings.append(
                        {
                            "check": "bridge_score_positive",
                            "message": "No nodes have bridge_score > 0",
                        }
                    )

                # Check correlation with betweenness (simplified)
                if all("betweenness_centrality" in n for n in nodes_with_bridge):
                    # Simple check: high betweenness should correlate with high bridge score
                    high_betweenness = sorted(
                        nodes_with_bridge,
                        key=lambda x: x.get("betweenness_centrality", 0),
                        reverse=True,
                    )[:10]
                    avg_bridge_top = sum(n.get("bridge_score", 0) for n in high_betweenness) / len(
                        high_betweenness
                    )
                    if avg_bridge_top < self.config["min_bridge_correlation"]:
                        self._log(
                            "CHECK", "⚠️  Low correlation between bridge_score and betweenness"
                        )
                        self.warnings.append(
                            {
                                "check": "bridge_correlation",
                                "message": (
                                    "Low correlation between bridge_score "
                                    "and betweenness_centrality"
                                ),
                            }
                        )

            # 4. Check for singleton clusters
            from collections import Counter

            cluster_sizes = Counter(n.get("cluster_id") for n in nodes if "cluster_id" in n)
            singletons = [cid for cid, size in cluster_sizes.items() if size == 1]
            if singletons:
                self._log("CHECK", f"⚠️  Found {len(singletons)} singleton clusters")
                self.warnings.append(
                    {
                        "check": "singleton_clusters",
                        "message": f"Found {len(singletons)} clusters with only 1 node",
                    }
                )
        else:
            self._log("INFO", "Clustering not applied to graph")
            self.info_messages.append(
                {
                    "check": "clustering",
                    "message": "Clustering metrics (cluster_id, bridge_score) not found",
                }
            )

        # 5. Outlier detection for all metrics
        self._detect_outliers(nodes)

    def _detect_outliers(self, nodes: List[Dict]):
        """Detect outliers in metrics using IQR or 3-sigma method."""
        method = self.config["outlier_method"]
        threshold = self.config["outlier_threshold"]

        outliers_found = {}

        for metric in self.REQUIRED_NODE_METRICS:
            # Handle naming variations
            if metric == "out-closeness":
                # Try both names
                values = []
                for n in nodes:
                    if "out-closeness" in n:
                        values.append(n["out-closeness"])
                    elif "closeness_centrality" in n:
                        values.append(n["closeness_centrality"])
            else:
                values = [n.get(metric, 0) for n in nodes if metric in n]

            if len(values) < 4:  # Need at least 4 values for IQR
                continue

            values_sorted = sorted(values)

            if method == "iqr":
                # IQR method
                q1_idx = len(values) // 4
                q3_idx = 3 * len(values) // 4
                q1 = values_sorted[q1_idx]
                q3 = values_sorted[q3_idx]
                iqr = q3 - q1

                if iqr > 0:
                    lower_bound = q1 - threshold * iqr
                    upper_bound = q3 + threshold * iqr
                    outliers = [v for v in values if v < lower_bound or v > upper_bound]

                    if outliers:
                        outliers_found[metric] = len(outliers)

            elif method == "3sigma":
                # 3-sigma method
                mean_val = sum(values) / len(values)
                variance = sum((v - mean_val) ** 2 for v in values) / len(values)
                std_dev = math.sqrt(variance)

                if std_dev > 0:
                    lower_bound = mean_val - 3 * std_dev
                    upper_bound = mean_val + 3 * std_dev
                    outliers = [v for v in values if v < lower_bound or v > upper_bound]

                    if outliers:
                        outliers_found[metric] = len(outliers)

        if outliers_found:
            for metric, count in outliers_found.items():
                self._log("CHECK", f"⚠️  Found {count} outliers in {metric} ({method} method)")
                self.warnings.append(
                    {
                        "check": "outliers",
                        "metric": metric,
                        "message": f"Found {count} outliers in {metric}",
                        "method": method,
                    }
                )
        else:
            self._log("CHECK", f"✅ No significant outliers detected ({method} method)")

        # Check for orphan nodes (no edges at all)
        orphan_nodes = [
            n for n in nodes if n.get("degree_in", 0) == 0 and n.get("degree_out", 0) == 0
        ]

        if orphan_nodes:
            self._log(
                "CHECK",
                f"⚠️  Found {len(orphan_nodes)} orphan nodes " f"(no incoming or outgoing edges)",
            )
            self.warnings.append(
                {
                    "check": "orphan_nodes",
                    "message": f"Found {len(orphan_nodes)} orphan nodes with no edges",
                    "count": len(orphan_nodes),
                }
            )
        else:
            self._log("CHECK", "✅ No orphan nodes found")

        # Check for dangling Assessment nodes (no TESTS edges)
        edges = self.graph_data.get("edges", [])
        assessments = [n for n in nodes if n.get("type") == "Assessment"]

        if assessments:
            # Find Assessment nodes that are source of TESTS edges
            assessments_with_tests = set()
            for edge in edges:
                if edge.get("type") == "TESTS":
                    assessments_with_tests.add(edge["source"])

            # Find dangling assessments (Assessment nodes NOT in the set)
            assessment_ids = {n["id"] for n in assessments}
            dangling_assessments = assessment_ids - assessments_with_tests

            if dangling_assessments:
                self._log(
                    "CHECK",
                    f"⚠️  Found {len(dangling_assessments)} dangling Assessment nodes "
                    f"(no TESTS edges)",
                )
                self.warnings.append(
                    {
                        "check": "dangling_assessments",
                        "message": (
                            f"Found {len(dangling_assessments)} Assessment nodes "
                            f"without TESTS edges"
                        ),
                        "count": len(dangling_assessments),
                    }
                )
            else:
                self._log("CHECK", "✅ All Assessment nodes have TESTS edges")

    def _detect_cycles_dfs(self, edges: List[Tuple[str, str]], nodes: List[Dict]) -> bool:
        """Detect cycles in directed graph using DFS.

        Args:
            edges: List of (source, target) tuples
            nodes: List of node dictionaries

        Returns:
            True if cycle found, False otherwise
        """
        # Build adjacency list
        graph = {}
        node_ids = {n["id"] for n in nodes}

        for node_id in node_ids:
            graph[node_id] = []

        for source, target in edges:
            if source in graph:
                graph[source].append(target)

        # DFS with three states: unvisited (0), visiting (1), visited (2)
        state = {node_id: 0 for node_id in node_ids}

        def dfs(node_id):
            """DFS helper - returns True if cycle detected."""
            if state[node_id] == 1:  # Currently visiting - cycle found!
                return True
            if state[node_id] == 2:  # Already visited - no cycle here
                return False

            state[node_id] = 1  # Mark as visiting

            for neighbor in graph.get(node_id, []):
                if neighbor in state and dfs(neighbor):
                    return True

            state[node_id] = 2  # Mark as visited
            return False

        # Check all nodes
        for node_id in node_ids:
            if state[node_id] == 0:  # Unvisited
                if dfs(node_id):
                    return True

        return False

    def calculate_statistics(self):
        """Calculate and display statistics for all metrics."""
        self._log("", "")
        self._log("", "=== Statistics ===")

        nodes = self.graph_data.get("nodes", [])

        # Calculate stats for each metric
        for metric in self.REQUIRED_NODE_METRICS:
            # Handle naming variations
            if metric == "out-closeness":
                values = []
                for n in nodes:
                    if "out-closeness" in n:
                        values.append(n["out-closeness"])
                    elif "closeness_centrality" in n:
                        values.append(n["closeness_centrality"])
            else:
                values = [n.get(metric, 0) for n in nodes if metric in n]

            if values:
                stats = {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "count": len(values),
                }

                # Calculate std deviation
                variance = sum((v - stats["mean"]) ** 2 for v in values) / len(values)
                stats["std"] = math.sqrt(variance)

                self.statistics[metric] = stats

                # Display key metrics
                if metric in ["pagerank", "educational_importance", "betweenness_centrality"]:
                    self._log(
                        "INFO",
                        f"{metric}: min={stats['min']:.4f}, max={stats['max']:.4f}, "
                        f"mean={stats['mean']:.4f}",
                    )

        # Component statistics
        component_ids = [n.get("component_id") for n in nodes if "component_id" in n]
        if component_ids:
            from collections import Counter

            comp_sizes = Counter(component_ids)
            largest_comp = max(comp_sizes.values())
            self._log("INFO", f"Components: {len(comp_sizes)} total, largest: {largest_comp} nodes")
            # Multiline breakdown by component_id
            for comp_id in sorted(comp_sizes.keys()):
                size = comp_sizes[comp_id]
                self._log("INFO", f"  component_id = {comp_id} [{size}]")

        # Concept Coverage Statistics
        chunks = [n for n in nodes if n.get("type") == "Chunk"]
        assessments = [n for n in nodes if n.get("type") == "Assessment"]

        if chunks or assessments:
            self._log("INFO", "Concept Coverage:")

            # Process Chunks
            if chunks:
                chunks_with_concepts = [
                    n for n in chunks if n.get("concepts") and len(n["concepts"]) > 0
                ]
                coverage_pct = (len(chunks_with_concepts) / len(chunks)) * 100
                total_concepts = sum(len(n.get("concepts", [])) for n in chunks)
                avg_concepts = total_concepts / len(chunks) if chunks else 0

                self._log(
                    "INFO",
                    f"  Chunks: {coverage_pct:.1f}% "
                    f"({len(chunks_with_concepts)}/{len(chunks)} nodes), "
                    f"avg {avg_concepts:.1f} concepts/node",
                )
            else:
                self._log("INFO", "  Chunks: N/A")

            # Process Assessments
            if assessments:
                assessments_with_concepts = [
                    n for n in assessments if n.get("concepts") and len(n["concepts"]) > 0
                ]
                coverage_pct = (len(assessments_with_concepts) / len(assessments)) * 100
                total_concepts = sum(len(n.get("concepts", [])) for n in assessments)
                avg_concepts = total_concepts / len(assessments) if assessments else 0

                self._log(
                    "INFO",
                    f"  Assessments: {coverage_pct:.1f}% "
                    f"({len(assessments_with_concepts)}/{len(assessments)} nodes), "
                    f"avg {avg_concepts:.1f} concepts/node",
                )
            else:
                self._log("INFO", "  Assessments: N/A")

        # Cluster statistics
        cluster_ids = [n.get("cluster_id") for n in nodes if "cluster_id" in n]
        if cluster_ids:
            from collections import Counter

            cluster_sizes = Counter(cluster_ids)
            sizes_list = sorted(cluster_sizes.values(), reverse=True)
            sizes_str = ", ".join(str(s) for s in sizes_list[:10])
            if len(sizes_list) > 10:
                sizes_str += ", ..."
            self._log("INFO", f"Clusters: {len(cluster_sizes)} total, sizes: [{sizes_str}]")

    def save_json_report(self):
        """Save detailed JSON report to logs directory."""
        if not self.config["save_json_report"]:
            return

        # Create logs directory
        logs_dir = Path("viz/logs")
        logs_dir.mkdir(exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = logs_dir / f"anomaly_report_{timestamp}.json"

        # Determine overall status
        if self.critical_issues:
            status = "CRITICAL_ISSUES"
        elif self.warnings:
            status = "PASSED_WITH_WARNINGS"
        else:
            status = "PASSED"

        # Build report
        report = {
            "timestamp": datetime.now().isoformat(),
            "graph_file": self.graph_file,
            "summary": {
                "nodes": len(self.graph_data.get("nodes", [])),
                "edges": len(self.graph_data.get("edges", [])),
                "status": status,
                "critical_issues": len(self.critical_issues),
                "warnings": len(self.warnings),
                "info_messages": len(self.info_messages),
            },
            "critical": self.critical_issues,
            "warnings": self.warnings,
            "info": self.info_messages,
            "statistics": self.statistics,
            "config": self.config,
        }

        # Save report
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self._log("INFO", f"Report saved: {report_file}")

    def run(self, graph_file: Path) -> int:
        """Run complete anomaly detection pipeline."""
        self._log("START", "Anomaly Detection for Knowledge Graph")

        # Load graph
        if not self.load_graph(graph_file):
            return 1

        # Run checks
        critical_passed = self.run_critical_checks()
        self.run_warning_checks()
        self.calculate_statistics()

        # Summary
        self._log("", "")
        self._log("", "=== Summary ===")

        if critical_passed:
            self._log("SUCCESS", "✅ All critical checks passed")
        else:
            self._log("ERROR", f"❌ {len(self.critical_issues)} critical issues found")

        if self.warnings:
            self._log("WARNING", f"⚠️  {len(self.warnings)} warnings detected")

        # Save JSON report
        self.save_json_report()

        # Determine exit code
        if self.critical_issues:
            return 1
        elif self.warnings and self.config["strict_mode"]:
            return 1
        else:
            return 0


def main():
    """CLI entry point."""
    # Default file path
    default_file = Path("viz/data/out/LearningChunkGraph_wow.json")

    # Check if file argument provided
    if len(sys.argv) > 1:
        graph_file = Path(sys.argv[1])
    else:
        graph_file = default_file

    # Run detector
    detector = AnomalyDetector()
    exit_code = detector.run(graph_file)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

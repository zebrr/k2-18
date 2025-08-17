# viz_anomaly_detector.md

## Status: READY

Anomaly detection utility for enriched knowledge graphs. Validates computed metrics, detects anomalies, and generates comprehensive reports with both console output and JSON logs.

## CLI Interface

### Usage
```bash
python -m viz.anomaly_detector [graph_file]
```

### Input
- **graph_file** (optional): Path to enriched JSON graph file
  - Default: `viz/data/out/LearningChunkGraph_wow.json`
  - Format: JSON with nodes and edges containing computed metrics

### Output Files
- **JSON Report**: `viz/logs/anomaly_report_YYYYMMDD_HHMMSS.json`
  - Detailed analysis results
  - All detected issues and statistics
  - Configuration used

### Exit Codes
- **0**: Success (no critical issues)
- **1**: Critical issues found OR warnings in strict_mode

## Terminal Output

### Format
```
[HH:MM:SS] LEVEL    | Message
```

### Levels
- **START**: Beginning of analysis (cyan+bright)
- **INFO**: Informational messages (blue)
- **CHECK**: Check results (white)
- **WARNING**: Warning messages (yellow)
- **ERROR**: Error messages (red+bright)
- **SUCCESS**: Success message (green+bright)

### Example Output
```
[12:34:56] START    | Anomaly Detection for Knowledge Graph
[12:34:56] INFO     | Loading: LearningChunkGraph_wow.json
[12:34:56] INFO     | Graph: 547 nodes, 1823 edges

=== Critical Checks ===
[12:34:56] CHECK    | ✅ PageRank sum: 1.0000 (tolerance: ±0.01)
[12:34:56] CHECK    | ✅ Educational importance sum: 0.9998
[12:34:56] CHECK    | ✅ Component IDs: sequential 0-3
[12:34:56] CHECK    | ✅ Prerequisite depth: all ≥ 0
[12:34:56] CHECK    | ✅ No NaN/Inf values found
[12:34:56] CHECK    | ✅ All required node metrics present
[12:34:56] CHECK    | ✅ Inverse weight: 1823/1823 edges

=== Warning Checks ===
[12:34:57] CHECK    | ⚠️  Weak cluster modularity estimate: 0.08
[12:34:57] CHECK    | ✅ Bridge scores: valid range [0, 1]
[12:34:57] CHECK    | ⚠️  Found 5 outliers in pagerank (iqr method)

=== Statistics ===
[12:34:57] INFO     | pagerank: min=0.0001, max=0.0234, mean=0.0018
[12:34:57] INFO     | betweenness_centrality: min=0.0000, max=0.1234, mean=0.0089
[12:34:57] INFO     | educational_importance: min=0.0001, max=0.0234, mean=0.0018
[12:34:57] INFO     | Components: 3 total, largest: 523 nodes
[12:34:57] INFO     | Clusters: 8 total, sizes: [124, 89, 76, 65, ...]

=== Summary ===
[12:34:58] SUCCESS  | ✅ All critical checks passed
[12:34:58] WARNING  | ⚠️  2 warnings detected
[12:34:58] INFO     | Report saved: viz/logs/anomaly_report_20250117_123458.json
```

## Core Algorithm

### Check Categories

#### 1. Critical Checks (stop execution)
- **PageRank sum**: Must be 1.0 ± tolerance
- **Educational importance sum**: Must be 1.0 ± tolerance
- **NaN/Inf values**: None allowed in any metric
- **Component IDs**: Must be sequential from 0
- **Prerequisite depth**: All values ≥ 0
- **Required metrics**: All 10 node metrics + inverse_weight on edges

#### 2. Warning Checks
- **Cluster modularity**: Above minimum threshold (if clustering applied)
- **Bridge scores**: Valid range [0, 1] and at least one > 0
- **Bridge correlation**: Correlation with betweenness centrality
- **Inter-cluster consistency**: Edges between clusters
- **Singleton clusters**: No clusters with size = 1
- **Outlier detection**: Statistical outliers in metrics

#### 3. Information Checks
- **Basic statistics**: min, max, mean, std for all metrics
- **Component sizes**: Number and sizes of connected components
- **Cluster sizes**: Number and sizes of clusters (if applicable)
- **Missing optional metrics**: cluster_id, bridge_score

### Outlier Detection Methods

#### IQR Method (default)
- Calculate Q1 (25th percentile) and Q3 (75th percentile)
- IQR = Q3 - Q1
- Outliers: values outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]

#### 3-Sigma Method
- Calculate mean and standard deviation
- Outliers: values outside [mean - 3×σ, mean + 3×σ]

## Public Functions/Classes

### AnomalyDetector

Main class for anomaly detection.

#### `__init__(config_path: Path = Path("viz/config.toml")) -> None`
Initialize detector with configuration.
- **Input**: 
  - config_path: Path to TOML configuration file
- **Attributes**:
  - config: Loaded configuration dictionary
  - critical_issues: List of critical problems found
  - warnings: List of warning issues
  - info_messages: List of informational messages
  - statistics: Computed statistics for metrics

#### `load_graph(file_path: Path) -> bool`
Load graph from JSON file.
- **Input**: file_path - Path to graph JSON file
- **Returns**: True if successful, False otherwise

#### `run_critical_checks() -> bool`
Execute all critical validation checks.
- **Returns**: True if all passed, False if any failed

#### `run_warning_checks() -> None`
Execute warning-level checks for potential issues.

#### `calculate_statistics() -> None`
Calculate and display statistics for all metrics.

#### `save_json_report() -> None`
Save detailed JSON report to logs directory.

#### `run(graph_file: Path) -> int`
Run complete anomaly detection pipeline.
- **Input**: graph_file - Path to graph file
- **Returns**: Exit code (0 or 1)

### Required Node Metrics
```python
REQUIRED_NODE_METRICS = [
    "degree_in",
    "degree_out",
    "degree_centrality",
    "pagerank",
    "betweenness_centrality",
    "out-closeness",  # Note: may be "closeness_centrality" in data
    "component_id",
    "prerequisite_depth",
    "learning_effort",
    "educational_importance"
]
```

### Optional Node Metrics
```python
OPTIONAL_NODE_METRICS = ["cluster_id", "bridge_score"]
```

## Configuration

Configuration loaded from `/viz/config.toml` section `[anomaly_detection]`:

```toml
[anomaly_detection]
# Critical thresholds (violation = stop)
pagerank_sum_tolerance = 0.01      # Allowed deviation from 1.0
educational_sum_tolerance = 0.01   # Allowed deviation from 1.0

# Warning thresholds
min_modularity = 0.1               # Minimum cluster modularity
min_bridge_correlation = 0.3       # Minimum correlation bridge_score vs betweenness
outlier_method = "iqr"             # Outlier detection: "iqr" or "3sigma"
outlier_threshold = 1.5            # Multiplier for IQR method

# Behavior
strict_mode = false                # true = WARNINGs are also critical for exit code
save_json_report = true            # Save JSON report to logs
```

## JSON Report Structure

```json
{
  "timestamp": "2025-01-17T12:34:58",
  "graph_file": "LearningChunkGraph_wow.json",
  "summary": {
    "nodes": 547,
    "edges": 1823,
    "status": "PASSED|PASSED_WITH_WARNINGS|CRITICAL_ISSUES",
    "critical_issues": 0,
    "warnings": 2,
    "info_messages": 1
  },
  "critical": [],
  "warnings": [
    {
      "check": "cluster_modularity",
      "message": "Weak cluster modularity: 0.08",
      "threshold": 0.10,
      "actual": 0.08
    }
  ],
  "info": [
    {
      "check": "clustering",
      "message": "Clustering metrics not found"
    }
  ],
  "statistics": {
    "pagerank": {
      "min": 0.0001,
      "max": 0.0234,
      "mean": 0.0018,
      "count": 547,
      "std": 0.0023
    },
    // ... other metrics
  },
  "config": {
    // Configuration used
  }
}
```

## Error Handling & Exit Codes

### Exit Codes
- **0**: Success
  - All critical checks passed
  - No warnings OR strict_mode=false
- **1**: Failure
  - Critical issues found OR
  - Warnings found AND strict_mode=true

### Error Conditions
- **Missing config file**: Exit with code 1
- **Missing graph file**: Exit with code 1
- **Invalid JSON**: Exit with code 1
- **Critical check failures**: Continue all checks, exit 1

## Test Coverage

- **test_viz_anomaly_detector**: Unit tests
  - test_load_config
  - test_critical_checks
  - test_warning_checks
  - test_outlier_detection
  - test_statistics_calculation
  - test_json_report_generation
  - test_exit_codes

- **Integration tests**:
  - test_full_pipeline_success
  - test_full_pipeline_with_warnings
  - test_full_pipeline_with_critical_issues
  - test_strict_mode_behavior

## Dependencies

- **Standard Library**: json, math, sys, datetime, pathlib, collections.Counter
- **External**: 
  - tomli (TOML parser)
  - colorama (optional, for colored output)
- **Internal**: None (standalone utility)

## Usage Examples

### Basic Usage
```bash
# Use default file
python -m viz.anomaly_detector

# Specify custom file
python -m viz.anomaly_detector viz/data/out/my_graph.json

# Check exit code
python -m viz.anomaly_detector && echo "Success" || echo "Issues found"
```

### In Scripts
```python
from viz.anomaly_detector import AnomalyDetector
from pathlib import Path

# Create detector
detector = AnomalyDetector()

# Run analysis
graph_file = Path("viz/data/out/LearningChunkGraph_wow.json")
exit_code = detector.run(graph_file)

# Access results
if detector.critical_issues:
    print(f"Critical issues: {len(detector.critical_issues)}")
if detector.warnings:
    print(f"Warnings: {len(detector.warnings)}")

# Statistics available in detector.statistics
```

### With Custom Config
```python
from viz.anomaly_detector import AnomalyDetector
from pathlib import Path

# Use custom config file
config_path = Path("custom_config.toml")
detector = AnomalyDetector(config_path)

# Run analysis
exit_code = detector.run(graph_file)
```

## Notes

- Handles both `out-closeness` and `closeness_centrality` naming variations
- Clustering checks are skipped if clustering wasn't applied
- Logs directory (`viz/logs/`) is created automatically if needed
- Colorama is optional - falls back to plain text if not available
- All metrics should be numeric (float/int) for proper validation
# K2-18-GRAPH_SPLIT-REFACTOR-001: Add Cluster Dictionary and Zero-Padding

## References

**Read before starting:**
- `/docs/specs/viz_graph_split.md` — updated spec (Status: IN_PROGRESS)
- `/viz/graph_split.py` — current implementation
- `/tests/viz/test_graph_split.py` — existing tests
- `/src/schemas/ConceptDictionary.schema.json` — dictionary schema (for reference)

## Context

The `graph_split` utility currently splits enriched graph into cluster files. We need to extend it with:

1. **Cluster dictionaries** — extract concepts used by each cluster into separate `_dict.json` files
2. **Zero-padding in filenames** — consistent naming like `cluster_00.json` instead of `cluster_0.json`

This enables LLM scenarios where working with a cluster requires access to related concepts without loading the full dictionary.

### Current state
- Utility loads `LearningChunkGraph_wow.json` and splits by `cluster_id`
- Output: `LearningChunkGraph_cluster_{ID}.json`
- Title is already extracted from source graph's `_meta.title`

### Target state
- Load both graph AND `ConceptDictionary_wow.json`
- Output two files per cluster:
  - `LearningChunkGraph_cluster_{ID}.json` (graph)
  - `LearningChunkGraph_cluster_{ID}_dict.json` (dictionary)
- Filenames use zero-padding based on max cluster ID

## Steps

### 1. Add dictionary loading

Add function `load_dictionary(input_file: Path, logger: Logger) -> Dict`:
- Load `/viz/data/out/ConceptDictionary_wow.json`
- Validate against `ConceptDictionary` schema
- Return dictionary data
- Handle errors same as `load_graph()`

Update `main()` to load dictionary after graph.

### 2. Add zero-padding utility

Add function `get_filename_padding(cluster_ids: List[int]) -> int`:
- Calculate `len(str(max(cluster_ids)))` 
- Return 1 if empty list
- Example: 16 clusters (0-15) → returns 2

Update `save_cluster_graph()` signature to accept `padding: int` parameter.
Update filename generation: `f"LearningChunkGraph_cluster_{cluster_id:0{padding}d}.json"`

### 3. Add concept extraction

Add function `extract_cluster_concepts(cluster_nodes: List[Dict], concepts_data: Dict, logger: Logger) -> Tuple[List[Dict], int]`:

```python
def extract_cluster_concepts(cluster_nodes, concepts_data, logger):
    # 1. Collect unique concept IDs from all nodes' concepts: [] field
    concept_ids = set()
    for node in cluster_nodes:
        concept_ids.update(node.get("concepts", []))
    
    # 2. Build lookup map from concepts_data
    concept_map = {c["concept_id"]: c for c in concepts_data.get("concepts", [])}
    
    # 3. Extract matching concepts
    concepts_list = []
    for cid in sorted(concept_ids):  # sorted for deterministic output
        if cid in concept_map:
            concepts_list.append(concept_map[cid])
        else:
            logger.warning(f"Concept {cid} not found in dictionary")
    
    return concepts_list, len(concepts_list)
```

### 4. Add cluster dictionary creation

Add function `create_cluster_dictionary(cluster_id: int, concepts_list: List[Dict], original_title: str) -> Dict`:

```python
def create_cluster_dictionary(cluster_id, concepts_list, original_title):
    return {
        "_meta": {
            "title": original_title,
            "cluster_id": cluster_id,
            "concepts_used": len(concepts_list)
        },
        "concepts": concepts_list
    }
```

### 5. Add dictionary saving

Add function `save_cluster_dictionary(cluster_dict: Dict, cluster_id: int, output_dir: Path, padding: int, logger: Logger) -> None`:
- Build filename with padding: `f"LearningChunkGraph_cluster_{cluster_id:0{padding}d}_dict.json"`
- Save with `ensure_ascii=False, indent=2`
- Handle IO errors with `EXIT_IO_ERROR`

### 6. Update main loop

In `main()`, for each cluster:
1. Extract cluster graph (existing)
2. Skip if single node (existing) — skip BOTH files
3. Extract cluster concepts (NEW)
4. Create cluster dictionary (NEW)
5. Save cluster graph with padding (UPDATED)
6. Save cluster dictionary with padding (NEW)
7. Log: `Cluster {id:0{padding}d}: {nodes} nodes, {edges} edges, {concepts} concepts`

### 7. Update terminal output

Update log messages to show:
- Dictionary loading: `Loading: ConceptDictionary_wow.json`
- Dictionary stats: `Dictionary: 115 concepts`
- Padding info: `Found 16 clusters (zero-padding: 2 digits)`
- Per-cluster: `Cluster 00: 45 nodes, 123 edges, 23 concepts`
- Summary: `Created 15 cluster graphs + 15 cluster dictionaries (1 skipped)`

### 8. Add new tests

Add to `/tests/viz/test_graph_split.py`:

**Unit tests:**
- `test_get_filename_padding` — verify 1, 2, 3 digit cases
- `test_get_filename_padding_empty` — empty list returns 1
- `test_extract_cluster_concepts` — normal extraction
- `test_extract_cluster_concepts_missing` — warning on missing concept
- `test_create_cluster_dictionary` — verify structure and metadata

**Integration tests:**
- `test_dictionary_files_created` — verify `_dict.json` files created
- `test_zero_padding_in_filenames` — verify consistent padding

**Boundary tests:**
- `test_cluster_with_no_concepts` — nodes with empty `concepts: []`

## Testing

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows

# Run code quality checks FIRST
ruff check viz/graph_split.py
black --check viz/graph_split.py
isort --check-only viz/graph_split.py
mypy viz/graph_split.py --ignore-missing-imports

# Run tests AFTER quality checks pass
pytest tests/viz/test_graph_split.py -v

# Run specific new tests
pytest tests/viz/test_graph_split.py -v -k "padding or dictionary or concepts"

# Integration test with real data (if available)
python -m viz.graph_split
ls viz/data/out/LearningChunkGraph_cluster_*
```

**Expected test results:**
- All existing tests pass (no regression)
- All new tests pass
- Zero-padded filenames in output
- Dictionary files created alongside graph files

## Deliverables

1. **Updated** `/viz/graph_split.py` with:
   - `load_dictionary()` function
   - `get_filename_padding()` function
   - `extract_cluster_concepts()` function
   - `create_cluster_dictionary()` function
   - `save_cluster_dictionary()` function
   - Updated `save_cluster_graph()` with padding parameter
   - Updated `main()` with new flow

2. **Updated** `/tests/viz/test_graph_split.py` with 8 new tests

3. **Report** in `/docs/tasks/K2-18-GRAPH_SPLIT-REFACTOR-001_REPORT.md`:
   - Summary of changes
   - Test results
   - Any issues or decisions made

4. **Updated spec status** in `/docs/specs/viz_graph_split.md`:
   - Change `Status: IN_PROGRESS` → `Status: READY` after successful implementation

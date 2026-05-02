# CLAUDE.md - Claude Code Instructions - Project K2-18 v4.0

## Project Overview

K2-18 is a semantic knowledge graph converter that processes educational content into:
- **ConceptDictionary** - structured vocabulary of concepts with definitions and synonyms
- **LearningChunkGraph** - semantic graph with relationships between knowledge chunks

## Critical Rules

1. **NEVER commit** changes - user controls git
2. **NEVER create** directories not specified in technical requirements
3. **NEVER run** production CLI utilities without user permission
4. **ALWAYS backup** files before modification: `<filename>_backup_<TASK-ID>.*`
5. **ALWAYS activate** venv before Python commands (see Environment Detection)
6. **ALWAYS do** quality checks → tests → spec updates → report (in this order)

## Environment Detection

At session start, detect environment:

### Check 1: Platform
- `$OSTYPE` contains `darwin` → macOS
- `$OSTYPE` contains `msys` or `cygwin` → Windows

### Check 2: Agent SDK Worktree or Terminal mode
- `$PWD` contains `.claude-worktrees` → Agent SDK sandbox mode
  - venv does NOT exist, must create:
    - macOS: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pip install -r requirements-dev.txt`
    - Windows: `python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt && pip install -r requirements-dev.txt`
- Otherwise → normal terminal mode
  - venv EXISTS, activate: `source .venv/bin/activate` (macOS) or `.venv\Scripts\activate` (Windows)

## Project Structure

```
/src/              → Main pipeline (slicer, itext2kg_*, dedup, refiner)
/src/utils/        → Shared utilities  
/src/schemas/      → JSON schemas for data structures
/src/prompts/      → LLM prompts
/viz/              → Visualization subproject (separate config.toml)
/tests/            → All tests (pytest)
/docs/             → Project docs and API references
/docs/specs/       → Module specifications (READ FIRST!)
/docs/issues/      → Known issues and problems (researched, not yet solved)
/docs/tasks/       → Task assignments and reports for Claude Code
/data/raw/         → Input files
/data/staging/     → *.slice.json (intermediate)
/data/out/         → Final outputs
```

### Key Components

**Main Pipeline** (`/src/`) — data in `/data/raw/` → `/data/staging/` → `/data/out/`:
- `slicer.py` - Splits content into chunks → `slice_*.slice.json`
- `itext2kg_concepts.py` - Extracts concepts → `ConceptDictionary.json`
- `itext2kg_graph.py` - Builds graph → `LearningChunkGraph_raw.json`
- `dedup.py` - Removes duplicates using embeddings → `LearningChunkGraph_dedup.json`
- `refiner_longrange.py` - Adds long-range connections → `LearningChunkGraph_longrange.json`

**Visualization** (`/viz/`) — data in `/viz/data/in/` → `/viz/data/out/`, tests in `/viz/data/test/`:
- `graph2metrics.py` - Computes graph metrics → `LearningChunkGraph_wow.json` (enriched)
- `graph_fix.py` - Marks LLM-generated content → `LearningChunkGraph_wow.json` (with markers)
- `anomaly_detector.py` - Validates graph quality → report in `/viz/logs/`
- `graph2html.py` - Interactive graph visualization → `knowledge_graph.html`
- `graph2viewer.py` - 3-column detailed viewer → `knowledge_graph_viewer.html`

**Data handoff**: Copy final artifacts from `/data/out/` to `/viz/data/in/` (without `_*` postfixes):
- `ConceptDictionary.json` → `ConceptDictionary.json`
- `LearningChunkGraph_longrange.json` → `LearningChunkGraph.json`

All tools use `/src/config.toml` for main pipeline, `/viz/config.toml` for visualization

### Exit Codes

- 0 - SUCCESS; 1 - CONFIG_ERROR; 2 - INPUT_ERROR; 3 - RUNTIME_ERROR; 4 - API_LIMIT_ERROR; 5 - IO_ERROR

Read `/docs/specs/util_exit_codes.md` for details

## Standard Workflow

 1. **READ** task assignment in `/docs/tasks/K2-18-<milestone>-XXX.md`
 2. **CHECK** specs in `/docs/specs/` for modules you'll touch
 3. **PLAN** prepare implementation plan, present to user for approval (use `plan mode`)
 4. **EXECUTE** after user confirms:
    - Backup files you'll modify
    - Implement according to plan
    - Run quality checks: `ruff check && ruff format && mypy`
    - Write/update tests, run pytest
 5. **UPDATE** spec to match implementation, set status:
    - `READY` — module fully matches spec, work complete
    - `IN_PROGRESS` — work will continue in follow-up tasks
 6. **REPORT** create `/docs/tasks/K2-18-<milestone>-XXX_REPORT.md`
 7. **NEVER** proceed beyond the specified scope

## Documentation First Policy

**Key Project Docs**:
- `/docs/K2-18 Architecture.md` - Main project architecture (build + viz pipelines)
- `/docs/K2-18 Specs Writing Guide.md` - How to write specs
- `/docs/k2-18 LLM Reference.md` - Unified LLM reference for the project
- `/docs/K2-18 Graph Metrics Computation Algorithms Reference.md` - Graph metrics algorithms
- `/docs/K2-18 Guide to Domain Adaptation of Prompts.md` - Domain adaptation guide

**ALWAYS** check specs before reading .py files:
- `/docs/specs/cli_*.md` - CLI tools documentation
- `/docs/specs/util_*.md` - Utilities documentation  
- `/docs/specs/viz_*.md` - Visualization documentation

**API References**:
- `/docs/K2-18 OpenAI Responses API Reference.md` - OpenAI Responses API
- `/docs/K2-18 OpenAI Embeddings API Reference.md` - OpenAI Embeddings API
- `/docs/K2-18 Cytoscape.js Reference.md` - Cytoscape.js Reference

If the documentation is incomplete or unclear, don't make assumptions - **ASK** user!

To create new specifications **ALWAYS** use `/docs/K2-18 Specs Writing Guide.md`

## Communication Guidelines

- Match user's language (usually Russian for discussion, English for code/docs)
- When uncertain: **ASK**, don't make assumptions
- Explain architectural decisions step by step
- Propose alternatives: rank them by priority with pros/cons explained
- Mention potential performance impacts
- Report blockers immediately

## Coding Style and Key Standards

### Principles

- **PEP 8**: Follow Python code-style standards
- **Typing**: Use type hints for all functions and methods
- **Docstrings**: Document all public functions, classes, and modules
- **Naming**: Use clear and descriptive names for variables and functions
- **Comments**: Explain complex logic and algorithms, not obvious code
- **Backups**: Create backups before making changes
- **Commit Types**: feat, fix, docs, test, refactor, style, chore

### Imports

- Place imports in the following order: standard library, third-party packages, local imports
- Use absolute imports instead of relative ones whenever possible
- Group imports and separate them with blank lines

### Linting and Formatting

- **Linting**: `ruff check .`
- **Formatting**: `ruff format .`
- **Type checking**: `mypy`

## Task Type Patterns

**New CLI Tool / Module** (new module):
1. Create spec following `/docs/K2-18 Specs Writing Guide.md` → status `DRAFT`
2. After approval: implement according to plan
3. Write tests
4. Update spec to match implementation, set status: `READY` or `IN_PROGRESS`

**Bug Fix** (existing module):
1. Read current spec
2. Write failing test that reproduces bug
3. Fix implementation
4. Verify all tests pass
5. Update spec to match implementation, set status: `READY` or `IN_PROGRESS`

**Refactoring** (existing module):
1. Read current spec, ensure tests pass
2. Backup original files
3. Refactor incrementally, verify tests after each step
4. Update spec to match implementation, set status: `READY` or `IN_PROGRESS`

**New Feature** (existing module):
1. Read current spec
2. Implement according to plan
3. Write/update tests
4. Update spec to match implementation, set status: `READY` or `IN_PROGRESS`

## Common Pitfalls

- **config.toml files are READ-ONLY** - never modify if not required by the task
- **Tests must run sequentially** - no parallel execution
- **Integration tests need real API** - .env must have valid keys
- **viz/ has separate config** - don't confuse with /src/config.toml
- **Specs before code** - always read documentation first
- **Backup before modify** - create _backup_<TASK-ID> files

## Report Template

The report `/docs/tasks/K2-18-<milestone>-XXX_REPORT.md` should follow this structure:

```markdown
# Task K2-18-<milestone>-XXX Completion Report

## Summary
[Brief overview of what was accomplished]

## Changes Made
- File 1: [what changed and why]
- File 2: [what changed and why]

## Tests
- Result: PASS/FAIL
- Existed tests modified: [list if any]
- New tests added: [list if any]

## Quality Checks
- ruff check: PASS/FAIL
- ruff format: PASS/FAIL
- mypy: PASS/FAIL

## Issues Encountered
[Any problems and resolutions, or "None"]

## Next Steps
[If any follow-up needed, or "None"]

## Commit Proposal
`type: brief description`

## Specs Updated
[Brief overview of what was updated or added]
```

## Commands Reference

```bash
# Environment Setup (see Environment Detection section)
# macOS:
source .venv/bin/activate              # bash/zsh
source .venv/bin/activate.fish         # fish shell
source .venv/bin/activate.csh          # csh/tcsh
# Windows:
.venv\Scripts\activate                 # cmd/powershell

# Quality Checks (run ALL before tests)
ruff check src/                        # Linting
ruff format src/                       # Formatting
mypy src/                              # Type checking

# Testing
python -m pytest -v                       # All tests
python -m pytest -v -s                    # With stdout
python -m pytest -v -m "not integration"  # Skip integration
python -m pytest -v -m "integration"      # Only integration
python -m pytest -k "test_name" -v        # Specific test by name
python -m pytest tests/test_module.py::test_function -v  # Specific test

# Visualization Tests
python -m pytest -m viz -v             # Only viz tests
python -m pytest -m "not viz" -v       # Exclude viz tests

# Coverage
python -m pytest --cov=src --cov-report=term-missing -v  # With details
python -m pytest --cov=src --cov-report=html            # HTML report
```

## Appendix 1: Core Python Standards

- Changes to existing code structure require clear, documented justification
- Every new feature must include unit tests
- Every bug must be reproduced by a unit test before being fixed
- Each class must include a docstring stating its purpose with usage example
- Each public method or function should include a docstring
- Docstrings, specs, and comments must be in English, UTF-8 encoding
- Favor "fail fast" over "fail safe": throw exceptions earlier
- Exception messages must include as much context as possible
- Error and log messages should not end with a period
- Constructors (`__init__`) should be lightweight: attribute assignments and simple validation only
- Prefer composition; use inheritance only when it adds clear value
- Favor immutable data objects where practical (e.g., `@dataclass(frozen=True)`)
- Provide only one primary constructor; use `@classmethod` factories for alternatives
- Do not create "utility" classes; use module-level functions instead
- Avoid `@staticmethod`; prefer `@classmethod` or standalone functions

## Appendix 2: Testing Standards

- Every change must be covered by a unit test
- Test cases must be as short as possible
- Every test must assert at least once
- Tests must use irregular inputs (e.g., non-ASCII strings)
- Tests must close resources they use (files, sockets, connections)
- Each test must verify only one specific behavioral pattern
- Tests should store temporary files in temporary directories, not in the codebase
- Tests must not wait indefinitely; always use timeouts
- Tests must assume absence of Internet connection (for unit tests)
- Tests must not rely on default configurations — provide explicit arguments
- Prefer pytest fixtures over `setUp`/`tearDown` methods
- Name tests descriptively: `test_returns_none_if_empty`
- Use mocks sparingly — favor lightweight fakes or stubs

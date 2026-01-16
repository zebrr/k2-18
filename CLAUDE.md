# CLAUDE.md - Claude Code Instructions - Project K2-18 v3.0

Workspace: `/Users/askold.romanov/code/k2-18`

## Project Overview

K2-18 is a semantic knowledge graph converter that processes educational content into:
- **ConceptDictionary** - structured vocabulary of concepts with definitions and synonyms
- **LearningChunkGraph** - semantic graph with relationships between knowledge chunks

## Critical Rules

1. **NEVER commit** changes - user controls git
2. **NEVER create** directories not specified in technical requirements
3. **NEVER run** production CLI utilities without user permission
4. **ALWAYS backup** files before modification: `<filename>_backup_<TASK-ID>.*`
5. **ALWAYS activate** venv before Python commands: `source .venv/bin/activate`
6. **ALWAYS do** quality checks → tests → spec updates → report (in this order)

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
/data/raw/         → Input files
/data/staging/     → *.slice.json (intermediate)
/data/out/         → Final outputs
```

### Key Components

**Main Pipeline** (`/src/`):
- `slicer.py` - Splits content into chunks
- `itext2kg_concepts.py` - Extracts concepts → ConceptDictionary  
- `itext2kg_graph.py` - Builds graph → LearningChunkGraph
- `dedup.py` - Removes duplicates using embeddings
- `refiner_longrange.py` - Adds long-range connections

**Visualization** (`/viz/`):
- `graph2metrics.py` - Computes graph metrics
- `graph_fix.py` - Marks LLM-generated content and updates Concept nodes' text
- `graph2html.py` - Generates interactive HTML with graphical representation
- `graph2viewer.py` - Generates interactive HTML with 3-column representation
- Separate pipeline with own config.toml

All tools use `/src/config.toml` for main pipeline, `/viz/config.toml` for visualization

### Exit Codes

- 0 - SUCCESS; 1 - CONFIG_ERROR; 2 - INPUT_ERROR; 3 - RUNTIME_ERROR; 4 - API_LIMIT_ERROR; 5 - IO_ERROR

Read `/docs/specs/util_exit_codes.md` for details

## Standard Workflow

 1. **READ** task assignment in `/CLAUDE_CODE_<TASK-ID>.md`
 2. **CHECK** specifications in `/docs/` and `/docs/specs/` for modules you'll use
 3. **PLAN** approach (use `think`, `think hard`, `ultrathink` for complex tasks)
 4. **BACKUP** any files you'll modify
 5. **CODE** implementation with tests
 6. **QUALITY** run all checks: `flake8 && black && isort && mypy && ruff check`
 7. **TEST** run pytest: `python -m pytest -v`
 8. **UPDATE** specifications for modules you've modified
 9. **REPORT** create `/CLAUDE_CODE_<TASK-ID>_REPORT.md`
10. **NEVER** proceed beyond the specified scope

## Documentation First Policy

**Key Project Docs**:
- `/docs/Technical_Specification_K2-18.md` - Main project architecture
- `/docs/Technical_Specification_VIZ_K2-18.md` - Visualization architecture
- `/docs/Technical_Specification_VIEW_K2-18.md` - Viewer architecture
- `/docs/Spec_Writing_Guide_K2-18.md` - How to write specs

**ALWAYS** check specs before reading .py files:
- `/docs/specs/cli_*.md` - CLI tools documentation
- `/docs/specs/util_*.md` - Utilities documentation  
- `/docs/specs/viz_*.md` - Visualization documentation

**API References**:
- `/docs/OpenAI_Responses_API_K2-18_Reference.md` - OpenAI Responses API
- `/docs/OpenAI_Embeddings_API_K2-18_Reference.md` - OpenAI Embeddings API
- `/docs/Cytoscape.js_LLM-Oriented_Reference.md` - Cytoscape.js Reference

If the documentation is incomplete or unclear, don't make assumptions - **ASK** user!

To create new specifications **ALWAYS** use `/docs/Spec_Writing_Guide_K2-18.md`

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

- **Style check**: `flake8 .` or `ruff check .`
- **Formatting**: `black .` or `ruff format .`
- **Type checking**: `mypy`
- **Import sorting**: `isort`

## Task Type Patterns

**New CLI Tool / Module**:
1. Write spec following `/docs/Spec_Writing_Guide_K2-18.md`
2. Create tests in `/tests/test_<tool/module>.py`
3. Implement in `/src/<tool/module>.py`
4. Update `/docs/specs/cli_<tool/module>.md`

**Bug Fix**:
1. Reproduce with failing test
2. Fix implementation
3. Verify all tests pass
4. Update affected specs

**Refactoring**:
1. Ensure tests exist
2. Backup original files
3. Refactor incrementally
4. Verify tests still pass

**New Feature**:
1. Update or create spec
2. Write tests first (TDD)
3. Implement feature
4. Integration test if needed

## Common Pitfalls

- **config.toml files are READ-ONLY** - never modify if not required by the task
- **Tests must run sequentially** - no parallel execution
- **Integration tests need real API** - .env must have valid keys
- **viz/ has separate config** - don't confuse with /src/config.toml
- **Specs before code** - always read documentation first
- **Backup before modify** - create _backup_<TASK-ID> files

## Report Template

The report `/CLAUDE_CODE_<TASK-ID>_REPORT.md` should follow this structure:

```markdown
# Task <TASK-ID> Completion Report

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
- flake8: PASS/FAIL
- black: PASS/FAIL
- isort: PASS/FAIL
- mypy: PASS/FAIL
- ruff: PASS/FAIL

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
# Environment Setup
source .venv/bin/activate              # Always first
source .venv/bin/activate.fish         # For fish shell
source .venv/bin/activate.csh          # For csh/tcsh

# Quality Checks (run ALL before tests)
# `.flake8` - flake8 settings
# `pyproject.toml` - settings for black, isort, mypy, and ruff
flake8 src/                            # Check style
black src/                             # Format code
isort src/                             # Sort imports
mypy src/                              # Type checking
ruff check src/                        # Combined linting

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

## Appendix 1: Core Pythonic Codebase Standards

- All CI workflows must pass before code changes may be reviewed
- Changes to the existing code structure require a clear, documented justification
- Every new feature must include unit tests
- Every bug must be reproduced by a unit test before being fixed
- Minor inconsistencies and typos in the existing code may be fixed
- The README.md file must explain the purpose of the repository
- The README.md file must be free of typos, grammar mistakes, and broken English
- Keep README.md concise—avoid duplicating details that belong in separate documentation
- Each class must include an English docstring that states its purpose and shows a usage example
- Each public method or function should include an English docstring; trivial one-liners may be exempt
- Docstrings, specs, and comments must be written in English only, using UTF-8 encoding
- Favor the “fail fast” paradigm over “fail safe”: throw exceptions earlier
- Exception messages must include as much context as possible
- Error and log messages should not end with a period
- Constructors (`__init__`) should be lightweight: limit them to attribute assignments and simple validation.
- Prefer composition; use class inheritance only when it adds clear value
- Avoid explicit “getter” methods; expose read-only data via `@property` where necessary
- Apply Domain-Driven Design concepts when the domain complexity warrants it
- Avoid explicit “setter” methods; favor immutability or controlled updates through properties
- Favor immutable data objects where practical (e.g., `@dataclass(frozen=True)`)
- Provide only one primary constructor; additional constructors must delegate to it via `@classmethod` factories
- Do not create “utility” classes; use module-level functions instead
- Avoid `@staticmethod`; prefer `@classmethod` or standalone functions
- Do not store public constants inside classes; place constants at the top level of the module

## Appendix 2: Detailed Pythonic Testing Standards

- Every change must be covered by a unit test to guarantee repeatability
- Test cases must be as short as possible
- Every test must assert at least once
- Tests must use irregular inputs, such as non-ASCII strings
- Tests may not share object attributes
- Tests must close resources they use, such as file handlers, sockets, and database connections
- Objects must not provide functionality used only by tests
- Tests may not test functionality irrelevant to their stated purpose
- Tests must not clean up after themselves; instead, they must prepare a clean state at the start
- Each test must verify only one specific behavioral pattern of the object it tests
- Tests should store temporary files in temporary directories, not in the codebase directory
- Tests are not allowed to print any log messages
- Tests must not wait indefinitely for any event; they must always stop waiting on a timeout
- Tests must assume the absence of an Internet connection
- Tests must not rely on default configurations of the objects they test, providing custom arguments
- Tests must use ephemeral TCP ports, generated using appropriate library functions
- Tests may create supplementary fixture objects to avoid code duplication
- Each test should target one logical behavior; multiple `assert` statements are fine if they validate that same behavior
- Organize test modules to reflect application modules where practical, but allow flexible grouping when clearer
- Prefer pytest fixtures instead of `setUp`/`tearDown` methods for arranging and cleaning state
- Name tests descriptively in English using snake\_case (e.g., `test_returns_none_if_empty`)
- Avoid asserting on logging unless the log output is part of the public contract
- Skip trivial getters/setters; test them only when they contain meaningful logic
- Use mocks sparingly—favor lightweight fakes or stubs when they keep tests simpler
- When property-based testing is helpful, generate random inputs with a fixed seed to avoid flakiness
- Configure logging in tests to suppress noisy output unless the log is being tested
- For concurrent code, include tests that cover multithreaded scenarios
- Prefer asserting on exception types; check message text only when it forms part of the API contract
- Inline very small fixture data directly in the test; load larger sample files only when needed for clarity
- Generate large test datasets programmatically at runtime to keep the repository lean

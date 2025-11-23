# CLAUDE.md - Project Context for AI Assistant

This file provides context about the `extract-slides` project to help Claude AI assistant understand the codebase and development practices.

## Project Overview

**extract-slides** is a modern Python CLI tool for extracting slides from presentation files (PowerPoint, PDF, Google Slides). The project emphasizes code quality, type safety, and modern Python development practices.

### Key Characteristics

- **Language**: Python 3.9+
- **Type**: Command-line application
- **Status**: Alpha/Early development
- **Package Manager**: uv (modern, fast Python package manager)
- **Architecture**: Simple modular design with clear separation of concerns

## Project Structure

```
extract-slides-py/
├── src/extract_slides/          # Main package (src layout)
│   ├── __init__.py             # Package initialization, exports __version__
│   ├── cli.py                  # Click-based CLI interface
│   ├── extractor.py            # Core slide extraction logic
│   └── py.typed                # PEP 561 marker for type hints
├── tests/                       # Test suite
│   ├── test_cli.py            # CLI tests using Click's CliRunner
│   └── test_extractor.py      # Extractor unit tests
├── pyproject.toml              # Project config (dependencies, tool settings)
├── .python-version             # Python version specification
├── README.md                   # User-facing documentation
└── CLAUDE.md                   # This file
```

### Why src Layout?

The project uses the "src layout" (code in `src/` directory) rather than flat layout. This prevents accidentally importing from the source directory during testing, ensuring tests run against the installed package.

## Technology Stack

### Core Dependencies

- **click** (>=8.1.7): CLI framework for building command-line interfaces
  - Provides decorators for commands, arguments, options
  - Built-in help generation and validation
  - `CliRunner` for testing

### Development Tools

- **uv**: Modern Python package installer (replaces pip, faster resolver)
- **pytest**: Testing framework with fixtures and parameterization
- **pytest-cov**: Code coverage reporting
- **mypy**: Static type checker for type safety
- **ruff**: Fast all-in-one linter and formatter (replaces black, flake8, isort)

## Code Standards

### Type Safety (Strict)

- **All functions must have type annotations** (enforced by mypy)
- Uses `disallow_untyped_defs = true` in mypy config
- Tests are exempt from strict typing requirements
- Package is marked as typed with `py.typed` file

Example:
```python
def extract_slides(input_file: str, output_dir: str, format: str, verbose: bool) -> dict[str, int]:
    """Extract slides with full type annotations."""
    # Implementation
```

### Code Style

- **Line length**: 100 characters (ruff enforced)
- **Quote style**: Double quotes
- **Import sorting**: Handled by ruff (isort rules)
- **Python version**: Target 3.9+ features

### Linting Rules (Ruff)

Active rule sets:
- `E/W`: pycodestyle (PEP 8 compliance)
- `F`: pyflakes (error detection)
- `I`: isort (import sorting)
- `B`: flake8-bugbear (common bugs)
- `C4`: flake8-comprehensions (better comprehensions)
- `UP`: pyupgrade (modern Python idioms)
- `ARG`: flake8-unused-arguments
- `SIM`: flake8-simplify

Exception: `E501` (line length) deferred to formatter

## Development Workflow

### Essential Commands

```bash
# Setup (one-time)
uv venv                          # Create virtual environment
source .venv/bin/activate        # Activate (Unix/macOS)
uv pip install -e ".[dev]"      # Install package + dev deps

# Development cycle
ruff format .                    # Format code
ruff check --fix .              # Lint and auto-fix
mypy src/extract_slides         # Type check
pytest                          # Run tests with coverage

# Run the CLI during development
extract-slides --help           # Installed via pip -e
python -m extract_slides.cli    # Direct module execution
```

### Testing Strategy

- **Framework**: pytest with coverage reporting
- **Coverage target**: >80%
- **CLI testing**: Uses Click's `CliRunner` for isolated filesystem tests
- **Test location**: `tests/` directory, mirroring `src/` structure
- **Naming**: `test_*.py` files, `test_*` functions

Example test pattern:
```python
from click.testing import CliRunner
from extract_slides.cli import main

def test_extract_command_with_temp_file():
    """Tests use isolated filesystem and descriptive names."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        # Test implementation
        result = runner.invoke(main, ["extract", "test.pptx"])
        assert result.exit_code == 0
```

## CLI Architecture

### Command Structure

The CLI uses Click's group/command pattern:

```
extract-slides                   # Main group (entry point)
├── --version                   # Version option
├── extract <input_file>        # Extract command
│   ├── --output-dir, -o       # Output directory
│   ├── --format, -f           # Output format (png|jpg|pdf)
│   └── --verbose, -v          # Verbose logging
└── info                        # Tool information
```

### Design Patterns

- **Entry point**: Defined in `pyproject.toml` under `[project.scripts]`
- **Command groups**: Using `@click.group()` decorator
- **Validation**: Click handles path existence, choices, etc.
- **Error handling**: Exceptions caught and converted to user-friendly messages
- **Output**: Uses `click.echo()` for consistent output/error streams

## Current Implementation Status

### Implemented

- Basic CLI scaffolding with Click
- Command structure (extract, info, version)
- Type-safe foundation with mypy
- Test framework setup
- Development tooling (ruff, mypy, pytest)

### Planned (from README roadmap)

- PowerPoint (.pptx) extraction (core feature)
- PDF presentation support
- Google Slides support
- Batch processing
- Image optimization
- Custom output file naming

## Common Development Tasks

### Adding a New Feature

1. Write tests first (TDD approach preferred)
2. Implement feature in appropriate module
3. Add type annotations
4. Run quality checks: `ruff format . && ruff check . && mypy src/extract_slides && pytest`
5. Update README if user-facing

### Adding a Dependency

- **Runtime**: Add to `dependencies` in `pyproject.toml`, run `uv pip install -e .`
- **Development**: Add to `dev` list in `optional-dependencies`, run `uv pip install -e ".[dev]"`

### Modifying CLI Interface

- Edit `src/extract_slides/cli.py`
- Use Click decorators for new commands/options
- Update tests in `tests/test_cli.py`
- Update README usage examples

## Important Context

### Package vs Module Name

- **Package name**: `extract-slides` (PyPI, pip install)
- **Module name**: `extract_slides` (Python import)
- This is a common pattern (hyphens in package, underscores in code)

### Version Management

- Version defined in `src/extract_slides/__init__.py`
- Exported as `__version__`
- Accessible via CLI: `extract-slides --version`

### Configuration Centralization

All tool configurations are in `pyproject.toml`:
- Project metadata
- Dependencies
- pytest settings
- mypy configuration
- ruff rules
- coverage settings

This follows modern Python best practices (PEP 518).

## When Working on This Project

### Before Making Changes

1. Read existing code to understand patterns
2. Check tests to understand expected behavior
3. Ensure virtual environment is activated
4. Install with `-e` flag for live editing

### Quality Checklist

All changes must:
- [ ] Pass type checking (mypy)
- [ ] Pass linting (ruff check)
- [ ] Be formatted (ruff format)
- [ ] Pass all tests (pytest)
- [ ] Include tests for new functionality
- [ ] Use type hints on all function signatures
- [ ] Follow existing code patterns

### Anti-Patterns to Avoid

- Don't add code without type hints
- Don't skip running tests before committing
- Don't add dependencies without justification
- Don't bypass mypy errors with `# type: ignore` without good reason
- Don't create circular imports between modules
- Don't mix tabs and spaces (use spaces, enforced by ruff)

## Resources

- **uv documentation**: https://github.com/astral-sh/uv
- **Click documentation**: https://click.palletsprojects.com/
- **Ruff rules**: https://docs.astral.sh/ruff/rules/
- **mypy documentation**: https://mypy.readthedocs.io/

## Notes for Claude

- This project is in early development; core extraction logic is not yet implemented
- Tests exist but may be placeholder tests
- Focus on maintainability and code quality over speed of development
- When suggesting changes, always consider type safety implications
- Use the development workflow commands exactly as specified
- Respect the project's strict quality standards (mypy, ruff, tests)

# extract-slides

A modern CLI tool for extracting slides from presentations, built with Python and best practices.

## Features

- 🚀 Fast and efficient slide extraction
- 🎨 Multiple output formats (PNG, JPG, PDF)
- 🔧 Modern Python tooling (uv, ruff, mypy)
- ✅ Comprehensive test coverage
- 📦 Type-safe with full type annotations

## Installation

### For Users

```bash
# Install using uv (recommended)
uv pip install extract-slides

# Or using pip
pip install extract-slides
```

### For Developers

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

#### Prerequisites

- Python 3.9 or higher
- uv (install with: `pip install uv`)

#### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd extract-slides-py
```

2. Create a virtual environment and install dependencies:
```bash
# Create virtual environment
uv venv

# Activate virtual environment
# On Unix/macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install the package with dev dependencies
uv pip install -e ".[dev]"
```

## Usage

### Basic Usage

```bash
# Extract slides from a presentation
extract-slides extract presentation.pptx

# Specify output directory
extract-slides extract presentation.pptx -o my-slides

# Change output format
extract-slides extract presentation.pptx -f jpg

# Verbose output
extract-slides extract presentation.pptx -v
```

### Available Commands

- `extract` - Extract slides from a presentation file
- `info` - Display tool information
- `--version` - Show version information
- `--help` - Show help message

## Development

### Project Structure

```
extract-slides-py/
├── src/
│   └── extract_slides/
│       ├── __init__.py      # Package initialization
│       ├── cli.py           # CLI interface
│       └── extractor.py     # Core extraction logic
├── tests/
│   ├── __init__.py
│   ├── test_cli.py          # CLI tests
│   └── test_extractor.py    # Extractor tests
├── pyproject.toml           # Project configuration
├── README.md
└── .gitignore
```

### Development Workflow

#### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=extract_slides --cov-report=html

# Run specific test file
pytest tests/test_cli.py

# Run specific test
pytest tests/test_cli.py::test_main_help
```

#### Type Checking

```bash
# Run mypy type checker
mypy src/extract_slides

# Check specific file
mypy src/extract_slides/cli.py
```

#### Code Formatting and Linting

```bash
# Format code with ruff
ruff format .

# Run linter
ruff check .

# Fix auto-fixable issues
ruff check --fix .
```

#### Running All Checks

```bash
# Run tests, type checking, and linting
pytest && mypy src/extract_slides && ruff check .
```

### Development Tools

This project uses modern Python development tools:

- **uv**: Fast Python package installer and resolver
- **pytest**: Testing framework with coverage support
- **mypy**: Static type checker for type safety
- **ruff**: Fast Python linter and formatter (replaces black, flake8, isort)

### Code Quality Standards

- All code must pass type checking with mypy
- All code must be formatted with ruff
- All tests must pass
- Maintain test coverage above 80%
- Use type hints for all function signatures

### Making Changes

1. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes following the code quality standards

3. Run all checks:
```bash
# Format code
ruff format .

# Check linting
ruff check --fix .

# Type check
mypy src/extract_slides

# Run tests
pytest
```

4. Commit your changes:
```bash
git add .
git commit -m "Description of your changes"
```

5. Push and create a pull request

### Adding Dependencies

```bash
# Add a new runtime dependency
# Edit pyproject.toml and add to dependencies list, then:
uv pip install -e .

# Add a new dev dependency
# Edit pyproject.toml and add to dev dependencies, then:
uv pip install -e ".[dev]"
```

### Configuration Files

- **pyproject.toml**: Central configuration for project metadata, dependencies, and all tools
- **.python-version**: Specifies Python version for the project
- **src/extract_slides/py.typed**: Marks package as typed for mypy

## Testing

### Test Structure

Tests are organized by module:
- `test_cli.py`: Tests for CLI interface
- `test_extractor.py`: Tests for extraction logic

### Writing Tests

- Use descriptive test names: `test_<what>_<condition>`
- Use pytest fixtures for common setup
- Test both success and error cases
- Use Click's `CliRunner` for CLI tests

Example test:
```python
def test_extract_command_with_temp_file():
    """Test extract command with a temporary file."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("test.pptx", "w") as f:
            f.write("dummy content")

        result = runner.invoke(main, ["extract", "test.pptx"])
        assert result.exit_code == 0
```

## Continuous Integration

The project is set up for CI/CD with the following checks:
- Code formatting (ruff)
- Linting (ruff)
- Type checking (mypy)
- Tests (pytest)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Ensure all tests pass and code is formatted
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'extract_slides'`
- **Solution**: Make sure you've installed the package with `uv pip install -e .`

**Issue**: Tests fail with import errors
- **Solution**: Activate your virtual environment and reinstall dependencies

**Issue**: Type checking errors
- **Solution**: Run `mypy src/extract_slides` to see specific issues

### Getting Help

- Check the [documentation](docs/)
- Open an issue on GitHub
- Review existing issues and pull requests

## Roadmap

- [ ] Support for PowerPoint (.pptx) files
- [ ] Support for PDF presentations
- [ ] Support for Google Slides
- [ ] Batch processing
- [ ] Image optimization options
- [ ] Custom naming schemes for output files

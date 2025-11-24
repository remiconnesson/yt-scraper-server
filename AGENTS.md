- Use Python uv for dependency management.
- Read the README.md
- Enforce strict type checking with `ty` and formatting/linting with `ruff` (add them as dev dependencies).
- Run tests with `pytest` to validate changes.
- Prefer test-driven development when applicable.
- If your made changes that need to be reflected in the README.md, update it.


How to write tests:
- do not test that python features or libraries work, test the code you wrote.
- we are using type-checking, do not write tests that are redundant with the type-checking.

Available Makefile targets:
- `make typecheck` - Run type checking with ty
- `make lint` - Run linting with ruff
- `make format` - Format code with ruff
- `make check` - Run both typecheck and lint
- `make test` - Run tests with pytest
- `make all` - Run format, typecheck, lint, and test
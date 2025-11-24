.PHONY: help typecheck lint format check test all

help:
	@echo "Available targets:"
	@echo "  typecheck  - Run type checking with ty"
	@echo "  lint       - Run linting with ruff"
	@echo "  format     - Format code with ruff"
	@echo "  check      - Run typecheck and lint"
	@echo "  test       - Run tests with pytest"
	@echo "  all        - Run format, typecheck, lint, and test"

typecheck:
	uv run ty check

lint:
	uv run ruff check

format:
	uv run ruff format

check: typecheck lint

test:
	uv run pytest

all: format check test


TAG ?= latest
.PHONY: help typecheck lint format check test all slides-image

help:
	@echo "Available targets:"
	@echo "  typecheck  - Run type checking with ty"
	@echo "  lint       - Run linting with ruff"
	@echo "  format     - Format code with ruff"
	@echo "  check      - Run typecheck and lint"
	@echo "  test       - Run tests with pytest"
	@echo "  all        - Run format, typecheck, lint, and test"
	@echo "  slides-image - Build/push slides-extractor image"

typecheck:
	uv run ty check

lint:
	uv run ruff check

format:
	uv run ruff format

check: typecheck lint

test:
	uv run pytest

slides-image:
ifndef TAG
	$(error TAG must be provided, e.g. make slides-image TAG=v0.0.0)
endif
	sudo docker build -t registry.localhost:5000/slides-extractor:$(TAG) .
	sudo docker push registry.localhost:5000/slides-extractor:$(TAG)
	sed -i "s#image: .*slides-extractor.*#image: registry.localhost:5000/slides-extractor:$(TAG)#g" deploy/prod.yaml

all: format check test


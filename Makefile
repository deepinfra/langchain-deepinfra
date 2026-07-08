.PHONY: help install test unit_test integration_test lint format format_check type_check check build clean

help:
	@echo "install           - sync all dependency groups with uv"
	@echo "test / unit_test  - run offline unit tests"
	@echo "integration_test  - run live tests (needs DEEPINFRA_API_TOKEN)"
	@echo "lint              - ruff check + format check"
	@echo "format            - ruff format + autofix"
	@echo "type_check        - mypy"
	@echo "check             - lint + type_check + unit_test"
	@echo "build             - build sdist + wheel"

install:
	uv sync --all-groups

TEST_FILE ?= tests/unit_tests/

test unit_test:
	uv run --group test pytest $(TEST_FILE)

integration_test:
	uv run --group test pytest tests/integration_tests/

lint:
	uv run --group lint ruff check .
	uv run --group lint ruff format --check .

format:
	uv run --group lint ruff format .
	uv run --group lint ruff check --fix .

type_check:
	uv run --group typing mypy langchain_deepinfra

check: lint type_check unit_test

build:
	uv build

clean:
	rm -rf dist build *.egg-info .pytest_cache .mypy_cache .ruff_cache

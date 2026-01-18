.PHONY: help install install-dev install-enhanced sync test format lint clean

help:  ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install core dependencies only
	uv sync --no-dev

install-dev:  ## Install with development dependencies
	uv sync

install-enhanced:  ## Install with enhanced features (langchain, openai, etc.)
	uv sync --extra enhanced

sync:  ## Sync dependencies with uv.lock
	uv sync

test:  ## Run tests with pytest
	uv run pytest

test-cov:  ## Run tests with coverage report
	uv run pytest --cov=genai_evaluation --cov-report=html --cov-report=term

format:  ## Format code with black
	uv run black genai_evaluation examples

format-check:  ## Check code formatting without modifying
	uv run black --check genai_evaluation examples

lint:  ## Lint code with ruff
	uv run ruff check genai_evaluation examples

lint-fix:  ## Lint and auto-fix issues
	uv run ruff check --fix genai_evaluation examples

typecheck:  ## Type check with mypy
	uv run mypy genai_evaluation

clean:  ## Clean up generated files
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:  ## Build the package
	uv build

run-example:  ## Run the example script
	uv run python examples/prompt_management_evaluation_example.py

shell:  ## Open a shell in the virtual environment
	uv run python

lock:  ## Update uv.lock file
	uv lock

upgrade:  ## Upgrade all dependencies
	uv lock --upgrade

add:  ## Add a new dependency (usage: make add PACKAGE=package-name)
	uv add $(PACKAGE)

add-dev:  ## Add a new dev dependency (usage: make add-dev PACKAGE=package-name)
	uv add --dev $(PACKAGE)

remove:  ## Remove a dependency (usage: make remove PACKAGE=package-name)
	uv remove $(PACKAGE)

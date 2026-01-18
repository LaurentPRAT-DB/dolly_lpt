# Using UV with Dolly LPT

This project uses [uv](https://github.com/astral-sh/uv) - a fast Python package installer and resolver written in Rust.

## Quick Start

### Installation

If you don't have `uv` installed:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew
brew install uv

# Or with pip
pip install uv
```

### Setting Up the Project

```bash
# Clone the repository
git clone https://github.com/LaurentPRAT-DB/dolly_lpt.git
cd dolly_lpt

# Install dependencies (creates .venv automatically)
uv sync

# Or install with development tools
uv sync --dev
```

## Common Commands

### Dependency Management

```bash
# Install core dependencies only
uv sync --no-dev

# Install with dev dependencies
uv sync

# Install with enhanced features (langchain, openai, tiktoken)
uv sync --extra enhanced

# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Remove a dependency
uv remove package-name

# Update all dependencies
uv lock --upgrade

# Update specific package
uv lock --upgrade-package package-name
```

### Running Python Code

```bash
# Run Python scripts
uv run python examples/prompt_management_evaluation_example.py

# Run Python interactively
uv run python

# Run a specific module
uv run -m genai_evaluation.prompt_manager
```

### Development Tasks

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=genai_evaluation --cov-report=html

# Format code
uv run black genai_evaluation examples

# Lint code
uv run ruff check genai_evaluation examples

# Type check
uv run mypy genai_evaluation
```

### Using the Makefile

This project includes a Makefile for convenience:

```bash
# Show all available commands
make help

# Install core dependencies
make install

# Install with dev dependencies
make install-dev

# Install with enhanced features
make install-enhanced

# Run tests
make test

# Run tests with coverage
make test-cov

# Format code
make format

# Lint code
make lint

# Type check
make typecheck

# Clean generated files
make clean

# Build the package
make build
```

## Project Structure

```
dolly_lpt/
├── pyproject.toml          # Project metadata and dependencies
├── uv.lock                 # Locked dependency versions
├── .python-version         # Python version (3.11)
├── .venv/                  # Virtual environment (auto-created by uv)
├── genai_evaluation/       # Main package
│   ├── __init__.py
│   ├── prompt_manager.py
│   ├── rag_evaluator.py
│   ├── config.py
│   └── utils.py
├── examples/               # Example scripts
└── test/                   # Tests
```

## Configuration

### pyproject.toml

The project is configured in `pyproject.toml` with:

- **Core dependencies**: mlflow, databricks-sdk, pandas, numpy
- **Dev dependencies**: pytest, black, ruff, mypy
- **Optional dependencies**: langchain, openai, tiktoken (install with `--extra enhanced`)

### Python Version

This project requires Python >=3.9. The `.python-version` file specifies Python 3.11.

```bash
# Check your Python version
uv run python --version
```

## Virtual Environment

`uv` automatically creates and manages a `.venv` directory in your project:

```bash
# Activate manually (if needed)
source .venv/bin/activate

# Deactivate
deactivate

# uv commands automatically use the virtual environment
```

## Why UV?

- **Fast**: 10-100x faster than pip
- **Reliable**: Uses a deterministic resolver
- **Compatible**: Works with existing pip/pyproject.toml projects
- **Simple**: Single binary, no Python installation needed

### Speed Comparison

```
pip install:     ~30-60 seconds
uv sync:         ~5-10 seconds
```

## Databricks Usage

When using on Databricks, you can install the package in a notebook:

```python
# In a Databricks notebook
%pip install uv
!uv pip install -e /Workspace/Repos/your-username/dolly_lpt

# Or use traditional pip
%pip install -e /Workspace/Repos/your-username/dolly_lpt
```

## Troubleshooting

### Issue: "No Python interpreter found"

```bash
# uv will automatically download Python if needed
uv python install 3.11
```

### Issue: "Package conflict"

```bash
# Clear the lock file and resolve again
rm uv.lock
uv sync
```

### Issue: "Virtual environment is broken"

```bash
# Remove and recreate
rm -rf .venv
uv sync
```

### Issue: "Command not found: uv"

```bash
# Install or update uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or add to PATH
export PATH="$HOME/.cargo/bin:$PATH"
```

## CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh

      - name: Install dependencies
        run: uv sync

      - name: Run tests
        run: uv run pytest
```

## Advanced Usage

### Working with Multiple Python Versions

```bash
# Install specific Python version
uv python install 3.9
uv python install 3.11

# Use specific Python version
uv venv --python 3.9
```

### Custom Virtual Environment Location

```bash
# Create venv in custom location
uv venv /path/to/venv

# Use custom venv
UV_PROJECT_ENVIRONMENT=/path/to/venv uv sync
```

### Dependency Groups

```bash
# Install specific dependency groups
uv sync --group dev
uv sync --group enhanced
```

## Migration from pip

If migrating from pip:

```bash
# Generate uv.lock from existing environment
pip freeze > requirements.txt
uv pip compile requirements.txt -o uv.lock

# Or let uv auto-detect from pyproject.toml
uv sync
```

## Resources

- **UV Documentation**: https://docs.astral.sh/uv/
- **Project Repository**: https://github.com/LaurentPRAT-DB/dolly_lpt
- **Issue Tracker**: https://github.com/LaurentPRAT-DB/dolly_lpt/issues

## Support

For questions or issues:
1. Check the UV documentation: https://docs.astral.sh/uv/
2. Open an issue: https://github.com/LaurentPRAT-DB/dolly_lpt/issues
3. Consult the project README: [README.md](README.md)

# Rizemind

## Prerequisites

First, install the [`uv`](https://github.com/astral-sh/uv) package manager:

- **macOS/Linux**:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- **Windows**:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Installation

Use `uv sync` to install dependencies. Choose your configuration based on your needs:

- **Base installation** (only essential dependencies):

```bash
uv sync --no-group
```

- **Full standard installation** (base + testing + linting):

```bash
uv sync
```

- **ML-specific installation** (includes ML libraries and above):

```bash
uv sync --group ml
```

- **Documentation generation installation** (includes ML and documentation libraries):

```bash
uv sync --group ml --group docs
```

- **All packages** (complete environment including tests, development tools, and documentation):

```bash
uv sync --all-groups
```

## Usage

Run project commands via `uv run --`:

```bash
uv run -- <command>
```

Examples:

- **Run Ruff formatter:**

```bash
 uv run -- ruff check .
```

```bash
 uv run -- ruff check --fix .
```

- **Run Flower:**

```bash
uv run -- flwr run
```

## Running Tests

Automated tests use `pytest`. Execute the full test suite with:

### Unit Tests

```bash
uv run pytest tests/unit
```

### Integration Tests

Requires Anvil installed.

```bash
uv run pytest tests/integration
```

Run specific tests with:

```bash
uv run -- pytest path/to/test_example
```

## Linting and Formatting

Project linting and formatting are handled by [Ruff](https://github.com/astral-sh/ruff):

```bash
uv run -- ruff check --fix .

uv run -- ruff format
```

### VSCode Integration

For smoother workflow in VSCode:

1. Install the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff).
2. Enable "formatting on save" by navigating to **Settings (Ctrl+,)**, search for `ruff save`, and toggle it on.

## Examples

Check the [examples](https://github.com/T-RIZE-Group/rizemind/tree/main/examples) directory for detailed instructions on running specific examples.

## Documentation

Documentation is generated using [Sphinx](https://www.sphinx-doc.org/) and written in reStructuredText (reST). To build and preview documentation locally:

```bash
cd sphinx
uv run -- sphinx-autobuild source build/html
```

Access the generated documentation via `http://localhost:8000` in your browser.

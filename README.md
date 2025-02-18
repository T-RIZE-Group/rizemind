# T-RIZE DML Infra

## Requirements

1. Install Poetry:
## Install instructions

1. Create venv
```shell 
python -m venv .venv
```

2. Activate the python venv environment
**Linux/MacOS**:
```shell
source .venv/bin/activate
```
**windows**:
```bash
.venv\Scripts\activate
```

3. Install local packages
```
pip install -e .
```
4. Select the `.venv` interpreter for your IDE.

5. Install the linter
```
curl -LsSf https://astral.sh/ruff/install.sh | sh
```

**VSCode**: `ctrl+shift+p` search for `python: select interpreter` then select the `.venv` conda env.

## Running tests
Automated tests are written using pytest.

1. Install dev dependencies
```shell
pip install -e ".[dev]"
```
2. ```shell
pytest
```

## Run the linter

Use Ruff to run and fix linting errors

```shell
ruff check --fix
```
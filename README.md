# T-RIZE DML Infra

## Requirements

1. Install Miniconda: https://docs.anaconda.com/free/miniconda/#quick-command-line-install

## Install instructions

1. Activate the python venv environment
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

**VSCode**: `ctrl+shift+p` search for `python: select interpreter` then select the `.venv` conda env.

## Project Structure

- /packages/authentication: main package containing the code extending the Flower framework for authentication
- /packages/examples: examples on how to use code from other packages
- /packages/notebooks: keep track of your notebooks
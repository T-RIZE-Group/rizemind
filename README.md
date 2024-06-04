# T-RIZE DML Infra

## Requirements

1. Install Miniconda: https://docs.anaconda.com/free/miniconda/#quick-command-line-install

## Install instructions
1. Create Conda virtual environment
```shell
conda env create -f env.yml
```
2. Activate the environment
```shell
conda activate rize-dml
```
3. Install pip dependencies
```
pip install -r requirements.txt
```

### Export environment after changes

```shell
pip freeze > requirements.txt
```
```shell
conda env export >> env.yml
```

## Project Structure

- /packages/dml: main package containing the code extending the Flower framework
- /packages/examples: examples on how to use code from other packages
- /packages/notebooks: keep track of your notebooks
- /env.yml: Conda environment config
- /requirements.txt: pip list of dependencies
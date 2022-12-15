#!/bin/bash
VENV_PATH=.DLA_env_linux

echo "Python version: $(python3 --version)"
echo "Initializing virtual environment in $VENV_PATH ..."

python3 -m venv $VENV_PATH

$VENV_PATH/bin/python -m pip install -U pip
$VENV_PATH/bin/pip install -U wheel setuptools
$VENV_PATH/bin/pip install -r ./requirements.txt

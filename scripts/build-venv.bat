@echo off
set VENV_PATH=.DLA_env

echo Python version:
python --version
echo Creating virtual environment in %VENV_PATH% ...

python -m venv %VENV_PATH%

%VENV_PATH%\Scripts\python.exe -m pip install -U pip
%VENV_PATH%\Scripts\pip install -U wheel setuptools
%VENV_PATH%\Scripts\pip install -r .\requirements.txt

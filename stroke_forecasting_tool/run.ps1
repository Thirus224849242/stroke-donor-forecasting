# Always runs the app through the project's venv, regardless of what
# `python`/`streamlit` resolve to on PATH in the current shell -- avoids the
# "ModuleNotFoundError: No module named 'sqlalchemy'" (or any other missing
# dependency) that happens when this is launched with system Python instead,
# which never has this project's packages installed into it.
#
# Usage: from this folder, run  .\run.ps1
$ErrorActionPreference = 'Stop'
$venvPython = Join-Path $PSScriptRoot '..\venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    Write-Error "Venv not found at $venvPython -- create it first: python -m venv ..\venv; ..\venv\Scripts\pip install -r requirements.txt"
    exit 1
}
& $venvPython -m streamlit run (Join-Path $PSScriptRoot 'app.py') --server.port 8501 --server.headless true

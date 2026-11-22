$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    python -m venv (Join-Path $projectRoot ".venv")
}

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r (Join-Path $projectRoot "requirements-dev.txt")
& $venvPython -m pip install -e (Join-Path $projectRoot "vendor\BeyondBench") --no-deps
& $venvPython -m pip install -e $projectRoot --no-deps

Write-Host "Environment ready: $venvPython"


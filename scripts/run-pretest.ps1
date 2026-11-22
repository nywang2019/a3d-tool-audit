param(
    [string]$Config = "configs\pretest.yaml",
    [string]$ResultsDir = "results"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

Push-Location $projectRoot
try {
    & $python -m llm_eval.cli validate --config $Config
    & $python -m llm_eval.cli run --config $Config --results-dir $ResultsDir
}
finally {
    Pop-Location
}


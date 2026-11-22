param(
    [string]$Config = "configs\pretest.yaml"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

Push-Location $projectRoot
try {
    & $python -m llm_eval.cli plan --config $Config
}
finally {
    Pop-Location
}


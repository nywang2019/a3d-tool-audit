param(
    [switch]$Full
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$arguments = @("-m", "llm_eval.cli", "status")
if ($Full) {
    $arguments += "--full"
}

Push-Location $projectRoot
try {
    & $python @arguments
}
finally {
    Pop-Location
}


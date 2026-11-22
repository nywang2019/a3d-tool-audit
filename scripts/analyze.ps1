param(
    [Parameter(Mandatory = $true)]
    [string]$Results,
    [string]$StaticScores
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$arguments = @("-m", "llm_eval.cli", "analyze", "--results", $Results)
if ($StaticScores) {
    $arguments += @("--static-scores", $StaticScores)
}

Push-Location $projectRoot
try {
    & $python @arguments
}
finally {
    Pop-Location
}


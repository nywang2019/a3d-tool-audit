param(
    [string]$Config = "configs\main_experiment.yaml",
    [string]$ResultsDir = "results",
    [string]$StaticScores = "data\static_scores.csv"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

Push-Location $projectRoot
try {
    & $python -m llm_eval.cli validate --config $Config
    $runArguments = @("-m", "llm_eval.cli", "run", "--config", $Config, "--results-dir", $ResultsDir)
    if (Test-Path -LiteralPath $StaticScores) {
        $runArguments += @("--static-scores", $StaticScores)
    }
    else {
        Write-Warning "Static score file not found: $StaticScores. Dynamic results will run without rank analysis."
    }
    & $python @runArguments
}
finally {
    Pop-Location
}


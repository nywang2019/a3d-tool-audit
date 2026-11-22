param(
    [Parameter(Mandatory = $true)]
    [string]$RunRoot,
    [Parameter(Mandatory = $true)]
    [string]$OutputPath,
    [int]$PollSeconds = 5
)

$ErrorActionPreference = "Stop"
$manifestPath = Join-Path $RunRoot "run_manifest.json"
$modelNames = @(
    "deepseek-flash",
    "kimi-k3.1-pro",
    "glm-5.3-flashx",
    "minimax-m3.1-flash",
    "qwen3.8-max"
)

while ($true) {
    $models = [ordered]@{}
    $completed = 0
    $correct = 0
    $technical = 0
    $requests = 0
    foreach ($model in $modelNames) {
        $trialsPath = Join-Path $RunRoot "$model\trials.json"
        $trials = @()
        if (Test-Path -LiteralPath $trialsPath) {
            $loaded = Get-Content -Raw -LiteralPath $trialsPath | ConvertFrom-Json
            $trials = @($loaded)
        }
        $modelCorrect = @($trials | Where-Object { $_.correct -eq $true }).Count
        $modelTechnical = @(
            $trials | Where-Object { $_.outcome -eq "technical_failure" }
        ).Count
        $rawPath = Join-Path $RunRoot "$model\raw"
        $modelRequests = if (Test-Path -LiteralPath $rawPath) {
            @(Get-ChildItem -LiteralPath $rawPath -Filter "*.json" -File).Count
        } else {
            0
        }
        $models[$model] = [ordered]@{
            completed = $trials.Count
            expected = 10
            correct = $modelCorrect
            technical_failures = $modelTechnical
            requests = $modelRequests
        }
        $completed += $trials.Count
        $correct += $modelCorrect
        $technical += $modelTechnical
        $requests += $modelRequests
    }
    $payload = [ordered]@{
        status = if (Test-Path -LiteralPath $manifestPath) {
            "completed"
        } else {
            "running"
        }
        completed = $completed
        expected = 50
        correct = $correct
        technical_failures = $technical
        requests = $requests
        models = $models
        updated_at = (Get-Date).ToUniversalTime().ToString("o")
        run_root = $RunRoot
    }
    $payload | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 -LiteralPath $OutputPath
    if ($payload.status -eq "completed") { break }
    Start-Sleep -Seconds $PollSeconds
}

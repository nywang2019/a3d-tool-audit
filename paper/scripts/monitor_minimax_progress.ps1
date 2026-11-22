param(
    [Parameter(Mandatory = $true)]
    [string]$RunRoot,
    [Parameter(Mandatory = $true)]
    [string]$OutputPath,
    [int]$PollSeconds = 5
)

$ErrorActionPreference = "Stop"
$trialsPath = Join-Path $RunRoot "minimax-m3.1-flash\trials.json"
$rawPath = Join-Path $RunRoot "minimax-m3.1-flash\raw"
$manifestPath = Join-Path $RunRoot "run_manifest.json"

while ($true) {
    $trials = @()
    if (Test-Path -LiteralPath $trialsPath) {
        $loaded = Get-Content -Raw -LiteralPath $trialsPath | ConvertFrom-Json
        $trials = @($loaded)
    }
    $rawCount = 0
    if (Test-Path -LiteralPath $rawPath) {
        $rawCount = @(Get-ChildItem -LiteralPath $rawPath -Filter "*.json" -File).Count
    }

    $baseline = @($trials | Where-Object { [int]$_.distractor_count -eq 0 })
    $compatible = @($trials | Where-Object { [int]$_.distractor_count -eq 5 })
    $status = if (Test-Path -LiteralPath $manifestPath) { "completed" } else { "running" }
    $payload = [ordered]@{
        title = "MiniMax 正式对比实验"
        status = $status
        completed = $trials.Count
        expected = 40
        correct = @($trials | Where-Object { $_.correct -eq $true }).Count
        technical_failures = @($trials | Where-Object { $_.outcome -eq "technical_failure" }).Count
        requests = $rawCount
        baseline = [ordered]@{
            completed = $baseline.Count
            correct = @($baseline | Where-Object { $_.correct -eq $true }).Count
        }
        compatible_5 = [ordered]@{
            completed = $compatible.Count
            correct = @($compatible | Where-Object { $_.correct -eq $true }).Count
        }
        updated_at = (Get-Date).ToUniversalTime().ToString("o")
        run_root = $RunRoot
    }
    $payload | ConvertTo-Json | Set-Content -Encoding utf8 -LiteralPath $OutputPath
    if ($status -eq "completed") { break }
    Start-Sleep -Seconds $PollSeconds
}

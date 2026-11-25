param(
    [string]$Timestamp = "20261101"
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$submission = Join-Path $root "paper\submission"
$figures = Join-Path $root "paper\figures"
$tables = Join-Path $root "paper\tables"
$supplement = Join-Path $root "paper\supplement"
$tmp = Join-Path $root "tmp"
$sourceStage = Join-Path $tmp "submission-source-$Timestamp"
$supplementStage = Join-Path $tmp "submission-supplement-$Timestamp"

New-Item -ItemType Directory -Force $sourceStage | Out-Null
New-Item -ItemType Directory -Force $supplementStage | Out-Null
New-Item -ItemType Directory -Force (Join-Path $supplementStage "tables") | Out-Null
New-Item -ItemType Directory -Force (Join-Path $supplementStage "case_studies") | Out-Null
New-Item -ItemType Directory -Force (Join-Path $supplementStage "figures") | Out-Null

foreach ($name in @("main.tex", "references.bib", "sn-jnl.cls", "sn-basic.bst", "VERSION.txt")) {
    Copy-Item -LiteralPath (Join-Path $submission $name) -Destination $sourceStage
}
Copy-Item -Path (Join-Path $submission "figure*.pdf") -Destination $sourceStage

foreach ($name in @("dataset_card.md", "diagnostic_protocol.md", "derived_analysis.md",
        "asp_method.md",
        "sealed_probe.md",
        "gating_error_transitions.csv", "gating_error_transitions.md",
        "repeatability_summary.json", "repeatability_summary.md",
        "reproducibility.md", "supplement_README.md")) {
    Copy-Item -LiteralPath (Join-Path $supplement $name) -Destination $supplementStage
}
foreach ($pattern in @("*.csv", "*.md", "*.tex")) {
    Copy-Item -Path (Join-Path $tables $pattern) -Destination (Join-Path $supplementStage "tables")
}
Copy-Item -Path (Join-Path $root "paper\supplement\case_studies\*.md") -Destination (Join-Path $supplementStage "case_studies")
Copy-Item -LiteralPath (Join-Path $root "paper\generated_facts.json") -Destination $supplementStage
Copy-Item -LiteralPath (Join-Path $root "paper\model_coverage_audit.md") -Destination $supplementStage
Copy-Item -LiteralPath (Join-Path $root "paper\model_coverage_audit.json") -Destination $supplementStage
Copy-Item -LiteralPath (Join-Path $root "docs\funcbench_control15_results.md") -Destination (Join-Path $supplementStage "control15_results.md")
Copy-Item -LiteralPath (Join-Path $root "docs\funcbench_minimax_main20_results.md") -Destination (Join-Path $supplementStage "minimax_main20_results.md")
Copy-Item -LiteralPath (Join-Path $figures "figure2_dose_response.pdf") -Destination (Join-Path $supplementStage "figures")

$sourceZip = Join-Path $submission "applied_intelligence_source.zip"
$supplementZip = Join-Path $submission "applied_intelligence_supplement.zip"
Compress-Archive -Path (Join-Path $sourceStage "*") -DestinationPath $sourceZip -Force
Compress-Archive -Path (Join-Path $supplementStage "*") -DestinationPath $supplementZip -Force

$hashRows = foreach ($path in @($sourceZip, $supplementZip, (Join-Path $submission "main.pdf"))) {
    $item = Get-Item -LiteralPath $path
    [ordered]@{
        file = $item.Name
        bytes = $item.Length
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash
    }
}
$hashRows | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $submission "package_hashes.json")
$hashRows | Format-Table -AutoSize

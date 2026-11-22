param(
    [Parameter(Mandatory = $true)]
    [string]$Version,
    [Parameter(Mandatory = $true)]
    [string]$Date,
    [Parameter(Mandatory = $true)]
    [string]$Summary,
    [string]$Status = "candidate"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$submission = Join-Path $root "paper\submission"
$versions = Join-Path $root "paper\versions"
$archive = Join-Path $versions $Version

if (Test-Path -LiteralPath $archive) {
    throw "Version already exists: $Version"
}

$required = @(
    "main.pdf",
    "main.tex",
    "applied_intelligence_source.zip",
    "applied_intelligence_supplement.zip",
    "package_hashes.json"
)
foreach ($name in $required) {
    if (-not (Test-Path -LiteralPath (Join-Path $submission $name))) {
        throw "Missing active artifact: $name"
    }
}

New-Item -ItemType Directory -Path $archive | Out-Null
foreach ($name in $required) {
    Copy-Item -LiteralPath (Join-Path $submission $name) -Destination $archive
}

$hashes = @{}
foreach ($name in @("main.pdf", "applied_intelligence_source.zip", "applied_intelligence_supplement.zip")) {
    $hashes[$name] = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $submission $name)).Hash
}
$manifest = [ordered]@{
    version = $Version
    date = $Date
    status = $Status
    summary = $Summary
    active_source = "paper/submission"
    artifacts = $hashes
}
$manifest | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $archive "VERSION_MANIFEST.json")
@"
# $Version

- Date: $Date
- Status: $Status
- Main changes: $Summary

## SHA-256

- main.pdf: $($hashes['main.pdf'])
- source ZIP: $($hashes['applied_intelligence_source.zip'])
- supplement ZIP: $($hashes['applied_intelligence_supplement.zip'])
"@ | Set-Content -Encoding utf8 (Join-Path $archive "CHANGELOG.md")

$historyCsv = Join-Path $versions "VERSION_HISTORY.csv"
$row = [pscustomobject]@{
    version = $Version
    date = $Date
    status = $Status
    main_changes = $Summary
    main_pdf_sha256 = $hashes["main.pdf"]
    source_zip_sha256 = $hashes["applied_intelligence_source.zip"]
    supplement_zip_sha256 = $hashes["applied_intelligence_supplement.zip"]
    archive = "paper/versions/$Version"
}
$row | Export-Csv -LiteralPath $historyCsv -NoTypeInformation -Append -Encoding utf8

$historyMarkdown = Join-Path $versions "VERSION_HISTORY.md"
$historyRows = Import-Csv -LiteralPath $historyCsv
$markdownLines = @(
    "# Paper version history",
    "",
    "| Version | Date | Status | Main changes | Frozen archive |",
    "|---|---|---|---|---|"
)
foreach ($historyRow in $historyRows) {
    $archiveCell = if ($historyRow.archive) {
        "``$($historyRow.archive)/``"
    } else {
        "Not retained"
    }
    $summaryCell = $historyRow.main_changes.Replace("|", "\|")
    $markdownLines += "| $($historyRow.version) | $($historyRow.date) | $($historyRow.status) | $summaryCell | $archiveCell |"
}
$markdownLines += @(
    "",
    "The first frozen artifact snapshot begins with v0.5.0."
)
$markdownLines | Set-Content -LiteralPath $historyMarkdown -Encoding utf8
Set-Content -LiteralPath (Join-Path $versions "ACTIVE_VERSION") -Value $Version -Encoding utf8
Set-Content -LiteralPath (Join-Path $submission "VERSION.txt") -Value $Version -Encoding utf8
Copy-Item -LiteralPath (Join-Path $submission "VERSION.txt") -Destination $archive

Write-Output "Archived $Version at $archive"

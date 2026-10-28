param(
    [string]$ReportRoot
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($ReportRoot)) {
    $stamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssfffZ") + "-$PID"
    $ReportRoot = Join-Path $repoRoot "test_reports/Facade_parameter_loop_release/$stamp"
}

$technicalRoot = Join-Path $ReportRoot "technical"
New-Item -ItemType Directory -Path $technicalRoot -Force | Out-Null
$logPath = Join-Path $technicalRoot "facade_parameter_loop_tests.log"
Push-Location $repoRoot
try {
    Write-Host "Running only the facade parameter-loop Release contract tests..."
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & cargo test --release --features perf-gates --bin perf-gates facade_parameter_loop -- --nocapture --test-threads=1 2>&1 |
            Tee-Object -FilePath $logPath | ForEach-Object { Write-Host $_ }
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

$testSummary = Get-Content -LiteralPath $logPath |
    Where-Object { $_ -match '^test result:' } |
    Select-Object -Last 1
$status = if ($exitCode -eq 0) { "passed" } else { "failed" }
$summary = @(
    "# Facade Parameter-Loop Release Matrix",
    "",
    "Focused Release correctness run for the BVP_api and StiffAPI large parameter-loop perf-gate scenarios. This does not record or compare performance baselines.",
    "",
    "| status | exit code | test summary | log |",
    "|---|---:|---|---|",
    "| $status | $exitCode | $(if ($testSummary) { $testSummary.Trim() } else { 'no test summary emitted' }) | technical/facade_parameter_loop_tests.log |"
)
$summaryPath = Join-Path $ReportRoot "summary.md"
Set-Content -LiteralPath $summaryPath -Value $summary -Encoding utf8
Write-Host "`nFacade parameter-loop Release status=$status"
Write-Host "summary=$summaryPath"
if ($exitCode -ne 0) { exit $exitCode }
exit 0

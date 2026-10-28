param(
    [string]$ReportRoot
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($ReportRoot)) {
    $stamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssfffZ") + "-$PID"
    $ReportRoot = Join-Path $repoRoot "test_reports/IVP_parameter_loop_release/$stamp"
}

$technicalRoot = Join-Path $ReportRoot "technical"
New-Item -ItemType Directory -Path $technicalRoot -Force | Out-Null
$steps = @(
    [pscustomobject]@{ Name = "BDF"; Filter = "numerical::BDF::parameter_loop" },
    [pscustomobject]@{ Name = "Backward Euler"; Filter = "numerical::BE::parameter_loop" },
    [pscustomobject]@{ Name = "LSODE2"; Filter = "numerical::LSODE2::parameter_loop" },
    [pscustomobject]@{ Name = "Radau"; Filter = "numerical::Radau::parameter_loop" },
    [pscustomobject]@{ Name = "StiffAPI"; Filter = "stiff_parameter_loop" }
)

$results = [System.Collections.Generic.List[object]]::new()
Push-Location $repoRoot
try {
    foreach ($step in $steps) {
        $logPath = Join-Path $technicalRoot ("{0}.log" -f ($step.Name -replace '\s+', '_'))
        $watch = [Diagnostics.Stopwatch]::StartNew()
        Write-Host "`n=== IVP parameter-loop Release: $($step.Name) ==="
        # Windows PowerShell 5 promotes native stderr to NativeCommandError when
        # ErrorActionPreference is Stop; Cargo warnings are expected stderr.
        $previousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & cargo test --release --lib $step.Filter -- --nocapture --test-threads=1 2>&1 |
                Tee-Object -FilePath $logPath | ForEach-Object { Write-Host $_ }
        }
        finally {
            $ErrorActionPreference = $previousErrorActionPreference
        }
        $exitCode = $LASTEXITCODE
        $watch.Stop()
        $testSummary = Get-Content -LiteralPath $logPath |
            Where-Object { $_ -match '^test result:' } |
            Select-Object -Last 1
        $results.Add([pscustomobject][ordered]@{
            area = $step.Name
            status = if ($exitCode -eq 0) { "passed" } else { "failed" }
            exit_code = $exitCode
            duration_s = [math]::Round($watch.Elapsed.TotalSeconds, 1)
            tests = if ($testSummary) { $testSummary.Trim() } else { "no test summary emitted" }
            log = "technical/$([IO.Path]::GetFileName($logPath))"
        })
    }
}
finally {
    Pop-Location
}

$table = @(
    "# IVP Parameter-Loop Release Matrix",
    "",
    "Each native solver route and the StiffAPI facade is tested independently. A failed matrix point does not suppress later checks.",
    "",
    "| area | status | exit code | duration (s) | test summary | log |",
    "|---|---|---:|---:|---|---|"
)
foreach ($result in $results) {
    $summary = $result.tests.Replace('|', '\|')
    $table += "| $($result.area) | $($result.status) | $($result.exit_code) | $($result.duration_s) | $summary | $($result.log) |"
}
$failed = @($results | Where-Object status -ne "passed").Count
$table += ""
$table += "Outcome: **$(if ($failed -eq 0) { 'passed' } else { "failed ($failed check(s))" })**"
$summaryPath = Join-Path $ReportRoot "summary.md"
Set-Content -LiteralPath $summaryPath -Value $table -Encoding utf8

Write-Host "`nIVP parameter-loop Release summary"
$results | Format-Table area, status, exit_code, duration_s -AutoSize
Write-Host "summary=$summaryPath"
if ($failed -gt 0) { exit 1 }
exit 0

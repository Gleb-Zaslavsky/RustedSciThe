param(
    [string]$Python = ".\.venv-gavin-win\Scripts\python.exe",
    [switch]$SkipBench,
    [switch]$SkipPythonReference,
    [switch]$FailOnAny
)

# Gavin LM focused release runner. Each step is isolated so one failure does
# not suppress the remaining compact evidence reports.
$ErrorActionPreference = "Continue"
$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$root = Join-Path (Get-Location) "test_reports\Optimization_Gavin_release_manual\$stamp"
$reports = Join-Path $root "reports"
$technical = Join-Path $root "technical"
New-Item -ItemType Directory -Force -Path $root, $reports, $technical | Out-Null

$env:RST_TEST_REPORT_DIR = $reports
$env:RST_TEST_REPORT_PROFILE = "release"
$env:RST_TEST_REPORT_ARCHIVE = "always"
$env:RST_TEST_REPORT_STDOUT = "off"
$env:GAVIN_REFERENCE_REPORT = Join-Path $reports "python_reference.md"

$results = New-Object System.Collections.Generic.List[object]
function Invoke-GavinStep {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][scriptblock]$Action,
        [switch]$RequireRustTests
    )

    $log = Join-Path $technical "$Name.log"
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    $exitCode = 0
    $testsPassed = $null
    $testsFailed = $null
    try {
        & $Action *> $log
        if ($null -ne $LASTEXITCODE) { $exitCode = [int]$LASTEXITCODE }
    }
    catch {
        $exitCode = 1
        Add-Content -Path $log -Value ("`n[runner exception] " + $_.Exception.Message)
    }

    if ($RequireRustTests -and $exitCode -eq 0) {
        $logText = Get-Content -LiteralPath $log -Raw
        $testResults = [regex]::Matches(
            $logText,
            '(?m)^test result: .*?(?<passed>\d+) passed; (?<failed>\d+) failed;'
        )
        if ($testResults.Count -eq 0) {
            $exitCode = 1
            Add-Content -Path $log -Value "`n[runner validation] no Rust test result line found"
        }
        else {
            $lastTestResult = $testResults[$testResults.Count - 1]
            $testsPassed = [int]$lastTestResult.Groups['passed'].Value
            $testsFailed = [int]$lastTestResult.Groups['failed'].Value
            if (($testsPassed + $testsFailed) -eq 0) {
                $exitCode = 1
                Add-Content -Path $log -Value "`n[runner validation] Cargo matched zero tests; treating this step as failed"
            }
        }
    }

    $watch.Stop()
    $status = if ($exitCode -eq 0) { "passed" } else { "failed" }
    $results.Add([pscustomobject][ordered]@{
        step = $Name
        status = $status
        exit_code = $exitCode
        tests_passed = $testsPassed
        tests_failed = $testsFailed
        duration_s = [math]::Round($watch.Elapsed.TotalSeconds, 1)
        technical_log = $log
    })
    $testCount = if ($null -eq $testsPassed) { "n/a" } else { "$testsPassed passed, $testsFailed failed" }
    Write-Host ("[{0}] {1} exit={2} tests={3}" -f $Name, $status, $exitCode, $testCount)
}

Invoke-GavinStep -Name "universal_api" -RequireRustTests -Action {
    cargo test --release --lib --no-default-features numerical::optimization::universal_fitting::tests -- --test-threads=1
}
Invoke-GavinStep -Name "gavin_controller" -RequireRustTests -Action {
    cargo test --release --lib --no-default-features numerical::optimization::lm_gavin::tests -- --test-threads=1
}
Invoke-GavinStep -Name "symbolic_builder" -RequireRustTests -Action {
    cargo test --release --lib --no-default-features numerical::optimization::gavin_fitting::tests -- --test-threads=1
}
Invoke-GavinStep -Name "frontend_story" -RequireRustTests -Action {
    cargo test --release --lib --no-default-features numerical::optimization::gavin_story_tests::gavin_universal_fitting_frontend_performance_story -- --ignored --nocapture --test-threads=1
}
Invoke-GavinStep -Name "python_controller_parity_story" -RequireRustTests -Action {
    cargo test --release --lib --no-default-features numerical::optimization::gavin_story_tests::gavin_python_controller_history_parity_story -- --ignored --nocapture --test-threads=1
}
Invoke-GavinStep -Name "active_bounds_parity_story" -RequireRustTests -Action {
    cargo test --release --lib --no-default-features numerical::optimization::gavin_story_tests::gavin_python_active_coefficient_bounds_parity_story -- --ignored --nocapture --test-threads=1
}

if (-not $SkipBench) {
    Invoke-GavinStep -Name "performance_matrix" -Action {
        cargo bench --no-default-features --bench gavin_fitting
    }
}

if (-not $SkipPythonReference) {
    if (Test-Path -LiteralPath $Python) {
        Invoke-GavinStep -Name "python_reference" -Action {
            & $Python .\src\numerical\optimization\gavin_reference\examples\exponential_sinusoidal_fit.py
        }
    }
    else {
        Invoke-GavinStep -Name "python_reference" -Action {
            throw "Python interpreter not found: $Python"
        }
    }
}

$summaryPath = Join-Path $root "release_summary.md"
$table = @(
    "| step | status | exit_code | tests_passed | tests_failed | duration_s | technical_log |",
    "|---|---|---:|---:|---:|---:|---|"
)
foreach ($row in $results) {
    $logPath = $row.technical_log -replace '\\', '/'
    $passed = if ($null -eq $row.tests_passed) { "-" } else { $row.tests_passed }
    $failedCount = if ($null -eq $row.tests_failed) { "-" } else { $row.tests_failed }
    $table += "| $($row.step) | $($row.status) | $($row.exit_code) | $passed | $failedCount | $($row.duration_s) | $logPath |"
}
$failed = @($results | Where-Object { $_.exit_code -ne 0 }).Count
$overall = if ($failed -eq 0) { "completed" } else { "completed_with_failures" }
$summary = @(
    "# Gavin LM Release Matrix",
    "",
    "- status: $overall",
    "- recorded_at_utc: $((Get-Date).ToUniversalTime().ToString('o'))",
    "- report_root: $root",
    "- compact_reports: $reports",
    "- technical_logs: $technical",
    "",
    "## Steps",
    "",
    ($table -join "`n"),
    "",
    "Compact tables are stored separately from Cargo/Python technical logs."
)
Set-Content -Path $summaryPath -Value ($summary -join "`n") -Encoding UTF8
Set-Content -Path (Join-Path $reports "release_summary.md") -Value ($summary -join "`n") -Encoding UTF8
Write-Host "reports=$root"
Write-Host "technical=$technical"
Write-Host "summary=$summaryPath"
if ($FailOnAny -and $failed -gt 0) { exit 1 }
exit 0

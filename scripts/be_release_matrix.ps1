param(
    [string]$Workloads = "diffusion-chain,combustion-like,robertson,three-body",
    [string]$Dimensions = "8,32",
    [string]$Routes = "native-analytic,native-fd,lambdify-expr-legacy,lambdify-atom-native",
    [int]$Continuation = 4,
    [switch]$IncludeIgnoredStories,
    [switch]$IncludeAot,
    [switch]$IncludeCriterion,
    [switch]$IncludeAllocationAudit,
    [switch]$FailOnAny
)

# BE release campaign runner. Each step is independent so one failure does
# not suppress later correctness, lifecycle, or performance evidence.
$ErrorActionPreference = "Continue"
$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$root = Join-Path (Get-Location) "test_reports\BE_release_manual\$stamp"
$reports = Join-Path $root "reports"
$technical = Join-Path $root "technical"
New-Item -ItemType Directory -Force -Path $reports, $technical | Out-Null

$env:RST_TEST_REPORT_DIR = $reports
$env:RST_TEST_REPORT_PROFILE = "release"
$env:RST_TEST_REPORT_ARCHIVE = "always"
$env:RST_TEST_REPORT_STDOUT = "off"
$env:BE_BENCH_WORKLOADS = $Workloads
$env:BE_BENCH_DIMENSIONS = $Dimensions
$env:BE_BENCH_CONTINUATION = [string]$Continuation
$env:BE_BENCH_ROUTES = if ($IncludeAot) {
    $Routes + ",aot-expr-legacy-tcc,aot-atom-native-tcc"
} else {
    $Routes
}

$results = New-Object System.Collections.Generic.List[object]
function Invoke-BeStep([string]$Name, [scriptblock]$Action) {
    $technicalLog = Join-Path $technical "$Name.log"
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    $exitCode = 0
    try {
        & $Action *> $technicalLog
        if ($null -ne $LASTEXITCODE) { $exitCode = [int]$LASTEXITCODE }
    }
    catch {
        $exitCode = 1
        Add-Content -Path $technicalLog -Value ("`n[runner exception] " + $_.Exception.Message)
    }
    $watch.Stop()
    $status = if ($exitCode -eq 0) { "passed" } else { "failed" }
    $results.Add([pscustomobject][ordered]@{
        step = $Name
        status = $status
        exit_code = $exitCode
        duration_s = [math]::Round($watch.Elapsed.TotalSeconds, 1)
        technical_log = $technicalLog
    })
    Write-Host ("[{0}] {1} exit={2}" -f $Name, $status, $exitCode)
}

Invoke-BeStep "story_core" {
    cargo test --release --lib --no-default-features numerical::BE::tests -- --nocapture --test-threads=1
}
Invoke-BeStep "story_performance" {
    cargo test --release --lib --no-default-features numerical::BE::performance_story_tests -- --nocapture --test-threads=1
}
Invoke-BeStep "story_cold_process" {
    cargo test --release --lib --no-default-features numerical::BE::cold_process_story_tests -- --nocapture --test-threads=1
}

if ($IncludeIgnoredStories) {
    Invoke-BeStep "story_ignored" {
        cargo test --release --lib --no-default-features numerical::BE:: -- --ignored --nocapture --test-threads=1
    }
}

Invoke-BeStep "compact_workload_matrix" {
    cargo bench --no-default-features --bench be_workloads -- --noplot
}

if ($IncludeCriterion) {
    Invoke-BeStep "criterion_workloads" {
        cargo bench --no-default-features --bench be_workload_benches -- --noplot
    }
    Invoke-BeStep "criterion_symbolic_frontends" {
        cargo bench --no-default-features --bench be_symbolic_frontend_benches -- --noplot
    }
    Invoke-BeStep "criterion_symbolic_execution" {
        cargo bench --no-default-features --bench be_symbolic_execution_benches -- --noplot
    }
}

if ($IncludeAllocationAudit) {
    Invoke-BeStep "criterion_history_allocation_audit" {
        cargo bench --no-default-features --bench be_history_allocation_audit
    }
}

$summaryPath = Join-Path $reports "release_summary.md"
$table = @(
    "| step | status | exit_code | duration_s | technical_log |",
    "|---|---|---:|---:|---|"
)
foreach ($row in $results) {
    $log = $row.technical_log -replace '\\', '/'
    $table += "| $($row.step) | $($row.status) | $($row.exit_code) | $($row.duration_s) | $log |"
}
$failed = @($results | Where-Object { $_.exit_code -ne 0 }).Count
$overall = if ($failed -eq 0) { "completed" } else { "completed_with_failures" }
$summary = @(
    "# BE Release Matrix",
    "",
    "- status: $overall",
    "- recorded_at_utc: $((Get-Date).ToUniversalTime().ToString('o'))",
    "- report_root: $root",
    "- reports: $reports",
    "- technical: $technical",
    "- compact matrix: workloads=$Workloads; dimensions=$Dimensions; routes=$($env:BE_BENCH_ROUTES); continuation=$Continuation",
    "",
    "## Steps",
    "",
    ($table -join "`n"),
    "",
    "Compact Markdown reports are stored below `reports`; compiler and Cargo transcripts are stored only below `technical`."
)
Set-Content -Path $summaryPath -Value ($summary -join "`n") -Encoding UTF8
Write-Host "reports=$reports"
Write-Host "technical=$technical"
Write-Host "summary=$summaryPath"

if ($FailOnAny -and $failed -gt 0) { exit 1 }
exit 0

param(
    [string]$Steps = "oscillator,nonlinear-exact",
    [string]$Matrices = "dense,sparse,banded",
    [string]$Frontends = "exprlegacy,atomview",
    [string]$Executions = "sequential,parallel,auto",
    [string]$Solvers = "damped,frozen",
    [int]$Continuation = 4,
    [string]$NSteps = "32",
    [switch]$IncludeIgnoredStories,
    [switch]$IncludeAot,
    [switch]$IncludeCriterion,
    [switch]$FailOnAny
)

# BVP_Damp release campaign runner. It is deliberately non-fail-fast:
# numerical/story failures are recorded in technical logs and later steps run.
$ErrorActionPreference = "Continue"
$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$root = Join-Path (Get-Location) "test_reports\BVP_Damp_release_manual\$stamp"
$reports = Join-Path $root "reports"
$technical = Join-Path $root "technical"
New-Item -ItemType Directory -Force -Path $reports, $technical | Out-Null

$env:RST_TEST_REPORT_DIR = $reports
$env:RST_TEST_REPORT_PROFILE = "release"
$env:RST_TEST_REPORT_ARCHIVE = "always"
$env:RST_TEST_REPORT_STDOUT = "off"
$env:BVP_DAMP_BENCH_WORKLOADS = $Steps
$env:BVP_DAMP_BENCH_MATRICES = $Matrices
$env:BVP_DAMP_BENCH_FRONTENDS = $Frontends
$env:BVP_DAMP_BENCH_EXECUTIONS = $Executions
$env:BVP_DAMP_BENCH_SOLVERS = $Solvers
$env:BVP_DAMP_BENCH_CONTINUATION = [string]$Continuation
$env:BVP_DAMP_BENCH_N_STEPS = [string]$NSteps
$env:BVP_DAMP_BENCH_ROUTES = if ($IncludeAot) { "lambdify,aot" } else { "lambdify" }

$results = New-Object System.Collections.Generic.List[object]
function Invoke-BvpStep([string]$Name, [scriptblock]$Action) {
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

$storyModules = @(
    "NR_Damp_solver_damped::tests",
    "NR_Damp_solver_frozen::tests",
    "test_correctness",
    "test_classic_examples",
    "test_lambdify_acceptance",
    "test_lambdify_cross_product",
    "test_lambdify_callback_matrix",
    "test_lambdify_lifecycle",
    "test_parameter_rebind",
    "test_parity_corpus",
    "test_telemetry_story",
    "test_factorization_cache",
    "test_frozen_runtime_story",
    "test_backend_compare",
    "test_linear_solve_boundary",
    "test_validation",
    "test_aot_diagnostics",
    "test_aot_runtime_contract",
    "test_aot_race_stress"
)

foreach ($module in $storyModules) {
    Invoke-BvpStep "story_$module" {
        cargo test --release --lib --no-default-features ("numerical::BVP_Damp::$module") -- --nocapture --test-threads=1
    }
}

if ($IncludeIgnoredStories) {
    Invoke-BvpStep "story_ignored" {
        cargo test --release --lib --no-default-features numerical::BVP_Damp:: -- --ignored --nocapture --test-threads=1
    }
}

Invoke-BvpStep "compact_workload_matrix" {
    cargo bench --no-default-features --bench bvp_damp_workloads -- --noplot
}

if ($IncludeCriterion) {
    Invoke-BvpStep "criterion_bvp" {
        cargo bench --no-default-features --bench bvp_benches -- --noplot
    }
    Invoke-BvpStep "criterion_frozen_runtime" {
        cargo bench --no-default-features --bench bvp_frozen_runtime_benches -- --noplot
    }
    Invoke-BvpStep "criterion_numeric_assembly" {
        cargo bench --no-default-features --bench bvp_numeric_assembly -- --noplot
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
    "# BVP_Damp Release Matrix",
    "",
    "- status: $overall",
    "- recorded_at_utc: $((Get-Date).ToUniversalTime().ToString('o'))",
    "- report_root: $root",
    "- reports: $reports",
    "- technical: $technical",
    "- compact matrix: BVP_DAMP_BENCH_WORKLOADS=$Steps; matrices=$Matrices; frontends=$Frontends; executions=$Executions; solvers=$Solvers; n_steps=$NSteps; continuation=$Continuation",
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

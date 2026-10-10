param(
    [string]$Dimensions = "3,16,64",
    [string]$Routes = "lambdify-expr-legacy,lambdify-atom-native",
    [string]$Policies = "sequential,parallel",
    [string]$Methods = "newton,damped-newton,lm-minpack",
    [int]$Continuation = 4,
    [string]$FrontendContinuation = "",
    [ValidateSet("build-if-missing", "rebuild-always")]
    [string]$FrontendAotLifecycle = "build-if-missing",
    [switch]$IncludeAot,
    [switch]$IncludeIgnoredStories,
    [switch]$IncludeCriterion,
    [switch]$IncludeAllocationAudit,
    [switch]$FailOnAny
)

# Nonlinear-system release runner. Every step is independent: a failed story
# or optional benchmark is recorded in technical/ and does not suppress the
# remaining evidence-producing steps.
$ErrorActionPreference = "Continue"
$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$root = Join-Path (Get-Location) "test_reports\Nonlinear_systems_release_manual\$stamp"
$technical = Join-Path $root "technical"
New-Item -ItemType Directory -Force -Path $root, $technical | Out-Null

# The reporting helper appends the module/profile directories itself. Keep
# compact reports beside `technical/`, matching the Radau release layout.
$env:RST_TEST_REPORT_DIR = $root
$env:RST_TEST_REPORT_PROFILE = "release"
$env:RST_TEST_REPORT_ARCHIVE = "always"
$env:RST_TEST_REPORT_STDOUT = "off"
$env:NONLINEAR_BENCH_DIMENSIONS = $Dimensions
$env:NONLINEAR_BENCH_ROUTES = if ($IncludeAot) { "$Routes,aot-expr-legacy,aot-atom-native" } else { $Routes }
$env:NONLINEAR_BENCH_POLICIES = $Policies
$env:NONLINEAR_BENCH_METHODS = $Methods
$env:NONLINEAR_BENCH_CONTINUATION = [string]$Continuation
$env:NONLINEAR_BENCH_INCLUDE_AOT = if ($IncludeAot) { "1" } else { "0" }
if ($FrontendContinuation) {
    $env:NONLINEAR_FRONTEND_CONTINUATION = $FrontendContinuation
}
$env:NONLINEAR_FRONTEND_AOT_LIFECYCLE = $FrontendAotLifecycle

$results = New-Object System.Collections.Generic.List[object]
function Invoke-NonlinearStep([string]$Name, [scriptblock]$Action) {
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

Invoke-NonlinearStep "story_core" {
    cargo test --release --lib --no-default-features numerical::Nonlinear_systems -- --nocapture --test-threads=1
}

if ($IncludeIgnoredStories) {
    Invoke-NonlinearStep "story_ignored" {
        cargo test --release --lib --no-default-features numerical::Nonlinear_systems -- --ignored --nocapture --test-threads=1
    }
}

Invoke-NonlinearStep "compact_workload_matrix" {
    cargo bench --no-default-features --bench nonlinear_workloads -- --noplot
}

Invoke-NonlinearStep "compact_frontend_matrix" {
    $env:NONLINEAR_FRONTEND_INCLUDE_AOT = if ($IncludeAot) { "1" } else { "0" }
    cargo bench --no-default-features --bench nonlinear_frontend_matrix -- --noplot
}

Invoke-NonlinearStep "preparation_telemetry" {
    cargo bench --no-default-features --bench nonlinear_preparation_telemetry -- --noplot
}

if ($IncludeCriterion) {
    Invoke-NonlinearStep "criterion_solver_matrix" {
        cargo bench --no-default-features --bench nonlinear_systems_benches -- --noplot
    }
}

if ($IncludeAllocationAudit) {
    Invoke-NonlinearStep "allocation_audit" {
        cargo bench --no-default-features --bench nonlinear_systems_allocation_audit -- --noplot
    }
}

$summaryPath = Join-Path $root "release_summary.md"
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
    "# Nonlinear Systems Release Matrix",
    "",
    "- status: $overall",
    "- recorded_at_utc: $((Get-Date).ToUniversalTime().ToString('o'))",
    "- report_root: $root",
    "- reports: $root",
    "- technical: $technical",
    "- dashboard: dimensions=$Dimensions; routes=$($env:NONLINEAR_BENCH_ROUTES); policies=$Policies; methods=$Methods; continuation=$Continuation; frontend_continuation=$($env:NONLINEAR_FRONTEND_CONTINUATION); frontend_aot_lifecycle=$FrontendAotLifecycle",
    "",
    "## Steps",
    "",
    ($table -join "`n"),
    "",
    "Compact Tabled reports are stored below the timestamp root; Cargo/compiler transcripts are stored only below ``technical``."
)
Set-Content -Path $summaryPath -Value ($summary -join "`n") -Encoding UTF8
Write-Host "reports=$root"
Write-Host "technical=$technical"
Write-Host "summary=$summaryPath"

if ($FailOnAny -and $failed -gt 0) { exit 1 }
exit 0

param(
    [string]$Dimensions = "32,128",
    [string]$Workloads = "stiff-scalar,robertson,combustion-like,three-body,diffusion-chain",
    [string]$Matrices = "dense,sparse,banded",
    [string]$Routes = "lambdify-sequential,lambdify-auto,aot-whole",
    [string]$ContinuationCounts = "1,4",
    [switch]$IncludeIgnoredStories,
    [switch]$IncludeDetailedCriterion,
    [switch]$IncludeLongContinuation,
    [switch]$FailOnAny
)

$ErrorActionPreference = "Continue"
$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$root = Join-Path (Get-Location) "test_reports\LSODE2_release_manual\$stamp"
$reports = Join-Path $root "reports"
$technical = Join-Path $root "technical"
New-Item -ItemType Directory -Force -Path $reports, $technical | Out-Null

$env:RST_TEST_REPORT_DIR = $reports
$env:RST_TEST_REPORT_PROFILE = "release"
$env:RST_TEST_REPORT_ARCHIVE = "always"
$env:RST_TEST_REPORT_STDOUT = "off"
$env:LSODE2_BENCH_COMPACT_DIFFUSION_DIMENSIONS = $Dimensions
$env:LSODE2_BENCH_COMPACT_WORKLOADS = $Workloads
$env:LSODE2_BENCH_COMPACT_MATRICES = $Matrices
$env:LSODE2_BENCH_COMPACT_ROUTES = $Routes
$env:LSODE2_BENCH_COMPACT_CONTINUATION_COUNTS = $ContinuationCounts
$env:LSODE2_COMPACT_REPORT_NAME = "lsode2_workload_matrix"

$results = New-Object System.Collections.Generic.List[object]
function Invoke-Step([string]$Name, [scriptblock]$Action) {
    $log = Join-Path $technical "$Name.log"
    $started = Get-Date
    & $Action *> $log
    $exitCode = $LASTEXITCODE
    $results.Add([pscustomobject]@{
        step = $Name
        status = $(if ($exitCode -eq 0) { "passed" } else { "failed" })
        exit_code = $exitCode
        seconds = [math]::Round(((Get-Date) - $started).TotalSeconds, 1)
        technical_log = $log
    })
    Write-Host ("[{0}] {1} exit={2}" -f $Name, $(if ($exitCode -eq 0) { "passed" } else { "failed" }), $exitCode)
}

Invoke-Step "story_correctness" {
    cargo test --release --lib --no-default-features numerical::LSODE2::correctness_story_tests -- --nocapture --test-threads=1
}
Invoke-Step "story_lambdify" {
    cargo test --release --lib --no-default-features numerical::LSODE2::lambdify_stage_story_tests -- --nocapture --test-threads=1
}
Invoke-Step "story_lifecycle" {
    cargo test --release --lib --no-default-features numerical::LSODE2::lifecycle_story_tests -- --nocapture --test-threads=1
}
Invoke-Step "story_parallel_policy" {
    cargo test --release --lib --no-default-features numerical::LSODE2::evaluator_policy_story_tests -- --nocapture --test-threads=1
}
Invoke-Step "story_telemetry" {
    cargo test --release --lib --no-default-features numerical::LSODE2::telemetry_stage_story_tests -- --nocapture --test-threads=1
}
Invoke-Step "story_aot_core" {
    cargo test --release --lib --no-default-features numerical::LSODE2::aot_correctness_story_tests -- --nocapture --test-threads=1
}
Invoke-Step "story_aot_lifecycle" {
    cargo test --release --lib --no-default-features numerical::LSODE2::aot_lifecycle_story_tests -- --nocapture --test-threads=1
}
if ($IncludeIgnoredStories) {
    Invoke-Step "story_ignored" {
        cargo test --release --lib --no-default-features numerical::LSODE2:: -- --ignored --nocapture --test-threads=1
    }
}
Invoke-Step "compact_workload_matrix" {
    cargo bench --no-default-features --bench lsode2_workloads -- --noplot
}
if ($IncludeDetailedCriterion) {
    Invoke-Step "criterion_callbacks" {
        cargo bench --no-default-features --bench lsode2_workload_callbacks -- --noplot
    }
    Invoke-Step "criterion_aot" {
        cargo bench --no-default-features --bench lsode2_workload_aot -- --noplot
    }
}
if ($IncludeLongContinuation) {
    Invoke-Step "criterion_parameter_continuation" {
        cargo bench --no-default-features --bench lsode2_parameter_continuation -- --noplot
    }
}

$results | Format-Table -AutoSize | Out-File (Join-Path $reports "nightly_summary.txt") -Encoding utf8
@("# LSODE2 release matrix", "", "- stamp: $stamp", "- reports: $reports", "- technical: $technical", "", "## Steps", "", ($results | ConvertTo-Csv -NoTypeInformation | Out-String)) | Set-Content (Join-Path $reports "nightly_summary.md") -Encoding utf8
Write-Host "reports=$reports"
Write-Host "technical=$technical"
if ($FailOnAny -and (($results | Where-Object { $_.exit_code -ne 0 }).Count -gt 0)) { exit 1 }

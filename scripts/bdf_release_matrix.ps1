param(
    [string]$Dimensions = "16,64",
    [string]$Workloads = "stiff-scalar,robertson,combustion-like,diffusion-chain",
    [string]$ContinuationCounts = "1,4,16",
    [string]$Routes = "lambdify-sequential,lambdify-parallel,lambdify-auto,aot-whole,aot-parallel2",
    [int]$SampleSize = 10,
    [int]$MeasurementSeconds = 1,
    [switch]$IncludeIgnoredStories,
    [switch]$IncludeDetailedBenches,
    [switch]$PlanOnly,
    [switch]$FailOnAny
)

# BDF release runner. Compact reports are written by the benchmark target;
# compiler/Criterion output is kept in technical logs. Every step continues
# after a failure so an overnight campaign retains the rest of its evidence.
$ErrorActionPreference = "Continue"

$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmss'Z'")
$captureRoot = Join-Path (Get-Location) "test_reports\BDF_release_manual\$stamp"
$reportRoot = Join-Path $captureRoot "reports"
$logRoot = Join-Path $captureRoot "technical"
$summaryPath = Join-Path $captureRoot "nightly_summary.md"
New-Item -ItemType Directory -Force -Path $reportRoot, $logRoot | Out-Null

$env:RST_TEST_REPORT_DIR = $reportRoot
$env:RST_TEST_REPORT_PROFILE = "release"
$env:RST_TEST_REPORT_STDOUT = "off"
$env:RST_TEST_REPORT_ARCHIVE = "always"
$env:BDF_BENCH_DIFFUSION_DIMENSIONS = $Dimensions
$env:BDF_BENCH_COMPACT_WORKLOADS = $Workloads
$env:BDF_BENCH_AOT_WORKLOADS = $Workloads
$env:BDF_BENCH_AOT_CONTINUATION_WORKLOADS = "combustion-like,three-body,diffusion-chain"
$env:BDF_BENCH_AOT_CONTINUATION_DIFFUSION_DIMENSIONS = $Dimensions
$env:BDF_BENCH_CONTINUATION_COUNTS = $ContinuationCounts
$env:BDF_COMPACT_CONTINUATION_COUNTS = $ContinuationCounts
$env:BDF_BENCH_COMPACT_ROUTES = $Routes
$env:BDF_BENCH_SAMPLE_SIZE = [Math]::Max(10, $SampleSize).ToString()
$env:BDF_BENCH_MEASUREMENT_SECONDS = [Math]::Max(1, $MeasurementSeconds).ToString()

$steps = New-Object System.Collections.Generic.List[object]

function Add-BdfStep {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Kind,
        [Parameter(Mandatory = $true)][scriptblock]$Action
    )

    $logPath = Join-Path $logRoot "$Name.log"
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    $status = "passed"
    $exitCode = 0
    if ($PlanOnly) {
        $status = "planned"
        $logPath = "-"
    }
    else {
        try {
            & $Action 2>&1 | Tee-Object -FilePath $logPath |
                ForEach-Object {
                    $line = $_.ToString()
                    if ($line -match '^\[BDF compact benchmark\]' -or
                        $line -match '^test result:') {
                        Write-Host $line
                    }
                }
            if ($null -ne $LASTEXITCODE) {
                $exitCode = [int]$LASTEXITCODE
            }
            if ($exitCode -ne 0) {
                $status = "failed"
            }
        }
        catch {
            $status = "failed"
            $exitCode = 1
            Add-Content -Path $logPath -Value "`n[runner exception] $($_.Exception.Message)"
        }
    }
    $watch.Stop()
    $steps.Add([pscustomobject][ordered]@{
        step = $Name
        kind = $Kind
        status = $status
        exit_code = $exitCode
        duration_s = [string]::Format(
            [Globalization.CultureInfo]::InvariantCulture,
            "{0:F1}",
            $watch.Elapsed.TotalSeconds
        )
        technical_log = $logPath
    })
}

Add-BdfStep "stories_fast" "story" {
    cargo test --release --lib --no-default-features numerical::BDF:: -- --nocapture --test-threads=1
}

if ($IncludeIgnoredStories) {
    Add-BdfStep "stories_ignored" "story-ignored" {
        cargo test --release --lib --no-default-features numerical::BDF:: -- --ignored --nocapture --test-threads=1
    }
}

$env:BDF_BENCH_COMPACT_REPORT = "1"
$env:BDF_COMPACT_REPORT_NAME = "bdf_workload_matrix"
Add-BdfStep "bench_compact_workloads" "bench-compact" {
    cargo bench --no-default-features --bench bdf_workloads -- --noplot
}

if ($IncludeDetailedBenches) {
    $env:BDF_BENCH_COMPACT_REPORT = "0"
    Add-BdfStep "bench_symbolic_frontends" "bench-detailed" {
        cargo bench --no-default-features --bench bdf_symbolic_frontends -- --noplot
    }
    Add-BdfStep "bench_aot_frontends" "bench-detailed-aot" {
        cargo bench --no-default-features --bench bdf_aot_frontends -- --noplot
    }
    Add-BdfStep "bench_backend_matrix" "bench-detailed-matrix" {
        cargo bench --no-default-features --bench bdf_backend_matrix -- --noplot
    }
    Add-BdfStep "bench_aot_continuation" "bench-detailed-continuation" {
        cargo bench --no-default-features --bench bdf_aot_continuation -- --noplot
    }
}

$passed = @($steps | Where-Object { $_.status -eq "passed" }).Count
$failed = @($steps | Where-Object { $_.status -eq "failed" }).Count
$planned = @($steps | Where-Object { $_.status -eq "planned" }).Count
$overall = if ($failed -eq 0 -and $planned -eq 0) { "completed" }
           elseif ($failed -gt 0) { "completed_with_failures" }
           else { "planned" }

$rows = @(
    "| step | kind | status | exit_code | duration_s | technical_log |",
    "|---|---|---|---:|---:|---|"
)
foreach ($step in $steps) {
    $log = $step.technical_log.Replace((Get-Location).Path + "\", "").Replace("\", "/")
    $rows += "| $($step.step) | $($step.kind) | $($step.status) | $($step.exit_code) | $($step.duration_s) | $log |"
}

$reports = @(Get-ChildItem -Path $reportRoot -Recurse -File -Filter '*.md' -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -ne $summaryPath } |
    ForEach-Object { $_.FullName.Replace((Get-Location).Path + '\', '').Replace('\', '/') })

$summary = @(
    "# BDF Release Matrix Summary",
    "",
    "- status: $overall",
    "- recorded_at_utc: $((Get-Date).ToUniversalTime().ToString('o'))",
    "- profile: release",
    "- dimensions: $Dimensions",
    "- workloads: $Workloads",
    "- continuation_counts: $ContinuationCounts",
    "- routes: $Routes",
    "- passed_steps: $passed",
    "- failed_steps: $failed",
    "- planned_steps: $planned",
    "",
    "## Steps",
    "",
    ($rows -join "`n"),
    "",
    "## Compact Reports",
    "",
    "Tables are stored under reports/; compiler and Criterion technical logs are under technical/.",
    $(if ($reports.Count -gt 0) { $reports -join "`n" } else { "- none" })
)
Set-Content -LiteralPath $summaryPath -Value ($summary -join "`n") -Encoding UTF8
Write-Output "[BDF release] status=$overall passed=$passed failed=$failed reports=$reportRoot"
Write-Output "[BDF release] summary=$summaryPath"

if ($FailOnAny -and $failed -gt 0) {
    exit 1
}
exit 0

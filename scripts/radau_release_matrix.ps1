param(
    [string]$Dimensions = "128,512,1024,2048",
    [string]$AotPolicies = "sequential,auto",
    [string]$LambdifyPolicies = "all",
    [string]$ContinuationCounts = "1,4,16,64",
    [switch]$SkipAot,
    [switch]$SkipIgnoredStories,
    [switch]$FailOnAny,
    [switch]$PlanOnly
)

# This is an overnight campaign runner, not a fail-fast CI command. Every
# step gets an isolated technical log and a compact status row. Use -FailOnAny
# when the caller wants the process exit code to enforce a gate.
$ErrorActionPreference = "Continue"

$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmss'Z'")
$reportRoot = Join-Path (Get-Location) "test_reports\Radau_release_manual\$stamp"
$logRoot = Join-Path $reportRoot "technical"
$summaryPath = Join-Path $reportRoot "nightly_summary.md"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

$env:RST_TEST_REPORT_DIR = $reportRoot
$env:RST_TEST_REPORT_PROFILE = "release"
$env:RST_TEST_REPORT_STDOUT = "off"
$env:RST_TEST_REPORT_ARCHIVE = "always"
$env:RADAU_BENCH_CONTINUATION_COUNTS = $ContinuationCounts

$results = New-Object System.Collections.Generic.List[object]

function Add-RadauStep {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Kind,
        [Parameter(Mandatory = $true)][scriptblock]$Action
    )

    $technicalLog = Join-Path $logRoot "$Name.log"
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    $exitCode = 0
    $status = "passed"
    $errorText = ""

    if (-not $PlanOnly) {
        try {
            # Keep the complete transcript in the technical log, but surface
            # only compact benchmark heartbeats and process summaries.
            & $Action 2>&1 |
                Tee-Object -FilePath $technicalLog |
                ForEach-Object {
                    $line = $_.ToString()
                    if ($line -match '^\[Radau compact progress\]' -or
                        $line -match '^\[Radau compact benchmark\]' -or
                        $line -match '^\[Radau process\]') {
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
            $errorText = $_.Exception.Message
            Add-Content -Path $technicalLog -Value "`n[runner exception] $errorText"
        }
    }
    else {
        $status = "planned"
        $technicalLog = "-"
    }

    $watch.Stop()
    $duration = [string]::Format([Globalization.CultureInfo]::InvariantCulture, "{0:F1}", $watch.Elapsed.TotalSeconds)
    $results.Add([pscustomobject][ordered]@{
        step = $Name
        kind = $Kind
        status = $status
        exit_code = $exitCode
        duration_s = $duration
        technical_log = $technicalLog
        error = $errorText
    })
}

Add-RadauStep "stories_fast" "story" {
    cargo test --release --lib --no-default-features numerical::Radau::tests:: -- --nocapture --test-threads=1
}

if (-not $SkipIgnoredStories) {
    Add-RadauStep "stories_ignored" "story-ignored" {
        cargo test --release --lib --no-default-features numerical::Radau::tests:: -- --ignored --nocapture --test-threads=1
    }
}

# Each worker count is a separate process. Compact benchmark reports contain
# measured rows; the technical log contains only the optional raw process
# transcript and never pollutes the terminal or the summary table.
foreach ($workers in @("1", "4", "12")) {
    $env:RAYON_NUM_THREADS = $workers
    $env:RADAU_BENCH_COMPACT_REPORT = "1"
    $env:RADAU_COMPACT_AOT = "0"
    $env:RADAU_COMPACT_DIMENSIONS = $Dimensions
    $env:RADAU_COMPACT_POLICIES = $LambdifyPolicies
    $env:RADAU_COMPACT_REPORT_NAME = "lambdify_matrix_w$workers"
    Add-RadauStep "bench_lambdify_w$workers" "bench" {
        cargo bench --no-default-features --bench radau_workloads -- --noplot
    }
}

if (-not $SkipAot) {
    $env:RAYON_NUM_THREADS = "12"
    $env:RADAU_BENCH_COMPACT_REPORT = "1"
    $env:RADAU_COMPACT_AOT = "1"
    $env:RADAU_COMPACT_DIMENSIONS = $Dimensions
    $env:RADAU_COMPACT_POLICIES = $AotPolicies
    $env:RADAU_COMPACT_REPORT_NAME = "aot_lambdify_matrix_w12"
    Add-RadauStep "bench_aot_lambdify_w12" "bench-aot" {
        cargo bench --no-default-features --bench radau_workloads -- --noplot
    }
}

Add-RadauStep "process_isolated_release" "story-process" {
    cargo test --release --lib --no-default-features numerical::Radau::tests::process_isolated::radau_aot_process_isolated_producer_consumer_handoff -- --ignored --nocapture --test-threads=1
}

$passed = @($results | Where-Object { $_.status -eq "passed" }).Count
$failed = @($results | Where-Object { $_.status -eq "failed" }).Count
$planned = @($results | Where-Object { $_.status -eq "planned" }).Count
$overall = if ($failed -eq 0 -and $planned -eq 0) { "completed" } elseif ($failed -gt 0) { "completed_with_failures" } else { "planned" }

$tableRows = @(
    "| step | kind | status | exit_code | duration_s | technical_log | error |",
    "|---|---|---|---:|---:|---|---|"
)
foreach ($row in $results) {
    $log = $row.technical_log -replace '\\', '/'
    $errorCell = ($row.error -replace '\|', '/')
    $tableRows += "| $($row.step) | $($row.kind) | $($row.status) | $($row.exit_code) | $($row.duration_s) | $log | $errorCell |"
}

$compactReports = @()
if (Test-Path $reportRoot) {
    $compactReports = @(Get-ChildItem -Path $reportRoot -Recurse -File -Filter '*.md' |
        Where-Object { $_.FullName -ne $summaryPath } |
        ForEach-Object { $_.FullName.Replace((Get-Location).Path + '\', '').Replace('\', '/') })
}

$summary = @(
    "# Radau Overnight Release Summary",
    "",
    "- status: $overall",
    "- profile: release",
    "- recorded_at_utc: $((Get-Date).ToUniversalTime().ToString('o'))",
    "- report_root: $reportRoot",
    "- passed_steps: $passed",
    "- failed_steps: $failed",
    "- planned_steps: $planned",
    "",
    "## Steps",
    "",
    ($tableRows -join "`n"),
    "",
    "## Compact Reports",
    "",
    "These files contain the measured tables. Cargo/compiler output is kept only in the technical logs above.",
    ""
)
if ($compactReports.Count -gt 0) {
    foreach ($path in $compactReports) {
        $summary += "- $path"
    }
}
else {
    $summary += "- none"
}
Set-Content -Path $summaryPath -Value ($summary -join "`n") -Encoding UTF8

Write-Output "[Radau overnight] status=$overall passed=$passed failed=$failed reports=$reportRoot"
Write-Output "[Radau overnight] summary=$summaryPath"

if ($FailOnAny -and $failed -gt 0) {
    exit 1
}
exit 0

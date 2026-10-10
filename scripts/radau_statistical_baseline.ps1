param(
    [string]$Dimensions = "512",
    [string]$ContinuationCounts = "4,16",
    [int]$SampleSize = 20,
    [int]$MeasurementSeconds = 3,
    [switch]$IncludePolicy,
    [switch]$IncludeAot,
    [switch]$PlanOnly,
    [string]$ConvertExistingRoot = "",
    [switch]$FailOnAny
)

# This is a bounded repeated-sampling campaign. It is deliberately separate
# from radau_release_matrix.ps1: compact release rows are one-shot wall-clock
# evidence, while this script produces Criterion's statistical estimates.
$ErrorActionPreference = "Continue"

$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmss'Z'")
$reportRoot = Join-Path (Get-Location) "test_reports\Radau_statistical_baseline\$stamp"
$logRoot = Join-Path $reportRoot "technical"
$tableRoot = Join-Path $reportRoot "tables"
$summaryPath = Join-Path $reportRoot "statistical_summary.md"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
New-Item -ItemType Directory -Force -Path $tableRoot | Out-Null

$env:RADAU_BENCH_COMPACT_REPORT = "0"
$env:RADAU_BENCH_DIFFUSION_DIMENSIONS = $Dimensions
$env:RADAU_BENCH_AOT_DIFFUSION_DIMENSIONS = $Dimensions
$env:RADAU_BENCH_POLICY_DIFFUSION_DIMENSIONS = $Dimensions
$env:RADAU_BENCH_CONTINUATION_COUNTS = $ContinuationCounts
$env:RADAU_BENCH_SAMPLE_SIZE = [Math]::Max(10, $SampleSize).ToString()
$env:RADAU_BENCH_MEASUREMENT_SECONDS = [Math]::Max(1, $MeasurementSeconds).ToString()

$steps = New-Object System.Collections.Generic.List[object]

function Convert-CriterionLogToTable {
    param(
        [Parameter(Mandatory = $true)][string]$LogPath,
        [Parameter(Mandatory = $true)][string]$TablePath
    )

    $rows = New-Object System.Collections.Generic.List[object]
    $current = $null
    foreach ($line in (Get-Content -LiteralPath $LogPath -Encoding Unicode)) {
        if ($line -match '^\s*(radau_[^\s]+)\s*$') {
            if ($null -ne $current) {
                $rows.Add([pscustomobject]$current)
            }
            $current = [ordered]@{
                benchmark = $Matches[1]
                low = "-"
                median = "-"
                high = "-"
                unit = "-"
                samples = "-"
                outliers = "-"
                verdict = "-"
            }
            continue
        }
        if ($null -eq $current) {
            continue
        }
        if ($line -match 'time:\s+\[\s*([0-9.eE+-]+)\s+\S+\s+([0-9.eE+-]+)\s+\S+\s+([0-9.eE+-]+)\s+\S+\s*\]') {
            $current.low = $Matches[1]
            $current.median = $Matches[2]
            $current.high = $Matches[3]
            $current.unit = if ($line -match '┬╡s') { "us" }
                            elseif ($line -match '\bms\b') { "ms" }
                            elseif ($line -match '\bns\b') { "ns" }
                            elseif ($line -match '\bs\b') { "s" }
                            else { "unknown" }
            # Normalize Criterion's UTF-8 microsecond marker after the
            # transcript is written as UTF-16. Keep the source ASCII-only.
            if ($line -match '\u252c\u2561s') {
                $current.unit = "us"
            }
            continue
        }
        if ($line -match 'Found\s+(\d+)\s+outliers\s+among\s+(\d+)\s+measurements') {
            $current.outliers = $Matches[1]
            $current.samples = $Matches[2]
            continue
        }
        if ($line -match '^\s*Performance has (improved|regressed)\.') {
            $current.verdict = $Matches[1]
            continue
        }
        if ($line -match '^\s*No change in performance detected\.') {
            $current.verdict = "no-change"
        }
    }
    if ($null -ne $current) {
        $rows.Add([pscustomobject]$current)
    }

    $tableRows = @(
        "| benchmark | low | median | high | unit | samples | outliers | verdict |",
        "|---|---:|---:|---:|---|---:|---:|---|"
    )
    foreach ($row in $rows) {
        $tableRows += "| $($row.benchmark) | $($row.low) | $($row.median) | $($row.high) | $($row.unit) | $($row.samples) | $($row.outliers) | $($row.verdict) |"
    }
    if ($rows.Count -eq 0) {
        $tableRows += "| no Criterion measurements found | - | - | - | - | - | - | error |"
    }

    $body = @(
        "# Criterion Statistical Results",
        "",
        "- source_log: $LogPath",
        "- rows: $($rows.Count)",
        "- low/median/high: Criterion's reported interval values",
        "- note: statistical evidence only; not a portable wall-clock gate",
        "",
        ($tableRows -join "`n"),
        ""
    )
    Set-Content -LiteralPath $TablePath -Value ($body -join "`n") -Encoding UTF8
}

if ($ConvertExistingRoot -ne "") {
    $existingRoot = (Resolve-Path -LiteralPath $ConvertExistingRoot).Path
    $existingTechnical = Join-Path $existingRoot "technical"
    $existingTables = Join-Path $existingRoot "tables"
    New-Item -ItemType Directory -Force -Path $existingTables | Out-Null
    $indexRows = @(
        "| source log | table report |",
        "|---|---|"
    )
    foreach ($log in (Get-ChildItem -LiteralPath $existingTechnical -Filter '*.log' -File | Sort-Object Name)) {
        $table = Join-Path $existingTables ($log.BaseName + ".md")
        Convert-CriterionLogToTable -LogPath $log.FullName -TablePath $table
        $logName = "technical/$($log.Name)"
        $tableName = "tables/$($log.BaseName).md"
        $indexRows += "| $logName | $tableName |"
    }
    $index = @(
        "# Existing Criterion Tables",
        "",
        "Generated from the archived UTF-16 technical logs without rerunning benchmarks.",
        "",
        ($indexRows -join "`n"),
        ""
    )
    Set-Content -LiteralPath (Join-Path $existingTables 'index.md') -Value ($index -join "`n") -Encoding UTF8
    Write-Output "[Radau statistical baseline] converted=$existingRoot tables=$existingTables"
    exit 0
}

function Add-StatisticalStep {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
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
            & $Action 2>&1 | Tee-Object -FilePath $logPath
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
    $tablePath = "-"
    if (-not $PlanOnly -and (Test-Path -LiteralPath $logPath)) {
        $tablePath = Join-Path $tableRoot "$Name.md"
        Convert-CriterionLogToTable -LogPath $logPath -TablePath $tablePath
    }
    $steps.Add([pscustomobject][ordered]@{
        step = $Name
        status = $status
        exit_code = $exitCode
        duration_s = [string]::Format(
            [Globalization.CultureInfo]::InvariantCulture,
            "{0:F1}",
            $watch.Elapsed.TotalSeconds
        )
        technical_log = $logPath
        table_report = $tablePath
    })
}

$env:RADAU_BENCH_POLICY_MATRIX = "0"
$env:RADAU_BENCH_AOT = "0"
Add-StatisticalStep "lambdify" {
    cargo bench --no-default-features --bench radau_workloads -- --noplot
}

if ($IncludePolicy) {
    $env:RADAU_BENCH_POLICY_MATRIX = "1"
    Add-StatisticalStep "lambdify_policy" {
        cargo bench --no-default-features --bench radau_workloads -- --noplot
    }
}

if ($IncludeAot) {
    $env:RADAU_BENCH_POLICY_MATRIX = if ($IncludePolicy) { "1" } else { "0" }
    $env:RADAU_BENCH_AOT = "1"
    Add-StatisticalStep "aot" {
        cargo bench --no-default-features --bench radau_workloads -- --noplot
    }
}

$passed = @($steps | Where-Object { $_.status -eq "passed" }).Count
$failed = @($steps | Where-Object { $_.status -eq "failed" }).Count
$rows = @(
    "| step | status | exit_code | duration_s | table_report | technical_log |",
    "|---|---|---:|---:|---|---|"
)
foreach ($step in $steps) {
    $log = $step.technical_log.Replace((Get-Location).Path + "\", "").Replace("\", "/")
    $table = $step.table_report.Replace((Get-Location).Path + "\", "").Replace("\", "/")
    $rows += "| $($step.step) | $($step.status) | $($step.exit_code) | $($step.duration_s) | $table | $log |"
}

$summary = @(
    "# Radau Statistical Baseline",
    "",
    "- recorded_at_utc: $((Get-Date).ToUniversalTime().ToString('o'))",
    "- dimensions: $Dimensions",
    "- continuation_counts: $ContinuationCounts",
    "- sample_size: $($env:RADAU_BENCH_SAMPLE_SIZE)",
    "- measurement_seconds: $($env:RADAU_BENCH_MEASUREMENT_SECONDS)",
    "- compact_report: disabled; Criterion repeated sampling is the measurement source",
    "- interpretation: compare medians and confidence intervals on the same host/profile; do not gate tiny absolute timings by percentage alone",
    "- passed_steps: $passed",
    "- failed_steps: $failed",
    "",
    "## Steps",
    "",
    ($rows -join "`n"),
    "",
    "Compact statistical tables are in tables/. Raw Criterion output is retained in technical/ and target/criterion."
)
Set-Content -Path $summaryPath -Value ($summary -join "`n") -Encoding UTF8
Write-Output "[Radau statistical baseline] passed=$passed failed=$failed report=$summaryPath"

if ($FailOnAny -and $failed -gt 0) {
    exit 1
}
exit 0

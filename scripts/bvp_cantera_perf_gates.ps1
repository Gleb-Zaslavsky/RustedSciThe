param(
    [Parameter(Mandatory)][ValidateSet("Record", "Compare", "RecordCompare")][string]$Action,
    [Parameter(Mandatory)][string]$Machine,
    [ValidateSet("bvp-cantera-matrix", "bvp-cantera-parameter-continuation")]
    [string]$Group = "bvp-cantera-matrix",
    [ValidateRange(0, 1000)][int]$Warmups = 10,
    [ValidateRange(3, 10001)][int]$Runs = 101,
    [ValidateRange(0.0, 1.0)][double]$MaxRelativeMad = 0.05,
    [ValidateRange(0.0, 1.0)][double]$Tolerance = 0.10,
    [switch]$Force,
    [switch]$AllowDirtySource,
    [string]$Toolchain,
    [switch]$PlanOnly
)

$ErrorActionPreference = "Continue"
if ($Action -eq "Compare" -and ($Force -or $AllowDirtySource)) {
    throw "-Force and -AllowDirtySource apply only to actions containing Record."
}
$steps = if ($Action -eq "RecordCompare") { @("Record", "Compare") } else { @($Action) }
if ($PlanOnly) {
    Write-Host "BVP_Cantera perf-gates plan: machine=$Machine group=$Group"
    foreach ($step in $steps) {
        Write-Host " - $step group=$Group warmups=$Warmups runs=$Runs"
    }
    exit 0
}

$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$reportRoot = Join-Path (Get-Location) "test_reports\BVP_Cantera_perf_gates\$stamp"
$technical = Join-Path $reportRoot "technical"
New-Item -ItemType Directory -Force -Path $technical | Out-Null
$runner = Join-Path $PSScriptRoot "perf_gates_run.ps1"
$results = New-Object System.Collections.Generic.List[object]
$recordFailed = $false

foreach ($step in $steps) {
    if ($step -eq "Compare" -and $recordFailed) {
        $results.Add([pscustomobject][ordered]@{
            group = $Group; action = $step; status = "blocked"; exit_code = "";
            duration_s = 0; technical_log = "record action did not complete successfully"
        })
        continue
    }

    $log = Join-Path $technical ("{0}_{1}.log" -f $Group, $step.ToLowerInvariant())
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    $arguments = @{
        Action = $step
        Machine = $Machine
        Group = $Group
        Warmups = $Warmups
        Runs = $Runs
        MaxRelativeMad = $MaxRelativeMad
        Tolerance = $Tolerance
    }
    if ($step -eq "Record") {
        if ($Force) { $arguments.Force = $true }
        if ($AllowDirtySource) { $arguments.AllowDirtySource = $true }
    }
    if (-not [string]::IsNullOrWhiteSpace($Toolchain)) { $arguments.Toolchain = $Toolchain }

    Write-Host "`n=== BVP_Cantera perf-gates: $Group / $step ==="
    $exitCode = 0
    try {
        & $runner @arguments 2>&1 | Tee-Object -FilePath $log
        if ($null -ne $LASTEXITCODE) { $exitCode = [int]$LASTEXITCODE }
    }
    catch {
        $exitCode = 1
        Add-Content -Path $log -Value ("`n[runner exception] " + $_.Exception.Message)
        Write-Error $_
    }
    $watch.Stop()
    $status = if ($exitCode -eq 0) { "passed" } else { "failed" }
    if ($step -eq "Record" -and $exitCode -ne 0) { $recordFailed = $true }
    $results.Add([pscustomobject][ordered]@{
        group = $Group
        action = $step
        status = $status
        exit_code = $exitCode
        duration_s = [math]::Round($watch.Elapsed.TotalSeconds, 1)
        technical_log = $log
    })
}

$summaryPath = Join-Path $reportRoot "summary.md"
$table = @(
    "| group | action | status | exit_code | duration_s | technical_log |",
    "|---|---|---|---:|---:|---|"
)
foreach ($row in $results) {
    $logPath = ([string]$row.technical_log) -replace '\\', '/'
    $table += "| $($row.group) | $($row.action) | $($row.status) | $($row.exit_code) | $($row.duration_s) | $logPath |"
}
$failed = @($results | Where-Object { $_.status -in @("failed", "blocked") }).Count
$status = if ($failed -eq 0) { "completed" } else { "completed_with_failures" }
$summary = @(
    "# BVP_Cantera Performance-Gate Matrix",
    "",
    "- status: $status",
    "- recorded_at_utc: $((Get-Date).ToUniversalTime().ToString('o'))",
    "- machine: $Machine",
    "- group: $Group",
    "- report_root: $reportRoot",
    "- performance_reports: performance/runs",
    "",
    "## Actions",
    "",
    ($table -join "`n"),
    "",
    "Scenario-level TOML/JSON reports remain in performance/runs; this matrix archives the command transcript and status table."
)
Set-Content -Path $summaryPath -Value ($summary -join "`n") -Encoding UTF8
$results | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $reportRoot "results.csv")
Write-Host "`nBVP_Cantera perf-gates summary"
$results | Format-Table -AutoSize | Out-String | Write-Host
Write-Host "summary=$summaryPath"
if ($failed -gt 0) { exit 1 }
exit 0

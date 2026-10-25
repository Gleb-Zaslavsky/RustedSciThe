param(
    [Parameter(Mandatory)][ValidateSet("Record", "Compare", "RecordCompare")][string]$Action,
    [Parameter(Mandatory)][string]$Machine,
    [ValidateSet("bdf", "radau", "lsode2", "be")]
    [string[]]$Solvers = @("bdf", "radau", "lsode2", "be"),
    [ValidateRange(0, 1000)][int]$Warmups = 20,
    [ValidateRange(3, 10001)][int]$Runs = 301,
    [ValidateRange(1, 1000)][int]$RadauWarmups = 5,
    [ValidateRange(3, 10001)][int]$RadauRuns = 31,
    [ValidateRange(0.0, 1.0)][double]$MaxRelativeMad = 0.05,
    [ValidateRange(0.0, 1.0)][double]$Tolerance = 0.10,
    [switch]$Force,
    [switch]$AllowDirtySource,
    [string]$Toolchain,
    [string]$ReportRoot,
    [switch]$PlanOnly,
    # Dependency injection for orchestration tests; normally use the maintained runner.
    [string]$RunnerScript
)

$ErrorActionPreference = "Stop"
if ($Action -eq "Compare" -and ($Force -or $AllowDirtySource)) {
    throw "-Force and -AllowDirtySource apply only to actions containing Record."
}
if ($Solvers.Count -eq 0) { throw "Select at least one IVP solver." }
$Solvers = @($Solvers | Select-Object -Unique)
$actions = if ($Action -eq "RecordCompare") { @("Record", "Compare") } else { @($Action) }
if ($PlanOnly) {
    Write-Host "IVP perf-gates plan: machine=$Machine"
    foreach ($solver in $Solvers) {
        $solverWarmups = if ($solver -eq 'radau') { $RadauWarmups } else { $Warmups }
        $solverRuns = if ($solver -eq 'radau') { $RadauRuns } else { $Runs }
        foreach ($step in $actions) { Write-Host " - $solver / $step (warmups=$solverWarmups runs=$solverRuns)" }
    }
    exit 0
}

. (Join-Path $PSScriptRoot "ivp_perf_gates_reporting.ps1")
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($RunnerScript)) {
    $RunnerScript = Join-Path $PSScriptRoot "perf_gates_first_tier.ps1"
}
$RunnerScript = (Resolve-Path -LiteralPath $RunnerScript).Path
if ([string]::IsNullOrWhiteSpace($ReportRoot)) {
    $ReportRoot = Join-Path $repoRoot "test_reports/IVP_perf_gates"
}
$stamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssfffZ") + "-$PID"
$root = Join-Path $ReportRoot $stamp
New-Item -ItemType Directory -Path (Join-Path $root "technical") -Force | Out-Null
$root = (Resolve-Path -LiteralPath $root).Path
$powerShellHost = (Get-Process -Id $PID).Path
$steps = [System.Collections.Generic.List[object]]::new()
$scenarios = [System.Collections.Generic.List[object]]::new()
foreach ($solver in $Solvers) {
    foreach ($step in $actions) {
        $steps.Add([pscustomobject][ordered]@{
            solver = $solver; action = $step.ToLowerInvariant(); status = "pending"
            warmups = 0; runs = 0; exit_code = $null; duration_s = 0; detail = ""; log = ""
        })
    }
}
$settings = [ordered]@{
    machine = $Machine; action = $Action; warmups = $Warmups; runs = $Runs
    radau_warmups = $RadauWarmups; radau_runs = $RadauRuns
    max_relative_mad = $MaxRelativeMad; tolerance = $Tolerance
    force = [bool]$Force; allow_dirty_source = [bool]$AllowDirtySource; toolchain = $Toolchain
}
Write-IvpGateSummary $root $settings $steps $scenarios "running"
Write-Host "IVP matrix reports: $root"

foreach ($step in $steps) {
    $step.warmups = if ($step.solver -eq 'radau') { $RadauWarmups } else { $Warmups }
    $step.runs = if ($step.solver -eq 'radau') { $RadauRuns } else { $Runs }
    if ($Action -eq "RecordCompare" -and $step.action -eq "compare") {
        $record = $steps | Where-Object { $_.solver -eq $step.solver -and $_.action -eq "record" }
        if ($record.status -ne "passed") {
            $step.status = "blocked"
            $step.detail = "Record did not complete successfully; comparison would mix old and new baselines."
            Write-IvpGateSummary $root $settings $steps $scenarios "running"
            continue
        }
    }
    $step.status = "running"
    $step.log = "technical/$($step.solver)_$($step.action).log"
    $logPath = Join-Path $root $step.log
    Write-IvpGateSummary $root $settings $steps $scenarios "running"
    Write-Host "`n=== IVP perf-gates: $($step.solver) / $($step.action) ==="
    $arguments = @(
        "-Solver", $step.solver, "-Action", $step.action, "-Machine", $Machine,
        "-Warmups", [string]$step.warmups, "-Runs", [string]$step.runs,
        "-MaxRelativeMad", $MaxRelativeMad.ToString([Globalization.CultureInfo]::InvariantCulture),
        "-Tolerance", $Tolerance.ToString([Globalization.CultureInfo]::InvariantCulture)
    )
    if ($step.action -eq "record") {
        if ($Force) { $arguments += "-Force" }
        if ($AllowDirtySource) { $arguments += "-AllowDirtySource" }
    }
    if ($Toolchain) { $arguments += @("-Toolchain", $Toolchain) }
    $watch = [Diagnostics.Stopwatch]::StartNew()
    try {
        # Child isolation preserves exit codes and lets the next solver run after failure.
        # PowerShell 5 wraps native stderr in ErrorRecords; those are log data here.
        $ErrorActionPreference = "Continue"
        & $powerShellHost -NoProfile -ExecutionPolicy Bypass -File $RunnerScript @arguments 2>&1 |
            Tee-Object -FilePath $logPath | ForEach-Object { Write-Host $_ }
        $step.exit_code = $LASTEXITCODE
        $ErrorActionPreference = "Stop"
        $reportPaths = @(Get-Content -LiteralPath $logPath | ForEach-Object {
            if ($_ -match '^run report:\s*(.+?)\s*$') { $Matches[1] }
        } | Select-Object -Unique)
        if ($reportPaths.Count -eq 0) { throw (Get-IvpGateLaunchFailure $logPath) }
        foreach ($path in $reportPaths) {
            $rows = @(Read-IvpGateReport $path $step.solver $step.action $root $step.log)
            foreach ($row in $rows) { $scenarios.Add($row) }
        }
        $rows = @($scenarios | Where-Object { $_.solver -eq $step.solver -and $_.action -eq $step.action })
        $step.status = if (@($rows | Where-Object status -eq "error").Count) { "error" }
            elseif (@($rows | Where-Object status -eq "regression").Count) { "regression" }
            elseif (@($rows | Where-Object status -eq "unstable").Count) { "unstable" }
            else { "passed" }
        $issues = @($rows | Where-Object { $_.status -ne 'passed' } | ForEach-Object {
            if ($_.detail) { "$($_.scenario) [$($_.status)]: $($_.detail)" }
            else { "$($_.scenario) [$($_.status)]" }
        })
        $step.detail = $issues -join '; '
        if (($step.exit_code -eq 0) -ne ($step.status -eq "passed")) {
            throw "Process exit and native report disagree (exit $($step.exit_code), report $($step.status))."
        }
    }
    catch {
        $step.status = "error"
        $step.detail = $_.Exception.Message
        Add-Content -LiteralPath $logPath -Value "[matrix error] $($step.detail)"
    }
    finally {
        $ErrorActionPreference = "Stop"
        $watch.Stop()
        $step.duration_s = [math]::Round($watch.Elapsed.TotalSeconds, 1)
    }
    Write-IvpGateSummary $root $settings $steps $scenarios "running"
}
$failed = @($steps | Where-Object { $_.status -ne "passed" }).Count
$outcome = if ($failed) { "completed_with_failures" } else { "completed" }
Write-IvpGateSummary $root $settings $steps $scenarios $outcome
Write-Host "`nIVP performance-gate summary"
$steps | Format-Table solver, action, status, exit_code, duration_s -AutoSize
foreach ($step in $steps | Where-Object { $_.status -eq 'error' -and $_.detail }) {
    Write-Host "$($step.solver) / $($step.action): $($step.detail)"
}
Write-Host "summary=$(Join-Path $root 'summary.md')"
if ($failed) { exit 1 }
exit 0

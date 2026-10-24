param(
    [Parameter(Mandatory)][string]$Machine,
    [string]$PythonPath,
    [ValidateRange(1, 64)][int]$CargoBuildJobs = 8,
    [ValidateRange(0, 1000)][int]$PerfWarmups = 20,
    [ValidateRange(3, 10001)][int]$PerfRuns = 301,
    [string]$BdfDimensions = "64,256,1024",
    [string]$RadauDimensions = "128,512,1024,2048",
    [string]$Lsode2Dimensions = "128,512,1024",
    [string]$BeDimensions = "32,128,512",
    [switch]$SkipReleaseTests,
    [switch]$SkipPerfCompare,
    [switch]$SkipReleaseMatrices,
    [switch]$RefreshBaselines,
    [string]$Toolchain,
    [switch]$PlanOnly
)

# Overnight IVP symbolic stabilization campaign. Existing solver scripts own
# their compact Tabled reports; this orchestrator owns only the campaign-level
# status and technical transcripts. STORY_TESTS remain manually curated.
$ErrorActionPreference = "Continue"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$root = Join-Path $repoRoot "test_reports\IVP_symbolic_nightly\$stamp"
$technical = Join-Path $root "technical"
$summaryPath = Join-Path $root "nightly_summary.md"
New-Item -ItemType Directory -Force -Path $technical | Out-Null

$releaseTests = Join-Path $PSScriptRoot "ivp_symbolic_release_tests.ps1"
$perfGates = Join-Path $PSScriptRoot "ivp_perf_gates.ps1"
$bdfMatrix = Join-Path $PSScriptRoot "bdf_release_matrix.ps1"
$radauMatrix = Join-Path $PSScriptRoot "radau_release_matrix.ps1"
$lsode2Matrix = Join-Path $PSScriptRoot "lsode2_release_matrix.ps1"
$beMatrix = Join-Path $PSScriptRoot "be_release_matrix.ps1"
$powerShellHost = (Get-Process -Id $PID).Path

function Invoke-ChildPowerShell {
    param(
        [Parameter(Mandatory)][string]$ScriptPath,
        [string[]]$ScriptArguments = @()
    )

    # Several established release runners deliberately call `exit` to expose
    # their gate result. A child process isolates that lifecycle so one solver
    # cannot terminate the campaign before the remaining solvers run.
    & $powerShellHost -NoProfile -ExecutionPolicy Bypass -File $ScriptPath @ScriptArguments
}

function Resolve-ScipyPython([string]$Requested) {
    $candidates = [System.Collections.Generic.List[string]]::new()
    if (-not [string]::IsNullOrWhiteSpace($Requested)) { $candidates.Add($Requested) }
    if (-not [string]::IsNullOrWhiteSpace($env:IVP_SCIPY_PYTHON)) {
        $candidates.Add($env:IVP_SCIPY_PYTHON)
    }
    $candidates.Add((Join-Path $repoRoot ".venv-gavin-win\Scripts\python.exe"))
    $candidates.Add((Join-Path $repoRoot ".venv\Scripts\python.exe"))
    $candidates.Add((Join-Path $repoRoot ".venv/bin/python"))
    $candidates.Add("python3")
    $candidates.Add("python")

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
        $command = Get-Command $candidate -CommandType Application -ErrorAction SilentlyContinue
        if ($null -ne $command) { return $command.Source }
    }
    throw "IVP parity requires Python with NumPy and SciPy. Pass -PythonPath or set IVP_SCIPY_PYTHON."
}

if (-not $PlanOnly) {
    # Resolve once for the whole campaign. BDF/Radau matrix scripts inherit the
    # same interpreter even when the standalone correctness stage is skipped.
    $env:IVP_SCIPY_PYTHON = Resolve-ScipyPython $PythonPath
    Write-Host "Validating IVP SciPy parity Python: $env:IVP_SCIPY_PYTHON"
    & $env:IVP_SCIPY_PYTHON -c "import sys, numpy, scipy; print(sys.executable); print('NumPy', numpy.__version__); print('SciPy', scipy.__version__)"
    if ($LASTEXITCODE -ne 0) {
        throw "The configured Python cannot import NumPy and SciPy; the nightly campaign was not started."
    }
}

$steps = [System.Collections.Generic.List[object]]::new()
function Invoke-IvpNightlyStep {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$Kind,
        [Parameter(Mandatory)][scriptblock]$Action
    )

    $log = Join-Path $technical "$Name.log"
    if ($PlanOnly) {
        $steps.Add([pscustomobject][ordered]@{
            step = $Name; kind = $Kind; status = "planned"; exit_code = 0
            duration_s = 0; technical_log = "-"; detail = ""
        })
        return
    }

    Write-Host "`n=== IVP nightly: $Name ==="
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    $exitCode = 0
    $detail = ""
    try {
        & $Action 2>&1 | Tee-Object -FilePath $log | ForEach-Object { Write-Host $_ }
        if ($null -ne $LASTEXITCODE) { $exitCode = [int]$LASTEXITCODE }
    }
    catch {
        $exitCode = 1
        $detail = $_.Exception.Message
        Add-Content -LiteralPath $log -Value "`n[runner exception] $detail"
    }
    $watch.Stop()
    $steps.Add([pscustomobject][ordered]@{
        step = $Name
        kind = $Kind
        status = $(if ($exitCode -eq 0) { "passed" } else { "failed" })
        exit_code = $exitCode
        duration_s = [math]::Round($watch.Elapsed.TotalSeconds, 1)
        technical_log = $log
        detail = $detail
    })
}

if (-not $SkipReleaseTests) {
    Invoke-IvpNightlyStep "release_tests" "correctness" {
        $arguments = @("-CargoBuildJobs", [string]$CargoBuildJobs)
        if (-not [string]::IsNullOrWhiteSpace($PythonPath)) {
            $arguments += @("-PythonPath", $PythonPath)
        }
        Invoke-ChildPowerShell $releaseTests $arguments
    }
}

if (-not $SkipPerfCompare) {
    Invoke-IvpNightlyStep "perf_compare" "performance-gate" {
        $arguments = @(
            "-Action", "Compare", "-Machine", $Machine,
            "-Warmups", [string]$PerfWarmups, "-Runs", [string]$PerfRuns
        )
        if (-not [string]::IsNullOrWhiteSpace($Toolchain)) {
            $arguments += @("-Toolchain", $Toolchain)
        }
        Invoke-ChildPowerShell $perfGates $arguments
    }
}

if (-not $SkipReleaseMatrices) {
    Invoke-IvpNightlyStep "bdf_release_matrix" "story-and-bench" {
        Invoke-ChildPowerShell $bdfMatrix @(
            "-Dimensions", $BdfDimensions, "-IncludeIgnoredStories",
            "-IncludeSciPyBenchmark", "-IncludeDetailedBenches", "-FailOnAny"
        )
    }
    Invoke-IvpNightlyStep "radau_release_matrix" "story-and-bench" {
        Invoke-ChildPowerShell $radauMatrix @(
            "-Dimensions", $RadauDimensions, "-FailOnAny"
        )
    }
    Invoke-IvpNightlyStep "lsode2_release_matrix" "story-and-bench" {
        Invoke-ChildPowerShell $lsode2Matrix @(
            "-Dimensions", $Lsode2Dimensions, "-IncludeIgnoredStories",
            "-IncludeDetailedCriterion", "-IncludeLongContinuation", "-FailOnAny"
        )
    }
    Invoke-IvpNightlyStep "be_release_matrix" "story-and-bench" {
        Invoke-ChildPowerShell $beMatrix @(
            "-Dimensions", $BeDimensions, "-IncludeIgnoredStories", "-IncludeAot",
            "-IncludeCriterion", "-IncludeAllocationAudit", "-FailOnAny"
        )
    }
}

$priorFailures = @($steps | Where-Object { $_.status -eq "failed" }).Count
if ($RefreshBaselines) {
    if ($PlanOnly -or $priorFailures -eq 0) {
        Invoke-IvpNightlyStep "perf_record_force" "baseline-refresh" {
            $arguments = @(
                "-Action", "Record", "-Machine", $Machine,
                "-Warmups", [string]$PerfWarmups, "-Runs", [string]$PerfRuns,
                "-Force"
            )
            if (-not [string]::IsNullOrWhiteSpace($Toolchain)) {
                $arguments += @("-Toolchain", $Toolchain)
            }
            Invoke-ChildPowerShell $perfGates $arguments
        }

        $recordFailed = @(
            $steps | Where-Object {
                $_.step -eq "perf_record_force" -and $_.status -eq "failed"
            }
        ).Count -gt 0
        if ($PlanOnly -or -not $recordFailed) {
            Invoke-IvpNightlyStep "perf_compare_refreshed" "baseline-verification" {
                $arguments = @(
                    "-Action", "Compare", "-Machine", $Machine,
                    "-Warmups", [string]$PerfWarmups, "-Runs", [string]$PerfRuns
                )
                if (-not [string]::IsNullOrWhiteSpace($Toolchain)) {
                    $arguments += @("-Toolchain", $Toolchain)
                }
                Invoke-ChildPowerShell $perfGates $arguments
            }
        }
        else {
            $steps.Add([pscustomobject][ordered]@{
                step = "perf_compare_refreshed"; kind = "baseline-verification"
                status = "blocked"; exit_code = 0; duration_s = 0; technical_log = "-"
                detail = "Baseline refresh failed; the replacement references were not accepted."
            })
        }
    }
    else {
        $steps.Add([pscustomobject][ordered]@{
            step = "perf_record_force"; kind = "baseline-refresh"; status = "blocked"
            exit_code = 0; duration_s = 0; technical_log = "-"
            detail = "Earlier nightly stages failed; existing baselines were preserved."
        })
    }
}

$rows = @(
    "| step | kind | status | exit_code | duration_s | technical_log | detail |",
    "|---|---|---|---:|---:|---|---|"
)
foreach ($step in $steps) {
    $log = $step.technical_log.ToString().Replace($repoRoot + "\", "").Replace("\", "/")
    $detail = $step.detail.ToString().Replace("|", "/").Replace("`r", " ").Replace("`n", " ")
    $rows += "| $($step.step) | $($step.kind) | $($step.status) | $($step.exit_code) | $($step.duration_s) | $log | $detail |"
}
$failed = @($steps | Where-Object { $_.status -eq "failed" }).Count
$blocked = @($steps | Where-Object { $_.status -eq "blocked" }).Count
$planned = @($steps | Where-Object { $_.status -eq "planned" }).Count
$overall = if ($PlanOnly) { "planned" } elseif ($failed -eq 0 -and $blocked -eq 0) { "completed" } else { "completed_with_failures" }
$summary = @(
    "# IVP Symbolic Nightly Summary", "",
    "- status: $overall",
    "- recorded_at_utc: $((Get-Date).ToUniversalTime().ToString('o'))",
    "- machine: $Machine",
    "- release_profile: true",
    "- SciPy_python: $($env:IVP_SCIPY_PYTHON)",
    "- failed_steps: $failed",
    "- blocked_steps: $blocked",
    "- planned_steps: $planned", "",
    "## Steps", "", ($rows -join "`n"), "",
    "## Interpretation", "",
    "Solver matrix scripts own their compact Tabled reports below test_reports/.",
    "This campaign does not rewrite STORY_TESTS; accepted results must be reviewed and recorded manually.",
    "A baseline refresh is blocked after any earlier failure and still requires a clean source revision."
)
Set-Content -LiteralPath $summaryPath -Value ($summary -join "`n") -Encoding UTF8

Write-Host "`nIVP symbolic nightly summary"
$steps | Format-Table step, kind, status, exit_code, duration_s -AutoSize
Write-Host "summary=$summaryPath"
if (-not $PlanOnly -and ($failed -gt 0 -or $blocked -gt 0)) { exit 1 }
exit 0

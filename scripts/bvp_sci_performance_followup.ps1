param(
    [string]$SuperLuLibraryPath = $env:SUPERLU_LIBRARY,
    [string]$SciPyPython = $env:BVP_SCIPY_PYTHON,
    [string]$WorkerCounts = "1,4,12",
    [string]$Dimensions = "32,64",
    [string]$Nodes = "128,256",
    [int]$Repeats = 5,
    [ValidateSet("tcc", "gcc")][string]$AotCompiler = "tcc"
)

# Focused release evidence for the outstanding BVP_sci performance questions.
# Each Cargo step is isolated so failures do not suppress the remaining runs;
# compact benchmark reports and noisy build logs have separate destinations.
$ErrorActionPreference = "Continue"

if ([string]::IsNullOrWhiteSpace($SuperLuLibraryPath)) {
    throw "Set SUPERLU_LIBRARY or pass -SuperLuLibraryPath to the SuperLU bridge library."
}
if ([string]::IsNullOrWhiteSpace($SciPyPython)) {
    throw "Set BVP_SCIPY_PYTHON or pass -SciPyPython to the SciPy 1.18.1 environment."
}
if (-not (Test-Path -LiteralPath $SuperLuLibraryPath -PathType Leaf)) {
    throw "SuperLU bridge library does not exist: $SuperLuLibraryPath"
}
if (-not (Test-Path -LiteralPath $SciPyPython -PathType Leaf)) {
    throw "SciPy Python executable does not exist: $SciPyPython"
}

$superLuPath = (Resolve-Path -LiteralPath $SuperLuLibraryPath).Path
$pythonPath = (Resolve-Path -LiteralPath $SciPyPython).Path
$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmss'Z'")
$runRoot = Join-Path (Get-Location) "test_reports\BVP_sci_performance_followup\$stamp"
$reportRoot = Join-Path $runRoot "reports"
$technicalRoot = Join-Path $runRoot "technical"
$summaryPath = Join-Path $runRoot "summary.md"
$null = New-Item -ItemType Directory -Force -Path $reportRoot, $technicalRoot

$env:SUPERLU_LIBRARY = $superLuPath
$env:BVP_SCIPY_PYTHON = $pythonPath
$env:SOMELINALG_PYTHON = $pythonPath
$env:RST_TEST_REPORT_DIR = $reportRoot
$env:RST_TEST_REPORT_PROFILE = "release"
$env:RST_TEST_REPORT_STDOUT = "off"
$env:RST_TEST_REPORT_ARCHIVE = "always"
$env:BVP_SCI_SCIPY_BENCH_WARMUPS = "2"
$env:BVP_SCI_SCIPY_BENCH_REPEATS = [Math]::Max(3, $Repeats).ToString()
$env:BVP_SCI_AOT_COMPILER = $AotCompiler
$env:BVP_SCI_AOT_POLICY_FULL_SOLVE_REPEATS = [Math]::Max(3, $Repeats).ToString()
$env:BVP_SCI_AOT_FULL_SOLVE_WORKLOAD = "coupled-chain"
$env:BVP_SCI_AOT_FULL_SOLVE_DIMENSIONS = $Dimensions
$env:BVP_SCI_AOT_FULL_SOLVE_NODES = $Nodes
$env:BVP_SCI_AOT_BENCH_FRONTENDS = "aot-expr-legacy,aot-atom-native"
$env:BVP_SCI_AOT_BENCH_LAYOUTS = "sparse"
$env:BVP_SCI_AOT_BENCH_POLICIES = "sequential,parallel,auto"
$env:BVP_SCI_AOT_BENCH_OUTPUT = Join-Path (Get-Location) "target\bvp-sci-performance-followup\$stamp"

$steps = [System.Collections.Generic.List[object]]::new()

function Invoke-LoggedCargo {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string[]]$CargoArgs
    )
    $logPath = Join-Path $technicalRoot "$Name.log"
    & cargo @CargoArgs *> $logPath
    $code = $LASTEXITCODE
    $steps.Add([pscustomobject]@{ Name = $Name; ExitCode = $code; Log = $logPath })
    Write-Host "$Name exit=$code log=$logPath"
}

# This route uses exact symbolic ExprLegacy derivatives with the same official
# BVP as the SciPy reference; it fails if SuperLU silently falls back to Faer.
Invoke-LoggedCargo "python_vs_rust_superlu" @(
    "bench", "--no-default-features", "--features", "superlu",
    "--bench", "bvp_sci_scipy_parity", "--", "--noplot"
)

# Fresh artifact roots are created internally per AOT row. Successful build,
# link, publication and runtime-ready lifecycle counters validate cold work;
# resolver cache-hit/miss counters are retained as diagnostics only.
$env:BVP_SCI_AOT_BENCH_PHASE = "cold-amortization"
$env:BVP_SCI_AOT_BENCH_REPORT = "aot_cold_callback_amortization"
$env:BVP_SCI_AOT_BENCH_DIMENSIONS = $Dimensions
$env:BVP_SCI_AOT_BENCH_FRONTENDS = "lambdify-expr-legacy,lambdify-atom-native,aot-expr-legacy,aot-atom-native"
$env:BVP_SCI_AOT_BENCH_LAYOUTS = "sparse"
Invoke-LoggedCargo "aot_cold_callback_amortization" @(
    "bench", "--no-default-features", "--bench", "bvp_sci_aot", "--", "--noplot"
)

# The worker count is process-global. Use one Cargo process per count and keep
# the same large coupled-chain workload, frontend, layout and solver policies.
$env:BVP_SCI_AOT_BENCH_PHASE = "policy-full-solve"
$env:BVP_SCI_AOT_BENCH_DIMENSIONS = $Dimensions
$env:BVP_SCI_AOT_BENCH_NODES = $Nodes
foreach ($workerText in ($WorkerCounts -split ',')) {
    $worker = $workerText.Trim()
    if ([string]::IsNullOrWhiteSpace($worker) -or [int]$worker -lt 1) {
        continue
    }
    $env:RAYON_NUM_THREADS = $worker
    $env:BVP_SCI_AOT_POLICY_FULL_SOLVE_WORKER_COUNT = $worker
    $env:BVP_SCI_AOT_BENCH_REPORT = "aot_coupled_chain_full_solve_workers_$worker"
    Invoke-LoggedCargo "aot_full_solve_workers_$worker" @(
        "bench", "--no-default-features", "--bench", "bvp_sci_aot", "--", "--noplot"
    )
}

$lines = @(
    "# BVP_sci Performance Follow-Up ($stamp)",
    "",
    "- SciPy Python: ``$pythonPath``",
    "- SuperLU library: ``$superLuPath``",
    "- AOT full-solve workload: ``coupled-chain``",
    "- dimensions/nodes: ``$Dimensions`` / ``$Nodes``",
    "- worker counts: ``$WorkerCounts``",
    "",
    "| step | exit | technical log |",
    "|---|---:|---|"
)
foreach ($step in $steps) {
    $status = if ($step.ExitCode -eq 0) { "passed" } else { "failed" }
    $relativeLog = Resolve-Path -LiteralPath $step.Log -Relative
    $lines += "| ``$($step.Name)`` ($status) | $($step.ExitCode) | ``$relativeLog`` |"
}
$lines += ""
$lines += "Compact Tabled benchmark reports are under ``$reportRoot``; build/compiler output is isolated under ``$technicalRoot``. Performance break-even remains unclaimed until the reported rows are reviewed."
Set-Content -LiteralPath $summaryPath -Value $lines -Encoding utf8

$failed = @($steps | Where-Object { $_.ExitCode -ne 0 }).Count
Write-Host "reports=$reportRoot"
Write-Host "technical=$technicalRoot"
Write-Host "summary=$summaryPath"
if ($failed -gt 0) {
    Write-Host "status=completed_with_failures failed=$failed"
    exit 1
}
Write-Host "status=completed passed=$($steps.Count) failed=0"

param(
    [string]$Python = ".venv-gavin-win\Scripts\python.exe",
    [int]$Warmups = 2,
    [int]$Repeats = 9
)

$ErrorActionPreference = "Continue"
$repoRoot = (Get-Location).Path
$pythonPath = if ([System.IO.Path]::IsPathRooted($Python)) {
    $Python
} else {
    Join-Path $repoRoot $Python
}
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Python interpreter not found: $pythonPath"
}

$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmss'Z'")
$captureRoot = Join-Path $repoRoot "test_reports\BVP_sci_scipy_benchmark\$stamp"
$reportRoot = Join-Path $captureRoot "reports"
$technicalRoot = Join-Path $captureRoot "technical"
New-Item -ItemType Directory -Force -Path $reportRoot, $technicalRoot | Out-Null

$env:RST_TEST_REPORT_DIR = $reportRoot
$env:RST_TEST_REPORT_PROFILE = "release"
$env:RST_TEST_REPORT_STDOUT = "off"
$env:RST_TEST_REPORT_ARCHIVE = "always"
$env:BVP_SCI_SCIPY_PYTHON = $pythonPath
$env:BVP_SCI_SCIPY_BENCH_WARMUPS = [Math]::Max(0, $Warmups).ToString()
$env:BVP_SCI_SCIPY_BENCH_REPEATS = [Math]::Max(1, $Repeats).ToString()
$env:BVP_SCI_SCIPY_BENCH_REPORT = "python_vs_rust_numerical_core_$stamp"

$technicalLog = Join-Path $technicalRoot "cargo_bench.log"
cargo bench --no-default-features --bench bvp_sci_scipy_parity -- --noplot 2>&1 |
    Tee-Object -FilePath $technicalLog |
    ForEach-Object {
        $line = $_.ToString()
        if ($line -match '^report=' -or $line -match '^error:') {
            Write-Host $line
        }
    }
$exitCode = [int]$LASTEXITCODE

$summary = @(
    "# BVP_sci SciPy parity benchmark run",
    "",
    "- status: $(if ($exitCode -eq 0) { 'passed' } else { 'failed' })",
    "- benchmark: Python SciPy wall-clock versus Rust numerical core",
    "- warmups: $($env:BVP_SCI_SCIPY_BENCH_WARMUPS)",
    "- measured repeats: $($env:BVP_SCI_SCIPY_BENCH_REPEATS)",
    "- reports: ``reports/``",
    "- compiler/build chatter: ``technical/cargo_bench.log``",
    "- report rows require identical successful status and final node count",
    ""
) -join "`n"
[System.IO.File]::WriteAllText(
    (Join-Path $captureRoot "release_summary.md"),
    $summary,
    [System.Text.UTF8Encoding]::new($false)
)

Write-Host "status=$(if ($exitCode -eq 0) { 'passed' } else { 'failed' }) exit=$exitCode reports=$reportRoot technical=$technicalRoot"
exit $exitCode

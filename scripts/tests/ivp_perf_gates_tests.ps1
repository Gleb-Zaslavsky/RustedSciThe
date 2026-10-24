# Run in Windows PowerShell 5.1 or PowerShell 7. No Cargo, Python or numerical work.
$ErrorActionPreference = 'Stop'
$scripts = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $scripts 'ivp_perf_gates.ps1'
$fake = Join-Path $PSScriptRoot 'fixtures/ivp_perf_gates_fake_runner.ps1'
$hostExe = (Get-Process -Id $PID).Path
$root = Join-Path ([IO.Path]::GetTempPath()) ('ivp gate tests ' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $root | Out-Null
$previousRoot = $env:IVP_GATE_TEST_ROOT
$previousMode = $env:IVP_GATE_TEST_MODE
$env:IVP_GATE_TEST_ROOT = $root
function Assert-Gate($Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}
function Invoke-MatrixTest([string]$Name, [string[]]$Arguments, [int]$ExpectedExit) {
    $reports = Join-Path $root $Name
    & $hostExe -NoProfile -ExecutionPolicy Bypass -File $runner -Machine test `
        -RunnerScript $fake -ReportRoot $reports @Arguments > (Join-Path $root "$Name.log") 2>&1
    Assert-Gate ($LASTEXITCODE -eq $ExpectedExit) "$Name exit code $LASTEXITCODE, expected $ExpectedExit"
    $summary = @(Get-ChildItem -LiteralPath $reports -Recurse -Filter summary.json)
    Assert-Gate ($summary.Count -eq 1) "$Name must produce one matrix summary"
    return Get-Content -LiteralPath $summary[0].FullName -Raw | ConvertFrom-Json
}
try {
    $env:IVP_GATE_TEST_MODE = 'mixed'
    $mixed = Invoke-MatrixTest 'mixed' @('-Action', 'Compare') 1
    Assert-Gate ($mixed.steps.Count -eq 4) 'All four solvers must be attempted'
    Assert-Gate ($mixed.steps[1].status -eq 'passed') 'Radau must run after BDF failure'
    Assert-Gate ($mixed.steps[2].status -eq 'error') 'Preflight failure must be explicit'
    Assert-Gate ($mixed.steps[3].status -eq 'error') 'BE must run after preflight failure'
    Assert-Gate ($mixed.counts.unstable -eq 1 -and $mixed.counts.regression -eq 1) 'Distinct noise and regression outcomes'
    Assert-Gate ($mixed.counts.error -eq 4) 'Missing/incompatible/prerequisite/timeout outcomes'
    $noisy = $mixed.scenarios | Where-Object scenario -eq 'bdf_noisy'
    Assert-Gate ($noisy.status -eq 'unstable') 'Confirmation must not turn UNSTABLE into PASS'
    Assert-Gate ($noisy.wall_ms -eq 123.456 -and $noisy.confirmation_wall_ms -eq 120) 'Both series must survive'
    Assert-Gate ($noisy.samples -eq 301) 'Sample count must survive'
    $campaign = (Get-ChildItem -LiteralPath (Join-Path $root 'mixed') -Directory)[0].FullName
    Assert-Gate (Test-Path -LiteralPath (Join-Path $campaign $noisy.measurements)) 'Native evidence must be copied'
    Assert-Gate (Test-Path -LiteralPath (Join-Path $campaign 'scenarios.csv')) 'CSV must exist'
    $markdown = Get-Content -LiteralPath (Join-Path $campaign 'summary.md') -Raw
    Assert-Gate ($markdown.Contains('unstable') -and $markdown.Contains('metric set changed')) 'Markdown must explain failures'

    . (Join-Path $scripts 'ivp_perf_gates_reporting.ps1')
    $nativeSummary = Join-Path $noisy.native_report 'summary.md'
    $savedSummary = Get-Content -LiteralPath $nativeSummary -Raw
    Add-Content -LiteralPath $nativeSummary -Value '- exit `5`: unknown_scenario: measurements are unstable'
    $rejected = $false
    try { $null = @(Read-IvpGateReport $noisy.native_report 'bdf' 'compare' $campaign 'test.log') }
    catch { $rejected = $true }
    Assert-Gate $rejected 'Unknown error ownership must fail closed'
    Set-Content -LiteralPath $nativeSummary -Value $savedSummary
    Set-Content -LiteralPath (Join-Path $noisy.native_report 'bdf_matrix_ok.md') -Value '# Missing measurements'
    $invalidRows = @(Read-IvpGateReport $noisy.native_report 'bdf' 'compare' $campaign 'test.log')
    Assert-Gate (($invalidRows | Where-Object scenario -eq 'bdf_matrix_ok').status -eq 'error') 'Missing success evidence must be an error'

    $paired = Invoke-MatrixTest 'paired' @('-Action', 'RecordCompare', '-Force', '-AllowDirtySource') 1
    Assert-Gate ($paired.steps.Count -eq 8) 'Record/compare pairs must be planned'
    Assert-Gate ($paired.steps[1].status -eq 'blocked') 'Failed record must block same-solver compare'
    Assert-Gate ($paired.steps[2].status -eq 'passed' -and $paired.steps[3].status -eq 'passed') 'Next solver must record and compare'
    $calls = Get-Content -LiteralPath (Join-Path $root 'calls.txt')
    Assert-Gate (@($calls | Where-Object { $_ -match '/compare force=True|/compare .*dirty=True' }).Count -eq 0) 'Record flags must not reach compare'

    $env:IVP_GATE_TEST_MODE = 'pass'
    $passed = Invoke-MatrixTest 'passed' @('-Action', 'RecordCompare', '-Solvers', 'be', '-Force') 0
    Assert-Gate ($passed.status -eq 'completed' -and $passed.scenarios.Count -eq 4) 'Both BE groups must be collected in both actions'

    $env:IVP_GATE_TEST_MODE = 'missing'
    $missing = Invoke-MatrixTest 'missing' @('-Action', 'Compare', '-Solvers', 'bdf') 1
    Assert-Gate ($missing.steps[0].detail -like '*No native run report*') 'Missing report must not become a pass'
    Assert-Gate ($missing.steps[0].detail -like '*refusing to record a baseline from dirty source*') 'Native preflight cause must appear in the summary'
    $missingCampaign = (Get-ChildItem -LiteralPath (Join-Path $root 'missing') -Directory)[0].FullName
    $missingMarkdown = Get-Content -LiteralPath (Join-Path $missingCampaign 'summary.md') -Raw
    Assert-Gate ($missingMarkdown.Contains('Step errors: 1')) 'A preflight error must be counted even without scenarios'
    $planRoot = Join-Path $root 'plan'
    & $hostExe -NoProfile -ExecutionPolicy Bypass -File $runner -Machine test -Action Compare -PlanOnly -ReportRoot $planRoot
    Assert-Gate ($LASTEXITCODE -eq 0 -and -not (Test-Path -LiteralPath $planRoot)) 'PlanOnly must not create artifacts'
    Write-Host "PASS: IVP matrix reporting, continuation, confirmation, partial record, preflight and multi-group contracts. Fixtures: $root"
}
finally {
    $env:IVP_GATE_TEST_ROOT = $previousRoot
    $env:IVP_GATE_TEST_MODE = $previousMode
}

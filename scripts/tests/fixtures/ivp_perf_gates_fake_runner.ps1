# Synthetic native report producer for orchestration tests. No solver is run.
param(
    [string]$Solver, [string]$Action, [string]$Machine, [int]$Warmups, [int]$Runs,
    [double]$MaxRelativeMad, [double]$Tolerance, [switch]$Force,
    [switch]$AllowDirtySource, [string]$Toolchain
)
$ErrorActionPreference = 'Stop'
Add-Content -LiteralPath (Join-Path $env:IVP_GATE_TEST_ROOT 'calls.txt') -Value "$Solver/$Action force=$Force dirty=$AllowDirtySource"
if ($env:IVP_GATE_TEST_MODE -eq 'missing' -or $Solver -eq 'lsode2') {
    [Console]::Error.WriteLine('perf-gates: invalid configuration: refusing to record a baseline from dirty source revision synthetic; commit the source or pass --allow-dirty-source explicitly')
    exit 2
}
$allPass = $env:IVP_GATE_TEST_MODE -eq 'pass'
$groups = if ($Solver -eq 'be') { @('matrix', 'large') } else { @('matrix') }
$exitCode = 0
foreach ($group in $groups) {
    $path = Join-Path $env:IVP_GATE_TEST_ROOT "$Solver-$Action-$group-$PID"
    New-Item -ItemType Directory -Path $path | Out-Null
    $ids = @("${Solver}_${group}_ok")
    $errors = @()
    if (-not $allPass -and $Solver -eq 'bdf') {
        $ids += 'bdf_noisy'
        $errors += '- exit `5`: bdf_noisy: measurements are unstable'
        $exitCode = 5
        if ($Action -eq 'Compare') {
            $ids += 'bdf_slow'
            $errors += '- exit `1`: bdf_slow: confirmed performance regression'
        }
    }
    if (-not $allPass -and $Solver -eq 'be' -and $group -eq 'large') {
        $ids += 'be_missing', 'be_incompatible', 'be_tool', 'be_timeout'
        $errors += '- exit `3`: be_missing: no baseline: missing.toml'
        $errors += '- exit `4`: be_incompatible: incompatible baseline: metric set changed'
        $errors += '- exit `2`: scenario be_tool requires `tcc` on PATH; install it or exclude this scenario'
        $errors += '- exit `2`: be_timeout confirmation: isolated scenario process failed: timeout'
        $exitCode = 4
    }
    $outcome = if ($errors.Count) { 'failed' } else { 'ok' }
    $summary = @(
        "# Performance $($Action.ToLowerInvariant()) run", '',
        '- machine: `test`', '- source: `test-revision`',
        ('- selected scenarios: ' + ($ids -join ', ')),
        ('- outcome: `' + $outcome + '`'), '', '## Errors'
    ) + $errors
    $summary | Set-Content -LiteralPath (Join-Path $path 'summary.md')
    'schema_version = 1' | Set-Content -LiteralPath (Join-Path $path 'summary.toml')
    foreach ($id in $ids) {
        if ($id -in @('be_missing', 'be_incompatible', 'be_tool', 'be_timeout')) { continue }
        $measure = @(
            "# Performance run: $id", '', '## Primary', '',
            '| Metric | Phase | Scope | Parent scope | Samples | Median | MAD | Min | Max |',
            '| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |',
            '| scenario_wall_ms (ms/off) | gate_off | scenario_outer_wall | - | 301 | 123.456000 | 6.700000 | 110 | 140 |'
        )
        if ($id -eq 'bdf_noisy') {
            # A passing confirmation must not erase the final UNSTABLE outcome.
            $measure += '', '## Confirmation', '',
                '| scenario_wall_ms (ms/off) | gate_off | scenario_outer_wall | - | 301 | 120.000000 | 1.000000 | 118 | 122 |'
        }
        $measure | Set-Content -LiteralPath (Join-Path $path "$id.md")
        'scenario_id = "synthetic"' | Set-Content -LiteralPath (Join-Path $path "$id.toml")
    }
    Write-Output "run report: $path"
}
exit $exitCode

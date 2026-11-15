$ErrorActionPreference = "Continue"

$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$root = Join-Path (Get-Location) "test_reports\AI_agent_interop_perf_manual\$stamp"
$reports = Join-Path $root "reports"
$technical = Join-Path $root "technical"
New-Item -ItemType Directory -Force -Path $reports, $technical | Out-Null

$log = Join-Path $technical "criterion.log"
$watch = [System.Diagnostics.Stopwatch]::StartNew()
$exitCode = 0
try {
    & cargo bench --no-default-features --bench ai_agent_interop_json -- --noplot *> $log
    if ($null -ne $LASTEXITCODE) { $exitCode = [int]$LASTEXITCODE }
}
catch {
    $exitCode = 1
    Add-Content -Path $log -Value ("`n[runner exception] " + $_.Exception.Message)
}
$watch.Stop()

$benchmarks = @(
    @{ Id = "agent_json_task_preflight_ivp"; Family = "IVP"; Operation = "task preflight" },
    @{ Id = "agent_json_task_preflight_bvp"; Family = "BVP"; Operation = "task preflight" },
    @{ Id = "agent_json_task_preflight_nonlinear_root"; Family = "Nonlinear root"; Operation = "task preflight" },
    @{ Id = "agent_json_task_preflight_fitting"; Family = "Fitting"; Operation = "task preflight" },
    @{ Id = "agent_json_result_json_encode_ivp"; Family = "IVP"; Operation = "result JSON serialization" },
    @{ Id = "agent_json_result_json_encode_bvp"; Family = "BVP"; Operation = "result JSON serialization" },
    @{ Id = "agent_json_result_json_encode_nonlinear_root"; Family = "Nonlinear root"; Operation = "result JSON serialization" },
    @{ Id = "agent_json_result_json_encode_fitting"; Family = "Fitting"; Operation = "result JSON serialization" },
    @{ Id = "agent_json_full_run_ivp"; Family = "IVP"; Operation = "JSON parse + solve + envelope" },
    @{ Id = "agent_json_full_run_bvp"; Family = "BVP"; Operation = "JSON parse + solve + envelope" },
    @{ Id = "agent_json_full_run_nonlinear_root"; Family = "Nonlinear root"; Operation = "JSON parse + solve + envelope" },
    @{ Id = "agent_json_full_run_fitting"; Family = "Fitting"; Operation = "JSON parse + solve + envelope" },
    @{ Id = "agent_json_outer_job_preflight_ivp"; Family = "Outer job"; Operation = "job preflight (IVP)" }
)

$rows = [System.Collections.Generic.List[string]]::new()
$rows.Add("| Family | Operation | Mean (us) | Median (us) | Mean 95% CI (us) |")
$rows.Add("|---|---|---:|---:|---:|")
$missing = [System.Collections.Generic.List[string]]::new()
foreach ($benchmark in $benchmarks) {
    $estimatePath = Join-Path (Get-Location) ("target\criterion\{0}\new\estimates.json" -f $benchmark.Id)
    if (-not (Test-Path $estimatePath)) {
        $missing.Add($benchmark.Id)
        continue
    }
    $estimate = Get-Content $estimatePath -Raw | ConvertFrom-Json
    $mean = [double]$estimate.mean.point_estimate / 1000.0
    $median = [double]$estimate.median.point_estimate / 1000.0
    $low = [double]$estimate.mean.confidence_interval.lower_bound / 1000.0
    $high = [double]$estimate.mean.confidence_interval.upper_bound / 1000.0
    $rows.Add(("| {0} | {1} | {2:F3} | {3:F3} | [{4:F3}, {5:F3}] |" -f $benchmark.Family, $benchmark.Operation, $mean, $median, $low, $high))
}

$status = if ($exitCode -eq 0 -and $missing.Count -eq 0) { "passed" } else { "failed" }
$summaryPath = Join-Path $reports "performance_summary.md"
$summary = @(
    "# AI-Agent Interop JSON Performance Matrix",
    "",
    "- status: $status",
    "- recorded_at_utc: $((Get-Date).ToUniversalTime().ToString('o'))",
    "- profile: Criterion bench profile, no default features",
    "- duration_s: $([math]::Round($watch.Elapsed.TotalSeconds, 1))",
    "- technical_log: $log",
    "- benchmark_count: $($rows.Count - 2) / $($benchmarks.Count)",
    "",
    "Times are Criterion estimates; full-run rows include numerical work and must not be interpreted as JSON-adapter overhead.",
    "",
    ""
)
$summary += $rows.ToArray()
$summary += ""
if ($missing.Count -gt 0) {
    $summary += "Missing estimate files: $($missing -join ', ')"
}
Set-Content -Path $summaryPath -Value ($summary -join "`n") -Encoding UTF8
Write-Host ("[ai-agent-interop] {0} exit={1} duration_s={2}" -f $status, $exitCode, [math]::Round($watch.Elapsed.TotalSeconds, 1))
Write-Host "reports=$reports"
Write-Host "technical=$technical"
Write-Host "summary=$summaryPath"
if ($exitCode -ne 0 -or $missing.Count -gt 0) { exit 1 }
exit 0

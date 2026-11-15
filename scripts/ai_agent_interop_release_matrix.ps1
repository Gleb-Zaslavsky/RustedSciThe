$ErrorActionPreference = "Continue"
$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$root = Join-Path (Get-Location) "test_reports\AI_agent_interop_release_manual\$stamp"
$reports = Join-Path $root "reports"
$technical = Join-Path $root "technical"
New-Item -ItemType Directory -Force -Path $reports, $technical | Out-Null

$env:RST_TEST_REPORT_DIR = $reports
$env:RST_TEST_REPORT_PROFILE = "release"
$env:RST_TEST_REPORT_ARCHIVE = "always"
$env:RST_TEST_REPORT_STDOUT = "off"

$groups = @(
    @{ Name = "task_json_worker_and_contract"; Filter = "command_interpreter::" },
    @{ Name = "agent_task_shell_cli"; Filter = "task_shell_cli::" },
    @{ Name = "dop853_facade_telemetry_and_cancel"; Filter = "universal_dop853" },
    @{ Name = "dop853_native_telemetry_opt_in"; Filter = "dop853_statistics_follow_opt_in_telemetry_mode_without_changing_solution" },
    @{ Name = "native_cancellation_and_snapshots"; Filter = "cancel" }
)
$rows = [System.Collections.Generic.List[object]]::new()

foreach ($group in $groups) {
    $log = Join-Path $technical ("{0}.log" -f $group.Name)
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    $exitCode = 0
    try {
        & cargo test --release --lib $group.Filter -- --include-ignored --nocapture --test-threads=1 *> $log
        if ($null -ne $LASTEXITCODE) { $exitCode = [int]$LASTEXITCODE }
    }
    catch {
        $exitCode = 1
        Add-Content -Path $log -Value ("`n[runner exception] " + $_.Exception.Message)
    }
    $watch.Stop()
    $status = if ($exitCode -eq 0) { "passed" } else { "failed" }
    $duration = [math]::Round($watch.Elapsed.TotalSeconds, 1)
    $testResult = Select-String -Path $log -Pattern '^test result:' | Select-Object -Last 1
    $counts = [regex]::Match(
        [string]$testResult,
        '(?<passed>\d+) passed; (?<failed>\d+) failed; (?<ignored>\d+) ignored'
    )
    $passed = if ($counts.Success) { [int]$counts.Groups['passed'].Value } else { 0 }
    $failedTests = if ($counts.Success) { [int]$counts.Groups['failed'].Value } else { 0 }
    $ignored = if ($counts.Success) { [int]$counts.Groups['ignored'].Value } else { 0 }
    $rows.Add([pscustomobject]@{
        suite = $group.Name
        filter = $group.Filter
        status = $status
        exit_code = $exitCode
        duration_s = $duration
        passed = $passed
        failed = $failedTests
        ignored = $ignored
        technical_log = $log
    })
    Write-Host ("[{0}] {1} exit={2} passed={3} failed={4} ignored={5} duration_s={6}" -f $group.Name, $status, $exitCode, $passed, $failedTests, $ignored, $duration)
}

$failedSuites = @($rows | Where-Object status -ne "passed").Count
$overall = if ($failedSuites -eq 0) { "passed" } else { "failed" }
$summary = [System.Collections.Generic.List[string]]::new()
foreach ($line in @(
    "# AI-Agent Interop Release Test Matrix",
    "",
    "- status: $overall",
    "- recorded_at_utc: $((Get-Date).ToUniversalTime().ToString('o'))",
    "- profile: release",
    "- include_ignored: true",
    "- report_root: $root",
    "",
    "The command-interpreter group covers task JSON, preflight, native runners, outer jobs, worker protocol and acceptance contracts. The task-shell group covers the agent-facing batch and persistent-worker CLI. DOP853 facade tests cover cancellation forwarding; its native telemetry opt-in parity test has a dedicated group. The native cancellation group discovers cancellation/snapshot tests across solver modules. Groups run serially to keep compiler/AOT and telemetry fixtures isolated.",
    "",
    "| suite | filter | status | passed | failed | ignored | duration_s | technical_log |",
    "|---|---|---|---:|---:|---:|---:|---|"
)) { $summary.Add($line) }
foreach ($row in $rows) {
    $relativeLog = $row.technical_log -replace '\\', '/'
    $summary.Add("| $($row.suite) | ``$($row.filter)`` | $($row.status) | $($row.passed) | $($row.failed) | $($row.ignored) | $($row.duration_s) | $relativeLog |")
}
$summary.Add("")
$summaryPath = Join-Path $reports "release_summary.md"
Set-Content -Path $summaryPath -Value ($summary.ToArray() -join "`n") -Encoding UTF8
Write-Host "reports=$reports"
Write-Host "technical=$technical"
Write-Host "summary=$summaryPath"
if ($failedSuites -gt 0) { exit 1 }
exit 0

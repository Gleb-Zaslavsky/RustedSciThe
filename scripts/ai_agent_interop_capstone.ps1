param(
    [ValidateSet("Debug", "Release")]
    [string]$Profile = "Release",
    [string]$ResumeFrom
)

$ErrorActionPreference = "Continue"
$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$root = Join-Path (Get-Location) "test_reports\AI_agent_interop_capstone\$stamp"
$reports = Join-Path $root "reports"
$technical = Join-Path $root "technical"
New-Item -ItemType Directory -Force -Path $reports, $technical | Out-Null
$workloadReports = if ([string]::IsNullOrWhiteSpace($ResumeFrom)) {
    Join-Path $reports "workloads"
} else {
    (Resolve-Path -LiteralPath $ResumeFrom).Path
}
New-Item -ItemType Directory -Force -Path $workloadReports | Out-Null

$env:RST_TEST_REPORT_DIR = $reports
$env:RST_TEST_REPORT_PROFILE = $Profile.ToLowerInvariant()
$env:RST_TEST_REPORT_ARCHIVE = "always"
$env:RST_TEST_REPORT_STDOUT = "off"
$env:RST_AI_AGENT_CAPSTONE_REPORT_DIR = $workloadReports

$log = Join-Path $technical "four_family_json_worker_capstone.log"
$watch = [System.Diagnostics.Stopwatch]::StartNew()
$exitCode = 0
try {
    $cargoArgs = @("test", "--test", "ai_agent_interop_capstone")
    if ($Profile -eq "Release") { $cargoArgs += "--release" }
    $cargoArgs += @("--", "--ignored", "--nocapture", "--test-threads=1")
    & cargo @cargoArgs *> $log
    if ($null -ne $LASTEXITCODE) { $exitCode = [int]$LASTEXITCODE }
}
catch {
    $exitCode = 1
    Add-Content -Path $log -Value ("`n[runner exception] " + $_.Exception.Message)
}
$watch.Stop()
$status = if ($exitCode -eq 0) { "passed" } else { "failed" }
$duration = [math]::Round($watch.Elapsed.TotalSeconds, 1)
$resultLine = Select-String -Path $log -Pattern '^test result:' | Select-Object -Last 1
$summary = @(
    "# AI-Agent Interop Capstone Matrix",
    "",
    "- status: $status",
    "- profile: $Profile",
    "- includes_ignored: true",
    "- exit_code: $exitCode",
    "- duration_s: $duration",
    "- test_result: $($resultLine.Line)",
    "- workload_report_dir: $workloadReports",
    "- resume_from: $(if ([string]::IsNullOrWhiteSpace($ResumeFrom)) { 'none' } else { $ResumeFrom })",
    "- technical_log: $log",
    "",
    "The gate submits versioned JSON jobs through the public sequential JSONL worker, runs preflight, executes IVP/BVP/nonlinear/fitting workloads, checks declared acceptance and independent numerical invariants, and checks default-deny AOT policy behavior."
)
$summaryPath = Join-Path $reports "summary.md"
Set-Content -Path $summaryPath -Value ($summary -join "`n") -Encoding UTF8
Write-Host "[$status] ai-agent-four-family-capstone duration_s=$duration"
Write-Host "reports=$reports"
Write-Host "technical=$technical"
Write-Host "summary=$summaryPath"
if ($exitCode -ne 0) { exit $exitCode }
exit 0

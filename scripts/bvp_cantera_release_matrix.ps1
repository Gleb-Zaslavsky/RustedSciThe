$ErrorActionPreference = "Continue"
$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$root = Join-Path (Get-Location) "test_reports\BVP_Cantera_release_manual\$stamp"
$reports = Join-Path $root "reports"
$technical = Join-Path $root "technical"
New-Item -ItemType Directory -Force -Path $reports, $technical | Out-Null

$env:RST_TEST_REPORT_DIR = $reports
$env:RST_TEST_REPORT_PROFILE = "release"
$env:RST_TEST_REPORT_ARCHIVE = "always"
$env:RST_TEST_REPORT_STDOUT = "off"

$results = New-Object System.Collections.Generic.List[object]
function Invoke-Step([string]$Name, [scriptblock]$Action) {
    $log = Join-Path $technical "$Name.log"
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    $exitCode = 0
    try {
        & $Action *> $log
        if ($null -ne $LASTEXITCODE) { $exitCode = [int]$LASTEXITCODE }
    }
    catch {
        $exitCode = 1
        Add-Content -Path $log -Value ("`n[runner exception] " + $_.Exception.Message)
    }
    $watch.Stop()
    $status = if ($exitCode -eq 0) { "passed" } else { "failed" }
    $results.Add([pscustomobject][ordered]@{
        step = $Name
        status = $status
        exit_code = $exitCode
        duration_s = [math]::Round($watch.Elapsed.TotalSeconds, 1)
        technical_log = $log
    })
    Write-Host ("[{0}] {1} exit={2}" -f $Name, $status, $exitCode)
}

Invoke-Step "cantera_release_all_including_ignored" {
    cargo test --release --lib numerical::BVP_Cantera:: -- --include-ignored --nocapture --test-threads=1
}
Invoke-Step "bvp_api_cantera_facade" {
    cargo test --release --lib numerical::BVP_api:: -- --include-ignored --nocapture --test-threads=1
}
Invoke-Step "cantera_perf_gate_correctness_and_registry" {
    cargo test --release --features perf-gates --bin perf-gates bvp_cantera -- --nocapture --test-threads=1
}
Invoke-Step "cantera_full_task_document_parser" {
    cargo test --release --lib cantera_full_document -- --nocapture --test-threads=1
}
Invoke-Step "cantera_hybrid_settings_only_parser" {
    cargo test --release --lib cantera_hybrid_settings -- --nocapture --test-threads=1
}
Invoke-Step "python_binding_compile" {
    cargo check --manifest-path bindings/python/Cargo.toml --no-default-features
}
Invoke-Step "cantera_facade_example" {
    cargo run --release --example bvp_api_cantera
}
Invoke-Step "cantera_combustion_parameter_continuation_example" {
    cargo run --release --example bvp_api_cantera_combustion_continuation
}
Invoke-Step "cantera_bratu_parameter_continuation_example" {
    cargo run --release --example bvp_api_cantera_bratu_continuation
}
Invoke-Step "cantera_lambdify_aot_frontend_backend_matrix" {
    cargo bench --no-default-features --bench bvp_cantera_workloads
}
Invoke-Step "cantera_aot_expr_atom_layout_parity" {
    cargo test --release --lib numerical::BVP_Cantera::tests::symbolic::aot_frontends_match_lambdify_across_linear_layouts -- --include-ignored --nocapture --test-threads=1
}
Invoke-Step "cantera_aot_compiler_profile_matrix" {
    cargo test --release --lib numerical::BVP_Cantera::tests::symbolic::aot_compiler_and_profile_matrix_matches_lambdify -- --include-ignored --nocapture --test-threads=1
}

$summaryPath = Join-Path $reports "release_summary.md"
$table = @(
    "| step | status | exit_code | duration_s | technical_log |",
    "|---|---|---:|---:|---|"
)
foreach ($row in $results) {
    $logPath = $row.technical_log -replace '\\', '/'
    $table += "| $($row.step) | $($row.status) | $($row.exit_code) | $($row.duration_s) | $logPath |"
}
$failed = @($results | Where-Object { $_.exit_code -ne 0 }).Count
$status = if ($failed -eq 0) { "completed" } else { "completed_with_failures" }
$summary = @(
    "# BVP_Cantera Release Matrix",
    "",
    "- status: $status",
    "- recorded_at_utc: $((Get-Date).ToUniversalTime().ToString('o'))",
    "- report_root: $root",
    "- reports: $reports",
    "- technical: $technical",
    "",
    "## Steps",
    "",
    ($table -join "`n"),
    "",
    "Compiler and Cargo transcripts are in the technical directory; structured benchmark output is in reports."
)
Set-Content -Path $summaryPath -Value ($summary -join "`n") -Encoding UTF8
Write-Host "reports=$reports"
Write-Host "technical=$technical"
Write-Host "summary=$summaryPath"

if ($failed -gt 0) { exit 1 }
exit 0

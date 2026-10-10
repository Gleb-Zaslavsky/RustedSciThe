param(
    [string]$Nodes = "8,32,128",
    [int]$Continuation = 4,
    [string]$Workloads = "linear,parameterized-linear,oscillator,stiff-decay,bratu-like,combustion-like,stiff-coupled",
    [string]$ContinuationWorkloads = "parameterized-linear",
    [string]$ContinuationNodes = "64,256,1024",
    [string]$ContinuationCounts = "1,4,16",
    [string]$ContinuationModes = "fresh,prepared,warm",
    [string]$ScaleNodes = "20,64,256,1024",
    [string]$ScaleWorkloads = "stiff-decay,stiff-coupled",
    [string]$PolicyFullSolveWorkers = "1,2,4,8,12",
    [string]$PolicyFullSolveNodes = "64,256",
    [string]$PolicyFullSolveWorkloads = "stiff-coupled,combustion-like",
    [string]$PolicyFullSolveLayouts = "sparse,banded",
    [string]$PolicyFullSolveFrontends = "expr-legacy,atom-native",
    [int]$PolicyFullSolveRepeats = 3,
    [switch]$IncludeAot,
    [string]$AotDimensions = "2,8,32",
    [string]$AotFullSolveDimensions = "8,32",
    [string]$AotFullSolveNodes = "16,64",
    [string]$AotPolicyFullSolveWorkers = "1,2,4,8,12",
    [int]$AotPolicyFullSolveRepeats = 3,
    [int]$AotContinuationCount = 16,
    [string]$AotFrontends = "lambdify-expr-legacy,lambdify-atom-native,aot-expr-legacy,aot-atom-native",
    [string]$AotLayouts = "dense,sparse,banded",
    [string]$AotPolicies = "sequential,parallel,auto",
    [string]$AotPhases = "matrix,continuation,policy",
    [string]$MatchedDimensions = "8,32",
    [string]$MatchedNodes = "16,64",
    [string]$MatchedCounts = "1,4,16",
    [int]$MatchedRepeats = 3,
    [ValidateSet("tcc", "gcc")][string]$AotCompiler = "tcc",
    [ValidateSet("sequential", "parallel", "auto")][string]$Policy = "sequential",
    [int]$PolicyMinWork = 64,
    [switch]$SkipIgnoredStories,
    [switch]$SkipBench,
    [switch]$SkipContinuationBench,
    [switch]$SkipContinuationLifecycleBench,
    [switch]$SkipMatchedFrontendBench,
    [switch]$SkipPolicyFullSolveBench,
    [switch]$SkipScaleEvidence,
    [switch]$PlanOnly,
    [switch]$FailOnAny
)

# BVP_sci release runner. Each step is independent: a failing story gate or
# benchmark does not suppress the remaining evidence. Compact tables are
# written below reports/; compiler and Criterion chatter stays in technical/
# and is never mixed into measured Markdown artifacts.
$ErrorActionPreference = "Continue"

$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmss'Z'")
$captureRoot = Join-Path (Get-Location) "test_reports\BVP_sci_release_manual\$stamp"
$reportRoot = Join-Path $captureRoot "reports"
$technicalRoot = Join-Path $captureRoot "technical"
$summaryPath = Join-Path $captureRoot "release_summary.md"
New-Item -ItemType Directory -Force -Path $reportRoot, $technicalRoot | Out-Null

$env:RST_TEST_REPORT_DIR = $reportRoot
$env:RST_TEST_REPORT_PROFILE = "release"
$env:RST_TEST_REPORT_STDOUT = "off"
$env:RST_TEST_REPORT_ARCHIVE = "always"
$env:BVP_SCI_BENCH_NODES = $Nodes
$env:BVP_SCI_BENCH_CONTINUATION = [Math]::Max(1, $Continuation).ToString()
$env:BVP_SCI_BENCH_WORKLOADS = $Workloads
$env:BVP_SCI_BENCH_CONTINUATION_WORKLOADS = $ContinuationWorkloads
$env:BVP_SCI_BENCH_CONTINUATION_NODES = $ContinuationNodes
$env:BVP_SCI_BENCH_CONTINUATION_COUNTS = $ContinuationCounts
$env:BVP_SCI_BENCH_CONTINUATION_MODES = $ContinuationModes
$env:BVP_SCI_BENCH_POLICY = $Policy
$env:BVP_SCI_BENCH_POLICY_MIN_WORK = [Math]::Max(0, $PolicyMinWork).ToString()
$env:BVP_SCI_BENCH_POLICY_FULL_SOLVE_NODES = $PolicyFullSolveNodes
$env:BVP_SCI_BENCH_POLICY_FULL_SOLVE_WORKLOADS = $PolicyFullSolveWorkloads
$env:BVP_SCI_BENCH_POLICY_FULL_SOLVE_LAYOUTS = $PolicyFullSolveLayouts
$env:BVP_SCI_BENCH_POLICY_FULL_SOLVE_FRONTENDS = $PolicyFullSolveFrontends
$env:BVP_SCI_BENCH_POLICY_FULL_SOLVE_REPEATS = [Math]::Max(1, $PolicyFullSolveRepeats).ToString()
$env:BVP_SCI_AOT_BENCH_DIMENSIONS = $AotDimensions
$env:BVP_SCI_AOT_BENCH_FRONTENDS = $AotFrontends
$env:BVP_SCI_AOT_BENCH_LAYOUTS = $AotLayouts
$env:BVP_SCI_AOT_BENCH_POLICIES = $AotPolicies
$env:BVP_SCI_AOT_COMPILER = $AotCompiler
$env:BVP_SCI_AOT_FULL_SOLVE_DIMENSIONS = $AotFullSolveDimensions
$env:BVP_SCI_AOT_FULL_SOLVE_NODES = $AotFullSolveNodes
$env:BVP_SCI_AOT_POLICY_FULL_SOLVE_REPEATS = [Math]::Max(1, $AotPolicyFullSolveRepeats).ToString()
$env:BVP_SCI_AOT_CONTINUATION_COUNT = [Math]::Max(1, $AotContinuationCount).ToString()
$env:BVP_SCI_MATCHED_DIMENSIONS = $MatchedDimensions
$env:BVP_SCI_MATCHED_NODES = $MatchedNodes
$env:BVP_SCI_MATCHED_COUNTS = $MatchedCounts
$env:BVP_SCI_MATCHED_REPEATS = [Math]::Max(1, $MatchedRepeats).ToString()
$env:BVP_SCI_MATCHED_COMPILER = $AotCompiler
$env:BVP_SCI_MATCHED_FAIL_ON_ERROR = "1"

$steps = New-Object System.Collections.Generic.List[object]

function Add-BvpSciStep {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Kind,
        [Parameter(Mandatory = $true)][scriptblock]$Action
    )

    $technicalLog = Join-Path $technicalRoot "$Name.log"
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    $watchStartUtc = [DateTime]::UtcNow
    $status = "passed"
    $exitCode = 0
    $errorText = ""
    $rowFailureReports = @()

    if ($PlanOnly) {
        $status = "planned"
        $technicalLog = "-"
    }
    else {
        try {
            & $Action 2>&1 |
                Tee-Object -FilePath $technicalLog |
                ForEach-Object {
                    $line = $_.ToString()
                    if ($line -match '^test result:' -or $line -match '^\[BVP_sci') {
                        Write-Host $line
                    }
                }
            if ($null -ne $LASTEXITCODE) {
                $exitCode = [int]$LASTEXITCODE
            }
            if ($exitCode -ne 0) {
                $status = "failed"
            }

            # Criterion and the compact story reports can complete their
            # process successfully while recording failed rows in Markdown.
            # Treat those rows as evidence failures, but keep executing later
            # independent steps so an overnight matrix remains useful.
            $rowFailureReports = @(Get-ChildItem -Path $reportRoot -Recurse -File -Filter '*.md' -ErrorAction SilentlyContinue |
                Where-Object { $_.FullName -notmatch '[\\/]archive[\\/]' } |
                Where-Object { $_.LastWriteTimeUtc -ge $watchStartUtc } |
                Where-Object {
                    $hasErrorRow = Select-String -LiteralPath $_.FullName -Pattern '\|\s*error:' -Quiet -ErrorAction SilentlyContinue
                    $hasFailedStatus = Select-String -LiteralPath $_.FullName -Pattern '^status:\s*failed\s*$' -Quiet -ErrorAction SilentlyContinue
                    $hasErrorRow -or $hasFailedStatus
                } |
                ForEach-Object { $_.FullName })
            if ($rowFailureReports.Count -gt 0) {
                $status = "failed"
                $rowError = "row-level failures in: " + (($rowFailureReports | ForEach-Object {
                    $_.Replace((Get-Location).Path + '\', '').Replace('\', '/')
                }) -join ', ')
                if ([string]::IsNullOrWhiteSpace($errorText)) {
                    $errorText = $rowError
                }
                else {
                    $errorText = "$errorText; $rowError"
                }
            }
        }
        catch {
            $status = "failed"
            $exitCode = 1
            $errorText = $_.Exception.Message
            Add-Content -LiteralPath $technicalLog -Value "`n[runner exception] $errorText"
        }
    }

    $watch.Stop()
    $steps.Add([pscustomobject][ordered]@{
        step = $Name
        kind = $Kind
        status = $status
        exit_code = $exitCode
        duration_s = [string]::Format(
            [Globalization.CultureInfo]::InvariantCulture,
            "{0:F1}",
            $watch.Elapsed.TotalSeconds
        )
        technical_log = $technicalLog
        error = $errorText
        row_failure_reports = $rowFailureReports.Count
    })
}

Add-BvpSciStep "stories_fast" "story" {
    cargo test --release --lib --no-default-features numerical::BVP_sci::new:: -- --nocapture --test-threads=1
}

if (-not $SkipIgnoredStories) {
    Add-BvpSciStep "stories_ignored" "story-ignored" {
        cargo test --release --lib --no-default-features numerical::BVP_sci::new:: -- --ignored --nocapture --test-threads=1
    }
    if (-not $SkipScaleEvidence) {
        Add-BvpSciStep "stories_banded_scale" "story-ignored" {
            $env:BVP_SCI_STORY_SCALE_NODES = $ScaleNodes
            $env:BVP_SCI_STORY_SCALE_WORKLOADS = $ScaleWorkloads
            cargo test --release --lib --no-default-features numerical::BVP_sci::new::story_tests::lambdify_layout_scale_matrix_reports_banded_route_costs -- --ignored --nocapture --test-threads=1
        }
    }
    if ($IncludeAot) {
        Add-BvpSciStep "stories_aot_contract" "story-ignored-aot" {
            cargo test --release --lib --no-default-features numerical::BVP_sci::new::story_tests::aot_frontend_backend_contract_matrix_is_compact_and_parity_checked -- --ignored --nocapture --test-threads=1
        }
        Add-BvpSciStep "stories_aot_solver" "story-ignored-aot" {
            cargo test --release --lib --no-default-features numerical::BVP_sci::new::story_tests::aot_solver_route_continuation_and_policy_matrix_is_compact -- --ignored --nocapture --test-threads=1
        }
        Add-BvpSciStep "stories_aot_full_solve" "story-ignored-aot" {
            cargo test --release --lib --no-default-features numerical::BVP_sci::new::story_tests::aot_full_solve_matches_lambdify_on_medium_parameterized_systems -- --ignored --nocapture --test-threads=1
        }
        Add-BvpSciStep "stories_aot_lifecycle" "story-ignored-aot" {
            cargo test --release --lib --no-default-features numerical::BVP_sci::new::story_tests::aot_lifecycle_policies_and_typed_failure_matrix_is_compact -- --ignored --nocapture --test-threads=1
        }
        Add-BvpSciStep "stories_aot_continuation_retention" "story-ignored-aot" {
            cargo test --release --lib --no-default-features numerical::BVP_sci::new::story_tests::aot_continuation_restart_retention_matrix_is_compact -- --ignored --nocapture --test-threads=1
        }
        Add-BvpSciStep "stories_aot_process_handoff" "story-ignored-aot-process" {
            cargo test --release --lib --no-default-features numerical::BVP_sci::new::story_tests::aot_process_isolated_producer_consumer_handoff_matrix_is_compact -- --ignored --nocapture --test-threads=1
        }
    }
}

if (-not $SkipBench -and $IncludeAot) {
    foreach ($aotPhase in ($AotPhases -split ',')) {
        $aotPhase = $aotPhase.Trim().ToLowerInvariant()
        if ([string]::IsNullOrWhiteSpace($aotPhase)) {
            continue
        }
        $env:BVP_SCI_AOT_BENCH_PHASE = $aotPhase
        $env:BVP_SCI_AOT_BENCH_REPORT = "aot_$aotPhase"
        Add-BvpSciStep "bench_aot_$aotPhase" "bench-aot-$aotPhase" {
            cargo bench --no-default-features --bench bvp_sci_aot -- --noplot
        }
    }

    if (-not $SkipPolicyFullSolveBench) {
        $previousRayonThreads = $env:RAYON_NUM_THREADS
        foreach ($workerText in ($AotPolicyFullSolveWorkers -split ',')) {
            $worker = $workerText.Trim()
            if ([string]::IsNullOrWhiteSpace($worker) -or [int]$worker -lt 1) {
                continue
            }
            # AOT chunk policy and Rayon worker count are process-lifetime
            # configuration. Use one child process per worker count so the
            # observed chunk counters and full-solve timings are honest.
            $env:RAYON_NUM_THREADS = $worker
            $env:BVP_SCI_AOT_POLICY_FULL_SOLVE_WORKER_COUNT = $worker
            $env:BVP_SCI_AOT_BENCH_PHASE = "policy-full-solve"
            $env:BVP_SCI_AOT_BENCH_REPORT = "aot_policy_full_solve_workers_$worker"
            Add-BvpSciStep "bench_aot_policy_full_solve_workers_$worker" "bench-aot-policy-full-solve" {
                cargo bench --no-default-features --bench bvp_sci_aot -- --noplot
            }
        }
        if ($null -eq $previousRayonThreads) {
            Remove-Item Env:RAYON_NUM_THREADS -ErrorAction SilentlyContinue
        }
        else {
            $env:RAYON_NUM_THREADS = $previousRayonThreads
        }
        Remove-Item Env:BVP_SCI_AOT_POLICY_FULL_SOLVE_WORKER_COUNT -ErrorAction SilentlyContinue
    }
}

if (-not $SkipBench) {
    if ($IncludeAot -and -not $SkipMatchedFrontendBench) {
        $env:BVP_SCI_MATCHED_REPORT = "matched_full_solve_continuation"
        Add-BvpSciStep "bench_matched_frontend_full_solve" "bench-matched-frontend" {
            cargo bench --no-default-features --bench bvp_sci_matched -- --noplot
        }
    }
    $env:BVP_SCI_BENCH_PHASE = "matrix"
    $env:BVP_SCI_BENCH_REPORT = "lambdify_matrix"
    Add-BvpSciStep "bench_lambdify_matrix" "bench-lambdify" {
        cargo bench --no-default-features --bench bvp_sci_lambdify -- --noplot
    }
}

if (-not $SkipBench -and -not $SkipContinuationBench) {
    $env:BVP_SCI_BENCH_PHASE = "continuation"
    $env:BVP_SCI_BENCH_REPORT = "lambdify_continuation"
    Add-BvpSciStep "bench_lambdify_continuation" "bench-lambdify-continuation" {
        cargo bench --no-default-features --bench bvp_sci_lambdify -- --noplot
    }
}

if (-not $SkipBench -and -not $SkipContinuationLifecycleBench) {
    $env:BVP_SCI_BENCH_PHASE = "continuation-lifecycle"
    $env:BVP_SCI_BENCH_REPORT = "lambdify_continuation_lifecycle"
    Add-BvpSciStep "bench_lambdify_continuation_lifecycle" "bench-lambdify-continuation-lifecycle" {
        cargo bench --no-default-features --bench bvp_sci_lambdify -- --noplot
    }
}

if (-not $SkipBench -and -not $SkipPolicyFullSolveBench) {
    $previousRayonThreads = $env:RAYON_NUM_THREADS
    foreach ($workerText in ($PolicyFullSolveWorkers -split ',')) {
        $worker = $workerText.Trim()
        if ([string]::IsNullOrWhiteSpace($worker) -or [int]$worker -lt 1) {
            continue
        }
        # Rayon reads this at process initialization. A separate child process
        # per worker count is therefore required for an honest matrix; changing
        # the variable inside one already-running bench cannot resize Rayon.
        $env:RAYON_NUM_THREADS = $worker
        $env:BVP_SCI_BENCH_POLICY_FULL_SOLVE_WORKER_COUNT = $worker
        $env:BVP_SCI_BENCH_PHASE = "policy-full-solve"
        $env:BVP_SCI_BENCH_REPORT = "lambdify_policy_full_solve_workers_$worker"
        Add-BvpSciStep "bench_lambdify_policy_full_solve_workers_$worker" "bench-lambdify-policy-full-solve" {
            cargo bench --no-default-features --bench bvp_sci_lambdify -- --noplot
        }
    }
    if ($null -eq $previousRayonThreads) {
        Remove-Item Env:RAYON_NUM_THREADS -ErrorAction SilentlyContinue
    }
    else {
        $env:RAYON_NUM_THREADS = $previousRayonThreads
    }
    Remove-Item Env:BVP_SCI_BENCH_POLICY_FULL_SOLVE_WORKER_COUNT -ErrorAction SilentlyContinue
}

if (-not $SkipBench -and -not $SkipScaleEvidence) {
    $env:BVP_SCI_BENCH_PHASE = "banded-scale"
    $env:BVP_SCI_BENCH_NODES = $ScaleNodes
    $env:BVP_SCI_BENCH_WORKLOADS = $ScaleWorkloads
    $env:BVP_SCI_BENCH_CONTINUATION = "1"
    $env:BVP_SCI_BENCH_REPORT = "lambdify_banded_scale"
    Add-BvpSciStep "bench_lambdify_banded_scale" "bench-lambdify-banded-scale" {
        cargo bench --no-default-features --bench bvp_sci_lambdify -- --noplot
    }
}

$passed = @($steps | Where-Object { $_.status -eq "passed" }).Count
$failed = @($steps | Where-Object { $_.status -eq "failed" }).Count
$planned = @($steps | Where-Object { $_.status -eq "planned" }).Count
$overall = if ($failed -eq 0 -and $planned -eq 0) { "completed" }
           elseif ($failed -gt 0) { "completed_with_failures" }
           else { "planned" }

$rows = @(
    "| step | kind | status | exit_code | duration_s | row_failures | technical_log | error |",
    "|---|---|---|---:|---:|---:|---|---|"
)
foreach ($step in $steps) {
    $log = $step.technical_log.Replace((Get-Location).Path + "\", "").Replace("\", "/")
    $errorCell = ($step.error -replace '\|', '/')
    $rows += "| $($step.step) | $($step.kind) | $($step.status) | $($step.exit_code) | $($step.duration_s) | $($step.row_failure_reports) | $log | $errorCell |"
}

$reports = @(Get-ChildItem -Path $reportRoot -Recurse -File -Filter '*.md' -ErrorAction SilentlyContinue |
    ForEach-Object { $_.FullName.Replace((Get-Location).Path + '\', '').Replace('\', '/') })
$summary = @(
    "# BVP_sci Lambdify Release Summary",
    "",
    "- status: $overall",
    "- profile: release",
    "- recorded_at_utc: $((Get-Date).ToUniversalTime().ToString('o'))",
    "- nodes: $Nodes",
    "- continuation: $Continuation",
    "- workloads: $Workloads",
    "- continuation_workloads: $ContinuationWorkloads",
    "- continuation_nodes: $ContinuationNodes",
    "- continuation_counts: $ContinuationCounts",
    "- continuation_modes: $ContinuationModes",
    "- scale_nodes: $ScaleNodes",
    "- policy_full_solve_workers: $PolicyFullSolveWorkers",
    "- policy_full_solve_nodes: $PolicyFullSolveNodes",
    "- policy_full_solve_workloads: $PolicyFullSolveWorkloads",
    "- policy_full_solve_layouts: $PolicyFullSolveLayouts",
    "- policy_full_solve_frontends: $PolicyFullSolveFrontends",
    "- policy_full_solve_repeats: $PolicyFullSolveRepeats",
    "- scale_workloads: $ScaleWorkloads",
    "- execution_policy: $Policy (min_work=$PolicyMinWork)",
    "- include_aot: $IncludeAot",
    "- aot_dimensions: $AotDimensions",
    "- aot_full_solve_dimensions: $AotFullSolveDimensions",
    "- aot_full_solve_nodes: $AotFullSolveNodes",
    "- aot_policy_full_solve_workers: $AotPolicyFullSolveWorkers",
    "- aot_policy_full_solve_repeats: $AotPolicyFullSolveRepeats",
    "- aot_continuation_count: $AotContinuationCount",
    "- aot_frontends: $AotFrontends",
    "- aot_layouts: $AotLayouts",
    "- aot_policies: $AotPolicies",
    "- aot_phases: $AotPhases",
    "- aot_compiler: $AotCompiler",
    "- matched_dimensions: $MatchedDimensions",
    "- matched_nodes: $MatchedNodes",
    "- matched_counts: $MatchedCounts",
    "- matched_repeats: $MatchedRepeats",
    "- passed_steps: $passed",
    "- failed_steps: $failed",
    "- planned_steps: $planned",
    "",
    "## Steps",
    "",
    ($rows -join "`n"),
    "",
    "## Compact Reports",
    "",
    "Measured tables are under reports/; compiler and Criterion output is under technical/.",
    $(if ($reports.Count -gt 0) { $reports -join "`n" } else { "- none" })
)
Set-Content -LiteralPath $summaryPath -Value ($summary -join "`n") -Encoding UTF8

Write-Output "[BVP_sci release] status=$overall passed=$passed failed=$failed reports=$reportRoot"
Write-Output "[BVP_sci release] summary=$summaryPath"

if ($FailOnAny -and $failed -gt 0) {
    exit 1
}
exit 0

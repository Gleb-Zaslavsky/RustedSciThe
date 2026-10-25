# Adapter for the runner's generated summary/measurement Markdown protocol.
# It never reads research STORY documents or infers a verdict from a timing.
# Keep format assumptions here and fail visibly if the native protocol changes.
function Get-IvpGateLaunchFailure {
    param([string]$LogPath)
    $lines = @(Get-Content -LiteralPath $LogPath)
    # Preflight failures happen before the native summary is created. Preserve
    # the native diagnostic; Cargo warnings and the wrapper's generic throw
    # are not the cause. This text is explanatory, never a verdict classifier.
    $diagnostics = @($lines | Where-Object { $_ -match '^(?:.* : )?perf-gates: ' })
    if ($diagnostics.Count -eq 0) {
        $diagnostics = @($lines | Where-Object { $_ -match '^error(?:\[[^\]]+\])?: ' })
    }
    if ($diagnostics.Count) {
        return 'No native run report was emitted. ' + (($diagnostics | Select-Object -First 5) -join ' ')
    }
    return 'No native run report was emitted. See technical log.'
}

function Read-IvpGateReport {
    param([string]$Path, [string]$Solver, [string]$Action, [string]$Root, [string]$Log)
    $lines = @(Get-Content -LiteralPath (Join-Path $Path "summary.md"))
    $selected = @($lines | Where-Object { $_ -match '^- selected scenarios: ' })
    $outcome = @($lines | Where-Object { $_ -match '^- outcome: `(ok|failed)`$' })
    if ($selected.Count -ne 1 -or $outcome.Count -ne 1 -or
        $lines[0] -ne "# Performance $Action run") {
        throw "Unrecognized native summary: $Path"
    }
    $ids = @($selected[0].Substring('- selected scenarios: '.Length).Split(',') | ForEach-Object { $_.Trim() })
    if ($ids.Count -eq 0 -or @($ids | Where-Object { $_ -notmatch '^[a-zA-Z0-9_]+$' }).Count -or
        @($ids | Select-Object -Unique).Count -ne $ids.Count) {
        throw "Invalid or duplicated scenario IDs in $Path"
    }
    $errors = @{}
    foreach ($line in $lines) {
        if ($line -notmatch '^- exit `(\d+)`: (.*)$') { continue }
        $code = [int]$Matches[1]
        $detail = $Matches[2]
        $matched = @($ids | Where-Object {
            $detail.StartsWith($_ + ':') -or $detail.StartsWith($_ + ' confirmation:') -or
            $detail.StartsWith('scenario ' + $_ + ' requires ')
        })
        if ($matched.Count -ne 1 -or $errors.ContainsKey($matched[0]) -or $code -eq 0) {
            throw "Unmapped or duplicate native error in ${Path}: $detail"
        }
        $errors[$matched[0]] = [pscustomobject]@{ code = $code; detail = $detail }
    }
    if (($errors.Count -eq 0) -ne ($outcome[0] -eq '- outcome: `ok`')) {
        throw "Native outcome and error list disagree: $Path"
    }
    $relative = "reports/$Solver/$Action/$(Split-Path -Leaf $Path)"
    $destination = Join-Path $Root $relative
    New-Item -ItemType Directory -Path $destination -Force | Out-Null
    # Archive only this invocation's reported evidence, never the latest directory
    # from another process. Original reports and baselines remain untouched.
    Get-ChildItem -LiteralPath $Path -File | Where-Object { $_.Extension -in @('.md', '.toml') } |
        Copy-Item -Destination $destination
    foreach ($id in $ids) {
        $errorInfo = $errors[$id]
        $status = "passed"
        $detail = ""
        $code = 0
        if ($null -ne $errorInfo) {
            $code = $errorInfo.code
            $detail = $errorInfo.detail
            $status = if ($code -eq 5) { "unstable" }
                elseif ($code -eq 1 -and $Action -eq 'compare') { "regression" }
                else { "error" }
        }
        $measurePath = Join-Path $destination "$id.md"
        $metrics = @()
        if (Test-Path -LiteralPath $measurePath) {
            # Row positions are part of append_metrics() in performance/runner/cli.rs.
            # The first wall row is primary; the second, when present, is confirmation.
            $metrics = @(Get-Content -LiteralPath $measurePath | ForEach-Object {
                if ($_ -match '^\| scenario_wall_ms \(ms(?:/off)?\) \|') {
                    $cells = $_.Split('|')
                    if ($cells.Count -ne 11) { throw "Unexpected measurement table: $measurePath" }
                    [pscustomobject]@{
                        samples = [int]$cells[5].Trim()
                        median = [double]::Parse($cells[6].Trim(), [Globalization.CultureInfo]::InvariantCulture)
                        mad = [double]::Parse($cells[7].Trim(), [Globalization.CultureInfo]::InvariantCulture)
                    }
                }
            })
        }
        if ($status -eq 'passed' -and $metrics.Count -eq 0) {
            $status = 'error'
            $detail = 'Native success has no readable wall measurement; inspect the archived report.'
        }
        [pscustomobject][ordered]@{
            solver = $Solver; action = $Action; scenario = $id; status = $status; exit_code = $code
            samples = $(if ($metrics.Count) { $metrics[0].samples } else { $null })
            wall_ms = $(if ($metrics.Count) { $metrics[0].median } else { $null })
            mad_ms = $(if ($metrics.Count) { $metrics[0].mad } else { $null })
            confirmation_wall_ms = $(if ($metrics.Count -gt 1) { $metrics[1].median } else { $null })
            confirmation_mad_ms = $(if ($metrics.Count -gt 1) { $metrics[1].mad } else { $null })
            detail = $detail; native_report = $Path
            report = "$relative/summary.md"; measurements = $(if (Test-Path -LiteralPath $measurePath) { "$relative/$id.md" } else { '' })
            log = $Log
        }
    }
}

function ConvertTo-IvpGateCell($Value) {
    if ($null -eq $Value) { return '-' }
    if ($Value -is [double]) { return $Value.ToString('0.0000', [Globalization.CultureInfo]::InvariantCulture) }
    return $Value.ToString().Replace('|', '\|').Replace("`r", ' ').Replace("`n", ' ')
}

function Write-IvpGateSummary {
    param([string]$Root, $Settings, $Steps, $Scenarios, [string]$Outcome)
    $stepArray = $Steps.ToArray()
    $scenarioArray = $Scenarios.ToArray()
    $counts = [ordered]@{}
    foreach ($status in @('passed', 'unstable', 'regression', 'error')) {
        $counts[$status] = @($scenarioArray | Where-Object { $_.status -eq $status }).Count
    }
    $document = [ordered]@{
        schema_version = 1; status = $Outcome; updated_at_utc = [DateTime]::UtcNow.ToString('o')
        settings = $Settings; counts = $counts; steps = $stepArray; scenarios = $scenarioArray
    }
    $document | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $Root 'summary.json') -Encoding UTF8
    if ($scenarioArray.Count) {
        $scenarioArray | Export-Csv -LiteralPath (Join-Path $Root 'scenarios.csv') -NoTypeInformation -Encoding UTF8
    }
    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add('# IVP Performance Gate Matrix')
    $lines.Add('')
    $lines.Add("- Status: $Outcome")
    $lines.Add("- Machine: $($Settings.machine); action: $($Settings.action); default warmups/runs: $($Settings.warmups)/$($Settings.runs); Radau: $($Settings.radau_warmups)/$($Settings.radau_runs)")
    $lines.Add("- Relative MAD limit: $($Settings.max_relative_mad); record tolerance: $($Settings.tolerance)")
    $lines.Add("- Scenario outcomes: $($counts.passed) passed, $($counts.unstable) unstable, $($counts.regression) regression, $($counts.error) error")
    $stepErrors = @($stepArray | Where-Object status -eq 'error').Count
    $blocked = @($stepArray | Where-Object status -eq 'blocked').Count
    $lines.Add("- Step errors: $stepErrors; blocked steps: $blocked (failures before measurement have no scenario outcomes)")
    $lines.Add('')
    $lines.Add('Pending/running steps mean the campaign has not finished. UNSTABLE is not a pass. No automatic retry selects a more favorable result.')
    $lines.Add('Native reports contain source/toolchain identity, counters, stage timings and raw samples. Technical logs retain baseline comparisons and confirmation output.')
    $lines.Add('')
    $lines.Add('| Solver | Action | Warmups | Runs | Status | Exit | Seconds | Log | Detail |')
    $lines.Add('| --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- |')
    foreach ($step in $stepArray) {
        $link = if ($step.log) { "[log]($($step.log))" } else { '-' }
        $lines.Add("| $($step.solver) | $($step.action) | $(ConvertTo-IvpGateCell $step.warmups) | $(ConvertTo-IvpGateCell $step.runs) | $($step.status) | $(ConvertTo-IvpGateCell $step.exit_code) | $($step.duration_s) | $link | $(ConvertTo-IvpGateCell $step.detail) |")
    }
    $lines.Add('')
    $lines.Add('| Solver | Action | Scenario | Status | N | Wall ms | MAD ms | Confirmation ms | Confirmation MAD | Evidence | Detail |')
    $lines.Add('| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |')
    foreach ($row in $scenarioArray) {
        $evidence = "[summary]($($row.report))"
        if ($row.measurements) { $evidence += " [measurements]($($row.measurements))" }
        $lines.Add("| $($row.solver) | $($row.action) | $($row.scenario) | $($row.status) | $(ConvertTo-IvpGateCell $row.samples) | $(ConvertTo-IvpGateCell $row.wall_ms) | $(ConvertTo-IvpGateCell $row.mad_ms) | $(ConvertTo-IvpGateCell $row.confirmation_wall_ms) | $(ConvertTo-IvpGateCell $row.confirmation_mad_ms) | $evidence | $(ConvertTo-IvpGateCell $row.detail) |")
    }
    $lines | Set-Content -LiteralPath (Join-Path $Root 'summary.md') -Encoding UTF8
}

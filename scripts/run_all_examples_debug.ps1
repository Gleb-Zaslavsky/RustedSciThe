param(
    [ValidateRange(1, 86400)][int]$TimeoutSeconds = 600,
    [string[]]$OnlyExample = @(),
    [string[]]$ExcludeExample = @(),
    [string]$ReportDirectory
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$cargoCommand = Get-Command cargo.exe -CommandType Application -ErrorAction SilentlyContinue
if ($null -eq $cargoCommand) {
    $cargoCommand = Get-Command cargo -CommandType Application -ErrorAction Stop
}

if ([string]::IsNullOrWhiteSpace($ReportDirectory)) {
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
    $ReportDirectory = Join-Path $repoRoot "test_reports\examples_debug\$stamp"
}
$reportRoot = [System.IO.Path]::GetFullPath($ReportDirectory)
$logsRoot = Join-Path $reportRoot "logs"
New-Item -ItemType Directory -Force -Path $logsRoot | Out-Null

$metadataOutput = & $cargoCommand.Source metadata --no-deps --format-version 1 --quiet 2>&1
if ($LASTEXITCODE -ne 0) {
    $metadataOutput | ForEach-Object { Write-Error $_ }
    throw "cargo metadata failed with exit code $LASTEXITCODE"
}
try {
    $metadata = ($metadataOutput -join [Environment]::NewLine) | ConvertFrom-Json
}
catch {
    throw "Could not parse cargo metadata: $($_.Exception.Message)"
}

$package = @($metadata.packages | Where-Object { $_.name -eq "RustedSciThe" }) | Select-Object -First 1
if ($null -eq $package) { throw "RustedSciThe package was not found in cargo metadata." }
$examples = @($package.targets | Where-Object { $_.kind -contains "example" } | Sort-Object name)
if ($examples.Count -eq 0) { throw "Cargo metadata registered no examples." }

$knownNames = @{};
foreach ($target in $examples) { $knownNames[$target.name] = $true }
foreach ($name in $OnlyExample) {
    if (-not $knownNames.ContainsKey($name)) { throw "Unknown Cargo example '$name'." }
}
foreach ($name in $ExcludeExample) {
    if (-not $knownNames.ContainsKey($name)) { throw "Unknown Cargo example in -ExcludeExample: '$name'." }
}
$excluded = @{}
foreach ($name in $ExcludeExample) { $excluded[$name] = $true }
$selected = @($examples | Where-Object {
    ($OnlyExample.Count -eq 0 -or $OnlyExample -contains $_.name) -and -not $excluded.ContainsKey($_.name)
})
if ($selected.Count -eq 0) { throw "No examples remain after applying -ExcludeExample." }

$results = [System.Collections.Generic.List[object]]::new()
Write-Host "Running $($selected.Count) Cargo examples in Debug mode; timeout=${TimeoutSeconds}s each."
Write-Host "Report: $reportRoot"

foreach ($example in $selected) {
    $stdoutLog = Join-Path $logsRoot "$($example.name).stdout.log"
    $stderrLog = Join-Path $logsRoot "$($example.name).stderr.log"
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    $status = "failed"
    $exitCode = 1

    Write-Host "[START] $($example.name)"
    try {
        $process = Start-Process `
            -FilePath $cargoCommand.Source `
            -ArgumentList @("run", "--example", $example.name) `
            -WorkingDirectory $repoRoot `
            -RedirectStandardOutput $stdoutLog `
            -RedirectStandardError $stderrLog `
            -NoNewWindow `
            -PassThru

        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            & taskkill.exe /PID $process.Id /T /F *> $null
            $process.WaitForExit()
            $status = "timeout"
            $exitCode = -1
        }
        else {
            $process.Refresh()
            $exitCode = [int]$process.ExitCode
            $status = if ($exitCode -eq 0) { "passed" } else { "failed" }
        }
    }
    catch {
        $status = "runner_error"
        $exitCode = 1
        Add-Content -LiteralPath $stderrLog -Value ("[runner exception] " + $_.Exception.Message)
    }
    finally {
        $watch.Stop()
    }

    $duration = [math]::Round($watch.Elapsed.TotalSeconds, 1)
    $results.Add([pscustomobject][ordered]@{
        example = $example.name
        status = $status
        exit_code = $exitCode
        duration_s = $duration
        stdout_log = "logs/$($example.name).stdout.log"
        stderr_log = "logs/$($example.name).stderr.log"
    })
    Write-Host ("[{0}] {1} exit={2} duration={3}s" -f $status.ToUpperInvariant(), $example.name, $exitCode, $duration)
}

$passed = @($results | Where-Object status -eq "passed").Count
$failed = $results.Count - $passed
$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add("# Debug Example Run")
$lines.Add("")
$lines.Add("- Date (UTC): $((Get-Date).ToUniversalTime().ToString('yyyy-MM-dd HH:mm:ss'))")
$lines.Add("- Profile: Debug (no ``--release``)")
$lines.Add("- Examples discovered: $($examples.Count)")
$lines.Add("- Examples selected: $($selected.Count)")
$lines.Add("- Timeout per example: $TimeoutSeconds seconds")
$lines.Add("- Summary: $passed passed, $failed failed/timed out")
$lines.Add("")
$lines.Add("| Example | Status | Exit | Seconds | Logs |")
$lines.Add("| --- | --- | ---: | ---: | --- |")
foreach ($result in $results) {
    $lines.Add("| ``$($result.example)`` | $($result.status) | $($result.exit_code) | $($result.duration_s) | [stdout]($($result.stdout_log)) / [stderr]($($result.stderr_log)) |")
}
$summaryPath = Join-Path $reportRoot "summary.md"
$lines | Set-Content -LiteralPath $summaryPath -Encoding utf8

Write-Host ""
Write-Host "Debug examples: $passed passed, $failed failed/timed out."
Write-Host "Summary: $summaryPath"
if ($failed -gt 0) { exit 1 }

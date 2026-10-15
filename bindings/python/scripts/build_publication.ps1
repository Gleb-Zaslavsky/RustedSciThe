[CmdletBinding()]
param(
    [string]$OutputDirectory = "",
    [switch]$SkipWheel
)

$ErrorActionPreference = "Stop"

function Invoke-NativeTool {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Label
    )

    & $Executable @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$Label failed with exit code $exitCode"
    }
}

function Invoke-NativeToolOutput {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $result = & $Executable @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$Label failed with exit code $exitCode"
    }
    return $result
}

$bindingRoot = (Resolve-Path (Join-Path $PSScriptRoot ".."))
$repoRoot = (Resolve-Path (Join-Path $bindingRoot "..\.."))
$python = if ($env:PYTHON) { $env:PYTHON } else {
    Join-Path $repoRoot ".venv-gavin-win\Scripts\python.exe"
}

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python executable was not found: $python"
}

if (-not $OutputDirectory) {
    $runId = (Get-Date -Format "yyyyMMdd-HHmmss") + "-" + [Guid]::NewGuid().ToString("N").Substring(0, 8)
    $OutputDirectory = Join-Path $bindingRoot ("target\publication-" + $runId)
}

$packageVersion = ((Get-Content (Join-Path $bindingRoot "pyproject.toml") |
        Select-String '^version\s*=\s*"([^"]+)"').Matches | Select-Object -First 1).Groups[1].Value
if (-not $packageVersion) {
    throw "Cannot determine the Python package version from pyproject.toml"
}

if (Test-Path -LiteralPath $OutputDirectory) {
    $existing = Get-ChildItem -LiteralPath $OutputDirectory -Force
    if ($existing.Count -gt 0) {
        throw "Output directory must be empty to prevent stale artifacts: $OutputDirectory"
    }
    $output = (Resolve-Path -LiteralPath $OutputDirectory).Path
} else {
    $output = (New-Item -ItemType Directory -Path $OutputDirectory).FullName
}
$rawDirectory = (New-Item -ItemType Directory -Path (Join-Path $output "raw-sdist")).FullName
$stageDirectory = (New-Item -ItemType Directory -Path (Join-Path $output "sdist-stage")).FullName

Push-Location $repoRoot
try {
    Invoke-NativeTool -Executable $python -Label "maturin sdist" -Arguments @(
        "-m", "maturin", "sdist",
        "--manifest-path", (Join-Path $bindingRoot "Cargo.toml"),
        "--out", $rawDirectory
    )
} finally {
    Pop-Location
}

$rawSdists = @(Get-ChildItem -LiteralPath $rawDirectory -Filter "rustedscithe-$packageVersion.tar.gz" -File)
if ($rawSdists.Count -ne 1) {
    throw "Expected exactly one rustedscithe-$packageVersion.tar.gz from maturin; found $($rawSdists.Count)"
}
$rawSdist = $rawSdists[0]

Invoke-NativeTool -Executable "tar" -Label "sdist extraction" -Arguments @(
    "-xzf", $rawSdist.FullName, "-C", $stageDirectory
)
$archiveRoots = @(Get-ChildItem -LiteralPath $stageDirectory -Directory)
if ($archiveRoots.Count -ne 1) {
    throw "Expected exactly one extracted sdist root; found $($archiveRoots.Count)"
}
$archiveRoot = $archiveRoots[0]

$pythonReadme = Join-Path $archiveRoot.FullName "bindings\python\README.md"
$archiveReadme = Join-Path $archiveRoot.FullName "README.md"
if (-not (Test-Path -LiteralPath $pythonReadme)) {
    throw "Binding README is missing from the sdist: $pythonReadme"
}

# Maturin's path-dependent sdist is rooted at the core Cargo workspace. Replace
# only the archive root README so source-built wheel metadata describes Python.
Copy-Item -LiteralPath $pythonReadme -Destination $archiveReadme -Force

$fixedSdist = Join-Path $output $rawSdist.Name
Push-Location $stageDirectory
try {
    Invoke-NativeTool -Executable "tar" -Label "fixed sdist archive" -Arguments @(
        "-czf", $fixedSdist, $archiveRoot.Name
    )
} finally {
    Pop-Location
}

Invoke-NativeTool -Executable $python -Label "twine check sdist" -Arguments @(
    "-m", "twine", "check", $fixedSdist
)
$sdistReadme = (Invoke-NativeToolOutput -Executable "tar" -Label "sdist README verification" -Arguments @(
    "-xOf", $fixedSdist, "$($archiveRoot.Name)/README.md"
)) -join "`n"
if ($sdistReadme -notmatch "# RustedSciThe for Python") {
    throw "The fixed sdist root README does not contain the Python package description"
}

if (-not $SkipWheel) {
    $wheelDirectory = (New-Item -ItemType Directory -Path (Join-Path $output "wheels-from-sdist")).FullName
    Invoke-NativeTool -Executable $python -Label "pip wheel from sdist" -Arguments @(
        "-m", "pip", "wheel", $fixedSdist, "--no-deps", "--no-cache-dir",
        "--wheel-dir", $wheelDirectory
    )
    $wheels = @(Get-ChildItem -LiteralPath $wheelDirectory -Filter "rustedscithe-$packageVersion-*.whl" -File)
    if ($wheels.Count -ne 1) {
        throw "Expected exactly one rustedscithe $packageVersion wheel from the sdist; found $($wheels.Count)"
    }
    $wheel = $wheels[0]

    $metadataPath = "rustedscithe-$packageVersion.dist-info/METADATA"
    $metadata = (Invoke-NativeToolOutput -Executable "tar" -Label "wheel metadata verification" -Arguments @(
        "-xOf", $wheel.FullName, $metadataPath
    )) -join "`n"
    if ($metadata -notmatch "# RustedSciThe for Python") {
        throw "The wheel metadata does not contain the Python README"
    }
    if ($metadata -match "# RustedSciThe\r?\nRustedSciThe is a Rust framework") {
        throw "The wheel metadata still contains the core README"
    }
    Invoke-NativeTool -Executable $python -Label "twine check wheel" -Arguments @(
        "-m", "twine", "check", $wheel.FullName
    )
    Write-Host "wheel=$($wheel.FullName)"
}

Write-Host "sdist=$fixedSdist"

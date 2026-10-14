[CmdletBinding()]
param(
    [string]$OutputDirectory = "",
    [switch]$SkipWheel
)

$ErrorActionPreference = "Stop"

$bindingRoot = (Resolve-Path (Join-Path $PSScriptRoot ".."))
$repoRoot = (Resolve-Path (Join-Path $bindingRoot "..\.."))
$python = if ($env:PYTHON) { $env:PYTHON } else {
    Join-Path $repoRoot ".venv-gavin-win\Scripts\python.exe"
}

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python executable was not found: $python"
}

if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $bindingRoot ("target\publication-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
}

$packageVersion = ((Get-Content (Join-Path $bindingRoot "pyproject.toml") |
        Select-String '^version\s*=\s*"([^"]+)"').Matches | Select-Object -First 1).Groups[1].Value
if (-not $packageVersion) {
    throw "Cannot determine the Python package version from pyproject.toml"
}

$output = (New-Item -ItemType Directory -Force -Path $OutputDirectory).FullName
$rawDirectory = (New-Item -ItemType Directory -Force -Path (Join-Path $output "raw-sdist")).FullName
$stageDirectory = (New-Item -ItemType Directory -Force -Path (Join-Path $output "sdist-stage")).FullName

Push-Location $repoRoot
try {
    & $python -m maturin sdist `
        --manifest-path (Join-Path $bindingRoot "Cargo.toml") `
        --out $rawDirectory
} finally {
    Pop-Location
}

$rawSdist = Get-ChildItem -LiteralPath $rawDirectory -Filter "*.tar.gz" | Select-Object -First 1
if (-not $rawSdist) {
    throw "maturin did not produce an sdist in $rawDirectory"
}

& tar -xzf $rawSdist.FullName -C $stageDirectory
$archiveRoot = Get-ChildItem -LiteralPath $stageDirectory -Directory | Select-Object -First 1
if (-not $archiveRoot) {
    throw "Cannot locate the extracted sdist root"
}

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
    & tar -czf $fixedSdist $archiveRoot.Name
} finally {
    Pop-Location
}

& $python -m twine check $fixedSdist

if (-not $SkipWheel) {
    $wheelDirectory = (New-Item -ItemType Directory -Force -Path (Join-Path $output "wheels-from-sdist")).FullName
    & $python -m pip wheel $fixedSdist --no-deps --no-cache-dir --wheel-dir $wheelDirectory
    $wheel = Get-ChildItem -LiteralPath $wheelDirectory -Filter "*.whl" | Select-Object -First 1
    if (-not $wheel) {
        throw "pip did not produce a wheel from the fixed sdist"
    }

    $metadataPath = "rustedscithe-$packageVersion.dist-info/METADATA"
    $metadata = (& tar -xOf $wheel.FullName $metadataPath) -join "`n"
    if ($metadata -notmatch "# RustedSciThe Python Package") {
        throw "The wheel metadata does not contain the Python README"
    }
    if ($metadata -match "# RustedSciThe\r?\nRustedSciThe is a Rust framework") {
        throw "The wheel metadata still contains the core README"
    }
    & $python -m twine check $wheel.FullName
    Write-Host "wheel=$($wheel.FullName)"
}

Write-Host "sdist=$fixedSdist"

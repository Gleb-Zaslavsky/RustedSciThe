# Backwards-compatible entry point; use build_superlu_bridge.ps1 for new setups.
param([string]$OutputDirectory = "test_reports/somelinalg_sparse_lu_reference/native")
$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "build_superlu_bridge.ps1") -OutputDirectory $OutputDirectory
if ($LASTEXITCODE -ne 0) { throw "SuperLU bridge build failed ($LASTEXITCODE)." }

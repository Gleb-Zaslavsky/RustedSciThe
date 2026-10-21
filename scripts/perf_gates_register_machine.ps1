param(
    [Parameter(Mandatory)][ValidatePattern('^[A-Za-z0-9_-]+$')][string]$Name,
    [string]$Toolchain
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$previousToolchain = $env:RUSTUP_TOOLCHAIN

try {
    if (-not [string]::IsNullOrWhiteSpace($Toolchain)) {
        $env:RUSTUP_TOOLCHAIN = $Toolchain
    }

    Push-Location $repoRoot
    try {
        Write-Host "Machine alias: $Name"
        if (-not [string]::IsNullOrWhiteSpace($Toolchain)) {
            Write-Host "Rustup toolchain: $Toolchain"
        }
        & rustc --version
        if ($LASTEXITCODE -ne 0) { throw "rustc is unavailable or failed." }
        & cargo --version
        if ($LASTEXITCODE -ne 0) { throw "cargo is unavailable or failed." }

        & cargo run --release --features perf-gates --bin perf-gates -- machine register --name $Name
        if ($LASTEXITCODE -ne 0) { throw "Machine profile registration failed." }

        Write-Host "Current target fingerprint and scenario coverage:"
        & cargo run --release --features perf-gates --bin perf-gates -- list
        if ($LASTEXITCODE -ne 0) { throw "Could not inspect the registered machine profile." }
    }
    finally {
        Pop-Location
    }
}
finally {
    if ($null -eq $previousToolchain) {
        Remove-Item Env:RUSTUP_TOOLCHAIN -ErrorAction SilentlyContinue
    }
    else {
        $env:RUSTUP_TOOLCHAIN = $previousToolchain
    }
}

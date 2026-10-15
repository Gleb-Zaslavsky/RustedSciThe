param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$consumerContract = Join-Path $projectRoot "tests\typing_consumer.py"

Push-Location $projectRoot
try {
    & $Python -m mypy --config-file pyproject.toml $consumerContract
    if ($LASTEXITCODE -ne 0) {
        throw "mypy failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

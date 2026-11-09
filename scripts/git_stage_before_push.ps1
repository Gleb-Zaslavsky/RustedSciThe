$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$gitCommand = Get-Command git.exe -CommandType Application -ErrorAction SilentlyContinue
if ($null -eq $gitCommand) {
    $gitCommand = Get-Command git -CommandType Application -ErrorAction Stop
}

$pathsToStage = @(
    "src/",
    "benches/",
    "bindings/",
    "performance/",
    "examples/",
    "scripts/",
    ".gitignore",
    "Cargo.toml",
    "Cargo.lock"
)

& $gitCommand.Source -C $repoRoot add -- @pathsToStage
if ($LASTEXITCODE -ne 0) {
    throw "git add failed with exit code $LASTEXITCODE."
}

Write-Host "Staged publication paths. Review the status below before committing:"
& $gitCommand.Source -C $repoRoot status --short
if ($LASTEXITCODE -ne 0) {
    throw "git status failed with exit code $LASTEXITCODE."
}

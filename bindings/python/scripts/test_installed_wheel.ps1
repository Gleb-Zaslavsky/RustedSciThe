param(
    [Parameter(Mandatory = $true)]
    [string] $WheelPath,

    [Parameter(Mandatory = $true)]
    [string] $PythonExe
)

$ErrorActionPreference = "Stop"
$wheel = (Resolve-Path -LiteralPath $WheelPath).Path
$python = (Resolve-Path -LiteralPath $PythonExe).Path
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$venv = Join-Path $env:TEMP ("rustedscithe-wheel-contract-" + [guid]::NewGuid().ToString("N"))
$venvPython = Join-Path $venv "Scripts\python.exe"
$smokeScript = Join-Path $venv "wheel_smoke.py"
$tests = Join-Path $repoRoot "bindings\python\tests\test_contract.py"
$pytestConfig = Join-Path $repoRoot "bindings\python\pyproject.toml"

& $python -m venv $venv
if ($LASTEXITCODE -ne 0) { throw "Could not create the clean wheel-test environment." }
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip bootstrap failed." }
& $venvPython -m pip install $wheel pytest
if ($LASTEXITCODE -ne 0) { throw "Could not install the wheel and contract-test dependency." }

$smoke = @'
import importlib.metadata as metadata
import pathlib
import sys
import numpy
import rustedscithe
import rustedscithe._native as native
prefix = pathlib.Path(sys.prefix).resolve()
package = pathlib.Path(rustedscithe.__file__).resolve()
assert prefix in package.parents, (prefix, package)
assert pathlib.Path(native.__file__).suffix.lower() in {".pyd", ".dll"}
print("version:", metadata.version("rustedscithe"))
print("package:", package)
print("native:", native.__file__)
print("NumPy:", numpy.__version__)
'@
$utf8WithoutBom = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText($smokeScript, $smoke, $utf8WithoutBom)

$oldPythonPath = $env:PYTHONPATH
try {
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    Push-Location $env:TEMP
    & $venvPython $smokeScript
    if ($LASTEXITCODE -ne 0) { throw "Installed-wheel import smoke failed." }
    & $venvPython -m pytest $tests -c $pytestConfig -q
    if ($LASTEXITCODE -ne 0) { throw "Installed-wheel contract suite failed." }
}
finally {
    Pop-Location
    if ($null -ne $oldPythonPath) { $env:PYTHONPATH = $oldPythonPath }
    else { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue }
}

Write-Host "wheel=$wheel"
Write-Host "python=$python"
Write-Host "venv=$venv"

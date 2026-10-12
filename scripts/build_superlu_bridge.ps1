param(
    [string]$OutputDirectory = "test_reports/somelinalg_sparse_lu_reference/native",
    [string]$SourceDirectory = "$env:LOCALAPPDATA/RustedSciThe/SuperLU-6.0.1",
    [string]$BuildDirectory = "$env:LOCALAPPDATA/RustedSciThe/SuperLU-6.0.1-build"
)

$ErrorActionPreference = "Stop"
$superluCommit = "d6a031ca56499eefa7c742d51fae5e8c644138cc"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$outputPath = Join-Path $repositoryRoot $OutputDirectory

function Resolve-Tool([string]$Name, [string[]]$Candidates = @()) {
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    foreach ($candidate in $Candidates) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    throw "Required build tool '$Name' was not found. Install it or add it to PATH."
}

$cmake = Resolve-Tool "cmake.exe" @(
    "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
)
$ninja = Resolve-Tool "ninja.exe" @(
    "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
)
$gcc = Resolve-Tool "gcc.exe" @("C:\msys64\ucrt64\bin\gcc.exe")

if (-not (Test-Path -LiteralPath $SourceDirectory)) {
    New-Item -ItemType Directory -Force -Path (Split-Path $SourceDirectory) | Out-Null
    git clone --depth 1 --branch v6.0.1 https://github.com/xiaoyeli/superlu.git $SourceDirectory
    if ($LASTEXITCODE -ne 0) { throw "Could not clone SuperLU 6.0.1." }
}
$actualCommit = (git -C $SourceDirectory rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $superluCommit) {
    throw "Expected SuperLU 6.0.1 commit $superluCommit, got '$actualCommit'."
}

& $cmake -S $SourceDirectory -B $BuildDirectory -G Ninja `
    "-DCMAKE_MAKE_PROGRAM=$ninja" `
    "-DCMAKE_C_COMPILER=$($gcc -replace '\\', '/')" `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON `
    -DBUILD_SHARED_LIBS=OFF `
    -Denable_internal_blaslib=ON `
    -Denable_single=OFF `
    -Denable_double=ON `
    -Denable_complex=OFF `
    -Denable_complex16=OFF `
    -Denable_fortran=OFF `
    -Denable_tests=OFF `
    -Denable_examples=OFF `
    -Denable_doc=OFF
if ($LASTEXITCODE -ne 0) { throw "SuperLU CMake configuration failed ($LASTEXITCODE)." }

& $cmake --build $BuildDirectory --config Release --parallel
if ($LASTEXITCODE -ne 0) { throw "SuperLU build failed ($LASTEXITCODE)." }

New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
$bridge = Join-Path $outputPath "superlu_bridge.dll"
$sourceHeaderDirectory = Join-Path $SourceDirectory "SRC"
$superluArchive = Join-Path $BuildDirectory "SRC/libsuperlu.a"
$blasArchive = Join-Path $BuildDirectory "CBLAS/libblas.a"
foreach ($archive in @($superluArchive, $blasArchive)) {
    if (-not (Test-Path -LiteralPath $archive)) { throw "Expected static archive not found: $archive" }
}

& $gcc -O3 -DNDEBUG -shared -static-libgcc "-Wl,--no-undefined" `
    "-I$sourceHeaderDirectory" `
    (Join-Path $repositoryRoot "src/somelinalg/sparse_lu_reference/superlu_bridge.c") `
    $superluArchive $blasArchive -lm -o $bridge
if ($LASTEXITCODE -ne 0) { throw "SuperLU C ABI bridge build failed ($LASTEXITCODE)." }

Write-Host "SuperLU 6.0.1 commit=$actualCommit"
Write-Host "target=Windows x86_64 MinGW-w64"
Write-Host "bridge=$bridge"
Write-Host "Set `$env:SUPERLU_LIBRARY to this path before selecting BvpSciSparseFactorization::SuperLu."

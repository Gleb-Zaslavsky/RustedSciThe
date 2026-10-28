param()

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

Push-Location $repoRoot
try {
    $tests = @(
        @{
            Name = "BVP_sci native parameter-loop success/failure recovery"
            Filter = "external_parameter_loop"
        },
        @{
            Name = "BVP_api BVP_sci parameter-loop forwarding"
            Filter = "bvp_sci_facade_parameter_loop_prepares_fixed_inputs_once_and_keeps_route_report"
        }
    )

    foreach ($test in $tests) {
        Write-Host "[BVP_sci parameter-loop Release] $($test.Name)"
        & cargo test --release --lib $test.Filter -- --nocapture --test-threads=1
        if ($LASTEXITCODE -ne 0) {
            throw "Release test failed: $($test.Name) (exit code $LASTEXITCODE)"
        }
    }
}
finally {
    Pop-Location
}

Write-Host "[BVP_sci parameter-loop Release] all focused tests passed"

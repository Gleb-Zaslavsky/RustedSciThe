# RustedSciThe Python Package Publication Guide

This document is the release runbook for the separate `rustedscithe` Python
distribution. The binding crate is published as a Python wheel and sdist; it
is not published as part of the core Rust crate. Commands below are intended
to be run from the repository root unless stated otherwise.

## Windows

### Prerequisites

Install or verify:

- CPython 3.10 or newer, with `pip` and `venv`;
- Rust and Cargo on `PATH`;
- `maturin` for building and `twine` for metadata validation;
- NumPy for runtime smoke tests.

The base wheel does not require TCC, GCC, Zig or SuperLU. Those tools are
optional and are needed only for explicitly selected AOT or native backend
routes. A missing optional tool must produce the documented typed diagnostic
or fallback, not make the base wheel uninstallable.

Create or select a build environment:

```powershell
$env:PYTHON = (Resolve-Path .\.venv-gavin-win\Scripts\python.exe).Path
$env:MATURIN = (Resolve-Path .\.venv-gavin-win\Scripts\maturin.exe).Path
& $env:PYTHON -m pip install maturin twine numpy
```

For Rust unit tests which link the PyO3 extension directly, a virtual
environment is not enough on Windows. The Python development import library
must be installed and discoverable. Set `PYO3_PYTHON` to the selected CPython
and verify the corresponding `python*.lib`/`python3.lib` path before running
Rust tests. Wheel builds use the extension-module build path and are separate
from this linker requirement.

### Build and audit

Use a fresh output directory for each release candidate:

```powershell
$ErrorActionPreference = "Stop"
$dist = ".\bindings\python\target\publication-20261015"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\bindings\python\scripts\build_publication.ps1 `
  -OutputDirectory $dist
```

The helper builds the sdist, replaces the path-dependent archive root README
with `bindings/python/README.md`, rebuilds the wheel from that corrected sdist
and verifies its metadata. This step is required for the current local
path-dependency layout; uploading a wheel built from an uncorrected sdist can
expose the core Rust README on PyPI.

The release candidate is acceptable only when:

- the wheel has the expected `cp310-abi3-win_amd64` tag;
- the sdist contains source and build metadata but no local venv, `target`,
  `test_reports`, generated AOT output, DLL, log or reference environment;
- `twine check` passes for every artifact;
- the wheel can be rebuilt directly from the sdist.

The repository Cargo allowlist and the archive audit are part of this gate.
Do not replace the allowlist with a broad `src/**/*` pattern: it can pull
local reference environments into the sdist.

### Clean installation and smoke test

Run the wheel outside the checkout. Use an ordinary internet-enabled clean
environment so that the declared NumPy dependency is tested too:

```powershell
$smoke = Join-Path $env:TEMP "rustedscithe-wheel-smoke"
py -3.14 -m venv $smoke
& "$smoke\Scripts\python.exe" -m pip install --upgrade pip
& "$smoke\Scripts\python.exe" -m pip install (Get-ChildItem "$dist\*.whl").FullName
& "$smoke\Scripts\python.exe" -c `
  "import numpy, rustedscithe; from rustedscithe import Expr; print(rustedscithe.__file__); print(numpy.__version__); print(Expr.parse('x^2 + 1'))"
```

Then run the Python contract suite against the installed wheel. Ensure the
test process cannot import `bindings/python/python` from the checkout, for
example by running from `$env:TEMP` and setting `PYTHONPATH` explicitly only
when the test harness requires it.

### Windows status recorded on 2026-10-15

- [x] Windows `cp310-abi3-win_amd64` wheel built with maturin.
- [x] sdist built and audited: no forbidden local/build/report artifacts.
- [x] wheel rebuilt directly from the sdist.
- [x] `twine check` passed for the sdist and wheel.
- [x] wheel installed outside the checkout in a clean temporary venv.
- [x] installed-wheel NumPy and `Expr` import smoke test passed.
- [x] Python contract suite passed: `87 passed`.
- [ ] Rust unit-test link gate with the installed Python development library.
- [x] Initial Windows TestPyPI `0.1.0` upload and clean install passed; it is
  staging-only because its PyPI description used the core README.
- [x] Corrected Windows `0.1.1` artifacts uploaded to TestPyPI and clean
  installed successfully with the Python-specific README.
- [x] Windows-first `0.1.1` sdist and wheel uploaded to production PyPI and
  clean installed from the public index.
- [x] Windows `0.1.2` release wheel built with Maturin's explicit `release`
  profile; editable builds remain on `dev`.
- [x] `0.1.2` release wheel installed and smoke-tested in a clean CPython 3.14
  environment; production PyPI hashes match the locally verified artifacts.

### Windows release 0.1.2

Release evidence (2026-10-15): the sdist was rebuilt and the Windows
`cp310-abi3-win_amd64` wheel was rebuilt from that sdist. The wheel build log
confirmed Maturin's `release` profile from `pyproject.toml`; editable builds
use `dev`. Both artifacts passed `twine check`. The exact release wheel was
installed in a clean CPython 3.14 / NumPy 2.5.4 environment outside the
checkout and passed the `Expr` import/evaluation smoke test. After upload, the
PyPI JSON hashes were checked against these local files and a fresh venv
successfully imported both `rustedscithe` and its native `_native.pyd` module.
An earlier pre-release `0.1.2` candidate had a full Python contract result of
`99 passed, 2 skipped`; its artifacts were superseded and were not uploaded to
production. The README metadata check confirms the Python README, not the core
Rust README, is embedded. The build used core version `0.4.24`.

Published artifact hashes:

| Artifact | SHA-256 |
| --- | --- |
| `rustedscithe-0.1.2.tar.gz` | `3db8d959ffa1b786ffc7e93cf548ca142bcf66786daa4123121c28de3e657fc` |
| `rustedscithe-0.1.2-cp310-abi3-win_amd64.whl` | `5cdf42657c253ab347be4726d43f3b4c4ad539b6b4f521bd6ff5ce7b7d1859d7` |

Recorded Windows toolchain for this evidence: Rust/Cargo `1.98.1`, Python
`3.14.6`, NumPy `2.5.4`, PyO3 `0.27.2`, maturin `1.15.0`, twine `7.0.0`,
binding/core version `0.1.1`/`0.4.23`.
The published Windows `0.1.1` artifact hashes are recorded in the binding
TODO and should be retained with the release report.

## Linux

### Prerequisites

Use a supported CPython development installation, not only a runtime venv.
For Debian-like systems the starting point is:

```bash
sudo apt-get update
sudo apt-get install -y build-essential pkg-config python3-dev python3-venv
python3 -m venv .venv-publication-linux
. .venv-publication-linux/bin/activate
python -m pip install --upgrade pip maturin twine numpy
```

Verify that `python3-config --includes` and `python3-config --ldflags` point to
the same interpreter used by `PYO3_PYTHON`. Record the Rust, Cargo, Python,
PyO3 and maturin versions in the release report.

### Build and audit

```bash
set -euo pipefail
export PYO3_PYTHON="$(command -v python3)"
dist="$(pwd)/bindings/python/target/publication-linux"
mkdir -p "$dist"

maturin build \
  --manifest-path bindings/python/Cargo.toml \
  --release \
  --out "$dist"
maturin sdist \
  --manifest-path bindings/python/Cargo.toml \
  --out "$dist"
python -m twine check "$dist"/*
```

For PyPI publication, Linux wheels must be built in a documented
manylinux-compatible environment or by the selected CI toolchain. A local
Linux wheel is evidence for that platform only; it is not automatically a
manylinux wheel.

### Clean installation and smoke test

```bash
smoke="$(mktemp -d)"
python3 -m venv "$smoke"
"$smoke/bin/python" -m pip install --upgrade pip
"$smoke/bin/python" -m pip install "$dist"/*.whl
"$smoke/bin/python" -c \
  "import numpy, rustedscithe; from rustedscithe import Expr; print(rustedscithe.__file__); print(numpy.__version__); print(Expr.parse('x^2 + 1'))"
```

The Linux gate is complete only after the installed-wheel contract suite,
typed-error matrix and representative solver smoke tests pass in the clean
environment. A platform-specific failure must be recorded with its compiler,
Python and wheel tags rather than hidden by a successful Windows run.

### Linux status

- [ ] Build the supported manylinux-compatible wheel matrix.
- [ ] Build and audit the Linux sdist.
- [ ] Run `twine check` on Linux artifacts.
- [ ] Install and test each wheel in a clean Linux venv.
- [ ] Run the Python contract, continuation, telemetry and typed-error gates.
- [ ] Upload and validate the Linux artifacts on TestPyPI.

## TestPyPI and production handoff

For the current Windows-first release, upload the Windows wheel and matching
sdist only after the local gates above pass. Linux wheels are not part of this
release. The sdist is still a source-build artifact; Linux source installation
is not advertised as validated until the Linux section is completed.

Create separate accounts/tokens for TestPyPI and PyPI. For a new project, the
first token generally needs account-level scope because the project does not
exist yet. Use the PyPI username `__token__` and paste the token, including
the `pypi-` prefix, when Twine prompts. Never commit a token or put it into a
tracked file.

### Windows upload commands

Run from the repository root after replacing only the Python executable path
if necessary:

```powershell
$ErrorActionPreference = "Stop"
$python = (Resolve-Path .\.venv-gavin-win\Scripts\python.exe).Path
$dist = (Resolve-Path .\bindings\python\target\publication-0.1.2-release-fixed).Path
$sdist = (Get-ChildItem $dist -Filter "*.tar.gz").FullName
$wheel = (Get-ChildItem (Join-Path $dist "wheels-from-sdist") -Filter "*.whl").FullName

& $python -m twine check $sdist $wheel
& $python -m twine upload --repository testpypi $sdist $wheel
```

After TestPyPI finishes, wait for the project page to appear and install the
uploaded version in a clean venv. The extra index is needed because NumPy is
resolved from the normal PyPI index:

```powershell
$smoke = Join-Path $env:TEMP "rustedscithe-testpypi-smoke"
py -3.14 -m venv $smoke
$smokePython = Join-Path $smoke "Scripts/python.exe"
& $smokePython -m pip install --upgrade pip
& $smokePython -m pip install --no-cache-dir `
  --index-url https://test.pypi.org/simple/ `
  --extra-index-url https://pypi.org/simple/ `
  rustedscithe==0.1.2
& $smokePython -c "import numpy, rustedscithe; from rustedscithe import Expr; print(rustedscithe.__file__); print(numpy.__version__); print(Expr.parse('x^2 + 1'))"
```

Only after that smoke test passes, publish the same immutable files to
production:

```powershell
& $python -m twine upload $sdist $wheel
```

PyPI release files cannot be replaced under the same version. If an upload or
metadata mistake is found after publication, increment the package version,
rebuild both artifacts, rerun the gates and publish a new release instead of
reusing an already-published version.

For later CI publication, prefer Trusted Publishing or a short-lived scoped
credential; never commit a long-lived upload token.

```bash
python -m twine upload --repository testpypi dist/*
python -m pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ rustedscithe
```

Record artifact hashes, wheel tags, install commands, smoke results and known
optional-backend limitations. Production PyPI publication is a separate final
step and must publish the matching sdist and complete supported wheel matrix
as one versioned release.

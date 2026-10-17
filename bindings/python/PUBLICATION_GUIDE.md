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

For the full contract suite, pass the exact wheel and interpreter to the clean
environment runner. It clears `PYTHONPATH`, creates an isolated venv outside
the checkout, verifies that both the package and native extension load from
that venv, and runs the contract tests against the installed artifact:

```powershell
$wheel = (Resolve-Path .\bindings\python\target\publication\YOUR_RELEASE_WHEEL.whl).Path
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\bindings\python\scripts\test_installed_wheel.ps1 `
  -WheelPath $wheel `
  -PythonExe (Get-Command python.exe).Source
```

Repeat with the minimum supported CPython (3.10) and the newest supported
CPython executable. The current development machine has CPython 3.14 only, so
the ABI3 minimum-version run must be completed on a 3.10 Windows environment
before using it as release evidence.

Then run the Python contract suite against the installed wheel. Ensure the
test process cannot import `bindings/python/python` from the checkout, for
example by running from `$env:TEMP` and setting `PYTHONPATH` explicitly only
when the test harness requires it.

The animation contract suite also verifies renderer startup failure without
opening a desktop window: it sets `WGPU_BACKEND` to an unknown backend, calls
the public animation facade with a valid trajectory, and checks that adapter
initialization failure returns as `RuntimeError` from the isolated worker.
This exercises the installed extension and Bevy renderer boundary, not just a
mocked subprocess response.

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

### Windows release 0.1.3

On 2026-10-17, the corrected sdist and `cp310-abi3-win_amd64` release wheel
were built locally. The wheel was built from the corrected sdist with the
Maturin `release` profile and passed `twine check`. The exact wheel was
installed outside the checkout in a clean CPython 3.14 environment; package
and native extension paths resolved inside that environment, NumPy `2.5.4`
imported, and the installed-wheel contract suite passed (`224 passed,
2 skipped`). The wheel and sdist were subsequently uploaded to TestPyPI and
production PyPI. The TestPyPI clean-install smoke passed; the post-upload
production artifacts were downloaded from PyPI and matched these SHA-256
hashes. The exact production wheel was installed outside the checkout and the
full contract suite passed on both CPython 3.14 / NumPy 2.5.4 and CPython
3.10.20 / NumPy 2.2.6 (`224 passed, 2 skipped` on each). The two skips are the
private panic-probe tests, which require a non-default test-only build feature
and are intentionally absent from the published wheel. The 3.10 run also
confirms the advertised `cp310-abi3` lower bound on Windows x64.

| Artifact | SHA-256 |
| --- | --- |
| `rustedscithe-0.1.3.tar.gz` | `56fabd8c036373b0b42b4a4f9f1db966703ece168be01c2ed0b0139901b139c6` |
| `rustedscithe-0.1.3-cp310-abi3-win_amd64.whl` | `604bc1302725389eb7396fd368de6cf24cabe67f2afb1a53b27c78fbd65e7a13` |

## Linux

**Release gate:** the published `0.1.3` release is Windows-only. Package
metadata in the working tree is set to the unreleased `0.1.4` candidate, but
do not build or upload release artifacts until the non-cosmetic debts in
`TODO.md` are closed. The agreed cross-platform target is `0.1.4`: build its
Windows and Linux wheels plus one sdist from the same release source, validate
the complete artifact set on TestPyPI, and publish the set together. Do not
attempt to append Linux to
`0.1.3`. The older `0.1.3` instructions below are retained only as release
history and diagnostic guidance.

### Prerequisites

The instructions below target Linux x86_64/glibc and a portable
`manylinux2014` wheel. Build inside the official Maturin manylinux container,
not directly on a recent host distribution: a host-built wheel can receive a
plain `linux` tag and fail on older glibc systems. The container provides the
Rust/manylinux toolchain; host Python is used for sdist handling, installation
tests and upload. Python development headers are needed for direct Rust/PyO3
test linking, not for this containerized ABI3 wheel build.

Install Docker and a supported host CPython. On Debian-like systems:

```bash
sudo apt-get update
sudo apt-get install -y docker.io python3 python3-venv python3-pip
python3 -m venv .venv-publication-linux
. .venv-publication-linux/bin/activate
python -m pip install --upgrade pip maturin twine numpy pytest
```

Record the Linux distribution, Docker version, resolved Maturin image digest,
Rust/Cargo, Python, PyO3 and Maturin versions. Do not add SuperLU as a build
dependency for the default wheel; it is an optional runtime backend.

### Build and audit

For the coordinated `0.1.4` release, create and audit one corrected sdist
from the immutable release tag, then use that exact archive for the Linux
build. Build the Windows wheel from the same tagged source and record all
artifact hashes. The README-at-archive-root correction below is required by
this repository's path-dependent package layout.

```bash
set -euo pipefail
VERSION=0.1.4
dist="$(pwd)/bindings/python/target/publication-linux-$VERSION"
raw_sdist="/path/to/rustedscithe-$VERSION.tar.gz"
expected_sdist_sha256="<SHA-256 recorded for the corrected 0.1.4 sdist>"
test "$(uname -m)" = x86_64
mkdir -p "$dist/sdist-stage" "$dist/wheels"
printf '%s  %s\n' "$expected_sdist_sha256" "$raw_sdist" | sha256sum --check -
tar -xzf "$raw_sdist" -C "$dist/sdist-stage"
root="$(find "$dist/sdist-stage" -mindepth 1 -maxdepth 1 -type d -print -quit)"
test -n "$root"
root_name="$(basename "$root")"
test -f "$root/bindings/python/README.md"
cp "$root/bindings/python/README.md" "$root/README.md"

# Use the same corrected sdist contents, but compile inside manylinux2014.
MATURIN_IMAGE=ghcr.io/pyo3/maturin
docker pull "$MATURIN_IMAGE"
docker image inspect "$MATURIN_IMAGE" --format '{{index .RepoDigests 0}}'
docker run --rm \
  -v "$dist:/io" \
  "$MATURIN_IMAGE" \
  build --release \
  --manifest-path "/io/sdist-stage/$root_name/bindings/python/Cargo.toml" \
  --manylinux 2014 \
  --out /io/wheels

wheel="$(find "$dist/wheels" -maxdepth 1 -type f \
  -name "rustedscithe-$VERSION-cp310-abi3-manylinux*_x86_64.whl" \
  -print -quit)"
test -n "$wheel"
python -m twine check "$raw_sdist" "$wheel"
python -m auditwheel show "$wheel"
printf 'sdist=%s\nwheel=%s\n' "$raw_sdist" "$wheel"
sha256sum "$wheel"
```

To create the corrected `0.1.4` sdist once from its release tag, apply the
archive-root README correction, then record and reuse that exact archive for
every platform build:

```bash
mkdir -p "$dist/raw-sdist" "$dist/sdist-new-stage"
maturin sdist --manifest-path bindings/python/Cargo.toml --out "$dist/raw-sdist"
new_sdist="$dist/raw-sdist/rustedscithe-$VERSION.tar.gz"
tar -xzf "$new_sdist" -C "$dist/sdist-new-stage"
new_root="$(find "$dist/sdist-new-stage" -mindepth 1 -maxdepth 1 -type d -print -quit)"
cp "$new_root/bindings/python/README.md" "$new_root/README.md"
corrected_sdist="$dist/rustedscithe-$VERSION.tar.gz"
tar -czf "$corrected_sdist" -C "$dist/sdist-new-stage" "$(basename "$new_root")"
python -m twine check "$corrected_sdist"
sha256sum "$corrected_sdist"
```

Use the resulting corrected archive as `raw_sdist` in the build below and
verify the recorded hash before each platform build. Do not regenerate an
already-published sdist: build from the exact uploaded file.

The expected wheel is tagged `cp310-abi3-manylinux*_x86_64`. Stop if it is
tagged only `linux_x86_64`, if the architecture is not x86_64, or if Maturin/
`auditwheel` reports non-compliant external shared-library dependencies. The
container build enforces manylinux2014 (glibc 2.17 baseline); it does not
produce an aarch64 wheel. Treat aarch64 as a separate build and validation
target.

### Clean installation and smoke test

Test the exact wheel outside the checkout. Use CPython 3.10 and the newest
supported CPython in separate clean environments; the `abi3` tag is not a
substitute for testing the minimum version. The example below uses the
currently available `python3`; select `python3.10` for the minimum-version
run when installed.

```bash
repo="$(git rev-parse --show-toplevel)"
smoke="$(mktemp -d)"
python3 -m venv "$smoke/venv"
cd "$smoke"
"$smoke/venv/bin/python" -m pip install --upgrade pip
"$smoke/venv/bin/python" -m pip install "$wheel" pytest
env -u PYTHONPATH "$smoke/venv/bin/python" -c \
  "import importlib.metadata as m, numpy, pathlib, sys, rustedscithe, rustedscithe._native as n; p=pathlib.Path(rustedscithe.__file__).resolve(); assert pathlib.Path(sys.prefix).resolve() in p.parents; print('version:', m.version('rustedscithe')); print('package:', p); print('native:', n.__file__); print('NumPy:', numpy.__version__); print('Expr:', rustedscithe.Expr.parse('x^2 + 1'))"
env -u PYTHONPATH "$smoke/venv/bin/python" -m pytest \
  -c "$repo/bindings/python/pyproject.toml" \
  "$repo/bindings/python/tests" -q
```

Run this for every supported Linux Python version. The Linux gate requires the
installed-wheel suite, typed-error/telemetry contracts and representative
solver smoke tests to pass; record any platform-specific failure with the
distribution, glibc, compiler, Python and exact wheel tag.

### TestPyPI and PyPI upload

Upload the complete `0.1.4` artifact set to TestPyPI and validate clean
installation on both Windows and Linux before production publication. Do not
upload only the Linux wheel to `0.1.3`.

```bash
set -euo pipefail
repo="$(git rev-parse --show-toplevel)"
python -m twine upload --repository testpypi "$wheel"

test_install="$(mktemp -d)"
python3 -m venv "$test_install/venv"
cd "$test_install"
"$test_install/venv/bin/python" -m pip install --upgrade pip
"$test_install/venv/bin/python" -m pip install --no-cache-dir \
  -v \
  --only-binary=rustedscithe \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "rustedscithe==$VERSION"
"$test_install/venv/bin/python" -c \
  "import importlib.metadata as m, numpy, rustedscithe, rustedscithe._native as n; print('version:', m.version('rustedscithe')); print('package:', rustedscithe.__file__); print('native:', n.__file__); print('NumPy:', numpy.__version__); print('Expr:', rustedscithe.Expr.parse('x^2 + 1')); assert m.version('rustedscithe') == '$VERSION'"
env -u PYTHONPATH "$test_install/venv/bin/python" -m pytest \
  -c "$repo/bindings/python/pyproject.toml" \
  "$repo/bindings/python/tests" -q
```

Confirm the verbose pip output selected the `manylinux` wheel, not a source
build. Only after the TestPyPI clean install and contract suite pass, upload the
same SHA-256-verified Linux wheel to production PyPI:

```bash
python -m twine upload "$wheel"
```

After the production upload, repeat the clean installation and contract suite
against the production index (not the local wheel or TestPyPI):

```bash
prod_install="$(mktemp -d)"
python3 -m venv "$prod_install/venv"
cd "$prod_install"
"$prod_install/venv/bin/python" -m pip install --upgrade pip
"$prod_install/venv/bin/python" -m pip install --no-cache-dir \
  --only-binary=rustedscithe "rustedscithe==$VERSION"
"$prod_install/venv/bin/python" -c \
  "import importlib.metadata as m, numpy, pathlib, sys, rustedscithe, rustedscithe._native as n; p=pathlib.Path(rustedscithe.__file__).resolve(); assert pathlib.Path(sys.prefix).resolve() in p.parents; print('version:', m.version('rustedscithe')); print('package:', p); print('native:', n.__file__); print('NumPy:', numpy.__version__); print('Expr:', rustedscithe.Expr.parse('x^2 + 1')); assert m.version('rustedscithe') == '$VERSION'"
env -u PYTHONPATH "$prod_install/venv/bin/python" -m pytest \
  -c "$repo/bindings/python/pyproject.toml" \
  "$repo/bindings/python/tests" -q
```

The Windows wheel and sdist already uploaded for `0.1.3` are immutable and
must not be uploaded again. The next coordinated release is `0.1.4`; upload
its validated sdist and complete supported wheel set together. See the [PyPI
release-file policy](https://blog.pypi.org/posts/2026-07-22-releases-now-reject-new-files-after-14-days/) and [Maturin's Linux wheel guidance](https://www.maturin.rs/distribution.html).

### Linux status

- [ ] Build the x86_64 manylinux2014 wheel from the exact release sdist and
  record the resolved container image digest.
- [ ] Run `twine check`/`auditwheel show`, clean-install, full contract,
  continuation, telemetry and typed-error gates on Linux.
- [ ] Upload the Linux wheel to TestPyPI and confirm pip selects that wheel
  rather than falling back to the sdist.
- [ ] Upload the same validated Linux wheel to production PyPI and record its
  SHA-256 and clean-install evidence.
- [ ] Decide whether Linux aarch64 is in the supported platform matrix; do not
  imply support until it has its own built and installed-wheel evidence.

## TestPyPI and production handoff

The Windows `0.1.2` upload commands below are retained as release history.
`0.1.3` is the current Windows-only production release. For `0.1.4`, build and
validate all intended platform wheels before uploading the sdist and complete
wheel set. Linux source installation is not considered validated merely
because an sdist is available.

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

## Recovery from a defective release

1. Stop further uploads for the affected version and preserve its wheel, sdist,
   hashes, build logs and the source revision used to produce it.
2. Classify severity. For a severe correctness, security or installation defect,
   yank the affected release on PyPI with a short reason; do not attempt to
   delete or replace its immutable files. For a non-blocking defect, leave the
   release available and publish a clearly documented patch version.
3. Fix the source, increment the Python package version, rebuild the sdist and
   every supported wheel, then rerun metadata, clean-install, contract and
   parity gates against those exact artifacts. Stage the same artifacts on
   TestPyPI before production upload.
4. Publish the replacement version and announce the affected version, impact,
   workaround and upgrade command in the release notes/project news. Record
   both the yanked artifact hashes and replacement hashes in the publication
   evidence.

Never reuse a published version number. A yank is a consumer warning and
rollback signal; it does not make the old artifact mutable or erase its release
history.

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

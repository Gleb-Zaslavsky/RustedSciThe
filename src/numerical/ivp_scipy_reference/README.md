# SciPy IVP Numerical Reference

This repository-only Python project will host the minimal SciPy source subset
needed to compare RustedSciThe's Radau and BDF numerical cores against the
SciPy algorithms. It is an independent oracle, not a production dependency or
a wrapper used by the Rust solvers.

The initial vendored subset is deliberately limited to `radau.py`, `bdf.py`,
and their shared `base.py` and `common.py`. The higher-level `ivp.py` dispatcher
is not included: it imports unrelated RK and LSODA implementations and is not
needed to exercise the Radau/BDF solver classes directly. Dense, sparse, and
linear algebra helpers remain imported from the installed SciPy runtime where
the upstream solver source itself uses them.

The source of truth is the official SciPy `v1.18.1` release. The upstream
solver files are copied from
`https://github.com/scipy/scipy/tree/v1.18.1/scipy/integrate/_ivp`; `bdf.py`,
`base.py`, and `common.py` were verified against the pinned source. `radau.py`
and `bdf.py` add opt-in telemetry hooks only; their numerical formulas and
controller branches remain the upstream implementation. Clean upstream and
instrumented SHA-256 values are recorded in `TODO.md`, and the applicable
SciPy BSD license is preserved in `SCIPY_LICENSE.txt`.

The previous local candidates were from SciPy 1.13 and are no longer the
reference. Their historical hashes remain in `TODO.md` only to identify the
superseded inputs. Old parity measurements must not be treated as evidence for
the current `v1.18.1` reference.

The Python environment is pinned by `.python-version`, exact project
dependencies, and `uv.lock`. From the repository root, recreate it with
`uv sync --project src/numerical/ivp_scipy_reference --locked --no-install-project`.
Run the contract tests from `src/numerical` so the reference package is
importable as a package:
`uv --project ivp_scipy_reference run --locked --no-sync python -m unittest ivp_scipy_reference.tests.test_bdf_telemetry ivp_scipy_reference.tests.test_radau_telemetry`.

The Rust crate excludes this directory from Cargo packaging. Git intentionally
tracks the reference `.py` sources while ignoring local virtual environments,
bytecode, and generated Python package metadata.

The matched Radau runner is `python -m ivp_scipy_reference.radau_parity` from
`src/numerical`; `--format tsv` is the machine-readable interface used by the
Rust story. Set `IVP_SCIPY_PYTHON` to the isolated interpreter when running
`radau_rust_vs_scipy_matched_workload_and_counter_semantics_story`.

The vendored BDF numerical implementation is based on byte-for-byte SciPy
`v1.18.1` source. Its opt-in `BdfTelemetry` hooks are an additive
instrumentation patch; with
`telemetry=None` or a disabled telemetry instance, the original RHS/LU
callbacks are passed through unchanged.
The BDF native-counter and instrumentation contracts are tested separately
from the Rust/Python parity story. The matched numerical-core parity campaign
against SciPy 1.18.1 now covers ten workloads; its measured deltas and explicit
scope limits are recorded in `BDF/BDF_STORY_TESTS.md` and the BDF user guides.

## Current Focus

The Radau and BDF numerical-core parity campaigns have matched Rust against the
pinned SciPy 1.18.1 reference. BDF's remaining counter deltas are localized and
explained. This does not claim full SciPy API parity: Rust-only budgets/errors
and the lack of a public Rust BDF dense-output evaluator are documented scope
boundaries. The broader BDF release matrix still needs a post-fix rerun after
two test-only expectation corrections.

The parity preflight requires the locked Python, NumPy, and SciPy versions and
prints the upstream tag plus vendored-source hashes. The helper compatibility tests exercise `group_columns`, dense
`lu_factor`/`lu_solve`, and sparse `csc_matrix`/`eye`/`issparse`/`splu` APIs.
The lockfile pins their implementation provider to SciPy 1.18.1; these tests
are an early compatibility contract, not a claim that every SciPy release is
supported.

## Radau Telemetry Semantics

The optional `telemetry.RadauTelemetry` instrumentation is observational;
SciPy's native `nfev`, `njev`, and `nlu` values are left intact. In particular,
`OdeSolver.fun_vectorized` calls the underlying RHS directly, so SciPy's
`nfev` intentionally excludes finite-difference Jacobian probes. The added
`rhs_callback_calls` and `fd_rhs_callback_calls` expose those calls separately.

| Quantity | SciPy reference meaning | Rust Radau counterpart |
| --- | --- | --- |
| RHS calls | `nfev`; ordinary solver calls, excluding FD probes | `residual_calls` |
| Jacobian refresh | `njev`; initial Jacobian plus refreshes | `jacobian_calls` |
| LU count | `nlu`; individual real or complex shifted LU factorization | `2 * factorizations`; Rust counts one real/complex pair backend event |
| Linear solve calls | Instrumented calls to SciPy's `solve_lu` | `linear_solves` (real plus complex) |
| Newton iterations | Sum of `solve_collocation_system` iteration counts | `newton_iterations` |
| Attempts/rejections | Added controller counters; rejection means a discarded trial attempt | `attempts` / `rejected_steps` |

These mappings are hypotheses to verify in the matched story, not promises that
adaptive trajectories or every work count are identical. Python timings are
diagnostic and are not a cross-language performance claim.

The first matched Radau run found and corrected two Rust controller deviations:
error rejections now include SciPy's Newton safety factor, and retries preserve
the last accepted step/error history. The ignored story retains per-attempt
`h`, error, Newton-iteration, acceptance, and Jacobian-refresh traces. One
Robertson terminal-step stale-J refresh still differs because the two dense
linear-algebra implementations make a different convergence decision at a
roundoff-sensitive boundary; this is reported as a localized work-count delta,
not hidden behind aggregate counters.

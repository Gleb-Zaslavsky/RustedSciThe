# SciPy BVP Reference

This repository-only reference supports numerical-core parity work for the
new Rust `BVP_sci` implementation. The source of truth is the official SciPy
`v1.18.1` implementation of `scipy.integrate._bvp`, copied without numerical
edits from
[`scipy/integrate/_bvp.py`](https://github.com/scipy/scipy/blob/v1.18.1/scipy/integrate/_bvp.py)
to [bvp.py](bvp.py). Its SHA-256 is
`f8c4125dabc65bfcffbb9dcd2d1f438f6fc34089a4117af741a85ccc25506e92`.

The older `src/numerical/BVP_sci/_bvp.py` is preserved for history but is not
the current reference. The upstream BSD license is retained in
[SCIPY_LICENSE.txt](SCIPY_LICENSE.txt).

`telemetry.py` instruments calls around an isolated import of the vendored
module. It reports diagnostic stage timings and callback/factorization counts;
it does not modify SciPy's formulas or controller. Timing scopes are inclusive
where noted and must not be summed as if they were disjoint. Counter mappings
to Rust are a separate parity contract, not assumed from similar names.

The Python runtime is the pinned SciPy/NumPy environment already used by
`src/numerical/ivp_scipy_reference`. Run the reference contract tests from the
repository root:

```powershell
& .\.venv-gavin-win\Scripts\python.exe -m unittest discover -s src/numerical/BVP_sci/scipy_reference/tests -v
```

The compact parity fixture can be queried independently:

```powershell
& .\.venv-gavin-win\Scripts\python.exe src/numerical/BVP_sci/scipy_reference/bvp_parity.py linear --format tsv
```

Run the ignored Rust/Python parity stories in debug mode with:

```powershell
cargo test --lib --no-default-features numerical::BVP_sci::new::story_tests::scipy_118_ -- --ignored --nocapture --test-threads=1
```

The current matched set includes exact linear, exponential, unknown-parameter,
Sturm-Liouville, Bratu, official shock-layer, singular-term, nonlinear-boundary,
60-state, 60-state/two-parameter, and failure/status cases. Separate gates
compare the full global collocation Jacobian at a fixed parameterized state and
compare exponential dense-output
derivatives plus per-interval RMS residuals. A strict Bratu refinement gate
also matches the 181-node mesh, sampled derivatives and interval defects;
status-3 boundary-tolerance exhaustion is checked with exact outer-pass and
factorization counts. The shock-layer has one documented linear-solve count
difference: nalgebra and SuperLU land on opposite sides of a strict
component-wise Newton stopping threshold by a small margin; Rust takes one extra correction,
while the final mesh and solution remain matched. The stage-by-stage source
map and scope differences are in
[NUMERICAL_CORE_PARITY_LEDGER.md](NUMERICAL_CORE_PARITY_LEDGER.md); the active
checklist is in [TODO.md](TODO.md).

No release runs are part of the initial parity setup. The benchmark campaign
will be added only after the debug numerical and telemetry contracts are
stable. Rust compares solutions on common evaluation points because adaptive
mesh node counts are not required to match SciPy.

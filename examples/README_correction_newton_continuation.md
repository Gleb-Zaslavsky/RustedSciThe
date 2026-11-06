# Fixed-Mesh Correction-Newton Continuation Examples

These paired Rust/Python examples use the public `BVP_api` facade, prepare one
AtomView/Lambdify/banded solver, and request ten ordered parameter points.
The default is 256 intervals (257 nodes); 512 or 1024 can be selected explicitly.
They print per-point telemetry, check the fixed mesh, finite values, endpoint
conditions and independently evaluated trapezoidal residuals, and verify that
each point names the previous successful point as its guess source.

## Models

- `bvp_api_correction_newton_combustion_continuation`: a synthetic six-state fixture from
  `src/numerical/BVP_CorrectionNewton/tests/large_workloads.rs`, with a source multiplier
  `heat_scale` swept from 0.8 to 1.25. This is **not real combustion kinetics**.
  The state labels have no assigned physical units or calibrated meaning.
- `bvp_api_correction_newton_bratu_continuation`: `u'' + lambda*exp(u) = 0`, on the symmetric
  half-domain `[0, 0.5]` with `u(0)=0` and `u'(0.5)=0`. Reflection gives the usual
  Dirichlet problem on `[0, 1]`. The lower branch is checked against its analytic
  profile for ten `lambda` values from 0.3 to 3.0. This is not pseudo-arclength
  continuation, and does not follow the turning point or upper branch.

## Run

From the repository root, with the Python bindings installed for Python runs:

```text
cargo run --release --example bvp_api_correction_newton_combustion_continuation -- 256
cargo run --release --example bvp_api_correction_newton_bratu_continuation -- 256
python bindings/python/examples/bvp_api_correction_newton_combustion_continuation.py --steps 256
python bindings/python/examples/bvp_api_correction_newton_bratu_continuation.py --steps 256
```

Run these commands sequentially when sharing Cargo artifacts with other agents.
Use ordinary Python execution, not `python -O`, so result assertions stay active.

## Layout And Checkpoints

The Rust initial guess is state-major, `states x steps`. Python supplies `steps`
reduced node-major rows of `states` values. Within each state, the prescribed
endpoint is omitted: left-constrained components start at node 1, while
right-constrained components start at node 0. A reduced row therefore need not
represent one common physical coordinate across components. Successful output
in both examples is a full node-major `(steps + 1) x states` profile.

Continuation is solver-managed: examples do not overwrite the guess after each
point. A failed point must not replace the last successful checkpoint. The
report walkers retain their expected checkpoint on failure and check it at the
next point; the default schedules are expected to have ten successes. They do
not deliberately provoke a failure or claim failure recovery has been exercised.

The continuation loop is implemented in the native solver and exposed through
the Rust and Python `BVP_api` facades. Rust and Python examples share the same
fixed-mesh parameter schedule and validate the profile/checkpoint contract.
Run the focused Rust facade tests with
`cargo test --lib correction_newton -- --test-threads=1`; Python contract coverage is in
`bindings/python/tests/test_bvp_correction_newton_continuation.py`.

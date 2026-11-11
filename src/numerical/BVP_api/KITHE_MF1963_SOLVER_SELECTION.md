# KiThe: MF-1963 BVP Solver Selection Findings

Date: 2026-11-11  
Scope: direct public RST `BVP_api`, MF-1963 six-state LOW/HIGH combustion
fixtures. This report summarizes the RST-side reproductions and gives
workload-specific guidance; it does not claim that every BVP method must solve
every branch or initial guess.

## Summary

Both regimes were exercised through the public symbolic BVP facade with
ExprLegacy/Lambdify and no Dense backend. BVP_sci used Sparse storage,
CorrectionNewton used BandedAuto, and Damped/Frozen used Banded Trapezoid.
Solver tolerances were kept at `2e-7`; no tolerance was relaxed to force a
result. The tests and all detailed output are kept together in
[`mf1963_nonuniform_story_tests.rs`](mf1963_nonuniform_story_tests.rs), with
the persistent measurements in [`STORY_TESTS.md`](STORY_TESTS.md).

The seed categories are important. "KiThe profile" is a caller-supplied,
physically plausible profile fixture, not a claim of exact solution. The
MF-HIGH smooth seed is independently reconstructed from a smooth logistic
temperature/conversion profile. SciPy and CorrectionNewton cross-seeds are
solver-generated warm starts, deliberately separated from normal physical
initial guesses.

## LOW/HIGH Outcome Matrix

| Regime | Solver / settings | Starting guess | Outcome |
|---|---|---|---|
| LOW | BVP_sci / Sparse, ExprLegacy + Lambdify | KiThe clustered physical-profile seed, 2049 nodes | Passed; 3807 nodes; max state error vs SciPy `1.29e-8` |
| LOW | CorrectionNewton / BandedAuto, fixed mesh | KiThe clustered physical-profile seed, 2049 nodes | Passed; trapezoid system converged, but profile differs from collocation reference |
| LOW | CorrectionNewton / BandedAuto | Same profile linearly bisected to 4097 nodes | Passed; reference `theta` error improved from `8.36` to `1.85` |
| LOW | CorrectionNewton / BandedAuto | SciPy reference profile, solver-generated seed | Passed; converged to the same fixed-mesh branch |
| LOW | Damped / Banded Trapezoid / TwoPoint `(0.01,0.01,2.0)`, 2 refinements | KiThe clustered physical-profile seed | Passed; 2207 nodes |
| LOW | Damped / Banded Trapezoid / Pearson `(1e-3,1.4)`, 2 refinements | KiThe clustered physical-profile seed | Passed; 4574 nodes |
| LOW | Damped / Banded Trapezoid / Grcar-Smooke `(1e-3,1e-2,1.4)`, 2 refinements | KiThe clustered physical-profile seed | Passed; 4574 nodes |
| LOW | Damped / Banded Trapezoid / DoublePoints, 4 refinements | KiThe clustered physical-profile seed | Passed; 32769 nodes; best tested Damped agreement with SciPy |
| LOW | Frozen / Banded Trapezoid / default `Frozen_naive` | KiThe clustered physical-profile seed | Failed safely on non-finite residual; no solution published |
| LOW | Frozen / Banded Trapezoid / `at_high_morm=1e-6` | Same physical-profile seed | Passed in 8 iterations with 8 Jacobian refreshes |
| LOW | Frozen / Banded Trapezoid / `every_m=1` | Same physical-profile seed | Did not converge within 100 iterations |
| LOW | Frozen / Banded Trapezoid / default | CorrectionNewton converged profile, solver-generated cross-seed | Passed; retained supplied root within about `3.4e-6` |
| HIGH | BVP_sci / Sparse, ExprLegacy + Lambdify | KiThe profile seed, 301 nodes | Passed; 823 nodes; max state error vs SciPy `2.84e-14` |
| HIGH | BVP_sci / Sparse, ExprLegacy + Lambdify | Independently reconstructed smooth logistic physical seed | Passed; same reference profile; max error `4.26e-14` |
| HIGH | CorrectionNewton / BandedAuto, fixed mesh | KiThe profile seed | Passed; tiny fixed-grid residual; max profile error vs refined collocation `2.45e-2` |
| HIGH | Damped / Banded Trapezoid / TwoPoint `(0.01,0.01,2.0)`, 2 refinements | KiThe profile seed | Passed; 424 nodes; max reference error `2.86e-3` |
| HIGH | Damped / Banded Trapezoid / Pearson `(1e-3,1.4)`, 2 refinements | KiThe profile seed | Passed; 1810 nodes; max reference error `3.52e-4` |
| HIGH | Damped / Banded Trapezoid / Grcar-Smooke `(1e-3,1e-2,1.4)`, 2 refinements | KiThe profile seed | Passed; 1810 nodes; max reference error `3.52e-4` |
| HIGH | Damped / Banded Trapezoid / DoublePoints, 4 refinements | KiThe profile seed | Passed; 4801 nodes; max reference error `9.57e-5` |
| HIGH | Frozen / Banded Trapezoid / default, `at_high_morm=1e-6`, or `every_m=1` | KiThe profile seed | All passed in 3 iterations; fixed-grid profiles agree with CorrectionNewton within `9.1e-6` |

Max reference errors are maximum absolute errors over the six state variables
on common seed coordinates. The LOW Damped DoublePoints absolute trapezoid
defect `7.666e-4` is localized to `q` on interval 31472 (`h=3.134e-7`); its
scaled defect is `1.552e-8`. This does not indicate a degraded discrete solve.
For MF-HIGH, CorrectionNewton and Frozen solve the fixed-grid trapezoid system
accurately; their `2.45e-2` difference from the refined SciPy collocation
profile is a discretization/resolution comparison, not a failed Newton solve.

## What Changed In The RST Route

### Explicit nonuniform mesh and full node-aligned guess

The mesh is an explicit argument to `BvpProblem::new`. It must be finite and
strictly increasing. For Newton BVP routes, the full public guess is a
state-major `state_count x mesh.len()` matrix aligned with every coordinate,
including the boundary nodes. The facade validates that the entries at fixed
boundary nodes agree with the boundary conditions within floating-point
roundoff, then removes those constrained values and packs the free unknowns in
the native ordering. Callers should pass the full profile; they should not
manually reproduce the native compact vector layout.

```rust
use nalgebra::DMatrix;
use RustedSciThe::numerical::BVP_api::{BvpProblem, BvpSolver, BvpSolverSettings};

let mesh: Vec<f64> = load_mesh()?; // finite, strictly increasing coordinates
let full_guess: DMatrix<f64> =
    DMatrix::from_fn(state_count, mesh.len(), |state, node| profile[state][node]);

let problem = BvpProblem::new(
    equations,
    state_names,
    "z",
    mesh,
    boundary_conditions,
)?;

let config = BvpSolverSettings::Damped(damped_options)
    .with_node_aligned_initial_guess(full_guess);
let mut solver = BvpSolver::new(problem, config)?;
let solution = solver.solve()?;
```

The same full-node-aligned builder is available for Frozen and CorrectionNewton.
BVP_sci also accepts the explicit mesh through `BvpProblem` and its own
full-node guess contract. Damped prepares on the supplied mesh and then owns
its adaptive refinement lifecycle; Frozen and CorrectionNewton retain a fixed
mesh. CorrectionNewton assembles each interval with its actual local width.
Existing reduced-guess APIs keep their previous semantics.

Runnable public-facade examples: [Rust](../../../examples/bvp_api_nonuniform_mesh.rs)
and [Python](../../../bindings/python/examples/bvp_api_nonuniform_mesh.py).
Rust's full guess is state-major (`[state][node]`); Python takes a node-major
list (`[node][state]`) and selects the same full-profile semantics with
`initial_guess_layout="node_aligned"`.

### Mixed-endpoint guess packing fix

The MF-LOW reproducer exposed a real Frozen initial-guess packing bug. For
mixed left/right boundary conditions, flattening the state-major matrix by
columns did not match the callback's node-major ordering of free unknowns. At a
known tight CorrectionNewton root, the incorrectly flattened Frozen callback
residual had infinity norm `1.525e3`; the corrected node-major packing reduced
it to `3.55e-8`. The fix is in the facade conversion and preserves the public
state-major contract. It also changed the original-seed Frozen outcome from a
false-looking convergence to a safe typed failure, so failed profiles are not
published.

### Frozen Jacobian refresh and observability

Frozen exposes strategy parameters through its native options. For MF-LOW,
`at_high_morm=1e-6` caused refresh after large corrections and changed the
physical-profile seed run from a non-finite residual failure to convergence in
8 steps. `every_m=1` refreshed too late for that trajectory and did not
converge within 100 iterations. The bounded opt-in `BvpFrozenIterationTrace`
records residual/correction norms, Jacobian age and refresh decisions without
re-evaluating callbacks.

The opt-in policy is set through native options, without changing the BVP
problem or guess layout:

```rust
let options = FrozenSolverOptions::banded_frozen()
    .with_banded_lambdify()
    .trapezoid_derivative()
    .with_tolerance(2e-7)
    .with_max_iterations(100)
    .with_strategy_params(Some(HashMap::from([(
        "at_high_morm".to_owned(),
        Some(vec![1e-6]),
    )])));
let config = BvpSolverSettings::Frozen(options)
    .with_node_aligned_initial_guess(full_guess);
```

MF-HIGH does not support changing the default: default, adaptive refresh and
`every_m=1` all converged in 3 iterations; adaptive refresh increased
Jacobians from 1 to 3 without materially changing the profile. Therefore the
refresh setting remains workload-specific and opt-in. The test's Damped
bounds `[-1e6,1e6]` were required by this setup but do not enforce physical
concentration positivity; KiThe should apply domain-appropriate bounds or
acceptance checks when physical nonnegativity is required.

## Practical Recommendations

- Use BVP_sci Sparse when the desired result is the MF-HIGH collocation
  reference; it reproduced the independent SciPy profile from both the KiThe
  profile and a distinct smooth physical guess.
- For Damped/Trapezoid on MF-HIGH, choose Pearson or Grcar for the measured
  middle point, DoublePoints for the closest tested profile at higher mesh
  cost, and TwoPoint for the coarser/cheaper tested result. These are
  case-specific tradeoffs, not universal rankings.
- CorrectionNewton and Frozen can solve their fixed-mesh discrete equations
  accurately while differing from a refined collocation solution. Compare
  like discretizations and mesh resolutions before classifying that gap as a
  defect.
- Frozen's refresh behavior depends on the trajectory: the LOW case benefits
  decisively from large-correction refresh, while HIGH does not. Keep the
  default unchanged and select the opt-in policy based on workload evidence.
- MF-LOW Frozen with the ordinary physical-profile seed remains a robustness
  limitation under its default refresh policy. A solver-generated
  CorrectionNewton profile is an effective warm start, but this is not proof
  that an ordinary caller guess will converge.

The complete MF-LOW/HIGH ignored Debug matrix passed: 17/17 in `102.74 s`;
the ordinary BVP_api Debug module passed 47 tests with 17 ignored. These are
correctness/suitability results, not performance benchmarks. The tested
frontend/backend scope is ExprLegacy/Lambdify with Sparse/Banded layouts; no
AtomView or AOT parity claim is made.

Reproduce the complete matrix from the crate root:

```powershell
cargo test --lib --no-default-features mf1963_mf -- --ignored --nocapture --test-threads=1
```

# `somelinalg` architecture TODO

This file tracks native-core architecture and solver classification. Python
bindings have their own contract checklist in `bindings/python/TODO.md` and
must not be used to hide unfinished native implementations.

## Capability inventory and public-scope policy

- [ ] Inventory every public or reachable `somelinalg` module and classify it
  as `stable`, `experimental`, `legacy`, `solver-internal` or
  `not suitable for public exposure`.
- [ ] Record ownership, storage layout, allocation behavior, mutation rules,
  typed errors, logging, telemetry and thread-safety for each stable candidate.
- [ ] Mark `bvp_sparse_lu`, sparse reference fixtures, benchmark helpers,
  `sparse_lu_superlu_experimental`, GPU/CUDA routes and unfinished iterative
  implementations explicitly as non-production unless their own review closes
  the required gates.
- [ ] Keep the faithful LAPACK-style banded, pivoted tridiagonal and
  consistent block-tridiagonal routes separate from legacy/experimental
  alternatives; do not silently change solver policy selection.
- [x] Keep SuperLU as an opt-in BVP_sci bridge rather than duplicating its
  external Python API in `somelinalg`.
- [x] Keep `least_squares_qr` outside the current public binding scope until
  its native result/error contract receives a dedicated review.

## Structured-memory and performance evidence

- [ ] Add allocator or instrumented medium/large evidence for the faithful
  banded workspace bound `O(n * (kl + ku))`.
- [ ] Audit all structured paths for accidental dense `n x n` materialization
  in normal factorization, solve, diagnostics and telemetry code.
- [ ] Add release-grade Tabled measurements for factorization, warm solve,
  multiple RHS, refinement and retained workspace for selected structured
  kernels. Keep dense reference/oracle work outside normal solve timings.
- [ ] Measure block-tridiagonal storage and factor reuse on representative
  medium/large block workloads; separate factorization, solve and refinement.

## `linear_sys_diagnostics`

- [ ] Decide whether this module remains a native-only diagnostic helper or is
  redesigned as a production diagnostic API.
- [ ] Remove unconditional `println!` output from any production route; use
  optional logging and/or a returned structured report.
- [ ] Replace boolean-only results with typed diagnostics containing solvability,
  singularity/conditioning metrics, finite-input status and source context.
- [ ] Validate matrix/RHS dimensions, finiteness and thresholds before dense
  operations; define ownership and allocation semantics explicitly.
- [ ] Add tests for singular, inconsistent, ill-conditioned and finite regular
  systems before reconsidering Python exposure.

## Error and lifecycle consistency

- [ ] Ensure structured kernels reject non-finite input before mutating a
  factorization workspace and report factor/solution failures distinctly.
- [ ] Standardize dimension, pivot, RHS-layout and backend-unavailable error
  semantics across banded families without breaking faithful LAPACK behavior.
- [ ] Document which diagnostics are cached at factorization time and which
  are intentionally offline/dense or test-only.

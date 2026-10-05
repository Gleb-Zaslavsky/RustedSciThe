# BVP Damp AOT Story Test Taxonomy

This ledger keeps the AOT stories separate from the pure-Lambdify stories.
The existing story ledgers remain historical evidence and are not replaced by
this taxonomy.

## Route Categories

| Category | Question | Typical tests |
| --- | --- | --- |
| lambdify_vs_aot | Does AOT agree with the prepared Lambdify oracle? | backend_compare, lambdify_cross_product, aot_diagnostics |
| aot_only_correctness | Does one AOT route preserve residual/Jacobian/layout and solve correctness without a Lambdify reference? | aot_diagnostics, frozen_runtime_story |
| aot_lifecycle | Are materialization, build, publication, linking, cache reuse and invalidation typed and reproducible? | aot_race_stress, generated lifecycle tests |
| aot_toolchain | Do Rust, C and Zig consume the same manifest/ABI and expose equivalent diagnostics? | codegen cross-language stories |
| aot_performance | What is cold preparation/build/link cost versus warm callback cost? | dated ignored stories only |

## Rules

1. A lambdify_vs_aot story must report componentwise residual and Jacobian
   drift, matrix layout, route marker, and integer solver counters.
2. An aot_only_correctness test must not use a successful Lambdify solve as
   an implicit oracle. It checks the prepared artifact contract, callback
   shapes, finite outputs, and the selected solver result independently.
3. Lifecycle tests must keep build/link/publication time outside warm callback
   measurements. Failure-injection and cross-process tests are lifecycle
   tests, not solver performance tests.
4. Dense is a small control route only. Production-sized AOT stories use
   Sparse/faer or Banded; no large Dense matrix should be added to the corpus.
5. Every printed result belongs in the dated AOT story ledger and in the
   dedicated BVP report file. Report I/O is outside measured scopes.

## Commands

Debug correctness smoke:

    cargo test --lib --no-default-features numerical::BVP_Damp::test_aot_diagnostics -- --nocapture --test-threads=1
    cargo test --lib --no-default-features symbolic::bvp::atom_aot::tests -- --nocapture --test-threads=1

Release stories are intentionally opt-in and should be run only after the
debug gates pass:

    cargo test --release --lib --no-default-features numerical::BVP_Damp::test_aot_diagnostics -- --nocapture --test-threads=1
    cargo test --release --lib --no-default-features numerical::BVP_Damp::test_backend_compare -- --ignored --nocapture --test-threads=1

## Current Boundary

As of 2026-09-21, AtomView owns a validated prepared plan and typed cold
telemetry. The sparse prepared provider also exposes fallible residual and
Jacobian callback boundaries; the old methods remain compatibility wrappers.
This is a correctness boundary, not a production-ready claim for native
AtomView linked callbacks. Full compiler/linker/warm-runtime and parity stories
remain open in TODO.md.

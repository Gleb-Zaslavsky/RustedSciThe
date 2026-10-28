# Independent IVP Parameter Loops

The first-tier stiff IVP solvers (BDF, Backward Euler, LSODE2 and Radau) offer
`try_solve_parameter_loop(y0, parameter_vectors)`. The prepared symbolic/AOT
model is reused; every vector starts a fresh integration from the same `y0`
over the solver's configured interval. No previous terminal state is used.

Each report point retains its input index and parameter vector. A successful
point contains `x_mesh` and row-oriented state values with shape
`(time_points, state_variables)`, plus the solver's native telemetry snapshot.
A failed point keeps its typed native error and does not prevent later points
from running. Empty or malformed vectors are handled by the native solver and
recorded as point failures.

Telemetry is captured after each attempt, but counter reset and accumulation
scope retain each solver's native semantics; the loop does not convert
cumulative counters into per-point deltas.

Radau follows its configured output policy: Dense retains accepted-step
boundaries, Sampled retains requested sample times, and FinalOnly returns just
the terminal sample. The common result axis therefore represents retained
solver output, not an inferred internal mesh when the policy discards it.

The shared StiffAPI exposes the same operation after route preparation. Use
the direct solver APIs when solver-specific controls are needed. Runnable
examples are `examples/bdf_parameter_loop.rs`,
`examples/be_parameter_loop.rs`, `examples/lsode2_parameter_loop.rs`,
`examples/radau_parameter_loop.rs` and
`examples/stiff_api_parameter_loop.rs`.

# Independent IVP Parameter Loops

Use `solve_parameter_loop(y0, parameter_vectors)` when one prepared IVP model
must be solved for several parameter vectors. BDF, Backward Euler, LSODE2 and
Radau all expose this operation directly, and `PreparedStiffSolver` provides
the same operation after selecting a route.

Every point is a fresh integration from the same `y0` over the configured
time interval. Prepared symbolic/AOT callbacks are reused, but the previous
terminal state is not used as a guess. This is a parameter sweep, not
continuation.

The returned ordered `IvpParameterLoopPoint` objects retain the index and
parameter vector. A successful point exposes the ordinary detached solution
through `.solution`; its owned NumPy trajectory and native telemetry remain
available on that solution. A failed point exposes its typed exception through
`.error`, with `stage`, `solver`, `route` and `native_kind` context, and later
points still run. Radau follows its configured output policy: dense output
returns accepted-step boundaries, sampled output returns requested times, and
final-only output contains just the terminal row.

Telemetry is attached to each successful solution using the native solver's
snapshot semantics. Some counters may be cumulative across repeated calls;
the loop does not infer or manufacture per-point deltas.

See [`ivp_parameter_loop.py`](../examples/ivp_parameter_loop.py) for direct
solver selection and [`stiff_api_parameter_loop.py`](../examples/stiff_api_parameter_loop.py)
for the shared facade.

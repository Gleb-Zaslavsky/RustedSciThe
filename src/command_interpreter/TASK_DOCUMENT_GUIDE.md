# Task Document Guide

Task files use sections followed by `key: value` fields. Control keys are
case-insensitive; names inside `equations`, `parameters`, `where`, and
`substitute` are preserved because they are symbolic names.

## Continuation

Continuation is an explicit section, not a second interpretation of
`parameter_values`:

```text
continuation
parameter: rate
values: 0.5, 1.0, 2.0
mode: prepared
restart_each: false
```

The parser retains the symbolic RHS and returns a typed `ContinuationSpec`.
`fresh` means rebuild for every value, `warm` permits reuse of a warm solver,
and `prepared` means reuse the prepared symbolic/backend model. The actual
solver adapter must still decide which state can safely be reused.

## Routing policy

`UniversalODESolver` is useful for simple nonstiff prototyping. For stiff
problems use the dedicated API of LSODE2, BDF, Radau, or Backward Euler. Those
APIs expose policies that a universal facade cannot represent faithfully,
including linear layout, symbolic frontend, AOT lifecycle, continuation, and
telemetry.

The task parser therefore normalizes documents first and delegates execution
to solver-specific adapters. It must never silently route a BVP_sci document
to the historical BVP implementation.

## Collision policy

Conflicting aliases and case-normalized duplicate keys are errors. The parser
does not use “last write wins” semantics for task configuration.

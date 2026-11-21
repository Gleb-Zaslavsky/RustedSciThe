# Pre-preparation symbolic substitutions for IVPs

`try_with_substitutions` expands named subexpressions in source `Expr` values
**before** solver preparation. It is a symbolic input convenience only: it does
not change the numerical algorithm and is not parameter binding/rebinding or
continuation. Alias definitions are expanded recursively, including chains.

```rust
let request = StiffSolverRequest::Bdf { options }
    .try_with_substitutions([("loss", Expr::parse_expression("-rate*y"))])?;
let mut solver = request.prepare(None)?;
solver.try_solve()?;
```

`rate` remains an ordinary runtime parameter and is configured through the
native parameter API. Alias names must not collide with state variables, the
independent variable, or declared parameters. Duplicate and cyclic definitions
return typed `SymbolicSubstitutionError` before callback preparation.

Supported by BDF, Backward Euler, LSODE2 and Radau, and by the common
`StiffSolverRequest`. Their native options/config types expose the same
`try_with_substitutions` operation. The nonstiff `nonstiffODE` and universal
`UniversalODESolver` also expose this preparation-time operation for symbolic
RHS inputs. The universal route rejects substitutions when native callbacks or
an independent LSODE2 problem config are installed. Direct Python solver constructors accept
`substitutions={"loss": Expr.parse("-rate*y")}`; Python `StiffSolverRequest`
uses `request.with_substitutions({...})`; `UniversalNonStiffSolver` accepts the
same constructor keyword.

Substitution applies to symbolic `Expr` problems. It cannot rewrite
user-provided numerical callbacks; LSODE2 rejects that combination with a
typed error rather than silently ignoring the aliases.

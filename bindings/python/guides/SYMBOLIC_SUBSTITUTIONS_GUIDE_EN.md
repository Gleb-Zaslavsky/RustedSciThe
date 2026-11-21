# Symbolic substitutions before solver preparation

Named subexpressions can be supplied as `substitutions` when creating a
symbolic BDF, Backward Euler, LSODE2 or Radau solver, or attached to a BVP API
request with `with_substitutions`. Nonlinear solvers accept the same mapping
at construction, and `UniversalFitting` exposes it as a builder method. In
each case aliases are recursively expanded before
numerical preparation:

```python
from rustedscithe import BdfSolver, Expr

solver = BdfSolver(
    [Expr.parse("loss")], ["y"], "t", [1.0], 0.0, 1.0, 0.01,
    parameters=["rate"], parameter_values=[2.0],
    substitutions={"loss": Expr.parse("-rate*y")},
)
solution = solver.solve()
```

`rate` and other free symbols remain ordinary problem variables/parameters.
Substitution is not bind/rebind and does not run the solve. With the common
facade, call `request.with_substitutions({...})` before `prepare()`.

For example, a BVP request can use an alias in its RHS without rebuilding the
expression in client code:

```python
request = BvpSolverRequest.sci(
    [Expr.parse("decay")], ["y"], "x", mesh,
    [(0, "left", 1.0)], initial_guess,
)
request.with_substitutions({"decay": Expr.parse("-y")})
solution = request.prepare().solve()
```

For a symbolic nonlinear system, aliases are expanded before unknown
inference, so the alias itself does not become a variable:

```python
system = SymbolicSystem([Expr.parse("force - 2")])
solver = NonlinearSolver(
    system,
    parameters=["rate"],
    parameter_values=[4.0],
    substitutions={"force": Expr.parse("rate*x")},
)
result = solver.solve([1.0])
```

`UniversalFitting.with_substitutions({...})` also applies aliases to LM/Gavin
equations and VarPro basis/invariant expressions before model preparation.

Empty, duplicate, cyclic, or state/time/parameter-colliding names are rejected
as typed configuration errors. LSODE2 substitutions cannot be combined with
user-provided numerical callbacks, which are opaque to symbolic expansion;
that combination is explicitly rejected.

"""Small Python-native convenience layer over the compiled extension."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable, Mapping
from typing import Any

from numpy.typing import ArrayLike

from . import _native

Expr = _native.Expr
SymbolicSystem = _native.SymbolicSystem
SolverOptions = _native.SolverOptions
TaskDocument = _native.TaskDocument
BvpSciSolver = _native.BvpSciSolver
BvpDampSolver = _native.BvpDampSolver


def _owned_sequence(value: Any) -> Any:
    """Convert NumPy-like containers before crossing into the Rust API."""

    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else value


def _method_name(method: str) -> str:
    normalized = method.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "bdf": "bdf", "radau": "radau5", "radau5": "radau5",
        "be": "backward_euler", "backward_euler": "backward_euler",
        "lsode": "lsode2", "lsode2": "lsode2",
    }
    if normalized not in aliases:
        raise ValueError(
            "unsupported IVP method; expected bdf, radau5, backward_euler or lsode2"
        )
    return aliases[normalized]


def create_ivp(
    method: str,
    expressions: list[Expr],
    state_variables: list[str],
    time_variable: str,
    y0: ArrayLike,
    t0: float,
    t_end: float,
    max_step: float,
    *,
    parameters: list[str] | None = None,
    parameter_values: ArrayLike | None = None,
    options: SolverOptions | None = None,
    **solver_options: Any,
) -> Any:
    """Construct a native IVP solver with one common Python signature."""

    family = _method_name(method)
    if family == "bdf":
        return _native.BdfSolver(
            expressions, state_variables, time_variable, _owned_sequence(y0), t0, t_end, max_step,
            options=options, parameters=parameters,
            parameter_values=_owned_sequence(parameter_values), **solver_options,
        )
    if family == "radau5":
        # Radau owns y0 at solve time; it is not part of symbolic preparation.
        return _native.RadauSolver(
            expressions, state_variables, time_variable, t0, t_end, max_step,
            options=options, parameters=parameters, **solver_options,
        )
    if family == "backward_euler":
        return _native.BackwardEulerSolver(
            expressions, state_variables, time_variable, _owned_sequence(y0), t0, t_end,
            h=max_step, options=options, parameters=parameters,
            parameter_values=_owned_sequence(parameter_values), **solver_options,
        )
    lsode_method = solver_options.pop("method", "bdf")
    return _native.Lsode2PythonSolver(
        expressions, state_variables, time_variable, _owned_sequence(y0), t0, t_end, max_step,
        options=options, method=lsode_method, parameters=parameters,
        parameter_values=_owned_sequence(parameter_values), **solver_options,
    )


def solve_ivp(
    method: str,
    expressions: list[Expr],
    state_variables: list[str],
    time_variable: str,
    y0: ArrayLike,
    t0: float,
    t_end: float,
    max_step: float,
    *,
    parameters: list[str] | None = None,
    parameter_values: ArrayLike | None = None,
    options: SolverOptions | None = None,
    **solver_options: Any,
) -> Any:
    """Construct and execute one IVP through the native lifecycle."""

    solver = create_ivp(
        method, expressions, state_variables, time_variable, y0, t0, t_end,
        max_step, parameters=parameters, parameter_values=parameter_values,
        options=options, **solver_options,
    )
    if _method_name(method) == "radau5":
        return solver.solve(_owned_sequence(y0), _owned_sequence(parameter_values))
    return solver.solve()


def _bvp_method_name(method: str) -> str:
    """Normalize BVP aliases without hiding solver-specific capabilities."""

    normalized = method.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "bvp_sci": "bvp_sci",
        "scipy": "bvp_sci",
        "bvp_damp": "bvp_damp",
        "damped": "bvp_damp",
    }
    if normalized not in aliases:
        raise ValueError("unsupported BVP method; expected bvp_sci or bvp_damp")
    return aliases[normalized]


def create_bvp(
    method: str,
    expressions: list[Expr],
    state_variables: list[str],
    boundary_data: Any,
    mesh: ArrayLike,
    initial_state: Any,
    *,
    options: SolverOptions | None = None,
    **solver_options: Any,
) -> Any:
    """Construct a BVP solver while preserving native solver options.

    ``boundary_data`` is a list of boundary expressions for ``bvp_sci`` and a
    ``{state: [(position, value), ...]}`` mapping for ``bvp_damp``. Remaining
    keyword arguments are forwarded unchanged to the selected native adapter.
    """

    family = _bvp_method_name(method)
    solver_type = BvpSciSolver if family == "bvp_sci" else BvpDampSolver
    return solver_type(
        expressions,
        state_variables,
        boundary_data,
        _owned_sequence(mesh),
        _owned_sequence(initial_state),
        options=options,
        **solver_options,
    )


def solve_bvp(
    method: str,
    expressions: list[Expr],
    state_variables: list[str],
    boundary_data: Any,
    mesh: ArrayLike,
    initial_state: Any,
    *,
    options: SolverOptions | None = None,
    **solver_options: Any,
) -> Any:
    """Construct and execute one BVP through its native lifecycle."""

    solver = create_bvp(
        method,
        expressions,
        state_variables,
        boundary_data,
        mesh,
        initial_state,
        options=options,
        **solver_options,
    )
    return solver.solve()


def solve_nonlinear(
    system: SymbolicSystem,
    initial_guess: ArrayLike,
    *,
    parameters: ArrayLike | None = None,
    **solver_options: Any,
) -> Any:
    """Create and solve a square nonlinear system natively."""

    solver = create_nonlinear(system, **solver_options)
    return solver.solve(_owned_sequence(initial_guess), _owned_sequence(parameters))


def create_nonlinear(system: SymbolicSystem, **solver_options: Any) -> Any:
    """Construct a prepared-capable nonlinear solver without solving it."""

    return _native.NonlinearSolver(system, **solver_options)


def fit(
    equations: list[Expr],
    initial_guess: ArrayLike,
    *,
    parameters: ArrayLike | None = None,
    **solver_options: Any,
) -> Any:
    """Create and solve a symbolic least-squares problem natively."""

    solver = create_fit(equations, _owned_sequence(initial_guess), **solver_options)
    return solver.solve() if parameters is None else solver.solve_with_parameters(_owned_sequence(parameters))


def create_fit(
    equations: list[Expr],
    initial_guess: ArrayLike,
    **solver_options: Any,
) -> Any:
    """Construct a symbolic fitting solver while retaining its native API."""

    return _native.FittingSolver(
        equations,
        _owned_sequence(initial_guess),
        **solver_options,
    )


def task(source_or_path: str | Path) -> TaskDocument:
    """Parse a task document from text or a filesystem path."""

    raw = str(source_or_path)
    if isinstance(source_or_path, Path) or ("\n" not in raw and "\r" not in raw):
        candidate = Path(source_or_path)
        if candidate.exists() and candidate.is_file():
            return TaskDocument.from_file(str(candidate))
    return TaskDocument.parse(raw)


def capabilities(solver: str) -> Any:
    """Forward the canonical capability descriptor from the extension."""

    return _native.capabilities(solver)


def render_table(rows: Iterable[Mapping[str, Any]], columns: Iterable[str] | None = None) -> str:
    """Render compact deterministic ASCII tables without capturing process noise.

    The function is intentionally dependency-free and accepts detached scalar
    values, so it can be used in a long-running batch even when an optional
    plotting or reporting package is unavailable.
    """

    materialized = [dict(row) for row in rows]
    if not materialized:
        return ""
    names = list(columns) if columns is not None else list(materialized[0])
    if not names:
        return ""
    values = [[str(row.get(name, "")) for name in names] for row in materialized]
    widths = [max(len(name), *(len(row[index]) for row in values)) for index, name in enumerate(names)]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    line = lambda row: "| " + " | ".join(value.ljust(width) for value, width in zip(row, widths)) + " |"
    return "\n".join([border, line(names), border, *[line(row) for row in values], border])


def write_table(path: str | Path, rows: Iterable[Mapping[str, Any]], columns: Iterable[str] | None = None) -> None:
    """Write one compact table and nothing else to ``path``."""

    Path(path).write_text(render_table(rows, columns), encoding="utf-8")


def run_task_batch(
    sources: Iterable[str | Path],
    *,
    continue_on_error: bool = True,
) -> list[dict[str, Any]]:
    """Run task documents independently and return a compact typed summary.

    A failed task never prevents later tasks when ``continue_on_error`` is
    true. The returned row preserves the Python exception class so callers can
    route parser, BVP and solver failures without matching message text.
    """

    rows: list[dict[str, Any]] = []
    for index, source in enumerate(sources, start=1):
        raw_source = str(source)
        # A full inline document is useful to the parser but poisonous to a
        # compact report column. Keep paths readable and give inline jobs a
        # stable label; the caller still has the original source object.
        source_label = (
            f"<inline:{index}>"
            if "\n" in raw_source or "\r" in raw_source
            else raw_source
        )
        row: dict[str, Any] = {"source": source_label}
        try:
            document = task(source)
            result = document.run_bvp() if document.kind == "bvp" else document.run()
            route = document.route()
            row.update(
                {
                    "status": "ok",
                    "kind": document.kind,
                    "route": route,
                    "result": result,
                }
            )
        except Exception as error:  # keep the batch alive, but retain its type
            row.update(
                {
                    "status": "error",
                    "kind": "unknown",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            if not continue_on_error:
                rows.append(row)
                raise
        rows.append(row)
    return rows


def run_task_batch_report(
    sources: Iterable[str | Path],
    *,
    path: str | Path | None = None,
    continue_on_error: bool = True,
) -> str:
    """Run a task batch and return a compact, scalar-only report.

    Numerical result objects remain available in :func:`run_task_batch`, but
    this report deliberately contains only stable routing/status fields. It
    is therefore safe to write as a small Markdown/ASCII artifact without
    serializing trajectories or capturing compiler output.
    """

    rows = run_task_batch(sources, continue_on_error=continue_on_error)
    report_rows: list[dict[str, Any]] = []
    columns = [
        "source",
        "status",
        "kind",
        "method",
        "solver",
        "dispatch",
        "frontend",
        "execution",
        "matrix_layout",
        "error_type",
    ]
    for row in rows:
        route = row.get("route", {})
        report_rows.append(
            {
                "source": row.get("source", ""),
                "status": row.get("status", ""),
                "kind": row.get("kind", ""),
                "method": route.get("method", ""),
                "solver": route.get("solver", ""),
                "dispatch": route.get("dispatch", ""),
                "frontend": route.get("frontend", ""),
                "execution": route.get("execution", ""),
                "matrix_layout": route.get("matrix_layout", ""),
                "error_type": row.get("error_type", ""),
            }
        )
    report = render_table(report_rows, columns=columns)
    if path is not None:
        write_table(path, report_rows, columns=columns)
    return report

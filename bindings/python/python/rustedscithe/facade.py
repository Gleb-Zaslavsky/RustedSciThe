"""Small Python-native convenience layer over the compiled extension."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _native

Expr = _native.Expr
SymbolicSystem = _native.SymbolicSystem
SolverOptions = _native.SolverOptions
TaskDocument = _native.TaskDocument
BvpSciSolver = _native.BvpSciSolver
BvpDampSolver = _native.BvpDampSolver
Lsode2Solver = _native.Lsode2Solver
ShootingBvpSolver = _native.ShootingBvpSolver

_IvpSolver = (
    _native.UniversalNonStiffSolver
    | _native.BdfSolver
    | _native.RadauSolver
    | _native.BackwardEulerSolver
    | _native.Lsode2Solver
)
_IvpSolution = (
    _native.UniversalNonStiffSolution
    | _native.BdfSolution
    | _native.RadauSolution
    | _native.BackwardEulerSolution
    | _native.Lsode2Solution
)

_ANIMATION_ERROR_PREFIX = "RUSTEDSCITHE_ANIMATION_ERROR:"


def _require_animation_main_thread() -> None:
    """Keep desktop event-loop startup on the caller's Python main thread."""

    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("trajectory animation must be started from Python's main thread")


def _run_animation_process(
    dimension: int,
    points: NDArray[np.float64],
    times: NDArray[np.float64],
    axis_names: tuple[str, ...] | None,
    speed: float,
) -> None:
    """Run one Bevy event loop in an isolated process.

    Winit permits one event loop per process. A short-lived child keeps that
    restriction from leaking into the public API: callers can close one view
    and open another later in the same Python process.
    """

    _require_animation_main_thread()
    with tempfile.TemporaryDirectory(prefix="rustedscithe-animation-") as temporary:
        input_path = Path(temporary) / "trajectory.npz"
        np.savez(input_path, points=points, times=times)
        arguments = [
            sys.executable,
            "-m",
            "rustedscithe._animation_worker",
            "--input",
            str(input_path),
            "--dimension",
            str(dimension),
            "--axis-names",
            json.dumps(axis_names),
            "--speed",
            repr(float(speed)),
        ]
        completed = subprocess.run(
            arguments,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    if completed.returncode == 0:
        if completed.stdout:
            sys.stdout.write(completed.stdout)
            sys.stdout.flush()
        if completed.stderr:
            sys.stderr.write(completed.stderr)
            sys.stderr.flush()
        return

    for line in reversed(completed.stderr.splitlines()):
        if line.startswith(_ANIMATION_ERROR_PREFIX):
            try:
                error = json.loads(line[len(_ANIMATION_ERROR_PREFIX) :])
            except json.JSONDecodeError:
                break
            message = str(error.get("message", "animation worker failed"))
            error_type = error.get("type")
            if error_type == "ValueError":
                raise ValueError(message)
            if error_type == "TypeError":
                raise TypeError(message)
            raise RuntimeError(message)

    detail = completed.stderr.strip() or completed.stdout.strip()
    message = f"animation worker exited with status {completed.returncode}"
    if detail:
        message = f"{message}: {detail}"
    raise RuntimeError(message)


def animate_trajectory_2d(
    points: ArrayLike,
    times: ArrayLike,
    axis_names: tuple[str, str] | None = None,
    speed: float = 1.0,
) -> None:
    """Display a sample-major ``(N, 2)`` trajectory in a blocking Bevy window.

    Each display runs in an isolated child process because winit allows one
    event loop per process. Call from Python's main thread.
    """

    points_array = np.asarray(points, dtype=np.float64)
    times_array = np.asarray(times, dtype=np.float64)
    if points_array.ndim != 2 or points_array.shape[1] != 2:
        raise ValueError(f"points must have shape (samples, 2), got {points_array.shape}")
    if times_array.ndim != 1:
        raise ValueError(f"times must have shape (samples,), got {times_array.shape}")
    _run_animation_process(2, points_array, times_array, axis_names, speed)


def animate_trajectory_3d(
    points: ArrayLike,
    times: ArrayLike,
    axis_names: tuple[str, str, str] | None = None,
    speed: float = 1.0,
) -> None:
    """Display a sample-major ``(N, 3)`` trajectory in a blocking Bevy window.

    Each display runs in an isolated child process because winit allows one
    event loop per process. Call from Python's main thread.
    """

    points_array = np.asarray(points, dtype=np.float64)
    times_array = np.asarray(times, dtype=np.float64)
    if points_array.ndim != 2 or points_array.shape[1] != 3:
        raise ValueError(f"points must have shape (samples, 3), got {points_array.shape}")
    if times_array.ndim != 1:
        raise ValueError(f"times must have shape (samples,), got {times_array.shape}")
    _run_animation_process(3, points_array, times_array, axis_names, speed)


def _solution_trajectory(
    solution: Any, columns: tuple[int, ...]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Detach selected state columns without invoking solver work again."""

    states_method = getattr(solution, "states_array", None)
    times_method = getattr(solution, "times_array", None)
    if not callable(states_method) or not callable(times_method):
        raise TypeError("solution must provide states_array() and times_array()")
    states = np.asarray(states_method(), dtype=np.float64)
    times = np.asarray(times_method(), dtype=np.float64)
    if states.ndim != 2:
        raise ValueError("solution states must have shape (samples, state_dimension)")
    if len(columns) not in (2, 3):
        raise ValueError("select exactly two or three state columns")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in columns):
        raise TypeError("state column indexes must be integers")
    if any(index < 0 or index >= states.shape[1] for index in columns):
        raise ValueError("selected state column is outside the solution dimension")
    return states[:, columns], times


def animate_solution_2d(
    solution: Any,
    columns: tuple[int, int],
    axis_names: tuple[str, str] | None = None,
    speed: float = 1.0,
) -> None:
    """Animate two selected state components from a detached IVP solution."""

    points, times = _solution_trajectory(solution, columns)
    animate_trajectory_2d(points, times, axis_names, speed)


def animate_solution_3d(
    solution: Any,
    columns: tuple[int, int, int],
    axis_names: tuple[str, str, str] | None = None,
    speed: float = 1.0,
) -> None:
    """Animate three selected state components from a detached IVP solution."""

    points, times = _solution_trajectory(solution, columns)
    animate_trajectory_3d(points, times, axis_names, speed)


def _owned_sequence(value: Any) -> Any:
    """Convert NumPy-like containers before crossing into the Rust API."""

    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else value


def _method_name(method: str) -> str:
    normalized = method.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "bdf": "bdf",
        "radau": "radau5",
        "radau5": "radau5",
        "be": "backward_euler",
        "backward_euler": "backward_euler",
        "lsode": "lsode2",
        "lsode2": "lsode2",
        "lsoda": "lsode2",
        "rk45": "rk45",
        "dopri": "dopri",
        "ab4": "ab4",
    }
    if normalized not in aliases:
        raise ValueError(
            "unsupported IVP method; expected rk45, dopri, ab4, bdf, radau5, "
            "backward_euler, lsode2 or lsoda"
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
    controller: str | None = None,
    **solver_options: Any,
) -> _IvpSolver:
    """Construct a native IVP solver with one common Python signature.

    This convenience function normalizes solver-family names and forwards
    solver-specific options to the native adapter. It does not erase native
    lifecycle differences: call ``prepare()`` and ``solve()`` on the returned
    object when reuse, restart, or continuation is required.

    Parameters
    ----------
    method:
        Solver family: ``"rk45"``, ``"dopri"``, ``"ab4"``, ``"bdf"``,
        ``"radau5"``, ``"backward_euler"`` or ``"lsode2"``. ``"radau"``,
        ``"be"``, ``"lsode"`` and ``"lsoda"`` are accepted aliases. The
        ``"lsoda"`` alias selects the automatic Adams/BDF controller unless
        ``controller`` explicitly overrides it.
    expressions, state_variables, time_variable:
        Symbolic right-hand sides and their ordered state/time names.
    y0:
        Initial state. NumPy-like arrays are converted to owned native vectors.
        Radau accepts the value here for API symmetry but binds it at ``solve``.
    t0, t_end, max_step:
        Integration interval and step control, in the model's time units. For
        ODE_api2 nonstiff methods this is the native fixed starting step size;
        other solver families interpret it according to their own contracts.
    parameters, parameter_values:
        Optional ordered parameter schema and current numeric values.
    options:
        Optional ``SolverOptions`` for frontend, execution, layout, callback
        scheduling and telemetry. Solver-specific keywords remain available
        through ``solver_options``.
    controller:
        LSODE2 controller only: ``"bdf"``, ``"adams"`` or ``"auto"``. This
        is separate from ``method``, which selects the solver family.

    Returns
    -------
    _IvpSolver
        The selected native solver wrapper; its concrete class depends on
        ``method`` and retains that solver's prepared/restart/continuation API.

    Raises
    ------
    ValueError
        If the family is unknown or ``controller`` is supplied to a solver
        other than LSODE2. Native configuration failures retain their typed
        RustedSciThe exception class.
    """

    family = _method_name(method)
    if family in {"rk45", "dopri", "ab4"}:
        if controller is not None:
            raise ValueError("controller is only supported by the LSODE2 route")
        if parameters is not None or parameter_values is not None:
            raise _native.UnsupportedOptionError(
                "ODE_api2 nonstiff does not expose symbolic parameter rebinding"
            )
        return _native.UniversalNonStiffSolver(
            family, expressions, state_variables, time_variable,
            _owned_sequence(y0), t0, t_end, max_step,
            options=options, **solver_options,
        )
    if family == "bdf":
        if controller is not None:
            raise ValueError("controller is only supported by the LSODE2 route")
        return _native.BdfSolver(
            expressions, state_variables, time_variable, _owned_sequence(y0), t0, t_end, max_step,
            options=options, parameters=parameters,
            parameter_values=_owned_sequence(parameter_values), **solver_options,
        )
    if family == "radau5":
        if controller is not None:
            raise ValueError("controller is only supported by the LSODE2 route")
        # Radau owns y0 at solve time; it is not part of symbolic preparation.
        return _native.RadauSolver(
            expressions, state_variables, time_variable, t0, t_end, max_step,
            options=options, parameters=parameters, **solver_options,
        )
    if family == "backward_euler":
        if controller is not None:
            raise ValueError("controller is only supported by the LSODE2 route")
        return _native.BackwardEulerSolver(
            expressions, state_variables, time_variable, _owned_sequence(y0), t0, t_end,
            h=max_step, options=options, parameters=parameters,
            parameter_values=_owned_sequence(parameter_values), **solver_options,
        )
    lsoda_alias = (
        method.strip().lower().replace("-", "_").replace(" ", "_") == "lsoda"
    )
    selected_controller = (
        controller if controller is not None else ("auto" if lsoda_alias else "bdf")
    )
    return Lsode2Solver(
        expressions, state_variables, time_variable, _owned_sequence(y0), t0, t_end, max_step,
        options=options, method=selected_controller, parameters=parameters,
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
    controller: str | None = None,
    **solver_options: Any,
) -> _IvpSolution:
    """Solve one IVP and return the solver's detached native result.

    Arguments have the same meaning as :func:`create_ivp`. This helper is for
    one-shot solves; use ``create_ivp`` when you need explicit preparation,
    continuation or restart. For Radau, ``y0`` is passed to ``solve`` because
    the native solver binds its initial state at execution time. For other
    families it is retained by the constructed solver.

    Returns
    -------
    _IvpSolution
        A detached solver-specific solution containing trajectory data and
        the telemetry snapshot selected in ``options``.

    Raises
    ------
    RustedSciTheError
        Typed symbolic preparation, callback, numerical, exhaustion or AOT
        lifecycle failure from the native solver.
    """

    solver = create_ivp(
        method, expressions, state_variables, time_variable, y0, t0, t_end,
        max_step, parameters=parameters, parameter_values=parameter_values,
        options=options, controller=controller, **solver_options,
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
    expressions: Sequence[Expr],
    state_variables: Sequence[str],
    boundary_data: Sequence[Expr] | Mapping[str, Sequence[tuple[int, float]]],
    mesh: ArrayLike,
    initial_state: ArrayLike,
    *,
    options: SolverOptions | None = None,
    **solver_options: Any,
) -> _native.BvpSciSolver | _native.BvpDampSolver:
    """Construct a BVP solver while preserving native solver options.

    The two solver families intentionally retain different boundary-data
    contracts. ``bvp_sci`` accepts an ordered list of symbolic boundary
    residuals (using its native ``ya_<state>``/``yb_<state>`` symbols), while
    ``bvp_damp`` accepts a mapping from state name to ``(position, value)``
    observations. ``options`` and remaining keywords are forwarded to the
    selected adapter, so native layout/frontend controls are not hidden.

    Parameters
    ----------
    method:
        ``"bvp_sci"``/``"scipy"`` or ``"bvp_damp"``/``"damped"``.
    expressions, state_variables:
        Ordered symbolic RHS expressions and state names.
    boundary_data:
        For ``bvp_sci``, a sequence of symbolic boundary residuals using
        ``ya_<state>``/``yb_<state>`` names. For ``bvp_damp``, a mapping from
        state name to endpoint observations ``(0, value)`` or ``(1, value)``.
    mesh:
        Initial strictly increasing mesh; array-like inputs are copied before
        entering Rust.
    initial_state:
        Initial state values on the mesh, in the selected solver's native
        expected shape.
    options:
        Optional common ``SolverOptions``. Family-specific keywords are passed
        unchanged through ``solver_options``.

    Returns
    -------
    BvpSciSolver | BvpDampSolver
        The selected native solver object. Call ``solve()`` to compute the
        solution; retain the solver when using its supported continuation or
        restart API.

    Raises
    ------
    ValueError
        If ``method`` is not recognized. Native validation and solve failures
        use typed RustedSciThe exceptions.
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
    expressions: Sequence[Expr],
    state_variables: Sequence[str],
    boundary_data: Sequence[Expr] | Mapping[str, Sequence[tuple[int, float]]],
    mesh: ArrayLike,
    initial_state: ArrayLike,
    *,
    options: SolverOptions | None = None,
    **solver_options: Any,
) -> _native.BvpSciSolution | _native.BvpDampSolution:
    """Solve one BVP using the selected native solver family.

    The input contract matches :func:`create_bvp`; this convenience function
    immediately calls ``solve()`` and returns the detached solution. Use
    :func:`create_bvp` when solver state must be retained for later native
    lifecycle operations.

    Returns
    -------
    BvpSciSolution | BvpDampSolution
        A solver-specific result with mesh/state data and native diagnostics.

    Raises
    ------
    RustedSciTheError
        Typed preparation, configuration, callback, linear-solver or
        convergence failure.
    """

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
) -> _native.NonlinearSolution:
    """Create and solve a square symbolic nonlinear system.

    ``system`` is prepared through the native nonlinear frontend and the
    initial guess is copied from any NumPy-like input. Pass solver controls
    such as ``method``, ``frontend``, ``execution_policy``, ``tolerance`` and
    ``max_iterations`` as keywords; use :func:`create_nonlinear` if the
    prepared solver must be reused.

    Returns
    -------
    NonlinearSolution
        A detached native nonlinear-solve result with solution and diagnostics.

    Raises
    ------
    RustedSciTheError
        Typed configuration, preparation, callback, numerical or exhaustion
        failure from the native solver.
    """

    solver = create_nonlinear(system, **solver_options)
    return solver.solve(_owned_sequence(initial_guess), _owned_sequence(parameters))


def create_nonlinear(
    system: SymbolicSystem, **solver_options: Any
) -> _native.NonlinearSolver:
    """Construct a nonlinear solver while leaving preparation explicit.

    Native options are forwarded unchanged. Call ``prepare()`` before repeated
    solves when you want to make symbolic setup explicit, then use the solver's
    parameter-binding methods for subsequent numeric problems.

    Returns
    -------
    NonlinearSolver
        The native wrapper with its preparation and
        telemetry methods.

    Raises
    ------
    RustedSciTheError
        Typed configuration or symbolic preparation failure.
    """

    return _native.NonlinearSolver(system, **solver_options)


def fit(
    equations: list[Expr],
    initial_guess: ArrayLike,
    *,
    parameters: ArrayLike | None = None,
    **solver_options: Any,
) -> _native.FittingSolution:
    """Solve a symbolic least-squares fitting problem.

    ``equations`` are residual expressions, not model predictions paired with
    an implicit observation vector. The initial guess and optional bound
    parameters are copied from array-like inputs. Method, frontend, bounds,
    tolerances, telemetry and iteration controls are forwarded to the native
    fitting solver. Use :func:`create_fit` for explicit preparation or repeated
    parameter-bound solves.

    Returns
    -------
    FittingSolution
        A native fitting result with fitted values, termination and telemetry.

    Raises
    ------
    RustedSciTheError
        Typed invalid-problem, preparation, callback, numerical or budget
        exhaustion failure.
    """

    solver = create_fit(equations, _owned_sequence(initial_guess), **solver_options)
    return solver.solve() if parameters is None else solver.solve_with_parameters(_owned_sequence(parameters))


def create_fit(
    equations: list[Expr],
    initial_guess: ArrayLike,
    **solver_options: Any,
) -> _native.FittingSolver:
    """Construct the native symbolic least-squares fitting solver.

    Expressions and solver configuration are retained by the Rust object;
    Python does not build a duplicate symbolic tree. Construction performs
    native symbolic preparation eagerly; call ``solve()`` to fit, then use
    ``solve_with_parameters()`` for subsequent parameter rebinding. The native
    object does not expose a separate ``prepare()`` method. Solver-specific
    keywords are forwarded to ``FittingSolver`` without imposing a reduced
    facade API.

    Returns
    -------
    FittingSolver
        A native fitting solver with preparation, solve, telemetry and
        parameter-rebind methods.

    Raises
    ------
    RustedSciTheError
        Typed validation or symbolic preparation failure.
    """

    return _native.FittingSolver(
        equations,
        _owned_sequence(initial_guess),
        **solver_options,
    )


def task(source_or_path: str | Path) -> TaskDocument:
    """Parse a native task document from inline text or a file path.

    A string containing a newline is always treated as document text. A
    newline-free string is treated as a path only when it names an existing
    file; otherwise it is parsed as inline input. ``Path`` objects are checked
    as files first. Parsing does not execute the task; call ``preview()``,
    ``check()`` or the matching run method on the returned document.

    Raises
    ------
    TaskDocumentError
        Syntax, semantic validation, file access or unsupported-route failure.
    """

    raw = str(source_or_path)
    if isinstance(source_or_path, Path) or ("\n" not in raw and "\r" not in raw):
        candidate = Path(source_or_path)
        if candidate.exists() and candidate.is_file():
            return TaskDocument.from_file(str(candidate))
    return TaskDocument.parse(raw)


def task_dict(source_or_path: str | Path) -> dict[str, Any]:
    """Parse a native task document and return its detached inspection dict.

    Parsing and semantic normalization remain in the Rust command-interpreter
    parser. This helper only resolves the path-versus-inline convenience and
    serializes the already parsed ``TaskDocument``; it never executes a task.
    """

    return task(source_or_path).to_dict()


def capabilities(solver: str) -> _native.SolverCapabilities:
    """Return the native capability descriptor for a solver family.

    Use this before constructing a solver when configuration is selected
    dynamically. The descriptor reports supported frontends, execution routes,
    layouts, callbacks, continuation and parallel policies; it is authoritative
    for the Python wrapper as well as the Rust API.

    Raises
    ------
    ValueError
        If ``solver`` is not a recognized family name.
    """

    return _native.capabilities(solver)


def render_table(rows: Iterable[Mapping[str, Any]], columns: Iterable[str] | None = None) -> str:
    """Render compact deterministic ASCII tables without capturing process noise.

    The function is intentionally dependency-free and accepts detached scalar
    values, so it can be used in a long-running batch even when an optional
    plotting or reporting package is unavailable.

    Parameters
    ----------
    rows:
        Iterable of mappings. Values are converted with ``str``; nested solver
        objects should be reduced to scalar diagnostics by the caller.
    columns:
        Optional explicit column order. If omitted, keys from the first row
        define the order. Missing values render as empty strings.

    Returns
    -------
    str
        An ASCII table, or an empty string when ``rows`` or the selected
        columns are empty.
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
    """Write a compact deterministic ASCII table to ``path``.

    The file contains only the rendered table; this helper does not redirect
    stdout/stderr or capture native compiler output. ``columns`` fixes column
    order and missing row values render as empty cells. Parent directories are
    not created automatically.

    Raises
    ------
    OSError
        If the destination cannot be written.
    """

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

    Returns
    -------
    list[dict[str, Any]]
        One row per input, preserving the result object on success and the
        exception class/message on failure. When ``continue_on_error`` is
        false, the failing row is appended internally and the exception is
        re-raised instead of returning the partial batch.
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

    Returns
    -------
    str
        The compact table text, including an empty string for an empty batch.

    Raises
    ------
    OSError
        If ``path`` is supplied and the table cannot be written. Task failures
        remain rows when ``continue_on_error`` is true; otherwise the first
        task exception is re-raised by :func:`run_task_batch`.
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

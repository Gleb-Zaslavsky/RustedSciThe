"""Opt-in observational instrumentation for an isolated SciPy _bvp module.

By default the adapter wraps Python call boundaries. A parity-only option can
also observe line events within solve_newton to record trial and stopping
ratios. It does not patch formulas, controller decisions, callback arguments,
or return values. Timings are diagnostic and nested scopes are inclusive.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
import inspect
import sys
from time import perf_counter
from typing import Any, Callable, Iterator


_ACTIVE: ContextVar["BvpTelemetry | None"] = ContextVar("bvp_telemetry", default=None)
_PHASE: ContextVar[str | None] = ContextVar("bvp_telemetry_phase", default=None)
_MESH_PASS: ContextVar[dict[str, Any] | None] = ContextVar("bvp_telemetry_mesh_pass", default=None)

_STAGES = (
    "solve_bvp",
    "estimate_fun_jac",
    "estimate_bc_jac",
    "construct_global_jac",
    "collocation_fun",
    "prepare_sys",
    "solve_newton",
    "estimate_rms_residuals",
    "modify_mesh",
    "create_spline",
)


@dataclass
class BvpTelemetry:
    """Per-solve counters and inclusive stage timings for SciPy BVP."""

    enabled: bool = True
    counters: dict[str, int] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    capture_newton_trace: bool = False
    newton_trace: list[dict[str, Any]] = field(default_factory=list)
    newton_convergence_trace: list[dict[str, Any]] = field(default_factory=list)
    mesh_passes: list[dict[str, Any]] = field(default_factory=list)

    def count(self, name: str, amount: int = 1) -> None:
        if self.enabled:
            self.counters[name] = self.counters.get(name, 0) + amount
            active = _MESH_PASS.get()
            if active is not None and active.get("owner") == id(self):
                counters = active["counters"]
                counters[name] = counters.get(name, 0) + amount

    def begin_mesh_pass(self, state_dimension: int, mesh_nodes: int,
                        parameter_count: int) -> dict[str, Any]:
        """Start attribution for one SciPy outer solve pass on its current mesh."""
        self.finish_mesh_pass()
        record: dict[str, Any] = {
            "owner": id(self),
            "mesh_pass": len(self.mesh_passes),
            "mesh_nodes": int(mesh_nodes),
            "linear_system_dimension": int(state_dimension * mesh_nodes + parameter_count),
            "counters": {
                name: 0 for name in (
                    "solve_newton_calls", "construct_global_jac_calls",
                    "factorization_calls", "linear_solve_calls", "prepare_sys_calls",
                    "rhs_callback_calls", "rhs_callback_state_points",
                    "boundary_callback_calls", "rhs_jacobian_callback_calls",
                    "rhs_jacobian_callback_state_points",
                    "boundary_jacobian_callback_calls",
                )
            },
            "timings_ms": {},
            "_started_at": perf_counter(),
        }
        self.mesh_passes.append(record)
        _MESH_PASS.set(record)
        return record

    def finish_mesh_pass(self) -> None:
        active = _MESH_PASS.get()
        if active is not None and active.get("owner") == id(self):
            started = active.pop("_started_at", None)
            if started is not None:
                active["elapsed_ms"] = (perf_counter() - started) * 1e3

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        start = perf_counter()
        self.count(f"{name}_calls")
        try:
            yield
        finally:
            elapsed_ms = (perf_counter() - start) * 1e3
            self.timings_ms[name] = self.timings_ms.get(name, 0.0) + elapsed_ms
            active = _MESH_PASS.get()
            if (name != "solve_bvp" and active is not None
                    and active.get("owner") == id(self)):
                timings = active["timings_ms"]
                timings[name] = timings.get(name, 0.0) + elapsed_ms

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        token = _PHASE.set(name)
        try:
            yield
        finally:
            _PHASE.reset(token)

    def snapshot(self) -> dict[str, Any]:
        """Return stable, sorted counters and timings for compact reporting."""
        snapshot = {
            "counters": dict(sorted(self.counters.items())),
            "timings_ms": dict(sorted(self.timings_ms.items())),
        }
        if self.capture_newton_trace:
            snapshot["newton_trace"] = list(self.newton_trace)
            snapshot["newton_convergence_trace"] = list(
                self.newton_convergence_trace
            )
        if self.enabled:
            snapshot["mesh_passes"] = [
                {
                    **{key: value for key, value in record.items() if key != "owner"},
                    "counters": dict(sorted(record["counters"].items())),
                    "timings_ms": dict(sorted(record["timings_ms"].items())),
                }
                for record in self.mesh_passes
            ]
        return snapshot


def _call_with_newton_trace(
    function: Callable[..., Any], telemetry: BvpTelemetry, *args: Any, **kwargs: Any
) -> Any:
    """Capture SciPy trial decisions only for explicitly requested parity runs."""
    source, first_line = inspect.getsourcelines(function)
    cost_line = next(
        first_line + offset
        for offset, text in enumerate(source)
        if text.lstrip().startswith("cost_new = np.dot")
    )
    decision_line = next(
        first_line + offset
        for offset, text in enumerate(source)
        if first_line + offset > cost_line
        and text.lstrip().startswith("if cost_new <")
    )
    convergence_line = next(
        first_line + offset
        for offset, text in enumerate(source)
        if text.lstrip().startswith("if (np.all(np.abs(col_res)")
    )
    target_code = function.__code__
    previous_trace = sys.gettrace()

    def trace(frame: Any, event: str, _arg: Any) -> Any:
        if event == "call" and frame.f_code is target_code:
            def trace_newton(local_frame: Any, local_event: str, _local_arg: Any) -> Any:
                if local_event == "line" and local_frame.f_lineno == decision_line:
                    values = local_frame.f_locals
                    telemetry.newton_trace.append({
                        "mesh_pass": (_MESH_PASS.get() or {}).get("mesh_pass", 0),
                        "mesh_nodes": int(values["m"]),
                        "linear_system_dimension": int(values["n"] * values["m"] + values["p"].size),
                        "iteration": int(values["iteration"]),
                        "trial": int(values["trial"]),
                        "alpha": float(values["alpha"]),
                        "cost": float(values["cost"]),
                        "cost_new": float(values["cost_new"]),
                        "accepted": bool(values["cost_new"] <
                                          (1 - 2 * values["alpha"] * values["sigma"])
                                          * values["cost"]),
                        "njev": int(values["njev"]),
                    })
                elif local_event == "line" and local_frame.f_lineno == convergence_line:
                    values = local_frame.f_locals
                    collocation_ratio = max(
                        abs(float(values["col_res"][state, interval]))
                        / (
                            float(values["tol_r"][interval])
                            * (
                                1.0
                                + abs(float(values["f_middle"][state, interval]))
                            )
                        )
                        for state in range(values["n"])
                        for interval in range(values["m"] - 1)
                    )
                    boundary_ratio = max(
                        abs(float(value)) / float(values["bc_tol"])
                        for value in values["bc_res"]
                    )
                    telemetry.newton_convergence_trace.append({
                        "mesh_pass": (_MESH_PASS.get() or {}).get("mesh_pass", 0),
                        "mesh_nodes": int(values["m"]),
                        "linear_system_dimension": int(values["n"] * values["m"] + values["p"].size),
                        "iteration": int(values["iteration"]),
                        "njev": int(values["njev"]),
                        "max_normalized_collocation_residual": collocation_ratio,
                        "normalized_boundary_residual": boundary_ratio,
                        "converged": collocation_ratio < 1.0 and boundary_ratio < 1.0,
                    })
                    telemetry.count("newton_iterations")
                return trace_newton

            return trace_newton
        return None

    sys.settrace(trace)
    try:
        return function(*args, **kwargs)
    finally:
        sys.settrace(previous_trace)


def instrument_module(module: Any) -> Callable[..., Any]:
    """Wrap an independently loaded vendored module and return its solve API.

    Load a fresh module instance for each instrumented reference run. The
    ContextVars keep nested stage attribution isolated across threads/tasks;
    the module's function wrappers themselves are installed once.
    """

    for name in _STAGES[1:]:
        original = getattr(module, name)

        @wraps(original)
        def stage_wrapper(*args: Any, __name: str = name,
                          __original: Callable[..., Any] = original,
                          **kwargs: Any) -> Any:
            telemetry = _ACTIVE.get()
            if telemetry is None or not telemetry.enabled:
                return __original(*args, **kwargs)
            if __name == "prepare_sys":
                bound_stage = inspect.signature(__original).bind_partial(*args, **kwargs)
                telemetry.begin_mesh_pass(
                    int(bound_stage.arguments["n"]),
                    int(bound_stage.arguments["m"]),
                    int(bound_stage.arguments["k"]),
                )
            phase = telemetry.phase(__name) if __name in {
                "estimate_fun_jac", "estimate_bc_jac"
            } else _null_context()
            with telemetry.measure(__name), phase:
                if __name == "solve_newton" and telemetry.capture_newton_trace:
                    return _call_with_newton_trace(
                        __original, telemetry, *args, **kwargs
                    )
                return __original(*args, **kwargs)

        setattr(module, name, stage_wrapper)

    original_splu = module.splu

    @wraps(original_splu)
    def timed_splu(*args: Any, **kwargs: Any) -> Any:
        telemetry = _ACTIVE.get()
        if telemetry is None or not telemetry.enabled:
            return original_splu(*args, **kwargs)
        with telemetry.measure("factorization"):
            factor = original_splu(*args, **kwargs)

        class TimedFactor:
            def __getattr__(self, name: str) -> Any:
                return getattr(factor, name)

            def solve(self, *solve_args: Any, **solve_kwargs: Any) -> Any:
                with telemetry.measure("linear_solve"):
                    return factor.solve(*solve_args, **solve_kwargs)

        return TimedFactor()

    module.splu = timed_splu

    solve_bvp = module.solve_bvp

    @wraps(solve_bvp)
    def solve_with_telemetry(*args: Any, telemetry: BvpTelemetry | None = None,
                             **kwargs: Any) -> Any:
        if telemetry is None or not telemetry.enabled:
            return solve_bvp(*args, **kwargs)
        bound = inspect.signature(solve_bvp).bind_partial(*args, **kwargs)
        fun = bound.arguments.get("fun")
        bc = bound.arguments.get("bc")
        if fun is None or bc is None:
            raise TypeError("solve_bvp requires positional or keyword fun and bc")

        def timed_callback(kind: str, callback: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(callback)
            def call(*cb_args: Any, **cb_kwargs: Any) -> Any:
                phase = _PHASE.get()
                label = {
                    "rhs": "rhs_callback",
                    "boundary": "boundary_callback",
                    "rhs_jacobian": "rhs_jacobian_callback",
                    "boundary_jacobian": "boundary_jacobian_callback",
                }[kind]
                if phase == "estimate_fun_jac" and kind == "rhs":
                    telemetry.count("rhs_finite_difference_callback_calls")
                if phase == "estimate_bc_jac" and kind == "boundary":
                    telemetry.count("boundary_finite_difference_callback_calls")
                if kind in {"rhs", "rhs_jacobian"} and len(cb_args) > 1:
                    values = cb_args[1]
                    points = int(values.shape[-1]) if getattr(values, "ndim", 0) > 1 else 1
                    telemetry.count(f"{label}_state_points", points)
                with telemetry.measure(label):
                    return callback(*cb_args, **cb_kwargs)

            return call

        bound.arguments["fun"] = timed_callback("rhs", fun)
        bound.arguments["bc"] = timed_callback("boundary", bc)
        for argument, kind in (
            ("fun_jac", "rhs_jacobian"),
            ("bc_jac", "boundary_jacobian"),
        ):
            callback = bound.arguments.get(argument)
            if callback is not None:
                bound.arguments[argument] = timed_callback(kind, callback)

        token = _ACTIVE.set(telemetry)
        mesh_token = _MESH_PASS.set(None)
        try:
            with telemetry.measure("solve_bvp"):
                return solve_bvp(*bound.args, **bound.kwargs)
        finally:
            telemetry.finish_mesh_pass()
            _MESH_PASS.reset(mesh_token)
            _ACTIVE.reset(token)

    return solve_with_telemetry


@contextmanager
def _null_context() -> Iterator[None]:
    yield

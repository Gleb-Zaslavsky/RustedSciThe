"""Opt-in instrumentation for the vendored SciPy Radau reference.

This module records work around the reference solver without changing its
numerical formulas or SciPy's own ``nfev``, ``njev`` and ``nlu`` semantics.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterator


@dataclass
class RadauTelemetry:
    """Counters and non-additive inclusive timings for one Radau solve."""

    enabled: bool = True
    counters: dict[str, int] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    controller_trace: list[tuple[float, float, float, int, bool, bool]] = field(default_factory=list)
    _phase: str = "integration"

    def count(self, name: str, amount: int = 1) -> None:
        if self.enabled:
            self.counters[name] = self.counters.get(name, 0) + amount

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        started = perf_counter()
        try:
            yield
        finally:
            self.timings_ms[name] = self.timings_ms.get(name, 0.0) + (
                perf_counter() - started
            ) * 1_000.0

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        previous = self._phase
        self._phase = name
        try:
            yield
        finally:
            self._phase = previous

    def record_rhs(self, state_shape: tuple[int, ...]) -> None:
        """Count callback invocations separately from evaluated state columns."""
        if not self.enabled:
            return
        columns = state_shape[1] if len(state_shape) == 2 else 1
        self.count("rhs_callback_calls")
        self.count("rhs_state_evaluations", columns)
        if self._phase == "finite_difference":
            self.count("fd_rhs_callback_calls")
            self.count("fd_state_probes", columns)

    def record_controller_step(
        self,
        h_abs: float,
        error_norm: float,
        next_h_abs: float,
        newton_iterations: int,
        accepted: bool,
        jacobian_refreshed: bool,
    ) -> None:
        """Retain step and Newton-retry decisions for parity diagnosis."""
        if self.enabled:
            self.controller_trace.append(
                (
                    float(h_abs),
                    float(error_norm),
                    float(next_h_abs),
                    int(newton_iterations),
                    bool(accepted),
                    bool(jacobian_refreshed),
                )
            )

    def snapshot(self, solver: object) -> dict[str, object]:
        """Return standard SciPy counters alongside instrumentation counters."""
        return {
            "scipy_nfev": int(getattr(solver, "nfev")),
            "scipy_njev": int(getattr(solver, "njev")),
            "scipy_nlu": int(getattr(solver, "nlu")),
            "counters": dict(sorted(self.counters.items())),
            "timings_ms": dict(sorted(self.timings_ms.items())),
        }


@dataclass
class BdfTelemetry:
    """Opt-in BDF counters and inclusive stage timings for one solve."""

    enabled: bool = True
    max_trace_entries: int = 4096
    counters: dict[str, int] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    controller_trace: list[tuple[float, float, int, int, float, bool, bool, float]] = field(
        default_factory=list
    )
    trace_truncated: bool = False
    _phase: str = "integration"

    def count(self, name: str, amount: int = 1) -> None:
        if self.enabled:
            self.counters[name] = self.counters.get(name, 0) + amount

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        started = perf_counter()
        try:
            yield
        finally:
            self.timings_ms[name] = self.timings_ms.get(name, 0.0) + (
                perf_counter() - started
            ) * 1_000.0

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        previous = self._phase
        self._phase = name
        try:
            yield
        finally:
            self._phase = previous

    def instrument_rhs(self, fun):
        """Wrap the callback only when enabled, leaving SciPy counters intact."""
        if not self.enabled:
            return fun

        def counted_fun(t, y):
            columns = y.shape[1] if getattr(y, "ndim", 1) == 2 else 1
            self.count("rhs_callback_calls")
            self.count("rhs_state_evaluations", columns)
            if self._phase == "fd_baseline":
                self.count("fd_baseline_rhs_calls")
            elif self._phase == "fd_probe":
                self.count("fd_rhs_callback_calls")
                self.count("fd_state_probes", columns)
            with self.measure("rhs_callback"):
                if self._phase == "fd_baseline":
                    with self.measure("fd_baseline_rhs"):
                        return fun(t, y)
                if self._phase == "fd_probe":
                    with self.measure("fd_probe_rhs"):
                        return fun(t, y)
                if self._phase == "newton":
                    with self.measure("newton_rhs"):
                        return fun(t, y)
                return fun(t, y)

        return counted_fun

    def record_controller_attempt(
        self,
        t: float,
        h_abs: float,
        order: int,
        newton_iterations: int,
        error_norm: float,
        accepted: bool,
        jacobian_refreshed: bool,
        next_h_abs: float,
    ) -> None:
        if not self.enabled:
            return
        if len(self.controller_trace) >= self.max_trace_entries:
            self.trace_truncated = True
            return
        self.controller_trace.append(
            (
                float(t),
                float(h_abs),
                int(order),
                int(newton_iterations),
                float(error_norm),
                bool(accepted),
                bool(jacobian_refreshed),
                float(next_h_abs),
            )
        )

    def snapshot(self, solver: object) -> dict[str, object]:
        """Return unchanged SciPy-native counters plus diagnostic telemetry."""
        return {
            "scipy_nfev": int(getattr(solver, "nfev")),
            "scipy_njev": int(getattr(solver, "njev")),
            "scipy_nlu": int(getattr(solver, "nlu")),
            "status": str(getattr(solver, "status")),
            "t": float(getattr(solver, "t")),
            "order": int(getattr(solver, "order")),
            "h_abs": float(getattr(solver, "h_abs")),
            "counters": dict(sorted(self.counters.items())),
            "timings_ms": dict(sorted(self.timings_ms.items())),
            "controller_trace": list(self.controller_trace),
            "trace_truncated": self.trace_truncated,
        }

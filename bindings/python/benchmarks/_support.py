"""Strict accessors for telemetry consumed by Python boundary benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
import importlib.metadata
import math
from pathlib import Path
import sys

import rustedscithe
from rustedscithe import _native


def required_counter(snapshot: Mapping[str, int], key: str, route: str) -> float:
    """Read a measured counter without disguising a missing key as zero."""
    if key not in snapshot:
        raise RuntimeError(f"{route} telemetry is missing required counter {key!r}")
    return float(snapshot[key])


def required_timing(snapshot: Mapping[str, float], key: str, route: str) -> float:
    """Read a required timing stage from a route collecting detailed telemetry."""
    if key not in snapshot:
        raise RuntimeError(f"{route} telemetry is missing required timing {key!r}")
    return float(snapshot[key])


def wall_minus_native_ms(wall_ms: float, native_ms: float) -> float:
    """Return a signed wall/native attribution residual, not pure Python time.

    The residual can include Python dispatch, PyO3 conversion, native work
    outside solver timers, and measurement noise. Keep its sign: the measured
    wall interval and native timing scope are not guaranteed to be identical.
    """
    if not math.isfinite(wall_ms) or not math.isfinite(native_ms):
        raise ValueError("wall and native timings must be finite")
    return wall_ms - native_ms


def runtime_provenance() -> str:
    """Describe the interpreter and exact Python/native modules being timed."""
    try:
        package_version = importlib.metadata.version("rustedscithe")
    except importlib.metadata.PackageNotFoundError:
        package_version = "unknown"
    package_path = Path(rustedscithe.__file__).resolve()
    native_path = Path(_native.__file__).resolve()
    return (
        f"Python={sys.version.split()[0]}, rustedscithe={package_version}, "
        f"package_path=`{package_path}`, native_extension=`{native_path}`"
    )

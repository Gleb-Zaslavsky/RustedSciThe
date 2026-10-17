"""Structured preflight checks for optional RustedSciThe tools and bridges."""

from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Literal, TypedDict

from . import _native


class ToolProbe(TypedDict):
    """PATH executable availability and the result of its version probe."""

    status: Literal["available", "missing", "probe_error"]
    available: bool
    path: str | None
    version: str | None
    detail: str | None


class SuperLuProbe(TypedDict):
    """Build-feature and configured shared-library status for SuperLU."""

    feature_enabled: bool
    library_path: str | None
    status: Literal["not_configured", "missing", "loadable", "load_error"]
    loadable: bool | None
    detail: str | None


class DependencyReport(TypedDict):
    """Machine-readable, side-effect-free dependency preflight result."""

    platform: str
    tools: dict[str, ToolProbe]
    superlu: SuperLuProbe


_TOOLS: dict[str, tuple[str, tuple[str, ...]]] = {
    "gnuplot": ("gnuplot", ("--version",)),
    "tcc": ("tcc", ("-v",)),
    "gcc": ("gcc", ("--version",)),
    "zig": ("zig", ("version",)),
    "pandoc": ("pandoc", ("--version",)),
}


def _version_probe(executable: str, args: tuple[str, ...], timeout: float) -> ToolProbe:
    path = shutil.which(executable)
    if path is None:
        return {
            "status": "missing",
            "available": False,
            "path": None,
            "version": None,
            "detail": "executable was not found on PATH",
        }

    try:
        completed = subprocess.run(
            [path, *args],
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "status": "probe_error",
            "available": True,
            "path": path,
            "version": None,
            "detail": f"version probe failed: {error}",
        }

    version = next(
        (
            line.strip()
            for stream in (completed.stdout, completed.stderr)
            for line in stream.splitlines()
            if line.strip()
        ),
        None,
    )
    if completed.returncode != 0:
        return {
            "status": "probe_error",
            "available": True,
            "path": path,
            "version": version,
            "detail": f"version probe exited with status {completed.returncode}",
        }
    return {
        "status": "available",
        "available": True,
        "path": path,
        "version": version,
        "detail": None,
    }


def _superlu_probe() -> SuperLuProbe:
    feature_enabled = bool(_native.superlu_feature_enabled())
    configured = os.environ.get("SUPERLU_LIBRARY")
    if not configured:
        return {
            "feature_enabled": feature_enabled,
            "library_path": None,
            "status": "not_configured",
            "loadable": None,
            "detail": "SUPERLU_LIBRARY is not set",
        }

    path = Path(configured).expanduser()
    if not path.is_file():
        return {
            "feature_enabled": feature_enabled,
            "library_path": str(path),
            "status": "missing",
            "loadable": False,
            "detail": "configured SUPERLU_LIBRARY path is not a file",
        }

    try:
        ctypes.CDLL(str(path.resolve()))
    except OSError as error:
        return {
            "feature_enabled": feature_enabled,
            "library_path": str(path),
            "status": "load_error",
            "loadable": False,
            "detail": str(error),
        }
    return {
        "feature_enabled": feature_enabled,
        "library_path": str(path),
        "status": "loadable",
        "loadable": True,
        "detail": None,
    }


def check_dependencies(*, timeout: float = 5.0) -> DependencyReport:
    """Check PATH tools and the separately configured optional SuperLU bridge.

    Executable lookup uses :func:`shutil.which`, which honors ``PATHEXT`` on
    Windows and normal ``PATH`` resolution on Unix-like systems. SuperLU is
    not looked up as a command: its configured shared library is checked with
    the platform dynamic loader. This reports loadability, not ABI correctness
    or a successful numerical solve.

    The function has no console output and never changes process environment.
    """

    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")

    tools: dict[str, ToolProbe] = {
        name: _version_probe(executable, args, timeout)
        for name, (executable, args) in _TOOLS.items()
    }
    # Ensure the mapping stays a plain detached dict for Python callers.
    return {
        "platform": sys.platform,
        "tools": tools,
        "superlu": _superlu_probe(),
    }


__all__ = ["DependencyReport", "SuperLuProbe", "ToolProbe", "check_dependencies"]

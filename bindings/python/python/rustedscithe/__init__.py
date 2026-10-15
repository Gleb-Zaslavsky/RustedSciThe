"""Public Python facade for the RustedSciThe extension.

The compiled module is kept private as ``rustedscithe._native``.  Re-exporting
its classes here preserves the simple public import while allowing a small
Python-native convenience layer to live beside the extension in wheels.
"""

from . import _native
from ._native import *  # noqa: F401,F403 - stable extension surface
from .facade import (
    animate_solution_2d,
    animate_solution_3d,
    animate_trajectory_2d,
    animate_trajectory_3d,
    capabilities,
    create_bvp,
    create_ivp,
    create_fit,
    create_nonlinear,
    fit,
    render_table,
    run_task_batch,
    run_task_batch_report,
    solve_bvp,
    solve_ivp,
    solve_nonlinear,
    task,
    write_table,
)

__all__ = [
    name for name in dir(_native) if not name.startswith("_")
] + [
    "animate_solution_2d", "animate_solution_3d",
    "animate_trajectory_2d", "animate_trajectory_3d", "capabilities",
    "create_bvp", "create_fit", "create_ivp",
    "create_nonlinear", "fit", "render_table", "run_task_batch",
    "run_task_batch_report", "solve_bvp",
    "solve_ivp", "solve_nonlinear", "task", "write_table",
]

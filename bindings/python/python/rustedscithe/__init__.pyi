"""Typed public import surface for :mod:`rustedscithe`."""

from ._native import *
from .facade import (
    animate_solution_2d as animate_solution_2d,
    animate_solution_3d as animate_solution_3d,
    animate_trajectory_2d as animate_trajectory_2d,
    animate_trajectory_3d as animate_trajectory_3d,
    capabilities as capabilities,
    create_bvp as create_bvp,
    create_fit as create_fit,
    create_ivp as create_ivp,
    create_nonlinear as create_nonlinear,
    fit as fit,
    render_table as render_table,
    run_task_batch as run_task_batch,
    run_task_batch_report as run_task_batch_report,
    solve_bvp as solve_bvp,
    solve_ivp as solve_ivp,
    solve_nonlinear as solve_nonlinear,
    task as task,
    write_table as write_table,
)
from .facade import task_dict as task_dict
from .dependencies import check_dependencies as check_dependencies
from ._native import fit_quality as fit_quality
from ._native import LeastSquaresBSpline as LeastSquaresBSpline
from ._native import least_squares_bspline as least_squares_bspline
from ._native import lowess_smooth as lowess_smooth
from ._native import savgol_filter as savgol_filter
from ._native import spline_resample as spline_resample

__all__ = [
    "Expr", "ExprVector", "ExprMatrix", "BdfSolver", "BdfCallbackSolver", "BdfSolution",
    "BackwardEulerSolver", "BackwardEulerSolution", "UniversalNonStiffSolver", "UniversalNonStiffSolution", "BvpSciSolver",
    "BvpSciSolution", "BvpDampSolver", "BvpDampNaiveSolver", "BvpDampFrozenSolver", "BvpDampSolution", "RadauSolver",
    "RadauCallbackSolver", "RadauSolution", "LambdifiedExpr", "Lsode2Solver",
    "Lsode2Solution", "SymbolicSystem", "LambdifiedSystem", "NonlinearSolver",
    "NonlinearSolution", "FittingSolver", "FittingSolution", "GavinFittingSolver",
    "GavinFittingSolution", "VarProSolver", "VarProSolution", "UniversalFitting",
    "UniversalFittingSolution", "SymbolicFrontend",
    "ExecutionRoute", "MatrixLayout", "ExecutionPolicy", "TelemetryMode",
    "SolverOptions", "SolverCapabilities", "TaskDocument", "IvpTaskResult",
    "BvpTaskResult", "ShootingBvpSolver", "ShootingBvpSolution", "parse_expr", "symbol", "symbols", "indexed_symbols", "constant", "capabilities",
    "RustedSciTheError", "InvalidExpressionError", "UnsupportedOptionError",
    "UnsupportedSymbolicOperationError",
    "InvalidConfigurationError", "SolverError", "PreparationError",
    "AotArtifactError", "AotCompilerError", "AotLinkError", "AotTimeoutError",
    "AotLifecycleError", "CallbackError", "TaskDocumentError", "BvpError",
    "TelemetryError", "ExhaustionError", "PanicError", "PoisonedSolverError",
    "animate_solution_2d", "animate_solution_3d",
    "animate_trajectory_2d", "animate_trajectory_3d",
    "create_bvp", "create_fit", "create_ivp", "create_nonlinear", "fit",
    "render_table", "run_task_batch", "run_task_batch_report", "solve_bvp",
    "solve_ivp", "solve_nonlinear", "task", "task_dict", "check_dependencies",
    "superlu_feature_enabled", "write_table",
    "fit_quality", "lowess_smooth", "savgol_filter", "spline_resample",
    "LeastSquaresBSpline", "least_squares_bspline",
]


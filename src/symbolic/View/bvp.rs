//! View-native helpers for BVP discretization stages.
//!
//! This module hosts the Atom/AtomView equivalent of the legacy
//! `discretization_system_BVP_par()` path. The design goal is to move the BVP
//! pipeline onto the packed symbolic representation as early as possible:
//!
//! `Vec<Expr> -> Atom residual assembly -> Atom sparse Jacobian -> CodegenIR_atom`
//!
//! instead of materializing large intermediate `Expr` trees all the way until
//! Jacobian generation.
//!
//! ## Mathematical shape
//! For a first-order BVP system
//! `y'(x) = f(x, y(x))`
//! this module assembles the discrete residual equations
//! `y_{j+1} - y_j - h_j * Phi(f_j, f_{j+1}) = 0`,
//! where `Phi` is the selected one-step quadrature:
//! - `forward`: `Phi = f_j`
//! - `trapezoid`: `Phi = (f_j + f_{j+1}) / 2`
//!
//! Boundary conditions are applied symbolically after residual assembly, and
//! the set of active discrete unknowns per equation is tracked so later sparse
//! Jacobian construction can stay bandwidth- and sparsity-aware.
//!
//! ## Performance shape
//! Large BVP systems usually contain many residual rows, so the expensive
//! residual assembly stage is parallelized across mesh intervals. Timing
//! breakdown is preserved in the returned system so we can benchmark this View
//! path against the legacy `Expr` path stage by stage.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use super::{
    Atom,
    conversions::{approximate_f64_atom, expr_to_atom},
    state::Symbol,
    transform::{rename_and_bind_symbol, substitute_symbol_values},
};
use crate::symbolic::bvp::telemetry::BvpAtomDiscretizationTelemetrySnapshot;
use crate::symbolic::symbolic_engine::Expr;
use rayon::prelude::*;
use tabled::{builder::Builder, settings::Style};

/// Fallible construction errors for the Atom-native BVP discretization path.
///
/// The compatibility constructors below retain their historical panic-based
/// signatures. New solver code should use the `try_*` constructors so invalid
/// user configuration remains a typed error at the public boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BvpAtomDiscretizationError {
    /// The requested one-step discretization scheme is not supported.
    InvalidScheme(String),
    /// A residual references a discrete variable that is absent from the
    /// prepared active-variable layout.
    MissingVariables(Vec<String>),
}

impl std::fmt::Display for BvpAtomDiscretizationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidScheme(scheme) => {
                write!(formatter, "invalid BVP discretization scheme `{scheme}`")
            }
            Self::MissingVariables(names) => write!(
                formatter,
                "variables not found in atom-discretized system: {names:?}"
            ),
        }
    }
}

impl std::error::Error for BvpAtomDiscretizationError {}

fn validate_scheme(scheme: &str) -> Result<(), BvpAtomDiscretizationError> {
    match scheme {
        "forward" | "trapezoid" => Ok(()),
        other => Err(BvpAtomDiscretizationError::InvalidScheme(other.to_string())),
    }
}

/// Atom-native result of BVP discretization.
///
/// This mirrors the key assembled outputs of the legacy `Jacobian` path closely
/// enough that callers can compare and later bridge both implementations.
#[derive(Debug, Clone)]
pub struct DiscretizedBvpAtomSystem {
    /// Discrete residual equations assembled on the packed Atom path.
    pub vector_of_functions: Vec<Atom>,
    /// Free discrete unknowns after boundary-condition elimination.
    pub vector_of_variables: Vec<Atom>,
    /// Names of `vector_of_variables` in solver order.
    pub variable_string: Vec<String>,
    /// Per-row active variable names used later for sparse Jacobian construction.
    pub variables_for_all_discrete: Vec<Vec<String>>,
    /// Boundary conditions encoded as `(full_position, boundary_side, value)`.
    pub bc_pos_n_values: Vec<(usize, usize, f64)>,
    pub bounds: Option<Vec<(f64, f64)>>,
    pub rel_tolerance_vec: Option<Vec<f64>>,
    pub mesh_points: Vec<f64>,
    pub step_sizes: Vec<f64>,
    /// Raw stage timings in milliseconds.
    pub timer_hash: HashMap<String, f64>,
    /// Typed stage telemetry with sub-millisecond precision.
    pub telemetry: BvpAtomDiscretizationTelemetrySnapshot,
}

impl DiscretizedBvpAtomSystem {
    /// Returns the discretization timing table normalized to percent of total wall time.
    pub fn normalized_timer_hash(&self) -> HashMap<String, f64> {
        let mut timer_hash = self.timer_hash.clone();
        let total = *timer_hash.get("total time, ms").unwrap_or(&0.0);
        for key in [
            "bc handling",
            "discretization of equations",
            "flat list creation",
            "BC application",
            "consistency test",
            "bounds and tolerances",
        ] {
            if let Some(value) = timer_hash.get_mut(key) {
                normalize_timer_percent(value, total);
            }
        }
        timer_hash
    }

    /// Returns the typed Atom discretization telemetry used by new callers.
    pub fn telemetry_snapshot(&self) -> BvpAtomDiscretizationTelemetrySnapshot {
        self.telemetry
    }

    /// Renders the normalized timing table in the same human-readable style as the legacy path.
    pub fn render_timer_table(&self) -> String {
        let mut table = Builder::from(self.normalized_timer_hash()).build();
        table.with(Style::modern_rounded());
        table.to_string()
    }
}

/// Precomputed symbol-only input for Atom-native BVP discretization.
///
/// `Vec<Expr>` is intentionally absent here. It is accepted only by the
/// compatibility wrapper below, which converts the continuous equations once
/// before this input reaches the parallel row assembly.
#[derive(Debug, Clone)]
pub(crate) struct AtomDiscretizationInput {
    equations: Vec<Atom>,
    value_symbols: Vec<Symbol>,
    arg_symbol: Symbol,
    matrix_of_symbols: Vec<Vec<Symbol>>,
    matrix_of_atom_vars: Vec<Vec<Atom>>,
    rename_maps: Vec<HashMap<Symbol, Symbol>>,
}

impl AtomDiscretizationInput {
    fn from_atom_system(
        equations: Vec<Atom>,
        value_symbols: Vec<Symbol>,
        arg_symbol: Symbol,
        n_steps: usize,
    ) -> Self {
        let matrix_of_symbols = indexed_symbol_matrix_symbols(n_steps, &value_symbols);
        let matrix_of_atom_vars = matrix_of_symbols
            .iter()
            .map(|row| row.iter().copied().map(Atom::new_var).collect())
            .collect();
        let rename_maps = matrix_of_symbols
            .iter()
            .map(|row| {
                value_symbols
                    .iter()
                    .copied()
                    .zip(row.iter().copied())
                    .collect::<HashMap<_, _>>()
            })
            .collect();

        Self {
            equations,
            value_symbols,
            arg_symbol,
            matrix_of_symbols,
            matrix_of_atom_vars,
            rename_maps,
        }
    }

    fn from_expr_system(
        equations: Vec<Expr>,
        values: &[String],
        arg: &str,
        n_steps: usize,
    ) -> Self {
        let value_symbols = values
            .iter()
            .map(|value| Symbol::new(crate::wrap_symbol!(value.as_str())))
            .collect::<Vec<_>>();
        Self::from_atom_system(
            equations.iter().map(expr_to_atom).collect(),
            value_symbols,
            Symbol::new(crate::wrap_symbol!(arg)),
            n_steps,
        )
    }
}

/// Mesh-bound Atom-native BVP input used by the production Lambdify path.
#[derive(Debug, Clone)]
pub(crate) struct AtomBvpProblem {
    input: AtomDiscretizationInput,
    step_sizes: Vec<f64>,
    mesh_points: Vec<f64>,
}

impl AtomBvpProblem {
    fn from_atom_system(
        equations: Vec<Atom>,
        value_symbols: Vec<Symbol>,
        arg_symbol: Symbol,
        t0: f64,
        n_steps: Option<usize>,
        h: Option<f64>,
        mesh: Option<Vec<f64>>,
    ) -> Self {
        let (step_sizes, mesh_points, n_steps_total) = create_mesh_atom(n_steps, h, mesh, t0);
        Self {
            input: AtomDiscretizationInput::from_atom_system(
                equations,
                value_symbols,
                arg_symbol,
                n_steps_total,
            ),
            step_sizes,
            mesh_points,
        }
    }

    fn from_expr_system(
        equations: Vec<Expr>,
        values: &[String],
        arg: &str,
        t0: f64,
        n_steps: Option<usize>,
        h: Option<f64>,
        mesh: Option<Vec<f64>>,
    ) -> Self {
        let (step_sizes, mesh_points, n_steps_total) = create_mesh_atom(n_steps, h, mesh, t0);
        Self {
            input: AtomDiscretizationInput::from_expr_system(equations, values, arg, n_steps_total),
            step_sizes,
            mesh_points,
        }
    }
}

/// Build the discretized right-hand side for one equation and one mesh step.
///
/// This mirrors the legacy `Expr`-based `eq_step()` but stays on the packed
/// `Atom` path.
pub fn eq_step_atom(
    eq_i: &Atom,
    matrix_of_names: &[Vec<String>],
    values: &[String],
    arg: &str,
    j: usize,
    t: f64,
    scheme: &str,
) -> Atom {
    try_eq_step_atom(eq_i, matrix_of_names, values, arg, j, t, scheme)
        .unwrap_or_else(|error| panic!("Atom BVP discretization failed: {error}"))
}

/// Fallible Atom-native equivalent of [`eq_step_atom`].
pub fn try_eq_step_atom(
    eq_i: &Atom,
    matrix_of_names: &[Vec<String>],
    values: &[String],
    arg: &str,
    j: usize,
    t: f64,
    scheme: &str,
) -> Result<Atom, BvpAtomDiscretizationError> {
    validate_scheme(scheme)?;
    let value_symbols = values
        .iter()
        .map(|value| Symbol::new(crate::wrap_symbol!(value.as_str())))
        .collect::<Vec<_>>();
    let matrix_of_symbols = matrix_of_names
        .iter()
        .map(|row| {
            row.iter()
                .map(|name| Symbol::new(crate::wrap_symbol!(name.as_str())))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let rename_map = value_symbols
        .iter()
        .copied()
        .zip(matrix_of_symbols[j].iter().copied())
        .collect::<HashMap<_, _>>();
    let arg_symbol = Symbol::new(crate::wrap_symbol!(arg));
    let eq_step_j = rename_and_bind_symbol(eq_i, &rename_map, arg_symbol, t);

    Ok(match scheme {
        "forward" => eq_step_j,
        "trapezoid" => {
            let next_rename_map = value_symbols
                .iter()
                .copied()
                .zip(matrix_of_symbols[j + 1].iter().copied())
                .collect::<HashMap<_, _>>();
            let eq_step_j_plus_1 = rename_and_bind_symbol(eq_i, &next_rename_map, arg_symbol, t);
            let half = Atom::new_num(1) / Atom::new_num(2);
            atom_mul_raw(&half, &atom_add_raw(&eq_step_j, &eq_step_j_plus_1))
        }
        _ => unreachable!("scheme was validated before assembly"),
    })
}

fn eq_step_atom_with_maps(
    eq_i: &Atom,
    rename_maps: &[HashMap<Symbol, Symbol>],
    arg_symbol: Symbol,
    j: usize,
    t: f64,
    scheme: &str,
) -> Result<Atom, BvpAtomDiscretizationError> {
    validate_scheme(scheme)?;
    let eq_step_j = rename_and_bind_symbol(eq_i, &rename_maps[j], arg_symbol, t);

    Ok(match scheme {
        "forward" => eq_step_j,
        "trapezoid" => {
            let eq_step_j_plus_1 = rename_and_bind_symbol(eq_i, &rename_maps[j + 1], arg_symbol, t);
            let half = Atom::new_num(1) / Atom::new_num(2);
            atom_mul_raw(&half, &atom_add_raw(&eq_step_j, &eq_step_j_plus_1))
        }
        _ => unreachable!("scheme was validated before assembly"),
    })
}

fn normalize_timer_percent(value: &mut f64, total: f64) {
    if total <= f64::EPSILON {
        *value = 0.0;
    } else {
        *value /= total / 100.0;
    }
}

/// Build a product without invoking Atom arithmetic normalization.
///
/// This is intentionally local to discretization.  Normalized arithmetic is
/// preferable for ordinary symbolic work, but it cannot represent a singular
/// endpoint such as `z / x` after `x` has been bound to zero: coefficient
/// normalization would try to evaluate `0^-1`.  The resulting raw Atom is
/// still a valid symbolic structure and is handled by the same downstream
/// Atom/Lambdify pipeline as every other residual.
fn atom_mul_raw(lhs: &Atom, rhs: &Atom) -> Atom {
    let mut result = Atom::default();
    let product = result.to_mul();
    product.extend(lhs.as_view());
    product.extend(rhs.as_view());
    result
}

/// Build a raw n-ary sum without eager coefficient normalization.
fn atom_add_raw(lhs: &Atom, rhs: &Atom) -> Atom {
    let mut result = Atom::default();
    let sum = result.to_add();
    sum.extend(lhs.as_view());
    sum.extend(rhs.as_view());
    result
}

/// Build `lhs - rhs` as a raw Atom sum without eager normalization.
fn atom_sub_raw(lhs: &Atom, rhs: &Atom) -> Atom {
    let mut result = Atom::default();
    let sum = result.to_add();
    sum.extend(lhs.as_view());
    let minus_one = Atom::new_num(-1);
    let neg_rhs = atom_mul_raw(&minus_one, rhs);
    sum.extend(neg_rhs.as_view());
    result
}

/// Atom-native analogue of `discretization_system_BVP_par()`.
///
/// The heavy symbolic stages stay on the packed View path:
/// variable renaming, argument substitution, residual assembly, and BC
/// substitution all work on `Atom` directly. No conversion back to `Expr`
/// happens inside this function.
///
/// ## Algorithm
/// 1. Build the mesh and the step sizes `h_j`.
/// 2. Build the matrix of interned discretized symbols `y_i_j`.
/// 3. Pre-compute boundary-condition substitutions and the set of discrete
///    variables removed from the nonlinear unknown vector.
/// 4. Reuse the already converted RHS atoms and per-step rename maps.
/// 5. In parallel over mesh intervals, assemble residual rows:
///    - every row, including `t == 0`, uses the same Atom-native path.
/// 6. Apply boundary conditions to every residual row.
/// 7. Flatten the surviving discrete unknowns into solver order.
/// 8. Run a consistency check that every tracked active variable really exists
///    in the flattened solver variable list.
#[allow(clippy::too_many_arguments)]
pub fn discretization_system_bvp_par_atom(
    eq_system: Vec<Expr>,
    values: Vec<String>,
    arg: String,
    t0: f64,
    n_steps: Option<usize>,
    h: Option<f64>,
    mesh: Option<Vec<f64>>,
    border_conditions: HashMap<String, Vec<(usize, f64)>>,
    bounds: Option<HashMap<String, (f64, f64)>>,
    rel_tolerance: Option<HashMap<String, f64>>,
    scheme: String,
) -> DiscretizedBvpAtomSystem {
    try_discretization_system_bvp_par_atom(
        eq_system,
        values,
        arg,
        t0,
        n_steps,
        h,
        mesh,
        border_conditions,
        bounds,
        rel_tolerance,
        scheme,
    )
    .unwrap_or_else(|error| panic!("Atom BVP discretization failed: {error}"))
}

/// Fallible compatibility entry point for the Expr-to-Atom BVP route.
///
/// New solver/parser code should use this function so malformed user input is
/// returned as a typed error instead of becoming a process panic.
#[allow(clippy::too_many_arguments)]
pub fn try_discretization_system_bvp_par_atom(
    eq_system: Vec<Expr>,
    values: Vec<String>,
    arg: String,
    t0: f64,
    n_steps: Option<usize>,
    h: Option<f64>,
    mesh: Option<Vec<f64>>,
    border_conditions: HashMap<String, Vec<(usize, f64)>>,
    bounds: Option<HashMap<String, (f64, f64)>>,
    rel_tolerance: Option<HashMap<String, f64>>,
    scheme: String,
) -> Result<DiscretizedBvpAtomSystem, BvpAtomDiscretizationError> {
    let problem = AtomBvpProblem::from_expr_system(eq_system, &values, &arg, t0, n_steps, h, mesh);
    let border_conditions = border_conditions
        .into_iter()
        .map(|(name, conditions)| (Symbol::new(crate::wrap_symbol!(name.as_str())), conditions))
        .collect();
    try_assemble_atom_bvp_problem(problem, border_conditions, bounds, rel_tolerance, scheme)
}

/// Assemble an already converted and symbol-indexed BVP without crossing
/// through the compatibility `Expr` representation.
#[allow(clippy::too_many_arguments)]
pub fn discretization_system_bvp_par_atom_native(
    eq_system: Vec<Atom>,
    values: Vec<Symbol>,
    arg: Symbol,
    t0: f64,
    n_steps: Option<usize>,
    h: Option<f64>,
    mesh: Option<Vec<f64>>,
    border_conditions: HashMap<Symbol, Vec<(usize, f64)>>,
    bounds: Option<HashMap<String, (f64, f64)>>,
    rel_tolerance: Option<HashMap<String, f64>>,
    scheme: String,
) -> DiscretizedBvpAtomSystem {
    try_discretization_system_bvp_par_atom_native(
        eq_system,
        values,
        arg,
        t0,
        n_steps,
        h,
        mesh,
        border_conditions,
        bounds,
        rel_tolerance,
        scheme,
    )
    .unwrap_or_else(|error| panic!("Atom BVP discretization failed: {error}"))
}

/// Fallible entry point for the fully Atom-native BVP route.
#[allow(clippy::too_many_arguments)]
pub fn try_discretization_system_bvp_par_atom_native(
    eq_system: Vec<Atom>,
    values: Vec<Symbol>,
    arg: Symbol,
    t0: f64,
    n_steps: Option<usize>,
    h: Option<f64>,
    mesh: Option<Vec<f64>>,
    border_conditions: HashMap<Symbol, Vec<(usize, f64)>>,
    bounds: Option<HashMap<String, (f64, f64)>>,
    rel_tolerance: Option<HashMap<String, f64>>,
    scheme: String,
) -> Result<DiscretizedBvpAtomSystem, BvpAtomDiscretizationError> {
    let problem = AtomBvpProblem::from_atom_system(eq_system, values, arg, t0, n_steps, h, mesh);
    try_assemble_atom_bvp_problem(problem, border_conditions, bounds, rel_tolerance, scheme)
}

#[allow(clippy::too_many_arguments)]
fn try_assemble_atom_bvp_problem(
    problem: AtomBvpProblem,
    border_conditions: HashMap<Symbol, Vec<(usize, f64)>>,
    bounds: Option<HashMap<String, (f64, f64)>>,
    rel_tolerance: Option<HashMap<String, f64>>,
    scheme: String,
) -> Result<DiscretizedBvpAtomSystem, BvpAtomDiscretizationError> {
    validate_scheme(&scheme)?;
    let total_start = Instant::now();
    let mut timer_hash: HashMap<String, f64> = HashMap::new();
    let step_sizes = &problem.step_sizes;
    let mesh_points = &problem.mesh_points;
    let n_steps_total = mesh_points.len();
    let input = &problem.input;

    let bc_handling = Instant::now();
    let bc_lookup: HashMap<Symbol, HashMap<usize, f64>> = border_conditions
        .into_iter()
        .map(|(k, v)| (k, v.into_iter().collect()))
        .collect();

    let mut bc_value_map: HashMap<Symbol, f64> = HashMap::default();
    let mut vars_to_exclude: HashSet<Symbol> = HashSet::default();
    let mut bc_pos_n_values = Vec::new();

    for (var_name, conditions) in &bc_lookup {
        if let Some(var_idx) = input.value_symbols.iter().position(|v| v == var_name) {
            for (&pos, &value) in conditions {
                match pos {
                    0 => {
                        let discretized_symbol = input.matrix_of_symbols[0][var_idx];
                        bc_value_map.insert(discretized_symbol, value);
                        vars_to_exclude.insert(discretized_symbol);
                        let full_pos = var_idx;
                        bc_pos_n_values.push((full_pos, 0usize, value));
                    }
                    1 => {
                        let discretized_symbol =
                            input.matrix_of_symbols[n_steps_total - 1][var_idx];
                        bc_value_map.insert(discretized_symbol, value);
                        vars_to_exclude.insert(discretized_symbol);
                        let full_pos = (n_steps_total - 1) * input.value_symbols.len() + var_idx;
                        bc_pos_n_values.push((full_pos, 1usize, value));
                    }
                    _ => {}
                }
            }
        }
    }
    let bc_handling_elapsed = bc_handling.elapsed();
    timer_hash.insert(
        "bc handling".to_string(),
        bc_handling_elapsed.as_millis() as f64,
    );

    let discretization_start = Instant::now();
    let assembled_rows: Result<Vec<Vec<(Atom, Vec<Symbol>)>>, BvpAtomDiscretizationError> = (0
        ..(n_steps_total - 1))
        .into_par_iter()
        .map(|j| {
            let t = mesh_points[j];
            input
                .equations
                .iter()
                .enumerate()
                .map(|(i, eq_i)| {
                    let mut vars_in_equation = Vec::new();
                    let y_j_plus_1 = input.matrix_of_symbols[j + 1][i];
                    let y_j = input.matrix_of_symbols[j][i];

                    if !vars_to_exclude.contains(&y_j_plus_1) {
                        vars_in_equation.push(y_j_plus_1);
                    }
                    if !vars_to_exclude.contains(&y_j) {
                        vars_in_equation.push(y_j);
                    }
                    for var_idx in 0..input.value_symbols.len() {
                        let var_symbol = input.matrix_of_symbols[j][var_idx];
                        if !vars_to_exclude.contains(&var_symbol) {
                            vars_in_equation.push(var_symbol);
                        }
                    }

                    let eq_step_j = eq_step_atom_with_maps(
                        eq_i,
                        &input.rename_maps,
                        input.arg_symbol,
                        j,
                        t,
                        &scheme,
                    )?;
                    // Keep this assembly structural rather than normalizing it through
                    // Atom's arithmetic operators.  A singular endpoint can legitimately
                    // contain a symbolic `0^-1` term (the same representation retained by
                    // ExprLegacy); eager coefficient normalization would turn it into a
                    // division-by-zero panic before the solver gets a chance to apply its
                    // endpoint policy.
                    let residual = atom_sub_raw(
                        &atom_sub_raw(
                            &input.matrix_of_atom_vars[j + 1][i],
                            &input.matrix_of_atom_vars[j][i],
                        ),
                        &atom_mul_raw(&atom_num_from_f64(step_sizes[j]), &eq_step_j),
                    );

                    Ok((residual, vars_in_equation))
                })
                .collect::<Result<Vec<_>, BvpAtomDiscretizationError>>()
        })
        .collect();
    let assembled_rows = assembled_rows?;
    let discretization_elapsed = discretization_start.elapsed();
    timer_hash.insert(
        "discretization of equations".to_string(),
        discretization_elapsed.as_millis() as f64,
    );

    let (discretized_system, variables_for_all_discrete_symbols): (Vec<_>, Vec<_>) =
        assembled_rows.into_iter().flatten().unzip();

    let bc_application_start = Instant::now();
    let discretized_with_bc = discretized_system
        .into_par_iter()
        .map(|eq| substitute_symbol_values(&eq, &bc_value_map))
        .collect::<Vec<_>>();
    let bc_application_elapsed = bc_application_start.elapsed();
    timer_hash.insert(
        "BC application".to_string(),
        bc_application_elapsed.as_millis() as f64,
    );

    let flat_list_start = Instant::now();
    let total_vars = input.value_symbols.len() * n_steps_total;
    let mut flat_list_of_names = Vec::with_capacity(total_vars);
    let mut flat_list_of_expr = Vec::with_capacity(total_vars);
    for time_idx in 0..n_steps_total {
        for var_idx in 0..input.value_symbols.len() {
            let symbol = input.matrix_of_symbols[time_idx][var_idx];
            if !vars_to_exclude.contains(&symbol) {
                flat_list_of_names.push(symbol.get_stripped_name().to_string());
                flat_list_of_expr.push(input.matrix_of_atom_vars[time_idx][var_idx].clone());
            }
        }
    }

    let variables_for_all_discrete = variables_for_all_discrete_symbols
        .iter()
        .map(|row| {
            row.iter()
                .map(|symbol| symbol.get_stripped_name().to_string())
                .collect()
        })
        .collect::<Vec<Vec<String>>>();
    let flat_list_elapsed = flat_list_start.elapsed();
    timer_hash.insert(
        "flat list creation".to_string(),
        flat_list_elapsed.as_millis() as f64,
    );

    let consistency_start = Instant::now();
    let hashset_of_vars: HashSet<Symbol> = input
        .matrix_of_symbols
        .iter()
        .flat_map(|row| row.iter().copied())
        .filter(|symbol| !vars_to_exclude.contains(symbol))
        .collect();
    let mut missing_vars = Vec::new();
    for var_list in &variables_for_all_discrete_symbols {
        for var in var_list {
            if !hashset_of_vars.contains(var) {
                missing_vars.push(var.get_stripped_name().to_string());
            }
        }
    }
    if !missing_vars.is_empty() {
        missing_vars.sort_unstable();
        missing_vars.dedup();
        return Err(BvpAtomDiscretizationError::MissingVariables(missing_vars));
    }
    let consistency_elapsed = consistency_start.elapsed();
    timer_hash.insert(
        "consistency test".to_string(),
        consistency_elapsed.as_millis() as f64,
    );

    let bounds_start = Instant::now();
    let (bounds_vec, tolerance_vec) =
        process_bounds_and_tolerances_atom(bounds, rel_tolerance, &flat_list_of_names);
    let bounds_elapsed = bounds_start.elapsed();
    timer_hash.insert(
        "bounds and tolerances".to_string(),
        bounds_elapsed.as_millis() as f64,
    );

    let total_elapsed = total_start.elapsed();
    let total_end = total_elapsed.as_millis() as f64;
    timer_hash.insert("total time, ms".to_string(), total_end);

    let system = DiscretizedBvpAtomSystem {
        vector_of_functions: discretized_with_bc,
        vector_of_variables: flat_list_of_expr,
        variable_string: flat_list_of_names,
        variables_for_all_discrete,
        bc_pos_n_values,
        bounds: bounds_vec,
        rel_tolerance_vec: tolerance_vec,
        mesh_points: problem.mesh_points.clone(),
        step_sizes: problem.step_sizes.clone(),
        timer_hash,
        telemetry: BvpAtomDiscretizationTelemetrySnapshot {
            boundary_conditions: bc_handling_elapsed,
            discretization: discretization_elapsed,
            boundary_application: bc_application_elapsed,
            flat_list: flat_list_elapsed,
            consistency: consistency_elapsed,
            bounds_and_tolerances: bounds_elapsed,
            total: total_elapsed,
        },
    };
    println!("{}", system.render_timer_table());
    Ok(system)
}

fn create_mesh_atom(
    n_steps: Option<usize>,
    h: Option<f64>,
    mesh: Option<Vec<f64>>,
    t0: f64,
) -> (Vec<f64>, Vec<f64>, usize) {
    if let Some(mesh) = mesh {
        let n_steps = mesh.len();
        let h_values = mesh
            .windows(2)
            .map(|window| window[1] - window[0])
            .collect::<Vec<_>>();
        (h_values, mesh, n_steps)
    } else {
        let n_steps = n_steps.unwrap_or(100) + 1;
        let h = h.unwrap_or(1.0);
        let mesh = (0..n_steps).map(|i| t0 + h * i as f64).collect::<Vec<_>>();
        let h_values = vec![h; n_steps - 1];
        (h_values, mesh, n_steps)
    }
}

fn indexed_symbol_matrix_symbols(n_steps: usize, values: &[Symbol]) -> Vec<Vec<Symbol>> {
    (0..n_steps)
        .map(|step| {
            values
                .iter()
                .map(|name| {
                    let indexed_name = format!("{}_{}", name.get_stripped_name(), step);
                    Symbol::new(crate::wrap_symbol!(indexed_name.as_str()))
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn process_bounds_and_tolerances_atom(
    bounds: Option<HashMap<String, (f64, f64)>>,
    rel_tolerance: Option<HashMap<String, f64>>,
    flat_list_of_names: &[String],
) -> (Option<Vec<(f64, f64)>>, Option<Vec<f64>>) {
    let bounds_vec = bounds.as_ref().map(|bounds_map| {
        let mut vec_of_bounds = Vec::with_capacity(flat_list_of_names.len());
        for name in flat_list_of_names {
            let base_name = if let Some(pos) = name.rfind('_') {
                &name[..pos]
            } else {
                name.as_str()
            };
            if let Some(&bound_pair) = bounds_map.get(base_name) {
                vec_of_bounds.push(bound_pair);
            }
        }
        vec_of_bounds
    });

    let tolerance_vec = rel_tolerance.as_ref().map(|tolerance_map| {
        let mut vec_of_tolerance = Vec::with_capacity(flat_list_of_names.len());
        for name in flat_list_of_names {
            let base_name = if let Some(pos) = name.rfind('_') {
                &name[..pos]
            } else {
                name.as_str()
            };
            if let Some(&tolerance) = tolerance_map.get(base_name) {
                vec_of_tolerance.push(tolerance);
            }
        }
        vec_of_tolerance
    });

    (bounds_vec, tolerance_vec)
}

fn atom_num_from_f64(value: f64) -> Atom {
    approximate_f64_atom(value)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::{
        numerical::Examples_and_utils::NonlinEquation,
        symbolic::{
            View::{
                bvp::{
                    BvpAtomDiscretizationError, discretization_system_bvp_par_atom, eq_step_atom,
                    try_discretization_system_bvp_par_atom, try_eq_step_atom,
                },
                conversions::{atom_to_expr, expr_to_atom},
                state::Symbol,
            },
            symbolic_engine::Expr,
            symbolic_functions_BVP::Jacobian,
        },
    };

    fn eval_expr_numeric(expr: &Expr, vars: &HashMap<String, f64>) -> f64 {
        match expr {
            Expr::Var(name) => *vars
                .get(name)
                .unwrap_or_else(|| panic!("missing sample value for variable `{name}`")),
            Expr::Const(v) => *v,
            Expr::Add(l, r) => eval_expr_numeric(l, vars) + eval_expr_numeric(r, vars),
            Expr::Sub(l, r) => eval_expr_numeric(l, vars) - eval_expr_numeric(r, vars),
            Expr::Mul(l, r) => eval_expr_numeric(l, vars) * eval_expr_numeric(r, vars),
            Expr::Div(l, r) => eval_expr_numeric(l, vars) / eval_expr_numeric(r, vars),
            Expr::Pow(b, e) => eval_expr_numeric(b, vars).powf(eval_expr_numeric(e, vars)),
            Expr::Exp(x) => eval_expr_numeric(x, vars).exp(),
            Expr::Ln(x) => eval_expr_numeric(x, vars).ln(),
            Expr::sin(x) => eval_expr_numeric(x, vars).sin(),
            Expr::cos(x) => eval_expr_numeric(x, vars).cos(),
            Expr::tg(x) => eval_expr_numeric(x, vars).tan(),
            Expr::ctg(x) => 1.0 / eval_expr_numeric(x, vars).tan(),
            Expr::arcsin(x) => eval_expr_numeric(x, vars).asin(),
            Expr::arccos(x) => eval_expr_numeric(x, vars).acos(),
            Expr::arctg(x) => eval_expr_numeric(x, vars).atan(),
            Expr::arcctg(x) => std::f64::consts::FRAC_PI_2 - eval_expr_numeric(x, vars).atan(),
        }
    }

    fn sample_assignments(variable_names: &[String]) -> Vec<HashMap<String, f64>> {
        let bases = [0.23, 0.41, 0.77];
        bases
            .iter()
            .enumerate()
            .map(|(sample_idx, base)| {
                variable_names
                    .iter()
                    .enumerate()
                    .map(|(idx, name)| {
                        let value = base + idx as f64 * 0.013 + sample_idx as f64 * 0.017;
                        (name.clone(), value)
                    })
                    .collect::<HashMap<_, _>>()
            })
            .collect()
    }

    #[test]
    fn eq_step_atom_matches_forward_style_variable_renaming() {
        let eq = expr_to_atom(&Expr::parse_expression("y + t"));
        let values = vec!["y".to_string()];
        let matrix = vec![vec!["y_0".to_string()], vec!["y_1".to_string()]];

        let out = eq_step_atom(&eq, &matrix, &values, "t", 0, 3.0, "forward");
        let rendered = crate::symbolic::View::conversions::atom_to_expr(&out).to_string();

        assert!(rendered.contains("y_0"));
        assert!(!rendered.contains("t"));
    }

    #[test]
    fn eq_step_atom_supports_trapezoid_scheme() {
        let eq = expr_to_atom(&Expr::parse_expression("y"));
        let values = vec!["y".to_string()];
        let matrix = vec![vec!["y_0".to_string()], vec!["y_1".to_string()]];

        let out = eq_step_atom(&eq, &matrix, &values, "t", 0, 0.0, "trapezoid");
        let rendered = crate::symbolic::View::conversions::atom_to_expr(&out).to_string();

        assert!(rendered.contains("y_0"));
        assert!(rendered.contains("y_1"));
    }

    #[test]
    fn try_eq_step_atom_rejects_unknown_scheme_without_panicking() {
        let eq = expr_to_atom(&Expr::parse_expression("y"));
        let values = vec!["y".to_string()];
        let matrix = vec![vec!["y_0".to_string()], vec!["y_1".to_string()]];

        let error = try_eq_step_atom(&eq, &matrix, &values, "t", 0, 0.0, "midpoint")
            .expect_err("unknown scheme must be a typed error");
        assert_eq!(
            error,
            BvpAtomDiscretizationError::InvalidScheme("midpoint".to_string())
        );
    }

    #[test]
    fn try_atom_discretization_rejects_unknown_scheme_without_panicking() {
        let error = try_discretization_system_bvp_par_atom(
            vec![Expr::parse_expression("y")],
            vec!["y".to_string()],
            "x".to_string(),
            0.0,
            Some(2),
            None,
            None,
            HashMap::new(),
            None,
            None,
            "midpoint".to_string(),
        )
        .expect_err("unknown scheme must be a typed error");
        assert_eq!(
            error,
            BvpAtomDiscretizationError::InvalidScheme("midpoint".to_string())
        );
    }

    fn compare_atom_and_expr_discretization(
        eqs: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        n_steps: usize,
        border_conditions: HashMap<String, Vec<(usize, f64)>>,
        bounds: Option<HashMap<String, (f64, f64)>>,
        rel_tolerance: Option<HashMap<String, f64>>,
        scheme: String,
    ) {
        let atom_system = discretization_system_bvp_par_atom(
            eqs.clone(),
            values.clone(),
            arg.clone(),
            t0,
            Some(n_steps),
            None,
            None,
            border_conditions.clone(),
            bounds.clone(),
            rel_tolerance.clone(),
            scheme.clone(),
        );

        let mut legacy = Jacobian::new();
        legacy.discretization_system_BVP_par(
            eqs,
            values,
            arg,
            t0,
            Some(n_steps),
            None,
            None,
            border_conditions,
            bounds,
            rel_tolerance,
            scheme,
        );

        assert_eq!(atom_system.variable_string, legacy.variable_string);
        let mut atom_bc = atom_system.bc_pos_n_values.clone();
        let mut legacy_bc = legacy.BC_pos_n_values.clone();
        atom_bc.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        legacy_bc.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        assert_eq!(atom_bc, legacy_bc);
        assert_eq!(atom_system.bounds, legacy.bounds);
        assert_eq!(atom_system.rel_tolerance_vec, legacy.rel_tolerance_vec);
        assert_eq!(
            atom_system.variables_for_all_discrete,
            legacy.variables_for_all_disrete
        );
        assert_eq!(
            atom_system.vector_of_variables.len(),
            legacy.vector_of_variables.len()
        );
        assert_eq!(
            atom_system.vector_of_functions.len(),
            legacy.vector_of_functions.len()
        );

        let sample_sets = sample_assignments(&atom_system.variable_string);

        for (row_idx, (atom_expr, legacy_expr)) in atom_system
            .vector_of_functions
            .iter()
            .zip(legacy.vector_of_functions.iter())
            .enumerate()
        {
            let atom_as_expr = atom_to_expr(atom_expr);
            let mut finite_checks = 0usize;
            for sample_values in &sample_sets {
                let atom_value = eval_expr_numeric(&atom_as_expr, sample_values);
                let legacy_value = eval_expr_numeric(legacy_expr, sample_values);
                if atom_value.is_finite() && legacy_value.is_finite() {
                    finite_checks += 1;
                    assert!(
                        (atom_value - legacy_value).abs() < 1e-8,
                        "row {row_idx} diverged numerically: atom={atom_value}, legacy={legacy_value}\natom_expr={atom_as_expr}\nlegacy_expr={legacy_expr}"
                    );
                }
            }

            if finite_checks == 0 {
                let atom_rendered = atom_as_expr.to_string();
                let legacy_rendered = legacy_expr.to_string();
                for var in &atom_system.variables_for_all_discrete[row_idx] {
                    assert!(
                        atom_rendered.contains(var),
                        "row {row_idx} lost active variable `{var}` in atom discretization"
                    );
                    assert!(
                        legacy_rendered.contains(var),
                        "row {row_idx} lost active variable `{var}` in legacy discretization"
                    );
                }
            }
        }
        for (atom_var, legacy_var) in atom_system
            .vector_of_variables
            .iter()
            .zip(legacy.vector_of_variables.iter())
        {
            assert_eq!(atom_to_expr(atom_var).to_string(), legacy_var.to_string());
        }
    }

    #[test]
    fn atom_discretization_matches_legacy_lane_emden() {
        let ne = NonlinEquation::LaneEmden5;
        let (start, _) = ne.span(None, None);
        compare_atom_and_expr_discretization(
            ne.setup(),
            ne.values(),
            "x".to_string(),
            start,
            12,
            ne.boundary_conditions(),
            Some(ne.Bounds()),
            None,
            "trapezoid".to_string(),
        );
    }

    #[test]
    fn atom_discretization_matches_legacy_two_point_bvp() {
        let ne = NonlinEquation::TwoPointBVP;
        let (start, _) = ne.span(None, None);
        compare_atom_and_expr_discretization(
            ne.setup(),
            ne.values(),
            "x".to_string(),
            start,
            12,
            ne.boundary_conditions(),
            Some(ne.Bounds()),
            None,
            "trapezoid".to_string(),
        );
    }

    #[test]
    fn atom_discretization_preserves_fractional_flux_coefficients() {
        compare_atom_and_expr_discretization(
            vec![Expr::parse_expression("J / 0.000288"), Expr::Const(0.0)],
            vec!["C".to_string(), "J".to_string()],
            "x".to_string(),
            0.0,
            2,
            HashMap::from([
                ("C".to_string(), vec![(0, 0.001)]),
                ("J".to_string(), vec![(1, 0.0)]),
            ]),
            None,
            None,
            "forward".to_string(),
        );
    }

    #[test]
    fn atom_discretization_preserves_singular_zero_endpoint_without_panicking() {
        let system = discretization_system_bvp_par_atom(
            vec![Expr::parse_expression("z / x"), Expr::parse_expression("y")],
            vec!["y".to_string(), "z".to_string()],
            "x".to_string(),
            0.0,
            Some(1),
            None,
            None,
            HashMap::new(),
            None,
            None,
            "forward".to_string(),
        );

        // A singular endpoint is a symbolic responsibility of the caller
        // (regularization, limiting value, or a problem-specific condition).
        // Assembly must preserve the structure rather than evaluate 0^-1.
        let rendered = atom_to_expr(&system.vector_of_functions[0]).to_string();
        assert!(
            rendered.contains("0"),
            "singular endpoint was lost: {rendered}"
        );
        assert_eq!(system.variable_string.len(), 4);
    }

    #[test]
    fn atom_only_constructor_matches_expr_compatibility_wrapper() {
        let equations = vec![Expr::parse_expression("y + x"), Expr::parse_expression("z")];
        let values = vec!["y".to_string(), "z".to_string()];
        let border_conditions = HashMap::from([("y".to_string(), vec![(0, 1.0)])]);
        let atom_border_conditions =
            HashMap::from([(Symbol::new(crate::wrap_symbol!("y")), vec![(0, 1.0)])]);

        let compatibility = discretization_system_bvp_par_atom(
            equations.clone(),
            values.clone(),
            "x".to_string(),
            0.0,
            Some(3),
            None,
            None,
            border_conditions,
            None,
            None,
            "forward".to_string(),
        );
        let atom_native = super::discretization_system_bvp_par_atom_native(
            equations.iter().map(expr_to_atom).collect(),
            values
                .iter()
                .map(|name| Symbol::new(crate::wrap_symbol!(name.as_str())))
                .collect(),
            Symbol::new(crate::wrap_symbol!("x")),
            0.0,
            Some(3),
            None,
            None,
            atom_border_conditions,
            None,
            None,
            "forward".to_string(),
        );

        assert_eq!(compatibility.variable_string, atom_native.variable_string);
        assert_eq!(
            compatibility.vector_of_functions.len(),
            atom_native.vector_of_functions.len()
        );
        for (compatibility_row, native_row) in compatibility
            .vector_of_functions
            .iter()
            .zip(atom_native.vector_of_functions.iter())
        {
            assert_eq!(
                atom_to_expr(compatibility_row).to_string(),
                atom_to_expr(native_row).to_string()
            );
        }
        assert!(compatibility.telemetry.stages_fit_total());
        assert!(atom_native.telemetry.stages_fit_total());
        assert!(compatibility.telemetry.total > std::time::Duration::ZERO);
        assert!(atom_native.telemetry.total > std::time::Duration::ZERO);
    }
}

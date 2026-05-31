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
    atom::FunctionBuilder,
    conversions::approximate_f64_atom,
    state::Symbol,
    transform::{rename_symbols_view, substitute_symbol_values},
};
use crate::symbolic::symbolic_engine::Expr;
use rayon::prelude::*;
use tabled::{builder::Builder, settings::Style};

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

    /// Renders the normalized timing table in the same human-readable style as the legacy path.
    pub fn render_timer_table(&self) -> String {
        let mut table = Builder::from(self.normalized_timer_hash()).build();
        table.with(Style::modern_rounded());
        table.to_string()
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
    let eq_step_j = rename_and_bind_step(eq_i, &matrix_of_names[j], values, arg, t);

    match scheme {
        "forward" => eq_step_j,
        "trapezoid" => {
            let eq_step_j_plus_1 =
                rename_and_bind_step(eq_i, &matrix_of_names[j + 1], values, arg, t);
            Atom::new_num(1) / Atom::new_num(2) * (eq_step_j + eq_step_j_plus_1)
        }
        _ => panic!("Invalid scheme"),
    }
}

fn normalize_timer_percent(value: &mut f64, total: f64) {
    if total <= f64::EPSILON {
        *value = 0.0;
    } else {
        *value /= total / 100.0;
    }
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
/// 2. Build the matrix of discretized variable names `y_i_j`.
/// 3. Pre-compute boundary-condition substitutions and the set of discrete
///    variables removed from the nonlinear unknown vector.
/// 4. Convert the original RHS system from `Expr` to `Atom` once.
/// 5. In parallel over mesh intervals, assemble residual rows:
///    - regular rows use the View-native `eq_step_atom(...)`,
///    - the singular `t == 0` row currently uses a conservative fallback
///      through the legacy expression logic to preserve exact semantics.
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
    let total_start = Instant::now();
    let mut timer_hash: HashMap<String, f64> = HashMap::new();
    let (step_sizes, mesh_points, n_steps_total) = create_mesh_atom(n_steps, h, mesh, t0);
    let (matrix_of_names, matrix_of_atom_vars) = indexed_vars_matrix_atom(n_steps_total, &values);

    let bc_handling = Instant::now();
    let bc_lookup: HashMap<String, HashMap<usize, f64>> = border_conditions
        .into_iter()
        .map(|(k, v)| (k, v.into_iter().collect()))
        .collect();

    let mut vars_for_boundary_conditions: HashMap<String, f64> = HashMap::default();
    let mut vars_to_exclude: HashSet<String> = HashSet::default();
    let mut bc_pos_n_values = Vec::new();

    for (var_name, conditions) in &bc_lookup {
        if let Some(var_idx) = values.iter().position(|v| v == var_name) {
            for (&pos, &value) in conditions {
                match pos {
                    0 => {
                        let discretized_name = &matrix_of_names[0][var_idx];
                        vars_for_boundary_conditions.insert(discretized_name.clone(), value);
                        vars_to_exclude.insert(discretized_name.clone());
                        let full_pos = var_idx;
                        bc_pos_n_values.push((full_pos, 0usize, value));
                    }
                    1 => {
                        let discretized_name = &matrix_of_names[n_steps_total - 1][var_idx];
                        vars_for_boundary_conditions.insert(discretized_name.clone(), value);
                        vars_to_exclude.insert(discretized_name.clone());
                        let full_pos = (n_steps_total - 1) * values.len() + var_idx;
                        bc_pos_n_values.push((full_pos, 1usize, value));
                    }
                    _ => {}
                }
            }
        }
    }
    timer_hash.insert(
        "bc handling".to_string(),
        bc_handling.elapsed().as_millis() as f64,
    );

    let bc_value_map: HashMap<Symbol, f64> = vars_for_boundary_conditions
        .iter()
        .map(|(name, value)| (Symbol::new(crate::wrap_symbol!(name.as_str())), *value))
        .collect();

    let eq_atoms = eq_system
        .iter()
        .map(super::conversions::expr_to_atom)
        .collect::<Vec<_>>();

    let discretization_start = Instant::now();
    let assembled_rows = (0..(n_steps_total - 1))
        .into_par_iter()
        .map(|j| {
            let t = mesh_points[j];
            eq_atoms
                .iter()
                .enumerate()
                .map(|(i, eq_i)| {
                    let mut vars_in_equation = Vec::new();
                    let y_j_plus_1_name = &matrix_of_names[j + 1][i];
                    let y_j_name = &matrix_of_names[j][i];

                    if !vars_to_exclude.contains(y_j_plus_1_name) {
                        vars_in_equation.push(y_j_plus_1_name.clone());
                    }
                    if !vars_to_exclude.contains(y_j_name) {
                        vars_in_equation.push(y_j_name.clone());
                    }
                    for var_idx in 0..values.len() {
                        let var_name = &matrix_of_names[j][var_idx];
                        if !vars_to_exclude.contains(var_name) {
                            vars_in_equation.push(var_name.clone());
                        }
                    }

                    let residual = if t == 0.0 {
                        build_residual_atom_expr_fallback(
                            &eq_system[i],
                            &matrix_of_names,
                            &values,
                            i,
                            &arg,
                            j,
                            t,
                            step_sizes[j],
                            &scheme,
                            &vars_for_boundary_conditions,
                        )
                    } else {
                        let eq_step_j =
                            eq_step_atom(eq_i, &matrix_of_names, &values, &arg, j, t, &scheme);
                        let eq_step_j = substitute_symbol_values(&eq_step_j, &bc_value_map);
                        matrix_of_atom_vars[j + 1][i].clone()
                            - matrix_of_atom_vars[j][i].clone()
                            - atom_num_from_f64(step_sizes[j]) * eq_step_j
                    };

                    (residual, vars_in_equation)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    timer_hash.insert(
        "discretization of equations".to_string(),
        discretization_start.elapsed().as_millis() as f64,
    );

    let (discretized_system, variables_for_all_discrete): (Vec<_>, Vec<_>) =
        assembled_rows.into_iter().flatten().unzip();

    let bc_application_start = Instant::now();
    let discretized_with_bc = discretized_system
        .into_par_iter()
        .map(|eq| substitute_symbol_values(&eq, &bc_value_map))
        .collect::<Vec<_>>();
    timer_hash.insert(
        "BC application".to_string(),
        bc_application_start.elapsed().as_millis() as f64,
    );

    let flat_list_start = Instant::now();
    let total_vars = values.len() * n_steps_total;
    let mut flat_list_of_names = Vec::with_capacity(total_vars);
    let mut flat_list_of_expr = Vec::with_capacity(total_vars);
    for time_idx in 0..n_steps_total {
        for var_idx in 0..values.len() {
            let name = &matrix_of_names[time_idx][var_idx];
            if !vars_to_exclude.contains(name) {
                flat_list_of_names.push(name.clone());
                flat_list_of_expr.push(matrix_of_atom_vars[time_idx][var_idx].clone());
            }
        }
    }
    timer_hash.insert(
        "flat list creation".to_string(),
        flat_list_start.elapsed().as_millis() as f64,
    );

    let consistency_start = Instant::now();
    let hashset_of_vars: HashSet<&String> = flat_list_of_names.iter().collect();
    let mut missing_vars = Vec::new();
    for var_list in &variables_for_all_discrete {
        for var in var_list {
            if !hashset_of_vars.contains(var) {
                missing_vars.push(var.clone());
            }
        }
    }
    if !missing_vars.is_empty() {
        missing_vars.sort_unstable();
        missing_vars.dedup();
        panic!(
            "Variables not found in atom-discretized system: {:?}",
            missing_vars
        );
    }
    timer_hash.insert(
        "consistency test".to_string(),
        consistency_start.elapsed().as_millis() as f64,
    );

    let bounds_start = Instant::now();
    let (bounds_vec, tolerance_vec) =
        process_bounds_and_tolerances_atom(bounds, rel_tolerance, &flat_list_of_names);
    timer_hash.insert(
        "bounds and tolerances".to_string(),
        bounds_start.elapsed().as_millis() as f64,
    );

    let total_end = total_start.elapsed().as_millis() as f64;
    timer_hash.insert("total time, ms".to_string(), total_end);

    let system = DiscretizedBvpAtomSystem {
        vector_of_functions: discretized_with_bc,
        vector_of_variables: flat_list_of_expr,
        variable_string: flat_list_of_names,
        variables_for_all_discrete,
        bc_pos_n_values,
        bounds: bounds_vec,
        rel_tolerance_vec: tolerance_vec,
        mesh_points,
        step_sizes,
        timer_hash,
    };
    println!("{}", system.render_timer_table());
    system
}

fn rename_and_bind_step(
    eq_i: &Atom,
    renamed_values_for_step: &[String],
    values: &[String],
    arg: &str,
    t: f64,
) -> Atom {
    let rename_map: HashMap<Symbol, Symbol> = values
        .iter()
        .zip(renamed_values_for_step.iter())
        .map(|(src, dst)| {
            (
                Symbol::new(crate::wrap_symbol!(src.as_str())),
                Symbol::new(crate::wrap_symbol!(dst.as_str())),
            )
        })
        .collect();
    let renamed = rename_symbols_view(eq_i.as_view(), &rename_map);

    let mut value_map = HashMap::default();
    value_map.insert(Symbol::new(crate::wrap_symbol!(arg)), t);
    substitute_symbol_values(&renamed, &value_map)
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

fn indexed_vars_matrix_atom(
    n_steps: usize,
    values: &[String],
) -> (Vec<Vec<String>>, Vec<Vec<Atom>>) {
    let names = (0..n_steps)
        .map(|step| {
            values
                .iter()
                .map(|name| format!("{name}_{step}"))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let atoms = names
        .iter()
        .map(|row| {
            row.iter()
                .map(|name| Atom::new_var(Symbol::new(crate::wrap_symbol!(name.as_str()))))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    (names, atoms)
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

fn build_residual_atom_expr_fallback(
    eq_i: &Expr,
    matrix_of_names: &[Vec<String>],
    values: &[String],
    eq_index: usize,
    arg: &str,
    j: usize,
    t: f64,
    h: f64,
    scheme: &str,
    vars_for_boundary_conditions: &HashMap<String, f64>,
) -> Atom {
    let eq_step_j = build_eq_step_expr(matrix_of_names, eq_i, values, arg, j, t, scheme);
    let y_j_plus_1 = Expr::Var(matrix_of_names[j + 1][eq_index].clone());
    let y_j = Expr::Var(matrix_of_names[j][eq_index].clone());
    let residual = y_j_plus_1 - y_j - Expr::Const(h) * eq_step_j;
    let residual = residual
        .set_variable_from_map(vars_for_boundary_conditions)
        .simplify();
    expr_to_atom_no_norm(&residual)
}

fn build_eq_step_expr(
    matrix_of_names: &[Vec<String>],
    eq_i: &Expr,
    values: &[String],
    arg: &str,
    j: usize,
    t: f64,
    scheme: &str,
) -> Expr {
    let vec_of_names_on_step = &matrix_of_names[j];
    let hashmap_for_rename: HashMap<String, String> = values
        .iter()
        .zip(vec_of_names_on_step.iter())
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    let eq_step_j = eq_i
        .rename_variables(&hashmap_for_rename)
        .set_variable(arg, t);

    match scheme {
        "forward" => eq_step_j,
        "trapezoid" => {
            let vec_of_names_on_step = &matrix_of_names[j + 1];
            let hashmap_for_rename: HashMap<String, String> = values
                .iter()
                .zip(vec_of_names_on_step.iter())
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            let eq_step_j_plus_1 = eq_i
                .rename_variables(&hashmap_for_rename)
                .set_variable(arg, t);
            Expr::Const(0.5) * (eq_step_j + eq_step_j_plus_1)
        }
        _ => panic!("Invalid scheme"),
    }
}

fn expr_to_atom_no_norm(expr: &Expr) -> Atom {
    match expr {
        Expr::Var(name) => Atom::new_var(Symbol::new(crate::wrap_symbol!(name.as_str()))),
        Expr::Const(v) => atom_num_from_f64(*v),
        Expr::Add(l, r) => {
            let mut out = Atom::default();
            let add = out.to_add();
            let l_atom = expr_to_atom_no_norm(l);
            let r_atom = expr_to_atom_no_norm(r);
            add.extend(l_atom.as_view());
            add.extend(r_atom.as_view());
            out
        }
        Expr::Sub(l, r) => {
            let mut out = Atom::default();
            let add = out.to_add();
            let l_atom = expr_to_atom_no_norm(l);
            let mut neg_r = Atom::default();
            let mul = neg_r.to_mul();
            let minus_one = Atom::new_num(-1);
            let r_atom = expr_to_atom_no_norm(r);
            mul.extend(minus_one.as_view());
            mul.extend(r_atom.as_view());
            add.extend(l_atom.as_view());
            add.extend(neg_r.as_view());
            out
        }
        Expr::Mul(l, r) => {
            let mut out = Atom::default();
            let mul = out.to_mul();
            let l_atom = expr_to_atom_no_norm(l);
            let r_atom = expr_to_atom_no_norm(r);
            mul.extend(l_atom.as_view());
            mul.extend(r_atom.as_view());
            out
        }
        Expr::Div(l, r) => {
            let mut out = Atom::default();
            let mul = out.to_mul();
            let l_atom = expr_to_atom_no_norm(l);
            let r_atom = expr_to_atom_no_norm(r);
            let mut inv_r = Atom::default();
            inv_r.to_pow(r_atom.as_view(), Atom::new_num(-1).as_view());
            mul.extend(l_atom.as_view());
            mul.extend(inv_r.as_view());
            out
        }
        Expr::Pow(b, e) => {
            let mut out = Atom::default();
            let base = expr_to_atom_no_norm(b);
            let exp = expr_to_atom_no_norm(e);
            out.to_pow(base.as_view(), exp.as_view());
            out
        }
        Expr::Exp(x) => FunctionBuilder::new(Atom::EXP)
            .add_arg(expr_to_atom_no_norm(x))
            .finish(),
        Expr::Ln(x) => FunctionBuilder::new(Atom::LOG)
            .add_arg(expr_to_atom_no_norm(x))
            .finish(),
        Expr::sin(x) => FunctionBuilder::new(Atom::SIN)
            .add_arg(expr_to_atom_no_norm(x))
            .finish(),
        Expr::cos(x) => FunctionBuilder::new(Atom::COS)
            .add_arg(expr_to_atom_no_norm(x))
            .finish(),
        Expr::tg(x) => FunctionBuilder::new(Atom::TAN)
            .add_arg(expr_to_atom_no_norm(x))
            .finish(),
        Expr::ctg(x) => FunctionBuilder::new(Atom::COT)
            .add_arg(expr_to_atom_no_norm(x))
            .finish(),
        Expr::arcsin(x) => FunctionBuilder::new(Atom::ASIN)
            .add_arg(expr_to_atom_no_norm(x))
            .finish(),
        Expr::arccos(x) => FunctionBuilder::new(Atom::ACOS)
            .add_arg(expr_to_atom_no_norm(x))
            .finish(),
        Expr::arctg(x) => FunctionBuilder::new(Atom::ATAN)
            .add_arg(expr_to_atom_no_norm(x))
            .finish(),
        Expr::arcctg(x) => FunctionBuilder::new(Atom::ACOT)
            .add_arg(expr_to_atom_no_norm(x))
            .finish(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::{
        numerical::Examples_and_utils::NonlinEquation,
        symbolic::{
            View::{
                bvp::{discretization_system_bvp_par_atom, eq_step_atom},
                conversions::{atom_to_expr, expr_to_atom},
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
}

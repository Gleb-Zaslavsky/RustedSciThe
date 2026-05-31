use crate::numerical::BVP_Damp::BVP_traits::{
    Fun, VectorType, Vectors_type_casting, convert_to_fun, convert_to_jac,
};
use crate::numerical::BVP_Damp::generated_solver_handoff::DampedGeneratedSolverState;
use crate::symbolic::codegen::codegen_backend_selection::SelectedBackendKind;
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;
use std::sync::Arc;

/// Pure-numeric RHS callback used by the damped BVP solver.
///
/// Signature:
/// - `x`: mesh coordinate,
/// - `y`: state vector at that node (ordered as `values`),
/// - `params`: optional parameter slice from solver configuration.
pub type NumericBvpRhs =
    Arc<dyn Fn(f64, &DVector<f64>, Option<&[f64]>) -> DVector<f64> + Send + Sync + 'static>;

/// Continuous Jacobian callback for the pure-numeric damped BVP route.
///
/// The callback returns `df/dy` for the RHS closure at one mesh node. The
/// discretized Newton Jacobian is assembled by this module so the user does not
/// have to build the large global BVP matrix by hand.
pub type NumericBvpJacobian =
    Arc<dyn Fn(f64, &DVector<f64>, Option<&[f64]>) -> DMatrix<f64> + Send + Sync + 'static>;

fn boundary_position(
    n_vars: usize,
    n_steps: usize,
    var_index: usize,
    side_flag: usize,
) -> Option<usize> {
    match side_flag {
        0 => Some(var_index),
        1 => Some(n_steps * n_vars + var_index),
        _ => None,
    }
}

fn build_bc_layout(
    values: &[String],
    border_conditions: &HashMap<String, Vec<(usize, f64)>>,
    n_steps: usize,
) -> Result<Vec<(usize, usize, f64)>, String> {
    let mut bc = Vec::with_capacity(values.len());
    for (var_index, var_name) in values.iter().enumerate() {
        let Some(entries) = border_conditions.get(var_name) else {
            continue;
        };
        for &(side, value) in entries {
            let pos =
                boundary_position(values.len(), n_steps, var_index, side).ok_or_else(|| {
                    format!(
                        "boundary side flag must be 0 or 1 for variable '{var_name}', got {side}"
                    )
                })?;
            bc.push((pos, side, value));
        }
    }
    if bc.len() != values.len() {
        return Err(format!(
            "pure numeric route expects exactly {} total boundary conditions for a first-order system with {} variables, got {}",
            values.len(),
            values.len(),
            bc.len()
        ));
    }
    bc.sort_by_key(|(pos, _, _)| *pos);
    for pair in bc.windows(2) {
        if pair[0].0 == pair[1].0 {
            return Err(format!(
                "duplicate pure numeric boundary condition at flattened position {}",
                pair[0].0
            ));
        }
    }
    Ok(bc)
}

fn build_unknown_layout(
    values: &[String],
    n_steps: usize,
    bc_positions: &[(usize, usize, f64)],
) -> (Vec<usize>, Vec<String>) {
    let n_vars = values.len();
    let full_len = n_vars * (n_steps + 1);
    let mut is_bc = vec![false; full_len];
    for (pos, _, _) in bc_positions {
        if *pos < full_len {
            is_bc[*pos] = true;
        }
    }

    let mut unknown_positions = Vec::with_capacity(n_vars * n_steps);
    let mut variable_string = Vec::with_capacity(n_vars * n_steps);
    for node in 0..=n_steps {
        for (var_index, var_name) in values.iter().enumerate() {
            let full_pos = node * n_vars + var_index;
            if !is_bc[full_pos] {
                unknown_positions.push(full_pos);
                variable_string.push(format!("{var_name}_{node}"));
            }
        }
    }
    (unknown_positions, variable_string)
}

fn build_bounds_and_tolerances(
    values: &[String],
    unknown_positions: &[usize],
    bounds: &HashMap<String, (f64, f64)>,
    rel_tolerance: &HashMap<String, f64>,
) -> Result<(Vec<(f64, f64)>, Vec<f64>), String> {
    let n_vars = values.len();
    let mut bounds_vec = Vec::with_capacity(unknown_positions.len());
    let mut rel_tolerance_vec = Vec::with_capacity(unknown_positions.len());
    for &full_pos in unknown_positions {
        let var_index = full_pos % n_vars;
        let var_name = &values[var_index];
        let b = bounds
            .get(var_name)
            .ok_or_else(|| format!("missing bounds for variable '{var_name}'"))?;
        let rt = rel_tolerance
            .get(var_name)
            .ok_or_else(|| format!("missing relative tolerance for variable '{var_name}'"))?;
        bounds_vec.push(*b);
        rel_tolerance_vec.push(*rt);
    }
    Ok((bounds_vec, rel_tolerance_vec))
}

fn infer_bandwidth(n_vars: usize) -> (usize, usize) {
    // Conservative stencil-based estimate for first-order BVP systems.
    // Each interval equation couples two neighboring nodes.
    (n_vars, n_vars)
}

#[allow(clippy::too_many_arguments)]
pub fn build_numeric_generated_solver_state(
    rhs: NumericBvpRhs,
    jacobian: Option<NumericBvpJacobian>,
    method: &str,
    scheme: &str,
    values: &[String],
    border_conditions: &HashMap<String, Vec<(usize, f64)>>,
    bounds: &HashMap<String, (f64, f64)>,
    rel_tolerance: &HashMap<String, f64>,
    n_steps: usize,
    mesh: &[f64],
    bandwidth_hint: Option<(usize, usize)>,
    param_values: Option<Vec<f64>>,
) -> Result<DampedGeneratedSolverState, String> {
    if values.is_empty() {
        return Err("values must not be empty".to_string());
    }
    if n_steps < 2 {
        return Err("n_steps must be >= 2".to_string());
    }
    if mesh.len() != n_steps + 1 {
        return Err(format!(
            "mesh length mismatch: expected {}, got {}",
            n_steps + 1,
            mesh.len()
        ));
    }

    let bc_layout = build_bc_layout(values, border_conditions, n_steps)?;
    let (unknown_positions, variable_string) = build_unknown_layout(values, n_steps, &bc_layout);
    let (bounds_vec, rel_tolerance_vec) =
        build_bounds_and_tolerances(values, &unknown_positions, bounds, rel_tolerance)?;

    let n_vars = values.len();
    let full_len = n_vars * (n_steps + 1);
    let bc_lookup: HashMap<usize, f64> = bc_layout
        .iter()
        .map(|(pos, _, value)| (*pos, *value))
        .collect();
    let mesh_vec = mesh.to_vec();
    let method_owned = method.to_string();
    let scheme_owned = scheme.to_ascii_lowercase();
    let fun_unknown_positions = unknown_positions.clone();
    let fun_bc_lookup = bc_lookup.clone();
    let fun_mesh_vec = mesh_vec.clone();
    let fun_param_values = param_values.clone();
    let fun_scheme_owned = scheme_owned.clone();

    let fun = convert_to_fun(Box::new(move |_x: f64, vec: &dyn VectorType| {
        let reduced = vec.to_DVectorType();
        assert_eq!(
            reduced.len(),
            fun_unknown_positions.len(),
            "numeric BVP reduced state length mismatch"
        );

        let mut full = vec![0.0; full_len];
        for (pos, value) in &fun_bc_lookup {
            full[*pos] = *value;
        }
        for (idx, full_pos) in fun_unknown_positions.iter().enumerate() {
            full[*full_pos] = reduced[idx];
        }

        let mut residual = vec![0.0; n_vars * n_steps];
        for step in 0..n_steps {
            let base0 = step * n_vars;
            let base1 = (step + 1) * n_vars;
            let y0 = DVector::from_vec(full[base0..base0 + n_vars].to_vec());
            let y1 = DVector::from_vec(full[base1..base1 + n_vars].to_vec());

            let x0 = fun_mesh_vec[step];
            let x1 = fun_mesh_vec[step + 1];
            let h = x1 - x0;

            let f0 = rhs(x0, &y0, fun_param_values.as_deref());
            assert_eq!(
                f0.len(),
                n_vars,
                "numeric RHS output dim mismatch at step {step}"
            );

            match fun_scheme_owned.as_str() {
                "trapezoid" => {
                    let f1 = rhs(x1, &y1, fun_param_values.as_deref());
                    assert_eq!(
                        f1.len(),
                        n_vars,
                        "numeric RHS output dim mismatch at step {}",
                        step + 1
                    );
                    for var in 0..n_vars {
                        residual[base0 + var] = y1[var] - y0[var] - 0.5 * h * (f0[var] + f1[var]);
                    }
                }
                _ => {
                    for var in 0..n_vars {
                        residual[base0 + var] = y1[var] - y0[var] - h * f0[var];
                    }
                }
            }
        }

        Vectors_type_casting(&DVector::from_vec(residual), method_owned.clone())
    })) as Box<dyn Fun>;

    let jac = jacobian.map(|jacobian| {
        let jac_unknown_positions = unknown_positions.clone();
        let jac_bc_lookup = bc_lookup.clone();
        let jac_mesh_vec = mesh_vec.clone();
        let jac_param_values = param_values.clone();
        let jac_scheme_owned = scheme_owned.clone();

        convert_to_jac(Box::new(move |_x: f64, vec: &dyn VectorType| {
            let reduced = vec.to_DVectorType();
            let n_unknowns = jac_unknown_positions.len();
            assert_eq!(
                reduced.len(),
                n_unknowns,
                "numeric BVP reduced state length mismatch in Jacobian callback"
            );

            let mut full = vec![0.0; full_len];
            for (pos, value) in &jac_bc_lookup {
                full[*pos] = *value;
            }
            let mut full_to_unknown = vec![None; full_len];
            for (idx, full_pos) in jac_unknown_positions.iter().enumerate() {
                full[*full_pos] = reduced[idx];
                full_to_unknown[*full_pos] = Some(idx);
            }

            let mut dense = vec![0.0_f64; n_unknowns * n_unknowns];
            let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
            let mut add_entry = |row: usize, col: usize, value: f64| {
                if value == 0.0 {
                    return;
                }
                let slot = row * n_unknowns + col;
                dense[slot] += value;
            };

            for step in 0..n_steps {
                let base0 = step * n_vars;
                let base1 = (step + 1) * n_vars;
                let y0 = DVector::from_vec(full[base0..base0 + n_vars].to_vec());
                let y1 = DVector::from_vec(full[base1..base1 + n_vars].to_vec());

                let x0 = jac_mesh_vec[step];
                let x1 = jac_mesh_vec[step + 1];
                let h = x1 - x0;
                let j0 = jacobian(x0, &y0, jac_param_values.as_deref());
                assert_eq!(
                    j0.shape(),
                    (n_vars, n_vars),
                    "numeric RHS Jacobian shape mismatch at step {step}"
                );

                match jac_scheme_owned.as_str() {
                    "trapezoid" => {
                        let j1 = jacobian(x1, &y1, jac_param_values.as_deref());
                        assert_eq!(
                            j1.shape(),
                            (n_vars, n_vars),
                            "numeric RHS Jacobian shape mismatch at step {}",
                            step + 1
                        );
                        for row_var in 0..n_vars {
                            let row = base0 + row_var;
                            for col_var in 0..n_vars {
                                if let Some(col) = full_to_unknown[base0 + col_var] {
                                    let identity = if row_var == col_var { -1.0 } else { 0.0 };
                                    add_entry(
                                        row,
                                        col,
                                        identity - 0.5 * h * j0[(row_var, col_var)],
                                    );
                                }
                                if let Some(col) = full_to_unknown[base1 + col_var] {
                                    let identity = if row_var == col_var { 1.0 } else { 0.0 };
                                    add_entry(
                                        row,
                                        col,
                                        identity - 0.5 * h * j1[(row_var, col_var)],
                                    );
                                }
                            }
                        }
                    }
                    _ => {
                        for row_var in 0..n_vars {
                            let row = base0 + row_var;
                            for col_var in 0..n_vars {
                                if let Some(col) = full_to_unknown[base0 + col_var] {
                                    let identity = if row_var == col_var { -1.0 } else { 0.0 };
                                    add_entry(row, col, identity - h * j0[(row_var, col_var)]);
                                }
                                if let Some(col) = full_to_unknown[base1 + col_var] {
                                    let identity = if row_var == col_var { 1.0 } else { 0.0 };
                                    add_entry(row, col, identity);
                                }
                            }
                        }
                    }
                }
            }

            for row in 0..n_unknowns {
                for col in 0..n_unknowns {
                    let value = dense[row * n_unknowns + col];
                    if value != 0.0 {
                        triplets.push((row, col, value));
                    }
                }
            }

            vec.from_vector(n_unknowns, n_unknowns, &dense, triplets)
        }))
    });

    Ok(DampedGeneratedSolverState {
        fun,
        jac,
        bounds_vec,
        rel_tolerance_vec,
        variable_string,
        bandwidth: bandwidth_hint.unwrap_or_else(|| infer_bandwidth(n_vars)),
        bc_position_and_value: bc_layout,
        updated_resolver: None,
        selected_backend: SelectedBackendKind::Numeric,
        runtime_diagnostics: HashMap::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_discretization_builds_expected_layout_sizes() {
        let rhs: NumericBvpRhs = Arc::new(|_x, y, _params| DVector::from_vec(vec![y[1], 0.0]));
        let values = vec!["y".to_string(), "z".to_string()];
        let bc = HashMap::from([
            ("y".to_string(), vec![(0usize, 0.0)]),
            ("z".to_string(), vec![(1usize, 1.0)]),
        ]);
        let bounds = HashMap::from([
            ("y".to_string(), (-10.0, 10.0)),
            ("z".to_string(), (-10.0, 10.0)),
        ]);
        let rel = HashMap::from([("y".to_string(), 1e-6), ("z".to_string(), 1e-6)]);
        let mesh = vec![0.0, 0.25, 0.5, 0.75, 1.0];

        let state = build_numeric_generated_solver_state(
            rhs, None, "Sparse", "forward", &values, &bc, &bounds, &rel, 4, &mesh, None, None,
        )
        .expect("numeric discretization should build");

        assert_eq!(state.variable_string.len(), values.len() * 4);
        assert_eq!(state.bounds_vec.len(), values.len() * 4);
        assert_eq!(state.rel_tolerance_vec.len(), values.len() * 4);
        assert_eq!(state.bc_position_and_value.len(), values.len());
    }

    #[test]
    fn numeric_discretization_uses_user_jacobian_when_provided() {
        let rhs: NumericBvpRhs = Arc::new(|_x, y, _params| DVector::from_vec(vec![y[1], 0.0]));
        let jacobian: NumericBvpJacobian =
            Arc::new(|_x, _y, _params| DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]));
        let values = vec!["y".to_string(), "z".to_string()];
        let bc = HashMap::from([
            ("y".to_string(), vec![(0usize, 0.0)]),
            ("z".to_string(), vec![(1usize, 1.0)]),
        ]);
        let bounds = HashMap::from([
            ("y".to_string(), (-10.0, 10.0)),
            ("z".to_string(), (-10.0, 10.0)),
        ]);
        let rel = HashMap::from([("y".to_string(), 1e-6), ("z".to_string(), 1e-6)]);
        let mesh = vec![0.0, 0.25, 0.5, 0.75, 1.0];

        let state = build_numeric_generated_solver_state(
            rhs,
            Some(jacobian),
            "Sparse",
            "forward",
            &values,
            &bc,
            &bounds,
            &rel,
            4,
            &mesh,
            None,
            None,
        )
        .expect("numeric discretization should build");

        assert!(state.jac.is_some());
    }

    #[test]
    fn numeric_discretization_accepts_boundary_conditions_on_same_variable() {
        let rhs: NumericBvpRhs = Arc::new(|_x, y, _params| DVector::from_vec(vec![y[1], -y[0]]));
        let jacobian: NumericBvpJacobian =
            Arc::new(|_x, _y, _params| DMatrix::from_row_slice(2, 2, &[0.0, 1.0, -1.0, 0.0]));
        let values = vec!["y".to_string(), "z".to_string()];
        let bc = HashMap::from([("y".to_string(), vec![(0usize, 0.0), (1usize, 1.0)])]);
        let bounds = HashMap::from([
            ("y".to_string(), (-10.0, 10.0)),
            ("z".to_string(), (-10.0, 10.0)),
        ]);
        let rel = HashMap::from([("y".to_string(), 1e-6), ("z".to_string(), 1e-6)]);
        let mesh = vec![0.0, 0.25, 0.5, 0.75, 1.0];

        let state = build_numeric_generated_solver_state(
            rhs,
            Some(jacobian),
            "Sparse",
            "forward",
            &values,
            &bc,
            &bounds,
            &rel,
            4,
            &mesh,
            None,
            None,
        )
        .expect("numeric discretization should accept two BCs on y and none on z");

        assert_eq!(state.variable_string.len(), values.len() * 4);
        assert_eq!(state.bc_position_and_value.len(), values.len());
        assert!(state.variable_string.iter().any(|name| name == "z_0"));
        assert!(state.variable_string.iter().any(|name| name == "z_4"));
        assert!(state.jac.is_some());
    }

    #[test]
    fn numeric_discretization_rejects_underconstrained_boundary_layout() {
        let rhs: NumericBvpRhs = Arc::new(|_x, y, _params| DVector::from_vec(vec![y[1], -y[0]]));
        let values = vec!["y".to_string(), "z".to_string()];
        let bc = HashMap::from([("y".to_string(), vec![(0usize, 0.0)])]);
        let bounds = HashMap::from([
            ("y".to_string(), (-10.0, 10.0)),
            ("z".to_string(), (-10.0, 10.0)),
        ]);
        let rel = HashMap::from([("y".to_string(), 1e-6), ("z".to_string(), 1e-6)]);
        let mesh = vec![0.0, 0.25, 0.5, 0.75, 1.0];

        let err = match build_numeric_generated_solver_state(
            rhs, None, "Sparse", "forward", &values, &bc, &bounds, &rel, 4, &mesh, None, None,
        ) {
            Ok(_) => panic!("numeric discretization must reject underconstrained BVPs"),
            Err(err) => err,
        };

        assert!(err.contains("expects exactly 2 total boundary conditions"));
    }
}

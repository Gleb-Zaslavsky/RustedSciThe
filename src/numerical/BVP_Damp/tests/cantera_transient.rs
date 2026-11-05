//! Test-only spatial quadrature hypotheses, not a production recovery policy.
//!
//! For the chosen artificial PDE y_tau = f - y_x, RST's integrated residual
//! R = y_right - y_left - integral(f dx) gains integral(y_tau dx). The
//! component mask and the artificial PDE are assumptions of this fixture;
//! neither is inferred from a generic first-order BVP.

use super::*;
use nalgebra::DVector;

pub(super) fn quadrature_mass(
    mesh: &[f64],
    components: usize,
    unknown_full_positions: &[usize],
    differential: &[bool],
    trapezoid: bool,
) -> DMatrix<f64> {
    let mut mass = DMatrix::zeros((mesh.len() - 1) * components, unknown_full_positions.len());
    for interval in 0..mesh.len() - 1 {
        let h = mesh[interval + 1] - mesh[interval];
        for component in 0..components {
            if !differential[component] {
                continue;
            }
            let row = interval * components + component;
            for (column, &position) in unknown_full_positions.iter().enumerate() {
                let left = interval * components + component;
                let right = (interval + 1) * components + component;
                if position == left {
                    mass[(row, column)] += h * if trapezoid { 0.5 } else { 1.0 };
                } else if trapezoid && position == right {
                    mass[(row, column)] += h * 0.5;
                }
            }
        }
    }
    mass
}

fn fixed_grid_solver(scheme: &str, mesh: &[f64]) -> NRBVP {
    let mut solver = NRBVP::new_with_options(
        vec![
            Expr::parse_expression("-u+v"),
            Expr::parse_expression("u-v"),
        ],
        DMatrix::from_element(2, 2, 0.5),
        vec!["u".into(), "v".into()],
        "x".into(),
        HashMap::from([("u".into(), vec![(0, 1.0)]), ("v".into(), vec![(1, 0.25)])]),
        0.0,
        1.0,
        2,
        DampedSolverOptions::dense_damped()
            .with_scheme_name(scheme)
            .with_bounds(HashMap::from([
                ("u".into(), (-10.0, 10.0)),
                ("v".into(), (-10.0, 10.0)),
            ]))
            .with_rel_tolerance(HashMap::from([("u".into(), 1e-6), ("v".into(), 1e-6)])),
    );
    solver
        .try_eq_generate(Some(mesh.to_vec()), None)
        .expect("symbolic callbacks should prepare");
    solver
}

#[test]
fn interval_quadrature_with_mixed_boundaries_is_not_a_diagonal_shift() {
    let mesh = [0.0, 0.2, 1.0];
    let unknowns = [1, 2, 3, 4];
    let forward = quadrature_mass(&mesh, 2, &unknowns, &[true, true], false);
    let trapezoid = quadrature_mass(&mesh, 2, &unknowns, &[true, true], true);

    // Rows are (interval, equation), columns are free node-major values.
    assert_eq!(forward.row(0).norm(), 0.0); // u(left) is fixed.
    assert!((forward[(1, 0)] - 0.2).abs() < 1e-14);
    assert!((forward[(2, 1)] - 0.8).abs() < 1e-14);
    assert!((forward[(3, 2)] - 0.8).abs() < 1e-14);
    assert_eq!(forward.column(3).norm(), 0.0);
    assert!((trapezoid[(0, 1)] - 0.1).abs() < 1e-14);
    assert!((trapezoid[(2, 1)] - 0.4).abs() < 1e-14);
    assert!((trapezoid[(2, 3)] - 0.4).abs() < 1e-14);

    let mixed_mask = quadrature_mass(&mesh, 2, &unknowns, &[true, false], true);
    assert_eq!(mixed_mask.row(1).norm(), 0.0);
    assert_eq!(mixed_mask.row(3).norm(), 0.0);
    assert!(mixed_mask[(2, 3)] > 0.0);
}

#[test]
fn right_endpoint_time_quadrature_respects_mixed_boundary_elimination() {
    let mesh = [0.0, 0.2, 1.0];
    let unknowns = [1, 2, 3, 4];
    let left = quadrature_mass(&mesh, 2, &unknowns, &[true, true], false);
    let average = quadrature_mass(&mesh, 2, &unknowns, &[true, true], true);
    let right = 2.0 * average - left;
    assert!((right[(0, 1)] - 0.2).abs() < 1e-14);
    assert!((right[(1, 2)] - 0.2).abs() < 1e-14);
    assert!((right[(2, 3)] - 0.8).abs() < 1e-14);
    assert_eq!(right.row(3).norm(), 0.0);
    assert_eq!(right.column(0).norm(), 0.0);
}

fn check_candidate_transient_jacobian(scheme: &str, trapezoid: bool) {
    let mesh = [0.0, 0.2, 1.0];
    let unknowns = [1, 2, 3, 4];
    let state = DVector::from_vec(vec![0.4, 0.9, 0.6, 0.7]);
    let previous = DVector::from_vec(vec![0.2, 0.8, 0.5, 0.6]);
    let dt = 0.3;

    let mut solver = fixed_grid_solver(scheme, &mesh);
    let mut bc_positions: Vec<_> = solver.BC_position_and_value.iter().map(|bc| bc.0).collect();
    bc_positions.sort_unstable();
    assert_eq!(bc_positions, [0, 5]);
    let mass = quadrature_mass(&mesh, 2, &unknowns, &[true, false], trapezoid);
    let steady = |y: &DVector<f64>| solver.fun.try_call(0.0, y).unwrap().to_DVectorType();
    let old_residual = steady(&previous);
    let unchanged_at_previous = steady(&previous) + &mass * (&previous - &previous) / dt;
    assert!((&unchanged_at_previous - old_residual).norm() < 1e-14);

    let jacobian = solver
        .jac
        .as_mut()
        .expect("Jacobian callback should exist")
        .try_call(0.0, &state)
        .unwrap()
        .to_DMatrixType();
    let candidate_jacobian = jacobian + &mass / dt;
    let eps = 1e-6;
    for column in 0..state.len() {
        let mut plus = state.clone();
        let mut minus = state.clone();
        plus[column] += eps;
        minus[column] -= eps;
        let steady_plus = solver.fun.try_call(0.0, &plus).unwrap().to_DVectorType();
        let steady_minus = solver.fun.try_call(0.0, &minus).unwrap().to_DVectorType();
        let steady_difference = (&steady_plus - &steady_minus) / (2.0 * eps);
        let expected_steady = candidate_jacobian.column(column) - mass.column(column) / dt;
        let steady_error = (&steady_difference - expected_steady).amax();
        assert!(
            steady_error < 1e-7,
            "{scheme} steady Jacobian column {column}: error {steady_error:e}; finite_difference={steady_difference:?}, candidate={candidate_jacobian:?}, mass={mass:?}"
        );
        let residual_plus = steady_plus + &mass * (&plus - &previous) / dt;
        let residual_minus = steady_minus + &mass * (&minus - &previous) / dt;
        let finite_difference = (residual_plus - residual_minus) / (2.0 * eps);
        let error = (&finite_difference - candidate_jacobian.column(column)).amax();
        assert!(
            error < 1e-7,
            "{scheme} column {column}: finite-difference error {error:e}"
        );
    }
}

#[test]
fn forward_candidate_transient_jacobian_matches_finite_difference() {
    check_candidate_transient_jacobian("forward", false);
}

#[test]
fn numeric_trapezoid_jacobian_and_residual_confirm_the_independent_control() {
    use crate::numerical::BVP_Damp::numeric_discretization::{
        NumericBvpJacobian, NumericBvpRhs, build_numeric_generated_solver_state,
    };
    let mesh = [0.0, 0.2, 1.0];
    let values = vec!["u".to_string(), "v".to_string()];
    let bc = HashMap::from([("u".into(), vec![(0, 1.0)]), ("v".into(), vec![(1, 0.25)])]);
    let bounds = HashMap::from([("u".into(), (-10.0, 10.0)), ("v".into(), (-10.0, 10.0))]);
    let rel = HashMap::from([("u".into(), 1e-6), ("v".into(), 1e-6)]);
    let rhs: NumericBvpRhs = Arc::new(|_, y, _| DVector::from_vec(vec![-y[0] + y[1], y[0] - y[1]]));
    let jac: NumericBvpJacobian =
        Arc::new(|_, _, _| DMatrix::from_row_slice(2, 2, &[-1.0, 1.0, 1.0, -1.0]));
    let mut numeric = build_numeric_generated_solver_state(
        rhs,
        Some(jac),
        "Dense",
        "trapezoid",
        &values,
        &bc,
        &bounds,
        &rel,
        2,
        &mesh,
        None,
        None,
    )
    .expect("numeric control should prepare");
    let symbolic = fixed_grid_solver("trapezoid", &mesh);
    let state = DVector::from_vec(vec![0.4, 0.9, 0.6, 0.7]);
    let numeric_residual = numeric.fun.try_call(0.0, &state).unwrap().to_DVectorType();
    let symbolic_residual = symbolic.fun.try_call(0.0, &state).unwrap().to_DVectorType();
    assert!((&numeric_residual - &symbolic_residual).amax() < 1e-12);
    let numeric_jacobian = numeric
        .jac
        .as_mut()
        .unwrap()
        .try_call(0.0, &state)
        .unwrap()
        .to_DMatrixType();
    assert!((numeric_jacobian[(1, 1)] + 0.1).abs() < 1e-12);
    let mut plus = state.clone();
    let mut minus = state.clone();
    plus[1] += 1e-6;
    minus[1] -= 1e-6;
    let finite_difference = (symbolic.fun.try_call(0.0, &plus).unwrap().to_DVectorType()
        - symbolic.fun.try_call(0.0, &minus).unwrap().to_DVectorType())
        / 2e-6;
    assert!((finite_difference[1] - numeric_jacobian[(1, 1)]).abs() < 1e-7);

    let previous = DVector::from_vec(vec![0.2, 0.8, 0.5, 0.6]);
    let mass = quadrature_mass(&mesh, 2, &[1, 2, 3, 4], &[true, false], true);
    let dt = 0.3;
    let transient_jacobian = &numeric_jacobian + &mass / dt;
    let residual = |y: &DVector<f64>| {
        numeric.fun.try_call(0.0, y).unwrap().to_DVectorType() + &mass * (y - &previous) / dt
    };
    for column in 0..state.len() {
        let mut plus = state.clone();
        let mut minus = state.clone();
        plus[column] += 1e-6;
        minus[column] -= 1e-6;
        let fd = (residual(&plus) - residual(&minus)) / 2e-6;
        assert!((fd - transient_jacobian.column(column)).amax() < 1e-7);
    }
    let correction = transient_jacobian
        .lu()
        .solve(&residual(&state))
        .expect("the test-only linear transient system should be nonsingular");
    let next = &state - correction;
    assert!(residual(&next).norm() < 1e-12);
    assert_eq!(previous, DVector::from_vec(vec![0.2, 0.8, 0.5, 0.6]));
}

#[test]
#[ignore = "diagnostic: ExprLegacy Trapezoid dependency inspection"]
fn diagnose_trapezoid_dependency_tracking_for_cross_component_right_node() {
    use crate::symbolic::symbolic_functions_BVP::Jacobian;
    let mut symbolic = Jacobian::new();
    symbolic.discretization_system_BVP_par(
        vec![
            Expr::parse_expression("-u+v"),
            Expr::parse_expression("u-v"),
        ],
        vec!["u".into(), "v".into()],
        "x".into(),
        0.0,
        Some(2),
        None,
        Some(vec![0.0, 0.2, 1.0]),
        HashMap::from([("u".into(), vec![(0, 1.0)]), ("v".into(), vec![(1, 0.25)])]),
        None,
        None,
        "trapezoid".into(),
    );
    let derivative = Expr::diff(&symbolic.vector_of_functions[1], "u_1").simplify();
    assert!(!derivative.is_zero(), "the residual depends on u_1");
    println!(
        "row 1 derivative w.r.t. u_1: {derivative}; tracked dependencies: {:?}",
        symbolic.variables_for_all_disrete[1]
    );
}

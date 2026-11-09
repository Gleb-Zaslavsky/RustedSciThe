use crate::numerical::BVP_Damp::BVP_traits::{MatrixType, VectorType};

use nalgebra::DMatrix;

use crate::numerical::BVP_Damp::NR_Damp_solver_damped::SolverParams;
use crate::numerical::logging::solver_info as info;
use std::collections::HashMap;
////////////////////////////////////BOUND STEP  CONDITIONS////////////////////////////////////
/// Inspired by the Cantera MultiNewton bound-step guard.
///
/// This legacy helper takes an already proposed `y_new`. The active Damped
/// solver path uses [`bound_step_Cantera2`] instead, because Newton steps are
/// represented as additive updates there: `x_next = x + step`.
pub fn bound_step_Cantera(
    y: &dyn VectorType,
    y_new: &dyn VectorType,
    bounds: &Vec<(f64, f64)>,
) -> f64 {
    let mut fbound = 1.0;
    //println!("bounds = {:?}", bounds.len());
    //  println!("y_new = {:?}", y_new.to_DVectorType()  );

    for (i, y_i) in y.iterate().enumerate() {
        let below = bounds[i].0;
        let above = bounds[i].1;
        let y_new_i = y_new.get_val(i);
        if y_i < below {
            fbound = ((y_i - below) / (y_i - y_new_i)).min(fbound);
        }
        if y_i > above {
            fbound = (((above - y_i) / (y_new_i - y_i)).min(fbound)).max(0.0);
        }
    }
    fbound
}
/// Inspired by the Cantera MultiNewton bound-step guard.
///
/// Computes the feasible factor for an additive update. Newton routes whose
/// applied correction has a different sign must use
/// [`bound_step_Cantera2_with_update_sign`] so the guard checks the candidate
/// that will actually be applied.
pub fn bound_step_Cantera2(
    x: &dyn VectorType,
    step: &dyn VectorType,
    bounds: &Vec<(f64, f64)>,
) -> f64 {
    bound_step_Cantera2_with_update_sign(x, step, bounds, 1.0)
}

/// Computes the largest feasible factor for `x_next = x + sign * factor * step`.
///
/// The public compatibility helper above retains its additive-update contract.
/// Damped Newton uses a subtractive Newton correction and passes `sign = -1`;
/// sharing the scalar loop avoids allocating a negated vector per iteration.
pub(crate) fn bound_step_Cantera2_with_update_sign(
    x: &dyn VectorType,
    step: &dyn VectorType,
    bounds: &[(f64, f64)],
    sign: f64,
) -> f64 {
    bound_step_Cantera2_with_update_sign_and_limit(x, step, bounds, sign).0
}

pub(crate) fn bound_step_Cantera2_with_update_sign_and_limit(
    x: &dyn VectorType,
    step: &dyn VectorType,
    bounds: &[(f64, f64)],
    sign: f64,
) -> (f64, Option<usize>) {
    assert_eq!(x.len(), bounds.len());
    assert_eq!(x.len(), step.len());
    debug_assert!(sign == 1.0 || sign == -1.0);
    let mut fbound: f64 = 1.0;
    let mut limiting_index = None;

    for (i, val) in x.iterate().enumerate() {
        let below = bounds[i].0;
        let above = bounds[i].1;
        let newval = val + sign * step.get_val(i);

        if newval > above {
            let candidate_factor = ((above - val) / (newval - val)).max(0.0);
            if candidate_factor < fbound {
                fbound = candidate_factor;
                limiting_index = Some(i);
            }
        } else if newval < below {
            let candidate_factor = (val - below) / (val - newval);
            if candidate_factor < fbound {
                fbound = candidate_factor;
                limiting_index = Some(i);
            }
        }
    }

    (fbound, limiting_index)
}

pub(crate) fn first_bound_violation(
    state: &dyn VectorType,
    bounds: &[(f64, f64)],
) -> Option<(usize, f64, f64, f64)> {
    if bounds.is_empty() {
        return None;
    }
    debug_assert_eq!(state.len(), bounds.len());
    state.iterate().enumerate().find_map(|(index, value)| {
        let (lower, upper) = bounds[index];
        (!value.is_finite() || value < lower || value > upper)
            .then_some((index, value, lower, upper))
    })
}

/// Repair a fraction-to-boundary factor if floating-point rounding makes its
/// resulting state fall outside an inclusive bound. The common feasible path
/// is one allocation-free scan; the bitwise bisection runs only on rounding
/// edge cases and returns the largest representable feasible factor.
pub(crate) fn feasible_update_factor(
    state: &dyn VectorType,
    step: &dyn VectorType,
    bounds: &[(f64, f64)],
    sign: f64,
    maximum_factor: f64,
) -> f64 {
    debug_assert_eq!(state.len(), step.len());
    debug_assert_eq!(state.len(), bounds.len());

    let is_feasible = |factor: f64| {
        state.iterate().enumerate().all(|(index, value)| {
            let candidate = value + sign * (step.get_val(index) * factor);
            let (lower, upper) = bounds[index];
            candidate.is_finite() && candidate >= lower && candidate <= upper
        })
    };

    if is_feasible(maximum_factor) {
        return maximum_factor;
    }
    if !is_feasible(0.0) {
        return 0.0;
    }

    let mut feasible_bits = 0.0_f64.to_bits();
    let mut infeasible_bits = maximum_factor.to_bits();
    while infeasible_bits - feasible_bits > 1 {
        let midpoint_bits = feasible_bits + (infeasible_bits - feasible_bits) / 2;
        let midpoint = f64::from_bits(midpoint_bits);
        if is_feasible(midpoint) {
            feasible_bits = midpoint_bits;
        } else {
            infeasible_bits = midpoint_bits;
        }
    }
    f64::from_bits(feasible_bits)
}

// This function calculates the minimum damping factor necessary to keep the solution within specified bounds after taking a Newton step.
// twopnt version
pub fn bound_step(y: &dyn VectorType, step: &dyn VectorType, bounds: &Vec<(f64, f64)>) -> f64 {
    // Initialize no damping
    let mut fbound = 1.0;
    let mut _entry = 0;
    let mut _force = false;
    let mut _value = 0.0;
    let s0 = step;
    for (i, y_i) in y.iterate().enumerate() {
        let below = bounds[i].0;
        let above = bounds[i].1;

        let s_i = s0.get_val(i);
        if s_i > f64::max(y_i - below, 0.0) {
            let temp = (y_i - below) / s_i;
            if temp < fbound {
                fbound = temp;
                _entry = i + 1; //
                _force = true;
                _value = below;
            }
        } else if s_i < f64::min(y_i - above, 0.0) {
            let temp = (y_i - above) / s_i;
            if temp < fbound {
                fbound = temp;
                _entry = i + 1; //
                _force = true;
                _value = above;
            }
        }
    }
    fbound
}

#[cfg(test)]
mod bound_step_tests {
    use super::*;
    use nalgebra::DVector;

    fn dense(values: &[f64]) -> DVector<f64> {
        DVector::from_column_slice(values)
    }

    fn assert_close(lhs: f64, rhs: f64) {
        assert!((lhs - rhs).abs() < 1e-12, "expected {rhs}, got {lhs}");
    }

    #[test]
    fn reduced_weighted_norm_matches_full_grid_with_zero_boundary_corrections() {
        // Node-major state: the large values at (node 0, component 0) and
        // (node 1, component 1) are prescribed BCs, not Newton unknowns.
        let state = dense(&[10.0, 20.0, 30.0, 40.0]);
        let step = dense(&[0.5, 2.0, 1.0, 4.0]);
        let full_positions = [1, 2, 4, 5];
        let boundary_values = [(0, 0, 100.0), (3, 1, 200.0)];
        let rel = [0.01, 0.1, 0.1, 0.01];
        let mut sums = [0.0; 2];

        let actual = weighted_norm_reduced_unknowns(
            &state,
            &step,
            &rel,
            0.5,
            2,
            3,
            &full_positions,
            &boundary_values,
            &mut sums,
        )
        .expect("valid reduced layout should produce a norm");

        // Component scales follow Cantera: all nodes, including the eliminated
        // BC entries, contribute to the mean state magnitude.
        let scale_0 = 0.5 + 0.1 * ((100.0_f64 + 20.0 + 30.0) / 3.0);
        let scale_1 = 0.5 + 0.01 * ((10.0_f64 + 200.0 + 40.0) / 3.0);
        let expected = (((0.5_f64 / scale_1).powi(2)
            + (2.0_f64 / scale_0).powi(2)
            + (1.0_f64 / scale_0).powi(2)
            + (4.0_f64 / scale_1).powi(2))
            / 6.0_f64)
            .sqrt();

        assert_close(actual, expected);
        let full_state = dense(&[100.0, 10.0, 20.0, 200.0, 30.0, 40.0]);
        let full_step = dense(&[0.0, 0.5, 2.0, 0.0, 1.0, 4.0]);
        let full_norm = weighted_norm(&full_state, &full_step, &vec![0.1, 0.01], 0.5, 2, 3);
        assert_close(actual, full_norm);
        assert_eq!(sums, [150.0, 250.0]);
    }

    #[test]
    fn reduced_weighted_norm_is_invariant_to_consistent_unit_scaling() {
        let state = dense(&[2.0, 20.0, 6.0, 60.0]);
        let step = dense(&[0.2, 2.0, 0.4, 4.0]);
        let scaled_state = dense(&[2000.0, 20000.0, 6000.0, 60000.0]);
        let scaled_step = dense(&[200.0, 2000.0, 400.0, 4000.0]);
        let rel = [0.1; 4];
        let full_positions = [0, 1, 4, 5];
        let boundary_values = [(2, 0, 4.0), (3, 1, 40.0)];
        let scaled_boundary_values = [(2, 0, 4000.0), (3, 1, 40000.0)];
        let mut sums = [0.0; 2];

        let original = weighted_norm_reduced_unknowns(
            &state,
            &step,
            &rel,
            0.01,
            2,
            3,
            &full_positions,
            &boundary_values,
            &mut sums,
        )
        .unwrap();
        let scaled = weighted_norm_reduced_unknowns(
            &scaled_state,
            &scaled_step,
            &rel,
            10.0,
            2,
            3,
            &full_positions,
            &scaled_boundary_values,
            &mut sums,
        )
        .unwrap();

        assert_close(original, scaled);
    }

    #[test]
    fn reduced_weighted_norm_rejects_invalid_layout_and_nonfinite_data() {
        let state = dense(&[1.0, 2.0]);
        let step = dense(&[0.1, 0.2]);
        let boundary_values = [(2, 0, 3.0), (3, 1, 4.0)];
        let mut sums = [0.0; 2];
        let nonfinite_step = dense(&[f64::NAN, 0.2]);
        assert!(
            weighted_norm_reduced_unknowns(
                &state,
                &nonfinite_step,
                &[0.1, 0.1],
                1e-6,
                2,
                2,
                &[0, 1],
                &boundary_values,
                &mut sums,
            )
            .is_none()
        );
        assert!(
            weighted_norm_reduced_unknowns(
                &state,
                &step,
                &[0.1],
                1e-6,
                2,
                2,
                &[0, 1],
                &boundary_values,
                &mut sums,
            )
            .is_none()
        );
        assert!(
            weighted_norm_reduced_unknowns(
                &state,
                &step,
                &[0.1, 0.1],
                1e-6,
                2,
                1,
                &[0, 0],
                &boundary_values,
                &mut sums,
            )
            .is_none()
        );
        assert!(
            weighted_norm_reduced_unknowns(
                &state,
                &step,
                &[0.1, 0.1],
                1e-6,
                2,
                2,
                &[0, 0],
                &boundary_values,
                &mut sums,
            )
            .is_none()
        );
        assert!(
            weighted_norm_reduced_unknowns(
                &state,
                &step,
                &[0.1, 0.1],
                1e-6,
                2,
                2,
                &[0, 9],
                &boundary_values,
                &mut sums,
            )
            .is_none()
        );
    }

    #[test]
    fn cantera2_keeps_full_step_when_trial_point_is_inside_bounds() {
        let x = dense(&[0.5, -0.2]);
        let step = dense(&[0.1, -0.3]);
        let bounds = vec![(0.0, 1.0), (-1.0, 1.0)];

        assert_close(bound_step_Cantera2(&x, &step, &bounds), 1.0);
    }

    #[test]
    fn cantera2_clips_upper_and_lower_outward_steps() {
        let upper_x = dense(&[0.8]);
        let upper_step = dense(&[0.5]);
        let lower_x = dense(&[0.2]);
        let lower_step = dense(&[-0.5]);
        let bounds = vec![(0.0, 1.0)];

        assert_close(bound_step_Cantera2(&upper_x, &upper_step, &bounds), 0.4);
        assert_close(bound_step_Cantera2(&lower_x, &lower_step, &bounds), 0.4);
    }

    #[test]
    fn cantera2_uses_most_restrictive_component() {
        let x = dense(&[0.8, 0.1, 0.5]);
        let step = dense(&[0.5, -0.5, 0.1]);
        let bounds = vec![(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)];

        assert_close(bound_step_Cantera2(&x, &step, &bounds), 0.2);
    }

    #[test]
    fn cantera2_returns_zero_when_current_boundary_value_moves_outward() {
        let at_upper = dense(&[1.0]);
        let upper_step = dense(&[0.1]);
        let at_lower = dense(&[0.0]);
        let lower_step = dense(&[-0.1]);
        let bounds = vec![(0.0, 1.0)];

        assert_close(bound_step_Cantera2(&at_upper, &upper_step, &bounds), 0.0);
        assert_close(bound_step_Cantera2(&at_lower, &lower_step, &bounds), 0.0);
    }

    #[test]
    fn subtractive_newton_update_limits_the_actual_candidate_direction() {
        let x = dense(&[0.0, 1.0, 0.5]);
        // The runtime update is x - lambda * delta. The first two corrections
        // move inward from their active bounds; the third crosses the lower
        // bound and therefore determines the limiting factor.
        let delta = dense(&[-0.1, 0.1, 0.8]);
        let bounds = vec![(0.0, 1.0); 3];

        let factor = bound_step_Cantera2_with_update_sign(&x, &delta, &bounds, -1.0);

        assert_close(factor, 0.625);
        for i in 0..x.len() {
            let candidate = x[i] - factor * delta[i];
            assert!(candidate >= bounds[i].0 - 1e-12);
            assert!(candidate <= bounds[i].1 + 1e-12);
        }
    }

    #[test]
    fn feasible_factor_repairs_one_ulp_boundary_overshoot() {
        let state = dense(&[1.1]);
        let step = dense(&[1.2000000000000002]);
        let bounds = [(-0.1, 2.0)];
        let raw_factor = bound_step_Cantera2_with_update_sign(&state, &step, &bounds, -1.0);
        let raw_candidate = 1.1 - step.get_val(0) * raw_factor;
        assert!(
            raw_candidate < bounds[0].0,
            "fixture must reproduce rounding overshoot"
        );

        let safe_factor = feasible_update_factor(&state, &step, &bounds, -1.0, raw_factor);
        let safe_candidate = 1.1 - step.get_val(0) * safe_factor;
        assert!(safe_factor <= raw_factor);
        assert!(safe_candidate >= bounds[0].0 && safe_candidate <= bounds[0].1);
    }

    #[test]
    fn subtractive_newton_update_blocks_only_a_true_outward_boundary_direction() {
        let at_lower = dense(&[0.0]);
        let inward_delta = dense(&[-0.1]);
        let outward_delta = dense(&[0.1]);
        let bounds = vec![(0.0, 1.0)];

        assert_close(
            bound_step_Cantera2_with_update_sign(&at_lower, &inward_delta, &bounds, -1.0),
            1.0,
        );
        assert_close(
            bound_step_Cantera2_with_update_sign(&at_lower, &outward_delta, &bounds, -1.0),
            0.0,
        );
    }

    #[test]
    fn cantera2_scaled_step_does_not_cross_bounds() {
        let x = dense(&[0.8, 0.2]);
        let step = dense(&[0.5, -0.5]);
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let fbound = bound_step_Cantera2(&x, &step, &bounds);

        for i in 0..x.len() {
            let next = x[i] + fbound * step[i];
            assert!(
                next >= bounds[i].0 - 1e-12 && next <= bounds[i].1 + 1e-12,
                "component {i}: {next} is outside {:?}",
                bounds[i]
            );
        }
    }

    #[test]
    fn legacy_twopnt_bound_step_uses_different_step_sign_convention() {
        let x = dense(&[0.8]);
        let additive_step = dense(&[0.5]);
        let bounds = vec![(0.0, 1.0)];

        let additive_contract = bound_step_Cantera2(&x, &additive_step, &bounds);
        let legacy_twopnt_contract = bound_step(&x, &additive_step, &bounds);

        assert_close(additive_contract, 0.4);
        assert_close(legacy_twopnt_contract, 1.0);
    }
}

////////////////////////////////////JACOBIAN RECALCULATION STRATEGY////////////////////////////////////
pub fn jac_recalc(
    strategy_params: &Option<SolverParams>,
    m: usize,
    old_jac: &Option<Box<dyn MatrixType>>,
    old_recalc_flag: &mut bool,
) -> bool {
    let m_from_task = strategy_params
        .as_ref()
        .and_then(|p| p.max_jac)
        .unwrap_or(3);

    // when jac is None it means this is first iteration
    if old_jac.is_none() || m > m_from_task {
        info!(
            "\n number of iterations with old jac {} is higher then threshold {}",
            m, m_from_task
        );
        true
    } else {
        *old_recalc_flag
    }
}
/////////////////////////////////////////////CONVERGENCE CONDITIONS////////////////////////////////////
/// Legacy Damped scalar threshold, retained only by the legacy Newton policy.
/// The CorrectionNewton candidate uses a full-grid component-scaled RMS cutoff
/// of one instead.
pub fn convergence_condition(
    y: &dyn VectorType,
    abs_tolerance: &f64,
    rel_tolerance_vec: &Vec<f64>,
) -> f64 {
    let mut sum = 0.0;
    for (i, y_i) in y.iterate().enumerate() {
        sum += (y_i).abs() * rel_tolerance_vec[i];
    }
    // let sum = sum/(y.len() as f64);
    let conv = sum.max(*abs_tolerance);

    conv
}

// Usage in damped_step would be:
// let s1 = convergence_condition2(
//    &*y_k_plus_1,
//    &*undamped_step_k_plus_1,
//    &self.rel_tolerance_vec,
//    &self.abs_tolerance_vec, // You'd need this as Vec<f64>
//    self.values.len(),
//    self.n_steps
//);

/// convergence condition
///  insired by MuliNewton cpp version
pub fn weighted_norm(
    x: &dyn VectorType,
    step: &dyn VectorType,
    rel_tolerance_vec: &Vec<f64>,
    abs_tolerance: f64,
    n_components: usize,
    n_points: usize,
) -> f64 {
    let mut sum = 0.0;

    for n in 0..n_components {
        // Calculate average magnitude for component n
        let mut esum = 0.0;
        for j in 0..n_points {
            let idx = j * n_components + n;
            esum += x.get_val(idx).abs();
        }

        // Weight = rtol * average + atol
        let ewt = rel_tolerance_vec[n] * esum / (n_points as f64) + abs_tolerance;

        // Sum of squared normalized steps for component n
        for j in 0..n_points {
            let idx = j * n_components + n;
            let f = step.get_val(idx) / ewt;
            sum += f * f;
        }
    }

    // Average over all points and take square root
    sum /= (n_components * n_points) as f64;
    sum.sqrt()
}

/// Compute Cantera's weighted correction norm for the boundary-eliminated
/// Damped system without materializing a full state vector.
///
/// Component scales include every grid point, including prescribed boundary
/// values. Eliminated boundary corrections are zero, but Cantera's RMS
/// denominator includes the entire full-grid state. `full_positions` maps
/// each packed entry to its node-major full-state position; `component_sums`
/// is caller-owned scratch and is reused by line-search trials.
#[cfg(test)]
pub(crate) fn weighted_norm_reduced_unknowns(
    state: &dyn VectorType,
    step: &dyn VectorType,
    rel_tolerances: &[f64],
    abs_tolerance: f64,
    n_components: usize,
    n_points: usize,
    full_positions: &[usize],
    boundary_values: &[(usize, usize, f64)],
    component_sums: &mut [f64],
) -> Option<f64> {
    let len = step.len();
    if len == 0
        || state.len() != len
        || n_components == 0
        || n_points == 0
        || rel_tolerances.len() != len
        || full_positions.len() != len
        || component_sums.len() != n_components
        || len.checked_add(boundary_values.len())? != n_components.checked_mul(n_points)?
        || !abs_tolerance.is_finite()
        || abs_tolerance <= 0.0
    {
        return None;
    }

    component_sums.fill(0.0);
    let full_len = n_components.checked_mul(n_points)?;
    let mut occupied = vec![false; full_len];
    for &(position, _, value) in boundary_values {
        if position >= full_len || occupied[position] || !value.is_finite() {
            return None;
        }
        occupied[position] = true;
        component_sums[position % n_components] += value.abs();
    }
    for (index, &position) in full_positions.iter().enumerate() {
        let value = state.get_val(index);
        if position >= full_len || occupied[position] || !value.is_finite() {
            return None;
        }
        occupied[position] = true;
        component_sums[position % n_components] += value.abs();
    }

    let mut scaled_rms = 0.0_f64;
    for i in 0..len {
        let position = full_positions[i];
        let component = position % n_components;
        let tolerance = rel_tolerances[i];
        if !tolerance.is_finite() || tolerance < 0.0 {
            return None;
        }
        let mean_magnitude = component_sums[component] / n_points as f64;
        let scale = abs_tolerance + tolerance * mean_magnitude;
        let correction = step.get_val(i);
        if !correction.is_finite() || !scale.is_finite() || scale <= 0.0 {
            return None;
        }
        scaled_rms = scaled_rms.hypot(correction / scale);
    }

    let result = scaled_rms / (full_len as f64).sqrt();
    result.is_finite().then_some(result)
}
pub fn if_initial_guess_inside_bounds(
    initial_guess: &DMatrix<f64>,
    Bounds: &Option<HashMap<String, (f64, f64)>>,
    values: Vec<String>,
) -> () {
    //  initial_guess - matrix with number of rows equal to the number of unknown vars, and number of columns equal to the number of steps
    //  Bounds - hashmap where keys are variable names and values are tuples with lower and upper bounds.
    //  Function checks if initial guess is inside the bounds of the variables. If not, it panics.
    for (i, row) in initial_guess.row_iter().enumerate() {
        let var_name = values[i].clone();
        let bounds = Bounds.as_ref().unwrap().get(&var_name).unwrap();
        let (lower, upper) = bounds;
        if row.iter().any(|&x| x < *lower || x > *upper) {
            panic!(
                "\n, \n, Initial guess  of the variable {} is outside the bounds {:?}.",
                var_name, &bounds
            );
        }
    }
}

/*
pub fn count_left_and_right_BC(BorderConditions: &HashMap<String, Vec<(usize, f64)>>) -> (usize, usize) {

    let left_cond = BorderConditions.iter   ().map(|(index, value)|
        {if index==0
        {value.len()}
        })
        .sum();
        let right_condition = BorderConditions.iter   ().map(|(index, value)|
        {if index==0
        {value.len()}
        })
        .sum();
    (left_cond, right_condition)
}
*/
// sometimes order of indexed variables (and therefore order of values in numerical result y_DMatrix ) and order of original variables turns out to be
// not the same: in that case permutation is needed to transform result matrix into the same order of

/*
To avoid this difficulty, the solution of the problem is  carried out in E-steps.  The problem is first solved (via the above iterative process) for
a modest value of E (e.g.,  E  =  1e-1 or 1e-2), and then,  in turn,  for  values of  e smaller  than the preceding  value  of  E  by a
factor of about 3.  The mesh  point set used at the completion of  the preceding step forms the initial set for the new step
 ON  A  DIFFERENTIAL  EQUATION  OF  BOUNDARY  LAYER  TYPE
By CARL  E.  PEARSON,
p. 138
*/

use crate::numerical::BVP_Damp::BVP_traits::{MatrixType, VectorType};

use nalgebra::DMatrix;

use crate::numerical::BVP_Damp::NR_Damp_solver_damped::SolverParams;
use std::collections::HashMap;
////////////////////////////////////BOUND STEP  CONDITIONS////////////////////////////////////
/// insired by MuliNewton cpp version of bound step
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
/// insired by MuliNewton cpp version of bound step
pub fn bound_step_Cantera2(
    x: &dyn VectorType,
    step: &dyn VectorType,
    bounds: &Vec<(f64, f64)>,
) -> f64 {
    assert_eq!(x.len(), bounds.len());
    let mut fbound: f64 = 1.0;

    for (i, val) in x.iterate().enumerate() {
        let below = bounds[i].0;
        let above = bounds[i].1;
        let step_i = step.get_val(i);
        let newval = val + step_i;

        if newval > above {
            fbound = fbound.min((above - val) / (newval - val)).max(0.0);
        } else if newval < below {
            fbound = fbound.min((val - below) / (val - newval));
        }
    }

    fbound
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
        log::info!(
            "\n number of iterations with old jac {} is higher then threshold {}",
            m,
            m_from_task
        );
        true
    } else {
        *old_recalc_flag
    }
}
/////////////////////////////////////////////CONVERGENCE CONDITIONS////////////////////////////////////
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

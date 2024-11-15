use crate::numerical::BVP_Damp::BVP_traits::{
    MatrixType, VectorType
    ,
};
use crate::symbolic::symbolic_functions::Jacobian;
use nalgebra::DMatrix;
use std::collections::HashMap;
use simplelog::*;
use std::fs::File;
// This function calculates the minimum damping factor necessary to keep the solution within specified bounds after taking a Newton step.
/*
pub fn bound_step(y:&dyn VectorType,  y_new:&dyn VectorType, bounds:&Vec<(f64, f64)>) -> f64 {
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
        if  y_i > above {
            fbound =  (( (above - y_i) / (y_new_i- y_i) ).min(fbound)  ).max(0.0);
        }
    };
    fbound

}
*/
pub fn jac_recalc(
    strategy_params: &Option<HashMap<String, Option<Vec<f64>>>>,
    m: usize,
    old_jac: &Option<Box<dyn MatrixType>>,
    old_recalc_flag: &mut bool,
) -> bool {
    //let m_from_task = strategy_params.clone().unwrap().values().collect::<Vec<&Option<Vec<f64>>>>()[0].clone().unwrap()[0] as usize;
    let m_from_task =
        if let Some(m_from_task_vec) = strategy_params.clone().unwrap().get("max_jac").unwrap() {
            m_from_task_vec[0] as usize
        } else {
            3
        };
    // when jac is None it means this is first iteration
    if old_jac.is_none() || m > m_from_task {
        log::info!(
            "\n number of iterations with old jac {} is higher then threshold {}",
            m, m_from_task
        );
        true
    } else {
        *old_recalc_flag
    }
}

pub fn convergence_condition(
    y: &dyn VectorType,
    abs_tolerance: &f64,
    rel_tolerance_vec: &Vec<f64>,
) -> f64 {
    let mut sum = 0.0;
    for (i, y_i) in y.iterate().enumerate() {
        sum += y_i * rel_tolerance_vec[i];
    }
    let conv = sum.max(*abs_tolerance);

    conv
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
// sometimes order of indexed variables (and therefore order of values in numerical result y_DMatrix ) and order of original variables turns out to be
// not the same: in that case permutation is needed to transform result matrix into the same order of
pub fn interchange_columns(
    y_DMatrix: DMatrix<f64>,
    variables: Vec<String>,
    indexed_variables: Vec<String>,
) -> DMatrix<f64> {
    let mut reordering: Vec<usize> = Vec::new();
    // lets take the same quantity of indexed variables as the original variables
    let indexed_vars_for_reordering: Vec<String> = indexed_variables[0..variables.len()].to_vec();
    // remove numeric suffix from indexed variables and compare with original variables
    let unindexed_vars = indexed_vars_for_reordering
        .iter()
        .map(|x| Jacobian::remove_numeric_suffix(&x.clone()))
        .collect::<Vec<String>>();

    let reordered_result = if variables != unindexed_vars {
        // if they have the same oreder no permutation needed
        println!("permutation needed");
        for var in &unindexed_vars {
            let index = variables.iter().position(|x| x == var).unwrap();
            reordering.push(index);
        }
        let reordered_result = y_DMatrix.select_columns(&reordering);
        reordered_result
    } else {
        y_DMatrix
    };
    reordered_result
}

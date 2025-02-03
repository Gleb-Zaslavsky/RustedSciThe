use crate::numerical::BVP_Damp::BVP_traits::MatrixType;
use crate::symbolic::symbolic_functions::Jacobian;
use log::{info, warn};
use nalgebra::DMatrix;
use std::collections::HashMap;
use std::time::Duration;
use sysinfo::System;
pub fn elapsed_time(elapsed: Duration) {
    let time = elapsed.as_millis();
    if time < 1000 {
        info!("Elapsed {} ms", time)
    } else if time >= 1000 && time < 60_000 {
        info!("Elapsed {} s", elapsed.as_secs())
    } else if time >= 60_000 && time < 3600_000 {
        info!("Elapsed {} min", elapsed.as_secs() / 60)
    } else {
        info!("Elapsed {} h", elapsed.as_secs() / 3600)
    }
}
// FROZEN JACOBIAN TASK PARAMETERS CHECK
pub fn strategy_check(
    strategy: &String,
    strategy_params: &Option<HashMap<String, Option<Vec<f64>>>>,
) {
    if strategy == "Naive" {
        assert_eq!(
            strategy_params.is_none(),
            true,
            "strategy_params are not None"
        );
    } else if strategy == "Frozen" {
        assert_eq!(strategy_params.is_none(), false, "strategy_params are None");
        if let Some(strategy_params_) = strategy_params.clone() {
            let stategy_name =
                strategy_params_.clone().keys().collect::<Vec<&String>>()[0].to_owned();
            match stategy_name.as_str() {
                //calculate jacobian only 1st iteration when old_jac is None, after 1st iter we save jacobian into old_jac and use it all the time
                "Frozen_naive" => {
                    // recalculate jacobian only 1st iteration when old_jac is None, after 1st iter we save jacobian into old_jac
                    let value = strategy_params_
                        .values()
                        .collect::<Vec<&Option<Vec<f64>>>>()[0]
                        .clone();
                    assert_eq!(
                        value.is_none(),
                        true,
                        "value is not None.  Frozen_naive strategy requires no parameters"
                    );
                }
                //recalculate jacobian every m-th iteration
                "every_m" => {
                    // recalculate jacobian every m-th iteration the m value we get from task
                    let m_from_task = strategy_params_
                        .values()
                        .collect::<Vec<&Option<Vec<f64>>>>()[0]
                        .clone()
                        .unwrap();

                    assert_eq!(m_from_task.len(), 1, "m is not of size 1 ");
                    assert!(m_from_task[0] as usize > 0, "m must be > 0");
                    // when jac is None it means this is first iteration
                }
                "at_high_morm" => {
                    let norm_from_task = strategy_params_
                        .values()
                        .collect::<Vec<&Option<Vec<f64>>>>()[0]
                        .clone()
                        .unwrap();
                    assert_eq!(norm_from_task.len(), 1, "norm is not of size 1 ");
                    assert!(norm_from_task[0] > 0.0, "norm must be > 0");
                }
                "at_low_speed" => {
                    let speed_rate = strategy_params_
                        .values()
                        .collect::<Vec<&Option<Vec<f64>>>>()[0]
                        .clone()
                        .unwrap();
                    assert_eq!(speed_rate.len(), 1, "speed_rate is not of size 1 ");
                    assert!(speed_rate[0] <= 1.0, "speed_rate must be <= 1.0");
                }
                "complex" => {
                    let vec_task = strategy_params_
                        .values()
                        .collect::<Vec<&Option<Vec<f64>>>>()[0]
                        .clone()
                        .unwrap();
                    assert_eq!(vec_task.len(), 3, "vec_task is not of size 3 ");
                }
                _ => {
                    println!("Method not implemented: no such stratrgy!");
                    println!(
                        "There are strategies: 
                \n \n - Frozen_naive,
                \n \n - every_m,
                \n \n - at_high_morm,
                \n \n - at_low_speed,
                \n \n - complex"
                    );
                    std::process::exit(1);
                }
            } // end of match
        } // end of if let
    } // end of if Frozen
}
// FUNCTION RETURNS A FLAG FOR  (RE)CALCULATING JACOBIAN ON CERTAIN CONDITION

pub fn frozen_jac_recalc(
    strategy: &String,
    strategy_params: &Option<HashMap<String, Option<Vec<f64>>>>,
    old_jac: &Option<Box<dyn MatrixType>>,
    m: usize,
    error: f64,
    error_old: f64,
) -> bool {
    if let Some(strategy_params_) = strategy_params.clone() {
        let stategy_name = strategy_params_.clone().keys().collect::<Vec<&String>>()[0].to_owned();
        match stategy_name.as_str() {
            //calculate jacobian only 1st iteration when old_jac is None, after 1st iter we save jacobian into old_jac and use it all the time
            "Frozen_naive" => {
                // recalculate jacobian only 1st iteration when old_jac is None, after 1st iter we save jacobian into old_jac
                if old_jac.is_none() {
                    true
                } else {
                    false
                }
            }
            //recalculate jacobian every m-th iteration
            "every_m" => {
                // recalculate jacobian every m-th iteration the m value we get from task
                let m_from_task = strategy_params_
                    .values()
                    .collect::<Vec<&Option<Vec<f64>>>>()[0]
                    .clone()
                    .unwrap()[0] as usize;
                // when jac is None it means this is first iteration
                if old_jac.is_none() || m > m_from_task {
                    info!(
                        "\n number of iterations with old jac {} is higher then threshold {}",
                        m, m_from_task
                    );
                    true
                } else {
                    false
                }
            }
            "at_high_morm" => {
                let norm_from_task = strategy_params_
                    .values()
                    .collect::<Vec<&Option<Vec<f64>>>>()[0]
                    .clone()
                    .unwrap()[0];
                if error > norm_from_task {
                    info!(
                        "\n norm {} is higher then threshold {}",
                        error, norm_from_task
                    );
                    true
                } else {
                    false
                }
            }
            "at_low_speed" => {
                // when norm of (i-1) iter multiplied by certain value B(<1) is lower than norm of i-th iter
                let speed_rate = strategy_params_
                    .values()
                    .collect::<Vec<&Option<Vec<f64>>>>()[0]
                    .clone()
                    .unwrap()[0];
                if speed_rate * error_old < error {
                    info!("error of i-1 iter -({}) must be at least ({}) times less then of i- iter ({})",   error_old, speed_rate, error);

                    true
                } else {
                    false
                }
            }
            "complex" => {
                let vec_task = strategy_params_
                    .values()
                    .collect::<Vec<&Option<Vec<f64>>>>()[0]
                    .clone()
                    .unwrap();
                let m_from_task = vec_task[0] as usize;
                //  println!("m {}, m_from_task {}",m, m_from_task);
                let norm_from_task = vec_task[1];
                let speed_rate = vec_task[2];
                if (error > norm_from_task) || (m > m_from_task) || (speed_rate * error_old < error)
                {
                    if error > norm_from_task {
                        info!(
                            "\n norm {} is higher then threshold {}",
                            error, norm_from_task
                        );
                    }
                    if m >= m_from_task {
                        info!(
                            "\n number of iterations with old jac {} is higher then threshold {}",
                            m, m_from_task
                        );
                    }
                    if speed_rate * error_old < error {
                        info!("error of i-1 iter -({}) must be at least ({}) times less then of i- iter ({})",   error_old, speed_rate, error);
                    }
                    true
                } else {
                    false
                }
            }
            _ => {
                info!("Method not implemented");
                std::process::exit(1);
            }
        }
    } else {
        if strategy.as_str() == "Naive" {
            true
        } else {
            info!("Method not implemented");
            std::process::exit(1);
        }
    }
}

pub fn task_check_mem(n_steps: usize, number_of_y: usize, method: &String) {
    let required_matrix_memory =
        ((n_steps * number_of_y).pow(2) * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0); // Convert bytes to megabyte
    info!("Required matrix memory: {:.2} MB", required_matrix_memory);
    let mut sys = System::new_all();
    sys.refresh_all();

    let free_memory = sys.free_memory() as f64 / (1024.0 * 1024.0);
    if required_matrix_memory > 0.8 * free_memory {
        warn!(
            "Matrix requires  {:.2} MB, which is higher than 70% of free memory.",
            required_matrix_memory
        );
    }
    if method.starts_with("Dense") {
        info!("it is strongly recommended to use sparse matrices!");
    }
}

//calculates and logs the memory usage of a given matrix (mat) and compares it to the total system memory.
// If the matrix memory usage exceeds 80% of the total system memory, it issues a warning.

pub fn checkmem(mat: &dyn MatrixType) -> f64 {
    let (nrows, ncols) = mat.shape();
    // Assuming each element of the matrix takes 8 bytes of memory (size of f64)
    let matrix_memory = (nrows * ncols * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0); // Convert bytes to megabytes
    info!("Matrix memory usage: {:.2} MB", matrix_memory);
    // Create a System object
    // CPUs and processes are filled!
    let mut sys = System::new_all();

    // First we update all information of our `System` struct.
    sys.refresh_all();
    let total_memory = sys.total_memory() as f64 / (1024.0 * 1024.0);
    let used_memory = sys.used_memory() as f64 / (1024.0 * 1024.0);
    let total_swap = sys.total_swap() as f64 / (1024.0 * 1024.0);
    let used_swap = sys.used_swap() as f64 / (1024.0 * 1024.0);
    let free_memory = sys.free_memory() as f64 / (1024.0 * 1024.0);
    if matrix_memory > 0.8 * free_memory {
        warn!(
            "Matrix memory usage is {:.2} MB, which is higher than 70% of total memory.",
            matrix_memory
        );
    }
    // RAM and swap information:
    println!("total memory: {} bytes", total_memory);
    println!("used memory : {} bytes", used_memory);
    println!("total swap  : {} bytes", total_swap);
    println!("used swap   : {} bytes", used_swap);
    matrix_memory
}

pub fn round_to_n_digits(value: f64, n: usize) -> f64 {
    let format_string = format!("{:.1$}", value, n);
    format_string.parse::<f64>().unwrap()
}

pub fn variables_order(variables: Vec<String>, indexed_variables: Vec<String>) -> Vec<String> {
    // lets take the same quantity of indexed variables as the original variables
    let indexed_vars_for_reordering: Vec<String> = indexed_variables[0..variables.len()].to_vec();
    // remove numeric suffix from indexed variables and compare with original variables
    let unindexed_vars = indexed_vars_for_reordering
        .iter()
        .map(|x| Jacobian::remove_numeric_suffix(&x.clone()))
        .collect::<Vec<String>>();
    unindexed_vars
}

// solution = values of unknown variable + boundary condition;
// so we  get the values of the boundary condition and the calculated values of unknown variables and get the full solution
pub fn construct_full_solution(
    solution: DMatrix<f64>,
    BorderConditions: &HashMap<String, (usize, f64)>,
    indexed_variables: &Vec<String>,
    values: &Vec<String>,
) -> DMatrix<f64> {
    info!("Constructing full solution");
    //  println!("{:?}",  &solution.ncols());
    assert_eq!(solution.shape().0, values.len());
    let (n_rows, n_cols) = solution.shape();
    let real_order_of_variables =
        variables_order(values.clone().to_owned(), indexed_variables.to_owned());
    let mut constructed_solution = DMatrix::zeros(n_rows, n_cols + 1);
    for (i, sol_i) in solution.row_iter().enumerate() {
        let mut sol_i: Vec<f64> = sol_i.iter().map(|x| *x).collect();
        //  println!("i {}", i);
        // println!("{:?}", values[i]);
        let unknown_variable = real_order_of_variables[i].clone();
        if let Some(&(bc_type, bc_value)) = BorderConditions.get(&unknown_variable) {
            match bc_type {
                0 => {
                    sol_i.insert(0, bc_value);
                    assert_eq!(sol_i.len(), n_cols + 1);
                    constructed_solution.row_mut(i).copy_from_slice(&sol_i);
                    // println!("{:?}", constructed_solution);
                }
                1 => {
                    sol_i.push(bc_value);
                    assert_eq!(sol_i.len(), n_cols + 1);
                    constructed_solution.row_mut(i).copy_from_slice(&sol_i);
                    //    println!("{:?}", constructed_solution);
                }
                _ => {
                    panic!("Unsupported boundary condition type: {}", bc_type);
                }
            }
        } else {
        }
    }
    // println!("constructed_solution {}", constructed_solution);
    constructed_solution
}

pub fn extract_unknown_variables(
    full_solution: DMatrix<f64>,
    border_conditions: &HashMap<String, (usize, f64)>,
    indexed_variables: &Vec<String>,
    values: &Vec<String>,
) -> DMatrix<f64> {
    let (n_rows, n_cols) = full_solution.shape();
    let mut extracted_solution = DMatrix::zeros(n_rows, n_cols - 1);
    let real_order_of_variables =
        variables_order(values.clone().to_owned(), indexed_variables.to_owned());
    for (i, full_sol_i) in full_solution.row_iter().enumerate() {
        //   print!("{:}", full_sol_i);
        let mut full_sol_i: Vec<f64> = full_sol_i.iter().map(|x| *x).collect();
        let unknown_variable = real_order_of_variables[i].clone();

        if let Some(&(bc_type, _)) = border_conditions.get(&unknown_variable) {
            match bc_type {
                0 => {
                    // Remove the first element (boundary condition at the start)
                    full_sol_i.remove(0);
                    assert_eq!(full_sol_i.len(), n_cols - 1);
                    extracted_solution.row_mut(i).copy_from_slice(&full_sol_i);
                    //  println!("{:?}", extracted_solution);
                }
                1 => {
                    // Remove the last element (boundary condition at the end)
                    full_sol_i.pop();
                    assert_eq!(full_sol_i.len(), n_cols - 1);
                    extracted_solution.row_mut(i).copy_from_slice(&full_sol_i);
                    //  println!("{:?}", extracted_solution);
                }
                _ => {
                    panic!("Unsupported boundary condition type: {}", bc_type);
                }
            }
        } else {
            panic!(
                "No boundary condition found for variable: {}",
                unknown_variable
            );
        }
    }
    //println!("extracted_solution {:?}", extracted_solution);
    extracted_solution
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;
    use std::collections::HashMap;

    #[test]
    fn test_construct_full_solution() {
        let solution = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]).transpose();
        let (n_rows, n_cols) = solution.shape();
        let mut border_conditions = HashMap::new();
        border_conditions.insert("x".to_string(), (0, 0.0));
        border_conditions.insert("y".to_string(), (1, 5.0));
        let values = vec!["y".to_string(), "x".to_string()];
        let indexed_variables = vec!["x_1".to_string(), "y_1".to_string()];
        let constructed_solution =
            construct_full_solution(solution, &border_conditions, &indexed_variables, &values);

        let expected_solution =
            DMatrix::from_row_slice(n_rows, n_cols + 1, &[0.0, 1.0, 3.0, 2.0, 4.0, 5.0]);
        println!(
            "{:}, {:}",
            constructed_solution.column(0),
            expected_solution.column(0)
        );
        println!(
            "constructed_solution {:?}, {:?}",
            constructed_solution,
            constructed_solution.shape()
        );
        // println!("expected {:?}", expected_solution);
        assert_eq!(constructed_solution, expected_solution);
    }

    #[test]
    fn test_extract_unknown_variables() {
        let full_solution =
            DMatrix::from_row_slice(3, 2, &[0.0, 1.0, 3.0, 2.0, 4.0, 5.0]).transpose();
        //  println!("full_solution {:?}, {:?}", full_solution, full_solution.shape());
        let mut border_conditions = HashMap::new();
        border_conditions.insert("x".to_string(), (0, 0.0));
        border_conditions.insert("y".to_string(), (1, 5.0));
        let values = vec!["y".to_string(), "x".to_string()];
        let indexed_variables = vec!["x_1".to_string(), "y_1".to_string()];
        let expected_solution = DMatrix::from_row_slice(2, 2, &[3.0, 4.0, 1.0, 2.0]);

        // println!("expected_solution {:?}, {:?}", expected_solution, expected_solution.shape());
        let extracted_solution = extract_unknown_variables(
            full_solution,
            &border_conditions,
            &indexed_variables,
            &values,
        );
        assert_eq!(extracted_solution, expected_solution);
    }
    #[test]
    fn test_combine() {
        let solution = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        // let (n_rows, n_cols) = solution.shape();
        let mut border_conditions = HashMap::new();
        border_conditions.insert("x".to_string(), (0, 0.0));
        border_conditions.insert("y".to_string(), (1, 5.0));
        let values = vec!["y".to_string(), "x".to_string()];
        let indexed_variables = vec!["x_1".to_string(), "y_1".to_string()];
        let constructed_solution = construct_full_solution(
            solution.clone(),
            &border_conditions,
            &indexed_variables,
            &values,
        );
        let extracted_solution = extract_unknown_variables(
            constructed_solution,
            &border_conditions,
            &indexed_variables,
            &values,
        );
        assert_eq!(solution, extracted_solution);
    }
}

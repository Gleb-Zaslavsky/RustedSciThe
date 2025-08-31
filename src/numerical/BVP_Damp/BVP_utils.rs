use crate::numerical::BVP_Damp::BVP_traits::MatrixType;
use crate::symbolic::symbolic_functions::Jacobian;
use log::{info, warn};
use nalgebra::DMatrix;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use sysinfo::System;
use tabled::{builder::Builder, settings::Style};
use regex::Regex;
pub fn elapsed_time(elapsed: Duration) -> (String, f64) {
    let time = elapsed.as_millis();
    if time < 1000 {
        info!("Elapsed {} ms", time);
        (" ms ".to_string(), time as f64)
    } else if time >= 1000 && time < 60_000 {
        info!("Elapsed {} s", elapsed.as_secs());
        (" s".to_string(), elapsed.as_secs() as f64)
    } else if time >= 60_000 && time < 3600_000 {
        info!("Elapsed {} min", elapsed.as_secs() / 60);
        (" min".to_string(), elapsed.as_secs() as f64 / 60.0)
    } else {
        info!("Elapsed {} h", elapsed.as_secs() / 3600);
        (" h".to_string(), elapsed.as_secs() as f64 / 3600.0)
    }
}

#[derive(Debug, Clone)]
pub struct CustomTimer {
    pub start: Instant,
    pub jac_time: Instant,
    pub jac: Duration,
    pub fun_time: Instant,
    pub fun: Duration,
    pub linear_system_time: Instant,
    pub linear_system: Duration,
    pub symbolic_operations_time: Instant,
    pub symbolic_operations: Duration,
    pub grid_refinement_time: Instant,
    pub grid_refinement: Duration,
}

impl CustomTimer {
    pub fn new() -> CustomTimer {
        CustomTimer {
            start: Instant::now(),
            jac_time: Instant::now(),
            jac: Duration::from_secs(0),
            fun_time: Instant::now(),
            fun: Duration::from_secs(0),
            linear_system_time: Instant::now(),
            linear_system: Duration::from_secs(0),
            symbolic_operations_time: Instant::now(),
            symbolic_operations: Duration::from_secs(0),
            grid_refinement_time: Instant::now(),
            grid_refinement: Duration::from_secs(0),
        }
    }
    pub fn start(&mut self) {
        self.start = Instant::now();
        self.jac_time = Instant::now();
        self.jac = Duration::from_secs(0);
        self.fun_time = Instant::now();
        self.fun = Duration::from_secs(0);
        self.linear_system_time = Instant::now();
        self.linear_system = Duration::from_secs(0);
        self.symbolic_operations_time = Instant::now();
        self.symbolic_operations = Duration::from_secs(0);
        self.grid_refinement_time = Instant::now();
        self.grid_refinement = Duration::from_secs(0);
    }
    pub fn jac_tic(&mut self) {
        self.jac_time = Instant::now();
    }

    pub fn jac_tac(&mut self) {
        let jac = self.jac_time.elapsed();
        self.jac += jac;
    }

    pub fn fun_tic(&mut self) {
        self.fun_time = Instant::now();
    }
    pub fn fun_tac(&mut self) {
        let fun = self.fun_time.elapsed();
        self.fun += fun;
    }
    pub fn append_to_fun_time(&mut self, fun: Duration) {
        self.fun += fun;
    }
    pub fn linear_system_tic(&mut self) {
        self.linear_system_time = Instant::now();
    }
    pub fn linear_system_tac(&mut self) {
        let linear_system = self.linear_system_time.elapsed();
        self.linear_system += linear_system;
    }
    pub fn append_to_linear_sys_time(&mut self, linear_system: Duration) {
        self.linear_system += linear_system;
    }
    pub fn symbolic_operations_tic(&mut self) {
        self.symbolic_operations_time = Instant::now();
    }
    pub fn symbolic_operations_tac(&mut self) {
        let symbolic_operations = self.symbolic_operations_time.elapsed();
        self.symbolic_operations += symbolic_operations;
    }
    pub fn grid_refinement_tic(&mut self) {
        self.grid_refinement_time = Instant::now();
    }
    pub fn grid_refinement_tac(&mut self) {
        let grid_refinement = self.grid_refinement_time.elapsed();
        self.grid_refinement += grid_refinement;
    }
    pub fn get_all(&self) -> HashMap<String, String> {
        let mut timer_data: HashMap<String, String> = HashMap::new();

        let total_time = self.start.elapsed().as_nanos() as f64;
        let total_time_string = elapsed_time(self.start.elapsed());

        let jac_total_string = elapsed_time(self.jac);
        let jac_total = self.jac.as_nanos() as f64;
        let jac_time_percent = 100.0 * jac_total / total_time;

        let fun_total = self.fun.as_nanos() as f64;
        let fun_time_percent = 100.0 * fun_total / total_time;
        let fun_total_string = elapsed_time(self.fun);

        let linear_system_total = self.linear_system.as_nanos() as f64;
        let linear_system_time_percent = 100.0 * linear_system_total / total_time;
        let linear_system_total_string = elapsed_time(self.linear_system);

        let symbolic_operations_total = self.symbolic_operations.as_nanos() as f64;
        let symbolic_operations_time_percent = 100.0 * symbolic_operations_total / total_time;
        let symbolic_operations_total_string = elapsed_time(self.symbolic_operations);

        let grid_refinement_total = self.grid_refinement.as_nanos() as f64;
        let grid_refinement_time_percent = 100.0 * grid_refinement_total / total_time;
        let grid_refinement_total_string = elapsed_time(self.grid_refinement);

        let other = total_time
            - jac_total
            - fun_total
            - linear_system_total
            - symbolic_operations_total
            - grid_refinement_total;

        let other_percent = 100.0 * other / total_time;

        if other_percent > 0.5 {
            timer_data.insert(
                "other %".to_string(),
                format!("{} ", (other_percent * 1000.0).round() / 1000.0),
            );
        }
        timer_data.insert(
            "time elapsed, ".to_string() + total_time_string.0.as_str(),
            format!("{}", total_time_string.1),
        );

        if grid_refinement_time_percent > 0.5 {
            timer_data.insert(
                "Grid Refinement (%, ".to_string() + grid_refinement_total_string.0.as_str() + ")",
                format!(
                    "{}, {}",
                    (grid_refinement_time_percent * 1000.0).round() / 1000.0,
                    grid_refinement_total_string.1
                ),
            );
        }
        if jac_time_percent > 0.5 {
            timer_data.insert(
                "Jacobian (%, ".to_string() + jac_total_string.0.as_str() + ")",
                format!(
                    "{}, {}",
                    (jac_time_percent * 1000.0).round() / 1000.0,
                    jac_total_string.1
                ),
            );
        }
        if fun_time_percent > 0.5 {
            timer_data.insert(
                "Function (%, ".to_string() + fun_total_string.0.as_str() + ")",
                format!(
                    "{}, {}",
                    (fun_time_percent * 1000.0).round() / 1000.0,
                    fun_total_string.1
                ),
            );
        }
        if linear_system_time_percent > 0.5 {
            timer_data.insert(
                "Linear System (%, ".to_string() + linear_system_total_string.0.as_str() + ")",
                format!(
                    "{}, {}",
                    (linear_system_time_percent * 1000.0).round() / 1000.0,
                    linear_system_total_string.1
                ),
            );
        }
        if symbolic_operations_time_percent > 0.5 {
            timer_data.insert(
                "Symbolic Operations (%, ".to_string()
                    + symbolic_operations_total_string.0.as_str()
                    + ")",
                format!(
                    "{}, {}",
                    (symbolic_operations_time_percent * 1000.0).round() / 1000.0,
                    symbolic_operations_total_string.1
                ),
            );
        }
        let mut table = Builder::from(timer_data.clone()).build();
        table.with(Style::modern_rounded());
        info!("\n \n TIMER DATA \n \n {}", table.to_string());
        timer_data
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
                if old_jac.is_none() { true } else { false }
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
                    info!(
                        "error of i-1 iter -({}) must be at least ({}) times less then of i- iter ({})",
                        error_old, speed_rate, error
                    );

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
                        info!(
                            "error of i-1 iter -({}) must be at least ({}) times less then of i- iter ({})",
                            error_old, speed_rate, error
                        );
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
    pub fn remove_numeric_suffix(input: &str) -> String {
        let re = Regex::new(r"_\d+$").unwrap();
        re.replace(input, "").to_string()
    }
pub fn variables_order(variables: Vec<String>, indexed_variables: Vec<String>) -> Vec<String> {
    // lets take the same quantity of indexed variables as the original variables
    let indexed_vars_for_reordering: Vec<String> = indexed_variables[0..variables.len()].to_vec();
    // remove numeric suffix from indexed variables and compare with original variables
    let unindexed_vars = indexed_vars_for_reordering
        .iter()
        .map(|x| remove_numeric_suffix(&x.clone()))
        .collect::<Vec<String>>();
    unindexed_vars
}

// solution = values of unknown variable + boundary condition;
// so we  get the values of the boundary condition and the calculated values of unknown variables and get the full solution
pub fn construct_full_solution(
    solution: DMatrix<f64>,
    BorderConditions: &HashMap<String, Vec<(usize, f64)>>,
    values: &Vec<String>,
) -> DMatrix<f64> {
    info!("Constructing full solution");
    // dbg!(&solution);
    //panic!();
    assert_eq!(solution.shape().0, values.len());
    let (n_rows, n_cols) = solution.shape();

    // Calculate total number of boundary conditions to determine output matrix size
    let mut total_bc_count: usize = 0;
    for values in BorderConditions.values() {
        if values.len() > total_bc_count {
            total_bc_count = values.len();
        }
    }

    let output_cols = n_cols + total_bc_count;
    let mut constructed_solution = DMatrix::zeros(n_rows, output_cols);

    // Add boundary conditions to create full solution
    for (var_idx, var_name) in values.iter().enumerate() {
        if let Some(boundary_conditions) = BorderConditions.get(var_name) {
            let mut row_data: Vec<f64> = solution.row(var_idx).iter().cloned().collect();

            // Process boundary conditions
            for &(bc_type, bc_value) in boundary_conditions {
                match bc_type {
                    0 => row_data.insert(0, bc_value), // Left boundary condition
                    1 => row_data.push(bc_value),      // Right boundary condition
                    _ => panic!("Unsupported boundary condition type: {}", bc_type),
                }
            }

            constructed_solution
                .row_mut(var_idx)
                .copy_from_slice(&row_data);
        }
    }

    constructed_solution
}

pub fn extract_unknown_variables(
    full_solution: DMatrix<f64>,
    border_conditions: &HashMap<String, Vec<(usize, f64)>>,
    values: &Vec<String>,
) -> DMatrix<f64> {
    let (n_rows, n_cols) = full_solution.shape();

    // Calculate total number of boundary conditions to determine output matrix size
    let mut total_bc_count: usize = 0;
    for values in border_conditions.values() {
        if values.len() > total_bc_count {
            total_bc_count = values.len();
        }
    }

    let output_cols = n_cols - total_bc_count;
    let mut extracted_solution = DMatrix::zeros(n_rows, output_cols);

    for (var_idx, var_name) in values.iter().enumerate() {
        if let Some(boundary_conditions) = border_conditions.get(var_name) {
            let mut row_data: Vec<f64> = full_solution.row(var_idx).iter().cloned().collect();

            // Process boundary conditions - remove them
            for &(bc_type, _) in boundary_conditions {
                match bc_type {
                    0 => {
                        row_data.remove(0);
                    } // Remove left boundary condition
                    1 => {
                        row_data.pop();
                    } // Remove right boundary condition
                    _ => panic!("Unsupported boundary condition type: {}", bc_type),
                }
            }

            extracted_solution
                .row_mut(var_idx)
                .copy_from_slice(&row_data);
        }
    }

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
        border_conditions.insert("x".to_string(), vec![(0, 0.0)]);
        border_conditions.insert("y".to_string(), vec![(1, 5.0)]);
        let values = vec!["x".to_string(), "y".to_string()];

        let constructed_solution = construct_full_solution(solution, &border_conditions, &values);

        let expected_solution =
            DMatrix::from_row_slice(n_rows, n_cols + 1, &[0.0, 1.0, 3.0, 2.0, 4.0, 5.0]);
        println!("{:}, {:}", constructed_solution, expected_solution);

        // println!("expected {:?}", expected_solution);
        assert_eq!(constructed_solution, expected_solution);
    }

    #[test]
    fn test_extract_unknown_variables() {
        let full_solution =
            DMatrix::from_row_slice(3, 2, &[0.0, 1.0, 3.0, 2.0, 4.0, 5.0]).transpose();
        //  println!("full_solution {:?}, {:?}", full_solution, full_solution.shape());
        let mut border_conditions = HashMap::new();
        border_conditions.insert("x".to_string(), vec![(0, 0.0)]);
        border_conditions.insert("y".to_string(), vec![(1, 5.0)]);
        let values = vec!["x".to_string(), "y".to_string()];

        let expected_solution = DMatrix::from_row_slice(2, 2, &[3.0, 4.0, 1.0, 2.0]);
        println!(
            "full_solution{} \n, expected  {}",
            full_solution, expected_solution
        );
        // println!("expected_solution {:?}, {:?}", expected_solution, expected_solution.shape());
        let extracted_solution =
            extract_unknown_variables(full_solution, &border_conditions, &values);
        println!(
            "extracted_solution {}, {:?}",
            extracted_solution,
            extracted_solution.shape()
        );
        assert_eq!(extracted_solution, expected_solution);
    }
    #[test]
    fn test_combine() {
        let solution = DMatrix::from_row_slice(2, 2, &[3.0, 4.0, 1.0, 2.0]);
        // let (n_rows, n_cols) = solution.shape();
        let mut border_conditions = HashMap::new();
        border_conditions.insert("x".to_string(), vec![(0, 100.0)]);
        border_conditions.insert("y".to_string(), vec![(1, 500.0)]);
        let values = vec!["x".to_string(), "y".to_string()];

        let constructed_solution =
            construct_full_solution(solution.clone(), &border_conditions, &values);
        println!(
            " {} \n , constructed_solution {} \n",
            solution, constructed_solution
        );
        let extracted_solution =
            extract_unknown_variables(constructed_solution, &border_conditions, &values);
        assert_eq!(solution, extracted_solution);
    }

    #[test]
    fn test_multiple_boundary_conditions() {
        // Test case with multiple boundary conditions per variable
        let solution = DMatrix::from_row_slice(1, 2, &[1.0, 2.0]);
        let mut border_conditions = HashMap::new();
        border_conditions.insert(
            "y".to_string(),
            vec![(0, 0.36787944117144233), (1, 0.36787944117144233)],
        );
        let values = vec!["y".to_string()];

        let constructed_solution =
            construct_full_solution(solution.clone(), &border_conditions, &values);

        // Should have original 2 columns + 2 boundary conditions = 4 columns
        assert_eq!(constructed_solution.ncols(), 4);

        let extracted_solution =
            extract_unknown_variables(constructed_solution, &border_conditions, &values);

        // Should recover original solution
        assert_eq!(solution, extracted_solution);
    }
}

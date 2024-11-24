use crate::numerical::BVP_Damp::BVP_traits::MatrixType;
use std::collections::HashMap;
use std::time::Duration;
use simplelog::*;
pub fn elapsed_time(elapsed: Duration) {
    let time = elapsed.as_millis();
    if time < 1000 {
        log::info!("\n \n Elapsed {} ms", time)
    } else if time >= 1000 && time < 60_000 {
        log::info!("\n \n Elapsed {} s", elapsed.as_secs())
    } else if time >= 60_000 && time < 3600_000 {
        log::info!("\n \n Elapsed {} min", elapsed.as_secs() / 60)
    } else {
        log::info!("\n \n Elapsed {} h", elapsed.as_secs() / 3600)
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
                    println!(
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
                    println!(
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
                    println!("error of i-1 iter -({}) must be at least ({}) times less then of i- iter ({})",   error_old, speed_rate, error);

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
                        println!(
                            "\n norm {} is higher then threshold {}",
                            error, norm_from_task
                        );
                    }
                    if m >= m_from_task {
                        println!(
                            "\n number of iterations with old jac {} is higher then threshold {}",
                            m, m_from_task
                        );
                    }
                    if speed_rate * error_old < error {
                        println!("error of i-1 iter -({}) must be at least ({}) times less then of i- iter ({})",   error_old, speed_rate, error);
                    }
                    true
                } else {
                    false
                }
            }
            _ => {
                println!("Method not implemented");
                std::process::exit(1);
            }
        }
    } else {
        if strategy.as_str() == "Naive" {
            true
        } else {
            println!("Method not implemented");
            std::process::exit(1);
        }
    }
}

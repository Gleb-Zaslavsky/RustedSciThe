use crate::numerical::BVP_Damp::BVP_traits::MatrixType;
use crate::numerical::BVP_Damp::telemetry::{
    BvpCallbackStage, BvpCallbackStageTiming, BvpTelemetryMode, BvpTimingSnapshot,
};

use log::{info, warn};
use nalgebra::{DMatrix, DVector};
use regex::Regex;

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::time::{Duration, Instant};
use sysinfo::System;
use tabled::{builder::Builder, settings::Style};

fn percent_of_total(part: f64, total: f64) -> f64 {
    if total <= f64::EPSILON {
        0.0
    } else {
        100.0 * part / total
    }
}

#[derive(Clone, Copy, Debug)]
struct CallbackStageAccumulator {
    durations: [Duration; BvpCallbackStage::ALL.len()],
}

impl Default for CallbackStageAccumulator {
    fn default() -> Self {
        Self {
            durations: [Duration::ZERO; BvpCallbackStage::ALL.len()],
        }
    }
}

impl CallbackStageAccumulator {
    #[inline]
    fn add(&mut self, stage: BvpCallbackStage, duration: Duration) {
        self.durations[stage.index()] += duration;
    }

    fn typed_snapshot(self) -> Vec<BvpCallbackStageTiming> {
        let mut snapshot = Vec::with_capacity(BvpCallbackStage::ALL.len());
        for stage in BvpCallbackStage::ALL {
            let duration = self.durations[stage.index()];
            if duration > Duration::ZERO {
                snapshot.push(BvpCallbackStageTiming {
                    stage,
                    elapsed: duration,
                });
            }
        }
        snapshot
    }
}

thread_local! {
    // Fixed slots keep callback instrumentation allocation-free. Presentation
    // maps are materialized only when a caller requests a snapshot/report.
    static CALLBACK_STAGE_TIMERS: RefCell<CallbackStageAccumulator> =
        RefCell::new(CallbackStageAccumulator::default());
    // Compatibility bridge for callbacks that do not yet receive a timer
    // session explicitly. A running solver installs its own solve-local
    // accumulator here; old standalone callback users keep the fallback.
    static ACTIVE_CALLBACK_STAGE_SESSION: RefCell<Option<Rc<RefCell<CallbackStageAccumulator>>>> =
        RefCell::new(None);
    // A disabled solve masks the compatibility fallback as well. The prior
    // value is restored when the solve finishes, so standalone callbacks keep
    // their historical behaviour outside a solver session.
    static CALLBACK_STAGE_COLLECTION_ENABLED: Cell<bool> = const { Cell::new(true) };
}

pub fn reset_callback_stage_timers() {
    CALLBACK_STAGE_TIMERS.with(|timers| {
        *timers.borrow_mut() = CallbackStageAccumulator::default();
    });
    ACTIVE_CALLBACK_STAGE_SESSION.with(|session| {
        if let Some(session) = session.borrow().as_ref() {
            *session.borrow_mut() = CallbackStageAccumulator::default();
        }
    });
}

pub fn record_callback_stage_time(label: &'static str, duration: Duration) {
    if !CALLBACK_STAGE_COLLECTION_ENABLED.with(Cell::get) {
        return;
    }
    let stage = BvpCallbackStage::from_label(label);
    let recorded_in_session = ACTIVE_CALLBACK_STAGE_SESSION.with(|session| {
        if let Some(active) = session.borrow().as_ref() {
            active.borrow_mut().add(stage, duration);
            true
        } else {
            false
        }
    });
    if recorded_in_session {
        return;
    }
    CALLBACK_STAGE_TIMERS.with(|timers| {
        timers.borrow_mut().add(stage, duration);
    });
}

/// Returns fixed-vocabulary callback timings for typed consumers.
pub fn callback_stage_timings_snapshot() -> Vec<BvpCallbackStageTiming> {
    ACTIVE_CALLBACK_STAGE_SESSION.with(|session| {
        if let Some(active) = session.borrow().as_ref() {
            active.borrow().typed_snapshot()
        } else {
            CALLBACK_STAGE_TIMERS.with(|timers| timers.borrow().typed_snapshot())
        }
    })
}

/// Compatibility projection for existing table/report consumers.
pub fn callback_stage_timer_snapshot() -> HashMap<String, Duration> {
    callback_stage_timings_snapshot()
        .into_iter()
        .map(|timing| (timing.stage.label().to_string(), timing.elapsed))
        .collect()
}

fn insert_duration_timer(
    timer_data: &mut HashMap<String, String>,
    label: &str,
    duration: Duration,
    total_time_ns: f64,
) {
    let duration_ns = duration.as_nanos() as f64;
    let percent = percent_of_total(duration_ns, total_time_ns);
    let duration_ms = duration.as_secs_f64() * 1000.0;
    timer_data.insert(
        format!("{label} (%, ms)"),
        format!(
            "{:.3}, {:.6}",
            (percent * 1000.0).round() / 1000.0,
            duration_ms
        ),
    );
}

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
    total: Option<Duration>,
    pub jac_time: Instant,
    pub jac: Duration,
    pub fun_time: Instant,
    pub fun: Duration,
    pub linear_system_time: Instant,
    pub linear_system: Duration,
    /// Time spent constructing numeric linear factors.
    pub factorization: Duration,
    /// Time spent solving RHS vectors with already prepared factors.
    pub rhs_solve: Duration,
    pub symbolic_operations_time: Instant,
    pub symbolic_operations: Duration,
    pub grid_refinement_time: Instant,
    pub grid_refinement: Duration,
    callback_stage_session: Option<Rc<RefCell<CallbackStageAccumulator>>>,
    callback_stage_previous_session: Option<Option<Rc<RefCell<CallbackStageAccumulator>>>>,
    callback_stage_previous_enabled: Option<bool>,
    enabled: bool,
    callback_stages_enabled: bool,
}

impl CustomTimer {
    pub fn new() -> CustomTimer {
        CustomTimer {
            start: Instant::now(),
            total: None,
            jac_time: Instant::now(),
            jac: Duration::from_secs(0),
            fun_time: Instant::now(),
            fun: Duration::from_secs(0),
            linear_system_time: Instant::now(),
            linear_system: Duration::from_secs(0),
            factorization: Duration::from_secs(0),
            rhs_solve: Duration::from_secs(0),
            symbolic_operations_time: Instant::now(),
            symbolic_operations: Duration::from_secs(0),
            grid_refinement_time: Instant::now(),
            grid_refinement: Duration::from_secs(0),
            callback_stage_session: None,
            callback_stage_previous_session: None,
            callback_stage_previous_enabled: None,
            enabled: true,
            callback_stages_enabled: true,
        }
    }

    /// Enables or disables all timer collection for this solve session.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        self.callback_stages_enabled = enabled;
    }

    /// Applies the public telemetry policy to the timer without changing the
    /// numerical solver. `Counters` keeps major timings but skips callback
    /// sub-stage timing; `Detailed` enables both.
    pub fn set_telemetry_mode(&mut self, mode: BvpTelemetryMode) {
        self.enabled = mode != BvpTelemetryMode::Off;
        self.callback_stages_enabled = mode == BvpTelemetryMode::Detailed;
    }

    pub fn start(&mut self) {
        if let Some(previous) = self.callback_stage_previous_session.take() {
            ACTIVE_CALLBACK_STAGE_SESSION.with(|session| {
                session.replace(previous);
            });
        }
        CALLBACK_STAGE_TIMERS.with(|timers| {
            *timers.borrow_mut() = CallbackStageAccumulator::default();
        });
        let previous_enabled = CALLBACK_STAGE_COLLECTION_ENABLED.with(|enabled| {
            let previous = enabled.get();
            enabled.set(self.callback_stages_enabled);
            previous
        });
        self.callback_stage_previous_enabled = Some(previous_enabled);
        if self.enabled && self.callback_stages_enabled {
            let callback_stage_session = self
                .callback_stage_session
                .get_or_insert_with(|| Rc::new(RefCell::new(CallbackStageAccumulator::default())));
            *callback_stage_session.borrow_mut() = CallbackStageAccumulator::default();
            let previous = ACTIVE_CALLBACK_STAGE_SESSION
                .with(|session| session.replace(Some(callback_stage_session.clone())));
            self.callback_stage_previous_session = Some(previous);
        } else {
            self.callback_stage_previous_session = None;
        }
        if !self.enabled {
            self.total = Some(Duration::ZERO);
            self.jac = Duration::ZERO;
            self.fun = Duration::ZERO;
            self.linear_system = Duration::ZERO;
            self.factorization = Duration::ZERO;
            self.rhs_solve = Duration::ZERO;
            self.symbolic_operations = Duration::ZERO;
            self.grid_refinement = Duration::ZERO;
            return;
        }
        self.start = Instant::now();
        self.total = None;
        self.jac_time = Instant::now();
        self.jac = Duration::from_secs(0);
        self.fun_time = Instant::now();
        self.fun = Duration::from_secs(0);
        self.linear_system_time = Instant::now();
        self.linear_system = Duration::from_secs(0);
        self.factorization = Duration::from_secs(0);
        self.rhs_solve = Duration::from_secs(0);
        self.symbolic_operations_time = Instant::now();
        self.symbolic_operations = Duration::from_secs(0);
        self.grid_refinement_time = Instant::now();
        self.grid_refinement = Duration::from_secs(0);
    }

    /// Freezes the solve wall-clock measurement for stable later reads.
    pub fn finish(&mut self) {
        if self.enabled {
            self.total = Some(self.start.elapsed());
        }
        if let Some(previous) = self.callback_stage_previous_session.take() {
            ACTIVE_CALLBACK_STAGE_SESSION.with(|session| {
                session.replace(previous);
            });
        }
        if let Some(previous) = self.callback_stage_previous_enabled.take() {
            CALLBACK_STAGE_COLLECTION_ENABLED.with(|enabled| enabled.set(previous));
        }
    }
    pub fn jac_tic(&mut self) {
        if !self.enabled {
            return;
        }
        self.jac_time = Instant::now();
    }

    pub fn jac_tac(&mut self) {
        if !self.enabled {
            return;
        }
        let jac = self.jac_time.elapsed();
        self.jac += jac;
    }

    pub fn fun_tic(&mut self) {
        if !self.enabled {
            return;
        }
        self.fun_time = Instant::now();
    }
    pub fn fun_tac(&mut self) {
        if !self.enabled {
            return;
        }
        let fun = self.fun_time.elapsed();
        self.fun += fun;
    }
    pub fn append_to_fun_time(&mut self, fun: Duration) {
        if !self.enabled {
            return;
        }
        self.fun += fun;
    }
    pub fn linear_system_tic(&mut self) {
        if !self.enabled {
            return;
        }
        self.linear_system_time = Instant::now();
    }
    pub fn linear_system_tac(&mut self) {
        if !self.enabled {
            return;
        }
        let linear_system = self.linear_system_time.elapsed();
        self.linear_system += linear_system;
    }
    pub fn append_to_linear_sys_time(&mut self, linear_system: Duration) {
        if !self.enabled {
            return;
        }
        self.linear_system += linear_system;
    }
    pub fn append_to_factorization_time(&mut self, factorization: Duration) {
        if !self.enabled {
            return;
        }
        self.factorization += factorization;
    }
    pub fn append_to_rhs_solve_time(&mut self, rhs_solve: Duration) {
        if !self.enabled {
            return;
        }
        self.rhs_solve += rhs_solve;
    }
    pub fn symbolic_operations_tic(&mut self) {
        if !self.enabled {
            return;
        }
        self.symbolic_operations_time = Instant::now();
    }
    pub fn symbolic_operations_tac(&mut self) {
        if !self.enabled {
            return;
        }
        let symbolic_operations = self.symbolic_operations_time.elapsed();
        self.symbolic_operations += symbolic_operations;
    }
    pub fn grid_refinement_tic(&mut self) {
        if !self.enabled {
            return;
        }
        self.grid_refinement_time = Instant::now();
    }
    pub fn grid_refinement_tac(&mut self) {
        if !self.enabled {
            return;
        }
        let grid_refinement = self.grid_refinement_time.elapsed();
        self.grid_refinement += grid_refinement;
    }

    /// Returns a typed snapshot for programmatic diagnostics.
    ///
    /// `get_all` remains the presentation/compatibility adapter. Keeping the
    /// typed snapshot separate prevents solver code from depending on labels
    /// and formatted duration strings.
    pub fn snapshot(&self) -> BvpTimingSnapshot {
        if !self.enabled {
            return BvpTimingSnapshot::default();
        }
        let callback_stage_timings = self
            .callback_stage_session
            .as_ref()
            .map(|session| session.borrow().typed_snapshot())
            .unwrap_or_default();
        let mut callback_stages: Vec<(String, Duration)> = callback_stage_timings
            .iter()
            .map(|timing| (timing.stage.label().to_string(), timing.elapsed))
            .collect();
        callback_stages.sort_by(|left, right| left.0.cmp(&right.0));
        BvpTimingSnapshot {
            total: self.total.unwrap_or_else(|| self.start.elapsed()),
            residual: self.fun,
            jacobian: self.jac,
            linear_system: self.linear_system,
            factorization: self.factorization,
            rhs_solve: self.rhs_solve,
            symbolic_operations: self.symbolic_operations,
            grid_refinement: self.grid_refinement,
            callback_stage_timings,
            callback_stages,
        }
    }

    pub fn get_all(&self) -> HashMap<String, String> {
        let mut timer_data: HashMap<String, String> = HashMap::new();

        let total_duration = self.total.unwrap_or_else(|| self.start.elapsed());
        let total_time = total_duration.as_nanos() as f64;
        let total_time_string = elapsed_time(total_duration);

        let jac_total_string = elapsed_time(self.jac);
        let jac_total = self.jac.as_nanos() as f64;
        let jac_time_percent = percent_of_total(jac_total, total_time);

        let fun_total = self.fun.as_nanos() as f64;
        let fun_time_percent = percent_of_total(fun_total, total_time);
        let fun_total_string = elapsed_time(self.fun);

        let linear_system_total = self.linear_system.as_nanos() as f64;
        let linear_system_time_percent = percent_of_total(linear_system_total, total_time);
        let linear_system_total_string = elapsed_time(self.linear_system);

        let factorization_total = self.factorization.as_nanos() as f64;
        let factorization_time_percent = percent_of_total(factorization_total, total_time);
        let factorization_total_string = elapsed_time(self.factorization);

        let rhs_solve_total = self.rhs_solve.as_nanos() as f64;
        let rhs_solve_time_percent = percent_of_total(rhs_solve_total, total_time);
        let rhs_solve_total_string = elapsed_time(self.rhs_solve);

        let symbolic_operations_total = self.symbolic_operations.as_nanos() as f64;
        let symbolic_operations_time_percent =
            percent_of_total(symbolic_operations_total, total_time);
        let symbolic_operations_total_string = elapsed_time(self.symbolic_operations);

        let grid_refinement_total = self.grid_refinement.as_nanos() as f64;
        let grid_refinement_time_percent = percent_of_total(grid_refinement_total, total_time);
        let grid_refinement_total_string = elapsed_time(self.grid_refinement);

        let other = total_time
            - jac_total
            - fun_total
            - linear_system_total
            - factorization_total
            - rhs_solve_total
            - symbolic_operations_total
            - grid_refinement_total;

        let other_percent = percent_of_total(other, total_time);

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

        timer_data.insert(
            "Grid Refinement (%, ".to_string() + grid_refinement_total_string.0.as_str() + ")",
            format!(
                "{}, {}",
                (grid_refinement_time_percent * 1000.0).round() / 1000.0,
                grid_refinement_total_string.1
            ),
        );
        timer_data.insert(
            "Jacobian (%, ".to_string() + jac_total_string.0.as_str() + ")",
            format!(
                "{}, {}",
                (jac_time_percent * 1000.0).round() / 1000.0,
                jac_total_string.1
            ),
        );
        timer_data.insert(
            "Function (%, ".to_string() + fun_total_string.0.as_str() + ")",
            format!(
                "{}, {}",
                (fun_time_percent * 1000.0).round() / 1000.0,
                fun_total_string.1
            ),
        );
        timer_data.insert(
            "Linear System (%, ".to_string() + linear_system_total_string.0.as_str() + ")",
            format!(
                "{}, {}",
                (linear_system_time_percent * 1000.0).round() / 1000.0,
                linear_system_total_string.1
            ),
        );
        timer_data.insert(
            "Factorization (%, ".to_string() + factorization_total_string.0.as_str() + ")",
            format!(
                "{}, {}",
                (factorization_time_percent * 1000.0).round() / 1000.0,
                factorization_total_string.1
            ),
        );
        timer_data.insert(
            "RHS Solve (%, ".to_string() + rhs_solve_total_string.0.as_str() + ")",
            format!(
                "{}, {}",
                (rhs_solve_time_percent * 1000.0).round() / 1000.0,
                rhs_solve_total_string.1
            ),
        );
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
        timer_data.insert(
            "Backend Preparation (%, ".to_string()
                + symbolic_operations_total_string.0.as_str()
                + ")",
            format!(
                "{}, {}",
                (symbolic_operations_time_percent * 1000.0).round() / 1000.0,
                symbolic_operations_total_string.1
            ),
        );
        for (label, duration) in callback_stage_timer_snapshot() {
            insert_duration_timer(&mut timer_data, label.as_str(), duration, total_time);
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

/// Estimates dense-equivalent matrix memory in MiB for diagnostics.
///
/// Keep this helper lightweight: it is called from `get_statistics()`, so it
/// must not refresh full system/process information on every statistics read.
/// Global free-memory checks belong to `task_check_mem()`, which runs before
/// symbolic matrix construction rather than inside the reporting path.
pub fn checkmem(mat: &dyn MatrixType) -> f64 {
    let (nrows, ncols) = mat.shape();
    let matrix_memory = (nrows * ncols * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0);
    info!("Matrix memory usage: {:.2} MB", matrix_memory);
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
    solution: DVector<f64>,
    BC_position_and_value: Vec<(usize, usize, f64)>,
) -> DVector<f64> {
    info!("Constructing full solution");

    let full_size = solution.len() + BC_position_and_value.len();
    let mut full_solution = vec![0.0; full_size];

    // Insert boundary condition values at their positions
    for (pos, _, value) in &BC_position_and_value {
        full_solution[*pos] = *value;
    }

    // Insert solution values at remaining positions
    let mut solution_idx = 0;
    for i in 0..full_size {
        if !BC_position_and_value.iter().any(|(pos, _, _)| *pos == i) {
            full_solution[i] = solution[solution_idx];
            solution_idx += 1;
        }
    }

    DVector::from_vec(full_solution)
}

pub fn extract_unknown_variables(
    full_solution: DMatrix<f64>,
    BC_position_and_value: &Vec<(usize, usize, f64)>,
    number_of_nonzero_keys: usize,
) -> DVector<f64> {
    info!("Extracting unknown variables from full solution");
    let (n_rows, _) = full_solution.shape();
    // Convert matrix to vector (interleaved format)
    let full_vector: Vec<f64> = full_solution.iter().cloned().collect();

    let columns_added = number_of_nonzero_keys;

    // Calculate BC positions and create sorted vector of tuples (position, index, value)
    let mut biased_bc_tuples = if columns_added == 0 {
        // No grid expansion - use original positions
        BC_position_and_value
            .iter()
            .map(|(pos, index, value)| (*pos, *index, *value))
            .collect::<Vec<(usize, usize, f64)>>()
    } else {
        // Grid was expanded - adjust positions
        let right_bc_bias = (columns_added - 1) * n_rows;
        BC_position_and_value
            .iter()
            .map(|(pos, index, value)| {
                if *index == 0 {
                    (*pos, *index, *value)
                } else {
                    (*pos + index * n_rows + right_bc_bias, *index, *value)
                }
            })
            .collect::<Vec<(usize, usize, f64)>>()
    };

    // Sort by position for efficient iteration
    biased_bc_tuples.sort_by_key(|(pos, _, _)| *pos);

    info!(
        "Sorted biased BC tuples: {:?} and the length of full_vector {}",
        biased_bc_tuples,
        full_vector.len()
    );

    // Extract unknown values by iterating through full_vector and skipping BC positions
    let mut unknown_values = Vec::new();
    let mut bc_iter = biased_bc_tuples.iter().peekable();

    for (i, value) in full_vector.into_iter().enumerate() {
        // Check if current position matches next BC position
        if let Some((bc_pos, _, bc_value)) = bc_iter.peek() {
            if i == *bc_pos {
                // This is a BC position - verify value and skip
                if (value - bc_value).abs() < 1e-12 {
                    info!("found BC position {} with value {}", i, value);
                } else {
                    // Value doesn't match expected BC - treat as unknown
                    unknown_values.push(value);
                }
                bc_iter.next(); // Move to next BC
                continue;
            }
        }
        // Not a BC position - include as unknown
        unknown_values.push(value);
    }

    DVector::from_vec(unknown_values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_construct_full_solution() {
        // Test with simple boundary conditions: x at position 0 = 0.0, y at position 3 = 5.0
        let solution = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let bc_position_and_value = vec![(0, 0, 0.0), (5, 1, 5.0)];

        let constructed_solution = construct_full_solution(solution, bc_position_and_value);
        let expected = DVector::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

        assert_eq!(constructed_solution, expected);
    }

    #[test]
    fn test_extract_unknown_variables() {
        // Test extracting unknowns from a full solution matrix
        let full_solution =
            DMatrix::from_row_slice(2, 3, &[0.0, 1.0, 2.0, 5.0, 3.0, 4.0]).transpose();
        let bc_position_and_value = vec![(0, 0, 0.0), (3, 1, 5.0)];

        let extracted = extract_unknown_variables(full_solution, &bc_position_and_value, 0);
        let expected = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        println!(" {}", extracted);
        assert_eq!(extracted, expected);
    }

    #[test]
    fn test_combine() {
        // Test round-trip: construct full solution then extract unknowns
        let original_solution = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let bc_position_and_value = vec![(0, 0, 100.0), (5, 1, 500.0)];

        let full_solution =
            construct_full_solution(original_solution.clone(), bc_position_and_value.clone());

        // Convert to matrix format for extraction (2x3 matrix)
        let full_matrix = DMatrix::from_row_slice(2, 3, full_solution.as_slice()).transpose();
        let extracted = extract_unknown_variables(full_matrix, &bc_position_and_value, 0);
        println!(
            "full solution {} \n extracted {}",
            original_solution, extracted
        );
        assert_eq!(original_solution, extracted);
    }

    #[test]
    fn test_multiple_boundary_conditions() {
        // Test with multiple boundary conditions
        let solution = DVector::from_vec(vec![1.0, 2.0]);
        let bc_position_and_value = vec![(0, 0, 0.36787944117144233), (3, 1, 0.36787944117144233)];

        let full_solution =
            construct_full_solution(solution.clone(), bc_position_and_value.clone());

        // Should have 4 elements total
        assert_eq!(full_solution.len(), 4);

        // Convert to matrix and extract
        let full_matrix = DMatrix::from_row_slice(1, 4, full_solution.as_slice());
        let extracted = extract_unknown_variables(full_matrix, &bc_position_and_value, 0);

        assert_eq!(solution, extracted);
    }

    #[test]
    fn custom_timer_get_all_does_not_emit_nan_percentages() {
        let timer = CustomTimer::new();
        let data = timer.get_all();

        assert!(
            data.values().all(|value| !value.contains("NaN")),
            "timer output should not contain NaN values: {data:?}"
        );
        assert!(
            data.keys()
                .any(|key| key.starts_with("Symbolic Operations")),
            "legacy symbolic timer key should remain available for existing story tables"
        );
        assert!(
            data.keys()
                .any(|key| key.starts_with("Backend Preparation")),
            "backend preparation timer alias should be available for numeric/codegen diagnostics"
        );
    }

    #[test]
    fn custom_timer_reports_and_resets_callback_stage_timers() {
        reset_callback_stage_timers();
        record_callback_stage_time("Callback Jacobian Values", Duration::from_millis(2));
        record_callback_stage_time("Callback Jacobian Values", Duration::from_millis(3));

        let data = CustomTimer::new().get_all();
        let key = "Callback Jacobian Values (%, ms)";
        let value = data
            .get(key)
            .unwrap_or_else(|| panic!("callback stage timer should be exported under {key}"));
        let millis = value
            .split(',')
            .nth(1)
            .expect("timer value should use 'percent, milliseconds' format")
            .trim()
            .parse::<f64>()
            .expect("callback timer milliseconds should be numeric");
        assert!(
            (millis - 5.0).abs() < 1.0e-9,
            "callback timer should accumulate repeated measurements, got {millis}"
        );

        let mut timer = CustomTimer::new();
        timer.start();
        let data_after_reset = timer.get_all();
        assert!(
            !data_after_reset.contains_key(key),
            "CustomTimer::start should reset callback stage timers"
        );
    }

    #[test]
    fn custom_timer_callback_stages_are_solve_local_and_nested_sessions_do_not_mix() {
        let mut outer = CustomTimer::new();
        outer.start();
        record_callback_stage_time("Callback Residual Values", Duration::from_millis(2));

        let mut inner = CustomTimer::new();
        inner.start();
        record_callback_stage_time("Callback Jacobian Values", Duration::from_millis(3));
        inner.finish();

        record_callback_stage_time("Callback Residual Values", Duration::from_millis(4));
        outer.finish();

        let outer_stages = outer.snapshot().callback_stage_timings;
        let inner_stages = inner.snapshot().callback_stage_timings;
        assert_eq!(
            outer_stages
                .iter()
                .find(|timing| timing.stage == BvpCallbackStage::ResidualValues)
                .map(|timing| timing.elapsed),
            Some(Duration::from_millis(6))
        );
        assert_eq!(
            inner_stages
                .iter()
                .find(|timing| timing.stage == BvpCallbackStage::JacobianValues)
                .map(|timing| timing.elapsed),
            Some(Duration::from_millis(3))
        );
        assert!(
            inner_stages
                .iter()
                .all(|timing| timing.stage != BvpCallbackStage::ResidualValues)
        );
    }

    #[test]
    fn custom_timer_off_skips_stage_timing_and_callback_collection() {
        reset_callback_stage_timers();
        let mut timer = CustomTimer::new();
        timer.set_telemetry_mode(BvpTelemetryMode::Off);
        timer.start();
        timer.fun_tic();
        timer.fun_tac();
        record_callback_stage_time("Callback Residual Values", Duration::from_millis(5));
        timer.finish();

        assert_eq!(timer.snapshot(), BvpTimingSnapshot::default());
        assert!(callback_stage_timings_snapshot().is_empty());
    }

    #[test]
    fn callback_stage_snapshot_keeps_unknown_stages_visible_without_dynamic_hot_state() {
        reset_callback_stage_timers();
        record_callback_stage_time("Callback Future Stage", Duration::from_millis(7));

        let typed = callback_stage_timings_snapshot();
        assert_eq!(typed.len(), 1);
        assert_eq!(typed[0].stage.label(), "Callback Other");
        assert_eq!(typed[0].elapsed, Duration::from_millis(7));

        let snapshot = callback_stage_timer_snapshot();
        assert_eq!(
            snapshot.get("Callback Other").copied(),
            Some(Duration::from_millis(7))
        );
        assert!(!snapshot.contains_key("Callback Future Stage"));
    }
}

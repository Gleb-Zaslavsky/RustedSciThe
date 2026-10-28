use crate::command_interpreter::task_wizard::{self, Question};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IvpTemplateMethod {
    LsodaAuto,
    LsodeAdams,
    LsodeBdf,
    Bdf,
    Radau5,
    BackwardEuler,
    Dopri,
    Ab4,
    Rk45,
}

impl IvpTemplateMethod {
    fn task_value(self) -> &'static str {
        match self {
            Self::LsodaAuto => "LSODA",
            Self::LsodeAdams | Self::LsodeBdf => "LSODE",
            Self::Bdf => "BDF",
            Self::Radau5 => "Radau5",
            Self::BackwardEuler => "BackwardEuler",
            Self::Dopri => "DOPRI",
            Self::Ab4 => "AB4",
            Self::Rk45 => "RK45",
        }
    }

    fn is_lsode2(self) -> bool {
        matches!(self, Self::LsodaAuto | Self::LsodeAdams | Self::LsodeBdf)
    }

    fn lsode2_family_value(self) -> Option<&'static str> {
        match self {
            Self::LsodaAuto => Some("auto"),
            Self::LsodeAdams => Some("adams"),
            Self::LsodeBdf => Some("bdf"),
            _ => None,
        }
    }

    fn needs_bdf_execution_choice(self) -> bool {
        matches!(self, Self::LsodaAuto | Self::LsodeBdf)
    }

    fn has_native_controls(self) -> bool {
        matches!(self, Self::Bdf | Self::Radau5 | Self::BackwardEuler)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IvpSymbolicAssembly {
    AtomView,
    ExprLegacy,
}

impl IvpSymbolicAssembly {
    fn task_value(self) -> &'static str {
        match self {
            Self::AtomView => "AtomView",
            Self::ExprLegacy => "ExprLegacy",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IvpSymbolicExecution {
    LambdifyExpr,
    Aot,
}

impl IvpSymbolicExecution {
    fn task_value(self) -> &'static str {
        match self {
            Self::LambdifyExpr => "LambdifyExpr",
            Self::Aot => "AOT",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IvpAotToolchain {
    CTcc,
    CGcc,
    Zig,
    Rust,
}

impl IvpAotToolchain {
    fn task_value(self) -> &'static str {
        match self {
            Self::CTcc => "c_tcc",
            Self::CGcc => "c_gcc",
            Self::Zig => "zig",
            Self::Rust => "rust",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IvpAotProfile {
    Release,
    Debug,
}

impl IvpAotProfile {
    fn task_value(self) -> &'static str {
        match self {
            Self::Release => "release",
            Self::Debug => "debug",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IvpLinearStructure {
    Dense,
    Sparse,
    Banded,
}

impl IvpLinearStructure {
    fn task_value(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Sparse => "sparse",
            Self::Banded => "banded",
        }
    }

    fn default_policy(self) -> IvpLinearSolverPolicy {
        match self {
            Self::Dense => IvpLinearSolverPolicy::DenseLu,
            Self::Sparse => IvpLinearSolverPolicy::FaerSparseLu,
            Self::Banded => IvpLinearSolverPolicy::LapackFaithfulBandedLu,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IvpLinearSolverPolicy {
    Auto,
    DenseLu,
    FaerSparseLu,
    LapackFaithfulBandedLu,
}

impl IvpLinearSolverPolicy {
    fn task_value(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::DenseLu => "dense_lu",
            Self::FaerSparseLu => "faer_sparse_lu",
            Self::LapackFaithfulBandedLu => "lapack_faithful_banded_lu",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IvpNativeExecution {
    FaithfulBdfSolve,
    ProbeBeforeBridge,
    BridgeSolve,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IvpExecutionPolicy {
    Sequential,
    Parallel,
    Auto,
}

impl IvpExecutionPolicy {
    fn task_value(self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::Parallel => "parallel",
            Self::Auto => "auto",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IvpTelemetry {
    Off,
    Counters,
    Timings,
}

impl IvpTelemetry {
    fn task_value(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Counters => "counters",
            Self::Timings => "timings",
        }
    }
}

impl IvpNativeExecution {
    fn task_value(self) -> &'static str {
        match self {
            Self::FaithfulBdfSolve => "faithful_bdf_solve",
            Self::ProbeBeforeBridge => "probe_before_bridge",
            Self::BridgeSolve => "bridge_solve",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct IvpTemplateConfig {
    method: IvpTemplateMethod,
    symbolic_assembly: IvpSymbolicAssembly,
    symbolic_execution: IvpSymbolicExecution,
    aot_toolchain: IvpAotToolchain,
    aot_profile: IvpAotProfile,
    linear_structure: IvpLinearStructure,
    linear_solver_policy: IvpLinearSolverPolicy,
    native_execution: IvpNativeExecution,
    execution_policy: IvpExecutionPolicy,
    telemetry: IvpTelemetry,
    tolerance: f64,
    max_iterations: usize,
    rtol: f64,
    atol: f64,
    max_step: Option<f64>,
    first_step: Option<f64>,
    step_size: Option<f64>,
}

impl Default for IvpTemplateConfig {
    fn default() -> Self {
        let linear_structure = IvpLinearStructure::Banded;
        Self {
            method: IvpTemplateMethod::LsodaAuto,
            symbolic_assembly: IvpSymbolicAssembly::AtomView,
            symbolic_execution: IvpSymbolicExecution::LambdifyExpr,
            aot_toolchain: IvpAotToolchain::CTcc,
            aot_profile: IvpAotProfile::Release,
            linear_structure,
            linear_solver_policy: linear_structure.default_policy(),
            native_execution: IvpNativeExecution::FaithfulBdfSolve,
            execution_policy: IvpExecutionPolicy::Sequential,
            telemetry: IvpTelemetry::Timings,
            tolerance: 1e-6,
            max_iterations: 100,
            rtol: 1e-6,
            atol: 1e-9,
            max_step: Some(1e-3),
            first_step: Some(1e-6),
            step_size: Some(1e-3),
        }
    }
}

fn build_ivp_task_template(config: &IvpTemplateConfig) -> String {
    let mut template = String::new();
    template.push_str("task\n");
    template.push_str("solver: IVP\n");
    template.push_str(&format!("method: {}\n\n", config.method.task_value()));

    template.push_str("equations\n");
    template.push_str("arg: t\n");
    template.push_str("unknowns: y1, y2, y3\n");
    template.push_str("rhs: -0.04*y1 + 1e4*y2*y3, 0.04*y1 - 1e4*y2*y3 - 3e7*y2^2, 3e7*y2^2\n\n");

    template.push_str("initial_conditions\n");
    template.push_str("t0: 0.0\n");
    template.push_str("t_end: 1.0\n");
    template.push_str("y0: 1.0, 0.0, 0.0\n\n");

    template.push_str("solver_options\n");
    match config.method {
        IvpTemplateMethod::BackwardEuler => {
            template.push_str(&format!("tolerance: {}\n", config.tolerance));
            template.push_str(&format!("max_iterations: {}\n", config.max_iterations));
            if let Some(step_size) = config.step_size {
                template.push_str(&format!("step_size: {}\n", step_size));
            }
        }
        IvpTemplateMethod::Bdf | IvpTemplateMethod::Radau5 => {
            template.push_str(&format!("rtol: {}\n", config.rtol));
            template.push_str(&format!("atol: {}\n", config.atol));
            template.push_str(&format!("max_iterations: {}\n", config.max_iterations));
            if let Some(max_step) = config.max_step {
                template.push_str(&format!("max_step: {}\n", max_step));
            }
            if let Some(first_step) = config.first_step {
                template.push_str(&format!("first_step: Some({})\n", first_step));
            }
        }
        _ => {
            template.push_str("rtol: 1e-6\n");
            template.push_str("atol: 1e-9\n");
            template.push_str("max_step: 1e-3\n");
            template.push_str("first_step: Some(1e-6)\n");
        }
    }

    if config.method.is_lsode2() {
        if let Some(family) = config.method.lsode2_family_value() {
            template.push_str(&format!("lsode2_method_family: {family}\n"));
        }
        template.push_str(&format!(
            "lsode2_symbolic_assembly: {}\n",
            config.symbolic_assembly.task_value()
        ));
        template.push_str(&format!(
            "lsode2_symbolic_execution: {}\n",
            config.symbolic_execution.task_value()
        ));
        if matches!(config.symbolic_execution, IvpSymbolicExecution::Aot) {
            template.push_str(&format!(
                "lsode2_aot_toolchain: {}\n",
                config.aot_toolchain.task_value()
            ));
            template.push_str(&format!(
                "lsode2_aot_profile: {}\n",
                config.aot_profile.task_value()
            ));
        }
        template.push_str(&format!(
            "lsode2_linear_structure: {}\n",
            config.linear_structure.task_value()
        ));
        template.push_str(&format!(
            "lsode2_linear_solver_policy: {}\n",
            config.linear_solver_policy.task_value()
        ));
        if config.method.needs_bdf_execution_choice() {
            template.push_str(&format!(
                "lsode2_native_execution: {}\n",
                config.native_execution.task_value()
            ));
        }
    } else if config.method.has_native_controls() {
        template.push_str(&format!(
            "symbolic_frontend: {}\n",
            config.symbolic_assembly.task_value()
        ));
        template.push_str(&format!(
            "symbolic_execution: {}\n",
            config.symbolic_execution.task_value()
        ));
        if matches!(config.symbolic_execution, IvpSymbolicExecution::Aot) {
            template.push_str(&format!(
                "aot_toolchain: {}\n",
                config.aot_toolchain.task_value()
            ));
            template.push_str(&format!(
                "aot_profile: {}\n",
                config.aot_profile.task_value()
            ));
        }
        template.push_str(&format!(
            "matrix_layout: {}\n",
            if config.method == IvpTemplateMethod::BackwardEuler {
                "dense"
            } else {
                config.linear_structure.task_value()
            }
        ));
        if config.method != IvpTemplateMethod::BackwardEuler {
            template.push_str(&format!(
                "execution_policy: {}\n",
                config.execution_policy.task_value()
            ));
        }
        template.push_str(&format!("telemetry: {}\n", config.telemetry.task_value()));
    }

    template.push_str("\npostprocessing\n");
    template.push_str("save_csv: true\n");
    template.push_str("csv_path: ivp_result.csv\n");
    template.push_str("save_txt: false\n");
    template.push_str("write_report: false\n");
    template.push_str("plotters_png: false\n");
    template.push_str("gnuplot_png: false\n");
    template.push_str("terminal_plot: false\n");
    template.push_str("plot: false\n");
    template
}

pub fn run_ivp_template_dialogue(output_path: Option<PathBuf>) -> std::io::Result<()> {
    let Some(answers) = task_wizard::run("--- IVP task document template wizard ---", |answers| {
        let mut questions = vec![Question {
            key: "method",
            prompt: "Choose the IVP method",
            options: &[
                "LSODA-style automatic Adams/BDF switching (recommended for mixed IVPs)",
                "LSODE fixed Adams family (non-stiff IVPs)",
                "LSODE fixed BDF family (stiff IVPs)",
                "Native BDF",
                "Radau5",
                "Backward Euler",
                "DOPRI (Dormand-Prince, non-stiff)",
                "AB4 (Adams-Bashforth 4, non-stiff)",
                "RK45 (non-stiff)",
            ],
            default: 0,
        }];
        let method = answers.get("method").copied().unwrap_or(0);
        if method >= 3 && method <= 5 {
            questions.push(Question {
                key: "tolerance_profile",
                prompt: "Choose numerical tolerance",
                options: &["standard (1e-6)", "strict (1e-8)", "relaxed (1e-4)"],
                default: 0,
            });
            questions.push(Question {
                key: "iteration_budget",
                prompt: "Choose nonlinear/step iteration budget",
                options: &["standard (100)", "extended (1000)", "large (10000)"],
                default: 0,
            });
            questions.push(Question {
                key: "step_profile",
                prompt: if method == 5 {
                    "Choose Backward Euler step size"
                } else {
                    "Choose adaptive step limits"
                },
                options: if method == 5 {
                    &["1e-3 (standard)", "1e-2", "1e-1"]
                } else {
                    &[
                        "max=1e-3, first=1e-6",
                        "max=1e-2, first=1e-5",
                        "unbounded adaptive",
                    ]
                },
                default: 0,
            });
            questions.push(Question {
                key: "assembly",
                prompt: "Choose symbolic frontend",
                options: &[
                    "AtomView (native symbolic path)",
                    "ExprLegacy (compatibility path)",
                ],
                default: 0,
            });
            questions.push(Question {
                key: "execution",
                prompt: "Choose symbolic execution",
                options: &[
                    "Lambdify (no external compiler)",
                    "AOT (compiled generated callback)",
                ],
                default: 0,
            });
            if answers.get("execution") == Some(&1) {
                questions.push(Question {
                    key: "toolchain",
                    prompt: "Choose AOT toolchain",
                    options: &["c_tcc", "c_gcc", "zig", "rust"],
                    default: 0,
                });
                questions.push(Question {
                    key: "profile",
                    prompt: "Choose AOT build profile",
                    options: &["release", "debug"],
                    default: 0,
                });
            }
            if method != 5 {
                questions.push(Question {
                    key: "structure",
                    prompt: "Choose matrix layout",
                    options: &["banded", "sparse", "dense"],
                    default: 0,
                });
                questions.push(Question {
                    key: "execution_policy",
                    prompt: "Choose symbolic callback execution policy",
                    options: &["sequential", "parallel", "auto"],
                    default: 0,
                });
            }
            questions.push(Question {
                key: "telemetry",
                prompt: "Choose solver telemetry",
                options: &["off", "counters", "timings (recommended for diagnostics)"],
                default: 2,
            });
        }
        if method <= 2 {
            questions.push(Question {
                key: "assembly",
                prompt: "Choose LSODE2 symbolic assembly",
                options: &[
                    "AtomView (recommended for large symbolic systems)",
                    "ExprLegacy (compatibility path)",
                ],
                default: 0,
            });
            questions.push(Question {
                key: "execution",
                prompt: "Choose LSODE2 symbolic execution",
                options: &[
                    "LambdifyExpr (fast startup, no external compiler)",
                    "AOT (compiled generated callback)",
                ],
                default: 0,
            });
            if answers.get("execution") == Some(&1) {
                questions.push(Question {
                    key: "toolchain",
                    prompt: "Choose AOT toolchain",
                    options: &[
                        "c_tcc (fast cold build, good default for experiments)",
                        "c_gcc",
                        "zig",
                        "rust",
                    ],
                    default: 0,
                });
                questions.push(Question {
                    key: "profile",
                    prompt: "Choose AOT build profile",
                    options: &["release", "debug"],
                    default: 0,
                });
            }
            questions.push(Question {
                key: "structure",
                prompt: "Choose linear system structure",
                options: &["banded", "sparse", "dense"],
                default: 0,
            });
            questions.push(Question {
                key: "linear_policy",
                prompt: "Choose linear solver policy",
                options: &[
                    "auto",
                    "dense_lu",
                    "faer_sparse_lu",
                    "lapack_faithful_banded_lu",
                ],
                default: match answers.get("structure").copied().unwrap_or(0) {
                    0 => 0,
                    1 => 2,
                    _ => 1,
                },
            });
            if method == 0 || method == 2 {
                questions.push(Question {
                    key: "native_execution",
                    prompt: "Choose BDF execution mode",
                    options: &[
                        "faithful_bdf_solve (recommended)",
                        "probe_before_bridge",
                        "bridge_solve",
                    ],
                    default: 0,
                });
            }
        }
        questions
    }) else {
        println!("IVP template wizard cancelled; no file was written.");
        return Ok(());
    };

    let selected = |key| answers.get(key).copied().unwrap_or(0);
    let method_idx = selected("method");
    let method = match method_idx {
        0 => IvpTemplateMethod::LsodaAuto,
        1 => IvpTemplateMethod::LsodeAdams,
        2 => IvpTemplateMethod::LsodeBdf,
        3 => IvpTemplateMethod::Bdf,
        4 => IvpTemplateMethod::Radau5,
        5 => IvpTemplateMethod::BackwardEuler,
        6 => IvpTemplateMethod::Dopri,
        7 => IvpTemplateMethod::Ab4,
        _ => IvpTemplateMethod::Rk45,
    };

    let mut config = IvpTemplateConfig {
        method,
        ..IvpTemplateConfig::default()
    };

    if method.has_native_controls() {
        config.tolerance = match selected("tolerance_profile") {
            1 => 1e-8,
            2 => 1e-4,
            _ => 1e-6,
        };
        config.max_iterations = match selected("iteration_budget") {
            1 => 1_000,
            2 => 10_000,
            _ => 100,
        };
        if method == IvpTemplateMethod::BackwardEuler {
            config.step_size = Some(match selected("step_profile") {
                1 => 1e-2,
                2 => 1e-1,
                _ => 1e-3,
            });
        } else {
            match selected("step_profile") {
                1 => {
                    config.max_step = Some(1e-2);
                    config.first_step = Some(1e-5);
                }
                2 => {
                    config.max_step = None;
                    config.first_step = None;
                }
                _ => {}
            }
            config.rtol = config.tolerance;
            config.atol = config.tolerance * 1e-3;
        }
    }

    if method.is_lsode2() {
        config.symbolic_assembly = if selected("assembly") == 0 {
            IvpSymbolicAssembly::AtomView
        } else {
            IvpSymbolicAssembly::ExprLegacy
        };
        config.symbolic_execution = if selected("execution") == 0 {
            IvpSymbolicExecution::LambdifyExpr
        } else {
            IvpSymbolicExecution::Aot
        };

        if matches!(config.symbolic_execution, IvpSymbolicExecution::Aot) {
            config.aot_toolchain = match selected("toolchain") {
                0 => IvpAotToolchain::CTcc,
                1 => IvpAotToolchain::CGcc,
                2 => IvpAotToolchain::Zig,
                _ => IvpAotToolchain::Rust,
            };
            config.aot_profile = if selected("profile") == 0 {
                IvpAotProfile::Release
            } else {
                IvpAotProfile::Debug
            };
        }

        config.linear_structure = match selected("structure") {
            0 => IvpLinearStructure::Banded,
            1 => IvpLinearStructure::Sparse,
            _ => IvpLinearStructure::Dense,
        };
        config.linear_solver_policy = match selected("linear_policy") {
            0 => IvpLinearSolverPolicy::Auto,
            1 => IvpLinearSolverPolicy::DenseLu,
            2 => IvpLinearSolverPolicy::FaerSparseLu,
            _ => IvpLinearSolverPolicy::LapackFaithfulBandedLu,
        };

        if method.needs_bdf_execution_choice() {
            config.native_execution = match selected("native_execution") {
                0 => IvpNativeExecution::FaithfulBdfSolve,
                1 => IvpNativeExecution::ProbeBeforeBridge,
                _ => IvpNativeExecution::BridgeSolve,
            };
        }
    } else if method.has_native_controls() {
        config.symbolic_assembly = if selected("assembly") == 0 {
            IvpSymbolicAssembly::AtomView
        } else {
            IvpSymbolicAssembly::ExprLegacy
        };
        config.symbolic_execution = if selected("execution") == 0 {
            IvpSymbolicExecution::LambdifyExpr
        } else {
            IvpSymbolicExecution::Aot
        };
        if matches!(config.symbolic_execution, IvpSymbolicExecution::Aot) {
            config.aot_toolchain = match selected("toolchain") {
                0 => IvpAotToolchain::CTcc,
                1 => IvpAotToolchain::CGcc,
                2 => IvpAotToolchain::Zig,
                _ => IvpAotToolchain::Rust,
            };
            config.aot_profile = if selected("profile") == 0 {
                IvpAotProfile::Release
            } else {
                IvpAotProfile::Debug
            };
        }
        if method != IvpTemplateMethod::BackwardEuler {
            config.linear_structure = match selected("structure") {
                0 => IvpLinearStructure::Banded,
                1 => IvpLinearStructure::Sparse,
                _ => IvpLinearStructure::Dense,
            };
            config.execution_policy = match selected("execution_policy") {
                1 => IvpExecutionPolicy::Parallel,
                2 => IvpExecutionPolicy::Auto,
                _ => IvpExecutionPolicy::Sequential,
            };
        } else {
            config.linear_structure = IvpLinearStructure::Dense;
        }
        config.telemetry = match selected("telemetry") {
            0 => IvpTelemetry::Off,
            1 => IvpTelemetry::Counters,
            _ => IvpTelemetry::Timings,
        };
    }

    let template = build_ivp_task_template(&config);
    let final_path = output_path.unwrap_or_else(|| PathBuf::from("ivp_task_custom.txt"));

    match std::fs::write(&final_path, template) {
        Ok(()) => println!("IVP task template saved to {}", final_path.display()),
        Err(error) => return Err(error),
    }
    Ok(())
}

/// Writes the deterministic default IVP task document without opening a
/// terminal dialogue. Interactive customization belongs to `wizard`.
pub fn write_ivp_task_template(path: Option<PathBuf>) -> std::io::Result<PathBuf> {
    let path = path.unwrap_or_else(|| PathBuf::from("ivp_task_template.txt"));
    std::fs::write(
        &path,
        build_ivp_task_template(&IvpTemplateConfig::default()),
    )?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command_interpreter::task_parser_ivp::{
        IvpMethodSpec, Lsode2TaskExecutionSpec, parse_ivp_task_from_str,
    };
    use crate::command_interpreter::task_runner::{parse_task_spec_from_str, render_task_check};
    use crate::numerical::LSODE2::{
        Lsode2AotProfile, Lsode2AotToolchain, Lsode2ControllerMode, Lsode2LinearSolverChoice,
        Lsode2LinearSolverPolicy, Lsode2LinearSystemStructure,
    };

    fn assert_template_passes_cli_preflight(template: &str) {
        let spec = parse_task_spec_from_str(template).expect("generated IVP should parse");
        let report = render_task_check(&spec);
        assert!(report.contains("preflight=valid; integration has not been run"));
    }

    #[test]
    fn ivp_dialogue_lsoda_auto_lambdify_template_is_parser_ready() {
        let template = build_ivp_task_template(&IvpTemplateConfig::default());
        assert_template_passes_cli_preflight(&template);
        let spec = parse_ivp_task_from_str(&template).expect("dialogue template should parse");

        assert_eq!(spec.solver.method, IvpMethodSpec::Lsoda);
        let lsode2 = spec
            .solver_options
            .lsode2
            .expect("LSODE2 options should be present");
        assert_eq!(
            lsode2.controller.expect("controller should be parsed").mode,
            Lsode2ControllerMode::AutomaticAdamsBdf
        );
        assert_eq!(
            lsode2.linear_system_structure,
            Some(Lsode2LinearSystemStructure::BandedAuto)
        );
        assert_eq!(
            lsode2.linear_solver_policy,
            Some(Lsode2LinearSolverPolicy::Force(
                Lsode2LinearSolverChoice::LapackFaithfulBandedLu
            ))
        );
        assert!(matches!(
            lsode2.symbolic_execution,
            Some(Lsode2TaskExecutionSpec::LambdifyExpr)
        ));
    }

    #[test]
    fn ivp_dialogue_offers_only_runnable_nonstiff_methods() {
        for method in [
            IvpTemplateMethod::Rk45,
            IvpTemplateMethod::Dopri,
            IvpTemplateMethod::Ab4,
        ] {
            let config = IvpTemplateConfig {
                method,
                ..IvpTemplateConfig::default()
            };
            let template = build_ivp_task_template(&config);
            assert_template_passes_cli_preflight(&template);
            assert!(
                parse_ivp_task_from_str(&template).is_ok(),
                "{} template must be executable by the task runner",
                method.task_value()
            );
        }
    }

    #[test]
    fn ivp_dialogue_lsode2_aot_template_is_parser_ready() {
        let config = IvpTemplateConfig {
            symbolic_execution: IvpSymbolicExecution::Aot,
            aot_toolchain: IvpAotToolchain::CTcc,
            aot_profile: IvpAotProfile::Release,
            linear_structure: IvpLinearStructure::Sparse,
            linear_solver_policy: IvpLinearSolverPolicy::Auto,
            ..IvpTemplateConfig::default()
        };
        let template = build_ivp_task_template(&config);
        assert_template_passes_cli_preflight(&template);
        let spec = parse_ivp_task_from_str(&template).expect("AOT dialogue template should parse");
        let lsode2 = spec
            .solver_options
            .lsode2
            .expect("LSODE2 options should be present");

        assert_eq!(
            lsode2.linear_system_structure,
            Some(Lsode2LinearSystemStructure::Sparse)
        );
        assert_eq!(
            lsode2.linear_solver_policy,
            Some(Lsode2LinearSolverPolicy::Auto)
        );
        assert!(matches!(
            lsode2.symbolic_execution,
            Some(Lsode2TaskExecutionSpec::Aot {
                toolchain: Lsode2AotToolchain::CTcc,
                profile: Lsode2AotProfile::Release,
                ..
            })
        ));
    }

    #[test]
    fn ivp_dialogue_lsode_fixed_adams_template_selects_adams_controller() {
        let config = IvpTemplateConfig {
            method: IvpTemplateMethod::LsodeAdams,
            ..IvpTemplateConfig::default()
        };
        let template = build_ivp_task_template(&config);
        assert_template_passes_cli_preflight(&template);
        assert!(template.contains("method: LSODE"));
        assert!(template.contains("lsode2_method_family: adams"));
        assert!(!template.contains("lsode2_native_execution"));

        let spec = parse_ivp_task_from_str(&template).expect("fixed Adams template should parse");
        let lsode2 = spec
            .solver_options
            .lsode2
            .expect("LSODE2 options should be present");
        assert_eq!(
            lsode2.controller.expect("controller should be parsed").mode,
            Lsode2ControllerMode::AdamsOnly
        );
    }

    #[test]
    fn ivp_dialogue_lsode_fixed_bdf_template_selects_bdf_controller() {
        let config = IvpTemplateConfig {
            method: IvpTemplateMethod::LsodeBdf,
            native_execution: IvpNativeExecution::ProbeBeforeBridge,
            ..IvpTemplateConfig::default()
        };
        let template = build_ivp_task_template(&config);
        assert_template_passes_cli_preflight(&template);
        assert!(template.contains("method: LSODE"));
        assert!(template.contains("lsode2_method_family: bdf"));
        assert!(template.contains("lsode2_native_execution: probe_before_bridge"));

        let spec = parse_ivp_task_from_str(&template).expect("fixed BDF template should parse");
        let lsode2 = spec
            .solver_options
            .lsode2
            .expect("LSODE2 options should be present");
        assert_eq!(
            lsode2.controller.expect("controller should be parsed").mode,
            Lsode2ControllerMode::BdfOnly
        );
    }

    #[test]
    fn ivp_dialogue_non_lsode2_template_does_not_emit_lsode2_fields() {
        let config = IvpTemplateConfig {
            method: IvpTemplateMethod::Radau5,
            ..IvpTemplateConfig::default()
        };
        let template = build_ivp_task_template(&config);
        assert_template_passes_cli_preflight(&template);
        assert!(!template.contains("lsode2_"));
        let spec = parse_ivp_task_from_str(&template).expect("Radau template should parse");
        assert_eq!(spec.solver.method, IvpMethodSpec::Radau5);
    }

    #[test]
    fn ivp_native_templates_expose_solver_controls_and_remain_parser_ready() {
        for method in [IvpTemplateMethod::Bdf, IvpTemplateMethod::Radau5] {
            let config = IvpTemplateConfig {
                method,
                ..IvpTemplateConfig::default()
            };
            let template = build_ivp_task_template(&config);
            assert!(template.contains("max_iterations: 100"));
            assert!(template.contains("max_step: 0.001"));
            assert_template_passes_cli_preflight(&template);
            parse_ivp_task_from_str(&template).unwrap_or_else(|error| {
                panic!("{} template must parse: {error}", method.task_value())
            });
        }

        let config = IvpTemplateConfig {
            method: IvpTemplateMethod::BackwardEuler,
            ..IvpTemplateConfig::default()
        };
        let template = build_ivp_task_template(&config);
        assert!(template.contains("tolerance: 0.000001"));
        assert!(template.contains("step_size: 0.001"));
        assert!(!template.contains("max_step:"));
        assert_template_passes_cli_preflight(&template);
        parse_ivp_task_from_str(&template).expect("Backward Euler template must parse");
    }
}

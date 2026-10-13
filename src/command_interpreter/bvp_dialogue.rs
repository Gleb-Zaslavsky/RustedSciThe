use crate::command_interpreter::task_wizard::{self, Question};
use crate::numerical::BVP_sci::new::BvpSciExecutionPolicy;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BvpTemplateStrategy {
    Damped,
    Frozen,
    Naive,
}

impl BvpTemplateStrategy {
    fn task_value(self) -> &'static str {
        match self {
            Self::Damped => "Damped",
            Self::Frozen => "Frozen",
            Self::Naive => "Naive",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BvpDerivativeScheme {
    Forward,
    Trapezoid,
}

impl BvpDerivativeScheme {
    fn task_value(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::Trapezoid => "trapezoid",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BvpMatrixPath {
    Banded,
    Sparse,
    Dense,
}

impl BvpMatrixPath {
    fn task_value(self) -> &'static str {
        match self {
            Self::Banded => "Banded",
            Self::Sparse => "Sparse",
            Self::Dense => "Dense",
        }
    }

    fn matrix_backend_value(self) -> &'static str {
        match self {
            Self::Banded => "banded",
            Self::Sparse => "sparse",
            Self::Dense => "dense",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BvpSymbolicBackend {
    AtomView,
    ExprLegacy,
}

impl BvpSymbolicBackend {
    fn task_value(self) -> &'static str {
        match self {
            Self::AtomView => "atom_view",
            Self::ExprLegacy => "expr_legacy",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BvpExecutionRoute {
    Lambdify,
    AotBuildIfMissing,
    AotRequirePrebuilt,
    AotRebuildAlways,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BvpAotToolchain {
    CTcc,
    CGcc,
    Zig,
    Rust,
}

impl BvpAotToolchain {
    fn codegen_backend_value(self) -> &'static str {
        match self {
            Self::CTcc | Self::CGcc => "c",
            Self::Zig => "zig",
            Self::Rust => "rust",
        }
    }

    fn c_compiler_value(self) -> Option<&'static str> {
        match self {
            Self::CTcc => Some("tcc"),
            Self::CGcc => Some("gcc"),
            Self::Zig | Self::Rust => None,
        }
    }

    fn preset_suffix(self) -> Option<&'static str> {
        match self {
            Self::CTcc => Some("tcc"),
            Self::CGcc => Some("gcc"),
            Self::Zig => Some("zig"),
            Self::Rust => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BvpAotExecutionPolicy {
    Auto,
    Sequential,
}

impl BvpAotExecutionPolicy {
    fn task_value(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Sequential => "sequential",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BvpBandedLinearSolver {
    Auto,
    LapackFaithful,
    FaerSparse,
}

impl BvpBandedLinearSolver {
    fn task_value(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::LapackFaithful => "lapack",
            Self::FaerSparse => "faer_sparse_lu",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BvpTemplateConfig {
    strategy: BvpTemplateStrategy,
    scheme: BvpDerivativeScheme,
    matrix_path: BvpMatrixPath,
    symbolic_backend: BvpSymbolicBackend,
    execution_route: BvpExecutionRoute,
    aot_toolchain: BvpAotToolchain,
    aot_execution_policy: BvpAotExecutionPolicy,
    banded_linear_solver: BvpBandedLinearSolver,
    refinement_steps: usize,
}

impl Default for BvpTemplateConfig {
    fn default() -> Self {
        Self {
            strategy: BvpTemplateStrategy::Damped,
            scheme: BvpDerivativeScheme::Forward,
            matrix_path: BvpMatrixPath::Banded,
            symbolic_backend: BvpSymbolicBackend::AtomView,
            execution_route: BvpExecutionRoute::Lambdify,
            aot_toolchain: BvpAotToolchain::CTcc,
            aot_execution_policy: BvpAotExecutionPolicy::Auto,
            banded_linear_solver: BvpBandedLinearSolver::LapackFaithful,
            refinement_steps: 0,
        }
    }
}

fn generated_backend_preset(config: &BvpTemplateConfig) -> String {
    let matrix = match config.matrix_path {
        BvpMatrixPath::Banded => "banded",
        BvpMatrixPath::Sparse | BvpMatrixPath::Dense => "sparse",
    };

    match config.execution_route {
        BvpExecutionRoute::Lambdify => format!("{matrix}_lambdify"),
        BvpExecutionRoute::AotBuildIfMissing
        | BvpExecutionRoute::AotRequirePrebuilt
        | BvpExecutionRoute::AotRebuildAlways => match config.aot_toolchain.preset_suffix() {
            Some(suffix) => format!("{matrix}_aot_{suffix}"),
            None => format!("{matrix}_aot"),
        },
    }
}

fn backend_policy_value(route: BvpExecutionRoute) -> &'static str {
    match route {
        BvpExecutionRoute::Lambdify => "lambdify_only",
        BvpExecutionRoute::AotBuildIfMissing => "prefer_aot_then_lambdify",
        BvpExecutionRoute::AotRequirePrebuilt | BvpExecutionRoute::AotRebuildAlways => "aot_only",
    }
}

fn aot_build_policy_value(route: BvpExecutionRoute) -> Option<&'static str> {
    match route {
        BvpExecutionRoute::Lambdify => None,
        BvpExecutionRoute::AotBuildIfMissing => Some("build_if_missing"),
        BvpExecutionRoute::AotRequirePrebuilt => Some("require_prebuilt"),
        BvpExecutionRoute::AotRebuildAlways => Some("rebuild_always"),
    }
}

fn build_bvp_task_template(config: &BvpTemplateConfig) -> String {
    let mut template = String::new();
    template.push_str("task\n");
    template.push_str("solver: BVP\n");
    template.push_str(&format!("strategy: {}\n", config.strategy.task_value()));
    template.push_str(&format!("scheme: {}\n", config.scheme.task_value()));
    template.push_str(&format!("method: {}\n\n", config.matrix_path.task_value()));

    template.push_str("equations\n");
    template.push_str("arg: x\n");
    template.push_str("unknowns: y, z\n");
    template.push_str("rhs: z, -y\n\n");

    template.push_str("boundary_conditions\n");
    template.push_str("y_left: 0.0\n");
    template.push_str("y_right: 1.0\n\n");

    template.push_str("mesh\n");
    template.push_str("t0: 0.0\n");
    template.push_str("t_end: 1.5707963267948966\n");
    template.push_str("n_steps: 100\n\n");

    template.push_str("initial_guess\n");
    template.push_str("y: 0.5\n");
    template.push_str("z: 0.5\n\n");

    template.push_str("solver_options\n");
    template.push_str("tolerance: 1e-8\n");
    template.push_str("max_iterations: 40\n");
    template.push_str("loglevel: warn\n");
    template.push_str(&format!(
        "generated_backend: {}\n",
        generated_backend_preset(config)
    ));
    template.push_str(&format!(
        "matrix_backend: {}\n",
        config.matrix_path.matrix_backend_value()
    ));
    template.push_str(&format!(
        "backend_policy: {}\n",
        backend_policy_value(config.execution_route)
    ));
    template.push_str(&format!(
        "symbolic_backend: {}\n",
        config.symbolic_backend.task_value()
    ));

    if let Some(build_policy) = aot_build_policy_value(config.execution_route) {
        template.push_str(&format!(
            "aot_codegen_backend: {}\n",
            config.aot_toolchain.codegen_backend_value()
        ));
        if let Some(c_compiler) = config.aot_toolchain.c_compiler_value() {
            template.push_str(&format!("aot_c_compiler: {c_compiler}\n"));
        }
        template.push_str(&format!("aot_build_policy: {build_policy}\n"));
        template.push_str("aot_build_profile: release\n");
        template.push_str("aot_compile_preset: dev_fastest\n");
        template.push_str(&format!(
            "aot_execution_policy: {}\n",
            config.aot_execution_policy.task_value()
        ));
    }

    if matches!(config.matrix_path, BvpMatrixPath::Banded) {
        template.push_str(&format!(
            "banded_linear_solver: {}\n",
            config.banded_linear_solver.task_value()
        ));
        template.push_str(&format!("refinement_steps: {}\n", config.refinement_steps));
    }

    template.push_str("\npostprocessing\n");
    template.push_str("save_csv: true\n");
    template.push_str("csv_path: bvp_result.csv\n");
    template.push_str("save_txt: false\n");
    template.push_str("write_report: false\n");
    template.push_str("plotters_png: false\n");
    template.push_str("gnuplot_png: false\n");
    template.push_str("terminal_plot: false\n");
    template.push_str("plot: false\n");
    template
}

fn build_bvp_sci_task_template(
    layout: BvpMatrixPath,
    frontend: BvpSymbolicBackend,
    route: BvpExecutionRoute,
    toolchain: BvpAotToolchain,
    execution_policy: BvpSciExecutionPolicy,
    sparse_factorization: &str,
) -> String {
    let mut task = String::from("task\nsolver: BVP_sci\n");
    task.push_str(&format!("frontend: {}\n", frontend.task_value()));
    task.push_str(&format!(
        "execution: {}\n",
        if matches!(route, BvpExecutionRoute::Lambdify) {
            "Lambdify"
        } else {
            "AOT"
        }
    ));
    task.push_str(&format!("method: {}\n\n", layout.task_value()));
    task.push_str("equations\narg: x\nunknowns: y, z\nrhs: z, -y\n\n");
    task.push_str("boundary_conditions\ny_left: 0.0\nz_right: 0.0\n\n");
    task.push_str("mesh\nt0: 0.0\nt_end: 1.5707963267948966\nn_steps: 32\n\n");
    task.push_str("initial_guess\ny: 0.5\nz: 0.5\n\n");
    task.push_str("solver_options\ntolerance: 1e-3\nboundary_tolerance: 1e-3\n");
    task.push_str("max_nodes: 1024\nmax_newton_iterations: 8\n");
    task.push_str("max_jacobian_refreshes: 4\nmax_mesh_refinements: 10\n");
    task.push_str("max_backtracking_steps: 4\ntelemetry: off\noutput_policy: mesh_and_values\n");
    let execution_policy_text = match execution_policy {
        BvpSciExecutionPolicy::Sequential => "sequential",
        BvpSciExecutionPolicy::Parallel { .. } => "parallel",
        BvpSciExecutionPolicy::Auto { .. } => "auto",
    };
    task.push_str(&format!("execution_policy: {execution_policy_text}\n"));
    if matches!(layout, BvpMatrixPath::Banded) {
        task.push_str("banded_lower: 2\nbanded_upper: 2\n");
    }
    if matches!(layout, BvpMatrixPath::Sparse) {
        task.push_str(&format!("sparse_factorization: {sparse_factorization}\n"));
    }
    if !matches!(route, BvpExecutionRoute::Lambdify) {
        task.push_str(&format!(
            "aot_codegen_backend: {}\n",
            toolchain.codegen_backend_value()
        ));
        if let Some(compiler) = toolchain.c_compiler_value() {
            task.push_str(&format!("aot_c_compiler: {compiler}\n"));
        }
        task.push_str(&format!(
            "aot_build_policy: {}\n",
            aot_build_policy_value(route).unwrap_or("build_if_missing")
        ));
        task.push_str("aot_build_profile: release\naot_output_dir: generated/bvp_sci\n");
    }
    task.push_str("\npostprocessing\noutput_policy: none\nplot: false\n");
    task
}

/// Write a non-interactive BVP_sci starter document for CLI automation.
pub fn write_bvp_sci_task_template(path: Option<PathBuf>) -> std::io::Result<PathBuf> {
    let path = path.unwrap_or_else(|| PathBuf::from("bvp_sci_task_template.txt"));
    std::fs::write(
        &path,
        build_bvp_sci_task_template(
            BvpMatrixPath::Sparse,
            BvpSymbolicBackend::AtomView,
            BvpExecutionRoute::Lambdify,
            BvpAotToolchain::CTcc,
            BvpSciExecutionPolicy::Auto { min_work: 0 },
            "standard_faer",
        ),
    )?;
    Ok(path)
}

pub fn run_bvp_template_dialogue(output_path: Option<PathBuf>) -> std::io::Result<()> {
    run_bvp_template_dialogue_for_family(output_path, None)
}

/// Open the retained-answer BVP_Damp branch without asking for the solver
/// family a second time. The generic family dialogue remains available for
/// callers that intentionally want to choose between BVP families.
pub fn run_bvp_damp_template_dialogue(output_path: Option<PathBuf>) -> std::io::Result<()> {
    run_bvp_template_dialogue_for_family(output_path, Some(0))
}

/// Writes the deterministic default BVP_Damp task document without opening a
/// terminal dialogue. Interactive customization belongs to `wizard`.
pub fn write_bvp_task_template(path: Option<PathBuf>) -> std::io::Result<PathBuf> {
    let path = path.unwrap_or_else(|| PathBuf::from("bvp_task_template.txt"));
    std::fs::write(&path, build_bvp_task_template(&BvpTemplateConfig::default()))?;
    Ok(path)
}

fn run_bvp_template_dialogue_for_family(
    output_path: Option<PathBuf>,
    fixed_family: Option<usize>,
) -> std::io::Result<()> {
    let Some(answers) = task_wizard::run("--- BVP task document template wizard ---", |answers| {
        let mut questions = Vec::new();
        if fixed_family.is_none() {
            questions.push(Question {
                key: "family",
                prompt: "Choose BVP solver family",
                options: &[
                    "BVP_Damp (damped/frozen solver)",
                    "BVP_sci (SciPy-like collocation solver)",
                ],
                default: 0,
            });
        }
        if fixed_family.unwrap_or_else(|| answers.get("family").copied().unwrap_or(0)) == 0 {
            questions.extend([
                Question {
                    key: "strategy",
                    prompt: "Choose Newton strategy",
                    options: &[
                        "Damped Newton (recommended general route)",
                        "Frozen Newton",
                        "Naive Newton",
                    ],
                    default: 0,
                },
                Question {
                    key: "scheme",
                    prompt: "Choose finite-difference derivative scheme",
                    options: &["forward", "trapezoid"],
                    default: 0,
                },
                Question {
                    key: "matrix",
                    prompt: "Choose matrix backend",
                    options: &[
                        "Banded (recommended for mesh BVPs)",
                        "Sparse",
                        "Dense (small/debug problems)",
                    ],
                    default: 0,
                },
                Question {
                    key: "symbolic",
                    prompt: "Choose symbolic assembly backend",
                    options: &["AtomView (recommended)", "ExprLegacy (compatibility path)"],
                    default: 0,
                },
                Question {
                    key: "route",
                    prompt: "Choose symbolic execution route",
                    options: &[
                        "Lambdify",
                        "AOT BuildIfMissing",
                        "AOT RequirePrebuilt",
                        "AOT RebuildAlways",
                    ],
                    default: 0,
                },
            ]);
            if answers.get("route").copied().unwrap_or(0) != 0 {
                questions.extend([
                    Question {
                        key: "toolchain",
                        prompt: "Choose AOT toolchain",
                        options: &["c_tcc", "c_gcc", "zig", "rust"],
                        default: 0,
                    },
                    Question {
                        key: "aot_policy",
                        prompt: "Choose AOT execution policy",
                        options: &["auto", "sequential"],
                        default: 0,
                    },
                ]);
            }
            if answers.get("matrix").copied().unwrap_or(0) == 0 {
                questions.push(Question {
                    key: "banded_solver",
                    prompt: "Choose banded linear solver",
                    options: &["lapack (faithful banded LU)", "auto", "faer_sparse_lu"],
                    default: 0,
                });
            }
        } else {
            questions.extend([
                Question {
                    key: "layout",
                    prompt: "Choose native matrix layout",
                    options: &["Sparse", "Banded", "Dense"],
                    default: 0,
                },
                Question {
                    key: "frontend",
                    prompt: "Choose symbolic frontend",
                    options: &["AtomViewNative", "ExprLegacy"],
                    default: 0,
                },
                Question {
                    key: "route",
                    prompt: "Choose symbolic execution lifecycle",
                    options: &[
                        "Lambdify",
                        "AOT BuildIfMissing",
                        "AOT RequirePrebuilt",
                        "AOT RebuildAlways",
                    ],
                    default: 0,
                },
                Question {
                    key: "callback_policy",
                    prompt: "Choose callback execution policy",
                    options: &["sequential", "auto", "parallel"],
                    default: 0,
                },
            ]);
            if answers.get("layout").copied().unwrap_or(0) == 0 {
                questions.push(Question {
                    key: "sparse_factorization",
                    prompt: "Choose sparse factorization",
                    options: &[
                        "standard_faer",
                        "SuperLU (requires feature and native bridge)",
                    ],
                    default: 0,
                });
            }
            if answers.get("route").copied().unwrap_or(0) != 0 {
                questions.push(Question {
                    key: "toolchain",
                    prompt: "Choose AOT code-generation toolchain",
                    options: &["tcc", "gcc", "zig", "rust"],
                    default: 0,
                });
            }
        }
        questions
    }) else {
        println!("BVP template wizard cancelled; no file was written.");
        return Ok(());
    };

    let selected = |key| answers.get(key).copied().unwrap_or(0);
    let family = fixed_family.unwrap_or_else(|| selected("family"));
    let (template, default_name) = if family == 0 {
        let strategy = match selected("strategy") {
            0 => BvpTemplateStrategy::Damped,
            1 => BvpTemplateStrategy::Frozen,
            _ => BvpTemplateStrategy::Naive,
        };
        let scheme = if selected("scheme") == 0 {
            BvpDerivativeScheme::Forward
        } else {
            BvpDerivativeScheme::Trapezoid
        };
        let matrix_path = match selected("matrix") {
            0 => BvpMatrixPath::Banded,
            1 => BvpMatrixPath::Sparse,
            _ => BvpMatrixPath::Dense,
        };
        let symbolic_backend = if selected("symbolic") == 0 {
            BvpSymbolicBackend::AtomView
        } else {
            BvpSymbolicBackend::ExprLegacy
        };
        let execution_route = match selected("route") {
            0 => BvpExecutionRoute::Lambdify,
            1 => BvpExecutionRoute::AotBuildIfMissing,
            2 => BvpExecutionRoute::AotRequirePrebuilt,
            _ => BvpExecutionRoute::AotRebuildAlways,
        };
        let config = BvpTemplateConfig {
            strategy,
            scheme,
            matrix_path,
            symbolic_backend,
            execution_route,
            aot_toolchain: match selected("toolchain") {
                0 => BvpAotToolchain::CTcc,
                1 => BvpAotToolchain::CGcc,
                2 => BvpAotToolchain::Zig,
                _ => BvpAotToolchain::Rust,
            },
            aot_execution_policy: if selected("aot_policy") == 0 {
                BvpAotExecutionPolicy::Auto
            } else {
                BvpAotExecutionPolicy::Sequential
            },
            banded_linear_solver: match selected("banded_solver") {
                0 => BvpBandedLinearSolver::LapackFaithful,
                1 => BvpBandedLinearSolver::Auto,
                _ => BvpBandedLinearSolver::FaerSparse,
            },
            ..BvpTemplateConfig::default()
        };
        (build_bvp_task_template(&config), "bvp_task_custom.txt")
    } else {
        let layout = match selected("layout") {
            0 => BvpMatrixPath::Sparse,
            1 => BvpMatrixPath::Banded,
            _ => BvpMatrixPath::Dense,
        };
        let frontend = if selected("frontend") == 0 {
            BvpSymbolicBackend::AtomView
        } else {
            BvpSymbolicBackend::ExprLegacy
        };
        let route = match selected("route") {
            0 => BvpExecutionRoute::Lambdify,
            1 => BvpExecutionRoute::AotBuildIfMissing,
            2 => BvpExecutionRoute::AotRequirePrebuilt,
            _ => BvpExecutionRoute::AotRebuildAlways,
        };
        let policy = match selected("callback_policy") {
            0 => BvpSciExecutionPolicy::Sequential,
            1 => BvpSciExecutionPolicy::Auto { min_work: 0 },
            _ => BvpSciExecutionPolicy::Parallel { min_work: 0 },
        };
        let factorization =
            if layout == BvpMatrixPath::Sparse && selected("sparse_factorization") == 1 {
                "superlu"
            } else {
                "standard_faer"
            };
        let toolchain = match selected("toolchain") {
            0 => BvpAotToolchain::CTcc,
            1 => BvpAotToolchain::CGcc,
            2 => BvpAotToolchain::Zig,
            _ => BvpAotToolchain::Rust,
        };
        (
            build_bvp_sci_task_template(layout, frontend, route, toolchain, policy, factorization),
            "bvp_sci_task_custom.txt",
        )
    };
    let path = output_path.unwrap_or_else(|| PathBuf::from(default_name));
    match std::fs::write(&path, template) {
        Ok(()) => println!("BVP task template saved to {}", path.display()),
        Err(error) => return Err(error),
    }
    Ok(())
}

/// Open the interactive BVP_sci branch without asking for the solver family again.
pub fn run_bvp_sci_template_dialogue(output_path: Option<PathBuf>) -> std::io::Result<()> {
    run_bvp_template_dialogue_for_family(output_path, Some(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command_interpreter::task_parser_bvp::{
        parse_bvp_task_from_str, BvpLinearBackendSpec, BvpSolverFamilySpec, BvpStrategySpec,
    };
    use crate::command_interpreter::task_runner::{parse_task_spec_from_str, render_task_check};

    fn assert_template_passes_cli_preflight(template: &str) {
        let spec = parse_task_spec_from_str(template).expect("generated BVP should parse");
        let report = render_task_check(&spec);
        assert!(report.contains("preflight=valid;"));
    }

    #[test]
    fn bvp_dialogue_banded_lambdify_template_is_parser_ready() {
        let template = build_bvp_task_template(&BvpTemplateConfig::default());
        assert_template_passes_cli_preflight(&template);
        let spec = parse_bvp_task_from_str(&template).expect("BVP dialogue template should parse");

        assert_eq!(spec.solver.strategy, BvpStrategySpec::Damped);
        assert_eq!(spec.solver.backend, BvpLinearBackendSpec::Banded);
        assert_eq!(
            spec.solver_options.generated_backend.preset.as_deref(),
            Some("banded_lambdify")
        );
        assert_eq!(
            spec.solver_options
                .generated_backend
                .backend_policy
                .as_deref(),
            Some("lambdify_only")
        );
        assert!(
            spec.boundary_conditions
                .conditions
                .get("y")
                .expect("y boundary conditions should be present")
                .len()
                == 2
        );
    }

    #[test]
    fn bvp_sci_template_is_parser_ready_and_uses_native_defaults() {
        let template = build_bvp_sci_task_template(
            BvpMatrixPath::Sparse,
            BvpSymbolicBackend::AtomView,
            BvpExecutionRoute::Lambdify,
            BvpAotToolchain::CTcc,
            BvpSciExecutionPolicy::Sequential,
            "standard_faer",
        );
        assert_template_passes_cli_preflight(&template);
        let spec =
            parse_bvp_task_from_str(&template).expect("BVP_sci wizard template should parse");
        assert_eq!(spec.solver.family, BvpSolverFamilySpec::Sci);
        assert_eq!(spec.solver.backend, BvpLinearBackendSpec::Sparse);
        assert_eq!(
            spec.solver.frontend,
            Some(crate::command_interpreter::task_parser_bvp::BvpFrontendSpec::AtomViewNative)
        );
        assert!(spec.solver_options.max_nodes.is_some());
        assert!(spec.solver_options.max_newton_iterations.is_some());
    }

    #[test]
    fn noninteractive_bvp_sci_template_writer_creates_a_parseable_file() {
        let dir = tempfile::tempdir().expect("temporary directory should be created");
        let path = dir.path().join("bvp_sci.txt");
        let written = write_bvp_sci_task_template(Some(path.clone()))
            .expect("CLI template writer should create the task file");
        assert_eq!(written, path);
        let text = std::fs::read_to_string(&written).expect("generated task should be readable");
        assert_template_passes_cli_preflight(&text);
        let spec = parse_bvp_task_from_str(&text).expect("generated task should parse");
        assert_eq!(spec.solver.family, BvpSolverFamilySpec::Sci);
    }

    #[test]
    fn bvp_dialogue_banded_aot_tcc_template_is_parser_ready() {
        let config = BvpTemplateConfig {
            execution_route: BvpExecutionRoute::AotBuildIfMissing,
            aot_toolchain: BvpAotToolchain::CTcc,
            aot_execution_policy: BvpAotExecutionPolicy::Auto,
            ..BvpTemplateConfig::default()
        };
        let template = build_bvp_task_template(&config);
        assert_template_passes_cli_preflight(&template);
        let spec = parse_bvp_task_from_str(&template).expect("BVP AOT template should parse");

        assert_eq!(
            spec.solver_options.generated_backend.preset.as_deref(),
            Some("banded_aot_tcc")
        );
        assert_eq!(
            spec.solver_options
                .generated_backend
                .aot_build_policy
                .as_deref(),
            Some("build_if_missing")
        );
        assert_eq!(
            spec.solver_options
                .generated_backend
                .aot_execution_policy
                .as_deref(),
            Some("auto")
        );
    }

    #[test]
    fn bvp_dialogue_sparse_aot_zig_template_is_parser_ready() {
        let config = BvpTemplateConfig {
            matrix_path: BvpMatrixPath::Sparse,
            execution_route: BvpExecutionRoute::AotRequirePrebuilt,
            aot_toolchain: BvpAotToolchain::Zig,
            ..BvpTemplateConfig::default()
        };
        let template = build_bvp_task_template(&config);
        assert_template_passes_cli_preflight(&template);
        let spec = parse_bvp_task_from_str(&template).expect("Sparse AOT template should parse");

        assert_eq!(spec.solver.backend, BvpLinearBackendSpec::Sparse);
        assert_eq!(
            spec.solver_options.generated_backend.preset.as_deref(),
            Some("sparse_aot_zig")
        );
        assert_eq!(
            spec.solver_options
                .generated_backend
                .backend_policy
                .as_deref(),
            Some("aot_only")
        );
        assert_eq!(
            spec.solver_options
                .generated_backend
                .aot_build_policy
                .as_deref(),
            Some("require_prebuilt")
        );
    }
}

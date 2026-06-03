use dialoguer::{Confirm, Select, theme::ColorfulTheme};
use std::fs::File;
use std::io::Write;
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

pub fn run_bvp_template_dialogue(output_path: Option<PathBuf>) {
    let theme = ColorfulTheme::default();

    println!("--- BVP task document template wizard ---");
    println!(
        "This wizard writes a parser-ready task document for Damped/Frozen BVP solvers. It exposes symbolic Lambdify/AOT task-doc options; pure numerical closures are configured through the Rust API, not task documents.\n"
    );

    let strategy_items = [
        "Damped Newton (recommended general route)",
        "Frozen Newton",
        "Naive Newton",
    ];
    let strategy_idx = Select::with_theme(&theme)
        .with_prompt("Choose Newton strategy")
        .items(&strategy_items)
        .default(0)
        .interact()
        .unwrap();
    let strategy = match strategy_idx {
        0 => BvpTemplateStrategy::Damped,
        1 => BvpTemplateStrategy::Frozen,
        _ => BvpTemplateStrategy::Naive,
    };

    let scheme_items = ["forward", "trapezoid"];
    let scheme_idx = Select::with_theme(&theme)
        .with_prompt("Choose finite-difference derivative scheme")
        .items(&scheme_items)
        .default(0)
        .interact()
        .unwrap();
    let scheme = if scheme_idx == 0 {
        BvpDerivativeScheme::Forward
    } else {
        BvpDerivativeScheme::Trapezoid
    };

    let matrix_items = [
        "Banded (recommended for mesh BVPs)",
        "Sparse",
        "Dense (small/debug problems)",
    ];
    let matrix_idx = Select::with_theme(&theme)
        .with_prompt("Choose matrix backend")
        .items(&matrix_items)
        .default(0)
        .interact()
        .unwrap();
    let matrix_path = match matrix_idx {
        0 => BvpMatrixPath::Banded,
        1 => BvpMatrixPath::Sparse,
        _ => BvpMatrixPath::Dense,
    };

    let symbolic_items = ["AtomView (recommended)", "ExprLegacy (compatibility path)"];
    let symbolic_idx = Select::with_theme(&theme)
        .with_prompt("Choose symbolic assembly backend")
        .items(&symbolic_items)
        .default(0)
        .interact()
        .unwrap();
    let symbolic_backend = if symbolic_idx == 0 {
        BvpSymbolicBackend::AtomView
    } else {
        BvpSymbolicBackend::ExprLegacy
    };

    let route_items = [
        "Lambdify (no external compiler)",
        "AOT BuildIfMissing",
        "AOT RequirePrebuilt",
        "AOT RebuildAlways",
    ];
    let route_idx = Select::with_theme(&theme)
        .with_prompt("Choose symbolic execution route")
        .items(&route_items)
        .default(0)
        .interact()
        .unwrap();
    let execution_route = match route_idx {
        0 => BvpExecutionRoute::Lambdify,
        1 => BvpExecutionRoute::AotBuildIfMissing,
        2 => BvpExecutionRoute::AotRequirePrebuilt,
        _ => BvpExecutionRoute::AotRebuildAlways,
    };

    let mut config = BvpTemplateConfig {
        strategy,
        scheme,
        matrix_path,
        symbolic_backend,
        execution_route,
        ..BvpTemplateConfig::default()
    };

    if !matches!(execution_route, BvpExecutionRoute::Lambdify) {
        let toolchain_items = [
            "c_tcc (fast cold build, strong default)",
            "c_gcc",
            "zig",
            "rust",
        ];
        let toolchain_idx = Select::with_theme(&theme)
            .with_prompt("Choose AOT toolchain")
            .items(&toolchain_items)
            .default(0)
            .interact()
            .unwrap();
        config.aot_toolchain = match toolchain_idx {
            0 => BvpAotToolchain::CTcc,
            1 => BvpAotToolchain::CGcc,
            2 => BvpAotToolchain::Zig,
            _ => BvpAotToolchain::Rust,
        };

        println!(
            "Task documents currently expose AOT execution as `auto` or `sequential`. Full custom parallel executor configs are available through the Rust API."
        );
        let exec_policy_items = ["auto", "sequential"];
        let exec_policy_idx = Select::with_theme(&theme)
            .with_prompt("Choose AOT execution policy")
            .items(&exec_policy_items)
            .default(0)
            .interact()
            .unwrap();
        config.aot_execution_policy = if exec_policy_idx == 0 {
            BvpAotExecutionPolicy::Auto
        } else {
            BvpAotExecutionPolicy::Sequential
        };
    }

    if matches!(matrix_path, BvpMatrixPath::Banded) {
        let solver_items = [
            "lapack (faithful LAPACK-style banded LU, recommended)",
            "auto",
            "faer_sparse_lu",
        ];
        let solver_idx = Select::with_theme(&theme)
            .with_prompt("Choose banded linear solver")
            .items(&solver_items)
            .default(0)
            .interact()
            .unwrap();
        config.banded_linear_solver = match solver_idx {
            0 => BvpBandedLinearSolver::LapackFaithful,
            1 => BvpBandedLinearSolver::Auto,
            _ => BvpBandedLinearSolver::FaerSparse,
        };
    }

    let template = build_bvp_task_template(&config);
    let final_path = output_path.unwrap_or_else(|| PathBuf::from("bvp_task_custom.txt"));

    println!("\nConfiguration is ready.");
    if Confirm::with_theme(&theme)
        .with_prompt(format!(
            "Write the BVP task template to {}?",
            final_path.display()
        ))
        .default(true)
        .interact()
        .unwrap()
    {
        let mut file = File::create(&final_path).expect("failed to create BVP task file");
        file.write_all(template.as_bytes())
            .expect("failed to write BVP task file");
        println!("BVP task template saved to {}", final_path.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command_interpreter::task_parser_bvp::{
        BvpLinearBackendSpec, BvpStrategySpec, parse_bvp_task_from_str,
    };

    #[test]
    fn bvp_dialogue_banded_lambdify_template_is_parser_ready() {
        let template = build_bvp_task_template(&BvpTemplateConfig::default());
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
    fn bvp_dialogue_banded_aot_tcc_template_is_parser_ready() {
        let config = BvpTemplateConfig {
            execution_route: BvpExecutionRoute::AotBuildIfMissing,
            aot_toolchain: BvpAotToolchain::CTcc,
            aot_execution_policy: BvpAotExecutionPolicy::Auto,
            ..BvpTemplateConfig::default()
        };
        let template = build_bvp_task_template(&config);
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

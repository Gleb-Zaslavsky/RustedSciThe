use dialoguer::{Confirm, Select, theme::ColorfulTheme};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IvpTemplateMethod {
    LsodaAuto,
    LsodeAdams,
    LsodeBdf,
    Bdf,
    Radau5,
    BackwardEuler,
    Rk45,
    Rk4,
}

impl IvpTemplateMethod {
    fn task_value(self) -> &'static str {
        match self {
            Self::LsodaAuto => "LSODA",
            Self::LsodeAdams | Self::LsodeBdf => "LSODE",
            Self::Bdf => "BDF",
            Self::Radau5 => "Radau5",
            Self::BackwardEuler => "BackwardEuler",
            Self::Rk45 => "RK45",
            Self::Rk4 => "RK4",
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

impl IvpNativeExecution {
    fn task_value(self) -> &'static str {
        match self {
            Self::FaithfulBdfSolve => "faithful_bdf_solve",
            Self::ProbeBeforeBridge => "probe_before_bridge",
            Self::BridgeSolve => "bridge_solve",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct IvpTemplateConfig {
    method: IvpTemplateMethod,
    symbolic_assembly: IvpSymbolicAssembly,
    symbolic_execution: IvpSymbolicExecution,
    aot_toolchain: IvpAotToolchain,
    aot_profile: IvpAotProfile,
    linear_structure: IvpLinearStructure,
    linear_solver_policy: IvpLinearSolverPolicy,
    native_execution: IvpNativeExecution,
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
    template.push_str("rtol: 1e-6\n");
    template.push_str("atol: 1e-9\n");
    template.push_str("max_step: 1e-3\n");
    template.push_str("first_step: Some(1e-6)\n");
    template.push_str("parallel: false\n");

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

pub fn run_ivp_template_dialogue(output_path: Option<PathBuf>) {
    let theme = ColorfulTheme::default();

    println!("--- IVP task document template wizard ---");
    println!(
        "This wizard writes a parser-ready task document for src/main.rs. It only exposes fields currently supported by the IVP task parser.\n"
    );

    let method_items = [
        "LSODA-style automatic Adams/BDF switching (recommended for mixed IVPs)",
        "LSODE fixed Adams family (non-stiff IVPs)",
        "LSODE fixed BDF family (stiff IVPs)",
        "Native BDF",
        "Radau5",
        "Backward Euler",
        "RK45",
        "RK4",
    ];
    let method_idx = Select::with_theme(&theme)
        .with_prompt("Choose the IVP method")
        .items(&method_items)
        .default(0)
        .interact()
        .unwrap();
    let method = match method_idx {
        0 => IvpTemplateMethod::LsodaAuto,
        1 => IvpTemplateMethod::LsodeAdams,
        2 => IvpTemplateMethod::LsodeBdf,
        3 => IvpTemplateMethod::Bdf,
        4 => IvpTemplateMethod::Radau5,
        5 => IvpTemplateMethod::BackwardEuler,
        6 => IvpTemplateMethod::Rk45,
        _ => IvpTemplateMethod::Rk4,
    };

    let mut config = IvpTemplateConfig {
        method,
        ..IvpTemplateConfig::default()
    };

    if method.is_lsode2() {
        println!("\nLSODE2 symbolic frontend");
        let assembly_items = [
            "AtomView (recommended for large symbolic systems)",
            "ExprLegacy (compatibility path)",
        ];
        let assembly_idx = Select::with_theme(&theme)
            .with_prompt("Choose symbolic assembly")
            .items(&assembly_items)
            .default(0)
            .interact()
            .unwrap();
        config.symbolic_assembly = if assembly_idx == 0 {
            IvpSymbolicAssembly::AtomView
        } else {
            IvpSymbolicAssembly::ExprLegacy
        };

        let execution_items = [
            "LambdifyExpr (fast startup, no external compiler)",
            "AOT (compiled generated callback)",
        ];
        let execution_idx = Select::with_theme(&theme)
            .with_prompt("Choose symbolic execution")
            .items(&execution_items)
            .default(0)
            .interact()
            .unwrap();
        config.symbolic_execution = if execution_idx == 0 {
            IvpSymbolicExecution::LambdifyExpr
        } else {
            IvpSymbolicExecution::Aot
        };

        if matches!(config.symbolic_execution, IvpSymbolicExecution::Aot) {
            let toolchain_items = [
                "c_tcc (fast cold build, good default for experiments)",
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
                0 => IvpAotToolchain::CTcc,
                1 => IvpAotToolchain::CGcc,
                2 => IvpAotToolchain::Zig,
                _ => IvpAotToolchain::Rust,
            };

            let profile_items = ["release", "debug"];
            let profile_idx = Select::with_theme(&theme)
                .with_prompt("Choose AOT build profile")
                .items(&profile_items)
                .default(0)
                .interact()
                .unwrap();
            config.aot_profile = if profile_idx == 0 {
                IvpAotProfile::Release
            } else {
                IvpAotProfile::Debug
            };
        }

        println!("\nLSODE2 linear algebra");
        println!(
            "For `banded` task documents the dialogue does not ask for bandwidth. The parser accepts optional lsode2_banded_kl/ku for advanced fixed-width runs, but normal templates should let the LSODE2 route choose the configured banded path."
        );
        let structure_items = ["banded", "sparse", "dense"];
        let structure_idx = Select::with_theme(&theme)
            .with_prompt("Choose linear system structure")
            .items(&structure_items)
            .default(0)
            .interact()
            .unwrap();
        config.linear_structure = match structure_idx {
            0 => IvpLinearStructure::Banded,
            1 => IvpLinearStructure::Sparse,
            _ => IvpLinearStructure::Dense,
        };

        let default_policy = config.linear_structure.default_policy();
        let policy_items = [
            "auto",
            "dense_lu",
            "faer_sparse_lu",
            "lapack_faithful_banded_lu",
        ];
        let default_policy_idx = match default_policy {
            IvpLinearSolverPolicy::Auto => 0,
            IvpLinearSolverPolicy::DenseLu => 1,
            IvpLinearSolverPolicy::FaerSparseLu => 2,
            IvpLinearSolverPolicy::LapackFaithfulBandedLu => 3,
        };
        let policy_idx = Select::with_theme(&theme)
            .with_prompt("Choose linear solver policy")
            .items(&policy_items)
            .default(default_policy_idx)
            .interact()
            .unwrap();
        config.linear_solver_policy = match policy_idx {
            0 => IvpLinearSolverPolicy::Auto,
            1 => IvpLinearSolverPolicy::DenseLu,
            2 => IvpLinearSolverPolicy::FaerSparseLu,
            _ => IvpLinearSolverPolicy::LapackFaithfulBandedLu,
        };

        if method.needs_bdf_execution_choice() {
            let native_items = [
                "faithful_bdf_solve (native faithful path, recommended)",
                "probe_before_bridge",
                "bridge_solve",
            ];
            let native_idx = Select::with_theme(&theme)
                .with_prompt("Choose BDF execution mode")
                .items(&native_items)
                .default(0)
                .interact()
                .unwrap();
            config.native_execution = match native_idx {
                0 => IvpNativeExecution::FaithfulBdfSolve,
                1 => IvpNativeExecution::ProbeBeforeBridge,
                _ => IvpNativeExecution::BridgeSolve,
            };
        }
    }

    let template = build_ivp_task_template(&config);
    let final_path = output_path.unwrap_or_else(|| PathBuf::from("ivp_task_custom.txt"));

    println!("\nConfiguration is ready.");
    if Confirm::with_theme(&theme)
        .with_prompt(format!(
            "Write the IVP task template to {}?",
            final_path.display()
        ))
        .default(true)
        .interact()
        .unwrap()
    {
        let mut file = File::create(&final_path).expect("failed to create IVP task file");
        file.write_all(template.as_bytes())
            .expect("failed to write IVP task file");
        println!("IVP task template saved to {}", final_path.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command_interpreter::task_parser_ivp::{
        IvpMethodSpec, Lsode2TaskExecutionSpec, parse_ivp_task_from_str,
    };
    use crate::numerical::LSODE2::{
        Lsode2AotProfile, Lsode2AotToolchain, Lsode2ControllerMode, Lsode2LinearSolverChoice,
        Lsode2LinearSolverPolicy, Lsode2LinearSystemStructure,
    };

    #[test]
    fn ivp_dialogue_lsoda_auto_lambdify_template_is_parser_ready() {
        let template = build_ivp_task_template(&IvpTemplateConfig::default());
        let spec = parse_ivp_task_from_str(&template).expect("dialogue template should parse");

        assert_eq!(spec.solver.method, IvpMethodSpec::Lsode2);
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
            Some(Lsode2LinearSystemStructure::Banded { kl: 0, ku: 0 })
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
        assert!(!template.contains("lsode2_"));
        let spec = parse_ivp_task_from_str(&template).expect("Radau template should parse");
        assert_eq!(spec.solver.method, IvpMethodSpec::Radau5);
    }
}

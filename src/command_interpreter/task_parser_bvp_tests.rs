use super::*;
use std::fs;
use tempfile::tempdir;

fn parse_document_for_bvp(input: &str) -> DocumentMap {
    let mut parser = DocumentParser::new(input.to_string());

    let pseudonyms = default_bvp_pseudonyms();

    parser.with_pseudonims(Some(pseudonyms.0), Some(pseudonyms.1));

    parser.parse_document().expect("BVP document should parse");

    parser.keys_to_lower_case(Some(vec![
        "equations".to_string(),
        "boundary_conditions".to_string(),
        "initial_guess".to_string(),
        "where".to_string(),
        "substitute".to_string(),
    ]));

    parser
        .get_result()
        .expect("BVP document map should exist")
        .clone()
}

#[test]
fn bvp_task_parser_supports_file_roundtrip() {
    let dir = tempdir().expect("temp dir should be created");
    let path = dir.path().join("bvp_task.txt");
    fs::write(
        &path,
        r#"
task
solver: BVP
strategy: Damped
scheme: forward
method: Sparse

equations
arg: x
unknowns: y
rhs: -y

boundary_conditions
y_left: 1.0

mesh
t0: 0.0
t_end: 1.0
n_steps: 20

initial_guess
y: 0.0
        "#,
    )
    .expect("test task file should be written");

    let spec = parse_bvp_task_from_file(Some(path)).expect("BVP file task should parse");
    assert_eq!(spec.solver.strategy, BvpStrategySpec::Damped);
    assert_eq!(spec.equations.unknowns, vec!["y".to_string()]);
    assert_eq!(spec.mesh.n_steps, 20);
}

#[test]
fn bvp_task_parser_supports_pair_style_equations() {
    let input = r#"

task

solver: BVP

strategy: Damped

scheme: forward

method: Sparse



equations

arg: x

y: -2.0*y



boundary_conditions

y_left: 1.0



mesh

t0: 0.0

t_end: 1.0

n_steps: 20



initial_guess

y: 0.0

"#;

    let spec = parse_bvp_task_from_str(input).expect("pair-style BVP task should parse");

    assert_eq!(spec.equations.unknowns, vec!["y".to_string()]);

    assert_eq!(spec.mesh.n_steps, 20);

    assert_eq!(
        spec.boundary_conditions.conditions["y"],
        vec![(0usize, 1.0f64)]
    );
}

#[test]

fn bvp_task_parser_can_split_problem_and_solver_settings() {
    let input = r#"

task

solver: BVP

strategy: Damped

scheme: forward

method: Banded



equations

arg: x

unknowns: y

rhs: -y



boundary_conditions

y_left: 1.0



mesh

t0: 0.0

t_end: 1.0

n_steps: 12



initial_guess

y: 0.0



solver_options

tolerance: 1e-6

max_iterations: 24

generated_backend: banded_aot_tcc

aot_build_policy: build_if_missing

aot_build_profile: debug

aot_execution_policy: sequential

banded_linear_solver: faithful

refinement_steps: 2

"#;

    let document = parse_document_for_bvp(input);

    let problem = parse_bvp_problem_from_document(&document)
        .expect("problem section should be parsed independently");

    let settings = parse_bvp_solver_settings_from_document(&document)
        .expect("solver settings should be parsed independently");

    assert_eq!(problem.equations.unknowns, vec!["y".to_string()]);

    assert_eq!(problem.mesh.n_steps, 12);

    assert_eq!(settings.solver.strategy, BvpStrategySpec::Damped);

    assert_eq!(settings.solver.backend, BvpLinearBackendSpec::Banded);

    assert_eq!(settings.solver_options.tolerance, Some(1e-6));

    assert_eq!(settings.solver_options.max_iterations, Some(24));

    assert_eq!(
        settings.solver_options.generated_backend.preset.as_deref(),
        Some("banded_aot_tcc")
    );

    let solver = build_bvp_solver_from_problem_and_settings(&problem, &settings)
        .expect("split problem/settings should build a solver");

    drop(solver);
}

#[test]

fn bvp_task_parser_reads_solver_settings_without_problem_sections() {
    let input = r#"

task

solver: BVP

strategy: Frozen

scheme: forward

method: Sparse



solver_options

generated_backend: sparse_lambdify

tolerance: 1e-4

max_iterations: 10

"#;

    let document = parse_document_for_bvp(input);

    let settings = parse_bvp_solver_settings_from_document(&document)
        .expect("solver-only settings should parse without equations");

    assert_eq!(settings.solver.strategy, BvpStrategySpec::Frozen);

    assert_eq!(settings.solver.backend, BvpLinearBackendSpec::Sparse);

    assert_eq!(
        settings.solver_options.generated_backend.preset.as_deref(),
        Some("sparse_lambdify")
    );

    assert_eq!(settings.solver_options.tolerance, Some(1e-4));

    assert_eq!(settings.solver_options.max_iterations, Some(10));
}

#[test]

fn bvp_task_parser_supports_full_end_to_end_document_with_aot_settings() {
    let input = r#"

task

solver: BVP

strategy: Damped

scheme: forward

method: Banded



equations

arg: x

unknowns: z, y

rhs: y-z, -z^3



boundary_conditions

z_left: 1.0

y_right: 1.0



mesh

t0: 0.0

t_end: 1.0

n_steps: 20



initial_guess

guess: 0.0, 0.0



solver_options

tolerance: 1e-5

max_iterations: 20

generated_backend: banded_aot_tcc

aot_build_policy: build_if_missing

aot_build_profile: debug

aot_execution_policy: sequential

banded_linear_solver: faithful

refinement_steps: 1



postprocessing

save_csv: false

save_txt: false

write_report: false

plot: false

"#;

    let spec = parse_bvp_task_from_str(input)
        .expect("full end-to-end BVP document with AOT settings should parse");

    assert_eq!(spec.solver.backend, BvpLinearBackendSpec::Banded);

    assert_eq!(
        spec.equations.unknowns,
        vec!["z".to_string(), "y".to_string()]
    );

    assert_eq!(spec.mesh.n_steps, 20);

    assert_eq!(
        spec.solver_options.generated_backend.preset.as_deref(),
        Some("banded_aot_tcc")
    );

    let solver = build_bvp_solver_from_spec(&spec)
        .expect("full end-to-end BVP document with AOT settings should build");

    drop(solver);
}

#[test]

fn bvp_task_parser_can_build_solver_from_rust_problem_and_task_doc_settings() {
    let problem = BvpProblemSpec {
        equations: BvpEquationSpec {
            arg: "x".to_string(),

            unknowns: vec!["y".to_string()],

            rhs: vec![Expr::parse_expression("-y")],

            parameter_names: vec![],

            parameter_values: HashMap::new(),
        },

        boundary_conditions: BoundaryConditionSpec {
            conditions: HashMap::from([("y".to_string(), vec![(0usize, 1.0f64)])]),
        },

        mesh: BvpMeshSpec {
            t0: 0.0,

            t_end: 1.0,

            n_steps: 16,
        },

        initial_guess: BvpInitialGuessSpec { values: vec![0.0] },
    };

    let settings_doc = r#"

task

solver: BVP

strategy: Frozen

scheme: forward

method: Sparse



solver_options

generated_backend: sparse_aot_tcc

aot_build_policy: require_prebuilt

aot_build_profile: release

aot_execution_policy: sequential

tolerance: 1e-4

max_iterations: 12

"#;

    let document = parse_document_for_bvp(settings_doc);

    let settings = parse_bvp_solver_settings_from_document(&document)
        .expect("solver settings should parse independently of problem sections");

    assert_eq!(settings.solver.strategy, BvpStrategySpec::Frozen);

    assert_eq!(settings.solver.backend, BvpLinearBackendSpec::Sparse);

    assert_eq!(
        settings.solver_options.generated_backend.preset.as_deref(),
        Some("sparse_aot_tcc")
    );

    let solver = build_bvp_solver_from_problem_and_settings(&problem, &settings)
        .expect("Rust problem plus task-doc settings should build");

    drop(solver);
}

#[test]

fn bvp_task_parser_supports_where_symbolic_substitutions() {
    let input = r#"

task

solver: BVP

strategy: Damped

scheme: forward

method: Sparse



equations

arg: x

y: flux - y



where

flux: 3.0*x



boundary_conditions

y_left: 1.0



mesh

t0: 0.0

t_end: 1.0

n_steps: 8



initial_guess

y: 0.0

"#;

    let spec = parse_bvp_task_from_str(input).expect("BVP with where substitutions should parse");

    let expr = spec.equations.rhs[0].clone();

    let f = expr.lambdify_borrowed_thread_safe(&["x", "y"]);

    let value = f(&[2.0, 1.0]);

    assert!((value - 5.0).abs() < 1e-12);
}

#[test]
fn bvp_task_parser_resolves_multilevel_where_with_numeric_parameters() {
    let input = r#"
task
solver: BVP
strategy: Damped
scheme: forward
method: Sparse

equations
arg: x
parameters: gain
parameter_values: 2.0
y: source - y

where
base: gain * x
shifted: base + 1.0
source: 2.0 * shifted

boundary_conditions
y_left: 1.0

mesh
t0: 0.0
t_end: 1.0
n_steps: 8

initial_guess
y: 0.0
"#;

    let spec = parse_bvp_task_from_str(input)
        .expect("BVP parser should resolve a multilevel symbolic definition chain");
    let rhs = spec.equations.rhs[0].lambdify_borrowed_thread_safe(&["x", "y"]);

    // source = 2 * (gain * x + 1), so source - y = 13 at x=3, y=1.
    assert!((rhs(&[3.0, 1.0]) - 13.0).abs() < 1e-12);
}

#[test]

fn bvp_task_parser_accepts_params_alias_and_where_substitution() {
    let input = r#"

task

solver: BVP

strategy: Damped

scheme: forward

method: Sparse



equations

arg: x

params: a

parameter_values: 2.0

y: heat - a*y



where

heat: 1.0 + x



boundary_conditions

y_left: 1.0



mesh

t0: 0.0

t_end: 1.0

n_steps: 8



initial_guess

y: 0.0

"#;

    let spec = parse_bvp_task_from_str(input).expect("BVP should accept IVP-style params alias");

    assert_eq!(spec.equations.parameter_names, vec!["a".to_string()]);

    assert_eq!(spec.equations.parameter_values["a"], 2.0);

    let expr = spec.equations.rhs[0].clone();

    let f = expr.lambdify_borrowed_thread_safe(&["x", "y"]);

    let value = f(&[0.5, 1.0]);

    assert!((value + 0.5).abs() < 1e-12);
}

#[test]

fn bvp_task_parser_reports_bad_where_expression() {
    let input = r#"

task

solver: BVP

strategy: Damped

scheme: forward

method: Sparse



equations

arg: x

y: heat - y



where

heat: sin(



boundary_conditions

y_left: 1.0



mesh

t0: 0.0

t_end: 1.0

n_steps: 8



initial_guess

y: 0.0

"#;

    let err = parse_bvp_task_from_str(input)
        .expect_err("bad where expression should be reported as a BVP parser error");

    let message = err.to_string();

    assert!(message.contains("where/substitute"));

    assert!(message.contains("failed to parse symbolic expression"));
}

#[test]

fn bvp_task_parser_maps_banded_lambdify_generated_backend_config() {
    let input = r#"

task

solver: BVP

strategy: Damped

scheme: forward

method: Banded



equations

arg: x

unknowns: y

rhs: -y



boundary_conditions

y_left: 1.0



mesh

t0: 0.0

t_end: 1.0

n_steps: 8



initial_guess

y: 0.0



solver_options

generated_backend: banded_lambdify

banded_linear_solver: lapack

refinement_steps: 0

"#;

    let spec = parse_bvp_task_from_str(input).expect("banded BVP task should parse");

    assert_eq!(spec.solver.backend, BvpLinearBackendSpec::Banded);

    assert_eq!(
        spec.solver_options.generated_backend.preset.as_deref(),
        Some("banded_lambdify")
    );

    let config = generated_backend_config_from_spec(
        &spec.solver_options.generated_backend,
        &spec.solver.backend,
    )
    .expect("banded lambdify generated backend config should be valid");

    assert_eq!(config.matrix_backend_override, Some(MatrixBackend::Banded));

    assert_eq!(
        config.backend_policy_override,
        Some(BackendSelectionPolicy::LambdifyOnly)
    );

    assert_eq!(
        config.symbolic_assembly_backend,
        BvpSymbolicAssemblyBackend::AtomView
    );

    assert_eq!(
        config.banded_linear_solver_config.policy,
        LinearSolverPolicy::ForceBanded
    );

    assert_eq!(
        config
            .banded_linear_solver_config
            .iterative_refinement_steps,
        0
    );
}

#[test]

fn bvp_task_parser_maps_banded_aot_tcc_generated_backend_config() {
    let input = r#"

task

solver: BVP

strategy: Frozen

scheme: forward

method: Banded



equations

arg: x

unknowns: y

rhs: -y



boundary_conditions

y_left: 1.0



mesh

t0: 0.0

t_end: 1.0

n_steps: 8



initial_guess

y: 0.0



solver_options

generated_backend: banded_aot_tcc

aot_build_policy: rebuild

aot_build_profile: release

aot_compile_preset: dev_fastest

aot_execution_policy: sequential

symbolic_backend: atomview

banded_linear_solver: faithful

refinement_steps: 1

"#;

    let spec = parse_bvp_task_from_str(input).expect("banded AOT BVP task should parse");

    let config = generated_backend_config_from_spec(
        &spec.solver_options.generated_backend,
        &spec.solver.backend,
    )
    .expect("banded AOT/tcc generated backend config should be valid");

    assert_eq!(config.matrix_backend_override, Some(MatrixBackend::Banded));

    assert_eq!(
        config.backend_policy_override,
        Some(BackendSelectionPolicy::PreferAotThenLambdify)
    );

    assert_eq!(
        config.symbolic_assembly_backend,
        BvpSymbolicAssemblyBackend::AtomView
    );

    assert_eq!(config.aot_codegen_backend, AotCodegenBackend::C);

    assert_eq!(config.aot_c_compiler.as_deref(), Some("tcc"));

    assert_eq!(
        config.aot_execution_policy,
        AotExecutionPolicy::SequentialOnly
    );

    assert!(matches!(
        config.aot_build_policy,
        AotBuildPolicy::RebuildAlways {
            profile: AotBuildProfile::Release
        }
    ));

    assert_eq!(
        config.banded_linear_solver_config.policy,
        LinearSolverPolicy::ForceBanded
    );

    assert_eq!(
        config
            .banded_linear_solver_config
            .iterative_refinement_steps,
        1
    );
}

#[test]

fn bvp_task_docs_reject_numeric_backend_policies_because_they_cannot_carry_closures() {
    for policy in [
        "numeric_only",
        "prefer_aot_then_numeric",
        "prefer_lambdify_then_numeric",
    ] {
        let input = format!(
            r#"

task

solver: BVP

strategy: Damped

scheme: forward

method: Sparse



equations

arg: x

unknowns: y

rhs: -y



boundary_conditions

y_left: 1.0



mesh

t0: 0.0

t_end: 1.0

n_steps: 8



initial_guess

y: 0.5



solver_options

backend_policy: {policy}

"#
        );

        let spec = parse_bvp_task_from_str(&input).expect("syntax is valid");

        let err = match build_bvp_solver_from_spec(&spec) {
            Ok(_) => panic!("task docs cannot request pure numeric closure route: {policy}"),

            Err(err) => err,
        };

        match err {
            BvpTaskError::Semantic(message) => {
                assert!(
                    message.contains("cannot provide Rust numeric_rhs closures"),
                    "unexpected semantic error for {policy}: {message}"
                );
            }

            other => panic!("expected semantic error for {policy}, got {other:?}"),
        }
    }
}

#[test]

fn bvp_task_docs_accept_symbolic_backend_policies_after_numeric_guard() {
    for policy in ["lambdify_only", "prefer_aot_then_lambdify", "aot_only"] {
        let input = format!(
            r#"

task

solver: BVP

strategy: Damped

scheme: forward

method: Sparse



equations

arg: x

unknowns: y

rhs: -y



boundary_conditions

y_left: 1.0



mesh

t0: 0.0

t_end: 1.0

n_steps: 8



initial_guess

y: 0.5



solver_options

backend_policy: {policy}

"#
        );

        let spec = parse_bvp_task_from_str(&input).expect("syntax is valid");

        build_bvp_solver_from_spec(&spec)
            .unwrap_or_else(|err| panic!("symbolic policy {policy} should be accepted: {err}"));
    }
}

#[test]

fn bvp_task_runner_solves_reference_problem() {
    let input = r#"

task

solver: BVP

strategy: Damped

scheme: forward

method: Sparse



equations

arg: x

unknowns: z, y

rhs: y-z, -z^3



boundary_conditions

z_left: 1.0

y_right: 1.0



mesh

t0: 0.0

t_end: 1.0

n_steps: 20



initial_guess

guess: 0.0, 0.0



solver_options

tolerance: 1e-5

max_iterations: 20

loglevel: off

"#;

    let result = run_bvp_task_from_str(input).expect("BVP task should solve");

    let matrix = result
        .result
        .expect("solver should produce a result matrix");

    assert!(matrix.nrows() > 0);

    assert!(matrix.ncols() > 0);
}

#[test]

fn bvp_task_runner_can_save_csv() {
    let dir = tempdir().expect("tempdir should be created");

    let csv_path = dir.path().join("bvp_task_output.csv");

    let input = format!(
        r#"

task

solver: BVP

strategy: Damped

scheme: forward

method: Sparse



equations

arg: x

unknowns: z, y

rhs: y-z, -z^3



boundary_conditions

z_left: 1.0

y_right: 1.0



mesh

t0: 0.0

t_end: 1.0

n_steps: 20



initial_guess

z: 0.0

y: 0.0



solver_options

tolerance: 1e-5

max_iterations: 20

loglevel: off



postprocessing

save_csv: true

csv_path: {}

plot: false

"#,
        csv_path.display()
    );

    let result = run_bvp_task_from_str(&input).expect("BVP task should solve and save CSV");

    assert!(result.result.is_some());

    assert!(csv_path.exists());
}

#[test]

fn bvp_task_runner_can_execute_modern_postprocessing_plan() {
    let dir = tempdir().expect("tempdir should be created");

    let txt_path = dir.path().join("bvp_task_output.txt");

    let report_path = dir.path().join("bvp_task_report.md");

    let input = format!(
        r#"

task

solver: BVP

strategy: Damped

scheme: forward

method: Sparse



equations

arg: x

unknowns: z, y

rhs: y-z, -z^3



boundary_conditions

z_left: 1.0

y_right: 1.0



mesh

t0: 0.0

t_end: 1.0

n_steps: 20



initial_guess

z: 0.0

y: 0.0



solver_options

tolerance: 1e-5

max_iterations: 20

loglevel: off



postprocessing

save_txt: true

txt_path: {}

write_report: true

report_path: {}

plot: false

"#,
        txt_path.display(),
        report_path.display()
    );

    let result = run_bvp_task_from_str(&input).expect("BVP task should solve and postprocess");

    assert!(result.result.is_some());

    assert!(txt_path.exists());

    let report = std::fs::read_to_string(&report_path)
        .expect("report should be readable after postprocessing");

    assert!(report.contains("Solver Result Report"));

    assert!(report.contains("axis: x"));
}

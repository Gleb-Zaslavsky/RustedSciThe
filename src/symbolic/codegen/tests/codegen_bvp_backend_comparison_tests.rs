#![cfg(test)]

//! Backend/compiler comparison diagnostics for generated BVP residual/Jacobian code.
//!
//! This module avoids end-to-end solver timing and focuses on:
//! - generated module build latency,
//! - total time from symbolic BVP construction to first callable outputs,
//! - numerical agreement with the lambdify baseline,
//! - callback throughput after the generated backend is already built.
//!
//! Test map:
//! - `bvp_generated_backend_pipeline_comparison_table`
//!   Asks: which backend/compiler gets from symbolic BVP to first callable
//!   residual/Jacobian outputs fastest?
//!   Hypothesis: compiler choice dominates time-to-first-output on medium/large
//!   BVPs more than symbolic lowering does.
//! - `bvp_generated_backend_runtime_comparison_table`
//!   Asks: once callbacks are already available, which backend runs residual and
//!   sparse Jacobian evaluation fastest?
//!   Hypothesis: compiled backends strongly outperform the lambdify baseline.
//! - `bvp_generated_backend_compile_preset_tradeoff_table`
//!   Asks: how much compile-latency reduction do `Production/FastBuild/DevFastest`
//!   buy, and what runtime speed is lost in return?
//!   Hypothesis: aggressive presets can dramatically reduce build latency, but
//!   the payoff depends on backend/compiler choice.
//! - `bvp_lambdify_vs_atomview_callable_leaders_compare`
//!   Asks: does `AtomView + compiled native backend` beat the `ExprLegacy +
//!   Lambdify` path overall, from symbolic construction to callable functions?
//!   Hypothesis: `AtomView` plus fast native compilation nearly closes the
//!   bootstrap gap to `Lambdify`, while clearly winning on runtime throughput.

use crate::numerical::BVP_Damp::BVP_traits::MatrixType;
use crate::somelinalg::banded::banded_assembly::BandedAssembly;
use crate::somelinalg::banded::block_tridiagonal_lu_consistent::BlockTridiagonalLuConsistent;
use crate::somelinalg::banded::lapack_style_banded::LapackStyleBandedLuFaithful;
use crate::somelinalg::banded::linear_solver::{build_linear_solver, build_solver_for_system};
use crate::somelinalg::banded::node_major_layout::NodeMajorLayout;
use crate::somelinalg::banded::solver_policy::{
    FallbackPolicy, LinearSolverConfig, LinearSolverPolicy,
};
use crate::somelinalg::banded::superblock_layout::SuperBlockLayout;
use crate::somelinalg::banded::{solver_traits::DirectLinearSolver, LinearSystemRef};
use crate::symbolic::codegen::c_backend::codegen_c_aot_build::{
    CAotBuildProfile, CAotBuildRequest, CAotCompileConfig, ExecutedCAotBuild,
};
use crate::symbolic::codegen::c_backend::codegen_c_aot_registry::register_c_build_in_registry;
use crate::symbolic::codegen::c_backend::codegen_c_aot_runtime_link::register_generated_c_sparse_backend;
use crate::symbolic::codegen::codegen_adapters::{
    banded_ir_blocks, residual_ir_blocks, sparse_ir_blocks,
};
use crate::symbolic::codegen::codegen_aot_driver::{AotCodegenBackend, GeneratedAotArtifact};
use crate::symbolic::codegen::codegen_aot_registry::AotRegistry;
use crate::symbolic::codegen::codegen_aot_runtime_link::{
    register_generated_sparse_cdylib_backend, unregister_linked_sparse_backend,
    LinkedSparseAotBackend,
};
use crate::symbolic::codegen::codegen_manifest::PreparedProblemManifest;
use crate::symbolic::codegen::codegen_provider_api::{MatrixBackend, PreparedProblem};
use crate::symbolic::codegen::rust_backend::codegen_aot_build::{
    AotBuildProfile, AotBuildRequest, AotCompileConfig, ExecutedAotBuild,
};
use crate::symbolic::codegen::tests::codegen_test_support::{
    build_combustion_bvp_case, build_combustion_bvp_case_with_backend, build_real_bvp_damp1_case,
    build_real_bvp_damp1_case_with_backend, command_exists,
};
use crate::symbolic::codegen::zig_backend::codegen_zig_aot_build::{
    ExecutedZigAotBuild, ZigAotBuildProfile, ZigAotBuildRequest,
};
use crate::symbolic::codegen::zig_backend::codegen_zig_aot_registry::register_zig_build_in_registry;
use crate::symbolic::codegen::zig_backend::codegen_zig_aot_runtime_link::register_generated_zig_sparse_backend;
use crate::symbolic::codegen::CodegenIR::AtomOptimizationProfile;
use crate::symbolic::symbolic_functions_BVP::{
    BvpPreparedSparseAotProblem, BvpSymbolicAssemblyBackend, Jacobian,
};
use faer::col::Col;
use faer::linalg::solvers::Solve;
use nalgebra::DMatrix;
use std::env;
use std::hint::black_box;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;
use tabled::{builder::Builder, settings::Style};

type EvalFn<'a> = Box<dyn Fn(&[f64]) -> f64 + Send + Sync + 'a>;

#[derive(Clone)]
struct BackendVariant {
    label: String,
    backend: AotCodegenBackend,
    c_compiler: Option<String>,
}

struct ScenarioSpec {
    label: &'static str,
    runtime_iters: usize,
    runtime_samples: usize,
    build_jacobian: fn() -> Jacobian,
}

struct ScenarioData {
    label: &'static str,
    symbolic_backend: BvpSymbolicAssemblyBackend,
    runtime_iters: usize,
    runtime_samples: usize,
    symbolic_ms: f64,
    prepared: BvpPreparedSparseAotProblem,
    args: Vec<f64>,
    lambdify_residual: Vec<f64>,
    lambdify_jacobian: Vec<f64>,
}

struct BuildAndLinkMetrics {
    symbolic_backend: &'static str,
    variant_label: String,
    preset_label: &'static str,
    artifact_ms: f64,
    jacobian_prepare_ms: f64,
    atom_sparse_lookup_prepare_ms: f64,
    atom_sparse_jacobian_build_ms: f64,
    atom_finalize_codegen_plan_ms: f64,
    atom_lower_many_ms: f64,
    atom_peephole_ms: f64,
    atom_reuse_temps_ms: f64,
    atom_push_ms: f64,
    prepared_module_init_ms: f64,
    prepared_residual_blocks_ms: f64,
    prepared_jacobian_blocks_ms: f64,
    module_ms: f64,
    source_ms: f64,
    source_probe_emit_ms: f64,
    language_source_emit_ms: f64,
    c_header_emit_ms: f64,
    artifact_packaging_ms: f64,
    source_kb: f64,
    materialize_ms: f64,
    build_ms: f64,
    link_ms: f64,
    first_issue_ms: f64,
    total_to_outputs_ms: f64,
    residual_diff: f64,
    jacobian_diff: f64,
    status: String,
    linked: Option<LinkedSparseAotBackend>,
}

#[derive(Clone)]
struct PipelineBootstrapSample {
    route: &'static str,
    symbolic_backend: &'static str,
    variant_label: String,
    preset_label: &'static str,
    symbolic_ms: f64,
    callable_prep_ms: f64,
    artifact_ms: f64,
    jacobian_prepare_ms: f64,
    atom_sparse_lookup_prepare_ms: f64,
    atom_sparse_jacobian_build_ms: f64,
    atom_finalize_codegen_plan_ms: f64,
    atom_lower_many_ms: f64,
    atom_peephole_ms: f64,
    atom_reuse_temps_ms: f64,
    atom_push_ms: f64,
    prepared_module_init_ms: f64,
    prepared_residual_blocks_ms: f64,
    prepared_jacobian_blocks_ms: f64,
    module_ms: f64,
    source_ms: f64,
    source_probe_emit_ms: f64,
    language_source_emit_ms: f64,
    c_header_emit_ms: f64,
    artifact_packaging_ms: f64,
    source_kb: f64,
    materialize_ms: f64,
    build_ms: f64,
    link_ms: f64,
    first_issue_ms: f64,
    total_to_outputs_ms: f64,
    residual_diff: f64,
    jacobian_diff: f64,
    status: String,
}

#[derive(Clone, Copy)]
struct PipelineMetricAggregate {
    count: usize,
    mean: f64,
    std: f64,
    min: f64,
    max: f64,
}

struct PipelineBootstrapAggregate {
    route: &'static str,
    symbolic_backend: &'static str,
    variant_label: String,
    preset_label: &'static str,
    runs: usize,
    ok_runs: usize,
    symbolic_ms: PipelineMetricAggregate,
    callable_prep_ms: PipelineMetricAggregate,
    artifact_ms: PipelineMetricAggregate,
    jacobian_prepare_ms: PipelineMetricAggregate,
    atom_sparse_lookup_prepare_ms: PipelineMetricAggregate,
    atom_sparse_jacobian_build_ms: PipelineMetricAggregate,
    atom_finalize_codegen_plan_ms: PipelineMetricAggregate,
    atom_lower_many_ms: PipelineMetricAggregate,
    atom_peephole_ms: PipelineMetricAggregate,
    atom_reuse_temps_ms: PipelineMetricAggregate,
    atom_push_ms: PipelineMetricAggregate,
    prepared_module_init_ms: PipelineMetricAggregate,
    prepared_residual_blocks_ms: PipelineMetricAggregate,
    prepared_jacobian_blocks_ms: PipelineMetricAggregate,
    module_ms: PipelineMetricAggregate,
    source_ms: PipelineMetricAggregate,
    source_probe_emit_ms: PipelineMetricAggregate,
    language_source_emit_ms: PipelineMetricAggregate,
    c_header_emit_ms: PipelineMetricAggregate,
    artifact_packaging_ms: PipelineMetricAggregate,
    artifact_other_ms: PipelineMetricAggregate,
    source_kb: PipelineMetricAggregate,
    materialize_ms: PipelineMetricAggregate,
    build_ms: PipelineMetricAggregate,
    link_ms: PipelineMetricAggregate,
    first_issue_ms: PipelineMetricAggregate,
    total_to_outputs_ms: PipelineMetricAggregate,
    residual_diff: PipelineMetricAggregate,
    jacobian_diff: PipelineMetricAggregate,
    status: String,
}

struct RuntimeRow {
    label: String,
    residual_ms: f64,
    jacobian_ms: f64,
    total_ms: f64,
    residual_diff: f64,
    jacobian_diff: f64,
    status: String,
}

struct CompiledMatrixBackendRow {
    scenario: String,
    matrix_backend: String,
    variant: String,
    preset: String,
    compile_mode: String,
    build_ms: String,
    total_to_outputs_ms: String,
    runtime_total_ms: String,
    residual_diff: String,
    jacobian_diff: String,
    status: String,
}

struct CallableStoryRow {
    scenario: String,
    source: String,
    matrix_backend: String,
    variant: String,
    preset: String,
    compile_mode: String,
    symbolic_ms: String,
    callable_prep_ms: String,
    materialize_ms: String,
    build_ms: String,
    link_ms: String,
    first_issue_ms: String,
    total_to_outputs_ms: String,
    status: String,
}

struct CallbackRuntimeMatrixRow {
    scenario: String,
    source: String,
    matrix_backend: String,
    variant: String,
    preset: String,
    residual_ms: String,
    jacobian_ms: String,
    total_ms: String,
    residual_diff: String,
    jacobian_diff: String,
    status: String,
}

struct LinearSolverStoryRow {
    scenario: String,
    source: String,
    matrix_backend: String,
    variant: String,
    preset: String,
    linear_solver: String,
    layout: String,
    refinement: String,
    direct_rr: String,
    final_rr: String,
    solve_rr: String,
    total_ms: String,
    solve_diff: String,
    relative_x_diff: String,
    status: String,
}

struct LinearSolveMetrics {
    linear_solver: String,
    layout: String,
    refinement: String,
    direct_rr: f64,
    final_rr: f64,
    solve_rr: f64,
    total_ms: f64,
    solve_diff: f64,
    relative_x_diff: f64,
    status: String,
}

#[derive(Clone, Copy, Debug)]
enum BandedSolverUnderTest {
    LegacyAuto,
    LapackStyle { refinement_steps: usize },
    Consistent { refinement_steps: usize },
}

impl BandedSolverUnderTest {
    fn variants() -> [Self; 4] {
        [
            Self::LegacyAuto,
            Self::LapackStyle {
                refinement_steps: 0,
            },
            Self::LapackStyle {
                refinement_steps: 1,
            },
            Self::Consistent {
                refinement_steps: 1,
            },
        ]
    }
}

struct EndToEndCallableRow {
    variant: String,
    symbolic_backend: &'static str,
    preset: &'static str,
    symbolic_ms: f64,
    callable_prep_ms: f64,
    materialize_ms: f64,
    build_ms: f64,
    link_ms: f64,
    first_issue_ms: f64,
    total_to_outputs_ms: f64,
    residual_ms: f64,
    jacobian_ms: f64,
    total_ms: f64,
    speedup_vs_lambdify: f64,
    residual_diff: f64,
    jacobian_diff: f64,
    status: String,
}

fn emit_progress(message: impl AsRef<str>) {
    eprintln!("[BVP backend compare] {}", message.as_ref());
    let _ = io::stderr().flush();
}

fn flush_stdout() {
    let _ = io::stdout().flush();
}

fn available_backend_variants() -> Vec<BackendVariant> {
    emit_progress("probing available generated backends");
    let mut variants = vec![BackendVariant {
        label: "Rust".to_string(),
        backend: AotCodegenBackend::Rust,
        c_compiler: None,
    }];

    for compiler in ["gcc", "tcc"] {
        emit_progress(format!("probing C compiler `{compiler}`"));
        if command_exists(compiler) {
            variants.push(BackendVariant {
                label: format!("C-{compiler}"),
                backend: AotCodegenBackend::C,
                c_compiler: Some(compiler.to_string()),
            });
        }
    }

    emit_progress("probing Zig compiler `zig`");
    if command_exists("zig") {
        variants.push(BackendVariant {
            label: "Zig".to_string(),
            backend: AotCodegenBackend::Zig,
            c_compiler: None,
        });
    }

    emit_progress(format!(
        "finished backend probe: {} variant(s)",
        variants.len()
    ));
    variants
}

fn compiler_probe_status(program: &str) -> String {
    let override_value = match program {
        "tcc" => env::var("RUSTEDSCITHE_TCC")
            .ok()
            .or_else(|| env::var("RUSTEDSCITHE_C_COMPILER").ok()),
        "gcc" => env::var("RUSTEDSCITHE_GCC")
            .ok()
            .or_else(|| env::var("RUSTEDSCITHE_C_COMPILER").ok()),
        "zig" => env::var("RUSTEDSCITHE_ZIG").ok(),
        _ => None,
    }
    .filter(|value| !value.trim().is_empty());

    let base = if command_exists(program) {
        "detected"
    } else {
        "missing"
    };

    match override_value {
        Some(value) => format!("{base} ({value})"),
        None => base.to_string(),
    }
}

fn print_backend_probe_summary(variants: &[BackendVariant]) {
    let labels = variants
        .iter()
        .map(|variant| variant.label.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    println!(
        "[BVP backend compare] detected variants: {}",
        if labels.is_empty() { "<none>" } else { &labels }
    );
    println!(
        "[BVP backend compare] probes: gcc={}, tcc={}, zig={}",
        compiler_probe_status("gcc"),
        compiler_probe_status("tcc"),
        compiler_probe_status("zig")
    );
    flush_stdout();
}

fn short_scenario_label(label: &str) -> &'static str {
    match label {
        "small-damp1-24" => "sd24",
        "combustion-100" => "cb100",
        "combustion-1000" => "cb1000",
        "combustion-3000" => "cb3000",
        _ => "bvp",
    }
}

fn short_variant_label(variant: &BackendVariant) -> String {
    match (variant.backend, variant.c_compiler.as_deref()) {
        (AotCodegenBackend::Rust, _) => "rs".to_string(),
        (AotCodegenBackend::Zig, _) => "zig".to_string(),
        (AotCodegenBackend::C, Some("gcc")) => "cgcc".to_string(),
        (AotCodegenBackend::C, Some("tcc")) => "ctcc".to_string(),
        (AotCodegenBackend::C, Some(other)) => format!("c{}", sanitize_label(other)),
        (AotCodegenBackend::C, None) => "c".to_string(),
    }
}

fn short_preset_label(preset: BuildPreset) -> &'static str {
    match preset {
        BuildPreset::Production => "prod",
        BuildPreset::FastBuild => "fast",
        BuildPreset::DevFastest => "dev",
    }
}

fn unique_compare_artifact_dir(scenario: &str, variant: &str) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock should be after unix epoch")
        .as_nanos();
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("test-artifacts")
        .join("cbcmp")
        .join(format!(
            "{}-{}-{}-{nonce}",
            sanitize_label(scenario),
            sanitize_label(variant),
            std::process::id()
        ))
}

fn sanitize_label(label: &str) -> String {
    label
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' => ch.to_ascii_lowercase(),
            _ => '_',
        })
        .collect()
}

fn sample_args(input_len: usize) -> Vec<f64> {
    (0..input_len)
        .map(|index| 0.25 + index as f64 * 1.0e-4)
        .collect()
}

fn compile_residual_lambdify<'a>(prepared: &'a BvpPreparedSparseAotProblem) -> Vec<EvalFn<'a>> {
    let input_names = prepared.as_prepared_problem().input_names().to_vec();
    prepared
        .residuals
        .iter()
        .map(|expr| {
            let fun = expr.lambdify_borrowed_thread_safe(input_names.as_slice());
            Box::new(move |args: &[f64]| fun(args)) as EvalFn<'a>
        })
        .collect()
}

fn compile_sparse_lambdify<'a>(prepared: &'a BvpPreparedSparseAotProblem) -> Vec<EvalFn<'a>> {
    let input_names = prepared.as_prepared_problem().input_names().to_vec();
    prepared
        .sparse_entries
        .iter()
        .map(|(_, _, expr)| {
            let fun = expr.lambdify_borrowed_thread_safe(input_names.as_slice());
            Box::new(move |args: &[f64]| fun(args)) as EvalFn<'a>
        })
        .collect()
}

fn matrix_backend_label(matrix_backend: MatrixBackend) -> &'static str {
    match matrix_backend {
        MatrixBackend::Banded => "Banded",
        _ => "Sparse",
    }
}

fn compile_mode_label(variant: &BackendVariant, preset: BuildPreset) -> String {
    match variant.backend {
        AotCodegenBackend::Rust => match preset {
            BuildPreset::Production => "cargo-release(default)".to_string(),
            BuildPreset::FastBuild => "cargo-release(O1,cgu16)".to_string(),
            BuildPreset::DevFastest => "cargo-release(O0,cgu16)".to_string(),
        },
        AotCodegenBackend::C => {
            let compiler = variant.c_compiler.as_deref().unwrap_or("gcc");
            match preset {
                BuildPreset::Production => format!("{compiler} -O3"),
                BuildPreset::FastBuild => format!("{compiler} -O1"),
                BuildPreset::DevFastest => format!("{compiler} -O0"),
            }
        }
        AotCodegenBackend::Zig => match preset {
            BuildPreset::Production => "zig ReleaseFast".to_string(),
            BuildPreset::FastBuild => "zig ReleaseSmall".to_string(),
            BuildPreset::DevFastest => "zig Debug".to_string(),
        },
    }
}

fn compile_banded_lambdify<'a>(prepared: &'a BvpPreparedSparseAotProblem) -> Vec<EvalFn<'a>> {
    let input_names = prepared.as_prepared_banded_problem().input_names().to_vec();
    prepared
        .as_prepared_banded_problem()
        .jacobian_plan
        .chunks
        .iter()
        .flat_map(|chunk| chunk.entries.iter())
        .map(|entry| {
            let fun = entry
                .expr
                .lambdify_borrowed_thread_safe(input_names.as_slice());
            Box::new(move |args: &[f64]| fun(args)) as EvalFn<'a>
        })
        .collect()
}

fn eval_functions(functions: &[EvalFn<'_>], args: &[f64]) -> Vec<f64> {
    functions.iter().map(|fun| fun(args)).collect()
}

fn eval_ir_blocks_sequential(
    args: &[f64],
    blocks: &[(usize, crate::symbolic::codegen::CodegenIR::LinearBlock)],
    output_len: usize,
) -> Vec<f64> {
    let mut out = vec![0.0; output_len];
    for (offset, block) in blocks {
        let len = block.outputs.len();
        block.eval_into(args, &mut out[*offset..(*offset + len)]);
    }
    out
}

fn max_abs_diff(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

fn vector_linf_norm(values: &[f64]) -> f64 {
    values
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
}

fn relative_x_diff(lhs: &[f64], rhs: &[f64]) -> f64 {
    max_abs_diff(lhs, rhs) / vector_linf_norm(rhs).max(1.0)
}

fn relative_dense_residual(matrix: &DMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
    let mut rmax = 0.0_f64;
    let mut bmax = 0.0_f64;
    for row in 0..matrix.nrows() {
        let mut ax = 0.0;
        for col in 0..matrix.ncols() {
            ax += matrix[(row, col)] * x[col];
        }
        rmax = rmax.max((ax - b[row]).abs());
        bmax = bmax.max(b[row].abs());
    }
    rmax / bmax.max(1.0)
}

fn fmt_metric_or_dash(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.3e}")
    } else {
        "-".to_string()
    }
}

fn banded_assembly_to_dense(assembly: &BandedAssembly) -> DMatrix<f64> {
    let mut dense = DMatrix::zeros(assembly.n(), assembly.n());
    for offset in assembly.min_offset()..=assembly.max_offset() {
        if let Some(diag) = assembly.diag(offset) {
            for (pos, value) in diag.iter().enumerate() {
                let (row, col) = assembly
                    .diag_pos_to_ij(offset, pos)
                    .expect("banded diagonal position should map back to matrix coordinates");
                dense[(row, col)] = *value;
            }
        }
    }
    dense
}

fn sample_rhs(n: usize) -> Vec<f64> {
    (0..n).map(|index| 1.0 + index as f64 * 1.0e-3).collect()
}

fn vars_per_node_for_scenario(label: &str) -> usize {
    match label {
        "small-damp1-24" => 2,
        "combustion-100" | "combustion-1000" => 6,
        _ => 1,
    }
}

fn infer_node_major_layout_for_scenario(label: &str, n_total: usize) -> NodeMajorLayout {
    let vars_per_node = vars_per_node_for_scenario(label);
    assert!(
        vars_per_node > 0 && n_total % vars_per_node == 0,
        "failed to infer node-major layout for scenario `{label}`",
    );
    NodeMajorLayout::new(n_total / vars_per_node, vars_per_node)
        .expect("node-major layout should be valid for scenario")
}

fn preferred_superblock_group_for_scenario(label: &str, n_total: usize) -> usize {
    let layout = infer_node_major_layout_for_scenario(label, n_total);
    if layout.vars_per_node() >= 6 && layout.n_nodes() % 2 == 0 {
        2
    } else {
        1
    }
}

fn solve_sparse_lu(matrix: &faer::sparse::SparseColMat<usize, f64>, rhs: &[f64]) -> Vec<f64> {
    let lu = matrix
        .sp_lu()
        .expect("sparse LU factorization should succeed");
    let rhs_col = Col::from_fn(rhs.len(), |index| rhs[index]);
    let solution = lu.solve(&rhs_col);
    solution.iter().copied().collect()
}

fn solve_banded_legacy_for_scenario(
    scenario_label: &str,
    assembly: &BandedAssembly,
    rhs: &[f64],
) -> LinearSolveMetrics {
    let layout = infer_node_major_layout_for_scenario(scenario_label, assembly.n());
    let block = assembly
        .to_block_tridiagonal(layout.n_blocks(), layout.block_size())
        .expect("banded assembly should match node-major layout");
    let native_attempt = build_linear_solver(
        &block,
        LinearSolverConfig {
            policy: LinearSolverPolicy::ForceBlockTridiagonal,
            fallback: FallbackPolicy::Never,
            iterative_refinement_steps: 0,
        },
    );
    let solver = match build_linear_solver(
        &block,
        LinearSolverConfig {
            policy: LinearSolverPolicy::Auto,
            fallback: FallbackPolicy::ToFaerSparse,
            iterative_refinement_steps: 0,
        },
    ) {
        Ok(solver) => solver,
        Err(err) => {
            return LinearSolveMetrics {
                linear_solver: "legacy_banded".to_string(),
                layout: format!("{}x{}", layout.n_blocks(), layout.block_size()),
                refinement: "-".to_string(),
                direct_rr: f64::NAN,
                final_rr: f64::NAN,
                solve_rr: f64::NAN,
                total_ms: f64::NAN,
                solve_diff: f64::NAN,
                relative_x_diff: f64::NAN,
                status: format!("factorization_failed({err:?})"),
            };
        }
    };
    let mut rhs_owned = rhs.to_vec();
    let begin = Instant::now();
    let solve_status = solver.solve_in_place(rhs_owned.as_mut_slice());
    let total_ms = begin.elapsed().as_secs_f64() * 1_000.0;
    let status = match native_attempt {
        Ok(_) => match solve_status {
            Ok(()) => "ok".to_string(),
            Err(err) => format!("solve_failed({err:?})"),
        },
        Err(err) => match solve_status {
            Ok(()) => format!("fallback({err:?})"),
            Err(solve_err) => format!("fallback({err:?})+solve_failed({solve_err:?})"),
        },
    };
    LinearSolveMetrics {
        linear_solver: solver.backend_name().to_string(),
        layout: format!("{}x{}", layout.n_blocks(), layout.block_size()),
        refinement: "-".to_string(),
        direct_rr: f64::NAN,
        final_rr: f64::NAN,
        solve_rr: f64::NAN,
        total_ms,
        solve_diff: f64::NAN,
        relative_x_diff: f64::NAN,
        status,
    }
}

fn solve_banded_consistent_direct_for_scenario(
    scenario_label: &str,
    assembly: &BandedAssembly,
    rhs: &[f64],
    refinement_steps: usize,
) -> (LinearSolveMetrics, Vec<f64>) {
    let node_layout = infer_node_major_layout_for_scenario(scenario_label, assembly.n());
    let nodes_per_superblock =
        preferred_superblock_group_for_scenario(scenario_label, assembly.n());
    let (layout, block) = if nodes_per_superblock == 1 {
        (
            format!("{}x{}", node_layout.n_blocks(), node_layout.block_size()),
            assembly
                .to_block_tridiagonal(node_layout.n_blocks(), node_layout.block_size())
                .expect("banded assembly should match node-major layout"),
        )
    } else {
        let super_layout = SuperBlockLayout::new(
            node_layout.n_nodes(),
            node_layout.vars_per_node(),
            nodes_per_superblock,
        )
        .expect("superblock layout should be constructible");
        (
            format!("{}x{}", super_layout.n_blocks(), super_layout.block_size()),
            assembly
                .finalize_superblock_tridiagonal(&super_layout)
                .expect("banded assembly should match superblock layout"),
        )
    };

    let mut solver = match BlockTridiagonalLuConsistent::new(block.n_blocks(), block.block_size()) {
        Ok(solver) => solver,
        Err(err) => {
            return (
                LinearSolveMetrics {
                    linear_solver: format!(
                        "block_tridiagonal_lu_consistent+refine{refinement_steps}"
                    ),
                    layout,
                    refinement: format!("auto/{refinement_steps}"),
                    direct_rr: f64::NAN,
                    final_rr: f64::NAN,
                    solve_rr: f64::NAN,
                    total_ms: f64::NAN,
                    solve_diff: f64::NAN,
                    relative_x_diff: f64::NAN,
                    status: format!("workspace_failed({err:?})"),
                },
                rhs.to_vec(),
            );
        }
    };
    if let Err(err) = solver.factor_from(&block) {
        return (
            LinearSolveMetrics {
                linear_solver: format!("block_tridiagonal_lu_consistent+refine{refinement_steps}"),
                layout,
                refinement: format!("auto/{refinement_steps}"),
                direct_rr: f64::NAN,
                final_rr: f64::NAN,
                solve_rr: f64::NAN,
                total_ms: f64::NAN,
                solve_diff: f64::NAN,
                relative_x_diff: f64::NAN,
                status: format!("factorization_failed({err:?})"),
            },
            rhs.to_vec(),
        );
    }

    let mut rhs_owned = rhs.to_vec();
    let begin = Instant::now();
    let report = match solver.solve_in_place_with_iterative_refinement_report(
        &block,
        rhs_owned.as_mut_slice(),
        refinement_steps,
    ) {
        Ok(report) => report,
        Err(err) => {
            return (
                LinearSolveMetrics {
                    linear_solver: format!(
                        "block_tridiagonal_lu_consistent+refine{refinement_steps}"
                    ),
                    layout,
                    refinement: format!("auto/{refinement_steps}"),
                    direct_rr: f64::NAN,
                    final_rr: f64::NAN,
                    solve_rr: f64::NAN,
                    total_ms: begin.elapsed().as_secs_f64() * 1_000.0,
                    solve_diff: f64::NAN,
                    relative_x_diff: f64::NAN,
                    status: format!("solve_failed({err:?})"),
                },
                rhs_owned,
            );
        }
    };
    let total_ms = begin.elapsed().as_secs_f64() * 1_000.0;

    (
        LinearSolveMetrics {
            linear_solver: format!("block_tridiagonal_lu_consistent+refine{refinement_steps}"),
            layout,
            refinement: format!(
                "{}/{}{}",
                report.accepted_steps,
                report.requested_steps,
                if report.refinement_attempted {
                    ""
                } else {
                    " skipped"
                }
            ),
            direct_rr: report.direct_relative_residual,
            final_rr: report.final_relative_residual,
            solve_rr: f64::NAN,
            total_ms,
            solve_diff: f64::NAN,
            relative_x_diff: f64::NAN,
            status: "ok".to_string(),
        },
        rhs_owned,
    )
}

fn solve_banded_lapack_style_for_scenario(
    assembly: &BandedAssembly,
    rhs: &[f64],
    refinement_steps: usize,
) -> (LinearSolveMetrics, Vec<f64>) {
    let solver = match build_solver_for_system(
        LinearSystemRef::BandedAssembly(assembly),
        LinearSolverConfig {
            policy: LinearSolverPolicy::ForceBanded,
            fallback: FallbackPolicy::Never,
            iterative_refinement_steps: refinement_steps,
        },
    ) {
        Ok(solver) => solver,
        Err(err) => {
            return (
                LinearSolveMetrics {
                    linear_solver: if refinement_steps == 0 {
                        "lapack_style_banded_lu".to_string()
                    } else {
                        format!("lapack_style_banded_lu+refine{refinement_steps}")
                    },
                    layout: format!("n{} kl{} ku{}", assembly.n(), assembly.kl(), assembly.ku()),
                    refinement: if refinement_steps == 0 {
                        "-".to_string()
                    } else {
                        format!("auto/{refinement_steps}")
                    },
                    direct_rr: f64::NAN,
                    final_rr: f64::NAN,
                    solve_rr: f64::NAN,
                    total_ms: f64::NAN,
                    solve_diff: f64::NAN,
                    relative_x_diff: f64::NAN,
                    status: format!("factorization_failed({err:?})"),
                },
                rhs.to_vec(),
            );
        }
    };
    let mut rhs_owned = rhs.to_vec();
    let begin = Instant::now();
    let solve_status = solver.solve_in_place(rhs_owned.as_mut_slice());
    let total_ms = begin.elapsed().as_secs_f64() * 1_000.0;
    (
        LinearSolveMetrics {
            linear_solver: solver.backend_name().to_string(),
            layout: format!("n{} kl{} ku{}", assembly.n(), assembly.kl(), assembly.ku()),
            refinement: if refinement_steps == 0 {
                "-".to_string()
            } else {
                format!("auto/{refinement_steps}")
            },
            direct_rr: f64::NAN,
            final_rr: f64::NAN,
            solve_rr: f64::NAN,
            total_ms,
            solve_diff: f64::NAN,
            relative_x_diff: f64::NAN,
            status: match solve_status {
                Ok(()) => "ok".to_string(),
                Err(err) => format!("solve_failed({err:?})"),
            },
        },
        rhs_owned,
    )
}

fn metrics_solution_legacy(
    scenario_label: &str,
    assembly: &BandedAssembly,
    rhs: &[f64],
) -> Vec<f64> {
    let layout = infer_node_major_layout_for_scenario(scenario_label, assembly.n());
    let block = assembly
        .to_block_tridiagonal(layout.n_blocks(), layout.block_size())
        .expect("banded assembly should match node-major layout");
    let solver = build_linear_solver(
        &block,
        LinearSolverConfig {
            policy: LinearSolverPolicy::Auto,
            fallback: FallbackPolicy::ToFaerSparse,
            iterative_refinement_steps: 0,
        },
    )
    .expect("legacy banded solver should either factorize or fallback");
    let mut rhs_owned = rhs.to_vec();
    solver
        .solve_in_place(rhs_owned.as_mut_slice())
        .expect("legacy banded solve should succeed");
    rhs_owned
}

fn solve_banded_solver_for_scenario(
    scenario_label: &str,
    assembly: &BandedAssembly,
    rhs: &[f64],
    solver: BandedSolverUnderTest,
) -> (LinearSolveMetrics, Vec<f64>) {
    match solver {
        BandedSolverUnderTest::LegacyAuto => (
            solve_banded_legacy_for_scenario(scenario_label, assembly, rhs),
            metrics_solution_legacy(scenario_label, assembly, rhs),
        ),
        BandedSolverUnderTest::LapackStyle { refinement_steps } => {
            solve_banded_lapack_style_for_scenario(assembly, rhs, refinement_steps)
        }
        BandedSolverUnderTest::Consistent { refinement_steps } => {
            solve_banded_consistent_direct_for_scenario(
                scenario_label,
                assembly,
                rhs,
                refinement_steps,
            )
        }
    }
}

fn average_linear_solve_ms(samples: usize, iters: usize, mut run_once: impl FnMut()) -> f64 {
    let mut totals = Vec::with_capacity(samples);
    for _ in 0..samples {
        let begin = Instant::now();
        for _ in 0..iters {
            run_once();
        }
        totals.push(begin.elapsed().as_secs_f64() * 1_000.0 / iters as f64);
    }
    average_ms(&totals)
}

fn measure_lambdify_matrix_runtime(
    scenario: &ScenarioData,
    matrix_backend: MatrixBackend,
) -> (
    CallableStoryRow,
    CallbackRuntimeMatrixRow,
    Vec<f64>,
    Option<BandedAssembly>,
) {
    let compile_begin = Instant::now();
    let residual_fns = compile_residual_lambdify(&scenario.prepared);
    let jacobian_fns = match matrix_backend {
        MatrixBackend::Banded => compile_banded_lambdify(&scenario.prepared),
        _ => compile_sparse_lambdify(&scenario.prepared),
    };
    let callable_prep_ms = compile_begin.elapsed().as_secs_f64() * 1_000.0;

    let first_issue_begin = Instant::now();
    let residual_once = eval_functions(&residual_fns, scenario.args.as_slice());
    let jacobian_once = eval_functions(&jacobian_fns, scenario.args.as_slice());
    let first_issue_ms = first_issue_begin.elapsed().as_secs_f64() * 1_000.0;

    let mut residual_samples = Vec::with_capacity(scenario.runtime_samples);
    let mut jacobian_samples = Vec::with_capacity(scenario.runtime_samples);
    for _ in 0..scenario.runtime_samples {
        let residual_begin = Instant::now();
        for _ in 0..scenario.runtime_iters {
            black_box(eval_functions(&residual_fns, scenario.args.as_slice()));
        }
        residual_samples.push(residual_begin.elapsed().as_secs_f64() * 1_000.0);

        let jacobian_begin = Instant::now();
        for _ in 0..scenario.runtime_iters {
            black_box(eval_functions(&jacobian_fns, scenario.args.as_slice()));
        }
        jacobian_samples.push(jacobian_begin.elapsed().as_secs_f64() * 1_000.0);
    }
    let residual_ms = average_ms(&residual_samples);
    let jacobian_ms = average_ms(&jacobian_samples);

    let banded_assembly = match matrix_backend {
        MatrixBackend::Banded => Some(
            scenario
                .prepared
                .as_prepared_banded_problem()
                .jacobian_plan
                .assemble_banded_assembly(jacobian_once.as_slice()),
        ),
        _ => None,
    };

    (
        CallableStoryRow {
            scenario: scenario.label.to_string(),
            source: "Lambdify".to_string(),
            matrix_backend: matrix_backend_label(matrix_backend).to_string(),
            variant: "-".to_string(),
            preset: "-".to_string(),
            compile_mode: "n/a".to_string(),
            symbolic_ms: format!("{:.3}", scenario.symbolic_ms),
            callable_prep_ms: format!("{:.3}", callable_prep_ms),
            materialize_ms: "0.000".to_string(),
            build_ms: "0.000".to_string(),
            link_ms: "0.000".to_string(),
            first_issue_ms: format!("{:.3}", first_issue_ms),
            total_to_outputs_ms: format!(
                "{:.3}",
                scenario.symbolic_ms + callable_prep_ms + first_issue_ms
            ),
            status: "ok".to_string(),
        },
        CallbackRuntimeMatrixRow {
            scenario: scenario.label.to_string(),
            source: "Lambdify".to_string(),
            matrix_backend: matrix_backend_label(matrix_backend).to_string(),
            variant: "-".to_string(),
            preset: "-".to_string(),
            residual_ms: format!("{:.3}", residual_ms),
            jacobian_ms: format!("{:.3}", jacobian_ms),
            total_ms: format!("{:.3}", residual_ms + jacobian_ms),
            residual_diff: format!(
                "{:.3e}",
                max_abs_diff(&residual_once, &scenario.lambdify_residual)
            ),
            jacobian_diff: match matrix_backend {
                MatrixBackend::Banded => {
                    let dense = banded_assembly_to_dense(
                        banded_assembly
                            .as_ref()
                            .expect("banded assembly should exist"),
                    );
                    let sparse_dense = scenario
                        .prepared
                        .as_prepared_problem()
                        .jacobian_plan
                        .assemble_sparse_col_mat(scenario.lambdify_jacobian.as_slice())
                        .to_DMatrixType();
                    format!(
                        "{:.3e}",
                        max_abs_diff(dense.as_slice(), sparse_dense.as_slice())
                    )
                }
                _ => format!(
                    "{:.3e}",
                    max_abs_diff(&jacobian_once, &scenario.lambdify_jacobian)
                ),
            },
            status: "ok".to_string(),
        },
        jacobian_once,
        banded_assembly,
    )
}

fn build_jacobian_for_spec(
    spec: &ScenarioSpec,
    symbolic_backend: BvpSymbolicAssemblyBackend,
) -> Jacobian {
    match spec.label {
        "small-damp1-24" => build_real_bvp_damp1_case_with_backend(24, symbolic_backend),
        "combustion-100" => build_combustion_bvp_case_with_backend(100, symbolic_backend),
        "combustion-1000" => build_combustion_bvp_case_with_backend(1000, symbolic_backend),
        "combustion-3000" => build_combustion_bvp_case_with_backend(3000, symbolic_backend),
        _ => {
            let mut jac = (spec.build_jacobian)();
            jac.set_symbolic_assembly_backend(symbolic_backend);
            jac
        }
    }
}

fn build_scenario_data(spec: &ScenarioSpec) -> ScenarioData {
    build_scenario_data_with_backend(spec, BvpSymbolicAssemblyBackend::ExprLegacy)
}

fn build_scenario_data_with_backend(
    spec: &ScenarioSpec,
    symbolic_backend: BvpSymbolicAssemblyBackend,
) -> ScenarioData {
    emit_progress(format!("building symbolic scenario `{}`", spec.label));
    let symbolic_begin = Instant::now();
    let jac = build_jacobian_for_spec(spec, symbolic_backend);
    let symbolic_ms = symbolic_begin.elapsed().as_secs_f64() * 1_000.0;

    let prepared = jac.prepare_sparse_aot_problem(
        "eval_residual",
        "eval_sparse_values",
        crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy::Whole,
        crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy::Whole,
    );
    let args = sample_args(prepared.as_prepared_problem().input_names().len());
    let (lambdify_residual, lambdify_jacobian) = {
        let residual_fns = compile_residual_lambdify(&prepared);
        let jacobian_fns = compile_sparse_lambdify(&prepared);
        (
            eval_functions(&residual_fns, &args),
            eval_functions(&jacobian_fns, &args),
        )
    };

    let scenario = ScenarioData {
        label: spec.label,
        symbolic_backend,
        runtime_iters: spec.runtime_iters,
        runtime_samples: spec.runtime_samples,
        symbolic_ms,
        lambdify_residual,
        lambdify_jacobian,
        prepared,
        args,
    };
    emit_progress(format!(
        "scenario `{}` ready: residuals={}, vars={}, nnz={}",
        scenario.label,
        scenario.prepared.residuals.len(),
        scenario.prepared.shape.1,
        scenario.prepared.sparse_entries.len()
    ));
    scenario
}

fn symbolic_backend_label(backend: BvpSymbolicAssemblyBackend) -> &'static str {
    match backend {
        BvpSymbolicAssemblyBackend::ExprLegacy => "ExprLegacy",
        BvpSymbolicAssemblyBackend::AtomView => "AtomView",
    }
}

#[derive(tabled::Tabled)]
struct CorrectnessRow {
    scenario: String,
    symbolic: String,
    path: String,
    residual_diff: String,
    jacobian_diff: String,
    status: String,
}

#[test]
fn bvp_sparse_banded_correctness_matrix_table() {
    let scenarios = [
        ScenarioSpec {
            label: "small-damp1-24",
            runtime_iters: 1,
            runtime_samples: 1,
            build_jacobian: || build_real_bvp_damp1_case(24),
        },
        ScenarioSpec {
            label: "combustion-1000",
            runtime_iters: 1,
            runtime_samples: 1,
            build_jacobian: || build_combustion_bvp_case(1000),
        },
    ];
    let symbolic_backends = [
        BvpSymbolicAssemblyBackend::ExprLegacy,
        BvpSymbolicAssemblyBackend::AtomView,
    ];
    let mut rows = Vec::new();

    for symbolic_backend in symbolic_backends {
        for spec in &scenarios {
            let scenario = build_scenario_data_with_backend(spec, symbolic_backend);
            let symbolic_label = match symbolic_backend {
                BvpSymbolicAssemblyBackend::ExprLegacy => "ExprLegacy",
                BvpSymbolicAssemblyBackend::AtomView => "AtomView",
            };

            let residual_fns = compile_residual_lambdify(&scenario.prepared);
            let sparse_fns = compile_sparse_lambdify(&scenario.prepared);
            let banded_fns = compile_banded_lambdify(&scenario.prepared);

            let sparse_prepared = scenario.prepared.as_prepared_problem();
            let banded_prepared = scenario.prepared.as_prepared_banded_problem();

            let sparse_lambdify_values = eval_functions(&sparse_fns, &scenario.args);
            let banded_lambdify_values = eval_functions(&banded_fns, &scenario.args);
            let sparse_baseline_dense = sparse_prepared
                .jacobian_plan
                .assemble_sparse_col_mat(&scenario.lambdify_jacobian)
                .to_DMatrixType();
            let banded_lambdify_jacobian = banded_assembly_to_dense(
                &banded_prepared
                    .jacobian_plan
                    .assemble_banded_assembly(&banded_lambdify_values),
            );

            let sparse_ir_values = eval_ir_blocks_sequential(
                &scenario.args,
                &sparse_ir_blocks(&sparse_prepared.jacobian_plan),
                sparse_prepared.jacobian_plan.nnz(),
            );
            let banded_ir_values = eval_ir_blocks_sequential(
                &scenario.args,
                &banded_ir_blocks(&banded_prepared.jacobian_plan),
                banded_prepared.jacobian_plan.nnz(),
            );
            let banded_ir_jacobian = banded_assembly_to_dense(
                &banded_prepared
                    .jacobian_plan
                    .assemble_banded_assembly(&banded_ir_values),
            );

            let residual_ir = eval_ir_blocks_sequential(
                &scenario.args,
                &residual_ir_blocks(&sparse_prepared.residual_plan),
                sparse_prepared.residual_plan.output_len,
            );
            let residual_lambdify = eval_functions(&residual_fns, &scenario.args);

            let sparse_lambdify_diff =
                max_abs_diff(&sparse_lambdify_values, &scenario.lambdify_jacobian);
            let banded_lambdify_diff = max_abs_diff(
                banded_lambdify_jacobian.as_slice(),
                sparse_baseline_dense.as_slice(),
            );
            let sparse_ir_diff = max_abs_diff(&sparse_ir_values, &scenario.lambdify_jacobian);
            let banded_ir_diff = max_abs_diff(
                banded_ir_jacobian.as_slice(),
                sparse_baseline_dense.as_slice(),
            );
            let residual_lambdify_diff =
                max_abs_diff(&residual_lambdify, &scenario.lambdify_residual);
            let residual_ir_diff = max_abs_diff(&residual_ir, &scenario.lambdify_residual);

            rows.push(CorrectnessRow {
                scenario: spec.label.to_string(),
                symbolic: symbolic_label.to_string(),
                path: "Lambdify-Sparse".to_string(),
                residual_diff: format!("{:.3e}", residual_lambdify_diff),
                jacobian_diff: format!("{:.3e}", sparse_lambdify_diff),
                status: "ok".to_string(),
            });
            rows.push(CorrectnessRow {
                scenario: spec.label.to_string(),
                symbolic: symbolic_label.to_string(),
                path: "Lambdify-Banded".to_string(),
                residual_diff: format!("{:.3e}", residual_lambdify_diff),
                jacobian_diff: format!("{:.3e}", banded_lambdify_diff),
                status: "ok".to_string(),
            });
            rows.push(CorrectnessRow {
                scenario: spec.label.to_string(),
                symbolic: symbolic_label.to_string(),
                path: "AOTIR-Sparse".to_string(),
                residual_diff: format!("{:.3e}", residual_ir_diff),
                jacobian_diff: format!("{:.3e}", sparse_ir_diff),
                status: "ok".to_string(),
            });
            rows.push(CorrectnessRow {
                scenario: spec.label.to_string(),
                symbolic: symbolic_label.to_string(),
                path: "AOTIR-Banded".to_string(),
                residual_diff: format!("{:.3e}", residual_ir_diff),
                jacobian_diff: format!("{:.3e}", banded_ir_diff),
                status: "ok".to_string(),
            });

            let tol = if spec.label == "combustion-1000" {
                1.0e-9
            } else {
                1.0e-11
            };
            assert!(sparse_lambdify_diff <= tol);
            assert!(banded_lambdify_diff <= tol);
            assert!(sparse_ir_diff <= tol);
            assert!(banded_ir_diff <= tol);
            assert!(residual_lambdify_diff <= tol);
            assert!(residual_ir_diff <= tol);
        }
    }

    let mut builder = Builder::default();
    builder.push_record([
        "scenario",
        "symbolic",
        "path",
        "residual_diff",
        "jacobian_diff",
        "status",
    ]);
    for row in rows {
        builder.push_record([
            row.scenario,
            row.symbolic,
            row.path,
            row.residual_diff,
            row.jacobian_diff,
            row.status,
        ]);
    }
    let mut table = builder.build();
    table.with(Style::rounded());
    println!("[BVP backend compare] correctness table");
    println!("{table}");
}

fn build_and_link_backend(
    scenario: &ScenarioData,
    variant: &BackendVariant,
) -> BuildAndLinkMetrics {
    emit_progress(format!(
        "building backend `{}` for scenario `{}`",
        variant.label, scenario.label
    ));
    let preset = pipeline_build_preset(scenario, variant);
    let metrics = build_and_link_backend_with_preset(scenario, variant, preset);
    emit_progress(format!(
        "backend `{}` for scenario `{}` finished with status `{}` (build {:.3} ms, total {:.3} ms)",
        variant.label,
        scenario.label,
        metrics.status,
        metrics.build_ms,
        metrics.total_to_outputs_ms
    ));
    metrics
}

fn pipeline_build_preset(scenario: &ScenarioData, variant: &BackendVariant) -> BuildPreset {
    match scenario.label {
        "combustion-1000" | "combustion-3000" => BuildPreset::DevFastest,
        _ => match variant.backend {
            AotCodegenBackend::Rust => BuildPreset::FastBuild,
            _ => BuildPreset::Production,
        },
    }
}

fn failed_build_metrics(
    scenario: &ScenarioData,
    variant: &BackendVariant,
    preset_label: &'static str,
    breakdown: &crate::symbolic::symbolic_functions_BVP::BvpGeneratedAotCrateBreakdown,
    artifact_ms: f64,
    materialize_ms: f64,
    build_ms: f64,
    link_ms: f64,
    status: String,
) -> BuildAndLinkMetrics {
    BuildAndLinkMetrics {
        symbolic_backend: symbolic_backend_label(scenario.symbolic_backend),
        variant_label: variant.label.clone(),
        preset_label,
        artifact_ms,
        jacobian_prepare_ms: breakdown.jacobian_prepare_ms,
        atom_sparse_lookup_prepare_ms: breakdown.atom_sparse_lookup_prepare_ms,
        atom_sparse_jacobian_build_ms: breakdown.atom_sparse_jacobian_build_ms,
        atom_finalize_codegen_plan_ms: breakdown.atom_finalize_codegen_plan_ms,
        atom_lower_many_ms: breakdown.atom_residual_lower_many_ms
            + breakdown.atom_sparse_lower_many_ms,
        atom_peephole_ms: breakdown.atom_residual_peephole_ms + breakdown.atom_sparse_peephole_ms,
        atom_reuse_temps_ms: breakdown.atom_residual_reuse_temps_ms
            + breakdown.atom_sparse_reuse_temps_ms,
        atom_push_ms: breakdown.atom_residual_push_ms + breakdown.atom_sparse_push_ms,
        prepared_module_init_ms: breakdown.prepared_module_init_ms,
        prepared_residual_blocks_ms: breakdown.prepared_residual_blocks_ms,
        prepared_jacobian_blocks_ms: breakdown.prepared_jacobian_blocks_ms,
        module_ms: breakdown.module_build_ms,
        source_ms: breakdown.source_emit_ms,
        source_probe_emit_ms: breakdown.source_probe_emit_ms,
        language_source_emit_ms: breakdown.language_source_emit_ms,
        c_header_emit_ms: breakdown.c_header_emit_ms,
        artifact_packaging_ms: breakdown.artifact_packaging_ms,
        source_kb: breakdown.source_kb,
        materialize_ms,
        build_ms,
        link_ms,
        first_issue_ms: 0.0,
        total_to_outputs_ms: scenario.symbolic_ms
            + artifact_ms
            + materialize_ms
            + build_ms
            + link_ms,
        residual_diff: f64::NAN,
        jacobian_diff: f64::NAN,
        status,
        linked: None,
    }
}

fn summarize_process_failure(status_code: Option<i32>, stdout: &str, stderr: &str) -> String {
    let stderr = stderr.trim();
    let stdout = stdout.trim();
    let details = if !stderr.is_empty() {
        format!("stderr={stderr}")
    } else if !stdout.is_empty() {
        format!("stdout={stdout}")
    } else {
        "no compiler output".to_string()
    };
    format!("build failed: status={status_code:?}, {details}")
}

fn successful_build_metrics(
    scenario: &ScenarioData,
    variant: &BackendVariant,
    preset_label: &'static str,
    breakdown: &crate::symbolic::symbolic_functions_BVP::BvpGeneratedAotCrateBreakdown,
    artifact_ms: f64,
    materialize_ms: f64,
    build_ms: f64,
    link_ms: f64,
    linked: LinkedSparseAotBackend,
) -> BuildAndLinkMetrics {
    let issue_begin = Instant::now();
    let mut residual_out = vec![0.0; linked.residual_len];
    (linked.residual_eval)(scenario.args.as_slice(), &mut residual_out);
    let mut jacobian_out = vec![0.0; linked.nnz];
    (linked.jacobian_values_eval)(scenario.args.as_slice(), &mut jacobian_out);
    let first_issue_ms = issue_begin.elapsed().as_secs_f64() * 1_000.0;
    BuildAndLinkMetrics {
        symbolic_backend: symbolic_backend_label(scenario.symbolic_backend),
        variant_label: variant.label.clone(),
        preset_label,
        artifact_ms,
        jacobian_prepare_ms: breakdown.jacobian_prepare_ms,
        atom_sparse_lookup_prepare_ms: breakdown.atom_sparse_lookup_prepare_ms,
        atom_sparse_jacobian_build_ms: breakdown.atom_sparse_jacobian_build_ms,
        atom_finalize_codegen_plan_ms: breakdown.atom_finalize_codegen_plan_ms,
        atom_lower_many_ms: breakdown.atom_residual_lower_many_ms
            + breakdown.atom_sparse_lower_many_ms,
        atom_peephole_ms: breakdown.atom_residual_peephole_ms + breakdown.atom_sparse_peephole_ms,
        atom_reuse_temps_ms: breakdown.atom_residual_reuse_temps_ms
            + breakdown.atom_sparse_reuse_temps_ms,
        atom_push_ms: breakdown.atom_residual_push_ms + breakdown.atom_sparse_push_ms,
        prepared_module_init_ms: breakdown.prepared_module_init_ms,
        prepared_residual_blocks_ms: breakdown.prepared_residual_blocks_ms,
        prepared_jacobian_blocks_ms: breakdown.prepared_jacobian_blocks_ms,
        module_ms: breakdown.module_build_ms,
        source_ms: breakdown.source_emit_ms,
        source_probe_emit_ms: breakdown.source_probe_emit_ms,
        language_source_emit_ms: breakdown.language_source_emit_ms,
        c_header_emit_ms: breakdown.c_header_emit_ms,
        artifact_packaging_ms: breakdown.artifact_packaging_ms,
        source_kb: breakdown.source_kb,
        materialize_ms,
        build_ms,
        link_ms,
        first_issue_ms,
        total_to_outputs_ms: scenario.symbolic_ms
            + artifact_ms
            + materialize_ms
            + build_ms
            + link_ms
            + first_issue_ms,
        residual_diff: max_abs_diff(&residual_out, &scenario.lambdify_residual),
        jacobian_diff: max_abs_diff(&jacobian_out, &scenario.lambdify_jacobian),
        status: "ok".to_string(),
        linked: Some(linked),
    }
}

fn lambdify_pipeline_sample(scenario: &ScenarioData) -> PipelineBootstrapSample {
    let callable_begin = Instant::now();
    let residual_fns = compile_residual_lambdify(&scenario.prepared);
    let jacobian_fns = compile_sparse_lambdify(&scenario.prepared);
    let callable_prep_ms = callable_begin.elapsed().as_secs_f64() * 1_000.0;

    let first_issue_begin = Instant::now();
    let residual_out = eval_functions(&residual_fns, scenario.args.as_slice());
    let jacobian_out = eval_functions(&jacobian_fns, scenario.args.as_slice());
    let first_issue_ms = first_issue_begin.elapsed().as_secs_f64() * 1_000.0;

    PipelineBootstrapSample {
        route: "Lambdify",
        symbolic_backend: symbolic_backend_label(scenario.symbolic_backend),
        variant_label: "Lambdify".to_string(),
        preset_label: "n/a",
        symbolic_ms: scenario.symbolic_ms,
        callable_prep_ms,
        artifact_ms: 0.0,
        jacobian_prepare_ms: 0.0,
        atom_sparse_lookup_prepare_ms: 0.0,
        atom_sparse_jacobian_build_ms: 0.0,
        atom_finalize_codegen_plan_ms: 0.0,
        atom_lower_many_ms: 0.0,
        atom_peephole_ms: 0.0,
        atom_reuse_temps_ms: 0.0,
        atom_push_ms: 0.0,
        prepared_module_init_ms: 0.0,
        prepared_residual_blocks_ms: 0.0,
        prepared_jacobian_blocks_ms: 0.0,
        module_ms: 0.0,
        source_ms: 0.0,
        source_probe_emit_ms: 0.0,
        language_source_emit_ms: 0.0,
        c_header_emit_ms: 0.0,
        artifact_packaging_ms: 0.0,
        source_kb: 0.0,
        materialize_ms: 0.0,
        build_ms: 0.0,
        link_ms: 0.0,
        first_issue_ms,
        total_to_outputs_ms: scenario.symbolic_ms + callable_prep_ms + first_issue_ms,
        residual_diff: max_abs_diff(&residual_out, &scenario.lambdify_residual),
        jacobian_diff: max_abs_diff(&jacobian_out, &scenario.lambdify_jacobian),
        status: "ok".to_string(),
    }
}

fn atom_profile_artifact_pipeline_sample(
    scenario: &ScenarioData,
    backend: AotCodegenBackend,
    profile: AtomOptimizationProfile,
) -> PipelineBootstrapSample {
    let variant_label = match backend {
        AotCodegenBackend::Rust => "rust",
        AotCodegenBackend::C => "c_source",
        AotCodegenBackend::Zig => "zig_source",
    };
    let artifact_name = format!("profile_{}_{}", variant_label, profile.label());
    let module_name = format!("profile_module_{}_{}", variant_label, profile.label());
    let artifact_begin = Instant::now();
    let (_artifact, breakdown) = scenario
        .prepared
        .generated_aot_artifact_with_breakdown_for_matrix_backend_and_atom_profile(
            artifact_name,
            &module_name,
            backend,
            MatrixBackend::SparseCol,
            profile,
        );
    let artifact_ms = artifact_begin.elapsed().as_secs_f64() * 1_000.0;

    PipelineBootstrapSample {
        route: "AOT-profile",
        symbolic_backend: symbolic_backend_label(scenario.symbolic_backend),
        variant_label: format!("{variant_label}/{}", profile.label()),
        preset_label: "artifact_only",
        symbolic_ms: scenario.symbolic_ms,
        callable_prep_ms: artifact_ms,
        artifact_ms,
        jacobian_prepare_ms: breakdown.jacobian_prepare_ms,
        atom_sparse_lookup_prepare_ms: breakdown.atom_sparse_lookup_prepare_ms,
        atom_sparse_jacobian_build_ms: breakdown.atom_sparse_jacobian_build_ms,
        atom_finalize_codegen_plan_ms: breakdown.atom_finalize_codegen_plan_ms,
        atom_lower_many_ms: breakdown.atom_residual_lower_many_ms
            + breakdown.atom_sparse_lower_many_ms,
        atom_peephole_ms: breakdown.atom_residual_peephole_ms + breakdown.atom_sparse_peephole_ms,
        atom_reuse_temps_ms: breakdown.atom_residual_reuse_temps_ms
            + breakdown.atom_sparse_reuse_temps_ms,
        atom_push_ms: breakdown.atom_residual_push_ms + breakdown.atom_sparse_push_ms,
        prepared_module_init_ms: breakdown.prepared_module_init_ms,
        prepared_residual_blocks_ms: breakdown.prepared_residual_blocks_ms,
        prepared_jacobian_blocks_ms: breakdown.prepared_jacobian_blocks_ms,
        module_ms: breakdown.module_build_ms,
        source_ms: breakdown.source_emit_ms,
        source_probe_emit_ms: breakdown.source_probe_emit_ms,
        language_source_emit_ms: breakdown.language_source_emit_ms,
        c_header_emit_ms: breakdown.c_header_emit_ms,
        artifact_packaging_ms: breakdown.artifact_packaging_ms,
        source_kb: breakdown.source_kb,
        materialize_ms: 0.0,
        build_ms: 0.0,
        link_ms: 0.0,
        first_issue_ms: 0.0,
        total_to_outputs_ms: scenario.symbolic_ms + artifact_ms,
        residual_diff: 0.0,
        jacobian_diff: 0.0,
        status: "ok".to_string(),
    }
}

#[test]
fn generated_aot_artifact_skips_source_probe_without_changing_c_source() {
    let spec = ScenarioSpec {
        label: "small-damp1-probe-skip",
        runtime_iters: 1,
        runtime_samples: 1,
        build_jacobian: || build_real_bvp_damp1_case(8),
    };
    let scenario = build_scenario_data(&spec);

    let (mut probed_module, probed_breakdown) = scenario
        .prepared
        .codegen_module_with_breakdown_for_matrix_backend(
            "generated_probe_skip",
            MatrixBackend::SparseCol,
        );
    probed_module.set_language(crate::symbolic::codegen::CodegenIR::CodegenLanguage::C);
    let expected_c_source = probed_module.emit_source();

    let (artifact, artifact_breakdown) = scenario
        .prepared
        .generated_aot_artifact_with_breakdown_for_matrix_backend(
            "generated_probe_skip_artifact",
            "generated_probe_skip",
            AotCodegenBackend::C,
            MatrixBackend::SparseCol,
        );

    assert!(
        probed_breakdown.source_probe_emit_ms >= 0.0,
        "module-only diagnostic path should still own the source probe timing"
    );
    assert_eq!(
        artifact_breakdown.source_probe_emit_ms, 0.0,
        "artifact path should not emit source once for a diagnostic probe and once for the real artifact"
    );
    assert!(
        artifact_breakdown.language_source_emit_ms >= 0.0,
        "artifact path should record the one real language-source emission"
    );
    assert!(
        artifact_breakdown.source_kb > 0.0,
        "artifact path should fill source_kb from the real emitted source"
    );

    let GeneratedAotArtifact::C(c_artifact) = artifact else {
        panic!("C backend should produce a generated C artifact");
    };
    assert_eq!(
        c_artifact.c_source, expected_c_source,
        "skipping the diagnostic probe must not change generated C source"
    );
    assert!(c_artifact.c_source.contains("eval_residual"));
    assert!(c_artifact.c_source.contains("eval_sparse_values"));
}

fn compiled_pipeline_sample(
    metrics: &BuildAndLinkMetrics,
    symbolic_ms: f64,
) -> PipelineBootstrapSample {
    PipelineBootstrapSample {
        route: "AOT",
        symbolic_backend: metrics.symbolic_backend,
        variant_label: metrics.variant_label.clone(),
        preset_label: metrics.preset_label,
        symbolic_ms,
        callable_prep_ms: metrics.artifact_ms
            + metrics.materialize_ms
            + metrics.build_ms
            + metrics.link_ms,
        artifact_ms: metrics.artifact_ms,
        jacobian_prepare_ms: metrics.jacobian_prepare_ms,
        atom_sparse_lookup_prepare_ms: metrics.atom_sparse_lookup_prepare_ms,
        atom_sparse_jacobian_build_ms: metrics.atom_sparse_jacobian_build_ms,
        atom_finalize_codegen_plan_ms: metrics.atom_finalize_codegen_plan_ms,
        atom_lower_many_ms: metrics.atom_lower_many_ms,
        atom_peephole_ms: metrics.atom_peephole_ms,
        atom_reuse_temps_ms: metrics.atom_reuse_temps_ms,
        atom_push_ms: metrics.atom_push_ms,
        prepared_module_init_ms: metrics.prepared_module_init_ms,
        prepared_residual_blocks_ms: metrics.prepared_residual_blocks_ms,
        prepared_jacobian_blocks_ms: metrics.prepared_jacobian_blocks_ms,
        module_ms: metrics.module_ms,
        source_ms: metrics.source_ms,
        source_probe_emit_ms: metrics.source_probe_emit_ms,
        language_source_emit_ms: metrics.language_source_emit_ms,
        c_header_emit_ms: metrics.c_header_emit_ms,
        artifact_packaging_ms: metrics.artifact_packaging_ms,
        source_kb: metrics.source_kb,
        materialize_ms: metrics.materialize_ms,
        build_ms: metrics.build_ms,
        link_ms: metrics.link_ms,
        first_issue_ms: metrics.first_issue_ms,
        total_to_outputs_ms: metrics.total_to_outputs_ms,
        residual_diff: metrics.residual_diff,
        jacobian_diff: metrics.jacobian_diff,
        status: metrics.status.clone(),
    }
}

fn pipeline_metric_aggregate(values: impl IntoIterator<Item = f64>) -> PipelineMetricAggregate {
    let values = values
        .into_iter()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return PipelineMetricAggregate {
            count: 0,
            mean: f64::NAN,
            std: f64::NAN,
            min: f64::NAN,
            max: f64::NAN,
        };
    }
    let count = values.len();
    let mean = values.iter().sum::<f64>() / count as f64;
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / count as f64;
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    PipelineMetricAggregate {
        count,
        mean,
        std: variance.sqrt(),
        min,
        max,
    }
}

fn fmt_pipeline_metric(value: PipelineMetricAggregate) -> String {
    if value.count == 0 {
        "-".to_string()
    } else if value.count == 1 {
        format!("{:.3}", value.mean)
    } else {
        format!(
            "{:.3}+/-{:.3} [{:.3},{:.3}]",
            value.mean, value.std, value.min, value.max
        )
    }
}

fn fmt_pipeline_short(value: PipelineMetricAggregate) -> String {
    if value.count == 0 {
        "-".to_string()
    } else if value.count == 1 {
        format!("{:.3}", value.mean)
    } else {
        format!("{:.3}+/-{:.3}", value.mean, value.std)
    }
}

fn summarize_pipeline_status(samples: &[&PipelineBootstrapSample]) -> String {
    let ok_runs = samples
        .iter()
        .filter(|sample| sample.status == "ok")
        .count();
    if ok_runs == samples.len() {
        format!("ok {ok_runs}/{}", samples.len())
    } else {
        let first_failure = samples
            .iter()
            .find(|sample| sample.status != "ok")
            .map(|sample| sample.status.as_str())
            .unwrap_or("unknown");
        format!(
            "ok {ok_runs}/{}, first_failure={first_failure}",
            samples.len()
        )
    }
}

fn aggregate_pipeline_samples(
    samples: &[PipelineBootstrapSample],
) -> Vec<PipelineBootstrapAggregate> {
    let mut keys = Vec::<(&'static str, &'static str, String, &'static str)>::new();
    for sample in samples {
        let key = (
            sample.route,
            sample.symbolic_backend,
            sample.variant_label.clone(),
            sample.preset_label,
        );
        if !keys.contains(&key) {
            keys.push(key);
        }
    }
    keys.into_iter()
        .map(|(route, symbolic_backend, variant_label, preset_label)| {
            let rows = samples
                .iter()
                .filter(|sample| {
                    sample.route == route
                        && sample.symbolic_backend == symbolic_backend
                        && sample.variant_label == variant_label
                        && sample.preset_label == preset_label
                })
                .collect::<Vec<_>>();
            let ok_rows = rows
                .iter()
                .copied()
                .filter(|sample| sample.status == "ok")
                .collect::<Vec<_>>();
            PipelineBootstrapAggregate {
                route,
                symbolic_backend,
                variant_label,
                preset_label,
                runs: rows.len(),
                ok_runs: ok_rows.len(),
                symbolic_ms: pipeline_metric_aggregate(
                    rows.iter().map(|sample| sample.symbolic_ms),
                ),
                callable_prep_ms: pipeline_metric_aggregate(
                    rows.iter().map(|sample| sample.callable_prep_ms),
                ),
                artifact_ms: pipeline_metric_aggregate(
                    rows.iter().map(|sample| sample.artifact_ms),
                ),
                jacobian_prepare_ms: pipeline_metric_aggregate(
                    rows.iter().map(|sample| sample.jacobian_prepare_ms),
                ),
                atom_sparse_lookup_prepare_ms: pipeline_metric_aggregate(
                    rows.iter()
                        .map(|sample| sample.atom_sparse_lookup_prepare_ms),
                ),
                atom_sparse_jacobian_build_ms: pipeline_metric_aggregate(
                    rows.iter()
                        .map(|sample| sample.atom_sparse_jacobian_build_ms),
                ),
                atom_finalize_codegen_plan_ms: pipeline_metric_aggregate(
                    rows.iter()
                        .map(|sample| sample.atom_finalize_codegen_plan_ms),
                ),
                atom_lower_many_ms: pipeline_metric_aggregate(
                    rows.iter().map(|sample| sample.atom_lower_many_ms),
                ),
                atom_peephole_ms: pipeline_metric_aggregate(
                    rows.iter().map(|sample| sample.atom_peephole_ms),
                ),
                atom_reuse_temps_ms: pipeline_metric_aggregate(
                    rows.iter().map(|sample| sample.atom_reuse_temps_ms),
                ),
                atom_push_ms: pipeline_metric_aggregate(
                    rows.iter().map(|sample| sample.atom_push_ms),
                ),
                prepared_module_init_ms: pipeline_metric_aggregate(
                    rows.iter().map(|sample| sample.prepared_module_init_ms),
                ),
                prepared_residual_blocks_ms: pipeline_metric_aggregate(
                    rows.iter().map(|sample| sample.prepared_residual_blocks_ms),
                ),
                prepared_jacobian_blocks_ms: pipeline_metric_aggregate(
                    rows.iter().map(|sample| sample.prepared_jacobian_blocks_ms),
                ),
                module_ms: pipeline_metric_aggregate(rows.iter().map(|sample| sample.module_ms)),
                source_ms: pipeline_metric_aggregate(rows.iter().map(|sample| sample.source_ms)),
                source_probe_emit_ms: pipeline_metric_aggregate(
                    rows.iter().map(|sample| sample.source_probe_emit_ms),
                ),
                language_source_emit_ms: pipeline_metric_aggregate(
                    rows.iter().map(|sample| sample.language_source_emit_ms),
                ),
                c_header_emit_ms: pipeline_metric_aggregate(
                    rows.iter().map(|sample| sample.c_header_emit_ms),
                ),
                artifact_packaging_ms: pipeline_metric_aggregate(
                    rows.iter().map(|sample| sample.artifact_packaging_ms),
                ),
                artifact_other_ms: pipeline_metric_aggregate(rows.iter().map(|sample| {
                    (sample.artifact_ms
                        - sample.module_ms
                        - sample.source_probe_emit_ms
                        - sample.language_source_emit_ms
                        - sample.c_header_emit_ms
                        - sample.artifact_packaging_ms)
                        .max(0.0)
                })),
                source_kb: pipeline_metric_aggregate(rows.iter().map(|sample| sample.source_kb)),
                materialize_ms: pipeline_metric_aggregate(
                    rows.iter().map(|sample| sample.materialize_ms),
                ),
                build_ms: pipeline_metric_aggregate(rows.iter().map(|sample| sample.build_ms)),
                link_ms: pipeline_metric_aggregate(rows.iter().map(|sample| sample.link_ms)),
                first_issue_ms: pipeline_metric_aggregate(
                    rows.iter().map(|sample| sample.first_issue_ms),
                ),
                total_to_outputs_ms: pipeline_metric_aggregate(
                    rows.iter().map(|sample| sample.total_to_outputs_ms),
                ),
                residual_diff: pipeline_metric_aggregate(
                    ok_rows.iter().map(|sample| sample.residual_diff),
                ),
                jacobian_diff: pipeline_metric_aggregate(
                    ok_rows.iter().map(|sample| sample.jacobian_diff),
                ),
                status: summarize_pipeline_status(&rows),
            }
        })
        .collect()
}

fn pipeline_sample_runs() -> usize {
    env::var("BVP_PIPELINE_RUNS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|runs| *runs > 0)
        .unwrap_or(3)
}

fn print_pipeline_bootstrap_summary_table(
    scenario: &ScenarioData,
    rows: &[PipelineBootstrapAggregate],
) {
    println!(
        "[BVP backend pipeline compare] scenario={}, residuals={}, vars={}, nnz={}, multi-run bootstrap summary",
        scenario.label,
        scenario.prepared.residuals.len(),
        scenario.prepared.shape.1,
        scenario.prepared.sparse_entries.len()
    );
    println!(
        "route    | assembly   | variant        | preset      | ok/runs | symbolic_ms mean+/-std [min,max] | callable_prep_ms mean+/-std [min,max] | artifact_ms | materialize_ms | build_ms | link_ms | first_issue_ms | total_to_outputs_ms mean+/-std [min,max] | status"
    );
    println!("{}", "-".repeat(273));
    for row in rows {
        println!(
            "{:<8} | {:<10} | {:<14} | {:<11} | {:>3}/{:<3} | {:<34} | {:<39} | {:<11} | {:<14} | {:<8} | {:<7} | {:<14} | {:<43} | {}",
            row.route,
            row.symbolic_backend,
            row.variant_label,
            row.preset_label,
            row.ok_runs,
            row.runs,
            fmt_pipeline_metric(row.symbolic_ms),
            fmt_pipeline_metric(row.callable_prep_ms),
            fmt_pipeline_short(row.artifact_ms),
            fmt_pipeline_short(row.materialize_ms),
            fmt_pipeline_short(row.build_ms),
            fmt_pipeline_short(row.link_ms),
            fmt_pipeline_short(row.first_issue_ms),
            fmt_pipeline_metric(row.total_to_outputs_ms),
            row.status,
        );
    }
    flush_stdout();
}

fn print_pipeline_codegen_summary_table(rows: &[PipelineBootstrapAggregate]) {
    println!(
        "AtomView-only planning stages. ExprLegacy rows are expected to be zero here; use the module/source table below for the active legacy module-build cost."
    );
    println!(
        "route    | assembly   | variant        | preset      | jac_prepare | lookup | jac_build | chunk_plan | lower | peephole | temp_reuse | module_push"
    );
    println!("{}", "-".repeat(145));
    for row in rows {
        println!(
            "{:<8} | {:<10} | {:<14} | {:<11} | {:<11} | {:<6} | {:<9} | {:<10} | {:<6} | {:<8} | {:<10} | {:<11}",
            row.route,
            row.symbolic_backend,
            row.variant_label,
            row.preset_label,
            fmt_pipeline_short(row.jacobian_prepare_ms),
            fmt_pipeline_short(row.atom_sparse_lookup_prepare_ms),
            fmt_pipeline_short(row.atom_sparse_jacobian_build_ms),
            fmt_pipeline_short(row.atom_finalize_codegen_plan_ms),
            fmt_pipeline_short(row.atom_lower_many_ms),
            fmt_pipeline_short(row.atom_peephole_ms),
            fmt_pipeline_short(row.atom_reuse_temps_ms),
            fmt_pipeline_short(row.atom_push_ms),
        );
    }
    println!(
        "route    | assembly   | variant        | preset      | module_ms | module_init | residual_lower | jacobian_lower | source_probe | source_emit | c_header | packaging | artifact_other | source_kb"
    );
    println!("{}", "-".repeat(205));
    for row in rows {
        println!(
            "{:<8} | {:<10} | {:<14} | {:<11} | {:<9} | {:<11} | {:<14} | {:<14} | {:<12} | {:<11} | {:<8} | {:<9} | {:<14} | {:<9}",
            row.route,
            row.symbolic_backend,
            row.variant_label,
            row.preset_label,
            fmt_pipeline_short(row.module_ms),
            fmt_pipeline_short(row.prepared_module_init_ms),
            fmt_pipeline_short(row.prepared_residual_blocks_ms),
            fmt_pipeline_short(row.prepared_jacobian_blocks_ms),
            fmt_pipeline_short(row.source_probe_emit_ms),
            fmt_pipeline_short(row.language_source_emit_ms),
            fmt_pipeline_short(row.c_header_emit_ms),
            fmt_pipeline_short(row.artifact_packaging_ms),
            fmt_pipeline_short(row.artifact_other_ms),
            fmt_pipeline_short(row.source_kb),
        );
    }
    flush_stdout();
}

fn print_pipeline_correctness_summary_table(rows: &[PipelineBootstrapAggregate]) {
    println!("route    | assembly   | variant        | preset      | residual_diff | jacobian_diff | status");
    println!("{}", "-".repeat(117));
    for row in rows {
        println!(
            "{:<8} | {:<10} | {:<14} | {:<11} | {:<13.6e} | {:<13.6e} | {}",
            row.route,
            row.symbolic_backend,
            row.variant_label,
            row.preset_label,
            row.residual_diff.max,
            row.jacobian_diff.max,
            row.status,
        );
    }
    flush_stdout();
}

fn print_runtime_table(scenario: &ScenarioData, rows: &[RuntimeRow]) {
    println!(
        "[BVP backend runtime compare] scenario={}, residuals={}, vars={}, nnz={}, iters={}, samples={}",
        scenario.label,
        scenario.prepared.residuals.len(),
        scenario.prepared.shape.1,
        scenario.prepared.sparse_entries.len(),
        scenario.runtime_iters,
        scenario.runtime_samples
    );
    println!(
        "variant        | residual_ms(avg) | jacobian_ms(avg) | total_ms(avg) | speedup_vs_lambdify | residual_diff | jacobian_diff | status"
    );
    println!("{}", "-".repeat(160));
    let baseline = rows
        .iter()
        .find(|row| row.label == "Lambdify")
        .map(|row| row.total_ms)
        .unwrap_or(1.0);
    for row in rows {
        let speedup = if row.total_ms.is_finite() && row.total_ms > 0.0 {
            baseline / row.total_ms
        } else {
            f64::NAN
        };
        println!(
            "{:<14} | {:>11.3} | {:>11.3} | {:>8.3} | {:>18.3}x | {:>13.6e} | {:>13.6e} | {}",
            row.label,
            row.residual_ms,
            row.jacobian_ms,
            row.total_ms,
            speedup,
            row.residual_diff,
            row.jacobian_diff,
            row.status,
        );
    }
    flush_stdout();
}

#[derive(Clone, Copy)]
enum BuildPreset {
    Production,
    FastBuild,
    DevFastest,
}

impl BuildPreset {
    fn label(self) -> &'static str {
        match self {
            Self::Production => "Production",
            Self::FastBuild => "FastBuild",
            Self::DevFastest => "DevFastest",
        }
    }
}

struct PresetRuntimeRow {
    variant: String,
    preset: &'static str,
    build_ms: f64,
    total_to_outputs_ms: f64,
    residual_ms: f64,
    jacobian_ms: f64,
    total_ms: f64,
    speedup_vs_baseline: f64,
    residual_diff: f64,
    jacobian_diff: f64,
    status: String,
}

fn active_presets_for_variant(variant: &BackendVariant) -> Vec<BuildPreset> {
    match variant.backend {
        AotCodegenBackend::Rust => vec![BuildPreset::FastBuild, BuildPreset::DevFastest],
        _ => vec![
            BuildPreset::Production,
            BuildPreset::FastBuild,
            BuildPreset::DevFastest,
        ],
    }
}

fn active_presets_for_scenario_variant(
    scenario: &ScenarioData,
    variant: &BackendVariant,
) -> Vec<BuildPreset> {
    match scenario.label {
        "combustion-1000" => vec![BuildPreset::DevFastest],
        _ => active_presets_for_variant(variant),
    }
}

fn build_and_link_backend_with_preset(
    scenario: &ScenarioData,
    variant: &BackendVariant,
    preset: BuildPreset,
) -> BuildAndLinkMetrics {
    let scenario_tag = short_scenario_label(scenario.label);
    let variant_tag = short_variant_label(variant);
    let preset_tag = short_preset_label(preset);
    let artifact_name = format!("gbc_{}_{}_{}", scenario_tag, variant_tag, preset_tag);
    let module_name = format!("gbcm_{}_{}_{}", scenario_tag, variant_tag, preset_tag);
    let output_parent_dir =
        unique_compare_artifact_dir(scenario_tag, &format!("{variant_tag}-{preset_tag}"));

    let artifact_begin = Instant::now();
    let (artifact, breakdown) = scenario.prepared.generated_aot_artifact_with_breakdown(
        &artifact_name,
        &module_name,
        variant.backend,
    );
    let artifact_ms = artifact_begin.elapsed().as_secs_f64() * 1_000.0;

    let manifest = PreparedProblemManifest::from(&PreparedProblem::sparse(
        scenario.prepared.as_prepared_problem(),
    ));
    let mut registry = AotRegistry::new();

    match (variant.backend, artifact) {
        (AotCodegenBackend::Rust, GeneratedAotArtifact::Rust(crate_spec)) => {
            let compile = match preset {
                BuildPreset::Production => AotCompileConfig::production(),
                BuildPreset::FastBuild => AotCompileConfig::fast_build(),
                BuildPreset::DevFastest => AotCompileConfig::dev_fastest(),
            };
            let materialize_begin = Instant::now();
            let build = match AotBuildRequest::new(
                crate_spec,
                &output_parent_dir,
                AotBuildProfile::Release,
            )
            .with_compile_config(compile)
            .materialize()
            {
                Ok(build) => build,
                Err(err) => {
                    return failed_build_metrics(
                        scenario,
                        variant,
                        preset.label(),
                        &breakdown,
                        artifact_ms,
                        0.0,
                        0.0,
                        0.0,
                        format!("materialize failed: {err}"),
                    )
                }
            };
            let materialize_ms = materialize_begin.elapsed().as_secs_f64() * 1_000.0;
            let build_begin = Instant::now();
            let executed = match build.execute() {
                Ok(executed) => executed,
                Err(err) => {
                    return failed_build_metrics(
                        scenario,
                        variant,
                        preset.label(),
                        &breakdown,
                        artifact_ms,
                        materialize_ms,
                        build_begin.elapsed().as_secs_f64() * 1_000.0,
                        0.0,
                        format!("build execute failed: {err}"),
                    )
                }
            };
            let build_ms = build_begin.elapsed().as_secs_f64() * 1_000.0;
            if !executed.succeeded() {
                return failed_build_metrics(
                    scenario,
                    variant,
                    preset.label(),
                    &breakdown,
                    artifact_ms,
                    materialize_ms,
                    build_ms,
                    0.0,
                    summarize_process_failure(
                        executed.status_code,
                        &executed.stdout,
                        &executed.stderr,
                    ),
                );
            }
            let registered = registry
                .register_materialized_build(manifest, &build)
                .clone();
            let link_begin = Instant::now();
            let linked = match register_generated_sparse_cdylib_backend(&registered) {
                Ok(linked) => linked,
                Err(err) => {
                    return failed_build_metrics(
                        scenario,
                        variant,
                        preset.label(),
                        &breakdown,
                        artifact_ms,
                        materialize_ms,
                        build_ms,
                        link_begin.elapsed().as_secs_f64() * 1_000.0,
                        format!("link failed: {err}"),
                    )
                }
            };
            let link_ms = link_begin.elapsed().as_secs_f64() * 1_000.0;
            successful_build_metrics(
                scenario,
                variant,
                preset.label(),
                &breakdown,
                artifact_ms,
                materialize_ms,
                build_ms,
                link_ms,
                linked,
            )
        }
        (AotCodegenBackend::C, GeneratedAotArtifact::C(library_spec)) => {
            let compiler = variant
                .c_compiler
                .clone()
                .unwrap_or_else(|| "gcc".to_string());
            let compile = match preset {
                BuildPreset::Production => CAotCompileConfig::production(),
                BuildPreset::FastBuild => CAotCompileConfig::fast_build(),
                BuildPreset::DevFastest => CAotCompileConfig::dev_fastest(),
            }
            .with_compiler(compiler);
            let materialize_begin = Instant::now();
            let build = match CAotBuildRequest::new(
                library_spec,
                &output_parent_dir,
                CAotBuildProfile::Release,
            )
            .with_compile_config(compile)
            .materialize()
            {
                Ok(build) => build,
                Err(err) => {
                    return failed_build_metrics(
                        scenario,
                        variant,
                        preset.label(),
                        &breakdown,
                        artifact_ms,
                        0.0,
                        0.0,
                        0.0,
                        format!("materialize failed: {err}"),
                    )
                }
            };
            let materialize_ms = materialize_begin.elapsed().as_secs_f64() * 1_000.0;
            let build_begin = Instant::now();
            let executed = match build.execute() {
                Ok(executed) => executed,
                Err(err) => {
                    return failed_build_metrics(
                        scenario,
                        variant,
                        preset.label(),
                        &breakdown,
                        artifact_ms,
                        materialize_ms,
                        build_begin.elapsed().as_secs_f64() * 1_000.0,
                        0.0,
                        format!("build execute failed: {err}"),
                    )
                }
            };
            let build_ms = build_begin.elapsed().as_secs_f64() * 1_000.0;
            if !executed.succeeded() {
                return failed_build_metrics(
                    scenario,
                    variant,
                    preset.label(),
                    &breakdown,
                    artifact_ms,
                    materialize_ms,
                    build_ms,
                    0.0,
                    summarize_process_failure(
                        executed.status_code,
                        &executed.stdout,
                        &executed.stderr,
                    ),
                );
            }
            let registered = register_c_build_in_registry(&mut registry, manifest, &build).clone();
            let link_begin = Instant::now();
            let linked = match register_generated_c_sparse_backend(&registered) {
                Ok(linked) => linked,
                Err(err) => {
                    return failed_build_metrics(
                        scenario,
                        variant,
                        preset.label(),
                        &breakdown,
                        artifact_ms,
                        materialize_ms,
                        build_ms,
                        link_begin.elapsed().as_secs_f64() * 1_000.0,
                        format!("link failed: {err}"),
                    )
                }
            };
            let link_ms = link_begin.elapsed().as_secs_f64() * 1_000.0;
            successful_build_metrics(
                scenario,
                variant,
                preset.label(),
                &breakdown,
                artifact_ms,
                materialize_ms,
                build_ms,
                link_ms,
                linked,
            )
        }
        (AotCodegenBackend::Zig, GeneratedAotArtifact::Zig(library_spec)) => {
            let profile = match preset {
                BuildPreset::Production => ZigAotBuildProfile::ReleaseFast,
                BuildPreset::FastBuild => ZigAotBuildProfile::ReleaseSmall,
                BuildPreset::DevFastest => ZigAotBuildProfile::Debug,
            };
            let materialize_begin = Instant::now();
            let build = match ZigAotBuildRequest::new(library_spec, &output_parent_dir, profile)
                .materialize()
            {
                Ok(build) => build,
                Err(err) => {
                    return failed_build_metrics(
                        scenario,
                        variant,
                        preset.label(),
                        &breakdown,
                        artifact_ms,
                        0.0,
                        0.0,
                        0.0,
                        format!("materialize failed: {err}"),
                    )
                }
            };
            let materialize_ms = materialize_begin.elapsed().as_secs_f64() * 1_000.0;
            let build_begin = Instant::now();
            let executed = match build.execute() {
                Ok(executed) => executed,
                Err(err) => {
                    return failed_build_metrics(
                        scenario,
                        variant,
                        preset.label(),
                        &breakdown,
                        artifact_ms,
                        materialize_ms,
                        build_begin.elapsed().as_secs_f64() * 1_000.0,
                        0.0,
                        format!("build execute failed: {err}"),
                    )
                }
            };
            let build_ms = build_begin.elapsed().as_secs_f64() * 1_000.0;
            if !executed.succeeded() {
                return failed_build_metrics(
                    scenario,
                    variant,
                    preset.label(),
                    &breakdown,
                    artifact_ms,
                    materialize_ms,
                    build_ms,
                    0.0,
                    summarize_process_failure(
                        executed.status_code,
                        &executed.stdout,
                        &executed.stderr,
                    ),
                );
            }
            let registered =
                register_zig_build_in_registry(&mut registry, manifest, &build).clone();
            let link_begin = Instant::now();
            let linked = match register_generated_zig_sparse_backend(&registered) {
                Ok(linked) => linked,
                Err(err) => {
                    return failed_build_metrics(
                        scenario,
                        variant,
                        preset.label(),
                        &breakdown,
                        artifact_ms,
                        materialize_ms,
                        build_ms,
                        link_begin.elapsed().as_secs_f64() * 1_000.0,
                        format!("link failed: {err}"),
                    )
                }
            };
            let link_ms = link_begin.elapsed().as_secs_f64() * 1_000.0;
            successful_build_metrics(
                scenario,
                variant,
                preset.label(),
                &breakdown,
                artifact_ms,
                materialize_ms,
                build_ms,
                link_ms,
                linked,
            )
        }
        _ => failed_build_metrics(
            scenario,
            variant,
            preset.label(),
            &breakdown,
            artifact_ms,
            0.0,
            0.0,
            0.0,
            "artifact/backend mismatch".to_string(),
        ),
    }
}

fn average_ms(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        0.0
    } else {
        samples.iter().sum::<f64>() / samples.len() as f64
    }
}

fn measure_linked_runtime_avg(
    scenario: &ScenarioData,
    label: &str,
    linked: &LinkedSparseAotBackend,
) -> RuntimeRow {
    measure_linked_runtime_avg_against_baseline(
        scenario,
        label,
        linked,
        &scenario.lambdify_residual,
        &scenario.lambdify_jacobian,
    )
}

fn build_and_link_backend_with_preset_for_matrix_backend(
    scenario: &ScenarioData,
    variant: &BackendVariant,
    preset: BuildPreset,
    matrix_backend: MatrixBackend,
) -> BuildAndLinkMetrics {
    if matrix_backend == MatrixBackend::SparseCol {
        return build_and_link_backend_with_preset(scenario, variant, preset);
    }

    let scenario_tag = short_scenario_label(scenario.label);
    let variant_tag = short_variant_label(variant);
    let preset_tag = short_preset_label(preset);
    let backend_tag = matrix_backend_label(matrix_backend).to_ascii_lowercase();
    let artifact_name = format!(
        "gbc_{}_{}_{}_{}",
        scenario_tag, variant_tag, preset_tag, backend_tag
    );
    let module_name = format!(
        "gbcm_{}_{}_{}_{}",
        scenario_tag, variant_tag, preset_tag, backend_tag
    );
    let output_parent_dir = unique_compare_artifact_dir(
        scenario_tag,
        &format!("{variant_tag}-{preset_tag}-{backend_tag}"),
    );

    let artifact_begin = Instant::now();
    let (artifact, breakdown) = scenario
        .prepared
        .generated_aot_artifact_with_breakdown_for_matrix_backend(
            &artifact_name,
            &module_name,
            variant.backend,
            matrix_backend,
        );
    let artifact_ms = artifact_begin.elapsed().as_secs_f64() * 1_000.0;

    let manifest = PreparedProblemManifest::from(
        &scenario
            .prepared
            .as_prepared_problem_for_matrix_backend(matrix_backend),
    );
    let mut registry = AotRegistry::new();

    match (variant.backend, artifact) {
        (AotCodegenBackend::Rust, GeneratedAotArtifact::Rust(crate_spec)) => {
            let compile = match preset {
                BuildPreset::Production => AotCompileConfig::production(),
                BuildPreset::FastBuild => AotCompileConfig::fast_build(),
                BuildPreset::DevFastest => AotCompileConfig::dev_fastest(),
            };
            let materialize_begin = Instant::now();
            let build = match AotBuildRequest::new(
                crate_spec,
                &output_parent_dir,
                AotBuildProfile::Release,
            )
            .with_compile_config(compile)
            .materialize()
            {
                Ok(build) => build,
                Err(err) => {
                    return failed_build_metrics(
                        scenario,
                        variant,
                        preset.label(),
                        &breakdown,
                        artifact_ms,
                        0.0,
                        0.0,
                        0.0,
                        format!("materialize failed: {err}"),
                    )
                }
            };
            let materialize_ms = materialize_begin.elapsed().as_secs_f64() * 1_000.0;
            let build_begin = Instant::now();
            let executed = match build.execute() {
                Ok(executed) => executed,
                Err(err) => {
                    return failed_build_metrics(
                        scenario,
                        variant,
                        preset.label(),
                        &breakdown,
                        artifact_ms,
                        materialize_ms,
                        build_begin.elapsed().as_secs_f64() * 1_000.0,
                        0.0,
                        format!("build execute failed: {err}"),
                    )
                }
            };
            let build_ms = build_begin.elapsed().as_secs_f64() * 1_000.0;
            if !executed.succeeded() {
                return failed_build_metrics(
                    scenario,
                    variant,
                    preset.label(),
                    &breakdown,
                    artifact_ms,
                    materialize_ms,
                    build_ms,
                    0.0,
                    summarize_process_failure(
                        executed.status_code,
                        &executed.stdout,
                        &executed.stderr,
                    ),
                );
            }
            let registered = registry
                .register_materialized_build(manifest, &build)
                .clone();
            let link_begin = Instant::now();
            let linked = match register_generated_sparse_cdylib_backend(&registered) {
                Ok(linked) => linked,
                Err(err) => {
                    return failed_build_metrics(
                        scenario,
                        variant,
                        preset.label(),
                        &breakdown,
                        artifact_ms,
                        materialize_ms,
                        build_ms,
                        link_begin.elapsed().as_secs_f64() * 1_000.0,
                        format!("link failed: {err}"),
                    )
                }
            };
            let link_ms = link_begin.elapsed().as_secs_f64() * 1_000.0;
            successful_build_metrics(
                scenario,
                variant,
                preset.label(),
                &breakdown,
                artifact_ms,
                materialize_ms,
                build_ms,
                link_ms,
                linked,
            )
        }
        (AotCodegenBackend::C, GeneratedAotArtifact::C(library_spec)) => {
            let compiler = variant
                .c_compiler
                .clone()
                .unwrap_or_else(|| "gcc".to_string());
            let compile = match preset {
                BuildPreset::Production => CAotCompileConfig::production(),
                BuildPreset::FastBuild => CAotCompileConfig::fast_build(),
                BuildPreset::DevFastest => CAotCompileConfig::dev_fastest(),
            }
            .with_compiler(compiler);
            let materialize_begin = Instant::now();
            let build = match CAotBuildRequest::new(
                library_spec,
                &output_parent_dir,
                CAotBuildProfile::Release,
            )
            .with_compile_config(compile)
            .materialize()
            {
                Ok(build) => build,
                Err(err) => {
                    return failed_build_metrics(
                        scenario,
                        variant,
                        preset.label(),
                        &breakdown,
                        artifact_ms,
                        0.0,
                        0.0,
                        0.0,
                        format!("materialize failed: {err}"),
                    )
                }
            };
            let materialize_ms = materialize_begin.elapsed().as_secs_f64() * 1_000.0;
            let build_begin = Instant::now();
            let executed = match build.execute() {
                Ok(executed) => executed,
                Err(err) => {
                    return failed_build_metrics(
                        scenario,
                        variant,
                        preset.label(),
                        &breakdown,
                        artifact_ms,
                        materialize_ms,
                        build_begin.elapsed().as_secs_f64() * 1_000.0,
                        0.0,
                        format!("build execute failed: {err}"),
                    )
                }
            };
            let build_ms = build_begin.elapsed().as_secs_f64() * 1_000.0;
            if !executed.succeeded() {
                return failed_build_metrics(
                    scenario,
                    variant,
                    preset.label(),
                    &breakdown,
                    artifact_ms,
                    materialize_ms,
                    build_ms,
                    0.0,
                    summarize_process_failure(
                        executed.status_code,
                        &executed.stdout,
                        &executed.stderr,
                    ),
                );
            }
            let registered = register_c_build_in_registry(&mut registry, manifest, &build).clone();
            let link_begin = Instant::now();
            let linked = match register_generated_c_sparse_backend(&registered) {
                Ok(linked) => linked,
                Err(err) => {
                    return failed_build_metrics(
                        scenario,
                        variant,
                        preset.label(),
                        &breakdown,
                        artifact_ms,
                        materialize_ms,
                        build_ms,
                        link_begin.elapsed().as_secs_f64() * 1_000.0,
                        format!("link failed: {err}"),
                    )
                }
            };
            let link_ms = link_begin.elapsed().as_secs_f64() * 1_000.0;
            successful_build_metrics(
                scenario,
                variant,
                preset.label(),
                &breakdown,
                artifact_ms,
                materialize_ms,
                build_ms,
                link_ms,
                linked,
            )
        }
        (AotCodegenBackend::Zig, GeneratedAotArtifact::Zig(library_spec)) => {
            let profile = match preset {
                BuildPreset::Production => ZigAotBuildProfile::ReleaseFast,
                BuildPreset::FastBuild => ZigAotBuildProfile::ReleaseSmall,
                BuildPreset::DevFastest => ZigAotBuildProfile::Debug,
            };
            let materialize_begin = Instant::now();
            let build = match ZigAotBuildRequest::new(library_spec, &output_parent_dir, profile)
                .materialize()
            {
                Ok(build) => build,
                Err(err) => {
                    return failed_build_metrics(
                        scenario,
                        variant,
                        preset.label(),
                        &breakdown,
                        artifact_ms,
                        0.0,
                        0.0,
                        0.0,
                        format!("materialize failed: {err}"),
                    )
                }
            };
            let materialize_ms = materialize_begin.elapsed().as_secs_f64() * 1_000.0;
            let build_begin = Instant::now();
            let executed = match build.execute() {
                Ok(executed) => executed,
                Err(err) => {
                    return failed_build_metrics(
                        scenario,
                        variant,
                        preset.label(),
                        &breakdown,
                        artifact_ms,
                        materialize_ms,
                        build_begin.elapsed().as_secs_f64() * 1_000.0,
                        0.0,
                        format!("build execute failed: {err}"),
                    )
                }
            };
            let build_ms = build_begin.elapsed().as_secs_f64() * 1_000.0;
            if !executed.succeeded() {
                return failed_build_metrics(
                    scenario,
                    variant,
                    preset.label(),
                    &breakdown,
                    artifact_ms,
                    materialize_ms,
                    build_ms,
                    0.0,
                    summarize_process_failure(
                        executed.status_code,
                        &executed.stdout,
                        &executed.stderr,
                    ),
                );
            }
            let registered =
                register_zig_build_in_registry(&mut registry, manifest, &build).clone();
            let link_begin = Instant::now();
            let linked = match register_generated_zig_sparse_backend(&registered) {
                Ok(linked) => linked,
                Err(err) => {
                    return failed_build_metrics(
                        scenario,
                        variant,
                        preset.label(),
                        &breakdown,
                        artifact_ms,
                        materialize_ms,
                        build_ms,
                        link_begin.elapsed().as_secs_f64() * 1_000.0,
                        format!("link failed: {err}"),
                    )
                }
            };
            let link_ms = link_begin.elapsed().as_secs_f64() * 1_000.0;
            successful_build_metrics(
                scenario,
                variant,
                preset.label(),
                &breakdown,
                artifact_ms,
                materialize_ms,
                build_ms,
                link_ms,
                linked,
            )
        }
        _ => failed_build_metrics(
            scenario,
            variant,
            preset.label(),
            &breakdown,
            artifact_ms,
            0.0,
            0.0,
            0.0,
            "artifact/backend mismatch".to_string(),
        ),
    }
}

fn measure_linked_runtime_avg_against_baseline(
    scenario: &ScenarioData,
    label: &str,
    linked: &LinkedSparseAotBackend,
    baseline_residual: &[f64],
    baseline_jacobian: &[f64],
) -> RuntimeRow {
    emit_progress(format!(
        "measuring linked runtime for `{label}` on scenario `{}`",
        scenario.label
    ));
    let iters = scenario.runtime_iters;
    let mut residual_samples = Vec::with_capacity(scenario.runtime_samples);
    let mut jacobian_samples = Vec::with_capacity(scenario.runtime_samples);

    for _ in 0..scenario.runtime_samples {
        let residual_begin = Instant::now();
        for _ in 0..iters {
            let mut residual_out = vec![0.0; linked.residual_len];
            (linked.residual_eval)(
                black_box(scenario.args.as_slice()),
                black_box(&mut residual_out),
            );
            black_box(residual_out);
        }
        residual_samples.push(residual_begin.elapsed().as_secs_f64() * 1_000.0);

        let jacobian_begin = Instant::now();
        for _ in 0..iters {
            let mut jacobian_out = vec![0.0; linked.nnz];
            (linked.jacobian_values_eval)(
                black_box(scenario.args.as_slice()),
                black_box(&mut jacobian_out),
            );
            black_box(jacobian_out);
        }
        jacobian_samples.push(jacobian_begin.elapsed().as_secs_f64() * 1_000.0);
    }

    let mut residual_once = vec![0.0; linked.residual_len];
    (linked.residual_eval)(scenario.args.as_slice(), &mut residual_once);
    let mut jacobian_once = vec![0.0; linked.nnz];
    (linked.jacobian_values_eval)(scenario.args.as_slice(), &mut jacobian_once);

    let residual_ms = average_ms(&residual_samples);
    let jacobian_ms = average_ms(&jacobian_samples);

    let row = RuntimeRow {
        label: label.to_string(),
        residual_ms,
        jacobian_ms,
        total_ms: residual_ms + jacobian_ms,
        residual_diff: max_abs_diff(&residual_once, baseline_residual),
        jacobian_diff: max_abs_diff(&jacobian_once, baseline_jacobian),
        status: "ok".to_string(),
    };
    emit_progress(format!(
        "runtime for `{label}` on scenario `{}` finished: total {:.3} ms",
        scenario.label, row.total_ms
    ));
    row
}

fn measure_lambdify_runtime_avg(scenario: &ScenarioData) -> RuntimeRow {
    emit_progress(format!(
        "measuring lambdify baseline for scenario `{}`",
        scenario.label
    ));
    let residual_fns = compile_residual_lambdify(&scenario.prepared);
    let jacobian_fns = compile_sparse_lambdify(&scenario.prepared);
    let iters = scenario.runtime_iters;
    let mut residual_samples = Vec::with_capacity(scenario.runtime_samples);
    let mut jacobian_samples = Vec::with_capacity(scenario.runtime_samples);

    for _ in 0..scenario.runtime_samples {
        let residual_begin = Instant::now();
        for _ in 0..iters {
            black_box(eval_functions(&residual_fns, scenario.args.as_slice()));
        }
        residual_samples.push(residual_begin.elapsed().as_secs_f64() * 1_000.0);

        let jacobian_begin = Instant::now();
        for _ in 0..iters {
            black_box(eval_functions(&jacobian_fns, scenario.args.as_slice()));
        }
        jacobian_samples.push(jacobian_begin.elapsed().as_secs_f64() * 1_000.0);
    }

    let residual_ms = average_ms(&residual_samples);
    let jacobian_ms = average_ms(&jacobian_samples);

    let row = RuntimeRow {
        label: "Lambdify".to_string(),
        residual_ms,
        jacobian_ms,
        total_ms: residual_ms + jacobian_ms,
        residual_diff: 0.0,
        jacobian_diff: 0.0,
        status: "ok".to_string(),
    };
    emit_progress(format!(
        "lambdify baseline for scenario `{}` finished: total {:.3} ms",
        scenario.label, row.total_ms
    ));
    row
}

fn build_lambdify_end_to_end_row(scenario: &ScenarioData) -> EndToEndCallableRow {
    emit_progress(format!(
        "measuring lambdify end-to-end callable path for scenario `{}`",
        scenario.label
    ));
    let compile_begin = Instant::now();
    let residual_fns = compile_residual_lambdify(&scenario.prepared);
    let jacobian_fns = compile_sparse_lambdify(&scenario.prepared);
    let callable_prep_ms = compile_begin.elapsed().as_secs_f64() * 1_000.0;

    let first_issue_begin = Instant::now();
    let residual_once = eval_functions(&residual_fns, scenario.args.as_slice());
    let jacobian_once = eval_functions(&jacobian_fns, scenario.args.as_slice());
    let first_issue_ms = first_issue_begin.elapsed().as_secs_f64() * 1_000.0;

    let runtime = measure_lambdify_runtime_avg(scenario);

    EndToEndCallableRow {
        variant: "Lambdify".to_string(),
        symbolic_backend: "ExprLegacy",
        preset: "n/a",
        symbolic_ms: scenario.symbolic_ms,
        callable_prep_ms,
        materialize_ms: 0.0,
        build_ms: 0.0,
        link_ms: 0.0,
        first_issue_ms,
        total_to_outputs_ms: scenario.symbolic_ms + callable_prep_ms + first_issue_ms,
        residual_ms: runtime.residual_ms,
        jacobian_ms: runtime.jacobian_ms,
        total_ms: runtime.total_ms,
        speedup_vs_lambdify: 1.0,
        residual_diff: max_abs_diff(&residual_once, &scenario.lambdify_residual),
        jacobian_diff: max_abs_diff(&jacobian_once, &scenario.lambdify_jacobian),
        status: "ok".to_string(),
    }
}

fn build_atomview_compiled_end_to_end_row(
    baseline: &ScenarioData,
    compiled: &ScenarioData,
    variant: &BackendVariant,
    preset: BuildPreset,
) -> EndToEndCallableRow {
    let metrics = build_and_link_backend_with_preset(compiled, variant, preset);
    if let Some(linked) = metrics.linked.as_ref() {
        let runtime = measure_linked_runtime_avg_against_baseline(
            compiled,
            &variant.label,
            linked,
            &baseline.lambdify_residual,
            &baseline.lambdify_jacobian,
        );
        EndToEndCallableRow {
            variant: variant.label.clone(),
            symbolic_backend: "AtomView",
            preset: metrics.preset_label,
            symbolic_ms: compiled.symbolic_ms,
            callable_prep_ms: metrics.artifact_ms,
            materialize_ms: metrics.materialize_ms,
            build_ms: metrics.build_ms,
            link_ms: metrics.link_ms,
            first_issue_ms: metrics.first_issue_ms,
            total_to_outputs_ms: compiled.symbolic_ms
                + metrics.artifact_ms
                + metrics.materialize_ms
                + metrics.build_ms
                + metrics.link_ms
                + metrics.first_issue_ms,
            residual_ms: runtime.residual_ms,
            jacobian_ms: runtime.jacobian_ms,
            total_ms: runtime.total_ms,
            speedup_vs_lambdify: f64::NAN,
            residual_diff: runtime.residual_diff,
            jacobian_diff: runtime.jacobian_diff,
            status: runtime.status,
        }
    } else {
        EndToEndCallableRow {
            variant: variant.label.clone(),
            symbolic_backend: "AtomView",
            preset: metrics.preset_label,
            symbolic_ms: compiled.symbolic_ms,
            callable_prep_ms: metrics.artifact_ms,
            materialize_ms: metrics.materialize_ms,
            build_ms: metrics.build_ms,
            link_ms: metrics.link_ms,
            first_issue_ms: metrics.first_issue_ms,
            total_to_outputs_ms: metrics.total_to_outputs_ms,
            residual_ms: f64::NAN,
            jacobian_ms: f64::NAN,
            total_ms: f64::NAN,
            speedup_vs_lambdify: f64::NAN,
            residual_diff: f64::NAN,
            jacobian_diff: f64::NAN,
            status: metrics.status,
        }
    }
}

fn print_end_to_end_callable_pipeline_table(scenario: &ScenarioData, rows: &[EndToEndCallableRow]) {
    println!(
        "[BVP end-to-end callable compare] scenario={}, residuals={}, vars={}, nnz={}",
        scenario.label,
        scenario.prepared.residuals.len(),
        scenario.prepared.shape.1,
        scenario.prepared.sparse_entries.len()
    );
    println!(
        "variant        | sym_backend | preset      | symbolic_ms | callable_prep_ms | materialize_ms | build_ms | link_ms | first_issue_ms | total_to_outputs_ms | status"
    );
    println!("{}", "-".repeat(171));
    for row in rows {
        println!(
            "{:<14} | {:<11} | {:<11} | {:>11.3} | {:>16.3} | {:>14.3} | {:>8.3} | {:>7.3} | {:>14.3} | {:>19.3} | {}",
            row.variant,
            row.symbolic_backend,
            row.preset,
            row.symbolic_ms,
            row.callable_prep_ms,
            row.materialize_ms,
            row.build_ms,
            row.link_ms,
            row.first_issue_ms,
            row.total_to_outputs_ms,
            row.status
        );
    }
    flush_stdout();
}

fn print_end_to_end_callable_runtime_table(rows: &[EndToEndCallableRow]) {
    println!(
        "variant        | residual_ms(avg) | jacobian_ms(avg) | total_ms(avg) | speedup_vs_lambdify | residual_diff | jacobian_diff | status"
    );
    println!("{}", "-".repeat(140));
    for row in rows {
        println!(
            "{:<14} | {:>16.3} | {:>16.3} | {:>13.3} | {:>20.3}x | {:>13.6e} | {:>13.6e} | {}",
            row.variant,
            row.residual_ms,
            row.jacobian_ms,
            row.total_ms,
            row.speedup_vs_lambdify,
            row.residual_diff,
            row.jacobian_diff,
            row.status
        );
    }
    flush_stdout();
}

fn print_end_to_end_callable_break_even_table(rows: &[EndToEndCallableRow]) {
    let Some(lambdify) = rows.iter().find(|row| row.variant == "Lambdify") else {
        return;
    };

    println!(
        "variant        | extra_bootstrap_ms_vs_lambdify | runtime_gain_ms_per_call | break_even_calls | status"
    );
    println!("{}", "-".repeat(108));
    for row in rows.iter().filter(|row| row.variant != "Lambdify") {
        let extra_bootstrap_ms = row.total_to_outputs_ms - lambdify.total_to_outputs_ms;
        let runtime_gain_ms_per_call = lambdify.total_ms - row.total_ms;
        let break_even_calls = if extra_bootstrap_ms.is_finite()
            && runtime_gain_ms_per_call.is_finite()
            && extra_bootstrap_ms > 0.0
            && runtime_gain_ms_per_call > 0.0
        {
            extra_bootstrap_ms / runtime_gain_ms_per_call
        } else if extra_bootstrap_ms <= 0.0 {
            0.0
        } else {
            f64::NAN
        };

        println!(
            "{:<14} | {:>29.3} | {:>24.3} | {:>16.3} | {}",
            row.variant, extra_bootstrap_ms, runtime_gain_ms_per_call, break_even_calls, row.status
        );
    }
    flush_stdout();
}

fn print_preset_build_table(scenario: &ScenarioData, rows: &[PresetRuntimeRow]) {
    println!(
        "[BVP backend preset compare] scenario={}, residuals={}, vars={}, nnz={}",
        scenario.label,
        scenario.prepared.residuals.len(),
        scenario.prepared.shape.1,
        scenario.prepared.sparse_entries.len()
    );
    println!("variant        | preset      | build_ms | total_to_outputs_ms | status");
    println!("{}", "-".repeat(78));
    for row in rows {
        println!(
            "{:<14} | {:<11} | {:>8.3} | {:>19.3} | {}",
            row.variant, row.preset, row.build_ms, row.total_to_outputs_ms, row.status
        );
    }
    flush_stdout();
}

fn print_preset_runtime_table(scenario: &ScenarioData, rows: &[PresetRuntimeRow]) {
    println!(
        "variant        | preset      | residual_ms(avg) | jacobian_ms(avg) | total_ms(avg) | speedup_vs_baseline | residual_diff | jacobian_diff"
    );
    println!("{}", "-".repeat(140));
    let _ = scenario;
    for row in rows {
        println!(
            "{:<14} | {:<11} | {:>16.3} | {:>16.3} | {:>13.3} | {:>21.3}x | {:>13.6e} | {:>13.6e}",
            row.variant,
            row.preset,
            row.residual_ms,
            row.jacobian_ms,
            row.total_ms,
            row.speedup_vs_baseline,
            row.residual_diff,
            row.jacobian_diff
        );
    }
    flush_stdout();
}

fn scenario_specs() -> Vec<ScenarioSpec> {
    vec![
        ScenarioSpec {
            label: "small-damp1-24",
            runtime_iters: 200,
            runtime_samples: 5,
            build_jacobian: || build_real_bvp_damp1_case(24),
        },
        ScenarioSpec {
            label: "combustion-100",
            runtime_iters: 40,
            runtime_samples: 5,
            build_jacobian: || build_combustion_bvp_case(100),
        },
        ScenarioSpec {
            label: "combustion-1000",
            runtime_iters: 6,
            runtime_samples: 3,
            build_jacobian: || build_combustion_bvp_case(1000),
        },
    ]
}

/// Keeps the largest cold-bootstrap case out of the separate runtime and
/// compile-preset matrices: those stories answer different questions and
/// would otherwise rebuild this expensive artifact unnecessarily.
fn pipeline_scenario_specs() -> Vec<ScenarioSpec> {
    let mut specs = scenario_specs();
    specs.push(ScenarioSpec {
        label: "combustion-3000",
        runtime_iters: 1,
        runtime_samples: 1,
        build_jacobian: || build_combustion_bvp_case(3000),
    });
    specs
}

/// Answers:
/// - which backend/compiler reaches the first callable outputs fastest,
/// - whether compiler choice dominates total bootstrap latency,
/// - whether all generated outputs still numerically match the lambdify baseline.
///
/// This test is the main evidence for the "time-to-first-output" hypothesis.
/// Set `BVP_PIPELINE_SCENARIO_FILTER=combustion-3000` to isolate the largest
/// stress scenario without rebuilding the three smaller pipeline cases.
#[test]
#[ignore = "diagnostic pipeline/build/correctness compare for BVP backends and compilers"]
fn bvp_generated_backend_pipeline_comparison_table() {
    emit_progress("starting pipeline comparison test");
    let variants = available_backend_variants();
    assert!(
        !variants.is_empty(),
        "expected at least one generated backend variant"
    );
    print_backend_probe_summary(&variants);
    let runs = pipeline_sample_runs();
    emit_progress(format!(
        "pipeline comparison will collect {runs} run(s) per scenario/route; set BVP_PIPELINE_RUNS to override"
    ));

    let scenario_filter = env::var("BVP_PIPELINE_SCENARIO_FILTER")
        .ok()
        .map(|filter| filter.trim().to_ascii_lowercase())
        .filter(|filter| !filter.is_empty());
    let specs: Vec<_> = pipeline_scenario_specs()
        .into_iter()
        .filter(|spec| {
            scenario_filter.as_ref().map_or(true, |filter| {
                spec.label.to_ascii_lowercase().contains(filter)
            })
        })
        .collect();
    assert!(
        !specs.is_empty(),
        "BVP_PIPELINE_SCENARIO_FILTER={:?} did not match any pipeline scenario",
        scenario_filter
    );
    if let Some(filter) = &scenario_filter {
        emit_progress(format!(
            "pipeline comparison restricted to scenario labels containing `{filter}`"
        ));
    }

    for spec in specs {
        emit_progress(format!(
            "pipeline comparison entering scenario `{}`",
            spec.label
        ));
        let mut samples = Vec::new();
        let mut last_scenario = None;
        for run_idx in 0..runs {
            emit_progress(format!(
                "pipeline comparison scenario `{}` run {}/{}",
                spec.label,
                run_idx + 1,
                runs
            ));
            let scenario = build_scenario_data(&spec);
            samples.push(lambdify_pipeline_sample(&scenario));
            for variant in &variants {
                let metrics = build_and_link_backend(&scenario, variant);
                samples.push(compiled_pipeline_sample(&metrics, scenario.symbolic_ms));
                if metrics.linked.is_some() {
                    let _ = unregister_linked_sparse_backend(&scenario.prepared.problem_key());
                }
            }
            last_scenario = Some(scenario);
        }
        let scenario = last_scenario.expect("at least one pipeline run should execute");
        let rows = aggregate_pipeline_samples(&samples);
        print_pipeline_bootstrap_summary_table(&scenario, &rows);
        print_pipeline_codegen_summary_table(&rows);
        print_pipeline_correctness_summary_table(&rows);
        emit_progress(format!(
            "pipeline comparison finished scenario `{}`",
            scenario.label
        ));
    }
}

/// Answers:
/// - how much cold-start time is spent in optional AtomView IR cleanup passes,
/// - whether skipping those passes materially reduces AOT artifact preparation,
/// - and which source emitter should be used for the next compile-stage test.
///
/// This intentionally stops before compiler invocation: it isolates the second
/// AOT stage (`AtomView -> IR -> source/artifact`) that dominates tcc cold
/// bootstrap once the external compiler is already fast.
#[test]
#[ignore = "diagnostic AtomView AOT optimization-profile bootstrap compare"]
fn bvp_atomview_aot_optimization_profile_bootstrap_table() {
    emit_progress("starting AtomView AOT optimization-profile bootstrap compare");
    let runs = pipeline_sample_runs();
    emit_progress(format!(
        "optimization-profile comparison will collect {runs} run(s); set BVP_PIPELINE_RUNS to override"
    ));

    let spec = ScenarioSpec {
        label: "combustion-1000",
        runtime_iters: 1,
        runtime_samples: 1,
        build_jacobian: || {
            build_combustion_bvp_case_with_backend(1000, BvpSymbolicAssemblyBackend::AtomView)
        },
    };
    let profiles = [
        AtomOptimizationProfile::Full,
        AtomOptimizationProfile::FastBootstrap,
        AtomOptimizationProfile::NoPeephole,
        AtomOptimizationProfile::NoTempReuse,
        AtomOptimizationProfile::NoCse,
    ];
    let backends = [
        AotCodegenBackend::C,
        AotCodegenBackend::Rust,
        AotCodegenBackend::Zig,
    ];

    let mut samples = Vec::new();
    let mut last_scenario = None;
    for run_idx in 0..runs {
        emit_progress(format!(
            "optimization-profile comparison run {}/{}",
            run_idx + 1,
            runs
        ));
        let scenario =
            build_scenario_data_with_backend(&spec, BvpSymbolicAssemblyBackend::AtomView);
        for backend in backends {
            for profile in profiles {
                samples.push(atom_profile_artifact_pipeline_sample(
                    &scenario, backend, profile,
                ));
            }
        }
        last_scenario = Some(scenario);
    }

    let scenario = last_scenario.expect("at least one optimization-profile run should execute");
    let rows = aggregate_pipeline_samples(&samples);
    println!(
        "\n[BVP codegen] AtomView AOT optimization-profile bootstrap table; scenario={}",
        scenario.label
    );
    println!(
        "note: artifact_only isolates AtomView -> IR/source/artifact packaging; build/link are intentionally excluded"
    );
    print_pipeline_bootstrap_summary_table(&scenario, &rows);
    print_pipeline_codegen_summary_table(&rows);
    emit_progress("optimization-profile bootstrap compare finished");
}

/// Answers:
/// - how fast already-built residual/Jacobian callbacks run,
/// - whether compiled backends materially outperform lambdified callbacks,
/// - whether runtime speed stays correct across compilers.
///
/// This test isolates callback throughput after bootstrap cost is already paid.
#[test]
#[ignore = "diagnostic runtime throughput compare for lambdify vs generated BVP backends and compilers"]
fn bvp_generated_backend_runtime_comparison_table() {
    emit_progress("starting runtime comparison test");
    let variants = available_backend_variants();
    assert!(
        !variants.is_empty(),
        "expected at least one generated backend variant"
    );
    print_backend_probe_summary(&variants);

    for spec in scenario_specs() {
        emit_progress(format!(
            "runtime comparison entering scenario `{}`",
            spec.label
        ));
        let scenario = build_scenario_data(&spec);
        let mut rows = vec![measure_lambdify_runtime_avg(&scenario)];
        for variant in &variants {
            let metrics = build_and_link_backend(&scenario, variant);
            if let Some(linked) = metrics.linked.as_ref() {
                rows.push(measure_linked_runtime_avg(
                    &scenario,
                    &variant.label,
                    linked,
                ));
            } else {
                rows.push(RuntimeRow {
                    label: variant.label.clone(),
                    residual_ms: f64::NAN,
                    jacobian_ms: f64::NAN,
                    total_ms: f64::NAN,
                    residual_diff: f64::NAN,
                    jacobian_diff: f64::NAN,
                    status: metrics.status.clone(),
                });
            }
            let _ = unregister_linked_sparse_backend(&scenario.prepared.problem_key());
        }
        print_runtime_table(&scenario, &rows);
        emit_progress(format!(
            "runtime comparison finished scenario `{}`",
            scenario.label
        ));
    }
}

/// Answers:
/// - whether `ExprLegacy + Lambdify` or `AtomView + compiled backend` wins overall,
/// - how much bootstrap latency compiled backends add relative to lambdify,
/// - after how many repeated evaluations the compiled path pays back that cost.
///
/// This test is the practical "is it worth it?" compare for the current leaders.
#[test]
#[ignore = "diagnostic end-to-end callable compare for Lambdify vs AtomView C leaders"]
fn bvp_lambdify_vs_atomview_callable_leaders_compare() {
    emit_progress("starting lambdify vs AtomView callable leaders compare");
    let variants = available_backend_variants();
    let leader_variants = variants
        .into_iter()
        .filter(|variant| {
            variant.backend == AotCodegenBackend::C
                && matches!(variant.c_compiler.as_deref(), Some("gcc") | Some("tcc"))
        })
        .collect::<Vec<_>>();
    assert!(
        !leader_variants.is_empty(),
        "expected at least one compiled C leader backend (gcc or tcc)"
    );
    print_backend_probe_summary(&leader_variants);

    for spec in scenario_specs()
        .into_iter()
        .filter(|spec| matches!(spec.label, "combustion-100" | "combustion-1000"))
    {
        emit_progress(format!(
            "callable leaders compare entering scenario `{}`",
            spec.label
        ));
        let lambdify_scenario =
            build_scenario_data_with_backend(&spec, BvpSymbolicAssemblyBackend::ExprLegacy);
        let atom_scenario =
            build_scenario_data_with_backend(&spec, BvpSymbolicAssemblyBackend::AtomView);

        let mut rows = vec![build_lambdify_end_to_end_row(&lambdify_scenario)];
        for variant in &leader_variants {
            rows.push(build_atomview_compiled_end_to_end_row(
                &lambdify_scenario,
                &atom_scenario,
                variant,
                BuildPreset::DevFastest,
            ));
            let _ = unregister_linked_sparse_backend(&atom_scenario.prepared.problem_key());
        }

        let lambdify_total = rows
            .iter()
            .find(|row| row.variant == "Lambdify")
            .map(|row| row.total_ms)
            .unwrap_or(f64::NAN);
        for row in &mut rows {
            row.speedup_vs_lambdify = if row.total_ms.is_finite() && row.total_ms > 0.0 {
                lambdify_total / row.total_ms
            } else {
                f64::NAN
            };
        }

        print_end_to_end_callable_pipeline_table(&atom_scenario, &rows);
        print_end_to_end_callable_runtime_table(&rows);
        print_end_to_end_callable_break_even_table(&rows);
        emit_progress(format!(
            "callable leaders compare finished scenario `{}`",
            spec.label
        ));
    }
}

/// Answers:
/// - which compile preset minimizes build latency for each backend,
/// - how much runtime speed is sacrificed by `FastBuild`/`DevFastest`,
/// - whether that tradeoff is worth it for a given backend and problem size.
///
/// This test validates or refutes the preset tradeoff hypothesis.
#[test]
#[ignore = "diagnostic compile-preset vs runtime tradeoff compare for generated BVP backends"]
fn bvp_generated_backend_compile_preset_tradeoff_table() {
    emit_progress("starting compile-preset tradeoff test");
    let variants = available_backend_variants();
    assert!(
        !variants.is_empty(),
        "expected at least one generated backend variant"
    );
    print_backend_probe_summary(&variants);
    for spec in scenario_specs() {
        emit_progress(format!(
            "compile-preset comparison entering scenario `{}`",
            spec.label
        ));
        let scenario = build_scenario_data(&spec);
        let mut rows = Vec::new();

        for variant in &variants {
            let mut baseline_total = None;
            for preset in active_presets_for_scenario_variant(&scenario, variant) {
                let metrics = build_and_link_backend_with_preset(&scenario, variant, preset);
                if let Some(linked) = metrics.linked.as_ref() {
                    let runtime = measure_linked_runtime_avg(
                        &scenario,
                        &format!("{}-{}", variant.label, preset.label()),
                        linked,
                    );
                    let baseline = if let Some(existing) = baseline_total {
                        existing
                    } else {
                        runtime.total_ms
                    };
                    if baseline_total.is_none() {
                        baseline_total = Some(runtime.total_ms);
                    }
                    rows.push(PresetRuntimeRow {
                        variant: variant.label.clone(),
                        preset: preset.label(),
                        build_ms: metrics.build_ms,
                        total_to_outputs_ms: metrics.total_to_outputs_ms,
                        residual_ms: runtime.residual_ms,
                        jacobian_ms: runtime.jacobian_ms,
                        total_ms: runtime.total_ms,
                        speedup_vs_baseline: if runtime.total_ms.is_finite()
                            && runtime.total_ms > 0.0
                        {
                            baseline / runtime.total_ms
                        } else {
                            f64::NAN
                        },
                        residual_diff: runtime.residual_diff,
                        jacobian_diff: runtime.jacobian_diff,
                        status: runtime.status,
                    });
                } else {
                    rows.push(PresetRuntimeRow {
                        variant: variant.label.clone(),
                        preset: preset.label(),
                        build_ms: metrics.build_ms,
                        total_to_outputs_ms: metrics.total_to_outputs_ms,
                        residual_ms: f64::NAN,
                        jacobian_ms: f64::NAN,
                        total_ms: f64::NAN,
                        speedup_vs_baseline: f64::NAN,
                        residual_diff: f64::NAN,
                        jacobian_diff: f64::NAN,
                        status: metrics.status,
                    });
                }
                let _ = unregister_linked_sparse_backend(&scenario.prepared.problem_key());
            }
        }

        print_preset_build_table(&scenario, &rows);
        print_preset_runtime_table(&scenario, &rows);
        emit_progress(format!(
            "compile-preset comparison finished scenario `{}`",
            scenario.label
        ));
    }
}

/// Answers:
/// - how real compiled modules behave for `Sparse` vs `Banded`,
/// - which language/preset reaches the best build/runtime balance,
/// - and whether compiled banded values stay numerically aligned with lambdify.
#[test]
#[ignore = "diagnostic real compiled sparse/banded backend compare across languages and presets"]
fn bvp_generated_compiled_sparse_banded_matrix_backend_table() {
    emit_progress("starting compiled sparse/banded matrix-backend compare");
    let variants = available_backend_variants();
    assert!(
        !variants.is_empty(),
        "expected at least one generated backend variant"
    );
    print_backend_probe_summary(&variants);

    for spec in scenario_specs()
        .into_iter()
        .filter(|spec| matches!(spec.label, "small-damp1-24" | "combustion-100"))
    {
        emit_progress(format!(
            "compiled sparse/banded compare entering scenario `{}`",
            spec.label
        ));
        let scenario = build_scenario_data(&spec);
        let banded_baseline = eval_functions(
            &compile_banded_lambdify(&scenario.prepared),
            scenario.args.as_slice(),
        );
        let mut rows = Vec::new();

        for matrix_backend in [MatrixBackend::SparseCol, MatrixBackend::Banded] {
            for variant in &variants {
                for preset in active_presets_for_scenario_variant(&scenario, variant) {
                    let metrics = build_and_link_backend_with_preset_for_matrix_backend(
                        &scenario,
                        variant,
                        preset,
                        matrix_backend,
                    );
                    let problem_key = scenario
                        .prepared
                        .problem_key_for_matrix_backend(matrix_backend);
                    if let Some(linked) = metrics.linked.as_ref() {
                        let runtime = match matrix_backend {
                            MatrixBackend::Banded => measure_linked_runtime_avg_against_baseline(
                                &scenario,
                                &format!(
                                    "{}-{}-{}",
                                    matrix_backend_label(matrix_backend),
                                    variant.label,
                                    preset.label()
                                ),
                                linked,
                                &scenario.lambdify_residual,
                                &banded_baseline,
                            ),
                            _ => measure_linked_runtime_avg_against_baseline(
                                &scenario,
                                &format!(
                                    "{}-{}-{}",
                                    matrix_backend_label(matrix_backend),
                                    variant.label,
                                    preset.label()
                                ),
                                linked,
                                &scenario.lambdify_residual,
                                &scenario.lambdify_jacobian,
                            ),
                        };
                        rows.push(CompiledMatrixBackendRow {
                            scenario: spec.label.to_string(),
                            matrix_backend: matrix_backend_label(matrix_backend).to_string(),
                            variant: variant.label.clone(),
                            preset: preset.label().to_string(),
                            compile_mode: compile_mode_label(variant, preset),
                            build_ms: format!("{:.3}", metrics.build_ms),
                            total_to_outputs_ms: format!("{:.3}", metrics.total_to_outputs_ms),
                            runtime_total_ms: format!("{:.3}", runtime.total_ms),
                            residual_diff: format!("{:.3e}", runtime.residual_diff),
                            jacobian_diff: format!("{:.3e}", runtime.jacobian_diff),
                            status: runtime.status,
                        });
                    } else {
                        rows.push(CompiledMatrixBackendRow {
                            scenario: spec.label.to_string(),
                            matrix_backend: matrix_backend_label(matrix_backend).to_string(),
                            variant: variant.label.clone(),
                            preset: preset.label().to_string(),
                            compile_mode: compile_mode_label(variant, preset),
                            build_ms: format!("{:.3}", metrics.build_ms),
                            total_to_outputs_ms: format!("{:.3}", metrics.total_to_outputs_ms),
                            runtime_total_ms: "NaN".to_string(),
                            residual_diff: "NaN".to_string(),
                            jacobian_diff: "NaN".to_string(),
                            status: metrics.status.clone(),
                        });
                    }
                    let _ = unregister_linked_sparse_backend(&problem_key);
                }
            }
        }

        let mut builder = Builder::default();
        builder.push_record([
            "scenario",
            "matrix_backend",
            "variant",
            "preset",
            "compile_mode",
            "build_ms",
            "total_to_outputs_ms",
            "runtime_total_ms",
            "residual_diff",
            "jacobian_diff",
            "status",
        ]);
        for row in rows {
            builder.push_record([
                row.scenario,
                row.matrix_backend,
                row.variant,
                row.preset,
                row.compile_mode,
                row.build_ms,
                row.total_to_outputs_ms,
                row.runtime_total_ms,
                row.residual_diff,
                row.jacobian_diff,
                row.status,
            ]);
        }
        let mut table = builder.build();
        table.with(Style::rounded());
        println!(
            "[BVP compiled backend compare] real compiled sparse/banded modules scenario={}",
            spec.label
        );
        println!("{table}");
        flush_stdout();
        emit_progress(format!(
            "compiled sparse/banded compare finished scenario `{}`",
            spec.label
        ));
    }
}

/// Production-like callable + linear-solver story for BVP backends.
///
/// Answers in one run:
/// - how expensive callable preparation is for lambdify and compiled backends,
/// - how quickly residual/Jacobian callbacks run afterwards,
/// - whether `Sparse` / `Banded` callbacks stay numerically aligned,
/// - and how the downstream linear-solver paths behave against the `faer` sparse baseline.
#[test]
#[ignore = "diagnostic compiled/lambdify callable story with linear solver comparison"]
fn bvp_callable_and_linear_solver_story_table() {
    emit_progress("starting callable + linear-solver story diagnostic");
    let variants = available_backend_variants();
    assert!(
        !variants.is_empty(),
        "expected at least one generated backend variant"
    );
    print_backend_probe_summary(&variants);

    for spec in scenario_specs() {
        emit_progress(format!(
            "callable + linear-solver story entering scenario `{}`",
            spec.label
        ));
        let scenario = build_scenario_data(&spec);
        let rhs = sample_rhs(scenario.prepared.shape.0);

        let sparse_baseline_matrix = scenario
            .prepared
            .as_prepared_problem()
            .jacobian_plan
            .assemble_sparse_col_mat(scenario.lambdify_jacobian.as_slice());
        let baseline_solution = solve_sparse_lu(&sparse_baseline_matrix, rhs.as_slice());

        let mut stage_rows = Vec::new();
        let mut runtime_rows = Vec::new();
        let mut linear_rows = Vec::new();

        for matrix_backend in [MatrixBackend::SparseCol, MatrixBackend::Banded] {
            let (stage_row, runtime_row, _jacobian_once, banded_assembly) =
                measure_lambdify_matrix_runtime(&scenario, matrix_backend);
            stage_rows.push(stage_row);
            runtime_rows.push(runtime_row);

            match matrix_backend {
                MatrixBackend::SparseCol => {
                    let solve_ms = average_linear_solve_ms(
                        scenario.runtime_samples,
                        scenario.runtime_iters,
                        || {
                            black_box(solve_sparse_lu(&sparse_baseline_matrix, rhs.as_slice()));
                        },
                    );
                    linear_rows.push(LinearSolverStoryRow {
                        scenario: scenario.label.to_string(),
                        source: "Lambdify".to_string(),
                        matrix_backend: "Sparse".to_string(),
                        variant: "-".to_string(),
                        preset: "-".to_string(),
                        linear_solver: "faer_sparse_lu".to_string(),
                        layout: "-".to_string(),
                        refinement: "-".to_string(),
                        direct_rr: "-".to_string(),
                        final_rr: "-".to_string(),
                        solve_rr: "-".to_string(),
                        total_ms: format!("{:.3}", solve_ms),
                        solve_diff: format!("{:.3e}", 0.0),
                        relative_x_diff: format!("{:.3e}", 0.0),
                        status: "ok".to_string(),
                    });
                }
                MatrixBackend::Banded => {
                    let assembly = banded_assembly.expect("banded lambdify assembly should exist");
                    let dense = banded_assembly_to_dense(&assembly);
                    for banded_solver in BandedSolverUnderTest::variants() {
                        let (mut metrics, solution_once) = solve_banded_solver_for_scenario(
                            scenario.label,
                            &assembly,
                            rhs.as_slice(),
                            banded_solver,
                        );
                        metrics.solve_diff = max_abs_diff(&baseline_solution, &solution_once);
                        metrics.relative_x_diff =
                            relative_x_diff(&solution_once, &baseline_solution);
                        metrics.solve_rr =
                            relative_dense_residual(&dense, &solution_once, rhs.as_slice());
                        linear_rows.push(LinearSolverStoryRow {
                            scenario: scenario.label.to_string(),
                            source: "Lambdify".to_string(),
                            matrix_backend: "Banded".to_string(),
                            variant: "-".to_string(),
                            preset: "-".to_string(),
                            linear_solver: metrics.linear_solver,
                            layout: metrics.layout,
                            refinement: metrics.refinement,
                            direct_rr: fmt_metric_or_dash(metrics.direct_rr),
                            final_rr: fmt_metric_or_dash(metrics.final_rr),
                            solve_rr: fmt_metric_or_dash(metrics.solve_rr),
                            total_ms: format!(
                                "{:.3}",
                                average_linear_solve_ms(
                                    scenario.runtime_samples,
                                    scenario.runtime_iters,
                                    || {
                                        black_box(solve_banded_solver_for_scenario(
                                            scenario.label,
                                            &assembly,
                                            rhs.as_slice(),
                                            banded_solver,
                                        ));
                                    },
                                )
                            ),
                            solve_diff: format!("{:.3e}", metrics.solve_diff),
                            relative_x_diff: format!("{:.3e}", metrics.relative_x_diff),
                            status: metrics.status,
                        });
                    }
                }
                _ => {}
            }
        }

        for matrix_backend in [MatrixBackend::SparseCol, MatrixBackend::Banded] {
            let jacobian_baseline = match matrix_backend {
                MatrixBackend::Banded => eval_functions(
                    &compile_banded_lambdify(&scenario.prepared),
                    scenario.args.as_slice(),
                ),
                _ => scenario.lambdify_jacobian.clone(),
            };

            for variant in &variants {
                for preset in active_presets_for_scenario_variant(&scenario, variant) {
                    let metrics = build_and_link_backend_with_preset_for_matrix_backend(
                        &scenario,
                        variant,
                        preset,
                        matrix_backend,
                    );
                    let problem_key = scenario
                        .prepared
                        .problem_key_for_matrix_backend(matrix_backend);

                    stage_rows.push(CallableStoryRow {
                        scenario: scenario.label.to_string(),
                        source: "Compiled".to_string(),
                        matrix_backend: matrix_backend_label(matrix_backend).to_string(),
                        variant: variant.label.clone(),
                        preset: preset.label().to_string(),
                        compile_mode: compile_mode_label(variant, preset),
                        symbolic_ms: format!("{:.3}", scenario.symbolic_ms),
                        callable_prep_ms: format!("{:.3}", metrics.artifact_ms),
                        materialize_ms: format!("{:.3}", metrics.materialize_ms),
                        build_ms: format!("{:.3}", metrics.build_ms),
                        link_ms: format!("{:.3}", metrics.link_ms),
                        first_issue_ms: format!("{:.3}", metrics.first_issue_ms),
                        total_to_outputs_ms: format!("{:.3}", metrics.total_to_outputs_ms),
                        status: metrics.status.clone(),
                    });

                    if let Some(linked) = metrics.linked.as_ref() {
                        let runtime = measure_linked_runtime_avg_against_baseline(
                            &scenario,
                            &format!(
                                "{}-{}-{}",
                                matrix_backend_label(matrix_backend),
                                variant.label,
                                preset.label()
                            ),
                            linked,
                            &scenario.lambdify_residual,
                            jacobian_baseline.as_slice(),
                        );
                        runtime_rows.push(CallbackRuntimeMatrixRow {
                            scenario: scenario.label.to_string(),
                            source: "Compiled".to_string(),
                            matrix_backend: matrix_backend_label(matrix_backend).to_string(),
                            variant: variant.label.clone(),
                            preset: preset.label().to_string(),
                            residual_ms: format!("{:.3}", runtime.residual_ms),
                            jacobian_ms: format!("{:.3}", runtime.jacobian_ms),
                            total_ms: format!("{:.3}", runtime.total_ms),
                            residual_diff: format!("{:.3e}", runtime.residual_diff),
                            jacobian_diff: format!("{:.3e}", runtime.jacobian_diff),
                            status: runtime.status.clone(),
                        });

                        let mut compiled_values = vec![0.0; linked.nnz];
                        (linked.jacobian_values_eval)(
                            scenario.args.as_slice(),
                            &mut compiled_values,
                        );

                        match matrix_backend {
                            MatrixBackend::SparseCol => {
                                let sparse_matrix = scenario
                                    .prepared
                                    .as_prepared_problem()
                                    .jacobian_plan
                                    .assemble_sparse_col_mat(compiled_values.as_slice());
                                let solve_once = solve_sparse_lu(&sparse_matrix, rhs.as_slice());
                                let solve_ms = average_linear_solve_ms(
                                    scenario.runtime_samples,
                                    scenario.runtime_iters,
                                    || {
                                        black_box(solve_sparse_lu(&sparse_matrix, rhs.as_slice()));
                                    },
                                );
                                linear_rows.push(LinearSolverStoryRow {
                                    scenario: scenario.label.to_string(),
                                    source: "Compiled".to_string(),
                                    matrix_backend: "Sparse".to_string(),
                                    variant: variant.label.clone(),
                                    preset: preset.label().to_string(),
                                    linear_solver: "faer_sparse_lu".to_string(),
                                    layout: "-".to_string(),
                                    refinement: "-".to_string(),
                                    direct_rr: "-".to_string(),
                                    final_rr: "-".to_string(),
                                    solve_rr: "-".to_string(),
                                    total_ms: format!("{:.3}", solve_ms),
                                    solve_diff: format!(
                                        "{:.3e}",
                                        max_abs_diff(&baseline_solution, &solve_once)
                                    ),
                                    relative_x_diff: format!(
                                        "{:.3e}",
                                        relative_x_diff(&solve_once, &baseline_solution)
                                    ),
                                    status: "ok".to_string(),
                                });
                            }
                            MatrixBackend::Banded => {
                                let assembly = scenario
                                    .prepared
                                    .as_prepared_banded_problem()
                                    .jacobian_plan
                                    .assemble_banded_assembly(compiled_values.as_slice());
                                let dense = banded_assembly_to_dense(&assembly);
                                for banded_solver in BandedSolverUnderTest::variants() {
                                    let (mut metrics, solution_once) =
                                        solve_banded_solver_for_scenario(
                                            scenario.label,
                                            &assembly,
                                            rhs.as_slice(),
                                            banded_solver,
                                        );
                                    metrics.solve_rr = relative_dense_residual(
                                        &dense,
                                        &solution_once,
                                        rhs.as_slice(),
                                    );
                                    metrics.relative_x_diff =
                                        relative_x_diff(&solution_once, &baseline_solution);

                                    linear_rows.push(LinearSolverStoryRow {
                                        scenario: scenario.label.to_string(),
                                        source: "Compiled".to_string(),
                                        matrix_backend: "Banded".to_string(),
                                        variant: variant.label.clone(),
                                        preset: preset.label().to_string(),
                                        linear_solver: metrics.linear_solver,
                                        layout: metrics.layout,
                                        refinement: metrics.refinement,
                                        direct_rr: fmt_metric_or_dash(metrics.direct_rr),
                                        final_rr: fmt_metric_or_dash(metrics.final_rr),
                                        solve_rr: fmt_metric_or_dash(metrics.solve_rr),
                                        total_ms: format!(
                                            "{:.3}",
                                            average_linear_solve_ms(
                                                scenario.runtime_samples,
                                                scenario.runtime_iters,
                                                || {
                                                    black_box(solve_banded_solver_for_scenario(
                                                        scenario.label,
                                                        &assembly,
                                                        rhs.as_slice(),
                                                        banded_solver,
                                                    ));
                                                },
                                            )
                                        ),
                                        solve_diff: format!(
                                            "{:.3e}",
                                            max_abs_diff(&baseline_solution, &solution_once)
                                        ),
                                        relative_x_diff: format!("{:.3e}", metrics.relative_x_diff),
                                        status: metrics.status,
                                    });
                                }
                            }
                            _ => {}
                        }
                    } else {
                        runtime_rows.push(CallbackRuntimeMatrixRow {
                            scenario: scenario.label.to_string(),
                            source: "Compiled".to_string(),
                            matrix_backend: matrix_backend_label(matrix_backend).to_string(),
                            variant: variant.label.clone(),
                            preset: preset.label().to_string(),
                            residual_ms: "NaN".to_string(),
                            jacobian_ms: "NaN".to_string(),
                            total_ms: "NaN".to_string(),
                            residual_diff: "NaN".to_string(),
                            jacobian_diff: "NaN".to_string(),
                            status: metrics.status.clone(),
                        });
                    }

                    let _ = unregister_linked_sparse_backend(&problem_key);
                }
            }
        }

        let mut stage_builder = Builder::default();
        stage_builder.push_record([
            "scenario",
            "source",
            "matrix_backend",
            "variant",
            "preset",
            "compile_mode",
            "symbolic_ms",
            "callable_prep_ms",
            "materialize_ms",
            "build_ms",
            "link_ms",
            "first_issue_ms",
            "total_to_outputs_ms",
            "status",
        ]);
        for row in stage_rows {
            stage_builder.push_record([
                row.scenario,
                row.source,
                row.matrix_backend,
                row.variant,
                row.preset,
                row.compile_mode,
                row.symbolic_ms,
                row.callable_prep_ms,
                row.materialize_ms,
                row.build_ms,
                row.link_ms,
                row.first_issue_ms,
                row.total_to_outputs_ms,
                row.status,
            ]);
        }
        let mut stage_table = stage_builder.build();
        stage_table.with(Style::rounded());
        println!(
            "[BVP story] callable/bootstrap table scenario={}",
            scenario.label
        );
        println!("{stage_table}");

        let mut runtime_builder = Builder::default();
        runtime_builder.push_record([
            "scenario",
            "source",
            "matrix_backend",
            "variant",
            "preset",
            "residual_ms",
            "jacobian_ms",
            "total_ms",
            "residual_diff",
            "jacobian_diff",
            "status",
        ]);
        for row in runtime_rows {
            runtime_builder.push_record([
                row.scenario,
                row.source,
                row.matrix_backend,
                row.variant,
                row.preset,
                row.residual_ms,
                row.jacobian_ms,
                row.total_ms,
                row.residual_diff,
                row.jacobian_diff,
                row.status,
            ]);
        }
        let mut runtime_table = runtime_builder.build();
        runtime_table.with(Style::rounded());
        println!(
            "[BVP story] callback runtime/correctness table scenario={}",
            scenario.label
        );
        println!("{runtime_table}");

        let mut linear_builder = Builder::default();
        linear_builder.push_record([
            "scenario",
            "source",
            "matrix_backend",
            "variant",
            "preset",
            "linear_solver",
            "layout",
            "refinement",
            "direct_rr",
            "final_rr",
            "solve_rr",
            "total_ms",
            "solve_diff",
            "rel_x_diff",
            "status",
        ]);
        for row in linear_rows {
            linear_builder.push_record([
                row.scenario,
                row.source,
                row.matrix_backend,
                row.variant,
                row.preset,
                row.linear_solver,
                row.layout,
                row.refinement,
                row.direct_rr,
                row.final_rr,
                row.solve_rr,
                row.total_ms,
                row.solve_diff,
                row.relative_x_diff,
                row.status,
            ]);
        }
        let mut linear_table = linear_builder.build();
        linear_table.with(Style::rounded());
        println!(
            "[BVP story] linear solver table scenario={} (baseline=faer_sparse_lu)",
            scenario.label
        );
        println!("{linear_table}");
        flush_stdout();

        emit_progress(format!(
            "callable + linear-solver story finished scenario `{}`",
            scenario.label
        ));
    }
}
/*
#[test]
#[ignore = "heavy combustion-1000 diagnostic for lapack-style banded factorization on a real generated Jacobian"]
fn diagnose_combustion_1000_real_banded_jacobian_lapack_factorization() {
    let spec = ScenarioSpec {
        label: "combustion-1000",
        runtime_iters: 1,
        runtime_samples: 1,
        build_jacobian: || build_combustion_bvp_case(1000),
    };

    emit_progress("building combustion-1000 scenario for lapack-style banded diagnostic");
    let scenario = build_scenario_data_with_backend(&spec, BvpSymbolicAssemblyBackend::ExprLegacy);
    let (_callable_row, runtime_row, _jacobian_once, banded_assembly) =
        measure_lambdify_matrix_runtime(&scenario, MatrixBackend::Banded);
    let assembly =
        banded_assembly.expect("combustion-1000 banded lambdify path should produce BandedAssembly");
    let compact = assembly
        .to_banded()
        .expect("banded assembly should convert to compact banded storage");
    let dense = banded_assembly_to_dense(&assembly);

    let mut lapack = LapackStyleBandedLu::new(compact.n(), compact.kl(), compact.ku())
        .expect("lapack-style workspace should allocate for combustion-1000 compact band");
    let steps = lapack
        .factor_from_with_diagnostics(&compact)
        .expect("lapack-style factorization diagnostics should succeed");
    let factor_rel = lapack
        .factor_residual_relative(&compact)
        .expect("factor residual should be available after lapack-style diagnostics run");

    let rhs: Vec<f64> = scenario
        .lambdify_residual
        .iter()
        .map(|value| -*value)
        .collect();
    let mut x_lapack = rhs.clone();
    lapack
        .solve_in_place(&mut x_lapack)
        .expect("lapack-style solve should succeed after diagnostic factorization");
    let solve_rr = relative_dense_residual(&dense, &x_lapack, rhs.as_slice());

    let sparse = scenario
        .prepared
        .as_prepared_problem()
        .jacobian_plan
        .assemble_sparse_col_mat(&scenario.lambdify_jacobian);
    let x_sparse = solve_sparse_lu(&sparse, rhs.as_slice());
    let solve_diff = max_abs_diff(x_lapack.as_slice(), x_sparse.as_slice());
    let rel_x_diff = relative_x_diff(x_lapack.as_slice(), x_sparse.as_slice());

    let worst_multiplier = steps
        .iter()
        .max_by(|lhs, rhs| lhs.max_abs_multiplier.total_cmp(&rhs.max_abs_multiplier))
        .expect("factor diagnostics should contain at least one step");
    let worst_active = steps
        .iter()
        .max_by(|lhs, rhs| lhs.max_abs_active_entry.total_cmp(&rhs.max_abs_active_entry))
        .expect("factor diagnostics should contain at least one step");
    let min_pivot = steps
        .iter()
        .min_by(|lhs, rhs| lhs.pivot_value.abs().total_cmp(&rhs.pivot_value.abs()))
        .expect("factor diagnostics should contain at least one step");
    let last_step = steps
        .last()
        .expect("factor diagnostics should contain at least one step");

    println!(
        "[BVP story] combustion-1000 real banded Jacobian diagnostic via LapackStyleBandedLu"
    );
    println!(
        "scenario={} symbolic_backend={} matrix_backend={} callable_status={} residual_diff={} jacobian_diff={}",
        scenario.label,
        "ExprLegacy",
        "Banded",
        runtime_row.status,
        runtime_row.residual_diff,
        runtime_row.jacobian_diff
    );
    println!(
        "n={} kl={} ku={} steps={} factor_rel={:.3e} solve_rr={:.3e} solve_diff={:.3e} rel_x_diff={:.3e}",
        compact.n(),
        compact.kl(),
        compact.ku(),
        steps.len(),
        factor_rel,
        solve_rr,
        solve_diff,
        rel_x_diff
    );
    println!(
        "min_abs_pivot: step={} pivot_row={} value={:.3e}",
        min_pivot.step,
        min_pivot.pivot_row,
        min_pivot.pivot_value
    );
    println!(
        "worst_multiplier: step={} pivot_row={} max_abs_multiplier={:.3e} workspace_linf={:.3e}",
        worst_multiplier.step,
        worst_multiplier.pivot_row,
        worst_multiplier.max_abs_multiplier,
        worst_multiplier.workspace_linf
    );
    println!(
        "worst_active_region: step={} ju={} max_abs_active_entry={:.3e}",
        worst_active.step,
        worst_active.ju,
        worst_active.max_abs_active_entry
    );
    println!(
        "last_step: step={} ju={} pivot_row={} pivot_value={:.3e} workspace_linf={:.3e}",
        last_step.step,
        last_step.ju,
        last_step.pivot_row,
        last_step.pivot_value,
        last_step.workspace_linf
    );
    println!("[BVP story] suspicious factorization steps:");
    let mut suspicious_count = 0usize;
    for step in steps.iter().filter(|step| {
        step.pivot_value.abs() < 1.0e-8
            || step.max_abs_multiplier > 1.0e2
            || step.max_abs_active_entry > 1.0e4
    }) {
        suspicious_count += 1;
        eprintln!("{step:?}");
    }
    if suspicious_count == 0 {
        println!("none");
    } else {
        println!("count={suspicious_count}");
    }

    let mut smallest_pivots = steps.iter().collect::<Vec<_>>();
    smallest_pivots.sort_by(|lhs, rhs| lhs.pivot_value.abs().total_cmp(&rhs.pivot_value.abs()));
    println!("[BVP story] top-5 smallest pivots:");
    for step in smallest_pivots.into_iter().take(5) {
        println!(
            "step={} pivot_row={} pivot_value={:.3e} max_abs_multiplier={:.3e} max_abs_active_entry={:.3e} workspace_linf={:.3e}",
            step.step,
            step.pivot_row,
            step.pivot_value,
            step.max_abs_multiplier,
            step.max_abs_active_entry,
            step.workspace_linf
        );
    }

    let mut largest_multipliers = steps.iter().collect::<Vec<_>>();
    largest_multipliers.sort_by(|lhs, rhs| {
        rhs.max_abs_multiplier
            .total_cmp(&lhs.max_abs_multiplier)
    });
    println!("[BVP story] top-5 largest multipliers:");
    for step in largest_multipliers.into_iter().take(5) {
        println!(
            "step={} pivot_row={} pivot_value={:.3e} max_abs_multiplier={:.3e} max_abs_active_entry={:.3e} workspace_linf={:.3e}",
            step.step,
            step.pivot_row,
            step.pivot_value,
            step.max_abs_multiplier,
            step.max_abs_active_entry,
            step.workspace_linf
        );
    }

    assert_eq!(
        steps.len(),
        compact.n(),
        "lapack-style diagnostics should report one step per combustion-1000 Jacobian column"
    );
    assert!(
        factor_rel.is_finite() && solve_rr.is_finite() && rel_x_diff.is_finite(),
        "diagnostic metrics should stay finite on the real combustion-1000 banded Jacobian"
    );
}

fn real_combustion_1000_banded_jacobian_fixture() -> (ScenarioData, CallbackRuntimeMatrixRow, BandedAssembly, crate::somelinalg::banded::storage::Banded<f64>, Vec<f64>) {
    let spec = ScenarioSpec {
        label: "combustion-1000",
        runtime_iters: 1,
        runtime_samples: 1,
        build_jacobian: || build_combustion_bvp_case(1000),
    };

    emit_progress("building combustion-1000 scenario for real banded Jacobian fixture");
    let scenario = build_scenario_data_with_backend(&spec, BvpSymbolicAssemblyBackend::ExprLegacy);
    let (_callable_row, runtime_row, _jacobian_once, banded_assembly) =
        measure_lambdify_matrix_runtime(&scenario, MatrixBackend::Banded);
    let assembly =
        banded_assembly.expect("combustion-1000 banded lambdify path should produce BandedAssembly");
    let compact = assembly
        .to_banded()
        .expect("banded assembly should convert to compact banded storage");
    let rhs = scenario
        .lambdify_residual
        .iter()
        .map(|value| -*value)
        .collect::<Vec<_>>();
    (scenario, runtime_row, assembly, compact, rhs)
}

#[test]
#[ignore = "heavy combustion-1000 diagnostic for trailing dense tail inspection on a real generated Jacobian"]
fn debug_extract_trailing_dense_block_combustion_1000_real_banded_jacobian() {
    let (scenario, runtime_row, _assembly, compact, _rhs) =
        real_combustion_1000_banded_jacobian_fixture();

    let tail20 = extract_trailing_dense_block(&compact, 20);
    let tail40 = extract_trailing_dense_block(&compact, 40);
    let tail60 = extract_trailing_dense_block(&compact, 60);

    let (mn20, mx20) = dense_diag_abs_minmax(&tail20);
    let (mn40, mx40) = dense_diag_abs_minmax(&tail40);
    let (mn60, mx60) = dense_diag_abs_minmax(&tail60);

    println!(
        "[BVP story] combustion-1000 trailing dense block diagnostic"
    );
    println!(
        "scenario={} symbolic_backend={} matrix_backend={} callable_status={} residual_diff={} jacobian_diff={}",
        scenario.label,
        "ExprLegacy",
        "Banded",
        runtime_row.status,
        runtime_row.residual_diff,
        runtime_row.jacobian_diff
    );
    println!(
        "tail20: linf={:e}, min|diag|={:e}, max|diag|={:e}",
        dense_linf_norm(&tail20),
        mn20,
        mx20
    );
    println!(
        "tail40: linf={:e}, min|diag|={:e}, max|diag|={:e}",
        dense_linf_norm(&tail40),
        mn40,
        mx40
    );
    println!(
        "tail60: linf={:e}, min|diag|={:e}, max|diag|={:e}",
        dense_linf_norm(&tail60),
        mn60,
        mx60
    );
    println!("[BVP story] tail20 preview:");
    dense_print_small(&tail20, 20, 20);

    assert!(
        mn20.is_finite() && mn40.is_finite() && mn60.is_finite(),
        "trailing dense block diagnostics should stay finite"
    );
}

#[test]
#[ignore = "heavy combustion-1000 diagnostic for row scaling on a real generated Jacobian"]
fn diagnose_combustion_1000_real_banded_jacobian_with_row_scaling() {
    let (scenario, runtime_row, _assembly, compact, rhs) =
        real_combustion_1000_banded_jacobian_fixture();

    let sparse = scenario
        .prepared
        .as_prepared_problem()
        .jacobian_plan
        .assemble_sparse_col_mat(&scenario.lambdify_jacobian);
    let x_ref = solve_sparse_lu(&sparse, rhs.as_slice());

    let mut lu0 = LapackStyleBandedLu::new(compact.n(), compact.kl(), compact.ku())
        .expect("lapack-style workspace should allocate for unscaled combustion-1000 matrix");
    lu0.factor_from(&compact)
        .expect("unscaled lapack-style factorization should succeed");

    let mut x0 = rhs.clone();
    lu0.solve_in_place(&mut x0)
        .expect("unscaled lapack-style solve should succeed");

    let rr0 = relative_banded_residual(&compact, &x0, &rhs);
    let fr0 = lu0
        .factor_residual_relative(&compact)
        .expect("unscaled factor residual should be available");
    let diff0 = max_abs_diff(x0.as_slice(), x_ref.as_slice());
    let rel_diff0 = relative_x_diff(x0.as_slice(), x_ref.as_slice());

    let (scaled, row_scales) = row_scaled_banded_copy_with_scales(&compact);
    let rhs_scaled = apply_row_scales_to_rhs(&rhs, &row_scales);

    let mut lu1 = LapackStyleBandedLu::new(scaled.n(), scaled.kl(), scaled.ku())
        .expect("lapack-style workspace should allocate for row-scaled combustion-1000 matrix");
    lu1.factor_from(&scaled)
        .expect("row-scaled lapack-style factorization should succeed");

    let mut x1 = rhs_scaled.clone();
    lu1.solve_in_place(&mut x1)
        .expect("row-scaled lapack-style solve should succeed");

    let rr1 = relative_banded_residual(&compact, &x1, &rhs);
    let fr1 = lu1
        .factor_residual_relative(&scaled)
        .expect("row-scaled factor residual should be available");
    let diff1 = max_abs_diff(x1.as_slice(), x_ref.as_slice());
    let rel_diff1 = relative_x_diff(x1.as_slice(), x_ref.as_slice());
    let x_diff = max_abs_diff(x0.as_slice(), x1.as_slice());

    let seeds = [7_u64, 17_u64, 27_u64];
    let mut solve_rr_unscaled_samples = Vec::with_capacity(seeds.len());
    let mut solve_rr_scaled_samples = Vec::with_capacity(seeds.len());
    let mut solve_rel_x_unscaled_samples = Vec::with_capacity(seeds.len());
    let mut solve_rel_x_scaled_samples = Vec::with_capacity(seeds.len());
    let mut solve_rr_faer_samples = Vec::with_capacity(seeds.len());
    let mut probe_relx_vs_faer_unscaled_samples = Vec::with_capacity(seeds.len());
    let mut probe_relx_vs_faer_scaled_samples = Vec::with_capacity(seeds.len());
    let mut probe_xdiff_vs_faer_unscaled_samples = Vec::with_capacity(seeds.len());
    let mut probe_xdiff_vs_faer_scaled_samples = Vec::with_capacity(seeds.len());
    for seed in seeds {
        let (x_true, b_probe) = generate_rhs_from_known_solution(&compact, seed);
        let x_probe_faer = solve_sparse_lu(&sparse, &b_probe);

        solve_rr_faer_samples.push(relative_dense_residual(
            &sparse.to_DMatrixType(),
            x_probe_faer.as_slice(),
            &b_probe,
        ));

        let mut x_probe_unscaled = b_probe.clone();
        lu0.solve_in_place(&mut x_probe_unscaled)
            .expect("unscaled lapack-style probe solve should succeed");
        solve_rr_unscaled_samples.push(relative_banded_residual(
            &compact,
            &x_probe_unscaled,
            &b_probe,
        ));
        solve_rel_x_unscaled_samples.push(relative_x_diff(
            x_probe_unscaled.as_slice(),
            x_true.as_slice(),
        ));
        probe_relx_vs_faer_unscaled_samples.push(relative_x_diff(
            x_probe_unscaled.as_slice(),
            x_probe_faer.as_slice(),
        ));
        probe_xdiff_vs_faer_unscaled_samples.push(max_abs_diff(
            x_probe_unscaled.as_slice(),
            x_probe_faer.as_slice(),
        ));

        let b_probe_scaled = apply_row_scales_to_rhs(&b_probe, &row_scales);
        let mut x_probe_scaled = b_probe_scaled.clone();
        lu1.solve_in_place(&mut x_probe_scaled)
            .expect("row-scaled lapack-style probe solve should succeed");
        solve_rr_scaled_samples.push(relative_banded_residual(
            &compact,
            &x_probe_scaled,
            &b_probe,
        ));
        solve_rel_x_scaled_samples.push(relative_x_diff(
            x_probe_scaled.as_slice(),
            x_true.as_slice(),
        ));
        probe_relx_vs_faer_scaled_samples.push(relative_x_diff(
            x_probe_scaled.as_slice(),
            x_probe_faer.as_slice(),
        ));
        probe_xdiff_vs_faer_scaled_samples.push(max_abs_diff(
            x_probe_scaled.as_slice(),
            x_probe_faer.as_slice(),
        ));
    }

    let probe_rr_unscaled = solve_rr_unscaled_samples
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let probe_rr_scaled = solve_rr_scaled_samples
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let probe_relx_unscaled = solve_rel_x_unscaled_samples
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let probe_relx_scaled = solve_rel_x_scaled_samples
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let probe_rr_faer = solve_rr_faer_samples
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let probe_relx_vs_faer_unscaled = probe_relx_vs_faer_unscaled_samples
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let probe_relx_vs_faer_scaled = probe_relx_vs_faer_scaled_samples
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let probe_xdiff_vs_faer_unscaled = probe_xdiff_vs_faer_unscaled_samples
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let probe_xdiff_vs_faer_scaled = probe_xdiff_vs_faer_scaled_samples
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);

    let mut largest_scales = row_scales
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<_>>();
    largest_scales.sort_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1));

    let tail_start = row_scales.len().saturating_sub(20);

    println!(
        "[BVP story] combustion-1000 row scaling diagnostic via LapackStyleBandedLu"
    );
    println!(
        "scenario={} symbolic_backend={} matrix_backend={} callable_status={} residual_diff={} jacobian_diff={}",
        scenario.label,
        "ExprLegacy",
        "Banded",
        runtime_row.status,
        runtime_row.residual_diff,
        runtime_row.jacobian_diff
    );
    println!(
        "unscaled: factor_rel={:e}, solve_rr={:e}, solve_diff={:e}, rel_x_diff={:e}",
        fr0, rr0, diff0, rel_diff0
    );
    println!(
        "row-scaled: factor_rel={:e}, solve_rr={:e}, solve_diff={:e}, rel_x_diff={:e}",
        fr1, rr1, diff1, rel_diff1
    );
    println!("x_diff(unscaled vs scaled solve) = {:e}", x_diff);
    println!(
        "probe solves (max over seeds {:?}): unscaled solve_rr={:e}, scaled solve_rr={:e}, unscaled rel_x_diff={:e}, scaled rel_x_diff={:e}",
        seeds,
        probe_rr_unscaled,
        probe_rr_scaled,
        probe_relx_unscaled,
        probe_relx_scaled
    );
    println!(
        "probe solves vs faer (max over seeds {:?}): faer solve_rr={:e}, unscaled rel_x_diff_vs_faer={:e}, scaled rel_x_diff_vs_faer={:e}, unscaled x_diff_vs_faer={:e}, scaled x_diff_vs_faer={:e}",
        seeds,
        probe_rr_faer,
        probe_relx_vs_faer_unscaled,
        probe_relx_vs_faer_scaled,
        probe_xdiff_vs_faer_unscaled,
        probe_xdiff_vs_faer_scaled
    );
    println!("[BVP story] top-10 row scales:");
    for (row, scale) in largest_scales.iter().take(10) {
        println!(
            "row={} scale={:e} tail_row={}",
            row,
            scale,
            if *row >= tail_start { "yes" } else { "no" }
        );
    }
    println!("[BVP story] tail row scales (last 20 rows):");
    for (row, scale) in row_scales.iter().enumerate().skip(tail_start) {
        println!("row={} scale={:e}", row, scale);
    }

    assert!(
        fr0.is_finite()
            && rr0.is_finite()
            && fr1.is_finite()
            && rr1.is_finite()
            && rel_diff0.is_finite()
            && rel_diff1.is_finite()
            && probe_rr_unscaled.is_finite()
            && probe_rr_scaled.is_finite()
            && probe_relx_unscaled.is_finite()
            && probe_relx_scaled.is_finite()
            && probe_rr_faer.is_finite()
            && probe_relx_vs_faer_unscaled.is_finite()
            && probe_relx_vs_faer_scaled.is_finite()
            && probe_xdiff_vs_faer_unscaled.is_finite()
            && probe_xdiff_vs_faer_scaled.is_finite(),
        "row-scaling diagnostic metrics should stay finite"
    );
}

*/

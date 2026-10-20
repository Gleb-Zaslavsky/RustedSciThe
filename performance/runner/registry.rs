use crate::error::RunnerError;

#[derive(Clone, Debug, Default)]
pub struct ScenarioOutput {
    pub metrics: Vec<(String, String, f64)>,
    /// Native timing relationships for metrics whose solver API exposes them.
    /// This metadata stays inside the performance runner and is persisted via
    /// the existing metric scope fields in the baseline schema.
    pub native_scopes: Vec<NativeMetricScope>,
    pub counters: Vec<(String, u64)>,
    pub diagnostics: Vec<(String, String, f64)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeMetricScope {
    pub metric_id: String,
    pub stage: String,
    pub kind: String,
    pub parent: Option<String>,
}

pub struct Scenario {
    pub id: &'static str,
    pub solver: &'static str,
    pub groups: &'static [&'static str],
    /// Fixed equation family and size; changing either requires a new version.
    pub workload: &'static str,
    /// Symbolic frontend, or `not-applicable` for a numerical-only route.
    pub frontend: &'static str,
    /// Evaluator/backend lifecycle, kept distinct from the frontend.
    pub execution: &'static str,
    /// Linear algebra storage/layout, or `not-applicable`.
    pub linear_algebra: &'static str,
    /// Callback dispatch policy, or `not-applicable` where unsupported.
    pub parallel_policy: &'static str,
    /// Native telemetry state during the timed solve.
    pub telemetry: &'static str,
    pub version: u32,
    /// Only these timings influence PASS/FAIL; other timings remain diagnostic.
    pub gate_metrics: &'static [&'static str],
    pub description: &'static str,
    pub run: fn() -> Result<ScenarioOutput, RunnerError>,
    /// Optional instrumented pass, invoked after all Off-mode samples finish.
    pub diagnose: Option<fn(&Scenario) -> Result<ScenarioOutput, RunnerError>>,
}

macro_rules! radau_route {
    ($id:literal, $frontend:literal, $execution:literal, $layout:literal, $policy:literal, $run:path) => {
        Scenario {
            id: $id,
            solver: "radau",
            groups: &["ivp", "stiff", "radau-matrix"],
            workload: "diffusion-chain, dimension=16, t=[0,0.01]",
            frontend: $frontend,
            execution: $execution,
            linear_algebra: $layout,
            parallel_policy: $policy,
            telemetry: "off wall gate + Timings stage diagnostic",
            version: 1,
            gate_metrics: &["scenario_wall_ms"],
            description: "Matched Radau diffusion route; AOT requires tcc and uses the runner-owned warm cache",
            run: $run,
            diagnose: Some(crate::scenarios::radau::diagnose),
        }
    };
}

macro_rules! bdf_route {
    ($id:literal, $frontend:literal, $execution:literal, $policy:literal, $run:path) => {
        Scenario {
            id: $id,
            solver: "bdf",
            groups: &["ivp", "stiff", "bdf-matrix"],
            workload: "diffusion-chain, dimension=64, t=[0,0.01]",
            frontend: $frontend,
            execution: $execution,
            linear_algebra: "not-applicable (fixed Dense implementation)",
            parallel_policy: $policy,
            telemetry: "off wall gate + Detailed stage diagnostic",
            version: 1,
            gate_metrics: &["scenario_wall_ms"],
            description:
                "Matched BDF 64-state diffusion route; AOT uses runner-owned warm artifacts",
            run: $run,
            diagnose: Some(crate::scenarios::bdf::diagnose),
        }
    };
}

macro_rules! lsode2_route {
    ($id:literal, $frontend:literal, $execution:literal, $layout:literal, $policy:literal, $run:path) => {
        Scenario {
            id: $id,
            solver: "lsode2",
            groups: &["ivp", "stiff", "lsode2-matrix"],
            workload: "diffusion-chain, dimension=32, t=[0,0.01], fixed BDF-only faithful route",
            frontend: $frontend,
            execution: $execution,
            linear_algebra: $layout,
            parallel_policy: $policy,
            telemetry: "off wall gate + Detailed stage diagnostic",
            version: 1,
            gate_metrics: &["scenario_wall_ms"],
            description: "Matched LSODE2 diffusion route; AOT uses a runner-owned warm cache and TCC",
            run: $run,
            diagnose: Some(crate::scenarios::lsode2::diagnose),
        }
    };
}

static CORE_SCENARIOS: &[Scenario] = &[
    Scenario {
        id: "banded_faithful_tridiagonal",
        solver: "somelinalg",
        groups: &["linear-algebra", "banded"],
        workload: "4096-row compact tridiagonal known-solution system",
        frontend: "not-applicable",
        execution: "faithful-banded-lu",
        linear_algebra: "compact-banded",
        parallel_policy: "sequential",
        telemetry: "off",
        version: 2,
        gate_metrics: &["scenario_wall_ms"],
        description: "Faithful compact band LU factor and solve with known solution",
        run: crate::scenarios::banded::run_tridiagonal,
        diagnose: None,
    },
    Scenario {
        id: "bdf_robertson",
        solver: "bdf",
        groups: &["ivp", "stiff"],
        workload: "Robertson, dimension=3, t=[0,0.1]",
        frontend: "ExprLegacy",
        execution: "Lambdify",
        linear_algebra: "not-applicable (fixed Dense implementation)",
        parallel_policy: "sequential",
        telemetry: "off",
        version: 2,
        gate_metrics: &["scenario_wall_ms"],
        description: "BDF Robertson solve using the shared benchmark workload builder",
        run: crate::scenarios::bdf::run_robertson,
        diagnose: None,
    },
    bdf_route!(
        "bdf_diff64_lambdify_expr_seq",
        "ExprLegacy",
        "Lambdify",
        "Sequential",
        crate::scenarios::bdf::diffusion_expr_lambdify_seq
    ),
    bdf_route!(
        "bdf_diff64_lambdify_atom_seq",
        "AtomView",
        "Lambdify",
        "Sequential",
        crate::scenarios::bdf::diffusion_atom_lambdify_seq
    ),
    bdf_route!(
        "bdf_diff64_lambdify_expr_parallel",
        "ExprLegacy",
        "Lambdify",
        "Parallel(min_work=1)",
        crate::scenarios::bdf::diffusion_expr_lambdify_parallel
    ),
    bdf_route!(
        "bdf_diff64_lambdify_atom_parallel",
        "AtomView",
        "Lambdify",
        "Parallel(min_work=1)",
        crate::scenarios::bdf::diffusion_atom_lambdify_parallel
    ),
    bdf_route!(
        "bdf_diff64_aot_expr_whole",
        "ExprLegacy",
        "AOT(warm-cache)",
        "Whole(default chunks)",
        crate::scenarios::bdf::diffusion_expr_aot_whole
    ),
    bdf_route!(
        "bdf_diff64_aot_atom_whole",
        "AtomView",
        "AOT(warm-cache)",
        "Whole(default chunks)",
        crate::scenarios::bdf::diffusion_atom_aot_whole
    ),
    bdf_route!(
        "bdf_diff64_aot_expr_parallel2",
        "ExprLegacy",
        "AOT(warm-cache)",
        "Parallel2",
        crate::scenarios::bdf::diffusion_expr_aot_parallel2
    ),
    bdf_route!(
        "bdf_diff64_aot_atom_parallel2",
        "AtomView",
        "AOT(warm-cache)",
        "Parallel2",
        crate::scenarios::bdf::diffusion_atom_aot_parallel2
    ),
    Scenario {
        id: "lsode2_robertson",
        solver: "lsode2",
        groups: &["ivp", "stiff"],
        workload: "Robertson, dimension=3, t=[0,0.25]",
        frontend: "ExprLegacy",
        execution: "Lambdify",
        linear_algebra: "Dense",
        parallel_policy: "sequential",
        telemetry: "off",
        version: 2,
        gate_metrics: &["scenario_wall_ms"],
        description: "Fixed BDF-controller LSODE2 Robertson solve using shared IVP equations",
        run: crate::scenarios::lsode2::run_robertson,
        diagnose: None,
    },
    lsode2_route!(
        "lsode2_diff32_expr_dense_lambdify_seq",
        "ExprLegacy",
        "Lambdify",
        "Dense",
        "Sequential",
        crate::scenarios::lsode2::diff_expr_dense_lambdify_seq
    ),
    lsode2_route!(
        "lsode2_diff32_atom_dense_lambdify_seq",
        "AtomView",
        "Lambdify",
        "Dense",
        "Sequential",
        crate::scenarios::lsode2::diff_atom_dense_lambdify_seq
    ),
    lsode2_route!(
        "lsode2_diff32_expr_sparse_lambdify_seq",
        "ExprLegacy",
        "Lambdify",
        "Sparse",
        "Sequential",
        crate::scenarios::lsode2::diff_expr_sparse_lambdify_seq
    ),
    lsode2_route!(
        "lsode2_diff32_atom_sparse_lambdify_seq",
        "AtomView",
        "Lambdify",
        "Sparse",
        "Sequential",
        crate::scenarios::lsode2::diff_atom_sparse_lambdify_seq
    ),
    lsode2_route!(
        "lsode2_diff32_expr_banded_lambdify_seq",
        "ExprLegacy",
        "Lambdify",
        "Banded(1,1)",
        "Sequential",
        crate::scenarios::lsode2::diff_expr_banded_lambdify_seq
    ),
    lsode2_route!(
        "lsode2_diff32_atom_banded_lambdify_seq",
        "AtomView",
        "Lambdify",
        "Banded(1,1)",
        "Sequential",
        crate::scenarios::lsode2::diff_atom_banded_lambdify_seq
    ),
    lsode2_route!(
        "lsode2_diff32_expr_dense_lambdify_parallel",
        "ExprLegacy",
        "Lambdify",
        "Dense",
        "Parallel(min_work=1)",
        crate::scenarios::lsode2::diff_expr_dense_lambdify_parallel
    ),
    lsode2_route!(
        "lsode2_diff32_atom_dense_lambdify_parallel",
        "AtomView",
        "Lambdify",
        "Dense",
        "Parallel(min_work=1)",
        crate::scenarios::lsode2::diff_atom_dense_lambdify_parallel
    ),
    lsode2_route!(
        "lsode2_diff32_expr_sparse_lambdify_parallel",
        "ExprLegacy",
        "Lambdify",
        "Sparse",
        "Parallel(min_work=1)",
        crate::scenarios::lsode2::diff_expr_sparse_lambdify_parallel
    ),
    lsode2_route!(
        "lsode2_diff32_atom_sparse_lambdify_parallel",
        "AtomView",
        "Lambdify",
        "Sparse",
        "Parallel(min_work=1)",
        crate::scenarios::lsode2::diff_atom_sparse_lambdify_parallel
    ),
    lsode2_route!(
        "lsode2_diff32_expr_banded_lambdify_parallel",
        "ExprLegacy",
        "Lambdify",
        "Banded(1,1)",
        "Parallel(min_work=1)",
        crate::scenarios::lsode2::diff_expr_banded_lambdify_parallel
    ),
    lsode2_route!(
        "lsode2_diff32_atom_banded_lambdify_parallel",
        "AtomView",
        "Lambdify",
        "Banded(1,1)",
        "Parallel(min_work=1)",
        crate::scenarios::lsode2::diff_atom_banded_lambdify_parallel
    ),
    lsode2_route!(
        "lsode2_diff32_expr_dense_lambdify_auto",
        "ExprLegacy",
        "Lambdify",
        "Dense",
        "Auto(min_work=1)",
        crate::scenarios::lsode2::diff_expr_dense_lambdify_auto
    ),
    lsode2_route!(
        "lsode2_diff32_atom_dense_lambdify_auto",
        "AtomView",
        "Lambdify",
        "Dense",
        "Auto(min_work=1)",
        crate::scenarios::lsode2::diff_atom_dense_lambdify_auto
    ),
    lsode2_route!(
        "lsode2_diff32_expr_sparse_lambdify_auto",
        "ExprLegacy",
        "Lambdify",
        "Sparse",
        "Auto(min_work=1)",
        crate::scenarios::lsode2::diff_expr_sparse_lambdify_auto
    ),
    lsode2_route!(
        "lsode2_diff32_atom_sparse_lambdify_auto",
        "AtomView",
        "Lambdify",
        "Sparse",
        "Auto(min_work=1)",
        crate::scenarios::lsode2::diff_atom_sparse_lambdify_auto
    ),
    lsode2_route!(
        "lsode2_diff32_expr_banded_lambdify_auto",
        "ExprLegacy",
        "Lambdify",
        "Banded(1,1)",
        "Auto(min_work=1)",
        crate::scenarios::lsode2::diff_expr_banded_lambdify_auto
    ),
    lsode2_route!(
        "lsode2_diff32_atom_banded_lambdify_auto",
        "AtomView",
        "Lambdify",
        "Banded(1,1)",
        "Auto(min_work=1)",
        crate::scenarios::lsode2::diff_atom_banded_lambdify_auto
    ),
    lsode2_route!(
        "lsode2_diff32_expr_dense_aot_whole",
        "ExprLegacy",
        "AOT(warm-cache)",
        "Dense",
        "Whole(default chunks)",
        crate::scenarios::lsode2::diff_expr_dense_aot_whole
    ),
    lsode2_route!(
        "lsode2_diff32_atom_dense_aot_whole",
        "AtomView",
        "AOT(warm-cache)",
        "Dense",
        "Whole(default chunks)",
        crate::scenarios::lsode2::diff_atom_dense_aot_whole
    ),
    lsode2_route!(
        "lsode2_diff32_expr_sparse_aot_whole",
        "ExprLegacy",
        "AOT(warm-cache)",
        "Sparse",
        "Whole(default chunks)",
        crate::scenarios::lsode2::diff_expr_sparse_aot_whole
    ),
    lsode2_route!(
        "lsode2_diff32_atom_sparse_aot_whole",
        "AtomView",
        "AOT(warm-cache)",
        "Sparse",
        "Whole(default chunks)",
        crate::scenarios::lsode2::diff_atom_sparse_aot_whole
    ),
    lsode2_route!(
        "lsode2_diff32_expr_banded_aot_whole",
        "ExprLegacy",
        "AOT(warm-cache)",
        "Banded(1,1)",
        "Whole(default chunks)",
        crate::scenarios::lsode2::diff_expr_banded_aot_whole
    ),
    lsode2_route!(
        "lsode2_diff32_atom_banded_aot_whole",
        "AtomView",
        "AOT(warm-cache)",
        "Banded(1,1)",
        "Whole(default chunks)",
        crate::scenarios::lsode2::diff_atom_banded_aot_whole
    ),
    lsode2_route!(
        "lsode2_diff32_expr_dense_aot_parallel2",
        "ExprLegacy",
        "AOT(warm-cache)",
        "Dense",
        "Parallel2",
        crate::scenarios::lsode2::diff_expr_dense_aot_parallel2
    ),
    lsode2_route!(
        "lsode2_diff32_atom_dense_aot_parallel2",
        "AtomView",
        "AOT(warm-cache)",
        "Dense",
        "Parallel2",
        crate::scenarios::lsode2::diff_atom_dense_aot_parallel2
    ),
    lsode2_route!(
        "lsode2_diff32_expr_sparse_aot_parallel2",
        "ExprLegacy",
        "AOT(warm-cache)",
        "Sparse",
        "Parallel2",
        crate::scenarios::lsode2::diff_expr_sparse_aot_parallel2
    ),
    lsode2_route!(
        "lsode2_diff32_atom_sparse_aot_parallel2",
        "AtomView",
        "AOT(warm-cache)",
        "Sparse",
        "Parallel2",
        crate::scenarios::lsode2::diff_atom_sparse_aot_parallel2
    ),
    lsode2_route!(
        "lsode2_diff32_expr_banded_aot_parallel2",
        "ExprLegacy",
        "AOT(warm-cache)",
        "Banded(1,1)",
        "Parallel2",
        crate::scenarios::lsode2::diff_expr_banded_aot_parallel2
    ),
    lsode2_route!(
        "lsode2_diff32_atom_banded_aot_parallel2",
        "AtomView",
        "AOT(warm-cache)",
        "Banded(1,1)",
        "Parallel2",
        crate::scenarios::lsode2::diff_atom_banded_aot_parallel2
    ),
    Scenario {
        id: "radau_robertson",
        solver: "radau",
        groups: &["ivp", "stiff"],
        workload: "Robertson, dimension=3, t=[0,0.002]",
        frontend: "ExprLegacy",
        execution: "Lambdify",
        linear_algebra: "Dense",
        parallel_policy: "sequential",
        telemetry: "off",
        version: 2,
        gate_metrics: &["scenario_wall_ms"],
        description: "Radau Robertson solve using the shared IVP workload and analytic Jacobian",
        run: crate::scenarios::radau::run_robertson,
        diagnose: None,
    },
    radau_route!(
        "radau_diff16_lambdify_expr_dense_seq",
        "ExprLegacy",
        "Lambdify",
        "Dense",
        "Sequential",
        crate::scenarios::radau::lambdify_expr_dense_seq
    ),
    radau_route!(
        "radau_diff16_lambdify_expr_sparse_seq",
        "ExprLegacy",
        "Lambdify",
        "Sparse",
        "Sequential",
        crate::scenarios::radau::lambdify_expr_sparse_seq
    ),
    radau_route!(
        "radau_diff16_lambdify_expr_banded_seq",
        "ExprLegacy",
        "Lambdify",
        "Banded(1,1)",
        "Sequential",
        crate::scenarios::radau::lambdify_expr_banded_seq
    ),
    radau_route!(
        "radau_diff16_lambdify_atom_dense_seq",
        "AtomView",
        "Lambdify",
        "Dense",
        "Sequential",
        crate::scenarios::radau::lambdify_atom_dense_seq
    ),
    radau_route!(
        "radau_diff16_lambdify_atom_sparse_seq",
        "AtomView",
        "Lambdify",
        "Sparse",
        "Sequential",
        crate::scenarios::radau::lambdify_atom_sparse_seq
    ),
    radau_route!(
        "radau_diff16_lambdify_atom_banded_seq",
        "AtomView",
        "Lambdify",
        "Banded(1,1)",
        "Sequential",
        crate::scenarios::radau::lambdify_atom_banded_seq
    ),
    radau_route!(
        "radau_diff16_lambdify_expr_dense_parallel",
        "ExprLegacy",
        "Lambdify",
        "Dense",
        "Parallel",
        crate::scenarios::radau::lambdify_expr_dense_par
    ),
    radau_route!(
        "radau_diff16_lambdify_expr_sparse_parallel",
        "ExprLegacy",
        "Lambdify",
        "Sparse",
        "Parallel",
        crate::scenarios::radau::lambdify_expr_sparse_par
    ),
    radau_route!(
        "radau_diff16_lambdify_expr_banded_parallel",
        "ExprLegacy",
        "Lambdify",
        "Banded(1,1)",
        "Parallel",
        crate::scenarios::radau::lambdify_expr_banded_par
    ),
    radau_route!(
        "radau_diff16_lambdify_atom_dense_parallel",
        "AtomView",
        "Lambdify",
        "Dense",
        "Parallel",
        crate::scenarios::radau::lambdify_atom_dense_par
    ),
    radau_route!(
        "radau_diff16_lambdify_atom_sparse_parallel",
        "AtomView",
        "Lambdify",
        "Sparse",
        "Parallel",
        crate::scenarios::radau::lambdify_atom_sparse_par
    ),
    radau_route!(
        "radau_diff16_lambdify_atom_banded_parallel",
        "AtomView",
        "Lambdify",
        "Banded(1,1)",
        "Parallel",
        crate::scenarios::radau::lambdify_atom_banded_par
    ),
    radau_route!(
        "radau_diff16_aot_expr_dense_seq",
        "ExprLegacy",
        "AOT(warm-cache)",
        "Dense",
        "Sequential",
        crate::scenarios::radau::aot_expr_dense_seq
    ),
    radau_route!(
        "radau_diff16_aot_expr_sparse_seq",
        "ExprLegacy",
        "AOT(warm-cache)",
        "Sparse",
        "Sequential",
        crate::scenarios::radau::aot_expr_sparse_seq
    ),
    radau_route!(
        "radau_diff16_aot_expr_banded_seq",
        "ExprLegacy",
        "AOT(warm-cache)",
        "Banded(1,1)",
        "Sequential",
        crate::scenarios::radau::aot_expr_banded_seq
    ),
    radau_route!(
        "radau_diff16_aot_atom_dense_seq",
        "AtomView",
        "AOT(warm-cache)",
        "Dense",
        "Sequential",
        crate::scenarios::radau::aot_atom_dense_seq
    ),
    radau_route!(
        "radau_diff16_aot_atom_sparse_seq",
        "AtomView",
        "AOT(warm-cache)",
        "Sparse",
        "Sequential",
        crate::scenarios::radau::aot_atom_sparse_seq
    ),
    radau_route!(
        "radau_diff16_aot_atom_banded_seq",
        "AtomView",
        "AOT(warm-cache)",
        "Banded(1,1)",
        "Sequential",
        crate::scenarios::radau::aot_atom_banded_seq
    ),
    radau_route!(
        "radau_diff16_aot_expr_dense_parallel",
        "ExprLegacy",
        "AOT(warm-cache)",
        "Dense",
        "Parallel",
        crate::scenarios::radau::aot_expr_dense_par
    ),
    radau_route!(
        "radau_diff16_aot_expr_sparse_parallel",
        "ExprLegacy",
        "AOT(warm-cache)",
        "Sparse",
        "Parallel",
        crate::scenarios::radau::aot_expr_sparse_par
    ),
    radau_route!(
        "radau_diff16_aot_expr_banded_parallel",
        "ExprLegacy",
        "AOT(warm-cache)",
        "Banded(1,1)",
        "Parallel",
        crate::scenarios::radau::aot_expr_banded_par
    ),
    radau_route!(
        "radau_diff16_aot_atom_dense_parallel",
        "AtomView",
        "AOT(warm-cache)",
        "Dense",
        "Parallel",
        crate::scenarios::radau::aot_atom_dense_par
    ),
    radau_route!(
        "radau_diff16_aot_atom_sparse_parallel",
        "AtomView",
        "AOT(warm-cache)",
        "Sparse",
        "Parallel",
        crate::scenarios::radau::aot_atom_sparse_par
    ),
    radau_route!(
        "radau_diff16_aot_atom_banded_parallel",
        "AtomView",
        "AOT(warm-cache)",
        "Banded(1,1)",
        "Parallel",
        crate::scenarios::radau::aot_atom_banded_par
    ),
    Scenario {
        id: "bvp_sci_exponential_dense",
        solver: "bvp-sci",
        groups: &["bvp", "dense"],
        workload: "exponential two-point BVP, nodes=33",
        frontend: "ExprLegacy",
        execution: "Lambdify",
        linear_algebra: "Dense",
        parallel_policy: "sequential",
        telemetry: "off",
        version: 2,
        gate_metrics: &["scenario_wall_ms"],
        description: "Dense ExprLegacy BVP_sci solve for y'=-y with exact exponential reference",
        run: crate::scenarios::bvp_sci::run_exponential_dense,
        diagnose: None,
    },
    Scenario {
        id: "be_diffusion_chain_32",
        solver: "be",
        groups: &["ivp", "stiff", "large-system"],
        workload: "diffusion-chain, dimension=32, 16 fixed steps",
        frontend: "ExprLegacy",
        execution: "Lambdify",
        linear_algebra: "Dense",
        parallel_policy: "sequential",
        telemetry: "off",
        version: 2,
        gate_metrics: &["scenario_wall_ms"],
        description: "Backward Euler native diffusion solve reusing the Criterion benchmark fixture",
        run: crate::scenarios::be::run_diffusion_chain,
        diagnose: None,
    },
    Scenario {
        id: "nonlinear_quadratic_chain_16",
        solver: "nonlinear-systems",
        groups: &["nonlinear", "symbolic", "dense"],
        workload: "quadratic-chain, dimension=16, known root",
        frontend: "ExprLegacy",
        execution: "Lambdify",
        linear_algebra: "Dense",
        parallel_policy: "sequential",
        telemetry: "off",
        version: 2,
        gate_metrics: &["scenario_wall_ms"],
        description: "Prepared ExprLegacy quadratic-chain Newton solve with known root",
        run: crate::scenarios::nonlinear::run_quadratic_chain,
        diagnose: None,
    },
];

/// The fixed registry is stored as two compile-time tables so the BVP_Damp
/// cross-product remains isolated in its own scenario module.
pub struct ScenarioRegistry;

impl ScenarioRegistry {
    pub fn iter(&self) -> impl Iterator<Item = &'static Scenario> {
        CORE_SCENARIOS
            .iter()
            .chain(crate::scenarios::bvp_damp::SCENARIOS.iter().flatten())
            .chain(crate::scenarios::bvp_sci::MATRIX_SCENARIOS.iter().flatten())
            .chain(crate::scenarios::be::MATRIX_SCENARIOS.iter())
            .chain(crate::scenarios::nonlinear_matrix::MATRIX_SCENARIOS.iter())
            .chain(crate::scenarios::optimization::MATRIX_SCENARIOS.iter())
    }
}

impl std::ops::Index<usize> for ScenarioRegistry {
    type Output = Scenario;

    fn index(&self, index: usize) -> &Self::Output {
        self.iter()
            .nth(index)
            .expect("scenario registry index out of bounds")
    }
}

pub static SCENARIOS: ScenarioRegistry = ScenarioRegistry;

#[derive(Clone, Debug, Default)]
pub struct Selector {
    pub solver: Option<String>,
    pub scenario: Option<String>,
    pub group: Option<String>,
    pub all: bool,
}

pub fn select(selector: &Selector) -> Result<Vec<&'static Scenario>, RunnerError> {
    let kinds = usize::from(selector.solver.is_some())
        + usize::from(selector.scenario.is_some())
        + usize::from(selector.group.is_some())
        + usize::from(selector.all);
    if kinds != 1 {
        return Err(RunnerError::Configuration(
            "choose exactly one of --solver, --scenario, --group or --all".into(),
        ));
    }
    if selector.all {
        return Ok(SCENARIOS.iter().collect());
    }
    let selected: Vec<_> = SCENARIOS
        .iter()
        .filter(|scenario| {
            selector
                .solver
                .as_deref()
                .is_none_or(|solver| scenario.solver == solver)
                && selector
                    .scenario
                    .as_deref()
                    .is_none_or(|id| scenario.id == id)
                && selector
                    .group
                    .as_deref()
                    .is_none_or(|group| scenario.groups.contains(&group))
        })
        .collect();
    if selected.is_empty() {
        let selection = selector
            .solver
            .as_deref()
            .or(selector.scenario.as_deref())
            .or(selector.group.as_deref())
            .unwrap_or("<none>");
        return Err(RunnerError::UnknownSelection(selection.into()));
    }
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solver_selection_is_addressable_and_does_not_expand_to_all_scenarios() {
        let selected = select(&Selector {
            solver: Some("bdf".into()),
            ..Selector::default()
        })
        .unwrap();
        assert!(!selected.is_empty());
        assert!(selected.iter().all(|scenario| scenario.solver == "bdf"));
    }

    #[test]
    fn rejects_empty_and_unknown_selections() {
        assert!(matches!(
            select(&Selector::default()),
            Err(RunnerError::Configuration(_))
        ));
        assert!(matches!(
            select(&Selector {
                solver: Some("not-a-solver".into()),
                ..Selector::default()
            }),
            Err(RunnerError::UnknownSelection(_))
        ));
    }

    #[test]
    fn scenario_matrix_declares_all_execution_axes_explicitly() {
        let mut ids = SCENARIOS
            .iter()
            .map(|scenario| scenario.id)
            .collect::<Vec<_>>();
        ids.sort_unstable();
        assert!(ids.windows(2).all(|pair| pair[0] != pair[1]));
        for scenario in SCENARIOS.iter() {
            assert!(!scenario.workload.is_empty(), "{} workload", scenario.id);
            assert!(!scenario.frontend.is_empty(), "{} frontend", scenario.id);
            assert!(!scenario.execution.is_empty(), "{} execution", scenario.id);
            assert!(
                !scenario.linear_algebra.is_empty(),
                "{} linear algebra",
                scenario.id
            );
            assert!(
                !scenario.parallel_policy.is_empty(),
                "{} parallel policy",
                scenario.id
            );
            assert!(!scenario.telemetry.is_empty(), "{} telemetry", scenario.id);
        }
    }

    #[test]
    fn radau_matrix_has_matched_supported_route_cells() {
        let radau = SCENARIOS
            .iter()
            .filter(|scenario| {
                scenario.solver == "radau" && scenario.groups.contains(&"radau-matrix")
            })
            .collect::<Vec<_>>();
        assert_eq!(radau.len(), 24);
        assert!(
            radau
                .iter()
                .all(|scenario| scenario.workload == "diffusion-chain, dimension=16, t=[0,0.01]")
        );
        for frontend in ["ExprLegacy", "AtomView"] {
            for execution in ["Lambdify", "AOT(warm-cache)"] {
                for layout in ["Dense", "Sparse", "Banded(1,1)"] {
                    for policy in ["Sequential", "Parallel"] {
                        assert_eq!(
                            radau
                                .iter()
                                .filter(|scenario| {
                                    scenario.frontend == frontend
                                        && scenario.execution == execution
                                        && scenario.linear_algebra == layout
                                        && scenario.parallel_policy == policy
                                })
                                .count(),
                            1,
                            "missing or duplicate Radau matrix cell: {frontend}/{execution}/{layout}/{policy}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn bdf_matrix_covers_only_its_supported_routes_and_has_no_layout_axis() {
        let bdf = SCENARIOS
            .iter()
            .filter(|scenario| scenario.solver == "bdf" && scenario.groups.contains(&"bdf-matrix"))
            .collect::<Vec<_>>();
        assert_eq!(bdf.len(), 8);
        assert!(bdf.iter().all(|scenario| {
            scenario.workload == "diffusion-chain, dimension=64, t=[0,0.01]"
                && scenario.linear_algebra == "not-applicable (fixed Dense implementation)"
        }));
        for frontend in ["ExprLegacy", "AtomView"] {
            for (execution, policy) in [
                ("Lambdify", "Sequential"),
                ("Lambdify", "Parallel(min_work=1)"),
                ("AOT(warm-cache)", "Whole(default chunks)"),
                ("AOT(warm-cache)", "Parallel2"),
            ] {
                assert_eq!(
                    bdf.iter()
                        .filter(|scenario| scenario.frontend == frontend
                            && scenario.execution == execution
                            && scenario.parallel_policy == policy)
                        .count(),
                    1,
                    "missing or duplicate BDF route: {frontend}/{execution}/{policy}"
                );
            }
        }
    }

    #[test]
    fn lsode2_matrix_covers_each_supported_frontend_layout_and_dispatch_route() {
        let lsode2 = SCENARIOS
            .iter()
            .filter(|scenario| {
                scenario.solver == "lsode2" && scenario.groups.contains(&"lsode2-matrix")
            })
            .collect::<Vec<_>>();
        assert_eq!(lsode2.len(), 30);
        assert!(lsode2.iter().all(|scenario| {
            scenario.workload
                == "diffusion-chain, dimension=32, t=[0,0.01], fixed BDF-only faithful route"
        }));

        for frontend in ["ExprLegacy", "AtomView"] {
            for layout in ["Dense", "Sparse", "Banded(1,1)"] {
                for (execution, policy) in [
                    ("Lambdify", "Sequential"),
                    ("Lambdify", "Parallel(min_work=1)"),
                    ("Lambdify", "Auto(min_work=1)"),
                    ("AOT(warm-cache)", "Whole(default chunks)"),
                    ("AOT(warm-cache)", "Parallel2"),
                ] {
                    assert_eq!(
                        lsode2
                            .iter()
                            .filter(|scenario| {
                                scenario.frontend == frontend
                                    && scenario.execution == execution
                                    && scenario.linear_algebra == layout
                                    && scenario.parallel_policy == policy
                            })
                            .count(),
                        1,
                        "missing or duplicate LSODE2 matrix cell: {frontend}/{execution}/{layout}/{policy}"
                    );
                }
            }
        }
    }

    #[test]
    fn bvp_damp_matrix_covers_supported_damped_and_frozen_routes() {
        let damp = SCENARIOS
            .iter()
            .filter(|scenario| {
                scenario.solver == "bvp_damp" && scenario.groups.contains(&"bvp-damp-matrix")
            })
            .collect::<Vec<_>>();
        assert_eq!(damp.len(), 66);
        for (family, workload) in [
            ("damped", "oscillator"),
            ("damped", "nonlinear"),
            ("frozen", "linear_exact"),
        ] {
            for frontend in ["exprlegacy", "atomview"] {
                let family_workload_frontend = damp
                    .iter()
                    .filter(|scenario| {
                        scenario
                            .groups
                            .iter()
                            .any(|group| group.eq_ignore_ascii_case(family))
                            && scenario.workload.contains(workload)
                            && scenario.frontend.eq_ignore_ascii_case(frontend)
                    })
                    .collect::<Vec<_>>();
                for layout in ["dense", "sparse_faer", "banded_faithful"] {
                    for (execution, policy) in [
                        ("Lambdify", "Sequential"),
                        ("Lambdify", "Parallel(min_work=0)"),
                        ("Lambdify", "Auto(min_work=0)"),
                    ] {
                        assert_eq!(
                            family_workload_frontend
                                .iter()
                                .filter(|scenario| {
                                    scenario.linear_algebra == layout
                                        && scenario.execution == execution
                                        && scenario.parallel_policy == policy
                                })
                                .count(),
                            1,
                            "missing/duplicate BVP_Damp cell: {family}/{workload}/{frontend}/{layout}/{execution}/{policy}"
                        );
                    }
                    if layout != "dense" {
                        assert_eq!(
                            family_workload_frontend
                                .iter()
                                .filter(|scenario| {
                                    scenario.linear_algebra == layout
                                        && scenario.execution == "AOT(warm-cache)"
                                        && scenario.parallel_policy == "Auto"
                                })
                                .count(),
                            1,
                            "missing/duplicate BVP_Damp AOT cell: {family}/{workload}/{frontend}/{layout}"
                        );
                    }
                }
                assert!(!family_workload_frontend.iter().any(|scenario| {
                    scenario.linear_algebra == "dense" && scenario.execution.starts_with("AOT")
                }));
            }
        }
    }

    #[test]
    fn bvp_sci_matrix_covers_symbolic_frontends_layouts_and_dispatch_routes() {
        let scenarios = SCENARIOS
            .iter()
            .filter(|scenario| {
                scenario.solver == "bvp_sci" && scenario.groups.contains(&"bvp-sci-matrix")
            })
            .collect::<Vec<_>>();
        assert_eq!(scenarios.len(), 30);
        for frontend in ["ExprLegacy", "AtomViewNative"] {
            for layout in ["Dense", "Sparse", "Banded"] {
                for (execution, dispatches) in [
                    (
                        "Lambdify",
                        &["Sequential", "Parallel(min_work=0)", "Auto(min_work=0)"][..],
                    ),
                    (
                        "AOT(warm-cache)",
                        &["Sequential", "Parallel(min_work=0)"][..],
                    ),
                ] {
                    for dispatch in dispatches {
                        assert_eq!(
                            scenarios
                                .iter()
                                .filter(|scenario| {
                                    scenario.frontend == frontend
                                        && scenario.linear_algebra == layout
                                        && scenario.execution == execution
                                        && scenario.parallel_policy == *dispatch
                                })
                                .count(),
                            1,
                            "missing or duplicate BVP_sci matrix cell: {frontend}/{layout}/{execution}/{dispatch}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn be_matrix_covers_only_supported_frontend_and_execution_routes() {
        let be = SCENARIOS
            .iter()
            .filter(|scenario| scenario.solver == "be" && scenario.groups.contains(&"be-matrix"))
            .collect::<Vec<_>>();
        assert_eq!(be.len(), 4);
        assert!(be.iter().all(|scenario| {
            scenario.workload
                == "parameterized combustion chain, dimension=3, 40 fixed steps, t=[0,0.04]"
                && scenario.linear_algebra == "Dense (fixed implementation)"
                && scenario.parallel_policy == "Sequential (fixed; no parallel dispatch API)"
        }));
        for frontend in ["ExprLegacy", "AtomView"] {
            for execution in ["Lambdify", "AOT(warm-cache)"] {
                assert_eq!(
                    be.iter()
                        .filter(|scenario| scenario.frontend == frontend
                            && scenario.execution == execution)
                        .count(),
                    1,
                    "missing or duplicate BE matrix cell: {frontend}/{execution}"
                );
            }
        }
    }

    #[test]
    fn nonlinear_matrix_covers_supported_frontends_routes_and_dispatch_only() {
        let nonlinear = SCENARIOS
            .iter()
            .filter(|scenario| {
                scenario.solver == "nonlinear-systems"
                    && scenario.groups.contains(&"nonlinear-matrix")
            })
            .collect::<Vec<_>>();
        assert_eq!(nonlinear.len(), 6);
        assert!(nonlinear.iter().all(|scenario| {
            scenario.workload == "quadratic-chain, dimension=64, known root, parameter p0=1"
                && scenario.linear_algebra == "Dense (fixed nalgebra implementation)"
        }));
        for frontend in ["ExprLegacy", "AtomView"] {
            for (execution, policies) in [
                ("Lambdify", &["Sequential", "Parallel(min_work=1)"][..]),
                (
                    "AOT(warm-cache)",
                    &["not-applicable (AOT has no callback dispatch policy)"][..],
                ),
            ] {
                for policy in policies {
                    assert_eq!(
                        nonlinear
                            .iter()
                            .filter(|scenario| {
                                scenario.frontend == frontend
                                    && scenario.execution == execution
                                    && scenario.parallel_policy == *policy
                            })
                            .count(),
                        1,
                        "missing or duplicate nonlinear matrix cell: {frontend}/{execution}/{policy}"
                    );
                }
            }
        }
    }

    #[test]
    fn optimization_matrix_covers_native_and_universal_fitting_routes() {
        let optimization = SCENARIOS
            .iter()
            .filter(|scenario| {
                scenario.solver == "optimisation" && scenario.groups.contains(&"optimization")
            })
            .collect::<Vec<_>>();
        assert_eq!(optimization.len(), 14);
        assert!(optimization.iter().all(|scenario| {
            scenario.execution == "Lambdify"
                && scenario.workload
                    == "exponential-sinusoidal fit, 201 samples, 4 fitted parameters"
                && scenario.linear_algebra == "Dense (fixed fitting implementation)"
                && scenario.gate_metrics == ["scenario_wall_ms"]
        }));

        let gavin = optimization
            .iter()
            .filter(|scenario| {
                scenario.groups.contains(&"Gavin/Marquardt")
                    || scenario.groups.contains(&"Gavin/Quadratic")
                    || scenario.groups.contains(&"Gavin/Nielsen")
            })
            .collect::<Vec<_>>();
        assert_eq!(gavin.len(), 6);
        for frontend in ["ExprLegacy", "AtomView"] {
            for policy in ["Gavin/Marquardt", "Gavin/Quadratic", "Gavin/Nielsen"] {
                assert_eq!(
                    gavin
                        .iter()
                        .filter(|scenario| scenario.frontend == frontend
                            && scenario.groups.contains(&policy)
                            && scenario.parallel_policy == "Sequential")
                        .count(),
                    1,
                    "missing or duplicate Gavin matrix cell: {frontend}/{policy}"
                );
            }
        }

        let fitting = optimization
            .iter()
            .filter(|scenario| scenario.groups.contains(&"FittingSolver"))
            .collect::<Vec<_>>();
        assert_eq!(fitting.len(), 4);
        for frontend in ["ExprLegacy", "AtomView"] {
            for policy in ["Sequential", "Parallel(min_work=1)"] {
                assert_eq!(
                    fitting
                        .iter()
                        .filter(|scenario| scenario.frontend == frontend
                            && scenario.parallel_policy == policy)
                        .count(),
                    1,
                    "missing or duplicate FittingSolver route: {frontend}/{policy}"
                );
            }
        }

        for method in [
            "UniversalFitting/LM",
            "UniversalFitting/Gavin",
            "UniversalFitting/VarPro",
        ] {
            assert_eq!(
                optimization
                    .iter()
                    .filter(|scenario| scenario.groups.contains(&method))
                    .count(),
                1,
                "missing or duplicate UniversalFitting branch {method}"
            );
        }
        assert_eq!(
            optimization
                .iter()
                .filter(|scenario| {
                    scenario
                        .groups
                        .iter()
                        .any(|group| group.ends_with("VarPro"))
                })
                .count(),
            2,
            "VarPro is represented by its standalone API and universal facade"
        );
    }
}

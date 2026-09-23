#![cfg(test)]

//! Architecture lane: AOT preparation, runtime diagnostics, and telemetry
//! semantics, kept separate from correctness and race-stress stories.
//!
//! Verbose runs are mirrored to `test_reports/BVP_Damp_AOT/*.md` after each
//! test. The console remains available with:
//! `cargo test --lib --no-default-features numerical::BVP_Damp::test_aot_diagnostics -- --ignored --nocapture --test-threads=1`

mod tests {
    use crate::numerical::BVP_Damp::BVP_traits::{BandedMatrixType, Vectors_type_casting};
    use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{
        DampedBvpStatistics, DampedSolverOptions, NRBVP, SolverParams,
    };
    use crate::numerical::BVP_Damp::generated_solver_handoff::{
        AotBuildPolicy, AotChunkingPolicy, AotExecutionPolicy, BuildDampedSolverRequest,
        DampedSolverBuildRequest,
    };
    use crate::numerical::BVP_Damp::test_common::{AotStoryProtocol, uniform_initial_guess};
    use crate::numerical::Examples_and_utils::NonlinEquation;
    use crate::somelinalg::banded::LinearSystemRef;
    use crate::somelinalg::banded::banded_assembly::BandedAssembly;
    use crate::somelinalg::banded::block_tridiagonal_lu_consistent::BlockTridiagonalLuConsistent;
    use crate::somelinalg::banded::block_tridiagonal_lu_consistent::IterativeRefinementReport;
    use crate::somelinalg::banded::lapack_style_banded::LapackStyleBandedLuFaithful;
    use crate::somelinalg::banded::linear_solver::build_solver_for_system;
    use crate::somelinalg::banded::node_major_layout::NodeMajorLayout;
    use crate::somelinalg::banded::solver_policy::{
        FallbackPolicy, LinearSolverConfig, LinearSolverPolicy,
    };
    use crate::somelinalg::banded::solver_traits::DirectLinearSolver;
    use crate::somelinalg::banded::storage::Banded;
    use crate::somelinalg::banded::superblock_layout::SuperBlockLayout;
    use crate::symbolic::View::bvp::{discretization_system_bvp_par_atom, eq_step_atom};
    use crate::symbolic::View::conversions::atom_to_expr;
    use crate::symbolic::bvp::aot_adapters::BvpAotAdapter;
    use crate::symbolic::bvp::aot_telemetry::{BvpAotTelemetryMode, BvpAotTelemetrySnapshot};
    use crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend;
    use crate::symbolic::codegen::codegen_aot_runtime_link::{
        register_generated_sparse_cdylib_backend, resolve_linked_sparse_backend,
        unregister_linked_sparse_backend,
    };
    use crate::symbolic::codegen::codegen_backend_selection::{
        BackendSelectionPolicy, SelectedBackendKind,
    };
    use crate::symbolic::codegen::codegen_orchestrator::{
        ParallelExecutorConfig, ParallelFallbackPolicy,
    };
    use crate::symbolic::codegen::codegen_provider_api::MatrixBackend;
    use crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy;
    use crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy;
    use crate::symbolic::codegen::rust_backend::codegen_aot_build::{
        AotBuildProfile, AotBuildRequest,
    };
    use crate::symbolic::symbolic_engine::Expr;
    use crate::symbolic::symbolic_functions_BVP::{
        BvpBackendIntegrationError, BvpSparseSolverBundle, BvpSymbolicAssemblyBackend, Jacobian,
    };
    use faer::linalg::solvers::Solve;
    use faer::sparse::SparseColMat;
    use nalgebra::{DMatrix, DVector};
    use std::collections::HashMap;
    use std::fs;
    use std::hint::black_box;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::path::PathBuf;
    use std::process::Command;
    use std::thread;
    use std::time::Duration;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    macro_rules! println {
        () => {
            crate::Utils::test_reporting::capture_test_line(format_args!(""));
        };
        ($($arg:tt)*) => {
            crate::Utils::test_reporting::capture_test_line(format_args!($($arg)*));
        };
    }

    macro_rules! aot_test_report {
        ($name:ident) => {
            // Isolated AOT children emit protocol markers to their parent.
            // They must not replace the parent's canonical report with one
            // containing only a single child observation.
            let _test_report = if std::env::var_os(ISOLATED_TUNING_CHILD_INDEX_ENV).is_some() {
                None
            } else {
                Some(crate::Utils::test_reporting::TestReportCapture::new(
                    "BVP_Damp_AOT",
                    concat!(module_path!(), "::", stringify!($name)),
                ))
            };
        };
    }

    const ISOLATED_TUNING_CHILD_INDEX_ENV: &str = "BVP_DAMP_ISOLATED_TUNING_CHILD_INDEX";
    const ISOLATED_TUNING_CHILD_STEPS_ENV: &str = "BVP_DAMP_ISOLATED_TUNING_CHILD_STEPS";
    const ISOLATED_TUNING_MATRIX_ENV: &str = "BVP_DAMP_ISOLATED_TUNING_MATRIX";
    const ISOLATED_TUNING_TIME_MARKER: &str = "[BVP_DAMP_ISOLATED_TUNING_TIME]";
    const ISOLATED_TUNING_SOLUTION_MARKER: &str = "[BVP_DAMP_ISOLATED_TUNING_SOLUTION]";
    const ISOLATED_TUNING_METRICS_MARKER: &str = "[BVP_DAMP_ISOLATED_TUNING_METRICS]";
    const ISOLATED_TUNING_PID_MARKER: &str = "[BVP_DAMP_ISOLATED_TUNING_PID]";

    include!("aot_diagnostics/core.rs");
    include!("aot_diagnostics/runtime.rs");
    include!("aot_diagnostics/tuning.rs");
    include!("aot_diagnostics/symbolic.rs");
    include!("aot_diagnostics/toolchains.rs");
    include!("aot_diagnostics/acceptance.rs");
    include!("aot_diagnostics/end_to_end.rs");
}

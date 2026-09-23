#![cfg(test)]

//! Architecture lane: isolated AOT lifecycle and cold/warm stress stories.
//! The legacy route remains an oracle while AtomView parity is added.
//!
//! Verbose runs are mirrored to `test_reports/BVP_Damp_AOT_Race/*.md` after
//! each test. Run the complete ignored lane with:
//! `cargo test --lib --no-default-features numerical::BVP_Damp::test_aot_race_stress -- --ignored --nocapture --test-threads=1`

mod tests {
    use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{
        DampedBvpStatistics, DampedSolverOptions, NRBVP, SolverParams,
    };
    use crate::numerical::BVP_Damp::generated_solver_handoff::{
        AotBuildPolicy, AotBuildProfile, AotChunkingPolicy, AotExecutionPolicy,
        GeneratedBackendConfig,
    };
    use crate::numerical::BVP_Damp::test_common::{AotStoryProtocol, uniform_initial_guess};
    use crate::symbolic::codegen::CodegenIR::AtomOptimizationProfile;
    use crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend;
    use crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
    use crate::symbolic::codegen::codegen_orchestrator::{
        ParallelExecutorConfig, ParallelFallbackPolicy,
    };
    use crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy;
    use crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy;
    use crate::symbolic::symbolic_engine::Expr;
    use crate::symbolic::symbolic_functions_BVP::BvpSymbolicAssemblyBackend;
    use nalgebra::{DMatrix, DVector};
    use std::collections::HashMap;
    use std::fs;
    use std::io::{self, Write};
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::path::PathBuf;
    use std::process::Command;
    use std::thread;
    use std::time::{Duration, Instant};

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
            let _test_report = crate::Utils::test_reporting::TestReportCapture::new(
                "BVP_Damp_AOT_Race",
                concat!(module_path!(), "::", stringify!($name)),
            );
        };
    }

    const RACE_REPETITIONS: usize = 5;
    const ISOLATED_STRESS_CHILD_INDEX_ENV: &str = "BVP_DAMP_ISOLATED_STRESS_CHILD_INDEX";
    const ISOLATED_STRESS_CHILD_REPETITION_ENV: &str = "BVP_DAMP_ISOLATED_STRESS_CHILD_REPETITION";
    const ISOLATED_RACE_ROW_MARKER: &str = "[BVP_DAMP_ISOLATED_RACE_ROW]";
    const ISOLATED_RACE_SOLUTION_MARKER: &str = "[BVP_DAMP_ISOLATED_RACE_SOLUTION]";
    const ISOLATED_RACE_PID_MARKER: &str = "[BVP_DAMP_ISOLATED_RACE_PID]";

    include!("aot_race_stress/core.rs");
    include!("aot_race_stress/protocol.rs");
    include!("aot_race_stress/variants.rs");
    include!("aot_race_stress/stories.rs");
}

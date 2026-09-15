//! Cross-toolchain nonlinear AOT acceptance story.
//!
//! This is intentionally an ignored test: it builds real generated shared
//! libraries and therefore depends on the machine's external toolchains. A
//! missing compiler is reported as `skipped`; once a toolchain is available,
//! materialization, build, dynamic loading, and solve failures are real test
//! failures with captured diagnostics.

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::process::Command;
    use std::time::Instant;

    use approx::assert_relative_eq;
    use nalgebra::DVector;

    use crate::numerical::Nonlinear_systems::engine::{
        DiagnosticsOptions, NewtonMethod, SolveOptions,
    };
    use crate::numerical::Nonlinear_systems::prelude::{
        NonlinearSolverMethod, SymbolicDenseAotOptions, SymbolicNonlinearProblem,
        SymbolicProblemOptions,
    };
    use crate::numerical::Nonlinear_systems::problem::NonlinearProblem;
    use crate::symbolic::codegen::c_backend::codegen_c_aot_registry::register_c_build_in_registry;
    use crate::symbolic::codegen::c_backend::codegen_c_aot_runtime_link::register_generated_c_dense_backend;
    use crate::symbolic::codegen::codegen_aot_driver::{
        AotBuildPreset, AotCodegenBackend, ExecutedGeneratedAotBuild, GeneratedAotBuildResult,
        generated_aot_artifact_from_prepared_problem, generated_aot_build_request_from_artifact,
    };
    use crate::symbolic::codegen::codegen_aot_registry::AotRegistry;
    use crate::symbolic::codegen::codegen_aot_resolution::AotResolver;
    use crate::symbolic::codegen::codegen_aot_runtime_link::{
        register_generated_dense_cdylib_backend, unregister_linked_dense_backend,
    };
    use crate::symbolic::codegen::zig_backend::codegen_zig_aot_registry::register_zig_build_in_registry;
    use crate::symbolic::codegen::zig_backend::codegen_zig_aot_runtime_link::register_generated_zig_dense_backend;

    fn equations() -> Vec<String> {
        vec!["x^2+y^2-10".to_string(), "x-y-4".to_string()]
    }

    fn options() -> SymbolicProblemOptions {
        SymbolicProblemOptions::new().with_variables(vec!["x".to_string(), "y".to_string()])
    }

    fn solve_options() -> SolveOptions {
        SolveOptions {
            tolerance: 1e-11,
            max_iterations: 32,
            diagnostics: DiagnosticsOptions {
                collect_history: false,
                collect_statistics: true,
                ..DiagnosticsOptions::default()
            },
            ..SolveOptions::default()
        }
    }

    fn tool_available(program: &str, probe: &str) -> bool {
        Command::new(program)
            .arg(probe)
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    fn c_toolchain_available() -> bool {
        if let Some(compiler) = std::env::var_os("RUSTEDSCITHE_C_COMPILER") {
            return tool_available(&compiler.to_string_lossy(), "-v");
        }
        [
            ("tcc", "-v"),
            ("gcc", "-v"),
            ("clang", "-v"),
            ("cl", "?"),
            ("cc", "-v"),
        ]
        .into_iter()
        .any(|(program, probe)| tool_available(program, probe))
    }

    fn zig_available() -> bool {
        tool_available("zig", "version")
    }

    fn executed_diagnostics(executed: &ExecutedGeneratedAotBuild) -> String {
        match executed {
            ExecutedGeneratedAotBuild::Rust(build) => format!(
                "status={:?}\nstdout:\n{}\nstderr:\n{}",
                build.status_code, build.stdout, build.stderr
            ),
            ExecutedGeneratedAotBuild::C(build) => format!(
                "status={:?}\nstdout:\n{}\nstderr:\n{}",
                build.status_code, build.stdout, build.stderr
            ),
            ExecutedGeneratedAotBuild::Zig(build) => format!(
                "status={:?}\nstdout:\n{}\nstderr:\n{}",
                build.status_code, build.stdout, build.stderr
            ),
        }
    }

    fn build_and_solve(
        backend: AotCodegenBackend,
        baseline: &DVector<f64>,
        output_parent: &Path,
    ) -> Result<(f64, f64, f64, f64, String), String> {
        let problem = SymbolicNonlinearProblem::from_strings_with_options(equations(), options())
            .map_err(|error| format!("problem preparation failed: {error:?}"))?;
        let prepared_bridge = problem.prepare_dense_aot_problem(SymbolicDenseAotOptions::default());
        let prepared = crate::symbolic::codegen::codegen_provider_api::PreparedProblem::dense(
            prepared_bridge.as_prepared_problem(),
        );
        let manifest = prepared_bridge.manifest();
        let artifact_name = format!("generated_nonlinear_{backend:?}_acceptance").to_lowercase();
        let module_name = format!("nonlinear_{backend:?}_acceptance_module").to_lowercase();

        let materialize_started = Instant::now();
        let request = generated_aot_build_request_from_artifact(
            generated_aot_artifact_from_prepared_problem(
                &artifact_name,
                &module_name,
                &prepared,
                backend,
            ),
            output_parent,
            AotBuildPreset::DevFastest,
        );
        let materialized = request
            .materialize()
            .map_err(|error| format!("materialize failed: {error}"))?;
        let materialize_ms = materialize_started.elapsed().as_secs_f64() * 1e3;
        let command = materialized.command_line();

        let build_started = Instant::now();
        let executed = materialized
            .execute()
            .map_err(|error| format!("build spawn failed for `{command}`: {error}"))?;
        let build_ms = build_started.elapsed().as_secs_f64() * 1e3;
        if !executed.succeeded() {
            return Err(format!(
                "build failed for `{command}` in `{}`\n{}",
                materialized.workdir().display(),
                executed_diagnostics(&executed)
            ));
        }

        let mut registry = AotRegistry::new();
        let link_started = Instant::now();
        let problem_key = manifest.problem_key();
        match (materialized, backend) {
            (GeneratedAotBuildResult::Rust(build), AotCodegenBackend::Rust) => {
                let registered = registry
                    .register_materialized_build(manifest.clone(), &build)
                    .clone();
                assert!(registered.manifest_key_matches());
                assert!(registered.compiled_artifact_exists());
                register_generated_dense_cdylib_backend(&registered)
                    .map_err(|error| format!("Rust runtime link failed: {error}"))?;
            }
            (GeneratedAotBuildResult::C(build), AotCodegenBackend::C) => {
                let registered =
                    register_c_build_in_registry(&mut registry, manifest.clone(), &build).clone();
                assert!(registered.manifest_key_matches());
                assert!(registered.compiled_artifact_exists());
                register_generated_c_dense_backend(&registered)
                    .map_err(|error| format!("C runtime link failed: {error}"))?;
            }
            (GeneratedAotBuildResult::Zig(build), AotCodegenBackend::Zig) => {
                let registered =
                    register_zig_build_in_registry(&mut registry, manifest.clone(), &build).clone();
                assert!(registered.manifest_key_matches());
                assert!(registered.compiled_artifact_exists());
                register_generated_zig_dense_backend(&registered)
                    .map_err(|error| format!("Zig runtime link failed: {error}"))?;
            }
            (actual, expected) => {
                return Err(format!(
                    "materialized backend mismatch: expected {expected:?}, got {actual:?}"
                ));
            }
        }
        let link_ms = link_started.elapsed().as_secs_f64() * 1e3;

        let resolver = AotResolver::new(registry);
        let solve_started = Instant::now();
        let compiled = SymbolicNonlinearProblem::from_strings_with_backend_selection(
            equations(),
            options(),
            crate::numerical::Nonlinear_systems::symbolic_backend::
                SymbolicBackendSelectionPolicy::AotOnly,
            Some(&resolver),
            SymbolicDenseAotOptions::default(),
        )
        .map_err(|error| format!("AOT solver selection failed: {error:?}"))?;
        let result = NonlinearSolverMethod::Newton(NewtonMethod)
            .solve(
                &compiled,
                DVector::from_vec(vec![1.0, 1.0]),
                solve_options(),
            )
            .map_err(|error| format!("AOT solve failed: {error:?}"))?;
        let solve_ms = solve_started.elapsed().as_secs_f64() * 1e3;
        let max_error = result
            .x
            .iter()
            .zip(baseline.iter())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert_relative_eq!(max_error, 0.0, epsilon = 1e-9);
        let residual = compiled
            .residual(&result.x)
            .map_err(|error| format!("post-solve residual failed: {error:?}"))?;
        let residual_norm = residual.iter().map(|value| value.abs()).fold(0.0, f64::max);
        assert!(
            residual_norm < 1e-9,
            "nonlinear residual is {residual_norm}"
        );
        unregister_linked_dense_backend(&problem_key);
        Ok((materialize_ms, build_ms, link_ms, solve_ms, command))
    }

    #[test]
    #[ignore = "builds and links generated Rust/C/Zig nonlinear AOT libraries when toolchains are installed"]
    fn cross_toolchain_nonlinear_aot_lifecycle_correctness_and_diagnostics() {
        let baseline_problem =
            SymbolicNonlinearProblem::from_strings_with_options(equations(), options())
                .expect("Lambdify baseline should prepare");
        let baseline = NonlinearSolverMethod::Newton(NewtonMethod)
            .solve(
                &baseline_problem,
                DVector::from_vec(vec![1.0, 1.0]),
                solve_options(),
            )
            .expect("Lambdify baseline should solve")
            .x;
        let root = tempfile::tempdir().expect("temporary AOT root should exist");

        println!(
            "[Nonlinear external AOT] baseline={baseline:?}; C_available={}; Zig_available={}",
            c_toolchain_available(),
            zig_available()
        );
        println!(
            "backend | status | materialize_ms | build_ms | link_ms | solve_ms | max_error | command"
        );

        let rust = build_and_solve(AotCodegenBackend::Rust, &baseline, root.path())
            .expect("Rust AOT acceptance should succeed");
        println!(
            "Rust | ok | {:.3} | {:.3} | {:.3} | {:.3} | <1e-9 | {}",
            rust.0, rust.1, rust.2, rust.3, rust.4
        );

        for (backend, available) in [
            (AotCodegenBackend::C, c_toolchain_available()),
            (AotCodegenBackend::Zig, zig_available()),
        ] {
            if !available {
                println!(
                    "{backend:?} | skipped | - | - | - | - | - | required toolchain unavailable"
                );
                continue;
            }
            let output = tempfile::tempdir().expect("isolated external AOT root should exist");
            let metrics = build_and_solve(backend, &baseline, output.path())
                .unwrap_or_else(|error| panic!("{backend:?} AOT acceptance failed:\n{error}"));
            println!(
                "{backend:?} | ok | {:.3} | {:.3} | {:.3} | {:.3} | <1e-9 | {}",
                metrics.0, metrics.1, metrics.2, metrics.3, metrics.4
            );
        }
    }
}

//! Experimental isolated same-matrix sparse-LU microbenchmark for BVP-shaped CSC systems.

#[path = "../src/somelinalg/bvp_sparse_lu.rs"]
mod bvp_sparse_lu;
#[path = "../src/somelinalg/sparse_lu_reference.rs"]
mod fixture;

use bvp_sparse_lu::ReusableSparseLu;
use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::solvers::Solve;
use faer::sparse::linalg::SupernodalThreshold;
use faer::sparse::linalg::lu::{LuRef, LuSymbolicParams, NumericLu, factorize_symbolic_lu};
use faer::{Conj, MatMut, get_global_parallelism};
use std::hint::black_box as std_black_box;
use std::path::Path;
use std::time::{Duration, Instant};
use tabled::{Table, Tabled};

use RustedSciThe::Utils::test_reporting::write_test_report;

#[derive(Tabled)]
struct BenchmarkRow {
    case: String,
    dimension: usize,
    nnz: usize,
    assembly_median_us: String,
    faer_cold_factor_us: String,
    reusable_symbolic_us: String,
    reusable_cold_total_us: String,
    reusable_warm_numeric_us: String,
    faer_solve_us: String,
    reusable_solve_us: String,
    relative_residual: String,
    known_solution_max_diff: String,
}

#[derive(Tabled)]
struct StrategyRow {
    case: String,
    dimension: usize,
    stored_nnz: usize,
    numeric_nonzeros: usize,
    faer_mode: String,
    symbolic_median_us: String,
    cold_factor_median_us: String,
    warm_numeric_median_us: String,
    solve_median_us: String,
    relative_backward_error: String,
    known_solution_max_diff: String,
    status: String,
}

#[cfg(feature = "superlu")]
#[derive(Tabled)]
struct SuperLuControlRow {
    case: String,
    dimension: usize,
    stored_nnz: usize,
    csc_to_i32_median_us: String,
    ffi_factor_solve_median_us: String,
    native_factor_median_us: String,
    native_solve_median_us: String,
    factor_l_storage_nnz: usize,
    factor_u_storage_nnz: usize,
    factor_l_csc_nnz: usize,
    factor_u_csc_nnz: usize,
    backward_error: String,
    known_solution_max_diff: String,
    status: String,
}

#[derive(Clone, Copy)]
struct FaerMode {
    name: &'static str,
    threshold: SupernodalThreshold,
}

fn diagnostic_symbolic(
    matrix: faer::sparse::SparseColMatRef<'_, usize, f64>,
    mode: FaerMode,
) -> faer::sparse::linalg::lu::SymbolicLu<usize> {
    factorize_symbolic_lu(
        matrix.symbolic(),
        LuSymbolicParams {
            colamd_params: Default::default(),
            supernodal_flop_ratio_threshold: mode.threshold,
            supernodal_params: Default::default(),
        },
    )
    .expect("requested faer symbolic strategy should prepare")
}

fn diagnostic_numeric_factor(
    symbolic: &faer::sparse::linalg::lu::SymbolicLu<usize>,
    numeric: &mut NumericLu<usize, f64>,
    matrix: faer::sparse::SparseColMatRef<'_, usize, f64>,
    scratch: &mut MemBuffer,
) {
    symbolic
        .factorize_numeric_lu(
            numeric,
            matrix,
            get_global_parallelism(),
            &mut MemStack::new(scratch),
            Default::default(),
        )
        .expect("faer numeric factorization should succeed");
}

fn diagnostic_solve(
    symbolic: &faer::sparse::linalg::lu::SymbolicLu<usize>,
    numeric: &NumericLu<usize, f64>,
    rhs: &mut [f64],
    scratch: &mut MemBuffer,
) {
    let factors = LuRef::new_unchecked(symbolic, numeric);
    factors.solve_in_place_with_conj(
        Conj::No,
        MatMut::from_column_major_slice_mut(rhs, rhs.len(), 1),
        get_global_parallelism(),
        &mut MemStack::new(scratch),
    );
}

fn relative_backward_error(
    triplets: &[faer::sparse::Triplet<usize, usize, f64>],
    solution: &[f64],
    rhs: &[f64],
) -> f64 {
    let mut product = vec![0.0; rhs.len()];
    let mut row_sums = vec![0.0; rhs.len()];
    for entry in triplets {
        product[entry.row] += entry.val * solution[entry.col];
        row_sums[entry.row] += entry.val.abs();
    }
    let residual_inf = product
        .iter()
        .zip(rhs)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    let matrix_inf = row_sums.into_iter().fold(0.0, f64::max);
    let solution_inf = solution.iter().map(|value| value.abs()).fold(0.0, f64::max);
    let rhs_inf = rhs.iter().map(|value| value.abs()).fold(0.0, f64::max);
    residual_inf / (matrix_inf * solution_inf + rhs_inf).max(f64::MIN_POSITIVE)
}

fn benchmark_faer_strategy(
    system: &fixture::BvpSparseSystem,
    case: &str,
    mode: FaerMode,
    repeats: usize,
) -> StrategyRow {
    let matrix = system.matrix.as_ref();
    let dimension = system.dimension();
    let symbolic_samples = timed_samples(
        || {
            std_black_box(diagnostic_symbolic(matrix, mode));
        },
        repeats,
    );
    let symbolic = diagnostic_symbolic(matrix, mode);
    let parallelism = get_global_parallelism();
    let factor_req = symbolic.factorize_numeric_lu_scratch::<f64>(parallelism, Default::default());
    let solve_req = symbolic.solve_in_place_scratch::<f64>(1, parallelism);

    let cold_samples = timed_samples(
        || {
            let symbolic = diagnostic_symbolic(matrix, mode);
            let factor_req =
                symbolic.factorize_numeric_lu_scratch::<f64>(parallelism, Default::default());
            let mut factor_scratch = MemBuffer::try_new(factor_req)
                .expect("allocate cold numeric factorization workspace");
            let mut numeric = NumericLu::new();
            diagnostic_numeric_factor(&symbolic, &mut numeric, matrix, &mut factor_scratch);
            std_black_box(numeric);
        },
        repeats,
    );

    let mut factor_scratch =
        MemBuffer::try_new(factor_req).expect("allocate reusable factorization workspace");
    let mut solve_scratch = MemBuffer::try_new(solve_req).expect("allocate solve workspace");
    let mut numeric = NumericLu::new();
    diagnostic_numeric_factor(&symbolic, &mut numeric, matrix, &mut factor_scratch);
    let warm_samples = timed_samples(
        || diagnostic_numeric_factor(&symbolic, &mut numeric, matrix, &mut factor_scratch),
        repeats,
    );

    let solve_samples = timed_prepared_samples(
        || system.rhs.clone(),
        |mut solution| {
            diagnostic_solve(&symbolic, &numeric, &mut solution, &mut solve_scratch);
            std_black_box(solution);
        },
        repeats,
    );
    let mut solution = system.rhs.clone();
    diagnostic_solve(&symbolic, &numeric, &mut solution, &mut solve_scratch);
    let backward_error = relative_backward_error(&system.triplets, &solution, &system.rhs);
    let solution_diff = if system.exact_solution.is_empty() {
        None
    } else {
        Some(fixture::max_abs_difference(
            &solution,
            &system.exact_solution,
        ))
    };
    let numerical_nonzeros = matrix.val().iter().filter(|value| **value != 0.0).count();
    let status = if backward_error <= 1e-12 && solution_diff.is_none_or(|diff| diff <= 1e-9) {
        "ok"
    } else {
        "bad-numerics"
    };

    StrategyRow {
        case: case.into(),
        dimension,
        stored_nnz: system.matrix.symbolic().row_idx().len(),
        numeric_nonzeros: numerical_nonzeros,
        faer_mode: mode.name.into(),
        symbolic_median_us: format!("{:.3}", median_ms(symbolic_samples) * 1e3),
        cold_factor_median_us: format!("{:.3}", median_ms(cold_samples) * 1e3),
        warm_numeric_median_us: format!("{:.3}", median_ms(warm_samples) * 1e3),
        solve_median_us: format!("{:.3}", median_ms(solve_samples) * 1e3),
        relative_backward_error: format!("{backward_error:.3e}"),
        known_solution_max_diff: solution_diff
            .map(|diff| format!("{diff:.3e}"))
            .unwrap_or_else(|| "n/a (captured RHS)".into()),
        status: status.into(),
    }
}

#[cfg(feature = "superlu")]
fn benchmark_superlu_control(
    control: &RustedSciThe::somelinalg::superlu::SolverLibrary,
    system: &fixture::BvpSparseSystem,
    case: &str,
    repeats: usize,
) -> SuperLuControlRow {
    use RustedSciThe::somelinalg::superlu::CscMatrix;

    let symbolic = system.matrix.symbolic();
    let conversion_samples = timed_samples(
        || {
            let converted = CscMatrix::try_from_usize(
                system.dimension(),
                symbolic.col_ptr(),
                symbolic.row_idx(),
                system.matrix.val(),
            )
            .expect("convert same CSC input to SuperLU's int32 ABI");
            std_black_box(converted);
        },
        repeats,
    );
    let matrix = CscMatrix::try_from_usize(
        system.dimension(),
        symbolic.col_ptr(),
        symbolic.row_idx(),
        system.matrix.val(),
    )
    .expect("convert same CSC input to SuperLU's int32 ABI");

    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats.min(3) {
        std_black_box(
            control
                .solve(&matrix, &system.rhs)
                .expect("SuperLU warmup solve"),
        );
    }
    for _ in 0..repeats {
        let started = Instant::now();
        let solved = control
            .solve(&matrix, &system.rhs)
            .expect("SuperLU factor and solve");
        samples.push((started.elapsed(), solved));
    }
    let total_samples = samples.iter().map(|(elapsed, _)| *elapsed).collect();
    let total_median_us = median_ms(total_samples) * 1e3;
    let mut factor_samples = samples
        .iter()
        .map(|(_, solved)| solved.factor_elapsed)
        .collect::<Vec<_>>();
    factor_samples.sort_unstable();
    let factor_median_us = factor_samples[factor_samples.len() / 2].as_secs_f64() * 1e3;
    let mut solve_samples = samples
        .iter()
        .map(|(_, solved)| solved.solve_elapsed)
        .collect::<Vec<_>>();
    solve_samples.sort_unstable();
    let solve_median_us = solve_samples[solve_samples.len() / 2].as_secs_f64() * 1e3;

    let result = control
        .solve(&matrix, &system.rhs)
        .expect("SuperLU correctness solve");
    let backward_error = relative_backward_error(&system.triplets, &result.solution, &system.rhs);
    let solution_diff = if system.exact_solution.is_empty() {
        None
    } else {
        Some(fixture::max_abs_difference(
            &result.solution,
            &system.exact_solution,
        ))
    };
    let status = if backward_error <= 1e-12 && solution_diff.is_none_or(|diff| diff <= 1e-8) {
        "ok"
    } else {
        "bad-numerics"
    };

    SuperLuControlRow {
        case: case.to_owned(),
        dimension: system.dimension(),
        stored_nnz: matrix.nnz(),
        csc_to_i32_median_us: format!("{:.3}", median_ms(conversion_samples) * 1e3),
        ffi_factor_solve_median_us: format!("{total_median_us:.3}"),
        native_factor_median_us: format!("{factor_median_us:.3}"),
        native_solve_median_us: format!("{solve_median_us:.3}"),
        factor_l_storage_nnz: result.factor_l_storage_nnz,
        factor_u_storage_nnz: result.factor_u_storage_nnz,
        factor_l_csc_nnz: result.factor_l_csc_nnz,
        factor_u_csc_nnz: result.factor_u_csc_nnz,
        backward_error: format!("{backward_error:.3e}"),
        known_solution_max_diff: solution_diff
            .map(|diff| format!("{diff:.3e}"))
            .unwrap_or_else(|| "n/a (captured RHS)".into()),
        status: status.into(),
    }
}

fn zero_pruned_numeric_case(source: &fixture::BvpSparseSystem) -> fixture::BvpSparseSystem {
    let mut triplets = source.triplets.clone();
    for (index, entry) in triplets.iter_mut().enumerate() {
        if entry.row != entry.col && index % 100 != 0 {
            entry.val = 0.0;
        }
    }
    let mut rhs = vec![0.0; source.dimension()];
    for entry in &triplets {
        rhs[entry.row] += entry.val * source.exact_solution[entry.col];
    }
    let matrix = faer::sparse::SparseColMat::try_new_from_triplets(
        source.dimension(),
        source.dimension(),
        &triplets,
    )
    .expect("zero-valued entries should preserve the CSC pattern");
    assert_eq!(
        matrix.symbolic().row_idx().len(),
        source.matrix.symbolic().row_idx().len(),
        "zero control must retain every structural coordinate"
    );
    fixture::BvpSparseSystem {
        mesh_nodes: source.mesh_nodes,
        state_dimension: source.state_dimension,
        border_dimension: source.border_dimension,
        matrix,
        triplets,
        rhs,
        exact_solution: source.exact_solution.clone(),
    }
}

fn strategy_diagnostic(repeats: usize) -> Vec<StrategyRow> {
    let modes = [
        FaerMode {
            name: "auto",
            threshold: SupernodalThreshold::AUTO,
        },
        FaerMode {
            name: "simplicial",
            threshold: SupernodalThreshold::FORCE_SIMPLICIAL,
        },
        FaerMode {
            name: "supernodal",
            threshold: SupernodalThreshold::FORCE_SUPERNODAL,
        },
    ];
    let mut rows = Vec::new();
    for &(nodes, state, border) in &[(24, 2, 2), (128, 4, 2), (384, 4, 2)] {
        let source = fixture::bvp_sparse_system(nodes, state, border);
        let zeroed = zero_pruned_numeric_case(&source);
        for (case, system) in [("fixture", &source), ("same-pattern-93pct-zero", &zeroed)] {
            for mode in modes {
                rows.push(benchmark_faer_strategy(system, case, mode, repeats));
            }
        }
    }
    for path in captured_fixture_paths() {
        let system = fixture::load_fixture(&path).expect("load captured BVP_sci CSC fixture");
        let case = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("captured-bvp-jacobian");
        for mode in modes {
            rows.push(benchmark_faer_strategy(&system, case, mode, repeats));
        }
    }
    rows
}

fn captured_fixture_paths() -> Vec<std::path::PathBuf> {
    let value = std::env::var("SOMELINALG_SPARSE_LU_CAPTURE_FIXTURES")
        .or_else(|_| std::env::var("SOMELINALG_SPARSE_LU_CAPTURE_FIXTURE"));
    value
        .ok()
        .into_iter()
        .flat_map(|paths| {
            paths
                .split(';')
                .map(str::trim)
                .filter(|path| !path.is_empty())
                .map(std::path::PathBuf::from)
                .collect::<Vec<_>>()
        })
        .collect()
}

fn median_ms(mut samples: Vec<Duration>) -> f64 {
    samples.sort_unstable();
    samples[samples.len() / 2].as_secs_f64() * 1e3
}

fn timed_samples(mut operation: impl FnMut(), repeats: usize) -> Vec<Duration> {
    for _ in 0..repeats.min(3) {
        operation();
    }
    (0..repeats)
        .map(|_| {
            let started = Instant::now();
            operation();
            started.elapsed()
        })
        .collect()
}

fn timed_prepared_samples<T>(
    mut prepare: impl FnMut() -> T,
    mut operation: impl FnMut(T),
    repeats: usize,
) -> Vec<Duration> {
    for _ in 0..repeats.min(3) {
        operation(prepare());
    }
    (0..repeats)
        .map(|_| {
            let input = prepare();
            let started = Instant::now();
            operation(input);
            started.elapsed()
        })
        .collect()
}

fn benchmark_case(
    mesh_nodes: usize,
    state_dim: usize,
    border_dim: usize,
    repeats: usize,
) -> BenchmarkRow {
    let system = fixture::bvp_sparse_system(mesh_nodes, state_dim, border_dim);
    let dimension = system.dimension();
    let nnz = system.matrix.symbolic().row_idx().len();

    let assembly_us = median_ms(timed_samples(
        || {
            let rebuilt = faer::sparse::SparseColMat::<usize, f64>::try_new_from_triplets(
                dimension,
                dimension,
                &system.triplets,
            )
            .expect("fixture assembly should succeed");
            std_black_box(rebuilt);
        },
        repeats,
    )) * 1e3;

    let faer_factor_us = median_ms(timed_samples(
        || {
            let factor = system
                .matrix
                .sp_lu()
                .expect("faer sparse wrapper factorization should succeed");
            std_black_box(factor);
        },
        repeats,
    )) * 1e3;

    let symbolic_us = median_ms(timed_samples(
        || {
            let mut kernel = ReusableSparseLu::new(dimension);
            assert!(kernel.prepare_symbolic(system.matrix.as_ref()).unwrap());
            std_black_box(kernel);
        },
        repeats,
    )) * 1e3;

    let mut kernel = ReusableSparseLu::new(dimension);
    kernel.prepare_symbolic(system.matrix.as_ref()).unwrap();
    let cold_total_us = median_ms(timed_samples(
        || {
            let mut cold = ReusableSparseLu::new(dimension);
            cold.prepare_symbolic(system.matrix.as_ref()).unwrap();
            cold.factor_numeric(system.matrix.as_ref()).unwrap();
            std_black_box(cold);
        },
        repeats,
    )) * 1e3;
    let warm_numeric_us = median_ms(timed_samples(
        || kernel.factor_numeric(system.matrix.as_ref()).unwrap(),
        repeats,
    )) * 1e3;

    let factor = system.matrix.sp_lu().unwrap();
    let faer_solve_us = median_ms(timed_prepared_samples(
        || faer::Col::from_fn(system.dimension(), |i| system.rhs[i]),
        |rhs| {
            let solution = factor.solve(&rhs);
            std_black_box(solution);
        },
        repeats,
    )) * 1e3;

    let mut reusable_solve = ReusableSparseLu::new(dimension);
    reusable_solve
        .prepare_symbolic(system.matrix.as_ref())
        .unwrap();
    reusable_solve
        .factor_numeric(system.matrix.as_ref())
        .unwrap();
    let solve_us = median_ms(timed_prepared_samples(
        || system.rhs.clone(),
        |mut rhs| {
            reusable_solve.solve_in_place(&mut rhs).unwrap();
            std_black_box(rhs);
        },
        repeats,
    )) * 1e3;

    let mut solution = system.rhs.clone();
    kernel.factor_numeric(system.matrix.as_ref()).unwrap();
    kernel.solve_in_place(&mut solution).unwrap();
    let relative_residual =
        fixture::relative_residual_inf(&system.triplets, &solution, &system.rhs);
    let known_solution_max_diff = fixture::max_abs_difference(&solution, &system.exact_solution);

    BenchmarkRow {
        case: format!("mesh={mesh_nodes},state={state_dim},border={border_dim}"),
        dimension,
        nnz,
        assembly_median_us: format!("{assembly_us:.3}"),
        faer_cold_factor_us: format!("{faer_factor_us:.3}"),
        reusable_symbolic_us: format!("{symbolic_us:.3}"),
        reusable_cold_total_us: format!("{cold_total_us:.3}"),
        reusable_warm_numeric_us: format!("{warm_numeric_us:.3}"),
        faer_solve_us: format!("{faer_solve_us:.3}"),
        reusable_solve_us: format!("{solve_us:.3}"),
        relative_residual: format!("{relative_residual:.3e}"),
        known_solution_max_diff: format!("{known_solution_max_diff:.3e}"),
    }
}

fn main() {
    let repeats = std::env::var("SOMELINALG_SPARSE_LU_REPEATS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(9)
        .max(3);
    let cases = [(24, 2, 2), (128, 4, 2), (384, 4, 2)];
    let rows = cases
        .iter()
        .map(|&(nodes, state, border)| benchmark_case(nodes, state, border, repeats))
        .collect::<Vec<_>>();
    let strategy_rows = strategy_diagnostic(repeats);
    #[cfg(feature = "superlu")]
    let superlu_report = if let Some(library_path) = std::env::var_os("SUPERLU_LIBRARY")
        .or_else(|| std::env::var_os("SUPERLU_EXPERIMENTAL_LIBRARY"))
    {
        use RustedSciThe::somelinalg::superlu::SolverLibrary;

        let control = SolverLibrary::load(Path::new(&library_path))
            .expect("load SuperLU 6.0.1 diagnostic bridge");
        let version = control.version();
        let mut superlu_rows = cases
            .iter()
            .map(|&(nodes, state, border)| {
                let system = fixture::bvp_sparse_system(nodes, state, border);
                benchmark_superlu_control(&control, &system, "synthetic", repeats)
            })
            .collect::<Vec<_>>();
        for path in captured_fixture_paths() {
            let system = fixture::load_fixture(&path)
                .expect("load captured BVP_sci CSC fixture for SuperLU control");
            superlu_rows.push(benchmark_superlu_control(
                &control,
                &system,
                path.file_stem()
                    .and_then(|stem| stem.to_str())
                    .unwrap_or("captured-bvp-jacobian"),
                repeats,
            ));
        }
        format!(
            "## SuperLU FFI control\n\n- Native core: SuperLU {}.{}.{} sequential `gstrf`/`gstrs` sequence, COLAMD, pivot threshold 1.0; the call sequence mirrors SciPy's one-shot `gstrf` then solve.\n- Rust FFI wall time includes result allocation and bridge setup; int32 CSC conversion is reported separately. Native factor timing includes ordering, etree, and numeric LU; native solve timing covers triangular solve.\n- `factor_*_storage_nnz` reports SuperLU's native factor storage count. `factor_*_csc_nnz` reproduces SciPy's `LU_to_csc` semantics: removes exact zeros, moves the upper triangle from the L supernodes into U, and inserts L's unit diagonal. These CSC counts are directly comparable to SciPy `factor.L.nnz` / `factor.U.nnz`.\n- SuperLU control is sequential and built with bundled CBLAS. Set `SOMELINALG_SCIPY_BLAS_THREADS=1` to constrain SciPy OpenBLAS for a matched thread budget; BLAS implementations still differ. Python rows report BLAS vendor/version and thread policy.\n- Set `SOMELINALG_SPARSE_LU_CAPTURE_FIXTURES` to semicolon-separated fixture paths to replay several BVP Jacobians; the singular `..._FIXTURE` variable remains supported.\n\n{}\n",
            version.0,
            version.1,
            version.2,
            Table::new(&superlu_rows),
        )
    } else {
        "## SuperLU FFI control\n\nNot run: build with `scripts/build_superlu_bridge.ps1` or `.sh`, then set `SUPERLU_LIBRARY` and enable `superlu`.\n".to_owned()
    };
    #[cfg(not(feature = "superlu"))]
    let superlu_report = "## SuperLU FFI control\n\nNot compiled; enable Cargo feature `superlu` to use the opt-in bridge.\n".to_owned();
    let report = format!(
        "# Sparse LU microbenchmark\n\n- SciPy reference: 1.18.1 `splu` / SuperLU\n- Repeats: {repeats}\n- Matrix: same BVP-shaped CSC fixture and RHS for every Rust route\n- Scope: assembly and solve are separate; faer `sp_lu` is cold symbolic+numeric factorization\n- Warm numeric refactor has no direct `splu` equivalent\n\n## Reusable-kernel baseline\n\n{}\n\n## Faer strategy and explicit-zero diagnostic\n\nThe strategy table uses the same COLAMD ordering parameters while forcing AUTO, simplicial, or supernodal numeric LU. `same-pattern-zeroed` retains the exact CSC coordinates but sets most off-diagonal values to zero; the resulting fixtures have 92.8% structural zero values at n=514/1538 and form a sensitivity control, not captured BVP Jacobians. Symbolic and cold columns include the named stages; warm numeric reuses symbolic, numeric, and scratch storage. Solve timing excludes RHS allocation. Backward error is `||Ax-b||inf / (||A||inf ||x||inf + ||b||inf)`. Captured files listed in `SOMELINALG_SPARSE_LU_CAPTURE_FIXTURES` are replayed through faer and SuperLU.\n\n{}\n\n{}",
        Table::new(&rows),
        Table::new(&strategy_rows),
        superlu_report,
    );
    println!("{report}");
    let report_path = write_test_report(
        "somelinalg_sparse_lu_reference",
        "rust_microbenchmark",
        &report,
    )
    .expect("write compact Rust benchmark report");
    println!("report={}", report_path.display());

    if let Some(directory) = std::env::var_os("SOMELINALG_SPARSE_LU_EXPORT_DIR") {
        let directory = Path::new(&directory);
        std::fs::create_dir_all(directory).expect("create fixture export directory");
        for &(nodes, state, border) in &cases {
            let system = fixture::bvp_sparse_system(nodes, state, border);
            let path = directory.join(format!("mesh-{nodes}-state-{state}-border-{border}.txt"));
            fixture::write_fixture(&path, &system).expect("export shared Python fixture");
            println!("fixture={}", path.display());
            let zeroed = zero_pruned_numeric_case(&system);
            let path = directory.join(format!(
                "mesh-{nodes}-state-{state}-border-{border}-same-pattern-zeroed.txt"
            ));
            fixture::write_fixture(&path, &zeroed)
                .expect("export zero-preserving shared Python fixture");
            println!("fixture={}", path.display());
        }
    }
}

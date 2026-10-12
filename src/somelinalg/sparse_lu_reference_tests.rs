use super::bvp_sparse_lu::ReusableSparseLu;
use super::sparse_lu_reference::{
    bvp_sparse_system, max_abs_difference, relative_residual_inf, write_fixture,
};
use faer::linalg::solvers::Solve;
use std::process::Command;
use tabled::Tabled;

#[derive(Tabled)]
struct PythonParityRow {
    scipy: String,
    dimension: usize,
    mesh_nodes: usize,
    csc_nnz: usize,
    factor_l_nnz: usize,
    factor_u_nnz: usize,
    residual_inf: String,
    known_solution_max_diff: String,
    result: &'static str,
}

#[cfg(feature = "superlu")]
#[derive(Tabled)]
struct NativePythonParityRow {
    case: String,
    scipy_blas: String,
    rust_blas: String,
    blas_threads: String,
    dimension: usize,
    matrix_nnz: usize,
    scipy_l_nnz: usize,
    rust_l_nnz: usize,
    scipy_u_nnz: usize,
    rust_u_nnz: usize,
    solution_max_diff: String,
    relative_residual: String,
    status: &'static str,
}

#[test]
fn bvp_shaped_sparse_lu_matches_faer_wrapper_and_known_solution() {
    let system = bvp_sparse_system(12, 3, 2);
    let mut reusable = ReusableSparseLu::new(system.dimension());
    assert!(reusable.prepare_symbolic(system.matrix.as_ref()).unwrap());
    reusable.factor_numeric(system.matrix.as_ref()).unwrap();
    let mut reusable_solution = system.rhs.clone();
    reusable.solve_in_place(&mut reusable_solution).unwrap();

    let factor = system
        .matrix
        .sp_lu()
        .expect("faer wrapper factorization should succeed");
    let wrapper_rhs = faer::Col::from_fn(system.dimension(), |i| system.rhs[i]);
    let wrapper_solution = factor.solve(&wrapper_rhs);
    let wrapper_solution = (0..system.dimension())
        .map(|index| wrapper_solution[index])
        .collect::<Vec<_>>();

    assert!(max_abs_difference(&reusable_solution, &system.exact_solution) < 1e-11);
    assert!(max_abs_difference(&reusable_solution, &wrapper_solution) < 1e-12);
    assert!(relative_residual_inf(&system.triplets, &reusable_solution, &system.rhs) < 1e-13);
}

#[test]
#[ignore = "invokes the pinned SciPy reference; run explicitly in a Python-enabled environment"]
fn bvp_shaped_fixture_matches_scipy_splu_reference() {
    use crate::Utils::test_reporting::{TestReportCapture, capture_test_table};

    let _report = TestReportCapture::new(
        "somelinalg_sparse_lu_reference",
        "bvp_shaped_fixture_matches_scipy_splu_reference",
    );
    let system = bvp_sparse_system(24, 2, 2);
    let temp = tempfile::tempdir().expect("temporary fixture directory");
    let fixture_path = temp.path().join("bvp_sparse_system.txt");
    write_fixture(&fixture_path, &system).expect("write shared Rust fixture");
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let python_script = manifest.join("src/somelinalg/sparse_lu_reference/splu_benchmark.py");
    let python = std::env::var_os("SOMELINALG_PYTHON").unwrap_or_else(|| "python".into());
    let output = Command::new(python)
        .current_dir(manifest)
        .arg(python_script)
        .arg(&fixture_path)
        .arg("--repeats")
        .arg("3")
        .output()
        .expect("launch SciPy reference runner");
    assert!(
        output.status.success(),
        "SciPy runner failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("SciPy output is UTF-8");
    let fields = stdout
        .trim()
        .split('\t')
        .filter_map(|field| field.split_once('='))
        .collect::<std::collections::HashMap<_, _>>();
    assert_eq!(fields.get("status"), Some(&"ok"));
    assert_eq!(fields.get("scipy_version"), Some(&"1.18.1"));
    assert_eq!(
        fields.get("dimension").and_then(|value| value.parse().ok()),
        Some(system.dimension())
    );
    let rust_csc_nnz = system.matrix.symbolic().row_idx().len();
    assert_eq!(
        fields.get("csc_nnz").and_then(|value| value.parse().ok()),
        Some(rust_csc_nnz),
        "Rust and SciPy must see the same canonical CSC structural count"
    );
    let residual = fields
        .get("residual_inf")
        .expect("SciPy reports residual")
        .parse::<f64>()
        .expect("residual is numeric");
    assert!(
        residual < 1e-11,
        "SciPy residual is too large: {residual:e}"
    );
    let solution_diff = fields
        .get("known_solution_max_diff")
        .expect("SciPy reports known-solution difference")
        .parse::<f64>()
        .expect("known-solution difference is numeric");
    assert!(
        solution_diff < 1e-11,
        "SciPy solution drift is too large: {solution_diff:e}"
    );

    capture_test_table(
        "Same-matrix Rust fixture -> SciPy 1.18.1 SuperLU `splu`",
        &[PythonParityRow {
            scipy: fields["scipy_version"].to_owned(),
            dimension: system.dimension(),
            mesh_nodes: system.mesh_nodes,
            csc_nnz: fields["csc_nnz"].parse().expect("nnz is numeric"),
            factor_l_nnz: fields["factor_L_nnz"].parse().expect("L nnz is numeric"),
            factor_u_nnz: fields["factor_U_nnz"].parse().expect("U nnz is numeric"),
            residual_inf: format!("{residual:.3e}"),
            known_solution_max_diff: format!("{solution_diff:.3e}"),
            result: "ok",
        }],
    );
}

#[cfg(feature = "superlu")]
#[test]
#[ignore = "requires SuperLU bridge DLL and SciPy 1.18.1; supports captured BVP fixture via env"]
fn superlu_bridge_matches_scipy_factorization_and_solution_on_shared_matrix() {
    use crate::Utils::test_reporting::{TestReportCapture, capture_test_table};
    use crate::somelinalg::superlu::{CscMatrix, SolverLibrary};

    let _report = TestReportCapture::new(
        "somelinalg_sparse_lu_reference",
        "superlu_bridge_matches_scipy_factorization_and_solution_on_shared_matrix",
    );
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let temp = tempfile::tempdir().expect("create parity fixture directory");
    let (system, fixture_path, case) =
        if let Some(path) = std::env::var_os("SOMELINALG_SPARSE_LU_CAPTURE_FIXTURE") {
            let path = std::path::PathBuf::from(path);
            let system = super::sparse_lu_reference::load_fixture(&path)
                .expect("load captured BVP matrix and RHS");
            let case = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .map(str::to_owned)
                .unwrap_or_else(|| "captured-bvp".to_owned());
            (system, path, case)
        } else {
            let system = bvp_sparse_system(24, 2, 2);
            let path = temp.path().join("synthetic_bvp_fixture.txt");
            write_fixture(&path, &system).expect("write shared BVP fixture");
            (system, path, "synthetic-bvp-24x2".to_owned())
        };

    let python = std::env::var_os("SOMELINALG_PYTHON").unwrap_or_else(|| "python".into());
    let python_script = manifest.join("src/somelinalg/sparse_lu_reference/splu_benchmark.py");
    let solution_path = temp.path().join("scipy_solution.txt");
    let output = Command::new(python)
        .current_dir(manifest)
        .env("SOMELINALG_SCIPY_BLAS_THREADS", "1")
        .env("OPENBLAS_NUM_THREADS", "1")
        .arg(python_script)
        .arg(&fixture_path)
        .arg("--repeats")
        .arg("3")
        .arg("--solution-output")
        .arg(&solution_path)
        .output()
        .expect("launch SciPy 1.18.1 reference");
    assert!(
        output.status.success(),
        "SciPy runner failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("SciPy output is UTF-8");
    let fields = stdout
        .trim()
        .split('\t')
        .filter_map(|field| field.split_once('='))
        .collect::<std::collections::HashMap<_, _>>();
    assert_eq!(fields.get("status"), Some(&"ok"));
    assert_eq!(fields.get("scipy_version"), Some(&"1.18.1"));

    let library_path = std::env::var("SUPERLU_LIBRARY")
        .or_else(|_| std::env::var("SUPERLU_EXPERIMENTAL_LIBRARY"))
        .expect("set SUPERLU_LIBRARY to the built platform bridge");
    let control = SolverLibrary::load(library_path).expect("load pinned SuperLU bridge");
    let symbolic = system.matrix.symbolic();
    let matrix = CscMatrix::try_from_usize(
        system.dimension(),
        symbolic.col_ptr(),
        symbolic.row_idx(),
        system.matrix.val(),
    )
    .expect("convert shared CSC matrix to the SuperLU ABI");
    let rust = control
        .solve(&matrix, &system.rhs)
        .expect("solve shared matrix through native SuperLU");
    let scipy_solution = std::fs::read_to_string(&solution_path)
        .expect("read full SciPy solution vector")
        .split_whitespace()
        .map(|value| value.parse::<f64>().expect("SciPy solution is numeric"))
        .collect::<Vec<_>>();
    assert_eq!(scipy_solution.len(), system.dimension());

    let scipy_l_nnz = fields["factor_L_nnz"]
        .parse::<usize>()
        .expect("SciPy L nnz is numeric");
    let scipy_u_nnz = fields["factor_U_nnz"]
        .parse::<usize>()
        .expect("SciPy U nnz is numeric");
    assert_eq!(
        rust.factor_l_csc_nnz, scipy_l_nnz,
        "materialized L fill parity"
    );
    assert_eq!(
        rust.factor_u_csc_nnz, scipy_u_nnz,
        "materialized U fill parity"
    );
    let solution_max_diff = max_abs_difference(&rust.solution, &scipy_solution);
    let scipy_solution_inf = scipy_solution
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    let solution_tolerance = 1e-10 * scipy_solution_inf.max(1.0);
    assert!(
        solution_max_diff <= solution_tolerance,
        "Rust/SciPy solution drift {solution_max_diff:e} exceeds {solution_tolerance:e}"
    );
    let relative_residual = relative_residual_inf(&system.triplets, &rust.solution, &system.rhs);
    assert!(
        relative_residual < 1e-12,
        "SuperLU relative residual {relative_residual:e}"
    );

    capture_test_table(
        "Same CSC matrix/RHS: SuperLU bundled CBLAS vs SciPy 1.18.1 OpenBLAS; both sequential",
        &[NativePythonParityRow {
            case,
            scipy_blas: fields["blas"].to_owned(),
            rust_blas: "SuperLU-bundled-CBLAS".into(),
            blas_threads: fields["blas_threads"].to_owned(),
            dimension: system.dimension(),
            matrix_nnz: matrix.nnz(),
            scipy_l_nnz,
            rust_l_nnz: rust.factor_l_csc_nnz,
            scipy_u_nnz,
            rust_u_nnz: rust.factor_u_csc_nnz,
            solution_max_diff: format!("{solution_max_diff:.3e}"),
            relative_residual: format!("{relative_residual:.3e}"),
            status: "ok",
        }],
    );
}

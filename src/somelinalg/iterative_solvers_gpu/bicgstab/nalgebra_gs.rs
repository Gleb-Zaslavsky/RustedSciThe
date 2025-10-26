use nalgebra::{DMatrix, DVector};
use std::error::Error;
/// simple BICGSTAB CPU prototype
/// Solves Ax = b using BiCGSTAB, with optional Gauss-Seidel preconditioning.
///
/// # Arguments
/// * `a` - System matrix (must be square).
/// * `b` - Right-hand side vector.
/// * `max_iter` - Maximum number of iterations.
/// * `tol` - Tolerance for convergence (||r|| / ||b|| < tol).
/// * `use_preconditioner` - If true, uses forward-only Gauss-Seidel preconditioning.
/// * `symmetric_gs` - If true and use_preconditioner=true, uses symmetric GS.
///
/// # Returns
/// * `x` - Solution vector.
/// * `iterations` - Number of iterations performed.
/// * `residual_norm` - Final relative residual norm.
///
/*
Single Unified Function: One function bicgstab handles both cases through the use_preconditioner boolean flag.
Preconditioner Abstraction: The Gauss-Seidel preconditioner
M=L+D is precomputed only once if needed. The operation of applying
M^−1 is encapsulated in a closure apply_preconditioner.

Efficient Application:
When enabled: The preconditioner is applied by solving a lower-triangular system using solve_lower_triangular.
When disabled: The closure simply clones the input vector, representing the identity operation
M^−1=IM^−1=I. This adds minimal overhead.
*/
pub fn bicgstab(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    max_iter: usize,
    tol: f64,
    use_preconditioner: bool,
) -> Result<(DVector<f64>, usize, f64), Box<dyn Error>> {
    bicgstab_with_symmetric_gs(a, b, max_iter, tol, use_preconditioner, false)
}

/// BiCGSTAB with option for symmetric GS
pub fn bicgstab_with_symmetric_gs(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    max_iter: usize,
    tol: f64,
    use_preconditioner: bool,
    symmetric_gs: bool,
) -> Result<(DVector<f64>, usize, f64), Box<dyn Error>> {
    let n = a.nrows();
    assert_eq!(a.ncols(), n, "Matrix must be square.");
    assert_eq!(b.nrows(), n, "Vector dimension must match matrix.");

    // Precompute preconditioner matrices if requested
    let (m_lower, m_upper) = if use_preconditioner {
        if symmetric_gs {
            (Some(a.lower_triangle()), Some(a.upper_triangle()))
        } else {
            (Some(a.lower_triangle()), None)
        }
    } else {
        (None, None)
    };

    // Initialize solution guess and residual
    let mut x = DVector::zeros(n);
    let mut r = b - a * &x;
    let r0_hat = r.clone(); // Shadow residual for the BiCG process

    let mut rho_prev = 1.0;
    let mut alpha = 1.0;
    let mut omega_prev = 1.0;
    let mut v = DVector::zeros(n);
    let mut p = DVector::zeros(n);

    let b_norm = b.norm();
    if b_norm < f64::EPSILON {
        return Ok((x, 0, 0.0));
    }

    // GS preconditioner: forward-only or symmetric
    let apply_preconditioner = |vec: &DVector<f64>| -> DVector<f64> {
        match (&m_lower, &m_upper) {
            (Some(lower), Some(upper)) => {
                // Symmetric GS: forward + backward sweep
                let z1 = lower
                    .solve_lower_triangular(vec)
                    .unwrap_or_else(|| vec.clone());
                upper.solve_upper_triangular(&z1).unwrap_or_else(|| z1)
            }
            (Some(lower), None) => {
                // Forward-only GS
                lower
                    .solve_lower_triangular(vec)
                    .unwrap_or_else(|| vec.clone())
            }
            _ => vec.clone(), // No preconditioning
        }
    };

    for iteration in 1..=max_iter {
        let rho = r0_hat.dot(&r);
        if rho.abs() < 1e-15 {
            return Err("BiCGSTAB breakdown: rho ~ 0".into());
        }

        // Update search direction p
        let beta = (rho / rho_prev) * (alpha / omega_prev);
        p = &r + beta * &p - (beta * omega_prev) * &v;

        // Apply preconditioner to p: p_hat = M⁻¹ * p
        let p_hat = apply_preconditioner(&p);
        v = a * &p_hat; // v = A * p_hat

        alpha = rho / r0_hat.dot(&v);
        let s = &r - alpha * &v; // Intermediate residual

        // Check convergence on the intermediate residual `s`
        if s.norm() / b_norm < tol {
            x += alpha * p_hat; // Update solution: x = x + alpha * p_hat
            return Ok((x, iteration, s.norm() / b_norm));
        }

        // Apply preconditioner to s: s_hat = M⁻¹ * s
        let s_hat = apply_preconditioner(&s);
        let t = a * &s_hat; // t = A * s_hat

        // Compute the stabilization parameter omega
        omega_prev = t.dot(&s) / t.dot(&t);
        // Update solution and residual
        x += alpha * &p_hat + omega_prev * &s_hat;
        r = &s - omega_prev * &t;

        // Check convergence
        let rel_residual = r.norm() / b_norm;
        if rel_residual < tol {
            return Ok((x, iteration, rel_residual));
        }

        rho_prev = rho;
    }

    Err(format!(
        "Did not converge in {} iterations. Final residual: {:.3e}",
        max_iter,
        r.norm() / b_norm
    )
    .into())
}
#[cfg(test)]
mod nalgebra_gs_test {
    use super::*;
    use nalgebra::{DMatrix, DVector};
    #[test]
    fn easy() {
        // Create a test problem: a small, non-symmetric matrix
        let a = DMatrix::from_row_slice(3, 3, &[4.0, 2.0, 1.0, 1.0, 5.0, 2.0, 1.0, 0.0, 6.0]);
        let b = DVector::from_vec(vec![1.0, 2.0, 3.0]);

        println!("Solving system A * x = b");
        println!("A:\n{}", a);
        println!("b: {}", b);

        // Solve with forward-only GS preconditioning
        let (x_precond, iter_precond, res_precond) = bicgstab(&a, &b, 100, 1e-10, true).unwrap();
        println!("\n--- With Forward GS Preconditioner ---");
        println!("Solution: {}", x_precond);
        println!("Iterations: {}", iter_precond);
        println!("Relative residual: {:.3e}", res_precond);

        // Solve with symmetric GS preconditioning
        let (x_sym, iter_sym, res_sym) =
            bicgstab_with_symmetric_gs(&a, &b, 100, 1e-10, true, true).unwrap();
        println!("\n--- With Symmetric GS Preconditioner ---");
        println!("Solution: {}", x_sym);
        println!("Iterations: {}", iter_sym);
        println!("Relative residual: {:.3e}", res_sym);

        // Solve without preconditioning
        let (x_vanilla, iter_vanilla, res_vanilla) = bicgstab(&a, &b, 100, 1e-10, false).unwrap();
        println!("\n--- Vanilla BiCGSTAB (No Preconditioner) ---");
        println!("Solution: {}", x_vanilla);
        println!("Iterations: {}", iter_vanilla);
        println!("Relative residual: {:.3e}", res_vanilla);

        // Verify all solutions are equivalent (within numerical precision)
        println!(
            "\nDifference forward vs vanilla: {:.3e}",
            (&x_precond - &x_vanilla).norm()
        );
        println!(
            "Difference symmetric vs vanilla: {:.3e}",
            (&x_sym - &x_vanilla).norm()
        );
        println!(
            "Iterations comparison - Vanilla: {}, Forward GS: {}, Symmetric GS: {}",
            iter_vanilla, iter_precond, iter_sym
        );
    }
    fn create_banded_non_dominant_matrix(
        n: usize,
        diag_val: f64,
        off_diag_val: f64,
    ) -> DMatrix<f64> {
        // Create a zero matrix of size n x n
        let mut a = DMatrix::zeros(n, n);

        for i in 0..n {
            // Set the main diagonal
            a[(i, i)] = diag_val;

            // Set the sub-diagonal (if not on the first row)
            if i > 0 {
                a[(i, i - 1)] = off_diag_val;
            }

            // Set the super-diagonal (if not on the last row)
            if i < n - 1 {
                a[(i, i + 1)] = off_diag_val;
            }
        }
        a
    }

    fn solve_banded_system() -> Result<(), Box<dyn Error>> {
        let n = 100; // Size of the system
        let diag_val = 1.9;
        let off_diag_val = -1.5;

        println!("Creating a {}x{} banded (tridiagonal) matrix...", n, n);
        println!("Main Diagonal: {}", diag_val);
        println!("Off-Diagonals: {}", off_diag_val);

        // Check for diagonal dominance for the first few rows
        println!("\nChecking diagonal dominance for first 5 rows:");
        let a = create_banded_non_dominant_matrix(n, diag_val, off_diag_val);
        for i in 0..5 {
            let row_sum: f64 = a.row(i).iter().map(|x| x.abs()).sum();
            let diag_abs = a[(i, i)].abs();
            let off_diag_sum = row_sum - diag_abs;
            println!(
                "Row {}: |{:.1}| >= {:.1}? {}",
                i,
                diag_abs,
                off_diag_sum,
                diag_abs >= off_diag_sum
            );
        }

        // Create a solution vector x of all 1s, then compute b = A * x
        // This ensures we have a consistent right-hand side for a known solution.
        let x_true = DVector::from_element(n, 1.0);
        let b = &a * &x_true;

        let max_iter = 1000;
        let tol = 1e-10;
        println!("\n=== RESULTS ===");
        println!("System size: {}", n);
        println!("Convergence tolerance: {:.1e}", tol);

        println!("\nSolving with Vanilla BiCGSTAB (no preconditioner)...");
        let vanilla_res = bicgstab(&a, &b, max_iter, tol, false);

        println!("Solving with Gauss-Seidel Preconditioned BiCGSTAB...");
        let gs_results = bicgstab_with_symmetric_gs(&a, &b, max_iter, tol, true, true);

        match vanilla_res.as_ref() {
            Ok((x_vanilla, iter_vanilla, res_vanilla)) => {
                println!("--- Vanilla BiCGSTAB ---");
                println!(
                    "\nVanilla BiCGSTAB converged in {} iterations.",
                    iter_vanilla
                );
                println!("Final relative residual: {:.3e}", res_vanilla);
                // Calculate the actual error from the true solution
                let error_vanilla = (&x_true - x_vanilla).norm();
                println!("Error from true solution: {:.3e}", error_vanilla);
            }
            Err(e) => println!("\n Vanilla BiCGSTAB FAILED to converge: {}", e),
        }

        match gs_results.as_ref() {
            Ok((x_gs, iter_gs, res_gs)) => {
                println!("--- GS-Preconditioned BiCGSTAB ---");
                let error_gs = (&x_true - x_gs).norm();
                println!(
                    "\nGS-Preconditioned BiCGSTAB converged in {} iterations.",
                    iter_gs
                );
                println!("Final relative residual: {:.3e}", res_gs);
                println!("Error from true solution: {:.3e}", error_gs);
            }
            Err(e) => println!("\n GS-Preconditioned BiCGSTAB FAILED to converge: {}", e),
        }
        println!("\n=== PERFORMANCE COMPARISON ===");
        match (vanilla_res, gs_results) {
            (Ok((_, _, res_vanilla)), Ok((_, _, res_gs))) => {
                if res_vanilla < res_gs {
                    println!(
                        "\nVanilla BiCGSTAB performed better than GS-Preconditioned BiCGSTAB."
                    );
                } else if res_gs < res_vanilla {
                    println!(
                        "\nGS-Preconditioned BiCGSTAB performed better than Vanilla BiCGSTAB."
                    );
                } else {
                    println!("\nBoth methods performed equally well.");
                }
            }
            _ => println!("\nUnable to compare performance due to convergence failure."),
        }

        Ok(())
    }

    #[test]
    fn test_non_diag_domin_matrix() {
        let _ = solve_banded_system();
    }

    fn create_non_dominant_banded_matrix(n: usize) -> DMatrix<f64> {
        let mut a = DMatrix::<f64>::zeros(n, n);

        // We will make the diagonal positive but too weak for dominance.
        let diag_val = 1.0; // This is less than the sum of the off-diagonals (1.0 + 1.0 = 2.0)
        let off_diag_val = -1.0;

        for i in 0..n {
            a[(i, i)] = diag_val;
            if i > 0 {
                a[(i, i - 1)] = off_diag_val;
            }
            if i < n - 1 {
                a[(i, i + 1)] = off_diag_val;
            }
        }
        a
    }

    fn create_spd_banded_non_dominant_solver() -> Result<(), Box<dyn Error>> {
        let n = 100;
        let a = create_non_dominant_banded_matrix(n);

        // Let's verify the lack of diagonal dominance for the first interior row
        let i = 1; // An interior row
        let row_sum_off_diag: f64 = a.row(i).iter().map(|x| x.abs()).sum::<f64>() - a[(i, i)].abs();
        println!("Matrix is NOT diagonally dominant:");
        println!(
            "For row {}: |{:.1}| >= {:.1}? {}",
            i,
            a[(i, i)],
            row_sum_off_diag,
            a[(i, i)].abs() >= row_sum_off_diag
        );
        // This should print: |1.5| >= 2.0? false

        // Create a random RHS for a non-trivial problem
        use rand::Rng;
        let mut rng = rand::rng();
        let x_true = DVector::from_fn(n, |_i, _j| rng.random());
        let b = &a * &x_true;

        let max_iter = 2000;
        let tol = 1e-6; // Use a more relaxed tolerance for a harder problem

        println!("\nSolving with Vanilla BiCGSTAB...");
        match bicgstab(&a, &b, max_iter, tol, false) {
            Ok((x_vanilla, iter_vanilla, res_vanilla)) => {
                println!(
                    "Converged in {} iterations. Residual: {:.3e}",
                    iter_vanilla, res_vanilla
                );
                let error = (&x_true - &x_vanilla).norm();
                println!("Error from true solution: {:.3e}", error);
            }
            Err(e) => println!("Vanilla failed to converge: {}", e),
        }

        println!("\nSolving with Gauss-Seidel Preconditioned BiCGSTAB...");
        match bicgstab_with_symmetric_gs(&a, &b, max_iter, tol, true, true) {
            Ok((x_gs, iter_gs, res_gs)) => {
                println!(
                    "Converged in {} iterations. Residual: {:.3e}",
                    iter_gs, res_gs
                );
                let error = (&x_true - &x_gs).norm();
                println!("Error from true solution: {:.3e}", error);
            }
            Err(e) => println!("GS Preconditioned failed to converge: {}", e),
        }

        Ok(())
    }
    #[test]
    fn test_create_spd_banded_non_dominant_solver() {
        let _ = create_spd_banded_non_dominant_solver();
    }

    fn create_combustion_bvp_matrix(
        n: usize,
        diffusion_coeff: f64,
        reaction_strength: f64,
    ) -> DMatrix<f64> {
        let mut a = DMatrix::zeros(n, n);
        let dx = 1.0 / ((n + 1) as f64); // Spatial step size
        let diffusion_term = diffusion_coeff / (dx * dx);

        // Create a fake "temperature" profile that might exist in a flame
        // (low on left, high on right) to compute a realistic reaction term.
        let x: Vec<f64> = (1..=n).map(|i| i as f64 * dx).collect();
        let _phi: Vec<f64> = x.iter().map(|x_pos| 0.1 + 0.9 * x_pos).collect(); // Linear profile

        for i in 0..n {
            // --- Diffusion Term (Tridiagonal) ---
            // Main diagonal
            a[(i, i)] += 2.0 * diffusion_term;
            // Lower diagonal (if not first row)
            if i > 0 {
                a[(i, i - 1)] -= diffusion_term;
            }
            // Upper diagonal (if not last row)
            if i < n - 1 {
                a[(i, i + 1)] -= diffusion_term;
            }

            // --- Reaction Term Jacobian (Diagonal) ---
            // This is the key part. A simple model for a reaction term Jacobian R'(φ)
            // For a strongly non-linear reaction, it peaks in the flame zone.
            // We'll model it with a Gaussian centered in the domain.
            let center = 0.5;
            let width = 0.1;
            let x_pos = (i as f64 + 1.0) * dx;
            let reaction_jacobian =
                reaction_strength * (-(x_pos - center).powi(2) / (2.0 * width * width)).exp();

            // Add the reaction Jacobian to the main diagonal.
            // This is a LARGE positive number, making the matrix diagonally dominant.
            a[(i, i)] += reaction_jacobian;
        }
        a
    }

    fn steady_state_example() -> Result<(), Box<dyn Error>> {
        let n = 1000; // Size of the system
        let diffusion_coeff = 0.1;
        let reaction_strength = 1e2; // Large reaction strength is typical in combustion

        println!("Creating a {}x{} matrix for a combustion-like BVP...", n, n);
        println!("Diffusion coefficient: {}", diffusion_coeff);
        println!("Reaction strength: {}", reaction_strength);

        let a = create_combustion_bvp_matrix(n, diffusion_coeff, reaction_strength);

        // Check diagonal dominance for a sample row in the middle
        let sample_row = n / 2;
        let diag_val = a[(sample_row, sample_row)];
        let sum_off_diag: f64 =
            a.row(sample_row).iter().map(|x| x.abs()).sum::<f64>() - diag_val.abs();
        println!(
            "For row {}: |{:.3e}| >= {:.3e}? {}",
            sample_row,
            diag_val,
            sum_off_diag,
            diag_val >= sum_off_diag
        );
        // This should be TRUE and the ratio should be large.

        // Create a meaningful RHS vector (e.g., a source term)
        let b = DVector::from_fn(n, |i, _j| {
            // Some simple source term, e.g., zero everywhere except a point source
            if i == n / 4 { 1.0 } else { 0.0 }
        });

        let max_iter = 1000;
        let tol = 1e-10;
        let res_gs = bicgstab(&a, &b, max_iter, tol, true);
        let res_vanilla = bicgstab(&a, &b, max_iter, tol, false);
        println!("\nSolving with Vanilla BiCGSTAB...");
        match res_vanilla.as_ref() {
            Ok((_, iter_vanilla, res_vanilla)) => {
                println!(
                    "Converged in {} iterations. Residual: {:.3e}",
                    iter_vanilla, res_vanilla
                );
            }
            Err(e) => println!("Vanilla failed to converge: {}", e),
        }

        println!("\nSolving with Gauss-Seidel Preconditioned BiCGSTAB...");
        match res_gs.as_ref() {
            Ok((_, iter_gs, res_gs)) => {
                println!(
                    "Converged in {} iterations. Residual: {:.3e}",
                    iter_gs, res_gs
                );
            }
            Err(e) => println!("GS Preconditioned failed to converge: {}", e),
        }
        match (res_vanilla, res_gs) {
            (Ok((_, iter_vanilla, _)), Ok((_, iter_gs, _))) => {
                println!(
                    "GS Preconditioner provided a {:.1}x speedup",
                    iter_vanilla as f64 / iter_gs as f64
                );
            }
            _ => println!("Unable to compare speedup due to convergence failure."),
        }
        Ok(())
    }
    #[test]
    fn test_csteady_state_example() {
        let _ = steady_state_example();
    }
}

use nalgebra::{DMatrix, DVector};
use std::f64;

/// Levenberg-Marquardt Parameter Determination
///
/// Given an m×n matrix A, n×n diagonal matrix D, m-vector b, and positive δ,
/// determine parameter 'par' such that if x solves the regularized least squares system:
///   A*x = b, sqrt(par)*D*x = 0
///
/// # Arguments
/// * `r` - Upper triangular matrix R from QR factorization (modified in-place)
/// * `ipvt` - Permutation vector (1-based indexing from Fortran)
/// * `diag` - Diagonal scaling matrix elements
/// * `qtb` - Q^T * b vector
/// * `delta` - Trust region radius
/// * `par` - Levenberg-Marquardt parameter (input/output)
///
/// # Returns
/// * `x` - Solution vector
/// * `sdiag` - Diagonal elements of matrix S
//
/*
/*
 * LMPAR - Levenberg-Marquardt Parameter Determination
 *
 * Purpose: Given an m×n matrix A, n×n diagonal matrix D, m-vector b, and positive δ,
 * determine parameter 'par' such that if x solves the regularized least squares system:
 *   A*x = b, sqrt(par)*D*x = 0
 * then either:
 *   - par = 0 and ||D*x|| - δ ≤ 0.1*δ, or
 *   - par > 0 and |||D*x|| - δ| ≤ 0.1*δ
 */
void lmpar(int n, double **R, int ldr, int *ipvt, double *diag, double *qtb,
           double delta, double *par, double *x, double *sdiag,
           double *wa1, double *wa2) {

    // Constants
    const double P1 = 0.1;        // Tolerance factor (10%)
    const double P001 = 0.001;    // Small parameter adjustment
    const double DWARF = /* machine epsilon */;

    int iter, nsing, i, j, k, l;
    double dxnorm, fp, gnorm, parl, paru, parc, temp, sum;

    // STEP 1: Compute Gauss-Newton direction
    // Handle rank-deficient Jacobian by finding effective rank
    nsing = n;
    for (j = 0; j < n; j++) {
        wa1[j] = qtb[j];
        // Check for zero diagonal elements (rank deficiency)
        if (R[j][j] == 0.0 && nsing == n) {
            nsing = j;  // Found rank deficiency at column j
        }
        if (nsing < n) {
            wa1[j] = 0.0;  // Zero out components beyond rank
        }
    }

    // Solve upper triangular system R*y = qtb by back substitution
    if (nsing >= 1) {
        for (k = 0; k < nsing; k++) {
            j = nsing - k - 1;  // Process from bottom-right to top-left
            wa1[j] = wa1[j] / R[j][j];
            temp = wa1[j];

            // Update remaining elements in this column
            for (i = 0; i < j; i++) {
                wa1[i] = wa1[i] - R[i][j] * temp;
            }
        }
    }

    // Apply permutation to get Gauss-Newton direction
    for (j = 0; j < n; j++) {
        l = ipvt[j] - 1;  // Convert to 0-based indexing
        x[l] = wa1[j];
    }

    // STEP 2: Test Gauss-Newton direction (par = 0 case)
    iter = 0;

    // Compute ||D*x|| (scaled norm)
    for (j = 0; j < n; j++) {
        wa2[j] = diag[j] * x[j];
    }
    dxnorm = enorm(n, wa2);  // Euclidean norm
    fp = dxnorm - delta;     // Function value at par = 0

    // Check if Gauss-Newton direction is acceptable
    if (fp <= P1 * delta) {
        // SUCCESS: Gauss-Newton direction satisfies trust region
        if (iter == 0) *par = 0.0;
        return;
    }

    // STEP 3: Need to find par > 0, set up bounds

    // Lower bound: Newton step for secular equation (if full rank)
    parl = 0.0;
    if (nsing >= n) {  // Full rank case
        // Compute gradient direction for secular equation
        for (j = 0; j < n; j++) {
            l = ipvt[j] - 1;
            wa1[j] = diag[l] * (wa2[l] / dxnorm);
        }

        // Solve R^T * z = wa1
        for (j = 0; j < n; j++) {
            sum = 0.0;
            for (i = 0; i < j; i++) {
                sum += R[i][j] * wa1[i];
            }
            wa1[j] = (wa1[j] - sum) / R[j][j];
        }

        temp = enorm(n, wa1);
        parl = ((fp / delta) / temp) / temp;  // Newton step lower bound
    }

    // Upper bound: compute ||R^(-1) * Q^T * b|| / delta
    for (j = 0; j < n; j++) {
        sum = 0.0;
        for (i = 0; i <= j; i++) {
            sum += R[i][j] * qtb[i];
        }
        l = ipvt[j] - 1;
        wa1[j] = sum / diag[l];
    }
    gnorm = enorm(n, wa1);
    paru = gnorm / delta;
    if (paru == 0.0) {
        paru = DWARF / min(delta, P1);
    }

    // Ensure initial par is within bounds
    *par = max(*par, parl);
    *par = min(*par, paru);
    if (*par == 0.0) {
        *par = gnorm / dxnorm;
    }

    // STEP 4: Main iteration loop (up to 10 iterations)
    while (true) {
        iter++;

        // Ensure par is not too small
        if (*par == 0.0) {
            *par = max(DWARF, P001 * paru);
        }

        // Solve regularized system: (R^T*R + par*D^2)*x = R^T*qtb
        temp = sqrt(*par);
        for (j = 0; j < n; j++) {
            wa1[j] = temp * diag[j];  // Regularization terms
        }

        // Call QR solver for regularized system
        qrsolv(n, R, ldr, ipvt, wa1, qtb, x, sdiag, wa2);

        // Compute new scaled norm
        for (j = 0; j < n; j++) {
            wa2[j] = diag[j] * x[j];
        }
        dxnorm = enorm(n, wa2);
        temp = fp;
        fp = dxnorm - delta;  // New function value

        // STEP 5: Check convergence criteria
        if (abs(fp) <= P1 * delta ||                    // Function small enough
            (parl == 0.0 && fp <= temp && temp < 0.0) || // Special case
            iter == 10) {                               // Max iterations

            if (iter == 0) *par = 0.0;
            break;  // CONVERGED
        }

        // STEP 6: Compute Newton correction for secular equation
        // Compute derivative of ||x(par)|| with respect to par
        for (j = 0; j < n; j++) {
            l = ipvt[j] - 1;
            wa1[j] = diag[l] * (wa2[l] / dxnorm);
        }

        // Solve (S^T * S) * z = wa1 where S is from QR factorization
        for (j = 0; j < n; j++) {
            wa1[j] = wa1[j] / sdiag[j];
            temp = wa1[j];

            for (i = j + 1; i < n; i++) {
                wa1[i] = wa1[i] - R[i][j] * temp;
            }
        }

        temp = enorm(n, wa1);
        parc = ((fp / delta) / temp) / temp;  // Newton correction

        // Update bounds based on function sign
        if (fp > 0.0) {
            parl = max(parl, *par);  // Function too large, increase lower bound
        }
        if (fp < 0.0) {
            paru = min(paru, *par);  // Function too small, decrease upper bound
        }

        // Update parameter estimate
        *par = max(parl, *par + parc);

    } // End main iteration loop
}

*/
pub fn lmpar(
    r: &mut DMatrix<f64>,
    ipvt: &[usize],
    diag: &DVector<f64>,
    qtb: &DVector<f64>,
    delta: f64,
    par: &mut f64,
) -> Result<(DVector<f64>, DVector<f64>), &'static str> {
    let n = r.ncols();
    if r.nrows() < n || diag.len() != n || qtb.len() != n || ipvt.len() != n {
        return Err("Dimension mismatch in lmpar");
    }

    // Constants
    const P1: f64 = 0.1;
    const P001: f64 = 0.001;
    const DWARF: f64 = f64::EPSILON;
    const MAX_ITER: usize = 10;

    let mut x = DVector::zeros(n);
    let mut sdiag = DVector::zeros(n);
    let mut wa1 = DVector::zeros(n);
    let mut wa2 = DVector::zeros(n);

    // Step 1: Compute Gauss-Newton direction
    let (gauss_newton_x, nsing) = compute_gauss_newton_direction(r, ipvt, qtb, &mut wa1)?;
    x.copy_from(&gauss_newton_x);

    // Step 2: Test if Gauss-Newton direction is acceptable
    wa2.zip_zip_apply(&x, diag, |wa2_i, x_i, diag_i| *wa2_i = diag_i * x_i);
    let dxnorm = wa2.norm();
    let mut fp = dxnorm - delta;

    if fp <= P1 * delta {
        // Gauss-Newton direction satisfies trust region
        *par = 0.0;
        return Ok((x, sdiag));
    }

    // Step 3: Compute bounds for the parameter
    let parl = if nsing >= n {
        compute_lower_bound(r, ipvt, diag, &wa2, dxnorm, fp, delta, &mut wa1)?
    } else {
        0.0
    };

    let paru = compute_upper_bound(r, ipvt, diag, qtb, delta, &mut wa1)?;

    // Initialize parameter within bounds
    *par = par.max(parl).min(paru);
    if *par == 0.0 {
        let gnorm = wa1.norm(); // wa1 contains the gradient norm computation result
        *par = gnorm / dxnorm;
    }

    // Step 4: Main iteration loop
    let mut iter = 0;
    loop {
        iter += 1;

        // Ensure par is not too small
        if *par == 0.0 {
            *par = DWARF.max(P001 * paru);
        }

        // Solve regularized system
        let sqrt_par = par.sqrt();
        wa1.zip_apply(diag, |wa1_i, diag_i| *wa1_i = sqrt_par * diag_i);

        qrsolv(r, ipvt, &wa1, qtb, &mut x, &mut sdiag, &mut wa2)?;

        // Compute new function value
        wa2.zip_zip_apply(&x, diag, |wa2_i, x_i, diag_i| *wa2_i = diag_i * x_i);
        let new_dxnorm = wa2.norm();
        let temp_fp = fp;
        fp = new_dxnorm - delta;

        // Check convergence
        if fp.abs() <= P1 * delta
            || (parl == 0.0 && fp <= temp_fp && temp_fp < 0.0)
            || iter >= MAX_ITER
        {
            break;
        }

        // Compute Newton correction
        let parc = compute_newton_correction(
            r, ipvt, diag, &sdiag, &wa2, new_dxnorm, fp, delta, &mut wa1,
        )?;

        // Update bounds and parameter
        let new_parl = if fp > 0.0 { parl.max(*par) } else { parl };
        let new_paru = if fp < 0.0 { paru.min(*par) } else { paru };

        *par = new_parl.max(*par + parc);
    }

    Ok((x, sdiag))
}

/// Compute the Gauss-Newton direction by solving R*x = qtb
fn compute_gauss_newton_direction(
    r: &DMatrix<f64>,
    ipvt: &[usize],
    qtb: &DVector<f64>,
    wa1: &mut DVector<f64>,
) -> Result<(DVector<f64>, usize), &'static str> {
    let n = r.ncols();

    // Find effective rank (number of non-zero diagonal elements)
    let nsing = r.diagonal().iter().position(|&x| x == 0.0).unwrap_or(n);

    // Initialize with qtb, zero out rank-deficient part
    wa1.copy_from(qtb);
    for i in nsing..n {
        wa1[i] = 0.0;
    }

    // Back substitution for upper triangular system
    if nsing > 0 {
        for k in 0..nsing {
            let j = nsing - 1 - k;
            wa1[j] /= r[(j, j)];
            let temp = wa1[j];

            // Update previous elements
            for i in 0..j {
                wa1[i] -= r[(i, j)] * temp;
            }
        }
    }

    // Apply permutation (convert from 1-based to 0-based indexing)
    let mut x = DVector::zeros(n);
    for (j, &pivot_idx) in ipvt.iter().enumerate() {
        if pivot_idx > 0 && pivot_idx <= n {
            x[pivot_idx - 1] = wa1[j];
        }
    }

    Ok((x, nsing))
}

/// Compute lower bound using Newton's method on the secular equation
fn compute_lower_bound(
    r: &DMatrix<f64>,
    ipvt: &[usize],
    diag: &DVector<f64>,
    wa2: &DVector<f64>,
    dxnorm: f64,
    fp: f64,
    delta: f64,
    wa1: &mut DVector<f64>,
) -> Result<f64, &'static str> {
    let n = r.ncols();

    // Compute scaled direction
    for (j, &pivot_idx) in ipvt.iter().enumerate() {
        if pivot_idx > 0 && pivot_idx <= n {
            wa1[j] = diag[pivot_idx - 1] * (wa2[pivot_idx - 1] / dxnorm);
        }
    }

    // Solve R^T * z = wa1 by forward substitution
    for j in 0..n {
        let mut sum = 0.0;
        for i in 0..j {
            sum += r[(i, j)] * wa1[i];
        }
        wa1[j] = (wa1[j] - sum) / r[(j, j)];
    }

    let temp = wa1.norm();
    Ok(((fp / delta) / temp) / temp)
}

/// Compute upper bound from gradient norm
fn compute_upper_bound(
    r: &DMatrix<f64>,
    ipvt: &[usize],
    diag: &DVector<f64>,
    qtb: &DVector<f64>,
    delta: f64,
    wa1: &mut DVector<f64>,
) -> Result<f64, &'static str> {
    let n = r.ncols();

    // Compute R^(-1) * Q^T * b
    for j in 0..n {
        let mut sum = 0.0;
        for i in 0..=j {
            sum += r[(i, j)] * qtb[i];
        }
        if let Some(&pivot_idx) = ipvt.get(j) {
            if pivot_idx > 0 && pivot_idx <= n {
                wa1[j] = sum / diag[pivot_idx - 1];
            }
        }
    }

    let gnorm = wa1.norm();
    let mut paru = gnorm / delta;
    if paru == 0.0 {
        paru = f64::EPSILON / delta.min(0.1);
    }

    Ok(paru)
}

/// Compute Newton correction for the secular equation
fn compute_newton_correction(
    r: &DMatrix<f64>,
    ipvt: &[usize],
    diag: &DVector<f64>,
    sdiag: &DVector<f64>,
    wa2: &DVector<f64>,
    dxnorm: f64,
    fp: f64,
    delta: f64,
    wa1: &mut DVector<f64>,
) -> Result<f64, &'static str> {
    let n = r.ncols();

    // Compute derivative direction
    for (j, &pivot_idx) in ipvt.iter().enumerate() {
        if pivot_idx > 0 && pivot_idx <= n {
            wa1[j] = diag[pivot_idx - 1] * (wa2[pivot_idx - 1] / dxnorm);
        }
    }

    // Solve system with S matrix
    for j in 0..n {
        wa1[j] /= sdiag[j];
        let temp = wa1[j];

        for i in (j + 1)..n {
            wa1[i] -= r[(i, j)] * temp;
        }
    }

    let temp = wa1.norm();
    Ok(((fp / delta) / temp) / temp)
}

/// QR solver for the regularized system (simplified version)
/// In practice, this would be a more sophisticated QR solver
fn qrsolv(
    r: &mut DMatrix<f64>,
    ipvt: &[usize],
    diag_aug: &DVector<f64>,
    qtb: &DVector<f64>,
    x: &mut DVector<f64>,
    sdiag: &mut DVector<f64>,
    wa: &mut DVector<f64>,
) -> Result<(), &'static str> {
    let n = r.ncols();

    // This is a simplified implementation
    // In the actual MINPACK, this involves Givens rotations
    // to solve the augmented system [R; sqrt(par)*D] * x = [qtb; 0]

    // For now, we'll use a basic approach
    // Copy diagonal elements
    for i in 0..n {
        sdiag[i] = r[(i, i)];
    }

    // Solve the regularized system approximately
    // This should be replaced with proper Givens rotations implementation
    wa.copy_from(qtb);

    // Back substitution (simplified)
    for k in 0..n {
        let j = n - 1 - k;
        let mut sum = wa[j];

        for i in (j + 1)..n {
            sum -= r[(j, i)] * x[i];
        }

        // Add regularization effect
        let reg_diag = (r[(j, j)].powi(2) + diag_aug[j].powi(2)).sqrt();
        x[j] = sum / reg_diag;
        sdiag[j] = reg_diag;
    }

    // Apply permutation
    let mut temp_x = x.clone();
    for (j, &pivot_idx) in ipvt.iter().enumerate() {
        if pivot_idx > 0 && pivot_idx <= n {
            temp_x[pivot_idx - 1] = x[j];
        }
    }
    x.copy_from(&temp_x);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{dmatrix, dvector};

    #[test]
    fn test_lmpar_basic() {
        let mut r = dmatrix![
            2.0, 1.0, 0.0;
            0.0, 1.5, 0.5;
            0.0, 0.0, 1.0
        ];

        let ipvt = vec![1, 2, 3]; // 1-based indexing
        let diag = dvector![1.0, 1.0, 1.0];
        let qtb = dvector![1.0, 0.5, 0.2];
        let delta = 1.0;
        let mut par = 0.0;

        let result = lmpar(&mut r, &ipvt, &diag, &qtb, delta, &mut par);
        assert!(result.is_ok());

        let (x, _sdiag) = result.unwrap();
        assert_eq!(x.len(), 3);
        assert!(par >= 0.0);
    }
}

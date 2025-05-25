use log::warn;
use nalgebra::{DMatrix, DVector, LU, SVD, stack};
/// In mathematics, the Rouché–Capelli theorem is a fundamental result in linear algebra. It gives a necessary and sufficient condition for a system of
/// linear equations to have a solution. The theorem is a consequence of the rank theorem. Statement
/// The Rouché–Capelli theorem states that a system of linear equations Ax = b has a solution if and only if the rank of A is equal to the rank of [A b].
///Here, A is the coefficient matrix, x is the vector of unknowns, and b is the vector of constants.
/// More formally, let A be an m × n matrix, and let b be an m × 1 vector. Then the system of linear equations Ax = b has a solution if and only if the rank of
///A is equal to the rank of [A b]. This theorem is often referred to as the Rouché–Capelli theorem, named after Eugène Rouché and Giuseppe Peano, who first proved it.
pub fn Rouche_Capelli_theorem(A: &DMatrix<f64>, b: &DVector<f64>) -> bool {
    // (A: DMatrix<f64>, b: DVector<f64>)

    let Ab: DMatrix<f64> = stack![A, b];
    let eps = std::f64::EPSILON; // tolerance for rank calculation
    let rank_A = A.rank(eps);
    let rank_Ab = Ab.rank(eps);

    let result = rank_A == rank_Ab;
    if !result {
        warn!(
            "The system has no solution. rank(A) = {} != rank([A b]) = {}",
            rank_A, rank_Ab
        );
        println!(
            "The system has no solution. rank(A) = {} != rank([A b]) = {}",
            rank_A, rank_Ab
        );
    }
    return result;
}
pub fn is_singular(A: DMatrix<f64>, epsilon: f64) -> bool {
    // A system of linear equations is said to be poorly conditioned if the solution is sensitive to small changes in the input data.
    // The condition number of a matrix is a measure of the sensitivity of its solution to small perturbations in the input data.
    // The condition number of a matrix is defined as the ratio of the largest singular value of the matrix to the smallest singular value of the matrix.
    let det = A.determinant();
    let is_singular = det.abs() < epsilon; // tolerance for singularity check
    if is_singular {
        warn!("Matrix is singular. Determinant = {:.8}", det);
        println!("Matrix is singular. Determinant = {:.8}", det);
    }
    is_singular
}
pub fn poorly_conditioned(A: DMatrix<f64>, threshold: f64) -> bool {
    // A system of linear equations is said to be poorly conditioned if the solution is sensitive to small changes in the input data.
    // The condition number of a matrix is a measure of the sensitivity of its solution to small perturbations in the input data.
    // The condition number of a matrix is defined as the ratio of the largest singular value of the matrix to the smallest singular value of the matrix.
    let singular_values = A.singular_values();
    let max_sigma = singular_values[0];
    let min_sigma = singular_values[singular_values.len() - 1];
    let condition_number = max_sigma / min_sigma;

    let poorly_conditioned = condition_number > threshold; // tolerance for condition number check
    if poorly_conditioned {
        warn!(
            "The system of linear equations is poorly conditioned. Condition number = {:.2}",
            condition_number
        );
        println!(
            "The system of linear equations is poorly conditioned. Condition number = {:.2}",
            condition_number
        );
    }
    poorly_conditioned
}
/// Singular Value Decomposition (SVD) is a factorization of a matrix A into the product of three matrices:
/// A = U * Σ * V^T
/// where U and V are orthogonal matrices, and Σ is a diagonal matrix containing the singular values of A.
/// The singular values of a matrix are its eigenvalues, and the singular vectors of a matrix are its eigenvectors.
/// The singular values of a matrix are used to determine the condition number of the matrix, which is a measure of the sensitivity of the solution

pub fn SVD_diagnostics(A: DMatrix<f64>) -> bool {
    let svd = A.svd(true, true);
    let singular_values = svd.singular_values;
    let u = svd.u.expect("U matrix not found");
    let v = svd.v_t.expect("V^T matrix not found");
    println!("Singular values: {:?}", singular_values);
    println!("U matrix: {:?}", u);
    println!("V^T matrix: {:?}", v);
    true
}
pub fn linear_system_diagnostics(A: DMatrix<f64>, b: DVector<f64>, threshold: f64) -> bool {
    // Check if the system of linear equations has a solution using the Rouché-Capelli theorem and the condition number of the coefficient matrix.
    if Rouche_Capelli_theorem(&A.clone(), &b) {
        // If the system has a solution, check if the coefficient matrix is poorly conditioned.
        if poorly_conditioned(A, threshold) {
            warn!(
                "The system of linear equations has a solution and the coefficient matrix is poorly conditioned."
            );

            false
        } else {
            true
        }
    } else {
        // If the system does not have a solution, check if the coefficient matrix is poorly conditioned.
        if poorly_conditioned(A, threshold) {
            warn!(
                "The system of linear equations does not have a solution and the coefficient matrix is poorly conditioned."
            );
            false
        } else {
            warn!(
                "The system of linear equations does not have a solution and the coefficient matrix is not poorly conditioned."
            );
            false
        }
    }
}
/// famous example of ill-conditioned matrix
fn hilbert_matrix(n: usize) -> DMatrix<f64> {
    let mut A = DMatrix::zeros(n, n);
    for i in 1..n + 1 {
        for j in 1..n + 1 {
            A[(i - 1, j - 1)] = 1.0 / (i as f64 + j as f64 - 1.0);
        }
    }
    A
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_Rouche_Capelli_theorem() {
        // example https://en.wikipedia.org/wiki/Rouché–Capelli_theorem
        let A = DMatrix::from_vec(3, 3, vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0]);
        println!("{}", A);
        let b = DVector::from_vec(vec![3.0, 1.0, 5.0]);
        let test = Rouche_Capelli_theorem(&A, &b);
        assert_eq!(test, false);
    }
    #[test]
    fn test_poorly_conditioned() {
        let A = DMatrix::from_vec(2, 2, vec![1.0, 1.0, 1.00001, 1.0]);
        let threshold = 1e5;
        assert_eq!(poorly_conditioned(A, threshold), true);
    }
    #[test]
    fn test_poorly_conditioned_hilbert() {
        let A = hilbert_matrix(6);
        println!("{}", A);
        let threshold = 1e5;
        assert_eq!(poorly_conditioned(A, threshold), true);
    }
    #[test]
    fn test_is_singular() {
        let A = DMatrix::from_vec(2, 2, vec![1.0, 1.0, 1.00001, 1.0]);
        let epsilon = 1e-4;
        assert_eq!(is_singular(A, epsilon), true);
    }

    #[test]
    fn test_SVD_diagnostics() {
        let A = DMatrix::from_vec(2, 2, vec![1.0, 1.0, 1.00001, 1.0]);
        assert_eq!(SVD_diagnostics(A), true);
    }

    #[test]
    fn try_solve_ill_conditioned_system() {
        let A = hilbert_matrix(8);
        let b = DVector::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let is_it_solvable = Rouche_Capelli_theorem(&A, &b);
        assert_eq!(is_it_solvable, true);
        let lu = A.clone().lu();
        println!("{:?}", lu);
        let _ = lu.solve(&b).expect("Failed to solve linear system");
        let inv = A.try_inverse().expect("Failed to invert matrix");
        let _ = inv * &b;
    }
    #[test]
    fn try_solve_ill_conditioned_system2() {
        let A = DMatrix::from_vec(2, 2, vec![-1.732720588235294, 1.0, 6.26727941176470, -1.0]);
        let b = DVector::from_vec(vec![
            0.5702779655060555,
            -0.00000000000000011102230246251565,
        ]);
        println!("{}", A);
        let is_it_solvable = Rouche_Capelli_theorem(&A, &b);
        assert_eq!(is_it_solvable, true);
        let lu = A.clone().lu();
        println!("{:?}", lu);
        let res1 = lu.solve(&b).expect("Failed to solve linear system");
        let inv = A.try_inverse().expect("Failed to invert matrix");
        let res2 = inv * &b;
        println!("{}", res1);
        println!("{}", res2);
    }
}

/*
2. Use Stable Numerical Methods
Some methods are more stable and less sensitive to ill-conditioning:

QR Decomposition: Decomposes AAA into an orthogonal matrix QQQ and an upper triangular matrix RRR.
Advantage: More stable than directly inverting AAA.

Singular Value Decomposition (SVD):
Decomposes AAA into UΣVTU \Sigma V^TUΣV
T
 , where Σ\SigmaΣ contains singular values.
Advantage: Can identify and handle the small singular values that cause instability.

3. Regularization Techniques
Tikhonov Regularization (Ridge Regression):
Adds a small value λ\lambdaλ to the diagonal elements to stabilize the solution:
(ATA+λI)x=ATb(A^T A + \lambda I) x = A^T b
(A
T
 A+λI)x=A
T
 b

Advantage: Reduces the effect of small singular values.
4. Preconditioning
Transform the system into an equivalent form that is better conditioned before solving.
Example: Find a matrix PPP such that PAP APA is better conditioned, then solve PAx=PbP A x = P bPAx=Pb.
5. Increase Precision
Use higher-precision arithmetic (like double or quadruple precision) to reduce numerical errors.
6. Verify and Validate
Check the residual ∥Ax−b∥\|A x - b\|∥Ax−b∥ to see how accurate your solution is.
Use multiple methods to compare solutions for consistency.

*/

/*

This is a step-by-step solution to addressing the challenges of solving linear systems with ill-conditioned matrices.


Step 1: Understand the Problem
When solving a linear system
 where
 is an ill-conditioned matrix, small errors in
 or
 (e.g., due to measurement noise, rounding errors during computation, or representation errors in floating-point arithmetic) are magnified enormously in the solution
. This means that direct methods like Gaussian elimination can produce highly inaccurate results. The primary goal is to obtain a stable and accurate solution despite the matrix's sensitivity.


Step 2: Use Iterative Refinement (for direct methods)
If using a direct method (like LU decomposition), subsequent iterative refinement can improve accuracy:



Solve: Compute an approximate solution
 using a direct method (e.g.,
).

Calculate Residual: Compute the residual
 using higher precision arithmetic if possible.

Solve for Correction: Solve
 for the correction vector
.

Update Solution: Update the solution
.

Repeat: Repeat steps 2-4 until the residual is sufficiently small or the correction becomes negligible. This process helps to mitigate the propagation of errors from the initial solution steps.


Step 3: Employ Iterative Methods
For very large and sparse ill-conditioned systems, iterative methods are often preferred over direct methods, as they avoid explicit inversion or decomposition of the matrix. They start with an initial guess and iteratively refine it until convergence. Popular methods include:



Jacobi Method

Gauss-Seidel Method

Successive Over-Relaxation (SOR) Method

Conjugate Gradient (CG) Method (for symmetric positive definite matrices)

Bi-Conjugate Gradient Stabilized (BiCGSTAB) or Generalized Minimal Residual (GMRES) (for non-symmetric matrices)


However, iterative methods can converge very slowly or not at all for ill-conditioned matrices without proper preconditioning.


Step 4: Use Preconditioning (Crucial for Iterative Methods)
Preconditioning transforms the original system
 into an equivalent system
 (or similar) that has a significantly smaller condition number, making it easier for iterative methods to converge. The matrix
 is called the preconditioner. Its ideal properties are:



 should approximate the identity matrix (or be well-conditioned).

 should be easily invertible.
Common preconditioning techniques include:



Jacobi Preconditioner:

Incomplete LU (ILU) Factorization: Approximates
 by dropping "small" fill-in entries.

Successive Over-Relaxation (SOR) Preconditioner

Multigrid Methods: Highly effective for certain types of problems (e.g., those arising from PDEs).


Step 5: Regularization Techniques
When the matrix is severely ill-conditioned or even singular, regularization techniques are used to find a stable "approximate" solution by modifying the original problem. These methods introduce a penalty term to stabilize the solution. The most common is Tikhonov regularization:

where
 is the regularization parameter. This modifies the matrix
 by adding diagonal entries, making it better-conditioned and guaranteeing a unique solution even if
 is singular. The choice of
 is critical; too small, and it doesn't help conditioning; too large, and it distorts the solution.


Step 6: Re-formulate the Problem (if possible)
Sometimes, the ill-conditioning stems from the original mathematical formulation of the problem. If feasible, consider alternative formulations or choose different basis functions if the problem involves approximation. For instance, using orthogonal polynomials can sometimes lead to better-conditioned matrices than standard monomial bases.


Step 7: Increase Precision
Using higher precision floating-point arithmetic (e.g., double instead of float, or arbitrary precision libraries) can mitigate the effect of rounding errors, but it's often a last resort due to increased computational cost and memory usage. It addresses the symptom (numerical instability) rather than the root cause (inherent sensitivity of the problem).


Final Answer
To solve linear systems with ill-conditioned matrices, one should avoid naive direct methods. Instead, employ strategies such as:



Iterative Refinement (for direct methods).

Iterative Methods (e.g., CG, GMRES).

Preconditioning (essential for iterative methods' convergence).

Regularization Techniques (e.g., Tikhonov regularization) to stabilize the problem.

Re-formulate the underlying problem if its structure causes the ill-conditioning.

Increase numerical precision as a last resort.



Key Concept & Explanation
Robust Solutions for Ill-Conditioned Systems: Solving linear systems with ill-conditioned matrices means dealing with inherent sensitivity to perturbations.
 The key is to employ techniques that either transform the problem into a better-conditioned one (preconditioning, regularization),
  iteratively refine the solution to reduce accumulated errors (iterative methods, iterative refinement), or re-formulate the problem itself to avoid
  the sensitivity. These methods aim to compute a stable and accurate solution despite the challenging nature of the matrix.





*/

/*
3. Check Condition Number: Calculate the condition number $ \kappa(A) $ of the matrix $ A $:
   $$
   \kappa(A) = \|A\| \cdot \|A^{-1}\|
   $$
   If $ \kappa(A) $ is large (typically $ > 10^3 $), the matrix is considered ill-conditioned.
4. Use Regularization: To stabilize the solution, apply regularization techniques such as Tikhonov regularization:
   $$
   \min_x \|Ax - b\|^2 + \lambda \|x\|^2
   $$
   where $ \lambda $ is a small positive constant.
5. Use Pseudoinverse: If $ A $ is singular or nearly singular, use the Moore-Penrose pseudoinverse $ A^+ $:
   $$
   x = A^+b
   $$
   This can provide a least-squares solution.
6. Iterative Methods: Consider iterative methods like the Conjugate Gradient method or GMRES, which can be more stable for ill-conditioned systems.
7. Preconditioning: Apply a preconditioner $ M $ to transform the system into a better-conditioned one:
   $$
   M^{-1}Ax = M^{-1}b
   $$
   Choose $ M $ such that $ M^{-1}A $ has a lower condition number.
8. Check Solution Stability: After obtaining a solution, check its sensitivity to perturbations in $ b $ or $ A $ to ensure stability.
By following these steps, you can effectively tackle linear systems with ill-conditioned matrices.

*/

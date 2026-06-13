// local copy of lstsq crate to avoid problems with nalgebra's different versions.

use nalgebra::allocator::Allocator;
use nalgebra::base::{OMatrix, OVector};
use nalgebra::dimension::{Dim, DimDiff, DimMin, DimMinimum, DimSub, U1};
use nalgebra::{DefaultAllocator, RealField};

/// Results of [lstsq]
pub struct Lstsq<R: RealField, N: Dim>
where
    DefaultAllocator: Allocator<N>,
{
    /// Least-squares solution.
    ///
    /// This is the variable `x` that approximatively solves the equation `a * x = b`.
    pub solution: OVector<R, N>,
    /// Sums of squared residuals: Squared Euclidean 2-norm.
    pub residuals: R,
    /// Rank of matrix `a`.
    pub rank: usize,
}

/// Return the least-squares solution to a linear matrix equation.
///
/// Computes the vector x that approximatively solves the equation `a * x = b`.
/// Usage is maximally compatible with Python's `numpy.linalg.lstsq`.
///
/// Arguments:
///
/// - `a`: "Coefficient" matrix (shape: M rows, N columns)
/// - `b`: Ordinate or “dependent variable” values (shape: M dimensional)
/// - `epsilon`: singular values less than this are assumed to be zero.
///
/// Returns:
/// - `Result<`[Lstsq]`,&'static str>`
///
/// See the module level documentation for example of usage.
pub fn lstsq<R, M, N>(
    a: &OMatrix<R, M, N>,
    b: &OVector<R, M>,
    epsilon: R,
) -> Result<Lstsq<R, N>, &'static str>
where
    R: RealField,
    M: DimMin<N>,
    N: Dim,
    DimMinimum<M, N>: DimSub<U1>, // for Bidiagonal.
    DefaultAllocator: Allocator<M, N>
        + Allocator<N>
        + Allocator<M>
        + Allocator<DimDiff<DimMinimum<M, N>, U1>>
        + Allocator<DimMinimum<M, N>, N>
        + Allocator<M, DimMinimum<M, N>>
        + Allocator<DimMinimum<M, N>>,
{
    // calculate solution with epsilon
    let svd = nalgebra::linalg::SVD::new(a.clone(), true, true);
    let solution = svd.solve(b, epsilon.clone())?;

    // calculate residuals
    let model: OVector<R, M> = a * &solution;
    let l1: OVector<R, M> = model - b;
    let residuals: R = l1.dot(&l1);

    // calculate rank with epsilon
    let rank = svd.rank(epsilon);

    Ok(Lstsq {
        solution,
        residuals,
        rank,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use na::{OMatrix, OVector, RealField, U2};
    use nalgebra as na;

    fn check_residuals<R: RealField + Copy>(epsilon: R) {
        /*
        import numpy as np
        A = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]])
        b = np.array([2.5, 4.4, 6.6, 8.5])
        x,residuals,rank,s = np.linalg.lstsq(A,b)
        */
        let a: Vec<R> = vec![1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0]
            .into_iter()
            .map(na::convert)
            .collect();

        let a = OMatrix::<R, na::Dyn, U2>::from_row_slice(&a);

        let b_data: Vec<R> = vec![2.5, 4.4, 6.6, 8.5]
            .into_iter()
            .map(na::convert)
            .collect();
        let b = OVector::<R, na::Dyn>::from_row_slice(&b_data);

        let results = lstsq(&a, &b, R::default_epsilon()).unwrap();
        assert_eq!(results.solution.nrows(), 2);
        approx::assert_relative_eq!(results.solution[0], na::convert(2.02), epsilon = epsilon);
        approx::assert_relative_eq!(results.solution[1], na::convert(0.45), epsilon = epsilon);
        approx::assert_relative_eq!(results.residuals, na::convert(0.018), epsilon = epsilon);
        assert_eq!(results.rank, 2);
    }

    #[test]
    fn test_residuals_f64() {
        check_residuals::<f64>(1e-14)
    }

    #[test]
    fn test_residuals_f32() {
        check_residuals::<f32>(1e-5)
    }
}

//! Stabilized bi-conjugate gradient solver for solving Ax = b with x unknown. Suitable for non-symmetric matrices.
//! A simple, sparse-sparse, serial, un-preconditioned implementation.

//!
//! # Example
//! ```rust
//! use sprs::{CsMatI, CsVecI};
//! use sprs::linalg::bicgstab::BiCGSTAB;
//!
//! let a = CsMatI::new_csc(
//!     (4, 4),
//!     vec![0, 2, 4, 6, 8],
//!     vec![0, 3, 1, 2, 1, 2, 0, 3],
//!     vec![1.0, 2., 21., 6., 6., 2., 2., 8.],
//! );
//!
//! // Solve Ax=b
//! let tol = 1e-60;
//! let max_iter = 50;
//! let b = CsVecI::new(4, vec![0, 1, 2, 3], vec![1.0; 4]);
//! let x0 = CsVecI::new(4, vec![0, 1, 2, 3], vec![1.0, 1.0, 1.0, 1.0]);
//!
//! let res = BiCGSTAB::<'_, f64, _, _>::solve(
//!     a.view(),
//!     x0.view(),
//!     b.view(),
//!     tol,
//!     max_iter,
//! )
//! .unwrap();
//! let b_recovered = &a * &res.x();
//!
//! println!("Iteration count {:?}", res.iteration_count());
//! println!("Soft restart count {:?}", res.soft_restart_count());
//! println!("Hard restart count {:?}", res.hard_restart_count());
//!
//! // Make sure the solved values match expectation
//! for (input, output) in
//!     b.to_dense().iter().zip(b_recovered.to_dense().iter())
//! {
//!     assert!(
//!         (1.0 - input / output).abs() < tol,
//!         "Solved output did not match input"
//!     );
//! }
//! ```
//!

use log::info;
use num_traits::One;
use sprs::indexing::SpIndex;
use sprs::{CsMatViewI, CsVecI, CsVecViewI};
/// Stabilized bi-conjugate gradient solver
#[derive(Debug)]
pub struct BiCGSTAB<'a, T, I: SpIndex, Iptr: SpIndex> {
    // Configuration
    iteration_count: usize,
    soft_restart_threshold: T,
    soft_restart_count: usize,
    hard_restart_count: usize,
    // Problem statement: err = a * x - b
    err: T,
    a: CsMatViewI<'a, T, I, Iptr>,
    b: CsVecViewI<'a, T, I>,
    x: CsVecI<T, I>,
    // Intermediate vectors
    r: CsVecI<T, I>,
    rhat: CsVecI<T, I>, // Arbitrary w/ dot(rhat, r) != 0
    p: CsVecI<T, I>,
    // Intermediate scalars
    rho: T,
    // New field to store nonzero indexes
    nonzero_indexes: Vec<I>,
}

macro_rules! bicgstab_impl {
    ($T: ty) => {
        impl<'a, I: SpIndex, Iptr: SpIndex> BiCGSTAB<'a, $T, I, Iptr> {
            /// Initialize the solver with a fresh error estimate
            pub fn new(
                a: CsMatViewI<'a, $T, I, Iptr>,
                x0: CsVecViewI<'a, $T, I>,
                b: CsVecViewI<'a, $T, I>,
            ) -> Self {
                let r = &b - &(&a.view() * &x0.view()).view();
                let rhat = r.to_owned();
                let p = r.to_owned();
                let err = (&r).l2_norm();
                let rho = err * err;
                let x = x0.to_owned();
                Self {
                    iteration_count: 0,
                    soft_restart_threshold: 0.1 * <$T>::one(), // A sensible default
                    soft_restart_count: 0,
                    hard_restart_count: 0,
                    err,
                    a,
                    b,
                    x,
                    r,
                    rhat,
                    p,
                    rho,
                    nonzero_indexes: Vec::new(), // Initialize the new field
                }
            }

            /// Attempt to solve the system to the given tolerance on normed error,
            /// returning an error if convergence is not achieved within the given
            /// number of iterations.
            pub fn solve(
                a: CsMatViewI<'a, $T, I, Iptr>,
                x0: CsVecViewI<'a, $T, I>,
                b: CsVecViewI<'a, $T, I>,
                tol: $T,
                max_iter: usize,
            ) -> Result<Box<BiCGSTAB<'a, $T, I, Iptr>>, Box<BiCGSTAB<'a, $T, I, Iptr>>> {
                let mut solver = Self::new(a, x0, b);
                for _ in 0..max_iter {
                    solver.step();
                    if solver.err() < tol {
                        // Check true error, which may not match the running error estimate
                        // and either continue iterations or return depending on result.
                        solver.hard_restart();
                        if solver.err() < tol {
                            // filter

                            let mut x: CsVecI<$T, I> = solver.x().to_owned();
                            let mut filtered_x: CsVecI<$T, I> = CsVecI::empty(x.dim());
                            //   filtered_x.map_inplace(|x| if x>=&tol { *x} else {}    );

                            for (i, value) in x.iter_mut() {
                                //  info!("value {}", &value);
                                if (*value).abs() >= tol {
                                    filtered_x.append(i, *value);

                                    info!("Filtering {}", i);
                                }
                            }

                            /*
                            let mut filtered_x:  CsVecI<$T, I> = CsVecI::zero(solver.x().dim());

                            for (i, value) in solver.x().iter() {
                                if (*value).abs() > tol {
                                    filtered_x.push((i as I, *value));
                                }
                            }
                            let filtered_x: CsVecI<T, I> = CsVecI::from_vec(solver.x().dim(), filtered_x);
                            */
                            // Save filtered x into the x field of the struct
                            solver.x = filtered_x;

                            // Save nonzero indexes of x
                            solver.nonzero_indexes = solver.x.indices().iter().cloned().collect();
                            return Ok(Box::new(solver));
                        }
                    }
                }

                // If we ran past our iteration limit, error, but still return results
                Err(Box::new(solver))
            }

            /// Reset the reference direction `rhat` to be equal to `r`
            /// to prevent a singularity in `1 / rho`.
            pub fn soft_restart(&mut self) {
                self.soft_restart_count += 1;
                self.rhat = self.r.to_owned();
                self.rho = self.err * self.err; // Shortcut to (&self.r).squared_l2_norm();
                self.p = self.r.to_owned();
            }

            /// Recalculate the error vector from scratch using `a` and `b`.
            pub fn hard_restart(&mut self) {
                self.hard_restart_count += 1;
                // Recalculate true error
                self.r = &self.b - &(&self.a.view() * &self.x.view()).view();
                self.err = (&self.r).l2_norm();
                // Recalculate reference directions
                self.soft_restart();
                self.soft_restart_count -= 1; // Don't increment soft restart count for hard restarts
            }

            pub fn step(&mut self) -> $T {
                self.iteration_count += 1;

                // Gradient descent step
                let v = &self.a.view() * &self.p.view();
                let alpha = self.rho / ((&self.rhat).dot(&v));
                let h = &self.x + &self.p.map(|x| x * alpha); // latest estimate of `x`

                // Conjugate direction step
                let s = &self.r - &v.map(|x| x * alpha); // s = A*h
                let t = &self.a.view() * &s.view();
                let omega = t.dot(&s) / &t.squared_l2_norm();
                self.x = &h.view() + &s.map(|x| omega * x);

                // Check error
                self.r = &s - &t.map(|x| x * omega);
                self.err = (&self.r).l2_norm();

                // Prep for next pass
                let rho_prev = self.rho;
                self.rho = (&self.rhat).dot(&self.r);

                // Soft-restart if `rhat` is becoming perpendicular to `r`.
                if self.rho.abs() / (self.err * self.err) < self.soft_restart_threshold {
                    self.soft_restart();
                } else {
                    let beta = (self.rho / rho_prev) * (alpha / omega);
                    self.p = &self.r + (&self.p - &v.map(|x| x * omega)).map(|x| x * beta);
                }

                self.err
            }

            /// Set the minimum value of `rho` to trigger a soft restart
            pub fn with_restart_threshold(mut self, thresh: $T) -> Self {
                self.soft_restart_threshold = thresh;
                self
            }

            /// Iteration number
            pub fn iteration_count(&self) -> usize {
                self.iteration_count
            }

            /// The minimum value of `rho` to trigger a soft restart
            pub fn soft_restart_threshold(&self) -> $T {
                self.soft_restart_threshold
            }

            /// Number of soft restarts that have been done so far
            pub fn soft_restart_count(&self) -> usize {
                self.soft_restart_count
            }

            /// Number of soft restarts that have been done so far
            pub fn hard_restart_count(&self) -> usize {
                self.hard_restart_count
            }

            /// Latest estimate of normed error
            pub fn err(&self) -> $T {
                self.err
            }

            /// `dot(rhat, r)`, a measure of how well-conditioned the
            /// update to the gradient descent step direction will be.
            pub fn rho(&self) -> $T {
                self.rho
            }

            /// The problem matrix
            pub fn a(&self) -> CsMatViewI<'_, $T, I, Iptr> {
                self.a.view()
            }

            /// The latest solution vector
            pub fn x(&self) -> CsVecViewI<'_, $T, I> {
                self.x.view()
            }

            pub fn nonzero_indexes(&self) -> Vec<I> {
                self.nonzero_indexes.clone()
            }
            /// The objective vector
            pub fn b(&self) -> CsVecViewI<'_, $T, I> {
                self.b.view()
            }

            /// Latest residual error vector
            pub fn r(&self) -> CsVecViewI<'_, $T, I> {
                self.r.view()
            }

            /// Latest reference direction.
            /// `rhat` is arbitrary w/ dot(rhat, r) != 0,
            /// and is reset parallel to `r` when needed to avoid
            /// `1 / rho` becoming singular.
            pub fn rhat(&self) -> CsVecViewI<'_, $T, I> {
                self.rhat.view()
            }

            /// Gradient descent step direction, unscaled
            pub fn p(&self) -> CsVecViewI<'_, $T, I> {
                self.p.view()
            }
        }
    };
}

bicgstab_impl!(f64);
bicgstab_impl!(f32);

#[cfg(test)]
mod test {
    use super::*;
    use sprs::CsMatI;

    #[test]
    fn test_bicgstab_f32() {
        let a = CsMatI::new_csc(
            (4, 4),
            vec![0, 2, 4, 6, 8],
            vec![0, 3, 1, 2, 1, 2, 0, 3],
            vec![1.0, 2., 21., 6., 6., 2., 2., 8.],
        );

        // Solve Ax=b
        let tol = 1e-5;
        let max_iter = 50;
        let b = CsVecI::new(4, vec![0, 1, 2, 3], vec![1.0; 4]);
        let x0 = CsVecI::new(4, vec![0, 1, 2, 3], vec![1.0, 1.0, 1.0, 1.0]);

        let res =
            BiCGSTAB::<'_, f32, _, _>::solve(a.view(), x0.view(), b.view(), tol, max_iter).unwrap();
        let b_recovered = &a * &res.x();
        info!("result = {:?}", res.x());
        info!("nonzero = {:?}", res.nonzero_indexes);
        info!("Iteration count {:?}", res.iteration_count());
        info!("Soft restart count {:?}", res.soft_restart_count());
        info!("Hard restart count {:?}", res.hard_restart_count());

        // Make sure the solved values match expectation
        for (input, output) in b.to_dense().iter().zip(b_recovered.to_dense().iter()) {
            assert!(
                (1.0 - input / output).abs() < tol,
                "Solved output did not match input"
            );
        }
    }

    #[test]
    fn test_bicgstab_f64() {
        let a = CsMatI::new_csc(
            (4, 4),
            vec![0, 2, 4, 6, 8],
            vec![0, 3, 1, 2, 1, 2, 0, 3],
            vec![1.0, 2., 21., 6., 6., 2., 2., 8.],
        );

        // Solve Ax=b
        let tol = 1e-5;
        let max_iter = 50;
        let b = CsVecI::new(4, vec![0, 1, 2, 3], vec![1.0; 4]);
        let x0 = CsVecI::new(4, vec![0, 1, 2, 3], vec![1.0, 1.0, 1.0, 1.0]);

        let res =
            BiCGSTAB::<'_, f64, _, _>::solve(a.view(), x0.view(), b.view(), tol, max_iter).unwrap();
        let b_recovered = &a * &res.x();
        info!("result = {:?}", res.x());
        info!("nonzero = {:?}", res.nonzero_indexes);
        info!("Iteration count {:?}", res.iteration_count());
        info!("Soft restart count {:?}", res.soft_restart_count());
        info!("Hard restart count {:?}", res.hard_restart_count());

        // Make sure the solved values match expectation
        for (input, output) in b.to_dense().iter().zip(b_recovered.to_dense().iter()) {
            assert!(
                (1.0 - input / output).abs() < tol,
                "Solved output did not match input"
            );
        }
    }
}

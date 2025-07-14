#![allow(unexpected_cfgs)]

use crate::numerical::optimization::LM_optimization::MINPACK_COMPAT;
use nalgebra::{Dim, RealField, U1, Vector, convert, storage::Storage};
use num_traits::float::Float; // MINPACK_COMPAT
// mod derivest;

cfg_if::cfg_if! {
    if #[cfg(feature = "RUSTC_IS_NIGHTLY")] {
        pub use core::intrinsics::{likely, unlikely};
    } else {
        #[inline]
        pub fn likely(b: bool) -> bool {
            b
        }

        #[inline]
        pub fn unlikely(b: bool) -> bool {
            b
        }
    }
}

#[inline]
#[allow(clippy::unreadable_literal)]
pub(crate) fn epsmch<F: RealField>() -> F {
    if MINPACK_COMPAT {
        convert(2.22044604926e-16f64)
    } else {
        F::default_epsilon()
    }
}

#[inline]
#[allow(clippy::unreadable_literal)]
pub(crate) fn giant<F: Float>() -> F {
    if MINPACK_COMPAT {
        F::from(1.79769313485e+308f64).unwrap()
    } else {
        F::max_value()
    }
}

#[inline]
#[allow(clippy::unreadable_literal)]
pub(crate) fn dwarf<F: Float>() -> F {
    if MINPACK_COMPAT {
        F::from(2.22507385852e-308f64).unwrap()
    } else {
        F::min_positive_value()
    }
}

#[inline]
pub(crate) fn enorm<F, N, VS>(v: &Vector<F, N, VS>) -> F
where
    F: nalgebra::RealField + Float + Copy,
    N: Dim,
    VS: Storage<F, N, U1>,
{
    let mut s1 = F::zero();
    let mut s2 = F::zero();
    let mut s3 = F::zero();
    let mut x1max = F::zero();
    let mut x3max = F::zero();
    let agiant = if MINPACK_COMPAT {
        convert(1.304e19f64)
    } else {
        Float::sqrt(giant::<F>())
    } / convert(v.nrows() as f64);
    let rdwarf = if MINPACK_COMPAT {
        convert(3.834e-20f64)
    } else {
        Float::sqrt(dwarf())
    };
    for xi in v.iter() {
        let xabs = xi.abs();
        if unlikely(xabs.is_nan()) {
            return xabs;
        }
        if unlikely(xabs >= agiant || xabs <= rdwarf) {
            if xabs > rdwarf {
                // sum for large components
                if xabs > x1max {
                    s1 = F::one() + s1 * Float::powi(x1max / xabs, 2);
                    x1max = xabs;
                } else {
                    s1 += Float::powi(xabs / x1max, 2);
                }
            } else {
                // sum for small components
                if xabs > x3max {
                    s3 = F::one() + s3 * Float::powi(x3max / xabs, 2);
                    x3max = xabs;
                } else if xabs != F::zero() {
                    s3 += Float::powi(xabs / x3max, 2);
                }
            }
        } else {
            s2 += xabs * xabs;
        }
    }

    if unlikely(!s1.is_zero()) {
        x1max * Float::sqrt(s1 + (s2 / x1max) / x1max)
    } else if likely(!s2.is_zero()) {
        Float::sqrt(if likely(s2 >= x3max) {
            s2 * (F::one() + (x3max / s2) * (x3max * s3))
        } else {
            x3max * ((s2 / x3max) + (x3max * s3))
        })
    } else {
        x3max * Float::sqrt(s3)
    }
}

#[inline]
/// Dot product between two vectors
pub(crate) fn dot<F, N, AS, BS>(a: &Vector<F, N, AS>, b: &Vector<F, N, BS>) -> F
where
    F: nalgebra::RealField + Copy,
    N: Dim,
    AS: Storage<F, N, U1>,
    BS: Storage<F, N, U1>,
{
    // To achieve floating point equality with MINPACK
    // the dot product implementation from nalgebra must not
    // be used.
    let mut dot = F::zero();
    for (x, y) in a.iter().zip(b.iter()) {
        dot += *x * *y;
    }
    dot
}

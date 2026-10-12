use super::bvp_sparse_lu::{ReusableSparseLu, ReusableSparseLuError};
use faer::sparse::{SparseColMat, Triplet};

fn matrix(scale: f64, include_corner: bool) -> SparseColMat<usize, f64> {
    let mut entries = vec![
        Triplet::new(0, 0, 4.0 * scale),
        Triplet::new(1, 0, 2.0 * scale),
        Triplet::new(0, 1, 1.0 * scale),
        Triplet::new(1, 1, 5.0 * scale),
        Triplet::new(2, 1, 1.0 * scale),
        Triplet::new(2, 2, 3.0 * scale),
    ];
    if include_corner {
        entries.push(Triplet::new(0, 2, 0.25 * scale));
    }
    SparseColMat::try_new_from_triplets(3, 3, &entries).expect("test CSC matrix should assemble")
}

#[test]
fn repeated_numeric_factorization_reuses_pattern_and_solves_in_place() {
    let first = matrix(1.0, false);
    let second = matrix(2.0, false);
    let mut lu = ReusableSparseLu::new(3);

    assert!(lu.prepare_symbolic(first.as_ref()).unwrap());
    assert!(!lu.prepare_symbolic(second.as_ref()).unwrap());
    lu.factor_numeric(first.as_ref()).unwrap();
    let mut rhs = [6.0, 12.0, -1.0];
    lu.solve_in_place(&mut rhs).unwrap();
    for (actual, expected) in rhs.into_iter().zip([1.0, 2.0, -1.0]) {
        assert!((actual - expected).abs() < 1e-13);
    }

    lu.factor_numeric(second.as_ref()).unwrap();
    let mut scaled_rhs = [12.0, 24.0, -2.0];
    lu.solve_in_place(&mut scaled_rhs).unwrap();
    for (actual, expected) in scaled_rhs.into_iter().zip([1.0, 2.0, -1.0]) {
        assert!((actual - expected).abs() < 1e-13);
    }
}

#[test]
fn changed_csc_pattern_requires_new_symbolic_analysis() {
    let first = matrix(1.0, false);
    let changed = matrix(1.0, true);
    let mut lu = ReusableSparseLu::new(3);

    assert!(lu.prepare_symbolic(first.as_ref()).unwrap());
    lu.factor_numeric(first.as_ref()).unwrap();
    assert!(lu.prepare_symbolic(changed.as_ref()).unwrap());
    assert!(matches!(
        lu.solve_in_place(&mut [1.0, 2.0, 3.0]),
        Err(ReusableSparseLuError::NotFactorized)
    ));
    lu.factor_numeric(changed.as_ref()).unwrap();
}

#[test]
fn numeric_factorization_rejects_unprepared_pattern() {
    let first = matrix(1.0, false);
    let changed = matrix(1.0, true);
    let mut lu = ReusableSparseLu::new(3);
    lu.prepare_symbolic(first.as_ref()).unwrap();

    assert!(matches!(
        lu.factor_numeric(changed.as_ref()),
        Err(ReusableSparseLuError::PatternMismatch)
    ));
}

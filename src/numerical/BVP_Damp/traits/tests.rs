#[cfg(test)]
mod y_trait_object_clone_tests {
    use super::*;

    #[test]
    fn dense_callback_view_borrows_builtin_newton_state() {
        let vector: Box<dyn VectorType> =
            Box::new(YEnum::Banded(DVector::from_vec(vec![1.0, 2.0])));
        let view = dense_slice_or_copy(&*vector);

        assert!(matches!(&view, Cow::Borrowed(_)));
        assert_eq!(view.as_ref(), &[1.0, 2.0]);
    }

    #[test]
    fn boxed_y_clone_preserves_dense_payload_without_recursion() {
        let original: Box<dyn Y> = Box::new(DVector::from_vec(vec![1.0, -2.0, 3.5]));
        let cloned = original.clone();
        let any = cloned.as_any();
        let dense = (*any)
            .downcast_ref::<DVector<f64>>()
            .expect("cloned Box<dyn Y> should keep dense payload type");

        assert_eq!(dense.as_slice(), &[1.0, -2.0, 3.5]);
    }

    #[test]
    fn boxed_y_clone_preserves_sparse_payload_without_recursion() {
        let sparse = CsVec::new(5, vec![0, 3], vec![2.0, -4.0]);
        let original: Box<dyn Y> = Box::new(sparse);
        let cloned = original.clone();
        let any = cloned.as_any();
        let sparse = (*any)
            .downcast_ref::<CsVec<f64>>()
            .expect("cloned Box<dyn Y> should keep sparse payload type");

        assert_eq!(sparse.dim(), 5);
        assert_eq!(sparse.indices(), &[0, 3]);
        assert_eq!(sparse.data(), &[2.0, -4.0]);
    }

    #[test]
    fn sparse_vector_to_dense_preserves_implicit_zero_coordinates() {
        let expected = [2.0, 0.0, 0.0, -4.0, 0.0];

        let sprs = CsVec::new(5, vec![0, 3], vec![2.0, -4.0]);
        let sprs_direct = sprs.to_DVectorType();
        let sprs_enum = YEnum::Sparse_1(sprs).to_DVectorType();
        assert_eq!(sprs_direct.as_slice(), &expected);
        assert_eq!(sprs_enum.as_slice(), &expected);

        let faer = faer_col::from_fn(5, |index| match index {
            0 => 2.0,
            3 => -4.0,
            _ => 0.0,
        });
        let faer_direct = faer.to_DVectorType();
        let faer_enum = YEnum::Sparse_3(faer).to_DVectorType();
        assert_eq!(faer_direct.as_slice(), &expected);
        assert_eq!(faer_enum.as_slice(), &expected);
    }

    #[test]
    fn sparse_matrix_adapters_preserve_shape_and_coordinates() {
        let triplets = vec![(0, 1, 2.0), (0, 1, 3.0), (2, 0, -4.0), (1, 1, 0.0)];
        let dense_values = vec![0.0; 9];

        let sprs_matrix = from_vector_to_matrix(
            "Sparse_1".to_string(),
            3,
            3,
            &dense_values,
            triplets.clone(),
        );
        let sprs_matrix = sprs_matrix
            .as_any()
            .downcast_ref::<CsMat<f64>>()
            .expect("Sparse_1 adapter should produce CsMat");
        assert_eq!(sprs_matrix.shape(), (3, 3));
        assert_eq!(sprs_matrix.get(0, 1), Some(&5.0));
        assert_eq!(sprs_matrix.get(2, 0), Some(&-4.0));
        assert_eq!(sprs_matrix.get(1, 1).copied().unwrap_or(0.0), 0.0);

        let faer_matrix =
            from_vector_to_matrix("Sparse_3".to_string(), 3, 3, &dense_values, triplets);
        let faer_matrix = faer_matrix
            .as_any()
            .downcast_ref::<SparseColMat<usize, f64>>()
            .expect("Sparse_3 adapter should produce faer SparseColMat");
        assert_eq!((faer_matrix.nrows(), faer_matrix.ncols()), (3, 3));
        assert_eq!(faer_matrix.get(0, 1), Some(&5.0));
        assert_eq!(faer_matrix.get(2, 0), Some(&-4.0));
        assert_eq!(faer_matrix.get(1, 1).copied().unwrap_or(0.0), 0.0);
    }

    #[test]
    fn finite_difference_jacobian_reports_callback_shape_mismatch() {
        let fun = FunEnum::Dense(Box::new(|_, _| DVector::from_element(1, 0.0)));
        let vec = DVector::from_vec(vec![1.0, 2.0]);

        let error = try_finite_difference_jacobian(&fun, 0.0, &vec, 1e-8)
            .expect_err("malformed finite-difference residual must be typed");
        assert_eq!(
            error,
            BvpCallbackError::OutputShapeMismatch {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn callback_try_call_captures_legacy_panic() {
        let fun = FunEnum::Dense(Box::new(|_, _| {
            panic!("callback panic for typed-boundary test")
        }));
        let vec = DVector::from_element(1, 0.0);

        let error = fun
            .try_call(0.0, &vec)
            .expect_err("callback panic must be captured by try_call");
        assert!(matches!(
            error,
            BvpCallbackError::Panic { message }
                if message.contains("callback panic for typed-boundary test")
        ));
    }

    #[test]
    fn callback_try_call_reports_builtin_backend_mismatch() {
        let fun = FunEnum::Dense(Box::new(|_, value| value.clone()));
        let sparse = CsVec::new(1, vec![0], vec![1.0]);

        let error = fun
            .try_call(0.0, &sparse)
            .expect_err("a Dense callback must reject a sparse vector explicitly");
        assert!(matches!(
            error,
            BvpCallbackError::TypeMismatch { expected, actual }
                if expected == "DVector" && actual == "Sparse_1"
        ));
    }

    #[test]
    fn jacobian_try_call_reports_builtin_backend_mismatch() {
        let mut jacobian = JacEnum::Dense(Box::new(|_, value| {
            DMatrix::identity(value.len(), value.len())
        }));
        let sparse = CsVec::new(1, vec![0], vec![1.0]);

        let error = jacobian
            .try_call(0.0, &sparse)
            .expect_err("a Dense Jacobian callback must reject a sparse vector explicitly");
        assert!(matches!(
            error,
            BvpCallbackError::TypeMismatch { expected, actual }
                if expected == "DVector" && actual == "Sparse_1"
        ));
    }

    #[test]
    fn jacobian_try_call_captures_legacy_panic() {
        let mut jacobian = JacEnum::Dense(Box::new(|_, _| {
            panic!("Jacobian panic for typed-boundary test")
        }));
        let vector = DVector::from_element(1, 0.0);

        let error = jacobian
            .try_call(0.0, &vector)
            .expect_err("Jacobian callback panic must be captured by try_call");
        assert!(matches!(
            error,
            BvpCallbackError::Panic { message }
                if message.contains("Jacobian panic for typed-boundary test")
        ));
    }

    #[test]
    fn jacobian_linear_compatibility_calls_have_typed_error_boundaries() {
        let mut jacobian = JacEnum::Dense(Box::new(|_, value| {
            DMatrix::identity(value.len(), value.len())
        }));
        let matrix: Box<dyn MatrixType> = Box::new(DMatrix::identity(2, 2));
        let rhs: Box<dyn VectorType> = Box::new(DVector::from_vec(vec![3.0, -2.0]));

        let solution = jacobian
            .try_solve_sys(matrix.as_ref(), rhs.as_ref(), 1e-12, 10)
            .expect("valid Jacobian-owned linear solve should remain available");
        assert_eq!(
            solution.to_DVectorType(),
            DVector::from_vec(vec![3.0, -2.0])
        );

        let wrong_matrix: Box<dyn MatrixType> = Box::new(DMatrix::from_element(2, 1, 1.0));
        let error = jacobian
            .try_inv(wrong_matrix.as_ref(), 1e-12, 10)
            .expect_err("legacy inverse panic must cross a typed boundary");
        assert!(matches!(
            error,
            BvpLinearSolveError::LegacyPanic { backend, .. }
                if backend == "jacobian-callback"
        ));

        let short_rhs: Box<dyn VectorType> = Box::new(DVector::from_element(1, 1.0));
        let error = jacobian
            .try_solve_sys(matrix.as_ref(), short_rhs.as_ref(), 1e-12, 10)
            .expect_err("linear dimension mismatch must be typed");
        assert!(matches!(
            error,
            BvpLinearSolveError::DimensionMismatch { .. }
        ));
    }
}

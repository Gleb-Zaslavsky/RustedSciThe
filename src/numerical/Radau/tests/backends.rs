//! Backend parity stories: AOT/Lambdify, ExprLegacy/AtomViewNative, and
//! Dense/Sparse/Banded routes.

use super::super::new::coefficients::RadauIia5;
use super::super::new::config::RadauMatrixLayout;
use super::super::new::error::RadauError;
use super::super::new::linear::{
    BandedBackend, DenseBackend, JacobianValues, LinearSystemBackend, PreparedLinearBackend,
    RadauLinearWorkspace, SparseBackend,
};
use super::super::new::prepared::RadauPreparedModel;
use super::super::new::session::RadauSession;
use super::super::new::telemetry::{RadauTelemetry, RadauTelemetryMode};

#[test]
fn dense_backend_assembles_and_solves_without_intermediate_conversion() {
    let backend = DenseBackend;
    let mut workspace = backend.create_workspace(2).unwrap();
    let coefficients = RadauIia5::new();
    let jacobian = [-1.0, 0.0, 0.0, -2.0];

    backend
        .assemble_shifted(
            JacobianValues::Dense { values: &jacobian },
            0.1,
            &coefficients,
            &mut workspace,
        )
        .unwrap();
    backend.factor(&mut workspace).unwrap();

    let mut real_rhs = [1.0, 2.0];
    backend.solve_real(&mut workspace, &mut real_rhs).unwrap();
    assert!(real_rhs.iter().all(|value| value.is_finite()));

    let mut complex_rhs_real = [1.0, 0.0];
    let mut complex_rhs_imag = [0.0, 1.0];
    backend
        .solve_complex(&mut workspace, &mut complex_rhs_real, &mut complex_rhs_imag)
        .unwrap();
    assert!(
        complex_rhs_real
            .iter()
            .chain(complex_rhs_imag.iter())
            .all(|value| value.is_finite())
    );
}

#[test]
fn banded_backend_stores_and_solves_compact_shifted_system() {
    let backend = BandedBackend { lower: 1, upper: 1 };
    let mut workspace = backend.create_workspace(8).unwrap();
    let coefficients = RadauIia5::new();
    let compact = vec![0.0; 3 * 8];

    backend
        .assemble_shifted(
            JacobianValues::Banded {
                values: &compact,
                lower: 1,
                upper: 1,
            },
            0.1,
            &coefficients,
            &mut workspace,
        )
        .unwrap();
    assert_eq!(workspace.real.len(), 24);
    backend.factor(&mut workspace).unwrap();
    let mut real_rhs = [1.0; 8];
    backend.solve_real(&mut workspace, &mut real_rhs).unwrap();
    assert!(real_rhs.iter().all(|value| value.is_finite()));
    let mut complex_rhs_real = [1.0; 8];
    let mut complex_rhs_imag = [0.5; 8];
    backend
        .solve_complex(&mut workspace, &mut complex_rhs_real, &mut complex_rhs_imag)
        .unwrap();
    assert!(
        complex_rhs_real
            .iter()
            .chain(complex_rhs_imag.iter())
            .all(|value| value.is_finite())
    );

    let dense = vec![0.0; 64];
    assert!(matches!(
        backend.assemble_shifted(
            JacobianValues::Dense { values: &dense },
            0.1,
            &coefficients,
            &mut workspace,
        ),
        Err(RadauError::UnsupportedRoute(
            super::super::new::error::RadauUnsupportedRoute::BandedValuesRequired,
        ))
    ));
}

#[test]
fn asymmetric_banded_backend_uses_lower_and_upper_sides_correctly() {
    let backend = BandedBackend { lower: 2, upper: 1 };
    let mut workspace = backend.create_workspace(4).unwrap();
    let coefficients = RadauIia5::new();
    let values = vec![
        10.0, 20.0, 30.0, 40.0, // upper = 1, diagonal column slots follow
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];

    backend
        .assemble_shifted(
            JacobianValues::Banded {
                values: &values,
                lower: 2,
                upper: 1,
            },
            0.5,
            &coefficients,
            &mut workspace,
        )
        .unwrap();

    assert!(workspace.real.iter().all(|value| value.is_finite()));
    assert_eq!(workspace.real.len(), 4 * (2 + 1 + 1));
    assert_eq!(workspace.slot(0, 2), None);
    assert_eq!(workspace.slot(3, 0), None);
    assert_eq!(workspace.real[workspace.slot(2, 0).unwrap()], -9.0);
    assert_eq!(workspace.real[workspace.slot(0, 1).unwrap()], -20.0);
    assert_eq!(
        workspace.real[workspace.slot(0, 0).unwrap()],
        coefficients.mu_real / 0.5 - 1.0
    );
}

#[test]
fn sparse_backend_keeps_csc_pattern_and_solves_shifted_system() {
    let backend = SparseBackend {
        pattern: vec![(0, 1), (1, 0)],
    };
    let mut workspace = backend.create_workspace(4).unwrap();
    let coefficients = RadauIia5::new();
    let values = [2.0, 3.0];
    let entries = [(0, 1), (1, 0)];

    backend
        .assemble_shifted(
            JacobianValues::Sparse {
                values: &values,
                entries: &entries,
            },
            0.1,
            &coefficients,
            &mut workspace,
        )
        .unwrap();
    assert_eq!(workspace.row_indices.len(), 6);
    assert_eq!(workspace.column_offsets.len(), 5);
    assert!(workspace.value_index(2, 2).is_some());
    assert!(workspace.real[workspace.value_index(2, 2).unwrap()] > 0.0);
    backend.factor(&mut workspace).unwrap();
    let mut real_rhs = [1.0; 4];
    backend.solve_real(&mut workspace, &mut real_rhs).unwrap();
    assert!(real_rhs.iter().all(|value| value.is_finite()));
    let mut complex_rhs_real = [1.0; 4];
    let mut complex_rhs_imag = [0.5; 4];
    backend
        .solve_complex(&mut workspace, &mut complex_rhs_real, &mut complex_rhs_imag)
        .unwrap();
    assert!(
        complex_rhs_real
            .iter()
            .chain(complex_rhs_imag.iter())
            .all(|value| value.is_finite())
    );
}

#[test]
fn prepared_layout_selects_one_backend_variant_and_workspace() {
    let layout = RadauMatrixLayout::Banded { lower: 1, upper: 2 };
    let backend = PreparedLinearBackend::from_layout(layout);
    assert_eq!(backend.layout(), layout);

    let config = super::super::new::config::RadauConfig {
        matrix_layout: layout,
        ..super::super::new::config::RadauConfig::default()
    };
    let prepared = RadauPreparedModel::prepare(8, &config).unwrap();
    let mut session = RadauSession::new(prepared).unwrap();
    assert_eq!(session.linear_backend().layout(), layout);
    assert!(matches!(
        session.workspace().linear,
        RadauLinearWorkspace::Banded(_)
    ));
    assert_eq!(session.workspace().jacobian.len(), 4 * 8);
}

#[test]
fn dense_backend_dispatch_reports_assembly_factor_and_solve_scopes() {
    let backend = PreparedLinearBackend::Dense(DenseBackend);
    let mut workspace = backend.create_workspace(2).unwrap();
    let coefficients = RadauIia5::new();
    let jacobian = [-1.0, 0.0, 0.0, -2.0];
    let mut telemetry = RadauTelemetry::new(RadauTelemetryMode::Timings);

    backend
        .assemble_shifted_into(
            JacobianValues::Dense { values: &jacobian },
            0.1,
            &coefficients,
            &mut workspace,
            &mut telemetry,
        )
        .unwrap();
    backend.factor_into(&mut workspace, &mut telemetry).unwrap();

    let mut real_rhs = [1.0, 2.0];
    backend
        .solve_real_into(&mut workspace, &mut real_rhs, &mut telemetry)
        .unwrap();
    let mut complex_rhs_real = [1.0, 0.0];
    let mut complex_rhs_imag = [0.0, 1.0];
    backend
        .solve_complex_into(
            &mut workspace,
            &mut complex_rhs_real,
            &mut complex_rhs_imag,
            &mut telemetry,
        )
        .unwrap();
    backend
        .invalidate_into(&mut workspace, &mut telemetry)
        .unwrap();

    assert_eq!(telemetry.counters.jacobian_assemblies, 1);
    assert_eq!(telemetry.counters.factorizations, 1);
    assert_eq!(telemetry.counters.real_solves, 1);
    assert_eq!(telemetry.counters.complex_solves, 1);
    assert_eq!(telemetry.counters.invalidations, 1);
    assert!(telemetry.timings.jacobian_assembly_ms.is_finite());
    assert!(telemetry.timings.factorization_ms.is_finite());
    assert!(telemetry.timings.real_solve_ms.is_finite());
    assert!(telemetry.timings.complex_solve_ms.is_finite());
    assert!(telemetry.timings.invalidate_ms.is_finite());
}

#[test]
fn structured_backend_dispatch_reports_native_assembly_factor_and_solve_scopes() {
    let coefficients = RadauIia5::new();

    let banded = PreparedLinearBackend::Banded(BandedBackend { lower: 1, upper: 1 });
    let mut banded_workspace = banded.create_workspace(2).unwrap();
    let mut banded_telemetry = RadauTelemetry::new(RadauTelemetryMode::Counters);
    banded
        .assemble_shifted_into(
            JacobianValues::Banded {
                values: &[0.0, 1.0, 0.0, 0.0, 2.0, 0.0],
                lower: 1,
                upper: 1,
            },
            0.1,
            &coefficients,
            &mut banded_workspace,
            &mut banded_telemetry,
        )
        .unwrap();
    banded
        .factor_into(&mut banded_workspace, &mut banded_telemetry)
        .unwrap();
    let mut banded_rhs = [1.0, 1.0];
    banded
        .solve_real_into(
            &mut banded_workspace,
            &mut banded_rhs,
            &mut banded_telemetry,
        )
        .unwrap();
    banded
        .invalidate_into(&mut banded_workspace, &mut banded_telemetry)
        .unwrap();
    assert_eq!(banded_telemetry.counters.jacobian_assemblies, 1);
    assert_eq!(banded_telemetry.counters.factorizations, 1);
    assert_eq!(banded_telemetry.counters.real_solves, 1);
    assert_eq!(banded_telemetry.counters.invalidations, 1);

    let sparse = PreparedLinearBackend::Sparse(SparseBackend {
        pattern: vec![(0, 0)],
    });
    let mut sparse_workspace = sparse.create_workspace(2).unwrap();
    let mut sparse_telemetry = RadauTelemetry::new(RadauTelemetryMode::Counters);
    sparse
        .assemble_shifted_into(
            JacobianValues::Sparse {
                values: &[1.0],
                entries: &[(0, 0)],
            },
            0.1,
            &coefficients,
            &mut sparse_workspace,
            &mut sparse_telemetry,
        )
        .unwrap();
    sparse
        .factor_into(&mut sparse_workspace, &mut sparse_telemetry)
        .unwrap();
    let mut sparse_rhs = [1.0, 1.0];
    sparse
        .solve_real_into(
            &mut sparse_workspace,
            &mut sparse_rhs,
            &mut sparse_telemetry,
        )
        .unwrap();
    sparse
        .invalidate_into(&mut sparse_workspace, &mut sparse_telemetry)
        .unwrap();
    assert_eq!(sparse_telemetry.counters.jacobian_assemblies, 1);
    assert_eq!(sparse_telemetry.counters.factorizations, 1);
    assert_eq!(sparse_telemetry.counters.real_solves, 1);
    assert_eq!(sparse_telemetry.counters.invalidations, 1);
}

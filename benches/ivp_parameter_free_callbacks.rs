//! Workload-sensitive benchmark for the parameter-free native IVP callback.
//!
//! The parameter-free AtomViewNative route skips the shared parameter adapter.
//! The parameterized AtomViewNative row is a control for the same native
//! evaluator when continuation-compatible parameter binding is required.
//!
//! Run with:
//!
//! ```text
//! cargo bench --bench ivp_parameter_free_callbacks -- --noplot
//! ```

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nalgebra::DVector;
use std::hint::black_box;

use RustedSciThe::symbolic::symbolic_engine::Expr;
use RustedSciThe::symbolic::symbolic_ivp::{
    IvpSymbolicAssemblyBackend, SymbolicIvpProblemOptions, prepare_symbolic_ivp_problem,
};

const DIMENSIONS: &[usize] = &[16, 128, 512];

fn equations(dimension: usize, parameterized: bool) -> Vec<Expr> {
    (0..dimension)
        .map(|index| {
            let left = (index > 0)
                .then(|| format!("y{}", index - 1))
                .unwrap_or_else(|| "0".to_string());
            let right = (index + 1 < dimension)
                .then(|| format!("y{}", index + 1))
                .unwrap_or_else(|| "0".to_string());
            let source = if parameterized {
                format!("-k*y{index} + d*({left} - 2*y{index} + {right}) + q*exp(-t)")
            } else {
                format!("-20.0*y{index} + 4.0*({left} - 2*y{index} + {right}) + 0.20*exp(-t)")
            };
            Expr::parse_expression(&source)
        })
        .collect()
}

fn prepare(
    dimension: usize,
    backend: IvpSymbolicAssemblyBackend,
    parameterized: bool,
) -> RustedSciThe::symbolic::symbolic_ivp::PreparedSymbolicIvpProblem {
    let mut options = SymbolicIvpProblemOptions::new().with_symbolic_assembly_backend(backend);
    if parameterized {
        options = options
            .with_equation_parameters(vec!["k".into(), "d".into(), "q".into()])
            .with_equation_parameter_values(DVector::from_vec(vec![20.0, 4.0, 0.20]));
    }
    prepare_symbolic_ivp_problem(
        equations(dimension, parameterized),
        (0..dimension).map(|index| format!("y{index}")).collect(),
        "t".to_string(),
        options,
    )
    .expect("parameter-free callback benchmark preparation should succeed")
}

fn state(dimension: usize) -> DVector<f64> {
    DVector::from_iterator(
        dimension,
        (0..dimension).map(|index| 0.2 + 0.01 * (index % 11) as f64),
    )
}

fn benchmark_parameter_free_callbacks(c: &mut Criterion) {
    let mut group = c.benchmark_group("ivp_parameter_free_residual_into");
    for &dimension in DIMENSIONS {
        let state = state(dimension);
        let native = prepare(dimension, IvpSymbolicAssemblyBackend::AtomView, false);
        let native_parameterized = prepare(dimension, IvpSymbolicAssemblyBackend::AtomView, true);
        let legacy = prepare(dimension, IvpSymbolicAssemblyBackend::ExprLegacy, false);
        let mut native_output = DVector::zeros(dimension);
        let mut native_parameterized_output = DVector::zeros(dimension);
        let mut legacy_output = DVector::zeros(dimension);

        group.bench_with_input(
            BenchmarkId::new("atom-native-parameter-free", dimension),
            &dimension,
            |bencher, _| {
                bencher.iter(|| {
                    native
                        .try_evaluate_residual_into(
                            black_box(0.5),
                            black_box(&state),
                            &mut native_output,
                        )
                        .expect("native parameter-free residual should evaluate");
                    black_box(native_output.as_slice());
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("atom-native-parameterized-control", dimension),
            &dimension,
            |bencher, _| {
                bencher.iter(|| {
                    native_parameterized
                        .try_evaluate_residual_into(
                            black_box(0.5),
                            black_box(&state),
                            &mut native_parameterized_output,
                        )
                        .expect("native parameterized residual should evaluate");
                    black_box(native_parameterized_output.as_slice());
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("expr-legacy-parameter-free", dimension),
            &dimension,
            |bencher, _| {
                bencher.iter(|| {
                    legacy
                        .try_evaluate_residual_into(
                            black_box(0.5),
                            black_box(&state),
                            &mut legacy_output,
                        )
                        .expect("ExprLegacy parameter-free residual should evaluate");
                    black_box(legacy_output.as_slice());
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, benchmark_parameter_free_callbacks);
criterion_main!(benches);

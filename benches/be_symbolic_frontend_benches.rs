use RustedSciThe::symbolic::symbolic_engine::Expr;
use RustedSciThe::symbolic::symbolic_ivp::{
    IvpSymbolicAssemblyBackend, SymbolicIvpProblemOptions, prepare_symbolic_ivp_problem,
};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use nalgebra::DVector;
use std::hint::black_box;
use std::time::Duration;

fn dense_coupled_fixture(dimension: usize) -> (Vec<Expr>, Vec<String>) {
    let variables: Vec<String> = (0..dimension).map(|i| format!("y{i}")).collect();
    let total_state = variables.join("+");
    let equations = variables
        .iter()
        .enumerate()
        .map(|(i, variable)| {
            let next = &variables[(i + 1) % dimension];
            Expr::parse_expression(&format!(
                "-0.1*{variable}+0.00001*({total_state})*({total_state})+0.02*{variable}*{next}"
            ))
        })
        .collect();
    (equations, variables)
}

fn prepare(
    equations: Vec<Expr>,
    variables: Vec<String>,
    assembly: IvpSymbolicAssemblyBackend,
) -> RustedSciThe::symbolic::symbolic_ivp::PreparedSymbolicIvpProblem {
    prepare_symbolic_ivp_problem(
        equations,
        variables,
        "t".to_string(),
        SymbolicIvpProblemOptions::new().with_symbolic_assembly_backend(assembly),
    )
    .expect("symbolic frontend preparation should succeed")
}

fn benchmark_symbolic_frontends(c: &mut Criterion) {
    let mut group = c.benchmark_group("be_symbolic_frontend");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));

    for dimension in [3, 8, 16, 32] {
        let (equations, variables) = dense_coupled_fixture(dimension);
        let state = DVector::from_fn(dimension, |i, _| 0.5 + i as f64 / dimension as f64);
        let legacy = prepare(
            equations.clone(),
            variables.clone(),
            IvpSymbolicAssemblyBackend::ExprLegacy,
        );
        let atom = prepare(
            equations.clone(),
            variables.clone(),
            IvpSymbolicAssemblyBackend::AtomView,
        );

        let legacy_residual = (legacy.residual)(0.25, &state);
        let atom_residual = (atom.residual)(0.25, &state);
        let legacy_jacobian = (legacy.jacobian)(0.25, &state);
        let atom_jacobian = (atom.jacobian)(0.25, &state);
        assert!(
            legacy_residual
                .iter()
                .zip(atom_residual.iter())
                .all(|(a, b)| (a - b).abs() <= 1e-10)
        );
        assert!(
            legacy_jacobian
                .iter()
                .zip(atom_jacobian.iter())
                .all(|(a, b)| (a - b).abs() <= 1e-10)
        );

        for (frontend, assembly) in [
            ("expr-legacy", IvpSymbolicAssemblyBackend::ExprLegacy),
            ("atom-native", IvpSymbolicAssemblyBackend::AtomView),
        ] {
            group.bench_function(
                BenchmarkId::new("prepare", format!("{frontend}/{dimension}")),
                |bencher| {
                    bencher.iter_batched(
                        || (equations.clone(), variables.clone()),
                        |(equations, variables)| {
                            black_box(prepare(equations, variables, assembly));
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }

        group.bench_function(
            BenchmarkId::new("residual", format!("expr-legacy/{dimension}")),
            |bencher| {
                bencher.iter(|| black_box((legacy.residual)(0.25, black_box(&state))));
            },
        );
        group.bench_function(
            BenchmarkId::new("residual", format!("atom-native/{dimension}")),
            |bencher| {
                bencher.iter(|| black_box((atom.residual)(0.25, black_box(&state))));
            },
        );
        group.bench_function(
            BenchmarkId::new("jacobian", format!("expr-legacy/{dimension}")),
            |bencher| {
                bencher.iter(|| black_box((legacy.jacobian)(0.25, black_box(&state))));
            },
        );
        group.bench_function(
            BenchmarkId::new("jacobian", format!("atom-native/{dimension}")),
            |bencher| {
                bencher.iter(|| black_box((atom.jacobian)(0.25, black_box(&state))));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_symbolic_frontends);
criterion_main!(benches);

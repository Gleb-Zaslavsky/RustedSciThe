//! Warm callback comparison for Expr closures, Atom Lambdify and explicit
//! `PreparedAtomSystem` execution with caller-owned scratch.
//!
//! Preparation is outside the timed loop. All routes evaluate the same sparse
//! chain residuals and analytically differentiated Jacobian entries into
//! caller-owned output buffers. This isolates callback execution; it is not a
//! full solver or preparation-cost benchmark.

use std::hint::black_box;

use RustedSciThe::symbolic::{
    View::{PreparedAtomSystem, PreparedAtomSystemScratch, conversions::expr_to_atom},
    symbolic_engine::Expr,
};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

type ScalarClosure = Box<dyn Fn(&[f64]) -> f64 + Send + Sync>;

struct CallbackFixture {
    dimension: usize,
    input: Vec<f64>,
    expr_residuals: Vec<ScalarClosure>,
    expr_jacobian: Vec<ScalarClosure>,
    atom_residuals: Vec<ScalarClosure>,
    atom_jacobian: Vec<ScalarClosure>,
    prepared_system: PreparedAtomSystem,
}

impl CallbackFixture {
    fn new(dimension: usize) -> Self {
        let names = (0..dimension)
            .map(|index| format!("y{index}"))
            .collect::<Vec<_>>();
        let name_refs = names.iter().map(String::as_str).collect::<Vec<_>>();
        let residual_exprs = (0..dimension)
            .map(|row| {
                if row == 0 {
                    Expr::parse_expression("y0^2")
                } else {
                    Expr::parse_expression(&format!("y{row}^2 + 0.125*y{}", row - 1))
                }
            })
            .collect::<Vec<_>>();

        let mut jacobian_exprs = Vec::with_capacity(2 * dimension - 1);
        for row in 0..dimension {
            if row > 0 {
                jacobian_exprs.push(residual_exprs[row].diff(&names[row - 1]));
            }
            jacobian_exprs.push(residual_exprs[row].diff(&names[row]));
        }

        let expr_residuals = residual_exprs
            .iter()
            .map(|expr| Expr::lambdify_borrowed_thread_safe(expr, &name_refs))
            .collect::<Vec<_>>();
        let expr_jacobian = jacobian_exprs
            .iter()
            .map(|expr| Expr::lambdify_borrowed_thread_safe(expr, &name_refs))
            .collect::<Vec<_>>();

        let residual_atoms = residual_exprs.iter().map(expr_to_atom).collect::<Vec<_>>();
        let jacobian_atoms = jacobian_exprs.iter().map(expr_to_atom).collect::<Vec<_>>();
        let atom_residuals = residual_atoms
            .iter()
            .map(|atom| atom.lambdify_borrowed_thread_safe(&name_refs))
            .collect::<Vec<_>>();
        let atom_jacobian = jacobian_atoms
            .iter()
            .map(|atom| atom.lambdify_borrowed_thread_safe(&name_refs))
            .collect::<Vec<_>>();

        let prepared_system = PreparedAtomSystem::try_from_exprs(&residual_exprs, &names, None)
            .expect("sparse-chain prepared system should compile");

        Self {
            dimension,
            input: (0..dimension)
                .map(|index| 0.5 + index as f64 * 1.0e-3)
                .collect(),
            expr_residuals,
            expr_jacobian,
            atom_residuals,
            atom_jacobian,
            prepared_system,
        }
    }

    fn evaluate_closures(
        residuals: &[ScalarClosure],
        jacobian: &[ScalarClosure],
        input: &[f64],
        residual_output: &mut [f64],
        jacobian_output: &mut [f64],
    ) {
        for (output, evaluator) in residual_output.iter_mut().zip(residuals) {
            *output = evaluator(input);
        }
        for (output, evaluator) in jacobian_output.iter_mut().zip(jacobian) {
            *output = evaluator(input);
        }
    }

    fn evaluate_prepared(
        prepared: &PreparedAtomSystem,
        input: &[f64],
        residual_output: &mut [f64],
        jacobian_output: &mut [f64],
        scratch: &mut PreparedAtomSystemScratch,
    ) {
        prepared
            .residual_and_jacobian_into(input, residual_output, jacobian_output, scratch)
            .expect("finite sparse-chain callback should evaluate");
    }

    fn outputs(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![0.0; self.dimension], vec![0.0; 2 * self.dimension - 1])
    }

    fn assert_route_parity(&self) {
        let (mut residual_expected, mut jacobian_expected) = self.outputs();
        Self::evaluate_closures(
            &self.expr_residuals,
            &self.expr_jacobian,
            &self.input,
            &mut residual_expected,
            &mut jacobian_expected,
        );

        let (mut residual_atom, mut jacobian_atom) = self.outputs();
        Self::evaluate_closures(
            &self.atom_residuals,
            &self.atom_jacobian,
            &self.input,
            &mut residual_atom,
            &mut jacobian_atom,
        );

        let (mut residual_prepared, mut jacobian_prepared) = self.outputs();
        let mut scratch = PreparedAtomSystemScratch::default();
        Self::evaluate_prepared(
            &self.prepared_system,
            &self.input,
            &mut residual_prepared,
            &mut jacobian_prepared,
            &mut scratch,
        );

        for (route, residual, jacobian) in [
            ("Atom closure", residual_atom, jacobian_atom),
            ("PreparedEvaluator", residual_prepared, jacobian_prepared),
        ] {
            let max_diff = residual_expected
                .iter()
                .chain(&jacobian_expected)
                .zip(residual.iter().chain(&jacobian))
                .map(|(expected, actual)| (expected - actual).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                max_diff <= 1.0e-12,
                "dimension={} {route} parity failed: max diff={max_diff:e}",
                self.dimension
            );
        }
    }
}

fn benchmark_prepared_atom_callback_routes(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("symbolic_prepared_atom_sparse_chain_callbacks");
    for dimension in [128, 512] {
        let fixture = CallbackFixture::new(dimension);
        fixture.assert_route_parity();
        eprintln!(
            "[Symbolic prepared-runtime preflight] dimension={} jacobian_nnz={} Expr/Atom/scratch parity=ok; preparation excluded",
            dimension,
            2 * dimension - 1
        );

        let (mut expr_residual, mut expr_jacobian) = fixture.outputs();
        group.bench_with_input(
            BenchmarkId::new("expr_lambdify_closures", dimension),
            &dimension,
            |bencher, _| {
                bencher.iter(|| {
                    CallbackFixture::evaluate_closures(
                        &fixture.expr_residuals,
                        &fixture.expr_jacobian,
                        black_box(&fixture.input),
                        &mut expr_residual,
                        &mut expr_jacobian,
                    );
                    black_box((&expr_residual, &expr_jacobian));
                });
            },
        );

        let (mut atom_residual, mut atom_jacobian) = fixture.outputs();
        group.bench_with_input(
            BenchmarkId::new("atom_lambdify_closures", dimension),
            &dimension,
            |bencher, _| {
                bencher.iter(|| {
                    CallbackFixture::evaluate_closures(
                        &fixture.atom_residuals,
                        &fixture.atom_jacobian,
                        black_box(&fixture.input),
                        &mut atom_residual,
                        &mut atom_jacobian,
                    );
                    black_box((&atom_residual, &atom_jacobian));
                });
            },
        );

        let (mut prepared_residual, mut prepared_jacobian) = fixture.outputs();
        let mut scratch = PreparedAtomSystemScratch::default();
        group.bench_with_input(
            BenchmarkId::new("prepared_atom_system_caller_scratch", dimension),
            &dimension,
            |bencher, _| {
                bencher.iter(|| {
                    CallbackFixture::evaluate_prepared(
                        &fixture.prepared_system,
                        black_box(&fixture.input),
                        &mut prepared_residual,
                        &mut prepared_jacobian,
                        &mut scratch,
                    );
                    black_box((&prepared_residual, &prepared_jacobian));
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, benchmark_prepared_atom_callback_routes);
criterion_main!(benches);

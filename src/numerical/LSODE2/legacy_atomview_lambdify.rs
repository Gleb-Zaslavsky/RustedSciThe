//! Historical AtomView Lambdify adapter retained as a comparison oracle.
//!
//! This is intentionally isolated from the production route. It mirrors the
//! pre-telemetry implementation: AtomView builds a sparse symbolic Jacobian,
//! entries are converted back to `Expr` and simplified, and the resulting
//! expressions are compiled into direct callbacks. The adapter is test-only so
//! it cannot become an accidental default backend.

use crate::numerical::BDF::BDF_solver::BdfJacobian;
use crate::numerical::LSODE2::native_jacobian::NativeJacobianStorage;
use crate::somelinalg::banded::storage::Banded;
use crate::symbolic::View::conversions::atom_to_expr;
use crate::symbolic::View::jacobian::PreparedSparseAtomSystem;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp::SharedIvpParameterValues;
use faer::sparse::Triplet;
use nalgebra::DVector;

type CompiledEntry = (usize, usize, Box<dyn Fn(&[f64]) -> f64 + Send + Sync>);

pub(crate) struct LegacyAtomViewCallbacks {
    pub(crate) residual: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64> + Send + Sync>,
    pub(crate) jacobian: Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian>,
}

pub(crate) fn prepare(
    equations: &[Expr],
    variables: &[String],
    time_arg: &str,
    equation_parameters: Option<&[String]>,
    parameter_values: Option<DVector<f64>>,
    storage: NativeJacobianStorage,
) -> LegacyAtomViewCallbacks {
    let parameter_values_handle =
        parameter_values.map(|values| std::sync::Arc::new(std::sync::RwLock::new(values)));
    let symbolic_jacobian = symbolic_jacobian(equations, variables);
    let residual = compile_residual(
        equations,
        time_arg,
        variables,
        equation_parameters,
        parameter_values_handle.clone(),
    );
    let jacobian = compile_jacobian(
        &symbolic_jacobian,
        time_arg,
        variables,
        equation_parameters,
        parameter_values_handle,
        storage,
    );
    LegacyAtomViewCallbacks { residual, jacobian }
}

pub(crate) fn symbolic_jacobian(equations: &[Expr], variables: &[String]) -> Vec<Vec<Expr>> {
    let rows = equations.len();
    let cols = variables.len();
    let variables_for_all_discrete = vec![variables.to_vec(); rows];
    let sparse_entries =
        PreparedSparseAtomSystem::from_exprs(equations, variables, &variables_for_all_discrete)
            .calc_sparse_jacobian_with_bandwidth(None);

    let zero = Expr::parse_expression("0");
    let mut dense = vec![vec![zero.clone(); cols]; rows];
    for entry in sparse_entries {
        dense[entry.row][entry.col] = atom_to_expr(&entry.value).simplify();
    }
    dense
}

fn compile_residual(
    equations: &[Expr],
    time_arg: &str,
    variables: &[String],
    equation_parameters: Option<&[String]>,
    parameter_values_handle: Option<SharedIvpParameterValues>,
) -> Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64> + Send + Sync> {
    let name_refs = build_name_refs(time_arg, variables, equation_parameters);
    let compiled = equations
        .iter()
        .map(|expr| Expr::lambdify_borrowed_thread_safe(expr, &name_refs))
        .collect::<Vec<_>>();

    Box::new(move |t, y| {
        let parameter_values = parameter_values_handle.as_ref().map(|handle| {
            handle
                .read()
                .expect("historical AtomView parameter state lock poisoned")
                .clone()
        });
        let args = build_args(t, y, parameter_values.as_ref());
        DVector::from_vec(compiled.iter().map(|func| func(&args)).collect())
    })
}

fn compile_jacobian(
    symbolic_jacobian: &[Vec<Expr>],
    time_arg: &str,
    variables: &[String],
    equation_parameters: Option<&[String]>,
    parameter_values_handle: Option<SharedIvpParameterValues>,
    storage: NativeJacobianStorage,
) -> Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian> {
    let rows = symbolic_jacobian.len();
    let cols = symbolic_jacobian.first().map_or(0, |row| row.len());
    let name_refs = build_name_refs(time_arg, variables, equation_parameters);
    let entries = symbolic_jacobian
        .iter()
        .enumerate()
        .flat_map(|(row, symbolic_row)| {
            symbolic_row.iter().enumerate().filter_map({
                let name_refs = name_refs.clone();
                move |(col, expr)| {
                    (!expr.is_zero()).then(|| {
                        (
                            row,
                            col,
                            Expr::lambdify_borrowed_thread_safe(expr, &name_refs),
                        )
                    })
                }
            })
        })
        .collect::<Vec<CompiledEntry>>();

    match storage {
        NativeJacobianStorage::Dense => Box::new(move |t, y| {
            let parameter_values = parameter_values_handle.as_ref().map(|handle| {
                handle
                    .read()
                    .expect("historical AtomView parameter state lock poisoned")
                    .clone()
            });
            let args = build_args(t, y, parameter_values.as_ref());
            let mut matrix = nalgebra::DMatrix::<f64>::zeros(rows, cols);
            for (row, col, eval) in &entries {
                matrix[(*row, *col)] = eval(&args);
            }
            BdfJacobian::Dense(matrix)
        }),
        NativeJacobianStorage::SparseTriplets => Box::new(move |t, y| {
            let parameter_values = parameter_values_handle.as_ref().map(|handle| {
                handle
                    .read()
                    .expect("historical AtomView parameter state lock poisoned")
                    .clone()
            });
            let args = build_args(t, y, parameter_values.as_ref());
            let triplets = entries
                .iter()
                .map(|(row, col, eval)| Triplet::new(*row, *col, eval(&args)))
                .collect::<Vec<_>>();
            BdfJacobian::SparseTriplets { n: rows, triplets }
        }),
        NativeJacobianStorage::Banded { bandwidth } => {
            let (kl, ku) = bandwidth.unwrap_or_else(|| infer_bandwidth(rows, cols, &entries));
            Box::new(move |t, y| {
                let parameter_values = parameter_values_handle.as_ref().map(|handle| {
                    handle
                        .read()
                        .expect("historical AtomView parameter state lock poisoned")
                        .clone()
                });
                let args = build_args(t, y, parameter_values.as_ref());
                let mut banded = Banded::<f64>::zeros(rows, kl, ku)
                    .expect("historical AtomView bandwidth should be valid");
                for (row, col, eval) in &entries {
                    banded
                        .set(*row, *col, eval(&args))
                        .expect("historical AtomView entry should fit its bandwidth");
                }
                BdfJacobian::Banded(banded)
            })
        }
    }
}

fn build_name_refs<'a>(
    time_arg: &'a str,
    variables: &'a [String],
    equation_parameters: Option<&'a [String]>,
) -> Vec<&'a str> {
    let mut names = Vec::with_capacity(
        1 + variables.len() + equation_parameters.map_or(0, |parameters| parameters.len()),
    );
    names.push(time_arg);
    if let Some(parameters) = equation_parameters {
        names.extend(parameters.iter().map(String::as_str));
    }
    names.extend(variables.iter().map(String::as_str));
    names
}

fn build_args(t: f64, y: &DVector<f64>, parameter_values: Option<&DVector<f64>>) -> Vec<f64> {
    let mut args = Vec::with_capacity(1 + y.len() + parameter_values.map_or(0, DVector::len));
    args.push(t);
    if let Some(values) = parameter_values {
        args.extend(values.iter().copied());
    }
    args.extend(y.iter().copied());
    args
}

fn infer_bandwidth(rows: usize, cols: usize, entries: &[CompiledEntry]) -> (usize, usize) {
    let mut kl = 0;
    let mut ku = 0;
    for (row, col, _) in entries {
        kl = kl.max(row.saturating_sub(*col));
        ku = ku.max(col.saturating_sub(*row));
    }
    if rows == cols {
        (kl, ku)
    } else {
        (rows.saturating_sub(1), cols.saturating_sub(1))
    }
}

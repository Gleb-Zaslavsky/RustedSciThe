//! Typed ownership boundary for the AtomView-native BVP AOT route.
//!
//! This module deliberately does not build or link an artifact yet.  It keeps
//! the symbolic payload and its output layout separate from the historical
//! `Expr`-based AOT adapter so that later compiler integration cannot silently
//! reintroduce an `Atom -> Expr` conversion.

use std::fmt;

use super::aot_telemetry::{BvpAotTelemetry, BvpAotTelemetryMode, BvpAotTelemetrySnapshot};
use crate::symbolic::View::atom::Atom;
use crate::symbolic::View::bvp::DiscretizedBvpAtomSystem;
use crate::symbolic::View::jacobian::SparseAtomJacobianEntry;
use crate::symbolic::View::state::Symbol;
use crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy;
use crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy;

/// Output storage contract owned by an AtomView AOT plan.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum AtomAotMatrixLayout {
    /// Fixed coordinate/value ordering for a CSC-compatible sparse callback.
    SparseCsc {
        rows: usize,
        cols: usize,
        nnz: usize,
    },
    /// Native band-slot callback layout.  The slot ordering is defined by the
    /// entry vector and is not allowed to fall back to dense staging.
    Banded {
        rows: usize,
        cols: usize,
        kl: usize,
        ku: usize,
        slots: usize,
    },
}

impl AtomAotMatrixLayout {
    /// Matrix dimensions carried by the generated callback contract.
    pub const fn shape(&self) -> (usize, usize) {
        match *self {
            Self::SparseCsc { rows, cols, .. } | Self::Banded { rows, cols, .. } => (rows, cols),
        }
    }

    /// Number of values emitted by the Jacobian callback.
    pub const fn value_count(&self) -> usize {
        match *self {
            Self::SparseCsc { nnz, .. } => nnz,
            Self::Banded { slots, .. } => slots,
        }
    }
}

/// Typed preparation failure for the AtomView AOT plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AtomAotPlanError {
    EmptyResidualSystem,
    EmptyVariableSchema,
    InputSchemaMismatch {
        names: usize,
        symbols: usize,
    },
    ShapeMismatch {
        residuals: usize,
        rows: usize,
        cols: usize,
    },
    InvalidJacobianCoordinate {
        row: usize,
        col: usize,
        rows: usize,
        cols: usize,
    },
    InvalidBandedLayout {
        kl: usize,
        ku: usize,
        rows: usize,
        cols: usize,
    },
    JacobianValueCountMismatch {
        entries: usize,
        expected: usize,
    },
}

impl fmt::Display for AtomAotPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyResidualSystem => f.write_str("AtomView AOT residual system is empty"),
            Self::EmptyVariableSchema => f.write_str("AtomView AOT variable schema is empty"),
            Self::InputSchemaMismatch { names, symbols } => {
                write!(
                    f,
                    "AtomView AOT input schema has {names} names but {symbols} symbols"
                )
            }
            Self::ShapeMismatch {
                residuals,
                rows,
                cols,
            } => write!(
                f,
                "AtomView AOT shape {rows}x{cols} does not match {residuals} residuals"
            ),
            Self::InvalidJacobianCoordinate {
                row,
                col,
                rows,
                cols,
            } => write!(
                f,
                "AtomView AOT Jacobian coordinate ({row}, {col}) is outside {rows}x{cols}"
            ),
            Self::InvalidBandedLayout { kl, ku, rows, cols } => write!(
                f,
                "AtomView AOT Banded layout kl={kl}, ku={ku} is invalid for {rows}x{cols}"
            ),
            Self::JacobianValueCountMismatch { entries, expected } => write!(
                f,
                "AtomView AOT Jacobian has {entries} entries but layout requires {expected}"
            ),
        }
    }
}

impl std::error::Error for AtomAotPlanError {}

/// Prepared AtomView payload shared by future Rust/C/Zig AOT emitters.
///
/// The plan owns the symbolic values needed for both cold code generation and
/// warm callbacks.  It contains no `Expr`, trait object or compiler-specific
/// state.  Those belong to explicit adapters layered on top of this plan.
#[derive(Clone, Debug)]
pub struct AtomAotPreparedPlan {
    residuals: Vec<Atom>,
    jacobian_entries: Vec<SparseAtomJacobianEntry>,
    input_names: Vec<String>,
    input_symbols: Vec<Symbol>,
    parameter_count: usize,
    matrix_layout: AtomAotMatrixLayout,
    residual_strategy: ResidualChunkingStrategy,
    jacobian_strategy: SparseChunkingStrategy,
    telemetry: BvpAotTelemetry,
}

impl AtomAotPreparedPlan {
    /// Builds a sparse AtomView AOT plan from an already discretized system.
    ///
    /// The only conversion performed here is name-to-`Symbol` interning.  The
    /// residual and Jacobian payload remains packed Atom data throughout.
    pub fn from_discretized_sparse(
        discretized: &DiscretizedBvpAtomSystem,
        parameter_names: &[String],
        residual_strategy: ResidualChunkingStrategy,
        jacobian_strategy: SparseChunkingStrategy,
    ) -> Result<Self, AtomAotPlanError> {
        let input_names = parameter_names
            .iter()
            .chain(discretized.variable_string.iter())
            .cloned()
            .collect::<Vec<_>>();
        let input_symbols = input_names
            .iter()
            .map(|name| Symbol::new(crate::wrap_symbol!(name.as_str())))
            .collect::<Vec<_>>();
        let entries = crate::symbolic::View::jacobian::PreparedSparseAtomSystem::from_atoms(
            &discretized.vector_of_functions,
            &discretized.variable_string,
            &discretized.variables_for_all_discrete,
        )
        .calc_sparse_jacobian_with_bandwidth(None);
        let nnz = entries.len();

        Self::from_parts(
            discretized.vector_of_functions.clone(),
            entries,
            input_names,
            input_symbols,
            parameter_names.len(),
            AtomAotMatrixLayout::SparseCsc {
                rows: discretized.vector_of_functions.len(),
                cols: discretized.variable_string.len(),
                nnz,
            },
            residual_strategy,
            jacobian_strategy,
        )
    }

    /// Builds a plan from owned Atom payload and an explicit output layout.
    ///
    /// This constructor is also used by the codegen bridge so layout checks are
    /// performed before any language-specific source emitter is invoked.
    pub fn from_parts(
        residuals: Vec<Atom>,
        jacobian_entries: Vec<SparseAtomJacobianEntry>,
        input_names: Vec<String>,
        input_symbols: Vec<Symbol>,
        parameter_count: usize,
        matrix_layout: AtomAotMatrixLayout,
        residual_strategy: ResidualChunkingStrategy,
        jacobian_strategy: SparseChunkingStrategy,
    ) -> Result<Self, AtomAotPlanError> {
        validate_plan(
            &residuals,
            &jacobian_entries,
            &input_names,
            &input_symbols,
            matrix_layout.shape(),
            &matrix_layout,
        )?;
        if matrix_layout.value_count() != jacobian_entries.len() {
            return Err(AtomAotPlanError::JacobianValueCountMismatch {
                entries: jacobian_entries.len(),
                expected: matrix_layout.value_count(),
            });
        }
        if parameter_count > input_names.len() {
            return Err(AtomAotPlanError::InputSchemaMismatch {
                names: input_names.len(),
                symbols: input_symbols.len(),
            });
        }
        Ok(Self {
            residuals,
            jacobian_entries,
            input_names,
            input_symbols,
            parameter_count,
            matrix_layout,
            residual_strategy,
            jacobian_strategy,
            telemetry: BvpAotTelemetry::disabled(),
        })
    }

    /// Enables the typed AOT telemetry stream for this prepared plan.
    pub fn with_telemetry_mode(mut self, mode: BvpAotTelemetryMode) -> Self {
        self.telemetry = BvpAotTelemetry::with_mode(mode);
        self
    }

    pub fn residuals(&self) -> &[Atom] {
        &self.residuals
    }

    pub fn jacobian_entries(&self) -> &[SparseAtomJacobianEntry] {
        &self.jacobian_entries
    }

    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    pub fn input_symbols(&self) -> &[Symbol] {
        &self.input_symbols
    }

    pub const fn parameter_count(&self) -> usize {
        self.parameter_count
    }

    pub const fn matrix_layout(&self) -> &AtomAotMatrixLayout {
        &self.matrix_layout
    }

    pub const fn residual_strategy(&self) -> ResidualChunkingStrategy {
        self.residual_strategy
    }

    pub const fn jacobian_strategy(&self) -> SparseChunkingStrategy {
        self.jacobian_strategy
    }

    pub fn telemetry(&self) -> &BvpAotTelemetry {
        &self.telemetry
    }

    pub fn telemetry_snapshot(&self) -> BvpAotTelemetrySnapshot {
        self.telemetry.snapshot()
    }
}

fn validate_plan(
    residuals: &[Atom],
    entries: &[SparseAtomJacobianEntry],
    input_names: &[String],
    input_symbols: &[Symbol],
    (rows, cols): (usize, usize),
    layout: &AtomAotMatrixLayout,
) -> Result<(), AtomAotPlanError> {
    if residuals.is_empty() {
        return Err(AtomAotPlanError::EmptyResidualSystem);
    }
    if cols == 0 || input_names.is_empty() {
        return Err(AtomAotPlanError::EmptyVariableSchema);
    }
    if input_names.len() != input_symbols.len() {
        return Err(AtomAotPlanError::InputSchemaMismatch {
            names: input_names.len(),
            symbols: input_symbols.len(),
        });
    }
    if residuals.len() != rows {
        return Err(AtomAotPlanError::ShapeMismatch {
            residuals: residuals.len(),
            rows,
            cols,
        });
    }
    if let Some(entry) = entries
        .iter()
        .find(|entry| entry.row >= rows || entry.col >= cols)
    {
        return Err(AtomAotPlanError::InvalidJacobianCoordinate {
            row: entry.row,
            col: entry.col,
            rows,
            cols,
        });
    }
    if let AtomAotMatrixLayout::Banded { kl, ku, .. } = layout {
        if *kl >= rows || *ku >= cols {
            return Err(AtomAotPlanError::InvalidBandedLayout {
                kl: *kl,
                ku: *ku,
                rows,
                cols,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::View::parser::parse;

    fn atom(source: &str) -> Atom {
        parse(source).expect("test atom must parse")
    }

    fn entry(row: usize, col: usize, source: &str) -> SparseAtomJacobianEntry {
        SparseAtomJacobianEntry {
            row,
            col,
            value: atom(source),
        }
    }

    #[test]
    fn sparse_plan_owns_atom_payload_and_ordered_input_schema() {
        let plan = AtomAotPreparedPlan::from_parts(
            vec![atom("x + p"), atom("y - p")],
            vec![entry(0, 0, "1"), entry(1, 1, "1")],
            vec!["p".to_string(), "x".to_string(), "y".to_string()],
            vec![
                Symbol::new(crate::wrap_symbol!("p")),
                Symbol::new(crate::wrap_symbol!("x")),
                Symbol::new(crate::wrap_symbol!("y")),
            ],
            1,
            AtomAotMatrixLayout::SparseCsc {
                rows: 2,
                cols: 2,
                nnz: 2,
            },
            ResidualChunkingStrategy::Whole,
            SparseChunkingStrategy::Whole,
        )
        .expect("valid AtomView AOT plan");

        assert_eq!(plan.parameter_count(), 1);
        assert_eq!(plan.input_names(), &["p", "x", "y"]);
        assert_eq!(plan.residuals().len(), 2);
        assert_eq!(plan.jacobian_entries().len(), 2);
        assert_eq!(plan.matrix_layout().value_count(), 2);
    }

    #[test]
    fn plan_rejects_invalid_coordinates_before_codegen() {
        let error = AtomAotPreparedPlan::from_parts(
            vec![atom("x")],
            vec![entry(1, 0, "1")],
            vec!["x".to_string()],
            vec![Symbol::new(crate::wrap_symbol!("x"))],
            0,
            AtomAotMatrixLayout::SparseCsc {
                rows: 1,
                cols: 1,
                nnz: 1,
            },
            ResidualChunkingStrategy::Whole,
            SparseChunkingStrategy::Whole,
        )
        .expect_err("out-of-bounds Atom entry must be typed");

        assert!(matches!(
            error,
            AtomAotPlanError::InvalidJacobianCoordinate { .. }
        ));
    }

    #[test]
    fn banded_layout_is_explicit_and_not_sparse_fallback() {
        let layout = AtomAotMatrixLayout::Banded {
            rows: 3,
            cols: 3,
            kl: 1,
            ku: 1,
            slots: 3,
        };

        assert_eq!(layout.shape(), (3, 3));
        assert_eq!(layout.value_count(), 3);
        assert_ne!(
            layout,
            AtomAotMatrixLayout::SparseCsc {
                rows: 3,
                cols: 3,
                nnz: 3,
            }
        );
    }
}

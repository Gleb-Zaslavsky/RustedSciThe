//! Bordered-banded collocation backend.
//!
//! A collocation Jacobian has a block-tridiagonal interior, but endpoint
//! boundary equations couple distant mesh nodes. Treating that matrix as one
//! scalar band makes the bandwidth grow with the mesh and can hide a large
//! amount of fill-in. The production Lambdify path therefore uses the shared
//! `[T U; V D]` factorization from `somelinalg` and keeps the old scalar band
//! implementation only for the low-level compatibility constructor.

use crate::numerical::BVP_sci::new::{
    BvpSciBandedRoute, BvpSciMatrixLayout, BvpSciTelemetry, BvpSciTelemetrySnapshot,
    error::BvpSciNewError, linear::LinearSystemBackend,
};
use crate::somelinalg::banded::{
    Banded, BandedError, BorderedBlockTridiagonal, GeneralBandedLuPartialPivot,
    LapackStyleBandedLuFaithful,
};
use faer::linalg::solvers::Solve;
use faer::sparse::linalg::solvers::Lu;
use faer::sparse::{SparseColMat, Triplet};

#[derive(Clone, Debug)]
enum Storage {
    Scalar {
        matrix: Banded<f64>,
        factorization: LapackStyleBandedLuFaithful,
    },
    Bordered {
        system: BorderedBlockTridiagonal,
        partition: CollocationPartition,
        reordered_rhs: Vec<f64>,
        original_rhs: Vec<f64>,
        scalar_fallback: Option<ScalarFallback>,
        use_scalar_fallback: bool,
        scalar_fallback_factorized: bool,
        sparse_fallback: SparseSafetyFallback,
        use_sparse_fallback: bool,
        telemetry: BvpSciTelemetry,
    },
}

#[derive(Clone, Debug)]
struct ScalarFallback {
    matrix: Banded<f64>,
    factorization: GeneralBandedLuPartialPivot,
}

/// Sparse safety handoff for a bordered factorization whose interior `T` is
/// effectively singular.  The normal Banded path remains structured; this
/// route is only activated after the residual guard proves that the Schur
/// reduction cannot produce a usable correction.  It keeps the global matrix
/// sparse and never materializes a dense fallback matrix.
#[derive(Clone, Debug, Default)]
struct SparseSafetyFallback {
    entries: Vec<(usize, usize, f64)>,
    factorization: Option<Lu<usize, f64>>,
}

impl SparseSafetyFallback {
    fn assemble(&mut self, entries: &[(usize, usize, f64)]) {
        self.entries.clear();
        self.entries.extend_from_slice(entries);
        self.factorization = None;
    }

    fn factor(&mut self, dimension: usize) -> Result<(), BvpSciNewError> {
        if self.factorization.is_none() {
            let triplets = self
                .entries
                .iter()
                .copied()
                .map(|(row, column, value)| Triplet::new(row, column, value))
                .collect::<Vec<_>>();
            let matrix = SparseColMat::<usize, f64>::try_new_from_triplets(
                dimension,
                dimension,
                &triplets,
            )
            .map_err(|error| BvpSciNewError::LinearBackend {
                message: format!("Banded sparse safety assembly failed: {error:?}"),
            })?;
            self.factorization = Some(matrix.sp_lu().map_err(|error| {
                BvpSciNewError::LinearBackend {
                    message: format!("Banded sparse safety factorization failed: {error:?}"),
                }
            })?);
        }
        Ok(())
    }

    fn solve_in_place(
        &mut self,
        dimension: usize,
        rhs: &mut [f64],
    ) -> Result<(), BvpSciNewError> {
        let factorization = self.factorization.as_ref().ok_or_else(|| {
            BvpSciNewError::LinearBackend {
                message: "Banded sparse safety factorization is unavailable".into(),
            }
        })?;
        let solution = factorization.solve(&faer::Col::from_fn(dimension, |index| rhs[index]));
        for index in 0..dimension {
            rhs[index] = solution[index];
        }
        Ok(())
    }
}

/// Partition of the global collocation matrix into structured core and border.
///
/// For a mesh with at least three nodes, interior nodes form `T`; node zero,
/// the last node and parameters form the dense border. The interval-zero rows
/// and all boundary rows become the border rows. For a two-node mesh there is
/// no interior interval, so the first state block is used as the one-block
/// core and the remaining endpoint/parameter variables form the border.
#[derive(Clone, Copy, Debug)]
struct CollocationPartition {
    state_dimension: usize,
    node_count: usize,
    parameter_dimension: usize,
    core_dimension: usize,
    border_dimension: usize,
}

impl CollocationPartition {
    fn new(
        total_dimension: usize,
        state_dimension: usize,
        node_count: usize,
        parameter_dimension: usize,
    ) -> Result<Self, BvpSciNewError> {
        if state_dimension == 0 || node_count < 2 {
            return Err(BvpSciNewError::InvalidConfiguration(
                "bordered-banded collocation requires a positive state and two mesh nodes".into(),
            ));
        }
        let expected = state_dimension
            .checked_mul(node_count)
            .and_then(|value| value.checked_add(parameter_dimension))
            .ok_or_else(|| {
                BvpSciNewError::InvalidConfiguration(
                    "bordered-banded collocation dimension overflows usize".into(),
                )
            })?;
        if expected != total_dimension {
            return Err(BvpSciNewError::InvalidConfiguration(
                "bordered-banded collocation dimensions do not match the global system".into(),
            ));
        }
        let core_dimension = if node_count == 2 {
            state_dimension
        } else {
            state_dimension * (node_count - 2)
        };
        let border_dimension = total_dimension - core_dimension;
        Ok(Self {
            state_dimension,
            node_count,
            parameter_dimension,
            core_dimension,
            border_dimension,
        })
    }

    fn core_blocks(self) -> usize {
        self.core_dimension / self.core_block_size()
    }

    /// Aggregate adjacent mesh states into a larger block when possible.
    ///
    /// The collocation matrix is still block-tridiagonal after aggregation,
    /// but a larger local pivot scope is materially more stable for stiff,
    /// strongly coupled systems than a separate LU pivot at every state node.
    fn core_block_nodes(self) -> usize {
        let interior_nodes = self.core_dimension / self.state_dimension;
        if interior_nodes >= 2 && interior_nodes % 2 == 0 {
            2
        } else {
            1
        }
    }

    fn core_block_size(self) -> usize {
        self.state_dimension * self.core_block_nodes()
    }

    fn row(self, row: usize) -> Result<RowPart, BvpSciNewError> {
        let boundary_start = (self.node_count - 1) * self.state_dimension;
        if self.node_count == 2 {
            if row < self.state_dimension {
                Ok(RowPart::Core(row))
            } else if row < self.state_dimension + self.border_dimension {
                Ok(RowPart::Border(row - self.state_dimension))
            } else {
                Err(self.index_error("row", row))
            }
        } else if row < self.state_dimension {
            Ok(RowPart::Border(row))
        } else if row < boundary_start {
            Ok(RowPart::Core(row - self.state_dimension))
        } else if row < boundary_start + self.state_dimension {
            Ok(RowPart::Border(self.state_dimension + row - boundary_start))
        } else if row < self.node_count * self.state_dimension + self.parameter_dimension {
            Ok(RowPart::Border(row - boundary_start + self.state_dimension))
        } else {
            Err(self.index_error("row", row))
        }
    }

    fn column(self, column: usize) -> Result<ColumnPart, BvpSciNewError> {
        let right_start = (self.node_count - 1) * self.state_dimension;
        let parameter_start = self.node_count * self.state_dimension;
        if self.node_count == 2 {
            if column < self.state_dimension {
                Ok(ColumnPart::Core(column))
            } else if column < parameter_start {
                Ok(ColumnPart::Border(column - self.state_dimension))
            } else if column < parameter_start + self.parameter_dimension {
                Ok(ColumnPart::Border(
                    self.state_dimension + column - parameter_start,
                ))
            } else {
                Err(self.index_error("column", column))
            }
        } else if column < self.state_dimension {
            Ok(ColumnPart::Border(column))
        } else if column < right_start {
            Ok(ColumnPart::Core(column - self.state_dimension))
        } else if column < parameter_start {
            Ok(ColumnPart::Border(
                self.state_dimension + column - right_start,
            ))
        } else if column < parameter_start + self.parameter_dimension {
            Ok(ColumnPart::Border(
                2 * self.state_dimension + column - parameter_start,
            ))
        } else {
            Err(self.index_error("column", column))
        }
    }

    fn index_error(self, axis: &str, index: usize) -> BvpSciNewError {
        BvpSciNewError::LinearBackend {
            message: format!(
                "bordered-banded {axis} index {index} is outside dimension {}",
                self.core_dimension + self.border_dimension
            ),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum RowPart {
    Core(usize),
    Border(usize),
}

#[derive(Clone, Copy, Debug)]
enum ColumnPart {
    Core(usize),
    Border(usize),
}

/// Native collocation storage with a structured bordered-banded production
/// route and a scalar-band compatibility route for direct low-level callers.
#[derive(Clone, Debug)]
pub struct BandedBackend {
    dimension: usize,
    requested_lower: usize,
    requested_upper: usize,
    lower: usize,
    upper: usize,
    storage: Storage,
}

impl BandedBackend {
    /// Construct the scalar-band compatibility backend.
    pub fn new(dimension: usize, lower: usize, upper: usize) -> Result<Self, BvpSciNewError> {
        let (lower, upper, matrix, factorization) = scalar_storage(dimension, lower, upper)?;
        Ok(Self {
            dimension,
            requested_lower: lower,
            requested_upper: upper,
            lower,
            upper,
            storage: Storage::Scalar {
                matrix,
                factorization,
            },
        })
    }

    /// Construct the production collocation backend without global format
    /// conversion. Numeric entries are repartitioned into `T/U/V/D` directly.
    pub fn new_for_collocation(
        dimension: usize,
        lower: usize,
        upper: usize,
        state_dimension: usize,
        node_count: usize,
        parameter_dimension: usize,
        telemetry: BvpSciTelemetry,
    ) -> Result<Self, BvpSciNewError> {
        if lower == usize::MAX || upper == usize::MAX {
            return Err(BvpSciNewError::InvalidConfiguration(
                "banded backend bandwidth must be finite".into(),
            ));
        }
        let partition =
            CollocationPartition::new(dimension, state_dimension, node_count, parameter_dimension)?;
        let system = BorderedBlockTridiagonal::zeros(
            partition.core_blocks(),
            partition.core_block_size(),
            partition.border_dimension,
        )
        .map_err(|error| map_banded_error("bordered construction", error))?;
        let reordered_rhs = vec![0.0; dimension];
        let original_rhs = vec![0.0; dimension];
        // A compact full-band fallback is retained only for small systems.
        // It is assembled directly from the same structural entries, never
        // produced by converting the structured matrix after factorization.
        // Large systems stay strictly on the bordered block path.
        let scalar_fallback = if dimension <= 256 {
            let matrix = scalar_matrix(dimension, dimension - 1, dimension - 1)?;
            let mut factorization = GeneralBandedLuPartialPivot::new(
                dimension,
                dimension - 1,
                dimension - 1,
            )
            .map_err(|error| {
                BvpSciNewError::InvalidConfiguration(format!(
                    "invalid BVP scalar fallback factorization: {error:?}"
                ))
            })?;
            // This route is deliberately restricted to small systems.  Its
            // partial-pivot workspace is the robust O(n^3) safety net for a
            // wide scalar band; the normal production route remains the
            // compact bordered structured factorization above.
            factorization.set_pivot_epsilon(0.0);
            Some(ScalarFallback {
                matrix,
                factorization,
            })
        } else {
            None
        };
        Ok(Self {
            dimension,
            requested_lower: lower,
            requested_upper: upper,
            lower: lower.min(dimension - 1),
            upper: upper.min(dimension - 1),
            storage: Storage::Bordered {
                system,
                partition,
                reordered_rhs,
                original_rhs,
                scalar_fallback,
                use_scalar_fallback: false,
                scalar_fallback_factorized: false,
                sparse_fallback: SparseSafetyFallback::default(),
                use_sparse_fallback: false,
                telemetry,
            },
        })
    }

    pub fn bandwidth(&self) -> (usize, usize) {
        (self.lower, self.upper)
    }

    /// Returns whether this instance is using the collocation Schur route.
    pub fn uses_bordered_structure(&self) -> bool {
        matches!(self.storage, Storage::Bordered { .. })
    }

    /// Return a copy of backend-local telemetry for diagnostics and compact
    /// stage benchmarks. The production solver normally reads the shared plan
    /// snapshot instead; exposing this narrow accessor keeps the microbench
    /// from reaching into storage internals.
    pub fn telemetry_snapshot(&self) -> Option<BvpSciTelemetrySnapshot> {
        match &self.storage {
            Storage::Scalar { .. } => None,
            Storage::Bordered { telemetry, .. } => Some(telemetry.snapshot()),
        }
    }

    pub fn assemble(&mut self, entries: &[(usize, usize, f64)]) -> Result<(), BvpSciNewError> {
        match &mut self.storage {
            Storage::Scalar { matrix, .. } => {
                matrix.as_mut_slice().fill(0.0);
                for &(row, column, value) in entries {
                    if !matrix.in_band(row, column) {
                        return Err(BvpSciNewError::LinearBackend {
                            message: format!(
                                "BVP global entry ({row}, {column}) is outside requested band ({}, {})",
                                self.lower, self.upper
                            ),
                        });
                    }
                    matrix[(row, column)] = value;
                }
            }
            Storage::Bordered {
                system,
                partition,
                scalar_fallback,
                use_scalar_fallback,
                scalar_fallback_factorized,
                sparse_fallback,
                use_sparse_fallback,
                telemetry,
                ..
            } => {
                *use_scalar_fallback = false;
                *scalar_fallback_factorized = false;
                *use_sparse_fallback = false;
                // Keep this separately visible: retaining original triplets
                // is the price of a lazy safety handoff, not sparse factor or
                // solve work. It can be removed or made opt-in once release
                // evidence shows the structured route is universally safe.
                let sparse_fallback_assembly_started = telemetry.start_timing();
                sparse_fallback.assemble(entries);
                telemetry.record_banded_sparse_fallback_assembly(
                    sparse_fallback_assembly_started,
                );
                system.clear_numeric();
                if let Some(fallback) = scalar_fallback {
                    let fallback_assembly_started = telemetry.start_timing();
                    fallback.matrix.as_mut_slice().fill(0.0);
                    // Reuse the original global coordinates for the scalar
                    // safety path. This is direct native assembly, not a
                    // conversion from the structured blocks.
                    for &(row, column, value) in entries {
                        // Match SparseBackend's triplet contract: repeated
                        // global coordinates are additive contributions.
                        fallback.matrix[(row, column)] += value;
                    }
                    telemetry.record_banded_scalar_fallback_assembly(
                        fallback_assembly_started,
                    );
                }
                for &(row, column, value) in entries {
                    let row = partition.row(row)?;
                    let column = partition.column(column)?;
                    set_structured_entry(system, row, column, value)?;
                }
            }
        }
        Ok(())
    }

    pub fn factor(&mut self) -> Result<(), BvpSciNewError> {
        match &mut self.storage {
            Storage::Scalar {
                matrix,
                factorization,
            } => {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    factorization.factor_from(matrix)
                }));
                match result {
                    Ok(Ok(())) => Ok(()),
                    Ok(Err(error)) => Err(map_banded_error("factorization", error)),
                    Err(_) => Err(BvpSciNewError::LinearBackend {
                        message: format!(
                            "banded LU factorization rejected configured bandwidth ({}, {})",
                            self.lower, self.upper
                        ),
                    }),
                }
            }
            Storage::Bordered {
                system,
                scalar_fallback,
                use_scalar_fallback,
                scalar_fallback_factorized,
                use_sparse_fallback,
                telemetry,
                ..
            } => {
                *use_scalar_fallback = false;
                *scalar_fallback_factorized = false;
                *use_sparse_fallback = false;
                let structured_started = telemetry.start_timing();
                let structured_result = system.factor();
                telemetry
                    .record_banded_factorization(BvpSciBandedRoute::Structured, structured_started);
                if let Some((factor_residual, min_abs_pivot, max_multiplier_norm)) =
                    system.core_factor_diagnostics()
                {
                    telemetry.record_banded_factor_diagnostics(
                        factor_residual,
                        min_abs_pivot,
                        max_multiplier_norm,
                    );
                }
                if let Some((min_abs_pivot, max_multiplier_norm)) =
                    system.border_factor_diagnostics()
                {
                    telemetry.record_banded_border_factor_diagnostics(
                        min_abs_pivot,
                        max_multiplier_norm,
                    );
                }
                let needs_fallback = structured_result.is_err();
                if needs_fallback {
                    if let Some(fallback) = scalar_fallback {
                        let fallback_started = telemetry.start_timing();
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            fallback.factorization.factor_from(&fallback.matrix)
                        }));
                        telemetry.record_banded_factorization(
                            BvpSciBandedRoute::ScalarFallback,
                            fallback_started,
                        );
                        match result {
                            Ok(Ok(())) => {
                                *use_scalar_fallback = true;
                                *scalar_fallback_factorized = true;
                                return Ok(());
                            }
                            Ok(Err(error)) => {
                                return Err(map_banded_error(
                                    "scalar fallback factorization",
                                    error,
                                ));
                            }
                            Err(_) => {
                                return Err(BvpSciNewError::LinearBackend {
                                    message: "banded scalar fallback rejected the factorization workspace".into(),
                                });
                            }
                        }
                    }
                }
                structured_result.map_err(|error| map_banded_error("bordered factorization", error))
            }
        }
    }

    pub fn solve_in_place(&mut self, rhs: &mut [f64]) -> Result<(), BvpSciNewError> {
        if rhs.len() != self.dimension {
            return Err(BvpSciNewError::LinearBackend {
                message: "banded solve vector has an invalid length".into(),
            });
        }
        match &mut self.storage {
            Storage::Scalar { factorization, .. } => {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    factorization.solve_in_place(rhs)
                }));
                match result {
                    Ok(Ok(())) => Ok(()),
                    Ok(Err(error)) => Err(map_banded_error("solve", error)),
                    Err(_) => Err(BvpSciNewError::LinearBackend {
                        message: "banded solve rejected the factorization workspace".into(),
                    }),
                }
            }
            Storage::Bordered {
                system,
                partition,
                reordered_rhs,
                original_rhs,
                scalar_fallback,
                use_scalar_fallback,
                scalar_fallback_factorized,
                sparse_fallback,
                use_sparse_fallback,
                telemetry,
            } => {
                if *use_scalar_fallback {
                    let fallback =
                        scalar_fallback
                            .as_mut()
                            .ok_or_else(|| BvpSciNewError::LinearBackend {
                                message: "banded fallback was selected without storage".into(),
                            })?;
                    if !*scalar_fallback_factorized {
                        let fallback_started = telemetry.start_timing();
                        fallback
                            .factorization
                            .factor_from(&fallback.matrix)
                            .map_err(|error| {
                                map_banded_error("scalar fallback factorization", error)
                            })?;
                        telemetry.record_banded_factorization(
                            BvpSciBandedRoute::ScalarFallback,
                            fallback_started,
                        );
                        *scalar_fallback_factorized = true;
                    }
                    let fallback_started = telemetry.start_timing();
                    let result = fallback.factorization.solve_in_place(rhs);
                    telemetry
                        .record_banded_solve(BvpSciBandedRoute::ScalarFallback, fallback_started);
                    return result
                        .map_err(|error| map_banded_error("scalar fallback solve", error));
                }
                if *use_sparse_fallback {
                    let sparse_started = telemetry.start_timing();
                    sparse_fallback.solve_in_place(self.dimension, rhs)?;
                    telemetry.record_banded_solve(
                        BvpSciBandedRoute::SparseFallback,
                        sparse_started,
                    );
                    return Ok(());
                }
                // The structured factorization uses [core, border] ordering,
                // while the collocation residual uses the original global
                // row ordering. Reorder into the reusable scratch buffer
                // before solving and scatter back to the global column order
                // afterwards. This keeps the permutation off the heap hot
                // path: only scalar copies are performed per Newton solve.
                let permutation_started = telemetry.start_timing();
                for (global_index, value) in rhs.iter().copied().enumerate() {
                    let part = partition.row(global_index)?;
                    match part {
                        RowPart::Core(index) => reordered_rhs[index] = value,
                        RowPart::Border(index) => {
                            reordered_rhs[partition.core_dimension + index] = value
                        }
                    }
                }
                telemetry.record_banded_rhs_permutation(rhs.len(), permutation_started);
                original_rhs.copy_from_slice(reordered_rhs);
                let structured_started = telemetry.start_timing();
                let structured_result = system.solve_in_place(reordered_rhs);
                telemetry.record_banded_solve(BvpSciBandedRoute::Structured, structured_started);
                if let Err(_error) = structured_result {
                    // A factorization can succeed while a later RHS exposes
                    // a zero pivot or numerical breakdown. The residual guard
                    // below cannot run in that case, so use the native scalar
                    // safety matrix immediately instead of leaking a false
                    // SingularJacobian to the nonlinear controller.
                    if let Some(fallback) = scalar_fallback {
                        let fallback_factor_started = telemetry.start_timing();
                        fallback
                            .factorization
                            .factor_from(&fallback.matrix)
                            .map_err(|fallback_error| {
                                map_banded_error("scalar fallback factorization", fallback_error)
                            })?;
                        telemetry.record_banded_factorization(
                            BvpSciBandedRoute::ScalarFallback,
                            fallback_factor_started,
                        );
                        let fallback_solve_started = telemetry.start_timing();
                        fallback
                            .factorization
                            .solve_in_place(rhs)
                            .map_err(|fallback_error| {
                                map_banded_error("scalar fallback solve", fallback_error)
                            })?;
                        telemetry.record_banded_solve(
                            BvpSciBandedRoute::ScalarFallback,
                            fallback_solve_started,
                        );
                        telemetry.record_banded_fallback_switch();
                        *use_scalar_fallback = true;
                        *scalar_fallback_factorized = true;
                        return Ok(());
                    }
                    let sparse_factor_started = telemetry.start_timing();
                    sparse_fallback.factor(self.dimension)?;
                    telemetry.record_banded_factorization(
                        BvpSciBandedRoute::SparseFallback,
                        sparse_factor_started,
                    );
                    let sparse_solve_started = telemetry.start_timing();
                    sparse_fallback.solve_in_place(self.dimension, rhs)?;
                    telemetry.record_banded_solve(
                        BvpSciBandedRoute::SparseFallback,
                        sparse_solve_started,
                    );
                    telemetry.record_banded_fallback_switch();
                    *use_sparse_fallback = true;
                    return Ok(());
                }
                let residual_started = telemetry.start_timing();
                let residual = system
                    .residual_inf(reordered_rhs, original_rhs)
                    .map_err(|error| map_banded_error("bordered residual", error))?;
                let solution_inf = reordered_rhs
                    .iter()
                    .copied()
                    .map(f64::abs)
                    .fold(0.0, f64::max);
                telemetry.record_banded_solve_diagnostics(residual, solution_inf);
                telemetry.record_banded_residual_guard(residual_started);
                if !residual.is_finite() || residual > 1.0e-8 {
                    if let Some(fallback) = scalar_fallback {
                        let fallback_factor_started = telemetry.start_timing();
                        fallback
                            .factorization
                            .factor_from(&fallback.matrix)
                            .map_err(|error| {
                                map_banded_error("scalar fallback factorization", error)
                            })?;
                        telemetry.record_banded_factorization(
                            BvpSciBandedRoute::ScalarFallback,
                            fallback_factor_started,
                        );
                        let fallback_solve_started = telemetry.start_timing();
                        fallback
                            .factorization
                            .solve_in_place(rhs)
                            .map_err(|error| map_banded_error("scalar fallback solve", error))?;
                        telemetry.record_banded_solve(
                            BvpSciBandedRoute::ScalarFallback,
                            fallback_solve_started,
                        );
                        telemetry.record_banded_fallback_switch();
                        *use_scalar_fallback = true;
                        *scalar_fallback_factorized = true;
                        return Ok(());
                    }
                    let sparse_factor_started = telemetry.start_timing();
                    sparse_fallback.factor(self.dimension)?;
                    telemetry.record_banded_factorization(
                        BvpSciBandedRoute::SparseFallback,
                        sparse_factor_started,
                    );
                    let sparse_solve_started = telemetry.start_timing();
                    sparse_fallback.solve_in_place(self.dimension, rhs)?;
                    telemetry.record_banded_solve(
                        BvpSciBandedRoute::SparseFallback,
                        sparse_solve_started,
                    );
                    telemetry.record_banded_fallback_switch();
                    *use_sparse_fallback = true;
                    return Ok(());
                }
                for global_index in 0..rhs.len() {
                    let part = partition.column(global_index)?;
                    rhs[global_index] = match part {
                        ColumnPart::Core(index) => reordered_rhs[index],
                        ColumnPart::Border(index) => {
                            reordered_rhs[partition.core_dimension + index]
                        }
                    };
                }
                Ok(())
            }
        }
    }
}

fn scalar_storage(
    dimension: usize,
    lower: usize,
    upper: usize,
) -> Result<(usize, usize, Banded<f64>, LapackStyleBandedLuFaithful), BvpSciNewError> {
    if dimension == 0 {
        return Err(BvpSciNewError::InvalidConfiguration(
            "banded backend dimension must be positive".into(),
        ));
    }
    if lower == usize::MAX || upper == usize::MAX {
        return Err(BvpSciNewError::InvalidConfiguration(
            "banded backend bandwidth must be finite".into(),
        ));
    }
    let actual_lower = lower.min(dimension - 1);
    let actual_upper = upper.min(dimension - 1);
    let matrix = Banded::zeros(dimension, actual_lower, actual_upper).map_err(|error| {
        BvpSciNewError::InvalidConfiguration(format!(
            "invalid BVP band storage ({dimension}, {lower}, {upper}): {error:?}"
        ))
    })?;
    let factorization = LapackStyleBandedLuFaithful::new(dimension, actual_lower, actual_upper)
        .map_err(|error| {
            BvpSciNewError::InvalidConfiguration(format!(
                "invalid BVP band factorization ({dimension}, {lower}, {upper}): {error:?}"
            ))
        })?;
    Ok((actual_lower, actual_upper, matrix, factorization))
}

fn scalar_matrix(
    dimension: usize,
    lower: usize,
    upper: usize,
) -> Result<Banded<f64>, BvpSciNewError> {
    if dimension == 0 {
        return Err(BvpSciNewError::InvalidConfiguration(
            "banded fallback dimension must be positive".into(),
        ));
    }
    let actual_lower = lower.min(dimension - 1);
    let actual_upper = upper.min(dimension - 1);
    Banded::zeros(dimension, actual_lower, actual_upper).map_err(|error| {
        BvpSciNewError::InvalidConfiguration(format!(
            "invalid BVP scalar fallback storage ({dimension}, {lower}, {upper}): {error:?}"
        ))
    })
}

fn set_structured_entry(
    system: &mut BorderedBlockTridiagonal,
    row: RowPart,
    column: ColumnPart,
    value: f64,
) -> Result<(), BvpSciNewError> {
    match (row, column) {
        (RowPart::Core(row), ColumnPart::Core(column)) => {
            let block_size = system.core_mut().block_size();
            let block_row = row / block_size;
            let block_column = column / block_size;
            let local_row = row % block_size;
            let local_column = column % block_size;
            let index = local_row * block_size + local_column;
            let blocks = if block_row == block_column {
                system.core_mut().diag_blocks_mut().get_mut(block_row)
            } else if block_row + 1 == block_column {
                system.core_mut().upper_blocks_mut().get_mut(block_row)
            } else if block_column + 1 == block_row {
                system.core_mut().lower_blocks_mut().get_mut(block_column)
            } else {
                None
            };
            let block = blocks.ok_or_else(|| BvpSciNewError::LinearBackend {
                message: format!(
                    "collocation core entry ({row}, {column}) is outside block-tridiagonal structure"
                ),
            })?;
            // Multiple collocation intervals contribute to shared node
            // columns. Overwriting here silently drops part of the Newton
            // Jacobian and becomes catastrophic on long meshes.
            block[index] += value;
        }
        (RowPart::Core(row), ColumnPart::Border(column)) => {
            let border_size = system.border_dimension();
            system.core_to_border_mut()[row * border_size + column] += value;
        }
        (RowPart::Border(row), ColumnPart::Core(column)) => {
            let core_size = system.core_dimension();
            system.border_to_core_mut()[row * core_size + column] += value;
        }
        (RowPart::Border(row), ColumnPart::Border(column)) => {
            let border_size = system.border_dimension();
            system.border_mut()[row * border_size + column] += value;
        }
    }
    Ok(())
}

fn map_banded_error(operation: &str, error: BandedError) -> BvpSciNewError {
    match error {
        BandedError::Singular { .. } | BandedError::ZeroPivot { .. } => {
            BvpSciNewError::SingularJacobian
        }
        other => BvpSciNewError::LinearBackend {
            message: format!("banded LU {operation} failed: {other:?}"),
        },
    }
}

impl LinearSystemBackend for BandedBackend {
    fn layout(&self) -> BvpSciMatrixLayout {
        BvpSciMatrixLayout::Banded {
            lower: self.requested_lower,
            upper: self.requested_upper,
        }
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn assemble(&mut self, entries: &[(usize, usize, f64)]) -> Result<(), BvpSciNewError> {
        Self::assemble(self, entries)
    }

    fn factor(&mut self) -> Result<(), BvpSciNewError> {
        Self::factor(self)
    }

    fn solve_in_place(&mut self, rhs: &mut [f64]) -> Result<(), BvpSciNewError> {
        Self::solve_in_place(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::{BandedBackend, CollocationPartition, ColumnPart, RowPart};
    use crate::numerical::BVP_sci::new::telemetry::BvpSciTelemetry;

    #[test]
    fn bordered_collocation_partition_matches_known_large_solution() {
        // 260 unknowns deliberately exceed the scalar-fallback threshold. The
        // test therefore exercises the same structured route as large BVPs,
        // while the manufactured RHS gives us an exact independent oracle.
        let state_dimension = 2;
        let node_count = 130;
        let parameter_dimension = 0;
        let dimension = state_dimension * node_count + parameter_dimension;
        let partition = CollocationPartition::new(
            dimension,
            state_dimension,
            node_count,
            parameter_dimension,
        )
        .unwrap();
        let expected = (0..dimension)
            .map(|index| 0.5 + 0.001 * index as f64)
            .collect::<Vec<_>>();
        let mut entries = Vec::new();

        for row in 0..dimension {
            for column in 0..dimension {
                let row_part = partition.row(row).unwrap();
                let column_part = partition.column(column).unwrap();
                let value = match (row_part, column_part) {
                    (RowPart::Core(row), ColumnPart::Core(column)) => {
                        let block_row = row / state_dimension;
                        let block_column = column / state_dimension;
                        if block_row.abs_diff(block_column) > 1 {
                            0.0
                        } else if block_row == block_column {
                            if row % state_dimension == column % state_dimension {
                                4.0 + 0.001 * row as f64
                            } else {
                                0.05
                            }
                        } else {
                            0.10
                        }
                    }
                    (RowPart::Core(_), ColumnPart::Border(_))
                    | (RowPart::Border(_), ColumnPart::Core(_)) => 0.001,
                    (RowPart::Border(row), ColumnPart::Border(column)) => {
                        if row == column {
                            2.0
                        } else {
                            0.02
                        }
                    }
                };
                if value != 0.0 {
                    entries.push((row, column, value));
                }
            }
        }

        let mut rhs = vec![0.0; dimension];
        for &(row, column, value) in &entries {
            rhs[row] += value * expected[column];
        }

        let mut backend = BandedBackend::new_for_collocation(
            dimension,
            1,
            1,
            state_dimension,
            node_count,
            parameter_dimension,
            BvpSciTelemetry::disabled(),
        )
        .unwrap();
        backend.assemble(&entries).unwrap();
        backend.factor().unwrap();
        backend.solve_in_place(&mut rhs).unwrap();

        let max_difference = rhs
            .iter()
            .zip(expected.iter())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_difference < 1e-9,
            "structured BVP solve mismatch: max_difference={max_difference:e}"
        );
    }
}

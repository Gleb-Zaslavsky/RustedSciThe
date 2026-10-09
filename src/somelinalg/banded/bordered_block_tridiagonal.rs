//! Direct solver for a block-tridiagonal core with a dense border.
//!
//! BVP collocation systems are usually block-tridiagonal in the interior, but
//! boundary equations couple the first and last mesh nodes. Treating those
//! entries as a scalar band would make the global bandwidth grow with the
//! mesh. This module keeps the interior factorization compact and handles the
//! small boundary/parameter block through a Schur complement instead.

use super::{
    block_tridiagonal::BlockTridiagonal,
    block_tridiagonal_lu_consistent::BlockTridiagonalLuConsistent, error::BandedError,
    general_lu_partial_pivot::GeneralBandedLuPartialPivot, storage::Banded,
};

/// Factorization of
///
/// ```text
/// [ T U ] [ x ] = [ r_core ]
/// [ V D ] [ z ]   [ r_border]
/// ```
///
/// where `T` is block-tridiagonal and the border dimension is expected to be
/// small relative to the number of mesh blocks. The factorization stores
/// `T^-1 U` and the Schur factorization, so repeated Newton solves reuse both
/// the expensive interior factorization and the border reduction.
#[derive(Clone, Debug)]
pub struct BorderedBlockTridiagonal {
    core: BlockTridiagonal,
    core_lu: BlockTridiagonalLuConsistent,
    core_to_border: Vec<f64>,
    core_to_border_solved: Vec<f64>,
    border_to_core: Vec<f64>,
    border_matrix: Vec<f64>,
    schur_matrix: Banded<f64>,
    border_lu: GeneralBandedLuPartialPivot,
    core_workspace: Vec<f64>,
    border_workspace: Vec<f64>,
    original_rhs: Vec<f64>,
    border_size: usize,
    is_factorized: bool,
}

impl BorderedBlockTridiagonal {
    /// Allocate a zero system with `border_size` dense border unknowns.
    pub fn zeros(
        n_blocks: usize,
        block_size: usize,
        border_size: usize,
    ) -> Result<Self, BandedError> {
        if border_size == 0 {
            return Err(BandedError::DimensionMismatch);
        }
        let core = BlockTridiagonal::zeros(n_blocks, block_size)?;
        let core_dimension = core.n();
        let core_lu = BlockTridiagonalLuConsistent::new(n_blocks, block_size)?;
        let core_border_len = core_dimension
            .checked_mul(border_size)
            .ok_or(BandedError::DimensionMismatch)?;
        let border_len = border_size
            .checked_mul(border_size)
            .ok_or(BandedError::DimensionMismatch)?;
        let schur_matrix = Banded::zeros(border_size, border_size - 1, border_size - 1)?;
        let border_lu =
            GeneralBandedLuPartialPivot::new(border_size, border_size - 1, border_size - 1)?;
        Ok(Self {
            core,
            core_lu,
            core_to_border: vec![0.0; core_border_len],
            core_to_border_solved: vec![0.0; core_border_len],
            border_to_core: vec![0.0; core_border_len],
            border_matrix: vec![0.0; border_len],
            schur_matrix,
            border_lu,
            core_workspace: vec![0.0; core_dimension],
            border_workspace: vec![0.0; border_size],
            original_rhs: vec![0.0; core_dimension + border_size],
            border_size,
            is_factorized: false,
        })
    }

    /// Mutable access to the compact interior block-tridiagonal matrix.
    pub fn core_mut(&mut self) -> &mut BlockTridiagonal {
        &mut self.core
    }

    /// Mutable row-major access to `U`, with shape `core_n x border_size`.
    pub fn core_to_border_mut(&mut self) -> &mut [f64] {
        &mut self.core_to_border
    }

    /// Mutable row-major access to `V`, with shape `border_size x core_n`.
    pub fn border_to_core_mut(&mut self) -> &mut [f64] {
        &mut self.border_to_core
    }

    /// Mutable row-major access to the dense `D` border block.
    pub fn border_mut(&mut self) -> &mut [f64] {
        &mut self.border_matrix
    }

    /// Clear numeric coefficients before a new Newton/Jacobian assembly.
    ///
    /// The allocation and block structure are retained. This is deliberately
    /// separate from construction so a continuation or mesh-local Jacobian
    /// refresh does not rebuild the structured workspace.
    pub fn clear_numeric(&mut self) {
        for block in self.core.diag_blocks_mut() {
            block.fill(0.0);
        }
        for block in self.core.lower_blocks_mut() {
            block.fill(0.0);
        }
        for block in self.core.upper_blocks_mut() {
            block.fill(0.0);
        }
        self.core_to_border.fill(0.0);
        self.core_to_border_solved.fill(0.0);
        self.border_to_core.fill(0.0);
        self.border_matrix.fill(0.0);
        self.schur_matrix.as_mut_slice().fill(0.0);
        self.original_rhs.fill(0.0);
        self.is_factorized = false;
    }

    /// Number of scalar unknowns in the block-tridiagonal core.
    #[inline]
    pub fn core_dimension(&self) -> usize {
        self.core.n()
    }

    /// Number of dense border unknowns.
    #[inline]
    pub fn border_dimension(&self) -> usize {
        self.border_size
    }

    /// Relative factorization residual of the structured core `T`.
    ///
    /// BVP backends use this diagnostic to detect compact, ill-conditioned
    /// superblock chains where a scalar-band fallback is safer. It is kept on
    /// the structured primitive so the decision does not require rebuilding a
    /// global matrix outside the linear backend.
    pub fn core_factor_residual_relative(&self) -> Result<f64, BandedError> {
        self.core_lu.factor_residual_relative(&self.core)
    }

    /// Return cached factor quality diagnostics for telemetry and route
    /// selection without reconstructing the global collocation matrix.
    pub fn core_factor_diagnostics(&self) -> Option<(f64, f64, f64)> {
        let residual = self.core_lu.cached_factor_residual_relative()?;
        let (min_abs_pivot, max_multiplier_norm) = self.core_lu.condition_diagnostics();
        Some((residual, min_abs_pivot, max_multiplier_norm))
    }

    /// Return pivots and multiplier growth for the dense border Schur factor.
    ///
    /// The interior core can remain perfectly well-conditioned while the
    /// endpoint/parameter Schur complement is nearly singular. Keeping this
    /// diagnostic next to the factorization makes that distinction visible.
    pub fn border_factor_diagnostics(&self) -> Option<(f64, f64)> {
        self.border_lu.condition_diagnostics()
    }

    /// Return the infinity-norm residual of a structured solution without
    /// materializing the corresponding global matrix.
    pub fn residual_inf(&self, solution: &[f64], rhs: &[f64]) -> Result<f64, BandedError> {
        let core_dimension = self.core_dimension();
        let expected = core_dimension
            .checked_add(self.border_size)
            .ok_or(BandedError::DimensionMismatch)?;
        if solution.len() != expected || rhs.len() != expected {
            return Err(BandedError::DimensionMismatch);
        }
        let (core_solution, border_solution) = solution.split_at(core_dimension);
        let (core_rhs, border_rhs) = rhs.split_at(core_dimension);
        let bs = self.core.block_size();
        let mut residual = 0.0_f64;
        for block in 0..self.core.n_blocks() {
            for local_row in 0..bs {
                let row = block * bs + local_row;
                let mut value = 0.0;
                for local_column in 0..bs {
                    value += self.core.diag_blocks()[block][local_row * bs + local_column]
                        * core_solution[block * bs + local_column];
                }
                if block > 0 {
                    for local_column in 0..bs {
                        value += self.core.lower_blocks()[block - 1][local_row * bs + local_column]
                            * core_solution[(block - 1) * bs + local_column];
                    }
                }
                if block + 1 < self.core.n_blocks() {
                    for local_column in 0..bs {
                        value += self.core.upper_blocks()[block][local_row * bs + local_column]
                            * core_solution[(block + 1) * bs + local_column];
                    }
                }
                for border in 0..self.border_size {
                    value += self.core_to_border[row * self.border_size + border]
                        * border_solution[border];
                }
                residual = residual.max((value - core_rhs[row]).abs());
            }
        }
        for row in 0..self.border_size {
            let mut value = 0.0;
            for column in 0..core_dimension {
                value += self.border_to_core[row * core_dimension + column] * core_solution[column];
            }
            for column in 0..self.border_size {
                value +=
                    self.border_matrix[row * self.border_size + column] * border_solution[column];
            }
            residual = residual.max((value - border_rhs[row]).abs());
        }
        Ok(residual)
    }

    /// Factor the core and the dense Schur complement in place.
    pub fn factor(&mut self) -> Result<(), BandedError> {
        self.is_factorized = false;
        self.core_lu.factor_from(&self.core)?;

        // Build T^-1 U one column at a time. Keep U intact so a later Newton
        // refactorization can reuse the same object with fresh coefficients.
        for column in 0..self.border_size {
            for row in 0..self.core_dimension() {
                self.core_workspace[row] = self.core_to_border[row * self.border_size + column];
            }
            // Block-local pivoting is intentionally retained, but stiff
            // superblock chains can amplify its residual. The existing
            // native refinement path corrects that residual against `T`
            // without materializing a dense global matrix.
            self.core_lu.solve_in_place_with_iterative_refinement(
                &self.core,
                &mut self.core_workspace,
                2,
            )?;
            for row in 0..self.core_dimension() {
                self.core_to_border_solved[row * self.border_size + column] =
                    self.core_workspace[row];
            }
        }

        self.schur_matrix.as_mut_slice().fill(0.0);
        for row in 0..self.border_size {
            for column in 0..self.border_size {
                let reduction = (0..self.core_dimension())
                    .map(|core_row| {
                        self.border_to_core[row * self.core_dimension() + core_row]
                            * self.core_to_border_solved[core_row * self.border_size + column]
                    })
                    .sum::<f64>();
                self.schur_matrix[(row, column)] =
                    self.border_matrix[row * self.border_size + column] - reduction;
            }
        }
        self.border_lu.factor_from(&self.schur_matrix)?;
        self.is_factorized = true;
        Ok(())
    }

    /// Solve the full system in place, with core variables first and border
    /// variables last. No global dense matrix is materialized.
    pub fn solve_in_place(&mut self, rhs: &mut [f64]) -> Result<(), BandedError> {
        if !self.is_factorized {
            return Err(BandedError::NotFactorized);
        }
        let expected = self
            .core_dimension()
            .checked_add(self.border_size)
            .ok_or(BandedError::DimensionMismatch)?;
        if rhs.len() != expected {
            return Err(BandedError::DimensionMismatch);
        }

        let (core_rhs, border_rhs) = rhs.split_at_mut(self.core_dimension());
        self.core_workspace.copy_from_slice(core_rhs);
        #[cfg(test)]
        let core_report = self
            .core_lu
            .solve_in_place_with_iterative_refinement_report(
                &self.core,
                &mut self.core_workspace,
                2,
            )?;
        #[cfg(not(test))]
        self.core_lu.solve_in_place_with_iterative_refinement(
            &self.core,
            &mut self.core_workspace,
            2,
        )?;
        #[cfg(test)]
        if std::env::var_os("BVP_SCI_DEBUG_BANDED_PARITY").is_some() {
            eprintln!(
                "BVP Banded core solve: direct_rr={:e} final_rr={:e} refinement_steps={} accepted={}",
                core_report.direct_relative_residual,
                core_report.final_relative_residual,
                core_report.requested_steps,
                core_report.accepted_steps,
            );
        }
        ensure_finite_solution("bordered core solve", &self.core_workspace)?;
        self.border_workspace.copy_from_slice(border_rhs);
        for row in 0..self.border_size {
            let correction = (0..self.core_dimension())
                .map(|core_column| {
                    self.border_to_core[row * self.core_dimension() + core_column]
                        * self.core_workspace[core_column]
                })
                .sum::<f64>();
            self.border_workspace[row] -= correction;
        }
        self.border_lu.solve_in_place(&mut self.border_workspace)?;
        ensure_finite_solution("bordered Schur solve", &self.border_workspace)?;

        for row in 0..self.core_dimension() {
            let correction = (0..self.border_size)
                .map(|column| {
                    self.core_to_border_solved[row * self.border_size + column]
                        * self.border_workspace[column]
                })
                .sum::<f64>();
            core_rhs[row] = self.core_workspace[row] - correction;
        }
        border_rhs.copy_from_slice(&self.border_workspace);
        ensure_finite_solution("bordered assembled solve", rhs)?;
        Ok(())
    }
}

fn ensure_finite_solution(stage: &'static str, values: &[f64]) -> Result<(), BandedError> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(BandedError::NonFiniteSolution {
            stage,
            index,
            value,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::BorderedBlockTridiagonal;
    use faer::prelude::Solve;
    use faer::sparse::{SparseColMat, Triplet};

    struct ParityCase {
        structured: BorderedBlockTridiagonal,
        triplets: Vec<Triplet<usize, usize, f64>>,
        rhs: Vec<f64>,
    }

    fn push_dense_block(
        triplets: &mut Vec<Triplet<usize, usize, f64>>,
        row0: usize,
        col0: usize,
        rows: usize,
        cols: usize,
        values: &[f64],
    ) {
        for row in 0..rows {
            for column in 0..cols {
                let value = values[row * cols + column];
                if value != 0.0 {
                    triplets.push(Triplet::new(row0 + row, col0 + column, value));
                }
            }
        }
    }

    fn parity_case_with_blocks(n_blocks: usize) -> ParityCase {
        let block_size = 2;
        let border_size = 2;
        let core_dimension = n_blocks * block_size;
        let mut structured =
            BorderedBlockTridiagonal::zeros(n_blocks, block_size, border_size).unwrap();
        let mut triplets = Vec::new();

        for block in 0..n_blocks {
            let diagonal = [4.5 + 0.1 * block as f64, 0.2, 0.1, 4.0 + 0.1 * block as f64];
            for row in 0..block_size {
                for column in 0..block_size {
                    structured
                        .core_mut()
                        .set_diag(block, row, column, diagonal[row * block_size + column])
                        .unwrap();
                }
            }
            push_dense_block(
                &mut triplets,
                block * block_size,
                block * block_size,
                block_size,
                block_size,
                &diagonal,
            );

            if block + 1 < n_blocks {
                let lower = [-0.4, 0.05, 0.0, -0.3];
                let upper = [-0.2, 0.0, 0.08, -0.25];
                for row in 0..block_size {
                    for column in 0..block_size {
                        structured
                            .core_mut()
                            .set_lower(block, row, column, lower[row * block_size + column])
                            .unwrap();
                        structured
                            .core_mut()
                            .set_upper(block, row, column, upper[row * block_size + column])
                            .unwrap();
                    }
                }
                push_dense_block(
                    &mut triplets,
                    (block + 1) * block_size,
                    block * block_size,
                    block_size,
                    block_size,
                    &lower,
                );
                push_dense_block(
                    &mut triplets,
                    block * block_size,
                    (block + 1) * block_size,
                    block_size,
                    block_size,
                    &upper,
                );
            }
        }

        let core_to_border = (0..core_dimension * border_size)
            .map(|index| {
                let row = index / border_size;
                let column = index % border_size;
                match (row + 2 * column) % 7 {
                    0 => 0.30,
                    1 => -0.10,
                    2 => 0.20,
                    3 => -0.20,
                    4 => 0.15,
                    5 => 0.10,
                    _ => 0.05,
                }
            })
            .collect::<Vec<_>>();
        structured
            .core_to_border_mut()
            .copy_from_slice(&core_to_border);
        push_dense_block(
            &mut triplets,
            0,
            core_dimension,
            core_dimension,
            border_size,
            &core_to_border,
        );

        let border_to_core = (0..border_size * core_dimension)
            .map(|index| {
                let row = index / core_dimension;
                let column = index % core_dimension;
                match (row + column) % 8 {
                    0 => 0.20,
                    1 => 0.00,
                    2 => -0.10,
                    3 => 0.15,
                    4 => 0.10,
                    5 => 0.05,
                    6 => -0.05,
                    _ => -0.10,
                }
            })
            .collect::<Vec<_>>();
        structured
            .border_to_core_mut()
            .copy_from_slice(&border_to_core);
        push_dense_block(
            &mut triplets,
            core_dimension,
            0,
            border_size,
            core_dimension,
            &border_to_core,
        );

        let border = [3.0, 0.2, 0.1, 2.5];
        structured.border_mut().copy_from_slice(&border);
        push_dense_block(
            &mut triplets,
            core_dimension,
            core_dimension,
            border_size,
            border_size,
            &border,
        );

        let expected = (0..core_dimension + border_size)
            .map(|index| 0.25 + 0.17 * index as f64)
            .collect::<Vec<_>>();
        let mut rhs = vec![0.0; expected.len()];
        for triplet in &triplets {
            rhs[triplet.row] += triplet.val * expected[triplet.col];
        }

        ParityCase {
            structured,
            triplets,
            rhs,
        }
    }

    #[test]
    fn bordered_block_tridiagonal_matches_known_three_by_three_system() {
        let mut system = BorderedBlockTridiagonal::zeros(2, 1, 1).unwrap();
        system.core_mut().set_diag(0, 0, 0, 4.0).unwrap();
        system.core_mut().set_diag(1, 0, 0, 3.0).unwrap();
        system.core_mut().set_upper(0, 0, 0, 1.0).unwrap();
        system.core_mut().set_lower(0, 0, 0, 2.0).unwrap();
        system.core_to_border_mut().copy_from_slice(&[1.0, 2.0]);
        system.border_to_core_mut().copy_from_slice(&[1.0, 0.0]);
        system.border_mut().copy_from_slice(&[2.0]);
        system.factor().unwrap();

        let mut rhs = vec![9.0, 14.0, 7.0];
        system.solve_in_place(&mut rhs).unwrap();
        assert!((rhs[0] - 1.0).abs() < 1e-12);
        assert!((rhs[1] - 2.0).abs() < 1e-12);
        assert!((rhs[2] - 3.0).abs() < 1e-12);

        // Refactorization must still use the original U, not T^-1 U from the
        // previous factorization.
        system.factor().unwrap();
        let mut rhs = vec![9.0, 14.0, 7.0];
        system.solve_in_place(&mut rhs).unwrap();
        assert!((rhs[0] - 1.0).abs() < 1e-12);
        assert!((rhs[1] - 2.0).abs() < 1e-12);
        assert!((rhs[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn bordered_block_tridiagonal_matches_faer_sparse_lu() {
        let case = parity_case_with_blocks(4);
        let n = case.rhs.len();
        let sparse =
            SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &case.triplets).unwrap();
        let sparse_lu = sparse.sp_lu().unwrap();
        let sparse_solution = sparse_lu.solve(&faer::Col::from_fn(n, |i| case.rhs[i]));

        let mut structured = case.structured;
        structured.factor().unwrap();
        let rhs = case.rhs;
        let mut structured_solution = rhs.clone();
        structured.solve_in_place(&mut structured_solution).unwrap();

        for (index, actual) in structured_solution.iter().enumerate() {
            assert!(
                (actual - sparse_solution[index]).abs() < 1e-10,
                "solution mismatch at {index}: structured={actual:e}, faer={:e}",
                sparse_solution[index]
            );
        }
    }

    #[test]
    fn bordered_block_tridiagonal_long_chain_matches_faer_sparse_lu() {
        let case = parity_case_with_blocks(128);
        let n = case.rhs.len();
        let sparse =
            SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &case.triplets).unwrap();
        let sparse_lu = sparse.sp_lu().unwrap();
        let sparse_solution = sparse_lu.solve(&faer::Col::from_fn(n, |i| case.rhs[i]));

        let rhs = case.rhs;
        let mut structured = case.structured;
        structured.factor().unwrap();
        let mut structured_solution = rhs.clone();
        structured.solve_in_place(&mut structured_solution).unwrap();

        let max_difference = structured_solution
            .iter()
            .enumerate()
            .map(|(index, actual)| (actual - sparse_solution[index]).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_difference < 1e-8,
            "long bordered solve mismatch: max_difference={max_difference:e}"
        );
        let residual = structured
            .residual_inf(&structured_solution, &rhs)
            .unwrap();
        assert!(residual < 1e-8, "long bordered residual={residual:e}");
    }
}

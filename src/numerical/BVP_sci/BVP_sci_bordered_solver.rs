//! Structural extraction for a future BVP_sci bordered-banded solver.
//!
//! The production BVP_sci Newton matrix is not just a scalar banded matrix:
//! collocation rows form a compact block-bidiagonal body, while the final
//! boundary-condition rows couple endpoint unknowns (and optional parameters).
//! This module keeps the next solver step honest by extracting that structure
//! explicitly before any specialized factorization is attempted.

use crate::numerical::BVP_sci::BVP_sci_faer::{faer_col, faer_dense_mat, faer_mat};
use nalgebra::{DMatrix, DVector, Dyn, LU};

#[derive(Clone, Debug)]
pub struct BvpSciBorderedBandedBlocks {
    pub variable_count: usize,
    pub mesh_points: usize,
    pub parameter_count: usize,
    pub diagonal_blocks: Vec<faer_dense_mat>,
    pub offdiag_blocks: Vec<faer_dense_mat>,
    pub collocation_parameter_blocks: Option<Vec<faer_dense_mat>>,
    pub boundary_left: faer_dense_mat,
    pub boundary_right: faer_dense_mat,
    pub boundary_parameters: Option<faer_dense_mat>,
}

impl BvpSciBorderedBandedBlocks {
    pub fn total_size(&self) -> usize {
        self.variable_count * self.mesh_points + self.parameter_count
    }

    pub fn collocation_rows(&self) -> usize {
        self.variable_count * self.mesh_points.saturating_sub(1)
    }

    pub fn boundary_rows(&self) -> usize {
        self.variable_count + self.parameter_count
    }

    /// Rebuild the dense global matrix represented by the extracted blocks.
    ///
    /// This is intended for tests and diagnostics, not as a production storage
    /// format.
    pub fn reconstruct_dense(&self) -> faer_dense_mat {
        let n = self.variable_count;
        let m = self.mesh_points;
        let mut dense = faer_dense_mat::zeros(self.total_size(), self.total_size());

        for interval in 0..m.saturating_sub(1) {
            let row_start = interval * n;
            let diag_col_start = interval * n;
            let offdiag_col_start = (interval + 1) * n;
            copy_block_into(
                &mut dense,
                row_start,
                diag_col_start,
                &self.diagonal_blocks[interval],
            );
            copy_block_into(
                &mut dense,
                row_start,
                offdiag_col_start,
                &self.offdiag_blocks[interval],
            );
            if let Some(param_blocks) = &self.collocation_parameter_blocks {
                copy_block_into(&mut dense, row_start, n * m, &param_blocks[interval]);
            }
        }

        let bc_row_start = self.collocation_rows();
        copy_block_into(&mut dense, bc_row_start, 0, &self.boundary_left);
        copy_block_into(
            &mut dense,
            bc_row_start,
            n * m.saturating_sub(1),
            &self.boundary_right,
        );
        if let Some(boundary_parameters) = &self.boundary_parameters {
            copy_block_into(&mut dense, bc_row_start, n * m, boundary_parameters);
        }

        dense
    }

    pub fn max_abs_diff_against_sparse(&self, sparse: &faer_mat) -> Result<f64, String> {
        let (rows, cols) = sparse.shape();
        if rows != self.total_size() || cols != self.total_size() {
            return Err(format!(
                "sparse matrix shape {}x{} does not match extracted BVP_sci layout {}x{}",
                rows,
                cols,
                self.total_size(),
                self.total_size()
            ));
        }
        let dense = self.reconstruct_dense();
        let mut max_diff = 0.0_f64;
        for row in 0..rows {
            for col in 0..cols {
                let sparse_value = *sparse.get(row, col).unwrap_or(&0.0);
                let diff = (*dense.get(row, col) - sparse_value).abs();
                max_diff = max_diff.max(diff);
            }
        }
        Ok(max_diff)
    }
}

/// Correctness-only reference solve for the extracted bordered-banded layout.
///
/// This intentionally reconstructs a dense matrix and uses nalgebra LU.  It is
/// not the final production backend; it is the oracle that lets us build and
/// test the specialized bordered solver without touching the Newton path yet.
pub fn solve_bordered_banded_reference(
    blocks: &BvpSciBorderedBandedBlocks,
    rhs: &faer_col,
) -> Result<faer_col, String> {
    validate_bordered_blocks(blocks)?;
    if rhs.nrows() != blocks.total_size() {
        return Err(format!(
            "BVP_sci bordered-banded reference solve expected rhs length {}, got {}",
            blocks.total_size(),
            rhs.nrows()
        ));
    }

    let dense = blocks.reconstruct_dense();
    let matrix = dense_to_nalgebra(&dense);
    let rhs = DVector::from_iterator(rhs.nrows(), (0..rhs.nrows()).map(|row| rhs[row]));
    let Some(solution) = matrix.lu().solve(&rhs) else {
        return Err("BVP_sci bordered-banded reference LU solve failed".to_string());
    };

    Ok(faer_col::from_fn(solution.nrows(), |row| solution[row]))
}

/// Solve the extracted bordered block-bidiagonal system without using Sparse LU.
///
/// This is the first native correctness implementation for the BVP_sci
/// bordered-banded route. It eliminates the collocation body interval by
/// interval:
///
/// `D_i y_i + U_i y_{i+1} + P_i p = r_i`
///
/// and expresses every `y_i` through the endpoint state `y_0` and optional
/// parameters `p`. The final boundary rows then form a dense `(n+k) x (n+k)`
/// system for `(y_0, p)`.
///
/// It is still a correctness path, not a performance-tuned production backend:
/// interval and border blocks are dense nalgebra matrices. That is deliberate
/// for now; the next step can replace the dense block operations while keeping
/// these Sparse-LU parity tests unchanged.
pub fn solve_bordered_banded_structured(
    blocks: &BvpSciBorderedBandedBlocks,
    rhs: &faer_col,
) -> Result<faer_col, String> {
    let factorization = factor_bordered_banded_structured(blocks)?;
    factorization.solve(rhs)
}

#[derive(Clone, Debug)]
pub struct BvpSciBorderedStructuredFactorization {
    variable_count: usize,
    mesh_points: usize,
    parameter_count: usize,
    offdiag_lus: Vec<LU<f64, Dyn, Dyn>>,
    diagonal_blocks: Vec<DMatrix<f64>>,
    state_from_y0: Vec<DMatrix<f64>>,
    state_from_params: Vec<DMatrix<f64>>,
    boundary_right: DMatrix<f64>,
    border_lu: LU<f64, Dyn, Dyn>,
}

impl BvpSciBorderedStructuredFactorization {
    pub fn total_size(&self) -> usize {
        self.variable_count * self.mesh_points + self.parameter_count
    }

    pub fn solve(&self, rhs: &faer_col) -> Result<faer_col, String> {
        let n = self.variable_count;
        let m = self.mesh_points;
        let k = self.parameter_count;
        if rhs.nrows() != self.total_size() {
            return Err(format!(
                "BVP_sci bordered-banded structured factorization expected rhs length {}, got {}",
                self.total_size(),
                rhs.nrows()
            ));
        }

        let mut state_offsets = Vec::with_capacity(m);
        state_offsets.push(DVector::<f64>::zeros(n));

        for interval in 0..m.saturating_sub(1) {
            let rhs_interval = rhs_segment(rhs, interval * n, n);
            let offset_rhs =
                rhs_interval - &self.diagonal_blocks[interval] * &state_offsets[interval];
            let Some(next_offset) = self.offdiag_lus[interval].solve(&offset_rhs) else {
                return Err(format!(
                    "BVP_sci bordered-banded structured solve failed: singular cached offdiag rhs solve at interval {interval}"
                ));
            };
            state_offsets.push(next_offset);
        }

        let last = m - 1;
        let border_rhs =
            rhs_segment(rhs, n * (m - 1), n + k) - &self.boundary_right * &state_offsets[last];

        let Some(endpoint_solution) = self.border_lu.solve(&border_rhs) else {
            return Err(
                "BVP_sci bordered-banded structured solve failed: singular cached border system"
                    .to_string(),
            );
        };
        let y0 = endpoint_solution.rows(0, n).into_owned();
        let params = if k == 0 {
            DVector::<f64>::zeros(0)
        } else {
            endpoint_solution.rows(n, k).into_owned()
        };

        let mut solution = faer_col::zeros(self.total_size());
        for node in 0..m {
            let state = &self.state_from_y0[node] * &y0
                + &self.state_from_params[node] * &params
                + &state_offsets[node];
            for row in 0..n {
                solution[node * n + row] = state[row];
            }
        }
        for row in 0..k {
            solution[n * m + row] = params[row];
        }

        Ok(solution)
    }
}

pub fn factor_bordered_banded_structured(
    blocks: &BvpSciBorderedBandedBlocks,
) -> Result<BvpSciBorderedStructuredFactorization, String> {
    validate_bordered_blocks(blocks)?;
    let n = blocks.variable_count;
    let m = blocks.mesh_points;
    let k = blocks.parameter_count;

    let mut state_from_y0 = Vec::with_capacity(m);
    let mut state_from_params = Vec::with_capacity(m);
    let mut offdiag_lus = Vec::with_capacity(m.saturating_sub(1));
    let mut diagonal_blocks = Vec::with_capacity(m.saturating_sub(1));
    state_from_y0.push(DMatrix::<f64>::identity(n, n));
    state_from_params.push(DMatrix::<f64>::zeros(n, k));

    for interval in 0..m.saturating_sub(1) {
        let diag = dense_to_nalgebra(&blocks.diagonal_blocks[interval]);
        let offdiag = dense_to_nalgebra(&blocks.offdiag_blocks[interval]);
        let offdiag_lu = offdiag.lu();

        let next_from_y0_rhs = -(&diag * &state_from_y0[interval]);
        let Some(next_from_y0) = offdiag_lu.solve(&next_from_y0_rhs) else {
            return Err(format!(
                "BVP_sci bordered-banded structured solve failed: singular offdiag block at interval {interval}"
            ));
        };

        let param_rhs = if k == 0 {
            DMatrix::<f64>::zeros(n, 0)
        } else {
            let param_block = blocks
                .collocation_parameter_blocks
                .as_ref()
                .and_then(|blocks| blocks.get(interval))
                .map(dense_to_nalgebra)
                .unwrap_or_else(|| DMatrix::<f64>::zeros(n, k));
            -((&diag * &state_from_params[interval]) + param_block)
        };
        let Some(next_from_params) = offdiag_lu.solve(&param_rhs) else {
            return Err(format!(
                "BVP_sci bordered-banded structured solve failed: singular offdiag parameter solve at interval {interval}"
            ));
        };

        state_from_y0.push(next_from_y0);
        state_from_params.push(next_from_params);
        diagonal_blocks.push(diag);
        offdiag_lus.push(offdiag_lu);
    }

    let boundary_left = dense_to_nalgebra(&blocks.boundary_left);
    let boundary_right = dense_to_nalgebra(&blocks.boundary_right);
    let boundary_params = blocks
        .boundary_parameters
        .as_ref()
        .map(dense_to_nalgebra)
        .unwrap_or_else(|| DMatrix::<f64>::zeros(n + k, k));

    let last = m - 1;
    let border_left = boundary_left + &boundary_right * &state_from_y0[last];
    let border_params = boundary_right.clone() * &state_from_params[last] + boundary_params;

    let mut border = DMatrix::<f64>::zeros(n + k, n + k);
    copy_nalgebra_block_into(&mut border, 0, 0, &border_left);
    if k > 0 {
        copy_nalgebra_block_into(&mut border, 0, n, &border_params);
    }

    let border_lu = border.lu();
    if !border_lu.is_invertible() {
        return Err(
            "BVP_sci bordered-banded structured solve failed: singular border system".to_string(),
        );
    }

    Ok(BvpSciBorderedStructuredFactorization {
        variable_count: n,
        mesh_points: m,
        parameter_count: k,
        offdiag_lus,
        diagonal_blocks,
        state_from_y0,
        state_from_params,
        boundary_right,
        border_lu,
    })
}

pub fn extract_bordered_banded_blocks(
    mat: &faer_mat,
    variable_count: usize,
    mesh_points: usize,
    parameter_count: usize,
) -> Result<BvpSciBorderedBandedBlocks, String> {
    if variable_count == 0 {
        return Err("BVP_sci bordered-banded extraction requires variable_count > 0".to_string());
    }
    if mesh_points < 2 {
        return Err(
            "BVP_sci bordered-banded extraction requires at least 2 mesh points".to_string(),
        );
    }

    let total_size = variable_count * mesh_points + parameter_count;
    let (rows, cols) = mat.shape();
    if rows != total_size || cols != total_size {
        return Err(format!(
            "BVP_sci bordered-banded extraction expected {}x{} matrix, got {}x{}",
            total_size, total_size, rows, cols
        ));
    }

    let n = variable_count;
    let m = mesh_points;
    let k = parameter_count;
    let mut diagonal_blocks = Vec::with_capacity(m - 1);
    let mut offdiag_blocks = Vec::with_capacity(m - 1);
    let mut collocation_parameter_blocks = if k > 0 {
        Some(Vec::with_capacity(m - 1))
    } else {
        None
    };

    for interval in 0..(m - 1) {
        let row_start = interval * n;
        diagonal_blocks.push(copy_dense_block(mat, row_start, interval * n, n, n));
        offdiag_blocks.push(copy_dense_block(mat, row_start, (interval + 1) * n, n, n));
        if let Some(param_blocks) = collocation_parameter_blocks.as_mut() {
            param_blocks.push(copy_dense_block(mat, row_start, n * m, n, k));
        }
    }

    let bc_row_start = n * (m - 1);
    let boundary_rows = n + k;
    let boundary_left = copy_dense_block(mat, bc_row_start, 0, boundary_rows, n);
    let boundary_right = copy_dense_block(mat, bc_row_start, n * (m - 1), boundary_rows, n);
    let boundary_parameters = if k > 0 {
        Some(copy_dense_block(mat, bc_row_start, n * m, boundary_rows, k))
    } else {
        None
    };

    Ok(BvpSciBorderedBandedBlocks {
        variable_count,
        mesh_points,
        parameter_count,
        diagonal_blocks,
        offdiag_blocks,
        collocation_parameter_blocks,
        boundary_left,
        boundary_right,
        boundary_parameters,
    })
}

fn copy_dense_block(
    mat: &faer_mat,
    row_start: usize,
    col_start: usize,
    rows: usize,
    cols: usize,
) -> faer_dense_mat {
    faer_dense_mat::from_fn(rows, cols, |row, col| {
        *mat.get(row_start + row, col_start + col).unwrap_or(&0.0)
    })
}

fn copy_block_into(
    target: &mut faer_dense_mat,
    row_start: usize,
    col_start: usize,
    block: &faer_dense_mat,
) {
    for row in 0..block.nrows() {
        for col in 0..block.ncols() {
            *target.get_mut(row_start + row, col_start + col) = *block.get(row, col);
        }
    }
}

fn dense_to_nalgebra(dense: &faer_dense_mat) -> DMatrix<f64> {
    DMatrix::from_fn(dense.nrows(), dense.ncols(), |row, col| {
        *dense.get(row, col)
    })
}

fn validate_bordered_blocks(blocks: &BvpSciBorderedBandedBlocks) -> Result<(), String> {
    let n = blocks.variable_count;
    let m = blocks.mesh_points;
    let k = blocks.parameter_count;
    if n == 0 {
        return Err("BVP_sci bordered-banded blocks require variable_count > 0".to_string());
    }
    if m < 2 {
        return Err("BVP_sci bordered-banded blocks require at least 2 mesh points".to_string());
    }
    let intervals = m - 1;
    if blocks.diagonal_blocks.len() != intervals {
        return Err(format!(
            "BVP_sci bordered-banded blocks expected {intervals} diagonal blocks, got {}",
            blocks.diagonal_blocks.len()
        ));
    }
    if blocks.offdiag_blocks.len() != intervals {
        return Err(format!(
            "BVP_sci bordered-banded blocks expected {intervals} offdiag blocks, got {}",
            blocks.offdiag_blocks.len()
        ));
    }

    for (index, block) in blocks.diagonal_blocks.iter().enumerate() {
        validate_dense_shape(block, n, n, "diagonal block", index)?;
    }
    for (index, block) in blocks.offdiag_blocks.iter().enumerate() {
        validate_dense_shape(block, n, n, "offdiag block", index)?;
    }

    match (&blocks.collocation_parameter_blocks, k) {
        (None, 0) => {}
        (Some(param_blocks), _) => {
            if param_blocks.len() != intervals {
                return Err(format!(
                    "BVP_sci bordered-banded blocks expected {intervals} collocation parameter blocks, got {}",
                    param_blocks.len()
                ));
            }
            for (index, block) in param_blocks.iter().enumerate() {
                validate_dense_shape(block, n, k, "collocation parameter block", index)?;
            }
        }
        (None, _) => {
            return Err(format!(
                "BVP_sci bordered-banded blocks require collocation parameter blocks for parameter_count={k}"
            ));
        }
    }

    validate_dense_shape(&blocks.boundary_left, n + k, n, "boundary-left block", 0)?;
    validate_dense_shape(&blocks.boundary_right, n + k, n, "boundary-right block", 0)?;
    match (&blocks.boundary_parameters, k) {
        (None, 0) => Ok(()),
        (Some(block), _) => validate_dense_shape(block, n + k, k, "boundary-parameter block", 0),
        (None, _) => Err(format!(
            "BVP_sci bordered-banded blocks require boundary parameter block for parameter_count={k}"
        )),
    }
}

fn validate_dense_shape(
    block: &faer_dense_mat,
    expected_rows: usize,
    expected_cols: usize,
    label: &str,
    index: usize,
) -> Result<(), String> {
    if block.nrows() == expected_rows && block.ncols() == expected_cols {
        return Ok(());
    }
    Err(format!(
        "BVP_sci bordered-banded {label} #{index} expected shape {}x{}, got {}x{}",
        expected_rows,
        expected_cols,
        block.nrows(),
        block.ncols()
    ))
}

fn rhs_segment(rhs: &faer_col, start: usize, len: usize) -> DVector<f64> {
    DVector::from_iterator(len, (0..len).map(|offset| rhs[start + offset]))
}

fn copy_nalgebra_block_into(
    target: &mut DMatrix<f64>,
    row_start: usize,
    col_start: usize,
    block: &DMatrix<f64>,
) {
    for row in 0..block.nrows() {
        for col in 0..block.ncols() {
            target[(row_start + row, col_start + col)] = block[(row, col)];
        }
    }
}

//! Structural extraction for a future BVP_sci bordered-banded solver.
//!
//! The production BVP_sci Newton matrix is not just a scalar banded matrix:
//! collocation rows form a compact block-bidiagonal body, while the final
//! boundary-condition rows couple endpoint unknowns (and optional parameters).
//! This module keeps the next solver step honest by extracting that structure
//! explicitly before any specialized factorization is attempted.

use crate::numerical::BVP_sci::BVP_sci_faer::{faer_col, faer_dense_mat, faer_mat};
use crate::somelinalg::banded::banded_assembly::BandedAssembly;
use crate::somelinalg::banded::dense_block_kernels::idx;
use crate::somelinalg::banded::{dense_lu_pivot_in_place, dense_lu_pivot_solve_in_place};
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
    backend: BvpSciBorderedStructuredBackend,
}

#[derive(Clone, Debug, Default)]
pub struct BvpSciBorderedSolveWorkspace {
    state_offsets: Vec<f64>,
    next_offset: Vec<f64>,
    endpoint_state: Vec<f64>,
    reconstructed_state: Vec<f64>,
}

#[derive(Clone, Debug)]
enum BvpSciBorderedStructuredBackend {
    NativeParameterFree(BvpSciNativeBorderedParameterFreeFactorization),
    Nalgebra(BvpSciNalgebraBorderedStructuredFactorization),
}

#[derive(Clone, Debug)]
struct BvpSciNalgebraBorderedStructuredFactorization {
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

#[derive(Clone, Debug)]
struct NativeDenseLu {
    lu: Vec<f64>,
    pivots: Vec<usize>,
}

#[derive(Clone, Debug)]
struct BvpSciNativeBorderedParameterFreeFactorization {
    variable_count: usize,
    mesh_points: usize,
    offdiag_lus: Vec<NativeDenseLu>,
    diagonal_blocks: Vec<Vec<f64>>,
    state_from_y0: Vec<Vec<f64>>,
    boundary_right: Vec<f64>,
    border_lu: NativeDenseLu,
}

impl BvpSciBorderedStructuredFactorization {
    pub fn total_size(&self) -> usize {
        match &self.backend {
            BvpSciBorderedStructuredBackend::NativeParameterFree(factorization) => {
                factorization.variable_count * factorization.mesh_points
            }
            BvpSciBorderedStructuredBackend::Nalgebra(factorization) => {
                factorization.variable_count * factorization.mesh_points
                    + factorization.parameter_count
            }
        }
    }

    pub fn solve(&self, rhs: &faer_col) -> Result<faer_col, String> {
        let mut workspace = self.new_workspace();
        self.solve_with_workspace(rhs, &mut workspace)
    }

    pub fn new_workspace(&self) -> BvpSciBorderedSolveWorkspace {
        match &self.backend {
            BvpSciBorderedStructuredBackend::NativeParameterFree(factorization) => {
                BvpSciBorderedSolveWorkspace {
                    state_offsets: vec![
                        0.0;
                        factorization.variable_count * factorization.mesh_points
                    ],
                    next_offset: vec![0.0; factorization.variable_count],
                    endpoint_state: vec![0.0; factorization.variable_count],
                    reconstructed_state: vec![0.0; factorization.variable_count],
                }
            }
            BvpSciBorderedStructuredBackend::Nalgebra(_) => BvpSciBorderedSolveWorkspace::default(),
        }
    }

    pub fn solve_with_workspace(
        &self,
        rhs: &faer_col,
        workspace: &mut BvpSciBorderedSolveWorkspace,
    ) -> Result<faer_col, String> {
        match &self.backend {
            BvpSciBorderedStructuredBackend::NativeParameterFree(factorization) => {
                factorization.solve_with_workspace(rhs, workspace)
            }
            BvpSciBorderedStructuredBackend::Nalgebra(factorization) => factorization.solve(rhs),
        }
    }
}

impl BvpSciNalgebraBorderedStructuredFactorization {
    fn solve(&self, rhs: &faer_col) -> Result<faer_col, String> {
        let n = self.variable_count;
        let m = self.mesh_points;
        let k = self.parameter_count;
        if rhs.nrows() != self.variable_count * self.mesh_points + self.parameter_count {
            return Err(format!(
                "BVP_sci bordered-banded structured factorization expected rhs length {}, got {}",
                self.variable_count * self.mesh_points + self.parameter_count,
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

        let mut solution = faer_col::zeros(n * m + k);
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

impl BvpSciNativeBorderedParameterFreeFactorization {
    fn solve_with_workspace(
        &self,
        rhs: &faer_col,
        workspace: &mut BvpSciBorderedSolveWorkspace,
    ) -> Result<faer_col, String> {
        let n = self.variable_count;
        let m = self.mesh_points;
        let total_size = n * m;
        if rhs.nrows() != total_size {
            return Err(format!(
                "BVP_sci bordered-banded structured factorization expected rhs length {}, got {}",
                total_size,
                rhs.nrows()
            ));
        }

        workspace.state_offsets.resize(total_size, 0.0);
        workspace.state_offsets.fill(0.0);
        workspace.next_offset.resize(n, 0.0);
        workspace.endpoint_state.resize(n, 0.0);
        workspace.reconstructed_state.resize(n, 0.0);

        for interval in 0..m.saturating_sub(1) {
            for row in 0..n {
                workspace.next_offset[row] = rhs[interval * n + row];
            }
            matvec_sub_assign(
                &mut workspace.next_offset,
                &self.diagonal_blocks[interval],
                &workspace.state_offsets[interval * n..(interval + 1) * n],
                n,
            );
            self.offdiag_lus[interval]
                .solve_in_place(&mut workspace.next_offset)
                .map_err(|err| {
                    format!(
                        "BVP_sci bordered-banded structured solve failed: cached native offdiag solve failed at interval {interval}: {err:?}"
                    )
                })?;
            workspace.state_offsets[(interval + 1) * n..(interval + 2) * n]
                .copy_from_slice(&workspace.next_offset);
        }

        let last = m - 1;
        for row in 0..n {
            workspace.endpoint_state[row] = rhs[n * (m - 1) + row];
        }
        matvec_sub_assign(
            &mut workspace.endpoint_state,
            &self.boundary_right,
            &workspace.state_offsets[last * n..(last + 1) * n],
            n,
        );
        self.border_lu
            .solve_in_place(&mut workspace.endpoint_state)
            .map_err(|err| {
            format!("BVP_sci bordered-banded structured solve failed: cached native border solve failed: {err:?}")
        })?;

        let mut solution = faer_col::zeros(total_size);
        for node in 0..m {
            matvec_into(
                &mut workspace.reconstructed_state,
                &self.state_from_y0[node],
                &workspace.endpoint_state,
                n,
            );
            for row in 0..n {
                workspace.reconstructed_state[row] += workspace.state_offsets[node * n + row];
                solution[node * n + row] = workspace.reconstructed_state[row];
            }
        }

        Ok(solution)
    }
}

impl NativeDenseLu {
    fn factor(mut matrix: Vec<f64>, size: usize) -> Result<Self, String> {
        let mut pivots = vec![0; size];
        dense_lu_pivot_in_place(&mut matrix, size, &mut pivots)
            .map_err(|err| format!("native dense block LU factorization failed: {err:?}"))?;
        Ok(Self { lu: matrix, pivots })
    }

    fn solve_in_place(
        &self,
        rhs: &mut [f64],
    ) -> Result<(), crate::somelinalg::banded::BandedError> {
        dense_lu_pivot_solve_in_place(&self.lu, self.pivots.len(), &self.pivots, rhs)
    }
}

pub fn factor_bordered_banded_structured(
    blocks: &BvpSciBorderedBandedBlocks,
) -> Result<BvpSciBorderedStructuredFactorization, String> {
    validate_bordered_blocks(blocks)?;
    let n = blocks.variable_count;
    let m = blocks.mesh_points;
    let k = blocks.parameter_count;

    if k == 0 {
        return factor_bordered_banded_parameter_free_native(blocks);
    }

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
        backend: BvpSciBorderedStructuredBackend::Nalgebra(
            BvpSciNalgebraBorderedStructuredFactorization {
                variable_count: n,
                mesh_points: m,
                parameter_count: k,
                offdiag_lus,
                diagonal_blocks,
                state_from_y0,
                state_from_params,
                boundary_right,
                border_lu,
            },
        ),
    })
}

fn factor_bordered_banded_parameter_free_native(
    blocks: &BvpSciBorderedBandedBlocks,
) -> Result<BvpSciBorderedStructuredFactorization, String> {
    let n = blocks.variable_count;
    let m = blocks.mesh_points;

    let mut state_from_y0 = Vec::with_capacity(m);
    let mut offdiag_lus = Vec::with_capacity(m.saturating_sub(1));
    let mut diagonal_blocks = Vec::with_capacity(m.saturating_sub(1));
    state_from_y0.push(identity_block(n));

    for interval in 0..m.saturating_sub(1) {
        let diag = dense_to_row_major(&blocks.diagonal_blocks[interval]);
        let offdiag = dense_to_row_major(&blocks.offdiag_blocks[interval]);
        let offdiag_lu = NativeDenseLu::factor(offdiag, n).map_err(|err| {
            format!(
                "BVP_sci bordered-banded structured solve failed: singular offdiag block at interval {interval}: native block LU error: {err}"
            )
        })?;

        let mut next_from_y0 = matmul(&diag, &state_from_y0[interval], n);
        for value in &mut next_from_y0 {
            *value = -*value;
        }
        solve_block_rhs_in_place(&offdiag_lu, &mut next_from_y0, n).map_err(|err| {
            format!(
                "BVP_sci bordered-banded structured solve failed: native offdiag block solve failed at interval {interval}: {err:?}"
            )
        })?;

        state_from_y0.push(next_from_y0);
        diagonal_blocks.push(diag);
        offdiag_lus.push(offdiag_lu);
    }

    let last = m - 1;
    let boundary_left = dense_to_row_major(&blocks.boundary_left);
    let boundary_right = dense_to_row_major(&blocks.boundary_right);
    let mut border = boundary_left;
    matmul_add_assign(&mut border, &boundary_right, &state_from_y0[last], n);
    let border_lu = NativeDenseLu::factor(border, n).map_err(|err| {
        format!(
            "BVP_sci bordered-banded structured solve failed: singular border system: native block LU error: {err}"
        )
    })?;

    Ok(BvpSciBorderedStructuredFactorization {
        backend: BvpSciBorderedStructuredBackend::NativeParameterFree(
            BvpSciNativeBorderedParameterFreeFactorization {
                variable_count: n,
                mesh_points: m,
                offdiag_lus,
                diagonal_blocks,
                state_from_y0,
                boundary_right,
                border_lu,
            },
        ),
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

/// Builds the parameter-free bordered collocation matrix directly from native
/// pointwise banded Jacobians.
///
/// This is the production assembly path for BVP_sci Banded AOT. It deliberately
/// avoids materializing the full global sparse Jacobian only to extract the
/// same dense state blocks again.
pub fn assemble_bordered_banded_blocks_from_pointwise(
    h: &faer_col,
    df_dy: &[BandedAssembly],
    df_dy_middle: &[BandedAssembly],
    dbc_dya: &faer_mat,
    dbc_dyb: &faer_mat,
) -> Result<BvpSciBorderedBandedBlocks, String> {
    let mesh_points = df_dy.len();
    if mesh_points < 2 {
        return Err("BVP_sci direct banded assembly requires at least 2 mesh points".to_string());
    }
    if df_dy_middle.len() != mesh_points - 1 || h.nrows() != mesh_points - 1 {
        return Err(format!(
            "BVP_sci direct banded assembly shape mismatch: points={}, middle={}, intervals={}",
            mesh_points,
            df_dy_middle.len(),
            h.nrows()
        ));
    }

    let variable_count = df_dy[0].n();
    if variable_count == 0
        || df_dy.iter().any(|jac| jac.n() != variable_count)
        || df_dy_middle.iter().any(|jac| jac.n() != variable_count)
    {
        return Err(
            "BVP_sci direct banded assembly requires equal non-empty pointwise Jacobians"
                .to_string(),
        );
    }
    if dbc_dya.shape() != (variable_count, variable_count)
        || dbc_dyb.shape() != (variable_count, variable_count)
    {
        return Err(format!(
            "BVP_sci direct banded assembly expected {}x{} boundary Jacobians",
            variable_count, variable_count
        ));
    }

    let mut diagonal_blocks = Vec::with_capacity(mesh_points - 1);
    let mut offdiag_blocks = Vec::with_capacity(mesh_points - 1);
    for interval in 0..mesh_points - 1 {
        let h_i = h[interval];
        let left = &df_dy[interval];
        let right = &df_dy[interval + 1];
        let middle = &df_dy_middle[interval];

        let diagonal = faer_dense_mat::from_fn(variable_count, variable_count, |row, col| {
            let identity = if row == col { -1.0 } else { 0.0 };
            identity
                - h_i / 6.0 * banded_value(left, row, col)
                - h_i / 3.0 * banded_value(middle, row, col)
        });
        let offdiag = faer_dense_mat::from_fn(variable_count, variable_count, |row, col| {
            let identity = if row == col { 1.0 } else { 0.0 };
            identity
                - h_i / 6.0 * banded_value(right, row, col)
                - h_i / 3.0 * banded_value(middle, row, col)
        });
        diagonal_blocks.push(diagonal);
        offdiag_blocks.push(offdiag);
    }

    Ok(BvpSciBorderedBandedBlocks {
        variable_count,
        mesh_points,
        parameter_count: 0,
        diagonal_blocks,
        offdiag_blocks,
        collocation_parameter_blocks: None,
        boundary_left: sparse_to_dense(dbc_dya),
        boundary_right: sparse_to_dense(dbc_dyb),
        boundary_parameters: None,
    })
}

#[inline]
fn banded_value(matrix: &BandedAssembly, row: usize, col: usize) -> f64 {
    matrix.get(row, col).unwrap_or(0.0)
}

fn sparse_to_dense(matrix: &faer_mat) -> faer_dense_mat {
    faer_dense_mat::from_fn(matrix.nrows(), matrix.ncols(), |row, col| {
        matrix.get(row, col).copied().unwrap_or(0.0)
    })
}

#[cfg(test)]
mod direct_banded_assembly_tests {
    use super::{assemble_bordered_banded_blocks_from_pointwise, extract_bordered_banded_blocks};
    use crate::numerical::BVP_sci::BVP_sci_faer::{construct_global_jac, faer_col, faer_mat};
    use crate::somelinalg::banded::banded_assembly::BandedAssembly;
    use faer::sparse::{SparseColMat, Triplet};

    fn sparse(entries: &[(usize, usize, f64)], rows: usize, cols: usize) -> faer_mat {
        let triplets = entries
            .iter()
            .map(|&(row, col, value)| Triplet::new(row, col, value))
            .collect::<Vec<_>>();
        SparseColMat::try_new_from_triplets(rows, cols, &triplets).unwrap()
    }

    fn banded(entries: &[(usize, usize, f64)], n: usize, kl: usize, ku: usize) -> BandedAssembly {
        let mut matrix = BandedAssembly::zeros(n, kl, ku).unwrap();
        for &(row, col, value) in entries {
            matrix.set(row, col, value).unwrap();
        }
        matrix
    }

    #[test]
    fn direct_pointwise_banded_assembly_matches_sparse_global_oracle() {
        let n = 3;
        let m = 4;
        let h = faer_col::from_fn(m - 1, |i| [0.2, 0.3, 0.5][i]);
        let point_entries = [
            vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, -3.0), (2, 2, 4.0)],
            vec![(0, 0, 1.5), (0, 1, 2.5), (1, 0, -2.0), (2, 2, 3.0)],
            vec![(0, 0, 2.0), (0, 1, 1.0), (1, 0, -1.0), (2, 2, 2.0)],
            vec![(0, 0, 2.5), (0, 1, 0.5), (1, 0, -0.5), (2, 2, 1.0)],
        ];
        let middle_entries = [
            vec![(0, 0, 0.5), (0, 1, 1.0), (1, 0, -1.5), (2, 2, 2.0)],
            vec![(0, 0, 0.7), (0, 1, 1.2), (1, 0, -1.0), (2, 2, 1.5)],
            vec![(0, 0, 0.9), (0, 1, 0.8), (1, 0, -0.5), (2, 2, 1.0)],
        ];
        let sparse_points = point_entries
            .iter()
            .map(|entries| sparse(entries, n, n))
            .collect::<Vec<_>>();
        let sparse_middle = middle_entries
            .iter()
            .map(|entries| sparse(entries, n, n))
            .collect::<Vec<_>>();
        let banded_points = point_entries
            .iter()
            .map(|entries| banded(entries, n, 1, 1))
            .collect::<Vec<_>>();
        let banded_middle = middle_entries
            .iter()
            .map(|entries| banded(entries, n, 1, 1))
            .collect::<Vec<_>>();
        let dbc_dya = sparse(&[(0, 0, 1.0), (2, 2, 1.0)], n, n);
        let dbc_dyb = sparse(&[(1, 1, 1.0)], n, n);

        let global = construct_global_jac(
            n,
            m,
            0,
            &h,
            &sparse_points,
            &sparse_middle,
            None,
            None,
            &dbc_dya,
            &dbc_dyb,
            None,
        );
        let oracle = extract_bordered_banded_blocks(&global, n, m, 0).unwrap();
        let direct = assemble_bordered_banded_blocks_from_pointwise(
            &h,
            &banded_points,
            &banded_middle,
            &dbc_dya,
            &dbc_dyb,
        )
        .unwrap();

        let direct_dense = direct.reconstruct_dense();
        let oracle_dense = oracle.reconstruct_dense();
        let mut diff = 0.0_f64;
        for row in 0..direct_dense.nrows() {
            for col in 0..direct_dense.ncols() {
                diff = diff.max((*direct_dense.get(row, col) - *oracle_dense.get(row, col)).abs());
            }
        }
        assert!(diff <= 1e-14, "direct banded assembly diff={diff:e}");
    }
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

fn dense_to_row_major(dense: &faer_dense_mat) -> Vec<f64> {
    let mut values = vec![0.0; dense.nrows() * dense.ncols()];
    for row in 0..dense.nrows() {
        for col in 0..dense.ncols() {
            values[row * dense.ncols() + col] = *dense.get(row, col);
        }
    }
    values
}

fn identity_block(size: usize) -> Vec<f64> {
    let mut values = vec![0.0; size * size];
    for row in 0..size {
        values[idx(size, row, row)] = 1.0;
    }
    values
}

fn matmul(lhs: &[f64], rhs: &[f64], size: usize) -> Vec<f64> {
    let mut out = vec![0.0; size * size];
    for row in 0..size {
        for mid in 0..size {
            let lhs_value = lhs[idx(size, row, mid)];
            if lhs_value == 0.0 {
                continue;
            }
            for col in 0..size {
                out[idx(size, row, col)] += lhs_value * rhs[idx(size, mid, col)];
            }
        }
    }
    out
}

fn matmul_add_assign(dst: &mut [f64], lhs: &[f64], rhs: &[f64], size: usize) {
    for row in 0..size {
        for mid in 0..size {
            let lhs_value = lhs[idx(size, row, mid)];
            if lhs_value == 0.0 {
                continue;
            }
            for col in 0..size {
                dst[idx(size, row, col)] += lhs_value * rhs[idx(size, mid, col)];
            }
        }
    }
}

fn matvec_into(out: &mut [f64], lhs: &[f64], rhs: &[f64], size: usize) {
    for row in 0..size {
        let mut sum = 0.0;
        for col in 0..size {
            sum += lhs[idx(size, row, col)] * rhs[col];
        }
        out[row] = sum;
    }
}

fn matvec_sub_assign(dst: &mut [f64], lhs: &[f64], rhs: &[f64], size: usize) {
    for row in 0..size {
        let mut sum = 0.0;
        for col in 0..size {
            sum += lhs[idx(size, row, col)] * rhs[col];
        }
        dst[row] -= sum;
    }
}

fn solve_block_rhs_in_place(
    lu: &NativeDenseLu,
    block_rhs: &mut [f64],
    size: usize,
) -> Result<(), crate::somelinalg::banded::BandedError> {
    let mut column = vec![0.0; size];
    for col in 0..size {
        for row in 0..size {
            column[row] = block_rhs[idx(size, row, col)];
        }
        lu.solve_in_place(&mut column)?;
        for row in 0..size {
            block_rhs[idx(size, row, col)] = column[row];
        }
    }
    Ok(())
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

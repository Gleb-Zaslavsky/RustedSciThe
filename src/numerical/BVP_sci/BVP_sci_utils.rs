use crate::numerical::BVP_sci::BVP_sci_faer::faer_mat;
use faer::sparse::SparseColMat;
use log::{info, warn};
use sysinfo::System;

fn matrix_stats(mat: &SparseColMat<usize, f64>) -> (usize, usize, usize, f64, f64) {
    let (nrows, ncols) = mat.shape();
    let nnz = mat.compute_nnz() as usize;
    let total_elements = nrows * ncols;
    let sparsity = 1.0 - (nnz as f64) / (total_elements as f64);
    let matrix_memory_mb = (nnz * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0);
    (nrows, ncols, nnz, sparsity, matrix_memory_mb)
}

pub fn size_of_jacobian(jac: Vec<faer_mat>) -> (f64, f64) {
    let mut size_of_jac = 0.0;
    let mut average_sparsity = 0.0;
    let n = jac.len();
    for single_matrix in jac.iter() {
        let (single_matrix_mem, sparsity) = size_of_single_matrix(single_matrix);
        size_of_jac += single_matrix_mem;
        average_sparsity += sparsity;
    }
    final_jacobian_diagnostics(size_of_jac);
    let average_sparsity = average_sparsity / (n as f64);
    info!(
        "Jacobian ensemble: slices={}, memory={:.2} MB, avg_sparsity={:.2}%",
        n,
        size_of_jac,
        average_sparsity * 100.0
    );
    (size_of_jac, average_sparsity)
}
#[allow(dead_code)]
pub fn size_of_single_matrix(mat: &SparseColMat<usize, f64>) -> (f64, f64) {
    let (_, _, _, sparsity, matrix_memory_mb) = matrix_stats(mat);
    (matrix_memory_mb, sparsity)
}
#[allow(dead_code)]
pub fn size_of_matrix(mat: &SparseColMat<usize, f64>) -> f64 {
    let (nrows, ncols, nnz, sparsity, matrix_memory_mb) = matrix_stats(mat);
    info!(
        "Sparse matrix: size={}x{}, nnz={}, sparsity={:.2}%, value_memory={:.2} MB",
        nrows,
        ncols,
        nnz,
        sparsity * 100.0,
        matrix_memory_mb
    );
    matrix_memory_mb
}
#[allow(dead_code)]
pub fn final_jacobian_diagnostics(matrix_memory: f64) {
    // Create a System object
    // CPUs and processes are filled!
    let mut sys = System::new_all();

    // First we update all information of our `System` struct.
    sys.refresh_all();
    let total_memory = sys.total_memory() as f64 / (1024.0 * 1024.0);
    let used_memory = sys.used_memory() as f64 / (1024.0 * 1024.0);
    let total_swap = sys.total_swap() as f64 / (1024.0 * 1024.0);
    let used_swap = sys.used_swap() as f64 / (1024.0 * 1024.0);
    let free_memory = sys.free_memory() as f64 / (1024.0 * 1024.0);
    if matrix_memory > 0.8 * free_memory {
        warn!(
            "Matrix memory usage is {:.2} MB, which is higher than 70% of total memory.",
            matrix_memory
        );
    }
    info!(
        "System memory: total_ram={:.2} MB, used_ram={:.2} MB, total_swap={:.2} MB, used_swap={:.2} MB",
        total_memory, used_memory, total_swap, used_swap
    );
}

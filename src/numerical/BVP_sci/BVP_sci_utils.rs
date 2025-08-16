use faer::sparse::SparseColMat;
use log::{info, warn};
use sysinfo::System;
use crate::numerical::BVP_sci::BVP_sci_faer::faer_mat;

pub fn size_of_jacobian(jac: Vec<faer_mat>) -> (f64, f64){
    let mut size_of_jac = 0.0;
    let mut average_sparticity = 0.0;
    let n = jac.len();
    for  single_matrix in jac.iter() {
        let (single_matrix_mem, spartisity) = size_of_single_matrix(single_matrix);
        size_of_jac+= single_matrix_mem;
        average_sparticity+=spartisity;

    };
    final_jacobian_diagnostics(size_of_jac);
    let average_sparticity = average_sparticity/(n as f64);
    info!("Jacobian sparsity: {:.2}%", average_sparticity * 100.0);
    (size_of_jac, average_sparticity)
}
#[allow(dead_code)]
pub fn size_of_single_matrix(mat: &SparseColMat<usize, f64>) -> (f64, f64) {
    let (nrows, ncols) = mat.shape();
    // number nonzero elements
    let nnz = mat.compute_nnz() as usize;
    let total_elements = nrows * ncols;
    let sparsity = 1.0 - (nnz as f64) / (total_elements as f64);
    // Assuming each element of the matrix takes 8 bytes of memory (size of f64)
    let matrix_memory = (nnz * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0); // Convert bytes to megabytes
    (matrix_memory, sparsity)
}
#[allow(dead_code)]
pub fn size_of_matrix(mat: &SparseColMat<usize, f64>) -> f64 {
    let (nrows, ncols) = mat.shape();
    // Assuming each element of the matrix takes 8 bytes of memory (size of f64)
    let matrix_memory = (nrows * ncols * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0); // Convert bytes to megabytes
    info!("Matrix memory usage: {:.2} MB", matrix_memory);
    let (nrows, ncols) = mat.shape();
    // number nonzero elements
    let nnz = mat.compute_nnz() as usize;
    let total_elements = nrows * ncols;
    let sparsity = 1.0 - (nnz as f64) / (total_elements as f64);
    info!("Jacobian sparsity: {:.2}%", sparsity * 100.0);
    info!("Jacobian size: {}x{}", nrows, ncols);
    info!("Jacobian nnz: {}", nnz);
    info!("Jacobian memory usage: {:.2} MB", size_of_matrix(mat));
    matrix_memory
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
    // RAM and swap information:
    info!("total memory: {} bytes", total_memory);
      info!("used memory : {} bytes", used_memory);
     info!("total swap  : {} bytes", total_swap);
     info!("used swap   : {} bytes", used_swap);
}

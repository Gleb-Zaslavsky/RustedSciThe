#[cfg(feature = "arrayfire")]
pub mod bicgstab_with_preconditioneer;

pub mod ilu_preconditioner;

#[cfg(feature = "arrayfire")]
pub mod utils;

#[cfg(feature = "arrayfire")]
pub mod bicgstab_matrix_api;

#[cfg(feature = "cuda")]
pub mod cuda_lib_ffi;

pub mod nalgebra_gs;

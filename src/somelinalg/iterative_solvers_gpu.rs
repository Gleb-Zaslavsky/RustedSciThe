//! # Rust GPU Krylov Solvers (BiCGStab, GMRES) with ArrayFire
//!
//! This part of the crate crate provides GPU–accelerated Krylov solvers (BiCGStab, GMRES) and preconditioners
//! written in Rust on top of the [`arrayfire`](https://crates.io/crates/arrayfire) crate.
//!
//! ## Installation Pipeline
//!
//! ### 1. Rust `arrayfire` crate
//! The latest supported version is **`arrayfire = "3.8.0"`**.  
//! Add this to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! arrayfire = "3.8.0"
//! ```
//!
//! ### 2. ArrayFire C++ runtime library
//! The Rust crate is just a wrapper: you must install the **native ArrayFire 3.8.0 library**.
//!
//! - **Linux (Ubuntu/Debian)**  
//!   Download the `.deb` package from [ArrayFire releases](https://arrayfire.com/download/).  
//!   Example:
//!   ```bash
//!   wget https://arrayfire.s3.amazonaws.com/arrayfire-linux-3.8.0.deb
//!   sudo dpkg -i arrayfire-linux-3.8.0.deb
//!   sudo apt-get install -f
//!   ```
//!   Libraries are installed into `/opt/arrayfire` by default.
//!
//! - **Windows**  
//!   Download the Windows installer from [ArrayFire releases](https://arrayfire.com/download/).  
//!   Run the installer and ensure that the `ArrayFire\lib` directory is in your `PATH`.
//!
//! - **macOS**  
//!   Use the [prebuilt pkg installer](https://arrayfire.com/download/) or build from source.
//!
//! After installation, make sure the `AF_PATH` environment variable is set, e.g.:
//! ```bash
//! export AF_PATH=/opt/arrayfire
//! export LD_LIBRARY_PATH=$AF_PATH/lib:$LD_LIBRARY_PATH
//! ```
//!
//! ### 3. CUDA / OpenCL backend
//! ArrayFire can use **CPU**, **CUDA**, or **OpenCL** as compute backends.
//!
//! - To use **CUDA**, you must install the NVIDIA CUDA Toolkit.  
//!   For ArrayFire 3.8.0 the recommended CUDA versions are **10.x – 11.x**.  
//!   (CUDA ≥ 12 may work but is not officially guaranteed for AF 3.8.0.)
//!
//! - To use **OpenCL**, ensure you have OpenCL ICD drivers installed (on AMD/Intel GPUs).
//!
//! You can select backend at runtime in Rust:
//! ```rust, ignore
//! arrayfire::set_backend(arrayfire::Backend::CUDA);
//! ```
//!
//! ### 4. Writing custom CUDA kernels
//! If you want to extend solvers with your own CUDA kernels that interact with `arrayfire::Array`:
//!
//! - Install **NVIDIA CUDA Toolkit** (matching your GPU).  
//! - Use `array.device_ptr::<T>()` to extract raw CUDA pointers (`CUdeviceptr`) from ArrayFire arrays.  
//! - Write and compile CUDA `.cu` kernels with `nvcc`.  
//! - Launch kernels from Rust via FFI, passing the raw pointers.  
//!
//! Supported CUDA versions for AF 3.8.0:  
//! - **Linux/Windows**: CUDA 10.1 – CUDA 11.4 (most stable).  
//!
//!
//! ## Troubleshooting
//!
//! - **`symbol lookup error: libafcuda.so not found` (Linux)**  
//!   Ensure `LD_LIBRARY_PATH` points to `$AF_PATH/lib`. Example:  
//!   ```bash
//!   export LD_LIBRARY_PATH=/opt/arrayfire/lib:$LD_LIBRARY_PATH
//!   ```
//!
//! - **`cudart64_110.dll not found` (Windows)**  
//!   Make sure your CUDA Toolkit `bin` and `lib` directories are on the system `PATH`.
//!
//! - **Backend mismatch (e.g. CUDA version mismatch)**  
//!   Check your installed CUDA runtime version matches the one ArrayFire was built against.  
//!   Use `af::info()` in Rust to print backend info at runtime.
//!
//! - **Illegal memory access when launching custom kernels**  
//!   - Verify that your `nvcc`-compiled kernel matches the pointer types (`f32` vs `f64`).  
//!   - Ensure thread/block grid covers the whole problem size.  
//!   - Double-check `device_ptr()` usage: always call `array.unlock()` after the kernel finishes.
//!
//! - **Slow performance (falls back to CPU)**  
//!   ArrayFire may fall back if it cannot load CUDA/OpenCL drivers.  
//!   Call `af::get_active_backend()` to check which backend is actually being used.
//!
//! ## Summary
//! - Install **ArrayFire 3.8.0** (C++ library).  
//! - Install **Rust `arrayfire = "3.8.0"` crate**.  
//! - (Optional) Install **CUDA Toolkit 10–11** for custom GPU kernels.  
//! - Set `AF_PATH` and `LD_LIBRARY_PATH` / `PATH` correctly.  
//! - Choose backend at runtime: CPU, CUDA, or OpenCL.  
//!


pub mod bicgstab;

#[cfg(feature = "arrayfire")]
pub mod gmres;

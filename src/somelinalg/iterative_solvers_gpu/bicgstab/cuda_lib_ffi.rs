#![cfg(feature = "cuda")]

use arrayfire as af;
use std::ffi::c_int;
use std::vec::Vec;

#[link(name = "gsgpu", kind = "dylib")]
unsafe extern "C" {
    pub unsafe fn launch_multicolor_gs_fused(
        n: i32,
        num_diags: i32,
        offsets_dev: *const i32,
        a_diags_dev: *const f32,
        b_dev: *const f32,
        x_dev: *mut f32,
        color_count: i32,
        symmetric: i32,
    ) -> i32;
}

/// Convenience safe wrapper.
/// Assumes you already have `offsets_dev`, `a_diags_dev`, `b_dev`, `x_dev` on the device.
/// Safe wrapper that accepts device Arrays (already on GPU).
/// - `a_diags_dev` must be flattened diag-major (num_diags*n)
/// - `offsets_dev` must be device array of i32 offsets (len=num_diags)
#[allow(unused_assignments)]
pub fn multicolor_gs_fused_call(
    n: usize,
    num_diags: usize,
    offsets_dev: &af::Array<i32>,
    a_diags_dev: &af::Array<f32>,
    b_dev: &af::Array<f32>,
    x_dev: &mut af::Array<f32>,
    color_count: i32,
    symmetric: bool,
) -> Result<(), i32> {
    // sanity checks
    if offsets_dev.elements() as usize != num_diags {
        panic!("offsets_dev length mismatch");
    }
    if (a_diags_dev.elements() as usize) != num_diags * n {
        panic!("a_diags_dev size mismatch; expected num_diags * n");
    }
    if (b_dev.elements() as usize) != n || (x_dev.elements() as usize) != n {
        panic!("b_dev/x_dev length mismatch");
    }

    // lock device pointers (unsafe)
    let mut rc: i32 = -999;
    unsafe {
        // get device pointers (locks array in ArrayFire)
        let offsets_ptr = offsets_dev.device_ptr() as *const i32;
        let a_ptr = a_diags_dev.device_ptr() as *const f32;
        let b_ptr = b_dev.device_ptr() as *const f32;
        let x_ptr = x_dev.device_ptr() as *mut f32;

        // call DLL
        rc = launch_multicolor_gs_fused(
            n as c_int,
            num_diags as c_int,
            offsets_ptr,
            a_ptr,
            b_ptr,
            x_ptr,
            color_count as c_int,
            if symmetric { 1 } else { 0 },
        );

        // unlock arrays no matter what (avoid deadlocks)
        offsets_dev.unlock();
        a_diags_dev.unlock();
        b_dev.unlock();
        x_dev.unlock();
    }

    if rc == 0 { Ok(()) } else { Err(rc) }
}

//////////////////////////////////////////////////////////////////////////////////////////
pub fn cpu_multicolor_gs_sweep(
    offsets: &[i32],
    diags: &[Vec<f32>],
    b: &[f32],
    x: &mut [f32],
    color_count: usize,
    symmetric: bool,
) {
    let n = x.len();
    let num_diags = offsets.len();
    let mut update = |i: usize| {
        let mut diag = 0.0f32;
        let mut sum = 0.0f32;
        for k in 0..num_diags {
            let off = offsets[k];
            let j = i as isize + off as isize;
            let a_ij = diags[k][i];
            if off == 0 {
                diag = a_ij;
            } else if off < 0 {
                if j >= 0 && j < n as isize {
                    sum += a_ij * x[j as usize];
                }
            }
        }
        if diag != 0.0 {
            x[i] = (b[i] - sum) / diag;
        }
    };

    for color in 0..color_count {
        for i in 0..n {
            if i % color_count == color {
                update(i);
            }
        }
    }
    if symmetric {
        for color in (0..color_count).rev() {
            for i in 0..n {
                if i % color_count == color {
                    update(i);
                }
            }
        }
    }
}
// Helper: flatten diagonals column-major
pub fn flatten_diagonals(diags: &[Vec<f32>]) -> Vec<f32> {
    let n = diags[0].len();
    let num_diags = diags.len();
    let mut flat = Vec::with_capacity(n * num_diags);
    for k in 0..num_diags {
        flat.extend_from_slice(&diags[k]);
    }
    flat
}

////////////////////////tests///////////////////////

#[cfg(test)]
#[test]
fn test_gsgpu_dll_increment() {
    println!("[RUST] Starting test_gsgpu_dll_increment");
    af::set_backend(af::Backend::CUDA);
    af::info();

    let n = 100;
    let data_host: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let dims = af::Dim4::new(&[n as u64, 1, 1, 1]);
    println!("[RUST] Before ArrayFire: first 5 = {:?}", &data_host[0..5]);

    let arr = af::Array::new(&data_host, dims);

    unsafe {
        arr.lock();
        let dev_ptr = arr.device_ptr() as *mut f32;
        println!(
            "[RUST] About to call increment_device_data with dev_ptr={:p}",
            dev_ptr
        );

        af::sync(-1);
        increment_device_data(dev_ptr, n as i32);
        println!("[RUST] increment_device_data returned");

        arr.unlock();
    }

    let mut out_host = vec![0.0f32; n];
    arr.host(&mut out_host);
    println!("[RUST] After ArrayFire: first 5 = {:?}", &out_host[0..5]);

    for i in 0..std::cmp::min(n, 10) {
        if (out_host[i] - (i as f32 + 1.0)).abs() >= 1e-6 {
            println!(
                "[RUST] FAIL at i={}: got {} expected {}",
                i,
                out_host[i],
                i as f32 + 1.0
            );
        }
    }

    for i in 0..n {
        assert!((out_host[i] - (i as f32 + 1.0)).abs() < 1e-6);
    }
    println!("[RUST] test_gsgpu_dll_increment PASSED");
}

#[cfg(test)]
#[test]
fn test_increment_on_gpu() {
    println!("[RUST] Starting test_increment_on_gpu");
    let n = 100;
    let mut vec: Vec<f32> = (0..n).map(|i| i as f32).collect();
    println!("[RUST] Before increment: first 5 = {:?}", &vec[0..5]);

    increment_on_gpu(&mut vec);

    println!("[RUST] After increment: first 5 = {:?}", &vec[0..5]);
    for i in 0..std::cmp::min(n, 10) {
        if (vec[i] - (i as f32 + 1.0)).abs() >= 1e-6 {
            println!(
                "[RUST] FAIL at i={}: got {} expected {}",
                i,
                vec[i],
                i as f32 + 1.0
            );
        }
    }

    for i in 0..n {
        assert!(
            (vec[i] - (i as f32 + 1.0)).abs() < 1e-6,
            "mismatch at i={} got {} expected {}",
            i,
            vec[i],
            i as f32 + 1.0
        );
    }
    println!("[RUST] test_increment_on_gpu PASSED");
}
//
#[allow(dead_code)]
#[link(name = "gsgpu")] // looks for gsgpu.dll
unsafe extern "C" {
    fn launch_multicolor_gs_test(data: *mut f32, n: i32) -> c_int;

    fn increment_device_data(dev_data: *mut f32, n: i32);
    fn get_gpu_info();
}

pub fn increment_on_gpu(vec: &mut [f32]) -> i32 {
    unsafe { launch_multicolor_gs_test(vec.as_mut_ptr(), vec.len() as c_int) }
}

#[cfg(test)]
#[test]
fn test_simple_host_increment() {
    println!("[RUST] Starting test_simple_host_increment");
    let n = 10;
    let mut vec: Vec<f32> = (0..n).map(|i| i as f32).collect();
    println!("[RUST] Before increment: {:?}", &vec[0..5]);
    let rc = increment_on_gpu(&mut vec);
    println!("[RUST] launch_multicolor_gs_test returned rc={}", rc);
    println!("[RUST] After increment: {:?}", &vec[0..5]);
    assert_eq!(rc, 0, "DLL returned error code {}", rc);
    for i in 0..n {
        assert!(
            (vec[i] - (i as f32 + 1.0)).abs() < 1e-6,
            "mismatch at i={} got {} expected {}",
            i,
            vec[i],
            i as f32 + 1.0
        );
    }
    println!("[RUST] test_simple_host_increment PASSED");
}
#[cfg(test)]
#[test]
fn test_dll_loading() {
    println!("[RUST] Testing DLL loading...");
    let n = 5;
    let mut vec: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    println!("[RUST] Before DLL call: {:?}", vec);
    println!("[RUST] Host pointer: {:p}", vec.as_ptr());

    unsafe {
        launch_multicolor_gs_test(vec.as_mut_ptr(), n);
    }

    println!("[RUST] After DLL call: {:?}", vec);
    println!("[RUST] DLL loading test completed");
}

#[cfg(test)]
#[test]
fn test_memory_modification() {
    println!("[RUST] Testing direct memory modification");
    let mut vec: Vec<f32> = vec![10.0, 20.0, 30.0];
    println!("[RUST] Before: {:?}, ptr={:p}", vec, vec.as_ptr());

    unsafe {
        // Directly modify memory to test if the issue is in CUDA or memory access
        let ptr = vec.as_mut_ptr();
        *ptr = 999.0;
        *ptr.add(1) = 888.0;
    }

    println!("[RUST] After direct modification: {:?}", vec);
    assert_eq!(vec[0], 999.0);
    assert_eq!(vec[1], 888.0);
}

#[cfg(test)]
#[test]
fn test_gpu_info() {
    println!("[RUST] Getting GPU info...");
    unsafe {
        get_gpu_info();
    }
}

#[cfg(test)]
#[test]
fn test_arrayfire_context() {
    use af::Dim4;
    af::set_backend(af::Backend::CUDA);
    af::info();
    let arr = af::constant(1.0f32, Dim4::new(&[20, 1, 1, 1]));
    let mut host = vec![0.0f32; 20];
    arr.host(&mut host);
    assert_eq!(host[0], 1.0);
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_answer() {
        af::set_backend(af::Backend::CUDA);
        af::info();

        let n = 100;
        let data_host: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let dims = af::Dim4::new(&[n as u64, 1, 1, 1]);
        println!("[RUST] Before ArrayFire: first 5 = {:?}", &data_host[0..5]);

        let arr = af::Array::new(&data_host, dims);

        unsafe {
            arr.lock();
            let dev_ptr = arr.device_ptr() as *mut f32;
            println!(
                "[RUST] About to call increment_device_data with dev_ptr={:p}",
                dev_ptr
            );

            af::sync(-1);
            increment_device_data(dev_ptr, n as i32);
            println!("[RUST] increment_device_data returned");

            arr.unlock();
        }

        let mut out_host = vec![0.0f32; n];
        arr.host(&mut out_host);
        println!("[RUST] After ArrayFire: first 5 = {:?}", &out_host[0..5]);

        for i in 0..std::cmp::min(n, 10) {
            if (out_host[i] - (i as f32 + 1.0)).abs() >= 1e-6 {
                println!(
                    "[RUST] FAIL at i={}: got {} expected {}",
                    i,
                    out_host[i],
                    i as f32 + 1.0
                );
            }
        }

        for i in 0..n {
            assert!((out_host[i] - (i as f32 + 1.0)).abs() < 1e-6);
        }
    }

    #[test]
    fn test_multicolor_gs_gpu() {
        af::set_backend(af::Backend::CUDA);
        let n = 8;
        let offsets = vec![-1, 0, 1];
        let color_count = 2;

        // Build diagonals: simple tridiagonal system
        let mut diags: Vec<Vec<f32>> = Vec::new();
        diags.push(vec![-1.0; n]); // lower
        diags.push(vec![4.0; n]); // main
        diags.push(vec![-1.0; n]); // upper

        // RHS and initial guess
        let b: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut x_cpu = vec![0.0f32; n];
        let mut x_gpu_host = vec![0.0f32; n];

        // ---- Upload to GPU ----
        // Flatten diagonals
        let flat_diags = flatten_diagonals(&diags);
        let dims_flat = af::Dim4::new(&[(n * diags.len()) as u64, 1, 1, 1]);
        let a_diags_dev = af::Array::new(&flat_diags, dims_flat);

        // Diagnostic: copy flattened array back and print main diagonal interpretation
        {
            let num_diags = diags.len();
            let mut flat_host = vec![0.0f32; n * num_diags];
            a_diags_dev.host(&mut flat_host);

            let main_k = offsets
                .iter()
                .position(|&o| o == 0)
                .expect("main diagonal offset 0 missing");

            println!("=== flatten (raw) ===");
            for k in 0..num_diags {
                println!("diag k={} offset={}", k, offsets[k]);
            }

            println!("--- Interpret as diag-major A_diags[k*n + i] (kernel expectation) ---");
            for i in 0..n {
                let v = flat_host[main_k * n + i];
                println!("main[{}] = {}", i, v);
            }

            println!("--- Interpret as row-major A_rows[i*num_diags + k] (alternative) ---");
            for i in 0..n {
                let v = flat_host[i * num_diags + main_k];
                println!("main[{}] = {}", i, v);
            }
        }

        // Offsets
        let dims_offsets = af::Dim4::new(&[offsets.len() as u64, 1, 1, 1]);
        let offsets_dev = af::Array::new(&offsets, dims_offsets);

        // b and x on device
        let dims_vec = af::Dim4::new(&[n as u64, 1, 1, 1]);
        let b_dev = af::Array::new(&b, dims_vec);
        let x_dev = af::Array::new(&x_gpu_host, dims_vec);

        // ---- Raw device pointers ----
        let offsets_dev_ptr = unsafe { offsets_dev.device_ptr() as *const i32 };
        let a_diags_dev_ptr = unsafe { a_diags_dev.device_ptr() as *const f32 };
        let b_dev_ptr = unsafe { b_dev.device_ptr() as *const f32 };
        let x_dev_ptr = unsafe { x_dev.device_ptr() as *mut f32 };

        unsafe {
            launch_multicolor_gs_fused(
                n as i32,
                offsets.len() as i32,
                offsets_dev_ptr,
                a_diags_dev_ptr,
                b_dev_ptr,
                x_dev_ptr,
                color_count as i32,
                1,
            );
        }

        // Unlock arrays (important!)
        offsets_dev.unlock();
        a_diags_dev.unlock();
        b_dev.unlock();
        x_dev.unlock();

        // ---- Copy back result ----
        x_dev.host(&mut x_gpu_host);

        // ---- CPU reference ----
        cpu_multicolor_gs_sweep(&offsets, &diags, &b, &mut x_cpu, color_count, true);

        // ---- Compare ----
        for i in 0..n {
            let diff = (x_gpu_host[i] - x_cpu[i]).abs();
            println!(
                "i={} cpu={} gpu={} diff={}",
                i, x_cpu[i], x_gpu_host[i], diff
            );
            assert!(
                diff < 1e-4,
                "Mismatch at i={} cpu={} gpu={} diff={}",
                i,
                x_cpu[i],
                x_gpu_host[i],
                diff
            );
        }
    }

    #[test]
    fn test_multicolor_gs_fused_wrapper() {
        use arrayfire as af;
        af::set_backend(af::Backend::CUDA);

        let n = 8usize;
        let offsets = vec![-1i32, 0, 1];
        let color_count = 2i32;

        let mut diags: Vec<Vec<f32>> = Vec::new();
        diags.push(vec![-1.0; n]); // lower
        diags.push(vec![4.0; n]); // main
        diags.push(vec![-1.0; n]); // upper

        let b: Vec<f32> = (0..n).map(|i| i as f32).collect();

        // Flatten diag-major
        let flat = flatten_diagonals(&diags);
        let a_diags_dev =
            af::Array::new(&flat, af::Dim4::new(&[(n * diags.len()) as u64, 1, 1, 1]));
        let offsets_dev = af::Array::new(&offsets, af::Dim4::new(&[offsets.len() as u64, 1, 1, 1]));
        let b_dev = af::Array::new(&b, af::Dim4::new(&[n as u64, 1, 1, 1]));
        let mut x_dev = af::Array::new(&vec![0.0f32; n], af::Dim4::new(&[n as u64, 1, 1, 1]));

        // call fused wrapper
        multicolor_gs_fused_call(
            n,
            diags.len(),
            &offsets_dev,
            &a_diags_dev,
            &b_dev,
            &mut x_dev,
            color_count,
            true,
        )
        .expect("fused GS returned err");

        let mut gpu_x = vec![0.0f32; n];
        x_dev.host(&mut gpu_x);

        // CPU reference
        let mut cpu_x = vec![0.0f32; n];
        cpu_multicolor_gs_sweep(&offsets, &diags, &b, &mut cpu_x, color_count as usize, true);

        for i in 0..n {
            assert!(
                (gpu_x[i] - cpu_x[i]).abs() < 1e-6,
                "i={} gpu={} cpu={} diff={}",
                i,
                gpu_x[i],
                cpu_x[i],
                (gpu_x[i] - cpu_x[i]).abs()
            );
        }
    }
}

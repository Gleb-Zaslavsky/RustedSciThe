// fused_multicolor_gs.cu
/*
nvcc -shared -Xcompiler "/MD" ^
  -gencode arch=compute_52,code=sm_52 ^
  -gencode arch=compute_52,code=compute_52 ^
  multicolor_gs.cu -o gsgpu.dll


*/

// cuda/multicolor_gs.cu
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>

using namespace cooperative_groups;

extern "C" {

    // simple per-color kernel
__global__ void per_color_kernel(
    int n, int num_diags, const int *offsets,
    const float *A_diags, const float *b,
    float *x, int color, int color_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        if ((i % color_count) != color) continue;
        float diag = 0.0f;
        float sum = 0.0f;
        for (int k = 0; k < num_diags; ++k) {
            int off = offsets[k];
            int j = i + off;
            float a_ij = A_diags[k * n + i];
            if (off == 0) diag = a_ij;
            else if (off < 0) {
                if (j >= 0 && j < n) sum += a_ij * x[j];
            }
        }
        if (diag != 0.0f) x[i] = (b[i] - sum) / diag;
    }
}


// fused kernel: does forward pass; if symmetric==1 it does backward pass as well.
// A_diags layout: column-major (diag k at A_diags + k*n, element for row i is A_diags[k*n + i])
// offsets: offsets[k] corresponds to diagonal k; can be negative (lower), 0 (main), positive (upper).
__global__ void fused_multicolor_gs_kernel(
    int n,
    int num_diags,
    const int *offsets,          // length num_diags
    const float *A_diags,        // length num_diags * n, column-major by diag
    const float *b,
    float *x,
    int color_count,
    int symmetric)
{
    grid_group grid = this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // forward-sweep compute (lower part)
    auto compute_forward = [&](int idx) -> float {
        float diag = 0.0f;
        float sum  = 0.0f;
        for (int k = 0; k < num_diags; ++k) {
            int off = offsets[k];
            int j = idx + off;
            float a_ij = A_diags[k * n + idx];
            if (off == 0) {
                diag = a_ij;
            } else if (off < 0) {
                if (j >= 0 && j < n)
                    sum += a_ij * x[j];
            }
        }
        if (diag == 0.0f) return x[idx];
        return (b[idx] - sum) / diag;
    };

    // backward-sweep compute (upper part)
    auto compute_backward = [&](int idx) -> float {
        float diag = 0.0f;
        float sum  = 0.0f;
        for (int k = 0; k < num_diags; ++k) {
            int off = offsets[k];
            int j = idx + off;
            float a_ij = A_diags[k * n + idx];
            if (off == 0) {
                diag = a_ij;
            } else if (off > 0) { // use upper diags now
                if (j >= 0 && j < n)
                    sum += a_ij * x[j];
            }
        }
        if (diag == 0.0f) return x[idx];
        return (b[idx] - sum) / diag;
    };

    // Forward pass (lower)
    for (int color = 0; color < color_count; ++color) {
        for (int idx = tid; idx < n; idx += stride) {
            if ((idx % color_count) != color) continue;
            x[idx] = compute_forward(idx);
        }
        grid.sync();
    }

    // Backward pass (upper)
    if (symmetric) {
        for (int color = color_count - 1; color >= 0; --color) {
            for (int idx = tid; idx < n; idx += stride) {
                if ((idx % color_count) != color) continue;
                x[idx] = compute_backward(idx);
            }
            grid.sync();
        }
    }
}

// host-side launcher: tries cooperative launch and falls back to per-color launches
__declspec(dllexport)
int launch_multicolor_gs_fused(
    int n,
    int num_diags,
    const int *offsets_dev,
    const float *A_diags_dev,
    const float *b_dev,
    float *x_dev,
    int color_count,
    int symmetric)
{
    int device;
    cudaError_t cerr = cudaGetDevice(&device);
    if (cerr != cudaSuccess) {
        printf("[CUDA] cudaGetDevice failed: %s\n", cudaGetErrorString(cerr));
        return -10;
    }

    int attr = 0;
    cudaDeviceGetAttribute(&attr, cudaDevAttrCooperativeLaunch, device);
    int cooperative = attr;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

  // printf("[CUDA] launch_multicolor_gs_fused: n=%d, num_diags=%d, blocks=%d threads=%d coop=%d\n",
      //     n, num_diags, blocks, threads, cooperative);
   // fflush(stdout);

    if (cooperative) {
        void *kernelArgs[] = {
            (void *)&n,
            (void *)&num_diags,
            (void *)&offsets_dev,
            (void *)&A_diags_dev,
            (void *)&b_dev,
            (void *)&x_dev,
            (void *)&color_count,
            (void *)&symmetric};
        cerr = cudaLaunchCooperativeKernel(
            (void*)fused_multicolor_gs_kernel,
            blocks, threads,
            kernelArgs);
        if (cerr != cudaSuccess) {
            printf("[CUDA] cooperative launch failed: %s\n", cudaGetErrorString(cerr));
        } else {
            cudaDeviceSynchronize();
            printf("[CUDA] cooperative fused kernel finished\n");
            return 0;
        }
    }

    // Fallback: per-color launches (portable)
  //  printf("[CUDA] Falling back to per-color launches (color_count=%d)\n", color_count);
    for (int c = 0; c < color_count; ++c) {
        per_color_kernel<<<blocks, threads>>>(
            n, num_diags, offsets_dev, A_diags_dev, b_dev, x_dev, c, color_count);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("[CUDA] per-color launch error color %d: %s\n", c, cudaGetErrorString(err));
            return 2 + c;
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("[CUDA] per-color sync failed color %d: %s\n", c, cudaGetErrorString(err));
            return 100 + c;
        }
    }
        // if symmetric requested, do backward pass also
    if (symmetric) {
        for (int c = color_count - 1; c >= 0; --c) {
            per_color_kernel<<<blocks, threads>>>(
                n, num_diags, offsets_dev, A_diags_dev, b_dev, x_dev, c, color_count
            );
            
                  cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("[CUDA] per-color launch error color %d: %s\n", c, cudaGetErrorString(err));
            return 2 + c;
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("[CUDA] per-color sync failed color %d: %s\n", c, cudaGetErrorString(err));
            return 100 + c;
        }

        }
    }
    return 0;
}


// host helper: allocate device buffers, copy data, run kernel, copy back
__declspec(dllexport)
int launch_multicolor_gs_host(
    int n,
    int num_diags,
    const int *offsets_host,
    const float *a_diags_host,
    const float *b_host,
    float *x_host,
    int color_count,
    int symmetric)
{
    size_t bytes_diags = (size_t)num_diags * n * sizeof(float);
    size_t bytes_offs = (size_t)num_diags * sizeof(int);
    size_t bytes_vec = (size_t)n * sizeof(float);

    float *d_diags = nullptr, *d_b = nullptr, *d_x = nullptr;
    int *d_offsets = nullptr;

    cudaError_t err;
    err = cudaMalloc((void**)&d_diags, bytes_diags); if (err != cudaSuccess) return -1;
    err = cudaMalloc((void**)&d_b, bytes_vec);       if (err != cudaSuccess) { cudaFree(d_diags); return -2; }
    err = cudaMalloc((void**)&d_x, bytes_vec);       if (err != cudaSuccess) { cudaFree(d_diags); cudaFree(d_b); return -3; }
    err = cudaMalloc((void**)&d_offsets, bytes_offs);if (err != cudaSuccess) { cudaFree(d_diags); cudaFree(d_b); cudaFree(d_x); return -4; }

    err = cudaMemcpy(d_diags, a_diags_host, bytes_diags, cudaMemcpyHostToDevice); if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(d_b, b_host, bytes_vec, cudaMemcpyHostToDevice);             if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(d_x, x_host, bytes_vec, cudaMemcpyHostToDevice);             if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(d_offsets, offsets_host, bytes_offs, cudaMemcpyHostToDevice);if (err != cudaSuccess) goto cleanup;

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    // cooperative launch attempt
    int coop_attr = 0;
    cudaDeviceGetAttribute(&coop_attr, cudaDevAttrCooperativeLaunch, 0);
    if (coop_attr) {
        void* kernelArgs[] = {
            (void*)&n, (void*)&num_diags, (void*)&d_offsets,
            (void*)&d_diags, (void*)&d_b, (void*)&d_x,
            (void*)&color_count, (void*)&symmetric };
        cudaError_t cerr = cudaLaunchCooperativeKernel(
            (void*)fused_multicolor_gs_kernel,
            grid.x, block.x, kernelArgs);
        if (cerr == cudaSuccess) {
            cerr = cudaDeviceSynchronize();
            if (cerr != cudaSuccess) { err = cerr; goto cleanup; }
        } else {
            // fallback below
        }
    }

    // fallback per-color launches (forward)
    for (int c = 0; c < color_count; ++c) {
        per_color_kernel<<<grid, block>>>(n, num_diags, d_offsets, d_diags, d_b, d_x, c, color_count);
        err = cudaGetLastError(); if (err != cudaSuccess) goto cleanup;
        err = cudaDeviceSynchronize(); if (err != cudaSuccess) goto cleanup;
    }

    // if symmetric requested, do backward pass also
    if (symmetric) {
        for (int c = color_count - 1; c >= 0; --c) {
            per_color_kernel<<<grid, block>>>(n, num_diags, d_offsets, d_diags, d_b, d_x, c, color_count);
            err = cudaGetLastError(); if (err != cudaSuccess) goto cleanup;
            err = cudaDeviceSynchronize(); if (err != cudaSuccess) goto cleanup;
        }
    }

    err = cudaMemcpy(x_host, d_x, bytes_vec, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto cleanup;

    cudaFree(d_diags); cudaFree(d_b); cudaFree(d_x); cudaFree(d_offsets);
    return 0;

cleanup:
    cudaFree(d_diags); cudaFree(d_b); cudaFree(d_x); cudaFree(d_offsets);
    return 10 + (int)err;
}

} // extern "C"


////////////////////////////////TESTS////////////////////////////////////////////////////
#include <cuda_runtime.h>
#include <cstdio>

extern "C" {

__global__ void increment_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1.0f;
        if (idx < 3) {
            printf("[KERNEL] Thread %d: data[%d] = %.1f\n", idx, idx, data[idx]);
        }
    }
}

// Return codes:
//  0 = success
// <0 = cudaMalloc or cudaMemcpy H2D failed
// >0 = cudaMemcpy D2H or kernel sync failed (positive codes)
__declspec(dllexport)
int launch_multicolor_gs_test(float *host_data, int n) {
    printf("[CUDA] launch_multicolor_gs_test called with n=%d, host_data=%p\n", n, host_data);
    fflush(stdout);

    float *dev_data = nullptr;
    size_t bytes = (size_t)n * sizeof(float);
    cudaError_t err;

    err = cudaMalloc((void**)&dev_data, bytes);
    if (err != cudaSuccess) {
        printf("[CUDA] cudaMalloc failed: %s\n", cudaGetErrorString(err));
        fflush(stdout);
        return -1;
    }

    // Print host values we will copy
    printf("[CUDA] First 3 input values (host): %.6f, %.6f, %.6f\n",
           (n>0?host_data[0]:0.0f), (n>1?host_data[1]:0.0f), (n>2?host_data[2]:0.0f));
    fflush(stdout);

    err = cudaMemcpy(dev_data, host_data, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("[CUDA] cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        fflush(stdout);
        cudaFree(dev_data);
        return -2;
    }

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    printf("[CUDA] Launching kernel with blocks=%d, threads=%d\n", blocks, threads);
    fflush(stdout);

    increment_kernel<<<blocks, threads>>>(dev_data, n);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[CUDA] Kernel launch failed: %s\n", cudaGetErrorString(err));
        fflush(stdout);
        cudaFree(dev_data);
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[CUDA] Kernel execution failed: %s\n", cudaGetErrorString(err));
        fflush(stdout);
        cudaFree(dev_data);
        return 2;
    }

    err = cudaMemcpy(host_data, dev_data, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("[CUDA] cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        fflush(stdout);
        cudaFree(dev_data);
        return 3;
    }

    printf("[CUDA] First 3 output values (host after D2H): %.6f, %.6f, %.6f\n",
           (n>0?host_data[0]:0.0f), (n>1?host_data[1]:0.0f), (n>2?host_data[2]:0.0f));
    fflush(stdout);

    cudaFree(dev_data);
    return 0;
}

__declspec(dllexport)
void get_gpu_info() {
    printf("[CUDA] get_gpu_info called\n");
    
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        printf("[CUDA] cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return;
    }
    
    printf("[CUDA] Found %d CUDA devices\n", deviceCount);
    
    if (deviceCount == 0) {
        printf("[CUDA] No CUDA devices found!\n");
        return;
    }
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, i);
        if (err != cudaSuccess) {
            printf("[CUDA] cudaGetDeviceProperties failed for device %d: %s\n", i, cudaGetErrorString(err));
            continue;
        }
        
        printf("[CUDA] Device %d: %s\n", i, prop.name);
        printf("[CUDA] Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("[CUDA] SM count: %d\n", prop.multiProcessorCount);
        printf("[CUDA] Global memory: %.1f MB\n", prop.totalGlobalMem / (1024.0f * 1024.0f));
    }
}

__declspec(dllexport)
void increment_device_data(float *dev_data, int n) {
    printf("[CUDA] increment_device_data called with n=%d, dev_data=%p\n", n, dev_data);
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    increment_kernel<<<blocks, threads>>>(dev_data, n);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[CUDA] Kernel launch error: %s\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[CUDA] Sync error: %s\n", cudaGetErrorString(err));
        return;
    }
    
    printf("[CUDA] increment_device_data completed successfully\n");
}

} // extern "C"
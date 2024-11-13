#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void checkCudaError(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCublasError(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS error" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Matrix dimensions
    const int N = 1024; // Size of the NxN matrix
    const int SIZE = N * N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(SIZE);
    // Initialize the matrix (for example, identity matrix)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i * N + j] = (i == j) ? 1.0f : 0.0f; // Identity matrix
        }
    }

    // Allocate device memory
    float* d_A;
    checkCudaError(cudaMalloc((void**)&d_A, SIZE));
    
    // Copy matrix from host to device
    checkCudaError(cudaMemcpy(d_A, h_A, SIZE, cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    checkCublasError(cublasCreate(&handle));

    // Perform LU factorization
    int* d_P; // Pivot indices
    int* d_info; // Info about the factorization
    checkCudaError(cudaMalloc((void**)&d_P, N * sizeof(int)));
    checkCudaError(cudaMalloc((void**)&d_info, sizeof(int)));

    int* h_info = (int*)malloc(sizeof(int));
    cublasSgetrf(handle, N, N, d_A, N, NULL, d_P, d_info);

    // Compute the inverse
    cublasSgetri(handle, N, d_A, N, d_P, d_info);

    // Copy the result back to host
    checkCudaError(cudaMemcpy(h_A, d_A, SIZE, cudaMemcpyDeviceToHost));

    // Check for errors in the factorization
    checkCudaError(cudaMemcpy(h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (*h_info != 0) {
        std::cerr << "Matrix inversion failed with error: " << *h_info << std::endl;
    } else {
        std::cout << "Matrix inversion successful!" << std::endl;
    }

    // Cleanup
    free(h_A);
    free(h_info);
    cudaFree(d_A);
    cudaFree(d_P);
    cudaFree(d_info);
    cublasDestroy(handle);

    return 0;
}

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void checkCudaError(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCublasError(cublasStatus_t result, const char* msg) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error: " << msg << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int N = 3; // Size of the matrix
    std::vector<float> h_A = {1, 2, 3, 0, 1, 4, 5, 6, 0}; // Host matrix (row-major order)
    std::vector<float> h_Ainv(N * N); // Host matrix to store the inverse

    float* d_A = nullptr;
    int* d_pivotArray = nullptr;
    int* d_info = nullptr;
    float* d_work = nullptr;
    int lwork = 0;

    cublasHandle_t handle;
    checkCublasError(cublasCreate(&handle), "Failed to create cuBLAS handle");

    // Allocate device memory
    checkCudaError(cudaMalloc((void**)&d_A, N * N * sizeof(float)), "Failed to allocate device memory for matrix A");
    checkCudaError(cudaMalloc((void**)&d_pivotArray, N * sizeof(int)), "Failed to allocate device memory for pivot array");
    checkCudaError(cudaMalloc((void**)&d_info, sizeof(int)), "Failed to allocate device memory for info");

    // Copy matrix from host to device
    checkCudaError(cudaMemcpy(d_A, h_A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy matrix A to device");

    // Get the optimal workspace size
    checkCublasError(cublasSgetrf_bufferSize(handle, N, N, d_A, N, &lwork), "Failed to get buffer size for LU decomposition");
    checkCudaError(cudaMalloc((void**)&d_work, lwork * sizeof(float)), "Failed to allocate device memory for workspace");

    // Perform LU decomposition
    checkCublasError(cublasSgetrf(handle, N, N, d_A, N, d_pivotArray, d_info), "Failed to perform LU decomposition");

    // Check if the matrix is singular
    int h_info;
    checkCudaError(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy info to host");
    if (h_info != 0) {
        std::cerr << "Matrix is singular and cannot be inverted" << std::endl;
        return EXIT_FAILURE;
    }

    // Perform matrix inversion
    checkCublasError(cublasSgetri(handle, N, d_A, N, d_pivotArray, d_work, lwork, d_info), "Failed to perform matrix inversion");

    // Copy the result back to the host
    checkCudaError(cudaMemcpy(h_Ainv.data(), d_A, N * N * sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy inverted matrix to host");

    // Print the inverted matrix
    std::cout << "Inverted matrix:" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << h_Ainv[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_pivotArray);
    cudaFree(d_info);
    cudaFree(d_work);
    cublasDestroy(handle);

    return 0;
}

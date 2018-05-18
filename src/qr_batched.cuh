// C libraries
#include <stdio.h>

// CUDA libraries
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuComplex.h"
#include <cublas_v2.h>

// kblas libraries
//#include "kblas_struct.h"
#include "kblas.h"
#include "kblas_batch.h"
//#include "batch_qr.cu"

#include "matrixOp_utility.cuh"


#define QR_ITERATIONS 1

// nvcc -gencode arch=compute_50,code=sm_50 -o test test_cu.cu -I/home/travis/Documents/kblas-gpu/include -I/home/travis/Documents/kblas-gpu/src -I/home/travis/Documents/kblas-gpu/src/batch_svd/ -lcublas




/**
 * QR factorization routine for doubles on the GPU utilizing kblas. Given a matrix A, return the upper triangular
 * matrix R and orthogonal matrix Q.
 * @param kblas_handle Kblas handle to kblas instance.
 * @param d_A Device pointer to input matrix A.
 * @param d_Q Device pointer to output orthogonal matrix Q.
 * @param d_R Device pointer to output upper triangular matrix R.
 * @param N Number of elements in a row.
 * @param batchSize Number of matrices to be calculated.
 */
void QR_d_Batched(kblasHandle_t kblas_handle, double *d_A, double *d_Q, double *d_R, int N, int batchSize){

    // Create batched tau array for workspace.
    double *d_tau;
    // Allocate batched array on device.
    cudaMalloc((void**)&d_tau, batchSize*N*sizeof(double));

    // Call QR decomposition routine in kblas-gpu.
    int res = kblasDgeqrf_batch_strided(kblas_handle, N, N, d_A, N, N*N, d_tau, N, batchSize);

    // Copy upper triangular portion of matrix A to matrix R.
    copy_UpperTriangular_Strided<double>(d_A, d_R, N, batchSize);

    // Call orthogonal matrix Q retriever routine in kblas-gpu to get the full Q matrix.
    res = kblasDorgqr_batch_strided(kblas_handle, N, N, d_A, N, N*N, d_tau, N, batchSize);

    // Copy all of matrix A into matrix Q.
    dCopy_Strided<double>(d_A, d_Q, N, batchSize);

    cudaFree(d_tau);
}




/**
 * QR factorization routine for doubles on the GPU utilizing kblas. Given a matrix A, return the upper triangular
 * matrix R and orthogonal matrix Q.
 * @param kblas_handle Kblas handle to kblas instance.
 * @param d_A Device pointer to input matrix A.
 * @param d_Q Device pointer to output orthogonal matrix Q.
 * @param d_R Device pointer to output upper triangular matrix R.
 * @param N Number of elements in a row.
 * @param batchSize Number of matrices to be calculated.
 */
void QR_f_Batched(kblasHandle_t kblas_handle, float *d_A, float *d_Q, float *d_R, int N, int batchSize){

    // Create batched tau array for workspace.
    float *d_tau;
    // Allocate batched array on device.
    cudaMalloc((void**)&d_tau, batchSize*N*sizeof(float));

    // Call QR decomposition routine in kblas-gpu.
    int res = kblasSgeqrf_batch_strided(kblas_handle, N, N, d_A, N, N*N, d_tau, N, batchSize);

    // Copy upper triangular portion of matrix A to matrix R.
    copy_UpperTriangular_Strided(d_A, d_R, N, batchSize);

    // Call orthogonal matrix Q retriever routine in kblas-gpu to get the full Q matrix.
    res = kblasSorgqr_batch_strided(kblas_handle, N, N, d_A, N, N*N, d_tau, N, batchSize);

    // Copy all of matrix A into matrix Q.
    dCopy_Strided<float>(d_A, d_Q, N, batchSize);

    cudaFree(d_tau);
}





void QR_Iterative_Batched(kblasHandle_t kblas_handle, double *d_A, double *d_Q, double *d_Eigenvalues, int N, int batchSize){
    
    // Get cuBLAS handle.
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    cublasOperation_t CUBLAS_OP_T;
    cublasOperation_t CUBLAS_OP_N;

    // Initialize alpha and beta as one.
    double alpha = 1.0, beta = 0.0;
    long stride = N*N;

    // Initialize Q and R batched matrices on device.
    double *d_R, *d_QQ, *d_temp;


    // Allocate Q and R batched matrices on device.
    //cudaMalloc((void**)&d_Q,    batchSize*N*N*sizeof(double));
    cudaMalloc((void**)&d_R,    batchSize*N*N*sizeof(double));
    cudaMalloc((void**)&d_QQ,   batchSize*N*N*sizeof(double));
    cudaMalloc((void**)&d_temp, batchSize*N*N*sizeof(double));


    // Initialize QQ matrix as an identity matrix.
    dInitialize_Identity_Batched<double>(d_QQ, N, batchSize);

    unsigned int iterations = 0;
    bool convergence = false;
    // Loop until convergence.
    while (convergence != true) {
    
        for (int ii = 0; ii < QR_ITERATIONS; ii++) {
    
            // Perform QR decomposition.
            QR_d_Batched(kblas_handle, d_A, d_Q, d_R, N, batchSize);
    
            // Multiply A(i+1) = R(i) * Q_(i).
            kblasDgemm_batch_strided(kblas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, alpha,
                                     (const double *) d_R, N, stride,
                                     (const double *) d_Q, N, stride, beta,
                                     d_A, N, stride, batchSize);
    
            // Multiply QQ * Q
            kblasDgemm_batch_strided(kblas_handle, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, alpha,
                                     (const double *) d_QQ, N, stride,
                                     (const double *) d_Q, N, stride, beta,
                                     d_temp, N, stride, batchSize);
    
    
            // Update QQ matrix as temp.
            dCopy_Strided<double>(d_temp, d_QQ, N, batchSize);
        }
        
         // Check for convergence.
        convergence = check_Convergence <double> (d_A, N, batchSize);
    
        // Update iterations taken.
        iterations += QR_ITERATIONS;
    }
    
    printf("Iterations: %i\n", iterations);

    // Update Q matrix as Q.
    dCopy_Strided<double>(d_QQ, d_Q, N, batchSize);

}










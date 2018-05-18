// C libraries
#include <stdio.h>

// CUDA libraries
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuComplex.h"
#include <cublas_v2.h>

#include "kblas.h"

#include "src/qr_batched.cuh"


// OFFICE
// nvcc -gencode arch=compute_50,code=sm_50 test_cu.cu -I/home/travis/Documents/kblas-gpu/include -I/home/travis/Documents/kblas-gpu/src -I/home/travis/Documents/magma-2.2.0/include/ -I/home/travis/Documents/lapack-3.7.1/include -L/home/travi s/Documents/kblas-gpu/lib -L/home/travis/Documents/magma-2.2.0/lib -lkblas-gpu -lcublas -lcusparse -lmagma

// HOME
// nvcc -gencode arch=compute_50,code=sm_50 test_cu.cu -I/home/travis/Documents/kblas-gpu/include -L/home/travis/Documents/kblas-gpu/lib -L/home/travis/Documents/magma-2.2.0/lib -L/home/travis/Documents/OpenBLAS -L/home/travis/Documents/lapack-3.7.1 -lkblas-gpu -lcublas -lcusparse -lopenblas -lmagma -o TEST_QR





kblasHandle_t createHandle(){
    // Create kblas handle (internally creates cublas handle as well).
    kblasHandle_t kblas_handle;
    kblasCreate(&kblas_handle);
    // Return kblas handle.
    return kblas_handle;
}


void printStridedMatrix(double *matrix, int N, int batchSize){
    int i, j, k;
    for (k = 0; k < batchSize; k++){
        for (i = k*N; i < (k+1)*N; i++){
            for (j = 0; j < N; j++){
                double m = matrix[i*N + j];
                printf("%g\t", m);
            }
            printf("\n");
        }
        printf("\n");
    }
}



int main() {
    const int N = 3;
    const int batchSize = 2;
    double A[batchSize*N*N]; //= { 1.0, 4.0, 2.0, 2.0, 5.0, 1.0, 3.0, 6.0, 1.0};
    double Q[batchSize*N*N]; // orthonormal columns
    double U[batchSize*N*N]; // Unitary matrix check.
    double eigenvalues[batchSize*N];


    A[0] = 1.0; A[1] = 4.0; A[2] = 2.0;
    A[3] = 2.0; A[4] = 5.0; A[5] = 1.0;
    A[6] = 3.0; A[7] = 6.0; A[8] = 1.0;

    A[9] = 2.0; A[10] = 4.0; A[11] = 5.0;
    A[12] = 7.0; A[13] = 1.0; A[14] = 6.0;
    A[15] = 6.0; A[16] = 2.0; A[17] = 9.0;


    // Initialize device pointers
    double *d_A;
    double *d_Q, *d_QQ, *d_U;
    double *d_R;
    double *d_Eigenvalues;

    // Allocate device memory for arrays and matrices.
    cudaMalloc((void**)&d_A,  batchSize*N*N*sizeof(double));
    cudaMalloc((void**)&d_Q,  batchSize*N*N*sizeof(double));
    cudaMalloc((void**)&d_QQ, batchSize*N*N*sizeof(double));
    cudaMalloc((void**)&d_U,  batchSize*N*N*sizeof(double));
    cudaMalloc((void**)&d_R,  batchSize*N*N*sizeof(double));
    cudaMalloc((void**)&d_Eigenvalues, batchSize*N*sizeof(double));

    // Copy batched matrices from host memory to device memory.
    cudaMemcpy(d_A, A, batchSize*N*N*sizeof(double), cudaMemcpyHostToDevice);
    
    // Create kblas handle instance.
    kblasHandle_t kblas_handle;
    kblas_handle = createHandle();

    // Print batched matrices before operations.
    printf("A Full: \n");
    printStridedMatrix(A, N, batchSize);

    // Call Iterative QR routine on device.
    QR_Iterative_Batched(kblas_handle, d_A, d_Q, d_Eigenvalues, N, batchSize);
    
    // Transpose Q matrix for unitary matrix check.
    transpose_strided_batched(d_Q, d_QQ, N, N, batchSize);

    // Multiply Q*Q^T = I to ensure Q is unitary.
    kblasDgemm_batch_strided(kblas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, 1.0,
                             (const double*) d_Q, N, N*N,
                             (const double*) d_QQ, N, N*N, 0.0,
                             d_U, N, N*N, batchSize);

    // Copy A, Q, and R matrices back.
    cudaMemcpy(A, d_A, sizeof(double)*N*N*batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(Q, d_Q, sizeof(double)*N*N*batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(U, d_U, sizeof(double)*N*N*batchSize, cudaMemcpyDeviceToHost);

    printf("A Full: \n");
    printStridedMatrix(A, N, batchSize);

    printf("Q Full: \n");
    printStridedMatrix(Q, N, batchSize);

    printf("Q * Q^T Full: \n");
    printStridedMatrix(U, N, batchSize);
    


    return 0;
}


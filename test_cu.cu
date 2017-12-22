// C libraries
#include <stdio.h>

// CUDA libraries
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuComplex.h"

#include "src/qr_batched.cuh"


//void printStridedMatrix(double *matrix, int N, int batchSize){
//    int i, j, k;
//    for (k = 0; k < batchSize; k++){
//        for (i = k*N; i < (k+1)*N; i++){
//            for (j = 0; j < N; j++){
//                double m = matrix[i*N + j];
//                printf("%g\t", m);
//            }
//            printf("\n");
//        }
//        printf("\n");
//    }
//}



int main() {
//    const int N = 3;
//    const int batchSize = 2;
//    double A[batchSize*N*N]; //= { 1.0, 4.0, 2.0, 2.0, 5.0, 1.0, 3.0, 6.0, 1.0};
//    double Q[batchSize*N*N]; // orthonormal columns
//    double R[batchSize*N*N]; // R = I - Q**T*Q
//
//    A[0] = 1.0; A[1] = 4.0; A[2] = 2.0;
//    A[3] = 2.0; A[4] = 5.0; A[5] = 1.0;
//    A[6] = 3.0; A[7] = 6.0; A[8] = 1.0;
//
//    A[9] = 2.0; A[10] = 4.0; A[11] = 5.0;
//    A[12] = 7.0; A[13] = 1.0; A[14] = 6.0;
//    A[15] = 6.0; A[16] = 2.0; A[17] = 9.0;
//
//
//    double *d_A;
//    double *d_Q;
//    double *d_R;
//
//    cudaMalloc((void**)&d_A, batchSize*N*N*sizeof(double));
//    cudaMalloc((void**)&d_Q, batchSize*N*N*sizeof(double));
//    cudaMalloc((void**)&d_R, batchSize*N*N*sizeof(double));
//
//    cudaMemcpy(d_A, A, batchSize*N*N*sizeof(double), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_Q, Q, batchSize*N*N*sizeof(double), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_R, R, batchSize*N*N*sizeof(double), cudaMemcpyHostToDevice);
//
//
//    // Create kblas handle instance.
//    kblasHandle_t kblas_handle;
//    kblas_handle = createHandle();
//
//
//
//    printf("A Full: \n");
//    printStridedMatrix(A, N, batchSize);
//
//
//    QR_Batched(kblas_handle, d_A, d_Q, d_R, N, batchSize);
//
//
//    // Copy Q and R matrices back.
//    cudaMemcpy(R, d_A, sizeof(double)*N*N*batchSize, cudaMemcpyDeviceToHost);
//    cudaMemcpy(Q, d_A, sizeof(double)*N*N*batchSize, cudaMemcpyDeviceToHost);
//
//    printf("R Full: \n");
//    printStridedMatrix(R, N, batchSize);
//
//    printf("Q Full: \n");
//    printStridedMatrix(A, N, batchSize);


    printf("Hello world.\n");

    return 0;
}


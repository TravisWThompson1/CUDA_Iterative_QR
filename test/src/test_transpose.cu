// C libraries
#include <stdio.h>

// CUDA libraries
#include "cuComplex.h"

// Local header files
#include "../../src/matrixOp_utility.cuh"


template <class T>
bool checkTranspose(T *A, T *A_trans, int N, int batchSize){
    double m, m_trans;
    int i, j, k;
    // Iterate matrix
    for (k = 0; k < batchSize; k++){
        // Iterate row
        for (i = 0; i < N; i++){
            // Iterate column
            for (j = 0; j < N; j++){
                // Get element at original index and element at its transposed index.
                m = A[k*N*N + i*N + j];
                m_trans = A_trans[k*N*N + j*N + i];
                // Check if the elements are the same.
                if ( m != m_trans )
                    return false;
            }
        }
    }
    return true;
}


template <class T>
void printStridedMatrix(T *matrix, int N, int batchSize){
    T m;
    int i, j, k;
    // Iterate matrix
    for (k = 0; k < batchSize; k++){
        // Iterate row
        for (i = 0; i < N; i++){
            // Iterate column
            for (j = 0; j < N; j++){
                m = matrix[k*N*N + i*N + j];
                printf("%g ", (double) m);
            }
            printf("\n");
        }
        printf("\n");
    }
}



int main() {
    
    int row_size[10] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    
    //for (int k = 0; k < 10; k++) {
    
        // Matrix constants
        unsigned int rows = 512; //row_size[k];
        int batchSize = 60;
        if ( rows == 256 ) {
            batchSize = 10;
        } else if (rows == 512){
            batchSize = 2;
        }
    
        // Initialize strided matrices
        float A[batchSize * rows * rows];
        float A_T[batchSize * rows * rows];
    
        // Initialize batched matrices as {0,1,2,3,4,5,...},{0,1,2,3,4,5,...},...
        for (int batch = 0; batch < batchSize; batch++) {
            for (int i = 0; i < rows * rows; i++) {
                A[batch * rows * rows + i] = batch * rows * rows + i;
            }
        }
    
        // Initialize device pointer
        float *d_A, *d_AT;
    
        // Allocate device memory
        cudaMalloc((void **) &d_A, batchSize * rows * rows * sizeof(float));
        cudaMalloc((void **) &d_AT, batchSize * rows * rows * sizeof(float));
    
        // Copy from host memory to device memory
        cudaMemcpy(d_A, A, batchSize * rows * rows * sizeof(float), cudaMemcpyHostToDevice);
    
        printf("======== RESULTS ========\n");
        // Transpose matrix
        transpose_strided_batched<float>(d_A, d_AT, rows, rows, batchSize);
    
        // Copy from device memory to host memory
        cudaMemcpy(A_T, d_AT, rows * rows * batchSize * sizeof(float), cudaMemcpyDeviceToHost);
    
        // Free CUDA memory.
        cudaFree(d_A);
        cudaFree(d_AT);
    
        // Print matrix after operation
        //printf("A after transpose: \n");
        // printStridedMatrix<float>(A_T, rows, batchSize);
    
        bool transposeSuccess = checkTranspose(A, A_T, rows, batchSize);
        printf("Rows: %i, Batchsize: %i, Success: %s\n", rows, batchSize, transposeSuccess ? "true" : "false");
    
    //}


    return 0;
}


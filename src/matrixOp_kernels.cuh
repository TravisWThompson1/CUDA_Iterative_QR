//
// Created by travis on 12/8/17.
//



// C libraries
#include <stdio.h>

// CUDA libraries
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuComplex.h"
#include <cublas_v2.h>

#define MAX_BLOCKSIZE 512
#define CONVERGENCE_EPS 1e-5




/**
 * Copies the input strided matrices to the strided output matrices.
 * @tparam T Data type template (float, double, cuFloatComplex, cuDoubleComplex, etc.).
 * @param d_input Stided input matrices to be copied.
 * @param d_output Strided output matrices that are copied to.
 * @param totalElem Total number of elements in the input strided matrices.
 */
template <class T>
__global__ void d_Copy_Strided(T *d_input, T *d_output, long int totalElem){
    // Thread id number tid:
    int tid =  blockDim.x * blockIdx.x + threadIdx.x;

    // Ensure thread is within memory allocation.
    if (tid < totalElem){
        // Retrieve input element.
        T temp = d_input[tid];
        // Write output element.
        d_output[tid] = temp;
    }
}


/**
 * Copies the upper-triangular elements in a set of strided column-major matrices of a double data type, while adding
 * zeros in the elements below the diagonal.
 * @param d_input Strided input matrix whose upper triangular elements will be copied to the output strided matrix.
 * @param d_output Strided output matrix that will be copied to.
 * @param N Number of columns or rows in one of the square matrices.
 * @param batchSize Number of matrices in the strided input or output matrices.
 */
template <class T>
__global__ void d_Copy_UpperTriangular_Strided(T *d_input, T *d_output, int N, int batchSize){
    // Thread id number tid:
    int tid =  blockDim.x * blockIdx.x + threadIdx.x;

    // Calculate the total number of elements in the strided matrices.
    long int totalElem = N * N * batchSize;

    // Calculate thread's local column (i) and row (j) index.
    int local_tid = tid % (N*N);
    int i = local_tid / N;
    int j = local_tid % N;

    // Ensure thread is within memory allocation.
    if (tid < totalElem){
        // Retrieve input element.
        T temp;
        // Determine whether local index is upper-triangular or not.
        if (i >= j){
            // Upper-triangular element.
            temp = d_input[tid];
        } else {
            // Below diagonal element. (Only passed temp into setZero() for type declaration)
            temp = setZero(temp);
        }
        // Write output element.
        d_output[tid] = temp;
    }
}





/**
 * Initializes the given strided matrices as identity matrices.
 * @param d_A Strided matrices to be set to identity matrices.
 * @param N Number of rows/columns in each matrix.
 * @param batchSize Number of matrices.
 */
template <class T>
__global__ void d_Initialize_Identity_Batched(T *d_A, int N, int batchSize){
    // Thread id number tid:
    int tid =  blockDim.x * blockIdx.x + threadIdx.x;

    // Calculate the total number of elements in the strided matrices.
    long int totalElem = N * N * batchSize;

    // Calculate thread's local column (i) and row (j) index.
    int local_tid = tid % (N*N);
    int i = local_tid / N;
    int j = local_tid % N;

    // Ensure thread is within memory allocation.
    if (tid < totalElem) {
        // Initialize possible outputs.
        T temp; // (Only passed temp into setZero() for type declaration)
        T one = setOne(temp);
        T zero = setZero(temp);
        // Write identity matrix.
        if (i == j) {
            // Diagonal element.
            d_A[tid] = one;
        } else {
            // Non-diagonal element.
            d_A[tid] = zero;
        }
    }
}





/**
 * Check batched matrices for convergence (0 = converged, 1 = not converged).
 * @param d_A Input batched matrices.
 * @param d_INFO Integer result of convergence.
 * @param N Number of rows/columns.
 * @param batchSize Number of matrices.
 */
template <class T>
__global__ void d_convergence_check(T *d_A, int *d_INFO, int N, int batchSize){
    // Thread id number tid:
    int tid =  blockDim.x * blockIdx.x + threadIdx.x;

    // Calculate the total number of elements in the strided matrices.
    long int totalElem = N * N * batchSize;

    // Calculate thread's local column (i) and row (j) index.
    int local_tid = tid % (N*N);
    int i = local_tid / N;
    int j = local_tid % N;

    // Ensure thread is within memory allocation.
    if (tid < totalElem) {
        // Only upper non-diagonal elements.
        if (i < j){
            T value = d_A[tid];
            // Check for convergence.
            if (fabs(value) > CONVERGENCE_EPS) {
                // Convergence not met.
                d_INFO[0] = 1;
            }
        }
    }
}



template <class T>
__global__ void d_maxBandwidth(T *input, T *output){
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    T temp = input[tid];
    output[tid] = temp;
};





/**
 * CUDA kernel for a strided batched transpose matrix operation (less than 32 rows) on a given strided square matrices.
 * @tparam T Template parameter type of strided matrices.
 * @tparam BLOCKSIZE CUDA blocksize used for kernel launch.
 * @param input Input non-transposed strided matrices of type T.
 * @param output Output transposed strided matrices of type T.
 * @param rows Number of rows in the input/output square matrices.
 * @param batchSize Number of strided matrices.
 * @return
 */
template <class T, unsigned int BLOCKSIZE>
__global__ void d_transposeSquare_strided_batched(T *input, T *output, int rows, int batchSize){
    
    if (threadIdx.x < rows) {
    
        // Batch ID
        unsigned int batchShift = blockIdx.x * rows * rows;
    
        // Initialize shared memory.
        __shared__
        T temp[BLOCKSIZE][BLOCKSIZE + 1];
    
        // Read elements from global memory and write them to shared memory.
        #pragma unroll
        for (int i = 0; i < rows; i++) {
        
            temp[i][threadIdx.x] = input[batchShift + i * rows + threadIdx.x];
        }
    
        // Read elements from shared memory and write them back to global memory.
        #pragma unroll
        for (int i = 0; i < rows; i++) {
        
            output[batchShift + i * rows + threadIdx.x] = temp[threadIdx.x][i];
        }
    }
}





/**
 * CUDA kernel for a strided batched transpose matrix operation (greater than 32 rows) on a given strided square matrices.
 * @tparam T Template parameter type of strided matrices.
 * @tparam BLOCKSIZE CUDA blocksize used for kernel launch.
 * @param input Input non-transposed strided matrices of type T.
 * @param output Output transposed strided matrices of type T.
 * @param rows Number of rows in the input/output square matrices.
 * @param batchSize Number of strided matrices.
 * @return
 */
template <class T, unsigned int BLOCKSIZE>
__global__ void d_transposeSquare_tiling_strided_batched(T *input, T *output, int rows, int batchSize){

    // Matrix Id = blockIdx.x;
    // Local linear block Id = blockIdx.y;

    // Tiles per matrix
    unsigned int tileDim = ceilf( rows / (float) blockDim.x );
    // Local block Id
    uint2 blockId, local_tid;
    blockId.x = blockIdx.y % tileDim;
    blockId.y = blockIdx.y / tileDim;
    // Local thread Id
    //local_tid.x = blockId.x * blockDim.x + threadIdx.x;
    local_tid.x = blockId.x * BLOCKSIZE + threadIdx.x;
    local_tid.y = blockId.y * BLOCKSIZE;
    // Global thread Id                matrix shift           +          local tile y-shift       +       local tile x-shift   +  warp-level thread Id
    unsigned int global_tid = ( blockIdx.x * rows * rows ) + ( blockId.y * blockDim.x * rows ) + ( blockId.x * blockDim.x ) + threadIdx.x;

    // Initialize shared memory.
    __shared__ T temp[BLOCKSIZE][BLOCKSIZE+1];

    // Check for overreach in x direction.
    if ( local_tid.x < rows ) {

        // Read elements from global memory and write them to shared memory.
        #pragma unroll
        for (int k = 0; k < BLOCKSIZE; k++) {

            if (local_tid.y + k < rows)
                temp[threadIdx.x][k] = input[global_tid + k * rows];

        }
    }
    

    // Transposed indices.
    // Global thread Id:    matrix shift         +        local tile y-shift         +     local tile x-shift     +  warp-level thread Id
    global_tid = ( blockIdx.x * rows * rows ) + ( blockId.x * blockDim.x * rows ) + ( blockId.y * blockDim.x ) + threadIdx.x;

    // Update local thread Id transposed.
    local_tid.x = blockId.y * BLOCKSIZE + threadIdx.x;
    local_tid.y = blockId.x * BLOCKSIZE;

    // Check for overreach in x direction.
    if ( local_tid.x < rows ) {

        // Read elements from shared memory in a transposed fashion and write them back to global memory.
        #pragma unroll
        for (int k = 0; k < BLOCKSIZE; k++) {

            if ( local_tid.y + k < rows ) {
                output[global_tid + k * rows] = temp[k][threadIdx.x];

            }
        }
    }
}









__device__ double setZero(double a){
    return 0.0;
}

__device__ float setZero(float a){
    return 0.0;
}

__device__ cuDoubleComplex setZero(cuDoubleComplex a){
    return make_cuDoubleComplex(0.0, 0.0);
}

__device__ cuFloatComplex setZero(cuFloatComplex a){
    return make_cuFloatComplex(0.0, 0.0);
}

__device__ double setOne(double a){
    return 1.0;
}

__device__ float setOne(float a){
    return 1.0;
}

__device__ cuDoubleComplex setOne(cuDoubleComplex a){
    return make_cuDoubleComplex(1.0, 0.0);
}

__device__ cuFloatComplex setOne(cuFloatComplex a){
    return make_cuFloatComplex(1.0, 0.0);
}






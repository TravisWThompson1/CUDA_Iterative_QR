/**
 * CUDA kernel utility routines.
 *
 *
 *
 * @author Travis W. Thompson
 * @date 12/07/2017
 *
 * @file kernels.cu
 * @version 1.0
 */


#include "kernel_utility.cuh"




/**
 * Copies the input strided matrices to the strided output matrices.
 * @tparam T Data type template (float, double, cuFloatComplex, cuDoubleComplex, etc.).
 * @param d_input Stided input matrices to be copied.
 * @param d_output Strided output matrices that are copied to.
 * @param totalElem Total number of elements in the input strided matrices.
 */
template <class T>
__global__ void d_Copy_Strided(double* d_input, double* d_output, long int totalElem){
    // Thread id number tid:
    int tid =  blockDim.x * blockIdx.x + threadIdx.x;

    // Ensure thread is within memory allocation.
    if (tid < totalElem){
        // Retrieve input element.
        double temp = d_input[tid];
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
__global__ void d_dCopy_UpperTriangular_Strided(double* d_input, double* d_output, int N, int batchSize){
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
        double temp;
        // Determine whether local index is upper-triangular or not.
        if (i >= j){
            // Upper-triangular element.
            temp = d_input[tid];
        } else {
            // Below diagonal element.
            temp = 0.0;
        }
        // Write output element.
        d_output[tid] = temp;
    }
}


/**
 * Copies the upper-triangular elements in a set of strided column-major matrices of a float data type, while adding
 * zeros in the elements below the diagonal.
 * @param d_input Strided input matrix whose upper triangular elements will be copied to the output strided matrix.
 * @param d_output Strided output matrix that will be copied to.
 * @param N Number of columns or rows in one of the square matrices.
 * @param batchSize Number of matrices in the strided input or output matrices.
 */
__global__ void d_fCopy_UpperTriangular_Strided(float* d_input, float* d_output, int N, int batchSize){
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
        float temp;
        // Determine whether local index is upper-triangular or not.
        if (i >= j){
            // Upper-triangular element.
            temp = d_input[tid];
        } else {
            // Below diagonal element.
            temp = 0.0;
        }
        // Write output element.
        d_output[tid] = temp;
    }
}





/**
 * Copies the upper-triangular elements in a set of strided column-major matrices of a cuFloatData data type, while
 * adding zeros in the elements below the diagonal.
 * @param d_input Strided input matrix whose upper triangular elements will be copied to the output strided matrix.
 * @param d_output Strided output matrix that will be copied to.
 * @param N Number of columns or rows in one of the square matrices.
 * @param batchSize Number of matrices in the strided input or output matrices.
 */
__global__ void d_cCopy_UpperTriangular_Strided(cuFloatComplex* d_input, cuFloatComplex* d_output, int N, int batchSize){
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
        cuFloatComplex temp;
        // Determine whether local index is upper-triangular or not.
        if (i >= j){
            // Upper-triangular element.
            temp = d_input[tid];
        } else {
            // Below diagonal element.
            temp = make_cuFloatComplex(0.0, 0.0);
        }
        // Write output element.
        d_output[tid] = temp;
    }
}




/**
 * Copies the upper-triangular elements in a set of strided column-major matrices of a cuDoubleData data type, while
 * adding zeros in the elements below the diagonal.
 * @param d_input Strided input matrix whose upper triangular elements will be copied to the output strided matrix.
 * @param d_output Strided output matrix that will be copied to.
 * @param N Number of columns or rows in one of the square matrices.
 * @param batchSize Number of matrices in the strided input or output matrices.
 */
__global__ void d_zCopy_UpperTriangular_Strided(cuDoubleComplex* d_input, cuDoubleComplex* d_output, int N, int batchSize){
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
        cuDoubleComplex temp;
        // Determine whether local index is upper-triangular or not.
        if (i >= j){
            // Upper-triangular element.
            temp = d_input[tid];
        } else {
            // Below diagonal element.
            temp = make_cuDoubleComplex(0.0, 0.0);
        }
        // Write output element.
        d_output[tid] = temp;
    }
}





















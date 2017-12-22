//
// Created by travis on 12/8/17.
//

#ifndef KERNEL_UTILITY_CUH
#define KERNEL_UTILITY_CUH

// C libraries
#include <stdio.h>
//#include <stdlib.h>

// CUDA libraries
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuComplex.h"
#include <cublas_v2.h>



/**
 * Copies the input strided matrices to the strided output matrices.
 * @tparam T Data type template (float, double, cuFloatComplex, cuDoubleComplex, etc.).
 * @param d_input Stided input matrices to be copied.
 * @param d_output Strided output matrices that are copied to.
 * @param totalElem Total number of elements in the input strided matrices.
 */
//template <class T>
__global__ void d_Copy_Strided(double* d_input, double* d_output, long int totalElem);




/**
 * Copies the upper-triangular elements in a set of strided column-major matrices of a double data type, while adding
 * zeros in the elements below the diagonal.
 * @param d_input Strided input matrix whose upper triangular elements will be copied to the output strided matrix.
 * @param d_output Strided output matrix that will be copied to.
 * @param N Number of columns or rows in one of the square matrices.
 * @param batchSize Number of matrices in the strided input or output matrices.
 */
__global__ void d_dCopy_UpperTriangular_Strided(double* d_input, double* d_output, int N, int batchSize);





/**
 * Copies the upper-triangular elements in a set of strided column-major matrices of a float data type, while adding
 * zeros in the elements below the diagonal.
 * @param d_input Strided input matrix whose upper triangular elements will be copied to the output strided matrix.
 * @param d_output Strided output matrix that will be copied to.
 * @param N Number of columns or rows in one of the square matrices.
 * @param batchSize Number of matrices in the strided input or output matrices.
 */
__global__ void d_fCopy_UpperTriangular_Strided(float* d_input, float* d_output, int N, int batchSize);





/**
 * Copies the upper-triangular elements in a set of strided column-major matrices of a cuFloatData data type, while
 * adding zeros in the elements below the diagonal.
 * @param d_input Strided input matrix whose upper triangular elements will be copied to the output strided matrix.
 * @param d_output Strided output matrix that will be copied to.
 * @param N Number of columns or rows in one of the square matrices.
 * @param batchSize Number of matrices in the strided input or output matrices.
 */
__global__ void d_cCopy_UpperTriangular_Strided(cuFloatComplex* d_input, cuFloatComplex* d_output, int N, int batchSize);





/**
 * Copies the upper-triangular elements in a set of strided column-major matrices of a cuDoubleData data type, while
 * adding zeros in the elements below the diagonal.
 * @param d_input Strided input matrix whose upper triangular elements will be copied to the output strided matrix.
 * @param d_output Strided output matrix that will be copied to.
 * @param N Number of columns or rows in one of the square matrices.
 * @param batchSize Number of matrices in the strided input or output matrices.
 */
__global__ void d_zCopy_UpperTriangular_Strided(cuDoubleComplex* d_input, cuDoubleComplex* d_output, int N, int batchSize);






#endif //KERNEL_UTILITY_CUH

// C libraries
#include <stdio.h>

// CUDA libraries
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuComplex.h"
#include <cublas_v2.h>

// kblas libraries
#include "kblas_struct.h"
//#include "kblas_common.h"
//#include "kblas_common.cpp"
#include "kblas.h"
//#include "batch_qr.cu"

#include "kernel_utility.cuh"

#define MAX_BLOCKSIZE 512




kblasHandle_t createHandle();

//template <class T>
void copy_Strided(double *d_input, double *d_output, int N, int batchSize);

void copy_dUpperTriangular_Strided(double *d_input, double *d_output, int N, int batchSize);

void QR_Batched(kblasHandle_t kblas_handle, double *d_A, double *d_Q, double *d_R, int N, int batchSize);





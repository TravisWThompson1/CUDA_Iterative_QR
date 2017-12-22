

#include "qr_batched.cuh"



// nvcc -gencode arch=compute_50,code=sm_50 -o test test_cu.cu -I/home/travis/Documents/kblas-gpu/include -I/home/travis/Documents/kblas-gpu/src -I/home/travis/Documents/kblas-gpu/src/batch_svd/ -lcublas


kblasHandle_t createHandle(){
    // Create kblas handle (internally creates cublas handle as well).
    kblasHandle_t kblas_handle;
    kblasCreate(&kblas_handle);
    // Return kblas handle.
    return kblas_handle;
}



//template <class T>
void copy_Strided(double *d_input, double *d_output, int N, int batchSize){
    // Initialize blocksize as the number of columns * batchSize.
    dim3 blockSize(N*N*batchSize, 1, 1);
    dim3 gridSize(blockSize.x / 512 + 1, 1);

    // Calculate total number of elements.
    long int totalElem = N * N * batchSize;

    // Call batched copy kernel.
    d_Copy_Strided<<<gridSize, blockSize>>>(d_input, d_output, totalElem);

}



void copy_dUpperTriangular_Strided(double *d_input, double *d_output, int N, int batchSize){
    // Initialize blocksize as the number of columns * batchSize.
    dim3 blockSize(N*N*batchSize, 1, 1);
    dim3 gridSize(blockSize.x / 512 + 1, 1);

    // Call batched copy kernel.
    d_dCopy_UpperTriangular_Strided<<<gridSize, blockSize>>>(d_input, d_output, N, batchSize);

}



void QR_Batched(kblasHandle_t kblas_handle, double *d_A, double *d_Q, double *d_R, int N, int batchSize){



    // Create batched tau array for workspace.
    double *d_tau;
    // Allocate batched array on device.
    cudaMalloc((void**)&d_tau, batchSize*N*sizeof(double));

    // Call QR decomposition routine in kblas-gpu.
    int res = kblasDgeqrf_batch_strided(kblas_handle, N, N, d_A, N, N*N, d_tau, N, batchSize);

    // Copy upper triangular portion of matrix A to matrix R.
    copy_dUpperTriangular_Strided(d_A, d_R, N, batchSize);


    // Call orthogonal matrix Q retriever routine in kblas-gpu to get the full Q matrix.
    res = kblasDorgqr_batch_strided(kblas_handle, N, N, d_A, N, N*N, d_tau, N, batchSize);

    // Copy all of matrix A into matrix Q.
    copy_Strided(d_A, d_R, N, batchSize);

    cudaFree(d_tau);

}
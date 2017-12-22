

#include "qr_batched.cuh"



// nvcc -gencode arch=compute_50,code=sm_50 -o test test_cu.cu -I/home/travis/Documents/kblas-gpu/include -I/home/travis/Documents/kblas-gpu/src -I/home/travis/Documents/kblas-gpu/src/batch_svd/ -lcublas


kblasHandle_t createHandle(){
    // Create kblas handle (internally creates cublas handle as well).
    kblasHandle_t kblas_handle;
    kblasCreate(&kblas_handle);
    // Return kblas handle.
    return kblas_handle;
}




void copy_dUpperTriangular_Strided(double *d_input, double *d_output, int N, int batchSize){
    // Initialize blocksize as the number of total elemenets.
    dim3 blockSize(N*N*batchSize, 1, 1);
    dim3 gridSize(blockSize.x / MAX_BLOCKSIZE + 1, 1);

    // Call batched copy kernel.
    d_dCopy_UpperTriangular_Strided<<<gridSize, blockSize>>>(d_input, d_output, N, batchSize);

}


void dInitialize_Identity_Batched(double *d_A, int N, int batchSize){
    // Initialize blocksize as the number of total elements.
    dim3 blockSize(N*N*batchSize, 1, 1);
    dim3 gridSize(blockSize.x / MAX_BLOCKSIZE + 1, 1);

    // Call batched initialize identity kernel.
    d_dInitialize_Identity_Batched<<<gridSize, blockSize>>>(d_A, N, batchSize);
}


//template <class T>
void dCopy_Strided(double *d_input, double *d_output, int N, int batchSize){
    // Initialize blocksize as the number of total elements.
    dim3 blockSize(N*N*batchSize, 1, 1);
    dim3 gridSize(blockSize.x / MAX_BLOCKSIZE + 1, 1);

    // Calculate total number of elements.
    long int totalElem = N * N * batchSize;

    // Call batched initialize identity kernel.
    d_dCopy_Strided<<<gridSize, blockSize>>>(d_input, d_output, totalElem);
}



bool check_Convergence(double *d_A, int *d_INFO, int N, int batchSize){
    // Initialize blocksize as the number of total elements.
    dim3 blockSize(N*N*batchSize, 1, 1);
    dim3 gridSize(blockSize.x / MAX_BLOCKSIZE + 1, 1);

    // Initialize INFO as zero.
    int INFO = 0;

    // Initialize INFO device integer.
    int *d_INFO;

    // Allocate INFO on device.
    cudaMalloc((void**)&d_INFO, sizeof(int));

    // Copy INFO from host memory to device memory.
    cudaMemcpy(d_INFO, INFO, sizeof(int), cudaMemcpyHostToDevice);

    d_dConvergence_Check<<<gridSize, blockSize>>>(d_A, d_INFO, N, batchSize);

    // Copy INFO from device memory to host memory.
    cudaMemcpy(INFO, d_INFO, sizeof(int), cudaMemcpyDeviceToHost);

    if (INFO == 0){
        return true;
    } else {
        return false;
    }
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





void QR_Eigenvalue_Batched(kblasHandle_t kblas_handle, double *d_A, double *d_Eigenvalues, int N, int batchSize){
    // Initialize alpha and beta as one.
    double alpha = 1.0, beta = 1.0;

    // Initialize Q and R batched matrices on device.
    double *d_Q, *d_R, *d_RQ, *d_QQ, *d_temp;
    double *d_alpha, *d_beta;

    // Allocate Q and R batched matrices on device.
    cudaMalloc((void**)&d_Q,    batchSize*N*N*sizeof(double));
    cudaMalloc((void**)&d_R,    batchSize*N*N*sizeof(double));
    cudaMalloc((void**)&d_RQ,   batchSize*N*N*sizeof(double));
    cudaMalloc((void**)&d_QQ,   batchSize*N*N*sizeof(double));
    cudaMalloc((void**)&d_temp, batchSize*N*N*sizeof(double));

    // Allocate alpha and beta on device.
    cudaMalloc((void**)&d_alpha, sizeof(double));
    cudaMalloc((void**)&d_beta,  sizeof(double));

    // Copy alpha and beta values.
    cudaMemcpy(alpha, d_alpha, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(beta,  d_beta,  sizeof(double), cudaMemcpyHostToDevice);

    // Initialize QQ matrix as an identity matrix.
    dInitialize_Identity_Batched(d_QQ, N, batchSize);

    bool convergence = false;
    // Loop until convergence.
    while (convergence != true){

        // Perform QR decomposition.
        QR_Batched(d_A, d_Q, d_R, N, batchSize);

        // Multiply R * Q.
        cublasXgemm(kblas_handle.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                    d_alpha, d_R, N, d_Q, N, d_beta, d_RQ, N);

        // Multiply QQ * Q
        cublasXgemm(kblas_handle.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                    d_alpha, d_QQ, N, d_Q, N, d_beta, d_temp, N);

        // Update A matrix as RQ.
        dCopy_Strided(d_QQ, d_temp, N, batchSize);

        // Check for convergence.
        convergence = check_Convergence(d_A);
    }



}














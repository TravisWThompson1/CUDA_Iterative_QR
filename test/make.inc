###########################################################

# USER SPECIFIC DIRECTORIES

# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda

##########################################################

# NVCC COMPILER OPTIONS
# nvcc -gencode arch=compute_50,code=sm_50 test_cu.cu -I/home/travis/Documents/kblas-gpu/include -I/home/travis/Documents/kblas-gpu/src -I/home/travis/Documents/magma-2.2.0/include/ -I/home/travis/Documents/lapack-3.7.1/include -L/home/travi s/Documents/kblas-gpu/lib -L/home/travis/Documents/magma-2.2.0/lib -lkblas-gpu -lcublas -lcusparse -lmagma

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS= -gencode arch=compute_30,code=sm_30
NVCC_LIBS=

# NVCC library directories:
NVCC_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
NVCC_INC_DIR= -I$(CUDA_ROOT_DIR)/include

##########################################################














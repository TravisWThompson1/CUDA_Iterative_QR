###########################################################

# USER SPECIFIC DIRECTORIES

# KBLAS Directory:
KBLAS_ROOT_DIR=/home/travis/Documents/kblas-gpu

# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda

##########################################################

# NVCC COMPILER OPTIONS

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS= -gencode arch=compute_50,code=sm_50 -std=c++11 --shared --compiler-options -fPIC
NVCC_LIBS= -lcudart -lcublas

# NVCC library directories:
NVCC_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
NVCC_INC_DIR= -I$(CUDA_ROOT_DIR)/include

##########################################################

# Additional variables:

# kblas library directory.
KBLAS_LIB_DIR= -L$(KBLAS_ROOT_DIR)/lib

# kblas library flag:
KBLAS_LIB_FLAG=-lkblas-gpu

# Include directories.
KBLAS_INC_DIR= -I$(KBLAS_ROOT_DIR)/src
KBLAS_INC_DIR+= -I$(KBLAS_ROOT_DIR)/include
KBLAS_INC_DIR+= -I$(KBLAS_ROOT_DIR)/src/batch_svd



##########################################################







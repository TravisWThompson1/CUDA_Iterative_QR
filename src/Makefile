# Include externeal makefile variables:
include ../make.inc

# Define objects directory:
OBJ_DIR = "../obj"

# Define source files directory:
SRC_DIR = "../src"

# Define include header files directory:
INC_DIR = "../include"

# Define objects to be compiled:
OBJS = $(OBJ_DIR)/kernel_utility.cu_o $(OBJ_DIR)/qr_batched.cu_o


# Compile all objects:
all: $(OBJS)

# Compile kernel utilities:
$(OBJ_DIR)/kernel_utility.cu_o: kernel_utility.cu kernel_utility.cuh
	$(NVCC) $(NVCC_FLAGS) -c $(CUDA_INC_DIR) $< -o $@

# Compile QR Batched routines:
$(OBJ_DIR)/qr_batched.cu_o: qr_batched.cu qr_batched.cuh
	$(NVCC) $(NVCC_FLAGS) $(KBLAS_LIB_DIR) $(KBLAS_LIB_FLAG) -c $(KBLAS_INC_DIR) $< -o $@

# Clean objects in object directory.
clean:
	$(RM) $(OBJ_DIR)/*.cu_o





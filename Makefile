# Include externeal makefile variables:
include make.inc

# Define objects directory:
OBJ_DIR = "obj"

# Define source files directory:
SRC_DIR = "src"

# Define include header files directory:
INC_DIR = "include"

# Define all objects in object directory:
OBJS = $(OBJ_DIR)/kernel_utility.cu_o $(OBJ_DIR)/qr_batched.cu_o

# Define executable name.
EXE = test_run


all:
	nvcc -gencode arch=compute_50,code=sm_50 test_cu.cu -I/home/travis/Documents/kblas-gpu/include -L/home/travis/Documents/kblas-gpu/lib -L/home/travis/Documents/magma-2.2.0/lib -L/home/travis/Documents/OpenBLAS -L/home/travis/Documents/lapack-3.7.1 -lkblas-gpu -lcublas -lcusparse -lopenblas -lmagma -o TEST_QR



#
## Compile all.
#all : SRC_CODE test_cu.o $(EXE)
#
## Link object files.
#$(EXE): test_cu.o
#	$(NVCC) test_cu.o -o $(EXE)
#
## Compile test case.
#test_cu.o: test_cu.cu
#	$(NVCC) $(NVCC_FLAGS) $(KBLAS_LIB_DIR) -c $(KBLAS_INC_DIR) $< -o $@
#
## Compile source code with sub-makefile.
#SRC_CODE:
#	make -C $(SRC_DIR) all

# Clean compilation files and run files.
clean:
	make -C $(SRC_DIR) clean
	rm -f *o $(EXE)
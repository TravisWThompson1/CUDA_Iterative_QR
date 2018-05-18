# Include externeal makefile variables:
include make.inc

# Define objects directory:
OBJ_DIR = "obj"

# Define source files directory:
SRC_DIR = "src"

# Define include header files directory:
INC_DIR = "../include"

# Define all objects in object directory:
OBJS = $(OBJ_DIR)/test_transpose.cu_o


all:
	$(NVCC) $(NVCC_FLAGS) -I$(INC_DIR) $(SRC_DIR)/test_transpose.cu -o TEST_TRANSPOSE


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
	rm -f *o TEST_TRANSPOSE
CUDA_HOME := /usr/local/cuda
NVCC := $(CUDA_HOME)/bin/nvcc

# Prefer repository-local CGBN headers so tests compile with supported sizes
# In WSL this resolves to /mnt/d/code/MPA-OpenCl/cgbn
CGBN_INCLUDE := ../cgbn
# CGBN_INCLUDE := /home/elysia/Math/CGBN/include/cgbn

INCLUDES := -I$(CGBN_INCLUDE)
LIBS := -lgmp

ARCH := -arch=sm_89

NVCC_FLAGS := $(ARCH) -O3 \
				--ptxas-options=-v

TARGET := bench
SRC_CPP := main.cpp
SRC_CU := bench_cgbn_2048.cu

BENCH4096_TARGET := bench4096
BENCH4096_CPP := main_4096.cpp
BENCH4096_CU := bench_cgbn_4096.cu

all: $(TARGET)

all: $(BENCH4096_TARGET)

$(TARGET): $(SRC_CPP) $(SRC_CU)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) \
	$(SRC_CPP) $(SRC_CU) \
	-o $(TARGET) $(LIBS)

$(BENCH4096_TARGET): $(BENCH4096_CPP) $(BENCH4096_CU)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) \
	$(BENCH4096_CPP) $(BENCH4096_CU) \
	-o $(BENCH4096_TARGET) $(LIBS)

clean:
	rm -f $(TARGET)

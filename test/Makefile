CUDA_HOME := /usr/local/cuda
NVCC := $(CUDA_HOME)/bin/nvcc

CGBN_INCLUDE := /home/elysia/Math/CGBN/include/cgbn

INCLUDES := -I$(CGBN_INCLUDE)
LIBS := -lgmp

ARCH := -arch=sm_89

NVCC_FLAGS := $(ARCH) -O3 \
				--ptxas-options=-v

TARGET := bench
SRC_CPP := main.cpp
SRC_CU := bench_cgbn_2048.cu

all: $(TARGET)

$(TARGET): $(SRC_CPP) $(SRC_CU)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) \
	$(SRC_CPP) $(SRC_CU) \
	-o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)

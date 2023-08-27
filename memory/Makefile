NVCC_PATH=$(shell which nvcc)
CUDA_PATH=$(shell dirname $(shell dirname $(NVCC_PATH)))

$(info CUDA Toolkit path [${CUDA_PATH}])


CXXFLAGS = -std=c++17 -O3 -I$(CUDA_PATH)/include -I../
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart


TARGET = memcpy

SRC = memcpy.cpp

all: $(TARGET)
$(TARGET): $(SRC)
	$(NVCC_PATH) $(CXXFLAGS) -x cu $(SRC) -o $(TARGET) $(LDFLAGS)
clean:
	rm -f $(TARGET)

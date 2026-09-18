CUDA_ARCH ?= sm_50
NVCC ?= nvcc
CC ?= gcc
CFLAGS ?= -O2 -std=c11
CUDAFLAGS ?= -O2 -std=c++14 -arch=$(CUDA_ARCH)

.PHONY: all build data test benchmark heavy clean

all: build

build: build/cart_cuda build/cart_sequential

build/cart_cuda: src/cart_cuda.cu
	mkdir -p build
	$(NVCC) $(CUDAFLAGS) $< -o $@

build/cart_sequential: src/cart_sequential.c
	mkdir -p build
	$(CC) $(CFLAGS) $< -o $@

data:
	mkdir -p data
	python3 scripts/generate_data.py

test: build data
	: > results/sequential.txt
	: > results/cuda.txt
	./build/cart_sequential 1000 4 200 data/train.bin data/test.bin results/sequential.txt
	./build/cart_cuda 1000 4 200 data/train.bin data/test.bin results/cuda.txt
	python3 scripts/compare_results.py results/sequential.txt results/cuda.txt

benchmark: build
	python3 scripts/benchmark_compare.py

heavy: build
	python3 scripts/tests_heavy.py

clean:
	rm -rf build data/*.bin results/*.txt
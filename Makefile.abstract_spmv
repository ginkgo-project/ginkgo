# Makefile for standalone abstract_spmv implementation
# Extracted from Ginkgo CSR kernels

# CUDA compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -std=c++14 -O3

# Architecture (adjust based on your GPU)
# Common options:
#   sm_60: Pascal (GTX 10xx, Tesla P100)
#   sm_70: Volta (Tesla V100)
#   sm_75: Turing (RTX 20xx, Tesla T4)
#   sm_80: Ampere (A100, RTX 30xx)
#   sm_86: Ampere (RTX 30xx mobile)
#   sm_89: Ada Lovelace (RTX 40xx)
ARCH = sm_70

# Default target
.PHONY: all
all: test

# Compile the test program
test: abstract_spmv_test.cu
	@echo "Compiling abstract_spmv test..."
	$(NVCC) $(NVCC_FLAGS) -arch=$(ARCH) $< -o abstract_spmv_test
	@echo "Done! Run with: ./abstract_spmv_test"

# Compile the standalone library as an object file
lib: abstract_spmv_standalone.cu
	@echo "Compiling abstract_spmv as library..."
	$(NVCC) $(NVCC_FLAGS) -arch=$(ARCH) -dc $< -o abstract_spmv.o
	@echo "Done! Link with: nvcc your_code.cu abstract_spmv.o -o your_program"

# Run the test
.PHONY: run
run: test
	@echo "Running test..."
	@./abstract_spmv_test

# Clean build artifacts
.PHONY: clean
clean:
	rm -f abstract_spmv_test abstract_spmv.o

# Show GPU information
.PHONY: gpu-info
gpu-info:
	@echo "GPU Information:"
	@nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

# Help message
.PHONY: help
help:
	@echo "Makefile for standalone abstract_spmv"
	@echo ""
	@echo "Targets:"
	@echo "  all (default) - Compile the test program"
	@echo "  test          - Compile the test program"
	@echo "  lib           - Compile as a library object file"
	@echo "  run           - Compile and run the test"
	@echo "  clean         - Remove build artifacts"
	@echo "  gpu-info      - Display GPU information"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  ARCH          - CUDA architecture (default: sm_70)"
	@echo "                  Override with: make ARCH=sm_80"
	@echo ""
	@echo "Examples:"
	@echo "  make              # Compile test with default settings"
	@echo "  make ARCH=sm_80   # Compile for Ampere architecture"
	@echo "  make run          # Compile and run test"
	@echo "  make lib          # Compile as library"

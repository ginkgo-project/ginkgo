#!/bin/bash

# Compilation test script for abstract_spmv standalone implementation

set -e  # Exit on error

echo "========================================"
echo "Testing abstract_spmv compilation"
echo "========================================"
echo

# Check if nvcc is available
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please install CUDA Toolkit."
    exit 1
fi

echo "NVCC version:"
nvcc --version
echo

# Test 1: Minimal test
echo "Test 1: Compiling minimal test..."
nvcc -std=c++14 -arch=sm_70 test_minimal.cu -o test_minimal 2>&1 | tee compile_minimal.log
if [ $? -eq 0 ]; then
    echo "✓ Minimal test compiled successfully!"
    ./test_minimal
    echo
else
    echo "✗ Minimal test failed to compile"
    echo "See compile_minimal.log for details"
    exit 1
fi

# Test 2: Standalone implementation
echo "Test 2: Compiling standalone implementation..."
nvcc -std=c++14 -arch=sm_70 -c abstract_spmv_standalone.cu -o abstract_spmv_standalone.o 2>&1 | tee compile_standalone.log
if [ $? -eq 0 ]; then
    echo "✓ Standalone implementation compiled successfully!"
    echo
else
    echo "✗ Standalone implementation failed to compile"
    echo "See compile_standalone.log for details"
    exit 1
fi

# Test 3: Full test
echo "Test 3: Compiling full test..."
nvcc -std=c++14 -arch=sm_70 abstract_spmv_test.cu -o abstract_spmv_test 2>&1 | tee compile_test.log
if [ $? -eq 0 ]; then
    echo "✓ Full test compiled successfully!"
    echo
    echo "Running test..."
    ./abstract_spmv_test
    if [ $? -eq 0 ]; then
        echo
        echo "✓ Test passed!"
    else
        echo
        echo "✗ Test failed"
        exit 1
    fi
else
    echo "✗ Full test failed to compile"
    echo "See compile_test.log for details"
    exit 1
fi

echo
echo "========================================"
echo "All tests passed successfully!"
echo "========================================"

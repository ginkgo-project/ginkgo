// Copyright (c) 2022, NVIDIA CORPORATION.
//
// SPDX-License-Identifier: Apache-2.0

#include <hip/hip_runtime.h>

#include <stdexcept>

__global__ void kernel() { printf("The kernel ran!\n"); }

void test_hipLaunchKernel()
{
    hipStream_t stream;
    hipStreamCreate(&stream);
    kernel<<<1, 1, 0, stream>>>();
    hipError_t err{hipDeviceSynchronize()};
    if (err != hipSuccess) {
        throw std::runtime_error("Kernel failed on non-default stream!");
    }
    err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error("Kernel failed on non-default stream!");
    }

    try {
        kernel<<<1, 1>>>();
    } catch (std::runtime_error) {
        return;
    }
    throw std::runtime_error(
        "No exception raised for kernel on default stream!");
}

int main() { test_hipLaunchKernel(); }
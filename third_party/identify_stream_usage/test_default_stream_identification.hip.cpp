/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
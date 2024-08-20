// SPDX-FileCopyrightText: 2024 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#if ENABLE_CUDA
#include <cuda_runtime.h>

#define GKO_ASSERT_NO_GPU_ERRORS(__call) GKO_ASSERT_NO_CUDA_ERRORS(__call)
#define gpuMemcpyFromSymbol(...) cudaMemcpyFromSymbol(__VA_ARGS__)

#elif ENABLE_HIP
#include <hip/hip_runtime.h>

#define GKO_ASSERT_NO_GPU_ERRORS(__call) GKO_ASSERT_NO_HIP_ERRORS(__call)
#define gpuMemcpyFromSymbol(...) hipMemcpyFromSymbol(__VA_ARGS__)

#endif

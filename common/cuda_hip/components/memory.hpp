// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_COMPONENTS_MEMORY_HPP_
#define GKO_COMMON_CUDA_HIP_COMPONENTS_MEMORY_HPP_


#if defined(GKO_COMPILING_CUDA)
#include "cuda/components/memory.cuh"
#elif defined(GKO_COMPILING_HIP)
#include "hip/components/memory.hip.hpp"
#else
#error "Executor definition missing"
#endif


#endif  // GKO_COMMON_CUDA_HIP_COMPONENTS_MEMORY_HPP_

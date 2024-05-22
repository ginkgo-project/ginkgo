// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_BASE_RUNTIME_HPP_
#define GKO_COMMON_CUDA_HIP_BASE_RUNTIME_HPP_


#ifdef GKO_COMPILING_CUDA
// nothing needed here
#elif defined(GKO_COMPILING_HIP)
#include <hip/hip_runtime.h>
#else
#error "Executor definition missing"
#endif


#endif  // GKO_COMMON_CUDA_HIP_BASE_RUNTIME_HPP_

// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_BASE_POINTER_MODE_GUARD_HPP_
#define GKO_COMMON_CUDA_HIP_BASE_POINTER_MODE_GUARD_HPP_


#if defined(GKO_COMPILING_CUDA)
#include "cuda/base/pointer_mode_guard.hpp"
#elif defined(GKO_COMPILING_HIP)
#include "hip/base/pointer_mode_guard.hip.hpp"
#else
#error "Executor definition missing"
#endif


#endif  // GKO_COMMON_CUDA_HIP_BASE_POINTER_MODE_GUARD_HPP_

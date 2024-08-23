// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_BASE_TYPES_HPP_
#define GKO_COMMON_CUDA_HIP_BASE_TYPES_HPP_


#if defined(GKO_COMPILING_CUDA)
#include "cuda/base/types.hpp"
#define device_type cuda_type
#define as_device_type as_cuda_type
#elif defined(GKO_COMPILING_HIP)
#include "hip/base/types.hip.hpp"
#define device_type hip_type
#define as_device_type as_hip_type
#else
#error "Executor definition missing"
#endif


#endif  // GKO_COMMON_CUDA_HIP_BASE_TYPES_HPP_

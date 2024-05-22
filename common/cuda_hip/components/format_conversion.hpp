// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_COMPONENTS_FORMAT_CONVERSION_HPP_
#define GKO_COMMON_CUDA_HIP_COMPONENTS_FORMAT_CONVERSION_HPP_


#ifdef GKO_COMPILING_CUDA
#include "cuda/components/format_conversion.cuh"
#elif defined(GKO_COMPILING_HIP)
#include "hip/components/format_conversion.hip.hpp"
#else
#error "Executor definition missing"
#endif


#endif  // GKO_COMMON_CUDA_HIP_COMPONENTS_FORMAT_CONVERSION_HPP_

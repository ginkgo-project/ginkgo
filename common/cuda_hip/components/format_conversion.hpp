// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_COMPONENTS_FORMAT_CONVERSION_HPP_
#define GKO_COMMON_CUDA_HIP_COMPONENTS_FORMAT_CONVERSION_HPP_


#ifdef GKO_COMPILING_HIP
#include "hip/components/format_conversion.hip.hpp"
#else  // GKO_COMPILING_CUDA
#include "cuda/components/format_conversion.cuh"
#endif


#endif  // GKO_COMMON_CUDA_HIP_COMPONENTS_FORMAT_CONVERSION_HPP_

// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_BASE_ACCESSOR_HPP_
#define GKO_COMMON_CUDA_HIP_BASE_ACCESSOR_HPP_


#include <ginkgo/core/base/bfloat16.hpp>
#include <ginkgo/core/base/half.hpp>

#include "common/cuda_hip/base/bf16_alias.hpp"

#ifdef GKO_COMPILING_CUDA
#include <cuda_fp16.h>

#include "accessor/cuda_helper.hpp"
#elif defined(GKO_COMPILING_HIP)
#include <hip/hip_fp16.h>

#include "accessor/hip_helper.hpp"
#endif

namespace acc {


#ifdef GKO_COMPILING_CUDA

template <>
struct cuda_type<gko::half> {
    using type = __half;
};

template <>
struct cuda_type<gko::bfloat16> {
    using type = gko::vendor_bf16;
};

#elif defined(GKO_COMPILING_HIP)

template <>
struct hip_type<gko::half> {
    using type = __half;
};

template <>
struct hip_type<gko::bfloat16> {
    using type = gko::vendor_bf16;
};

#endif


}  // namespace acc


#include "accessor/cuda_hip_helper.hpp"


#endif  // GKO_COMMON_CUDA_HIP_BASE_ACCESSOR_HPP_

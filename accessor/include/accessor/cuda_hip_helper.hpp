// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_ACCESSOR_CUDA_HIP_HELPER_HPP_
#define GKO_ACCESSOR_CUDA_HIP_HELPER_HPP_


#include <utility>


#if defined(__HIPCC__)
#include "accessor/hip_helper.hpp"
#elif defined(__CUDACC__)
#include "accessor/cuda_helper.hpp"
#else
#error \
    "cuda_hip_helper.hpp requires compilation with a CUDA (__CUDACC__) or HIP (__HIPCC__) compiler"
#endif


namespace gko {
namespace acc {


template <typename AccType>
GKO_ACC_INLINE auto as_device_range(AccType&& acc)
{
#if defined(__HIPCC__)
    return as_hip_range(std::forward<AccType>(acc));
#elif defined(__CUDACC__)
    return as_cuda_range(std::forward<AccType>(acc));
#endif
}


}  // namespace acc
}  // namespace gko


#endif  // GKO_ACCESSOR_CUDA_HIP_HELPER_HPP_

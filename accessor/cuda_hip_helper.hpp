// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_ACCESSOR_CUDA_HIP_HELPER_HPP_
#define GKO_ACCESSOR_CUDA_HIP_HELPER_HPP_


#include <utility>


#ifdef GKO_COMPILING_HIP
#include "accessor/hip_helper.hpp"
#else  // GKO_COMPILING_CUDA
#include "accessor/cuda_helper.hpp"
#endif


namespace gko {
namespace acc {


template <typename AccType>
GKO_ACC_INLINE auto as_device_range(AccType&& acc)
{
#ifdef GKO_COMPILING_HIP
    return as_hip_range(std::forward<AccType>(acc));
#else  // GKO_COMPILING_CUDA
    return as_cuda_range(std::forward<AccType>(acc));
#endif
}


}  // namespace acc
}  // namespace gko


#endif  // GKO_ACCESSOR_CUDA_HIP_HELPER_HPP_

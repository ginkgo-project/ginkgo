// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_UNIFIED_BASE_KERNEL_LAUNCH_HPP_
#error \
    "This file can only be used from inside common/unified/base/kernel_launch.hpp"
#endif


#include <thrust/tuple.h>


#include "accessor/cuda_helper.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/thread_ids.cuh"


namespace gko {
namespace kernels {
namespace cuda {


template <typename AccessorType>
struct to_device_type_impl<gko::acc::range<AccessorType>&> {
    using type = std::decay_t<decltype(gko::acc::as_cuda_range(
        std::declval<gko::acc::range<AccessorType>>()))>;
    static type map_to_device(gko::acc::range<AccessorType>& range)
    {
        return gko::acc::as_cuda_range(range);
    }
};

template <typename AccessorType>
struct to_device_type_impl<const gko::acc::range<AccessorType>&> {
    using type = std::decay_t<decltype(gko::acc::as_cuda_range(
        std::declval<gko::acc::range<AccessorType>>()))>;
    static type map_to_device(const gko::acc::range<AccessorType>& range)
    {
        return gko::acc::as_cuda_range(range);
    }
};


namespace device_std = thrust;


constexpr int default_block_size = 512;


#include "common/cuda_hip/base/kernel_launch.hpp.inc"


}  // namespace cuda
}  // namespace kernels
}  // namespace gko

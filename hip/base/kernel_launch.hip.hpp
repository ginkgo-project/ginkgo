// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_UNIFIED_BASE_KERNEL_LAUNCH_HPP_
#error \
    "This file can only be used from inside common/unified/base/kernel_launch.hpp"
#endif


#include <hip/hip_runtime.h>
#include <thrust/tuple.h>


#include "accessor/hip_helper.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {


template <typename AccessorType>
struct to_device_type_impl<gko::acc::range<AccessorType>&> {
    using type = std::decay_t<decltype(gko::acc::as_hip_range(
        std::declval<gko::acc::range<AccessorType>>()))>;
    static type map_to_device(gko::acc::range<AccessorType>& range)
    {
        return gko::acc::as_hip_range(range);
    }
};

template <typename AccessorType>
struct to_device_type_impl<const gko::acc::range<AccessorType>&> {
    using type = std::decay_t<decltype(gko::acc::as_hip_range(
        std::declval<gko::acc::range<AccessorType>>()))>;
    static type map_to_device(const gko::acc::range<AccessorType>& range)
    {
        return gko::acc::as_hip_range(range);
    }
};


namespace device_std = thrust;


constexpr int default_block_size = 512;


#include "common/cuda_hip/base/kernel_launch.hpp.inc"


}  // namespace hip
}  // namespace kernels
}  // namespace gko

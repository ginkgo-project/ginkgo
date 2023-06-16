/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

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

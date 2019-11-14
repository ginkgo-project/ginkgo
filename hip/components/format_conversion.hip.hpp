/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#ifndef GKO_HIP_COMPONENTS_FORMAT_CONVERSION_HIP_HPP_
#define GKO_HIP_COMPONENTS_FORMAT_CONVERSION_HIP_HPP_


#include <ginkgo/core/base/std_extensions.hpp>


#include "hip/base/config.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
namespace coo {
namespace host_kernel {


/**
 * @internal
 *
 * It calculates the number of warps used in Coo Spmv depending on the GPU
 * architecture and the number of stored elements.
 */
template <size_type subwarp_size = config::warp_size>
__host__ size_type calculate_nwarps(std::shared_ptr<const HipExecutor> exec,
                                    const size_type nnz)
{
    size_type nwarps_in_hip = exec->get_num_multiprocessor() *
                              exec->get_num_warps_per_sm() * config::warp_size /
                              subwarp_size;
#if GINKGO_HIP_PLATFORM_NVCC
    size_type multiple = 8;
    if (nnz >= 2e6) {
        multiple = 128;
    } else if (nnz >= 2e5) {
        multiple = 32;
    }
#else
    size_type multiple = 2;
    if (nnz >= 1e7) {
        multiple = 32;
    } else if (nnz >= 1e5) {
        multiple = 8;
    }
#endif  // GINKGO_HIP_PLATFORM_NVCC
    return std::min(multiple * nwarps_in_hip,
                    size_type(ceildiv(nnz, config::warp_size)));
}


}  // namespace host_kernel
}  // namespace coo
}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_COMPONENTS_FORMAT_CONVERSION_HIP_HPP_
